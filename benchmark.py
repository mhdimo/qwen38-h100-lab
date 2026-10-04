"""Run one isolated vLLM configuration. GPU imports occur only inside run()."""
import argparse
import asyncio
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

from measurement import consume, run_load, summarize

HERE = Path(__file__).resolve().parent
MODEL = os.environ.get('LAB_MODEL_PATH',
    '/models/Qwen3.8-27B-FP8/snapshots/017b9c7af6b5689d5dd426a76e0bc077eb5ca20a')


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, default=str) + '\n')
    temporary.replace(path)


def hardware():
    return subprocess.check_output(['nvidia-smi', '--query-gpu=name,uuid,driver_version,memory.total,memory.used,utilization.gpu,power.draw,power.limit,clocks.current.sm,clocks.current.memory,temperature.gpu', '--format=csv'], text=True).strip()


def check_idle_gpu():
    processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory', '--format=csv,noheader'], text=True).strip()
    if processes:
        raise RuntimeError('GPU already has compute processes; refusing contention: ' + processes)


async def run(args):
    # Absolute Python invocation alone does not expose its installed JIT helpers.
    os.environ['PATH'] = str(Path(sys.executable).parent) + os.pathsep + os.environ.get('PATH', '')
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    os.environ.setdefault('VLLM_WORKER_MULTIPROC_METHOD', 'spawn')
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    check_idle_gpu()
    from vllm import AsyncEngineArgs, SamplingParams
    from vllm.v1.engine.async_llm import AsyncLLM
    config = json.loads(args.config.read_text())
    workloads = json.loads(args.workloads.read_text())
    engine_options = dict(
        model=MODEL, tensor_parallel_size=1, max_model_len=10240,
        gpu_memory_utilization=.85, enforce_eager=False,
        enable_prefix_caching=True, enable_chunked_prefill=True,
        disable_log_stats=False, seed=0,
        mm_processor_cache_gb=1,
        mm_processor_kwargs={'min_pixels': 1024, 'max_pixels': 2621440},
        limit_mm_per_prompt={'video': 1, 'image': 1},
    )
    engine_options.update(config['engine'])
    if args.profile:
        engine_options['profiler_config'] = {
            'profiler': 'torch', 'torch_profiler_dir': str(args.output.parent.resolve() / 'traces'),
            'torch_profiler_with_stack': False, 'torch_profiler_record_shapes': False,
            'ignore_frontend': True, 'max_iterations': 12,
        }
        if args.profiler_kind == 'cuda':
            engine_options['profiler_config'] = {'profiler':'cuda','max_iterations':12}
    metadata = {
        'started_at': datetime.now(timezone.utc).isoformat(), 'config': config,
        'engine_options': engine_options, 'gpu_before': hardware(),
        'versions': {x: importlib.metadata.version(x) for x in
                     ['vllm', 'torch', 'triton', 'transformers', 'flashinfer-python']},
        'method': 'AsyncLLM direct, closed-loop concurrency, cumulative output, fixed length, seeded unique cold prompts. TTFT includes scheduler queue. TPOT=(completion-first token)/(output count-1).',
        'profiled': args.profile,
        'python_executable': sys.executable,
        'source_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in [Path(__file__), HERE / 'measurement.py', args.config, args.workloads]},
        'status': 'starting', 'results': [],
    }
    save(args.output, metadata)
    started = time.perf_counter()
    engine_config = AsyncEngineArgs(**engine_options).create_engine_config()
    metadata['resolved_engine_config'] = str(engine_config)
    engine = AsyncLLM.from_vllm_config(engine_config)
    metadata['load_seconds'] = time.perf_counter() - started
    try:
        # Natural-text smoke outputs are kept for comparison across configurations.
        tokenizer = engine.get_tokenizer()
        checks = []
        for index, (question, expected) in enumerate([
            ('Reply with only the city name: What is the capital of France?', 'Paris'),
            ('Reply with only the number: What is 7 multiplied by 8?', '56'),
            ('Reply with only the word: What is the opposite of hot?', 'cold'),
        ]):
            prompt = tokenizer.apply_chat_template([{'role':'user','content':question}],
                tokenize=False, add_generation_prompt=True, enable_thinking=False)
            final = None
            async for output in engine.generate({'prompt': prompt}, SamplingParams(temperature=0, max_tokens=32), f'quality-{index}'):
                final = output
            answer = final.outputs[0]
            checks.append({'question':question, 'expected':expected, 'text':answer.text,
                           'token_ids': list(answer.token_ids), 'passed':expected.lower() in answer.text.lower()})
        metadata['quality_smoke'] = checks
        if not all(x['passed'] for x in checks):
            raise RuntimeError('quality smoke check failed')
        for case_index, case in enumerate(workloads):
            for repetition in range(args.repetitions):
                sampling = SamplingParams(temperature=0, max_tokens=case['output_tokens'],
                                          min_tokens=case['output_tokens'], ignore_eos=True, detokenize=False)
                # All generation/seed work happens before timing, identical per candidate.
                rng = random.Random(20261001 + case_index * 1000 + repetition)
                warmup_count = min(case['concurrency'], engine_options['max_num_seqs'])
                prompts = [[rng.randrange(1000, 20000) for _ in range(case['input_tokens'])]
                           for _ in range(case['requests'] + warmup_count)]
                async def request(index):
                    rid = f"{case['name']}-r{repetition}-n{index}"
                    started = time.perf_counter()
                    async def events():
                        async for output in engine.generate({'prompt_token_ids':prompts[index]}, sampling, rid):
                            tokens = output.outputs[0].token_ids if output.outputs else []
                            yield time.perf_counter(), tokens, output.finished, int(output.num_cached_tokens or 0)
                    row, tokens = await consume(events(), started, case['input_tokens'], case['output_tokens'])
                    return {'sample': asdict(row), 'token_sha256':hashlib.sha256(json.dumps(tokens).encode()).hexdigest(),
                            'request_index':index}
                # Prime every workload shape with a distinct prompt; never measure JIT cold start.
                await run_load(lambda index: request(case['requests'] + index), warmup_count,
                               warmup_count, args.timeout)
                await engine.reset_prefix_cache()
                if args.profile:
                    await engine.start_profile(profile_prefix=case['name'])
                progress = {}
                try:
                    rows, elapsed = await run_load(request, case['requests'], case['concurrency'],
                                                   args.timeout, progress=progress)
                except BaseException as exc:
                    metadata['failed_case'] = {'workload':case, 'repetition':repetition,
                                                'progress':progress, 'error':repr(exc)}
                    raise
                if args.profile:
                    await engine.stop_profile()
                from measurement import Sample
                summary = summarize([Sample(**x['sample']) for x in rows], elapsed)
                if summary['cached_tokens']:
                    raise RuntimeError('cold-cache benchmark unexpectedly reused cached tokens')
                entry = {'workload':case, 'repetition':repetition, 'summary':summary, 'requests':rows,
                         'gpu_after_case':hardware()}
                metadata['results'].append(entry)
                metadata['status'] = 'running'
                save(args.output, metadata)
                print(json.dumps({'config':config['name'], 'case':case['name'], 'repetition':repetition,
                                  'summary':summary}), flush=True)
        metadata['status'] = 'complete'
    except BaseException as exc:
        metadata['status'] = 'failed'
        metadata['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        metadata['finished_at'] = datetime.now(timezone.utc).isoformat()
        save(args.output, metadata)
        engine.shutdown()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--workloads', type=Path, default=HERE / 'workloads.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repetitions', type=int, default=2)
    parser.add_argument('--timeout', type=float, default=600)
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--profiler-kind', choices=['torch','cuda'], default='torch')
    args = parser.parse_args()
    if args.repetitions < 1:
        parser.error('repetitions must be positive')
    if args.output.exists():
        parser.error('output already exists; choose a new path to preserve evidence')
    asyncio.run(run(args))


if __name__ == '__main__':
    main()
