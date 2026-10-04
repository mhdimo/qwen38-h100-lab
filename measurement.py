"""CPU-only measurement and bounded concurrent load generation."""
import asyncio
import statistics
import time
from dataclasses import dataclass


@dataclass(frozen=True)
class Sample:
    input_tokens: int
    output_tokens: int
    ttft_s: float
    latency_s: float
    cached_tokens: int


def distribution(values):
    values = sorted(values)
    if not values:
        return dict(mean=None, p50=None, p95=None, count=0)
    def percentile(q):
        index = (len(values) - 1) * q
        lo = int(index)
        hi = min(lo + 1, len(values) - 1)
        return values[lo] + (values[hi] - values[lo]) * (index - lo)
    return dict(mean=statistics.mean(values), p50=percentile(.5),
                p95=percentile(.95), count=len(values))


def summarize(rows, elapsed_s):
    if not rows or elapsed_s <= 0:
        raise ValueError('nonempty samples and positive wall time required')
    return {
        'completed': len(rows), 'elapsed_s': elapsed_s,
        'output_tokens_per_s': sum(x.output_tokens for x in rows) / elapsed_s,
        'input_tokens_per_s': sum(x.input_tokens for x in rows) / elapsed_s,
        'requests_per_s': len(rows) / elapsed_s,
        'cached_tokens': sum(x.cached_tokens for x in rows),
        'ttft_ms': distribution([x.ttft_s * 1000 for x in rows]),
        'latency_ms': distribution([x.latency_s * 1000 for x in rows]),
        'tpot_ms': distribution([(x.latency_s - x.ttft_s) * 1000 /
                                 (x.output_tokens - 1) for x in rows
                                 if x.output_tokens > 1]),
    }


async def consume(events, started, input_tokens, expected_output):
    first = None
    tokens = []
    finished = False
    cached = 0
    last = started
    async for timestamp, cumulative, final, cached_count in events:
        last = timestamp
        if cumulative and first is None:
            first = timestamp
        tokens = list(cumulative)
        finished = final
        cached = cached_count
    if first is None or not finished or len(tokens) != expected_output:
        raise RuntimeError(f'incomplete stream: {len(tokens)}/{expected_output}, finished={finished}')
    return Sample(input_tokens, len(tokens), first - started, last - started, cached), tokens


async def run_load(request, count, concurrency, timeout=600, progress=None):
    if count <= 0 or concurrency <= 0 or timeout <= 0:
        raise ValueError('count, concurrency, and timeout must be positive')
    # Worker pool keeps no unbounded task queue; TTFT starts at engine submission.
    indices = iter(range(count))
    values = []
    if progress is None:
        progress = {}
    progress.update(submitted=0, completed=values, errors=[], cancelled=0)
    async def worker():
        for index in indices:
            progress['submitted'] += 1
            try:
                values.append(await request(index))
            except asyncio.CancelledError:
                progress['cancelled'] += 1
                raise
            except Exception as exc:
                progress['errors'].append({'index':index, 'error':repr(exc)})
                raise
    started = time.perf_counter()
    async with asyncio.timeout(timeout):
        async with asyncio.TaskGroup() as group:
            for _ in range(min(count, concurrency)):
                group.create_task(worker())
    return values, time.perf_counter() - started
