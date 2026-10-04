"""Sequential experiments with one cooperating lab lock and process timeout."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent


def terminate_group(process, grace=30):
    """Bound lifetime of all our workers, even if their parent exits first."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        process.wait()
        return
    deadline = time.monotonic() + grace
    while time.monotonic() < deadline:
        process.poll()  # Reap parent but still check the independent process group.
        try:
            os.killpg(process.pid, 0)
        except ProcessLookupError:
            process.wait()
            return
        time.sleep(min(.1, max(0, deadline - time.monotonic())))
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--configs', nargs='+', default=['baseline', 'batch64_t2048', 'batch64_t8192', 'batch64_t16384'])
    parser.add_argument('--workloads', type=Path, default=HERE / 'workloads.json')
    parser.add_argument('--repetitions', type=int, default=2)
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--kernel-action', choices=['validate', 'bench', 'gdn-validate', 'gdn-bench', 'gdn-sanitize', 'pipeline-validate', 'pipeline-bench', 'pipeline-profile', 'pipeline-sanitize'])
    args = parser.parse_args()
    if args.kernel_action and args.profile:
        parser.error('--profile applies to engine workloads, not --kernel-action')
    for name in args.configs:
        if not (HERE / 'configs' / f'{name}.json').is_file() or Path(name).name != name:
            parser.error(f'unknown configuration {name}')
    if args.repetitions < 1:
        parser.error('repetitions must be positive')
    with (HERE / '.gpu-lab.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
        directory = HERE / 'results' / run_id
        directory.mkdir(parents=True)
        manifest = []
        for name in ([f'kernel_{args.kernel_action}'] if args.kernel_action else args.configs):
            command = [sys.executable, str(HERE / 'benchmark.py'), '--config', str(HERE / 'configs' / f'{name}.json'),
                       '--workloads', str(args.workloads), '--output', str(directory / f'{name}.json'),
                       '--repetitions', str(args.repetitions)]
            if args.kernel_action:
                module = 'gdn_pass.experiment' if args.kernel_action.startswith('gdn-') else 'custom_kernels.experiment'
                action = args.kernel_action.removeprefix('gdn-')
                command = [sys.executable, '-m', module,
                           '--action', action, '--output', str(directory / f'{name}.json')]
                if args.kernel_action=='gdn-sanitize':
                    command=['/usr/local/cuda/bin/compute-sanitizer','--tool','memcheck',
                             '--error-exitcode','99','--kernel-name','kns=gdn_decode_warp_kernel',
                             sys.executable,'-m','gdn_pass.sanitize','--output',str(directory/f'{name}.json')]
                if args.kernel_action.startswith('pipeline-'):
                    action=args.kernel_action.removeprefix('pipeline-')
                    module='pipeline_pass.experiment' if action in ('validate','bench') else 'pipeline_pass.diagnostics'
                    command=[sys.executable,'-m',module,'--action',action,'--output',str(directory/f'{name}.json')]
            if args.profile:
                command.append('--profile')
            print(f'START {name} directory={directory}', flush=True)
            with (directory / f'{name}.log').open('w') as log:
                process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                           start_new_session=True, cwd=HERE)
                try:
                    exit_code = process.wait(timeout=2400)
                except BaseException as exc:
                    terminate_group(process)
                    manifest.append({'config':name, 'exit_code':process.returncode,
                                     'error':repr(exc), 'command':command})
                    (directory / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
                    raise
            manifest.append({'config':name, 'exit_code':exit_code, 'command':command})
            (directory / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
            print(f'END {name} exit={exit_code}', flush=True)
            # Process termination reclaims GPU contexts; verify idleness in next benchmark.
            time.sleep(3)
        if any(x['exit_code'] for x in manifest):
            raise SystemExit(1)


if __name__ == '__main__':
    main()
