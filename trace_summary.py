"""Summed GPU kernel durations from PyTorch Chrome traces, not wall-clock shares."""
import argparse
from collections import defaultdict
import gzip
import json
from pathlib import Path


def summarize_kernels(trace):
    timings = defaultdict(list)
    for event in trace['traceEvents']:
        if event.get('cat') == 'kernel' and event.get('ph') == 'X':
            timings[event['name']].append(float(event['dur']))
    total = sum(sum(values) for values in timings.values())
    if total <= 0:
        raise ValueError('no timed GPU kernels in trace')
    kernels = [{'name':name, 'count':len(values), 'total_ms':sum(values)/1000,
                'mean_us':sum(values)/len(values), 'percent_kernel_time':100*sum(values)/total}
               for name,values in timings.items()]
    kernels.sort(key=lambda row:row['total_ms'], reverse=True)
    return {'total_kernel_ms':total/1000, 'kernels':kernels,
            'interpretation':'Sum of kernel durations, not exclusive wall time; concurrent kernels may overlap.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trace', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    opener = gzip.open if args.trace.suffix == '.gz' else open
    with opener(args.trace, 'rt') as stream:
        result = summarize_kernels(json.load(stream))
    result['source'] = str(args.trace)
    args.output.write_text(json.dumps(result,indent=2) + '\n')
    for row in result['kernels'][:15]:
        print(f"{row['percent_kernel_time']:6.2f}% {row['total_ms']:10.2f} ms {row['count']:7} {row['name'][:200]}")


if __name__ == '__main__':
    main()
