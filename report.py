"""Aggregate only completed, unprofiled cold-cache runs."""
import argparse
import json
from pathlib import Path

from measurement import Sample, summarize


def aggregate(data):
    if data['status'] != 'complete' or data['profiled']:
        raise ValueError('only complete unprofiled runs are comparable')
    if not data['quality_smoke'] or not all(x['passed'] for x in data['quality_smoke']):
        raise ValueError('quality smoke failed')
    groups = {}
    for entry in data['results']:
        case, summary = entry['workload'], entry['summary']
        if summary['cached_tokens'] or summary['completed'] != case['requests'] or len(entry['requests']) != case['requests']:
            raise ValueError('partial or cache-contaminated result')
        group = groups.setdefault(case['name'], {'rows':[], 'seconds':0, 'rates':[]})
        group['rows'].extend(Sample(**x['sample']) for x in entry['requests'])
        group['seconds'] += summary['elapsed_s']
        group['rates'].append(sum(x['sample']['output_tokens'] for x in entry['requests']) / summary['elapsed_s'])
    if not groups:
        raise ValueError('no measured cases')
    return {name: {**summarize(group['rows'], group['seconds']),
                   'repetition_output_rates':group['rates']} for name,group in groups.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('files', nargs='+', type=Path)
    parser.add_argument('--output', type=Path, default=Path('comparison.json'))
    args = parser.parse_args()
    comparison = {}
    print('| Configuration | Workload | Output tok/s | Input tok/s | TTFT p50/p95 ms | TPOT p50/p95 ms | N |')
    print('|---|---|---:|---:|---:|---:|---:|')
    def fmt(value):
        return '—' if value is None else f'{value:.2f}'
    for path in args.files:
        data = json.loads(path.read_text())
        try:
            values = aggregate(data)
        except ValueError as exc:
            print(f'Excluded {path}: {exc}')
            continue
        name = data['config']['name']
        key = f'{path.parent.name}/{name}'
        comparison[key] = {'path':str(path), 'workloads':values}
        for case, result in values.items():
            ttft, tpot = result['ttft_ms'], result['tpot_ms']
            print(f"| {name} | {case} | {result['output_tokens_per_s']:.2f} | {result['input_tokens_per_s']:.2f} | {fmt(ttft['p50'])}/{fmt(ttft['p95'])} | {fmt(tpot['p50'])}/{fmt(tpot['p95'])} | {result['completed']} |")
    args.output.write_text(json.dumps(comparison, indent=2) + '\n')


if __name__ == '__main__':
    main()
