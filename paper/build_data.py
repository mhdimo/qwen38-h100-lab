"""Rebuild paper numbers directly from the three complete raw experiment files."""
import csv
import hashlib
import io
import json
from pathlib import Path
import statistics
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from report import aggregate

FILES = {
    'Base': ROOT / 'results/20261001T113011.178231Z/baseline.json',
    'Small': ROOT / 'results/20261001T102025.566538Z/batch64_t8192.json',
    'Large': ROOT / 'results/20261001T102025.566538Z/batch64_t16384.json',
}
CASE_NAMES = {'Single':'decode_c1', 'Prefill':'prefill_c32',
              'Decode':'decode_c32', 'Mixed':'mixed_c32'}


def main():
    raw = {name:json.loads(path.read_text()) for name,path in FILES.items()}
    data = {name:aggregate(value) for name,value in raw.items()}
    signatures = None
    for name,value in raw.items():
        actual = [(r['workload'],r['repetition']) for r in value['results']]
        if signatures is None:
            signatures = actual
        assert actual == signatures, f'mismatched workloads/repetitions: {name}'
        assert sum(x['completed'] for x in data[name].values()) == 400
        assert value['versions'] == raw['Base']['versions']
        assert value['engine_options']['model'] == raw['Base']['engine_options']['model']
    macros = {}
    def put(name, value, decimals=2):
        macros[name] = f'{value:,.{decimals}f}' if isinstance(value,(int,float)) else value
    for name,groups in data.items():
        for label,case in CASE_NAMES.items():
            stats=groups[case]
            put(name+label,stats['input_tokens_per_s'] if label=='Prefill' else stats['output_tokens_per_s'])
            for metric,heading in [('ttft_ms','TTFT'),('tpot_ms','TPOT')]:
                values=stats[metric]
                if values['p50'] is not None:
                    put(name+label+heading, f"{values['p50']:,.2f} / {values['p95']:,.2f}")
                    put(name+label+heading+'Median',values['p50'])
        spreads=[]
        for group in groups.values():
            rates=group['repetition_output_rates']
            spreads.append(100*(max(rates)-min(rates))/statistics.mean(rates))
        put(name+'Spread',max(spreads))
        memory=[]
        prefill_power=[]
        for entry in raw[name]['results']:
            gpu=next(csv.DictReader(io.StringIO(entry['gpu_after_case']),skipinitialspace=True))
            memory.append(float(gpu['memory.used [MiB]'].split()[0])/1024)
            if entry['workload']['name']=='prefill_c32':
                prefill_power.append(float(gpu['power.draw [W]'].split()[0]))
        put(name+'Memory',max(memory))
        put(name+'Power',f'{min(prefill_power):.1f}--{max(prefill_power):.1f}')
    ratio=lambda case,metric,a,b:data[a][case][metric]/data[b][case][metric]
    put('DecodeSpeedup',ratio('decode_c32','output_tokens_per_s','Large','Base'))
    put('MixedSpeedup',ratio('mixed_c32','output_tokens_per_s','Large','Base'))
    pr=ratio('prefill_c32','input_tokens_per_s','Large','Base')
    sr=ratio('decode_c1','output_tokens_per_s','Large','Base')
    put('PrefillRatio',pr)
    put('PrefillGain',100*(pr-1))
    put('SingleRatio',sr)
    put('SingleLoss',100*(1-sr))
    for label,case,metric in [('Prefill','prefill_c32','input_tokens_per_s'),
                              ('Mixed','mixed_c32','output_tokens_per_s'),
                              ('Decode','decode_c32','output_tokens_per_s')]:
        put('Budget'+label+('Change' if label=='Decode' else 'Gain'),
            100*(ratio(case,metric,'Large','Small')-1))
    (HERE/'metrics.tex').write_text('% Generated from raw results; do not hand edit.\n'+
        ''.join('\\newcommand{\\'+key+'}{'+value+'}\n' for key,value in macros.items()))
    with (HERE/'throughput.csv').open('w') as stream:
        writer=csv.writer(stream)
        writer.writerow(['config','decode','mixed','prefill'])
        for label,name in [('Base','Base'),('8k','Small'),('16k','Large')]:
            writer.writerow([label,data[name]['decode_c32']['output_tokens_per_s'],
                             data[name]['mixed_c32']['output_tokens_per_s'],
                             data[name]['prefill_c32']['input_tokens_per_s']])
    (HERE/'data.json').write_text(json.dumps(data,indent=2)+'\n')
    (HERE/'source-checksums.txt').write_text(''.join(
        hashlib.sha256(path.read_bytes()).hexdigest()+'  '+str(path.relative_to(ROOT))+'\n'
        for path in FILES.values()))
    print(json.dumps({key:macros[key] for key in ['DecodeSpeedup','MixedSpeedup',
          'PrefillGain','SingleLoss','BaseMemory','SmallMemory','LargeMemory']},indent=2))


if __name__=='__main__':
    main()
