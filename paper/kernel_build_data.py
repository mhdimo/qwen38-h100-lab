"""Generate kernel-paper numbers from validated raw evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
sys.path.insert(0,str(ROOT))
from report import aggregate


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--engine-run',type=Path,required=True)
    args=p.parse_args()
    paths={
        'reference':args.engine_run/'batch64_t16384.json',
        'candidate':args.engine_run/'batch64_t16384_custom_silu.json',
        'reference_repeat':ROOT/'results/20261001T102025.566538Z/batch64_t16384.json',
        'numerical':ROOT/'results/20261001T121017.707611Z/kernel_validate.json',
        'micro':ROOT/'results/20261001T121049.464234Z/kernel_bench.json',
        'profile':ROOT/'results/20261001T121125.290923Z/trace_index.json',
    }
    raw={name:json.loads(path.read_text()) for name,path in paths.items()}
    base,custom=raw['reference'],raw['candidate']
    assert base['versions']==custom['versions']
    assert base['engine_options']['model']==custom['engine_options']['model']
    assert [r['workload'] for r in base['results']]==[r['workload'] for r in custom['results']]
    assert [r['repetition'] for r in base['results']]==[r['repetition'] for r in custom['results']]
    bopts,copts=json.loads(json.dumps(base['engine_options'])),json.loads(json.dumps(custom['engine_options']))
    copts.pop('worker_cls')
    copts['compilation_config'].pop('cache_dir')
    assert bopts==copts,'unexpected engine differences beyond worker and cache path'
    assert raw['numerical']['status']=='complete' and raw['numerical']['results']['passed']
    assert raw['micro']['status']=='complete'
    assert all(r['custom_kernel_launches']>0 for r in raw['profile'])
    before,after=aggregate(base),aggregate(custom)
    assert sum(r['completed'] for r in before.values())==400
    assert sum(r['completed'] for r in after.values())==400
    agree,total=0,0
    for a,b in zip(base['results'],custom['results']):
        aa={r['request_index']:r['token_sha256'] for r in a['requests']}
        bb={r['request_index']:r['token_sha256'] for r in b['requests']}
        assert aa.keys()==bb.keys()
        agree+=sum(aa[k]==bb[k] for k in aa);total+=len(aa)
    repeated=raw['reference_repeat']
    assert repeated['status']=='complete' and not repeated['profiled']
    assert repeated['versions']==base['versions'] and repeated['engine_options']==base['engine_options']
    assert [r['workload'] for r in repeated['results']]==[r['workload'] for r in base['results']]
    repeat_agree=0
    for a,b in zip(repeated['results'],base['results']):
        aa={r['request_index']:r['token_sha256'] for r in a['requests']}
        bb={r['request_index']:r['token_sha256'] for r in b['requests']}
        repeat_agree+=sum(aa[k]==bb[k] for k in aa)
    numbers=[]
    def macro(name,value):numbers.append('\\newcommand{\\'+name+'}{'+str(value)+'}')
    micro=raw['micro']['results']
    largest=next(r for r in micro if r['m']==16384)
    macro('MicroBase',f"{largest['median_us']['baseline']:.2f}")
    macro('MicroCustom',f"{largest['median_us']['custom']:.2f}")
    macro('MicroSpeedup',f"{largest['speedup']:.2f}")
    def change(case):
        return 100*(after[case]['output_tokens_per_s']/before[case]['output_tokens_per_s']-1)
    macro('PrefillKernelGain',f"{change('prefill_c32'):+.2f}")
    macro('DecodeKernelGain',f"{change('decode_c32'):+.2f}")
    macro('NumericalCases',len(raw['numerical']['results']['cases']))
    macro('CustomProfileCalls',sum(r['custom_kernel_launches'] for r in raw['profile']))
    macro('MicroRows','\n'.join(f"{r['m']:,} & {r['median_us']['baseline']:.2f} & "
        f"{r['median_us']['custom']:.2f} & {r['speedup']:.2f}$\\times$ \\\\" for r in micro))
    engine_rows=[];latency_rows=[]
    cases=[('decode_c1','Single / output tok/s'),('prefill_c32','Prefill / input tok/s'),
           ('decode_c32','Decode / output tok/s'),('mixed_c32','Mixed / output tok/s')]
    fmt=lambda x:'---' if x is None else f'{x:.2f}'
    for name,label in cases:
        metric='input_tokens_per_s' if name=='prefill_c32' else 'output_tokens_per_s'
        a,b=before[name],after[name]
        engine_rows.append(f"{label} & {a[metric]:.2f} & {b[metric]:.2f} & {change(name):+.2f}\\% \\\\")
        latency_rows.append(f"{label.split(' /')[0]} & {fmt(a['ttft_ms']['p95'])} & {fmt(b['ttft_ms']['p95'])} & "
                            f"{fmt(a['tpot_ms']['p95'])} & {fmt(b['tpot_ms']['p95'])} \\\\")
    macro('EngineRows','\n'.join(engine_rows));macro('LatencyRows','\n'.join(latency_rows))
    macro('TokenComparison',f'{agree} of {total} matched measured requests have identical generated-token hashes. '
          f'Two unchanged reference runs agree on {repeat_agree}/{total} requests. '
          'Thus the full engine is not bitwise deterministic in these runs. This control does not '
          'prove all differences are harmless or caused by scheduling; representative quality evaluation remains necessary. '
          'Both configurations pass the same three text smoke checks with identical answers.')
    (HERE/'kernel_metrics.tex').write_text('\n'.join(numbers)+'\n')
    summary={'sources':{k:{'path':str(v.resolve()),'sha256':hashlib.sha256(v.read_bytes()).hexdigest()} for k,v in paths.items()},
             'reference':before,'custom':after,'token_hash_agreement':{'equal':agree,'total':total},
             'unchanged_reference_agreement':{'equal':repeat_agree,'total':total},
             'microbench':micro,'numerical':raw['numerical'],'profile':raw['profile'],
             'current_source_sha256':{str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest()
                                     for f in (ROOT/'custom_kernels').iterdir() if f.suffix in ('.py','.cu')}}
    (HERE/'kernel_data.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({'gains_percent':{k:change(k) for k in before},'token_agreement':[agree,total]},indent=2))


if __name__=='__main__':main()
