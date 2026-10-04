"""Generate GDN report tables from complete, matched H100 evidence."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
sys.path.insert(0,str(ROOT))
from report import aggregate


def main():
    if not __debug__:raise RuntimeError('Report validation requires Python assertions; do not use -O')
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--engine-run',type=Path,required=True)
    p.add_argument('--micro',type=Path,default=ROOT/'results/20261001T124251.173582Z/kernel_gdn-bench.json')
    p.add_argument('--profile',type=Path,default=ROOT/'results/20261001T124418.339742Z/gdn_trace_index.json')
    args=p.parse_args()
    paths={'baseline':args.engine_run/'batch64_t16384.json',
        'candidate':args.engine_run/'batch64_t16384_gdn.json',
        'numerical':ROOT/'results/20261001T124343.661727Z/kernel_gdn-validate.json',
        'micro':args.micro,'profile':args.profile,
        'baseline_profile':ROOT/'results/20261001T115943.440807Z/trace_index.json',
        'sanitizer':ROOT/'results/20261001T125119.656720Z/kernel_gdn-sanitize.json',
        'sanitizer_manifest':ROOT/'results/20261001T125119.656720Z/manifest.json',
        'baseline_repeat':ROOT/'results/20261001T121440.767119Z/batch64_t16384.json'}
    raw={k:json.loads(v.read_text()) for k,v in paths.items()}
    base,custom=raw['baseline'],raw['candidate']
    assert base['versions']==custom['versions']
    assert [r['workload'] for r in base['results']]==[r['workload'] for r in custom['results']]
    assert [r['repetition'] for r in base['results']]==[r['repetition'] for r in custom['results']]
    bopts,copts=json.loads(json.dumps(base['engine_options'])),json.loads(json.dumps(custom['engine_options']))
    assert copts.pop('worker_cls')=='gdn_pass.worker.GDNWorker'
    copts['compilation_config'].pop('cache_dir')
    assert bopts==copts,'unexpected engine differences'
    before,after=aggregate(base),aggregate(custom)
    assert sum(v['completed'] for v in before.values())==sum(v['completed'] for v in after.values())==400
    numerical=raw['numerical']
    assert numerical['status']=='complete' and all(v['passed'] for v in numerical['results'].values())
    assert raw['micro']['status']=='complete'
    assert raw['sanitizer']['status']=='complete' and raw['sanitizer_manifest'][0]['exit_code']==0
    sanitizer_log=paths['sanitizer'].with_suffix('.log')
    assert 'ERROR SUMMARY: 0 errors' in sanitizer_log.read_text()
    paths['sanitizer_log']=sanitizer_log
    profile=raw['profile'];decode=next(r for r in profile if r['phase']=='decode_with_prompt')
    assert decode['custom_gdn_calls']>0 and decode['reference_gdn_calls']==0
    selected=numerical['results']['rows_4']
    macro_lines=[]
    def macro(name,value):macro_lines.append('\\newcommand{\\'+name+'}{'+str(value)+'}')
    micro=raw['micro']['results']
    macro('MicroRows','\n'.join(f"{r['batch']} & {r['median_us']['baseline']:.2f} & {r['median_us']['custom_r4']:.2f} & {r['speedup']['custom_r4']:.3f}$\\times$ \\\\" for r in micro))
    macro('NumericalCases',len(selected['cases']))
    macro('NumericalEdges',len(selected['edge_cases']))
    macro('StateError',f"{selected['trajectory_max_state_abs']:.3g}")
    macro('OutputError',f"{selected['trajectory_max_output_abs']:.3g}")
    macro('ProfileCalls',decode['custom_gdn_calls'])
    macro('ProfileGraphs',len(decode['decode_graph_correlations']))
    profile_kernel=next(k for k in decode['steady_decode']['kernels'] if 'gdn_decode_warp_kernel' in k['name'])
    macro('ProfileMean',f"{profile_kernel['mean_us']:.2f}")
    engine_rows=[];latency_rows=[];gains={}
    fmt=lambda x:'---' if x is None else f'{x:.2f}'
    for key,label in [('decode_c1','Single / output tok/s'),('prefill_c32','Prefill / input tok/s'),
                      ('decode_c32','Decode / output tok/s'),('mixed_c32','Mixed / output tok/s')]:
        a,b=before[key],after[key]
        metric='input_tokens_per_s' if key=='prefill_c32' else 'output_tokens_per_s'
        gain=100*(b[metric]/a[metric]-1);gains[key]=gain
        engine_rows.append(f"{label} & {a[metric]:.2f} & {b[metric]:.2f} & {gain:+.2f}\\% \\\\")
        latency_rows.append(f"{label.split(' /')[0]} & {fmt(a['ttft_ms']['p95'])} & {fmt(b['ttft_ms']['p95'])} & {fmt(a['tpot_ms']['p95'])} & {fmt(b['tpot_ms']['p95'])} \\\\")
    macro('EngineRows','\n'.join(engine_rows));macro('LatencyRows','\n'.join(latency_rows))
    macro('DecodeGain',f"{gains['decode_c32']:+.2f}");macro('MixedGain',f"{gains['mixed_c32']:+.2f}")
    def agreement(a,b):
        assert [r['workload'] for r in a['results']]==[r['workload'] for r in b['results']]
        equal=total=0
        for x,y in zip(a['results'],b['results']):
            xx={r['request_index']:r['token_sha256'] for r in x['requests']}
            yy={r['request_index']:r['token_sha256'] for r in y['requests']}
            assert xx.keys()==yy.keys()
            equal+=sum(xx[i]==yy[i] for i in xx);total+=len(xx)
        return {'equal':equal,'total':total}
    token=agreement(base,custom)
    repeat=raw['baseline_repeat'];assert repeat['engine_options']==base['engine_options'] and repeat['versions']==base['versions']
    aggregate(repeat)
    control=agreement(repeat,base)
    macro('TokenAgree',token['equal']);macro('ControlAgree',control['equal'])
    def peak_sample(data):
        values=[]
        for entry in data['results']:
            row=list(csv.DictReader(entry['gpu_after_case'].splitlines(),skipinitialspace=True))[0]
            values.append(float(row['memory.used [MiB]'].split()[0])/1024)
        return max(values)
    memory={'baseline_gib':peak_sample(base),'candidate_gib':peak_sample(custom)}
    macro('MemoryBase',f"{memory['baseline_gib']:.2f}");macro('MemoryCustom',f"{memory['candidate_gib']:.2f}")
    summary={'sources':{k:{'path':str(v.resolve()),'sha256':hashlib.sha256(v.read_bytes()).hexdigest()} for k,v in paths.items()},
             'baseline':before,'candidate':after,'gains_percent':gains,'token_agreement':token,
             'baseline_repeat_agreement':control,'memory_point_samples':memory,
             'numerical':numerical,'micro':micro,'profile':profile}
    (HERE/'gdn_data.json').write_text(json.dumps(summary,indent=2)+'\n')
    (HERE/'gdn_metrics.tex').write_text('\n'.join(macro_lines)+'\n')
    print(json.dumps({'gains_percent':gains,'token_agreement':token,'baseline_control':control},indent=2))


if __name__=='__main__':main()
