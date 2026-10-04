"""Index engine captures and isolate recurrent decode CUDA graph launches."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
from trace_summary import summarize_kernels


def decode_events(events):
    launches = {e['args']['correlation'] for e in events
                if e.get('cat') == 'cuda_runtime' and e['name'] == 'cudaGraphLaunch'}
    recurrent = {e['args']['correlation'] for e in events if e.get('cat') == 'kernel'
                 and 'fused_recurrent_gated_delta_rule_packed_decode_kernel' in e['name']}
    selected = launches & recurrent
    return [e for e in events if e.get('cat') == 'kernel'
            and e.get('args', {}).get('correlation') in selected], sorted(selected)


def analyze(directory):
    records=[]
    for source in sorted((directory/'traces').glob('*.pt.trace.json.gz')):
        trace=json.loads(gzip.decompress(source.read_bytes()))
        events=trace['traceEvents']
        summary=summarize_kernels(trace)
        decode,correlations=decode_events(events)
        phase='decode_with_prompt' if decode else 'prefill'
        record={'file':str(source),'sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
                'phase':phase,'all_kernels':summary}
        if decode:
            record['steady_decode']=summarize_kernels({'traceEvents':decode})
            record['decode_graph_launch_correlations']=correlations
        record['custom_kernel_launches']=sum(e.get('cat')=='kernel' and
            'silu_mul_fp8_warp_kernel' in e['name'] for e in events)
        records.append(record)
    if len(records)!=2 or {r['phase'] for r in records}!={'prefill','decode_with_prompt'}:
        raise ValueError('expected one prefill and one decode capture; inspect workload/logs')
    output=directory/'trace_index.json'
    output.write_text(json.dumps(records,indent=2)+'\n')
    for record in records:
        print(record['phase'],record['custom_kernel_launches'],'custom launches',record['file'])
    return records


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory',type=Path)
    analyze(p.parse_args().directory)
