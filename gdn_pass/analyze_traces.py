"""Index GDN captures; identify decode by actual reference or custom GPU kernels."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
from trace_summary import summarize_kernels


def analyze(directory):
    records=[]
    for path in sorted((directory/'traces').glob('*.pt.trace.json.gz')):
        events=json.loads(gzip.decompress(path.read_bytes()))['traceEvents']
        launches={e['args']['correlation'] for e in events if e.get('cat')=='cuda_runtime' and e['name']=='cudaGraphLaunch'}
        gdn=[e for e in events if e.get('cat')=='kernel' and
             ('gdn_decode_warp_kernel' in e['name'] or 'fused_recurrent_gated_delta_rule_packed_decode_kernel' in e['name'])]
        correlations=launches & {e['args']['correlation'] for e in gdn}
        decode=[e for e in events if e.get('cat')=='kernel' and e.get('args',{}).get('correlation') in correlations]
        records.append({'file':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'phase':'decode_with_prompt' if decode else 'prefill',
            'all_kernels':summarize_kernels({'traceEvents':events}),
            'steady_decode':summarize_kernels({'traceEvents':decode}) if decode else None,
            'decode_graph_correlations':sorted(correlations),
            'gdn_launch_metadata':[{key:e['args'].get(key) for key in
                ('registers per thread','shared memory','grid','block')} for e in gdn[:1]],
            'custom_gdn_calls':sum('gdn_decode_warp_kernel' in e['name'] for e in gdn),
            'reference_gdn_calls':sum('fused_recurrent_gated_delta_rule_packed_decode_kernel' in e['name'] for e in gdn)})
    assert len(records)==2 and {r['phase'] for r in records}=={'prefill','decode_with_prompt'}
    (directory/'gdn_trace_index.json').write_text(json.dumps(records,indent=2)+'\n')
    return records


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path)
    for r in analyze(p.parse_args().directory):print(r['phase'],r['custom_gdn_calls'],'custom,',r['reference_gdn_calls'],'reference')
