"""Independent numerical contract and matched graph timing for new GDN variants."""
import argparse
from datetime import datetime, timezone
from functools import partial
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import statistics
import sys
from benchmark import check_idle_gpu, hardware, save

HERE=Path(__file__).resolve().parent
VARIANTS={f'{kind}_t{tiles}_c{cap}':dict(tiles=tiles,cap=cap,async_copy=kind=='async')
          for kind in ('direct','async') for tiles in (1,2,4,8) for cap in (0,4)}

def implementations():
    from pipeline_pass.kernel import decode
    from gdn_pass.kernel import decode as previous
    from vllm.third_party.flash_linear_attention.ops.fused_recurrent import fused_recurrent_gated_delta_rule_packed_decode as stock
    return {'stock':stock,'previous':previous,**{name:partial(decode,**kw) for name,kw in VARIANTS.items()}}

def validate(torch,ops):
    from gdn_pass.experiment import validate as check
    results={}
    for name in VARIANTS:
        results[name]=check(torch,ops[name],ops['stock'])
        print(json.dumps({'variant':name,'passed':True}),flush=True)
    return results

def bench(torch,ops):
    from gdn_pass.experiment import inputs
    rows=[]
    for batch in (1,2,8,16,32,64,128):
        pool=[inputs(torch,batch,seed) for seed in range(8)]
        for case in pool:case['ssm_state_indices'].copy_(torch.arange(1,batch+1,device='cuda'))
        graphs={}
        for name,op in ops.items():
            for case in pool:op(**case)
            torch.cuda.synchronize();graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for case in pool:op(**case)
            graphs[name]=graph
        samples={name:[] for name in ops}
        for trial in range(10):
            names=list(ops)
            # Rotate as well as reverse: no variant always occupies an end position.
            names=names[trial%len(names):]+names[:trial%len(names)]
            for name in (names if trial%2==0 else names[::-1]):
                graph=graphs[name]
                for _ in range(3):graph.replay()
                start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(30):graph.replay()
                end.record();end.synchronize()
                samples[name].append(start.elapsed_time(end)*1000/240)
        med={k:statistics.median(v) for k,v in samples.items()}
        row={'batch':batch,'active_state_bytes':8*batch*48*128*128*4,
             'median_us':med,'samples_us':samples,
             'speedup_vs_stock':{k:med['stock']/v for k,v in med.items()}}
        rows.append(row);print(json.dumps(row),flush=True)
        del pool,graphs,graph,case
    return rows

def main():
    p=argparse.ArgumentParser();p.add_argument('--action',choices=['validate','bench'],required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if args.output.exists():p.error('output exists')
    check_idle_gpu()
    os.environ['PATH']=str(Path(sys.executable).parent)+os.pathsep+os.environ.get('PATH','')
    data={'status':'starting','action':args.action,'started_at':datetime.now(timezone.utc).isoformat(),'gpu_before':hardware()}
    save(args.output,data)
    try:
        if not (HERE/'gdn_pipeline.cu').is_file():raise AssertionError('Persistent/async CUDA kernel not implemented')
        import torch
        ops=implementations()
        data['source_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.iterdir() if p.suffix in ('.py','.cu')}
        data['versions']={n:importlib.metadata.version(n) for n in ('torch','vllm','triton')}
        data['results']=(validate if args.action=='validate' else bench)(torch,ops)
        data['status']='complete'
    except BaseException as exc:data['status']='failed';data['error']=repr(exc);raise
    finally:
        data['gpu_after']=hardware();data['finished_at']=datetime.now(timezone.utc).isoformat();save(args.output,data)

if __name__=='__main__':main()
