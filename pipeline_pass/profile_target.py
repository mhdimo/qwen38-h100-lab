"""Warmed, labelled kernel fixtures for Nsight; all timings are diagnostic only."""
import argparse
from functools import partial
import os
from pathlib import Path
import sys
from benchmark import check_idle_gpu,save

def main():
    p=argparse.ArgumentParser();p.add_argument('--kind',choices=['compute','systems'],required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    check_idle_gpu()
    os.environ['PATH']=str(Path(sys.executable).parent)+os.pathsep+os.environ.get('PATH','')
    import torch
    from pipeline_pass.experiment import implementations
    from gdn_pass.experiment import inputs
    from custom_kernels.experiment import allocate
    from custom_kernels.silu_mul_fp8 import load
    import vllm._custom_ops
    ops=implementations();cases=[]
    names=['stock','previous','direct_t1_c0','direct_t4_c4','async_t4_c4','direct_t8_c4','async_t8_c4']
    for batch in (1,32,128):
        for name in names:
            case=inputs(torch,batch,23)
            case['ssm_state_indices'].copy_(torch.arange(1,batch+1,device='cuda'))
            cases.append((f'gdn/{name}/B{batch}',partial(ops[name],**case)))
    silu={'stock':torch.ops._C.silu_and_mul_per_block_quant.default,'previous':load()}
    for m in (32,16384):
        x=torch.randn(m,34816,device='cuda',dtype=torch.bfloat16)
        for name,op in silu.items():
            y,s=allocate(torch,x,True)
            cases.append((f'silu/{name}/M{m}',partial(op,y,x,s,128,None,True)))
    for _,op in cases:
        for _ in range(3):op()
    torch.cuda.synchronize()
    graphs={}
    if args.kind=='systems':
        for name,op in cases:
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):op()
            for _ in range(3):graph.replay()
            graphs[name]=graph
        torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStart()
    for name,op in cases:
        for mode in (['eager'] if args.kind=='compute' else ['eager','graph']):
            torch.cuda.nvtx.range_push(name+'/'+mode)
            for _ in range(1 if args.kind=='compute' else 20):
                (op if mode=='eager' else graphs[name].replay)()
            torch.cuda.synchronize();torch.cuda.nvtx.range_pop()
    torch.cuda.cudart().cudaProfilerStop()
    save(args.output,{'status':'complete','profiled':True,'cases':[name for name,_ in cases],
                     'repeats':1 if args.kind=='compute' else 20,'cache_regime':'NCU default cold-cache replay; Systems same-buffer warm replay',
                     'purpose':'hardware diagnostics only; not throughput evidence'})

if __name__=='__main__':main()
