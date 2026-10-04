"""Exercise tile/grid tails, slot masks and shared-memory pipelines under sanitizer."""
import argparse
import os
from pathlib import Path
import sys
from benchmark import check_idle_gpu,save

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    check_idle_gpu();os.environ['PATH']=str(Path(sys.executable).parent)+os.pathsep+os.environ.get('PATH','')
    import torch
    from gdn_pass.experiment import inputs
    from pipeline_pass.experiment import VARIANTS
    from pipeline_pass.kernel import decode,load
    load();calls=0
    for name,kw in VARIANTS.items():
        for batch in (1,7,13):
            for fp32 in (False,True):
                case=inputs(torch,batch,17,True,fp32)
                case['use_qk_l2norm_in_kernel']=fp32
                if fp32:case['ssm_state_indices']=case['ssm_state_indices'].to(torch.int32)
                if batch>1:case['ssm_state_indices'][1]=case['initial_state'].size(0)
                decode(**case,**kw);calls+=1;torch.cuda.synchronize()
        case=inputs(torch,13,21,True);graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):decode(**case,**kw)
        for _ in range(3):graph.replay();calls+=1
        torch.cuda.synchronize()
    save(args.output,{'status':'complete','calls':calls,'variants':list(VARIANTS),
                     'scope':'pipeline kernels only; inspect tool log and exit status'})

if __name__=='__main__':main()
