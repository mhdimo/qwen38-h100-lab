"""Small memory-safety matrix; run under Compute Sanitizer via the shared suite."""
import argparse
import hashlib
import os
from pathlib import Path
import sys
from benchmark import check_idle_gpu,save


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():p.error('output exists')
    check_idle_gpu()
    os.environ['PATH']=str(Path(sys.executable).parent)+os.pathsep+os.environ.get('PATH','')
    import torch
    from gdn_pass.experiment import inputs
    from gdn_pass.kernel import decode,load
    load();calls=0
    for batch in (1,7):
        for rows in (4,8,16):
            for fp32 in (False,True):
                for int32 in (False,True):
                    for norm in (False,True):
                        case=inputs(torch,batch,17,True,fp32)
                        case['use_qk_l2norm_in_kernel']=norm
                        if int32:case['ssm_state_indices']=case['ssm_state_indices'].to(torch.int32)
                        # Exercise the explicit upper-bound guard too (not a reference-equivalence case).
                        if batch>1:case['ssm_state_indices'][1]=case['initial_state'].size(0)
                        decode(**case,rows_per_warp=rows);calls+=1
                        torch.cuda.synchronize()
    graph=torch.cuda.CUDAGraph()
    case=inputs(torch,7,21,True)
    with torch.cuda.graph(graph):decode(**case)
    for _ in range(8):graph.replay();calls+=1
    torch.cuda.synchronize()
    save(args.output,{'status':'complete','invocations':calls,'tool':'Compute Sanitizer memcheck',
        'scope':'custom GDN kernels only; final error count and exit status are in the log/manifest',
        'source_sha256':hashlib.sha256((Path(__file__).parent/'gdn_decode.cu').read_bytes()).hexdigest()})


if __name__=='__main__':main()
