"""Bounded profiler/sanitizer children; launched only by the locked shared suite."""
import argparse
import json
import os
import signal
from pathlib import Path
import subprocess
import sys
from benchmark import check_idle_gpu,save,hardware
from pipeline_pass.process import run_child

ROOT=Path(__file__).resolve().parents[1]
NCU='/opt/nvidia/nsight-compute/2026.1.1/ncu'

def main():
    p=argparse.ArgumentParser();p.add_argument('--action',choices=['profile','sanitize'],required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    check_idle_gpu();directory=args.output.parent
    os.environ['PATH']=str(Path(sys.executable).parent)+os.pathsep+os.environ.get('PATH','')
    data={'status':'starting','profiled':True,'gpu_before':hardware(),'commands':[]}
    save(args.output,data)
    def interrupted(signum,frame):raise InterruptedError(f'diagnostic received signal {signum}')
    signal.signal(signal.SIGTERM,interrupted)
    signal.signal(signal.SIGINT,interrupted)
    def run(name,command,required=True):
        entry={'name':name,'command':command,'status':'starting'}
        data['commands'].append(entry);save(args.output,data)
        try:
            with (directory/(name+'.log')).open('w') as log:
                code=run_child(command,log,timeout=1800,privileged=command[0]=='sudo')
            entry.update(exit_code=code,status='complete' if code==0 else 'failed')
            if required and code:raise RuntimeError(f'{name} failed: {code}')
            return code
        except BaseException as exc:entry.update(status='failed',error=repr(exc));raise
        finally:save(args.output,data);print(json.dumps(entry),flush=True)
    try:
        if args.action=='sanitize':
            for tool in ('memcheck','racecheck'):
                run(tool,['/usr/local/cuda/bin/compute-sanitizer','--tool',tool,'--error-exitcode','99',
                    '--kernel-name','kns=gdn_pipeline_kernel',sys.executable,'-m','pipeline_pass.sanitize',
                    '--output',str(directory/(tool+'.json'))])
        else:
            stage=os.environ.get('PIPELINE_DIAGNOSTIC','compute')
            if stage=='compute':
                target=[sys.executable,'-m','pipeline_pass.profile_target','--kind','compute','--output',str(directory/'compute_target.json')]
                profiler=[NCU,'--config-file','off','--section-folder','/opt/nvidia/nsight-compute/2026.1.1/sections',
                    '--profile-from-start','off','--target-processes','all','--nvtx',
                    '--section','SpeedOfLight','--section','MemoryWorkloadAnalysis','--section','Occupancy',
                    '--section','LaunchStats','--section','WarpStateStats','--section','SourceCounters',
                    '--export',str(directory/'compute')]
                # Counter access belongs to the CUDA target too; both target and
                # profiler run elevated, with their caches confined to the lab.
                env=[f'{k}={v}' for k,v in os.environ.items() if k in ('PATH','TMPDIR','VLLM_CACHE_ROOT',
                     'TRITON_CACHE_DIR','TORCHINDUCTOR_CACHE_DIR','CUDA_CACHE_PATH')]
                env += [f'XDG_CACHE_HOME={ROOT}/pipeline_pass/build/xdg', f'XDG_CONFIG_HOME={ROOT}/pipeline_pass/build/config',
                        f'FLASHINFER_WORKSPACE_BASE={ROOT}/pipeline_pass/build/flashinfer', f'TORCH_HOME={ROOT}/pipeline_pass/build/torch']
                report=directory/'compute.ncu-rep'
                try:
                    run('compute',['sudo','-n','env',*env,*profiler,*target])
                finally:
                    # Includes failed/partial outputs, so preservation and cleanup
                    # remain possible even when collection fails.
                    run('report_owner',['sudo','-n','chown','-R',f'{os.getuid()}:{os.getgid()}',str(directory),str(ROOT/'pipeline_pass/build'),str(ROOT/'gdn_pass/build'),str(ROOT/'custom_kernels/build')])
                run('compute_csv',[NCU,'--import',str(report),'--csv','--page','raw'])
                run('compute_details',[NCU,'--import',str(report),'--page','details'])
            elif stage in ('systems','engine'):
                prefix=directory/stage
                target=[sys.executable,'-m','pipeline_pass.profile_target','--kind','systems','--output',str(directory/'systems_target.json')]
                if stage=='engine':
                    target=[sys.executable,str(ROOT/'benchmark.py'),'--config',str(ROOT/'configs/batch64_t16384.json'),
                        '--workloads',str(ROOT/'workloads_profile.json'),'--output',str(directory/'engine.json'),
                        '--repetitions','1','--profile','--profiler-kind','cuda']
                run(stage,['nsys','profile','--trace=cuda,nvtx,osrt','--sample=none','--cpuctxsw=none',
                    '--capture-range=cudaProfilerApi','--capture-range-end=repeat','--cuda-graph-trace=node',
                    '--output='+str(prefix),*target])
                reports=sorted(directory.glob(stage+'*.nsys-rep'))
                if not reports:raise RuntimeError('Nsight Systems produced no reports')
                for report in reports:
                    run(report.stem+'_stats',['nsys','stats','--report','cuda_gpu_kern_sum,cuda_api_sum,nvtx_sum,cuda_gpu_trace',
                        '--format','csv','--output',str(directory/(report.stem+'_stats')),str(report)])
            else:raise ValueError('unknown diagnostic stage')
        data['status']='complete'
    except BaseException as exc:data['status']='failed';data['error']=repr(exc);raise
    finally:data['gpu_after']=hardware();save(args.output,data)

if __name__=='__main__':main()
