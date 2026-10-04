"""GDN numerical oracle, recurrent-state tests and CUDA-graph microbenchmarks."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
from benchmark import check_idle_gpu, hardware, save

HERE=Path(__file__).resolve().parent


def inputs(torch,batch,seed=0,padded=False,bias_fp32=False):
    torch.manual_seed(seed)
    device='cuda';dtype=torch.bfloat16
    def tensor(rows,cols):
        return torch.randn(rows,cols+(16 if padded else 0),device=device,dtype=dtype)[:,:cols]
    x=tensor(batch,10240);a=tensor(batch,48);b=tensor(batch,48)
    alog=torch.linspace(-3,2,48,device=device)
    bias=torch.linspace(-5,3,48,device=device).to(torch.float32 if bias_fp32 else dtype)
    state=torch.randn(batch+3,48*128*128+(128 if padded else 0),device=device)*.2
    state=state[:,:48*128*128].view(batch+3,48,128,128)
    idx=torch.randperm(batch,device=device,dtype=torch.int64)+1
    if batch>=3:idx[0]=0;idx[-1]=-1
    if padded:
        storage=torch.zeros(batch*2,device=device,dtype=torch.int64)
        storage[::2]=idx;idx=storage[::2]
    out=torch.empty(batch,1,48,128,device=device,dtype=dtype)
    return dict(mixed_qkv=x,a=a,b=b,A_log=alog,dt_bias=bias,scale=128**-.5,
                initial_state=state,out=out,ssm_state_indices=idx,use_qk_l2norm_in_kernel=True)


def clone_case(torch,case):
    result=dict(case)
    # Preserve cache-slot padding rather than silently testing only contiguous state.
    state=case['initial_state']
    backing=torch.empty(state.size(0),state.stride(0),device=state.device,dtype=state.dtype)
    copy=backing[:,:48*128*128].view_as(state);copy.copy_(state)
    result['initial_state']=copy;result['out']=torch.empty_like(case['out'])
    return result


def oracle(torch,case):
    state=case['initial_state'].clone();out=torch.zeros_like(case['out'],dtype=torch.float32)
    for n,slot in enumerate(case['ssm_state_indices'].tolist()):
        if slot<=0:continue
        x=case['mixed_qkv'][n].float()
        q=x[:2048].view(16,128);k=x[2048:4096].view(16,128);v=x[4096:].view(48,128)
        if case['use_qk_l2norm_in_kernel']:
            q=q/torch.sqrt(q.square().sum(-1,keepdim=True)+1e-6)
            k=k/torch.sqrt(k.square().sum(-1,keepdim=True)+1e-6)
        q=q.repeat_interleave(3,dim=0)*case['scale'];k=k.repeat_interleave(3,dim=0)
        sp=torch.nn.functional.softplus(case['a'][n].float()+case['dt_bias'].float(),threshold=20)
        decay=torch.exp(-torch.exp(case['A_log'])*sp)
        beta=torch.sigmoid(case['b'][n].float()).to(case['b'].dtype).float()
        h=state[slot]*decay[:,None,None]
        delta=(v-(h*k[:,None,:]).sum(-1))*beta[:,None]
        h=h+delta[:,:,None]*k[:,None,:]
        state[slot]=h;out[n,0]=(h*q[:,None,:]).sum(-1)
    return out.to(case['out'].dtype),state


def validate(torch,custom,baseline):
    records=[]
    # Fixed before timing: FP32 state rtol=3e-4/atol=3e-5; BF16 output rtol=.016/atol=.002.
    for seed in (0,19,2026):
        for batch in (0,1,2,7,32,64):
            for padded in (False,True):
                case=inputs(torch,batch,seed,padded,bias_fp32=padded)
                if seed==19:case['initial_state'].zero_()
                ref=clone_case(torch,case)
                initial=case['initial_state'].clone()
                expected_o,expected_s=oracle(torch,case)
                custom(**case)
                if batch:baseline(**ref)
                else:ref['out'].zero_()
                torch.testing.assert_close(case['initial_state'],ref['initial_state'],rtol=3e-4,atol=3e-5)
                torch.testing.assert_close(case['out'],ref['out'],rtol=.016,atol=.002)
                torch.testing.assert_close(case['initial_state'],expected_s,rtol=3e-4,atol=3e-5)
                torch.testing.assert_close(case['out'],expected_o,rtol=.016,atol=.002)
                touched={int(i) for i in case['ssm_state_indices'].tolist() if i>0}
                for slot in range(initial.size(0)):
                    if slot not in touched:assert torch.equal(initial[slot],case['initial_state'][slot])
                assert torch.isfinite(case['out']).all() and torch.isfinite(case['initial_state']).all()
                records.append({'seed':seed,'batch':batch,'padded':padded,
                    'max_state_abs':float((case['initial_state']-ref['initial_state']).abs().max()),
                    'max_output_abs':float((case['out'].float()-ref['out'].float()).abs().max()) if batch else 0})
    edges=[]
    for mode in ('int32','unnormalized','zero_qk','extreme_gates','misaligned_state','fp16_fallback'):
        from gdn_pass.kernel import dispatch,supported
        case=inputs(torch,7,31,True)
        if mode=='int32':case['ssm_state_indices']=case['ssm_state_indices'].to(torch.int32)
        if mode=='unnormalized':
            case['use_qk_l2norm_in_kernel']=False
            case['mixed_qkv'][:,:4096].mul_(.01)
        if mode=='zero_qk':case['mixed_qkv'][:,:4096].zero_()
        if mode=='extreme_gates':
            case['a'].copy_(torch.linspace(-80,80,48,device='cuda'))
            case['b'].copy_(torch.linspace(-80,80,48,device='cuda'))
        if mode=='misaligned_state':
            s=case['initial_state']
            backing=torch.empty(s.numel()+1,device='cuda')
            misaligned=backing[1:].view_as(s);misaligned.copy_(s)
            case['initial_state']=misaligned
        if mode=='fp16_fallback':
            for key in ('mixed_qkv','a','b','dt_bias','out'):case[key]=case[key].half()
        ref=clone_case(torch,case)
        expected_o,expected_s=oracle(torch,case)
        fallback=mode in ('misaligned_state','fp16_fallback')
        assert supported(**case) != fallback
        dispatch(**case);baseline(**ref)
        torch.testing.assert_close(case['initial_state'],ref['initial_state'],rtol=3e-4,atol=3e-5)
        torch.testing.assert_close(case['out'],ref['out'],rtol=.016,atol=.002)
        torch.testing.assert_close(case['initial_state'],expected_s,rtol=3e-4,atol=3e-5)
        torch.testing.assert_close(case['out'],expected_o,rtol=.016,atol=.002)
        edges.append({'mode':mode,'fallback':fallback,'max_state_abs':float((case['initial_state']-ref['initial_state']).abs().max()),
                      'max_output_abs':float((case['out'].float()-ref['out'].float()).abs().max())})
    # Long trajectories and CUDA graphs exercise in-place state mutation repeatedly.
    case=inputs(torch,7,77,True);ref=clone_case(torch,case)
    custom(**case);baseline(**ref);torch.cuda.synchronize()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):custom(**case)
    # Capture records commands; reset both states explicitly before the trajectory.
    case['initial_state'].copy_(ref['initial_state'])
    trajectory_state_max=0.;trajectory_output_max=0.
    for step in range(256):
        case['mixed_qkv'].normal_();case['a'].normal_();case['b'].normal_()
        graph.replay();baseline(**ref)
        torch.testing.assert_close(case['initial_state'],ref['initial_state'],rtol=3e-4,atol=3e-5)
        torch.testing.assert_close(case['out'],ref['out'],rtol=.016,atol=.002)
        trajectory_state_max=max(trajectory_state_max,float((case['initial_state']-ref['initial_state']).abs().max()))
        trajectory_output_max=max(trajectory_output_max,float((case['out'].float()-ref['out'].float()).abs().max()))
    torch.cuda.synchronize()
    return {'cases':records,'edge_cases':edges,'trajectory_steps':256,'passed':True,
            'trajectory_max_state_abs':trajectory_state_max,'trajectory_max_output_abs':trajectory_output_max,
            'state_tolerance':{'rtol':3e-4,'atol':3e-5},'output_tolerance':{'rtol':.016,'atol':.002}}


def bench(torch,custom,baseline):
    rows=[]
    for batch in (1,2,8,16,32,64,128):
        # Eight independent states model repeated layer calls and exceed L2 for B>=8.
        pool=[inputs(torch,batch,seed) for seed in range(8)]
        for case in pool:
            case['ssm_state_indices'].copy_(torch.arange(1,batch+1,device='cuda'))
        graphs={}
        for name,op in [('baseline',baseline),('custom',custom)]:
            for case in pool:op(**case)
            torch.cuda.synchronize();graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for case in pool:op(**case)
            graphs[name]=graph
        samples={'baseline':[],'custom':[]}
        for trial in range(10):
            for name in (('baseline','custom') if trial%2==0 else ('custom','baseline')):
                graph=graphs[name]
                for _ in range(3):graph.replay()
                start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(30):graph.replay()
                end.record();end.synchronize()
                samples[name].append(start.elapsed_time(end)*1000/240)
        med={k:statistics.median(v) for k,v in samples.items()}
        row={'batch':batch,'state_pool_bytes':sum(c['initial_state'].numel()*4 for c in pool),
             'median_us':med,'samples_us':samples,'speedup':med['baseline']/med['custom']}
        rows.append(row);print(json.dumps(row),flush=True)
        del pool,graphs,graph,case
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--action',choices=['validate','bench'],required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():p.error('result already exists')
    os.environ['PATH']=str(Path(sys.executable).parent)+os.pathsep+os.environ.get('PATH','')
    check_idle_gpu()
    data={'status':'starting','action':args.action,'started_at':datetime.now(timezone.utc).isoformat(),'gpu':hardware()}
    save(args.output,data)
    try:
        assert (HERE/'gdn_decode.cu').is_file(),'Custom GDN kernel not implemented'
        import torch
        from gdn_pass.kernel import decode
        from vllm.third_party.flash_linear_attention.ops.fused_recurrent import fused_recurrent_gated_delta_rule_packed_decode as baseline
        data['source_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.iterdir() if p.suffix in ('.py','.cu')}
        data['results']=validate(torch,decode,baseline) if args.action=='validate' else bench(torch,decode,baseline)
        data['status']='complete'
    except BaseException as exc:
        data['status']='failed';data['error']=repr(exc);raise
    finally:
        data['finished_at']=datetime.now(timezone.utc).isoformat();save(args.output,data)


if __name__=='__main__':main()
