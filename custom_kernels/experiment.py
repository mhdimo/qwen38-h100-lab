"""GPU numerical tests and matched microbenchmarks; invoked through run_suite.py."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import sys
import os

from benchmark import check_idle_gpu, hardware, save

HERE = Path(__file__).resolve().parent


def allocate(torch, x, transposed):
    m, twice_n = x.shape
    n = twice_n // 2
    y = torch.empty((m, n), device=x.device, dtype=torch.float8_e4m3fn)
    s = torch.empty((n // 128, m) if transposed else (m, n // 128),
                    device=x.device, dtype=torch.float32)
    return y, s.t() if transposed else s


def validate(torch, custom, baseline):
    """Faults caught: wrong SiLU/order, group scale, FP8 rounding, layout, tail, graph replay."""
    rows = []
    for seed in (0, 19, 2026):
        torch.manual_seed(seed)
        for m, n in [(0, 128), (1, 128), (3, 384), (7, 17408), (32, 17408),
                     (64, 17408), (129, 17408), (1024, 17408)]:
            for transposed in (False, True):
                x = torch.randn((m, 2*n), dtype=torch.bfloat16, device='cuda') * 3
                for pattern in ('random', 'zero', 'tiny', 'extreme') if m < 8 else ('random',):
                    if pattern == 'zero': x.zero_()
                    if pattern == 'tiny': x.fill_(1e-5)
                    if pattern == 'extreme':
                        x[:, :n] = torch.linspace(-80, 80, n, device='cuda')
                        x[:, n:] = 4
                    for upper in (None, .001) if m < 8 else (None,):
                        ub = None if upper is None else torch.tensor(upper, device='cuda')
                        y, s = allocate(torch, x, transposed)
                        y0, s0 = allocate(torch, x, transposed)
                        custom(y, x, s, 128, ub, transposed)
                        if m == 0:
                            rows.append({'m':m, 'n':n, 'passed':True})
                            continue  # Installed kernel cannot launch a zero-sized grid.
                        baseline(y0, x, s0, 128, ub, transposed)
                        torch.testing.assert_close(s, s0, rtol=2e-6, atol=1e-8)
                        # Exact FP8 equality is the goal; signed zero is also compared.
                        mismatch=y.view(torch.uint8)!=y0.view(torch.uint8)
                        assert not mismatch.any(), \
                            (f'FP8 bytes differ: seed={seed} M={m} N={n} {pattern} ub={upper}; '
                             f'count={int(mismatch.sum())}; custom={y.float()[mismatch][:8].tolist()}; '
                             f'baseline={y0.float()[mismatch][:8].tolist()}')
                        # Independent FP32 mathematical oracle, including scale floor and clamp.
                        g, u = x.float().chunk(2, -1)
                        act = (g * torch.sigmoid(g)) * u
                        expected_s = act.view(m, n//128, 128).abs().amax(-1) / 448
                        if ub is not None: expected_s.clamp_(max=upper)
                        expected_s.clamp_(min=1/(448*512))
                        torch.testing.assert_close(s, expected_s, rtol=2e-6, atol=1e-8)
                        expected = (act.view(m,n//128,128)/expected_s.unsqueeze(-1)).clamp(-448,448)
                        # FP8 values at rounding boundaries can differ by one representable step.
                        torch.testing.assert_close(y.float(), expected.to(y.dtype).view(m,n).float(), rtol=.13, atol=.002)
                        assert torch.isfinite(y.float()).all() and torch.isfinite(s).all()
                        rows.append({'seed':seed,'m':m,'n':n,'transposed':transposed,
                                     'pattern':pattern,'upper':upper,'fp8_exact':True,
                                     'scale_max_abs':float((s-s0).abs().max())})
    x = torch.randn((32, 34816), dtype=torch.bfloat16, device='cuda')
    y,s = allocate(torch,x,True)
    graph = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    with torch.cuda.graph(graph): custom(y,x,s,128,None,True)
    for seed in range(4):
        x.normal_()
        graph.replay()
        y0,s0 = allocate(torch,x,True)
        baseline(y0,x,s0,128,None,True)
        assert torch.equal(y.view(torch.uint8),y0.view(torch.uint8))
        torch.testing.assert_close(s,s0,rtol=2e-6,atol=1e-8)
    # Contiguous storage-offset views may still be unaligned for vector accesses.
    from custom_kernels.integration import install
    install()
    adapter=torch.ops.qwen_lab.silu_mul_fp8.default
    for shifted in ('input','output'):
        x=torch.randn(1+2*384*3,device='cuda',dtype=torch.bfloat16)
        x=x[1 if shifted=='input' else 0:][:2*384*3].view(3,768)
        y=torch.empty(1+384*3,device='cuda',dtype=torch.float8_e4m3fn)
        y=y[1 if shifted=='output' else 0:][:384*3].view(3,384)
        s=torch.empty(3,3,device='cuda',dtype=torch.float32)
        try:
            custom(y,x,s,128,None,False)
        except RuntimeError as exc:
            assert 'aligned' in str(exc)
        else:
            raise AssertionError('unaligned vector access was not rejected')
        adapter(y,x,s,128,None,False)
        y0,s0=allocate(torch,x,False)
        baseline(y0,x,s0,128,None,False)
        assert torch.equal(y.view(torch.uint8),y0.view(torch.uint8))
        torch.testing.assert_close(s,s0,rtol=0,atol=0)
    x=torch.randn(3,768,device='cuda',dtype=torch.float16)
    y,s=allocate(torch,x,False)
    adapter(y,x,s,128,None,False)
    y0,s0=allocate(torch,x,False)
    baseline(y0,x,s0,128,None,False)
    assert torch.equal(y.view(torch.uint8),y0.view(torch.uint8))
    torch.testing.assert_close(s,s0,rtol=0,atol=0)
    torch.cuda.synchronize()
    return {'cases':rows,'graph_replays':4,'alignment_fallback_cases':2,
            'fp16_fallback_cases':1,'passed':True}


def bench(torch, custom, baseline):
    rows=[]
    torch.manual_seed(20261001)
    for m in (1, 32, 64, 256, 1024, 4096, 8192, 16384):
        pool=[torch.randn((m,34816),device='cuda',dtype=torch.bfloat16) for _ in range(4)]
        buffers=[allocate(torch,x,True) for x in pool]
        graphs={}
        for name,op in [('baseline',baseline),('custom',custom)]:
            for _ in range(3):
                for x,(y,s) in zip(pool,buffers): op(y,x,s,128,None,True)
            torch.cuda.synchronize()
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for x,(y,s) in zip(pool,buffers): op(y,x,s,128,None,True)
            graphs[name]=graph
        samples={'baseline':[],'custom':[]}
        # Alternate execution order. Four distinct inputs per graph reduce warm-L2 bias
        # at prefill sizes. Small decode shapes remain explicitly cache-resident.
        for trial in range(10):
            for name in (('baseline','custom') if trial%2==0 else ('custom','baseline')):
                graph=graphs[name]
                for _ in range(3): graph.replay()
                start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(30): graph.replay()
                end.record(); end.synchronize()
                samples[name].append(start.elapsed_time(end)*1000/(30*len(pool)))
        med={name:statistics.median(values) for name,values in samples.items()}
        row={'m':m,'n':17408,'layout':'transposed_scales','mode':'CUDA graph replay',
             'input_pool_bytes':sum(x.numel()*x.element_size() for x in pool),
             'microseconds':samples,'median_us':med,'speedup':med['baseline']/med['custom']}
        rows.append(row); print(json.dumps(row),flush=True)
        del graphs,graph,buffers,pool
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--action',choices=['validate','bench'],required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists(): parser.error('refusing to overwrite evidence')
    os.environ['PATH']=str(Path(sys.executable).parent)+os.pathsep+os.environ.get('PATH','')
    check_idle_gpu()
    result={'status':'starting','action':args.action,'started_at':datetime.now(timezone.utc).isoformat(),
            'gpu':hardware()}
    save(args.output,result)
    try:
        assert (HERE/'silu_mul_fp8.cu').is_file(), 'Custom CUDA kernel has not been implemented yet'
        import torch
        import vllm._custom_ops  # registers the installed binary reference
        from custom_kernels.silu_mul_fp8 import load
        custom=load()
        baseline=torch.ops._C.silu_and_mul_per_block_quant.default
        result['source_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in HERE.iterdir() if p.suffix in ('.py','.cu','.cpp')}
        result['torch']=torch.__version__
        result['results']=validate(torch,custom,baseline) if args.action=='validate' else bench(torch,custom,baseline)
        result['status']='complete'
    except BaseException as exc:
        result['status']='failed';result['error']=repr(exc)
        raise
    finally:
        result['finished_at']=datetime.now(timezone.utc).isoformat()
        save(args.output,result)


if __name__=='__main__': main()
