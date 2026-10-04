"""Preload an isolated CUDA extension; preserve vLLM's packed-decode interface."""
from pathlib import Path
import os

_native=None


def load():
    global _native
    if _native is not None:return _native
    import torch
    from torch.utils.cpp_extension import load as build_extension
    here=Path(__file__).resolve().parent
    build=here/'build'/'cuda';build.mkdir(parents=True,exist_ok=True)
    os.environ.setdefault('CUDA_HOME','/usr/local/cuda')
    os.environ.setdefault('TORCH_CUDA_ARCH_LIST','9.0')
    os.environ.setdefault('MAX_JOBS','2')
    build_extension(name='qwen_gdn_decode',sources=[str(here/'gdn_decode.cu')],
        build_directory=str(build),with_cuda=True,is_python_module=False,
        extra_cflags=['-O3'],extra_cuda_cflags=['-O3','-lineinfo','-Xptxas=-v'],verbose=True)
    _native=torch.ops.qwen_gdn_native.decode.default
    return _native


def supported(mixed_qkv,a,b,A_log,dt_bias,scale,initial_state,out,ssm_state_indices,
              use_qk_l2norm_in_kernel=False):
    import torch
    x=mixed_qkv;s=initial_state;i=ssm_state_indices
    if x.ndim!=2:return False
    batch=x.shape[0]
    return (x.is_cuda and all(t.device==x.device for t in (a,b,A_log,dt_bias,s,out,i))
        and x.ndim==2 and x.shape[1]==10240 and x.stride(1)==1 and batch<=1365
        and a.shape==b.shape==(batch,48) and a.stride(1)==b.stride(1)==1
        and A_log.shape==dt_bias.shape==(48,) and A_log.is_contiguous() and dt_bias.is_contiguous()
        and s.ndim==4 and tuple(s.shape[1:])==(48,128,128) and s.stride()[1:]==(16384,128,1)
        and s.stride(0)>=48*16384 and s.stride(0)%4==0 and s.data_ptr()%16==0
        and out.shape==(batch,1,48,128) and out.is_contiguous() and i.shape==(batch,)
        and i.dtype in (torch.int32,torch.int64) and dt_bias.dtype in (torch.bfloat16,torch.float32)
        and A_log.dtype==s.dtype==torch.float32 and x.dtype==a.dtype==b.dtype==out.dtype==torch.bfloat16)


def decode(mixed_qkv,a,b,A_log,dt_bias,scale,initial_state,out,ssm_state_indices,
           use_qk_l2norm_in_kernel=False,rows_per_warp=4):
    load()(mixed_qkv,a,b,A_log,dt_bias,initial_state,out,ssm_state_indices,
           scale,use_qk_l2norm_in_kernel,rows_per_warp)
    return out,initial_state


def dispatch(*args,**kwargs):
    if supported(*args,**kwargs):return decode(*args,**kwargs)
    from vllm.third_party.flash_linear_attention.ops.fused_recurrent import fused_recurrent_gated_delta_rule_packed_decode
    return fused_recurrent_gated_delta_rule_packed_decode(*args,**kwargs)
