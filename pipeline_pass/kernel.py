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
    build_extension(name='qwen_gdn_pipeline',sources=[str(here/'gdn_pipeline.cu')],
        build_directory=str(build),with_cuda=True,is_python_module=False,
        extra_cflags=['-O3'],extra_cuda_cflags=['-O3','-lineinfo','-Xptxas=-v'],verbose=True)
    _native=torch.ops.qwen_pipeline_native.decode.default
    return _native


def decode(mixed_qkv,a,b,A_log,dt_bias,scale,initial_state,out,ssm_state_indices,
           use_qk_l2norm_in_kernel=False,tiles=8,cap=4,async_copy=False):
    load()(mixed_qkv,a,b,A_log,dt_bias,initial_state,out,ssm_state_indices,
           scale,use_qk_l2norm_in_kernel,tiles,cap,async_copy)
    return out,initial_state

