"""Build the CUDA source inside this disposable lab; no installed package edits."""
from pathlib import Path
import os

_loaded = None


def load():
    global _loaded
    if _loaded is not None:
        return _loaded
    import torch
    from torch.utils.cpp_extension import load as load_extension
    here = Path(__file__).resolve().parent
    build = here / 'build' / 'silu_mul_fp8'
    build.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('CUDA_HOME', '/usr/local/cuda')
    os.environ.setdefault('TORCH_CUDA_ARCH_LIST', '9.0')
    os.environ.setdefault('MAX_JOBS', '2')
    load_extension(name='qwen_lab_silu_fp8', sources=[str(here/'silu_mul_fp8.cu')],
                   build_directory=str(build), with_cuda=True, is_python_module=False,
                   extra_cflags=['-O3'],
                   extra_cuda_cflags=['-O3', '-lineinfo', '-Xptxas=-v'],
                   verbose=True)
    _loaded = torch.ops.qwen_lab_native.silu_mul_fp8.default
    return _loaded
