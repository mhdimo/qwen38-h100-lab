"""Opt-in activation fusion substitution in this worker process only."""
import torch
from custom_kernels.silu_mul_fp8 import load

_installed = False


def install():
    global _installed
    if _installed:
        return
    native = load()  # Compile/load before graph capture or measured inference.
    original = torch.ops._C.silu_and_mul_per_block_quant.default

    @torch.library.custom_op('qwen_lab::silu_mul_fp8', mutates_args=('out', 'scales'))
    def replacement(out: torch.Tensor, input: torch.Tensor, scales: torch.Tensor,
                    group_size: int, scale_ub: torch.Tensor | None,
                    is_scale_transposed: bool) -> None:
        if (input.dtype == torch.bfloat16 and out.dtype == torch.float8_e4m3fn
                and group_size == 128 and input.is_contiguous() and out.is_contiguous()
                and input.data_ptr() % 4 == 0 and out.data_ptr() % 4 == 0
                and input.shape[0] > 1):
            native(out, input, scales, group_size, scale_ub, is_scale_transposed)
        else:
            original(out, input, scales, group_size, scale_ub, is_scale_transposed)

    @replacement.register_fake
    def fake(out, input, scales, group_size, scale_ub, is_scale_transposed):
        return None

    from vllm.compilation.passes.fusion.act_quant_fusion import FUSED_OPS
    from vllm.model_executor.layers.quantization.utils.quant_utils import kFp8Dynamic128Sym
    FUSED_OPS[kFp8Dynamic128Sym] = torch.ops.qwen_lab.silu_mul_fp8.default
    _installed = True
    print('QWEN_LAB: enabled custom CUDA SiLU/FP8 kernel for 128-element groups', flush=True)
