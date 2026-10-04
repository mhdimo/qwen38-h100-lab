// Experimental lab kernel: BF16 SiLU(gate)*up -> grouped E4M3 FP8.
// One warp owns 128 values, four adjacent values per lane. No shared memory or
// block barriers: a warp shuffle computes the scale, then lanes pack four bytes.
// The installed vLLM 0.26.0 binary is the numerical/performance reference.
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/library.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

namespace {
constexpr int kGroup = 128;
constexpr int kWarps = 4;

__global__ void silu_mul_fp8_warp_kernel(
    const __nv_bfloat16* __restrict__ input,
    unsigned char* __restrict__ output, float* __restrict__ scales,
    const float* upper, long long tokens, long long width,
    long long scale_row_stride, long long scale_group_stride) {
  const int lane = threadIdx.x & 31;
  const long long groups_per_row = width / kGroup;
  const long long group = (long long)blockIdx.x * kWarps + threadIdx.x / 32;
  if (group >= tokens * groups_per_row) return;  // Uniform within every warp.
  const long long row = group / groups_per_row;
  const long long column = (group % groups_per_row) * kGroup + lane * 4;
  const __nv_bfloat16* gate = input + row * (2 * width) + column;
  const __nv_bfloat16* up = gate + width;
  const float2 g0 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(gate));
  const float2 g1 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(gate + 2));
  const float2 u0 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(up));
  const float2 u1 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(up + 2));
  const float g[4] = {g0.x, g0.y, g1.x, g1.y};
  const float u[4] = {u0.x, u0.y, u1.x, u1.y};
  float v[4];
  float maximum = 0.f;
  #pragma unroll
  for (int j = 0; j < 4; ++j) {
    const float sigmoid = 1.f / (1.f + expf(-g[j]));
    v[j] = (g[j] * sigmoid) * u[j];
    maximum = fmaxf(maximum, fabsf(v[j]));
  }
  #pragma unroll
  for (int offset = 16; offset; offset >>= 1)
    maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, offset));
  float scale = maximum / 448.f;
  if (upper) scale = fminf(scale, *upper);
  scale = fmaxf(scale, 1.f / (448.f * 512.f));
  if (lane == 0)
    scales[row * scale_row_stride + (group % groups_per_row) * scale_group_stride] = scale;
  unsigned int packed = 0;
  #pragma unroll
  for (int j = 0; j < 4; ++j) {
    const float normalized = fmaxf(-448.f, fminf(v[j] / scale, 448.f));
    const unsigned int byte = __nv_cvt_float_to_fp8(normalized, __NV_SATFINITE, __NV_E4M3);
    packed |= byte << (8 * j);
  }
  *reinterpret_cast<unsigned int*>(output + row * width + column) = packed;
}

void silu_mul_fp8(at::Tensor out, const at::Tensor& input, at::Tensor scales,
                  int64_t group_size, const std::optional<at::Tensor>& scale_ub,
                  bool is_scale_transposed) {
  TORCH_CHECK(input.is_cuda() && out.device() == input.device() && scales.device() == input.device(),
              "all tensors must be on the same CUDA device");
  TORCH_CHECK(input.scalar_type() == at::kBFloat16 && out.scalar_type() == at::kFloat8_e4m3fn &&
              scales.scalar_type() == at::kFloat, "expected BF16 input, E4M3 output, FP32 scales");
  TORCH_CHECK(input.dim() == 2 && out.dim() == 2 && scales.dim() == 2,
              "expected rank-two tensors");
  TORCH_CHECK(input.is_contiguous() && out.is_contiguous(), "input/output must be contiguous");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(input.data_ptr()) % 4 == 0 &&
              reinterpret_cast<uintptr_t>(out.data_ptr()) % 4 == 0,
              "vector loads/stores require four-byte aligned input and output");
  const int64_t m = input.size(0), n = out.size(1);
  TORCH_CHECK(group_size == kGroup && n > 0 && n % kGroup == 0 && input.size(1) == 2*n &&
              out.size(0) == m && scales.size(0) == m && scales.size(1) == n/kGroup,
              "incompatible shape or quantization group");
  TORCH_CHECK(m == 0 || (is_scale_transposed && scales.stride(0) == 1 && scales.stride(1) == m) ||
              (!is_scale_transposed && scales.is_contiguous()), "unsupported scale layout");
  if (scale_ub) {
    TORCH_CHECK(scale_ub->device() == input.device() && scale_ub->scalar_type() == at::kFloat &&
                scale_ub->numel() == 1, "scale upper bound must be one CUDA FP32 value");
  }
  if (m == 0) return;
  const c10::cuda::CUDAGuard guard(input.device());
  const int64_t blocks = (m * (n/kGroup) + kWarps-1)/kWarps;
  TORCH_CHECK(blocks <= 2147483647, "grid too large");
  silu_mul_fp8_warp_kernel<<<blocks, kWarps*32, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
      reinterpret_cast<unsigned char*>(out.data_ptr()), scales.data_ptr<float>(),
      scale_ub ? scale_ub->data_ptr<float>() : nullptr, m, n, scales.stride(0), scales.stride(1));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
}  // namespace

TORCH_LIBRARY(qwen_lab_native, m) {
  m.def("silu_mul_fp8(Tensor(a!) out, Tensor input, Tensor(b!) scales, int group_size, Tensor? scale_ub, bool is_scale_transposed) -> ()");
}
TORCH_LIBRARY_IMPL(qwen_lab_native, CUDA, m) {
  m.impl("silu_mul_fp8", &silu_mul_fp8);
}
