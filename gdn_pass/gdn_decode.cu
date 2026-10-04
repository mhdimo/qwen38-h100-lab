// Experimental packed Gated DeltaNet recurrence for Qwen3.8-27B on SM90.
// Four warps divide a (4*Rows)x128 state tile; each lane owns four adjacent K values.
// State remains FP32 and beta preserves the reference's intermediate BF16 rounding.
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/library.h>
#include <cuda_bf16.h>

namespace {
__device__ __forceinline__ float warp_sum(float x) {
  #pragma unroll
  for (int d=16; d; d>>=1) x += __shfl_xor_sync(0xffffffff, x, d);
  return x;
}
template<typename T> __device__ __forceinline__ float as_float(T v) { return float(v); }
template<> __device__ __forceinline__ float as_float(__nv_bfloat16 v) { return __bfloat162float(v); }

template<typename Index, typename Bias, bool Normalize, int Rows>
__global__ void gdn_decode_warp_kernel(
    const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ a,
    const __nv_bfloat16* __restrict__ b, const float* __restrict__ alog,
    const Bias* __restrict__ bias, float* __restrict__ state,
    __nv_bfloat16* __restrict__ out, const Index* __restrict__ indices,
    int64_t xs, int64_t as, int64_t bs, int64_t ss, int64_t ids,
    int64_t slots, float scale) {
  const int n=blockIdx.y/48, head=blockIdx.y%48, qhead=head/3;
  const int lane=threadIdx.x%32, warp=threadIdx.x/32;
  const int row0=blockIdx.x*(4*Rows)+warp*Rows, col=lane*4;
  const int64_t slot=indices[n*ids];
  if (slot<=0 || slot>=slots) {
    if (lane<Rows) out[(n*48+head)*128+row0+lane]=__float2bfloat16(0.f);
    return;
  }
  float q[4], k[4], qsq=0.f, ksq=0.f;
  #pragma unroll
  for (int j=0;j<4;++j) {
    q[j]=__bfloat162float(x[n*xs+qhead*128+col+j]);
    k[j]=__bfloat162float(x[n*xs+2048+qhead*128+col+j]);
    qsq+=q[j]*q[j]; ksq+=k[j]*k[j];
  }
  float qnorm=1.f, knorm=1.f;
  if constexpr (Normalize) {
    qnorm=1.f/sqrtf(warp_sum(qsq)+1e-6f);
    knorm=1.f/sqrtf(warp_sum(ksq)+1e-6f);
  }
  #pragma unroll
  for (int j=0;j<4;++j) { q[j]=(q[j]*qnorm)*scale; k[j]*=knorm; }
  const float gate=__bfloat162float(a[n*as+head])+as_float(bias[head]);
  const float softplus=gate<=20.f ? logf(1.f+expf(gate)) : gate;
  const float decay=expf(-expf(alog[head])*softplus);
  const float beta=__bfloat162float(__float2bfloat16_rn(1.f/(1.f+expf(-__bfloat162float(b[n*bs+head])))));
  float h[Rows][4];
  float* base=state+slot*ss+head*128*128+row0*128+col;
  #pragma unroll
  for (int r=0;r<Rows;++r) {
    const float4 v=*reinterpret_cast<const float4*>(base+r*128);
    h[r][0]=v.x*decay; h[r][1]=v.y*decay; h[r][2]=v.z*decay; h[r][3]=v.w*decay;
  }
  #pragma unroll
  for (int r=0;r<Rows;++r) {
    float dot=0.f;
    #pragma unroll
    for (int j=0;j<4;++j) dot+=h[r][j]*k[j];
    dot=warp_sum(dot);
    const float v=__bfloat162float(x[n*xs+4096+head*128+row0+r]);
    const float delta=(v-dot)*beta;
    float result=0.f;
    #pragma unroll
    for (int j=0;j<4;++j) {
      h[r][j]=fmaf(delta,k[j],h[r][j]);
      result+=h[r][j]*q[j];
    }
    *reinterpret_cast<float4*>(base+r*128)=make_float4(h[r][0],h[r][1],h[r][2],h[r][3]);
    result=warp_sum(result);
    if (lane==0) out[(n*48+head)*128+row0+r]=__float2bfloat16_rn(result);
  }
}

void decode(const at::Tensor& x, const at::Tensor& a, const at::Tensor& b,
            const at::Tensor& alog, const at::Tensor& bias, at::Tensor state,
            at::Tensor out, const at::Tensor& indices, double scale, bool normalize, int64_t rows) {
  TORCH_CHECK(x.is_cuda(), "CUDA required");
  for (const auto& t: {a,b,alog,bias,state,out,indices})
    TORCH_CHECK(t.device()==x.device(), "all tensors must share a CUDA device");
  TORCH_CHECK(x.dim()==2 && x.size(1)==10240 && x.stride(1)==1, "expected packed [B,10240]");
  const auto batch=x.size(0);
  TORCH_CHECK(a.sizes()==at::IntArrayRef({batch,48}) && b.sizes()==a.sizes() &&
              a.stride(1)==1 && b.stride(1)==1, "expected [B,48] gates");
  TORCH_CHECK(alog.sizes()==at::IntArrayRef({48}) && bias.sizes()==alog.sizes() &&
              alog.is_contiguous() && bias.is_contiguous(), "expected 48 contiguous gate parameters");
  TORCH_CHECK(state.dim()==4 && state.size(1)==48 && state.size(2)==128 && state.size(3)==128 &&
              state.stride(3)==1 && state.stride(2)==128 && state.stride(1)==16384 &&
              state.stride(0)>=48*16384 && state.stride(0)%4==0 &&
              reinterpret_cast<uintptr_t>(state.data_ptr())%16==0, "unsupported state layout/alignment");
  TORCH_CHECK(out.sizes()==at::IntArrayRef({batch,1,48,128}) && out.is_contiguous(), "unsupported output layout");
  TORCH_CHECK(indices.sizes()==at::IntArrayRef({batch}) &&
              (indices.scalar_type()==at::kInt || indices.scalar_type()==at::kLong), "expected integer slot indices");
  TORCH_CHECK(x.scalar_type()==at::kBFloat16 && a.scalar_type()==at::kBFloat16 &&
              b.scalar_type()==at::kBFloat16 && out.scalar_type()==at::kBFloat16 &&
              state.scalar_type()==at::kFloat && alog.scalar_type()==at::kFloat &&
              (bias.scalar_type()==at::kFloat || bias.scalar_type()==at::kBFloat16), "unsupported dtype");
  if (!batch) return;
  TORCH_CHECK(rows==4 || rows==8 || rows==16, "unsupported row tile");
  TORCH_CHECK(batch<=65535/48, "batch exceeds grid limit");
  const c10::cuda::CUDAGuard guard(x.device());
  const dim3 grid(128/(4*rows),batch*48);
  const auto stream=at::cuda::getCurrentCUDAStream();
  #define LAUNCH(I,D,N,R) gdn_decode_warp_kernel<I,D,N,R><<<grid,128,0,stream>>>( \
    reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),reinterpret_cast<const __nv_bfloat16*>(a.data_ptr()), \
    reinterpret_cast<const __nv_bfloat16*>(b.data_ptr()),alog.data_ptr<float>(),reinterpret_cast<const D*>(bias.data_ptr()), \
    state.data_ptr<float>(),reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),indices.data_ptr<I>(), \
    x.stride(0),a.stride(0),b.stride(0),state.stride(0),indices.stride(0),state.size(0),float(scale))
  #define ROWS(I,D,N) if (rows==4) { LAUNCH(I,D,N,4); } else if (rows==8) { LAUNCH(I,D,N,8); } else { LAUNCH(I,D,N,16); }
  #define NORM(I,D) if (normalize) { ROWS(I,D,true); } else { ROWS(I,D,false); }
  #define BIAS(I) if (bias.scalar_type()==at::kFloat) { NORM(I,float) } else { NORM(I,__nv_bfloat16) }
  if (indices.scalar_type()==at::kLong) { BIAS(int64_t) } else { BIAS(int32_t) }
  #undef BIAS
  #undef NORM
  #undef ROWS
  #undef LAUNCH
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
}
TORCH_LIBRARY(qwen_gdn_native,m) {
  m.def("decode(Tensor x, Tensor a, Tensor b, Tensor alog, Tensor bias, Tensor(s!) state, Tensor(o!) out, Tensor indices, float scale, bool normalize, int rows) -> ()");
}
TORCH_LIBRARY_IMPL(qwen_gdn_native,CUDA,m) { m.impl("decode",&decode); }
