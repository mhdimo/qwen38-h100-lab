# Gated DeltaNet CUDA decode experiment

The actual CUDA kernel is `gdn_decode.cu`. This is a new decode-only experiment,
separate from `custom_kernels/silu_mul_fp8.cu` and from the initial serving tuning.
No installed vLLM source or production service is modified.

## Implementation

For Qwen's packed BF16 QKV, H=16, HV=48 and K=V=128, each block has four warps.
Each lane owns four consecutive K values. The selected tile assigns four V rows
to each warp (16 rows/block). We also compile/test eight and sixteen rows/warp.
The selected version needs 40 registers/thread, compared with 62 and 104 for the
other two tiles and 205 in the profiled installed Triton kernel. No spills are
reported. Warp shuffles reduce dot products and aligned float4 state operations
provide vector loads/stores. The original was already fused; the change is its
thread/tile mapping, not an additional operator fusion.

The recurrence keeps FP32 state, including the reference's BF16 rounding of beta
before conversion to FP32. It accepts padded state-slot strides, strided input
and gate rows, BF16 or FP32 dt_bias, int32/int64 strided slot indices, optional Q/K
normalization and caller-provided scale. Output is contiguous BF16.
Positive slot indices must be unique and in range. Nonpositive padding indices
zero the output and leave state alone. Positive out-of-range indices are guarded
against memory access, but equivalence outside the valid-slot contract is not
claimed. Unsupported metadata uses the installed function via `dispatch`.

`worker.py` substitutes the function only in this lab worker process, inside
vLLM's existing opaque GDN attention operation. It preloads the native extension
before graph capture. `configs/batch64_t16384_gdn.json` enables the ablation.
The prior SiLU substitution is not enabled in this worker.

## Reproduce on the authorized H100

Use the pinned Python 3.14.6 interpreter and restore the source tree on the lab
host first.
The CUDA source targets SM90 and requires the already-installed CUDA/PyTorch
development headers. It does not run on the local Mac GPU.

```sh
python gdn_pass/run.py --kernel-action gdn-validate
python gdn_pass/run.py --kernel-action gdn-bench
python gdn_pass/run.py --kernel-action gdn-sanitize
python gdn_pass/run.py --configs batch64_t16384_gdn \
  --workloads workloads_profile.json --repetitions 1 --profile
python gdn_pass/run.py --configs batch64_t16384 batch64_t16384_gdn --repetitions 2
```

The launcher contains vLLM, Triton, Inductor, driver and temporary caches inside
the lab. The shared suite takes an exclusive cooperating lock, checks GPU
idleness and bounds subprocess lifetime. Never run suites concurrently or kill
another workload. Each run has its own timestamped JSON/log directory.

## Validation and measurement

All three tile sizes pass 42 matrix cases, six extra edge/fallback cases and 256
changing-input CUDA-graph recurrent steps. Matrix cases compare the installed
reference and an independent PyTorch oracle. FP32 state tolerances were fixed at
rtol=3e-4/atol=3e-5, output at rtol=.016/atol=.002 before measurement. These were
not relaxed. The selected graph trajectory's largest state error was 8.345e-6
and largest BF16 output difference 9.766e-4. This is not bitwise equivalence.
Compute Sanitizer 2026.1.1 reports zero errors for 48 direct custom-kernel calls
and eight graph replays; only custom GDN kernels are instrumented.

The exploratory microbenchmark alternates four implementations over ten trials;
each times 30 replays of an eight-operation graph on independent states. All
batch entries are active. Reported active bytes exclude unused sentinel slots.
Batch 1 can fit in L2; larger pools deliberately exceed L2. Fixed inputs make this
a steady-state recurrence benchmark, not a serving simulation. The selected
16-row tile is compared separately in the full model with changing activations.

See `paper/gdn.tex` and its generated report/data for the complete serving
comparison. Prefill-only results are a negative control: this kernel only changes
decode. Production traffic quality, multimodal behavior, structured outputs and
application admission remain outside this lab pass.

Matched serving results (`results/20261001T125154.014415Z`, 400 requests/config):
single throughput +0.71%, prefill -0.18%, concurrent decode -0.04%, mixed +1.38%.
Mixed TPOT p95 worsens from 18.40 to 21.48 ms. Concurrent decode is effectively
flat, so stock GDN remains the preferred baseline for the throughput
objective; the custom worker stays opt-in. Two repetitions and synthetic traffic
do not establish confidence in the small mixed gain. Generated-token agreement
is 232/400, versus 231/400 between unchanged reference runs; this does not certify
production quality. Both engines pass the same three text smokes.

## Local preservation and remote cleanup

The initial expected-failure test and every completed attempt are preserved in
`results/`; `variants/initial_r8/` keeps the first implementation. The temporary
remote lab was hash-verified locally and removed. The pinned numerical reference
is the installed vLLM v0.26.0 operator; its upstream source can be obtained from
the [vLLM repository](https://github.com/vllm-project/vllm).
