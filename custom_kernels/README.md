# Custom CUDA kernel experiment

Completed result: full-model throughput relative to tuned vLLM is +5.90% prefill,
+0.93% decode, +3.65% mixed and -0.15% single-request. The operation microbenchmark
reaches 2.26x at 16,384 rows, with 1.53x at 32 rows. The serving adapter keeps the
original kernel at batch one, where the standalone candidate was slightly slower.
See `../paper/kernel.pdf` and `../paper/kernel_data.json` for raw-source attribution.

There are no running remote experiments. All remote files and the tracked custom
AOT cache were verified locally before removal of the temporary lab workspace.

This is a new kernel implementation, separate from the earlier serving-configuration
study. Source: `silu_mul_fp8.cu`. Objective: aggregate serving throughput on the
existing Qwen3.8-27B-FP8/H100 stack, preserving the existing quantization semantics.

## Target selected from measured profiles

The original tuned `batch64_t16384` engine was profiled after warmup in
`../results/20261001T115943.440807Z`. The fused SiLU/multiply/FP8 quantization kernel
accounts for 14.54% of summed GPU kernel duration in the prefill capture. Selecting
CUDA graph launch correlations containing recurrent decode kernels isolates ten
decode steps: SiLU/FP8 is 2.10%, and GDN recurrence is 20.45% of their summed kernel
duration. These are profiling measurements, not uninstrumented serving times or
exclusive wall-clock fractions. No Nsight Compute hardware-counter study was run.

The second trace has the same `prefill_profile` filename prefix as the first because
the installed profiler wrapper reuses its initial name. The later timestamp is
the decode workload; preserve both files and use the explicit trace index.

## Implementation

The kernel computes `y = SiLU(gate) * up` in FP32 from BF16 input. Each group of
128 values has scale `max(max(abs(y))/448, 1/(448*512))`; an optional upper bound
applies before the lower floor, matching the installed CUDA operator. Values are
divided by the scale, saturated to [-448,448], and converted to FP8 E4M3.

One warp processes one quantization group. Each lane owns four adjacent values;
warp shuffles reduce the maximum. Four warps process adjacent groups per block.
There are no block barriers or shared-memory reductions. Outputs use packed
four-byte stores. Row-major and transposed scale layouts are supported, including
partial last blocks. This is not a new quantization algorithm or weight format.

`silu_mul_fp8.py` compiles for SM90 before measurement/capture, with all build
products inside this lab. `worker.py` selects an isolated worker that replaces the
activation fusion mapping with a separate custom operator. Other formats use the
installed operator. No installed vLLM file or production service is modified.
The compiled vLLM graph cache for this experiment is also kept inside the lab.
One vLLM AOT cache directory was created outside the lab by the installed
runtime; it was copied locally and removed along with the lab. Existing shared
caches and production files were preserved.

## Verification and timing

`experiment.py` compares to the installed vLLM binary and an independent PyTorch
formula. Acceptance requires identical FP8 bytes against the installed kernel,
FP32 scale agreement (rtol 2e-6, atol 1e-8), finite values, and correct CUDA graph
replay with changing inputs. Cases include three seeds, empty and partial tiles,
Qwen's 17,408-wide intermediate, multiple batch sizes, both scale layouts, zeros,
tiny/extreme inputs and upper-bound saturation. Failed attempts remain in results.
The final matrix has 168 nonempty reference comparisons and 48 empty-input checks;
all tested nonempty FP8 bytes and scales match exactly. Full-model output hashes
match on 259/400 requests, versus 264/400 between unchanged reference runs. Do not
infer deterministic generation or production quality from isolated kernel equality.

The microbenchmark uses warmed CUDA graphs with four distinct input buffers and
alternates baseline/custom measurement order across ten trials. Each sample times
30 graph replays with CUDA events. Small decode tensors remain cache-resident;
large prefill tensors exceed L2. Microbenchmark speedups must be confirmed by the
matched, unprofiled AsyncLLM workloads before drawing serving conclusions.

On an authorized, idle H100 with this lab restored and the pinned stack available:

```bash
python run_suite.py --kernel-action validate
python run_suite.py --kernel-action bench
python run_suite.py --configs batch64_t16384 batch64_t16384_custom_silu --repetitions 2
python run_suite.py --configs batch64_t16384_custom_silu --workloads workloads_profile.json --repetitions 1 --profile
```

These commands use the shared GPU lock and refuse an occupied GPU. Rebuild Python
extensions on the target runtime; the saved Linux binaries do not execute on macOS.
The baseline CUDA source can be inspected in
[vLLM upstream](https://github.com/vllm-project/vllm/blob/main/csrc/libtorch_stable/quantization/fused_kernels/fused_silu_mul_block_quant.cu);
the installed v0.26.0 binary, rather than upstream `main`, is the tested oracle.

This additional H100 pass preserved all artifacts locally, then removed the
temporary remote lab.
