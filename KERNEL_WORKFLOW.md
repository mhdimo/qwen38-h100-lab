# Procedure for the next kernel optimization

The serving and kernel questions have different measurements. Establish both
before accepting a change.

1. **Pin and reproduce.** Keep model snapshot, input/output lengths, dtype,
   concurrency, vLLM/Torch/Triton/FlashInfer versions, GPU, graph mode, and clocks/
   power observations alongside every run. Run one GPU experiment at a time.
2. **Profile after warmup.** Separate prefill and decode traces. Look for total
   kernel time, launch gaps, copies, synchronization, CPU scheduling and padding.
   CUDA-graph traces require inspecting GPU kernel events, not just host operator
   times. Profiled timings never enter throughput comparisons.
3. **Identify a specific kernel and shape.** Qwen3.8 includes GDN recurrence and
   chunked prefill as well as full attention, dense FP8 GEMMs, normalization,
   activation/quantization, and a large output projection. Rank actual measured
   time before changing anything. Existing backend ablations come first.
4. **Write a numerical oracle.** Compare the proposed implementation against the
   pinned reference on real activation shapes and strides, multiple seeds, small
   and large batches, variable sequence lengths, and edge tiles. Define FP32/
   BF16/FP8 tolerances before timing. Check NaN/Inf, masks, padding and state updates.
   End-to-end greedy text equality alone is not a numerical correctness test.
5. **Optimize one hypothesis.** For example, fuse an intermediate memory pass,
   improve coalescing or tensor-core tiling, remove a host synchronization, or
   reduce graph-padding work. Check register/shared-memory use and occupancy;
   avoid assuming maximum occupancy or the largest tile is always fastest.
6. **Time correctly.** Use CUDA events, device synchronization at measurement
   boundaries, many warm iterations and multiple independent runs. Measure the
   actual dtype, shape, concurrency and graph replay mode. Exclude JIT, autotune,
   allocation and compilation startup. Prevent cache-only synthetic inputs from
   exaggerating bandwidth. Keep a rollback backend.
7. **Measure end-to-end impact.** Re-run this lab's matched workload matrix.
   A microbenchmark win can disappear under continuous batching, graph capture,
   cache pressure or different shapes. Record regressions, TTFT/TPOT tails,
   memory consumption, and supported/fallback shapes.
8. **Validate deployment behavior.** Replay representative text, image/video,
   structured-output and cancellation traffic. Exercise admission/backpressure
   and co-resident model budgets. Only then integrate a measured configuration
   or kernel behind a reversible selection mechanism.

Compilation and CUDA graphs are complementary. Persist compatible compilation
artifacts; capture common execution shapes; warm each serving configuration
before admitting traffic. A cached binary can reduce startup, but it does not by
itself make an already warmed kernel faster. Graph replay principally reduces
launch overhead; it cannot remove a serialized application admission policy.

Available on the H100 host: Nsight Systems 2025.6.3 and the PyTorch profiler.
`ncu` was not on PATH during the first passes; the lab does not install tools
or change drivers. A later pass found an existing Nsight Compute 2026.1.1
installation outside PATH (see `pipeline_pass/README.md`).

References:

- [NVIDIA CUDA best practices](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [vLLM CUDA graphs](https://docs.vllm.ai/en/latest/design/cuda_graphs/)
- [vLLM optimization](https://docs.vllm.ai/en/latest/configuration/optimization/)
