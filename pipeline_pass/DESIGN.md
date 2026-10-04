# Persistent work and asynchronous state staging

Keep the existing CUDA GDN implementation and installed vLLM reference unchanged.
Add an opt-in CUDA kernel with the same packed recurrence, FP32 state, BF16 beta
rounding, metadata checks, and fixed numerical tolerances. Four warps each own
four state rows. A CTA processes 1, 2, 4 or 8 adjacent 16-row tiles, reusing Q/K,
normalization and gates. Optionally cap the grid to four CTAs per SM and loop
over work items (bounded persistent scheduling, not a resident serving loop).
For the same scheduling configuration, compare direct float4 loads against two
shared-memory buffers populated by 16-byte cp.async copies. Each thread reads
only its own copied locations; completion waits are per thread. Prefetch the
next tile before computing the current tile. Do not speculate future tokens.

Profile stock and existing custom GDN and SiLU with Nsight Compute; capture a
warmed graph micro-workload and actual engine prefill/decode with Nsight Systems.
Counter runs are diagnostic, never throughput measurements. Use the shared lab
lock and GPU-idle check. Scope elevated profiling to owned scripts/reports; do
not alter performance-counter permissions, clocks, services or installed tools.

Validate independent oracle, pinned reference, strided inputs, masks, BF16/FP32
bias, int32/int64 indices, graph trajectories, and memcheck/racecheck. Benchmark
eight independent state buffers, alternating orders, CUDA graphs and multiple
trials. Run matched engine workloads only for a credible winning candidate.
Preserve failures, profiler reports and build outputs locally, verify hashes,
then remove the task-owned remote lab. Production remains unchanged.
