# Persistent work, async state copies, and Nsight profiling

The tested persistent/async GDN variants do not improve the high-concurrency target. Keep the installed GDN decoder for serving; all new variants remain standalone research kernels. Earlier SiLU engine gains remain separate. Production was not modified.

## Implementation

`gdn_pipeline.cu` contains the actual CUDA kernel. Four warps process 16 FP32 state rows at a time. `tiles=1/2/4/8` controls adjacent-tile work reuse; Q/K normalization and gates are reused within each work item. `cap=4` bounds the grid to four CTAs per SM (528 CTAs on this H100) with a grid-stride work loop; `cap=0` launches all work items. This is bounded persistent scheduling within one invocation, not a permanently resident server kernel.

`async_copy=True` uses two 8 KiB shared buffers and 16-byte `__pipeline_memcpy_async` operations, prefetching the next tile while computing the current one. Each thread consumes only its own copied addresses and waits before use/reuse. With `tiles=1`, async staging is a control without within-item overlap. SASS confirms `LDGSTS.E.BYPASS.128`, `LDGDEPBAR`, and dependency waits; the dump is kept in the local lab archive. Direct variants use 43 registers/thread, async variants 40 with 16 KiB dynamic shared memory, with no compiler spills.

## Unprofiled timing

Microseconds per kernel; each cell shows run 1 / independent run 2. Each run uses 10 alternating/rotating trials, 30 replays of eight kernels per trial, eight independent state buffers, active slots 1..B, warm compilation and graphs. Active state exceeds L2 at B>=8. Inputs are fixed during timing and recurrent states converge; changing-input correctness tests are separate.

| Batch | Stock | Previous CUDA | Persistent (8 tiles, capped) | Persistent + async | Async (2 tiles, uncapped) |
|---:|---:|---:|---:|---:|---:|
| 1 | 6.00 / 6.03 | 4.73 / 4.75 | 12.48 / 12.60 | 10.54 / 10.46 | 5.40 / 5.38 |
| 2 | 9.46 / 9.45 | 7.94 / 7.92 | 15.85 / 15.82 | 12.88 / 12.89 | 8.55 / 8.58 |
| 8 | 23.21 / 23.25 | 21.04 / 21.04 | 24.85 / 24.87 | 22.36 / 22.41 | 20.52 / 20.53 |
| 16 | 39.84 / 39.89 | 37.99 / 37.98 | 45.22 / 45.25 | 41.55 / 41.61 | 38.01 / 38.04 |
| 32 | 72.85 / 72.87 | 71.52 / 71.55 | 78.95 / 79.00 | 75.79 / 75.85 | 72.70 / 72.71 |
| 64 | 139.50 / 139.54 | 138.61 / 138.61 | 152.34 / 152.46 | 145.92 / 146.07 | 140.64 / 140.65 |
| 128 | 272.71 / 272.73 | 272.48 / 272.50 | 298.13 / 298.36 | 288.02 / 288.04 | 275.77 / 275.81 |

At B32, persistent work reuse is about 10.4% slower than the previous CUDA kernel; adding async copies reduces that regression to about 6.0%. All new variants lose to the previous CUDA kernel at B32/64/128 in both sweeps. The B8 two-tile uncapped async variant is about 2.5% faster than previous CUDA and 13% faster than stock in the kernel fixture. No new end-to-end speedup is claimed, and no new kernel worker was installed: there was no winning concurrent-batch candidate to advance to the engine comparison.

## Correctness and safety

All 16 variants passed 42 matrix cases, four pipeline-specific edge cases, and 256 changing-input CUDA Graph steps against both the independent Torch oracle and the installed reference. Tolerances are unchanged: state rtol=3e-4/atol=3e-5, output rtol=.016/atol=.002. Two additional edge cases per variant exercise the previous adapter fallback, not a new pipeline fallback. New native metadata guards retain the previous contract; unique positive in-range slot indices are a caller precondition. Memcheck and racecheck each exercised 144 calls: 0 errors; racecheck also reported 0 warnings/hazards.

## Measured hardware counters

Nsight Compute 2026.1.1 captured 25 labelled GDN/SiLU cases, using 20 replay passes per kernel and SpeedOfLight, MemoryWorkloadAnalysis, Occupancy, LaunchStats, WarpStateStats, and SourceCounters. This is a synthetic shape-matched fixture, not counters sampled from inside the serving worker. Default NCU cache/clock behavior applies; diagnostic durations are excluded from speedup claims. Driver counter permissions remained unchanged; only the profiler/fixture ran elevated, with owned caches confined to the lab.

At B32, stock GDN measured 2.613 TB/s and 77.98% DRAM throughput, with 11.35% achieved occupancy and 209 registers/thread. Previous CUDA measured 2.662 TB/s, 79.49%, 65.25% occupancy, and 40 registers/thread. Lower register use improved occupancy dramatically but barely changed large-batch bandwidth/time. These measured cold-replay counters replace the earlier arithmetic-only bandwidth estimate for this experiment; they do not establish an absolute ceiling.

The capped 8-tile direct kernel measured 2.393 TB/s and 23.42% occupancy. Async staging raised it to 2.556 TB/s while using 16 KiB shared memory. Long-scoreboard cycles per issued instruction decreased from 18.96 to 13.03, while short-scoreboard rose from 2.39 to 3.85. Tensor-pipe utilization is zero in these recurrence/activation kernels; full-model GEMMs are separate. At M16384 the earlier SiLU kernel measured 2.037 TB/s versus 0.904 TB/s for stock. All raw counters and units are retained.

## Nsight Systems

Nsight Systems 2025.6.3 captured eager/graph kernel fixtures plus actual warmed vLLM prefill and decode workloads. Engine configuration is unchanged from `batch64_t16384`: FP8 weights, BF16 activations, FP32 state, max_num_seqs=64, 16 GiB cache, FULL_AND_PIECEWISE graphs. CPU sampling/context-switch tracing was disabled; CUDA, NVTX and OS runtime traces were enabled. Profiled request times are not throughput evidence.

The decode workload also includes initial prefill. The analyzer selects 10 CUDA graph launches containing GDN recurrence: 11,690 kernel events, including 480 GDN calls. GDN accounts for 20.44% of summed kernel durations; dense FP8 GEMMs account for 63.25%. These are summed GPU duration shares, not fractions of serving wall time. The separate prefill trace has 16,207 kernel events; its largest categories are dense FP8 GEMMs and SiLU/quantization.

In the micro trace, stock B1 mean inter-kernel gaps were 37.79 us eager versus 2.28 us with one-kernel graphs. At B32 they were 1.58 us eager versus 1.80 us graphed. These are instrumented Python fixtures, not serving speedups or pure CPU-overhead measurements. They show why graphs help small work more readily than a bandwidth-heavy recurrence. `analysis.json` retains interval-union calculations, CUDA API timings and exact graph correlations.

## Raw evidence

- Numerical validation: `results/20261001T131439.353720Z` (initial missing-source RED also preserved).
- NCU report/CSV/details: `results/20261001T131906.154994Z`; failed privilege-switch wrapper attempt: `results/20261001T131601.992791Z`.
- Memcheck/racecheck: `results/20261001T131942.649030Z`.
- Timing sweeps: `results/20261001T132029.342274Z` and `results/20261001T132744.505479Z`.
- Nsight Systems fixtures: `results/20261001T132126.515474Z`.
- Nsight Systems actual engine: `results/20261001T132220.555573Z`, two `.nsys-rep` reports, SQLite exports, CSV statistics and engine metadata.

## Reproduction

From the lab host, use the pinned Python 3.14.6 interpreter with:

```sh
python -m pipeline_pass.run --kernel-action pipeline-validate
python -m pipeline_pass.run --kernel-action pipeline-sanitize
python -m pipeline_pass.run --kernel-action pipeline-bench
PIPELINE_DIAGNOSTIC=compute python -m pipeline_pass.run --kernel-action pipeline-profile
PIPELINE_DIAGNOSTIC=systems python -m pipeline_pass.run --kernel-action pipeline-profile
PIPELINE_DIAGNOSTIC=engine python -m pipeline_pass.run --kernel-action pipeline-profile
```

Every action passes through the shared suite lock and idle-GPU guard. Commands, errors and raw artifacts are preserved. An independent review found a timeout cleanup bug; owned process-group termination now covers privileged targets, timeouts, and ordinary parent failure. Regression tests and the full CPU suite pass.

## Related paper

A suggested reference is [Megatron-LM (2019), section 3](https://arxiv.org/pdf/1909.08053). Its central optimization partitions MLP matrices and attention heads across GPUs while reducing synchronization. That is directly relevant to multi-GPU execution, not a direct solution for this one-H100 state-bandwidth problem. The transferable principle is reducing data movement. Lower-precision recurrent state or a different algorithm would require a separate accuracy/performance study and was not tested here.

NVIDIA method references: [Nsight Compute profiling guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html), [async copies](https://docs.nvidia.com/cuda/archive/13.1.1/cuda-programming-guide/04-special-topics/async-copies.html).

## Preservation and cleanup

All remote files and the tracked compilation caches were hash-verified locally
before removal of the temporary lab workspace. The remote lab is absent, the
GPU has no compute processes, the production service remains active, and
RmProfilingAdminOnly remains 1. A cache audit found no modified files outside
the lab in the checked cache/config directories.

Final CPU suite: 23 passed (`CPU_TESTS.log`). Running the same suite with the
required process-inspection permission passed. The review's cleanup issue is
covered by timeout and failed-parent regression tests.
