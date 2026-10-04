# Qwen3.8-27B-FP8 on a single H100 — serving study, a custom FP8 CUDA kernel, and honest negatives

Measured optimization study of a production Qwen3.8-27B-FP8 vLLM deployment
(vLLM 0.26.0, PyTorch 2.11.0, Triton 3.6.0) on one NVIDIA H100 80GB HBM3.
Four passes, each with retained raw evidence and explicit limits:

| Pass | Question | Result |
|---|---|---|
| 1. Serving configuration | What does the engine-capacity limit cost concurrent users? | Enabling concurrency (sequence capacity, KV cache, token budget) lifts decode-heavy aggregate throughput to **22.13×** the deployed single-sequence engine — a *configuration* gain, **not** a kernel speedup |
| 2. Custom CUDA kernel | Can the prefill activation + FP8 quantization op be beaten? | Warp-level SM90 kernel, **byte-exact** vs the installed vLLM operator across 168 reference comparisons → **2.26×** at the 16,384-row shape and **+5.90% prefill / +3.65% mixed** full-model throughput vs the already-tuned engine |
| 3. Gated DeltaNet decode kernel | Can the decode recurrence be beaten? | **No.** Registers 205 → 40/thread, but concurrent decode throughput is flat (−0.04%); mixed TPOT p95 regresses. Negative result kept, with full evidence |
| 4. Persistent/async + Nsight follow-up | Do scheduling/async-copy variants win at large batches? | **No** (best case ≈2.5% at B8 in the microbenchmark only). Nsight Compute counters and Nsight Systems timelines retained |

Headline document: [`paper/kernel.pdf`](paper/kernel.pdf).
Companion reports: [`paper/paper.pdf`](paper/paper.pdf) (serving configuration),
[`paper/gdn.pdf`](paper/gdn.pdf) (GDN ablation).

## The kernel (pass 2)

Warmed PyTorch profiles of the serving stack (selected via CUDA-graph launch
correlation) put fused SiLU/multiply/per-group FP8 quantization at **14.54%** of
prefill GPU kernel time. [`custom_kernels/silu_mul_fp8.cu`](custom_kernels/silu_mul_fp8.cu) replaces it:

- one warp per 128-value quantization group, four adjacent values per lane;
- warp-shuffle-only max reduction — no block barriers, no shared-memory reductions, 29 registers/thread;
- packed 4-byte FP8 stores; compact and transposed scale layouts; partial last blocks;
- FP32 arithmetic before the final cast; identical scale floor and optional upper-bound semantics as the installed operator.

Verification: 216 numerical cases, including 168 nonempty byte-exact FP8
comparisons against the installed CUDA operator and 48 empty-input checks,
FP32 scale agreement at rtol 2e-6 / atol 1e-8 (observed difference: zero),
edge values, both scale layouts, and changing-input CUDA-graph replays. An
initial fast-math build failed byte-exact comparison; the failure log is
retained and the flag was removed.

Serving impact (400 matched synthetic requests per config, two repetitions,
identical prompts and lengths): **+5.90% prefill**, **+0.93% decode**,
**+3.65% mixed**, **−0.15% single-request** (the adapter keeps the stock kernel
at batch one, where the standalone candidate is slightly slower).

## Repository map

- `benchmark.py`, `measurement.py`, `run_suite.py`, `report.py`, `trace_summary.py` — the measurement harness (async closed-loop client, GPU-idle guard, shared lab lock, profiler-trace summarizer). Unit tests live at the root (`test_*.py`).
- `configs/` — engine configurations compared in pass 1 and the opt-in kernel configs of passes 2–3.
- `custom_kernels/` — pass 2: the FP8 SiLU kernel, its verification/benchmark driver, and the serving-worker substitution.
- `gdn_pass/` — pass 3: the Gated DeltaNet decode kernel (selected tile plus variants) and its negative-result evidence.
- `pipeline_pass/` — pass 4: persistent/async scheduling variants, Nsight Compute counters and Nsight Systems timelines.
- `paper/` — the three LaTeX reports and the scripts that regenerate every number from raw evidence.
- `results/` — raw per-run JSON, logs and manifests for every configuration above.
- [`RESULTS.md`](RESULTS.md) — compact results summary; [`KERNEL_WORKFLOW.md`](KERNEL_WORKFLOW.md) — the measurement methodology used throughout.

## Reproducing

Everything requires an NVIDIA GPU (SM90 for the kernels) with the pinned stack
above and a local Qwen3.8-27B-FP8 checkpoint (`LAB_MODEL_PATH`). Regenerating
the papers and analyzing the included results needs no GPU. Command sequences
are in each pass README and in the papers themselves.

## Limits — read before quoting numbers

- **22.13×** is an engine-configuration change (sequence capacity 1 → 64, cache 2 → 16 GiB, graph capture policy) measured against a reproduced baseline that *queues* concurrent requests. It is not a kernel speedup and not production HTTP throughput.
- The kernel gains are incremental against an already-tuned engine: two repetitions, synthetic fixed-length traffic, unlocked clocks. Small differences are not significance-tested.
- The GDN and persistent/async passes are preserved negative results; stock GDN remains the preferred decoder at large batches.
- No production multimodal / structured-output quality validation was completed. Token-hash agreement against unchanged-reference controls is reported in the papers and does not certify determinism.

## Provenance

All artifacts were produced in one session on a rented, isolated H100 (1 October
2026); the temporary remote workspace was hash-verified locally and removed.
Large binary profiler artifacts (raw `.pt` trace exports, `.nsys-rep`/`.ncu-rep`
reports, SQLite exports) are kept in a local archive rather than this repository;
every number in the papers regenerates from the included JSON/log evidence via
the `paper/*_build_data.py` scripts.

## License

MIT (see [LICENSE](LICENSE)). This repository contains no vLLM source code; the
numerical and performance oracle is the installed vLLM 0.26.0 binary.
