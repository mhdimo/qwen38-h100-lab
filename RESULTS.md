# Completed measurements and tradeoffs

The GDN follow-up is also measured: see [the CUDA source](gdn_pass/gdn_decode.cu)
and [GDN report source](paper/gdn.tex). The selected tile reduces registers from
205 to 40/thread and passes numerical, graph and memory checks, but concurrent
decode throughput is effectively unchanged (-0.04%). Mixed throughput rises
1.38% while mixed TPOT p95 worsens (18.40 to 21.48 ms). Stock GDN remains the
preferred baseline for this throughput objective. These results are a separate
ablation, not combined with the SiLU gains below.

This file records the original serving-configuration phase. A subsequent,
user-authorized custom CUDA kernel pass is complete: see
[the kernel report](paper/kernel.pdf) and [implementation](custom_kernels/README.md).
Its incremental throughput gains against tuned vLLM are +5.90% prefill, +0.93%
decode and +3.65% mixed. The remote lab has since been verified locally and removed.

All remaining GPU experiments were stopped once the study scope was frozen, and
the lab was copied to a local archive. Transfer integrity: 43 files were
checked against remote SHA-256 hashes. The production service stayed up.

The three complete configurations each ran 400 measured synthetic requests,
with two repetitions per workload, unchanged Qwen3.8-27B-FP8 weights, BF16 cache,
and vLLM 0.26.0 on the same H100 80GB.

| Metric | Baseline | 8,192-token budget | 16,384-token budget |
|---|---:|---:|---:|
| Prefill-heavy input tokens/s, concurrency 32 | 11,919.10 | 17,683.68 | 18,092.46 |
| Decode-heavy output tokens/s, concurrency 32 | 74.37 | 1,647.90 | 1,645.66 |
| Mixed output tokens/s, concurrency 32 | 71.70 | 996.70 | 1,003.64 |
| Single-request output tokens/s | 74.31 | 72.83 | 72.49 |
| Single-request TTFT p50, ms | 62.89 | 98.54 | 98.00 |
| Decode-heavy TPOT p50, ms | 13.06 | 16.92 | 16.96 |
| Mixed TTFT p50, ms | 55,448.50 | 1,519.94 | 1,779.62 |
| Highest end-of-case memory sample, GiB | 33.36 | 48.81 | 50.97 |

The 16,384 candidate gives **22.13× decode-heavy aggregate throughput**,
**14.00× mixed throughput**, and **51.79% more prefill-heavy input throughput**
on these matched shapes. Single-request throughput drops 2.45%, and first-token
latency at concurrency one increases. Memory use and per-request token latency
also increase. The 8,192 candidate uses less memory and has lower median mixed TTFT.

The baseline reproduces deployed engine limits (one sequence, 2 GiB cache,
prefill graph policy) through the same asynchronous harness used for candidates.
It queues concurrent requests. The real production wrapper instead rejects
overlapping requests with a busy lock; these numbers are not production HTTP
throughput. Candidates allow 64 sequences, use 16 GiB cache, and capture decode
batches through 64. The main gain combines these changes; it is not a custom
kernel speedup. The token-budget comparison changes only that budget.

The 2,048-token candidate was cancelled after six of eight case repetitions;
its partial results are preserved and excluded here. Higher-concurrency runs,
GDN backend ablation and kernel profiles never started. The results must not be
read as kernel bottleneck attribution, a maximum supported concurrency, or a universal optimum.

The model passed three text smoke checks, but no production multimodal or
structured-output quality evaluation was completed. Two repetitions, fixed
synthetic lengths, closed-loop traffic and point resource samples limit conclusions.

Full methods, latency distributions, interpretation and reproducibility are in
[the LaTeX paper](paper/paper.tex). Raw sources are listed in
`paper/source-checksums.txt`. The paper's `build_data.py` generates its figures
and numeric macros directly from those files.


## Persistent/async and Nsight follow-up (1 October 2026)

Completed 16 scheduling/copy variants, both Nsight tools, memcheck and racecheck. No new B32/64/128 win over previous CUDA; a B8-only ~2.5% microbenchmark improvement was repeated. See [full evidence and tradeoffs](pipeline_pass/README.md) and [derived counter/timeline data](pipeline_pass/analysis.json). Prior SiLU/engine results are unchanged.
