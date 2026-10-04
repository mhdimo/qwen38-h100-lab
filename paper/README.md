# LaTeX papers

Three compiled reports covering the October 2026 single-H100 study of a
Qwen3.8-27B-FP8 vLLM deployment. All three are authored by Mihal Dimo.

- `kernel.tex` / `kernel.pdf` — the custom FP8 SiLU/multiply/quantization CUDA
  kernel: target selection from GPU profiles, numerical validation, the
  operation microbenchmark, and the matched end-to-end comparison. Rebuild its
  numbers with
  `python3 paper/kernel_build_data.py --engine-run results/20261001T121440.767119Z`,
  then compile `kernel.tex` with [Tectonic](https://tectonic-typesetting.github.io/)
  (`tectonic kernel.tex`).
- `gdn.tex` / `gdn.pdf` — the Gated DeltaNet decode-kernel ablation. Rebuild with
  `python3 paper/gdn_build_data.py --engine-run results/20261001T125154.014415Z`,
  then compile with Tectonic. Concurrent decode throughput is effectively
  unchanged despite much lower register use; the report preserves that negative
  result and the mixed-latency regression.
- `paper.tex` / `paper.pdf` — the earlier serving-configuration study
  (concurrency, cache and token-budget). Rebuild with `python3 paper/build_data.py`,
  then compile with Tectonic. Its large gains are configuration changes and must
  not be attributed to the custom CUDA kernels.

The `*_build_data.py` scripts reject incomplete runs, mismatched
workload/repetition definitions, and different model/runtime versions. They
generate `kernel_metrics.tex`, `gdn_metrics.tex`, `metrics.tex`,
`throughput.csv` and the `*_data.json` summaries from the raw files under
`results/`. Generating the papers and analyzing the data require no GPU.

The later persistent/async and Nsight profiling pass is documented in
[pipeline_pass/README.md](../pipeline_pass/README.md). The original GDN paper
describes the earlier pass before hardware counters were collected; its
archived results remain unchanged.
