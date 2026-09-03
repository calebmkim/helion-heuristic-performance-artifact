# Scripts

These are the adapters used for the included run:

- `run_benchmark.py` constructs the manifest workloads and co-measures raw
  default, heuristic seed, and optionally a Triton-only `torch.compile`
  max-autotune reference in a fresh subprocess per shape.
- `summarize_results.py` writes summary JSON, Markdown, and per-shape CSV.
- `plot_results.py` creates one graph containing every matmul kernel family.
- `plot_blog_figures.py` creates the publication-style graph used by the
  compile-time-heuristics blog artifacts.
- `starter.sh` composes those steps.

They accept checkout, GPU, manifest, and output paths at runtime. Treat them as
starting points when the current Helion examples or configuration APIs differ.
`starter.sh` includes the PyTorch arm by default; set
`INCLUDE_TORCH_COMPILE=0` for the faster original two-arm run. Both plotters
accept `--baseline default` and `--baseline torch_compile`.
