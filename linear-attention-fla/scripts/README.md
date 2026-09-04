# Linear-Attention Scripts

Both launchers use the repository's shared primary-stack checker and fail
before benchmarking unless they use a clean Helion checkout at
`fa2f62eb686ef846c76f8b9e18beec30fbc5bee1` and import PyTorch
`2.13.0+cu132`, CUDA `13.2`, and Triton `3.7.1`. Override the required
revisions only for an explicitly labeled historical run.

## Native Helion Harness

`run_native_harness.sh` is a thin launcher for the target checkout's own
`benchmarks.run_linattn` entry point. Use it to reproduce results that were
claimed using Helion's native benchmark methodology. It deliberately leaves
Helion config selection to the caller's environment and produces the native
`helionbench.json` format.

The native harness measures one active Helion configuration against FLA. It
does not co-measure default, seed, and AOT, and separate native invocations
should not be merged into a purported same-process four-arm comparison.
Use a fresh `OUTPUT_DIR`: some revisions of the native writer append to an
existing `helionbench.json`.

## Controlled Four-Arm Comparison

The remaining scripts form an adaptable pipeline:

1. `discover_manifest.py` reads the workload from the target Helion checkout.
2. `materialize_configs.py` discovers default, seed, and existing AOT selections
   in isolated processes and writes an explicit replay manifest.
3. `run_benchmark.py` starts one fresh process per cell and co-measures all
   three replayed Helion arms with FLA.
4. `summarize_results.py` writes higher-is-better tables.
5. `plot_results.py` plots per-kernel geometric means relative to FLA.
6. `plot_blog_figures.py` creates the blog-style H100 figure and can combine it
   with the B200 per-cell CSV.
7. `run_interleaved_comparison.sh` shows how those pieces are composed.

For a small compatibility run, set `VARIANTS`, `SHAPE_NAMES`, `MODES`, or
`LIMIT`; the launcher passes those filters to manifest discovery before config
materialization. For example, four dense variants at one shape produce eight
paired forward/forward-backward cells:

```bash
VARIANTS=vanilla_linear_attn,full_gla,gated_delta_rule,kda \
SHAPE_NAMES=B1_T8192_H96_D128 \
linear-attention-fla/scripts/run_interleaved_comparison.sh
```

The materializer and runner reuse Helion's example harness for workload
construction, FLA calls, gradients, tolerances, and timing conventions. The
replay layer only intercepts constituent Helion kernel dispatch. It audits
argument signatures and call order once, caches each explicit compiled
callable, and uses that cached callable while timing so config lookup is not
added to reported device latency. The runner then interleaves cold-L2 CUDA
event samples in rotated forward/reverse pairs, clearing gradients before the
start event for backward cells. The narrow timing adapter is needed because
Helion's generic interleaved helper uses a fixed arm order and cannot clear
backward gradients outside its timed region.

These are starting points, not stable interfaces. Inspect the current Helion
harness and adapt moved imports or semantics. Checkout paths are supplied
through CLI flags or `HELION_ROOT`/`FLA_ROOT`; no checkout layout is assumed.
Use a fresh output directory for a different revision.
`run_interleaved_comparison.sh` resumes only when explicitly invoked with
`RESUME=1`; doing so is valid only when the checkouts and workload are
unchanged.

The plot has separate forward and forward-plus-backward panels. Each kernel is
a cluster with default, seed, and AOT-tuned bars, where bar height is
`shared FLA latency / arm latency`; the FLA reference is the `1.00x` line.

`combine_results.py` remains only for reading older separate-arm runs. It is not
part of the current reproduction path.
