# Starter Scripts

These scripts form an adaptable pipeline:

1. `discover_manifest.py` reads the workload from the target Helion checkout.
2. `materialize_configs.py` discovers default, seed, and existing AOT selections
   in isolated processes and writes an explicit replay manifest.
3. `run_benchmark.py` starts one fresh process per cell and co-measures all
   three replayed Helion arms with FLA.
4. `summarize_results.py` writes higher-is-better tables.
5. `plot_results.py` plots per-kernel geometric means relative to FLA.
6. `starter.sh` shows how those pieces are composed.

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

The plot has separate forward and forward-plus-backward panels. Each kernel is
a cluster with default, seed, and AOT-tuned bars, where bar height is
`shared FLA latency / arm latency`; the FLA reference is the `1.00x` line.

`combine_results.py` remains only for reading older separate-arm runs. It is not
part of the current reproduction path.
