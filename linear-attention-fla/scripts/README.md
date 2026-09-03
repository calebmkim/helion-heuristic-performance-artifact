# Starter Scripts

These scripts form an adaptable pipeline:

1. `discover_manifest.py` reads the workload from the target Helion checkout.
2. `run_benchmark.py` measures one Helion arm and FLA in a long-lived process.
3. `combine_results.py` joins the raw arm outputs.
4. `summarize_results.py` writes higher-is-better tables.
5. `plot_results.py` plots per-kernel geometric means relative to FLA.
6. `starter.sh` shows how those pieces were composed.

`run_benchmark.py` is generalized from the exact adapter used for the retained
H100 PR #3546 result. It is a starting point, not a stable interface. Inspect
the current Helion harness and adapt moved imports or semantics. Checkout paths
are supplied through CLI flags or `HELION_ROOT`/`FLA_ROOT`; no checkout layout
is assumed.

The plot has separate forward and forward-plus-backward panels. Each kernel is
a cluster with default, seed, and AOT-tuned bars, where bar height is
`paired FLA latency / arm latency`; the FLA reference is the `1.00x` line.
For schema compatibility, `combine_results.py` also retains the seed process's
FLA result as a standalone arm. Plots and calibrated comparisons use each
Helion arm's own paired FLA timing.
