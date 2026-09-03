# Starter Scripts

The pipeline is:

1. `run_benchmark.py` expands `shapes.json`, runs one child per cell, and
   compiles/captures/times all three arms together.
2. `summarize_results.py` writes higher-is-better per-kernel geomeans and a
   raw per-cell CSV.
3. `plot_results.py` plots those geomeans with Torch compile at `1.00x`.
4. `starter.sh` runs the complete sequence.

`workloads.py` owns the exact argument construction, Torch references,
observable outputs, and tolerances for the ten kernels.

RMSNorm backward uses `rtol=0.03, atol=0.08` for BF16 `grad_x`. The absolute
tolerance covers reduction-order cancellation near zero; its mean absolute
error in the included run's diagnostic was `0.00339`. All correctness checks
run before capture and are outside the timed interval.

These scripts are starting points, not stable interfaces. Adapt them when the
target Helion revision changes APIs, while preserving arm semantics and the
fixed shape population.
