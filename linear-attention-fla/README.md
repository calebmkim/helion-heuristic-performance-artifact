# Linear Attention: Helion vs FLA

This comparison measures the current Helion linear-attention workload using
four configurations:

| Arm | Meaning |
|---|---|
| `default` | Helion's base configuration, without compiler heuristics or tuned results |
| `fla_triton` | The corresponding handwritten FLA Triton implementation |
| `seed` | Helion with autotuning disabled and compiler heuristics enabled |
| `aot_tuned` | Helion's checked-in architecture-specific pre-tuned selector |

Helion already has a linear-attention comparison runner. At the reference
revision it was `benchmarks/run_linattn.py`; an agent should locate the current
equivalent and follow its imports rather than assuming that path or API is
unchanged. FLA is available at
<https://github.com/fla-org/flash-linear-attention>.

## Reproduce

Give [AGENT_PROMPT.md](AGENT_PROMPT.md) to an agent with the requested Helion
and FLA revisions. The [scripts](scripts/README.md) are the actual adapters used
to reproduce that workflow; the runner is generalized from the exact adapter
used for the prior result. They discover checkouts and output locations at
runtime, and should be adapted when the current Helion or FLA APIs differ.

No particular checkout layout is required. Script inputs come from command-line
arguments or environment variables; [scripts/starter.sh](scripts/starter.sh)
shows one composition of the pipeline.

The resulting report should include:

- correctness and absolute latency for each current workload shape;
- the timing method already used by the current Helion linear-attention
  harness, applied consistently to every arm;
- higher-is-better geometric-mean comparisons of each baseline with the
  heuristic seed;
- a per-kernel clustered bar graph, with separate forward and
  forward-plus-backward panels, default/seed/AOT-tuned bars, and FLA fixed at
  the `1.00x` reference line;
- the number of shapes in each comparison;
- unsupported FLA operations and unavailable AOT-tuned operations;
- the adapted runner and raw output needed to audit or repeat the run.

Do not silently replace a missing FLA implementation or AOT-tuned
configuration with another baseline.

Each Helion arm runs in its own long-lived process to keep import-time
configuration and kernel caches isolated. FLA is timed beside that arm, and
the plotted bar uses the paired `FLA latency / arm latency` ratio. This makes
FLA the common `1.00x` calibration reference across the separate arm processes.

## Included H100 Run

The corrected H100 run includes the
[summary](generated/h100-corrected-aot/summary.md),
[per-kernel graph](generated/h100-corrected-aot/per-kernel-bars.png), and
[combined raw results](generated/h100-corrected-aot/results.json).
[H100_PR3546.md](reference-results/H100_PR3546.md) records why the earlier AOT
comparison was invalidated.
