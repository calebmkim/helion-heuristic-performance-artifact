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
and FLA revisions. The [scripts](scripts/README.md) are the adapters used for
the included run. They discover checkouts and output locations at runtime, and
should be adapted when the current Helion or FLA APIs differ.

No particular checkout layout is required. Script inputs come from command-line
arguments or environment variables; [scripts/starter.sh](scripts/starter.sh)
shows one composition of the pipeline.

The pipeline has two phases:

1. In isolated discovery processes, record the default and heuristic-seed
   configs and resolve the existing architecture AOT selector for every
   workload cell. This does not tune a new AOT table.
2. For each cell, start one fresh process and explicitly replay all three
   Helion config sets beside FLA. All four arms therefore share inputs,
   compilation state, GPU state, and one FLA timing for that cell. Their
   cold-L2 CUDA-event samples are interleaved in rotated forward/reverse order.

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

## Included H100 Run

The same-process H100 run includes the
[summary](generated/h100-same-process/summary.md),
[per-kernel graph](generated/h100-same-process/per-kernel-bars.png),
[blog-style graph](generated/h100-same-process/blog-figures/results-linattn-h100.png),
[combined raw results](generated/h100-same-process/results.json), and
[replay manifest](generated/h100-same-process/config-replay.json).
[H100_PR3546.md](reference-results/H100_PR3546.md) records why the earlier AOT
comparison was invalidated.
