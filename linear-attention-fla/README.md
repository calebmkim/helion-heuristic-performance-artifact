# Linear Attention: Helion vs FLA

> **For all new reproductions, use PyTorch `2.13.0+cu132` (CUDA `13.2`)
> with Triton `3.7.1`.**

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

## Choose a Workflow

Give [AGENT_PROMPT.md](AGENT_PROMPT.md) to an agent with the requested Helion
and FLA revisions. There are two intentionally different workflows.

### Native Helion Harness

Use [scripts/run_native_harness.sh](scripts/run_native_harness.sh) when the goal
is to reproduce a performance claim made with Helion's own linear-attention
runner. It invokes the checkout's runner directly and preserves its workload,
timing order, and output format. The native runner measures FLA against the one
Helion configuration selected by that checkout and its environment; it does
not compare default, seed, and AOT in one invocation.

Pin the Helion, FLA, PyTorch, and Triton revisions used by the claim. Also
preserve its Helion configuration environment: the native runner does not
label or change the active config. Do not combine separate native invocations
into a four-arm result and describe it as same-process measurement.

### Controlled Four-Arm Comparison

Use
[scripts/run_interleaved_comparison.sh](scripts/run_interleaved_comparison.sh)
when the goal is to compare default, FLA, seed, and AOT under one controlled
measurement protocol. The pipeline has two phases:

1. In isolated discovery processes, record the default and heuristic-seed
   configs and resolve the existing architecture AOT selector for every
   workload cell. This does not tune a new AOT table.
2. For each cell, start one fresh process and explicitly replay all three
   Helion config sets beside FLA. All four arms therefore share inputs,
   compilation state, GPU state, and one FLA timing for that cell. Their
   cold-L2 CUDA-event samples are interleaved in rotated forward/reverse order.

The split is intentional. Config discovery must isolate Helion's import-time
selection state, while performance measurement should hold process and GPU
state constant across the compared arms. Every replay arm is compiled and
checked before timing; backward gradients are cleared before the start event,
and every arm receives the same number of retained samples.

This supersedes the older separate-arm method. That method measured FLA and
each Helion arm in blocks in different long-lived processes. Its row ordering
also caused every dense backward cell to time the complete FLA block before
the Helion block. Per-sample rotation removes that fixed order, and a single
shared FLA measurement removes cross-process denominator drift.

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

## Primary Software Stack

Linear attention, vLLM reductions, and example reductions now share one
primary reproduction stack:

- PyTorch `2.13.0+cu132`
- CUDA `13.2`
- Triton `3.7.1`

Both linear-attention launchers call the repository's shared stack checker and
fail before benchmark imports when the environment differs. Required versions
may be overridden only for an explicitly labeled historical run.

## Unified-Stack Compatibility Check

An 8-cell H100 check reran four representative variants in both forward and
forward-plus-backward on the primary stack. It held the Helion and FLA
revisions fixed against the existing PyTorch 2.12 result. All cells completed,
all Helion arms passed correctness, and aggregate performance-versus-FLA moved
by at most `0.05%` across default, seed, and AOT.

The [compatibility artifact](generated/h100-torch213-triton371-compatibility-smoke/)
also verifies the separate native Helion timing workflow. This bounded check
validates the stack transition; it does not replace the full-population run.

## Included Full H100 Reference Run

The checked-in 96-cell H100 reference run predates the unified PyTorch pin and
used:

| Component | Version or revision |
|---|---|
| Helion | `eacfee67c0fdbc5a1c068f16a3b2f9f15ce23eb7` |
| FLA | `0.5.2` at `6bd90692588c81fe102ee6e12ac70686359658a2` |
| PyTorch | `2.12.0+cu132` |
| Triton | `3.7.1` |
| PyTorch CUDA build | `13.2` |
| GPU | NVIDIA H100 80GB HBM3, compute capability 9.0 |

All 96 workload cells completed successfully, with all replayed Helion arms
passing correctness. This is a controlled four-arm run, not a native-harness
run. Its recorded PyTorch version is historical and must not be relabeled.
It includes the
[summary](generated/h100-same-process/summary.md),
[per-kernel graph](generated/h100-same-process/per-kernel-bars.png),
[blog-style graph](generated/h100-same-process/blog-figures/results-linattn-h100.png),
[combined raw results](generated/h100-same-process/results.json), and
[replay manifest](generated/h100-same-process/config-replay.json).

[H100_SUPERSEDED_RUNS.md](reference-results/H100_SUPERSEDED_RUNS.md) records
why the earlier H100 artifacts must not be used. The plotting script also
accepts the B200
[per-cell CSV](https://github.com/calebmkim/helion/blob/pytorch-blog-heuristics-results/PYTORCH_BLOG_RAW_DATA/linear_attention_e2e_per_cell.csv)
to regenerate the cross-GPU blog figures.
