# Linear Attention: Helion vs FLA

> **For all new reproductions, use Helion
> `fa2f62eb686ef846c76f8b9e18beec30fbc5bee1`, PyTorch `2.13.0+cu132`
> (CUDA `13.2`) with Triton `3.7.1`.**

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
Helion configuration selected by that checkout and its environment; the
artifact launcher defaults to the checked-in AOT selector used by Helion's
current benchmark workflow. It does not compare default, seed, and AOT in one
invocation.

The pinned main AOT table has one syntactically stale no-op field on the varlen
KDA output kernel: `range_num_stages: [0]` remains after the kernel stopped
accepting a staged range. Both artifact workflows narrowly normalize it to
`[]`, reject any unexpected source value, and record the repair. No
performance-bearing AOT setting or heuristic seed is changed.

Pin the Helion, FLA, PyTorch, and Triton revisions used by the claim. The
artifact launcher selects and labels the checked-in AOT configuration by
default; an explicitly different native configuration must override both the
selection environment and its output label. Do not combine separate native
invocations into a four-arm result and describe it as same-process
measurement.

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

All four comparisons in this repository share one primary reproduction stack:

- Helion `fa2f62eb686ef846c76f8b9e18beec30fbc5bee1`
- PyTorch `2.13.0+cu132`
- CUDA `13.2`
- Triton `3.7.1`

Both linear-attention launchers call the repository's shared stack checker and
fail before benchmark imports when the Helion checkout is dirty or any required
revision differs. Required revisions may be overridden only for an explicitly
labeled historical run.

## Fresh H100 Results

Both timing workflows were rerun on an NVIDIA H100 80GB HBM3 with the primary
software stack and FLA `0.5.2` at
`6bd90692588c81fe102ee6e12ac70686359658a2`.

### Controlled Four-Arm Result

All 96 workload cells completed all four arms, and all 288 replayed Helion
arm-cells passed correctness.

| Population | Cells | Default | Seed | AOT-tuned | FLA Triton |
|---|---:|---:|---:|---:|---:|
| Overall | 96 | 0.5445x | 1.1573x | 1.2888x | 1.0000x |
| Forward | 54 | 0.5235x | 1.1843x | 1.2612x | 1.0000x |
| Forward + backward | 42 | 0.5729x | 1.1235x | 1.3252x | 1.0000x |

- [summary](generated/h100-fa2f62eb-torch213-triton371-interleaved/summary.md)
- [per-kernel graph](generated/h100-fa2f62eb-torch213-triton371-interleaved/per-kernel-bars.png)
- [blog-style graph](generated/h100-fa2f62eb-torch213-triton371-interleaved/blog-figures/results-linattn-h100.png)
- [combined raw results](generated/h100-fa2f62eb-torch213-triton371-interleaved/results.json)
- [replay manifest](generated/h100-fa2f62eb-torch213-triton371-interleaved/config-replay.json)

### Native Helion Result

The native block-ordered harness accepted all 96 cells and recorded a positive
Helion latency for every cell. Its single active Helion arm is the checked-in
H100 AOT selector, with the recorded no-op compatibility repair described
above.

| Population | Cells | Active Helion / FLA |
|---|---:|---:|
| Overall | 96 | 1.2860x |
| Forward | 54 | 1.2409x |
| Forward + backward | 42 | 1.3464x |

- [summary](generated/h100-fa2f62eb-torch213-triton371-native/summary.md)
- [per-kernel graph](generated/h100-fa2f62eb-torch213-triton371-native/per-kernel-bars.png)
- [native raw output](generated/h100-fa2f62eb-torch213-triton371-native/helionbench.json)
- [provenance](generated/h100-fa2f62eb-torch213-triton371-native/provenance.json)

These results must remain separate. The native path measures one selected
Helion configuration in blocks using Helion's own harness; the controlled path
interleaves four arms per sample and shares one FLA denominator within each
cell. Their AOT overall geomeans agree closely (`1.2860x` native versus
`1.2888x` controlled), but that agreement does not make their per-cell samples
interchangeable.

The plotting script also accepts the B200
[per-cell CSV](https://github.com/calebmkim/helion/blob/pytorch-blog-heuristics-results/PYTORCH_BLOG_RAW_DATA/linear_attention_e2e_per_cell.csv)
to regenerate the cross-GPU blog figures.
