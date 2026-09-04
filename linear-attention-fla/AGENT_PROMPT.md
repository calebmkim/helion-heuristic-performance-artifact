# Agent Prompt: Reproduce Linear-Attention Performance

## Inputs

```text
HELION_ROOT=<existing Helion checkout>
HELION_REVISION=<revision to measure>
FLA_ROOT=<FLA checkout or installed package>
FLA_REVISION=<revision or package version to measure>
ARTIFACT_ROOT=<this repository>
OUTPUT_DIR=<directory for scripts, raw results, and the report>
CUDA_VISIBLE_DEVICES=<GPU index>
```

## Important

The paths, flags, and starter script in this artifact reflect one previous
Helion revision. They are not a stable API and may be wrong for the requested
environment. Inspect the current checkouts and adapt them. Do not stop merely
because a command, module, or output format has changed.

## Task

Reproduce the current Helion linear-attention workload on the requested GPU and
compare:

1. Helion's base/default configuration, with compiler heuristics and tuned
   results disabled;
2. FLA's Triton implementation;
3. Helion's compiler heuristic with autotuning disabled;
4. Helion's architecture-specific AOT-tuned configuration.

Start by inspecting Helion's current linear-attention benchmark and its FLA
integration. Around PR #3546, `benchmarks/run_linattn.py` was the main entry
point, but use the current equivalent. Reuse the workload, input generation,
correctness checks, and timing utilities already present in Helion.

Use mutually compatible FLA and Triton revisions. In particular, do not
classify backward operations as inherently unavailable when an older FLA
version merely guards them on Hopper. Update the comparison environment or
record it as an environment incompatibility.

Use the files in `linear-attention-fla/scripts/` as starting points. The
orchestrator is `scripts/starter.sh`; modify or replace any adapter that no
longer matches the current checkout. The scripts show the intended arm
selection, but do not prove that those settings still have the same behavior.

Use the current benchmark population rather than a shape list copied from this
artifact. Include forward and forward-plus-backward operation measurements
where the current workload supports them.

First materialize a benchmark replay manifest. Use isolated discovery
processes for the three Helion config sources, execute complete workload cells,
and record every constituent kernel call and selected config:

- `default` must not accidentally receive a compiler heuristic or cached tuned
  result;
- `seed` should use the compiler-selected configuration without running a
  search;
- `aot_tuned` must use Helion's checked-in architecture-specific AOT selector
  with its normal selection semantics. Do not replace or restrict that
  selector in the benchmark adapter. Do not set `HELION_AUTOTUNE_EFFORT=none`
  for this arm: Helion then resolves an implicit compiler default before
  consulting `AOTAutotuneCache`. Audit selected bound configs against AOT cache
  results rather than inferring selection from correctness or timing.

Do not tune a replacement AOT table. The materialized file is a benchmark
replay manifest for the selected revision and workload, not a production AOT
selector.

Measure one workload cell per fresh process. In that process, explicitly replay
the materialized default, seed, and AOT configs and time FLA as a fourth arm.
Compile and audit every replay arm before timing, reject unplanned constituent
calls, and interleave cold-L2 device-event samples across the four arms in
rotated forward/reverse order. Clear backward gradients before each start
event. This is important: do not benchmark the Helion arms in separate
processes and then combine their results through separately observed FLA
timings.

Use Helion's existing correctness references and timing conventions. At the
reference revision, the helper used CUDA device events, cleared L2 before each
timed call, did not use CUDA Graph replay here, and reported the mean. Preserve
those choices in the narrow interleaved adapter; do not silently replace the
mean with nested median rounds. Save absolute per-shape latencies before
computing ratios.

Make all aggregate tables and plots higher-is-better. For a comparison against
the heuristic seed, use:

```text
heuristic speedup = baseline latency / heuristic latency
```

Compute geometric means only over shapes that are correct and available in
both compared arms, and state the shape count. Report overall results and each
kernel family separately. Include a four-way common-population comparison when
useful.

Create a clustered per-kernel bar graph with separate forward and
forward-plus-backward panels. Use the cell's shared FLA Triton measurement as a
`1.00x` horizontal reference and plot default, seed, and AOT-tuned as
`FLA latency / arm latency`.

Make reasonable implementation decisions when APIs have moved. Record the
important adaptations and any unsupported or failed cells instead of blocking
the entire run. Leave `OUTPUT_DIR` with:

- the adapted runner;
- the materialized config replay manifest;
- raw per-shape results;
- a short Markdown summary;
- the per-kernel bar graph;
- exact revisions, package versions, and GPU model.
