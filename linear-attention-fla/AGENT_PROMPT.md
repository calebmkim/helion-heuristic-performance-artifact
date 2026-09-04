# Agent Prompt: Reproduce Linear-Attention Performance

**For all new reproductions, use Helion
`fa2f62eb686ef846c76f8b9e18beec30fbc5bee1`, PyTorch `2.13.0+cu132`
(CUDA `13.2`) with Triton `3.7.1`.**

## Inputs

```text
HELION_ROOT=<existing Helion checkout>
HELION_REVISION=<revision to measure>
FLA_ROOT=<FLA checkout or installed package>
FLA_REVISION=<revision or package version to measure>
ARTIFACT_ROOT=<this repository>
OUTPUT_DIR=<directory for scripts, raw results, and the report>
CUDA_VISIBLE_DEVICES=<GPU index>
REQUIRED_TORCH_VERSION=2.13.0+cu132
REQUIRED_CUDA_VERSION=13.2
REQUIRED_TRITON_VERSION=3.7.1
REQUIRED_HELION_COMMIT=fa2f62eb686ef846c76f8b9e18beec30fbc5bee1
```

## Important

The paths, flags, and starter script in this artifact reflect one previous
Helion revision. They are not a stable API and may be wrong for the requested
environment. Inspect the current checkouts and adapt them. Do not stop merely
because a command, module, or output format has changed.

Unless the request explicitly asks for a historical run, require clean Helion
`fa2f62eb686ef846c76f8b9e18beec30fbc5bee1`, PyTorch `2.13.0+cu132`, CUDA
`13.2`, and Triton `3.7.1`. Both launchers invoke the repository's shared
stack checker before importing either benchmark.

Use a fresh `OUTPUT_DIR` for a new revision or environment. Set `RESUME=1` only
when continuing the same run with the same Helion, FLA, and workload
revisions; replaying stale materialized configs invalidates the comparison.

## Choose the Workflow

Use Helion's native linear-attention runner directly when the requested goal is
to reproduce a claim from the main Helion repository. Around PR #3546,
`benchmarks/run_linattn.py` was the entry point, but locate its current
equivalent. `scripts/run_native_harness.sh` is a thin starting point. Preserve
the target claim's revision, package versions, configuration environment,
workload, timing order, and output semantics. The native harness measures FLA
against one environment-selected Helion configuration; do not represent
separate native runs as a same-process four-arm comparison.

Use the controlled interleaved workflow when the goal is to compare all four
arms fairly in one measurement process. The remainder of this prompt describes
that workflow, whose orchestrator is
`scripts/run_interleaved_comparison.sh`.

## Controlled Comparison

Reproduce the current Helion linear-attention workload on the requested GPU and
compare:

1. Helion's base/default configuration, with compiler heuristics and tuned
   results disabled;
2. FLA's Triton implementation;
3. Helion's compiler heuristic with autotuning disabled;
4. Helion's architecture-specific AOT-tuned configuration.

Start by inspecting Helion's current linear-attention benchmark and its FLA
integration. Reuse the workload, input generation, correctness checks, and
timing utilities already present in Helion.

Assert the primary Helion, PyTorch, CUDA, and Triton revisions before
materializing configs. Record them in the result.

Use mutually compatible FLA and Triton revisions. In particular, do not
classify backward operations as inherently unavailable when an older FLA
version merely guards them on Hopper. Update the comparison environment or
record it as an environment incompatibility.

Use the files in `linear-attention-fla/scripts/` as starting points. Modify or
replace any adapter that no longer matches the current checkout. The scripts
show the intended arm selection, but do not prove that those settings still
have the same behavior.

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

The older separate-arm adapter also measured every dense backward FLA block
before its Helion block because of workload row parity. Do not restore that
fixed ordering: rotate the first arm and alternate forward/reverse order at
the retained-sample level.

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
