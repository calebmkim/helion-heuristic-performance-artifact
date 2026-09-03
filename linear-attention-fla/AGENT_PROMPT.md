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

For each arm, verify enough runtime state to know which configuration actually
ran. In particular:

- `default` must not accidentally receive a compiler heuristic or cached tuned
  result;
- `fla_triton` must actually use FLA's Triton path;
- `seed` should use the compiler-selected configuration without running a
  search;
- `aot_tuned` must use Helion's checked-in architecture-specific AOT selector
  with its normal selection semantics. Do not replace or restrict that
  selector in the benchmark adapter. Do not set `HELION_AUTOTUNE_EFFORT=none`
  for this arm: Helion then resolves an implicit compiler default before
  consulting `AOTAutotuneCache`. Audit selected bound configs against AOT cache
  results rather than inferring selection from correctness or timing.

Run each Helion arm in one long-lived process covering the full workload.
Separate processes are appropriate between arms because Helion selection and
bound-kernel caches are process state; a fresh process per shape is unnecessary.
Time FLA beside each arm and normalize that arm using its paired
`FLA latency / arm latency` ratio.

Use Helion's existing correctness references and its current linear-attention
timing helper rather than introducing a new timing protocol. Apply that method
consistently to every arm and report what it does. At the reference revision,
the helper used CUDA device events and cleared L2 before each timed call; it did
not use CUDA Graph replay here. Its default summary statistic was the mean; do
not silently replace that with nested median rounds. Save absolute per-shape
latencies before computing ratios.

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
forward-plus-backward panels. Use FLA Triton as a `1.00x` horizontal reference
and plot default, seed, and AOT-tuned as `FLA latency / arm latency`.

Make reasonable implementation decisions when APIs have moved. Record the
important adaptations and any unsupported or failed cells instead of blocking
the entire run. Leave `OUTPUT_DIR` with:

- the adapted runner;
- raw per-shape results;
- a short Markdown summary;
- the per-kernel bar graph;
- exact revisions, package versions, and GPU model.
