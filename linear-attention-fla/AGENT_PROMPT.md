# Agent Prompt: Reproduce Linear-Attention Performance

Copy this entire prompt to the agent running the benchmark and fill in the
values below.

## Inputs

```text
HELION_ROOT=<absolute path to an existing Helion clone>
HELION_REVISION=<commit or branch to test>
FLA_ROOT=<absolute path to an existing flash-linear-attention clone, or NONE if installed>
FLA_REVISION=<commit/tag/version to test>
ARTIFACT_ROOT=<absolute path to this artifact clone>
OUTPUT_DIR=<absolute path for durable results>
CUDA_VISIBLE_DEVICES=<one GPU index>
OUTER_ROUNDS=3
```

## Objective

Measure every linear-attention operation cell in the checked-out Helion
registry using:

1. raw Helion default;
2. FLA Triton;
3. Helion compiler heuristic;
4. exact architecture-specific Helion pre-tuned configuration.

Produce an auditable combined result and a higher-is-better report. Stay with
the run through correctness, timing, aggregation, and validation. If an arm is
unsupported, fails, or lacks an exact pre-tuned key, report that state rather
than replacing it with another configuration.

## Non-Negotiable Rules

- Treat Helion's current `benchmarks/run_linattn.py` and
  `examples/linear/linear_attention_harness.py` as the workload authority.
- Reuse Helion's input factories, FLA wrappers, references, tolerances, and
  `do_bench`. Do not copy kernels or freeze a private shape list.
- Do not edit Helion's production kernels or heuristic implementation.
- Put any thin benchmark adapter under `OUTPUT_DIR/tools/` and retain it with
  the results.
- Run every `(cell, Helion arm)` in a fresh subprocess.
- Never call a nearest-key AOT fallback `pre_tuned`.
- Preserve absolute latency, individual timing samples, correctness evidence,
  selected configs, and provenance.
- Do not silently drop cells. Every discovered cell must end as `ok`,
  `unsupported`, `missing_exact_config`, or `error`.
- All aggregate tables must be higher-is-better.

## Procedure

### 1. Pin and inspect

1. Resolve `HELION_REVISION` and `FLA_REVISION` to immutable commit IDs. Do not
   change a dirty checkout without first preserving and reporting its state.
2. Read:
   - `benchmarks/run_linattn.py`
   - `examples/linear/linear_attention_harness.py`
   - `examples/linear/linear_attention_fla.py`
   - the current architecture's generated linear-attention AOT module
3. Run:

   ```bash
   python "$ARTIFACT_ROOT/linear-attention-fla/scripts/check_environment.py" \
     --helion "$HELION_ROOT" \
     --fla "$FLA_ROOT" \
     --output "$OUTPUT_DIR/environment.json"
   ```

4. Record the exact Python executable, Helion and FLA commits and dirty states,
   imported module paths, Torch, Triton, CUDA, GPU model, compute capability,
   and the relevant environment variables.

Stop only for a genuinely unusable environment, such as no CUDA device or no
FLA import. A missing architecture AOT module makes only `pre_tuned`
unavailable.

### 2. Discover the population

Import the current runner and derive cells from its public registry:

- variants and their dense or variable-length shape definitions;
- `forward` for every supported variant;
- `forward_backward` only where the runner defines backward.

Use a stable ID such as:

```text
<variant>::<mode>::<shape-name>
```

Write the discovered manifest before timing. The final result must contain a
row for every manifest entry. Record the chunk size and dtype selected by the
current harness.

### 3. Build a thin adapter

Use current Helion APIs to make a resumable adapter. It should accept one cell
and one Helion arm, execute correctness before timing, write one result
atomically, and exit. A parent process should launch these children and merge
their files.

The adapter should only provide orchestration and selection audit. Delegate to
the existing harness for:

- deterministic input construction;
- Helion forward and forward-plus-backward operations;
- FLA forward and forward-plus-backward operations;
- recurrent and chunked correctness references;
- tolerances;
- `helion._testing.do_bench`.

Inspect the checked-out APIs and make the smallest compatible adapter. Do not
patch this artifact's protocol to match one Helion revision when adaptation in
`OUTPUT_DIR/tools/` is sufficient.

### 4. Implement and prove each arm

Before importing Helion kernels, clear unrelated `HELION_*` overrides and set
only the controls needed for the arm.

#### `default`

Start from:

```text
HELION_AUTOTUNE_EFFORT=none
HELION_DISABLE_AUTOTUNER_HEURISTICS=1
```

Prevent local, remote, and AOT configuration lookup. After compilation, inspect
each invoked bound kernel and record:

- selected `Config`;
- base fragment default;
- compiler default, which must be absent;
- `autotuner_heuristics`, which must be empty.

Fail the arm audit if the selected configuration is not the normalized base
default. This is the raw default, not whatever Helion ordinarily calls its
default when compiler promotion is enabled.

#### `heuristic`

Start from:

```text
HELION_AUTOTUNE_EFFORT=none
HELION_DISABLE_AUTOTUNER_HEURISTICS=0
```

Prevent AOT, local, and remote tuned configurations from winning. Record, for
every invoked kernel:

- selected `Config`;
- `compiler_default_config`;
- `autotuner_heuristics`;
- relevant compiler facts when exposed by the current revision.

The selected configuration should be the normalized compiler default layered
over base defaults when a heuristic fires. Some constituent kernels may
legitimately retain base defaults.

Time FLA in this child process as the `fla_triton` arm. Resolve it through
Helion's `linear_attention_fla.py` adapter. Alternate timing order by cell
index: even cells time Helion then FLA, odd cells time FLA then Helion.
Inspect the resolved FLA operation and confirm it uses FLA's Triton
implementation. If the selected FLA revision dispatches to TileLang, a compiled
extension, or another backend, record the actual backend and do not place that
latency in `fla_triton`.

#### `pre_tuned`

Start from:

```text
HELION_AUTOTUNE_CACHE=AOTAutotuneCache
HELION_AOT_MODE=evaluate
```

Locate the generated module matching the active compute capability. Inspect its
call-key function and key tables. Wrap or instrument selection so each call
records:

- generated kernel name;
- call key;
- whether the key is exactly present;
- selected config.

Generated modules may choose a nearest same-flags key for unseen calls. If any
kernel invoked by the operation lacks an exact key, mark the operation
`missing_exact_config`. A caught selector exception may cause Helion to fall
back to the compiler default, so explicitly discard any such fallback latency.

### 5. Correctness

Seed inputs consistently. Use the current harness's reference paths and
tolerances before timing:

- forward Helion output versus the recurrent reference;
- backward Helion gradients versus the chunked reference;
- Helion versus FLA output or gradients wherever FLA executes.

Record measured errors and finite-value checks. A reference OOM or FLA's
documented architecture guard is not a pass; classify it and preserve the
exception. Do not report a Helion latency when that Helion arm fails
correctness.

### 6. Timing

For each reportable callable:

1. Warm and synchronize through `do_bench`.
2. Run `OUTER_ROUNDS` independent calls to
   `do_bench(..., return_mode="median")`.
3. Store every returned millisecond value.
4. Report the median of the outer samples.

Keep compilation, input creation, correctness references, and metadata
collection outside the timed callable. For backward, clear gradients using the
same mechanism as Helion's harness.

Use only `CUDA_VISIBLE_DEVICES`. Check for unrelated GPU processes before the
campaign. Capture clocks, power limit, power draw, and temperature at the
beginning and end, and periodically during long runs.

### 7. Resume and merge

Write each child result to a temporary file and rename it only after valid JSON
is complete. The parent may skip a completed `(cell, arm)` on resume, but it
must rerun malformed or partial files.

Merge into:

```text
$OUTPUT_DIR/results.json
```

following
`$ARTIFACT_ROOT/linear-attention-fla/RESULT_SCHEMA.md`. FLA latency should come
from the heuristic child that co-benchmarked it. Retain all raw child files,
the adapter, the discovered manifest, environment capture, stdout, and stderr.

### 8. Validate and summarize

Run:

```bash
python "$ARTIFACT_ROOT/linear-attention-fla/scripts/summarize.py" \
  "$OUTPUT_DIR/results.json" \
  --markdown-out "$OUTPUT_DIR/summary.md" \
  --json-out "$OUTPUT_DIR/summary.json"
```

Manually audit at least:

- one raw-default cell;
- one cell where a compiler heuristic fired;
- one exact pre-tuned cell;
- one missing or unsupported cell, if present.

Confirm the report contains:

- default versus heuristic on their full correct common population;
- FLA Triton versus heuristic on their full correct common population;
- exact pre-tuned versus heuristic on their full correct common population;
- a four-way common-population table;
- overall, forward, forward-plus-backward, and per-variant geometric means;
- correctness, support, exact-key, and error counts.

### 9. Final response

Report:

- immutable Helion and FLA revisions;
- hardware and core package versions;
- discovered and successfully timed cell counts;
- all pairwise geometric means and their populations;
- the common four-way higher-is-better table;
- correctness and exact-AOT coverage;
- every limitation or unsupported class;
- paths to `results.json`, `summary.md`, the adapter, and environment capture.

Do not summarize a partial campaign as complete.
