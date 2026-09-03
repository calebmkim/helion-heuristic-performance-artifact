# Linear Attention: Helion vs FLA

This comparison measures a Helion revision four ways:

| Arm | Meaning |
|---|---|
| `default` | Helion's conservative fragment defaults, with compiler heuristics and tuned caches disabled |
| `fla_triton` | FLA's handwritten Triton operation reached through Helion's FLA adapters |
| `heuristic` | Helion with autotuning disabled and compiler heuristics enabled |
| `pre_tuned` | A checked-in, architecture-specific Helion AOT configuration, only when every invoked key is exact |

This is an operation-level comparison. A forward cell times the full Helion or
FLA forward operation. A forward-plus-backward cell times the corresponding
combined operation.

## Source Of Truth

Use the checked-out Helion files, rather than a workload copy in this artifact:

- `benchmarks/run_linattn.py` defines variants and production shapes.
- `examples/linear/linear_attention_harness.py` owns input construction,
  correctness references, operation wrappers, and timing helpers.
- `examples/linear/linear_attention_fla.py` maps each variant to FLA.
- `examples/linear/linear_attention_engine.py` owns the Helion kernels.
- `examples/linear/_helion_aot_linear_attention_engine_cuda_smXY.py` contains
  architecture-specific pre-tuned configurations when available.

At Helion commit `4f69931e12aa5650e5926ce9e5ac50c15881be70`, the registry
contained 96 operation cells: 54 forward and 42 forward-plus-backward. Do not
hardcode that count. Later Helion revisions may add variants, shapes, or modes.

## Setup

Use one Python environment for all four arms. Install Helion from the revision
under test and FLA from the baseline revision:

```bash
git clone https://github.com/pytorch/helion.git
git clone https://github.com/fla-org/flash-linear-attention.git

git -C helion checkout <helion-revision>
git -C flash-linear-attention checkout <fla-revision>
```

Follow each repository's installation instructions. The historical result in
this directory used FLA `0.5.1`; a new run may intentionally use another
revision, but it must report that revision and must not compare unlabeled FLA
versions.

Preflight the exact environment:

```bash
python scripts/check_environment.py \
  --helion /path/to/helion \
  --fla /path/to/flash-linear-attention \
  --output results/environment.json
```

The preflight must find the current GPU's AOT module for a four-way report.
Without it, the other three arms can still be measured, but `pre_tuned` is
unavailable.

## Run

The authoritative procedure is
[AGENT_PROMPT.md](AGENT_PROMPT.md). Fill in its inputs and give the entire file
to an agent. The agent should adapt a thin runner to the current Helion APIs
instead of maintaining a second linear-attention benchmark implementation.

The run should produce one combined JSON document conforming to
[RESULT_SCHEMA.md](RESULT_SCHEMA.md). Validate and summarize it with:

```bash
python scripts/summarize.py results/results.json \
  --markdown-out results/summary.md \
  --json-out results/summary.json
```

## Arm Isolation

`HELION_AUTOTUNE_EFFORT=none` does not, by itself, define a baseline. Helion's
compiler heuristic can become `ConfigSpec.default_config()`, and AOT modules
can silently fall back to a nearest key. The runner must establish and audit
these semantics:

### Raw default

- Disable autotuning and compiler heuristics.
- Prevent local, remote, and AOT caches from supplying a configuration.
- Verify every bound kernel reports no compiler heuristic and that its selected
  configuration equals the base fragment default after normal validation.

### FLA Triton

- Resolve FLA through `examples.linear.linear_attention_fla`, so both sides use
  Helion's operation adapter and input conventions.
- Verify that the selected FLA implementation is its Triton path. A newer FLA
  revision may dispatch the same public operation to another backend; record
  that backend and do not label it `fla_triton`.
- Time FLA in the same child process as the heuristic arm.
- Alternate Helion-first and FLA-first timing by cell.
- Record unsupported FLA operations explicitly. Do not substitute another
  implementation.

### Compiler heuristic

- Disable autotune search and persisted configuration lookup.
- Leave compiler heuristics enabled.
- Record each invoked kernel's selected configuration,
  `compiler_default_config`, and `autotuner_heuristics`.
- Do not require every constituent kernel to fire a matmul heuristic. The audit
  proves what happened rather than assuming every operation is homogeneous.

### Exact pre-tuned

- Select the current architecture's generated AOT module.
- Require an exact generated key for every Helion kernel invoked by the
  operation.
- The generated modules may intentionally choose a nearest key for unseen
  inputs. That behavior is useful in production but is not an exact pre-tuned
  reference. Intercept or audit key selection and reject nearest-key fallback.
- If any invoked kernel lacks an exact key, retain the reason but discard the
  fallback latency from `pre_tuned`.

Run each `(cell, Helion arm)` in a fresh subprocess. This prevents bound-kernel,
compiler-default, and AOT module state from crossing arms and also makes the
run resumable.

## Correctness And Timing

Reuse Helion's references and tolerances:

- Forward compares against the recurrent reference.
- Backward compares gradients against the chunked reference.
- FLA output or gradients are compared where FLA supports the operation.
- A failed Helion arm has no reportable latency.

The historical protocol used three outer rounds. Each round called Helion's
`do_bench(..., return_mode="median")`, and the median of those three values was
reported. Keep all samples. Do not include compilation or reference execution
in the timed callable.

Record GPU clocks, power limit, power draw, and temperature when available.
Avoid concurrent GPU work. Use one visible GPU for the entire campaign.

## Changing Baselines

Use a new output directory for every Helion and FLA revision pair. Do not resume
child results produced by another revision, Python environment, or GPU model.
The provenance block makes reports comparable without pretending that timings
from different campaigns were co-benchmarked.

When evaluating several FLA versions, keep the Helion revision and environment
fixed where possible. When evaluating several Helion revisions, rerun all four
arms so raw defaults, compiler heuristics, generated AOT modules, and FLA timing
are all measured under the same software stack.

## Aggregation

Store absolute milliseconds first. For a baseline `B` and heuristic `H`, the
heuristic speedup is:

```text
latency(B) / latency(H)
```

Values above `1.0` favor the heuristic. Aggregate shape-level ratios with a
geometric mean.

The four-way table uses only cells that are correct and available in all four
arms, with an exact pre-tuned key for every invoked kernel. Normalize each arm
to heuristic performance:

```text
normalized_performance(arm) = latency(heuristic) / latency(arm)
```

The heuristic is `1.0`; larger values are faster. Pairwise tables may use each
pair's larger common population, but must state its cell count.

## Reference

[H100_PR3546.md](reference-results/H100_PR3546.md) records the result that
motivated this artifact. The corresponding
[machine-readable result](reference-results/H100_PR3546.json.gz) contains the
96 per-cell records, samples, and selection evidence. The summarizer reads
plain JSON and gzip-compressed JSON. This reference is useful as a pipeline
sanity check, not as an acceptance threshold for future software or hardware
revisions.
