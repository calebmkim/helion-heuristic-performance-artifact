# Agent Prompt: Reproduce Example Reduction Performance

**For all new reproductions, use Helion
`fa2f62eb686ef846c76f8b9e18beec30fbc5bee1`, PyTorch `2.13.0+cu132`
(CUDA `13.2`) with Triton `3.7.1`.**

## Inputs

```text
HELION_ROOT=<existing Helion checkout>
HELION_REVISION=<revision to measure>
ARTIFACT_ROOT=<this repository>
OUTPUT_DIR=<directory for adapted scripts, raw results, and report>
CUDA_VISIBLE_DEVICES=<exactly one GPU index>
REQUIRED_TORCH_VERSION=2.13.0+cu132
REQUIRED_CUDA_VERSION=13.2
REQUIRED_TRITON_VERSION=3.7.1
REQUIRED_HELION_COMMIT=fa2f62eb686ef846c76f8b9e18beec30fbc5bee1
```

## Important

Unless the request explicitly asks for the historical run, use clean Helion
`fa2f62eb686ef846c76f8b9e18beec30fbc5bee1`, PyTorch `2.13.0+cu132`, its
CUDA `13.2` build, and Triton `3.7.1`. PyTorch 2.13 pins that exact Triton
version. Verify the checkout and all three imported versions before compiling
any cell; a run on a different stack does not replace the primary result.

The retained historical dataset used PyTorch `2.12.0+cu132`, Triton `3.7.0`,
and Helion commit `6ca445ca0605f703d44967dbedd153a0a89a5e00`.

The scripts reflect one Helion revision. Inspect the requested checkout and
adapt imports or kernel APIs when they change. Do not edit Helion's production
kernels or heuristic implementation to make the benchmark run.

## Task

Compare all three arms over every cell in `shapes.json`:

1. `default`: `ConfigSpec._base_default_config()`, explicitly replayed so
   heuristic promotion cannot alter it;
2. `seed`: the compiler heuristic's first seed with search and all tuned cache
   lookup disabled, explicitly replayed;
3. `torch_compile`: the complete Torch reference compiled with
   `mode="max-autotune-no-cudagraphs"`.

Normalize to `torch_compile = 1.00x`; higher is faster.

## Shape Policy

The `liger-mixed` manifest is fixed and carries provenance for every shape.
Liger model dimensions are the anchor. Supplements are included only where
Liger omits an operational regime used by the corresponding kernel family:
short/long attention softmax, wider vocabularies, varied outer populations,
and wider/longer low-level GRPO.

Do not silently substitute a different shape. Record an unavailable kernel or
unsupported cell as a failure.

## Procedure

### 1. Pin and inspect

- Resolve the requested revision to an immutable commit.
- Record dirty state without discarding changes.
- Assert clean Helion `fa2f62eb686ef846c76f8b9e18beec30fbc5bee1`,
  PyTorch `2.13.0+cu132`, CUDA `13.2`, and Triton `3.7.1` for the primary run.
- Verify every body named in `shapes.json`.
- Record Python, Torch, Triton, CUDA, GPU model, compute capability, and
  imported package paths.

### 2. Isolate the arms

Set `HELION_AUTOTUNE_EFFORT=none`, `HELION_SKIP_CACHE=1`, and disable AOT/local/
remote tuned cache lookup. Obtain both Helion configs from the same bound
kernel, then construct explicit fixed-config kernels for timing.

Compile Torch only after constructing the exact reference and inputs for the
cell. Compilation and warmup are not timed.

### 3. Correctness and timing

Run each arm against the eager Torch result before timing. Multi-output kernels
must compare every observable output, including JSD gradients, GRPO LSE, and
normalization parameter gradients.

All arms for one cell must be compiled and captured in the same child process.
Time their graph replays together using the shared balanced timer from
`pretuned_kernels/_bench.py`: rotating/reversed order, cold L2, and CUDA events.
Report CUDA device time. CPU graph-launch overhead is excluded.

Run one cell per child process to release compiled code and large tensors
between shapes. Honor the supplied `CUDA_VISIBLE_DEVICES`.

### 4. Report

For each cell, retain:

- absolute latency in microseconds;
- `torch_compile / arm_latency`, so higher is faster;
- default and seed configs and fired heuristic names;
- correctness and errors;
- shape provenance;
- environment and exact revision;
- all outer-round samples.

Geometric means use only cells valid in all three arms. Leave failures visible.
Produce a combined JSON result, per-cell CSV, Markdown report, and per-kernel
graph.
