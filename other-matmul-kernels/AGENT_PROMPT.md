# Agent Prompt: Reproduce Other Matmul Kernel Performance

**For all new reproductions, use Helion
`fa2f62eb686ef846c76f8b9e18beec30fbc5bee1`, PyTorch `2.13.0+cu132`
(CUDA `13.2`) with Triton `3.7.1`.**

## Inputs

```text
HELION_ROOT=<existing Helion checkout>
HELION_REVISION=<revision to measure>
ARTIFACT_ROOT=<this repository>
OUTPUT_DIR=<result directory>
CUDA_VISIBLE_DEVICES=<H100 index>
REQUIRED_TORCH_VERSION=2.13.0+cu132
REQUIRED_CUDA_VERSION=13.2
REQUIRED_TRITON_VERSION=3.7.1
REQUIRED_HELION_COMMIT=fa2f62eb686ef846c76f8b9e18beec30fbc5bee1
```

## Task

Reproduce the formula-matmul and multi-matmul seed comparison described by
`other-matmul-kernels/shapes.json`.

Use `other-matmul-kernels/scripts/` as the starting point. These scripts match
one Helion revision, not a stable API. Inspect the requested checkout and adapt
imports, configuration APIs, or example-kernel signatures when needed. Keep
meaningful adaptations with the output.

Unless the request explicitly asks for a historical run, require clean Helion
`fa2f62eb686ef846c76f8b9e18beec30fbc5bee1`, PyTorch `2.13.0+cu132`, CUDA
`13.2`, and Triton `3.7.1`. The launcher invokes the repository's shared stack
checker before importing the benchmark.

For every selected shape, compare these exact arms:

1. Helion's base/default configuration from
   `ConfigSpec._base_default_config()`;
2. the active architecture's promoted rank-0 compiler heuristic seed, with
   autotuning disabled;
3. the example's natural PyTorch reference under `torch.compile` max-autotune.

Do not let either arm consume a tuned cache or launch an autotuning search.
Record which heuristic fired, every compiler seed it generated, and the
effective config used by each arm. A formula-matmul cell may also trigger a
secondary heuristic such as `triton_skinny_gemm`; retain that fact.

Compile and time both arms in one process for a shape, but isolate different
shapes in fresh subprocesses. Correctness-check the seed output against the
default output, reject non-finite results, alternate timing order, and retain
absolute latency samples. Follow Helion's current device-timing helper unless
there is a concrete reason to adapt it.

Restrict TorchInductor GEMM choices to Triton and disable cuBLASLt autotuning.
Audit its generated source and reject any cell that calls through ATen/external
kernels or names cuBLAS, cuDNN, CUTLASS, NVGEMM, FlashAttention, or Efficient
Attention. Use explicit PyTorch matmul/softmax attention rather than SDPA.

Normalize every result as:

```text
seed speedup = default latency / seed latency
```

Thus default is `1.00x` and higher is faster. Report every shape and failure,
per-family geometric means with shape counts, separate formula-matmul and
multi-matmul aggregates, and a macro-geomean that gives each family equal
weight. Generate one combined graph containing every kernel family, plus the
publication-style blog graph following the conventions used by the other
compile-time-heuristics artifacts.

Also report `torch.compile = 1.00x`, with higher-is-better bars for Helion
default and heuristic seed. Keep the original default-normalized report and
graphs. State that recurrent, attention, and state-space references are
natural PyTorch formulations and may differ algorithmically from Helion.
Treat 30 minutes of compilation for one shape as a timeout.

This run intentionally omits autotuning. State clearly that a rigorous
extension should autotune every shape once and compare default and seed with
that best-known same-kernel configuration.

Do not silently add the workloads in `to-do-work/` to the aggregate. If one has
become supported, validate it first and make the population change explicit.

Leave `OUTPUT_DIR` with raw results, summary JSON, per-shape CSV, Markdown
report, graph, exact Helion revision, package versions, and GPU model.
