# Agent Prompt: Reproduce vLLM Reduction Performance

## Inputs

```text
HELION_ROOT=<existing Helion checkout>
HELION_REVISION=<revision to measure>
VLLM_ROOT=<existing vLLM checkout>
VLLM_REVISION=<revision to measure>
ARTIFACT_ROOT=<this repository>
OUTPUT_DIR=<directory for adapted scripts, raw results, and report>
CUDA_VISIBLE_DEVICES=<GPU index>
PROFILE=curated
REFERENCE=auto
VLLM_EXTENSION_PATH=<optional path to _C_stable_libtorch.abi3.so>
REQUIRED_TORCH_VERSION=2.13.0+cu132
REQUIRED_CUDA_VERSION=13.2
REQUIRED_TRITON_VERSION=3.7.1
REQUIRED_HELION_COMMIT=fa2f62eb686ef846c76f8b9e18beec30fbc5bee1
```

## Important

The scripts reflect one Helion and vLLM revision. Inspect both requested
checkouts and adapt the scripts when APIs, selector formats, kernel names, or
dispatch paths have changed. The scripts are intended to accelerate the work,
not constrain the reproducer.

Do not edit either project's production kernels or heuristic implementation to
make the benchmark run.

## Task

Always compare the first three arms below. Add the fourth when vLLM's compiled
extension is available:

1. `default`: Helion's base configuration, before compiler heuristic promotion;
2. `seed`: Helion's compiler-selected seed, without search or tuned caches;
3. `aot_tuned`: the exact checked-in AOT configuration for that shape;
4. `vllm_cuda` (optional): the matching CUDA/C++ custom operator shipped by
   vLLM.

Two outputs are valid:

- four arms normalized to `vllm_cuda = 1.00x`;
- three arms normalized to `aot_tuned = 1.00x` when the extension is absent or
  intentionally omitted.

Never substitute another backend under the `vllm_cuda` label.

Use the `curated` profile unless another profile is requested. It should be
derived from each current kernel module's `_bench_shapes()` and intersected
with the current architecture's exact AOT keys. Do not silently time a
nearest-key fallback and label it AOT tuned.

The expected kernel families are listed in `README.md`. Re-inventory the
target checkout. Include a newly added reduction-bearing vLLM pretuned kernel,
and record a removed or unavailable kernel rather than manufacturing a local
copy. Keep pointwise-only kernels out of this reduction comparison.

## Procedure

### 1. Pin and inspect

- Resolve Helion and vLLM revisions to immutable commits and report dirty
  states without discarding local changes.
- Run `scripts/inspect_vllm.py` and inspect every matching wrapper and native
  source file.
- Verify the external call's actual backend. A Python wrapper that can fall
  back to Triton does not make a contiguous NVIDIA `_C` call a Triton kernel.
- Record Python, Torch, Triton, CUDA, GPU model, compute capability, and
  imported package paths.
- For the primary H100 reproduction, require clean Helion
  `fa2f62eb686ef846c76f8b9e18beec30fbc5bee1`, PyTorch `2.13.0+cu132`,
  CUDA `13.2`, and Triton `3.7.1`. The starter checks all four before doing
  any benchmark work.
- Do not install vLLM 0.24.0's dependencies into that environment. Its wheel
  declares `torch==2.11.0`, which can replace the primary Torch/Triton pair
  with Torch 2.11.0 and Triton 3.6.0 and substantially regress
  `fused_qk_norm_rope`.
- Instead, download the vLLM 0.24.0 platform wheel with `pip download
  --no-deps`, extract it, and pass `_C_stable_libtorch.abi3.so` through
  `VLLM_EXTENSION_PATH`. vLLM builds this extension against the PyTorch 2.11
  C-shim and documents it as ABI-compatible with PyTorch 2.11 and newer.
- The stable guarantee covers libtorch, not arbitrary GPU/CUDA platforms.
  Require the probe and per-cell execution checks to pass. Record the wheel
  version, extension SHA-256, and exact vLLM source revision.
- Do not force only a newer Triton into the old Torch 2.11 environment. For an
  exact historical reproduction, use every version recorded in that raw
  dataset rather than the primary versions above.

At the reference vLLM revision, all six external calls are CUDA/C++ extension
kernels. If this changes, update the arm name and report rather than preserving
the old label.

### 2. Discover exact cells

Use `scripts/discover_manifest.py`. It reads `_KEYS` from the current
architecture's generated AOT modules and obtains the curated population from
the current kernel modules.

Available profiles:

- `curated`: checked-in `_bench_shapes()` intersected with exact AOT keys;
- `core`: a smaller exact-key decode/mid/prefill sample;
- `all`: every exact AOT key.

Write the manifest before timing. Report curated shapes missing from the exact
selector rather than allowing normal nearest-key selection.

### 3. Isolate and prove the arms

Run each cell in a fresh child process. Within that process, construct,
correctness-check, compile, and CUDA-graph-capture every available arm before
timing any arm. Do not run the arms in separate processes.

For `default`:

- set `HELION_AUTOTUNE_EFFORT=none`;
- disable all tuned cache lookup;
- obtain `ConfigSpec._base_default_config` from the cell's bound kernel;
- replay the kernel with that config explicitly so heuristic promotion cannot
  change the arm;
- record the selected config.

For `seed`:

- set `HELION_AUTOTUNE_EFFORT=none`;
- prevent local, remote, and AOT tuned configs from winning;
- leave compiler heuristics enabled;
- obtain the compiler-selected config and fired heuristic names;
- replay the kernel with that config explicitly.

For `aot_tuned`:

- require the manifest key to exist exactly in the generated selector;
- read the config at that exact selector index and replay it explicitly;
- record the selected config and selector file, and verify that the compiled
  runtime config matches the manifest's exact selector entry;
- do not expect `HELION_AUTOTUNE_EFFORT=none` to perform AOT cache lookup. It
  bypasses that lookup, which is why the combined runner forces the already
  validated exact selector config.

For `vllm_cuda`:

- run `scripts/probe_vllm_cuda.py`;
- preferably load `_C_stable_libtorch` directly from an explicitly supplied
  path extracted from the exact wheel without installing its dependencies;
- the vLLM source tree or an installed exact-revision wheel remains a valid
  fallback when it does not alter the benchmark's Torch/Triton pair;
- call the matching `torch.ops._C` operator directly through the existing
  Helion pretuned module's vLLM baseline adapter;
- verify the op is registered by the requested vLLM build;
- do not substitute a Triton, Torch, or Helion fallback.

If the probe fails under `REFERENCE=auto`, continue with the three required
arms. If `REFERENCE=vllm_cuda`, fail rather than silently changing the
reference.

The starter runner patches each module's benchmark population to one discovered
cell, then reuses its input factory, correctness check, Torch reference, and
call construction. Preserve that reuse when adapting it.

### 4. Correctness and timing

Run the module's correctness check before timing every arm, including the vLLM
operator. A failed arm has no reportable latency.

Use one timing protocol for every available arm. These kernels are small, so
report device time rather than host launch time. After every arm has been
pre-captured, pass all graph replays together to the same balanced timer used
by Helion's pretuned dashboard. It rotates implementations through every
position and then reverses those rotations, clearing L2 before each replay.
CPU graph-launch overhead is outside the CUDA-event interval. Keep all
outer-round samples and report their median in microseconds. Do not include
compilation, input generation, or correctness checks in the timed callable.

Avoid concurrent GPU work. Honor the supplied `CUDA_VISIBLE_DEVICES`; do not
move the campaign to another GPU.

### 5. Report

Keep absolute latency first. When the vLLM extension is available, normalize
the per-kernel comparison to the CUDA implementation:

```text
normalized performance = vLLM CUDA latency / arm latency
```

The vLLM CUDA arm is `1.00x`; higher is faster. Without the extension, use:

```text
normalized performance = AOT tuned latency / arm latency
```

The AOT arm is then `1.00x`. Compute geometric means only over cells that are
correct and available in every reported arm, and state the valid/total count.
Retain all failed cells.

Leave `OUTPUT_DIR` with:

- the discovered manifest;
- vLLM implementation inspection;
- one combined raw JSON result with every arm for each cell;
- a per-cell CSV;
- a concise Markdown report;
- any adapted scripts;
- exact revisions, package versions, GPU, and timing settings.
