# vLLM Reductions: Helion Configurations vs vLLM

> **For all new reproductions, use Helion
> `fa2f62eb686ef846c76f8b9e18beec30fbc5bee1`, PyTorch `2.13.0+cu132`
> (CUDA `13.2`) with Triton `3.7.1`, and load vLLM `0.24.0`'s
> stable-libtorch extension directly.**

This comparison measures the reduction-bearing vLLM kernels under
`pretuned_kernels/` in a Helion checkout:

| Arm | Meaning | Required |
|---|---|---|
| `default` | Helion's base configuration before compiler heuristic promotion | yes |
| `seed` | Helion's compiler-selected seed with search and tuned caches disabled | yes |
| `aot_tuned` | The exact checked-in architecture-specific AOT configuration | yes |
| `vllm_cuda` | The corresponding compiled CUDA/C++ operator shipped by vLLM | optional |

There are two supported report paths:

| Available reference | Arms | Normalization |
|---|---|---|
| vLLM stable-libtorch extension | all four | `vllm_cuda = 1.00x` |
| No vLLM extension | three Helion arms | `aot_tuned = 1.00x` |

In both cases higher is faster. Missing CUDA C++ support must not be replaced
with a Triton, Torch, or Helion implementation under the `vllm_cuda` name.

The target kernels are:

- `dynamic_per_token_scaled_fp8_quant`
- `per_token_group_fp8_quant`
- `rms_norm_dynamic_per_token_quant`
- `rms_norm_per_block_quant`
- `silu_and_mul_per_block_quant`
- `fused_qk_norm_rope`

`silu_mul_fp8` is not included because it is pointwise and does not exercise
the reduction heuristic.

## Population

The scripts discover shapes from the target Helion checkout and require every
reported AOT cell to be an exact key in the current GPU architecture's
generated selector. Three profiles are available:

| Profile | Purpose |
|---|---|
| `curated` | Default. Intersect each kernel's checked-in `_bench_shapes()` sweep with exact AOT keys. |
| `core` | Quick representative sweep using decode, middle, and prefill token counts. |
| `all` | Every exact key in the architecture-specific AOT selectors. |

At the H100 revision inspected while creating this artifact, these profiles
contain 153, 54, and 402 cells respectively. Those counts are descriptive, not
hardcoded acceptance criteria.

`curated` is the primary profile because it follows the existing Helion
pretuned-dashboard precedent. `all` is useful for an exhaustive selector audit,
but its dense tuning grid gives repeated powers-of-two token counts more weight
than a normal performance report should.

## Primary H100 Run

The primary four-arm run measured all 153 curated cells successfully on an
H100 using:

- Helion main `fa2f62eb686ef846c76f8b9e18beec30fbc5bee1`
- PyTorch `2.13.0+cu132`
- CUDA `13.2`
- Triton `3.7.1`
- vLLM `0.24.0` at `ee0da84ab9e04ac7610e28580af62c365e898389`
- vLLM stable-extension SHA-256
  `686ece5394839c1eb214ba91fdf0219e47fab4ccce0a69a10d3fba39db9a47fb`

The heuristic seed reached `1.268x` and exact AOT reached `1.308x` versus
vLLM CUDA. The base default reached `0.578x`.

- [report](generated/h100-fa2f62eb-torch213-triton371-vllm024-curated/REPORT.md)
- [per-kernel graph](generated/h100-fa2f62eb-torch213-triton371-vllm024-curated/per-kernel-performance.png)
- [blog graph](../blog/figures/results-vllm-h100.png)
- [combined raw results](generated/h100-fa2f62eb-torch213-triton371-vllm024-curated/benchmark.json)
- [per-cell CSV](generated/h100-fa2f62eb-torch213-triton371-vllm024-curated/per_cell.csv)

## vLLM Baseline

At vLLM `0.24.0` commit `ee0da84ab9e04ac7610e28580af62c365e898389`,
all six matching
NVIDIA operators are compiled CUDA/C++ extension kernels:

| Kernel | vLLM entry point | Implementation |
|---|---|---|
| Dynamic per-token FP8 quant | `torch.ops._C.dynamic_per_token_scaled_fp8_quant` | CUDA C++ |
| Per-token-group FP8 quant | `torch.ops._C.per_token_group_fp8_quant` | CUDA C++ |
| RMSNorm + dynamic FP8 quant | `torch.ops._C.rms_norm_dynamic_per_token_quant` | CUDA C++ |
| RMSNorm + block FP8 quant | `torch.ops._C.rms_norm_per_block_quant` | CUDA C++ |
| SiLU-and-mul + block quant | `torch.ops._C.silu_and_mul_per_block_quant` | CUDA C++ |
| Fused QK norm + RoPE | `torch.ops._C.fused_qk_norm_rope` | CUDA C++ (TensorRT-LLM-derived kernel) |

None of these six comparison calls use CuTe. vLLM also has a Triton fallback
for per-token-group quantization, but contiguous NVIDIA inputs take the
compiled `_C` operator. The benchmark calls `_C` directly so the measured
backend is unambiguous. vLLM also ships optional Helion replacements for some
of these operators; those are deliberately not used as the external baseline.

Run `scripts/inspect_vllm.py` against the requested vLLM revision to verify
that this classification is still current. The CUDA C++ arm is optional
because source-only environments may not have vLLM's
`_C_stable_libtorch` extension installed. `scripts/probe_vllm_cuda.py` loads
that extension directly and verifies all six operators without requiring the
full vLLM server dependency stack.

### Toolchain compatibility footgun

Keep the benchmark environment on the exact PyTorch `2.13.0+cu132` / CUDA
`13.2` / Triton `3.7.1` stack. The vLLM 0.24.0 wheel declares
`torch==2.11.0`; installing it
normally in that environment can replace Torch and consequently Triton with
3.6.0. In our diagnostic run that older compiler stack reduced the
`fused_qk_norm_rope` seed geomean from about `1.04x` to `0.69x` versus vLLM
CUDA.

The wheel dependency pin is not a binary requirement of
`_C_stable_libtorch.abi3.so`. vLLM 0.24.0 builds that CUDA extension against
the PyTorch 2.11 C-shim and documents it as ABI-compatible with PyTorch 2.11
and newer. We verified the v0.24.0 extension above under PyTorch 2.13.0 and
Triton 3.7.1: all six operators loaded, executed, passed correctness, and
completed all 153 cells.

Therefore, do not install vLLM's dependencies into the benchmark environment.
Download the platform wheel without dependencies, extract it, and pass the
stable extension's path explicitly:

```bash
python -m pip download --only-binary=:all: --no-deps \
  vllm==0.24.0 --dest /path/to/wheelhouse
python -m zipfile -e /path/to/wheelhouse/<downloaded-wheel>.whl \
  /path/to/extracted-vllm-wheel

export VLLM_EXTENSION_PATH=/path/to/extracted-vllm-wheel/vllm/_C_stable_libtorch.abi3.so
```

This stable guarantee covers the libtorch interface; the wheel must still
target the machine's supported GPU/CUDA platform. The preflight probe loads
the binary, verifies all six registrations, and records its SHA-256. Every arm
then receives a real execution and correctness check in each benchmark cell.
Do not force Triton 3.7 into a Torch 2.11 environment or accept a silent
three-arm fallback when reproducing the primary four-arm result.

## Reproduce

Give [AGENT_PROMPT.md](AGENT_PROMPT.md) to the reproducing agent with the
requested Helion and vLLM revisions. The files under [scripts](scripts/README.md)
are adaptable starting points rather than a stable benchmark API.
Change paths, environment setup, module/import locations, or minor
compatibility plumbing as needed for the reproducing machine. For a comparable
result, preserve and report the pinned revisions, workload and arms,
correctness checks, and timing semantics.

The normal pipeline is:

```bash
export HELION_ROOT=/path/to/helion
export VLLM_ROOT=/path/to/vllm
export OUTPUT_DIR=/path/to/results
export CUDA_VISIBLE_DEVICES=<gpu>
# Primary stack, checked by starter.sh:
export REQUIRED_TORCH_VERSION=2.13.0+cu132
export REQUIRED_CUDA_VERSION=13.2
export REQUIRED_TRITON_VERSION=3.7.1
# Extracted from the vLLM 0.24.0 wheel with --no-deps:
export VLLM_EXTENSION_PATH=/path/to/_C_stable_libtorch.abi3.so
# Require the external baseline; do not silently emit a three-arm report:
export REFERENCE=vllm_cuda

vllm-reductions/scripts/starter.sh
```

With `REFERENCE=auto`, the starter uses `vllm_cuda` when the extension probe
passes and otherwise emits a valid three-arm report normalized to exact AOT.
Use `REFERENCE=vllm_cuda` to require the external operator or
`REFERENCE=aot_tuned` to intentionally omit it.

These kernels are small enough that CPU launch overhead can obscure compiler
code-generation differences. The starter therefore reports CUDA device time.
For each cell, one process correctness-checks, compiles, and captures all
available arms before timing them together. The shared timer rotates and
reverses arm order, clears L2 before every measured replay, and uses CUDA
events. Host launch, compilation, input construction, and correctness checks
are outside the measured interval.

The final report should retain raw per-cell latency, correctness status,
selected Helion configurations, exact revisions, and higher-is-better
per-kernel geomeans normalized to the selected reference. It should also
produce a clustered per-kernel graph using the same common population as the
table. Failed arms must remain visible rather than being replaced with a
different implementation.
