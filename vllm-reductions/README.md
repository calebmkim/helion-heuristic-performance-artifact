# vLLM Reductions: Helion Configurations vs vLLM

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

## Included H100 Run

The included four-arm run measured all 153 curated cells successfully on an
H100 using Helion commit `746ee7c8a94fcc5fe5eab18356bde1aab69f9c43` and
vLLM commit `fc7fc421e98863c4ffb1aa02d46bd6e4d0202c26`:

- [report](generated/h100-pr3551-curated/REPORT.md)
- [per-kernel graph](generated/h100-pr3551-curated/per-kernel-performance.png)
- [combined raw results](generated/h100-pr3551-curated/benchmark.json)
- [per-cell CSV](generated/h100-pr3551-curated/per_cell.csv)

## vLLM Baseline

At vLLM commit `fc7fc421e98863c4ffb1aa02d46bd6e4d0202c26`, all six matching
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

## Reproduce

Give [AGENT_PROMPT.md](AGENT_PROMPT.md) to the reproducing agent with the
requested Helion and vLLM revisions. The files under [scripts](scripts/README.md)
are adaptable starting points rather than a stable benchmark API.

The normal pipeline is:

```bash
export HELION_ROOT=/path/to/helion
export VLLM_ROOT=/path/to/vllm
export OUTPUT_DIR=/path/to/results
export CUDA_VISIBLE_DEVICES=<gpu>
# Optional: explicit path to _C_stable_libtorch.abi3.so
export VLLM_EXTENSION_PATH=/path/to/_C_stable_libtorch.abi3.so
# auto (default), vllm_cuda, or aot_tuned
export REFERENCE=auto

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
