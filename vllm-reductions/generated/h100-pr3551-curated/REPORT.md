# vLLM Reduction Performance

- Profile: `curated`
- Timing: all arms are compiled and captured in one process per cell, then timed together with balanced ordering, cold L2, and CUDA events.
- CPU launch overhead is excluded.
- Normalization: `vllm_cuda = 1.00x`; higher is faster.
- 4-arm common population: 153/153 cells.

## Per-kernel geomeans

| Kernel | Valid | Default | Seed | AOT tuned | vLLM CUDA |
|---|---:|---:|---:|---:|---:|
| `dynamic_per_token_scaled_fp8_quant` | 24/24 | 0.322x | 1.113x | 1.123x | 1.000x |
| `fused_qk_norm_rope` | 21/21 | 1.005x | 1.039x | 1.075x | 1.000x |
| `per_token_group_fp8_quant` | 24/24 | 1.161x | 1.198x | 1.251x | 1.000x |
| `rms_norm_dynamic_per_token_quant` | 36/36 | 0.187x | 1.280x | 1.304x | 1.000x |
| `rms_norm_per_block_quant` | 24/24 | 0.723x | 1.456x | 1.507x | 1.000x |
| `silu_and_mul_per_block_quant` | 24/24 | 1.368x | 1.557x | 1.652x | 1.000x |
| **Overall** | **153/153** | **0.577x** | **1.267x** | **1.308x** | **1.000x** |

## Status

| Arm | Status counts |
|---|---|
| Default | `ok`: 153 |
| Seed | `ok`: 153 |
| AOT tuned | `ok`: 153 |
| vLLM CUDA | `ok`: 153 |

Absolute latency, outer-round samples, correctness results, selected configs, and environment metadata remain in `benchmark.json`.
