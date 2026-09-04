# vLLM Reduction Performance

- Profile: `curated`
- Helion commit: `fa2f62eb686ef846c76f8b9e18beec30fbc5bee1`
- Helion worktree dirty: `False`
- vLLM commit: `ee0da84ab9e04ac7610e28580af62c365e898389`
- vLLM worktree dirty: `False`
- Software: Torch `2.13.0+cu132`, Triton `3.7.1`, CUDA runtime `13.2`.
- GPU: `NVIDIA H100 80GB HBM3`, compute capability `[9, 0]`.
- Timing: all arms are compiled and captured in one process per cell, then timed together with balanced ordering, cold L2, and CUDA events.
- CPU launch overhead is excluded.
- Normalization: `vllm_cuda = 1.00x`; higher is faster.
- 4-arm common population: 153/153 cells.
- vLLM CUDA baseline: directly loaded stable-libtorch extension, distribution version `0.24.0`, SHA-256 `686ece5394839c1eb214ba91fdf0219e47fab4ccce0a69a10d3fba39db9a47fb`.

## Per-kernel geomeans

| Kernel | Valid | Default | Seed | AOT tuned | vLLM CUDA |
|---|---:|---:|---:|---:|---:|
| `dynamic_per_token_scaled_fp8_quant` | 24/24 | 0.323x | 1.117x | 1.123x | 1.000x |
| `fused_qk_norm_rope` | 21/21 | 1.005x | 1.040x | 1.075x | 1.000x |
| `per_token_group_fp8_quant` | 24/24 | 1.167x | 1.195x | 1.255x | 1.000x |
| `rms_norm_dynamic_per_token_quant` | 36/36 | 0.187x | 1.289x | 1.306x | 1.000x |
| `rms_norm_per_block_quant` | 24/24 | 0.723x | 1.449x | 1.510x | 1.000x |
| `silu_and_mul_per_block_quant` | 24/24 | 1.369x | 1.556x | 1.652x | 1.000x |
| **Overall** | **153/153** | **0.578x** | **1.269x** | **1.310x** | **1.000x** |

## Status

| Arm | Status counts |
|---|---|
| Default | `ok`: 153 |
| Seed | `ok`: 153 |
| AOT tuned | `ok`: 153 |
| vLLM CUDA | `ok`: 153 |

Absolute latency, outer-round samples, correctness results, selected configs, and environment metadata remain in `benchmark.json`.
