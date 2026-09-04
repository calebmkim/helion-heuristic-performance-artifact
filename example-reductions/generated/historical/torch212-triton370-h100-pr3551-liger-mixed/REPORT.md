# Example Reduction Performance

- Profile: `liger-mixed`
- Helion commit: `6ca445ca0605f703d44967dbedd153a0a89a5e00`
- Helion worktree dirty: `False`
- Software: Torch `2.12.0+cu132`, Triton `3.7.0`, CUDA runtime `13.2`.
- GPU: `NVIDIA H100 80GB HBM3`, compute capability `[9, 0]`.
- Torch reference: `torch.compile(mode="max-autotune-no-cudagraphs")`.
- Timing: all arms are compiled and captured in one process per cell, then timed together with balanced ordering, cold L2, and CUDA events.
- CPU launch overhead is excluded.
- Normalization: `torch.compile = 1.00x`; higher is faster.
- Three-arm common population: 80/80 cells.

## Per-kernel performance

| Kernel | Valid | Default | Heuristic seed | torch.compile |
|---|---:|---:|---:|---:|
| `cross_entropy` | 8/8 | 0.791x | 0.974x | 1.000x |
| `fused_linear_jsd` | 8/8 | 0.668x | 0.927x | 1.000x |
| `grpo` | 8/8 | 0.126x | 1.145x | 1.000x |
| `jsd` | 8/8 | 0.094x | 0.854x | 1.000x |
| `kl_div` | 8/8 | 0.069x | 0.977x | 1.000x |
| `layer_norm` | 8/8 | 1.133x | 1.203x | 1.000x |
| `layer_norm_bwd` | 8/8 | 0.044x | 1.259x | 1.000x |
| `rms_norm` | 8/8 | 1.018x | 1.091x | 1.000x |
| `rms_norm_bwd` | 8/8 | 0.045x | 1.062x | 1.000x |
| `softmax` | 8/8 | 1.101x | 1.335x | 1.000x |
| **Overall** | **80/80** | **0.253x** | **1.073x** | **1.000x** |

## Geometric-mean latency

| Kernel | Default (us) | Heuristic seed (us) | torch.compile (us) |
|---|---:|---:|---:|
| `cross_entropy` | 347.49 | 282.21 | 274.93 |
| `fused_linear_jsd` | 3306.49 | 2380.96 | 2207.55 |
| `grpo` | 2527.76 | 278.36 | 318.61 |
| `jsd` | 4579.27 | 502.20 | 429.12 |
| `kl_div` | 4802.67 | 337.10 | 329.46 |
| `layer_norm` | 19.18 | 18.07 | 21.74 |
| `layer_norm_bwd` | 1322.64 | 46.20 | 58.16 |
| `rms_norm` | 18.32 | 17.09 | 18.64 |
| `rms_norm_bwd` | 905.17 | 38.11 | 40.45 |
| `softmax` | 23.59 | 19.46 | 25.97 |
| **Overall** | **478.84** | **112.86** | **121.08** |

## Status

| Arm | Status counts |
|---|---|
| Default | `ok`: 80 |
| Heuristic seed | `ok`: 80 |
| torch.compile | `ok`: 80 |

Absolute per-cell latency, shape provenance, outer-round samples, correctness, selected configs, and environment metadata remain in `benchmark.json` and `per_cell.csv`.
