# Example Reduction Performance

- Profile: `liger-mixed`
- Helion commit: `fa2f62eb686ef846c76f8b9e18beec30fbc5bee1`
- Helion worktree dirty: `False`
- Software: Torch `2.13.0+cu132`, Triton `3.7.1`, CUDA runtime `13.2`.
- GPU: `NVIDIA H100 80GB HBM3`, compute capability `[9, 0]`.
- Torch reference: `torch.compile(mode="max-autotune-no-cudagraphs")`.
- Timing: all arms are compiled and captured in one process per cell, then timed together with balanced ordering, cold L2, and CUDA events.
- CPU launch overhead is excluded.
- Normalization: `torch.compile = 1.00x`; higher is faster.
- Three-arm common population: 80/80 cells.

## Per-kernel performance

| Kernel | Valid | Default | Heuristic seed | torch.compile |
|---|---:|---:|---:|---:|
| `cross_entropy` | 8/8 | 0.777x | 0.956x | 1.000x |
| `fused_linear_jsd` | 8/8 | 0.921x | 1.229x | 1.000x |
| `grpo` | 8/8 | 0.127x | 1.155x | 1.000x |
| `jsd` | 8/8 | 0.094x | 0.856x | 1.000x |
| `kl_div` | 8/8 | 0.068x | 0.963x | 1.000x |
| `layer_norm` | 8/8 | 1.118x | 1.190x | 1.000x |
| `layer_norm_bwd` | 8/8 | 0.046x | 1.085x | 1.000x |
| `rms_norm` | 8/8 | 1.000x | 1.075x | 1.000x |
| `rms_norm_bwd` | 8/8 | 0.056x | 1.086x | 1.000x |
| `softmax` | 8/8 | 1.130x | 1.377x | 1.000x |
| **Overall** | **80/80** | **0.268x** | **1.088x** | **1.000x** |

## Geometric-mean latency

| Kernel | Default (us) | Heuristic seed (us) | torch.compile (us) |
|---|---:|---:|---:|
| `cross_entropy` | 347.66 | 282.71 | 270.20 |
| `fused_linear_jsd` | 3181.31 | 2384.21 | 2929.53 |
| `grpo` | 2536.85 | 279.94 | 323.39 |
| `jsd` | 4576.04 | 501.89 | 429.57 |
| `kl_div` | 4805.68 | 337.56 | 325.19 |
| `layer_norm` | 19.31 | 18.15 | 21.60 |
| `layer_norm_bwd` | 1092.47 | 46.14 | 50.05 |
| `rms_norm` | 18.63 | 17.32 | 18.62 |
| `rms_norm_bwd` | 737.14 | 38.09 | 41.37 |
| `softmax` | 23.55 | 19.33 | 26.62 |
| **Overall** | **459.65** | **113.07** | **123.00** |

## Status

| Arm | Status counts |
|---|---|
| Default | `ok`: 80 |
| Heuristic seed | `ok`: 80 |
| torch.compile | `ok`: 80 |

Absolute per-cell latency, shape provenance, outer-round samples, correctness, selected configs, and environment metadata remain in `benchmark.json` and `per_cell.csv`.
