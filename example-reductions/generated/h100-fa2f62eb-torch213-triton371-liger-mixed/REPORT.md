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
| `cross_entropy` | 8/8 | 0.776x | 0.954x | 1.000x |
| `fused_linear_jsd` | 8/8 | 0.921x | 1.230x | 1.000x |
| `grpo` | 8/8 | 0.128x | 1.158x | 1.000x |
| `jsd` | 8/8 | 0.094x | 0.855x | 1.000x |
| `kl_div` | 8/8 | 0.068x | 0.963x | 1.000x |
| `layer_norm` | 8/8 | 1.141x | 1.211x | 1.000x |
| `layer_norm_bwd` | 8/8 | 0.046x | 1.082x | 1.000x |
| `rms_norm` | 8/8 | 1.004x | 1.078x | 1.000x |
| `rms_norm_bwd` | 8/8 | 0.056x | 1.083x | 1.000x |
| `softmax` | 8/8 | 1.151x | 1.385x | 1.000x |
| **Overall** | **80/80** | **0.269x** | **1.090x** | **1.000x** |

## Geometric-mean latency

| Kernel | Default (us) | Heuristic seed (us) | torch.compile (us) |
|---|---:|---:|---:|
| `cross_entropy` | 347.88 | 283.10 | 270.13 |
| `fused_linear_jsd` | 3179.61 | 2381.32 | 2929.58 |
| `grpo` | 2526.79 | 278.80 | 322.97 |
| `jsd` | 4564.93 | 501.30 | 428.83 |
| `kl_div` | 4802.07 | 337.13 | 324.64 |
| `layer_norm` | 19.16 | 18.05 | 21.85 |
| `layer_norm_bwd` | 1090.56 | 46.22 | 50.02 |
| `rms_norm` | 18.39 | 17.14 | 18.46 |
| `rms_norm_bwd` | 738.05 | 38.31 | 41.48 |
| `softmax` | 23.45 | 19.50 | 26.99 |
| **Overall** | **458.13** | **112.99** | **123.18** |

## Status

| Arm | Status counts |
|---|---|
| Default | `ok`: 80 |
| Heuristic seed | `ok`: 80 |
| torch.compile | `ok`: 80 |

Absolute per-cell latency, shape provenance, outer-round samples, correctness, selected configs, and environment metadata remain in `benchmark.json` and `per_cell.csv`.
