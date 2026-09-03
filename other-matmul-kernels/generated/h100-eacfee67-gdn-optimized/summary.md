# Other Matmul Kernel Performance

- Profile: `h100-off-corpus-v1`
- GPU: `NVIDIA H100 80GB HBM3` (compute capability `[9, 0]`)
- Helion commit: `eacfee67c0fdbc5a1c068f16a3b2f9f15ce23eb7`
- Helion worktree dirty: `True`
- Arms: raw `ConfigSpec._base_default_config()`, the promoted rank-0 compiler heuristic seed, and each example's natural PyTorch reference under `torch.compile(mode="max-autotune-no-cudagraphs")`. Helion autotuning and tuned-cache replay were disabled; TorchInductor GEMMs were max-autotuned with Triton as the only permitted backend.
- Timing: Helion do_bench CUDA device timing; median within do_bench, then median of balanced outer rounds; 3 outer rounds.
- Normalization: `default = 1.00x`; higher is faster.
- This is a seed-quality comparison, not an autotuned-quality comparison. A rigorous extension should autotune every shape.

## Aggregate performance

| Population | Valid shapes | Families | Default | Heuristic seed (cell geomean) | Heuristic seed (family macro-geomean) |
|---|---:|---:|---:|---:|---:|
| Multi-matmul | 6/6 | 1/1 | 1.000x | 1.933x | 1.933x |
| **Overall** | **6/6** | **1/1** | **1.000x** | **1.933x** | **1.933x** |

## Torch compile comparison

These ratios use `torch.compile = 1.00x`; higher is faster. Generated TorchInductor source must pass the Triton-only dispatch audit.

| Population | Valid shapes | Torch compile | Helion default (cell geomean) | Helion seed (cell geomean) | Helion default (family macro-geomean) | Helion seed (family macro-geomean) |
|---|---:|---:|---:|---:|---:|---:|
| Multi-matmul | 6/6 | 1.000x | 7.044x | 13.615x | 7.044x | 13.615x |
| **Overall** | **6/6** | **1.000x** | **7.044x** | **13.615x** | **7.044x** | **13.615x** |

The family macro-geomean gives every kernel family equal weight; the cell geomean gives every shape equal weight.

## Per-family performance

| Kernel family | Family ID | Valid shapes | Default | Heuristic seed | Default latency (us) | Seed latency (us) |
|---|---|---:|---:|---:|---:|---:|
| GDN forward-H | `gdn_forward_h` | 6/6 | 1.000x | 1.933x | 457.16 | 236.52 |

## Per-family torch compile comparison

| Kernel family | Valid shapes | Torch compile | Helion default | Helion seed | Torch compile latency (us) |
|---|---:|---:|---:|---:|---:|
| GDN forward-H | 6/6 | 1.000x | 7.044x | 13.615x | 3220.25 |

## Per-shape performance

| Family | Shape | Dtype | Fired heuristic(s) | Default (us) | Seed (us) | Torch compile (us) | Seed/default | Default/torch compile | Seed/torch compile | Status |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---|
| GDN forward-H | `b1_t8192_h64_c64_d64_s128` | `bfloat16` | `triton_h100_multi_matmul` | 404.67 | 158.43 | 9017.15 | 2.554x | 22.283x | 56.915x | `ok` |
| GDN forward-H | `b2_t8192_h32_c128_d128_s256` | `bfloat16` | `triton_h100_multi_matmul` | 753.30 | 287.39 | 4721.62 | 2.621x | 6.268x | 16.429x | `ok` |
| GDN forward-H | `b4_t4096_h64_c128_d64_s128` | `bfloat16` | `triton_h100_multi_matmul` | 387.76 | 240.98 | 2339.58 | 1.609x | 6.034x | 9.709x | `ok` |
| GDN forward-H | `b8_t2048_h32_c256_d64_s128` | `bfloat16` | `triton_h100_multi_matmul` | 222.59 | 167.30 | 732.75 | 1.331x | 3.292x | 4.380x | `ok` |
| GDN forward-H | `b8_t4096_h80_c256_d64_s128` | `bfloat16` | `triton_h100_multi_matmul` | 1438.72 | 804.98 | 3033.15 | 1.787x | 2.108x | 3.768x | `ok` |
| GDN forward-H | `b1_t4096_h32_c64_d128_s256` | `bfloat16` | `triton_h100_multi_matmul` | 241.17 | 118.48 | 5037.15 | 2.036x | 20.886x | 42.515x | `ok` |

## Audit notes

All compiler-generated seeds, selected effective configs, and absolute per-round latencies are retained in `raw-results.json`.
TorchInductor options, generated-source hashes, Triton kernel counts, and forbidden-dispatch matches are also retained per shape.
Attention, recurrent, and state-space references are natural PyTorch formulations, not tiled replicas of the Helion algorithms; their ratios therefore include fusion and algorithmic differences.
