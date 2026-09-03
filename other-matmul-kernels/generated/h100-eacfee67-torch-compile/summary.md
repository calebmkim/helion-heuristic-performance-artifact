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
| Formula matmul | 26/26 | 5/5 | 1.000x | 7.028x | 7.091x |
| Multi-matmul | 43/44 | 8/8 | 1.000x | 2.487x | 2.441x |
| **Overall** | **69/70** | **13/13** | **1.000x** | **3.679x** | **3.678x** |

## Torch compile comparison

These ratios use `torch.compile = 1.00x`; higher is faster. Generated TorchInductor source must pass the Triton-only dispatch audit.

| Population | Valid shapes | Torch compile | Helion default (cell geomean) | Helion seed (cell geomean) | Helion default (family macro-geomean) | Helion seed (family macro-geomean) |
|---|---:|---:|---:|---:|---:|---:|
| Formula matmul | 26/26 | 1.000x | 0.185x | 1.302x | 0.178x | 1.265x |
| Multi-matmul | 43/44 | 1.000x | 1.374x | 3.417x | 1.350x | 3.294x |
| **Overall** | **69/70** | **1.000x** | **0.646x** | **2.376x** | **0.620x** | **2.280x** |

The family macro-geomean gives every kernel family equal weight; the cell geomean gives every shape equal weight.

## Per-family performance

| Kernel family | Family ID | Valid shapes | Default | Heuristic seed | Default latency (us) | Seed latency (us) |
|---|---|---:|---:|---:|---:|---:|
| Plain BF16 matmul | `plain_matmul` | 5/5 | 1.000x | 12.163x | 1239.78 | 101.93 |
| BF16 x INT16 GEMM | `bf16xint16_gemm` | 5/5 | 1.000x | 7.907x | 1098.82 | 138.96 |
| Broadcast matmul | `broadcast_matmul` | 5/5 | 1.000x | 15.949x | 840.63 | 52.71 |
| Gather GEMV | `gather_gemv` | 5/5 | 1.000x | 2.084x | 155.07 | 74.41 |
| Mamba-2 chunk state | `mamba2_chunk_state` | 6/6 | 1.000x | 5.610x | 620.00 | 110.52 |
| Dense attention forward | `dense_attention` | 5/5 | 1.000x | 3.042x | 845.90 | 278.09 |
| Causal attention forward | `causal_attention` | 5/5 | 1.000x | 5.054x | 2108.57 | 417.18 |
| Biased attention forward | `biased_attention` | 5/5 | 1.000x | 1.488x | 117.83 | 79.18 |
| Attention backward | `attention_backward` | 5/5 | 1.000x | 1.999x | 452.91 | 226.61 |
| Jagged HSTU attention | `jagged_hstu` | 5/6 | 1.000x | 0.842x | 104.61 | 124.20 |
| Squeeze-and-excitation forward | `squeeze_excitation` | 6/6 | 1.000x | 4.961x | 75.09 | 15.14 |
| GDN forward-H | `gdn_forward_h` | 6/6 | 1.000x | 1.933x | 457.16 | 236.52 |
| Mamba-2 chunk scan | `mamba2_chunk_scan` | 6/6 | 1.000x | 3.408x | 797.34 | 233.93 |

## Per-family torch compile comparison

| Kernel family | Valid shapes | Torch compile | Helion default | Helion seed | Torch compile latency (us) |
|---|---:|---:|---:|---:|---:|
| Plain BF16 matmul | 5/5 | 1.000x | 0.096x | 1.163x | 118.57 |
| BF16 x INT16 GEMM | 5/5 | 1.000x | 0.107x | 0.848x | 117.77 |
| Broadcast matmul | 5/5 | 1.000x | 0.077x | 1.226x | 64.59 |
| Gather GEMV | 5/5 | 1.000x | 0.481x | 1.003x | 74.64 |
| Mamba-2 chunk state | 6/6 | 1.000x | 0.476x | 2.672x | 295.33 |
| Dense attention forward | 5/5 | 1.000x | 0.645x | 1.962x | 545.74 |
| Causal attention forward | 5/5 | 1.000x | 0.548x | 2.772x | 1156.36 |
| Biased attention forward | 5/5 | 1.000x | 0.839x | 1.249x | 98.91 |
| Attention backward | 5/5 | 1.000x | 1.088x | 2.174x | 492.72 |
| Jagged HSTU attention | 5/6 | 1.000x | 6.417x | 5.405x | 671.31 |
| Squeeze-and-excitation forward | 6/6 | 1.000x | 0.689x | 3.417x | 51.73 |
| GDN forward-H | 6/6 | 1.000x | 7.044x | 13.615x | 3220.25 |
| Mamba-2 chunk scan | 6/6 | 1.000x | 1.095x | 3.731x | 872.72 |

## Per-shape performance

| Family | Shape | Dtype | Fired heuristic(s) | Default (us) | Seed (us) | Torch compile (us) | Seed/default | Default/torch compile | Seed/torch compile | Status |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---|
| Plain BF16 matmul | `m1_k4096_n4096` | `bfloat16` | `triton_h100_formula_matmul`, `triton_skinny_gemm` | 148.16 | 23.17 | 38.53 | 6.395x | 0.260x | 1.663x | `ok` |
| Plain BF16 matmul | `m32_k4096_n11008` | `bfloat16` | `triton_h100_formula_matmul`, `triton_skinny_gemm` | 161.25 | 47.74 | 45.34 | 3.377x | 0.281x | 0.950x | `ok` |
| Plain BF16 matmul | `m512_k4096_n4096` | `bfloat16` | `triton_h100_formula_matmul`, `triton_skinny_gemm` | 674.37 | 32.61 | 33.98 | 20.681x | 0.050x | 1.042x | `ok` |
| Plain BF16 matmul | `m4096_k4096_n11008` | `bfloat16` | `triton_h100_formula_matmul` | 13304.11 | 568.64 | 640.61 | 23.396x | 0.048x | 1.127x | `ok` |
| Plain BF16 matmul | `m4096_k11008_n4096` | `bfloat16` | `triton_h100_formula_matmul` | 13665.42 | 536.42 | 616.26 | 25.475x | 0.045x | 1.149x | `ok` |
| BF16 x INT16 GEMM | `m1_k4096_n4096` | `bfloat16` | `triton_h100_formula_matmul`, `triton_skinny_gemm` | 148.42 | 25.50 | 37.66 | 5.819x | 0.254x | 1.477x | `ok` |
| BF16 x INT16 GEMM | `m32_k4096_n11008` | `bfloat16` | `triton_h100_formula_matmul`, `triton_skinny_gemm` | 160.42 | 49.98 | 48.74 | 3.209x | 0.304x | 0.975x | `ok` |
| BF16 x INT16 GEMM | `m512_k4096_n4096` | `bfloat16` | `triton_h100_formula_matmul`, `triton_skinny_gemm` | 675.42 | 64.22 | 56.10 | 10.517x | 0.083x | 0.873x | `ok` |
| BF16 x INT16 GEMM | `m4096_k4096_n11008` | `bfloat16` | `triton_h100_formula_matmul` | 13168.99 | 1125.44 | 704.96 | 11.701x | 0.054x | 0.626x | `ok` |
| BF16 x INT16 GEMM | `m65536_k1024_n1280` | `bfloat16` | `triton_h100_formula_matmul`, `triton_skinny_gemm` | 7564.35 | 562.40 | 312.16 | 13.450x | 0.041x | 0.555x | `ok` |
| Broadcast matmul | `b32_m1_k4096_n4096` | `bfloat16` | `triton_h100_formula_matmul`, `triton_skinny_gemm` | 150.98 | 25.47 | 30.43 | 5.927x | 0.202x | 1.195x | `ok` |
| Broadcast matmul | `b8_m128_k4096_n11008` | `bfloat16` | `triton_h100_formula_matmul`, `triton_skinny_gemm` | 3414.91 | 140.70 | 170.02 | 24.270x | 0.050x | 1.208x | `ok` |
| Broadcast matmul | `b16_m512_k768_n1024` | `bfloat16` | `triton_h100_formula_matmul`, `triton_skinny_gemm` | 475.23 | 29.63 | 33.66 | 16.038x | 0.071x | 1.136x | `ok` |
| Broadcast matmul | `b4_m1024_k4096_n4096` | `bfloat16` | `triton_h100_formula_matmul` | 5128.80 | 200.50 | 221.46 | 25.581x | 0.043x | 1.105x | `ok` |
| Broadcast matmul | `b64_m16_k1024_n4096` | `bfloat16` | `triton_h100_formula_matmul` | 334.05 | 19.10 | 29.15 | 17.486x | 0.087x | 1.526x | `ok` |
| Gather GEMV | `b8_s2048_n2` | `bfloat16` | `triton_h100_formula_matmul` | 47.68 | 14.56 | 30.69 | 3.275x | 0.644x | 2.108x | `ok` |
| Gather GEMV | `b8_s4096_n2` | `bfloat16` | `triton_h100_formula_matmul` | 99.23 | 36.86 | 30.99 | 2.692x | 0.312x | 0.841x | `ok` |
| Gather GEMV | `b8_s8192_n2` | `bfloat16` | `triton_h100_formula_matmul` | 216.80 | 108.13 | 94.91 | 2.005x | 0.438x | 0.878x | `ok` |
| Gather GEMV | `b8_s14336_n2` | `bfloat16` | `triton_h100_formula_matmul` | 504.21 | 339.87 | 269.82 | 1.484x | 0.535x | 0.794x | `ok` |
| Gather GEMV | `b64_s4096_n8` | `bfloat16` | `triton_h100_formula_matmul` | 173.38 | 115.65 | 95.10 | 1.499x | 0.549x | 0.822x | `ok` |
| Mamba-2 chunk state | `b1_h64_g8_t8192_c64_d64_s128` | `bfloat16` | `triton_h100_formula_matmul` | 419.58 | 117.76 | 224.19 | 3.563x | 0.534x | 1.904x | `ok` |
| Mamba-2 chunk state | `b2_h32_g8_t8192_c128_d128_s256` | `bfloat16` | `triton_h100_formula_matmul` | 1474.66 | 274.34 | 450.18 | 5.375x | 0.305x | 1.641x | `ok` |
| Mamba-2 chunk state | `b4_h64_g8_t4096_c128_d64_s128` | `bfloat16` | `triton_h100_formula_matmul` | 754.24 | 156.48 | 386.06 | 4.820x | 0.512x | 2.467x | `ok` |
| Mamba-2 chunk state | `b8_h32_g4_t2048_c256_d64_s128` | `bfloat16` | `triton_h100_formula_matmul` | 366.30 | 55.30 | 194.85 | 6.624x | 0.532x | 3.524x | `ok` |
| Mamba-2 chunk state | `b8_h80_g1_t4096_c256_d64_s128` | `bfloat16` | `triton_h100_formula_matmul` | 1780.16 | 227.58 | 865.09 | 7.822x | 0.486x | 3.801x | `ok` |
| Mamba-2 chunk state | `b1_h128_g8_t2048_c256_d64_s128` | `bfloat16` | `triton_h100_formula_matmul` | 186.66 | 28.64 | 101.02 | 6.517x | 0.541x | 3.527x | `ok` |
| Dense attention forward | `b1_h4_m512_n512_d64_f16` | `float16` | `triton_h100_multi_matmul` | 46.27 | 25.02 | 75.68 | 1.849x | 1.636x | 3.024x | `ok` |
| Dense attention forward | `b2_h32_m1024_n1024_d64_f16` | `float16` | `triton_h100_multi_matmul` | 424.37 | 107.58 | 244.80 | 3.945x | 0.577x | 2.275x | `ok` |
| Dense attention forward | `b8_h16_m2048_n2048_d64_bf16` | `bfloat16` | `triton_h100_multi_matmul` | 3204.58 | 753.95 | 1694.53 | 4.250x | 0.529x | 2.248x | `ok` |
| Dense attention forward | `b4_h32_m4096_n4096_d128_bf16` | `bfloat16` | `triton_h100_multi_matmul` | 20007.94 | 4912.93 | 7465.98 | 4.073x | 0.373x | 1.520x | `ok` |
| Dense attention forward | `b4_h16_m128_n4096_d64_bf16` | `bfloat16` | `triton_h100_multi_matmul` | 344.00 | 166.78 | 206.54 | 2.063x | 0.600x | 1.238x | `ok` |
| Causal attention forward | `b1_h8_s512_d128_f16` | `float16` | `triton_h100_multi_matmul` | 52.03 | 29.82 | 70.34 | 1.745x | 1.352x | 2.358x | `ok` |
| Causal attention forward | `b2_h32_s1024_d128_f16` | `float16` | `triton_h100_multi_matmul` | 720.48 | 117.34 | 323.78 | 6.140x | 0.449x | 2.759x | `ok` |
| Causal attention forward | `b8_h16_s2048_d128_bf16` | `bfloat16` | `triton_h100_multi_matmul` | 5190.42 | 807.04 | 2318.34 | 6.431x | 0.447x | 2.873x | `ok` |
| Causal attention forward | `b4_h32_s4096_d128_bf16` | `bfloat16` | `triton_h100_multi_matmul` | 20548.85 | 2998.46 | 8884.26 | 6.853x | 0.432x | 2.963x | `ok` |
| Causal attention forward | `b1_h16_s8192_d128_bf16` | `bfloat16` | `triton_h100_multi_matmul` | 10424.45 | 1492.06 | 4408.10 | 6.987x | 0.423x | 2.954x | `ok` |
| Biased attention forward | `b1_h4_m128_n128_d64_f16` | `float16` | `triton_h100_multi_matmul` | 24.35 | 24.80 | 67.01 | 0.982x | 2.752x | 2.702x | `ok` |
| Biased attention forward | `b2_h8_m512_n512_d64_f16` | `float16` | `triton_h100_multi_matmul` | 61.66 | 42.50 | 65.02 | 1.451x | 1.054x | 1.530x | `ok` |
| Biased attention forward | `b2_h16_m1024_n1024_d64_f16` | `float16` | `triton_h100_multi_matmul` | 247.73 | 189.71 | 160.74 | 1.306x | 0.649x | 0.847x | `ok` |
| Biased attention forward | `b1_h16_m1024_n2048_d128_bf16` | `bfloat16` | `triton_h100_multi_matmul` | 506.88 | 189.70 | 186.21 | 2.672x | 0.367x | 0.982x | `ok` |
| Biased attention forward | `b4_h8_m256_n1024_d64_bf16` | `bfloat16` | `triton_h100_multi_matmul` | 120.45 | 82.08 | 72.61 | 1.467x | 0.603x | 0.885x | `ok` |
| Attention backward | `b1_h4_s256_d64_f16` | `float16` | `triton_h100_multi_matmul` | 48.93 | 49.25 | 98.69 | 0.994x | 2.017x | 2.004x | `ok` |
| Attention backward | `b2_h16_s512_d64_f16` | `float16` | `triton_h100_multi_matmul` | 60.93 | 49.50 | 105.66 | 1.231x | 1.734x | 2.134x | `ok` |
| Attention backward | `b2_h32_s1024_d64_f16` | `float16` | `triton_h100_multi_matmul` | 389.10 | 159.81 | 471.94 | 2.435x | 1.213x | 2.953x | `ok` |
| Attention backward | `b4_h32_s2048_d128_bf16` | `bfloat16` | `triton_h100_multi_matmul` | 5551.87 | 1757.50 | 3218.13 | 3.159x | 0.580x | 1.831x | `ok` |
| Attention backward | `b1_h16_s4096_d128_bf16` | `bfloat16` | `triton_h100_multi_matmul` | 2959.18 | 872.75 | 1833.66 | 3.391x | 0.620x | 2.101x | `ok` |
| Jagged HSTU attention | `b4_l320_max128_h8_d64` | `bfloat16` | `triton_h100_multi_matmul` | 16.06 | 20.06 | 168.22 | 0.801x | 10.472x | 8.384x | `ok` |
| Jagged HSTU attention | `b16_l3440_max512_h16_d64` | `bfloat16` | `triton_h100_multi_matmul` | 56.03 | 92.74 | 649.06 | 0.604x | 11.584x | 6.999x | `ok` |
| Jagged HSTU attention | `b8_l6656_max2048_h32_d128` | `bfloat16` | `triton_h100_multi_matmul` | 701.81 | 587.42 | 977.28 | 1.195x | 1.393x | 1.664x | `ok` |
| Jagged HSTU attention | `b32_l2485_max256_h8_d128` | `bfloat16` | `triton_h100_multi_matmul` | 34.08 | 30.91 | 1281.92 | 1.102x | 37.615x | 41.470x | `ok` |
| Jagged HSTU attention | `b2_l7168_max4096_h16_d64` | `bfloat16` | `triton_h100_multi_matmul` | 582.00 | 874.59 | 996.70 | 0.665x | 1.713x | 1.140x | `ok` |
| Jagged HSTU attention | `b128_l131072_max1024_h4_d128` | `bfloat16` | - | - | - | - | - | - | - | `driver_error` |
| Squeeze-and-excitation forward | `m256_n256_k16` | `bfloat16` | `triton_h100_multi_matmul` | 16.03 | 7.36 | 48.13 | 2.178x | 3.002x | 6.539x | `ok` |
| Squeeze-and-excitation forward | `m256_n512_k32` | `bfloat16` | `triton_h100_multi_matmul` | 39.84 | 9.57 | 44.70 | 4.164x | 1.122x | 4.672x | `ok` |
| Squeeze-and-excitation forward | `m128_n1024_k64` | `bfloat16` | `triton_h100_multi_matmul` | 124.96 | 15.97 | 57.79 | 7.826x | 0.462x | 3.619x | `ok` |
| Squeeze-and-excitation forward | `m64_n2048_k128` | `bfloat16` | `triton_h100_multi_matmul` | 420.54 | 43.78 | 48.93 | 9.607x | 0.116x | 1.118x | `ok` |
| Squeeze-and-excitation forward | `m1024_n1024_k256` | `bfloat16` | `triton_h100_multi_matmul` | 370.14 | 32.77 | 49.79 | 11.296x | 0.135x | 1.520x | `ok` |
| Squeeze-and-excitation forward | `m1_n256_k16` | `bfloat16` | `triton_h100_multi_matmul` | 14.43 | 7.46 | 63.23 | 1.936x | 4.381x | 8.481x | `ok` |
| GDN forward-H | `b1_t8192_h64_c64_d64_s128` | `bfloat16` | `triton_h100_multi_matmul` | 404.67 | 158.43 | 9017.15 | 2.554x | 22.283x | 56.915x | `ok` |
| GDN forward-H | `b2_t8192_h32_c128_d128_s256` | `bfloat16` | `triton_h100_multi_matmul` | 753.30 | 287.39 | 4721.62 | 2.621x | 6.268x | 16.429x | `ok` |
| GDN forward-H | `b4_t4096_h64_c128_d64_s128` | `bfloat16` | `triton_h100_multi_matmul` | 387.76 | 240.98 | 2339.58 | 1.609x | 6.034x | 9.709x | `ok` |
| GDN forward-H | `b8_t2048_h32_c256_d64_s128` | `bfloat16` | `triton_h100_multi_matmul` | 222.59 | 167.30 | 732.75 | 1.331x | 3.292x | 4.380x | `ok` |
| GDN forward-H | `b8_t4096_h80_c256_d64_s128` | `bfloat16` | `triton_h100_multi_matmul` | 1438.72 | 804.98 | 3033.15 | 1.787x | 2.108x | 3.768x | `ok` |
| GDN forward-H | `b1_t4096_h32_c64_d128_s256` | `bfloat16` | `triton_h100_multi_matmul` | 241.17 | 118.48 | 5037.15 | 2.036x | 20.886x | 42.515x | `ok` |
| Mamba-2 chunk scan | `b1_h64_g8_t8192_c64_d64_s128` | `bfloat16` | `triton_h100_multi_matmul` | 373.70 | 114.91 | 468.42 | 3.252x | 1.253x | 4.076x | `ok` |
| Mamba-2 chunk scan | `b2_h32_g8_t8192_c128_d128_s256` | `bfloat16` | `triton_h100_multi_matmul` | 1286.98 | 389.02 | 908.46 | 3.308x | 0.706x | 2.335x | `ok` |
| Mamba-2 chunk scan | `b4_h64_g8_t4096_c128_d64_s128` | `bfloat16` | `triton_h100_multi_matmul` | 948.26 | 283.41 | 1043.65 | 3.346x | 1.101x | 3.682x | `ok` |
| Mamba-2 chunk scan | `b8_h32_g4_t2048_c256_d64_s128` | `bfloat16` | `triton_h100_multi_matmul` | 615.10 | 177.06 | 755.84 | 3.474x | 1.229x | 4.269x | `ok` |
| Mamba-2 chunk scan | `b8_h80_g1_t4096_c256_d64_s128` | `bfloat16` | `triton_h100_multi_matmul` | 2954.66 | 771.58 | 3443.73 | 3.829x | 1.166x | 4.463x | `ok` |
| Mamba-2 chunk scan | `b1_h128_g8_t2048_c256_d64_s128` | `bfloat16` | `triton_h100_multi_matmul` | 310.02 | 94.69 | 382.21 | 3.274x | 1.233x | 4.036x | `ok` |

## Audit notes

All compiler-generated seeds, selected effective configs, and absolute per-round latencies are retained in `raw-results.json`.
TorchInductor options, generated-source hashes, Triton kernel counts, and forbidden-dispatch matches are also retained per shape.
Attention, recurrent, and state-space references are natural PyTorch formulations, not tiled replicas of the Helion algorithms; their ratios therefore include fusion and algorithmic differences.
Secondary compiler heuristics also fired after the expected formula/multi-matmul heuristic: `triton_skinny_gemm` on 10 shapes.

## Failures

- `jagged_hstu:b128_l131072_max1024_h4_d128`: `driver_error`: cell timed out after 1800s
