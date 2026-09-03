# Other Matmul Kernel Performance

- Profile: `h100-off-corpus-v1`
- GPU: `NVIDIA H100 80GB HBM3` (compute capability `[9, 0]`)
- Helion commit: `eacfee67c0fdbc5a1c068f16a3b2f9f15ce23eb7`
- Helion worktree dirty: `True`
- Arms: raw `ConfigSpec._base_default_config()` and the promoted rank-0 compiler heuristic seed; autotuning and tuned-cache replay were disabled.
- Timing: Helion do_bench CUDA device timing; median within do_bench, then median of balanced outer rounds; 3 outer rounds.
- Normalization: `default = 1.00x`; higher is faster.
- This is a seed-quality comparison, not an autotuned-quality comparison. A rigorous extension should autotune every shape.

## Aggregate performance

| Population | Valid shapes | Families | Default | Heuristic seed (cell geomean) | Heuristic seed (family macro-geomean) |
|---|---:|---:|---:|---:|---:|
| Formula matmul | 26/26 | 5/5 | 1.000x | 6.998x | 7.069x |
| Multi-matmul | 44/44 | 8/8 | 1.000x | 2.454x | 2.469x |
| **Overall** | **70/70** | **13/13** | **1.000x** | **3.622x** | **3.700x** |

The family macro-geomean gives every kernel family equal weight; the cell geomean gives every shape equal weight.

## Per-family performance

| Kernel family | Family ID | Valid shapes | Default | Heuristic seed | Default latency (us) | Seed latency (us) |
|---|---|---:|---:|---:|---:|---:|
| Plain BF16 matmul | `plain_matmul` | 5/5 | 1.000x | 12.319x | 1235.17 | 100.27 |
| BF16 x INT16 GEMM | `bf16xint16_gemm` | 5/5 | 1.000x | 8.036x | 1096.29 | 136.42 |
| Broadcast matmul | `broadcast_matmul` | 5/5 | 1.000x | 15.823x | 835.35 | 52.79 |
| Gather GEMV | `gather_gemv` | 5/5 | 1.000x | 2.077x | 155.07 | 74.68 |
| Mamba-2 chunk state | `mamba2_chunk_state` | 6/6 | 1.000x | 5.429x | 622.93 | 114.75 |
| Dense attention forward | `dense_attention` | 5/5 | 1.000x | 3.134x | 844.68 | 269.55 |
| Causal attention forward | `causal_attention` | 5/5 | 1.000x | 5.153x | 2109.13 | 409.31 |
| Biased attention forward | `biased_attention` | 5/5 | 1.000x | 1.508x | 119.16 | 79.00 |
| Attention backward | `attention_backward` | 5/5 | 1.000x | 1.989x | 449.82 | 226.19 |
| Jagged HSTU attention | `jagged_hstu` | 6/6 | 1.000x | 0.875x | 167.35 | 191.22 |
| Squeeze-and-excitation forward | `squeeze_excitation` | 6/6 | 1.000x | 4.970x | 75.26 | 15.14 |
| GDN forward-H | `gdn_forward_h` | 6/6 | 1.000x | 1.928x | 457.55 | 237.26 |
| Mamba-2 chunk scan | `mamba2_chunk_scan` | 6/6 | 1.000x | 3.396x | 798.10 | 234.99 |

## Per-shape performance

| Family | Shape | Dtype | Fired heuristic(s) | Default (us) | Seed (us) | Seed speedup | Status |
|---|---|---|---|---:|---:|---:|---|
| Plain BF16 matmul | `m1_k4096_n4096` | `bfloat16` | `triton_h100_formula_matmul`, `triton_skinny_gemm` | 148.03 | 23.01 | 6.434x | `ok` |
| Plain BF16 matmul | `m32_k4096_n11008` | `bfloat16` | `triton_h100_formula_matmul`, `triton_skinny_gemm` | 160.80 | 47.55 | 3.382x | `ok` |
| Plain BF16 matmul | `m512_k4096_n4096` | `bfloat16` | `triton_h100_formula_matmul`, `triton_skinny_gemm` | 680.86 | 33.18 | 20.518x | `ok` |
| Plain BF16 matmul | `m4096_k4096_n11008` | `bfloat16` | `triton_h100_formula_matmul` | 13025.02 | 508.90 | 25.595x | `ok` |
| Plain BF16 matmul | `m4096_k11008_n4096` | `bfloat16` | `triton_h100_formula_matmul` | 13619.04 | 548.48 | 24.831x | `ok` |
| BF16 x INT16 GEMM | `m1_k4096_n4096` | `bfloat16` | `triton_h100_formula_matmul`, `triton_skinny_gemm` | 148.58 | 25.47 | 5.833x | `ok` |
| BF16 x INT16 GEMM | `m32_k4096_n11008` | `bfloat16` | `triton_h100_formula_matmul`, `triton_skinny_gemm` | 160.67 | 50.08 | 3.208x | `ok` |
| BF16 x INT16 GEMM | `m512_k4096_n4096` | `bfloat16` | `triton_h100_formula_matmul`, `triton_skinny_gemm` | 671.46 | 64.00 | 10.491x | `ok` |
| BF16 x INT16 GEMM | `m4096_k4096_n11008` | `bfloat16` | `triton_h100_formula_matmul` | 13071.01 | 1078.66 | 12.118x | `ok` |
| BF16 x INT16 GEMM | `m65536_k1024_n1280` | `bfloat16` | `triton_h100_formula_matmul`, `triton_skinny_gemm` | 7558.18 | 536.54 | 14.087x | `ok` |
| Broadcast matmul | `b32_m1_k4096_n4096` | `bfloat16` | `triton_h100_formula_matmul`, `triton_skinny_gemm` | 150.66 | 25.20 | 5.978x | `ok` |
| Broadcast matmul | `b8_m128_k4096_n11008` | `bfloat16` | `triton_h100_formula_matmul`, `triton_skinny_gemm` | 3362.16 | 140.02 | 24.013x | `ok` |
| Broadcast matmul | `b16_m512_k768_n1024` | `bfloat16` | `triton_h100_formula_matmul`, `triton_skinny_gemm` | 478.30 | 29.25 | 16.353x | `ok` |
| Broadcast matmul | `b4_m1024_k4096_n4096` | `bfloat16` | `triton_h100_formula_matmul` | 5037.92 | 203.26 | 24.785x | `ok` |
| Broadcast matmul | `b64_m16_k1024_n4096` | `bfloat16` | `triton_h100_formula_matmul` | 333.26 | 19.55 | 17.045x | `ok` |
| Gather GEMV | `b8_s2048_n2` | `bfloat16` | `triton_h100_formula_matmul` | 47.90 | 14.94 | 3.206x | `ok` |
| Gather GEMV | `b8_s4096_n2` | `bfloat16` | `triton_h100_formula_matmul` | 99.10 | 36.77 | 2.695x | `ok` |
| Gather GEMV | `b8_s8192_n2` | `bfloat16` | `triton_h100_formula_matmul` | 216.82 | 108.10 | 2.006x | `ok` |
| Gather GEMV | `b8_s14336_n2` | `bfloat16` | `triton_h100_formula_matmul` | 503.65 | 339.22 | 1.485x | `ok` |
| Gather GEMV | `b64_s4096_n8` | `bfloat16` | `triton_h100_formula_matmul` | 172.96 | 115.26 | 1.501x | `ok` |
| Mamba-2 chunk state | `b1_h64_g8_t8192_c64_d64_s128` | `bfloat16` | `triton_h100_formula_matmul` | 421.82 | 121.41 | 3.474x | `ok` |
| Mamba-2 chunk state | `b2_h32_g8_t8192_c128_d128_s256` | `bfloat16` | `triton_h100_formula_matmul` | 1469.70 | 273.86 | 5.367x | `ok` |
| Mamba-2 chunk state | `b4_h64_g8_t4096_c128_d64_s128` | `bfloat16` | `triton_h100_formula_matmul` | 764.99 | 156.93 | 4.875x | `ok` |
| Mamba-2 chunk state | `b8_h32_g4_t2048_c256_d64_s128` | `bfloat16` | `triton_h100_formula_matmul` | 366.40 | 58.21 | 6.295x | `ok` |
| Mamba-2 chunk state | `b8_h80_g1_t4096_c256_d64_s128` | `bfloat16` | `triton_h100_formula_matmul` | 1795.60 | 227.20 | 7.903x | `ok` |
| Mamba-2 chunk state | `b1_h128_g8_t2048_c256_d64_s128` | `bfloat16` | `triton_h100_formula_matmul` | 187.26 | 33.09 | 5.660x | `ok` |
| Dense attention forward | `b1_h4_m512_n512_d64_f16` | `float16` | `triton_h100_multi_matmul` | 46.59 | 21.95 | 2.122x | `ok` |
| Dense attention forward | `b2_h32_m1024_n1024_d64_f16` | `float16` | `triton_h100_multi_matmul` | 423.26 | 105.47 | 4.013x | `ok` |
| Dense attention forward | `b8_h16_m2048_n2048_d64_bf16` | `bfloat16` | `triton_h100_multi_matmul` | 3202.98 | 753.34 | 4.252x | `ok` |
| Dense attention forward | `b4_h32_m4096_n4096_d128_bf16` | `bfloat16` | `triton_h100_multi_matmul` | 19860.26 | 4880.40 | 4.069x | `ok` |
| Dense attention forward | `b4_h16_m128_n4096_d64_bf16` | `bfloat16` | `triton_h100_multi_matmul` | 342.77 | 167.17 | 2.050x | `ok` |
| Causal attention forward | `b1_h8_s512_d128_f16` | `float16` | `triton_h100_multi_matmul` | 51.68 | 28.03 | 1.844x | `ok` |
| Causal attention forward | `b2_h32_s1024_d128_f16` | `float16` | `triton_h100_multi_matmul` | 726.08 | 115.65 | 6.278x | `ok` |
| Causal attention forward | `b8_h16_s2048_d128_bf16` | `bfloat16` | `triton_h100_multi_matmul` | 5177.65 | 808.74 | 6.402x | `ok` |
| Causal attention forward | `b4_h32_s4096_d128_bf16` | `bfloat16` | `triton_h100_multi_matmul` | 20531.81 | 2979.22 | 6.892x | `ok` |
| Causal attention forward | `b1_h16_s8192_d128_bf16` | `bfloat16` | `triton_h100_multi_matmul` | 10462.94 | 1470.78 | 7.114x | `ok` |
| Biased attention forward | `b1_h4_m128_n128_d64_f16` | `float16` | `triton_h100_multi_matmul` | 25.41 | 24.96 | 1.018x | `ok` |
| Biased attention forward | `b2_h8_m512_n512_d64_f16` | `float16` | `triton_h100_multi_matmul` | 61.76 | 42.62 | 1.449x | `ok` |
| Biased attention forward | `b2_h16_m1024_n1024_d64_f16` | `float16` | `triton_h100_multi_matmul` | 246.50 | 186.05 | 1.325x | `ok` |
| Biased attention forward | `b1_h16_m1024_n2048_d128_bf16` | `bfloat16` | `triton_h100_multi_matmul` | 518.75 | 189.38 | 2.739x | `ok` |
| Biased attention forward | `b4_h8_m256_n1024_d64_bf16` | `bfloat16` | `triton_h100_multi_matmul` | 119.71 | 82.08 | 1.458x | `ok` |
| Attention backward | `b1_h4_s256_d64_f16` | `float16` | `triton_h100_multi_matmul` | 47.36 | 48.03 | 0.986x | `ok` |
| Attention backward | `b2_h16_s512_d64_f16` | `float16` | `triton_h100_multi_matmul` | 61.70 | 51.07 | 1.208x | `ok` |
| Attention backward | `b2_h32_s1024_d64_f16` | `float16` | `triton_h100_multi_matmul` | 389.60 | 158.66 | 2.456x | `ok` |
| Attention backward | `b4_h32_s2048_d128_bf16` | `bfloat16` | `triton_h100_multi_matmul` | 5544.10 | 1745.10 | 3.177x | `ok` |
| Attention backward | `b1_h16_s4096_d128_bf16` | `bfloat16` | `triton_h100_multi_matmul` | 2917.86 | 871.71 | 3.347x | `ok` |
| Jagged HSTU attention | `b4_l320_max128_h8_d64` | `bfloat16` | `triton_h100_multi_matmul` | 16.42 | 20.32 | 0.808x | `ok` |
| Jagged HSTU attention | `b16_l3440_max512_h16_d64` | `bfloat16` | `triton_h100_multi_matmul` | 55.42 | 92.32 | 0.600x | `ok` |
| Jagged HSTU attention | `b8_l6656_max2048_h32_d128` | `bfloat16` | `triton_h100_multi_matmul` | 703.52 | 587.44 | 1.198x | `ok` |
| Jagged HSTU attention | `b32_l2485_max256_h8_d128` | `bfloat16` | `triton_h100_multi_matmul` | 33.86 | 30.78 | 1.100x | `ok` |
| Jagged HSTU attention | `b2_l7168_max4096_h16_d64` | `bfloat16` | `triton_h100_multi_matmul` | 577.28 | 900.61 | 0.641x | `ok` |
| Jagged HSTU attention | `b128_l131072_max1024_h4_d128` | `bfloat16` | `triton_h100_multi_matmul` | 1755.87 | 1600.10 | 1.097x | `ok` |
| Squeeze-and-excitation forward | `m256_n256_k16` | `bfloat16` | `triton_h100_multi_matmul` | 16.16 | 7.42 | 2.177x | `ok` |
| Squeeze-and-excitation forward | `m256_n512_k32` | `bfloat16` | `triton_h100_multi_matmul` | 39.97 | 9.57 | 4.177x | `ok` |
| Squeeze-and-excitation forward | `m128_n1024_k64` | `bfloat16` | `triton_h100_multi_matmul` | 125.22 | 16.22 | 7.718x | `ok` |
| Squeeze-and-excitation forward | `m64_n2048_k128` | `bfloat16` | `triton_h100_multi_matmul` | 418.94 | 43.42 | 9.648x | `ok` |
| Squeeze-and-excitation forward | `m1024_n1024_k256` | `bfloat16` | `triton_h100_multi_matmul` | 372.32 | 33.31 | 11.177x | `ok` |
| Squeeze-and-excitation forward | `m1_n256_k16` | `bfloat16` | `triton_h100_multi_matmul` | 14.40 | 7.23 | 1.991x | `ok` |
| GDN forward-H | `b1_t8192_h64_c64_d64_s128` | `bfloat16` | `triton_h100_multi_matmul` | 408.10 | 158.08 | 2.582x | `ok` |
| GDN forward-H | `b2_t8192_h32_c128_d128_s256` | `bfloat16` | `triton_h100_multi_matmul` | 753.63 | 287.78 | 2.619x | `ok` |
| GDN forward-H | `b4_t4096_h64_c128_d64_s128` | `bfloat16` | `triton_h100_multi_matmul` | 387.26 | 240.93 | 1.607x | `ok` |
| GDN forward-H | `b8_t2048_h32_c256_d64_s128` | `bfloat16` | `triton_h100_multi_matmul` | 222.78 | 168.18 | 1.325x | `ok` |
| GDN forward-H | `b8_t4096_h80_c256_d64_s128` | `bfloat16` | `triton_h100_multi_matmul` | 1436.83 | 804.83 | 1.785x | `ok` |
| GDN forward-H | `b1_t4096_h32_c64_d128_s256` | `bfloat16` | `triton_h100_multi_matmul` | 240.66 | 120.26 | 2.001x | `ok` |
| Mamba-2 chunk scan | `b1_h64_g8_t8192_c64_d64_s128` | `bfloat16` | `triton_h100_multi_matmul` | 377.41 | 115.52 | 3.267x | `ok` |
| Mamba-2 chunk scan | `b2_h32_g8_t8192_c128_d128_s256` | `bfloat16` | `triton_h100_multi_matmul` | 1276.42 | 393.39 | 3.245x | `ok` |
| Mamba-2 chunk scan | `b4_h64_g8_t4096_c128_d64_s128` | `bfloat16` | `triton_h100_multi_matmul` | 940.32 | 283.94 | 3.312x | `ok` |
| Mamba-2 chunk scan | `b8_h32_g4_t2048_c256_d64_s128` | `bfloat16` | `triton_h100_multi_matmul` | 619.02 | 177.09 | 3.496x | `ok` |
| Mamba-2 chunk scan | `b8_h80_g1_t4096_c256_d64_s128` | `bfloat16` | `triton_h100_multi_matmul` | 2972.45 | 768.96 | 3.866x | `ok` |
| Mamba-2 chunk scan | `b1_h128_g8_t2048_c256_d64_s128` | `bfloat16` | `triton_h100_multi_matmul` | 310.05 | 95.84 | 3.235x | `ok` |

## Audit notes

All compiler-generated seeds, selected effective configs, and absolute per-round latencies are retained in `raw-results.json`.
Secondary compiler heuristics also fired after the expected formula/multi-matmul heuristic: `triton_skinny_gemm` on 10 shapes.
