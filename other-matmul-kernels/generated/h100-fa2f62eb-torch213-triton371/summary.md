# Other Matmul Kernel Performance

- Profile: `h100-off-corpus-v1`
- GPU: `NVIDIA H100 80GB HBM3` (compute capability `[9, 0]`)
- Helion commit: `fa2f62eb686ef846c76f8b9e18beec30fbc5bee1`
- Helion worktree dirty: `False`
- Arms: raw `ConfigSpec._base_default_config()`, the promoted rank-0 compiler heuristic seed, and each example's natural PyTorch reference under `torch.compile(mode="max-autotune-no-cudagraphs")`. Helion autotuning and tuned-cache replay were disabled; TorchInductor GEMMs were max-autotuned with Triton as the only permitted backend.
- Timing: Helion do_bench CUDA device timing; median within do_bench, then median of balanced outer rounds; 3 outer rounds.
- Normalization: `default = 1.00x`; higher is faster.
- This is a seed-quality comparison, not an autotuned-quality comparison. A rigorous extension should autotune every shape.

## Aggregate performance

| Population | Valid shapes | Families | Default | Heuristic seed (cell geomean) | Heuristic seed (family macro-geomean) |
|---|---:|---:|---:|---:|---:|
| Formula matmul | 26/26 | 5/5 | 1.000x | 6.986x | 7.052x |
| Multi-matmul | 44/44 | 8/8 | 1.000x | 2.444x | 2.458x |
| **Overall** | **70/70** | **13/13** | **1.000x** | **3.610x** | **3.686x** |

## Torch compile comparison

These ratios use `torch.compile = 1.00x`; higher is faster. Generated TorchInductor source must pass the Triton-only dispatch audit.

| Population | Valid shapes | Torch compile | Helion default (cell geomean) | Helion seed (cell geomean) | Helion default (family macro-geomean) | Helion seed (family macro-geomean) |
|---|---:|---:|---:|---:|---:|---:|
| Formula matmul | 26/26 | 1.000x | 0.170x | 1.188x | 0.163x | 1.151x |
| Multi-matmul | 44/44 | 1.000x | 1.082x | 2.644x | 1.038x | 2.551x |
| **Overall** | **70/70** | **1.000x** | **0.544x** | **1.964x** | **0.510x** | **1.878x** |

The family macro-geomean gives every kernel family equal weight; the cell geomean gives every shape equal weight.

## Per-family performance

| Kernel family | Family ID | Valid shapes | Default | Heuristic seed | Default latency (us) | Seed latency (us) |
|---|---|---:|---:|---:|---:|---:|
| Plain BF16 matmul | `plain_matmul` | 5/5 | 1.000x | 12.100x | 1239.66 | 102.45 |
| BF16 x INT16 GEMM | `bf16xint16_gemm` | 5/5 | 1.000x | 7.953x | 1102.62 | 138.64 |
| Broadcast matmul | `broadcast_matmul` | 5/5 | 1.000x | 15.747x | 842.45 | 53.50 |
| Gather GEMV | `gather_gemv` | 5/5 | 1.000x | 2.077x | 155.04 | 74.64 |
| Mamba-2 chunk state | `mamba2_chunk_state` | 6/6 | 1.000x | 5.539x | 623.16 | 112.50 |
| Dense attention forward | `dense_attention` | 5/5 | 1.000x | 3.077x | 851.75 | 276.79 |
| Causal attention forward | `causal_attention` | 5/5 | 1.000x | 5.098x | 2125.68 | 417.00 |
| Biased attention forward | `biased_attention` | 5/5 | 1.000x | 1.531x | 120.69 | 78.85 |
| Attention backward | `attention_backward` | 5/5 | 1.000x | 1.955x | 462.19 | 236.40 |
| Jagged HSTU attention | `jagged_hstu` | 6/6 | 1.000x | 0.877x | 168.10 | 191.59 |
| Squeeze-and-excitation forward | `squeeze_excitation` | 6/6 | 1.000x | 4.920x | 75.07 | 15.26 |
| GDN forward-H | `gdn_forward_h` | 6/6 | 1.000x | 1.930x | 456.80 | 236.69 |
| Mamba-2 chunk scan | `mamba2_chunk_scan` | 6/6 | 1.000x | 3.403x | 801.90 | 235.65 |

## Per-family torch compile comparison

| Kernel family | Valid shapes | Torch compile | Helion default | Helion seed | Torch compile latency (us) |
|---|---:|---:|---:|---:|---:|
| Plain BF16 matmul | 5/5 | 1.000x | 0.085x | 1.027x | 105.18 |
| BF16 x INT16 GEMM | 5/5 | 1.000x | 0.096x | 0.767x | 106.38 |
| Broadcast matmul | 5/5 | 1.000x | 0.067x | 1.048x | 56.07 |
| Gather GEMV | 5/5 | 1.000x | 0.453x | 0.942x | 70.29 |
| Mamba-2 chunk state | 6/6 | 1.000x | 0.469x | 2.598x | 292.32 |
| Dense attention forward | 5/5 | 1.000x | 0.611x | 1.881x | 520.62 |
| Causal attention forward | 5/5 | 1.000x | 0.507x | 2.583x | 1077.28 |
| Biased attention forward | 5/5 | 1.000x | 0.635x | 0.971x | 76.59 |
| Attention backward | 5/5 | 1.000x | 0.949x | 1.856x | 438.80 |
| Jagged HSTU attention | 6/6 | 1.000x | 3.722x | 3.265x | 625.64 |
| Squeeze-and-excitation forward | 6/6 | 1.000x | 0.400x | 1.969x | 30.04 |
| GDN forward-H | 6/6 | 1.000x | 4.469x | 8.624x | 2041.23 |
| Mamba-2 chunk scan | 6/6 | 1.000x | 1.086x | 3.696x | 870.88 |

## Per-shape performance

| Family | Shape | Dtype | Fired heuristic(s) | Default (us) | Seed (us) | Torch compile (us) | Seed/default | Default/torch compile | Seed/torch compile | Status |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---|
| Plain BF16 matmul | `m1_k4096_n4096` | `bfloat16` | `triton_h100_formula_matmul`, `triton_skinny_gemm` | 147.97 | 23.14 | 28.48 | 6.396x | 0.192x | 1.231x | `ok` |
| Plain BF16 matmul | `m32_k4096_n11008` | `bfloat16` | `triton_h100_formula_matmul`, `triton_skinny_gemm` | 160.93 | 46.75 | 44.70 | 3.442x | 0.278x | 0.956x | `ok` |
| Plain BF16 matmul | `m512_k4096_n4096` | `bfloat16` | `triton_h100_formula_matmul`, `triton_skinny_gemm` | 677.12 | 33.18 | 34.11 | 20.405x | 0.050x | 1.028x | `ok` |
| Plain BF16 matmul | `m4096_k4096_n11008` | `bfloat16` | `triton_h100_formula_matmul` | 13300.34 | 577.38 | 548.83 | 23.036x | 0.041x | 0.951x | `ok` |
| Plain BF16 matmul | `m4096_k11008_n4096` | `bfloat16` | `triton_h100_formula_matmul` | 13651.65 | 544.61 | 540.00 | 25.067x | 0.040x | 0.992x | `ok` |
| BF16 x INT16 GEMM | `m1_k4096_n4096` | `bfloat16` | `triton_h100_formula_matmul`, `triton_skinny_gemm` | 148.51 | 25.60 | 28.38 | 5.801x | 0.191x | 1.109x | `ok` |
| BF16 x INT16 GEMM | `m32_k4096_n11008` | `bfloat16` | `triton_h100_formula_matmul`, `triton_skinny_gemm` | 160.38 | 49.47 | 49.12 | 3.242x | 0.306x | 0.993x | `ok` |
| BF16 x INT16 GEMM | `m512_k4096_n4096` | `bfloat16` | `triton_h100_formula_matmul`, `triton_skinny_gemm` | 680.54 | 63.58 | 57.54 | 10.703x | 0.085x | 0.905x | `ok` |
| BF16 x INT16 GEMM | `m4096_k4096_n11008` | `bfloat16` | `triton_h100_formula_matmul` | 13297.15 | 1135.68 | 597.09 | 11.709x | 0.045x | 0.526x | `ok` |
| BF16 x INT16 GEMM | `m65536_k1024_n1280` | `bfloat16` | `triton_h100_formula_matmul`, `triton_skinny_gemm` | 7561.25 | 560.16 | 284.43 | 13.498x | 0.038x | 0.508x | `ok` |
| Broadcast matmul | `b32_m1_k4096_n4096` | `bfloat16` | `triton_h100_formula_matmul`, `triton_skinny_gemm` | 150.78 | 25.31 | 26.82 | 5.957x | 0.178x | 1.059x | `ok` |
| Broadcast matmul | `b8_m128_k4096_n11008` | `bfloat16` | `triton_h100_formula_matmul`, `triton_skinny_gemm` | 3415.68 | 145.46 | 146.53 | 23.483x | 0.043x | 1.007x | `ok` |
| Broadcast matmul | `b16_m512_k768_n1024` | `bfloat16` | `triton_h100_formula_matmul`, `triton_skinny_gemm` | 477.92 | 29.98 | 29.60 | 15.939x | 0.062x | 0.987x | `ok` |
| Broadcast matmul | `b4_m1024_k4096_n4096` | `bfloat16` | `triton_h100_formula_matmul` | 5159.97 | 204.42 | 205.18 | 25.242x | 0.040x | 1.004x | `ok` |
| Broadcast matmul | `b64_m16_k1024_n4096` | `bfloat16` | `triton_h100_formula_matmul` | 334.11 | 19.42 | 23.23 | 17.201x | 0.070x | 1.196x | `ok` |
| Gather GEMV | `b8_s2048_n2` | `bfloat16` | `triton_h100_formula_matmul` | 47.71 | 14.82 | 23.52 | 3.220x | 0.493x | 1.587x | `ok` |
| Gather GEMV | `b8_s4096_n2` | `bfloat16` | `triton_h100_formula_matmul` | 99.07 | 36.80 | 29.98 | 2.692x | 0.303x | 0.815x | `ok` |
| Gather GEMV | `b8_s8192_n2` | `bfloat16` | `triton_h100_formula_matmul` | 216.70 | 108.00 | 94.75 | 2.007x | 0.437x | 0.877x | `ok` |
| Gather GEMV | `b8_s14336_n2` | `bfloat16` | `triton_h100_formula_matmul` | 504.19 | 339.71 | 269.86 | 1.484x | 0.535x | 0.794x | `ok` |
| Gather GEMV | `b64_s4096_n8` | `bfloat16` | `triton_h100_formula_matmul` | 173.47 | 115.81 | 95.17 | 1.498x | 0.549x | 0.822x | `ok` |
| Mamba-2 chunk state | `b1_h64_g8_t8192_c64_d64_s128` | `bfloat16` | `triton_h100_formula_matmul` | 422.06 | 121.10 | 218.56 | 3.485x | 0.518x | 1.805x | `ok` |
| Mamba-2 chunk state | `b2_h32_g8_t8192_c128_d128_s256` | `bfloat16` | `triton_h100_formula_matmul` | 1491.71 | 276.54 | 446.42 | 5.394x | 0.299x | 1.614x | `ok` |
| Mamba-2 chunk state | `b4_h64_g8_t4096_c128_d64_s128` | `bfloat16` | `triton_h100_formula_matmul` | 755.17 | 157.54 | 387.71 | 4.794x | 0.513x | 2.461x | `ok` |
| Mamba-2 chunk state | `b8_h32_g4_t2048_c256_d64_s128` | `bfloat16` | `triton_h100_formula_matmul` | 369.41 | 54.62 | 193.06 | 6.763x | 0.523x | 3.534x | `ok` |
| Mamba-2 chunk state | `b8_h80_g1_t4096_c256_d64_s128` | `bfloat16` | `triton_h100_formula_matmul` | 1780.16 | 228.99 | 859.38 | 7.774x | 0.483x | 3.753x | `ok` |
| Mamba-2 chunk state | `b1_h128_g8_t2048_c256_d64_s128` | `bfloat16` | `triton_h100_formula_matmul` | 187.30 | 30.72 | 99.42 | 6.097x | 0.531x | 3.236x | `ok` |
| Dense attention forward | `b1_h4_m512_n512_d64_f16` | `float16` | `triton_h100_multi_matmul` | 46.37 | 23.90 | 47.65 | 1.940x | 1.028x | 1.993x | `ok` |
| Dense attention forward | `b2_h32_m1024_n1024_d64_f16` | `float16` | `triton_h100_multi_matmul` | 427.87 | 107.62 | 253.01 | 3.976x | 0.591x | 2.351x | `ok` |
| Dense attention forward | `b8_h16_m2048_n2048_d64_bf16` | `bfloat16` | `triton_h100_multi_matmul` | 3254.30 | 760.26 | 1862.05 | 4.281x | 0.572x | 2.449x | `ok` |
| Dense attention forward | `b4_h32_m4096_n4096_d128_bf16` | `bfloat16` | `triton_h100_multi_matmul` | 20124.62 | 4929.25 | 8257.68 | 4.083x | 0.410x | 1.675x | `ok` |
| Dense attention forward | `b4_h16_m128_n4096_d64_bf16` | `bfloat16` | `triton_h100_multi_matmul` | 345.02 | 168.51 | 206.34 | 2.047x | 0.598x | 1.224x | `ok` |
| Causal attention forward | `b1_h8_s512_d128_f16` | `float16` | `triton_h100_multi_matmul` | 51.94 | 29.98 | 49.28 | 1.732x | 0.949x | 1.644x | `ok` |
| Causal attention forward | `b2_h32_s1024_d128_f16` | `float16` | `triton_h100_multi_matmul` | 727.42 | 116.11 | 324.67 | 6.265x | 0.446x | 2.796x | `ok` |
| Causal attention forward | `b8_h16_s2048_d128_bf16` | `bfloat16` | `triton_h100_multi_matmul` | 5277.22 | 808.93 | 2315.04 | 6.524x | 0.439x | 2.862x | `ok` |
| Causal attention forward | `b4_h32_s4096_d128_bf16` | `bfloat16` | `triton_h100_multi_matmul` | 20632.40 | 3012.08 | 8867.42 | 6.850x | 0.430x | 2.944x | `ok` |
| Causal attention forward | `b1_h16_s8192_d128_bf16` | `bfloat16` | `triton_h100_multi_matmul` | 10550.59 | 1486.34 | 4417.49 | 7.098x | 0.419x | 2.972x | `ok` |
| Biased attention forward | `b1_h4_m128_n128_d64_f16` | `float16` | `triton_h100_multi_matmul` | 26.82 | 23.87 | 41.18 | 1.123x | 1.536x | 1.725x | `ok` |
| Biased attention forward | `b2_h8_m512_n512_d64_f16` | `float16` | `triton_h100_multi_matmul` | 61.89 | 42.94 | 47.71 | 1.441x | 0.771x | 1.111x | `ok` |
| Biased attention forward | `b2_h16_m1024_n1024_d64_f16` | `float16` | `triton_h100_multi_matmul` | 250.14 | 189.76 | 159.39 | 1.318x | 0.637x | 0.840x | `ok` |
| Biased attention forward | `b1_h16_m1024_n2048_d128_bf16` | `bfloat16` | `triton_h100_multi_matmul` | 511.65 | 190.32 | 184.70 | 2.688x | 0.361x | 0.970x | `ok` |
| Biased attention forward | `b4_h8_m256_n1024_d64_bf16` | `bfloat16` | `triton_h100_multi_matmul` | 120.54 | 82.34 | 45.57 | 1.464x | 0.378x | 0.553x | `ok` |
| Attention backward | `b1_h4_s256_d64_f16` | `float16` | `triton_h100_multi_matmul` | 53.02 | 54.56 | 62.85 | 0.972x | 1.185x | 1.152x | `ok` |
| Attention backward | `b2_h16_s512_d64_f16` | `float16` | `triton_h100_multi_matmul` | 62.18 | 50.59 | 78.62 | 1.229x | 1.265x | 1.554x | `ok` |
| Attention backward | `b2_h32_s1024_d64_f16` | `float16` | `triton_h100_multi_matmul` | 388.32 | 165.17 | 473.60 | 2.351x | 1.220x | 2.867x | `ok` |
| Attention backward | `b4_h32_s2048_d128_bf16` | `bfloat16` | `triton_h100_multi_matmul` | 5589.79 | 1826.58 | 3793.54 | 3.060x | 0.679x | 2.077x | `ok` |
| Attention backward | `b1_h16_s4096_d128_bf16` | `bfloat16` | `triton_h100_multi_matmul` | 2947.30 | 886.62 | 1832.48 | 3.324x | 0.622x | 2.067x | `ok` |
| Jagged HSTU attention | `b4_l320_max128_h8_d64` | `bfloat16` | `triton_h100_multi_matmul` | 16.70 | 20.77 | 98.66 | 0.804x | 5.906x | 4.750x | `ok` |
| Jagged HSTU attention | `b16_l3440_max512_h16_d64` | `bfloat16` | `triton_h100_multi_matmul` | 55.17 | 91.84 | 292.22 | 0.601x | 5.297x | 3.182x | `ok` |
| Jagged HSTU attention | `b8_l6656_max2048_h32_d128` | `bfloat16` | `triton_h100_multi_matmul` | 705.57 | 588.08 | 962.94 | 1.200x | 1.365x | 1.637x | `ok` |
| Jagged HSTU attention | `b32_l2485_max256_h8_d128` | `bfloat16` | `triton_h100_multi_matmul` | 33.86 | 30.66 | 528.82 | 1.104x | 15.620x | 17.250x | `ok` |
| Jagged HSTU attention | `b2_l7168_max4096_h16_d64` | `bfloat16` | `triton_h100_multi_matmul` | 581.73 | 897.65 | 1011.84 | 0.648x | 1.739x | 1.127x | `ok` |
| Jagged HSTU attention | `b128_l131072_max1024_h4_d128` | `bfloat16` | `triton_h100_multi_matmul` | 1762.24 | 1602.50 | 4037.34 | 1.100x | 2.291x | 2.519x | `ok` |
| Squeeze-and-excitation forward | `m256_n256_k16` | `bfloat16` | `triton_h100_multi_matmul` | 16.06 | 7.65 | 29.47 | 2.100x | 1.835x | 3.854x | `ok` |
| Squeeze-and-excitation forward | `m256_n512_k32` | `bfloat16` | `triton_h100_multi_matmul` | 39.71 | 9.79 | 25.34 | 4.056x | 0.638x | 2.588x | `ok` |
| Squeeze-and-excitation forward | `m128_n1024_k64` | `bfloat16` | `triton_h100_multi_matmul` | 124.00 | 16.32 | 30.21 | 7.598x | 0.244x | 1.851x | `ok` |
| Squeeze-and-excitation forward | `m64_n2048_k128` | `bfloat16` | `triton_h100_multi_matmul` | 419.65 | 43.55 | 29.79 | 9.636x | 0.071x | 0.684x | `ok` |
| Squeeze-and-excitation forward | `m1024_n1024_k256` | `bfloat16` | `triton_h100_multi_matmul` | 372.67 | 32.35 | 30.98 | 11.519x | 0.083x | 0.957x | `ok` |
| Squeeze-and-excitation forward | `m1_n256_k16` | `bfloat16` | `triton_h100_multi_matmul` | 14.46 | 7.33 | 35.30 | 1.974x | 2.440x | 4.817x | `ok` |
| GDN forward-H | `b1_t8192_h64_c64_d64_s128` | `bfloat16` | `triton_h100_multi_matmul` | 403.90 | 158.98 | 4242.94 | 2.541x | 10.505x | 26.689x | `ok` |
| GDN forward-H | `b2_t8192_h32_c128_d128_s256` | `bfloat16` | `triton_h100_multi_matmul` | 752.77 | 287.55 | 2485.92 | 2.618x | 3.302x | 8.645x | `ok` |
| GDN forward-H | `b4_t4096_h64_c128_d64_s128` | `bfloat16` | `triton_h100_multi_matmul` | 386.24 | 240.51 | 1576.29 | 1.606x | 4.081x | 6.554x | `ok` |
| GDN forward-H | `b8_t2048_h32_c256_d64_s128` | `bfloat16` | `triton_h100_multi_matmul` | 223.04 | 167.71 | 646.62 | 1.330x | 2.899x | 3.856x | `ok` |
| GDN forward-H | `b8_t4096_h80_c256_d64_s128` | `bfloat16` | `triton_h100_multi_matmul` | 1438.46 | 806.78 | 3023.44 | 1.783x | 2.102x | 3.748x | `ok` |
| GDN forward-H | `b1_t4096_h32_c64_d128_s256` | `bfloat16` | `triton_h100_multi_matmul` | 241.15 | 118.18 | 2225.41 | 2.041x | 9.228x | 18.831x | `ok` |
| Mamba-2 chunk scan | `b1_h64_g8_t8192_c64_d64_s128` | `bfloat16` | `triton_h100_multi_matmul` | 377.18 | 114.98 | 471.02 | 3.281x | 1.249x | 4.097x | `ok` |
| Mamba-2 chunk scan | `b2_h32_g8_t8192_c128_d128_s256` | `bfloat16` | `triton_h100_multi_matmul` | 1291.87 | 391.74 | 909.47 | 3.298x | 0.704x | 2.322x | `ok` |
| Mamba-2 chunk scan | `b4_h64_g8_t4096_c128_d64_s128` | `bfloat16` | `triton_h100_multi_matmul` | 948.11 | 284.10 | 1045.17 | 3.337x | 1.102x | 3.679x | `ok` |
| Mamba-2 chunk scan | `b8_h32_g4_t2048_c256_d64_s128` | `bfloat16` | `triton_h100_multi_matmul` | 619.46 | 179.26 | 748.96 | 3.456x | 1.209x | 4.178x | `ok` |
| Mamba-2 chunk scan | `b8_h80_g1_t4096_c256_d64_s128` | `bfloat16` | `triton_h100_multi_matmul` | 2995.10 | 782.14 | 3444.03 | 3.829x | 1.150x | 4.403x | `ok` |
| Mamba-2 chunk scan | `b1_h128_g8_t2048_c256_d64_s128` | `bfloat16` | `triton_h100_multi_matmul` | 310.21 | 95.46 | 377.76 | 3.250x | 1.218x | 3.957x | `ok` |

## Audit notes

All compiler-generated seeds, selected effective configs, and absolute per-round latencies are retained in `raw-results.json`.
TorchInductor options, generated-source hashes, Triton kernel counts, and forbidden-dispatch matches are also retained per shape.
Attention, recurrent, and state-space references are natural PyTorch formulations, not tiled replicas of the Helion algorithms; their ratios therefore include fusion and algorithmic differences.
Secondary compiler heuristics also fired after the expected formula/multi-matmul heuristic: `triton_skinny_gemm` on 10 shapes.
