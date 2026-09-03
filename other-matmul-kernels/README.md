# Other Matmul Kernels

This comparison measures Helion's H100 formula-matmul and multi-matmul
compiler heuristics outside the linear-attention corpus.

| Population | Kernel family | Shapes |
|---|---|---:|
| Formula matmul | Plain BF16 matmul | 5 |
| Formula matmul | BF16 x INT16 GEMM | 5 |
| Formula matmul | Broadcast matmul | 5 |
| Formula matmul | Gather GEMV | 5 |
| Formula matmul | Mamba-2 chunk state | 6 |
| Multi-matmul | Dense attention forward | 5 |
| Multi-matmul | Causal attention forward | 5 |
| Multi-matmul | Biased attention forward | 5 |
| Multi-matmul | Attention backward | 5 |
| Multi-matmul | Jagged HSTU attention | 6 |
| Multi-matmul | Squeeze-and-excitation forward | 6 |
| Multi-matmul | GDN forward-H | 6 |
| Multi-matmul | Mamba-2 chunk scan | 6 |
| **Total** | **13 families** | **70** |

The exact population and provenance are in [shapes.json](shapes.json). Most
shapes are the published B200 off-corpus cases. The manifest adds plain
matmuls, Mamba chunk-state counterparts, and one supplement each for four
multi-matmul families. These are realistic model regimes and deliberate
synthetic breadth, not a production trace.

## Comparison

Each shape compares:

| Arm | Meaning |
|---|---|
| `default` | `ConfigSpec._base_default_config()` |
| `seed` | The promoted rank-0 compiler heuristic seed, merged with base defaults |
| `torch_compile` | The example's natural PyTorch reference, compiled with max-autotune and Triton-only GEMM choices |

Helion autotuning and tuned-cache replay are disabled. The default report uses
`default = 1.00x`; the additional report uses `torch.compile = 1.00x`. Higher
is faster in both. All requested arms are compiled and timed in one fresh
process per shape.

TorchInductor GEMM and convolution choices are restricted to Triton. Generated
source is audited, and a cell is rejected if it dispatches through ATen or an
external GPU library. Attention uses explicit PyTorch matmul/softmax math, not
SDPA, so it cannot silently select FlashAttention.

The PyTorch references preserve the examples' natural formulations. Attention,
GDN, Mamba, and jagged HSTU are not algorithmically matched to Helion's tiled
implementations, so those ratios include algorithmic and fusion advantages.
Only the GDN reference has received a focused PyTorch optimization pass; the
other references may have additional optimization headroom. GDN uses PyTorch
`scan` to preserve its recurrence without unrolling every chunk. Epilogue
fusion is disabled for this reference because the current Inductor
max-autotune flop counter cannot handle data-dependent scan offsets during
fusion; its BMM choices remain Triton-only and max-autotuned.
This remains primarily a seed-quality artifact. A rigorous Helion quality
ceiling should also autotune every shape against the best same-kernel config.
Compilation is capped at 30 minutes per shape.

## Reproduce

Give [AGENT_PROMPT.md](AGENT_PROMPT.md) to an agent with the Helion revision
and H100 to measure. The files under [scripts](scripts/README.md) are adaptable
starting points and should be updated when Helion APIs move.

The included H100 run contains the
[report](generated/h100-eacfee67/summary.md),
[raw benchmark output](generated/h100-eacfee67/raw-results.json),
[summary JSON](generated/h100-eacfee67/results.json),
[per-shape CSV](generated/h100-eacfee67/per-shape.csv), and
[combined graph](generated/h100-eacfee67/per-family-bars.png).
The [blog-style graph](generated/h100-eacfee67/blog-figures/results-matmul-h100.png)
uses the same visual conventions as the other compile-time-heuristics figures.

A focused rerun of the six optimized GDN reference shapes is in
[generated/h100-eacfee67-gdn-optimized](generated/h100-eacfee67-gdn-optimized),
including raw output, summaries, and both graph styles.

Kernels that need workload or compiler work before they belong in the timed
population are tracked in [to-do-work](to-do-work/README.md).
