# Example Reductions

> **For all new reproductions, use PyTorch `2.13.0+cu132` (CUDA `13.2`)
> with Triton `3.7.1`.**

This comparison measures ten general reduction kernels from a Helion checkout:

| Arm | Meaning |
|---|---|
| `default` | Helion's base configuration before compiler heuristic promotion |
| `seed` | Helion's compiler-selected seed with search and tuned caches disabled |
| `torch_compile` | The equivalent Torch implementation compiled with `max-autotune-no-cudagraphs` |

All three arms are required. Results are normalized to
`torch_compile = 1.00x`; higher is faster.

## Kernels

- `rms_norm`
- `layer_norm`
- `softmax`
- `cross_entropy`
- `kl_div`
- `jsd`
- `fused_linear_jsd` (`jsd_kernel`, the reduction kernel inside the fused example)
- `grpo`
- `rms_norm_bwd`
- `layer_norm_bwd`

## Population

The fixed `liger-mixed` profile has eight shapes per kernel and 80 cells total.
The complete machine-readable population and per-shape provenance are in
[shapes.json](shapes.json).

| Group | Kernels | Shapes |
|---|---|---|
| Normalization | RMSNorm, LayerNorm, and their backward kernels | `(2048, H)` for Liger hidden sizes `H={2048,3584,4096,5120,7168,8192}`, plus `(128,4096)` and `(8192,4096)` |
| Softmax | Softmax | Four Liger widths, small/large row variants, and widths `256` and `32768` for short- and long-context softmax |
| Vocabulary | Cross entropy, KL, JSD, fused-example JSD | `(2048, V)` for every unique Liger vocabulary, small/large token variants at `V=128256`, and `V=256000` |
| GRPO | GRPO forward | Liger's four explicit `(B,512,32000)` shapes plus four model-anchored longer-sequence/wider-vocabulary shapes |

The primary source is Liger-Kernel's model registry and benchmark suite at
commit `64594266bbed8db2c428575f46298d77da94f7a7`. Liger does not cover the
full operational shape range of these kernels: its softmax benchmark omits
important attention widths, its low-level GRPO benchmark fixes
`T=512, V=32000`, its registry omits the 256k-vocabulary regime, and its fixed
model sweep has limited outer-shape variation. The supplements fill only those
gaps while retaining real model dimensions.

## Method

Each cell runs in a fresh process. That process constructs, correctness-checks,
compiles, and CUDA-graph-captures all three arms before timing any of them.
Their graph replays are then timed together with rotating and reversed order,
cold L2, and CUDA events. Reported latency is device time in microseconds;
compilation, input construction, correctness checks, and CPU launch overhead
are excluded.

The Torch arm is hot execution of:

```python
torch.compile(reference, mode="max-autotune-no-cudagraphs")
```

It implements every observable output of the corresponding Helion kernel.

## Primary Software Stack

The primary reproduction target is the matched software stack:

- PyTorch `2.13.0+cu132`
- CUDA `13.2`
- Triton `3.7.1`

PyTorch 2.13 declares an exact dependency on Triton 3.7.1. Do not install a
package into the benchmark environment that replaces either member of this
pair. The starter script checks both versions before launching.

## Primary H100 Run

The primary run measured all 80 cells successfully on an H100 using Helion
main commit `fa2f62eb686ef846c76f8b9e18beec30fbc5bee1`, PyTorch `2.13.0+cu132`,
CUDA `13.2`, and Triton `3.7.1`. The heuristic seed reached `1.088x` versus
`torch.compile`; the base default reached `0.268x`. The reported values are
CUDA device time; CPU launch overhead is excluded.

- [report](generated/h100-main-torch213-triton371-liger-mixed/REPORT.md)
- [per-kernel graph](generated/h100-main-torch213-triton371-liger-mixed/per-kernel-performance.png)
- [blog graph](generated/blog-figures/results-example-reductions-h100.png)
- [combined raw results](generated/h100-main-torch213-triton371-liger-mixed/benchmark.json)
- [per-cell CSV](generated/h100-main-torch213-triton371-liger-mixed/per_cell.csv)

## Historical H100 Run

The former primary dataset is retained at
[generated/historical/torch212-triton370-h100-pr3551-liger-mixed](generated/historical/torch212-triton370-h100-pr3551-liger-mixed/).
It used Helion commit `6ca445ca0605f703d44967dbedd153a0a89a5e00`,
PyTorch `2.12.0+cu132`, and Triton `3.7.0`, and reported a `1.073x`
heuristic seed and `0.253x` base default. Treat it as a historical result, not
the default reproduction target. In particular, the `torch.compile`
performance of fused JSD and LayerNorm backward changed materially between
the two software stacks.

## Reproduce

Give [AGENT_PROMPT.md](AGENT_PROMPT.md) to the reproducing agent. The files
under [scripts](scripts/README.md) are adaptable starting points:

```bash
export HELION_ROOT=/path/to/helion
export OUTPUT_DIR=/path/to/results
export CUDA_VISIBLE_DEVICES=<gpu>
# Defaults enforced by starter.sh:
export REQUIRED_TORCH_VERSION=2.13.0+cu132
export REQUIRED_CUDA_VERSION=13.2
export REQUIRED_TRITON_VERSION=3.7.1

example-reductions/scripts/starter.sh
```

The final report includes raw per-cell latency, selected configurations,
correctness, exact revisions, a per-kernel table, and a graph.
