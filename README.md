# Helion Heuristic Performance Artifact

This repository contains prompts and adaptable starter scripts for comparing
Helion compiler heuristics with defaults, tuned configurations, and external
implementations.

The scripts are examples, not stable interfaces. An agent should inspect the
requested checkout, use them as a starting point, and retain meaningful
adaptations with the results.

## Comparisons

| Comparison | Baselines | Status |
|---|---|---|
| [Linear attention vs FLA](linear-attention-fla/README.md) | Helion default, FLA Triton, heuristic seed, AOT-tuned Helion | Ready |
| [vLLM reductions](vllm-reductions/README.md) | Helion default, heuristic seed, exact AOT-tuned Helion, vLLM CUDA/C++ | Ready |
| [Example reductions](example-reductions/README.md) | Helion default, heuristic seed, torch.compile max-autotune | Ready |
| [Other matmul kernels](other-matmul-kernels/README.md) | Helion raw default, formula/multi-matmul heuristic seed, Triton-only torch.compile max-autotune | Ready |

## Use

Assume the project being measured is already cloned. Open the comparison
folder, give its prompt and starter scripts to an agent, and let the agent
adapt them to that revision. A useful run should leave behind the adapted
runner, raw per-shape measurements, a concise summary, and requested plots.
