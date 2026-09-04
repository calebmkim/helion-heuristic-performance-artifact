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
| [Linear attention vs FLA](linear-attention-fla/README.md) | Native Helion-vs-FLA reproduction; controlled default/FLA/seed/AOT comparison | Ready; new runs pin the shared Helion and compiler stack |
| [vLLM reductions](vllm-reductions/README.md) | Helion default, heuristic seed, exact AOT-tuned Helion, vLLM 0.24 CUDA/C++ | Ready; primary: Torch 2.13 / CUDA 13.2 / Triton 3.7.1 |
| [Example reductions](example-reductions/README.md) | Helion default, heuristic seed, torch.compile max-autotune | Ready; primary: Torch 2.13 / CUDA 13.2 / Triton 3.7.1 |
| [Other matmul kernels](other-matmul-kernels/README.md) | Helion raw default, formula/multi-matmul heuristic seed, Triton-only torch.compile max-autotune | Ready; primary: Torch 2.13 / CUDA 13.2 / Triton 3.7.1 |

The four primary launchers share
[`check_primary_stack.py`](check_primary_stack.py) as their
single executable stack contract: clean Helion
`fa2f62eb686ef846c76f8b9e18beec30fbc5bee1` from `pytorch/helion:main`,
PyTorch `2.13.0+cu132`, CUDA `13.2`, and Triton `3.7.1`.

## Documentation

Prose written to be read, as opposed to the generated `REPORT.md` and `summary.md`
files under each comparison's `generated/` directory. The two traces describe the
*structure* of each heuristic, which a port should preserve; the constants, cutoffs
and ramps in them are what a port is expected to re-fit.

- [Matmul heuristic trace](human-docs/MATMUL_HEURISTIC_HIGH_LEVEL_TRACE.md) -- the matmul and multi-matmul heuristics in decision order: what each stage knows, what it sets, and what a later stage may correct.
- [Reduction heuristic trace](human-docs/REDUCTION_HEURISTIC_HIGH_LEVEL_TRACE.md) -- the reduction heuristic after the candidate-resolved liveness rewrite, stage by stage.
- [Handoff](human-docs/human-written-handoff.md) -- what is benchmarked today, what is missing, and a prompt for porting the reduction heuristic to sm100/b200.
- [Blog post and supporting material](blog/README.md) -- the draft write-up, its figures, and the notes its numbers were checked against.

## Use

Assume the project being measured is already cloned. Open the comparison
folder, give its prompt and starter scripts to an agent, and let the agent
adapt them to that revision. A useful run should leave behind the adapted
runner, raw per-shape measurements, a concise summary, and requested plots.
