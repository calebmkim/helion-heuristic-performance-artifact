# Helion Heuristic Performance Artifact

This repository contains prompt-first protocols for measuring Helion compiler
heuristics against changing implementations and external baselines.

The workload remains in the project that owns it. These instructions tell an
agent how to discover that workload, select each comparison arm without cache
leakage, prove which configuration ran, and publish auditable results. Small
scripts validate the environment and aggregate a stable result format; they do
not duplicate Helion kernels or benchmark registries.

## Comparisons

| Comparison | Baselines | Status |
|---|---|---|
| [Linear attention vs FLA](linear-attention-fla/README.md) | Helion raw default, FLA Triton, Helion compiler heuristic, exact pre-tuned Helion | Ready |
| Matmul | To be added | Planned |
| Multi-matmul | To be added | Planned |
| Reduction | To be added | Planned |

## Measurement Contract

Every comparison follows the same rules:

1. Pin and report the Helion revision, baseline revision, Python packages, GPU,
   and timing method.
2. Treat the workload in the checked-out project as authoritative. Do not copy
   its shapes or kernels into this repository.
3. Run configuration-selection arms in isolated subprocesses.
4. Record the selected configuration and the evidence for its source.
5. Check correctness before including a latency.
6. Report missing or unsupported arms as unavailable, never as a fallback.
7. Derive ratios from absolute latency on an explicitly named common
   population.
8. Make every table higher-is-better.

## Quick Start

Clone Helion and FLA, prepare the environment using their own installation
instructions, then run:

```bash
python linear-attention-fla/scripts/check_environment.py \
  --helion /path/to/helion \
  --fla /path/to/flash-linear-attention
```

Give an agent [the linear-attention prompt](linear-attention-fla/AGENT_PROMPT.md)
with the local paths and desired revisions filled in. The expected output
contract is documented in
[RESULT_SCHEMA.md](linear-attention-fla/RESULT_SCHEMA.md).

The prior H100 run is retained as a
[reference result](linear-attention-fla/reference-results/H100_PR3546.md), not
as a timeless performance claim.
