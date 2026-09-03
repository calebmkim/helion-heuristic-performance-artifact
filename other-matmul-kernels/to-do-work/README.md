# To-Do Work

These kernels are relevant to matmul heuristics but are not included in the
timed 70-shape population.

| Kernel | Why it is deferred |
|---|---|
| FlexAttention | Its current unresolved tile extents prevent the H100 formula/multi-matmul heuristic from firing. |
| Jagged dense BMM | The target heuristic does not currently fire, and the workload has BF16 reduction-order accuracy sensitivity. |
| D64 causal attention | The multi-matmul heuristic fires, but both raw default and seed reproduce the same Triton reshape/layout compile failure. |
| Grouped GEMM | Needs a principled grouped workload population and clarity on which compiler heuristic should own it. |
| MoE/grouped-expert GEMM | Needs production-representative token distributions, expert counts, and routing imbalance rather than a nominal dense shape. |

The known D64 causal cases are:

```text
(B,H,S,D,dtype)
(1, 8, 512, 64, FP16)
(2, 32, 1024, 64, FP16)
(8, 16, 2048, 64, BF16)
```

The two larger published causal cases already use D128 and remain in the timed
population. Three D128 supplements replace the failing D64 rows so the current
artifact has five compilable causal comparisons.
