# Human-written documentation

Prose meant to be read by a person, as opposed to the generated `REPORT.md` and
`summary.md` files that sit under each comparison's `generated/` directory.

| File | What it is |
|---|---|
| [`MATMUL_HEURISTIC_HIGH_LEVEL_TRACE.md`](MATMUL_HEURISTIC_HIGH_LEVEL_TRACE.md) | The matmul and multi-matmul heuristics, in decision order: what each stage knows, what it sets, and what a later stage may correct. |
| [`REDUCTION_HEURISTIC_HIGH_LEVEL_TRACE.md`](REDUCTION_HEURISTIC_HIGH_LEVEL_TRACE.md) | The reduction heuristic after the candidate-resolved liveness rewrite, stage by stage. |
| [`human-written-handoff.md`](human-written-handoff.md) | What is benchmarked today, what is still missing, and a prompt for porting the reduction heuristic to sm100/b200. |

The two traces describe the *structure* of each heuristic and are the reference a
port should preserve; the constants, cutoffs and ramps in them are what a port is
expected to re-fit.
