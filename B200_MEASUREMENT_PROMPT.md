Please clone https://github.com/calebmkim/helion-heuristic-performance-artifact and reproduce all four performance experiments on an NVIDIA B200.

The repository contains four experiments grouped by kernel family. The H100 results have been reliably reproduced using these scripts; your job is to produce fresh B200 results using the same experiment definitions. This is a measurement task. Do not optimize kernels, retune heuristics, or run additional Helion autotuning searches. Poor performance is a valid result.

Run the experiments in this order:

1. `linear-attention-fla`
2. `vllm-reductions`
3. `example-reductions`
4. `other-matmul-kernels`

Read the root README and each experiment's README and AGENT_PROMPT.md. The documentation explains the goals and measurement semantics; the scripts provide concrete starting points. Set up the required repositories and dependencies yourself, including Helion, FLA, and vLLM. Preserve the documented software revisions, workload selection rules, baseline definitions, correctness checks, and timing methods.

You may edit or replace benchmark scripts and adapters as needed to make these experiments work on B200. Inspect the actual checkouts and use your judgment to resolve installation, build, API, and architecture compatibility issues. Update H100-specific assumptions in the harness, including heuristic-name checks and plot labels, for the actual B200 behavior. Use the existing B200 architecture-specific AOT configurations. Retain meaningful adaptations and record why they were necessary. This prompt takes precedence where the repository's instructions describe an H100-only run or permit omitting a baseline required below.

All baseline comparisons are required:

- For linear attention, run both the controlled four-arm comparison and the native Helion harness. The four-arm comparison must include the base default, FLA Triton, heuristic seed, and B200 AOT-tuned configurations. Use the native harness to cross-check the corresponding AOT-versus-FLA result. I expect the corresponding overall results to be similar; investigate discrepancies and report the measured values honestly. Use the four-arm results for the main report and blog graphs, and retain the native results separately with a comparison of the two workflows.
- For vLLM reductions, include the base default, heuristic seed, exact B200 AOT-tuned configuration, and the actual vLLM CUDA/C++ baseline. Obtaining and running the vLLM extension is part of the task. Use the documented curated workload selection rules with the B200 AOT tables.
- For example reductions, include the base default, heuristic seed, and `torch.compile` max-autotune reference, preserving the existing lack of a Triton-only restriction.
- For other matmul kernels, include the base default, heuristic seed, and the existing Triton-only `torch.compile` max-autotune reference, preserving its backend restrictions and generated-code audit.

Run the full intended workload population for each experiment and save its manifest. Correctness-check every required arm before counting its timing. Resolve failed measurements and retain their diagnostic history; do not reduce the workload or remove baselines to make a run appear complete. All reported B200 measurements must come from this run.

Each experiment is finished when its intended workloads and required baselines have fresh, correctness-checked B200 measurements, and you have produced and visually inspected graphs matching the format and conventions of the existing blog graphs. Save raw per-shape measurements, selected configurations, correctness results, exact software and GPU information, reproducible commands, script changes, and a concise Markdown report alongside the graphs. Preserve the existing H100 artifacts and write the new B200 results to separate output directories.

This is an autonomous run. Make implementation decisions using your best judgment and write important decisions in the log instead of stopping to ask questions. Treat problems as work to investigate and resolve, and keep working until all four experiments are finished. A launched job, a partial sweep, or graphs covering only a successful subset do not constitute completion.

At the beginning, create a durable Markdown run log and paste this entire prompt verbatim at its top. Keep a current checkpoint immediately below it with completed work, running jobs and their identifiers, output locations, unresolved issues, and the next action. Record important commands, findings, adaptations, and decisions as you go. Reread the prompt and checkpoint after context compaction, and resume from the saved state. The log should contain everything needed to continue the run without relying on conversational memory.

Please begin and run autonomously until all four experiments, their reports, and their graphs are finished.
