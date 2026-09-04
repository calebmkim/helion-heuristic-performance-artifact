# Benchmark Refresh Task

I want you to do one new run.
Re-run numbers using the new pinned versions (the pinned helion commit, torch/triton/cuda versions, etc. fla/vllm versions should be pinned as well — but they’re pretty consistent anyways).
For fla linear attention, please get numbers for *both* benchmarking results.

To be clear, you should run:
- vLLM numbers — and generate the new graph based on this — any old numbers from any stale commits can be removed (this applies to the other workloads as well — once we have up-to-date numbers, we can remove old legacy numbers since we have newer more up-to-date ones)
- Linear attention numbers — get both interleaved and legacy benchmarking numbers — and graphs for both.
- Examples reductions
- Other matmul kernels

Then report the results to me for each.

This is a long running task. That’s fine.
If there are unexpected crashes, failures, etc. try to fix it and continue autonomously.
I will NOT be babysitting you.
You have enough background to be able to do this task effectively.

Let me know if you have any questions.
Otherwise please begin.
