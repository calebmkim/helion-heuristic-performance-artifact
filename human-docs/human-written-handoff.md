## BENCHMARKING TODO

Currently we have comparisons for.

linear attention FLA: 
- default no-autotune vs. heuristic no-autotune (this work) vs. aot tuned config table vs. handwritten baseline (triton FLA)
- for both h100 and b200
- TODO: re-bench b200 numbers from scratch using this repo's harness. Two reasons. (a) They were produced by a different harness than the h100 numbers, with a different `default` arm -- here `default` is the unseeded base config, whereas the b200 run's baseline was the *previous compiler-selected default*. (b) The heuristic has moved since, so configs really are different and a few individual cells got slower -- follow-up work changed the emitted config on 41 of 432 linear-attention cells, and across all 432 that nets out to 0.991x. So I wouldn't expect perf to change very much -- but still good to have.

vLLM reductions:
- default no autotune vs. heuristic no-autotune (this work) vs. aot tuned config table vs. handwritten baseline (cuda/c++ vLLM kernels)
- only for h100
- TODO: get b200 numbers. Use vLLM's **b200** config table for the aot arm (`nvidia_b200.json`), not the h100 table replayed on blackwell. Make sure to use the existing harness (make changes if you need). 

other matmul kernels (attention + variants, mamba, squeeze-excitation, etc.):
- default no autotune vs. heuristic no-auttoune (this work) vs. torch.compile max-autotune
- only for h100
- TODO: get b200 numbers
- OPTIONAL TODO (will take a long time): run autotuning on each shape and obtain configs for each shape

example reductions (e.g., rms norm, layer norm, cross entropy, etc.): 
- default no autotune vs. heuristic no-auttoune (this work) vs. torch.compile max-autotune
- only for h100
- TODO: get b200 numbers
- OPTIONAL TODO (will tkae a long time): run autotuning on each shape and obtain configs for each shape

## ANOTHER TODO (NON-TRIVIAL, COULD TAKE LONGER)

currently there is 1 reduction heuristic: tuned only for h100. it might slightly underperform on b200.

My usually strategy for porting heuristics from one backend to another, is that you should keep the *structure* of the heuristic the same, you should just adjust empirically tuned constants. 
Especially for reductions, I've found heuristics translate reasonably well, so the port should (hopefully) be straightforward.

Here is one prompt you can give to create to port the h100 heuristic to b200.
(as a side note, I already did something similar in: https://github.com/pytorch/helion/pull/3546.)

### PROMPT: ###

Your job is to port the existing reduction heuristic in helion, which has been tuned for sm90/h100, to sm100/b200.
There is already an outline for this in https://github.com/pytorch/helion/pull/3546, which ported matmul/multi-matmul heuristics from b200 to h100.
Your job is porting reduction heuristics from h100 to b200 -- note that is the *opposite* direction to that PR, so use it as a template for the process, not for which arch is the source.
This should actualy be easier than https://github.com/pytorch/helion/pull/3546: that PR had to worry about tmem vs. registers, but because this is a reduction, you don't have to worry about that.
It can be more of a direct port.

NOTE: before https://github.com/pytorch/helion/pull/3551, there were separate h100 / b200 reductions.
If I remember correctly, the heuristics were similar in structure, it's just that the num warps ramp needed to be adjusted a bit.

However, that PR rewrote and simplified the reduction heuristics. 
The overall structure of the heuristic is specified here: https://github.com/calebmkim/helion-heuristic-performance-artifact/blob/main/human-docs/REDUCTION_HEURISTIC_HIGH_LEVEL_TRACE.md.

When you port the heuristic over, you should keep the same structure. 
You should only tune constants, cutoffs, ramps, etc. 
But the overall structure should be the same.
NOTE: this means there should actually be very few meaningful LoC changes.
If the performance of b200 underperforms compared to h100 heuristics, that's fine -- just report it honestly. 
Don't go making drastic changes to the heuristic just to improve perf.

You should evaluate the changes made based on the vllm-reduction kernels, shapes, and the examples-reduction kernels,shapes in this repo.

Here is the workflow. 

(1) Confirm the heuristic already fires on b200, and record the baseline.
Do not expect to find something broken here: `TritonReductionHeuristic` is registered once for both h100 and b200, both currently share one sizing and warp policy, and the real SM count already feeds the launch-wave math. So there is probably nothing to "port" in the enabling sense -- the actual work of this task is re-fitting the constants for b200, which is steps (2) onward.

(2) Check performance across the vllm-reduction and examples-reduction cirriculum, and compare performance.

(3) Look at the worst performing kernels and shapes (compare performance relative to aot config or torch.compile baseline for vllm and examples, respectively), and examine why they are losing. 
If there are aot pre-tuned configs, compare the configs.
Come up with an explanation as to why it is underperforming, based on what's going on the GPU.
Propose an adjustment to a constants, cutoffs, ramps, etc. that reflect this.
NOTE: your changes should never smuggle kernel identity into the heuristic, e.g., don't add a gate that specifically checks for a very narrow type of kernel as an escape hatch for a poorly performing cell. 
Just honestly report it.
Structural changes are not allowed -- the general structure of https://github.com/calebmkim/helion-heuristic-performance-artifact/blob/main/human-docs/REDUCTION_HEURISTIC_HIGH_LEVEL_TRACE.md should be the same across hardware backends.
Once you have made the change, go back to (2) -- if performance regresses then you should revert the change. 

(4) Eventually, once the constants and ramps are tuned to your local max, or you've done the best you can, then stop, and report the performance results for both vllm-reduction and examples-reduction.
It would probably be nice to report the numbers versus the state *before* https://github.com/pytorch/helion/pull/3551, i.e., at a previous iteration in the compiler where there was a b200 specialized path. Concretely, the heuristics to compare against are `triton_reduction_tile_sm100` and `triton_reduction_user_tile_sm100`, which existed before that PR collapsed the two arches into one path.
Ideally it should be similar in perf or better than that.

Two things about measurement, since you will be tuning constants against benchmark results:
- **Gate on correctness, not just latency.** You are changing block sizes and warp counts, so a config can get faster by being wrong. Every arm needs to pass a numerical check before its timing counts, and a cell where the heuristic fails accuracy is a failure, not a fast result. Report any such cell rather than dropping it.
- **Time it so the numbers mean something.** Use cold-L2 CUDA-graph timing, with the arms interleaved in one process rather than measured in separate runs. There is real per-cell measurement noise here, and it is larger than you would guess: work out what it is for your setup, and require a change to clear it before you accept the change. Otherwise you will spend the run chasing artifacts.

Guidelines:
- Have an overnight log, with this prompt copy pasted at the top. This serves as a durable log across context compaction. This prompt is your north star and you should follow it. 
- Any questions, judgements, updates, etc. should go in this overnight log.
- Please implement this in an overnight, autonomous, run.

Now, please go ahead and begin. Implement this change.

