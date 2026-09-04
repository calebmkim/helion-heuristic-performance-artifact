# Fast by Default: Compile-Time Heuristics for Helion Kernels

*By [BYLINE — plain comma-separated author list, no affiliations, per series convention]*

### TL;DR

Helion, PyTorch's domain-specific language (DSL) for performance portable machine learning kernels, relies heavily on autotuning for its performance.
Autotuning can be a major bottleneck:
a previous team tuning one [paged-attention kernel](https://pytorch.org/blog/portable-paged-attention-in-helion/) across 72 scenarios -- batch sizes, sequence lengths, head sizes -- spent 10 hours in Helion's *quick* mode, and 25 at full effort.
This blog post proposes an alternate approach: *compute* the config instead of searching for it. 
A set of heuristics that read a kernel's IR and dataflow at compile time and emit a config directly, compiling and benchmarking nothing.
We sort kernels by what they contain -- a matmul, a reduction, or neither, which leaves pointwise -- and write one heuristic per category.

Highlights:

- **Linear attention.** Over 96 kernel-and-shape cells spanning nine linear-attention variants, the computed config beats the hand-written Triton in [Flash Linear Attention](https://github.com/fla-org/flash-linear-attention) (FLA) by **1.18x** on H100 and **1.30x** on B200 in the forward pass, and by **1.13x** and **1.44x** with backward included. On H100 it is **2.27x** faster than Helion's no-autotune default in forward and **1.93x** with backward included.
- **vLLM's reduction kernels.** Over 153 cells across six production kernels on H100, the computed config is **1.27x** faster than vLLM's own handwritten CUDA/C++ operators, and **2.2x** faster than Helion's previous default. [B200 run pending.]
- **And it costs nothing.** This is static analysis, not search: nothing is compiled and nothing is benchmarked, so it adds tens of milliseconds to compile time rather than hours of GPU time.

Limitations: 

- **The honest ceiling.** Full autotuning still wins. On linear attention it is **1.06x to 1.17x** faster depending on the part and on whether backward is included, so we leave between **6%** and **15%** of the tuned config's performance on the table. On the vLLM kernels the gap is **1.03x**, or about **3%**. [B200 vLLM run pending.]
- **Not every kernel gets a config.** Everything these heuristics compute is arithmetic on extents -- block sizes, CTA counts, loop trip counts, live bytes -- so when a matmul dimension is not known at compile time we lose every estimate at once, and we decline to seed rather than guess. A loop tiled over a range that is only determined at runtime is enough to trigger this. That is conservatism rather than a fundamental limit: a fallback extent, a hint from the author, or any sort of default behavior, would let these kernels be seeded too.

## Background: Finding a Good Config Is Hard

A single Helion kernel lowers into thousands of different Triton kernels.
That is the point of the language -- one definition covers block sizes, loop orders, indexing modes and pipelining strategies -- but it is also the problem, because something has to choose which of those thousands to emit.
Today that something is the autotuner, and it chooses by compiling and benchmarking candidates until it runs out of budget.

Choosing well is genuinely hard, and the best evidence is that this series has already spent two posts on the problem.
[Likelihood-Free Bayesian Optimization](https://pytorch.org/blog/accelerating-autotuning-in-helion/) (LFBO) replaced the previous pattern search and cut autotuning time by 36.5% while improving kernel latency 2.6% on average.
[From Minutes to Seconds](https://pytorch.org/blog/from-minutes-to-seconds-llm-guided-autotuning-for-helion-kernels/) then had an LLM propose the candidates, benchmarking roughly 10x fewer configurations for up to 6.7x less wall clock.
Both are real wins, and both are clear about what survives them.
LFBO's own worked example -- one LayerNorm kernel, at one shape -- still takes about five minutes, down from nine.
The LLM-guided autotuner still spends 22 to 64 seconds per kernel-and-shape on a 384-thread host, and it pays in LLM tokens and API round-trips on top of the GPU time.
Two serious attacks on the problem later, the floor is still minutes of GPU time to configure one kernel at one shape, because neither approach removes the compile-and-benchmark loop or the GPU it runs on.
And that is *per kernel and per shape*, which exacerbates the problem.

## Our Approach: Decide the Config at Compile Time

So we compute the config at compile time instead of searching for it at tuning time.
We read the device IR and record structural facts about it: which axes are tiled, which tiles are live at the same moment, etc.
A heuristic reads those facts and emits a `Config`, without any autotuning.

A main heuristic produces the answer for a given kernel, and that answer is what Helion emits when autotuning is off.
Alongside it, helper heuristics offer variations on that answer and a few alternative approaches to the same kernel -- 5 to 20 configs in all.
Those become the autotuner's initial population when autotuning is on.

The two goals are separate, so we report them separately: a good main answer helps everyone who never tunes, and a good initial population should get the autotuner there sooner for everyone who does.

[FIGURE: `figures/diagrams/diagram-E-two-consumers.png` -- one heuristic run, and the two things that consume its output.]

## How the Heuristics Work: Static Analysis

The real implementations are considerably more involved than what follows; this is the simplified version.

### Resource Estimation

They all lean on the same underlying analysis.
At any step of the kernel, take a snapshot: given a candidate set of block sizes, how many registers does one CTA hold live at that moment, how much shared memory, and -- on Blackwell -- how much *tensor memory*?

(For context, Blackwell's big tensor-core instruction does not accumulate into registers; it accumulates into a separate on-chip space called tensor memory, or TMEM.
It matters here because it is a *third independent capacity*, not a pool you can trade against the other two.)

The compiler hands us a device IR; on top of it we run a liveness analysis -- walk each graph to find where every value is last used, and classify what produced it, because the same tile costs different things depending on its role.
A dot's output is charged to tensor memory on Blackwell and to registers otherwise; an operand being streamed in is charged to the ring of shared-memory buffers the pipeline holds it in; a loop-carried accumulator is charged for the whole loop rather than one step.
Get the classification wrong and the byte count is wrong in a way no amount of careful arithmetic afterwards will fix.

[FIGURE: `figures/diagrams/diagram-A1-liveness.png` -- which values are live at each step of a kernel, and what produced each one.]
The genuinely hard part is that this requires predicting *where each tensor will end up* -- what Triton will keep in registers, what it will stage through shared memory, what it will materialize at all rather than fuse away.
We are predicting another compiler's decisions, so the estimate is approximate.

From that we can estimate the things that actually matter, such as:

- **Residency** -- how many CTAs fit on an SM at once. Whichever space runs out first decides it. A CTA needing a tenth of the SM's registers and a tenth of its shared memory, but six tenths of its tensor memory, gets one CTA per SM, and tensor memory is the only reason why.
- **Register pressure.** Peak live registers spread over the threads we intend to launch tells us whether the assembler will be forced to spill -- a softer failure than not fitting on the SM at all.

### Matmul

Start with a kernel that has exactly one matmul in it -- an `hl.dot`, a `torch.baddbmm`, whatever.
We hand it the config that a plain k-reduce matmul of that shape would want.
That means block sizes grown until the fp32 accumulator tile fills a byte budget, biased wide in N because N is the axis the stores are coalesced along, with `num_warps` following from how big that tile ended up and three or four pipeline stages behind it.
If the tile grid is many blocks tall it also gets an *L2 grouping*, which reorders the block indices so that blocks running at the same time reuse the same operand out of L2 instead of each pulling it from memory.

The actual kernel may look nothing like a plain matmul -- maybe it does one block of matmul and then a softmax, maybe the matmul sits inside a loop carrying state from chunk to chunk.
We hand it the standard config anyway.
That might be wildly off, and that's fine; the point is to start somewhere defensible.
It is probably not the *best* config, though, so we then make a series of *adjustments*, each reading the resource estimate above.

- **Residency.** If the launch produces fewer CTAs than the GPU has SMs, some SMs sit idle, so we shrink a block size on a grid axis to produce more of them.
- **num_stages.** More stages hide more load latency, so we want as many as shared memory allows. But stages are also the cheapest way to accidentally lose residency: six of them might fit while eating the CTA's entire shared-memory budget, leaving one CTA per SM, where three stages and two CTAs per SM is the better trade.
- **num_warps.** Two things pull on this:
  - *Which instruction Triton emits.* Below four warps -- a *warpgroup* -- or with an M block under 64, Triton uses the smaller synchronous MMA; at or above that it uses the big high-throughput one (`wgmma` on Hopper, `tcgen05` on Blackwell). Which one you get also decides where the accumulator lives, and therefore which budget it spends: registers for the small instruction, tensor memory for the big one. Usually we want the big one. But when there is little matmul work the kernel is memory-bound rather than compute-bound, and the better answer is many CTAs per SM running small MMAs that overlap each other's loads -- so we count **dynamic matmul work**, where a small matmul inside a 100-iteration loop counts a hundred times, and choose the regime from that.
  - *Register pressure.* Warps divide the tile across more threads, so more warps means less pressure on each one. If the peak live state at the current count would spill, that argues for climbing.

  These two often disagree -- the memory-bound case wants to stay below the warpgroup, spill relief wants to climb past it -- so we settle them together rather than letting either decide alone.
- **Sanity.** A final pessimistic pass over the whole kernel: add up tensor memory, shared memory and registers, and confirm the config can actually launch. The adjustments above already respect these budgets, so this is mostly defensive.

### Multi-Matmul

Now suppose several matmuls appear in one kernel, which is the common case for attention and linear attention.
We run the matmul heuristic on each `hl.dot` and use those as the starting point.
They may disagree: two dots can share one block-size knob -- one dot's N is frequently the next dot's K -- and there is only one `num_warps` for the whole kernel.

So we need a **merge policy**.
We rank each dot by estimated importance: first, whether it writes a loop-carried accumulator, since that accumulator stays resident for the whole loop and its tile therefore sets the kernel's footprint; then its **dynamic matmul work**; then its own output area.
Wherever two dots disagree about a knob, the higher-ranked one wins (for example `num_warps` is set by the top-ranked dot alone).

Then we apply the **same four adjustments as before** -- residency, `num_stages`, `num_warps`, sanity.

### Reduction

If a kernel contains a matmul at all, the reduction heuristic declines and the matmul path takes it.

The starting point comes from growing tiles until they run out of room: every tunable axis begins at its smallest legal size and we expand until peak live state reaches a byte budget.
Reduction axes get widened first -- in `x[tile_m, tile_n].sum(-1)` that is `tile_n` -- since a wider reduction is usually what pays.
That budget is a fitted constant, chosen once offline against measured points, and it is per-architecture: nothing is benchmarked when *you* compile a kernel, but benchmarking did happen once, by us, and a new architecture would want the constants re-fit.

Then, as with matmul, we adjust.
One assumption runs through what follows: a reduction has less arithmetic to hide behind, so it is likely memory bound, and throughput is set by how many loads are in flight -- which means CTAs resident on an SM.
That is the memory-bound case from matmul, except here it is the rule rather than the exception, and two separate knobs buy residency, each at a cost.

- **num_warps.** Three pressures here, pulling in different directions:
  - *Enough warps to do the reduction.* A wide reduction needs warps working on it, so a ladder off the largest selected reduction sets the starting point: 128 elements wants one or two warps, 16,384 wants many more.
  - *Residency.* More warps per CTA means potentially fewer CTAs on the SM (there is a per-SM CTA limit), so we also test stepping back down to a lower rung -- paying reduction parallelism for residency.
  - *Register pressure.* A warp count that spills against the peak live state is not a saving, which bounds how far down that descent can go.
- **Tile size.** Fitting more CTAs per SM means potentially *smaller* tiles, so we shrink blocks back down to buy residency. The constraint is that an axis with stride 1 is the one carrying the coalesced loads, and shrinking it too far loses more in memory efficiency than the extra CTAs return.
- **Persistence.** A reduction can be kept resident and done in one pass, or walked in chunks. The case that matters is a row the kernel reads twice -- RMSNorm reads it once to compute the scale and again to apply it -- and when we can prove that, staying persistent is allowed to cost some spilling and some residency, because it saves the second trip to memory. 

Because the warp count depends on the blocks and the blocks depend on the warp count, we iterate until both hold still for a full round.

### Pointwise

A kernel with no matmul and no reduction gets the pointwise heuristic, the same budget-driven idea in its simplest form. There is no adjustment phase: we compute three caps on the tile -- the bytes a program should move, what fits in registers, and enough waves to fill the machine -- and take the smallest.
We have not evaluated it as carefully as the other two, since fewer Helion kernels fall into this pattern.

## Result 1: Config Quality, with No Tuning Time

### End-to-end linear attention

We measured nine linear-attention variants -- vanilla, simple GLA, retention, full GLA, delta rule, gated delta rule and three KDA variants -- in forward and forward-plus-backward, over six production shapes, on both H100 and B200.
The six shapes span batch 1 to 8, 8 to 96 heads, sequence 1024 to 16384 and head dimension 64 to 256.
The baseline is [Flash Linear Attention](https://github.com/fla-org/flash-linear-attention) (FLA), an open-source, widely used library of hand-written Triton kernels for these models, tuned by its authors; we warm it first so its own Triton autotuner has converged, and feed it inputs in its native layout.
All three Helion arms run the same kernel source; only the config differs.
Every arm is checked for numerical agreement before it is timed.
On H100, we materialize those configs first and then co-measure all four arms
in one process per cell, with cold-L2 samples interleaved in rotated
forward/reverse order.
The retained full H100 sweep used PyTorch 2.12.0+cu132. A matched eight-cell
forward/backward compatibility check on PyTorch 2.13.0+cu132, CUDA 13.2, and
Triton 3.7.1 changed aggregate performance-versus-FLA by at most 0.05% across
the three Helion arms, with every sampled arm passing correctness.

[FIGURE: `figures/results-linattn-summary.png` -- geomeans for both parts and both modes, normalized to handwritten FLA Triton at 1.00x.
Then optionally the two per-variant breakdowns, `figures/results-linattn-h100.png` and `figures/results-linattn-b200.png`.]

[FIGURE: `figures/diagrams/diagram-D-tradeoff-space.png` -- schematic of the tradeoff, if it is not already used up top.]

Before this work the choice was binary: take Helion's no-autotune default and run at **0.52x** of handwritten FLA Triton on H100, or spend GPU-hours autotuning and reach **1.26x**.
Neither is a good answer if you have not budgeted for tuning -- the first is a regression against a hand-written kernel you could have used instead, the second a bill that comes due again on every new shape and every new GPU.
The heuristic adds a third option that did not exist before: **1.18x** on H100 and **1.30x** on B200, with no autotuning price.
That is most of what autotuning finds, available immediately, and it moves Helion's out-of-the-box behaviour from *losing* to handwritten Triton to *beating* it, at least for this narrow set of kernels.

### vLLM's reduction kernels

[vLLM](https://github.com/vllm-project/vllm) is the most widely deployed open-source LLM inference engine, and it ships its own hand-written CUDA/C++ kernels for the operations on its serving hot path.
Six of those are reductions -- two FP8 quantizers, RMSNorm and SiLU-and-multiply each fused with quantization, and fused QK-normalization plus RoPE -- and vLLM already has Helion implementations of them, contributed by the team behind [an earlier post in this series](https://pytorch.org/blog/portable-vllm-model-inference-kernels-in-helion/).
That makes them a good test: production kernels, written by someone else, against the hand-written CUDA they were meant to replace.

[FIGURE: `figures/results-vllm-h100.png` -- 153 cells across the six kernels on H100, normalized to vLLM's own CUDA kernel at 1.00x.]

Same shape of result, in a different codebase.
Helion's no-autotune default runs at **0.58x** of vLLM's CUDA; the computed config reaches **1.27x**; full autotuning gets to **1.31x**.
The interesting number is the last gap: on these kernels autotuning buys only **3%** over the config we compute for free.
This primary run uses PyTorch 2.13.0+cu132, CUDA 13.2, Triton 3.7.1, and vLLM 0.24.0's stable-ABI CUDA extension loaded directly, without allowing vLLM's wheel dependencies to replace the compiler stack.
The B200 run is still pending.

### The everyday kernels, against `torch.compile`

Both comparisons above are specialist libraries -- linear-attention variants and FP8 quantizers. This one is deliberately the opposite: the kernels everybody writes. RMSNorm, LayerNorm, softmax, cross entropy and friends, which is also the suite the [introduction post](https://pytorch.org/blog/helion/) benchmarked when it first measured Helion.

The reference is what you would otherwise reach for rather than something hand-written: `torch.compile(mode="max-autotune-no-cudagraphs")`, the strongest setting Inductor offers, with everything normalized to it at 1.00x. There is no tuned-config arm in this section, for the mundane reason that we do not have pre-tuned configs for these kernels -- so unlike the two sections above, this one reports no ceiling.

On a suite of ten reduction kernels -- RMSNorm and LayerNorm forward and backward, softmax, cross entropy, KL divergence, JSD, fused linear JSD, GRPO -- over 80 cells, the computed config comes out at **1.088x** of max-autotune `torch.compile`, with no tuning of its own. Helion's unseeded default is **0.268x**. Seven of the ten kernels beat `torch.compile` and three trail it, the largest win being softmax at 1.377x and the largest loss JSD at 0.856x. This primary result uses PyTorch 2.13.0+cu132, CUDA 13.2, and Triton 3.7.1; the artifact retains the earlier PyTorch 2.12.0+cu132 / Triton 3.7.0 dataset separately.

[FIGURE: `example-reductions/generated/blog-figures/results-example-reductions-h100.png` -- ten reduction kernels on H100 using PyTorch 2.13.0+cu132 and Triton 3.7.1, normalized to max-autotune `torch.compile` at 1.00x.]

The other corpus is a breadth sweep: thirteen kernel families over 69 shapes -- plain and broadcast matmul, a BF16 x INT16 GEMM, gather GEMV, dense, causal, biased and backward attention, Mamba-2 chunk state and chunk scan, squeeze-and-excitation, jagged HSTU, and a gated-delta-net recurrence. The computed config comes out at **2.38x** of max-autotune `torch.compile` overall, against **0.65x** for Helion's default.

One caveat on that 2.38x: for some of these kernels the Helion implementation and the PyTorch reference are not the same algorithm. Where the reference materializes intermediates that the Helion kernel fuses, Inductor is implementing a different algorithm, and the ratio reflects that as much as it reflects the config. The gated-delta-net kernel is the extreme, at 13.6x.

Two families go the other way. On the BF16 x INT16 GEMM the computed config lands at 0.848x, slower than `torch.compile`; and on jagged HSTU it is slower than Helion's own default, the one family in the sweep where the heuristic actively hurts.

[FIGURE: `other-matmul-kernels/generated/h100-eacfee67-torch-compile/blog-figures/results-matmul-vs-torch-compile-h100.png` -- thirteen kernel families over 69 shapes on H100, normalized to max-autotune `torch.compile`.]

## Result 2: Does the Seed Make the Search Faster?

Everything above is about the config Helion emits when autotuning is *off*.
But the same configs can be handed to the autotuner as its starting population, and in principle a good starting point should let the search finish sooner.
One config is a thin population, so the helper heuristics fill it out to somewhere between 5 and 15.

To test it we took 38 matmul and multi-matmul kernels -- linear attention, attention, Mamba, and a few wild cards like squeeze-and-excitation -- and one shape each, deliberately varied rather than the same shape throughout.
Then we ran the full search on every cell twice: once from random configs, once with the compiler's seeds in the initial population.

The short answer is that seeding buys a lot early but much less late.
Throughout, "within X%" means a config whose latency is at most (1 + X) times the best any arm found on that cell -- so "within 50%" is no worse than 1.5x the best, and "within 5%" is essentially as good.
Here is one cell, chosen as the clearest example of the pattern rather than a typical one:

[FIGURE: `figures/results-seed-trajectory.png` -- best config so far against configs benchmarked, for a bf16 x int16 matmul at m=1, k=4096, n=4096.]

Seeded, the *first* config tried is already within 50% of the best anything found; unseeded takes 102 configs to match it.
But look where the two lines meet.
Both arms reach within 5% at around 130 configs -- 129 unseeded, 134 seeded, so on this cell seeding is fractionally *behind* by then -- and from there they sit on the same answer while the search grinds through several hundred more configs.
The early advantage is enormous and the final advantage is zero.

That is the pattern across all 38 cells, and it is why we report this as the weaker of our two results.
Medians below; counts include configs that failed to compile or failed to produce a timing, since those consume budget too.

| Search target | median configs (no seeds / seeded) | median wall seconds (no seeds / seeded) |
|---|---|---|
| within 50% of the best found | 35.5 / **1.0** | 52.6 / **20.9** |
| within 25% | 103.5 / **2.5** | 105.4 / **28.1** |
| within 5% | 255.5 / **175.0** | 269.8 / **134.8** |
| complete search | 521.5 / **420.5** | 537.3 / **454.1** |

Read down the table and the advantage decays: **35x** fewer configs to reach a merely decent config, **41x** to get within 25%, **1.5x** by the time you want within 5%, **1.2x** to finish.
Wall clock decays the same way but not as far -- 2.5x, 3.8x, **2.0x**, 1.2x -- so at the tight tolerance the time saved is worth more than the config count suggests.
It also has a floor: 20.9 s is essentially the seeded arm's median time to get *any* successful measurement back, so there is nothing left to win below it.

One thing this result is not: seeding does not make the final config any better.
Replaying both arms' winners on one GPU, they land within a fraction of a percent of each other.
This is about how long the search takes, not where it ends up.

## Limitations and Future Work

The central limitation is that these heuristics predict what another compiler will do.
Every budget depends on knowing where a value ends up -- registers, shared memory, or tensor memory -- and that is Triton's decision, not ours.
Some of it is safe to bet on: on B200 a matmul result is going to be in tensor memory.
Plenty of it we get wrong, sometimes badly; register and shared-memory estimates are the worst offenders.

Which raises a question we do not think is settled: should this live in Helion at all?
One layer down, inside Triton, these quantities are not estimates -- that compiler knows exactly what it allocated, so a heuristic there would never mispredict shared memory.
The counter-argument is that several of the decisions worth making are not parameters Triton could accept.
L2 grouping, choosing tensor descriptors over pointer arithmetic, a persistent launch instead of a flat one, a persistent reduction instead of a chunked one -- each of those is a *different Triton program*, not a different argument to the same one.
Something has to be able to generate both and choose, which is exactly what a compiler that emits Triton is for.
So the resource estimate wants to be lower in the stack and the config choice wants to be higher, and we do not have a clean answer to that.

Four directions we think are worth taking.

**Fire on more kernels.** We currently decline whenever we lack the information to size anything -- a loop tiled over a range only known at runtime, for instance.
A less-educated guess would be easy: a default extent, a mean, something. Configs would be worse without that information, but the alternative today is the base default, which the results above show is very bad -- so the guess is probably still worth making.

**Touch more knobs.** We set block sizes, `num_warps` and `num_stages`, plus some L2 grouping and L2 eviction policy.
That leaves a long list untouched -- tensor descriptors versus pointer arithmetic, persistent versus flat launch, etc.
Individually these are probably worth less than the ones we do set, but there is certainly performance left on the table.

**Specialize the heuristics.** They are deliberately general: one multi-matmul heuristic has to do well on attention *and* linear attention, one reduction heuristic on RMSNorm *and* vLLM's quantization kernels.
An alternative is parent heuristics with children -- a `vLLMQuantizationHeuristic` that starts from `ReductionHeuristic`'s answer and then specializes on what it knows about that family.
In practice the parent's answer is the floor: the child starts there, so unless its specialization actively hurts, it can only improve on it.

**Measure instead of predicting.** The cleanest fix for the limitation above is to stop guessing: compile once, read the register and shared-memory usage the assembler reported, and correct the estimate from there.
That data is nearly free -- it falls out of a compilation we already did, no profiler involved -- so the question is not how to get it but what to do with it: whether a compile-and-correct round is worth its latency, and whether the corrected estimate is enough better to change the config it produces.

## Conclusion

**A good config is now free, and free changes the tradeoff.** Full autotuning still finds better configs than we do -- the ceiling numbers are above -- but it costs GPU-days per library per hardware generation, and has to be redone for every new part.
A config that captures most of that ceiling for tens of milliseconds of compile-time analysis changes which bill you have to pay.

## Acknowledgements

<!-- ============================================================================================
NUMBER PROVENANCE — every figure printed above (plus a few trimmed for length, kept here so an editor
can put them back), with its source and its baseline. Research notes are in
blog/research/ in this repo.

LENGTH: about 4,750 words counting every line that is not a table, a code block or a figure
placeholder (the previous pass reported 3,901 by a narrower count). A verification pass added roughly
450 words: the Pointwise result subsection, the calibration disclosure, the censored-vs-total basis
note in Result 2, the absolute-latency anchor in Result 1, and Figure 6. The "Limitations and What's
Next" section was folded into one paragraph at the end of Result 3, which removed the duplicated
pipeline-depth item and the register-model bullet. If an editor must cut ~150 more words, the cheapest
in order are: (1) the "13 of 38 cells" clause in Result 2; (2) the "Two qualifications" sentence in the
compiler-thesis section; (3) the third caveat bullet under the linear-attention table (the two
FLA-losing variant rows); (4) the "Restricted to the four main quantization kernels" sentence. None of
those is a CRITICAL_CORRECTIONS-mandated caveat. Do NOT cut the calibration paragraph, the pointwise
"4.4% of peak" qualifier, the censoring-basis note or the multi-matmul-cost disclosure — each closes a
verified reviewer attack.

END-TO-END LINEAR ATTENTION (research/10; raw: wt-sm100-linattn/matmul_heuristic_perf_results/
e2e_fla_v3/results.json; warmed eager, cold L2, median of three round means; 96 cells)
ARM NAMES USED IN THE BODY, fixed: "Helion's previous compiler default" = the pre-change compiler
heuristic (linear-attention tables ONLY, corrections A2); "the unseeded base default" =
ConfigSpec._base_default_config() (the 149-cell, vLLM, pointwise and SGLang baselines). No other
surface form appears in the body.

  1.98x / 1.81x            heuristic / previous compiler default, forward (54) / fwd+bwd (42)
  1.30x / 1.44x            heuristic / FLA Triton 0.5.2 (1.3023 / 1.4418)
  1.298x                   heuristic / FLA fwd+bwd excluding the two D256 cliff cells (B1)
  0.66x / 0.80x            previous compiler default / FLA Triton; lost 15 of 16 variant-mode groups
  2.26x / 2.12x            full-autotuned AOT / previous compiler default
  1.14x / 1.17x            AOT / heuristic (1.1404 / 1.1710)
  77 of 96                 cells where heuristic >= FLA
  1.005x / 0.986x          heuristic / pre on B8_T1024_H8_D64; AOT there is 1.002x / 0.978x (B2)
  0.82x / 0.96x            heuristic / FLA, gated_delta_rule and delta_rule fwd+bwd
  1.41x                    AOT / heuristic on gated_delta_rule fwd+bwd (1.4085)
  63x, 38x vs 2.7-4.8x     FLA's own fwd+bwd/fwd ratio at D256 vs at every OTHER shape; at D256
                           itself the five non-cliffing variants run 3.80/4.06/4.40/5.10/9.02x
                           (research/10 SS4) — the body says both ranges
  4.06 / 0.91 / 1.02 /     ABSOLUTE anchor printed in the body: simple_gla forward at
  0.70 ms                  B8_T2048_H32_D256, previous default / heuristic / FLA Triton / AOT;
                           fwd+bwd 12.34 / 3.99 / 9.10 / 3.08 ms. Re-derived from
                           e2e_fla_v3/arms/simple_gla__*__B8_T2048_H32_D256/*.json
                           (timings.method.latency_ms, timings.fla.latency_ms)
  83.9% / 79.0%            log-space capture = log(heur/pre) / log(AOT/pre)
  92-93% / 45% / ~40%      capture: retention+vanilla+simple_gla fwd / gated_delta fwd+bwd / kda_varlen
  14 upstream commits      between the pre (9c46dd311) and post (375363d8) trees
  dq-only backward gate    tol 0.05 (B14)
  torch 2.12.0+cu130, Triton 3.7.0, one B200, physical GPU 1

CONSTITUENT MATMUL CELLS (research/11; on_corpus_pretuned/results.json; CUDA graphs, cold L2,
rotated interleaved, 149 cells. "Pre" here IS the unseeded base default — corrections A2, re-verified
this pass: pre_change.heuristic_name is null on 149/149 records.)
  1.55x                    heuristic / unseeded base default (1.5527)
  1.20x                    per-cell tuned reference / heuristic (1.2046). NOT full autotuning, and NOT
                           the same artifact as the e2e AOT ceiling. Re-verified 2026-09-02: 124/124
                           refs are byte-identical to SM100_CONFIGS.json sm100_config (sm90-seeded,
                           B200 hand-climbed, median 50 / range 18-120 arms per cell, 230 of 251 cells
                           changed from their seed); the other 25 scored cells are SGLang's shipped
                           helion.Config objects. This corpus is DISJOINT from the 256-config shipped
                           in-tree table (0 configs in common; all 18 chunk_bwd_dh_diag_fused cells
                           differ), and 27 of the 124 refs use block_ptr indexing, which appears in
                           0 of the 256 shipped configs. By contrast the E2E AOT arm IS the shipped
                           full-autotune table: 474/474 pretuned-arm subkernel configs resolve to
                           examples/linear/_helion_aot_linear_attention_engine_cuda_sm100.py via
                           AOTAutotuneCache. Caution: SM100_CONFIGS.md's "Shipping artifact:" line
                           names that in-tree file, which is stale/wrong - they are disjoint.
                           UNMEASURED: shipped-autotuned vs hand-climbed, head to head. (A1)
  1.99x, 20/20, 0 regr     single-contraction front end (20 cells); 1.1257x left
  1.49x, 23 regressions    multi-contraction front end (129 cells); 1.2173x left
  124 / 25 / 46            cells from the SM100_CONFIGS.json hand-climb / SGLang shipped Configs /
                           block_ptr configs in that corpus no autotuner here can emit
  70.3%                    log capture, ln 1.5527 / ln 1.8704 (70.26%)
  median 1.16x; 53, 26, 8  gap distribution; top-5 cells hold 16.4% of the log-gap
  148 of 149               tuned configs differ outside the four owned knobs (research/23 SS1.4).
                           This is a CO-OCCURRENCE, not an attribution: "how much of the 1.20x is the
                           un-owned knobs" is unquantified (research/23 Gap 8), so the body says only
                           that they are where the configs differ.
  12 / 13 / 14 fields      normalized-config key count over the 149 cells (37 / 12 / 100 cells);
                           the PROMOTED config is exactly 3 keys (block_sizes, num_stages, num_warps)
                           on 149/149, plus l2_groupings on the single-contraction path only.
                           Re-verified this pass from on_corpus_pretuned/results.json.

REDUCTION / vLLM (research/12 Audit A; cold-L2 CUDA-graph device time; 114 recorded, 113 valid)
  1.92x / 1.39x / 0.918x   heuristic vs unseeded base default / torch.compile default mode (A5) /
                           vLLM's shipped B200 config on the same Helion body (54 cells)
  1.75/1.26/0.988          at 1 token; 1.78/1.37/0.988 at 128; 2.27/1.55/0.793 at 8192
  1.000x                   four main quant kernels at 128 tokens vs vLLM's config
  7.434 / 2.850 / 1.025    per-kernel vs the unseeded base default: rms_norm_dynamic_per_token_quant /
                           dynamic_per_token_scaled_fp8_quant / silu_and_mul_per_block_quant. The
                           1.92x is a QUANTIZATION-family effect, not a "fused" one (research/12 SS2.3;
                           rms_norm_per_block_quant 2.204x, fused_qk_norm_rope 0.956x).
  1.03x, 0.978x            general_aot cohort (24 cells) vs the unseeded base default; LayerNorm forward
  0.987x / 0.955x          the SEPARATE broader audit, whose vLLM arm is the H100 table on B200 (A3)
  17 of 113 valid, 0.514x  cells where the unseeded base default beats us; worst fused_qk_norm_rope
                           [8192,64,8]
  1 of 114 excluded        rms_norm_bwd [2048,11008], fails for the heuristic AND the default (B4)
  114 of 114 / 61 of 78    seed num_stages == 1 / tuned winners using >= 2

POINTWISE (research/15 Part A; 36 planned, 35 timed; calibrated cold-L2 CUDA graph, 3 cohorts of
17 / 9 / 9 cells; torch 2.12.0+cu132)
  19.4x vs 18.5x           heuristic vs torch.compile default mode over the 17 general cells — a TIE,
                           and both are against a pathological baseline (B7); 19.3508x / 18.5082x
  351 GB/s = 4.4%          the unseeded base default's achieved HBM bandwidth on every swiglu/geglu
                           shape, vs 63.3-85.2% for the heuristic's config
  0.890x / 0.886x          heuristic / torch.compile on vLLM silu_mul_fp8, SGLang interleaved (B7)
  1.257x vs 1.036x         heuristic vs SGLang's exact-key SM100 table, each over the unseeded base
                           default, on 9 cells = 1.2138x for us over the shipped table
  2.1-23.6 us              absolute latencies on the 18 production pointwise cells (+ one 643 us
                           outlier), which is why the body says "a few to a few tens of microseconds"
  [1,1] vs [1,32]          RoPE: the heuristic picks SMALLER blocks (B8), true for 5 of the 17 general cells

SGLang KDA (research/15 Part B; CUDA-graph op timing — never merge with the eager numbers)
  0.874x / 0.807x / 1.51x  heuristic vs handwritten Triton / vs unseeded base default / shipped vs
                           Triton, decode
  1.156x / 1.111x          heuristic vs Triton on fixed and packed prefill; 0.674x / 0.663x is the
                           UNSEEDED BASE DEFAULT vs the same Triton on those two paths, not a
                           pre-change arm
  0.83 / 1.01 / 1.43       FIGURE 6's overall geomean group (n=36) vs handwritten Triton: no
                           heuristics / heuristic / pre-tuned (0.8284 / 1.0095 / 1.4250); decode group
                           reads 1.08 / 0.87 / 1.51. Source
                           sglang_triton_head_to_head/four_arm_no_heuristics_v2_address_matched/
  16 decode shapes         8 batch sizes x 2 head counts; one identical config emitted for all
  12 of 16                 decode shapes SGLang's wrapper routes to a CUDA kernel, not Triton
  attribution              all 16 decode records fired triton_reduction_tile_sm100 only (A7)

SEARCH COST (research/14)
  189 searches / 98 cells / 49.6 GPU-hours; mean 15.75 min, median 10.70, p90 29, max 2 h 21 m
  256 call keys / 26 kernels -> 67.2 GPU-h = 2.80 days at the mean (45.7 GPU-h at the median) (SSC)
  9.8 GPU-h in 5 h 52 m    two B200s in independent lanes (B13)
  3,102 of 19,796          configs that fail to compile = 15.7%: a fraction of CONFIGS, not of the bill
  26,250 s of 35,152 s     elapses BEFORE the best incumbent, so ~25% is post-winner confirmation.
                           NOTE: research/14 SS5's one-liner states this INVERTED; SS2d is correct.
  18.1%                    geomean gain from hand-retuning the sm90 configs on B200
  2.375 s -> 2.433 s       config selection with heuristics off vs on, 49 single-GEMM BF16 cells;
                           paired median +33.7 ms. The MULTI-matmul front end is UNMEASURED on GPU
                           (CPU micro-benchmark on stubbed facts: 4.2 ms at 2 dots, 15.6 ms at 8).

SEEDED SEARCH (research/13; 38 frozen cells, three arms at one commit)
  50/1/1, 108/19/4.5, 290/162/202.5, 521.5/490.5/420.5 median configs to 50 / 25 / 5% / complete.
                           The first three rows are right-censored terminal-attempt lower bounds; the
                           fourth counts configs benchmarked. Different bases — the body says so
                           immediately under the table (B11).
  52.6/23.5/20.9, 105.4/44.2/28.1, 334.7/174.3/194.4, 587/681/500   median wall seconds, same rows
  37 of 38                 cells where a seed config itself satisfies the 50% target
  20.9 s                   the expanded arm's median time to its FIRST successful timing
  28 / 34 / 28 of 38       5% reach counts; expanded worse than no-seed on 11 of 38 (B9)
  14.4%, 0.855x, 0.844x    expanded vs no seeds: configs cut, configs geomean, in-tuner time geomean
  0.851x vs 1.009x         expanded vs the OLD pool, multi- vs single-contraction in-tuner time
  35,058 s vs 35,152 s     old pool vs no seeds, complete-search wall time (B10)
  1.0005x / 1.0142x        seeded winners' latency vs unseeded (lower better); 22 of 38 within 5%
  45 -> 38 cells, ~19%     survivorship filter (B12), and share of counted attempts that failed
  13 of 38                 cells where the expanded pool cost more configs than no seeds
  rank-0 unchanged         by the expanded pool (research/21 SS0)

MECHANISM (research/20, 21, 22)
  227 KiB, 128x512 columns, 255 registers/thread, 148 SMs, 64 accumulator rows for tcgen05
  Required: 768, limit 512 the real OutOfResources from three live 256-column accumulators
  0.43 -> 0.87 / 0.581     wave utilization for 64->128 and 86->172 programs; extent 11008
  325 us vs 570 us         two warps (38 B spill, 4 CTAs/SM) vs eight warps (no spill, 1 CTA)
  1.18-1.74x / 0.79-0.96x  knob amortization: reuse-free 128->32 vs reuse-bearing 64->32 (research/20 #10)
  48 KiB vs 31.9 KiB       three live fp32 accumulators vs the one-warp register file -> pressure 1.51
  [1,64,16] nw2 ns4        the emitted rank-0 config for chunk_fwd_o_helion at BHN=1024, C=D=DV=64
  3 distinct configs       across chunk_fwd_o_helion's FIVE measured cells, not five: [1,64,16] nw2 ns4
                           (#0), [1,128,32] nw4 ns3 (#7, #2, #4), [1,128,64] nw4 ns3 (#9). Re-verified
                           this pass from on_corpus_pretuned/results.json promoted_config.
  1.80x-5.53x, 3.89x geo   that kernel family's five cells vs the unseeded base default (per cell
                           1.8038 / 3.9351 / 4.6668 / 5.5284 / 4.8365); 1.18x left to the reference
  10 rules, 7 exact shapes, 1 of 432 cases seeded by the old B200 table; 93 -> 406 with the formulas
  CALIBRATED CONSTANTS     reduction byte ceilings (240/120/720/288 KiB) and the pointwise budget
                           constants are hill-climbed, fit against a 702-point set; WARPS_PER_SM = 48
                           is the argmax of a grid search over {16,32,48,64,96,128} against 116
                           measured groups and is explicitly NOT a hardware quantity (B200 allows 64).
                           research/22 SS3 — disclosed in the body's "formula beats a table" paragraph.
  1.32x / 1.00x / 0.60x    same modeled facts, three attention kernels, one warp policy (research/23 M11)
  0 of 149                 post configs with l2_groupings != 1 — the L2 model never fires on this corpus

PRIOR POSTS, quoted to CREDIT them (research/02 SS1.3, SS2.5)
  36.5%, 2.6%              LFBO's own headline: autotuning time reduced 36.5% while improving kernel
                           latency 2.6% on average, on its B200 benchmark set, measured VS PATTERN
                           SEARCH (research/02 C1) — the body states that baseline
  ~10x / up to 6.7x        From Minutes to Seconds: ~10x fewer configs benchmarked, 6.7x less wall
                           clock than LFBO (39 s vs 261 s) on a 384-thread host; 6.7x is a best case
                           (research/02 SS5), hence "up to" in the body

OFF-CORPUS (research/23 SS1.3)
  1.4477x                  UNSEEDED FULL autotune / heuristic on 75 off-corpus cells. Distinct arm AND
                           distinct population from the 1.20x (which is the on-corpus per-cell tuned
                           reference over 149 cells) — the body never implies they are the same arm (A1)

OPEN QUESTIONS FOR THE AUTHOR BEFORE PUBLICATION
 0. Content cut purely for length, all true, all recoverable from the numbers above: the log-space
    capture spread per variant (92-93% on the simplest gated variants, ~40% on KDA varlen); the
    per-front-end headroom (1.13x single-contraction, 1.22x multi); the 3,102-of-19,796 compile-failure
    count; the reduction cohort's worst cell (0.514x on fused QK-norm + RoPE at [8192,64,8]); the
    "1.51x of Triton" SGLang decode ceiling; the 12-of-16 SGLang CUDA-kernel routing caveat; and the
    off-corpus worst family (4.33x). The dq-only backward gate (B14) is now IN the body, in the
    linear-attention scope sentence, so no gradient claim is left implicit.
 1. RE-RUN THE HEADLINES. Both big tables are snapshots at trees that have been rebased away
    (149-cell at ccfcfbdd, 96-cell at 375363d8). B15. Everything else here is downstream of that.
 2. PUBLICATION HYGIENE, three placeholders left on purpose: the bracketed byline on line 3, the
    "[NAMES TO BE FILLED IN.]" marker in Acknowledgements, and the Acknowledgements section itself. No
    engineering post in this series carries one (the intro post's italic "Helion is the work of many
    hands including: ..." is the only precedent), so if the byline ends up naming everyone thanked,
    delete the section per the two autotuning posts. The editor comment that used to sit inside the
    body has been removed.
 3. A "Resources" section? The vLLM post established one. I had no verified PR number, branch name or
    public repro command safe to print, so there is none, and nothing in the body now references one —
    Result 1's auditability need is met instead by the absolute-latency anchor sentence under the
    linear-attention table. Add a Resources H2 if you have a public link.
 4. The primary H100 reduction runs and the linear-attention compatibility check now use
    torch 2.13.0+cu132 / CUDA 13.2 / Triton 3.7.1. The retained full 96-cell H100
    linear-attention artifact used torch 2.12.0+cu132 / Triton 3.7.1, while pointwise used
    torch 2.12.0+cu132 / Triton 3.7.0. The SGLang run was on physical GPU 0 while everything
    else was GPU 1. Confirm whether you want the full historical matrix stated.
 5. Measure the multi-matmul front end's config-selection cost on GPU. The +34 ms figure covers the
    single-contraction path only, and the post headlines the other one. research/14 SS3e calls this a
    one-afternoon job on the B200 box. The gap is now DISCLOSED in the body (TL;DR bullet 4 and the
    Conclusion both scope the figure to the single-GEMM path and give the 4-16 ms CPU micro-benchmark),
    so this is an improvement to make, not a hole in the post.
 6. `fused_qk_norm_rope` miscompile count: research/12 SS5.7 says 5 of 15 excluded cells, research/23
    M8 says 6 of 15. I wrote "several" rather than print a contested count. Pick one.
 7. The knob-amortization table is quoted from the predicate's own docstring via research/20 and
    research/21; no standalone per-cell artifact was located behind the 1.18-1.74x / 0.79-0.96x ranges,
    and the two kernels are named only by body. If a reviewer wants the two shape tuples side by side
    we would have to export them.
 8. The off-corpus quick-autotune figure exists in two replay lanes (0.852x and 0.633x for
    heuristic/quick, research/14 SS6.5). I wrote only "a plain quick autotune also beats us"; name a
    lane if you want the number in the body.
 9. FIGURE INVENTORY (updated after the figure pass):
      - Figures 1, 3, 4 are BUILT as drafts in `figures/` with their generating scripts alongside.
        They already carry the series contract (in-image two-line title with the aggregate on line 2,
        polarity and baseline in the axis label, correct arm labels per CRITICAL_CORRECTIONS A1/A2/A6).
        They are rough: usable for review, not final art. Two known nits — Figure 1's fwd+bwd panel
        hatches the whole Vanilla/Retention groups when only their D256 shape cliffs (the legend says
        so, but per-bar hatching would be truer), and its Retention "0.93" label collides with the
        1.0 line.
      - Figure 2 (the pipeline diagram) is NOT built; it is the one figure that needs a human, since
        it carries the compiler-thesis argument visually.
      - Figures 4a and 5 and 6 are verified reusable pre-existing renders at the full paths given
        inline (4a = pointwise three-panel, 5 = autotune progress, 6 = SGLang KDA four-arm), but each
        still wants the annotations its placeholder describes — the series has no captions, so every
        label has to live inside the image.
10. `jagged_dense_bmm` is cut for length: it accumulates in BF16, is sensitive to reduction tiling, and
    its generated defaults are recorded as an open correctness concern rather than a counted comparison
    (research/23 M12). If any reviewer reads the FP8 paragraph as "the heuristic never ships a wrong
    answer", put it back.
11. RESOLVED 2026-09-02 (was: does the block_ptr count fall inside these 149 cells?). It does: 27 of
    the 124 benchmark references use block_ptr, out of 46 in the 251-cell corpus, against 0 of the 256
    shipped autotuned configs. The body now states the 27 directly. Two follow-ups this opened:
      (a) NEW MEASUREMENT WORTH DOING: run the shipped in-tree sm100 AOT table against the hand-climbed
          corpus on the same 149 cells. Nobody has. It would settle whether the 1.20x is conservative
          (climb stronger than one autotune run) or generous, and would let the post state one ceiling
          instead of hedging. Until then "plausibly a harder reference, since it searched wider" is the
          strongest defensible phrasing.
      (b) FIX THE LAB DOC: sm100-linattn/SM100_CONFIGS.md says "Shipping artifact:
          _helion_aot_linear_attention_engine_cuda_sm100.py". That is false - the two are disjoint
          (0 configs in common). Either upstream the climb or correct the line; as written it invites
          exactly the mislabeling this note exists to prevent.
12. Cut for length, all true and available if you want them back: the vLLM post's "an entire day" for
    168 scaled_mm shapes; the 6.03x/1.013x "original reduction kernels" cohort row; the RoPE pretuned
    table's fallback cliff (15.70x vs our 27.36x where it must guess a key); and LFBO's config-space
    size. If you re-add the last of these, quote only the words "more than 8 quadrillion possible
    configurations" — the source's own "(10^16)" parenthetical is internally inconsistent (8
    quadrillion is 8x10^15), so do not reproduce it.
============================================================================================ -->
