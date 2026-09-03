# Positioning Reference: The Two Prior PyTorch Helion Autotuning Posts

Research note for the compile-time-heuristics blog post. Everything below was
read from the live pages (fetched 2026-09-02) and from the published figure PNGs
(read as images, so figure-internal numbers are attributed as **[figure read]**).

Sources fetched:
- Post A: <https://pytorch.org/blog/accelerating-autotuning-in-helion/> — raw text extract kept at `/tmp/h1.txt`, raw HTML at `/tmp/helion1.html`
- Post B: <https://pytorch.org/blog/from-minutes-to-seconds-llm-guided-autotuning-for-helion-kernels/> — raw text extract at `/tmp/h2.txt`, raw HTML at `/tmp/helion2.html`
- Figure PNGs downloaded to `/tmp/figs/`

Both posts are ~1,400 (A) and ~1,950 (B) words including code blocks/tables.
Neither has an acknowledgements section, a references list, or figure captions
(`<figcaption>` count = 0 in both). Comments are closed on both.

---

## 0. The one-line positioning summary

- **Post A (Feb 2026)** made *search itself* smarter: swap exhaustive
  single-parameter Pattern Search for an ML-classifier-filtered search (LFBO).
  Result: **36.5% less autotuning time on B200, 25.9% on MI350** — but still
  minutes per kernel.
- **Post B (Jun 2026)** made the *proposal step* smarter: ask an LLM for
  configs. Result: **~10x fewer configs, ~6.7x less wall-clock (39 s vs 261 s)**
  — but still tens of seconds per kernel-and-shape, plus an LLM API dependency.
- **Neither post removes the compile-and-benchmark loop.** Both keep a GPU in
  the loop, both are per-kernel-per-shape, and Post B's *published prompt
  already contains a "Compiler Analysis / Compiler-derived seed config(s)"
  block* — i.e. Post B silently consumes an early version of the heuristics the
  new post is about, without naming, measuring, or ablating them. Post B's last
  sentence is an explicit promissory note: "we plan to enhance the heuristics to
  further boost the effectiveness of both the LLM-guided and hybrid autotuners."
  **The new post is the paper that cashes that note.** Do not re-explain LFBO or
  LLM search; cite them and move to "what if you don't search at all."

---

## 1. Post A — "Accelerating Autotuning in Helion with Bayesian Optimization"

**Byline (verbatim):** "By Ethan Che, Oguz Ulgen, Max Balandat, Jongsok Choi,
Jason Ansel" — **February 24, 2026**.

### 1.1 Problem stated

Verbatim (Introduction, para 2):

> "However, the performance gains from auto-tuning comes with a cost: long
> wall-clock times. A typical autotuning session can take 10+ minutes,
> evaluating thousands of candidate configurations, and can even take on the
> order of hours for complex kernels. Since its launch, long autotuning times
> have consistently surfaced as a user complaint and one of the biggest pain
> points in the kernel development cycle. While Helion provides developers
> options to shorten the auto-tuning process, e.g. by reducing the number of
> search steps, this typically leads to a loss in kernel performance, forcing an
> undesirable trade-off."

(Note the typo "comes" is in the original — quote it as-is or paraphrase.)

### 1.2 Mechanism (exact)

- **Baseline arm = `PatternSearch`**, the *previous default*: "starts from
  multiple promising configurations ('search copies') and explores neighboring
  configs by exhaustively evaluating all single-parameter perturbations."
  Criticism, verbatim: "While thorough, this approach is inefficient: the vast
  majority of neighbors offer no performance improvement, yet each is compiled
  and benchmarked. Furthermore, restricting moves to single-parameter changes
  limits the algorithm's ability to traverse the high-dimensional search space
  quickly."
- **New arm = `LFBOPatternSearch`** (`helion.autotuner.surrogate_pattern_search`).
  Inspiration: Bayesian Optimization (cites botorch / Ax / arXiv 1807.02811);
  the specific adaptation is **Likelihood-Free Bayesian Optimization**
  (arXiv 2206.13035), "which uses a lighter-weight classification model as a
  surrogate."
- The 5-step algorithm, verbatim bullets:
  1. "Similar to PatternSearch, we first benchmark a set of randomly generated
     configs, and identify a small set of the most promising configurations
     ('search copies')."
  2. "We generate candidates from the search copies, by making random
     perturbations across multiple parameters, exploring more widely than
     PatternSearch."
  3. "We train a classification model (RandomForest) on latency data collected
     so far. Instead of predicting latency directly, we predict a binary label
     indicating whether the config is in the top 10% in terms of latency."
  4. "We rank the candidates based on ML model predictions. Unlike typical LFBO,
     we also add a penalty for similarity to previously ranked candidates to
     encourage exploration."
  5. "We select the top 10% of them to compile and benchmark. We update the
     search copies based on the best performing configs and add the latencies to
     the dataset."
- **Crucial data-provenance claim** (relevant to the new post, which needs *no*
  data at all): "Importantly, the model only uses data collected during the
  search process, and doesn't need the user to provide any additional data."
- Two named design decisions: **Classification vs Regression** (classification
  focuses capacity on top configs *and* can learn to avoid configs that "error
  out or suffer compile timeouts (as these are assigned negative labels)",
  which regression cannot because those points have no latency) and
  **Encouraging Diversity** (similarity score from "leaf node co-occurrence"
  in the Random Forest, penalising clustered candidates that "waste the batch
  budget on redundant samples").
- Ablation: regression surrogates (Random Forest, Gradient-Boosting Tree, MLP)
  on PatternSearch logs, scored by "the expected improvement in kernel latency
  when using the surrogate to filter the next batch of candidates." Verbatim
  punchline: "Notably, when we only can select 10% of candidates, the
  regression-based methods perform equivalent or even worse than simple random
  selection, as regression is not always aligned with ranking performance."

### 1.3 Headline numbers, with baselines (all measured **vs PatternSearch**)

Verbatim lead-in + bullets:

> "Using ML, we can reduce autotuning time substantially without sacrificing
> performance:
> - On our set of benchmark NVIDIA B200 kernels, we reduce autotuning time by
>   36.5% while improving kernel latency by 2.6% on average.
> - On AMD MI350 kernels, we reduce autotuning time by 25.9% while improving
>   kernel latency by 1.7%."

> "For some kernels the improvements are especially significant: we see up to a
> 50% reduction in wall-clock time for B200 layer-norm kernels, and even a >15%
> improvement in kernel latency for B200 Helion FlashAttention kernels. Due to
> its enhanced performance, it is the default search algorithm at the time of
> writing."

**Baseline is explicitly the old default search, NOT an unseeded/untuned
kernel.** There is no torch.compile arm, no eager arm, no vendor-library arm
anywhere in Post A. All of Post A's numbers are search-vs-search.

**[figure read] Fig 1 ("LFBO vs Pattern Search [B200]")** — the B200 kernel set
is 8 kernels: `cross_entropy, flash_attention, int4_gemm, layer_norm, rms_norm,
rms_norm-bwd, softmax, welford`. Left panel "Performance Speedup" (Pattern
Search = 1.0, error bars): flash_attention ≈ 1.175, softmax ≈ 1.065,
welford ≈ 1.03, others ≈ 1.00–1.02, rms_norm-bwd very slightly < 1.0. Right
panel "Wallclock Time Fraction" (Method / Pattern Search): layer_norm ≈ 0.49,
welford ≈ 0.50, rms_norm-bwd ≈ 0.52, rms_norm ≈ 0.68, softmax ≈ 0.77,
cross_entropy ≈ 0.80, int4_gemm ≈ 0.81, flash_attention ≈ 0.93.

**[figure read] Fig 2 ("LFBO vs Pattern Search [MI350]")** — 7 kernels (no
flash_attention): wallclock fractions ≈ 0.59–0.96; layer_norm is the one kernel
where LFBO is slightly *slower* in kernel latency (≈0.995).

### 1.4 Heading sequence (Post A)

`h1` title → `h2 Introduction` → `h2 The Challenges of Kernel Autotuning` →
`h2 Likelihood-Free Bayesian Optimization Pattern Search` → `h2 Conclusion`.
No TL;DR box. No sub-`h3`s at all. 4 `h2`s total.

---

## 2. Post B — "From Minutes to Seconds: LLM-Guided Autotuning for Helion Kernels"

**Byline (verbatim):** "By Jongsok Choi, Ethan Che, Jason Ansel, Oguz Ulgen" —
**June 18, 2026**, with an update stamp of **June 24th, 2026**.

### 2.1 Problem stated

The TL;DR names LFBO as the incumbent and the residual cost as the problem, in
one sentence repeated twice in the post (TL;DR and Introduction, identical
wording — they clearly consider it the thesis sentence):

> "LFBO is a strong baseline which works well, but it still grinds through
> hundreds of compile-and-benchmark cycles per kernel."

Also, verbatim (Introduction): "Reducing the tuning time is also critical for
developer velocity and production deployment, which impacts Helion adoption."

### 2.2 TL;DR verbatim (this is the whole box)

> "Helion, PyTorch's domain-specific language (DSL) for performance portable
> machine learning kernels, heavily relies on autotuning for performance.
> Currently Helion searches utilize the Likelihood-Free Bayesian Optimization
> (LFBO) to find the most performant configs. LFBO is a strong baseline which
> works well, but it still grinds through hundreds of compile-and-benchmark
> cycles per kernel. To this end, we introduce an LLM-guided autotuner that
> matches LFBO-level kernel performance (geomean 1.009X) while benchmarking ~10X
> fewer configurations in ~6.7X less wall-clock time. For the handful of kernels
> where the LLM trails by >5%, a hybrid strategy (LLM seeding followed by LFBO
> refinement) closes the gap while remaining ~3X cheaper than the full LFBO
> search. Finally, the result is largely LLM model-independent — Opus-4.8,
> gpt-5.5, and Sonnet-4.6 perform within a couple percent of each other —
> showing that LLM-guided autotuning is a practical approach to dramatically
> faster kernel tuning at production quality."

### 2.3 Mechanism (exact)

- **Population-based search driven by prompt/feedback rounds.** "In the initial
  phase, Helion provides the kernel and the associated details to the LLM to ask
  for a set of candidate configurations. Once LLM responds, Helion compiles and
  benchmarks the configs, retaining the top-performing configurations.
  Subsequent refinement rounds then occur, where the LLM is given the most
  successful configs, their performance metrics, and an analysis of successful
  patterns to guide specific mutations. If no significant performance gains are
  detected, the process terminates early."
- Initial prompt contents (bulleted, verbatim labels): **Kernel source**
  ("The actual @helion.kernel source code"), **Input Tensors**
  ("arg[0]: shape=[4096, 1024], dtype=torch.float16, …"), **GPU Hardware**
  ("NVIDIA B200, 148 SMs, 178.4 GB, 2048 threads/SM"), **Configuration Space**
  ("Every tunable field with type/range"), **Default Configuration**
  ("The baseline config."). Output contract demands minified single-line JSON
  `{"configs":[...]}`.
- Model returns "A minified JSON with 15 configs" (per round).
- Refinement prompt fields: Search state, Anchor configs, Results, Top/failed
  config patterns, Next Step. Early stop: "The feedback loop stops early if
  relative improvement from each round drops below ~0.5%."
- **Hybrid = `LLM-Seeded LFBO Search`**: Stage 1 one round of LLM seeding →
  Handoff ("The most successful LLM-generated configs serve as the starting
  point for the next phase to train LFBO's surrogate model. This allows LFBO to
  begin with immediate knowledge of promising regions rather than starting from a
  blank slate.") → Stage 2 LFBO refinement, capped at 20 iterations.
- Stated motivation for the hybrid, verbatim: "to address LLM's tendency to
  leave micro-architectural knobs unexplored."

### 2.4 Benchmark methodology (exact arms)

> "We compare LFBO (LFBOTreeSearch with full effort) to LLM-Guided Search using
> Opus 4.8 across 11 kernels — matmul (square + split-K), grouped-GEMM,
> attention, fp8-attention, softmax, rms_norm, rope, swiglu, mamba2, and
> gated-delta-net — on small, medium, and large shapes on NVIDIA B200."

33 cases = 11 kernels x 3 shapes. **Again: no torch.compile / eager / vendor
baseline anywhere.** Everything is search-vs-search on the same kernel bodies.

### 2.5 Headline numbers, with baselines

**Result 1 (over all 33 cases, vs full LFBO):**

> "Geomean configs benchmarked: 9.8X fewer configs (~55 vs ~546 per kernel) for
> LLM-guided autotuner. This is a machine-independent metric that demonstrates
> the efficacy of the new approach."

> "Geomean wall-clock time: 6.7X less end-to-end tuning time (39 s vs 261 s),
> measured on a 384-thread host. The end-to-end tuning time consists of config
> generation (for the LLM, including its API round-trips), Triton/ptxas
> compilation of every candidate, and GPU benchmarking of every candidate."

**Result 2:** "Across all 12 convergence kernels, the LLM drops to its plateau
inside roughly the first ~7% of LFBO's budget. On grouped GEMM (g=4, m=512),
that's 18X fewer configs than LFBO at the same kernel performance."

**Result 3:** "On kernel performance, the LLM is roughly on-par with LFBO, with
the geomean performance of LLM kernel/LFBO kernel latency being 1.009X." …
"There are 8 cases where LLM loses to LFBO by more than 5%."

**Hybrid:** "The hybrid strategy improves kernel performance in all cases and
closes the gap to LFBO in 6/8 cases. The mamba2 family still does worse than
LFBO and we are investigating improving the LLM heuristics to close this gap."
"Across the 8 kernels, it explores 4X fewer configs LFBO, leading to 3X faster
end-to-end autotuning time."

**Geomean table (exact cells; this table is over the 8-kernel gap subset, see
caveat C4):**

|                 | LLM-only | Hybrid | Full LFBO |
|---|---:|---:|---:|
| Autotuning Time | 44 s | 111 s | 328 s |
| Explored Configs | 59 | 186 | 686 |

**Model-independence table (exact cells, full 33 cases, LLM-only):**

| model | geomean perf vs Opus-4.8 | Geomean configs explored |
|---|---:|---:|
| Opus-4.8 | 1.00 (baseline) | 55 |
| gpt-5.5 | 0.98 | 61 |
| Sonnet-4-6 | 1.03 | 51 |

(Post B writes "Sonnet-4-6" in the table and "Sonnet-4.6" in prose.)

### 2.6 Figure-internal numbers (the richest source of "search is expensive")

**[figure read] "Search efficiency: configs evaluated per kernel — geomean 9.8x
fewer configs"** (33 bars, sorted by LFBO cost). Full-LFBO configs benchmarked
per single kernel-and-shape, top of range: **mamba2 (S=2048, C=128) ≈ 1024
configs**; matmul square (8192³) ≈ 990; matmul split-K (K=1024) ≈ 977;
grouped-GEMM (g=8, M=512) ≈ 968; fp8-attention (S=512, D=64) ≈ 967. Bottom of
range ≈ 210 (gated-delta-net S=8192, C=128). LLM-guided bars are all ≈ 40–68.

**[figure read] "Autotuning cost per kernel: LLM-guided vs full LFBO — geomean
6.7x faster"** (log-scale seconds, per-bar labels legible). Full LFBO extremes:
**760 s (grouped-GEMM g=8, M=512) ≈ 12.7 minutes for ONE kernel at ONE shape**,
then 733, 595, 599, 589, 581, 574, 480, 470, 380, 370, 357, 355, 346, 333, 323,
303, 284, 264, 257, 203, 199, 194, 179, 177, 153, 148, 133, 113, 104, 60, 53,
52 s. LLM-guided range: **22 s (matmul square 1024³) to 64 s**; most bars
30–55 s. This is the single best "residual cost" figure for the new post:
*after* a 6.7x acceleration, the cheapest tuned cell still costs 22 s of
GPU+CPU work.

**[figure read] "Convergence vs search effort (all 12 convergence kernels)"** —
x-axis is "configs benchmarked / LFBO's total configs", y is "best config so far
/ best-ever (1.0 = optimal)". A dashed vertical annotation reads **"LLM's full
budget (~10% of LFBO's)"**. Note for skeptics: one full-LFBO trace plateaus at
**~10x worse than best-ever even at 1.0 of its own budget** — i.e. on at least
one kernel full LFBO never gets near the best config any arm found. (Figure
observation, not stated in the prose.)

**[figure read] "Per-kernel performance: LLM-guided autotuner vs full LFBO on
B200"** — 33 horizontal bars, ratio LLM ms / LFBO ms, with a ±2.5% "noise band".
Grouping: **3 "LLM faster", 18 "tie (±2.5%)", 12 "LFBO faster"**. Outlier worth
quoting: **matmul split-K (K=65536) at ≈ 0.45 — LLM-guided is ~2.2x faster than
full LFBO**, i.e. full LFBO badly missed on that cell. Worst LFBO-favouring
cells: mamba2 (S=8192, C=256) ≈ 1.22, mamba2 (S=4096, C=256) ≈ 1.13,
attention (S=8192, D=64) ≈ 1.13, fp8-attention (S=512, D=64) ≈ 1.13,
attention (S=4096, D=64) ≈ 1.12.

**[figure read] "Does the hybrid close the gap?"** — the "8 cases" set is
exactly: gated-delta-net (S=4096, C=64) 1.04→1.01; matmul square (8192³)
1.05→1.01; matmul square (4096³) 1.08→1.02; grouped-GEMM (g=8, M=512)
1.09→1.01; attention (S=4096, D=64) 1.12→1.02; attention (S=8192, D=64)
1.12→0.99; mamba2 (S=4096, C=256) 1.13→1.06; mamba2 (S=8192, C=256)
1.22→1.13. "Closed the gap" is judged against a **±2.5% noise band** (drawn in
the figure).

**[figure read] "LLM-only vs Hybrid vs Full LFBO autotuning cost"** (same 8
cells, two panels: configs and log-seconds). Configs: LLM-only 50–67; hybrid
140–284; full LFBO 222–991. Seconds: LLM-only 36–54; hybrid 44–235; full LFBO
**52–780**. Note the hybrid is *not* uniformly much cheaper: on
gated-delta-net (S=4096, C=64) it is 44 s vs LFBO's 52 s.

### 2.7 Heading sequence (Post B)

`h3 Featured projects` (Helion logo widget) → hero social image inside an empty
`h3` → **`h3 TL;DR`** → `h2 Introduction` → `h2 How the LLM-Guided Autotuner
Works` (with bolded inline subhead "The Initial Prompt", then a `<pre>` prompt
block) → *(a stray empty `h2` in the markup)* → `h2 What the Model Returns` →
`h2 Refinement Rounds` → `h2 LLM-Seeded LFBO: The Best of Both Worlds`
(+ `h3 The Hybrid Workflow`) → `h2 Benchmarking Results` (+ `h3 The Methodology`,
`h3 Result 1: The Efficiency Win`) → `h2 Result 2: LLM Converges in the First
~7% of LFBO Budget` → `h2 Result 3: LLM Delivers LFBO-level Performance` →
`h2 Can the Hybrid Search Close the Gap?` → `h2 Does the Model Matter?` →
`h2 Conclusions`.

Note the inconsistency to *avoid copying*: Result 1 is an `h3` under
"Benchmarking Results" while Results 2 and 3 are top-level `h2`s.

---

## 3. QUOTE BANK — every sentence in either post that quantifies the cost of autotuning

This is the deliverable the new post's "search is expensive even when
accelerated" argument rests on. All verbatim.

**Post A** (Accelerating Autotuning in Helion with Bayesian Optimization,
Che/Ulgen/Balandat/Choi/Ansel, Feb 24 2026):

1. "However, the performance gains from auto-tuning comes with a cost: long
   wall-clock times."
2. "A typical autotuning session can take 10+ minutes, evaluating thousands of
   candidate configurations, and can even take on the order of hours for complex
   kernels."
3. "Since its launch, long autotuning times have consistently surfaced as a user
   complaint and one of the biggest pain points in the kernel development
   cycle."
4. "While Helion provides developers options to shorten the auto-tuning process,
   e.g. by reducing the number of search steps, this typically leads to a loss
   in kernel performance, forcing an undesirable trade-off."
5. "While compiling and measuring the latency of a single configuration takes on
   the order of seconds, the autotuning engine typically searches through
   thousands of configurations to achieve the best possible performance."
6. "Even a simple kernel like LayerNorm has more than 8 quadrillion (10^16)
   possible configurations. However, while the search space is large, only a
   small fraction of configs have good performance."
7. "Long Compile Times: Certain kernel configurations can take a significant
   amount of time to compile, unnecessarily extending the autotuning process's
   wall-clock time."
8. "Config Errors and Timeouts: The search space can also include configs that
   have compilation errors, produce inaccurate results, or take too long to
   compile."
9. "While thorough, this approach is inefficient: the vast majority of neighbors
   offer no performance improvement, yet each is compiled and benchmarked."
10. "We see not only that LFBO completes auto-tuning earlier (~5 min instead of
    ~9 min), it finds better configurations faster with much larger jumps in
    performance compared to Pattern Search." — **this is the strongest single
    quote for the new post: the accelerated default still takes ~5 minutes on
    one layer-norm kernel at one shape.**
11. "we see up to a 50% reduction in wall-clock time for B200 layer-norm
    kernels" (i.e. best case of Post A's own technique is still a halving, not
    an elimination).
12. **[figure read, Post A Fig 4]** the trace is titled
    `layer_norm - 4096x10752_10752_10752`; x-axis "Wall-clock Time (m)" runs
    1→9; LFBO's first measured point lands at ~1.4 min and it terminates at
    ~5.2 min; Pattern Search terminates at ~8.7 min. Best latency found:
    ~0.0594 ms (LFBO) vs ~0.0686 ms (Pattern Search).

**Post B** (From Minutes to Seconds, Choi/Che/Ansel/Ulgen, Jun 18 2026):

13. "LFBO is a strong baseline which works well, but it still grinds through
    hundreds of compile-and-benchmark cycles per kernel." (appears twice —
    TL;DR and Introduction)
14. "Every Helion kernel is tuned across a vast, high-dimensional configuration
    space (tile sizes, block sizes, num_warps, num_stages, see documentation for
    more) to reach peak performance on the target hardware."
15. "Reducing the tuning time is also critical for developer velocity and
    production deployment, which impacts Helion adoption."
16. "Geomean configs benchmarked: 9.8X fewer configs (~55 vs ~546 per kernel)
    for LLM-guided autotuner."
17. "Geomean wall-clock time: 6.7X less end-to-end tuning time (39 s vs 261 s),
    measured on a 384-thread host. The end-to-end tuning time consists of config
    generation (for the LLM, including its API round-trips), Triton/ptxas
    compilation of every candidate, and GPU benchmarking of every candidate."
18. "End-to-end tuning time is dominated by compiling candidate configs, where
    Helion precompiles them in parallel across CPU cores. As this host has 100s
    of threads, compilation is heavily parallelized. On a machine with fewer
    cores, the LLM's ~10X fewer configs would translate into a proportionally
    larger wall-clock time reduction." — **use this: their own 39 s / 261 s
    numbers are a best case that assumes a 384-thread host.**
19. "Across all 12 convergence kernels, the LLM drops to its plateau inside
    roughly the first ~7% of LFBO's budget. On grouped GEMM (g=4, m=512),
    that's 18X fewer configs than LFBO at the same kernel performance."
20. "Hence LLM gives you a good config fast, while LFBO can outperform at the
    cost of more config exploration and tuning time."
21. "Across the 8 kernels, it explores 4X fewer configs LFBO, leading to 3X
    faster end-to-end autotuning time." (sic — "configs LFBO")
22. Table: Full LFBO "Autotuning Time 328 s" / "Explored Configs 686"; hybrid
    111 s / 186; LLM-only 44 s / 59.
23. "The efficiency gain is substantial: The LLM-guided autotuner converges to
    LFBO-quality results in 7% of LFBO's budget, explores ~10X fewer
    configurations, with ~6.7X reduction in wall-clock time, offering a massive
    boost in developer velocity."
24. **[figure read]** full LFBO peaks at **≈1024 configs** and **760 s** for a
    single kernel-and-shape (grouped-GEMM g=8, M=512 for the seconds;
    mamba2 S=2048, C=128 for the configs).
25. **[figure read]** even the *fast* arm costs **22–64 s per kernel-and-shape**
    (LLM-guided, 384-thread host).

**What is NOT in either post** (do not attribute these to them): GPU-hours,
dollar cost, CI-time budgets, per-shape retuning cost across a serving shape
distribution, cache-miss/cold-start rates in production, or any claim about
users skipping autotuning entirely. Post A's "user complaint … biggest pain
points in the kernel development cycle" is the closest thing to user friction
either post offers.

---

## 4. Residual cost after both techniques — what they explicitly do NOT solve

Ordered by how directly the posts admit it.

1. **The compile-and-benchmark loop is still mandatory, and still dominates.**
   Post B: "End-to-end tuning time is dominated by compiling candidate configs."
   Both techniques reduce the *count* of candidates; neither removes the need to
   compile, run, and time candidates on the target GPU. Floor after both:
   ~22–64 s and ~40–68 configs per kernel-and-shape (Post B figures).
2. **A GPU must be present at tune time.** Both papers' pipelines benchmark on
   the target hardware. Neither offers a no-GPU, no-benchmark path.
3. **Per-kernel *and* per-shape.** Post B's unit of work is a kernel x shape
   cell (33 = 11 x 3); Post A's traces are per-kernel-per-shape
   (`layer_norm - 4096x10752_10752_10752`). Neither post claims a config
   transfers across shapes, and neither measures shape transfer. The new post's
   compile-time f(shape)->config framing is untouched territory.
4. **Post B adds a *new* dependency and a new cost:** LLM API round-trips are
   inside its measured wall-clock ("config generation (for the LLM, including
   its API round-trips)"). Nothing in Post B addresses offline / air-gapped /
   no-API-key builds, rate limits, nondeterminism, or per-call token cost.
5. **Post B's speedup is host-dependent and its 6.7x is a best case.** "measured
   on a 384-thread host … On a machine with fewer cores, the LLM's ~10X fewer
   configs would translate into a proportionally larger wall-clock time
   reduction." (The direction favours them on small hosts for the *ratio*, but
   the *absolute* residual seconds get worse.)
6. **Search still misses.** Post B's own figure has full LFBO ending ~2.2x
   slower than LLM-guided on matmul split-K (K=65536), and one convergence trace
   plateauing ~10x off best-ever. Post A concedes rms_norm-bwd/layer_norm cells
   where LFBO is slightly worse than the search it replaced. So "just search
   harder" is not a guaranteed quality ceiling either.
7. **Named open weakness that the new post's families address directly:** "We
   also explore a hybrid strategy (LLM-Seeded LFBO Search) to address LLM's
   tendency to leave micro-architectural knobs unexplored." and "The mamba2
   family still does worse than LFBO and we are investigating improving the LLM
   heuristics to close this gap." **mamba2 and gated-delta-net are linear
   attention = the new post's multi-matmul family.** Post B leaves the two
   worst residual cells (mamba2 S=4096/8192, C=256: hybrid still 1.06x and
   1.13x off LFBO) unsolved and explicitly hands them to "heuristics".
8. **Neither post ever compares to a non-Helion baseline.** No torch.compile, no
   eager, no vendor kernel, no vLLM. Everything is search-vs-search inside
   Helion. The new post's torch.compile / AOT-pre-tuned / vLLM-shipped-config
   arms are therefore *new axes*, not a re-run of their charts — say so.

---

## 5. Their exact framing of the tradeoff

- **Post A's framing is a forced trade:** speed vs quality. "While Helion
  provides developers options to shorten the auto-tuning process, e.g. by
  reducing the number of search steps, this typically leads to a loss in kernel
  performance, forcing an undesirable trade-off." The whole post exists to bend
  that curve, not to escape it. It also frames autotuning as *the source of
  Helion's advantage*: "As a result, Helion can achieve significant speedups
  over torch.compile and even highly-optimized, hand-written kernels in Triton
  or CuTe DSL."
- **Post A's three named difficulty axes** (verbatim headings):
  "High-Dimensional, Combinatorial Space", "Long Compile Times",
  "Config Errors and Timeouts".
- **Post B's framing is cost-per-quality with a recipe:** "Hence LLM gives you a
  good config fast, while LFBO can outperform at the cost of more config
  exploration and tuning time." And the closing "The Practical Recipe": "For a
  streamlined workflow, we suggest trying the LLM-only search to rapidly
  identify a high-performance kernel. To maximize performance, users can apply
  the hybrid search to refine and capture the final performance gains."
- **Cold-start language:** Post B is the one that names it, but only for the
  *surrogate*, not for the user: LLM seeding "allows LFBO to begin with
  immediate knowledge of promising regions rather than starting from a blank
  slate", and "the hybrid search converges significantly faster than a cold LFBO
  search." Post B also frames the LLM as an alternative to blind starts:
  "What if, instead of starting the search blindly, you could ask an LLM to
  reason about the kernel and propose configurations?"
- **Adoption framing (Post B):** "Reducing the tuning time is also critical for
  developer velocity and production deployment, which impacts Helion adoption."
- **Neither post says users skip autotuning.** Post A says long tuning times are
  "a user complaint and one of the biggest pain points"; that is the strongest
  available claim. If the new post wants "users skip autotuning", it must source
  that itself or phrase it as an inference.

---

## 6. Existing "heuristics / seeded search / analytical config selection" language

**This section is the collision risk. Read it before drafting.**

### 6.1 Post A: none.

The word "heuristic" appears exactly once, and it means *local search operator*,
not compile-time analysis: "We combine the local search heuristic of Pattern
Search with the LFBO classifier model to filter only the most promising
candidates to benchmark, instead of exhaustive search." There is no mention of
compile-time analysis, static analysis, analytical config derivation, structural
priors, or seed configs. Post A's search starts from **random** configs: "we
first benchmark a set of randomly generated configs". **The new post's
contribution is entirely absent from Post A.**

### 6.2 Post B: yes — a compiler-heuristic seed block is already in the published prompt.

Post B prints, verbatim, under "The Initial Prompt":

> "The Helion compiler also analyzes the kernel to add heuristics to the prompt.
> For rms_norm:
>
> ```
> ## Compiler Analysis
> Helion's compiler statically analyzed this kernel's structure and derived the
> following structural priors. Treat them as strong starting points:
> Compiler-derived seed config(s):
>   {"block_sizes":[1],"load_eviction_policies":["last","last","last","last","last"],"reduction_loops":[null]}
> ```

Three things follow, and the new post must handle all three:

- **The heuristics already existed and already shipped as of June 2026**, at
  least for the reduction/pointwise family (that rms_norm seed —
  `reduction_loops: [null]`, one row per program — is the `triton_reduction_tile`
  heuristic; cf.
  `/home/dev/local/wt-sm100-linattn/helion/_compiler/autotuner_heuristics/`
  — files `triton.py`, `cute.py`, `pallas.py`, `registry.py`, `common.py`,
  `matmul_b200.json` — and the purpose string at
  `/home/dev/local/wt-sm100-linattn/helion/autotuner/llm/prompting.py:207-216`).
  So the new post should NOT
  claim to invent seeding; it should claim to (a) generalise it to four families
  including multi-matmul, (b) make it good enough to *replace* search, and
  (c) measure it, which Post B never does.
- **Post B never ablates the Compiler Analysis section.** Every LLM-guided
  number in Post B was produced with compiler-derived seeds already in the
  prompt. That means Post B's own headline is *partly* a heuristics result and
  nobody measured which part. This is a legitimate, generous framing for the new
  post: "the seeds were already helping; here is how much."
- **Post B's own source confirms the dual consumption.** Docstring in
  `/home/dev/local/wt-sm100-linattn/helion/autotuner/llm/prompting.py:229-231`
  (branch `calebmkim/stack/50`):
  "Helion fires eligible heuristics before autotuning; their names and
  analytical seed configs already feed the search seed path. This also shows
  them to the LLM as a structural prior." So the same heuristic output feeds
  (1) the autotuner seed pool and (2) the LLM prompt.

**Other seeding language in Post B to reference rather than reinvent:**

- "LLM-Seeded LFBO: The Best of Both Worlds" / "Stage 1 – LLM Seeding" —
  "seeding" in Post B means *the LLM seeds LFBO*. The new post's "seeding" means
  *the compiler seeds either search*. Disambiguate explicitly on first use, e.g.
  "compiler-derived seeds (distinct from the LLM seeding in the previous post)".
- "The kernel author provided the following seed config(s)…" — there is also an
  *author*-supplied `autotune_seed_configs` path
  (`build_author_seed_section`, same file). Post B does not mention it; the new
  post should not conflate it with compiler-derived seeds.
- **The promissory note to cash, verbatim (Post B's final sentence):** "Moving
  forward, we plan to enhance the heuristics to further boost the effectiveness
  of both the LLM-guided and hybrid autotuners." Also, one paragraph earlier:
  "we are investigating improving the LLM heuristics to close this gap" (about
  mamba2). Quote the first; it is the cleanest handoff sentence available and it
  names both downstream consumers.

**Suggested one-sentence positioning line** (draft, adapt):
> Two earlier posts made Helion's search cheaper — a learned surrogate
> ([Accelerating Autotuning in Helion with Bayesian Optimization]) and an LLM
> proposer ([From Minutes to Seconds]) — and the second one already fed the
> model a "Compiler Analysis" block of statically derived seed configs. This
> post is about that block: what the compiler can work out about a kernel before
> a single candidate is compiled, and how far it gets with no search at all.

---

## 7. Style notes for the drafting agents

**Openings.**
- Post A: no TL;DR. Opens by linking back to the earlier
  <https://pytorch.org/blog/helion/> intro post ("As introduced in a previous
  blog post, Helion is a high-level DSL that empowers developers to write
  high-performance ML kernels using a familiar PyTorch-like syntax, delegating
  the complex task of optimization to its autotuning engine."), then problem
  paragraph, then a 2-bullet results list, then the two headline figures, then a
  "for some kernels" paragraph. Headline bullets land **before** any mechanism.
- Post B: a "Featured projects" widget with the Helion logo, a hero social card
  image, then an `h3 TL;DR` — a single dense ~150-word paragraph (not bullets)
  that contains every headline number. Then Introduction repeats the thesis
  sentence verbatim. **Recommend copying Post B's TL;DR-paragraph pattern**
  (it is the newer house style and the one the same team most recently shipped)
  while keeping Post A's habit of a short bullet list of headline results
  immediately before the first figure.

**Results-section pattern (Post B, worth imitating):** numbered, named result
headings — "Result 1: The Efficiency Win", "Result 2: LLM Converges in the
First ~7% of LFBO Budget", "Result 3: LLM Delivers LFBO-level Performance" —
each one claim + numbers + one figure. Then an adversarial-question heading
("Can the Hybrid Search Close the Gap?", "Does the Model Matter?") that
pre-empts a reviewer objection and answers it with a table. Close with
"Conclusions" as bolded lead-in phrases ("The efficiency gain is substantial:",
"LLM reaches LFBO-level performance:", "Hybrid strategy bridges the gap:",
"The Practical Recipe:").

**Figure captions: there are none, in either post.** All figures are bare
centered `<img>`s. Post A's six figures have `alt=""` entirely — every label
lives *inside* the PNG (chart title, subtitle carrying the geomean, axis labels
that state the polarity). Post B's `alt` text is short and title-like
("Search efficiency: configs evaluated per kernel", "Autotuning cost per
kernel", "Convergence vs Search Effort", "Per Kernel Performance ",
"Does the Hybrid Close the Gap?", "LLM-only vs Hybrid vs LFBO-Autotuning").
**Implication for the new post: bake the caption into the image.** Post B's
figure titles do this well, with a two-line title where line 2 is the aggregate
("Search efficiency: configs evaluated per kernel / geomean 9.8x fewer
configs") and axis labels that state polarity ("configs benchmarked during
autotuning — lower is more efficient", "perf ratio (LLM ms / LFBO ms) — < 1.0
means LLM is faster"). Post B also draws an explicit **±2.5% noise band** and
colours bars by tie/win/loss with a legend — a good, honest device to copy.

**Tables.** Post B uses two small HTML tables (3-4 columns, 2-3 data rows) with
plain `<td>` header rows, units inline in the cells ("44 s", "1.00 (baseline)").
No table captions or numbering.

**Acknowledgements.** Neither post has an acknowledgements section, a
contributors list, or a references section. Citations are inline hyperlinks only
(Post A links botorch, Ax, arXiv 1807.02811, arXiv 2206.13035, and two
helionlang.com API anchors; Post B links helionlang.com autotuner docs, the
Post A blog, and the Helion project page). **If the new post wants an
acknowledgements block, it will be the first in the series** — that is fine but
it is a deviation, so flag it to the editor.

**Bylines.** Comma-separated author list, no affiliations, no links. Both posts
share three authors (Che, Ansel, Ulgen); Choi is on both. Overlapping authorship
means the new post should read as *the same team continuing*, not as a rebuttal.

**Tone/register.** Both are first-person-plural, present tense, no hedging
about motivation but explicit about limits ("There are cases that LFBO wins at
the cost of higher autotuning time"; "The mamba2 family still does worse").
Post A ends with a solicitation: "We are actively interested in applying
additional ML techniques to enhance the auto-tuner, including methods from
reinforcement learning (RL) and large language models (LLMs), and welcome any
contributions." Post B ends with a recipe + roadmap. Neither uses emoji, neither
uses a "Try it out" / install-snippet section.

---

## 8. Cross-checks and caveats a skeptical reviewer will raise

- **C1. Post A's numbers are search-vs-search only.** "36.5% less autotuning
  time, 2.6% better latency" is *vs PatternSearch*, on 8 B200 kernels. Do not
  paraphrase it as "vs no autotuning" or "vs default".
- **C2. Post A's kernel-latency wins are small and noisy.** Most per-kernel bars
  in Fig 1 sit within ~2% of 1.0 with error bars crossing 1.0; the 2.6% average
  is carried by flash_attention (~1.175) and softmax (~1.065). One cell
  (rms_norm-bwd on B200, layer_norm on MI350) is slightly *worse*.
- **C3. Post B's arithmetic checks out** on the stated geomeans: 261/39 = 6.69
  ("6.7X"); 546/55 = 9.9, reported as "9.8X" (they state geomeans are computed
  cell-by-cell, so a small mismatch against the ratio-of-geomeans is expected).
- **C4. Post B's two sets of aggregate numbers are easy to confuse.** "~55 vs
  ~546 configs / 39 s vs 261 s" is the **full 33-case** geomean. The
  "44 s / 111 s / 328 s, 59 / 186 / 686" table is the **8-case gap subset**
  (introduced by "Across the 8 kernels…"), though the post never labels the
  table as such. Never mix the two.
- **C5. The ">5%" gap set is really ≥~4%.** The prose says "8 cases where LLM
  loses to LFBO by more than 5%", but the hybrid figure's 8 cells start at
  1.04x (gated-delta-net S=4096, C=64) and 1.05x (matmul square 8192³). The set
  is best described as "the 8 largest LFBO-favouring gaps, 1.04x–1.22x".
- **C6. "Closes the gap in 6/8" is judged against a ±2.5% band**, per the
  figure's legend, not against exact parity.
- **C7. "Model-independent" is a geomean claim over 33 cells.** Sonnet-4.6's
  1.03 is *worse* than Opus-4.8 by 3% in geomean while using fewer configs; the
  post's own noise band is ±2.5%, so 0.98/1.03 straddles it. "within a couple
  percent of each other" is fair; "identical" would not be.
- **C8. Post A's 10^16 LayerNorm config-count claim is unsourced** in the post
  (no formula, no version pin), and internally slightly off: "more than
  8 quadrillion (10^16)" — 8 quadrillion is 8x10^15, not 10^16. If the new post
  cites a search-space size, derive and cite its own.
- **C10. "All 12 convergence kernels" (Post B, Result 2) is not the 33-case
  suite.** The post never says which 12, nor whether "kernel" here means kernel
  or kernel-and-shape cell. Treat "~7% of LFBO's budget" as a claim over an
  unspecified 12-item subset, and "18X fewer configs" as a single cell
  (grouped GEMM g=4, m=512).
- **C9. Post B's per-kernel timing is host-specific (384 threads)** and Post A
  publishes no host spec at all. Any absolute-second comparison the new post
  draws against them must carry a "different host" caveat.

---

## 9. Verbatim sentences most likely to be quoted in the new post

Ranked by usefulness for the "search is expensive even when accelerated" thesis:

1. Post A: "A typical autotuning session can take 10+ minutes, evaluating
   thousands of candidate configurations, and can even take on the order of
   hours for complex kernels."
2. Post B: "LFBO is a strong baseline which works well, but it still grinds
   through hundreds of compile-and-benchmark cycles per kernel."
3. Post A: "Since its launch, long autotuning times have consistently surfaced
   as a user complaint and one of the biggest pain points in the kernel
   development cycle."
4. Post A: "we see not only that LFBO completes auto-tuning earlier (~5 min
   instead of ~9 min)…"
5. Post A: "While Helion provides developers options to shorten the auto-tuning
   process, e.g. by reducing the number of search steps, this typically leads to
   a loss in kernel performance, forcing an undesirable trade-off."
6. Post B: "End-to-end tuning time is dominated by compiling candidate configs,
   where Helion precompiles them in parallel across CPU cores."
7. Post B: "Geomean wall-clock time: 6.7X less end-to-end tuning time
   (39 s vs 261 s), measured on a 384-thread host."
8. Post B: "Moving forward, we plan to enhance the heuristics to further boost
   the effectiveness of both the LLM-guided and hybrid autotuners."
9. Post A: "While compiling and measuring the latency of a single configuration
   takes on the order of seconds, the autotuning engine typically searches
   through thousands of configurations to achieve the best possible
   performance."
10. Post B: "The mamba2 family still does worse than LFBO and we are
    investigating improving the LLM heuristics to close this gap."
