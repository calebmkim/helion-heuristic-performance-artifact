# 01 — Voice + Structure Reference: the PyTorch/Helion blog series

**Purpose:** style + content reference for the new "compile-time heuristics" post. Everything below
was re-derived from the live HTML of the published posts (fetched 2026-09-02), not from summaries.
Verbatim quotes are marked `>` or "quoted"; anything I could not verify is in §9 Gaps.

---

## 0. PROVENANCE + A FETCH GOTCHA THAT WILL BITE THE NEXT AGENT

⚠️ **pytorch.org serves stale CDN content for these two URLs on a cold `curl`.** On first fetch,
both target URLs returned HTTP 200 with the *wrong article body*:

| Requested URL | First fetch returned (WRONG) | Actual post (after cache-bust) |
|---|---|---|
| `https://pytorch.org/blog/helion/` | "Accelerating Autotuning in Helion with Bayesian Optimization" | **"Helion: A High-Level DSL for Performant and Portable ML Kernels"** |
| `.../pytorch-foundation-welcomes-helion-as-a-foundation-hosted-project-to-standardize-open-portable-and-accessible-ai-kernel-authoring/` | "From Minutes to Seconds: LLM-Guided Autotuning for Helion Kernels" | **"PyTorch Foundation Welcomes Helion as a Foundation-Hosted Project…"** |

Fix: append a random query string and send no-cache headers, e.g.
`curl -sL -H "Cache-Control: no-cache" "https://pytorch.org/blog/helion/?cb=$RANDOM$RANDOM"`.
Verify with `grep -o '<link rel="canonical"[^>]*>'` — the canonical must match the slug you asked for.
(Both wrong-body responses had a canonical pointing at the *other* post, which is how I caught it.)

Local verbatim transcripts I extracted (markdown, from the correct bodies) are alongside this file:
`_source-intro-post.md`, `_source-foundation-post.md`, `_source-lfbo-post.md`,
`_source-llm-autotune-post.md`, `_source-vllm-post.md`.

### The full Helion series (chronological) — the new post is post #9

| Date | Title | Byline | Article words |
|---|---|---|---:|
| 2025-10-22 | Helion: A High-Level DSL for Performant and Portable ML Kernels | **By PyTorch Team at Meta** | 2,645 |
| 2026-02-03 | Portable Paged Attention in Helion | By Burkhard Ringlein (IBM Research) and the vLLM Team at IBM Research | 4,937 |
| 2026-02-24 | Accelerating Autotuning in Helion with Bayesian Optimization | By Ethan Che, Oguz Ulgen, Max Balandat, Jongsok Choi, Jason Ansel | 1,421 |
| 2026-04-07 | PyTorch Foundation Welcomes Helion as a Foundation-Hosted Project… | **By PyTorch Foundation** | 711 |
| 2026-06-10 | Portable vLLM Model Inference Kernels in Helion | By Sean Chen (Red Hat) and Yanan Cao (PyTorch, Meta Platforms) | 2,430 |
| 2026-06-18 | From Minutes to Seconds: LLM-Guided Autotuning for Helion Kernels | By Jongsok Choi, Ethan Che, Jason Ansel, Oguz Ulgen | 1,941 |
| 2026-07-10 | Towards Free Normalization: Fusing Normalization into GEMM and Attention Kernels | By Jacky (Junqing) Zhou, Hongtao Yu, … (10 names) | 5,525 |
| 2026-07-23 | Helion on TPU: Towards Hardware Heterogeneous Kernel Authoring | By Dunfan Lu, Yifei Xu, Jongsok Choi, Ethan Che, Oguz Ulgen, Jason Ansel, Yarong Mu, Theotime Combes, Emilio Cota, Freya Azad | 2,002 |

Slugs (for cross-linking): `helion/`, `portable-paged-attention-in-helion/`,
`accelerating-autotuning-in-helion/`,
`pytorch-foundation-welcomes-helion-as-a-foundation-hosted-project-to-standardize-open-portable-and-accessible-ai-kernel-authoring/`,
`portable-vllm-model-inference-kernels-in-helion/`,
`from-minutes-to-seconds-llm-guided-autotuning-for-helion-kernels/`,
`towards-free-normalization-fusing-normalization-into-gemm-and-attention-kernels/`,
`helion-on-tpu-towards-hardware-heterogeneous-kernel-authoring/`.

**Note:** the LLM-autotuning post links the LFBO post as
`https://pytorch.org/blog/accelerating-autotuning-in-helion/` — use that slug, not `/blog/helion/`,
when citing LFBO. The `/blog/helion/` slug is the *intro* post.

---

## 1. STRUCTURE

### 1a. Post A — "Helion: A High-Level DSL for Performant and Portable ML Kernels" (the canonical intro)

Exact heading sequence, with body-text word count, paragraph/list-item count, images, tables per
section (H1 = title, rendered by the site chrome; the body starts at H2):

```
H2  Introduction to Helion .................................. w=179  p=2   img=0 tbl=0
H2  Motivation for a New DSL ............................... w=159  p=4   img=0 tbl=0     (3 bulleted contrasts: CUDA/Gluon/TLX, Triton, PyTorch)
H2  Helion Programming Model: "PyTorch with Tiles" ......... w=397  p=8   img=0 tbl=0     (+ 1 code block: the matmul kernel)
H2  Helion's Autotuner: Generating Optimal Kernels via
      Implicit Search Spaces ............................... w=118  p=3   img=1 tbl=0
H3    The Autotuning Workflow .............................. w=148  p=3   img=0 tbl=0     (+ 2 code blocks: autotuner stdout, then the pasted @helion.kernel(config=...))
H3    The Configuration Space .............................. w=383  p=11  img=0 tbl=1     ← the ONE real HTML table in the post
H2  Performance Analysis and Benchmarks .................... w=50   p=1   img=0 tbl=0     (2-sentence methodology preamble)
H3    Performance on NVIDIA B200 ........................... w=93   p=1   img=0 tbl=0
H3    (empty heading) ...................................... w=0    p=0   img=1 tbl=0     ← the B200 results *screenshot*
H3    Performance on AMD MI350X ............................ w=68   p=1   img=0 tbl=0
H3    (empty heading) ...................................... w=0    p=0   img=1 tbl=0     ← the MI350X results *screenshot*
H3    Case Study 1: Outperforming Highly Optimized CuTe
        DSL Kernel ......................................... w=103  p=4   img=1 tbl=0
H3    Case Study 2: Benchmarking Helion to TileLang ........ w=313  p=12  img=2 tbl=0
   (bold <p>, NOT a heading:) High-Level Compiler Architecture  ← diagram + 5 numbered pipeline stages
H2  Conclusions ............................................ w=167  p=8   img=0 tbl=0
   (plain <p>, NOT a heading:) Acknowledgements → italic <p> of names
   (plain <p>:) "Note: B200 Performance numbers updated 10/23."
```

Load-bearing structural facts:
- **No TL;DR and no bulleted results up top.** The post opens with two dense
  problem-statement paragraphs ("In modern machine learning, the demand for high-performance
  computation has led to a proliferation of custom kernels…") and the numbers do not appear until
  ~55% of the way in. This is the *2025* convention.
- Section length: **50–400 body words per section.** Nothing is long. Median paragraph = **52
  words** (n=33, min 6, max 107).
- **Benchmark tables are screenshot PNGs, not HTML tables.** Both the B200 and MI350X result grids
  are `Screenshot-2025-10-2X-at-….png` dropped under an *empty* H3. The prose immediately above the
  screenshot states the geomeans in words. The only real `<table>` is the Configuration Space
  knob/description table (8 body rows + header).
- Figures land **immediately after** the paragraph that describes them, with **no captions and no
  "Figure N" numbering** anywhere in this post.
- Code blocks are plain `<pre>` with no syntax-highlight language tag, no line numbers, no caption.
- 7 of 14 headings are wrapped in `<strong>` (visually bolded H2s) — cosmetic WordPress artifact,
  not meaningful.

### 1b. Post B — "PyTorch Foundation Welcomes Helion as a Foundation-Hosted Project…"

**This is a Linux Foundation press release, not an engineering post.** Do not model the new post's
structure on it. 711 words total. Byline "By PyTorch Foundation". Structure:

```
(hero image: helion-1024x536.png, wrapped in italics)
italic one-line dek: "Helion joins community of leading open source AI projects to simplify kernel
                      development across the open AI ecosystem"
bold datelined lede: "PARIS – April 7, 2026 – The PyTorch Foundation, a community-driven hub for
                      open source AI under the Linux Foundation, today announced that it has
                      welcomed Helion as its newest foundation-hosted project alongside DeepSpeed,
                      PyTorch, Ray, and vLLM."
3 body paragraphs (positioning; one embedded executive quote from Matt White)
1 paragraph: what Helion is technically
1 paragraph: ExecuTorch → PyTorch Core
1 paragraph: upcoming conferences
bold  Supporting Quotes   (2 quotes, each followed by "– Name, Title, Org" in bold)
###   (bare hash separator)
H3    About the PyTorch Foundation
bold  About the Linux Foundation
      trademark boilerplate (italic)
bold  Media Contact  → Grace Lucier / The Linux Foundation / pr@linuxfoundation.org
```
No headings besides one H3, no figures besides the hero, **no tables, no code, no
acknowledgements section, no numbers except the two quoted claims.**

### 1c. What the *current* (mid-2026) series convention looks like — model the new post on this

Every engineering post from June 2026 onward opens with a **`TL;DR`** section as the very first
heading (H3 in the LLM and vLLM and TPU posts; H2 in the normalization post), containing **exactly
one paragraph of 67–191 words** that names the baseline, the headline ratio, and the caveat. The
vLLM post italicizes the whole TL;DR paragraph; the others do not.

Canonical TL;DR to imitate (LLM-autotuning post, 145 words) — note it puts the *win*, the *cost
metric*, the *exception*, and the *robustness check* all in one paragraph:

> Helion, PyTorch's domain-specific language (DSL) for performance portable machine learning
> kernels, heavily relies on autotuning for performance. Currently Helion searches utilize the
> Likelihood-Free Bayesian Optimization (LFBO) to find the most performant configs. LFBO is a strong
> baseline which works well, but it still grinds through hundreds of compile-and-benchmark cycles
> per kernel. To this end, we introduce an LLM-guided autotuner that matches LFBO-level kernel
> performance (geomean 1.009X) while benchmarking ~10X fewer configurations in ~6.7X less
> wall-clock time. For the handful of kernels where the LLM trails by >5%, a hybrid strategy (LLM
> seeding followed by LFBO refinement) closes the gap while remaining ~3X cheaper than the full LFBO
> search. Finally, the result is largely LLM model-independent — Opus-4.8, gpt-5.5, and Sonnet-4.6
> perform within a couple percent of each other — showing that LLM-guided autotuning is a practical
> approach to dramatically faster kernel tuning at production quality.

Result-section naming convention worth stealing (LLM post): **`Result 1: The Efficiency Win`**,
`Result 2: LLM Converges in the First ~7% of LFBO Budget`, `Result 3: LLM Delivers LFBO-level
Performance`, then a question-headed section `Can the Hybrid Search Close the Gap?`, then
`Does the Model Matter?` (an ablation framed as a reader's objection), then `Conclusions` as 4
**bold-lead-in** bullets (`**The efficiency gain is substantial:** …`, `**The Practical Recipe**: …`).

Two other conventions the new post should adopt:
- **`Caveats`** as its own H2 near the end (vLLM post) — a place to put the honest losses. The
  vLLM post's Caveats section is 172 words / 2 paragraphs and is where "an entire day" of
  autotuning and "tens of microseconds of CPU overhead per kernel launch" live.
- **`Resources`** H2 (vLLM post, 40 words) pointing at a GitHub issue/branch for reproduction.

Caption convention: **only the vLLM post numbers its exhibits**, as italic lines *below* the
element: `*Fig. 1: Total throughput speedup on H100 with per-token activation quantization enabled,
using the default vLLM setup as the baseline.*` and `*Tab. 2: A summary of the geometric-mean
speedups achieved by Helion kernels.*` The intro, LFBO, LLM, TPU and normalization posts use **zero
captions**. Since the new post has many arms, the vLLM post's `Fig. N:` / `Tab. N:` style is the
better precedent and is defensible as in-series.

Table style in the series: real HTML tables are used for (a) the knob/description reference table
(intro post), (b) small geomean roll-ups with **bold** headers and bold winning cells (LLM post:
3-column `LLM-only / Hybrid / Full LFBO`; 3-row model ablation), (c) wide per-cell benchmark grids
(vLLM post Tab. 2: 9 kernel rows × 6 speedup columns, `N/A` in cells with no applicable baseline;
Tab. 3: 12 columns of TTFT/TPOT/throughput with the three speedup columns bolded). All of these are
small enough to read — no post ships a table with more than ~12 rows.

---

## 2. VOICE

Measured, per 100 words:

| Post | we/our density | "you" | contractions | speedup token | hedges |
|---|---:|---:|---:|---|---|
| Intro (2025-10) | **0.2** (5 total) | 1 | 0 | lowercase `x` (18) | can×18, often×4, typically×2 |
| LFBO (2026-02) | **2.7** (38) | 0 | 1 | (percentages only) | typically×3, ~×2 |
| LLM (2026-06) | 0.8 (15) | 5 | 2 | uppercase `X` (16) | ~×12, roughly×2, largely×1 |
| vLLM (2026-06) | 1.4 (33) | 3 | 0 | lowercase `x` (45) | ~×3, up to×2 |

Conventions to follow:
- **Person.** The intro post is almost entirely *impersonal third person about the system*
  ("Helion resolves this conflict by…", "Helion changes this dynamic with implicit search spaces").
  First person appears **only** in the benchmark section and only as methodology: *"We benchmark the
  performance of Helion to `torch.compile` (with max-autotune), and hand-written Triton…"*,
  *"We implemented the Mamba-2-chunk-scan kernel … in Helion to compare against…"*. The
  results-heavy posts (LFBO, vLLM) use "we" freely for every methodological and measurement choice.
  **Recommended for the new post: mechanism described impersonally, every measurement choice in
  "we".** Never "I". Zero authorial personality; no jokes.
- **Tense.** Present tense for mechanism and for results ("Helion achieves the highest geomean
  speedup of 3.27x"); simple past for what the authors did ("We implemented…", "we observed up to
  approximately 1.09x"); present-progressive for roadmap ("Ongoing work on Helion's CuteDSL backend
  is expected to further improve…", "The Helion team is actively reducing the dispatch latency").
- **Hedging.** Numbers are stated flatly but wrapped in scope. The dominant devices are
  `geomean`/`geometric-mean` (mandatory for any cross-shape number), `approximately` / `~`,
  `up to`, `on average`, `across all benchmarks`, `in most cases`, and naming the arm inline
  (`over eager mode execution`, `against baseline`, `using the default vLLM setup as the baseline`).
  Losses are stated plainly and then attributed to a named external cause, never buried.
- **Jargon budget.** Very high assumed familiarity: `num_warps`, `num_stages`, `tl.load`
  `eviction_policy`, TMA/Tensor Memory Accelerator, PID swizzling, persistent kernels, SM,
  FX Graphs, SSA, TorchInductor, CUDA graph capture/replay, UE8M0, CUTLASS/DeepGEMM, geomean — all
  used with at most a half-sentence gloss. But *Helion-specific* terms always get an explicit
  definition on first use (`hl.tile`, host code vs device code, "implicit search spaces",
  "search copies").
- **Paragraph length.** Intro post median 52 words (max 107); LFBO median 81 (max 134); LLM median
  35; vLLM median 20 (very list-heavy). Target **40–70 words/paragraph, 2–5 sentences**, and lean on
  bullets for enumerations. Nothing in the series runs past ~135 words in a paragraph.
- **`x` vs `X`.** Intro + vLLM + paged-attention use lowercase `1.21x`; only the LLM post uses
  `10X`. **Use lowercase `x`.**

### 5–8 representative sentences, verbatim

1. > Helion resolves this conflict by compiling a high-level Python-embedded domain-specific
   > language (DSL) into automatically tuned Triton code. — *intro post*
2. > By automating tedious and error-prone tasks like tensor indexing, memory management, and
   > hardware-specific tuning, Helion empowers developers to focus on algorithmic logic rather than
   > hardware-specific implementation details. — *intro post*
3. > Helion makes kernel implementations radically simpler: the Attention kernel is just 30 lines in
   > Helion, compared to 120 lines in Triton and thousands of lines in CUDA. — *intro post*
4. > We benchmark the performance of Helion to `torch.compile`(with max-autotune), and hand-written
   > Triton to measure their respective speedups over eager mode execution across a wide variety of
   > kernels and shapes on NVIDIA B200 and AMD MI350X GPUs. — *intro post* (the canonical
   > methodology sentence: arms named, baseline named, hardware named, in one sentence)
5. > Across all benchmarks, Helion achieves the highest geomean speedup of 3.27x, followed by
   > `torch.compile` (with max-autotune) at 2.7x, and hand-written Triton kernels at 1.76x. — *intro post*
6. > However, the performance gains from auto-tuning comes with a cost: **long wall-clock times**.
   > — *LFBO post* (note: the series' standard way to introduce the autotuning-cost problem, and the
   > exact rhetorical slot the new post occupies)
7. > However, while the search space is large, only a small fraction of configs have good
   > performance. — *LFBO post*
8. > The primary limiting factor for B200 is the performance of Triton-generated GEMM kernels on
   > Blackwell GPUs rather than the Helion programming model itself. — *vLLM post* (the house style
   > for owning a loss: state it, then localize the cause outside the thing you're advocating)
9. > This means the baseline in this section is **not** the default vLLM configuration. However, the
   > comparison is still meaningful because we are able to use consistent CUTLASS kernels for the
   > linear layer for all runs. — *vLLM post* (the house style for a baseline caveat — bold "not",
   > then the justification. **The new post needs this exact move for its `torch.compile`
   > default-mode and vLLM-shipped-config arms.**)

---

## 3. THE CANONICAL FRAMING OF HELION'S CONFIGURATION SPACE

**Cite these; do not re-derive.** Every sentence in the corpus that quantifies the space:

From the intro post (`/blog/helion/`), §"Helion's Autotuner: Generating Optimal Kernels via Implicit
Search Spaces":
> Helion's key differentiator is its automated, ahead-of-time (AOT) autotuning engine. In Triton,
> developers are responsible for manually defining the search space for optimizations. This requires
> explicitly enumerating every configuration to be tested, a tedious process that limits the scope of
> exploration.

> Helion changes this dynamic with implicit search spaces. The high-level language automatically
> constructs a vast, multi-dimensional search space over implementation choices. For example, a
> single `hl.tile` call implicitly instructs the autotuner to explore different block sizes, loop
> orderings, and whether to flatten the iteration space into a single dimension. **One Helion kernel
> definition thus maps to thousands of Triton configurations**, allowing the autotuner to create a
> much larger and richer search space to discover a superior configuration.

> The configuration space represents the set of implementation choices that Helion automates. This
> space is the primary source of Helion's performance portability, as it allows a single kernel
> definition to be adapted to the unique characteristics of different hardware architectures and
> input tensor sizes. Exploring this space is what gives Helion its advantage over manually written
> kernels, which are often tuned for a specific set of conditions.

> The autotuner explores a wide range of parameters that control everything from data movement to
> thread mapping. The table below details the configuration options.

From the LFBO post (`/blog/accelerating-autotuning-in-helion/`) — **the biggest and most precise
number in the corpus**:
> **High-Dimensional, Combinatorial Space:** The space of all possible combinations of block sizes,
> unroll factors, etc. is high-dimensional and vast. **Even a simple kernel like LayerNorm has more
> than 8 quadrillion (10^16) possible configurations.** However, while the search space is large,
> only a small fraction of configs have good performance.

> This autotuner explores a vast, high-dimensional space of implementation choices—block sizes, loop
> orders, memory access patterns—to discover configurations that maximize performance on the target
> hardware.

From the LLM-autotuning post:
> Every Helion kernel is tuned across a vast, high-dimensional configuration space (tile sizes,
> block sizes, num_warps, num_stages, see [documentation](https://helionlang.com/api/autotuner.html)
> for more) to reach peak performance on the target hardware.

From the Foundation press release (Matt White, Global CTO of AI at the Linux Foundation and CTO of
the PyTorch Foundation):
> Helion gives engineers a much more productive path to writing high-performance kernels, including
> **autotuning across hundreds of candidate implementations for a single kernel**.

⚠️ **These three magnitudes are not the same quantity and the corpus is loose about it.** "8
quadrillion (10^16)" = size of the *combinatorial space* for LayerNorm; "thousands of Triton
configurations" = what one Helion kernel definition *maps to* / what a search *evaluates*
("evaluates thousands of candidate Triton kernel configurations"); "hundreds of candidate
implementations" = the press-release simplification. **Recommendation for the new post: quote the
LFBO post's `8 quadrillion (10^16)` when arguing the space is too big to enumerate, and the intro
post's `thousands of Triton configurations` when talking about what a search touches. Do not average,
reconcile, or restate them as one number.**

### The exact knob table from the intro post (8 body rows) — the canonical knob vocabulary

| Row label as printed | Gloss as printed (condensed; the post gives 1–6 sentences each) |
|---|---|
| `indexing` | Three memory-access methods: "pointer arithmetic, block pointers, and tensor descriptors, which leverage Tensor Memory Accelerators (TMAs) on NVIDIA Hopper/Blackwell GPUs"; values `'pointer'`, `'block_ptr'`, `'tensor_descriptor'` |
| `block_sizes` | "a list of tile sizes for each dimension in an `hl.tile` loop that determines the amount of data each thread block processes. This affects register usage, shared memory requirements, and parallelism." |
| `flatten_loops` | "controls flattening a multi-dimensional tiling space of a `hl.tile` loop into a single dimension" |
| `loop_orders,l2_grouping` | `loop_orders` "allows the autotuner to permute the iteration order of nested tiles"; `l2_grouping` "enables PID swizzling, a technique that reorders the assignment of thread blocks to improve data reuse in the L2 cache" |
| `reduction_loops` | persistent reduction vs looped reduction; persistent "can create high register pressure, leading to register spilling and low performance" for large reductions |
| `pid_type` | `'flat'` (1D grid), `'xyz'` (multi-dim grid), `'persistent_blocked'`/`'persistent_interleaved'` ("launch only one thread block per Streaming Multiprocessor (SM)") |
| `load_eviction_policy` | autotunes Triton `tl.load`'s `eviction_policy`, "influencing GPU L1 cache residency" |
| `Triton configs: num_warps, num_stages, range_unroll_factors, range_warp_specializes, range_num_stages, range_multi_buffers, range_flattens` | "Helion automatically explores standard Triton tunable parameters, alleviating the developer effort of manual tuning." |

⚠️ Two rows use **singular field names that do not match the real config fields**: the table says
`l2_grouping` and `load_eviction_policy`, but the autotuner output in the same post prints
`l2_groupings=[4]` and the paged-attention post prints `load_eviction_policies=['', …]`. The new post
should use the **plural, real** field names (`l2_groupings`, `load_eviction_policies`) and can note
they map to the intro post's table rows — but should not "correct" the old post out loud.

Also note `flatten_loops` appears in the table but **not** in either printed autotuner config
(replaced in practice by `range_flattens` / the `pid_type` + loop machinery). Don't claim the table
is an exhaustive current field list.

---

## 4. AUTOTUNING WALL-CLOCK COST — verbatim claims

Intro post, §"The Autotuning Workflow" (the claim the new post's premise rests on):
> When a kernel is run for the first time without a specified configuration, the autotuner initiates
> an automated search. **This process, which typically takes around 10 minutes, evaluates thousands
> of candidate Triton kernel configurations** using search strategies like Differential Evolution or
> Pattern Search to identify optimized sets of parameters for the given input shapes and hardware.
> Upon completion, the autotuner prints the single best configuration it discovered:

**Sample autotuner output, verbatim (intro post):**
```
[586s] Autotuning complete in 586.6s after searching 1520 configs.
One can hardcode the best config and skip autotuning with:
    @helion.kernel(config=helion.Config(block_sizes=[64, 64, 64], 
loop_orders=[[0, 1]], l2_groupings=[4], range_unroll_factors=[0, 1], 
range_warp_specializes=[None, False], range_num_stages=[0, 3], 
range_multi_buffers=[None, False], range_flattens=[None, None], 
num_warps=8, num_stages=6, indexing='block_ptr', pid_type='flat'))
```
Followed immediately by the same config pasted into a decorator (a second `<pre>`), then:
> The developer can copy this config into the `@helion.kernel()` decorator in their source code.
> This instructs Helion to bypass the search process entirely during subsequent runs. In a production
> environment, this results in fast, deterministic compilation that generates the single,
> pre-optimized Triton kernel, delivering performance equivalent to a meticulously hand-tuned kernel
> with far less effort.

> The developer can also specify a list of configs in `@helion.kernel()`, in which case Helion will
> explore only those configs to choose the fastest implementation.

**A second, larger sample autotuner output, verbatim (paged-attention post)** — useful because it
shows a real 2026-era field set and a 46-minute search:
```
[2752s] Autotuning complete in 2752.1s after searching 5353 configs.

One can hardcode the best config and skip autotuning with:

@helion.kernel(config=helion.Config(
     block_sizes=[32, 4], 
     indexing=['pointer', 'pointer', 'pointer', 'pointer', 
'tensor_descriptor', 'pointer', 'tensor_descriptor', 'pointer', 
'tensor_descriptor'], 
l2_groupings=[2],
load_eviction_policies=['', '', '', '', '', 'last', 'last', ''], 
loop_orders=[[1, 2, 0], [1,0]], 
num_stages=6, num_warps=8, pid_type='flat', 
range_flattens=[None, True, True, True], 
range_multi_buffers=[None,  None, None, False], 
range_num_stages=[], range_unroll_factors=[0, 1,  2, 1], 
range_warp_specializes=[]), static_shapes=False)
```

LFBO post — the strongest "autotuning is painful" framing available to quote:
> However, the performance gains from auto-tuning comes with a cost: **long wall-clock times**. **A
> typical autotuning session can take 10+ minutes, evaluating thousands of candidate configurations,
> and can even take on the order of hours for complex kernels.** Since its launch, long autotuning
> times have consistently surfaced as a user complaint and one of the biggest pain points in the
> kernel development cycle. While Helion provides developers options to shorten the auto-tuning
> process, e.g. by reducing the number of search steps, this typically leads to a loss in kernel
> performance, forcing an undesirable trade-off.

> While compiling and measuring the latency of a single configuration takes on the order of seconds,
> the autotuning engine typically searches through thousands of configurations to achieve the best
> possible performance.

> Long Compile Times: Certain kernel configurations can take a significant amount of time to
> compile, unnecessarily extending the autotuning process's wall-clock time.

> We see not only that LFBO completes auto-tuning earlier (~5 min instead of ~9 min), it finds better
> configurations faster with much larger jumps in performance compared to Pattern Search.

LLM post — the current state of the art the new post is adjacent to:
> **Geomean configs benchmarked: 9.8X fewer configs (~55 vs ~546 per kernel) for LLM-guided
> autotuner**. This is a machine-independent metric that demonstrates the efficacy of the new
> approach.

> **Geomean wall-clock time: 6.7X less end-to-end tuning time (39 s vs 261 s), measured on a
> 384-thread host**. The end-to-end tuning time consists of config generation (for the LLM, including
> its API round-trips), Triton/ptxas compilation of every candidate, and GPU benchmarking of every
> candidate.

> End-to-end tuning time is dominated by compiling candidate configs, where Helion precompiles them in
> parallel across CPU cores. As this host has 100s of threads, compilation is heavily parallelized. On
> a machine with fewer cores, the LLM's ~10X fewer configs would translate into a proportionally
> larger wall-clock time reduction. The machine-independent metrics are the number of configs
> benchmarked and how fast the best configs converge to their optimal results.

vLLM post, §Caveats — **the single best real-world hook for a heuristics post**:
> During our experiments, the majority of engineering time was spent on kernel autotuning. For large
> kernels such as scaled_mm, running a full-effort autotuning sweep across all three model sizes,
> covering a total of 168 distinct input shapes, **can take an entire day**, as Helion automatically
> generates and benchmarks thousands of candidate kernel implementations for each shape. Initial
> research suggests that exhaustive per-shape autotuning and dispatching may not always be necessary,
> and that reducing the number of specialization buckets may achieve a better tradeoff between
> autotuning cost and runtime performance with minimal performance degradation. **The Helion team is
> actively exploring additional techniques to further reduce tuning time, including search-space
> reduction strategies and LLM-guided autotuning approaches.**

⚠️ Wall-clock-cost numbers in the corpus, and their scope — keep them straight:
`~10 minutes` typical single-kernel search (intro, 2025) · `586.6s / 1520 configs` matmul example
(intro) · `2752.1s / 5353 configs` paged attention (Feb 2026) · `10+ minutes … order of hours for
complex kernels` (LFBO) · `~9 min → ~5 min` B200 LayerNorm, Pattern Search → LFBO (LFBO post) ·
`261 s → 39 s` geomean LFBO → LLM on a **384-thread host** (LLM post) · `328 s / 111 s / 44 s` for
LFBO / hybrid / LLM-only on the 8 hard cases (LLM post table) · `an entire day` for 168 `scaled_mm`
shapes (vLLM post).

---

## 5. ACKNOWLEDGEMENTS CONVENTION

Format varies by post; both spellings are in use.

**Intro post (the "project" style).** Not a heading at all — a plain `<p>` reading
`Acknowledgements`, followed by a fully **italic** `<p>`:

> *Helion is the work of many hands including: Jason Ansel, Oguz Ulgen, Will Feng, Jongsok Choi,
> Markus Hoehnerbach, Manman Ren, Jie Liu, Paul Zhang, Driss Guessous, Joy Dong, Xuan Zhang, Karthick
> Panner Selvam, Peng Wu, Hongtao Yu, Neil Dhar, Nick Riasanovsky, Shane Nay, Alexey Loginov, as well
> as teams at Meta, NVIDIA, AMD, and Intel*

Then a separate plain `<p>`: `Note: B200 Performance numbers updated 10/23.` (the series' convention
for post-publication number corrections).

**Contribution posts (the "thanks" style)** — an actual `H2`, plain (non-italic) single paragraph,
"we would like to thank our colleagues:" + first-and-last names, closing with "for their feedback and
support":

- `## Acknowledgments` (vLLM post): > This work was supported by many contributors across the OCTO
  and vLLM teams at Red Hat, as well as the Helion team at Meta. In particular, we would like to
  thank our colleagues: Luka Govedič, Richard Zou and Will Feng for their feedback and support
  throughout this work.
- `## Acknowledgments` (paged-attention post): > This work was supported by the AI platform team at
  IBM Research, in particular we would like to thank our colleagues: Thomas Parnell, Jan van
  Lunteren, Mudhakar Srivatsa, and Raghu Ganti. Also, we would like to thank Jason Ansel and the
  Helion team at Meta for their feedback and support, especially the fast fixing of the bugs we
  reported, sometimes even within 24 hours.
- `## Acknowledgements` (TPU post): > This project was made possible through the invaluable
  collaboration and technical insights of our peers. A special thank you to Joe Pamer, Robert Hundt,
  Claudio Basile and Adam Paszke at Google, as well as Jana van Greunen, Gregory Chanan, Peng Wu, and
  Zongwei Zhou at Meta, for their feedback and support in bringing this to fruition.
- `## Acknowledgements` (normalization post) thanks *external open-source authors* by name for prior
  art (Tri Dao, Markus Hoehnerbach, Jay Shah, Ted Zadouri, Vijay Thakkar, Wentao Guo for FlashAttention
  and Quack) and then "the Pytorch and Triton teams". This post also ships a `## References` section
  with `[1] Title. https://…` numbered entries — the only post in the series that does.

**Neither of the two autotuning posts (LFBO, LLM-guided) has an acknowledgements section at all** —
they instead carry a long author byline (5 and 4 names). **The two `-ing` autotuning posts are the
closest analogues to the new post, so the byline-only pattern is fully in-series.**

**Author-list style.** Comma-separated first-and-last names, no titles, no "et al.", prefixed "By ",
technical lead typically last or near-last (`Jason Ansel` is last in the LFBO byline, third in the
LLM byline). Cross-org posts append parenthetical affiliations: `By Sean Chen (Red Hat) and Yanan Cao
(PyTorch, Meta Platforms)`. Institutional posts use an org as the author: `By PyTorch Team at Meta`,
`By PyTorch Foundation`. Bylines run 1–10 names.

---

## 6. CLAIMS THE NEW POST MUST NOT CONTRADICT

Ordered by how easy each is to trip over.

1. **The heuristics are already publicly described — as an *input to the LLM autotuner*.** The
   LLM-guided post (2026-06-18) prints, verbatim, part of its rms_norm prompt:
   > ## Compiler Analysis
   > Helion's compiler statically analyzed this kernel's structure and derived the following
   > structural priors. Treat them as strong starting points.
   > Compiler-derived seed config(s):
   > {"block_sizes":[1],"load_eviction_policies":["last","last","last","last","last"],"reduction_loops":[null]}

   and prose: *"The Helion compiler also analyzes the kernel to add heuristics to the prompt."* The
   new post is the deep dive on that already-shipped machinery — it should **say so and link that
   post**, and its description of "emit a seed config" must be compatible with (a) the seed being a
   *partial* config dict (only the fields the heuristic owns), (b) `reduction_loops:[null]` style
   values, (c) heuristics feeding both the LLM prompt and the search. Do **not** frame the heuristics
   as brand-new or as replacing the autotuner.
2. **The autotuner default is LFBO, not Differential Evolution / Pattern Search.** The intro post
   (Oct 2025) says "search strategies like Differential Evolution or Pattern Search"; the LFBO post
   says LFBO "is the default search algorithm at the time of writing" and the LLM post says
   "Helion's current default autotuner uses LFBO". The vLLM post pins the exact default:
   `LFBOTreeSearch` with `initial_population=FROM_RANDOM, copies=5, max_generations=20,
   similarity_penalty=1.0`. **Any seeding/search-efficiency claim in the new post must name LFBO /
   LFBOTreeSearch as the search it seeds.**
3. **`torch.compile` baselines are not interchangeable across the series.** The intro post's
   `1.21x over torch.compile` is **max-autotune**. The vLLM post's `torch.compile` baseline is
   `fullgraph=True, dynamic=False, backend="inductor"` with `combo_kernels: True` +
   `benchmark_combo_kernel: True` matching vLLM's own setup. The new post's `torch.compile` arm is
   **default mode**. The new post must label its arm as default mode and must not present a
   default-mode ratio as an improvement on the intro post's max-autotune ratio.
4. **Helion already claims 1.21x over torch.compile(max-autotune) and 3.27x over eager on B200**
   (intro post, geomean, whole suite; 2.37x/1.05x/1.44x on MI350X). A heuristics post claiming very
   large ratios "vs default" must make clear that the default it beats is the *unseeded/pre-change
   compiler config*, not the autotuned Helion the intro post benchmarked. (This is already the
   ground-truth doc's own warning: `Pre` = "pre-PR compiler heuristic", not
   `_base_default_config()`; the general-pointwise 19.351x "primarily reflect[s] how poor the tiny
   unseeded base configurations are".)
5. **vLLM arms.** The published vLLM post's Tab. 2 geomeans are for **fully autotuned Helion
   kernels** vs CUTLASS / `torch.compile` / `torch.ops._C` — e.g. `rms_norm_dynamic_per_token_quant`
   1.180x vs torch.compile and 1.802x vs `torch.ops._C` on H100; 1.240x / 1.969x on B200;
   `silu_and_mul_per_block_quant` 1.731x / 2.269x (H100). It also states
   > For non-GEMM kernels, Helion consistently demonstrates strong performance and outperforms both
   > TorchInductor-generated kernels and the existing vLLM CUDA implementations.
   The new post's vLLM numbers are a **different thing**: same Helion kernel body, *vLLM's shipped
   config* vs *heuristic config* (a configuration comparison, per the ground-truth doc's own
   instruction: "not comparisons against vLLM's native CUDA operators"). Say that explicitly, in the
   vLLM post's own "the baseline in this section is **not** …" style, or readers will read the new
   heuristic/vLLM ratios (0.918x, 0.956x, 0.987x) as Helion regressing against the published post.
6. **GEMM on Blackwell via Triton is a known Helion weak spot, publicly owned.** The vLLM post:
   `scaled_mm` 0.739x and `scaled_mm_blockwise` 0.782x vs CUTLASS on B200, with
   > The primary limiting factor for B200 is the performance of Triton-generated GEMM kernels on
   > Blackwell GPUs rather than the Helion programming model itself.
   A matmul-heuristic post should not imply that Helion Triton matmul on B200 is at CUTLASS parity.
7. **Helion runtime dispatch has real CPU overhead.** vLLM post: *"Helion runtime dispatching itself
   introduces tens of microseconds of CPU overhead per kernel launch. For small kernels, this
   overhead can dominate the end-to-end latency. As a result, CUDA graph capture and replay are
   essential."* This matters for the new post's mix of eager-dispatch (linear attention) and
   CUDA-graph (SGLang KDA) timing — the series has already told readers the two are not comparable.
8. **Autotuning is framed as a *cost*, never as worthless.** Every autotuning post takes the line
   that search finds the best configs and the goal is to get there cheaper (LFBO: "saving time and
   discovering faster kernel configs"; LLM: "LFBO can outperform at the cost of more config
   exploration and tuning time"). The new post's own data agrees (AOT is 1.14–1.17x above the
   heuristic on linear attention; 1.20x above on the matmul corpus). Frame heuristics as
   *zero-tuning-time quality + a better search start*, not as "autotuning was unnecessary".
9. **Backend/portability claims.** Intro post: compiles to **Triton**. Foundation press release:
   *"designed to compile down to multiple backends for hardware heterogeneity (Triton, TileIR, and
   more coming soon)"*. TPU post: a **Pallas** backend. vLLM post: *"Ongoing work on Helion's CuteDSL
   backend"*. The new post's heuristics are Triton-backend heuristics (`autotuner_heuristics/triton.py`)
   — say "the Triton backend" rather than implying they cover all backends.
10. **Attribution/credit claims.** Intro post: *"the Attention kernel is just 30 lines in Helion,
    compared to 120 lines in Triton and thousands of lines in CUDA"*; *"written in less than a day"*
    for the RMSNorm-backward CuTe comparison; vLLM post: *"most kernels could be implemented and
    validated within a single day"*. Consistent "one day" productivity motif available to reuse.
11. **The LayerNorm example is already spoken for.** LFBO uses B200 LayerNorm as its running example
    (8 quadrillion configs; ~9→~5 min; the PCA plot). If the new post uses LayerNorm, keep the
    numbers consistent with that post or pick a different exemplar (the LLM post picked `rms_norm`).
12. **Minor: the press release's "hundreds of candidate implementations for a single kernel"** is a
    quote from Matt White. Don't quote it as a technical figure; use the intro/LFBO framings.

---

## 7. RECOMMENDED SKELETON FOR THE NEW POST (synthesized from the series)

```
TL;DR                                  (H3, 1 paragraph, 120–190 words: 4 families, the headline
                                        linear-attention + vLLM numbers, the AOT gap, the losses)
Introduction                           (H2, ~300 words: quote the ~10 min / entire-day cost, cite
                                        LFBO + LLM posts, state the thesis: static analysis, no search)
Why a Config Is Hard to Guess          (H2: quote 8 quadrillion + the intro post's knob table;
                                        do not re-derive the space)
How the Heuristics Work                (H2, then one H3 per family: matmul / multi-matmul /
                                        reduction / pointwise; 150–400 words each; 1 diagram)
Result 1: Linear Attention             (H2, "Result N:" naming per the LLM post; arm table +
                                        Fig. N/Tab. N captions per the vLLM post)
Result 2: vLLM Reduction Kernels
Result 3: Seeding the Autotuner        (search-efficiency framing, all 38 cells)
Caveats                                (H2, per the vLLM post: decode loss, AOT gap, censoring,
                                        timing-scope non-comparability)
Resources                              (H2: repo/PR links for reproduction)
Conclusion(s)                          (H2, 3–4 bold-lead-in bullets)
Acknowledgements                       (H2 or plain <p>+italic; or byline-only per the two
                                        autotuning posts)
```

## 8. QUICK REUSE CHECKLIST

- Lowercase `x` for ratios; always say `geomean` for cross-shape numbers.
- Name the arm inside the sentence ("over eager mode execution", "using the default vLLM setup as
  the baseline").
- 40–70-word paragraphs; bullets for enumerations; nothing over ~135 words.
- Impersonal for mechanism, "we" for every measurement decision. No "I". No jokes.
- Figures immediately after the paragraph that describes them; if numbering, use the vLLM post's
  italic `*Fig. N: …*` / `*Tab. N: …*` below the element.
- Losses get one plain sentence + a localized cause, in a `Caveats` H2.
- Post-publication number fixes go in a trailing `Note: … updated MM/DD.` paragraph.

---

## 9. GAPS / NOT ESTABLISHED

- Neither target post contains a **figure-generation or plotting style guide**; the intro post's
  result grids are opaque screenshots, so I cannot recover their color/typography conventions.
- The intro post's B200/MI350X per-kernel numbers are **inside PNGs** (`Screenshot-2025-10-23-at-8.44.36-AM.png`,
  `Screenshot-2025-10-21-at-10.31.36-AM.png`); only the prose geomeans (3.27x / 2.7x / 1.76x and
  2.37x / 2.26x / 1.65x, plus softmax 2.28x and jsd 6.22x) are machine-readable. I did not OCR them.
- No post states an **editorial word-count limit or review process**, so the 700–5,500-word range
  above is empirical, not a rule.
- The **canonical `<h1>` styling / hero-image requirement** is unclear: the LLM, vLLM, TPU and
  normalization posts each open with a 1024×576-ish social/hero image and an auto-generated
  "Featured projects" logo strip (site chrome, inserted by the CMS from the project taxonomy — not
  authored). The intro and LFBO posts have no hero. Whether a hero is now mandatory is unknown.
- I could not find any published Helion post that states a **per-knob search-space cardinality**
  (e.g. "block_sizes has N choices"), so the new post cannot cite one; only "8 quadrillion for
  LayerNorm" and "thousands" exist.
- The `Note: B200 Performance numbers updated 10/23.` convention appears **once** (intro post); I
  cannot confirm it is a house rule vs a one-off.
- The two autotuning posts have no acknowledgements section, so I cannot tell whether that is an
  editorial choice for author-heavy posts or an omission.
