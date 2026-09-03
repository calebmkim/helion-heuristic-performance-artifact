# Style reference: two PyTorch Helion blog posts (voice, mechanism depth, tables, limitations)

Research note for the "compile-time heuristics for Helion" PyTorch blog post.
Source material = the published HTML of two posts, fetched and converted locally.
All structural counts below were computed from the HTML DOM (BeautifulSoup walk), not eyeballed.

| | Post A | Post B |
|---|---|---|
| Title | **Helion on TPU: Towards Hardware Heterogeneous Kernel Authoring** | **Portable Paged Attention in Helion** |
| URL | https://pytorch.org/blog/helion-on-tpu-towards-hardware-heterogeneous-kernel-authoring/ | https://pytorch.org/blog/portable-paged-attention-in-helion/ |
| Byline (verbatim) | "By Dunfan Lu, Yifei Xu, Jongsok Choi, Ethan Che, Oguz Ulgen, Jason Ansel, Yarong Mu, Theotime Combes, Emilio Cota, Freya Azad" | "By Burkhard Ringlein (IBM Research) and the vLLM Team at IBM Research" |
| Body prose words (p + li text only; excludes code, tables, captions) | **1,538** | **4,418** |
| Sentences / avg words per sentence | 65 / **18.4** | 202 / **21.9** |
| Paragraphs | 30 | 73 |
| Code blocks (lines each) | **5** (6, 13, 13, 18, 17) | **6** (21, 32, 8, 1, 17, 8) |
| Bullet lists / list items | **9 / 24** | **0 / 0** |
| Tables / total rows | **3 / 23** (2×3-row micro tables + 1×17-row results table) | **0** |
| Figures (unique images, excl. logo + social card) | **5** | **8** |
| First-person density | "we" ×8, "our" ×2 | "we" ×**87**, "our" ×**42** |
| Voice | first-person-plural but restrained, product-team register | narrative lab-notebook register, "we tried / we were surprised" |

Local artifacts from this session (regenerate-able):
`/tmp/tpu.md`, `/tmp/paged.md` (markdown conversions), `/tmp/blogimg/*.png` (figures),
`/home/dev/local/pytorch-blog-heuristics/research/_img/crop_{left,right}.png` (zoomed chart reads).

---

## 1. Heading sequence (exact, in document order)

### Post A — Helion on TPU
```
H3  TL;DR                                   (81 words, one paragraph, no bullets)
H2  Introduction                            (188 words + 3-bullet "three use cases" list)
H2  TPU Primer                              (161 words + 3-row TPU-vs-GPU table + memory-hierarchy diagram)
H2  Helion's Pallas Codegen                 (91 words + 3-bullet "three-fold strategy" list)
H3    Example: add                          (154 words + 2 code blocks + 3 explainer bullets)
H3    (image-only heading: add pipeline diagram)
H3    Example: Flash Attention              (617 words + 3 code blocks + 2 pipeline diagrams + 3-row table + TFLOPs line chart)
H2  Broader Kernel Benchmarks               (95 words + 17-row results table)
H2  (image-only heading: speedup bar chart)
H2  What's Next                             (42 words, 4 bullets)
H2  Getting Started                         (53 words, 4 resource links)
H2  Acknowledgements                        (56 words)
```
Notes: there are literally two **empty H2/H3 headings whose only content is an `<img>`** — the
figures are hung off headings rather than captioned. Post A has **no figure captions at all**
(every `alt` is empty except the social card).

### Post B — Portable Paged Attention
```
(no TL;DR, no intro heading — 131-word untitled lede that states the challenge + links the vLLM PR)
H2  Brief Background to vLLM, Triton, and Helion          (328 words, no code)
H2  Implementation Details: How to write Paged Attention in Tiled PyTorch   (0 words — pure container)
H3    Launch Grid and approach of parallelization         (593 words, 53 code lines, Figure 1)
H3    Tiling in Helion                                   (438 words, 8 code lines, Figure 2)
H3    Autotuning                                         (724 words, 18 code lines)
H2  Performance Evaluation                               (99 words — methodology preamble)
H3    Micro-Benchmarks                                   (727 words, Figures 3-6)
H3    End-to-End in vLLM                                 (1,313 words, Figures 7-8)
p>b "Lessons Learned and Conclusion"                     (bold paragraph, NOT a heading — 610 words)
H2  Acknowledgments                                      (65 words)
```
Notes: Post B captions **every** figure as a paragraph immediately below the image, format
`Figure N: <one-line what> <two-to-three sentences of setup>.` "Lessons Learned and Conclusion"
is `<p><b>…</b></p>` — an authoring slip; it does not appear in the ToC/heading tree. Post B has
**no bulleted lists anywhere** — everything is running prose.

---

## 2. How they present a technical mechanism inside the compiler

### Post A is the closest available template for a "compiler mechanism" section
Post A's mechanism section is about **codegen strategies the compiler chooses between**; that is
structurally the same shape as "a heuristic that statically picks a config." Its recipe:

1. **One-sentence objective statement up front**, then an N-way bullet taxonomy of the strategy:
   > "To extract maximum performance out of a TPU, Helion's Pallas codegen aims to maximize
   > **software pipelining**, ensuring that memory transfers and computation overlap as much as
   > possible. This section illustrates Helion's three-fold strategy for generating pipelined kernels:"
   followed by a 3-item nested bullet list (outer loop / inner loop autotuned between two options /
   auto-tuned buffer sizes). **No numbered "Stage 1…Stage N" walkthrough anywhere** — the numbered-
   stage function is served by (a) short bullet taxonomies and (b) parenthetical labels.
2. **Escalating worked examples**: a trivial kernel first (`add`, 6-line Helion source), then the
   real one (`attention`). The trivial example exists only to teach the reader to read the
   generated-code diff.
3. **Source → generated-code pairs.** Every mechanism claim is anchored to a code block. Both
   sides are shown: the Helion kernel the user writes, then "The Helion compiler translates this
   into two functions…". Generated code is **abridged pseudo-code, not real emitted output**:
   `_BLOCK_SIZE_0 = <autotuner-selected value>`, `<qk matmul, online softmax, v matmul, update acc>`,
   `out = torch.empty(...)`, `launcher( # wraps around pallas_call`. Blocks are 6-18 lines. Nothing
   longer than 18 lines appears in the entire post.
4. **Post-code explainer bullets**: after each block, "Within the generated code:" + 3 bullets, one
   per moving part. For the nested case they use **parenthetical stage labels instead of numbers**:
   "- (Outer pipeline) The host uses `pallas_call` to invoke `_helion_attention`…" /
   "- (Inner Pipeline) Within `_helion_attention`, the device uses `emit_pipeline`…"
5. **Diagram-as-timeline**: each strategy gets a hand-drawn Google-Drawings-style figure with two
   rows, `Memory` and `Compute`, of rounded boxes on a timeline. The `add` figure shows
   `Load tile 0..3` staggered above `Add tile 0..3`. The `emit_pipeline` figure shows
   `Q0 | KV0 | KV1 | KV2 | Q1 | KV0 | KV1 | KV2` on the Memory row and gaps on the Compute row; the
   `unroll` figure shows `Entire KV | Q0 | … | Q1` and a **gapless** Compute row. The *visual
   difference between the two figures is the whole argument*; the prose just names it ("bubbles").
   Diagrams are simple boxes-and-arrows, greyscale + two accent fills, no axes, no data.
6. **Name the config knob explicitly.** The post surfaces the literal autotuner key:
   "This is keyed on the `pallas_loop_type` autotuner config" and then discusses
   `pallas_loop_type == emit_pipeline` vs `== unroll` by name.
7. **State the mechanism's cost/limit immediately after showing the win** (see §3).

**Length ratio, measured (Post A):** mechanism (`Helion's Pallas Codegen` + both examples) =
**862 prose words, 67 code lines, 3 diagrams, 1 micro-table, 1 line chart**; results
(`Broader Kernel Benchmarks`) = **95 prose words, 1 table (16 kernels), 1 bar chart**. That is
≈ **9 : 1 mechanism-to-results in prose words**, with the results section carrying its weight in a
table + chart and only two sentences of interpretation. The `Example: Flash Attention` subsection
alone (617 words) is **40% of the entire post's prose**.

### Post B's "mechanism" is kernel-authoring, not compiler internals
Post B explains *how the author wrote the kernel and fought the DSL*, never how Helion's compiler
works internally. Its mechanism devices worth stealing:
- **A "concept figure" before any code** (Figure 1: a 3-axis isometric box diagram labelling
  `query_length` / `num_query_heads` / `sequence_length` with the tunables `TILE_Q=tuneable`,
  `TILE_M = Q heads per KV head`, `TILE_N=tuneable` drawn as arrows on the axes). The tunables are
  drawn *onto the problem geometry*.
- **A concrete failing example as a figure** (Figure 2: queries of length 7, 2, 1 packed into a
  varlen tensor, tile size 4, showing tokens of two requests colliding in one tile) — then the
  fix in 8 lines of real code (`hl.load(..., extra_mask=q_load_mask)`).
- **Real generated Triton, with provenance comments kept in**:
  `# src[helion_unified_attention.py:129]: for seq_tile, tile_m in hl.tile(` … then
  `for offset_9 in tl.range(0, v_0.to(tl.int64), _BLOCK_SIZE_0, loop_unroll_factor=2, num_stages=2, disallow_acc_multi_buffer=False, flatten=False):`
  followed by one sentence of interpretation ("this uses the pid_type='flat' version of program
  launch, as determined by the Helion autotuner").
- **Dumping a full `helion.Config(...)` verbatim** (17 lines: `block_sizes`, `indexing`,
  `l2_groupings`, `load_eviction_policies`, `loop_orders`, `num_stages`, `num_warps`, `pid_type`,
  `range_flattens`, `range_multi_buffers`, `range_num_stages`, `range_unroll_factors`,
  `range_warp_specializes`) purely to make the size of the search space visceral, then explicitly
  declining to explain it: "A detailed discussion of all the knobs is beyond the scope of this blog post."
- **A line-count comparison as the developer-experience metric**: "133 lines of code (vLLM
  formatting with comments) vs. 295 in Triton", with a deep link to the PR diff.

**Length ratio, measured (Post B):** implementation/mechanism = **1,755 words + 79 code lines +
2 figures**; performance evaluation = **2,139 words + 6 figures + 0 tables**; lessons/conclusion =
**610 words**. Roughly **1 : 1.2 mechanism-to-results**, i.e. the opposite balance from Post A.

---

## 3. Performance tables, geomeans, normalization, arm labels

### Post A — three distinct result presentations, each with a different baseline
**(a) Mechanism micro-table (3 rows), absolute TFLOPs, no baseline at all.** Used to justify one
compiler decision, placed *inside* the mechanism section:

| | S = 8k | S = 32k |
|---|---|---|
| emit_pipeline TFLOPs | 653 | 695 |
| unroll TFLOPs | 892 | **OOM** |

Prose framing: "The performance difference between these strategies is significant — the table
below shows results on workloads with B=8, H=32, D=256". `OOM` is printed as a cell value; that
one cell carries the entire "why we still need the fallback" argument.

**(b) Absolute-throughput line chart vs. handwritten baselines.** Chart title band reads
"Forward pass, non-causal, B=8, H=32, D=256, bfloat16"; y = "Speed (TFLOPs/s)", x = "Sequence
Length" (512…32768, categorical). Four arms, each labelled in the legend:
`Helion` / `Pallas reference (JAX Repo)` / `Tokamax` / `Pure JAX (no Pallas)`. **Every point is
data-labelled with 2-decimal TFLOPs.** Values I read off the figure:

| S | Helion | Pallas reference | Tokamax | Pure JAX |
|---|---|---|---|---|
| 512 | 306.99 | 227.17 | 162.26 | 122.79 |
| 1024 | 571.95 | 482.31 | 311.92 | 161.74 |
| 2048 | 746.16 | 505.16 | 476.25 | 185.05 |
| 4096 | 830.74 | 546.17 | 636.28 | 195.87 |
| 8192 | **892.01** | 573.10 | 692.17 | 201.37 |
| 16384 | 696.72 | 575.85 | **726.29** | 203.58 |
| 32768 | 695.16 | 578.86 | **747.55** | 205.13 |

The Helion series is the **autotuned best-of-both-strategies** (892.01 at 8k == the `unroll` row;
695.16 at 32k == the `emit_pipeline` row), which is exactly the "the compiler picks per shape"
story rendered as one line.

**(c) Whole-suite table normalized to BOTH eager and torch.compile, plus a bar chart normalized
only to eager.** Columns, verbatim: `kernel | shape | torch_tpu eager (ms) | torch.compile(tpu)
(ms) | Helion (ms) | Helion vs torch_tpu eager | Helion vs torch.compile`. 16 kernel rows.
Absolute milliseconds are given for all three arms *and* two ratio columns; shapes are printed as
raw lists (`[8,32,8192,256]`, `[65536,2560]`). Ratios are formatted with the multiplication sign
`4.45×` (a few rows slip to `2.71x` — inconsistent glyph). **Rows are sorted descending by the
LAST column (Helion vs torch.compile)**, i.e. best-vs-compile first, so the four losing rows
(0.82×, 0.80×, 0.69×, 0.68×) sit at the bottom of the table and are not hidden.
Geomean claim: "Helion shows a geometric average speed-up of 1.55x compared to eager, and 1.12x
compared to compiled." **I recomputed both from the table's own millisecond columns: 1.5474 →
1.55x and 1.1158 → 1.12x. Both check out** (n=16, unweighted geometric mean of per-row ratios).
The bar chart (`Helion speedup on TPUv7`, y = "Speedup vs torch_tpu eager", series
`torch_tpu eager` / `torch.compile(tpu)` / `Helion (Pallas backend)`) plots the *same* data but is
**sorted descending by Helion-vs-eager** and annotates only the Helion bar (`4.45×`, `2.71×`, …),
with a dashed 1.0 reference line. Baseline arm is drawn as a literal 1.0 bar rather than omitted.

Two-sentence interpretation is all they give: "Helion shows the largest gains on kernels that
employ fusion or optimization patterns that are difficult for XLA to discover automatically —
flash attention is a prominent example. For the more standard operations like `matmul` and
`layer_norm`, XLA's compiler already produces high-quality code, and Helion performs comparably."

### Post B — zero tables; everything is normalized-to-Triton figures plus prose ranges
- **Normalization rule is stated explicitly before the first plot**: "Since we focus on the
  comparison between Triton and Helion kernels, we normalized all data with the leftmost Triton
  result in each plot." (Micro-benchmarks normalize to *one anchor point*, not per-x — so the
  curves show absolute scaling, and only the ratio between curves is meaningful.)
- **The measurement contract is stated in the same paragraph**: "Please note that all measurements
  are done with CUDA/HIP graphs enabled, so we do not evaluate any software overhead like compile
  time or launch time, just pure kernel performance."
- **Arms are named, and each arm is defined by (shapes mode × autotune effort × config set)**:
  legend labels are `triton_unified_2d (vllm)`, `helion_unified (dynamic shapes & select. conf.)`,
  `helion_unified (static shapes & full tun.)`; end-to-end legend is `triton_attn`,
  `helion_attn (static shapes)`, `helion_attn (dynamic shapes)`. The post spends a full paragraph
  justifying *why only two of four cells of the (shapes × tuning) matrix are plotted*: "we
  selected the both 'extremes': First, 'fewer' optimizations per request … And second the scenario
  with the most possible optimizations".
- **Axis labels carry the polarity**: "normalized latency in ms (lower is better)",
  "normalized throughput, higher is better (triton == 1.0)", "normalized latency, lower is better
  (triton == 1.0), log scale".
- **The same data is plotted twice under different sort keys** ("in one plot sorted by the share of
  decode requests within a batch, and in the other by maximum sequence length"), with facets
  `decode_share 0.0 / 0.5 / 1.0` and x = "number of tokens" on log-log.
- **No geomean anywhere.** Aggregate performance is reported as prose min–max ranges per regime:
  "the performance of the Helion kernel ranges between 29% and 137% vs Triton for prefill
  requests, and between 132% and 153% for decode-only requests" (H100); "between 13% and 75% for
  prefill requests and between 58% and 107% for decode-only batches" (MI300X).
- **End-to-end is reported as three scalar metrics** (total token throughput, median TTFT, median
  ITL), all normalized with triton == 1.0, and the losing arm is stated first: "Helion with static
  shapes achieves only roughly 26% of Triton's total throughput, while Helion with dynamic shapes
  achieves 96% of the total token throughput".
- Benchmark reproducibility is a code block of the actual commands
  (`VLLM_ATTENTION_BACKEND=EXPERIMENTAL_HELION_ATTN vllm serve …` / `vllm bench serve … --dataset-name sharegpt --ignore_eos`),
  plus versions pinned in prose: "We used Helion 0.2.4, Triton 3.5.0 and PyTorch 2.9 for all experiments."
- Fairness caveats are given for the *baseline* arm too: "the Triton attention backend in vLLM
  today does not do 'live' tuning and instead selects between four different configurations with
  if-else statements", and "To be fair for both implementations, we always do two warmup benchmark
  runs".

---

## 4. Limitations and future work

**Post A** — limitations are *inline with the mechanism*, then a terse forward-looking list:
- The cost of the fast strategy is stated in the same paragraph as its win: "The trade-off is that
  this requires more VMEM usage, as the entire K and V sequences need to be present. This means
  that although more performant, this translation isn't always possible. The VMEM usage is linear
  with respect to the input sequence lengths (as opposed to tile size), which is prohibitive with
  longer sequences."
- Losses are left visible in the table (4 of 16 kernels are < 1.0× vs torch.compile) and
  explained in one sentence by attributing them to the baseline being already good ("XLA's
  compiler already produces high-quality code").
- `What's Next` is **42 words, 4 bullets, no dates, no numbers**: expand kernel coverage, further
  performance improvements, better support for jagged and sparse operations, distributed TPU.
- Availability caveat is honest and specific: the TPU backend "has a dependency on TorchTPU, which
  is expected to be released publicly later this year."

**Post B** — limitations are a full ~610-word "Lessons Learned and Conclusion" plus caveats
embedded in the results:
- Root-causes its own gap architecturally: "The gap in the prefill can be explained by the smaller
  launch grid of the Helion kernel vs the Triton kernel. … Hence, optimizing the launch grid of the
  Helion kernel is another optimization avenue."
- Names a *product-level* limitation of the tool: "one disadvantage for the current version of
  Helion is that it expects one configuration for the complete kernel and we cannot differentiate
  between configurations that would be beneficial for a prefill batch or decode batch, as we can
  do in Triton."
- Publishes the autotuning cost as a cost: quick mode "still required 10 hours to tune our kernel
  for 72 different scenarios … Which was faster than the 25 hours it takes with the 'full' (or
  default) setting, but maybe not as 'quick' as we would have wished." Also quotes the tuner banner
  verbatim: "[2752s] Autotuning complete in 2752.1s after searching 5353 configs."
- Discloses an experiment defect that could bias a number, and bounds it: "due to unrecoverable
  crashes of the generated Triton code during 'live' autotuning, we had to disable it for the
  end-to-end experiments with static shapes … This limitation of the experiments could explain a
  small part of the gap in TTFT between static and dynamic shapes, but not the gap in ITL nor the
  big difference in throughput."
- Contradicts the tool's own documented advice from evidence: "Static shapes are usually mentioned
  as performance optimization in Helion, but for highly dynamic usage scenarios like vLLM they are not."
- Flags a bug they hit and its fix landing upstream (`autotune_baseline_fn=callable()`), framed
  generously: "Yes, Helion is still only beta and in active development and this issue was later
  fixed by allowing the user to define an external function as autotune baseline".
- Future work is *concrete asks*, and the last one is literally this blog's subject:
  "Finally, we would like to add some pre-trained heuristics or decision trees to Helion, to have a
  middle ground between hour-long autotuning and just one configuration for all cases in
  low-latency scenarios such as vLLM." Also earlier: "Our experience with autotuning the Triton
  kernels taught us that it is a good trade-off to tune for a broad range of scenarios (in a
  microbenchmark setting) and then select only a handful of configurations to be used in vLLM and
  use decision trees or other heuristics to select between them."
- Links open GitHub issues instead of hand-waving (helion#1286, helion#1242, helion#1249).
- Includes a tooling tip as a takeaway: "The single most-useful 'Helion command' turned out to be
  `tensor.view`, to understand early if the Helion compiler considers the shape of tensors to be
  the same as we expect."
- Sizes the effort: "Overall we spent less than three weeks on the experiments described in this
  blog post and are quite surprised by the impressive results."

---

## 5. Hardware-specific resource reasoning and vocabulary level

Measured vocabulary counts over the body text of each post:

| term | Post A (TPU) | Post B (paged) |
|---|---|---|
| VMEM | 33 (incl. code identifiers) | 0 |
| pipeline / pipelined / pipelining | 34 | 1 |
| bubble(s) | 3 | 0 |
| TFLOPs | 4 | 0 |
| MFU | 1 | 0 |
| tensor core | 3 | 0 |
| MXU / VREG | 1 / 1 (figure only) | 0 |
| OOM | 1 | 0 |
| **occupancy** | **0** | **0** |
| **shared memory / SMEM** | **0** | **0** |
| **register pressure / spill** | **0** | **0** |
| **bank conflict** | **0** | **0** |
| **TMEM** | **0** | 0 ("tensor memory allocation" ×1, meaning Helion's allocation, not SM100 TMEM) |
| warps | 0 | 3 (only as the config key `num_warps` / "number of warps") |
| L2 | 0 | 1 (only inside the dumped `l2_groupings=[2]`) |

**Pitch level: one level above the ISA, and only for resources the argument needs.**
- Post A confines hardware reasoning to a **capacity/overlap** model: a 5-box memory-hierarchy
  diagram (`HBM (Large, off-chip)` → `VMEM (Fast, on-chip)` → `VREGs (Fastest, on-chip)` →
  `MXU` + `Vector Compute`), a 3-row TPU-vs-GPU contrast table (rows: `Threading`,
  `Memory Hierarchy`; cells like "Sequential Few large workers" vs "Parallel SIMT ( + tensor core)
  Many small workers", "Explicit memory spaces (persistent vs scratchpad memory), Async mem copies
  required for pipelining." vs "Implicit caches, HW-managed"), and then exactly one quantitative
  resource claim: VMEM usage is **linear in sequence length, not tile size**, therefore `unroll`
  OOMs. No bytes, no occupancy tables, no register counts.
- Post A's only efficiency-normalized number is in the TL;DR: "838 TFLOPs (~79% MFU of one tensor
  core) on TPU v7" — note the denominator is **one tensor core**, not the chip, and the peak used
  is never given.
- Post A grounds the cross-vendor comparison in exactly two metrics and says so: "TPU7x and NVIDIA
  B200 have very similar BF16 compute TFLOPS and HBM bandwidth — the two most important hardware
  metrics for modern ML workloads."
- Post B does **no** hardware-resource reasoning. Its resource story is *shape/space* reasoning:
  KV pages must be loaded whole ("we always have to load full pages from the KV cache, since we
  cannot determine at compile-time if we need the complete page or not"), one tile size per
  compiled kernel ("remember, it is a compile time constant"), padding tile upper bounds to
  powers of two to keep JIT/autotune counts finite, and shape-constrainedness explaining a null
  result ("the size of the matrix multiplications always need to align with the KV cache page size
  (or block size) of vLLM").

**Implication for the heuristics post:** SM100-specific vocabulary (TMEM columns, SMEM budget,
register/spill, cluster dims, `tcgen05`) has **no precedent** in either post. If the heuristic's
reasoning is genuinely occupancy/SMEM/TMEM-driven, expect to have to introduce each term with a
one-clause gloss the way Post A glosses VMEM ("Fast, on-chip"), and to state at most one
quantitative resource law per mechanism (Post A's model: "VMEM usage is linear with respect to the
input sequence lengths (as opposed to tile size)").

---

## 6. Acknowledgements format (verbatim)

Post A, heading `Acknowledgements` (British spelling), 56 words, single paragraph, individuals
named with affiliation grouped by company, no roles:
> "This project was made possible through the invaluable collaboration and technical insights of
> our peers. A special thank you to Joe Pamer, Robert Hundt, Claudio Basile and Adam Paszke at
> Google, as well as Jana van Greunen, Gregory Chanan, Peng Wu, and Zongwei Zhou at Meta, for their
> feedback and support in bringing this to fruition."

Post B, heading `Acknowledgments` (US spelling), 65 words, single paragraph, thanks own team then
the upstream project team, and ends with a link as evidence of responsiveness:
> "This work was supported by the AI platform team at IBM Research, in particular we would like to
> thank our colleagues: Thomas Parnell, Jan van Lunteren, Mudhakar Srivatsa, and Raghu Ganti. Also,
> we would like to thank Jason Ansel and the Helion team at Meta for their feedback and support,
> especially the fast fixing of the bugs we reported, sometimes even within 24 hours." (the last
> phrase hyperlinks to helion issue #1249)

Post A additionally has a `Getting Started` section immediately before acknowledgements: 53 words,
availability caveat, then 4 bare links (GitHub repo, docs, TPU examples, performance dashboard).
Post B has no such section; it links the vLLM PR (#27293) inline in the lede and again in the
end-to-end section.

---

## 7. Verbatim quotes to model on

### Post A — Helion on TPU (7 quotes)
1. "Helion is PyTorch's high-level DSL for writing performance-portable ML kernels. Partnering with
   Google, we have built a TPU backend that compiles Helion kernels to Pallas, providing a
   PyTorch-friendly way to author performant TPU kernels."
2. "On different input shapes, Helion autotunes over different code-generation strategies to select
   the optimal pipelining schema, making the most use of TPU's available VMEM and compute."
3. "To extract maximum performance out of a TPU, Helion's Pallas codegen aims to maximize
   **software pipelining**, ensuring that memory transfers and computation overlap as much as
   possible. This section illustrates Helion's three-fold strategy for generating pipelined kernels:"
4. "`_BLOCK_SIZE_0` (the tile/buffer size) is selected by the autotuner, which explores different
   sizes to find the best overlap between memory transfers and compute for the target hardware."
5. "One obvious point of inefficiency in this pipeline is that there are bubbles in the compute
   units – for every new Q tile, while we fetch the 0th KV tile, there is no work available for the
   compute units. This comes down to the fact that we are re-loading the KV tiles from HBM to VMEM
   for every new Q tile."
6. "The trade-off is that this requires more VMEM usage, as the entire K and V sequences need to be
   present. This means that although more performant, this translation isn't always possible."
7. "The benefit of Helion lies in its ability to autotune and select the best autotuner config. So
   that with smaller sequences, it makes use of the VMEM available and generates pipelined code
   with no compute bubbles. For longer sequences, it falls back to `emit_pipeline` which scales to
   arbitrary context lengths."
8. "Helion shows the largest gains on kernels that employ fusion or optimization patterns that are
   difficult for XLA to discover automatically — flash attention is a prominent example. For the
   more standard operations like `matmul` and `layer_norm`, XLA's compiler already produces
   high-quality code, and Helion performs comparably."

### Post B — Portable Paged Attention (8 quotes)
1. "To test this promise (and learn Helion), we embarked on the challenge to write one of AI's most
   performance-critical kernels in Helion: Paged Attention, the core of vLLM."
2. "In contrast to Triton, Helion's autotuner has not only a usable caching mechanism, but the
   autotuner also has a lot more degrees of freedom. This larger freedom comes from the fact that
   in Helion, the autotuner can also change algorithmic aspects of an implementation, in addition
   to lower-level compile flags like the number of warps or pipeline depths."
3. "Overall, this kernel implementation in Helion requires 133 lines of code (vLLM formatting with
   comments) vs. 295 in Triton."
4. "But as usual, terms like 'quick' and 'slow' are relative, and the quick mode of Helions
   autotuner still required 10 hours to tune our kernel for 72 different scenarios (like batch
   size, sequence lengths, head size, etc.)."
5. "Since we focus on the comparison between Triton and Helion kernels, we normalized all data with
   the leftmost Triton result in each plot."
6. "Our data shows that the performance of the Helion kernel ranges between 29% and 137% vs Triton
   for prefill requests, and between 132% and 153% for decode-only requests."
7. "As can be seen in the Figure, Helion with static shapes achieves only roughly 26% of Triton's
   total throughput, while Helion with dynamic shapes achieves 96% of the total token throughput
   and is also on par in TTFT (Time-to-first-token, i.e. the prefill time) and very close in ITL
   (Inter-token-latency, i.e. the time to decode one more token)."
8. "This experiment highlighted one important reality of inference servers: Request shapes are
   diverse, plentiful, and not known in advance."
9. "Finally, we would like to add some pre-trained heuristics or decision trees to Helion, to have
   a middle ground between hour-long autotuning and just one configuration for all cases in
   low-latency scenarios such as vLLM."

---

## 8. Defects / inconsistencies in the source posts (things NOT to copy)

1. **Post A's headline TFLOPs number does not appear anywhere else in the post.** TL;DR says "838
   TFLOPs (~79% MFU of one tensor core)". The in-post `unroll` micro-table says **892** at S=8k;
   the line chart's Helion peak is **892.01** at S=8192; and the 16-kernel table's attention row
   (`[8,32,8192,256]`, 19.72 ms) back-computes to **892 TFLOPs** using non-causal fwd
   `4·B·H·S²·D = 1.759e13` FLOPs / 19.72 ms. **838 is unreconciled with every other number in the
   post.** Lesson: make the TL;DR number literally equal to a cell in a table below it.
2. **Post A's prose overclaims relative to its own chart.** "The autotuner's ability codegen
   different loop and pipelining strategies depending on the input length is what gives Helion its
   edge even when compared to highly optimized implementations such as Tokamax." But the chart it
   sits under shows **Tokamax ahead of Helion at S=16384 (726.29 vs 696.72) and S=32768 (747.55 vs
   695.16)** — a crossover the prose never mentions. Lesson: name your crossovers before a reader
   finds them in your own figure.
3. **Post A has ratio-glyph and rounding inconsistency** in the results table (`4.45×` vs
   `2.71x`; `matmul_layernorm` prints `1.10×` where the millisecond columns give 1.094, and the
   bar chart for the same cell prints `1.09×`).
4. **Post A's figures have no captions and empty `alt` text**, and two figures are attached to
   otherwise-empty headings — an accessibility and referenceability miss. Post B's
   `Figure N: <claim> <setup>` captions are strictly better and are what to copy.
5. **Post B's "Lessons Learned and Conclusion" is `<p><b>…</b></p>`, not a heading**, so the
   post's most important section is missing from the heading tree.
6. **Post B has a broken link** in the Launch Grid section: the "here" URL is
   `…/enabling-vllm-v1-on-amd-gpus-with-triton/)%20and%20we%20selected%20this%20kernel%20version,%20sin`
   — a sentence fragment got URL-encoded into the href.
7. **Post B's normalization anchor is a single leftmost point**, not per-x normalization, which
   makes the y-axis a mix of "scaling with tokens" and "ratio to Triton"; the reader must do the
   ratio by eye between two curves. Post A's explicit two-ratio-column table is easier to audit.

---

## 9. Direct implications for the heuristics blog

Structure that follows Post A most closely (it is the compiler-mechanism precedent):

- **TL;DR of ~80 words** that names the mechanism, the headline number, and the "without
  autotuning" claim — and whose number is copied from a table below.
- **A short primer section** (≈160 words + 1 diagram + 1 small contrast table) for whatever the
  reader must know before the mechanism. Post A's "TPU Primer" is the slot; the analogue is
  "what a Helion config is / what autotuning costs".
- **Mechanism section ≈ 55-60% of prose**, structured as: objective sentence → N-way bullet
  taxonomy of the strategy → trivial worked example → real worked example (multi-matmul / linear
  attention) → per-part explainer bullets with parenthetical stage labels → cost/limit paragraph.
  Precedent says **no numbered stage lists**; if the heuristic really is a pipeline of stages, the
  established device is `- (Stage name) …` bullets, ≤4 of them.
- **Code blocks 6-18 lines, abridged with `<placeholder>` and `...`.** Post A never shows real
  emitted code; Post B shows real emitted Triton but only 32 lines with `# src[...]` provenance
  comments and heavy `...`. Either is precedented; nothing longer than ~32 lines is.
- **Put one micro-table inside the mechanism section** to justify the key decision
  (Post A: `emit_pipeline` vs `unroll` × S=8k/32k, with `OOM` as a printed cell). For a heuristics
  post the analogue is *heuristic config vs the config it rejected*, on 2 shapes.
- **Results section can be tiny in prose (Post A: 95 words) if the table + chart carry it**, but
  give absolute times for every arm *plus* ratio columns, print the shape for every row, sort
  descending by the ratio you care most about, and leave the losing rows in.
- **Arm labelling must be self-describing in the legend, Post-B-style**: Post B's
  `helion_unified (static shapes & full tun.)` encodes shapes-mode × tuning-effort in the label
  itself, and the axis label carries the polarity and the baseline (`(triton == 1.0)`,
  `lower is better`). For this blog that means labels that cannot be confused, e.g.
  `heuristic (no autotuning)`, `AOT / pre-tuned config (tuning time not counted)`,
  `pre-change compiler default`, `raw unseeded default`, `torch.compile (default mode)`,
  `vLLM shipped config, same Helion kernel body`. Post B also devotes a paragraph to *why* only
  some cells of the arm matrix are plotted — copy that.
- **Both posts state the measurement contract in the results preamble** (Post B: CUDA/HIP graphs
  on, so no compile/launch overhead is measured; versions pinned as "Helion 0.2.4, Triton 3.5.0
  and PyTorch 2.9"). Do the same, and additionally state that tuning time is excluded from the
  AOT arm, since that is the load-bearing asymmetry in a heuristics story.
- **Geomean is precedented (Post A) but must be re-derivable from the printed table** — Post A's
  1.55x/1.12x reproduce exactly from its own millisecond columns. Post B prints no geomean and
  uses min–max ranges per regime instead; ranges are the honest option when the distribution is
  bimodal (prefill vs decode), which is worth remembering for reduction/pointwise families.
- **Limitations belong inline with the mechanism** (Post A puts the VMEM-linear-in-S limit in the
  same paragraph as the win) **and again in a short list at the end** (`What's Next`: 42 words, 4
  bullets, no dates). Post B's long lessons section is the alternative if the post has an
  experiential arc.
- **Post B is a ready-made external endorsement of this blog's thesis** — its final future-work ask
  is verbatim "pre-trained heuristics or decision trees … a middle ground between hour-long
  autotuning and just one configuration for all cases in low-latency scenarios such as vLLM", and
  its measured autotune cost is "10 hours … for 72 different scenarios" (quick mode) vs 25 hours
  (full). That is the strongest available quotable motivation for a compile-time heuristic, from a
  third party, in a PyTorch-blog voice.
