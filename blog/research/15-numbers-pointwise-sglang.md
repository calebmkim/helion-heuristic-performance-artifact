# Research notes: POINTWISE heuristic + SGLang KDA four-arm head-to-head

Author: research subagent. Date of analysis: 2026-09-02.
Everything below is re-derived from raw JSON (`summary.json`, `results.json`, per-cell JSON)
and from the heuristic/kernel source, not from prose summaries. Where I derived something
that no checked-in report states, it is explicitly labelled **[derived here]**.

---

# PART A — POINTWISE HEURISTIC (36-cell B200 audit)

## A.0 Provenance and exact file paths

| What | Path |
|---|---|
| Human report | `/home/dev/local/wt-b200-pointwise-audit/perf-repro/b200-pointwise-audit/results/full/SUMMARY.md` |
| Raw aggregate | `.../results/full/summary.json` (495 KB, 36 `rows`) |
| One JSON per cell | `.../results/full/raw/{cohort}__{kernel}__{NN}.json` |
| Per-cell worker logs | `.../results/full/logs/` |
| Plan (arm definitions) | `.../b200-pointwise-audit/PLAN.md` |
| Harness | `.../b200-pointwise-audit/bench.py`, `run_all.py`, `aggregate.py` |
| Charts | `.../results/charts/{all_cohorts,general,vllm,sglang}_relative_performance.png` |

Environment (from `summary.json:environment`, identical on every cell,
`environment_consistent: true`):

- Helion `61f4058f3610e1b2bbabc82df8267ac450f591df`, branch `b200-pointwise-audit-run`
  (this is a **different worktree/commit** from the linear-attention/SGLang tree
  `wt-sm100-linattn` @ `375363d8663af60d71a0869003db7954f22c42ca`)
- PyTorch `2.12.0+cu132` (`7661cd9c6b841b62b7f411aa52ec51f05457263b`), Triton `3.7.0`
- CUDA runtime 13.2, driver 595.71.05
- **NVIDIA B200, physical GPU 1** (`CUDA_VISIBLE_DEVICES=1`), cc `(10,0)`, 191.5 GB
- 148 SMs (confirmed independently from the SGLang `results.json:gpu.sm_count`, same box family)

## A.1 Arm definitions (verified against `bench.py`, not just PLAN.md)

| Arm | Exact definition | Verified at |
|---|---|---|
| `default` | `config_spec._base_default_config()` — the **raw unseeded compiler default**, no heuristic of any kind | `bench.py:236` |
| `seed` | `config_spec.compiler_seed_configs[0]` — the config the `triton_pointwise` heuristic emits. Because `promote_seed_to_default=True` with `PROMOTE_TARGETS = (("cuda","sm90"),("cuda","sm100"))`, **this is what a user actually gets on B200 with no autotuning** | `bench.py:229-239` |
| `torch_compile` | `torch.compile(case.ref_fn)` — **default mode, not max-autotune**, on the eager torch reference (not on the Helion body) | `bench.py:690` |
| `aot` | explicit replay of a checked-in pretuned config table (see A.6 for provenance per kernel) | `bench.py:_select_aot` |
| `default_null` | independently compiled duplicate of `default`; **timing-noise control only**, never a result | `bench.py:654-655` |

- Reported metric: `default_latency / arm_latency`. >1 = faster than the unseeded default.
- `HELION_AUTOTUNE_EFFORT=none` is set (`bench.py:35`). **No online autotuning anywhere in this experiment.**
- Timing: "calibrated cold-L2 CUDA graph".
  `(flush_plus_operation_graph_ms - flush_only_graph_ms) / batch`, flush **inside** the timed
  graph, batch=16, 20 interleaved iterations/round, median of 9 round medians, escalate to 15
  rounds if any arm's spread > 5%.
- Accuracy gate runs before timing on every arm; an inaccurate arm is dropped but recorded.

## A.2 The headline table — RE-DERIVED, all values match to 4 decimals

I recomputed every geomean from `summary.json:rows[*].relative_performance_vs_default`.
**All cohort and per-kernel numbers in SUMMARY.md reproduce exactly.**

| Cohort | Measured cells | Heuristic (seed) | torch.compile | AOT |
|---|---:|---:|---:|---:|
| General pointwise | 17 | **19.3508x** [14.284, 40.623] | 18.5082x [14.221, 44.731] | 22.1446x [11.182, 39.213] (n=**5**, RoPE only) |
| vLLM `silu_mul_fp8` | 9 | **1.0960x** [0.958, 1.251] | 1.2316x [0.988, 1.487] | 1.1297x [0.847, 1.388] |
| SGLang `silu_and_mul_interleaved` | 9 | **1.2572x** [1.005, 1.669] | 1.4197x [1.215, 1.758] | 1.0358x [0.807, 1.187] |
| Overall | 35 | 4.5790x | 4.7640x | 2.0852x (n=23) |

Per kernel (all re-derived, all match):

| Kernel | Dtype | Cells recorded / measured | Heuristic | torch.compile | AOT |
|---|---|---|---:|---:|---:|
| `swiglu` | bf16 | 6 / 6 | **18.7405x** | 16.5624x | n/a |
| `geglu` | bf16 | 6 / 6 | **14.7109x** | 14.5386x | n/a |
| `rope` (fwd) | bf16 | 6 / **5** | **27.9430x** | 28.2530x | 22.1446x (sm90 table) |
| `silu_mul_fp8` | bf16→fp8 | 9 / 9 | **1.0960x** | 1.2316x | 1.1297x (sm90 table) |
| `silu_and_mul_interleaved` | bf16 | 9 / 9 | **1.2572x** | 1.4197x | 1.0358x (sm100 table) |

Cell accounting: 36 planned = 6 swiglu + 6 geglu + 6 rope + 9 vllm + 9 sglang.
36 recorded, **35 timed**, 1 timeout. `heuristic_counts: {"triton_pointwise": 35}` and
`heuristic_no_seed: []` — the pointwise heuristic fired on 35/35 measured cells.

## A.3 ★ WHY GENERAL POINTWISE IS ~19x — the honest answer, quantified

The 19.351x is **almost entirely a statement about how bad `_base_default_config()` is**, and
the audit's own numbers prove it two independent ways.

### A.3.1 The configs, verbatim

For **swiglu and geglu** (a flat 1-D `hl.tile(total_elements)` loop over the whole tensor):

| Arm | config |
|---|---|
| unseeded default | `block_sizes=[32]`, `num_warps=4`, `num_stages=1`, `pid_type='flat'`, all-`pointer` indexing |
| heuristic seed | `block_sizes=[2048]` — **and nothing else** (`num_warps`/`num_stages` left at default) |

Identical on all 12 swiglu+geglu cells. The heuristic's whole output is **one number: a 64x
wider block**. The union of keys the pointwise heuristic ever emits across all 35 cells is
`['block_sizes', 'num_warps']` — that's it.

### A.3.2 What `block_sizes=[32]` costs, in hardware terms **[derived here]**

`block_sizes=[32]` on a flat elementwise loop means:

- **Grid = numel/32 = 1.47M – 4.72M programs** (e.g. swiglu (32768,1536) → 1,572,864 programs;
  geglu (4096,36864) → 4,718,592).
- Each program moves **32 elements per operand = 64 bytes of bf16** — *half of one 128-byte
  cache line*, so no operand load can ever fill a full sector run.
- `num_warps=4` = 128 threads for a 32-element tile → **~75% of every CTA's lanes are masked off**.

Achieved HBM bandwidth I computed from the recorded latencies (bytes = 2 reads + 1 write ×
2 B; B200 HBM3e nominal ≈ 8 TB/s):

| Kernel | Shape | Default grid | Seed grid | Bytes | Default | Seed |
|---|---|---:|---:|---:|---:|---:|
| swiglu | (32768,1536) | 1,572,864 | 24,576 | 302 MB | **351 GB/s (4.4% of peak)** | **6,385 GB/s (79.8%)** |
| swiglu | (8192,14336) | 3,670,016 | 57,344 | 705 MB | 351 GB/s (4.4%) | 6,808 GB/s (85.1%) |
| swiglu | (4096,28672) | 3,670,016 | 57,344 | 705 MB | 351 GB/s (4.4%) | 6,812 GB/s (85.2%) |
| geglu | (16384,2880) | 1,474,560 | 23,040 | 283 MB | 354 GB/s (4.4%) | 5,064 GB/s (63.3%) |
| geglu | (4096,36864) | 4,718,592 | 73,728 | 906 MB | 358 GB/s (4.5%) | 5,425 GB/s (67.8%) |

The unseeded default is **pinned at 351–358 GB/s on every single swiglu/geglu shape** —
i.e. 4.4–4.5% of the part. The heuristic lands at 63–85%. **That ratio *is* the 14–19x.**

The heuristic's own docstring says the same thing and is directly quotable:

> "A pointwise kernel (no reduction / matmul / accumulator) is BANDWIDTH-bound, but the compiler
> defaults it to `block_size=32` (~10% of HBM)."
> — `helion/_compiler/autotuner_heuristics/triton.py`, `TritonPointwiseSeedHeuristic`

Caveat worth a footnote: the docstring's "~10% of HBM" was calibrated on H100 (3.35 TB/s).
On B200 the same 351 GB/s is only **4.4%**, so the *same bad default gets worse as HBM gets
faster* — that is genuinely part of why the B200 number is so large.

### A.3.3 The second, independent proof: torch.compile is at 18.508x

On the same 17 cells `torch.compile` (default mode) reaches **18.508x** — statistically the
same place as the heuristic. Per-cell geomean of seed/torch.compile on general pointwise is
**1.0455x**. So the honest framing is:

> On flat elementwise kernels the heuristic does not beat Inductor; it moves Helion's
> zero-autotune default *from ~1/19th of Inductor's throughput to 4.6% ahead of it*.
> The 19x is the size of the hole, not the size of the win over a real competitor.

(This also means "19.351x" and "18.508x" should never be presented as if the heuristic beat
torch.compile by 0.84x-worth of anything. They are both ~19x over the same broken baseline.)

### A.3.4 RoPE is a 27.9x for the *opposite* reason — the heuristic goes SMALLER

Worth including because it stops the story from being "just use bigger blocks":

`rope_fwd` tiles `hl.tile([batch, seq_len])` and each program then handles **all heads ×
head_dim** internally (`slab_numel = 33280` for the 256-head_dim shapes).

| Arm | config (shape (1,32,2048,256)) |
|---|---|
| unseeded default | `block_sizes=[1, 32]`, `num_warps=4` |
| heuristic seed | `block_sizes=[1, 1]`, `num_warps=2` |
| AOT (sm90 table) | `block_sizes=[1, 1]`, `num_warps=16`, `loop_orders=[[1,0]]`, two `tensor_descriptor` slots |

So here the default's 32-wide *outer* tile means **32 × 33,280 = 1.06 M fp32 values of working
set per program**, and the grid collapses to only **64–256 programs on 148 SMs** — under-parallel
*and* massively over-subscribed per program at the same time. Measured default bandwidth:
**115–298 GB/s (1.4–3.7% of peak)**; the heuristic's `[1,1]` reaches **4.68–5.86 TB/s (58–73%)**.

The heuristic gets this right because the same formula has a *register* cap, not only a byte
budget (below).

### A.3.5 The formula reproduces every seed exactly **[derived here]**

I re-ran the heuristic's arithmetic by hand from the recorded `pointwise_fact` for all 35 cells.
Constants (sm100 branch of `TritonPointwiseSeedHeuristic`): `TILE_BYTES_SM100=16384`,
`REGISTER_BYTES=65536`, `MIN_WAVES_SM100=4`, `BLOCK_FLOOR=256`, `num_sm=148`.

```
fetch_bytes   = slab_numel * storage_itemsize * max(1, gather_stride)
budget_target = 16384 // fetch_bytes          # bandwidth: bytes in flight per program
reg_cap       = 65536 // (slab_numel * 4)     # fp32 working set before spill
occ_cap       = total_numel // (148 * 4)      # keep the grid >= 4 waves
target        = max(1, min(budget_target, reg_cap, occ_cap))
```

| Kernel | slab | gstr | budget | reg_cap | occ_cap | target | emitted seed |
|---|---:|---:|---:|---:|---:|---:|---|
| swiglu / geglu (all 12) | 3 | 1 | 2730 | 5461 | 79.7k–255k | 2730 | `[2048]` (pow2 floor) |
| rope 256-dim | 33280 | 1 | **1** | **1** | 3–13 | 1 | `[1,1]` |
| rope (4,8,4096,128) | 4352 | 1 | 3 | 3 | 27 | 1 | `[1,2]` (`inner_floor=pow2_floor(reg_cap)=2`) |
| sglang interleaved, small rows | 5 | **2** | 819 | 3276 | 20–664 | 20–664 | `[1,256]` / `[1,512]` (`BLOCK_FLOOR=256`) |
| sglang interleaved, large rows | 5 | 2 | 819 | 3276 | 1992–510k | 819 | `[1,512]` |
| silu_mul_fp8, ≤128 tokens | 3 | 1 | 2730 | 5461 | 3–442 | 3–442 | `[1,256]` (floor) |
| silu_mul_fp8, ≥256 tokens | 3 | 1 | 2730 | 5461 | 3324–12398 | 2730 | `[1,2048]` |

Three levers, three mechanisms, and each one actually binds on a different kernel family:
**bandwidth budget** binds swiglu/geglu, **register cap** binds RoPE, **occupancy cap +
`BLOCK_FLOOR`** binds the small-token vLLM/SGLang cells, and the **gather-stride** term
(`ELEMS_PER_THREAD_BY_STRIDE`) is what halves the budget for SGLang's stride-2 interleaved
load. That's a good, concrete "the analysis is doing real work" story.

## A.4 vLLM `silu_mul_fp8` = 1.096x and SGLang interleaved = 1.257x — the honest reading

These two cohorts are the *real* comparison, because their base default is not pathological:
they are 2-D tiled kernels whose default `[N, 32]` inner block is already 256 B wide, so the
unseeded default is already within ~10–25% of good.

| Cohort | Heuristic vs default | Heuristic vs torch.compile | Heuristic vs AOT | Cells heuristic loses to default |
|---|---:|---:|---:|---|
| vLLM `silu_mul_fp8` (9) | 1.0960x | **0.8899x** | 0.9702x | 2: `(1,2048)` 0.958x, `(256,7688)` 0.979x |
| SGLang interleaved (9) | 1.2572x | **0.8856x** | **1.2138x** | 0 (min 1.005x) |

**The heuristic is ~11% SLOWER than default-mode torch.compile on both of these cohorts.**
That is not stated anywhere in `PYTORCH_BLOG_HEURISTICS_RESULTS.md` (it is derivable from its
own table: 1.096 vs 1.232, 1.257 vs 1.420). The blog should not claim a torch.compile win for
pointwise except on the flat general kernels.

**Positive that IS quotable:** on the 9 SGLang keys the heuristic (1.2572x) beats SGLang's own
**exact-key, SM100-tuned** table (1.0358x) by **1.2138x**. And the reason is documented in the
harness source:

> "The SM100 table key is `(hidden, (itemsize,), (has_topk_weights,))` and **does not include
> rows**."
> — `perf-repro/b200-pointwise-audit/sglang_kernel.py` docstring

i.e. one shipped config per hidden width has to cover 6 → 98,304 rows, while the heuristic's
`occ_cap` term adapts to the row count per invocation. That is a genuinely strong argument
for compile-time analysis over a static shipped table.

Measurement floor caveat for these two cohorts: absolute latencies are 2.1–23.6 µs (one 643 µs
outlier). Max `default_null_delta` across all 35 cells is **1.32%**; the worst single-arm
round-median spread is **4.97%** (`silu_mul_fp8 (16,11008)`), right at the 5% escalation gate,
and `high_spread: []` means no cell escalated to 15 rounds. So the 9.6% and 25.7% cohort
geomeans are safe, but the two sub-1.0 vLLM cells (0.958x with a 1.15% null delta) are small
enough that they should be called "flat/noise-level", not "a regression".

## A.5 The timed-out RoPE cell

- Cell: `rope`, shape `[2, 32, 2048, 256]` (the largest general shape in the curriculum).
- Raw record is literally only:
  `{"cohort":"general","kernel":"rope","shape_index":3,"shape":[2,32,2048,256],"dtype":"bf16","fatal_error":"timeout after 300s"}`
- Worker log `logs/general__rope__03.log` is **0 bytes**.
- The timeout is a **whole-cell subprocess timeout** (`run_all.py:136`, `--timeout-seconds` default
  300), so **the artifact does not attribute it to any one arm** — it is not recoverable from the
  data which arm hung. Do not claim it was the default arm.
- **[derived here, plausible not proven]** the default arm for this shape would be
  `block_sizes=[2, 32]`, i.e. 64 tiled positions × `slab_numel=33280` ≈ **2.1 M fp32 values of
  per-program working set** — 2x the (1,32,2048,256) cell that *did* complete at 1182 µs. A
  Triton compile blowup on that arm is the natural suspect. State it as a suspicion only.
- Consequence for the tables: `rope` reads "6/6 cells" but **every RoPE geomean is n=5**, and the
  general cohort's `recorded_cells` is 18 while every general geomean is **n=17**. Anyone quoting
  "17 cells" is right; anyone quoting "18" is quoting the recorded count.

## A.6 AOT provenance — three separate traps

| Cohort | AOT table | Arch | Tuned on | Trap |
|---|---|---|---|---|
| RoPE | `pretuned_kernels/rope/_helion_aot_rope_cuda_sm90.py` | **sm90** | H100 | Replayed on B200. "B200 runtime fallback would select it," but it is **not a B200-tuned result** (PLAN.md: "SM90 AOT results must not be described as B200-tuned results.") |
| vLLM `silu_mul_fp8` | `pretuned_kernels/silu_mul_fp8/_helion_aot_silu_mul_fp8_cuda_sm90.py` | **sm90** | H100 | Header says: "Re-tuned on NVIDIA H100 (sm90) with **Helion's** autotuner … seeded from the previous vLLM-converted configs." → this is **Helion's own H100 table, not vLLM's shipped config**. The blog's global rule "vLLM arms run vLLM's shipped config" does **not** apply to this pointwise cell set. There is no SM100 table. |
| SGLang interleaved | `silu_and_mul_interleaved_sm_100.json` from SGLang commit `5f79cf35110d6a0be828f266160b75d83a2a6276` | **sm100** | B200 | Genuinely SM100 and exact-key on all 9 cells — and it still loses to the heuristic (1.036x vs 1.257x) because its key omits the row count. |

### A.6.1 ★ Do not put 19.351x and 22.145x in the same sentence

SUMMARY.md's `general` cohort row reads `Seed 19.351 (n=17) | AOT 22.145 (n=5)`. Those are
**different cell sets** — the AOT column is RoPE-only. On the SAME 5 RoPE cells the heuristic
is **27.943x** and the SM90 AOT table is **22.145x**, i.e. the heuristic is **1.2618x faster
than the AOT table**. The index doc does label the AOT column "(5 RoPE cells)", but the row still
reads as if AOT beat the heuristic. It did not.

### A.6.2 The RoPE AOT number decomposes cleanly **[derived here]**

2 of the 5 RoPE AOT cells are **not exact-key** — the SM90 table has no entry and falls back to
key `[8192, 2048]` (`aot_selector.exact_key=false`, `fallback_key=[8192,2048]`):

| RoPE AOT subset | n | AOT geomean | Heuristic on the same cells |
|---|---:|---:|---:|
| exact SM90 key | 3 | **27.854x** | 28.336x (AOT is 1.7% behind — effectively a tie) |
| fallback key | 2 | **15.698x** | 27.363x (AOT is 43% behind) |
| all 5 (the reported 22.145x) | 5 | 22.145x | 27.943x |

Great one-liner: *a pretuned table ties the heuristic exactly where it has an entry and falls
off a cliff (15.7x vs 27.4x) where it has to guess — the worst cell, (4,8,4096,128), drops to
11.182x.* That is the strongest single argument in the pointwise data for a formula over a table.

## A.7 Charts available for pointwise

- `results/charts/all_cohorts_relative_performance.png` — 3 stacked panels (general / vLLM / SGLang),
  bars = Pointwise seed / torch.compile / Pretuned AOT, dashed line at Default 1.00x. Labels
  read 18.7 / 16.6 (SwiGLU), 14.7 / 14.5 (GEGLU), 27.9 / 28.3 / 22.1 (RoPE), overall 19.4 / 18.5 / 22.1;
  1.10 / 1.23 / 1.13 (vLLM); 1.26 / 1.42 / 1.04 (SGLang). All consistent with the JSON.
- Per-cohort versions: `general_relative_performance.png`, `vllm_relative_performance.png`,
  `sglang_relative_performance.png`.
- **There are no per-shape charts.** All charts are per-kernel geomeans + cohort overall.

---

# PART B — SGLang KDA FOUR-ARM HEAD-TO-HEAD (the "we lose here" datapoint)

## B.0 Provenance

| What | Path |
|---|---|
| Final report | `.../sglang_triton_head_to_head/four_arm_no_heuristics_v2_address_matched/RESULTS.md` |
| Raw aggregate | `.../four_arm_no_heuristics_v2_address_matched/results.json` (845 KB, 36 `records`) |
| Per-cell JSON | `.../four_arm_no_heuristics_v2_address_matched/cells/*.json` (36 files) |
| Campaign report | `.../sglang_triton_head_to_head/REPORT.md` |
| Plan | `/home/dev/local/wt-sm100-linattn/SGLANG_KDA_TRITON_HEAD_TO_HEAD_PLAN.md` |
| Earlier 3-arm runs | `.../current_fixed/`, `.../current_packed/`, `.../pr_v1_4_fixed/` |
| 40-cell config replay | `.../config_recheck/RESULTS.md` |
| Superseded 4-arm v1 | `.../four_arm_no_heuristics/` (kept for auditability; do NOT quote) |
| Chart | `.../sglang_kda_four_arm_speedup_vs_triton.png` |
| Prefill-only chart | `.../sglang_kda_prefill_only_speedup_vs_triton.png` |

- GPU: **physical GPU 0**, NVIDIA B200, `GPU-dbc66c8d-…-088b6cebe52f`, `sm_count: 148`.
- Helion `375363d8663af60d71a0869003db7954f22c42ca` (PR #3409), SGLang PR #32593 snapshot
  `8ad8d192c1741c12ecbef955c1cfaad7e4751401`, torch `2.12.0+cu132`, Triton `3.7.0`.
- BF16 activations, **FP32 recurrent state**, K = V = 128. Decode B ∈ {1,4,8,16,32,64,128,256};
  prefill T ∈ {512,1024,2048,4096,8192}; H ∈ {16,32}.

## B.1 The four arms — exact definitions

1. **`triton`** — SGLang's handwritten Triton. For decode this launches
   `fused_recurrent_kda_packed_decode_kernel` **directly** (see B.5 for why that matters).
   Source: `sglang/kernels/ops/attention/fla/fused_recurrent.py` and `.../fla/kda.py`.
2. **`no_heuristics`** — canonical `ConfigSpec._base_default_config()`. Verbatim from
   `results.json`: *"Canonical `ConfigSpec._base_default_config()`; all promoted compiler defaults
   bypassed, including reduction and matmul/multi-matmul heuristics."* This is the **raw unseeded
   default**, NOT the pre-change compiler heuristic.
3. **`heuristic`** — the current compiler-promoted default on this branch (what a user gets with
   no autotuning).
4. **`shipped`** / "pre-tuned" — the fixed `helion.Config` embedded in SGLang PR #32593
   (`kda_decode.py:_KDA_CONFIG`, `kda_prefill.py`), i.e. **the product of full autotuning,
   replayed with zero tuning time counted**.

Arms 2–4 compile the **same Helion kernel bodies**; only the config differs. Arm 1 is a
different source (SGLang's vendored FLA Triton).

Protocol facts (from `measurement_metadata`, per cell):
`method: "rotated_triton_do_bench_cudagraph"`, `rep_ms: 20`, `rounds: 4`,
`samples_per_round: 10`, `warm_l2: true`, `address_matched_timing_inputs: true`,
`mutable_state_reset_before_each_measurement: true`,
`all_helion_policies_precompiled_before_timing: true`,
`configs_canonicalized_before_compilation: true`, `single_process_four_arm: true`,
`timing_rounds_balanced_by_arm: true`. Each shape ran in a **fresh process** so its heuristic
config derives from that shape.

## B.2 The aggregate table — RE-DERIVED, all 24 values match exactly

Recomputed as per-cell geometric means from `results.json:records[*].ratios`:

| Scope | N | Unseeded/Triton | **Heuristic/Triton** | Pre-tuned/Triton | Heuristic/unseeded | Pre-tuned/unseeded | Pre-tuned/heuristic |
|---|---:|---:|---:|---:|---:|---:|---:|
| Decode | 16 | 1.0829x | **0.8739x** | 1.5061x | **0.8070x** | 1.3909x | 1.7235x |
| Fixed / preactivated prefill | 10 | 0.6741x | **1.1556x** | 1.5509x | 1.7143x | 2.3007x | 1.3420x |
| Packed / raw prefill (production) | 10 | 0.6632x | **1.1110x** | 1.1982x | 1.6752x | 1.8067x | 1.0785x |
| Overall | 36 | 0.8284x | **1.0095x** | 1.4250x | 1.2186x | 1.7201x | 1.4115x |

Win counts **[derived here]**: heuristic beats Triton on **3/16 decode**, **9/10 fixed prefill**,
**8/10 packed prefill**. The 2 packed-prefill losses are the biggest shapes at H=32:
T=4096 (0.9802x) and T=8192 (0.9217x); the 1 fixed-prefill loss is T=8192, H=32 (0.9561x).

Reproducibility gate (from `prior_run_overlap_audit`, limit 2%, result PASS): across all 36 cells,
latency drift vs the prior 3-arm run is **−0.05% Triton, −0.06% heuristic, +0.03% pre-tuned**;
ratio drift +0.01% / −0.08%. All 108 correctness comparisons (output + mutated recurrent state)
passed; the no-heuristics config differs from the heuristic config for **all 126** generated
Helion kernel instances.

### B.2.1 Timing: CUDA-graph op timing vs eager — must be stated

`REPORT.md` is explicit and should be quoted:

> "The headline three-arm results use `triton.testing.do_bench_cudagraph`, which captures many
> unrolled operation calls and amortizes graph-replay submission over them. **Treat these as
> GPU-pipeline/kernel measurements, not literal single-wrapper-call latency.**"

> "Decode's PR timing is also not a pure kernel metric. It times repeated eager wrapper calls with
> CUDA events, so host launch spacing is included."

Concretely, the same comparison under the two protocols disagrees a lot:

| Protocol | Decode, shipped/Triton |
|---|---:|
| SGLang PR's own claim (repeated eager + CUDA events) | 1.118x |
| Reproduced on Helion **v1.4.0**, PR's eager protocol | **1.140x** (+1.9% vs claim) |
| Same eager protocol on **current** Helion | **0.962x** |
| CUDA-graph kernel timing (the headline arm) | **1.5061x** |

So: never merge these SGLang CUDA-graph numbers with the eager-dispatch linear-attention/FLA
numbers, and never call 1.5061x "1.5x lower decode latency in production" — it is a kernel-pipeline
measurement with host dispatch amortized away.

Also `REPORT.md`, prefill boundary: the PR's published `67/42 µs` H=16 T=512 row only reproduces
with **fixed, preactivated** inputs (`68.98 / 44.36 µs`). With packed metadata, raw gates, and
in-operation Q/K normalization — the production boundary — shipped Helion is only **1.203x** over
Triton. *"This is an operation-boundary mismatch, not a shape difference."*

## B.3 ★ THE MOST IMPORTANT PRECISION POINT: decode is a REDUCTION-heuristic cell, not multi-matmul

I read `records[*].configs[*].heuristic_names` for all 36 cells. Complete attribution:

| Scope | Helion kernel | Cells | Heuristic that fired |
|---|---|---:|---|
| **Decode** | `_helion_fused_recurrent_kda_packed_decode_body` | 16 | **`triton_reduction_tile_sm100`** (the reduction heuristic) |
| Fixed prefill | `_gate_cumsum_operands` | 10 | `triton_reduction_tile_sm100` |
| Fixed prefill | `_intra_matrices_wide`, `_intra_solve_recompute`, `_chunk_state`, `_chunk_output` | 10 each | **`triton_b200_multi_matmul`** |
| Packed prefill | `_l2norm_qk`, `_gate_cumsum_operands` | 10 each | `triton_reduction_tile_sm100` |
| Packed prefill | `_intra_matrices_wide`, `_intra_solve_recompute`, `_chunk_state`, `_chunk_output` | 10 each | **`triton_b200_multi_matmul`** |

**The multi-matmul heuristic never fires on the decode path at all.** The decode body has no
`tl.dot`; its two contractions are broadcast-multiply-then-`.sum(-1)`
(`(state * k[None,:]).sum(-1)` and `(state * q[None,:]).sum(-1)`), which is a reduction, not a
matmul. Corroborating evidence: the 149-cell on-corpus matmul study
(`matmul_heuristic_perf_results/on_corpus_pretuned/RESULTS.md`) lists
`sglang.kda_prefill.*` and `sglang.kda_replayssm.replayssm_decode_body` but **not**
`sglang.kda_decode.packed_decode_body`; and `config_recheck/RESULTS.md`'s 25-cell "matmul subset"
excludes the decode-body cells (they are the `Config changed = yes` rows).

**Blog consequence:** if the post headlines multi-matmul/linear attention, then "we lose to
handwritten Triton on SGLang decode" is *not* a multi-matmul result. On SGLang, multi-matmul
touches only the prefill kernels, and there it wins (1.1556x fixed / 1.1110x packed vs Triton,
and 1.71x / 1.68x vs the unseeded default). The decode loss belongs to the reduction heuristic.
The index doc `PYTORCH_BLOG_HEURISTICS_RESULTS.md §2` does not make this distinction — it should.

## B.4 DIAGNOSIS of the decode regression

### B.4.1 The kernel has exactly ONE tunable block size

`kda_decode.py`:

```python
block_v = hl.register_block_size(1, V)
for tile_b, tile_hv, tile_v in hl.tile([B, HV, V], block_size=[1, 1, block_v]):
```

The author pinned the B and HV tiles to 1. So `block_sizes` is a length-1 list holding only the
**V tile**, and the entire config space the compiler can move is
{V tile, num_warps, num_stages, loop_orders, pid_type, indexing, eviction}.
With V = 128 the grid is `B * HV * (128 / block_v)` and, since `HV == H` for these shapes,
**latency is a function of `B*H` only** — verified: B=4,H=32 and B=8,H=16 differ by 0.07%
(6.775 vs 6.780 µs); B=8,H=32 and B=16,H=16 by 0.01%.

### B.4.2 The four configs, side by side

| Arm | V tile | num_warps | num_stages | loop_orders | pid_type | tiles per (b,hv) | warps per (b,hv) |
|---|---:|---:|---:|---|---|---:|---:|
| SGLang **Triton** | 32 (`BV=min(next_pow2(V),32)`) | **1** | **3** | — | 2-D grid `(NV=4, B*HV)` | 4 | 4 |
| Helion **unseeded** | 32 | 4 | 1 | `[[0,1,2]]` | `flat` | 4 | 16 |
| Helion **heuristic** | **4** | **2** | 1 | `[[0,1,2]]` | `flat` | **32** | **64** |
| Helion **pre-tuned** | 8 | **1** | 1 | `[[2,1,0]]` | **`xyz`** | 16 | 16 |

The heuristic emits **the identical config on all 16 decode shapes** — `block_sizes=[4]`,
`num_warps=2`, `num_stages=1`, `loop_orders=[[0,1,2]]`, `pid_type='flat'`, no eviction override.
It is completely **shape-invariant** here: nothing about B or H moves it. So all 16 cells are one
config decision, and the cross-shape spread in the ratio is purely how that one config scales.

### B.4.3 What actually goes wrong: fine on fixed cost, bad on marginal cost **[derived here]**

Least-squares fit of latency vs `B*H` over the 16 decode cells:

| Arm | marginal cost (ns per (b,hv) pair) | fixed cost (µs intercept) |
|---|---:|---:|
| SGLang Triton | 23.57 | **4.16** |
| Helion unseeded | 27.76 | 0.56 |
| **Helion heuristic** | **35.80** (1.52x Triton, **1.81x pre-tuned**) | 1.44 |
| Helion pre-tuned | **19.74** | 0.70 |

That is the whole story in two numbers:

- **Triton's fixed overhead is ~3x Helion's** (4.16 µs vs 0.56–1.44 µs). That is why the heuristic
  *wins* on the three smallest shapes (B=1,H=16 → 1.7616x; B=1,H=32 → 1.6529x; B=4,H=16 → 1.3161x),
  and why the unseeded default also looks fine at 1.0829x geomean.
- **The heuristic's marginal cost is the worst of all four arms.** Once `B*H ≥ 128` (i.e. the grid
  is saturated: 4096 programs on 148 SMs) the extra programs stop buying fill and become pure
  overhead. The ratio then decays monotonically: 0.9722 → 0.7276 → 0.7381 → 0.7003 → **0.6523x**
  at B=256, H=32.

Achieved bandwidth (counting the FP32 state read + in-place write, which dominates:
`2·V·K·4 = 128 KiB` per (b,hv) pair) **[derived here]**:

| B*H | Triton | Unseeded | **Heuristic** | Pre-tuned |
|---:|---:|---:|---:|---:|
| 512 | 4.55 TB/s | 4.97 | **3.53** | 7.33 |
| 1024 | 5.08 | 5.54 | **3.76** | **8.41** ← best observed |
| 4096 | 5.24 | 4.75 | **3.68** | 6.57 |
| 8192 | 5.65 | 4.74 | **3.68** | 6.75 |

The heuristic **plateaus at 3.7 TB/s — about 44% of the 8.4 TB/s the autotuned config reaches**
on the same body. Independent cross-check that these bandwidth numbers are sane: SGLang's own
source comment on the CUDA fast path says its Triton kernel *"tops out at ~5 TB/s holding a
`[BV, K]` register tile per warp"* — and I measure Triton at 5.08–5.65 TB/s.

### B.4.4 Mechanism candidates, ranked, with honesty about what is proven

**Proven from the artifacts:**

- **Wrong warp count is the single biggest historical error.** On Helion **v1.4.0** the *same*
  `triton_reduction_tile_sm100` heuristic fired on this body and emitted `block_sizes=[4]` with
  **`num_warps=8`** (plus a `load_eviction_policies` list). Its decode geomean was
  **0.3321x of Triton** (B=256,H=32: 913.87 µs). Current emits `[4]` with **`num_warps=2`** →
  **0.8739x** (295.50 µs). **Same block size, warps 8→2, 2.63x faster.** Both SGLang's Triton and
  the autotuned Helion config use **`num_warps=1`**. Every good config on this kernel is one warp;
  the heuristic's warp ramp is pushing the wrong direction.
- **The heuristic is 1.5–2.3x behind the shipped config at the kernel level too**, independently of
  the operation harness. `config_recheck/RESULTS.md`, `sglang.kda_decode.packed_decode_body`,
  Shipped/current column: **1.7857x, 1.7308x, 2.2554x, 1.5557x, 1.0155x** over five cases.
- **`num_stages` is never considered.** The reduction seed hardcodes `"num_stages": 1`
  (`triton.py`, `TritonStandardReductionHeuristicSM90.get_seed_config`). SGLang's Triton uses
  **`num_stages=3`** on this exact kernel. A software-pipelined recurrent state update is a real
  lever the heuristic structurally cannot reach on this track.
- **Grid mapping differs and is un-isolated.** Heuristic: `pid_type='flat'` with
  `loop_orders=[[0,1,2]]`. Pre-tuned: `pid_type='xyz'` with `loop_orders=[[2,1,0]]`, and SGLang's
  own comment explains why: *"Tile V on the CUDA x axis so tensor-parallel head counts do not
  change the grid width."* The reduction seed hardcodes `'flat'` with the comment "these reductions
  are grid-saturated at the M-grid" — an assumption that is false for a 3-D user-tiled decode loop.

**Plausible, consistent with the slope, NOT isolated by any measurement in the tree
[derived here]:**

- **Redundant per-program prologue.** Everything except the state tile is `tile_v`-independent:
  `a[K]`, `dt_bias[K]`, `A_log`, `b`, `k[K]`, `q[K]`, plus the whole gate pipeline
  (`exp2`, `log1p`-style softplus, two `sigmoid`s) over K=128. At `block_v=4` that work is redone
  **32x per (b,hv)** instead of 16x (pre-tuned) or 4x (unseeded). The measured
  heuristic/pre-tuned slope ratio is **1.81x** against a **2.0x** tile-count ratio — suggestively
  close, but redundancy alone cannot be the whole story, because the unseeded arm has 4x *less*
  redundancy than pre-tuned and is still 1.41x worse per unit. The optimum is interior
  (block_v ≈ 8–32 with one warp), and the two Helion non-tuned arms sit on opposite sides of it.
- **Warp-slot waste.** Warps consumed per (b,hv) pair: heuristic **64**, unseeded 16, pre-tuned 16,
  Triton 4. The heuristic asks for 4x the warp slots of every other Helion-viable config for the
  same useful work.
- **Per-thread state residency.** Pre-tuned holds `[8,128]` fp32 = 1024 values in 32 threads
  (32/thread); Triton holds `[32,128]` = 4096 in 32 threads (128/thread — the "register tile per
  warp" SGLang describes). The heuristic holds `[4,128]` = 512 across 64 threads (8/thread),
  i.e. it *spreads* the recurrent state instead of keeping it resident.

**No report in the tree writes down a diagnosis of this regression.** I grepped every `*.md`
under `wt-sm100-linattn` for a decode-regression explanation; the strongest existing statements
are just the observations:

> "The current heuristic helps both prefill boundaries but is slower than handwritten Triton for
> decode in aggregate. SGLang's shipped configs remain faster than the heuristic in all three
> scopes." — `sglang_triton_head_to_head/REPORT.md`

So everything in B.4.3/B.4.4 marked **[derived here]** is my derivation from raw latencies and
configs, not a checked-in finding. Present it as analysis, not as a measured attribution.

## B.5 A large caveat on the decode BASELINE that cuts the other way

`REPORT.md` states, and I confirmed in the SGLang source:

> "the final SGLang PR wrapper silently dispatches 12 of the 16 tested BF16/FP32 shapes (B>=8) to
> a custom CUDA kernel, despite the benchmark column being described as packed Triton."

Verified in `sglang/kernels/ops/attention/fla/fused_recurrent.py` → `kda_packed_decode.covered()`
with `_MIN_BATCH = 8`: for `use_qk_l2norm_in_kernel`, K=V=128, bf16 in / fp32 state,
**B ≥ 8 goes to a hand-written CUDA kernel**, not Triton. That is 6 batch sizes × 2 head counts
= **12 of the 16 decode cells**. SGLang's own comment on that path:

> "row-streaming state update reaches the in-place R+W bandwidth of the part (~9.6 TB/s) where
> this triton kernel tops out at ~5 TB/s holding a `[BV, K]` register tile per warp … small
> batches keep triton (launch-bound anyway)."

Both implications matter:

- The study deliberately measures the **handwritten Triton** kernel (the plan says so explicitly:
  "Launch `fused_recurrent_kda_packed_decode_kernel` directly as the unambiguous handwritten Triton
  baseline"). That is a fair and *harder* Triton baseline than the wrapper would give.
- But it means the decode arm we lose to is **a baseline SGLang itself no longer runs in production
  for 12 of these 16 shapes**. Nothing in the campaign measured that CUDA kernel, so the blog must
  not say "we're 13% behind SGLang's production decode." It is "13% behind SGLang's handwritten
  Triton decode kernel," and by SGLang's own account their CUDA kernel is ~1.9x faster than that
  Triton kernel on bandwidth. Also note the pre-tuned Helion arm (6.6–8.4 TB/s) is already well
  past Triton's ~5 TB/s ceiling — so the *body* is fine; only the config choice is not.

## B.6 Other caveats for the SGLang section

- **Launch counts** (steady-state profiler audit, `REPORT.md`): packed decode 1 Helion launch vs
  1 Triton launch. Fixed prefill 5 Helion vs 5 Triton (small grid) / 7 (large grid). Packed prefill
  6 Helion vs 7 / 9. "Small grid" = `ceil(T/64)*H ≤ 256` = 3 of the 10 prefill shapes. Helion needs
  *fewer* launches on prefill, which is part of the prefill win.
- **This study ≠ the FLA end-to-end study.** Different kernel source (SGLang's specialized
  `kda_decode.py`/`kda_prefill.py` vs the generic `examples/linear/linear_attention_engine.py`),
  different Triton baseline (SGLang's vendored FLA vs installed `flash-linear-attention==0.5.2`),
  different timing (CUDA graph vs warmed eager). Do not merge the numbers.
- **Harness bug that was found and fixed** — worth one sentence as a methodology flex: the
  provisional harness allocated one input clone per arm, which made timing addresses depend on arm
  count and shifted shipped `decode H=16,B=8` from ~4.15 µs to ~4.36 µs. With one shared allocation
  the 3-arm and 4-arm protocols measured 4.1422 and 4.1437 µs (**0.036%**). The final harness uses
  identical tensor addresses for every arm. Only the
  `four_arm_no_heuristics_v2_address_matched/` directory should be quoted;
  `four_arm_no_heuristics/` is the superseded provisional run (its decode geomeans are
  1.0807 / 0.8740 / 1.4993 — close, but not the number of record).
- **Worst measurement spread**: "the worst three-round median spread was 4.2% on a roughly 5 µs
  decode cell; the two steady-state rounds agreed and determine the reported median."
- SGLang fix PR #35197 was audited and does not affect these T≥512, K=V=128 shapes.
- Config drift note: `config_recheck` found **15 of 40** reduction/default configs changed after
  the originally recorded study, which is why "all SGLang kernels shipped/current" is 1.2415x
  while the frozen matmul subset reproduces at 1.2017x (−0.04% from the prior 1.2022x).

## B.7 ★ Honest one-paragraph framing of the SGLang result for the blog

> SGLang's KDA kernels are the case where compile-time analysis is clearly not enough. Across all
> 36 operation cells the heuristic lands at 1.0095x of SGLang's handwritten Triton — a dead heat —
> but that average hides two opposite outcomes. On both prefill boundaries the heuristic wins:
> 1.1556x versus Triton on the fixed/preactivated path and 1.1110x on the packed/raw production
> path, up from 0.6741x and 0.6632x for Helion's unseeded default, so the heuristic is worth about
> 1.7x on prefill and is the difference between losing badly and winning. On decode it loses:
> 0.8739x versus Triton, and — the part worth admitting out loud — 0.8070x versus Helion's own
> unseeded default, which the analysis made *worse*. Full autotuning on the identical kernel body
> reaches 1.5061x on decode, so the ceiling is there and the heuristic is leaving 1.72x of it on
> the table. Two structural reasons, both honest: the SGLang decode body exposes a single tunable
> block size (the author pinned the batch and head tiles to 1), so there is almost no analysis to
> do and one wrong guess is the whole result — the heuristic emits the same V-tile of 4 with 2
> warps for all 16 shapes, where every good config on that kernel uses 1 warp and a wider tile; and
> the decode body is a reduction, not a matmul, so this is the reduction heuristic's miss, not the
> multi-matmul heuristic's. The failure mode is also specific rather than diffuse: Helion's fixed
> per-call overhead is about a third of Triton's, so the heuristic actually wins the three smallest
> batches (up to 1.76x at B=1), and only loses once the grid saturates, where its marginal cost per
> (batch, head) pair is 1.8x the autotuned config's and it plateaus at 3.7 TB/s against the 8.4 TB/s
> the same body reaches when tuned. Two caveats in the other direction: these are CUDA-graph kernel
> measurements with host dispatch amortized away, not wrapper latencies; and SGLang's own wrapper
> routes 12 of these 16 decode shapes to a hand-written CUDA kernel rather than the Triton kernel we
> benchmarked, so "we lose to handwritten Triton on decode" is the accurate claim and "we lose to
> SGLang's production decode path" is not something this data can support either way.

---

# Cross-cutting arm-hygiene reminders (things I checked and can confirm)

- Pointwise `default` = `_base_default_config()` = **raw unseeded**, verified in `bench.py:236`.
  SGLang `no_heuristics` = same thing, verified in `results.json:no_heuristics_definition`.
  Neither is the "pre-change compiler heuristic". The only *pre-change* arm anywhere in these two
  datasets is the Helion **v1.4.0** run in `pr_v1_4_fixed/` (decode 0.3321x, prefill 0.6456x vs
  Triton), and it is a genuinely different and much worse thing than the unseeded default.
- `torch.compile` in the pointwise audit is `torch.compile(ref_fn)` in **default mode** on the
  eager reference, verified at `bench.py:690`. Not max-autotune. There is no torch.compile arm in
  the SGLang study.
- "AOT" / "pre-tuned" = full-autotune output replayed with zero tuning time counted, in both
  datasets.
- Pointwise vLLM AOT is **Helion's own H100 (sm90) table**, not vLLM's shipped config. SGLang
  pointwise AOT *is* SGLang's shipped SM100 JSON table. SGLang KDA `shipped` *is* the config
  embedded in SGLang PR #32593.
