# 13 — Does the heuristic make the autotuner find the best config FASTER?

**Blog result-question (b). This is the weakest result in the set; these notes are written so the
blog cannot overclaim it.**

Everything below was re-derived from the **raw per-attempt CSV search logs**
(`autotune.csv`, one row per config attempt) plus each arm's `result.json`, *not* from the prose
reports. My re-derivation reproduces every published number in
`THREE_WAY_AUTOTUNE_TABLES.json` **exactly** (0 mismatches on shared-best latency, terminal-attempt
counts, `num_configs_tested`, configs-to-5%, and wall-seconds-to-5% across all 38 cells × 3 arms),
so the published tables are trustworthy at the arithmetic level. The caveats below are about
*metric design*, not arithmetic errors.

---

## 0. The experiment, exactly

Two GPU runs, later merged into one three-way report.

| Arm (report name) | Where the search log lives | Source | Seeds |
|---|---|---|---|
| **No seed** (`no_seed`, a.k.a. "zero seed") | `zero_seed_full_autotune/cells/*/zero_seed/autotune.csv` | clean `40151a23f`, **compiler heuristics explicitly disabled** | 0 raw / 0 author / 0 normalized (hard-asserted per cell) |
| **Old seeds** (`old_seeds`, a.k.a. `main`, `baseline_40151`) | `multi_seed_full_autotune_ablation/cells/*/main/autotune.csv` | clean `40151a23f`, unpatched | normalized 1–3 per cell (median **2**, total **63**) |
| **Expanded seeds** (`expanded_seeds`, a.k.a. `branch`) | `multi_seed_full_autotune_ablation/cells/*/branch/autotune.csv` | `40151a23f` **+ a 3-file working-tree patch** | normalized 4–15 per cell (median **7**, total **317**) |

Verified facts about the arms (from `result.json → source` / `arm_source_fingerprints`):

- All three arms are the **same git commit** `40151a23f680fff13583cbf64d3c58e0222e19a0`.
- No-seed and old-seed used the *unpatched* tree (patch sha256 = `e3b0c442…`, i.e. the SHA-256 of the
  empty string). Expanded used patch sha256 `d8a97063c1aa7bbfcef30e2a71be6889e098d8e37fd31b6e9d266d9cd3978576`
  touching exactly `helion/_compiler/autotuner_heuristics/triton.py`,
  `helion/_compiler/device_ir_analysis.py`, `helion/autotuner/config_spec.py`.
- **I confirmed that patch is still byte-identical to the current `wt-sm100-linattn` working tree**
  (`git diff` on those three files hashes to `d8a97063…`). So what was measured is what the blog's
  heuristic is.
- No-seed differs from old-seed *only* by turning heuristics off — same tree, same commit. Clean
  ablation.
- Searcher: `LFBOTreeSearch`, `autotune_effort='full'`, `autotune_random_seed=0`,
  `autotune_accuracy_check=True`, `static_shapes=True`, `dot_precision='tf32'`, on **NVIDIA B200**
  (2 physical GPUs, per-cell GPU affinity frozen across arms).
- 38 frozen `(kernel, shape)` cells, selection hash
  `60c8b0756490344bb2182c48dcd667e5c4db8eff7a3fb11e1e5fbff9cc4d0230`. **30 are `multi` (multi-matmul)
  and 8 are `single`.**

Corpus note: the 38 cells are what *survived*. The paired run started from a 45-cell manifest and
excluded 6 kernel families because **every** generation-0 config failed the accuracy gate (NaNs in
the randomly reconstructed inputs) — `chunk_bwd_dstate_delta`, `chunk_bwd_state_du_kda`,
`chunk_fwd_h_delta(_varlen)`, `chunk_fwd_wy_delta(_varlen)`, `_intra_matrices_wide`. This is a
**survivorship filter on the corpus**, documented in
`MATMUL_MULTI_SEED_FULL_AUTOTUNE_OVERNIGHT_LOG.md`.

---

## 1. The metric, precisely (and its four weaknesses)

### Definition, verbatim behaviour (verified against the CSVs)

1. Every config attempt writes a `started` row and then exactly one **terminal** row with
   `status ∈ {ok, error, timeout}`. **Terminal attempts** = the non-`started` rows, in log order.
2. For a cell, `shared_raw_best_ms` = **min `perf_ms` over every successful config in all three
   arms' raw logs**. Targets are `(1+X) · shared_raw_best`.
3. **"Configs to within X%" = the 1-based index, among that arm's terminal attempts, of the first
   terminal row whose `perf_ms ≤ target`.** Failed attempts (compile error, compile/benchmark
   timeout, accuracy rejection) advance the counter and can never improve the incumbent.
4. If an arm never reaches the target, the table substitutes **that arm's full terminal-attempt
   count** and appends `+`. This is a **right-censored lower bound**, not a measurement.
5. "Complete full search" is `AutotuneMetrics.num_configs_tested` — a total, not another tolerance.
6. Aggregate geomeans are **cell-wise arm/no-seed ratios** (38 ratios, then geometric mean), not
   ratios of the sums. Both are published and they differ.

### Weakness 1 — ~19% of the budget is failures

Across all 114 searches the CSVs contain **56,024 terminal attempts: 45,236 `ok` (80.7%),
10,357 `error` (18.5%), 431 `timeout` (0.8%)**. So roughly **one in five "configs" counted by this
metric produced no timing at all.** Reported failure totals (`result.json`): compile failures
3,102 / 2,822 / 2,576 and accuracy failures 450 / 948 / 459 (no-seed / old / expanded). The metric
is therefore "attempts consumed", closer to a wall-clock proxy than to "configs evaluated".

### Weakness 2 — right-censoring, and it is *biased toward the expanded arm*

At the 5% target the substituted lower bounds are:

| Arm | cells that never reach 5% | sum of substituted bounds | mean substituted bound |
|---|---:|---:|---:|
| No seed | 10 | 4,617 | 461.7 |
| Old seeds | 4 | 2,455 | 613.8 |
| Expanded seeds | 10 | 3,585 | **358.5** |

No-seed and expanded **both** fail on 10 cells, but expanded's substituted penalty is ~22% smaller
*because its full runs are shorter*. **An arm that stops searching earlier gets a smaller penalty
for never reaching the target.** That is a structural bias in favour of the expanded arm on the
censored rows, and it is the single most important caveat in this dataset.

### Weakness 3 — the 5% row is the least trustworthy

- Target reach at 5%: **28/38 no-seed, 34/38 old, 28/38 expanded.** Old seeds reach the shared 5%
  target on *more* cells than either other arm, yet the censored sum shows expanded "winning"
  (8,419 vs 8,674). The two facts point in opposite directions purely because of Weakness 2.
- Per-cell sign test at 5% (censored): expanded is better than no-seed on **26 cells, worse on 11,
  tied on 1**. Old seeds are better on **31, worse on 5, tied on 2**. At the tight tolerance the
  *old* pool is the more reliable one.
- Censoring-free sensitivity (restrict to cells where **all three** arms reached the target):

  | target | n cells | sum no / old / exp | sum ratio old, exp | geomean old, exp |
  |---|---:|---|---|---|
  | within 50% | 38 | 2,683 / 513 / 78 | 0.1912, 0.0291 | 0.1006, 0.0626 |
  | within 25% | 35 | 3,676 / 2,072 / 1,197 | 0.5637, 0.3256 | 0.2024, 0.0991 |
  | within 12.5% | 30 | 4,701 / 3,317 / 2,484 | 0.7056, 0.5284 | 0.3005, 0.2105 |
  | within 5% | **21** | 4,863 / 3,460 / 2,897 | 0.7115, 0.5957 | 0.4488, 0.3527 |

  The 5% row now covers only **21 of 38 cells** — a survivor-selected subset. It is not a fix, just
  a bound on how much the censoring moved things (expanded's 5% geomean swings 0.4478 → 0.3527).

### Weakness 4 — "configs to within 50%" is mostly re-reporting result (a)

The seeded arms hit the loose targets **inside their own seed budget**, i.e. a *seed config itself*
qualifies:

| target | hit inside own seed budget (no-seed / old / expanded) |
|---|---|
| within 50% | 0/38 · 27/38 · **37/38** |
| within 25% | 0/38 · 17/38 · 27/38 |
| within 12.5% | 0/38 · 9/38 · 12/38 |
| within 5% | 0/38 · 3/38 · 5/38 |

So "expanded seeds get within 50% after 2.1 attempts on average" is essentially the statement *the
heuristic config is within 50% of the best config anyone found* — that is result (a) (config quality)
expressed in search coordinates. Only the 12.5% and 5% rows are genuinely about search *dynamics*,
and those are the weakest rows.

### Minor internal inconsistency worth knowing (not an error the blog will trip on, but real)

The censored substitution uses **`terminal_attempts`** while the "Complete full search" row uses
**`num_configs_tested`**. These differ in 26 of 114 arms (369 extra rows in total), because
compile/benchmark **timeout** rows are logged in the CSV but not counted as configs tested. Effect:
- The full-run sums would be 19,932 / 19,070 / 17,022 on the terminal basis vs the published
  19,796 / 18,923 / 16,936 (≤0.7% difference; ratios 0.9568/0.8540 vs 0.9559/0.8555).
- In **3 published rows the censored `+` bound exceeds that arm's own published full-run total**
  (`biased_attention` expanded 416+ vs 404; `blackwell_attention` no-seed 582+ vs 580;
  `bf16xint16` no-seed 547+ vs 546). Don't put those two columns side by side in the blog.

---

## 2. All-38-cell threshold table — VERIFIED (re-derived from raw CSVs)

Every number below was recomputed from the per-attempt logs and matches
`THREE_WAY_AUTOTUNE_SIGNIFICANT_SEED_ADVANTAGE.md` § "All Cells (38)" and
`PYTORCH_BLOG_HEURISTICS_RESULTS.md` § 4 **exactly**.

### Configs (terminal attempts) to target, n = 38

| Search target | No-seed sum | Old-seed sum | Expanded-seed sum | Old/no sum | Exp/no sum | Old/no **geomean** | Exp/no **geomean** | Reach (no/old/exp) |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Within 50% | 2,683 | 513 | 78 | 0.1912x | **0.0291x** | 0.1006x | **0.0626x** | 38/38/38 |
| Within 25% | 5,131 | 2,960 | 2,097 | 0.5769x | 0.4087x | 0.2184x | 0.1131x | 36/37/37 |
| Within 12.5% | 8,767 | 6,046 | 4,867 | 0.6896x | 0.5551x | 0.3413x | 0.2553x | 32/36/35 |
| Within 5% | 12,124 | 8,674 | 8,419 | 0.7154x | 0.6944x | 0.5044x | 0.4478x | 28/34/28 |
| Complete full search | 19,796 | 18,923 | 16,936 | 0.9559x | 0.8555x | 0.9643x | 0.8551x | 38/38/38 |

Derived reductions (expanded vs no seed, on the sums): **97.1% / 59.1% / 44.5% / 30.6% / 14.4%** —
all five reproduce the blog index exactly.

Medians and means (censored), which are more honest than sums:

| target | median configs (no / old / exp) | mean configs (no / old / exp) |
|---|---|---|
| within 50% | **50 / 1 / 1** | 70.6 / 13.5 / 2.1 |
| within 25% | 108 / 19 / 4.5 | 135.0 / 77.9 / 55.2 |
| within 12.5% | 193 / 123 / 108 | 230.7 / 159.1 / 128.1 |
| within 5% | 290 / 162 / **202.5** | 319.1 / 228.3 / 221.6 |

Note the 5% median **inverts**: expanded's median (202.5) is worse than old's (162).

### Per-cell sign test vs no-seed (censored)

| target | old seeds better / worse / tie | expanded better / worse / tie |
|---|---|---|
| within 50% | 25 / 4 / 9 | 29 / 3 / 6 |
| within 25% | 22 / 9 / 7 | 27 / 7 / 4 |
| within 12.5% | 25 / 9 / 4 | 26 / 11 / 1 |
| within 5% | **31 / 5 / 2** | **26 / 11 / 1** |
| complete search, configs | 23 fewer / 15 more | 25 fewer / 13 more |
| complete search, wall | 24 fewer / 14 more | 21 fewer / 17 more |

**This is the sentence a skeptical reviewer will want: on the complete search, expanded seeds cost
MORE configs than no seeds on 13 of 38 cells and more wall time on 17 of 38.** Named regressions
from the overnight log: `jagged_bmm` (544 → 734 configs), `gdn_fwd_h` (492 → 657),
`_chunk_output::packed_h12_t2048` (240 → 277), `gather_gemv` (336 → 425).

### Same table on WALL SECONDS (derived here; only the 5% row is published)

Censored with the arm's own full `autotune_time`.

| Search target | No-seed sum (s) | Old sum (s) | Exp sum (s) | Old/no sum | Exp/no sum | Old/no geo | Exp/no geo | median s (no / old / exp) |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Within 50% | 4,670.9 | 1,593.1 | 1,153.3 | 0.3411x | 0.2469x | 0.4843x | 0.4120x | 52.6 / 23.5 / **20.9** |
| Within 25% | 7,472.1 | 5,024.5 | 3,901.2 | 0.6724x | 0.5221x | 0.5702x | 0.4073x | 105.4 / 44.2 / 28.1 |
| Within 12.5% | 13,052.3 | 10,854.8 | 8,106.2 | 0.8316x | 0.6211x | 0.6090x | 0.4924x | 199.7 / 118.7 / 72.5 |
| Within 5% | 19,885.5 | 15,210.1 | 13,901.0 | 0.7649x | 0.6991x | **0.6230x** | **0.6141x** | 334.7 / 174.3 / 194.4 |
| Complete | 35,151.5 | 35,058.0 | 29,180.6 | **0.9973x** | 0.8301x | 0.9572x | 0.8440x | 587.2 / 680.8 / 500.5 |

The 5% row (19,885.5 / 15,210.1 / 13,901.0 s, geomeans 0.6230x / 0.6141x) matches
`THREE_WAY_AUTOTUNE_TABLES.md` exactly.

**Two things the blog must not miss here.**
1. The spectacular **0.0291x config ratio at the 50% target collapses to 0.2469x on wall clock**,
   because the search has a fixed start-up floor: the first *successful* config's timestamp is a
   median of **19.3 s (no-seed) / 20.8 s (old) / 20.9 s (expanded)** — and expanded's median
   time-to-within-50% is **20.9 s**, i.e. identical to its floor. The seeded arms are already at the
   physical limit of "how fast can this tuner produce any timing at all". Quote the wall-clock
   number, not the config-count number, if the claim is about time.
2. **The old seed pool bought essentially zero total wall time**: 35,058.0 s vs 35,151.5 s = a
   **0.997x** sum ratio (the 0.9572x is a per-cell geomean over a very skewed distribution). Only
   the expanded pool moves the complete-search cost (0.830x sum / 0.844x geomean).

---

## 3. Final-winner quality — VERIFIED. This is NOT a quality win.

Method (verified from all 38 `winner_replay/result.json`): each cell's three final winners were
replayed **in one process, on the same physical GPU, using the single clean baseline tree**
(`/home/dev/local/wt-multiseed-baseline` @ `40151a23f`), `rotated_interleaved_cuda_events`,
**cold L2**, CUDA-graph captured for all three arms, **7 rounds** on every cell,
`repeat_per_round` 10–1,000 (median 267), target 25 ms per arm-round. So this is a pure
*config* comparison under identical lowering — methodologically clean.

| Quantity | Value | Verified |
|---|---|---|
| Winner-latency geomean, old / no-seed | **1.0005x** (lower = better) | ✔ recomputed 1.000522 |
| Winner-latency geomean, expanded / no-seed | **1.0142x** | ✔ recomputed 1.014167 |
| Cells where all three winners are within 5% (`max/min ≤ 1.05`) | **22/38** | ✔ |
| Of the remaining 16, fastest winner: no-seed | **5** | ✔ |
| … old-seed | **8** (includes **2 exact old/expanded ties**: `broadcast_matmul_b16_m512_k768_n1024`, `packed_h12_t2048`, resolved to old by deterministic arm order) | ✔ |
| … expanded-seed | **3** (`chunk_bwd_dk_delta#2`, `chunk_bwd_dqk#3`, `blackwell_attention_b4_h32_s2048_d64_bf16`) | ✔ |

**Conclusion the blog must state: with a full-effort search on both sides, seeds do not make the
final config better. The unseeded arm's winners are, in geomean, 0.05% and 1.4% *faster* than the
seeded arms'. This is a search-EFFICIENCY result only.**

Two extra honest details:
- Worst expanded regression: `bf16_regular_h16_b64` (SGLang ReplaySSM decode) — no-seed winner
  32.768 us vs old 49.152 us (1.500x) and expanded 47.104 us (1.438x). `mamba_scan` is 1.163x for
  both seeded arms.
- **Do not quote the "winner-latency raw total (sum)" row** (128,968 / 130,940 / 129,814 us). One
  cell, `jagged_hstu_b2_l7168_max4096_h16_d64` at 119,064 us, is **92.3%** of that sum. The report
  itself says the sums are "for completeness, not a workload-weighted performance score".
- Accuracy footnote: all three winners independently passed the autotuner accuracy gate on all 38
  cells. On `jagged_bmm_b8_l255_d128_k64_bias` the old and expanded winners agree with each other
  but both differ from the no-seed winner on **7 / 16,320 values** under the stricter pairwise
  replay tolerance. One pairwise disagreement total.

---

## 4. The filtered 26-cell cohort — VERIFIED, and the selection rule verbatim

I reproduced the 26-cell membership, the 12/7/7 split, and **every number in all four aggregate
tables** of `THREE_WAY_AUTOTUNE_SIGNIFICANT_SEED_ADVANTAGE.md` from the raw CSVs.

### Selection rule, verbatim from the report (lines 5–8)

> - For each cell, `s` is the expanded arm's `normalized_compiler_seed_count`: the complete expanded
>   seed pool after normalization and deduplication.
> - All three raw-search incumbents are evaluated after the same first `s` terminal config attempts.
>   Failed attempts consume budget but cannot improve the incumbent.
> - A cell qualifies when `L_no_seed(s) / min(L_old_seed(s), L_expanded_seed(s)) > 1.50`. Thus the
>   no-seed incumbent is more than 50% slower at the equal budget.
> - Sections are determined by which seeded arm supplies the lower raw incumbent latency at `s`, not
>   by the eventual replay winner.

`s` ranges 4–15 (median 7). **12 cells** have expanded supplying the better incumbent at `s`,
**7** old, **7** exactly tied.

### The 26-cell table (verified)

| Search target | No-seed sum | Old-seed sum | Expanded-seed sum | Exp/no sum | Exp/no geomean |
|---|---:|---:|---:|---:|---:|
| Within 50% | 2,646 | 494 | 53 | 0.0200x | 0.0201x |
| Within 25% | 4,631 | 2,520 | 1,951 | 0.4213x | 0.0707x |
| Within 12.5% | 7,599 | 4,800 | 3,929 | 0.5170x | 0.1896x |
| Within 5% | 9,751 | 6,605 | 6,720 | 0.6892x | 0.4429x |
| Complete full search | 15,186 | 13,909 | 11,745 | 0.7734x | 0.7493x |

(Old/no sum ratios, also verified: 0.1867 / 0.5442 / 0.6317 / 0.6774 / 0.9159; old/no geomeans
0.0427 / 0.1330 / 0.2814 / 0.4609 / 0.9034.)

**Why the blog must label this cohort explicitly:** the filter is *not* independent of the outcome.
It selects on **early incumbent quality at the seed budget** — the very quantity the "search
efficiency" claim is measuring. It is a post-hoc, outcome-correlated filter that keeps 26/38 = 68%
of cells. The blog index already says "Use all 38 cells for the expanded-seed headline. The 26-cell
table is deliberately filtered for a greater-than-1.5x early seeded advantage" — keep that wording.
Note also that within this cohort the 5% row **still inverts** (expanded 6,720 > old 6,605) and
expanded reaches 5% on only **17/26** cells vs old's 23/26.

Also inside this cohort: `bf16_regular_h16_b64` and `mamba_scan` are cells where the seeded arm has
a better *early incumbent* yet the no-seed arm has the *better final winner* by 1.44–1.50x /
1.16x. The cohort is about early progress, full stop.

---

## 5. WALL CLOCK — every measured number

### Per-cell full autotune time (`AutotuneMetrics.autotune_time`, measured inside the tuner)

| Arm | n | min | p25 | median | p75 | max | mean | sum |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| No seed | 38 | 241.2 s | 397.1 s | **587.2 s** | 1,057.2 s | **5,733.8 s** | 925.0 s | **35,151.5 s = 9.76 GPU-h** |
| Old seeds | 38 | 202.7 s | 373.7 s | 680.8 s | 967.2 s | **7,058.8 s** | 922.6 s | **35,058.0 s = 9.74 GPU-h** |
| Expanded seeds | 38 | **167.2 s** | 366.3 s | 500.5 s | 907.7 s | 6,044.4 s | 767.9 s | **29,180.6 s = 8.11 GPU-h** |
| **All 114 searches** | 114 | 167.2 s | — | 560.9 s | — | 7,058.8 s | 871.8 s | **99,390.1 s = 27.61 GPU-hours** |

`process_wall_seconds` (includes process overhead) is only ~0.3% higher: 35,254.2 / 35,164.3 /
29,299.0 s → 99,717.4 s = 27.70 h.

**Quotable single-kernel worst cases** (one kernel, one shape, one full-effort autotune):
`jagged_hstu_b2_l7168_max4096_h16_d64` = **7,058.8 s ≈ 1 h 58 m** (old-seed arm), 6,044.4 s
(expanded), 5,733.8 s (no-seed). Runners-up on the no-seed arm: `chunk_bwd_o_kda#4` 2,312.7 s,
`biased_attention` 2,141.4 s, `chunk_bwd_wu_kda#3` 1,922.3 s, `chunk_bwd_dqkw_delta#2` 1,900.0 s,
`causal_attention_b4_h32_s4096_d128_bf16` 1,721.3 s.

### Configs benchmarked per cell (`num_configs_tested`)

| Arm | min | median | max | mean | sum |
|---|---:|---:|---:|---:|---:|
| No seed | 147 | 521.5 | 992 | 520.9 | 19,796 |
| Old seeds | 240 | 490.5 | 948 | 498.0 | 18,923 |
| Expanded seeds | 205 | 420.5 | 755 | 445.7 | 16,936 |
| All 114 | 147 | 475.5 | 992 | 488.2 | **55,655** |

Generations explored: no-seed 2–20 (median 8), old 4–19 (median 8), expanded 2–17 (median 7).

### End-to-end scheduler elapsed time (from `events.jsonl` `time_utc`)

| Run | Contents | Elapsed on 2 × B200 |
|---|---|---|
| `zero_seed_full_autotune` | 38 full-effort searches + 38 three-way replays | 2026-08-24T07:21:18Z → 13:12:57Z = **5 h 51 m 39 s** |
| `multi_seed_full_autotune_ablation` | 76 full-effort searches (38 pairs) + 38 replays, plus 36 failed arms / 6 excluded kernel families / retries | 2026-08-22T20:49:10Z → 2026-08-23T08:24:50Z = **11 h 35 m 40 s** |

So the whole three-way study is **17.45 h of elapsed time on 2 B200s = ~34.9 GPU-hours ≈ 1.45
GPU-days of allocated GPU time**, of which **27.6 GPU-hours is pure in-tuner autotune time**.
Tuning the 38-kernel corpus **once**, at full effort, costs **~10 GPU-hours**.

### Contrast arm for "quick" effort (from the broader 244-cell run, `results.json`)

178 cells were tuned at `effort='quick'`:
**min 27.8 s, median 118.3 s, mean 140.9 s, max 642.9 s, total 25,085.9 s = 6.97 GPU-hours.**
So full effort is roughly **5x** the per-cell cost of quick effort (587 s vs 118 s median) — useful
if the blog wants to say "even the cheap autotune setting costs minutes per kernel/shape".

### Time-to-target wall clock (see the table in §2)

Headline honest numbers: median seconds to get within 50% of the best config any arm found —
**52.6 s (no seed) → 23.5 s (old seeds) → 20.9 s (expanded)**; within 25% —
**105.4 s → 44.2 s → 28.1 s**; within 12.5% — **199.7 s → 118.7 s → 72.5 s**.

### A third, self-referential metric that exists in the reports — do NOT use it as a headline

`RESULTS.md` for both runs reports **"time to best search incumbent"**: no-seed 26,249.5 s,
old 23,714.9 s, expanded 18,769.3 s, with geomeans no/old = 1.2764x and no/expanded = 1.4483x
(equivalently old/no = 0.7835x, expanded/no = 0.6905x). This is time until each arm reached **its
own** final best incumbent, so an arm that settles for a worse config "wins". It is not comparable
to the shared-target rows above. The multi-seed run's own branch/main figure is 0.8813x.

### Direction-of-ratio warning (a real trap in these files)

`MATMUL_ZERO_SEED_FULL_AUTOTUNE_OVERNIGHT_LOG.md` and `zero_seed_full_autotune/RESULTS.md` quote
**zero-seed / seeded** ratios (bigger = seeds better): "1.0371x configs, 1.0447x autotune time,
1.2764x time to best, 0.9995x winner latency" vs baseline and "1.1694x, 1.1848x, 1.4483x, 0.9860x"
vs expanded. The three-way tables quote **seeded / no-seed** (smaller = seeds better): 0.9643x,
0.9572x, 1.0005x and 0.8551x, 0.8440x, 1.0142x. **Same data, reciprocal presentation.** I verified
1/0.9643 = 1.0370, 1/0.9572 = 1.0447, 1/0.8551 = 1.1694, 1/0.8440 = 1.1848, 1/1.0005 = 0.9995,
1/1.0142 = 0.9860. Pick one direction for the blog and stick to it.

### Single-matmul vs multi-matmul split (this is the "linear attention" angle)

Geomeans over the 38 cells, split by kernel `kind` (30 multi, 8 single):

| Metric | multi (n=30) | single (n=8) |
|---|---|---|
| expanded/old, configs | 0.8684x | 0.9596x |
| expanded/old, autotune time | **0.8506x** | **1.0088x** |
| expanded/no-seed, configs | 0.8814x | 0.7633x |
| expanded/no-seed, autotune time | 0.8547x | 0.8052x |
| old/no-seed, autotune time | 1.0047x | 0.7982x |

Matches the overnight log's "Multi-only autotune time is 0.8506x; single-only is 1.0088x". **The
expanded (multi-matmul) seed pool's search-cost win is entirely on multi-matmul kernels; on
single-contraction kernels it is a wash (1.009x, slightly worse).** That is exactly the story the
blog wants — but say it, don't let a reader infer that the win is universal.

---

## 6. Three honest one-sentence framings

1. "Seeding the search does not make the *final* tuned kernel faster — on 22 of 38 kernel/shape
   cells all three fully-searched winners land within 5% of each other, and the seeded arms' winner
   latency geomeans are 1.0005x and 1.0142x of the unseeded arm's — but it does move early progress
   forward sharply: the median cell needs **1 benchmarked config instead of 50**, and **20.9 s
   instead of 52.6 s**, to get within 50% of the best config any arm ever found."

2. "On the cost of a *complete* full-effort search the effect is real but modest and not uniform:
   the expanded seed pool cut total benchmarked configs 14.4% (geomean 0.855x) and total in-tuner
   autotune time 17% (geomean 0.844x) across 38 cells, concentrated entirely in the multi-matmul
   kernels (0.851x autotune time vs 1.009x for single-contraction kernels), and it was *more*
   expensive than no seeds on 13 of 38 cells."

3. "The tight-tolerance rows are suggestive, not conclusive: no arm reaches within 5% of the shared
   best on every cell (28/34/28 of 38), the non-reaching cells are right-censored at each arm's own
   run length — which mechanically rewards whichever arm stops earliest — and at that tolerance the
   expanded pool is worse than no seeds on 11 of 38 cells while the *old* pool reaches the target on
   the most cells (34/38)."

Sentence the blog can also use verbatim from the source report:
> "This supports a search-efficiency claim, not a default-runtime claim: expanded seeds usually find
> good configurations much earlier while the final full-search winners remain close in aggregate."

---

## 7. Per-cell threshold table (configs) — derived here; only the 25% and 5% columns were published

`n+` = right-censored at that arm's terminal-attempt count. Within each group the order is
**no-seed / old-seed / expanded-seed**. `s` = the expanded arm's `normalized_compiler_seed_count`.
Only the 25% and 5% columns exist in the published reports; the 50% and 12.5% columns are derived here.

| Cell | s (exp seeds) | 50%: no / old / exp | 25%: no / old / exp | 12.5%: no / old / exp | 5%: no / old / exp | full: no / old / exp |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| chunk_bwd_dh_diag_fused#8:B4_T2048_H16_D128 | 5 | 100 / 1 / 1 | 100 / 1 / 5 | 105 / 192 / 5 | 287 / 192 / 5 | 630 / 460 / 464 |
| chunk_bwd_dk_delta_helion#2:B2_T16384_H16_D128 | 6 | 1 / 1 / 1 | 1 / 1 / 1 | 152 / 321 / 150 | 436+ / 373+ / 295 | 436 / 373 / 447 |
| chunk_bwd_dqk_helion#3:B8_T2048_H32_D256 | 14 | 117 / 102 / 5 | 267 / 211 / 377 | 524+ / 552+ / 595 | 524+ / 552+ / 595 | 524 / 552 / 632 |
| chunk_bwd_dqkg_scalar_helion#0:B8_T1024_H8_D64 | 11 | 2 / 3 / 5 | 32 / 33 / 5 | 124 / 33 / 42 | 177 / 156 / 133 | 459 / 564 / 427 |
| chunk_bwd_dqkw_delta_helion#2:B2_T16384_H16_D128 | 13 | 21 / 22 / 5 | 201 / 113 / 5 | 204 / 113 / 145 | 285 / 148 / 302 | 671 / 580 / 425 |
| chunk_bwd_dv_helion#0:B8_T1024_H8_D64 | 11 | 103 / 101 / 5 | 116 / 108 / 6 | 153 / 120 / 6 | 183 / 151 / 115 | 430 / 528 / 605 |
| chunk_bwd_gram2_kda_helion#2:B2_T16384_H16_D128 | 7 | 1 / 1 / 1 | 103 / 105 / 2 | 119 / 220 / 104 | 177 / 295 / 202 | 519 / 739 / 755 |
| chunk_bwd_o_kda_helion#4:B1_T8192_H96_D128 | 13 | 65 / 1 / 1 | 121 / 1 / 1 | 182 / 1 / 1 | 417 / 213 / 103 | 992 / 637 / 508 |
| chunk_bwd_state_dwk_kda_helion#2:B4_T2048_H16_D128 | 10 | 10 / 1 / 1 | 95 / 96 / 4 | 128 / 150 / 144 | 310 / 369 / 244 | 584 / 474 / 416 |
| chunk_bwd_wu_kda_helion#3:B8_T2048_H32_D256 | 8 | 1 / 1 / 1 | 27 / 1 / 1 | 27 / 1 / 1 | 27 / 1 / 1 | 494 / 328 / 332 |
| chunk_bwd_wy_dL_delta_helion#7:B4_T2048_H16_D128 | 15 | 55 / 20 / 4 | 254 / 264 / 4 | 314 / 302 / 7 | 439 / 403 / 205+ | 486 / 592 / 205 |
| chunk_cumsum_gc_helion#7:B4_T2048_H16_D128 | 4 | 8 / 1 / 1 | 8 / 1 / 1 | 72 / 72 / 75 | 246 / 158 / 229+ | 443 / 309 / 229 |
| chunk_cumsum_gc_varlen_helion#3:ragged_T8192_H96_D128 | 4 | 2 / 2 / 5 | 221 / 190 / 112 | 293 / 264 / 112 | 293 / 264 / 112 | 346 / 333 / 291 |
| chunk_fwd_A_diag_anchored_helion#0:B8_T1024_H8_D64 | 6 | 1 / 1 / 1 | 1 / 1 / 1 | 33 / 1 / 1 | 216+ / 33 / 117 | 216 / 477 / 610 |
| chunk_fwd_A_diag_anchored_varlen_helion#1:ragged_T8192_H64_D128 | 6 | 1 / 1 / 1 | 1 / 1 / 1 | 1 / 1 / 1 | 147+ / 104 / 4 | 147 / 502 / 628 |
| chunk_fwd_h_diag_fused#8:B4_T2048_H16_D128 | 5 | 104 / 1 / 1 | 104 / 1 / 1 | 113 / 1 / 1 | 113 / 1 / 1 | 585 / 316 / 329 |
| chunk_fwd_o_diag_anchored_helion#4:B1_T8192_H96_D128 | 10 | 62 / 1 / 1 | 116 / 130 / 103 | 237 / 179 / 148 | 237 / 218 / 567+ | 421 / 948 / 567 |
| chunk_fwd_o_diag_anchored_varlen_helion#0:uniform_T8192_H64_D128 | 7 | 1 / 1 / 1 | 1 / 1 / 1 | 5 / 5 / 11 | 116 / 111 / 112 | 311 / 339 / 258 |
| chunk_fwd_o_helion#2:B2_T16384_H16_D128 | 12 | 45 / 1 / 1 | 103 / 1 / 1 | 684+ / 213 / 168 | 684+ / 321 / 401+ | 684 / 370 / 401 |
| attention_backward_b2_h32_s1024_d64_f16 | 10 | 112 / 1 / 1 | 117 / 1 / 1 | 263 / 1 / 1 | 352 / 109 / 256+ | 566 / 627 / 256 |
| biased_attention_b1_h16_m1024_n2048_d128_bf16 | 7 | 147 / 1 / 1 | 160 / 128 / 7 | 312 / 143 / 330 | 380 / 324 / 416+ | 725 / 386 / 404 |
| blackwell_attention_backward_b1_h8_s256_d64_f16 | 10 | 157 / 1 / 1 | 279 / 1 / 1 | 437 / 1 / 1 | 437 / 1 / 1 | 472 / 427 / 289 |
| blackwell_attention_b4_h32_s2048_d64_bf16 | 12 | 151 / 1 / 1 | 250 / 199 / 128 | 290 / 199 / 219 | 582+ / 524 / 627 | 580 / 610 / 680 |
| causal_attention_b4_h32_s4096_d128_bf16 | 11 | 35 / 1 / 1 | 112 / 164 / 238 | 112 / 164 / 319 | 112 / 164 / 319 | 653 / 585 / 350 |
| attention_b1_h4_m512_n512_d64_f16 | 6 | 36 / 37 / 5 | 88 / 89 / 93 | 103 / 101 / 203 | 103 / 101 / 203 | 350 / 352 / 398 |
| flex_dense_b2_h32_s1024_d64_f16 | 8 | 32 / 32 / 4 | 48 / 48 / 5 | 276 / 48 / 55 | 466 / 373 / 217 | 485 / 410 / 376 |
| jagged_hstu_b2_l7168_max4096_h16_d64 | 6 | 113 / 1 / 1 | 117 / 1 / 1 | 251 / 313 / 112 | 555 / 374 / 300 | 641 / 777 / 717 |
| bf16xint16_m1_k4096_n4096 | 9 | 117 / 102 / 2 | 129 / 107 / 5 | 547+ / 107 / 134 | 547+ / 107 / 324+ | 546 / 492 / 324 |
| broadcast_matmul_b16_m512_k768_n1024 | 8 | 115 / 1 / 1 | 187 / 1 / 1 | 246 / 220 / 129 | 461 / 222 / 188 | 553 / 489 / 460 |
| gather_gemv_b64_s4096_n8 | 7 | 4 / 5 / 6 | 4 / 5 / 6 | 20 / 89 / 94 | 101 / 101 / 101 | 453 / 336 / 425 |
| jagged_bmm_b8_l255_d128_k64_bias | 7 | 245 / 1 / 1 | 531+ / 172 / 114 | 531+ / 172 / 114 | 531+ / 172 / 114 | 531 / 544 / 734 |
| se_net_m256_n512_k32 | 10 | 176 / 1 / 1 | 693+ / 126 / 185 | 693+ / 126 / 185 | 693+ / 126 / 185 | 693 / 581 / 542 |
| gdn_b8_t4096_h80_c256_d64_s128 | 6 | 60 / 60 / 5 | 60 / 60 / 65 | 60 / 60 / 65 | 60 / 60 / 65 | 630 / 492 / 657 |
| mamba_scan_b1_h64_g8_t8192_c64_d64_s128 | 7 | 218 / 1 / 1 | 218 / 1 / 1 | 219 / 809 / 309+ | 367 / 940+ / 309+ | 903 / 940 / 309 |
| packed_h12_t2048 | 7 | 4 / 1 / 1 | 5 / 5 / 11 | 257+ / 160 / 277+ | 257+ / 160 / 277+ | 257 / 240 / 277 |
| fixed_h16_t512_float32 | 6 | 21 / 1 / 1 | 21 / 1 / 1 | 60 / 1 / 1 | 67 / 61 / 72 | 336 / 315 / 283 |
| ragged_h16_t2048 | 7 | 9 / 1 / 1 | 9 / 1 / 1 | 9 / 1 / 1 | 116 / 102 / 101 | 388 / 309 / 325 |
| bf16_regular_h16_b64 | 6 | 231 / 1 / 1 | 231 / 590+ / 601+ | 611 / 590+ / 601+ | 623 / 590+ / 601+ | 656 / 590 / 601 |

## 8. Per-cell wall seconds to target — derived here; only the 5% column was published

Censored values (`+`) are that arm's full `autotune_time`. Only the 5% column (not shown here, see
`THREE_WAY_AUTOTUNE_TABLES.md`) was published; these three columns are derived here.

| Cell | 50% wall s: no / old / exp | 25% wall s: no / old / exp | 12.5% wall s: no / old / exp | full wall s: no / old / exp |
| --- | ---: | ---: | ---: | ---: |
| chunk_bwd_dh_diag_fused#8:B4_T2048_H16_D128 | 39.3 / 12.1 / 11.8 | 39.3 / 12.1 / 14.8 | 46.9 / 121.2 / 14.8 | 460.3 / 372.9 / 411.9 |
| chunk_bwd_dk_delta_helion#2:B2_T16384_H16_D128 | 10.8 / 12.4 / 12.5 | 10.8 / 12.4 / 12.5 | 157.2 / 346.8 / 230.1 | 440.6 / 422.3 / 489.0 |
| chunk_bwd_dqk_helion#3:B8_T2048_H32_D256 | 202.5 / 219.8 / 62.3 | 350.7 / 382.5 / 536.8 | 729.3+ / 857.3+ / 972.2 | 729.3 / 857.3 / 1119.7 |
| chunk_bwd_dqkg_scalar_helion#0:B8_T1024_H8_D64 | 12.6 / 15.6 / 17.2 | 26.8 / 32.2 / 17.2 | 88.7 / 32.2 / 39.7 | 446.0 / 602.8 / 455.1 |
| chunk_bwd_dqkw_delta_helion#2:B2_T16384_H16_D128 | 30.0 / 35.8 / 29.4 | 209.2 / 116.2 / 29.4 | 211.0 / 116.2 / 170.0 | 1900.0 / 817.7 / 696.1 |
| chunk_bwd_dv_helion#0:B8_T1024_H8_D64 | 53.0 / 59.2 / 12.9 | 60.2 / 63.9 / 13.3 | 96.8 / 71.1 / 13.3 | 334.9 / 415.8 / 467.9 |
| chunk_bwd_gram2_kda_helion#2:B2_T16384_H16_D128 | 28.2 / 27.8 / 28.0 | 171.5 / 161.2 / 29.1 | 219.9 / 421.3 / 162.6 | 1182.2 / 3662.5 / 1431.2 |
| chunk_bwd_o_kda_helion#4:B1_T8192_H96_D128 | 63.5 / 24.3 / 24.3 | 118.2 / 24.3 / 24.3 | 193.3 / 24.3 / 24.3 | 2312.7 / 985.7 / 876.0 |
| chunk_bwd_state_dwk_kda_helion#2:B4_T2048_H16_D128 | 21.1 / 19.9 / 20.0 | 79.0 / 86.3 / 27.1 | 134.4 / 197.1 / 167.3 | 1060.5 / 659.4 / 511.9 |
| chunk_bwd_wu_kda_helion#3:B8_T2048_H32_D256 | 52.1 / 45.2 / 44.9 | 90.3 / 45.2 / 44.9 | 90.3 / 45.2 / 44.9 | 1922.3 / 761.7 / 733.1 |
| chunk_bwd_wy_dL_delta_helion#7:B4_T2048_H16_D128 | 196.9 / 49.9 / 32.8 | 892.5 / 460.0 / 32.8 | 1143.4 / 535.8 / 36.7 | 1650.0 / 1332.6 / 381.8 |
| chunk_cumsum_gc_helion#7:B4_T2048_H16_D128 | 9.9 / 9.2 / 9.3 | 9.9 / 9.2 / 9.3 | 26.7 / 29.9 / 31.9 | 396.1 / 256.7 / 178.6 |
| chunk_cumsum_gc_varlen_helion#3:ragged_T8192_H96_D128 | 10.3 / 10.3 / 13.6 | 167.0 / 146.2 / 82.0 | 219.5 / 203.8 / 82.0 | 303.4 / 300.0 / 265.0 |
| chunk_fwd_A_diag_anchored_helion#0:B8_T1024_H8_D64 | 19.4 / 21.5 / 21.7 | 19.4 / 21.5 / 21.7 | 42.8 / 21.5 / 21.7 | 332.4 / 705.5 / 918.3 |
| chunk_fwd_A_diag_anchored_varlen_helion#1:ragged_T8192_H64_D128 | 41.0 / 43.2 / 43.7 | 41.0 / 43.2 / 43.7 | 41.0 / 43.2 / 43.7 | 484.7 / 1540.9 / 1560.6 |
| chunk_fwd_h_diag_fused#8:B4_T2048_H16_D128 | 53.3 / 13.6 / 13.5 | 53.3 / 13.6 / 13.5 | 60.6 / 13.6 / 13.5 | 550.3 / 338.8 / 367.7 |
| chunk_fwd_o_diag_anchored_helion#4:B1_T8192_H96_D128 | 35.2 / 14.7 / 14.6 | 92.6 / 105.3 / 70.4 | 206.0 / 148.3 / 124.0 | 449.1 / 783.7 / 520.2 |
| chunk_fwd_o_diag_anchored_varlen_helion#0:uniform_T8192_H64_D128 | 7.4 / 8.6 / 8.7 | 7.4 / 8.6 / 8.7 | 14.2 / 15.8 / 23.9 | 364.4 / 421.7 / 334.9 |
| chunk_fwd_o_helion#2:B2_T16384_H16_D128 | 33.3 / 11.8 / 11.8 | 65.2 / 11.8 / 11.8 | 708.5+ / 158.8 / 118.8 | 708.5 / 376.1 / 386.4 |
| attention_backward_b2_h32_s1024_d64_f16 | 126.2 / 20.2 / 20.2 | 129.8 / 20.2 / 20.2 | 352.5 / 20.2 / 20.2 | 840.5 / 1108.8 / 571.4 |
| biased_attention_b1_h16_m1024_n2048_d128_bf16 | 445.3 / 72.7 / 72.7 | 503.3 / 411.6 / 79.7 | 817.4 / 432.6 / 920.9 | 2141.4 / 1083.0 / 1096.7 |
| blackwell_attention_backward_b1_h8_s256_d64_f16 | 195.2 / 19.3 / 19.4 | 343.7 / 19.3 / 19.4 | 537.8 / 19.3 / 19.4 | 624.0 / 496.1 / 337.8 |
| blackwell_attention_b4_h32_s2048_d64_bf16 | 175.2 / 32.8 / 33.4 | 305.4 / 321.0 / 206.8 | 340.5 / 321.0 / 339.8 | 691.5 / 800.1 / 954.7 |
| causal_attention_b4_h32_s4096_d128_bf16 | 176.2 / 71.5 / 71.9 | 377.6 / 501.3 / 639.8 | 377.6 / 501.3 / 732.7 | 1721.3 / 1695.8 / 832.9 |
| attention_b1_h4_m512_n512_d64_f16 | 143.4 / 117.0 / 73.2 | 178.3 / 157.5 / 160.2 | 193.4 / 172.5 / 332.1 | 436.3 / 473.4 / 577.6 |
| flex_dense_b2_h32_s1024_d64_f16 | 102.5 / 110.7 / 75.8 | 129.6 / 142.7 / 76.5 | 482.0 / 142.7 / 148.3 | 968.7 / 890.2 / 995.5 |
| jagged_hstu_b2_l7168_max4096_h16_d64 | 1296.8 / 73.3 / 73.4 | 1374.1 / 73.3 / 73.4 | 2572.2 / 3230.5 / 859.6 | 5733.8 / 7058.8 / 6044.4 |
| bf16xint16_m1_k4096_n4096 | 36.7 / 40.2 / 9.5 | 39.5 / 41.9 / 9.7 | 312.2+ / 41.9 / 30.7 | 312.2 / 250.8 / 167.2 |
| broadcast_matmul_b16_m512_k768_n1024 | 28.9 / 8.2 / 8.3 | 61.6 / 8.2 / 8.3 | 99.5 / 95.1 / 56.7 | 267.2 / 254.1 / 233.6 |
| gather_gemv_b64_s4096_n8 | 20.5 / 23.6 / 24.3 | 20.5 / 23.6 / 24.3 | 23.6 / 57.1 / 58.0 | 241.2 / 202.7 / 334.2 |
| jagged_bmm_b8_l255_d128_k64_bias | 139.9 / 19.2 / 19.4 | 310.9+ / 99.4 / 63.0 | 310.9+ / 99.4 / 63.0 | 310.9 / 249.6 / 359.7 |
| se_net_m256_n512_k32 | 71.3 / 9.8 / 9.8 | 400.2+ / 51.5 / 90.8 | 400.2+ / 51.5 / 90.8 | 400.2 / 339.6 / 365.8 |
| gdn_b8_t4096_h80_c256_d64_s128 | 148.6 / 176.0 / 63.7 | 148.6 / 176.0 / 199.1 | 148.6 / 176.0 / 199.1 | 1047.2 / 1004.5 / 1181.1 |
| mamba_scan_b1_h64_g8_t8192_c64_d64_s128 | 185.2 / 15.6 / 15.9 | 185.2 / 15.6 / 15.9 | 185.8 / 723.8 / 328.2+ | 691.6 / 911.5 / 328.2 |
| packed_h12_t2048 | 8.2 / 7.9 / 8.1 | 8.4 / 10.2 / 13.8 | 336.9+ / 170.8 / 414.2+ | 336.9 / 331.7 / 414.2 |
| fixed_h16_t512_float32 | 36.9 / 23.3 / 24.1 | 36.9 / 23.3 / 24.1 | 74.3 / 23.3 / 24.1 | 425.2 / 461.4 / 426.6 |
| ragged_h16_t2048 | 56.3 / 42.4 / 42.9 | 56.3 / 42.4 / 42.9 | 56.3 / 42.4 / 42.9 | 794.5 / 702.2 / 763.1 |
| bf16_regular_h16_b64 | 358.0 / 54.6 / 54.5 | 358.0 / 1129.8+ / 1090.3+ | 1000.0 / 1129.8+ / 1090.3+ | 1139.3 / 1129.8 / 1090.3 |

## 9. Reproduction recipe

```
# arm → raw log
no_seed        : matmul_heuristic_perf_results/zero_seed_full_autotune/cells/<CELL>/zero_seed/autotune.csv
old_seeds      : matmul_heuristic_perf_results/multi_seed_full_autotune_ablation/cells/<CELL>/main/autotune.csv
expanded_seeds : matmul_heuristic_perf_results/multi_seed_full_autotune_ablation/cells/<CELL>/branch/autotune.csv
```
Terminal attempts = rows with `status != 'started'`, in file order. `shared_raw_best_ms` =
min `perf_ms` over the three logs. Configs-to-target = 1-based index of the first terminal row with
`perf_ms <= (1+X)*shared_raw_best`. Censoring substitutes `len(terminal rows)`. Full-run totals come
from `result.json → autotune.num_configs_tested`; full wall from
`result.json → autotune.autotune_time`. `s` = `result.json → autotune.normalized_compiler_seed_count`
of the **branch** arm.

Files read for these notes:
- `/home/dev/PYTORCH_BLOG_HEURISTICS_RESULTS.md`
- `/home/dev/local/wt-sm100-linattn/matmul_heuristic_perf_results/zero_seed_full_autotune/THREE_WAY_AUTOTUNE_SIGNIFICANT_SEED_ADVANTAGE.md`
- `…/zero_seed_full_autotune/THREE_WAY_AUTOTUNE_TABLES.md` and `.json`
- `…/zero_seed_full_autotune/THREE_WAY_AUTOTUNE_BY_WINNER.md`
- `…/zero_seed_full_autotune/RESULTS.md`, `events.jsonl`, `cells/*/{zero_seed,winner_replay}/…`
- `…/multi_seed_full_autotune_ablation/RESULTS.md`, `events.jsonl`, `cells/*/{main,branch}/…`
- `…/zero_seed_full_autotune/autotune_progress/README.md`
- `/home/dev/local/wt-sm100-linattn/MATMUL_ZERO_SEED_FULL_AUTOTUNE_OVERNIGHT_LOG.md`
- `/home/dev/local/wt-sm100-linattn/MATMUL_MULTI_SEED_FULL_AUTOTUNE_OVERNIGHT_LOG.md`
- `…/matmul_heuristic_perf_results/results.json` (quick-effort tuning durations)
- `/home/dev/local/sm100-linattn/MATMUL_HEURISTIC_HIGH_LEVEL_TRACE.md` (seed-emission stage 6)

## 10. Figures already rendered (usable in the blog)

- `…/zero_seed_full_autotune/autotune_progress/all_cells_three_way.png` (+ pairwise overlays,
  `individual_progress.pdf`, 38 per-cell PNGs). Y axis = `best raw-search latency across all three
  arms / current incumbent`, X axis = terminal config attempts; the shared best is exactly 1.00x and
  failed attempts advance X without moving Y. **This is the right figure for result (b).**
- `…/zero_seed_full_autotune/configs_to_own_5pct_vs_total_all_cells.png`,
  `configs_to_own_5pct_vs_total_equivalent_winners.png`,
  `configs_to_5pct_vs_total_equivalent_winners.png`. Careful: two of the three are the
  **"own 5%"** (self-referential) variant, not the shared-best 5% used in the tables.
