# Headline dataset #2 — constituent-kernel matmul / multi-matmul on B200

Research notes for the PyTorch blog post. Everything below was re-derived from
`results.json` (not from the prose reports) unless explicitly marked as a quote
from a `RESULTS.md`.

Primary sources:

- `/home/dev/local/wt-sm100-linattn/matmul_heuristic_perf_results/on_corpus_pretuned/results.json`
  (149 records + 20 `excluded_records`; schema `matmul-heuristic-pretuned-results/1`)
- `/home/dev/local/wt-sm100-linattn/matmul_heuristic_perf_results/on_corpus_pretuned/RESULTS.md`
- `/home/dev/local/wt-sm100-linattn/matmul_heuristic_perf_results/results.json`
  (244 records; schema `matmul-heuristic-perf-results/1`)
- `/home/dev/local/wt-sm100-linattn/matmul_heuristic_perf_results/RESULTS.md`
- Harness: `/home/dev/local/wt-sm100-linattn/scripts/matmul_heuristic_perf_experiment.py`
- Protocol: `/home/dev/local/wt-sm100-linattn/MATMUL_HEURISTIC_PERF_CURRICULUM.{md,json}`
- Reference-config corpus: `/home/dev/local/sm100-linattn/SM100_CONFIGS.json`
- Reference-corpus provenance notes: `/home/dev/local/sm100-linattn/LINATTN_HEURISTIC_IMPLEMENTATION_PLAN.md` §"in-tree known-good oracle",
  `/home/dev/local/sm100-linattn/CORPUSA_RESOLUTION_NOTES.md` §D6

Directory listing (`on_corpus_pretuned/`): `RESULTS.md`, `results.json`,
`manifest.json`, `cells/` (one JSON per cell). Parent
`matmul_heuristic_perf_results/`: `RESULTS.md`, `results.json`, `manifest.json`,
`events.jsonl`, `cells/`, `configs/{pre,post}/`, `quick_configs/`,
`autotune_logs/`, `worker_logs/`, `triton_cache/`, plus 20+ later follow-up
sub-runs (`e2e_fla*`, `zero_seed_full_autotune`, `off_corpus_*`,
`sglang_triton_head_to_head`, `tmem_regime_fix`, `mamba2_*`,
`multi_seed_*`).

---

## 1. Arm definitions — read this before quoting any ratio

### On-corpus 149-cell run (`on_corpus_pretuned/`)

Three arms, all three **replayed in one process on one bound kernel** on the
same inputs, so the comparison is config-vs-config, not tree-vs-tree.

| Arm | What it is (verified from JSON) |
|---|---|
| **Pre** | `pre_sha = 9c46dd311bcba596324fe5cba996495bda3e65fb`, `HELION_AUTOTUNE_EFFORT=none`. For **all 149/149** cells `config_origin = "default_config"`, `heuristic_name = null`, `promoted_config = null`. i.e. **the raw unseeded `_base_default_config()`** — no matmul heuristic existed for any of these kernels at the base revision. |
| **Post** | `post_sha = ccfcfbdd86e3cda4667e92bc3225185e1d4df5a7`, `HELION_AUTOTUNE_EFFORT=none`. All 149 cells `config_origin = "current_heuristic"`; 20 fired `triton_b200_formula_matmul`, 129 fired `triton_b200_multi_matmul`. Zero autotuning; the heuristic *is* the compiler default (`promoted_config` non-null on all 149). |
| **Pre-tuned** | The exact per-cell reference config. **124 cells** (`config_origin = "known_good_config_key"`) are verbatim `SM100_CONFIGS.json[cell]["sm100_config"]` — verified byte-identical for 124/124. **25 cells** (`config_origin = "shipped_helion_config"`) are the `helion.Config` objects shipped in SGLang PR #32593 (`_KDA_REPLAYSSM_BF16_CONFIG`, `_KDA_REPLAYSSM_BF16_SMALL_HEAD_CONFIG`, etc.). No tuning time is counted — it is a replay of a stored config. |

**IMPORTANT — the "Pre-tuned" arm on the 124 Helion cells is NOT a Helion
full-autotune output.** `SM100_CONFIGS.json` self-describes as
`seeded_from: "shipped sm90 AOT heuristic"`, `cells: 251`,
`geomean_speedup: 1.1814`. Each cell records `sm90_seed` (the shipped
`HELION_AUTOTUNE_EFFORT=full` H100/sm90 AOT config, verified to match
`examples/linear/_helion_aot_linear_attention_engine_cuda_sm90.py`),
an `sm100_config`, `arms_tried`, `speedup_vs_seed`, `null_spread_pct`, and a
long prose `note` describing a per-axis measured ablation climb. Aggregates:
230/251 `improved`, 21/251 `seed_kept`, **12,989 total measured arms** (median 50
per cell, range 18–120), geomean **1.1814x over the sm90 full-autotune seed**.
The project's own plan calls it "251 **hand-tuned** B200 config cells" and
explicitly warns: *"Do not replace it with the similarly named 256-config SM100
AOT table in the Helion worktree; that is a different autotuned dataset."*
Verified: for `chunk_bwd_dh_diag_fused` the shipped sm100 AOT table entry is
different from the SM100_CONFIGS entry, and `CORPUSA_RESOLUTION_NOTES.md` §D6
records that the two differ in **262/262** cases.

Safe wording: *"the best config we know of for that exact kernel and shape —
an sm90 full-autotune config then hand-climbed per cell on B200 over ~50
measured arms"*, or simply **"best-known / pre-tuned reference"**. It is a
reference at least as strong as one full-autotune run, so the direction of the
1.2046x headroom claim is not at risk — only the label is.

### Broader 244-cell run (`matmul_heuristic_perf_results/`)

| Arm | What it is |
|---|---|
| **Pre** | Same `pre_sha`, effort `none`. **Mixed**: 209/244 cells `default_config` (raw unseeded default); 35/244 `existing_heuristic` — 20 fired `triton_reduction_tile_sm100`, 14 fired `triton_b200_formula_matmul`, 1 fired `triton_b200_matmul,triton_b200_formula_matmul`. |
| **Post** | Same `post_sha`, effort `none`. 184 `multi_matmul`, 39 `formula_matmul`, 20 `reduction_tile_sm100`, 1 `matmul,formula_matmul`. |
| **Quick-autotune** | `HELION_AUTOTUNE_EFFORT=quick`, `bound.autotune(args, force=True)`, `HELION_AUTOTUNE_RANDOM_SEED=0`, **run on the post tree** — so it is *seeded by the new heuristic*, not an independent search. Completed on 178/244 cells; 66 still `pending`. |

---

## 2. On-corpus table — VERIFIED

Every number in the task prompt reproduces exactly from `results.json`.
Wins/regressions use a **>1% / <1%** threshold.

### Aggregate (149 cells)

| Group | Comparison | N | Geomean | Wins >1% | Regr >1% | re-derived? |
|---|---|---:|---:|---:|---:|---|
| All | Post / Pre | 149 | **1.5527x** | 111 | 23 | ✅ exact |
| All | Pre-tuned / Pre | 149 | **1.8704x** | 139 | 4 | ✅ exact |
| All | Pre-tuned / Post | 149 | **1.2046x** | 123 | 3 | ✅ exact |
| Helion linear attention | Post / Pre | 124 | **1.5358x** | 88 | 22 | ✅ exact |
| Helion linear attention | Pre-tuned / Pre | 124 | 1.8508x | 115 | 4 | ✅ exact |
| Helion linear attention | Pre-tuned / Post | 124 | 1.2051x | 107 | 2 | ✅ exact |
| SGLang | Post / Pre | 25 | **1.6391x** | 23 | 1 | ✅ exact |
| SGLang | Pre-tuned / Pre | 25 | 1.9705x | 24 | 0 | ✅ exact |
| SGLang | Pre-tuned / Post | 25 | 1.2022x | 16 | 1 | ✅ exact |

Extra derived stats (not in RESULTS.md):

- **Median** post/pre = 1.4803x; median pre-tuned/post = 1.1558x.
- Distribution of post/pre over 149: 111 cells >1.01, 15 cells in [0.99, 1.01],
  23 cells <0.99. 29 cells are strictly below 1.0000. 11 cells are 1.0000 ± 0.001.
- 4 cells (all `chunk_fwd_A_diag_anchored_varlen_helion`) have **byte-identical
  pre and post configs**, so their 1.0000x is by construction, not measurement
  (`accuracy.pre_vs_post.status = "identical_config"`, one shared timing).

### By heuristic (149 cells)

| Post heuristic | N | Post/Pre geomean | Wins >1% | Regr >1% | Pre-tuned/Post geomean | re-derived? |
|---|---:|---:|---:|---:|---:|---|
| `triton_b200_formula_matmul` (single contraction) | 20 | **1.9948x** | 20 | 0 | **1.1257x** | ✅ exact |
| `triton_b200_multi_matmul` (multi contraction) | 129 | **1.4935x** | 91 | 23 | **1.2173x** | ✅ exact |

The single-contraction front end is a clean sweep — 20/20 cells win by >1% and
**zero** regressions — and it also leaves less on the table (1.1257x headroom vs
1.2173x). **Every one of the 23 regressions is on the multi-matmul path.**

### Scope / exclusions

- 169 on-corpus cells were measured (169/169 complete). 149 are in scope
  ("a cell where `triton_b200_formula_matmul` or `triton_b200_multi_matmul`
  fired"); 20 are preserved under `excluded_records`.
- The 20 excluded cells are `l2norm_fwd_helion` (5),
  `sglang.kda_prefill._l2norm_qk` (5),
  `sglang.kda_prefill._gate_cumsum_operands` (5),
  `sglang.kda_decode.packed_decode_body` (5). All 20 fired
  `triton_reduction_tile_sm100` on **both** pre and post with byte-identical
  configs, so post/pre = exactly 1.0 on all 20. Their pre-tuned/pre geomean is
  1.4747x (informational only — those cells belong to the reduction story).
- Case selection: max 5 cases per kernel body. Dense linear-attention shapes are
  `B8_T1024_H8_D64`, `B4_T2048_H16_D128`, `B2_T16384_H16_D128`,
  `B8_T2048_H32_D256`, `B1_T8192_H96_D128`; varlen profiles are
  fixed-H64 / uniform-H64 / ragged-H64 / fixed-H96 / ragged-H96.
  `chunk_fwd_A_diag_anchored_varlen_helion` has only 4 cases (there is a known,
  documented hole: no tuned reference for `fixed`/`uniform_T8192_H96_D128`).

---

## 3. All 30 kernel-family rows (on-corpus)

`hl.dot` counts are static source counts, computed by AST from
`examples/linear/linear_attention_engine.py` and the SGLang
`helion/kda_prefill.py` / `helion/kda_replayssm.py`.

| Kernel | N | Post/Pre | Pre-tuned/Pre | Pre-tuned/Post | Post regr >1% | Heuristic | source `hl.dot` count |
|---|---:|---:|---:|---:|---:|---|---|
| `helion.linear_attention_engine.chunk_bwd_dh_diag_fused` | 5 | 1.7744x | 2.0799x | 1.1722x | 0 | `formula_matmul` | 1 |
| `helion.linear_attention_engine.chunk_bwd_dk_delta_helion` | 5 | **0.8635x** | 0.9981x | 1.1558x | 5 | `multi_matmul` | 2 |
| `helion.linear_attention_engine.chunk_bwd_dqk_helion` | 5 | 2.1640x | 2.9951x | 1.3841x | 0 | `multi_matmul` | 7 |
| `helion.linear_attention_engine.chunk_bwd_dqkg_scalar_helion` | 5 | 1.9238x | 2.3958x | 1.2453x | 0 | `multi_matmul` | 7 |
| `helion.linear_attention_engine.chunk_bwd_dqkw_delta_helion` | 5 | 1.6143x | 2.7357x | 1.6946x | 1 | `multi_matmul` | 6 |
| `helion.linear_attention_engine.chunk_bwd_dstate_delta_helion` | 5 | 1.7570x | 2.2190x | 1.2629x | 0 | `multi_matmul` | 7 |
| `helion.linear_attention_engine.chunk_bwd_dv_helion` | 5 | **3.0871x** | 3.7680x | 1.2206x | 0 | `multi_matmul` | 6 |
| `helion.linear_attention_engine.chunk_bwd_gram2_kda_helion` | 5 | 1.0647x | 1.2434x | 1.1678x | 0 | `multi_matmul` | 8 |
| `helion.linear_attention_engine.chunk_bwd_o_kda_helion` | 5 | 1.8068x | 2.0208x | 1.1184x | 0 | `multi_matmul` | 4 |
| `helion.linear_attention_engine.chunk_bwd_state_du_kda_helion` | 5 | 1.0204x | 1.3511x | 1.3241x | 3 | `multi_matmul` | 2 |
| `helion.linear_attention_engine.chunk_bwd_state_dwk_kda_helion` | 5 | 1.5034x | 1.8587x | 1.2363x | 0 | `multi_matmul` | 2 |
| `helion.linear_attention_engine.chunk_bwd_wu_kda_helion` | 5 | 1.2443x | 1.3169x | 1.0583x | 0 | `multi_matmul` | 6 |
| `helion.linear_attention_engine.chunk_bwd_wy_dL_delta_helion` | 5 | 1.0785x | 1.4109x | 1.3082x | 3 | `multi_matmul` | 9 |
| `helion.linear_attention_engine.chunk_cumsum_gc_helion` | 5 | 1.7303x | 1.8221x | 1.0531x | 0 | `formula_matmul` | 1 |
| `helion.linear_attention_engine.chunk_cumsum_gc_varlen_helion` | 5 | 2.3525x | 2.7137x | 1.1536x | 0 | `formula_matmul` | 1 |
| `helion.linear_attention_engine.chunk_fwd_A_diag_anchored_helion` | 5 | **0.9390x** | 1.0503x | 1.1185x | 2 | `multi_matmul` | 8 |
| `helion.linear_attention_engine.chunk_fwd_A_diag_anchored_varlen_helion` | 4 | 1.0000x | 1.0589x | 1.0589x | 0 | `multi_matmul` | 8 |
| `helion.linear_attention_engine.chunk_fwd_h_delta_helion` | 5 | 1.7694x | 2.1260x | 1.2015x | 0 | `multi_matmul` | 2 |
| `helion.linear_attention_engine.chunk_fwd_h_delta_varlen_helion` | 5 | 1.0501x | 1.4998x | 1.4282x | 2 | `multi_matmul` | 2 |
| `helion.linear_attention_engine.chunk_fwd_h_diag_fused` | 5 | 2.1923x | 2.4724x | 1.1278x | 0 | `formula_matmul` | 1 |
| `helion.linear_attention_engine.chunk_fwd_o_diag_anchored_helion` | 5 | **3.3069x** | 3.9768x | 1.2026x | 0 | `multi_matmul` | 2 |
| `helion.linear_attention_engine.chunk_fwd_o_diag_anchored_varlen_helion` | 5 | 1.0000x | 1.1523x | 1.1523x | 0 | `multi_matmul` | 2 |
| `helion.linear_attention_engine.chunk_fwd_o_helion` | 5 | **3.8856x** | 4.5880x | 1.1808x | 0 | `multi_matmul` | 3 |
| `helion.linear_attention_engine.chunk_fwd_wy_delta_helion` | 5 | 1.0603x | 1.1666x | 1.1003x | 1 | `multi_matmul` | 5 |
| `helion.linear_attention_engine.chunk_fwd_wy_delta_varlen_helion` | 5 | **0.8905x** | 1.0171x | 1.1422x | 5 | `multi_matmul` | 5 |
| `sglang.kda_prefill._chunk_output` | 5 | 1.4240x | 1.4416x | 1.0123x | 0 | `multi_matmul` | 2 |
| `sglang.kda_prefill._chunk_state` | 5 | 1.7242x | 2.0358x | 1.1807x | 0 | `multi_matmul` | 2 |
| `sglang.kda_prefill._intra_matrices_wide` | 5 | **2.1442x** | 3.1909x | 1.4882x | 0 | `multi_matmul` | 4 |
| `sglang.kda_prefill._intra_solve_recompute` | 5 | 1.6397x | 1.6642x | 1.0150x | 0 | `multi_matmul` | 0 direct; ~24 via inlined `_assemble_lower_64_inverse` + `_apply_lower_64_blocks` |
| `sglang.kda_replayssm.replayssm_decode_body` | 5 | 1.3708x | 1.9064x | 1.3908x | 1 | `multi_matmul` | 2 |

(The `Pre-tuned/Post` column is derived here; RESULTS.md's per-kernel table
omits it. The `Post` and `Pre-tuned` columns match RESULTS.md exactly.)

### Five biggest family wins (Post/Pre geomean)

1. `chunk_fwd_o_helion` — **3.8856x** (5 cells; best cell 5.5284x)
2. `chunk_fwd_o_diag_anchored_helion` — **3.3069x** (best cell 5.8398x)
3. `chunk_bwd_dv_helion` — **3.0871x** (best cell 4.4759x)
4. `chunk_cumsum_gc_varlen_helion` — **2.3525x** (formula matmul, single dot)
5. `chunk_fwd_h_diag_fused` — **2.1923x** (formula matmul, single dot)

(6th–7th for reference: `sglang.kda_prefill._intra_matrices_wide` 2.1442x,
`chunk_bwd_dqk_helion` 2.1640x.)

### Five biggest single-cell wins

| Post/Pre | Pre us | Post us | Pre-tuned us | Cell |
|---:|---:|---:|---:|---|
| 5.8398x | 3072.880 | 526.192 | 534.512 | `chunk_fwd_o_diag_anchored_helion#3:B8_T2048_H32_D256` |
| 5.5284x | 3032.992 | 548.624 | 407.968 | `chunk_fwd_o_helion#9:B8_T2048_H32_D256` |
| 4.8365x | 1089.568 | 225.280 | 192.544 | `chunk_fwd_o_helion#4:B1_T8192_H96_D128` |
| 4.6668x | 735.184 | 157.536 | 151.408 | `chunk_fwd_o_helion#2:B2_T16384_H16_D128` |
| 4.4759x | 1539.840 | 344.032 | 343.968 | `chunk_bwd_dv_helion#10:B1_T8192_H96_D128` |

Note the first row is one of only **three** cells in the whole 149 where the
heuristic **beats** the pre-tuned reference by >1% (pre-tuned/post = 0.9844x).

### Five worst families and the five worst cells

Worst families (Post/Pre geomean): `chunk_bwd_dk_delta_helion` **0.8635x**
(+15.81% latency, 5/5 cells regress), `chunk_fwd_wy_delta_varlen_helion`
**0.8905x** (+12.30%, 5/5 regress), `chunk_fwd_A_diag_anchored_helion`
**0.9390x** (+6.50%, 2/5 regress), then two families at exactly 1.0000x
(`chunk_fwd_A_diag_anchored_varlen_helion` — 4 byte-identical configs;
`chunk_fwd_o_diag_anchored_varlen_helion` — configs differ but latency lands on
the same value). The three genuine regressing families are exactly the three
listed in RESULTS.md's "Regression Analysis".

Worst individual cells:

| Post/Pre | Pre us | Post us | Pre-tuned us | Pre-tuned/Post | Cell |
|---:|---:|---:|---:|---:|---|
| **0.5358x** | 30.688 | 57.280 | 22.496 | 2.5462x | `sglang.kda_replayssm.replayssm_decode_body / bfloat16_small_head_b16` |
| **0.6978x** | 307.200 | 440.224 | 309.152 | 1.4240x | `chunk_bwd_dk_delta_helion#3:B8_T2048_H32_D256` |
| **0.7507x** | 909.152 | 1211.120 | 865.216 | 1.3998x | `chunk_fwd_A_diag_anchored_helion#9:B8_T2048_H32_D256` |
| **0.7874x** | 22.400 | 28.448 | 24.480 | 1.1621x | `chunk_bwd_dk_delta_helion#0:B8_T1024_H8_D64` |
| **0.8464x** | 327.584 | 387.040 | 240.672 | 1.6082x | `chunk_fwd_h_delta_varlen_helion#2:ragged_T8192_H64_D128` |

### Cells where the heuristic is SLOWER than Pre

**23 cells regress by >1%; 29 cells are strictly below 1.0000x.** The full list
of 23 is in `on_corpus_pretuned/RESULTS.md` §"Regressing Cells" (verified
row-for-row against `results.json`). Structure:

- 5/5 cells of `chunk_bwd_dk_delta_helion`
- 5/5 cells of `chunk_fwd_wy_delta_varlen_helion`
- 3/5 of `chunk_bwd_state_du_kda_helion`, 3/5 of `chunk_bwd_wy_dL_delta_helion`
- 2/5 of `chunk_fwd_A_diag_anchored_helion`, 2/5 of `chunk_fwd_h_delta_varlen_helion`
- 1/5 each of `chunk_bwd_dqkw_delta_helion`, `chunk_fwd_wy_delta_helion`,
  `sglang.kda_replayssm.replayssm_decode_body`
- 0 regressions on any of the 20 `formula_matmul` cells.

Mechanism (verified from the JSON configs, matching RESULTS.md's diagnosis):
across the 23 regressing cells the heuristic **raises `num_stages` in 19**,
changes `block_sizes` in 9, and changes `num_warps` in 9.

Three quotable, fully verified single-mechanism cases:

- `chunk_fwd_wy_delta_varlen_helion`: **the only config difference in all five
  cells is `num_stages: 1 -> 3`** (block `[32]` and 4 warps unchanged), and all
  five lose 9.8–11.7% (`0.8832x`–`0.9019x`). Every pre-tuned config for that
  kernel returns to one stage. Cleanest "pipeline depth that fits but doesn't
  pay" example in the dataset.
- `chunk_fwd_A_diag_anchored_helion`: `block_sizes = []` (no tunable block
  sizes at all), so the heuristic's only levers are stages/warps. It raises
  1→2 stages on every shape; the B8/T2048/H32/D256 cell also goes 4→8 warps and
  falls to `0.7507x`, and its pre-tuned config is 1 stage / 4 warps. Note the
  pre-tuned family for this kernel is 1, 1, 4, 1, 6 stages across the 5 shapes —
  i.e. there is no single right depth, and the failure is the *uniform* choice.
- `sglang.kda_replayssm.replayssm_decode_body / bfloat16_small_head_b16`
  (worst cell, `0.5358x`): pre `[32,32]`/1 stage/4 warps → post
  `[128,64]`/2 stages/4 warps; pre-tuned is `[32,128]`/1 stage/**1 warp** with
  loop order `[1,2,0]`. Post grew the tile on the wrong axis for a small-head
  decode shape. Pre-tuned is `1.3642x` vs pre and `2.5462x` vs post.
- Largest absolute headroom left on the table:
  `chunk_bwd_dqkw_delta_helion#9:B8_T2048_H32_D256` — pre 8213.616 us, post
  9059.232 us (`0.9067x`), pre-tuned **2232.224 us** (`3.6796x` vs pre,
  `4.0584x` vs post). Post grew only the first three of five block sizes
  (`[16,16,16,16,16]` → `[64,64,64,16,16]`); pre-tuned uses
  `[128,128,64,256,128]`, 8 warps, loop order `[1,0]`, and range scheduling.

Fairness caveat that RESULTS.md itself states and I confirmed: **the pre-tuned
configs also change indexing, loop order, eviction policy and range scheduling**,
so "pre-tuned/post" headroom is not attributable to block sizes / stages / warps
alone. Only the pre→post diffs isolate the heuristic's decisions.

### Cells where the heuristic BEATS the best-known config (3 cells, >1%)

| Pre-tuned/Post | Cell |
|---:|---|
| 0.9435x | `chunk_fwd_h_diag_fused#8:B4_T2048_H16_D128` (formula matmul; 34.720 us vs 36.800 us) |
| 0.9752x | `sglang.kda_prefill._chunk_output / fixed_h32_t8192` (81.760 us vs 83.840 us) |
| 0.9844x | `chunk_fwd_o_diag_anchored_helion#3:B8_T2048_H32_D256` (526.192 us vs 534.512 us) |

### Cells where the pre-tuned reference is SLOWER than Pre (4 cells, >1%)

`chunk_bwd_dk_delta_helion#0` (0.9150x), `chunk_bwd_dk_delta_helion#4`
(0.9584x), `chunk_fwd_wy_delta_helion#0` (0.9620x),
`chunk_fwd_A_diag_anchored_helion#7` (0.9812x). Useful honesty point: on a few
tiny/awkward cells the raw default is already near-optimal and the whole
exercise is a wash.

---

## 4. Broader 244-cell corpus — VERIFIED

| Comparison | N | Geomean | Wins >1% | Regr >1% | re-derived? |
|---|---:|---:|---:|---:|---|
| Post / Pre | 244 | **1.4547x** | 159 | 35 | ✅ exact |
| Quick / Pre | 178 | **1.8997x** | 155 | 8 | ✅ exact |
| Quick / Post | 178 | **1.3283x** | 125 | 1 | ✅ exact |

Corpus composition: 169 on-corpus cells over 34 bodies (26
`helion.linear_attention_engine` + 8 `sglang`) + 75 off-corpus cells over 15
bodies × 5 cases (`off.attention.*`, `off.matmul.*`, `off.state_space.*`).
244 cells over 49 kernel bodies. Median post/pre = 1.4258x. 42/244 cells are
exactly 1.0000x (37 of those have byte-identical pre/post configs).

### What "quick-autotune" is, exactly

`HELION_AUTOTUNE_EFFORT=quick`, default autotuner = `LFBOTreeSearch`, seed 0,
`force=True`, run **on the post tree**. From
`helion/autotuner/effort_profile.py`:

| Knob | `quick` | `full` |
|---|---|---|
| (lfbo) pattern search initial population | **30** | 100 |
| copies | **2** | 5 |
| max generations | **5** | 20 |
| initial population strategy | `from_best_available` (pad-random **off**) | `from_random` |
| differential evolution | pop 20 × 8 gens | pop 40 × 40 gens |
| random search count | 100 | 1000 |
| LLM search | `max_rounds=1` | `max_rounds=4` |
| `finishing_rounds` | 0 | 0 |
| `rebenchmark_threshold` | 0.9 (i.e. rebenchmarking effectively **disabled**) | 1.5 |

So quick is ~3.3x smaller initial population, 4x fewer generations, no
re-benchmark of the winner, and it starts `from_best_available` — meaning
**quick-autotune is warm-started from the new heuristic's own seed**. It is
neither the full-autotune ceiling nor an independent search.

Measured tuning cost of the quick arm: 178 completed cells,
**median 118.3 s, mean 140.9 s, min 27.8 s, max 642.9 s, 6.97 GPU-hours total.**
(This makes the "no tuning at all" contrast concrete: the heuristic's cost is
one compile-time analysis pass.)

Direct evidence that quick < the best-known reference, measured on the
**118 cells present in both runs** (149-cell scope ∩ quick-complete):

| Comparison on the same 118 cells | Geomean |
|---|---:|
| Quick / Post | 1.1734x |
| Pre-tuned (best-known) / Post | **1.2022x** |
| latency(quick) / latency(pre-tuned) | **1.0262x** — pre-tuned faster on 57 cells, quick faster on 24, tied on 37 |
| Post / Pre in broader run | 1.5146x |
| Post / Pre in pretuned run (same cells) | 1.5098x |

The last two lines are a useful cross-run reproducibility check: the same 118
cells give 1.5146x and 1.5098x in two independent measurement passes — **0.3%
run-to-run drift.**

### Why 1.3283x (quick/heuristic) must NOT be read as "the autotuner is 33% better"

The 178-cell quick cohort is a *different, off-corpus-heavy* set:

| Cohort | N (quick complete) | Post/Pre | Quick/Post | Quick/Pre |
|---|---:|---:|---:|---:|
| on-corpus | 133 | 1.4453x | **1.1896x** | 1.7194x |
| off-corpus | 45 | 1.3864x | **1.8399x** | 2.5507x |

The whole gap is off-corpus, and it is dominated by a handful of cells where the
heuristic itself catastrophically regresses (see below). On the on-corpus
constituent kernels — the blog's subject — quick-autotune buys only 1.19x over
the heuristic, and the best-known configs buy 1.20x.

Also note the 66 pending quick cells are **not random**: phase 2 was
round-robin by case ordinal across all 49 bodies, and 4 of 5 rounds finished, so
the missing cells are the 4th/5th case of each body (30 off-corpus + 36
on-corpus). Off-corpus bodies each have only 3/5 cases tuned.

### Where the 1.4547x comes from: split by pre-arm origin (important!)

| Pre arm | N | Post/Pre geomean | Wins >1% | Regr >1% |
|---|---:|---:|---:|---:|
| `default_config` (raw unseeded default) | 209 | **1.5509x** | 159 | 33 |
| `existing_heuristic` (a heuristic already fired pre-change) | 35 | **0.9925x** | **0** | 2 |
| — of which `triton_reduction_tile_sm100` | 20 | 1.0000x (20/20 byte-identical configs) | 0 | 0 |
| — of which pre-existing `triton_b200_formula_matmul` | 14 | 0.9814x (12/14 byte-identical configs) | 0 | 2 |

**The headline is entirely a "heuristic vs. no heuristic" number.** On the 35
cells where a matmul/reduction heuristic already existed at the base revision,
this branch is a wash (0.9925x). The two non-identical cells there are
`broadcast_matmul_b32_m1_k4096_n4096` (0.9087x; 20.384 → 22.432 us) and
`bf16xint16_m1_k4096_n4096` (0.8460x; 22.496 → 26.592 us) — both M=1 GEMV
shapes at ~20 us, i.e. 1–2 steps of the timer lattice (see §6).

Sensitivity: 244-cell geomean excluding the 20 unchanged reduction cells =
1.5042x; excluding `jagged_hstu` = 1.5270x.

### Off-corpus per-family rows (broader run) — the honest bad news

| Kernel | N | Post/Pre | Nq | Quick/Post | Quick/Pre | pre-heuristic |
|---|---:|---:|---:|---:|---:|---|
| `off.attention.backward` | 5 | 1.9527x | 3 | 1.1602x | 2.0099x | none |
| `off.attention.biased_forward` | 5 | 1.5516x | 3 | 2.0511x | 2.6842x | none |
| `off.attention.blackwell_backward` | 5 | 2.1366x | 3 | 1.1186x | 2.2521x | none |
| `off.attention.blackwell_forward` | 5 | 3.4856x | 3 | 1.3042x | 3.7409x | none |
| `off.attention.causal_forward` | 5 | 1.3427x | 3 | 3.0271x | 3.0960x | none |
| `off.attention.dense_forward` | 5 | 1.5492x | 3 | 8.2814x | 5.9541x | none |
| `off.attention.flex_forward` | 5 | 1.4042x | 3 | 1.3674x | 2.2742x | none |
| `off.attention.jagged_hstu` | 5 | **0.1434x** | 3 | **17.4333x** | 3.4362x | none |
| `off.matmul.bf16xint16` | 5 | 0.9671x | 3 | 1.1450x | 1.0829x | formula_matmul |
| `off.matmul.broadcast` | 5 | 0.9810x | 3 | 1.1022x | 1.0676x | formula_matmul |
| `off.matmul.gather_gemv` | 5 | 2.0633x | 3 | 1.0299x | 3.3129x | none |
| `off.matmul.jagged_dense_bmm` | 5 | 1.0000x | 3 | 1.3455x | 1.3455x | formula_matmul |
| `off.matmul.squeeze_excitation_forward` | 5 | 1.4520x | 3 | 2.4580x | 3.9823x | none |
| `off.state_space.gdn_fwd_h` | 5 | 2.0043x | 3 | 1.0028x | 2.4497x | none |
| `off.state_space.mamba2_chunk_scan` | 5 | 3.0688x | 3 | 1.0480x | 3.8242x | none |

**`off.attention.jagged_hstu` is a 7x slowdown** (family geomean 0.1434x;
individual cells 0.0660x, 0.0879x, 0.0902x, 0.0915x, 1.2687x — i.e. up to a
**15x** regression). The 4 cells with post/pre < 0.5 in the whole 244-cell run
are all jagged HSTU. Four `off.attention.dense_forward`/`causal_forward` cells
also regress (0.5510x–0.8306x). If the blog quotes the 244-cell number it should
say off-corpus attention was not the target and had known regressions at that
SHA — and note the later work-model follow-up partly fixed them (§7).

Biggest single off-corpus win: `off.attention.dense_forward /
attention_b4_h32_m4096_n4096_d128_bf16` at **8.1914x**.

---

## 5. Which heuristic fires where — corpus structure

The front-end split is structural, not name-based
(`helion/_compiler/autotuner_heuristics/triton.py`):
`TritonB200FormulaMatmulHeuristic` ("front end 1", single contraction) owns a
kernel when there is one understandable contraction with known per-program
M/N/K extents (including the generalized fixed-full-extent axis form);
`TritonB200MultiMatmulHeuristic` ("front end 2") explicitly **declines whenever
front end 1 fires** ("so exactly one of the two owns any given kernel"), and
requires at least one dot with known M/N/K. Both decline all-FP8 contractions
(Helion cannot force full-precision fp8 accumulate under a promoted default).
Both promote their seed to the no-autotune compiler default.

Empirically on this corpus the split lines up **exactly** with the static
`hl.dot` count:

- **`formula_matmul` (single contraction), 20 cells / 4 families — all have
  exactly 1 `hl.dot`:** `chunk_bwd_dh_diag_fused`, `chunk_fwd_h_diag_fused`,
  `chunk_cumsum_gc_helion`, `chunk_cumsum_gc_varlen_helion`.
- **`multi_matmul`, 129 cells / 26 families — all have ≥2 `hl.dot`** (2 to 9 in
  the Helion engine; `sglang._intra_solve_recompute` inlines ~24 via
  `_assemble_lower_64_inverse` + `_apply_lower_64_blocks`, a blocked 64×64
  lower-triangular inverse).
- **0 `hl.dot` → the reduction heuristic** (`triton_reduction_tile_sm100`):
  `l2norm_fwd_helion`, `sglang._l2norm_qk`, `sglang._gate_cumsum_operands`,
  `sglang.kda_decode.packed_decode_body` — these are the 20 excluded cells.

Off-corpus: `off.matmul.bf16xint16`, `off.matmul.broadcast`,
`off.matmul.jagged_dense_bmm`, `off.matmul.gather_gemv` are formula-matmul
(single-contraction) families; all `off.attention.*`, `off.state_space.*`, and
`off.matmul.squeeze_excitation_forward` are multi-matmul.

Workload provenance of the 124 Helion cells (from `pretuned.capture`):
`dense_kda` 33, `varlen_kda` 24, `dense_delta_rule` 21, `dense_full_gla` 17,
`dense_vanilla_linear_attn` 11, `dense_gated_delta_rule` 10, `dense_retention` 7,
`fused_kda` 1. The 25 SGLang cells span fixed / packed-varlen / ragged prefill
plus regular and small-head ReplaySSM decode, in FP32 and BF16 state dtypes.

---

## 6. Hardware and timing methodology (exact)

From `results.json.measurement_metadata` and `_bench_configs()` in
`scripts/matmul_heuristic_perf_experiment.py`:

- **Device:** one NVIDIA B200 (sm100), **physical GPU 1**, launched with
  `CUDA_VISIBLE_DEVICES=1`, so it is `cuda:0` in-process. `num_sm = 148`;
  `shared_memory_per_block_optin = 232,448 B`; TMEM = 128 lanes × 512 columns.
- **Same-process, same-bound-kernel replay.** One `kernel.bind(args)`, then
  `bound.set_config(cfg)` per arm. All three configs are compiled and warmed
  **outside** the timed region.
- **CUDA graphs:** `_make_cudagraph_replay` per config;
  `cuda_graph_status = "captured"` for **all 149 × 3 arms** (no fallbacks).
- **Cold L2:** Triton's `driver.clear_cache(empty_cache_for_benchmark)` before
  **every individual timed call**.
- **Rotated / interleaved CUDA events:** 5 rounds; the arm order is rotated by
  round index; within a round, `repeat` calls alternate across arms. Per round:
  median of the per-call event times. Reported `latency_us` = **median of the 5
  round medians**.
- **Repeat count** = `clamp(ceil(25 ms / pilot_ms), 10, 1000)`
  (`target_ms_per_arm_round = 25.0`). Measured range over the 149 cells:
  min 10, median 67, max 1000.
- **Accuracy gate before timing**: `gate.compare.check_two_configs` with
  `rtol = 5e-2`, `atol = 1e-1`. Result on 149 cells: 145 `pass` + 4
  `identical_config` for pre-vs-post; 149 `pass` for pre-vs-pre-tuned. **Zero
  failures.**
- **Byte-identical configs share one timing** (4 cells) and are recorded via
  `shared_measurement_with`.
- Reproducibility within a run is excellent: **per-arm round-median spread is
  median 0.04%, p90 0.36%, max 5.62%.**

### Timer lattice caveat (my own analysis, not in RESULTS.md)

The reported latencies quantize onto a ~2.02–2.08 us lattice: the distinct small
values are 12.192, 14.208, 16.288, 18.304, 20.352, 22.400, 24.448, 26.464,
28.448 us. That is the cudaEvent/dispatch floor, not kernel time resolution.
Consequence: **on the 10 cells with pre < 30 us, one lattice step is 7–15% of
the measured value**, and on the 27 cells with pre < 60 us it is 3–15%. Cells
below ~60 us should not carry a precise percentage claim. The 149-cell pre
latency distribution is min 14.208 us, p25 107.4 us, median 372.8 us, p75 949.8
us, max 8213.6 us, so the geomean is dominated by cells well above the lattice.
Examples of lattice-limited rows to avoid quoting precisely:
`sglang._chunk_output / fixed_h16_t512` (all three arms exactly 14.208 us → all
ratios 1.0000x), and the two `existing_heuristic` M=1 "regressions" in §4.

---

## 7. Corrections, must-not-merge, and staleness

### Corrections to `/home/dev/PYTORCH_BLOG_HEURISTICS_RESULTS.md`

1. **"Full-autotuned / Pre" and "Full-autotuned / heuristic" are mislabeled for
   this dataset** (index §3 table and the Quoting Note "`AOT` and `pre-tuned`
   are configurations produced by full autotuning"). Verified: the 124 Helion
   cells' pre-tuned configs are verbatim `SM100_CONFIGS.json[cell].sm100_config`
   — a **hand-tuned** per-cell corpus (the project's own plan says "251
   hand-tuned B200 config cells"), seeded from the sm90 `EFFORT=full` AOT config
   and climbed with a median of 50 measured arms per cell. The 25 SGLang cells
   are SGLang PR #32593's **shipped** `helion.Config` objects. The shipped
   Helion sm100 AOT table (`examples/linear/_helion_aot_..._sm100.py`, which
   *is* documented as `EFFORT=full`) is a **different** dataset — it differs from
   `SM100_CONFIGS.json` in 262/262 cases. Recommended label: **"pre-tuned
   best-known config"** / **"per-cell tuned reference"**.
2. **"Pre" in *this* dataset IS the raw unseeded default.** Index §1 correctly
   warns that in the *e2e linear-attention* run `Pre` is the old compiler
   heuristic and not `_base_default_config()`. That warning does **not** apply
   here: for all 149 on-corpus matmul cells, `pre.heuristic_name = null`,
   `pre.promoted_config = null`, `config_origin = "default_config"` — the raw
   base default. In the broader 244-cell run, `Pre` is a **mix** (209 raw
   default, 35 pre-existing heuristic), and the 1.4547x headline is 1.5509x on
   the 209 and 0.9925x on the 35.
3. **"Quick-autotune is not the full-autotune ceiling" understates the issue.**
   Quick is also (a) *seeded by the very heuristic it is being compared to*
   (`initial_population_strategy = from_best_available` on the post tree), and
   (b) measured on a **different, off-corpus-heavy 178-cell subset**. The
   1.3283x figure is 1.1896x on-corpus and 1.8399x off-corpus. Do not present
   1.3283x as the headroom over the heuristic on the linear-attention corpus.

### Do not numerically merge

- The **149-cell on_corpus_pretuned run** and the **244-cell broader run** are
  separate measurement passes over overlapping cells with independently
  re-measured pre/post latencies (e.g. `chunk_bwd_dh_diag_fused#1` pre = 24.448
  us in one and 24.544 us in the other). Never average their aggregates. Their
  agreement on shared cells is 1.5098x vs 1.5146x (0.3%), which is a *check*,
  not a licence to pool.
- These constituent-kernel numbers are **isolated per-kernel config replays with
  CUDA graphs and cold L2**. The e2e linear-attention numbers (`e2e_fla_v3/`)
  use **normal eager dispatch, 3 rounds** on whole operations. Different timing
  scope; keep separate. Likewise the SGLang four-arm operation benchmark uses
  CUDA-graph operation timing over a different arm set.
- The 20 excluded reduction cells belong to the reduction story, not here.
- `pretuned` here ≠ `AOT` in the e2e report (different config tables).

### Staleness / reproducibility (flag this)

- The measurements were taken at `post_sha = ccfcfbdd8...` (files dated
  **Aug 17**). The current worktree HEAD is **`40151a23f`** on branch
  `calebmkim/stack/50`, and `ccfcfbdd8` is **not an ancestor of HEAD** — the
  branch was rebased/squashed from a 12-commit development line into
  `9d835ee5f` + `40151a23f`. `helion/_compiler/autotuner_heuristics/triton.py`
  differs by **+554 / −270 lines** between the measured SHA and HEAD.
- Follow-up work landed *after* the run and **changed the emitted configs**:
  `MULTI_MATMUL_WORK_MODEL_FOLLOWUP_RESULTS.md` (Aug 21) reports generated
  defaults changed on **41/432 linear-attention cells** (geomean over those
  changed pairs **0.913x**, 21 wins / 18 regressions) and **22/75 off-corpus
  cells** (geomean **1.401x**, incl. `causal_attention` +1.78x…+3.65x,
  `squeeze_excitation` up to 5.62x, `jagged_hstu` ~1.07x). So the current tree
  would likely score *better* off-corpus and *marginally worse* on the changed
  linear-attention cells than the tables above.
- **There is no re-run of the 149-cell pre-tuned comparison at HEAD** in these
  directories.

---

## 8. Quick-reference number sheet

- 149 cells, 30 kernel families, B200 GPU 1, zero autotuning at compile time.
- Heuristic vs raw unseeded default: **1.5527x** geomean (median 1.4803x),
  111 wins >1%, 23 regressions >1%.
- Best-known pre-tuned config vs raw default: **1.8704x**.
- Best-known pre-tuned config vs heuristic: **1.2046x** — i.e. the compile-time
  heuristic captures **~83%** of the tuned config's speedup over the default
  in geomean-log terms (`ln 1.5527 / ln 1.8704 = 0.702`; on the ratio scale
  1.5527/1.8704 = 0.830).
- Linear attention (124): 1.5358x / 1.8508x / 1.2051x.
- SGLang KDA (25): 1.6391x / 1.9705x / 1.2022x.
- Single-contraction front end (20 cells): **1.9948x**, 20/20 wins, 0
  regressions, only 1.1257x left to the tuned config.
- Multi-contraction front end (129 cells): **1.4935x**, 91 wins, all 23
  regressions, 1.2173x left to the tuned config.
- Broader 244 cells: **1.4547x** (1.5509x on the 209 cells that previously had
  no heuristic; 0.9925x on the 35 that did).
- Quick-autotune (178 cells, ~2 min median tuning each, 6.97 GPU-h total):
  1.8997x vs pre, 1.3283x vs heuristic — but 1.1896x vs heuristic on-corpus and
  1.8399x off-corpus.
- Best cell: 5.8398x (`chunk_fwd_o_diag_anchored_helion#3:B8_T2048_H32_D256`,
  3072.880 → 526.192 us). Worst cell: 0.5358x
  (`replayssm_decode_body / bfloat16_small_head_b16`, 30.688 → 57.280 us).
- Dominant regression mechanism: excess `num_stages` (raised in 19 of the 23
  regressing cells).
