# RESULT-QUESTION (c): Where the heuristics mispredict — honest failure catalogue

Research notes for the PyTorch blog post on Helion's compile-time config heuristics.
Everything below is re-derived from raw artifacts (`results.json` / `summary.json` /
per-cell JSON / source) unless explicitly marked as prose-only.

**Tree**: `/home/dev/local/wt-sm100-linattn`, branch `calebmkim/stack/50`,
HEAD `40151a23f` ("[autotuner] harden multi-matmul work and grid modeling"),
with ~1,713 lines of *uncommitted* working-tree changes to
`helion/_compiler/autotuner_heuristics/triton.py`, `device_ir_analysis.py`,
`autotuner/config_spec.py`, `test/test_matmul_heuristics.py`.

---

## 0. Arm definitions used below (do not blur these)

| Label | Exact meaning |
|---|---|
| `pre` / pre-change | Config selection behaviour at base commit `9c46dd311`. May be an *existing* heuristic or the compiler default — recorded per cell as `existing_heuristic` vs `default_config`. **Not** the raw unseeded default. |
| `post` / heuristic | The new formula-matmul / multi-matmul heuristic's rank-0 config, used as the no-autotune compiler default. |
| unseeded default | `ConfigSpec._base_default_config()` with `HELION_DISABLE_AUTOTUNER_HEURISTICS=1`. |
| AOT / pre-tuned | A config produced by full autotuning (or a shipped hand-tuned config), replayed with **zero** tuning time counted. |
| quick-autotune | Helion quick-autotune mode, run on a clean pinned `main` so the branch seed never enters the search. |
| full-autotune (unseeded) | Full-effort `LFBOTreeSearch`, `HELION_AUTOTUNER_INITIAL_POPULATION=from_random`, heuristics disabled, no cached/author seeds. 22.016 GPU-hours for 75 off-corpus cells. |
| `torch.compile` | **Default mode** in the reduction / pointwise / linear-attention audits. **NOT default mode** in the 49-cell formula-matmul report — see Correction C4. |
| vLLM / SGLang shipped | The *same Helion kernel body* with vLLM's/SGLang's shipped config. Never a native CUDA kernel. |

Key sources:
- `matmul_heuristic_perf_results/on_corpus_pretuned/{RESULTS.md,results.json}` (149 matmul cells)
- `matmul_heuristic_perf_results/e2e_fla_v3/{RESULTS.md,results.json}` (96 e2e cells)
- `matmul_heuristic_perf_results/off_corpus_full_autotune_unseeded/RESULTS.md` (75 off-corpus)
- `matmul_heuristic_perf_results/off_corpus_main_autotune/RESULTS.md` (75 off-corpus, earlier tree)
- `matmul_heuristic_perf_results/multi_seed_quality_50/{REPORT.md,PRIMARY_VS_POOL_BREAKDOWN.md,results.json}`
- `matmul_heuristic_perf_results/tmem_regime_fix/`, `.../mamba2_chunk_scan_followup/REPORT.md`
- `/home/dev/local/wt-b200-reduction-audit-run/perf-repro/b200-reduction-audit/results/full/{SUMMARY.md,summary.json}` (114 reduction cells)
- `/home/dev/local/wt-b200-perf-report-repro-plan/perf-repro/results/{SUMMARY.md,PERF_TABLES.md}` (455-cell repro+generalization)
- `/home/dev/local/wt-b200-pointwise-audit/perf-repro/b200-pointwise-audit/results/full/{SUMMARY.md,summary.json}` (36 pointwise cells)
- `/home/dev/local/sm100-linattn/MATMUL_HEURISTIC_HIGH_LEVEL_TRACE.md` (two appendices = the author's own limitation list)
- `/home/dev/local/sm100-linattn/TUNABLE_BK_GRADED_STAGE_ABLATION.md`
- `helion/_compiler/autotuner_heuristics/triton.py` (fp8 decline comment L3247-3274; `_register_live_bytes` L1570-1624)

---

## 1. The residual full-autotune gap: is it concentrated or spread out?

### 1.1 Matmul cells (149 on-corpus, pre-tuned / heuristic)

Headline `1.2046x` reproduces exactly from `results.json` (geomean of per-cell
`pretuned_vs_post`). Distribution re-derived from the 149 raw ratios:

| Band (pretuned/post) | Cells |
|---|---:|
| < 1.00 (heuristic *wins*) | 8 |
| 1.00 – 1.01 | 18 |
| 1.01 – 1.05 | 12 |
| 1.05 – 1.10 | 14 |
| 1.10 – 1.25 | **53** |
| 1.25 – 1.50 | 30 |
| 1.50 – 2.00 | 10 |
| ≥ 2.00 | 4 |

median `1.1558`, p25 `1.0479`, p75 `1.2996`, min `0.9435`, max `4.0584`.
26/149 cells have `pretuned/post < 1.01` (AOT at most 1% faster — this bucket
includes the 8 cells where the heuristic is faster); 38/149 have `< 1.05`;
97/149 are `≥ 1.10`.

**Verdict: mostly SPREAD OUT with a modest long tail.** Top-5 cells hold only
16.4% of the total log-gap; top-10 → 26.3%; top-30 → 52.9%; top-75 → 85.7%.
There is no small set of pathological cells you can excise to make the gap go away.
The *latency-weighted* gap is bigger than the geomean: Σ post 73,277 µs vs
Σ pre-tuned 53,402 µs = **1.372x**.

Where it is worst, by shape and by front end:

| Slice | Cells | pretuned/post geomean |
|---|---:|---:|
| head dim D256 (`B8_T2048_H32_D256`) | 20 | **1.3688** |
| head dim D128 | 84 | 1.1794 |
| head dim D64 | 20 | 1.1618 |
| SGLang (`h*_t*` cases) | 25 | 1.2022 |
| `triton_b200_formula_matmul` (single contraction) | 20 | **1.1257** |
| `triton_b200_multi_matmul` | 129 | 1.2173 |

So: the single-contraction front end is much closer to AOT than the
multi-contraction one, and the gap grows with head dimension / kernel size.

### 1.2 End-to-end linear attention (96 cells, AOT / heuristic)

Recomputed per-cell from `e2e_fla_v3/results.json`: forward `1.1404x` (n=54),
forward+backward `1.1710x` (n=42), all-96 `1.1537x`, median `1.1445`.
**Here the gap IS concentrated:**

- 15 `B8_T2048_H32_D256` cells: **1.3705x** → 34.4% of the total log-gap.
- 12 `gated_delta_rule` cells: 24.4% of the log-gap
  (fwd+bwd `1.4085x`, fwd `1.2408x` — the worst variant by a wide margin).
- Union of those two groups (25 of 96 cells) = **50.4% of the whole e2e gap.**
- The *smallest* shape `B8_T1024_H8_D64` has an AOT/heuristic **group geomean of
  `0.9942x` over its 15 cells** — i.e. on that shape the heuristic is, in
  aggregate, already at parity with the full-autotuned AOT config. (Group
  geomean, not a per-cell claim.)
- In 14 of the 96 individual cells the heuristic is faster than AOT.
- Worst single cell: `gated_delta_rule / forward_backward / B8_T2048_H32_D256`
  at `2.1796x` (post 16,254.7 µs vs AOT 7,457.7 µs).

**One-line story for the blog**: at the *kernel* level the residual is a broad
~15% tax; at the *model* level it collapses onto big-state (D256) and
gated-delta-rule backward.

### 1.3 The same gap measured off-corpus is 2-3x larger

| Reference | Population | Reference / heuristic |
|---|---:|---:|
| Shipped/AOT pre-tuned, on-corpus | 149 | 1.2046x |
| Unseeded **full** autotune, off-corpus | 75 | **1.4477x** |
| ↳ `squeeze_excitation_forward` | 5 | **4.3268x** |
| ↳ `causal_forward` | 5 | 2.7731x |
| ↳ `flex_forward` | 5 | 1.7070x |
| ↳ `dense_forward` | 5 | 1.5372x |
| ↳ best kernel (`attention.backward`) | 5 | 1.0667x |

On the same 75 off-corpus cells the heuristic is also *worse than a plain
quick-autotune done on main*: `current heuristic / quick = 0.8521x`
(`off_corpus_full_autotune_unseeded/RESULTS.md`).

### 1.4 A big share of the gap is in knobs the heuristic does not even set

The heuristic owns exactly four fields: `block_sizes`, `num_warps`,
`num_stages`, `l2_groupings`. Field-level diff of post vs pre-tuned over all
149 on-corpus cells (after normalising empty-list ≡ all-None/all-zero):

| Field | Cells where post ≠ pre-tuned |
|---|---:|
| `indexing` | 142 |
| `load_eviction_policies` | 117 |
| `range_unroll_factors` | 106 |
| `num_stages` | 101 |
| `range_multi_buffers` | 92 |
| `range_num_stages` | 91 |
| `block_sizes` | 84 |
| `range_flattens` | 81 |
| `num_warps` | 78 |
| `loop_orders` | 66 |
| `l2_groupings` | 61 |
| `pid_type` | 34 |
| `atomic_indexing` | 17 |
| `num_sm_multiplier` | 9 |
| `maxnreg` | 4 |

**148 of 149 cells differ in at least one field outside the four owned knobs;
only 1 cell differs *only* in owned knobs; 5 cells differ *only* in un-owned
knobs; 0 cells are identical.** The 1.20x therefore cannot be read as "the tile
formula is 20% off" — it is an upper bound that also contains loop order,
indexing (tensor descriptors), range pipelining, eviction hints, PID schedule and
SM oversubscription, none of which any of the four heuristics predict.
`multi_seed_quality_50/PRIMARY_VS_POOL_BREAKDOWN.md` says this explicitly:
"an auxiliary-policy miss is not evidence that the matmul formula predicted that
policy incorrectly; the current seed path did not try to predict it."

### 1.5 The independent 50-cell random-sample quantification

`multi_seed_quality_50` samples 50 random (kernel, shape) cells and measures
`coverage = reference latency / seed latency`:

- **Shipped rank-0 primary coverage geomean `0.7859x`** (n=49; 1 cell's primary
  failed the accuracy gate) → the reference is 1.272x faster. Median `0.7997`,
  min `0.4666`.
- 17/49 cells below `0.75` coverage; only 11/49 at or above `0.95`.
- Best-of-the-entire-13-seed-pool coverage is only `0.8411x` (n=50).
- Against genuine unseeded full-autotune references (16 cells): primary
  `0.7284x`, pool-best `0.8324x`. Against pretuned known-good references
  (34 cells): primary `0.8128x`, pool-best `0.8452x`.
- **`exact_match` count = 0/50.** No cell's seed pool contains the reference config.

Structured miss taxonomy over the 20 hand-diagnosed cells in that report:

| Tag | Miss | Cells |
|---|---|---:|
| TG | no valid seed emitted the reference `block_sizes` vector | 14/20 |
| LS | no valid seed emitted the reference `(num_warps, num_stages)` pair | 15/20 |
| GS | winner differed in loop order / L2 grouping / PID type / SM multiplier | 12/20 |
| AP | winner differed in range transforms / indexing / eviction / `maxnreg` | **20/20** |
| JM | merged/focal recipes never formed the asymmetric whole-kernel block vector | 10/20 |
| SR | mixed-dtype / GEMV / jagged / persistent regime outside the seed frontier | 5/20 |

and the report's own conclusion: **"none of the 20 pools contained the reference
`(B,W,S)` tuple"** — a useful tile appears in one recipe and a useful warp/stage
point in another, but never their combination.

---

## 2. The catalogue

Each entry: (i) misprediction, (ii) measured evidence + source, (iii) root cause
in the model, (iv) accepted approximation vs open bug.

### M1 — `num_stages`: "deepest feasible", not "most profitable"

(i) The stage solver is a **feasibility** solver: it picks the deepest pipeline
depth that loop trips, estimated shared memory and estimated CTA residency
permit, and never asks whether the extra stage overlaps enough useful loading to
repay its prologue/epilogue, barrier and register cost.

(ii) Evidence, three independent measurements:

- On-corpus: post raises `num_stages` vs pre in **117 of 149** cells. That is on
  average *good* — `post/pre` geomean is `1.5863x` where stages were raised vs
  `1.4355x` where they were not. But of the **23 cells that regress >1%,
  post raises `num_stages` in 19** (block sizes change in 9, warps in 9, and
  post never *lowers* stages in a regressing cell)
  (`on_corpus_pretuned/results.json`, re-derived; matches RESULTS.md prose).
- Cleanest single case: `chunk_fwd_wy_delta_varlen_helion`, all 5 shapes. The
  *only* config difference from pre is `num_stages: 1 → 3`; block `[32]` and 4
  warps are unchanged. All five cells lose 9.8–11.7% (`0.8832x`–`0.9019x`), and
  **every** pre-tuned config for that kernel goes back to one stage and usually
  unrolls the enclosing range.
- The opposing ablation (`TUNABLE_BK_GRADED_STAGE_ABLATION.md`): letting the
  graded stage model also run for *tunable* BK changes 15/432 configs and gives
  geomean `1.0157` overall but splits sharply by transition —
  `ns4→ns3` 7 cells `0.8880`, `ns2→ns4` 4 cells `1.1105`, `ns2→ns6` 4 cells
  `1.1751`. Per body: plain `matmul` `0.8908` (all four affected plain GEMMs
  prefer the existing `ns4` by 16–20%), `bmm` `0.9978`, Mamba2 chunk-state
  `1.0631`, split-K `1.1216` (`m512_k32768_n512` alone `1.4657`). Conclusion in
  the doc: "the one-line universal change should not ship."
- A second, opposite-direction ablation (`LINATTN_HEURISTIC_RUN_LOG.md`,
  "Re-evaluating an `ns2` occupancy-relaxation ceiling"): capping
  `OCCUPANCY_RELAXED_MAX_STAGES` at 2 changes 79 configs, geomean `ns2/ns3`
  `0.9333` (67 of 79 regress) — yet it repairs the three known occupancy cliffs
  (`dqkw#9` 1.5906x, `dqkw#3` 1.4354x, `dv#3` 1.3542x). The tree was restored
  to `ns3`.

(iii) Root cause (documented by the author in
`MATMUL_HEURISTIC_HIGH_LEVEL_TRACE.md` "Appendix: Stage-depth limitation"): the
missing profitability question is coupled and discontinuous — extra async copies
and barriers, discrete occupancy thresholds the resource estimate misses,
register/spill changes in the generated schedule, a possible change of warp or
tcgen05 regime, whether staged loads are a meaningful fraction of runtime, and
operand reuse across dots making a many-trip loop compute-bound anyway.

(iv) **Known-and-accepted approximation.** The author deliberately keeps the
simpler policy rather than encode kernel-specific exceptions, and the ablation
data is the stated justification.

### M2 — Joint (BK, stages, warps) selection is split across code paths

(i) A tunable-K loop gets a *joint* BK+`num_stages` proposal from
`_matmul_tile` using dot-local SMEM and coarse binary occupancy signals; a
fixed-full-extent K gets the candidate-real *graded* stage model using
whole-kernel SMEM and estimated residency but with no BK to choose. Neither path
searches the (BK, stages, warps) frontier jointly.

(ii) The tunable-BK ablation above is precisely the "swap in only the stage
half" experiment, and its opposing results (plain GEMM −11%, split-K +12%,
Mamba +6%) are cited in the trace as motivation for the joint experiment rather
than a refutation of it. Independent corroboration from the 50-cell diagnosis:
14/20 cells lack the reference tile *and* 15/20 lack its warp/stage pair, so
"more BK perturbations alone will not close these gaps."

(iii) Circular dependency: BK and stages set SMEM; warps set thread/register
limits; those set residency; desired residency changes acceptable BK and depth.
The candidate space is small enough to enumerate — the hard part is *ranking*
the feasible frontier, because maximum residency is not always fastest.

(iv) **Explicitly listed future opportunity**, not in the current implementation.

### M3 — The register model charges nothing for load-produced values

(i) `_register_live_bytes` skips every `LiveTile(kind="load")`
(`triton.py:1613`: `if tile.kind == "load": continue`). Those bytes go to the
shared-memory model instead. That is right for pipelined matmul operands Triton
promotes into an SMEM ring, but `kind="load"` describes *every* load — including
ordinary pointwise loads that must pass through registers on the way to a store.

(ii) The trace's second appendix gives the minimal repro:
```python
value = y[0, tile_m, tile_n]
x[tile_m, tile_n] = value
```
`value` is charged 0 register bytes. The exclusion is deliberate and tested
(`test_register_estimate_excludes_loads_and_unregisterable_values`,
`test/test_matmul_heuristics.py:1926`; `_register_live_bytes(...load 64x64..., 8) == 0`).
**I found no benchmark cell in any artifact whose regression is attributed to
this undercount** — see Gaps.

(iii) The model knows an op is a load but not its eventual *storage role*. A
correct model would distinguish SMEM-promoted pipeline operands from
register-resident loads. Charging every load to both budgets would be too crude
(double-counting promoted operands; the compiler may stream/reuse registers
rather than materialise the whole logical tile).

(iv) **Known-and-accepted approximation with a named, unimplemented fix.**
Note the same function's other two exclusions exist for measured reasons: a
value bigger than the largest possible register file is excluded because a
varlen packed `[T,C,D]` buffer measured 256 MiB pinned every warp count to the
maximum; and the *previous* rank-profile selector under-counted a kernel that
spills 540 registers at one warp (43,520 B against a 32,640 B one-warp file)
while over-counting one that spills none (49,152 B).

### M4 — SGLang KDA decode: loses to handwritten Triton *and* to the unseeded default

(i) On KDA decode the shipped Helion config is **19.3% slower than the raw
unseeded compiler default** and **12.6% slower than handwritten SGLang Triton**,
and the loss grows monotonically with batch size.

(ii) `sglang_triton_head_to_head/four_arm_no_heuristics_v2_address_matched/`
(CUDA-graph op timing, warm L2, all arms on identical tensor addresses,
4 balanced rotated rounds, every output *and mutated-state* comparison passed):

| Scope | N | unseeded/Triton | heuristic/Triton | shipped/Triton | heuristic/unseeded | shipped/heuristic |
|---|---:|---:|---:|---:|---:|---:|
| Decode | 16 | 1.0829x | **0.8739x** | 1.5061x | **0.8070x** | 1.7235x |
| Fixed prefill | 10 | 0.6741x | 1.1556x | 1.5509x | 1.7143x | 1.3420x |
| Packed prefill | 10 | 0.6632x | 1.1110x | 1.1982x | 1.6752x | 1.0785x |
| Overall | 36 | 0.8284x | 1.0095x | 1.4250x | 1.2186x | 1.4115x |

Per shape, heuristic/Triton: B=1 → `1.7616x` (H16) / `1.6529x` (H32), then
`0.9706` at B=8, `0.7287` at B=16, `0.7753` at B=32, `0.7396` at B=64,
`0.7619` at B=128, `0.7033` at B=256, bottoming at `0.6523` (B=256, H=32).

(iii) **Root cause, and an important attribution correction: this is the
REDUCTION heuristic, not the matmul heuristics.** In all 16 decode records
`heuristic_names == ['triton_reduction_tile_sm100']` for the single kernel
`_helion_fused_recurrent_kda_packed_decode_body`. It emits **one
shape-invariant config for all 16 shapes**: `block_sizes=[4]`, `num_warps=2`,
`num_stages=1`, `pid_type='flat'`, `loop_orders=[[0,1,2]]`. The unseeded default
is `[32]` / 4 warps. The shipped SGLang config is `[8]` / **1 warp** /
`loop_orders=[[2,1,0]]` / `pid_type='xyz'` — and *two of those four differences
(`loop_orders`, `pid_type='xyz'`) are knobs the reduction heuristic does not
choose at all* (its knob list is `block_sizes`, `reduction_loops`, `num_warps`,
`num_stages`≡1, `pid_type` proposed as `flat`, `load_eviction_policies`).
So the decode body's config is insensitive to the only parameter that changes
(batch), and the winning schedule lives outside its knob set.

(iv) **Open bug / real gap** (the tuned config is 1.72x faster, so it is not a
measurement artifact). Two caveats worth carrying: the Triton arm launches
`fused_recurrent_kda_packed_decode_kernel` *directly*, because the shipped
SGLang wrapper silently dispatches 12 of these 16 BF16/FP32 shapes (B≥8) to a
custom CUDA kernel; and these are CUDA-graph numbers that must not be merged
with the eager FLA e2e results.

### M5 — The reduction heuristic never emits a pipeline depth > 1

(i) `num_stages` is fixed to 1 by construction, and the AOT configs it loses to
are overwhelmingly multi-stage.

(ii) Re-derived from `b200-reduction-audit/results/full/summary.json`
(114 cells, exact-key checked-in `_helion_aot_<kernel>_cuda_sm100.py` configs,
78 of them AOT-comparable):

- seed `num_stages` histogram: **`{1: 114}`**. Unseeded default: `{1: 114}`.
- AOT `num_stages` histogram: `{1: 17, 2: 8, 3: 9, 4: 7, 5: 7, 6: 15, 7: 6, 8: 9}`
  → **61 of 78 AOT winners use ≥2 stages.**
- Cohort ratios: general AOT `heuristic/AOT = 0.917x`, vLLM `0.918x`.
- 33 of 78 AOT-comparable cells have AOT strictly faster than the seed.
- The trace confirms this is by design:
  `REDUCTION_HEURISTIC_HIGH_LEVEL_TRACE.md` lists `num_stages` as
  "**Fixed to `1` by these reduction seeds**".

Worst AOT/seed cells (seed → AOT config):

| Kernel | Shape | AOT/seed | seed | AOT |
|---|---|---:|---|---|
| `silu_and_mul_per_block_quant` | `[8192,25600,128]` | **0.434** | `bs=[64] w=2 s=1` | `bs=[8] w=2 s=1` |
| `fused_qk_norm_rope` | `[8192,64,8]` | **0.441** | `bs=[128] w=2 s=1` | `bs=[2] w=1 s=6` |
| `silu_and_mul_per_block_quant` | `[8192,12288,128]` | 0.599 | `bs=[64] w=2` | `bs=[8] w=2 s=1` |
| `dynamic_per_token_scaled_fp8_quant` | `[8192,5120]` | 0.630 | `bs=[8192,4096] w=16` | `bs=[1024,512] w=1 s=1` |
| `rms_norm_dynamic_per_token_quant` | `[8192,5120]` | 0.643 | `w=16` | `w=8 s=1` |
| `layer_norm` | `[1024,36864]` | 0.647 | `bs=[1] w=32` | `bs=[1] w=16 s=2` |
| `cross_entropy` | `[2048,256000]` | 0.682 | `bs=[1] w=32 s=1` | `bs=[1] w=16 s=6` |

(iii) Two distinct mechanisms are visible: (a) no pipeline depth at all, and
(b) on the *largest* shape of each family the heuristic over-tiles the outer/row
axis (`[128]` vs AOT `[2]`; `[64]` vs `[8]`) and/or over-provisions warps
(32 vs 4–16). Both bite specifically at big row counts.

(iv) `num_stages=1` is an **accepted design decision**; the large-shape
over-tiling on `fused_qk_norm_rope` / `silu_and_mul_per_block_quant` /
`layer_norm[1024,36864]` is a **real modeling miss** (see M6, since the unseeded
default also beats it there).

### M6 — 17 of 114 reduction cells: the raw unseeded default beats the heuristic

(i) The heuristic is a net *regression* versus doing nothing on a specific
family of shapes.

(ii) Re-derived from `summary.json` (`default/seed < 1.0`), worst first:

| Kernel | Shape | default/seed | seed | unseeded default |
|---|---|---:|---|---|
| `fused_qk_norm_rope` | `[8192,64,8]` | **0.514** | `bs=[128] w=2` | `bs=[32] w=4` |
| `silu_and_mul_per_block_quant` | `[8192,25600,128]` | **0.558** | `bs=[64] w=2` | `bs=[32] w=4` |
| `silu_and_mul_per_block_quant` | `[8192,12288,128]` | 0.660 | `bs=[64] w=2` | `bs=[32] w=4` |
| `layer_norm` | `[1024,36864]` | 0.766 | `bs=[1] w=32` | `bs=[1] w=4` |
| `cross_entropy` | `[2048,256000]` | 0.854 | `bs=[1] w=32` | `bs=[1] w=4` |
| `cross_entropy` | `[4096,152064]` | 0.903 | `bs=[1] w=32` | `bs=[1] w=4` |
| `layer_norm` | `[4096,12288]` | 0.932 | `bs=[1] w=8` | `bs=[1] w=4` |
| ...10 more between 0.947 and 0.999 | | | | |

Kernel-level consequences in the same audit: **LayerNorm forward 0.978x vs
unseeded default, 0.961x vs `torch.compile`, 0.945x vs AOT**; cross entropy
0.974x vs default and **0.762x vs AOT**; `fused_qk_norm_rope` 0.956x vs default.

(iii) The B200 warp correction is a 4-rung traffic table
(`row_traffic = full_extent × itemsize × n_loads`; ≤64 KiB & extent ≤1024 → 2
warps; ≤64 KiB & >1024 → 4; >64 KiB & extent ≤16384 → 8; extent >16384 → keep the
base ramp). Very wide rows (vocab 152k–256k, `N=36864`) fall into the "keep the
base ramp" branch, which climbs to 32 warps; AOT prefers 4–16 warps with real
pipelining. Symmetrically, on very tall grids the row-tile allocator gives the
grid axis "the remainder" and lands at `[64]`/`[128]` where the tuned answer is
`[2]`/`[8]`.

(iv) **Open modeling miss** — small (17/114 cells, aggregate cohorts still
1.03x/1.92x/6.03x above the default), but genuinely a wrong-direction
prediction, and it is the same regime as the M4 decode loss.

### M7 — The vLLM/reduction gap widens off the tuned shapes (generalization)

(i) The heuristic is ~1.3% behind vLLM's shipped configs on the shapes the
constants were fit against, and ~4.5% behind on unseen shapes.

(ii) `wt-b200-perf-report-repro-plan/perf-repro/results/SUMMARY.md`
(455 cells; cold-L2 interleaved median-of-9 `do_bench`, arms round-robined):

| Scope | seed/torch.compile | seed/unseeded default | seed/vLLM tuned |
|---|---:|---:|---:|
| reproduction_overall (n=188) | 1.042 | 2.589 | 0.987 (n=16) |
| generalization_overall (n=252/257) | 1.102 | 2.899 | **0.955** (n=106) |
| `vllm` posted shapes (n=16) | 1.249 | 3.816 | 0.987 |
| `vllm_gen` full tuned-grid (n=96) | 1.192 | 3.523 | **0.956** |
| `qk_norm_rope_gen` (n=10) | 1.919 | **0.996** | 0.942 |
| generalization_fp32 (n=74/77) | **0.997** | 2.827 | — |

Per-kernel `vllm_gen` seed/vLLM: `dynamic_per_token_scaled_fp8_quant` 0.938
(n=35), `rms_norm_dynamic_per_token_quant` 0.948 (n=19),
`rms_norm_per_block_quant` 0.975 (n=30), `per_token_group_fp8_quant` 0.977 (n=12).
The "per-shape disasters" table (`G_tc < 0.75`) is dominated by **fp32
`cross_entropy`** (curriculum geomean `G_tc` 0.726; gen cells down to 0.516) and
**`long_sum`** (down to 0.415).

(iii) Two things: the hand-tuned constants are fit on the curriculum (unseen
fp32 shapes drop to parity with default-mode `torch.compile`, `0.997`), and the
same wide-row / long-reduction regime as M5/M6 is the failure mode.

(iv) **Accepted, honestly reported.** Caveat for the blog: this audit's vLLM arm
is described as "vLLM's tuned **H100** JSON" with "nearest-shape / exact-key
lookup", replayed on B200 — a *weaker/differently-provenanced* reference than the
114-cell audit's exact-key SM100 AOT configs (all 78 AOT cells there are
`aot_exact_key_or_sweep_shape = True`). Do not present `0.955` and `0.918` as
the same measurement.

### M8 — `fused_qk_norm_rope`: the heuristic picks tile sizes that miscompile

(i) On 6 of 15 generalization cells the heuristic-selected block size produces a
numerically **wrong** result, while the unseeded default's block size does not.

(ii) `wt-b200-perf-report-repro-plan/perf-repro/notes/QK_NORM_ROPE_FINDING.md`:
hand-sweeping the token-axis block at (q=32, kv=8, tok=512) gives maxabs vs a
pure-torch reference of 0.0156 at block 8, **1.90 at 16**, 0.0156 at 32,
**1.59 at 64**, 0.0156 at 128 — an *alternating* power-of-two pattern, so it is
a Helion codegen bug, not precision drift and not a seed bug. The vendored
kernel body is byte-identical to upstream vLLM (72/72 lines). Per-shape: 9/15
seed configs correct, 6 wrong; the pattern is not a clean `block∈{16,64}` rule
(q=16/tok=512 picks `[8]` and is wrong; q=64/tok=64 picks `[4]` and is wrong).

(iii) Suspected (unconfirmed) in-place read-after-write on `qkv` in the RoPE
epilogue aliasing across certain tile sizes.

(iv) **Open upstream codegen bug, exposed by the heuristic.** For the blog this
is a *consequence* of prediction, not a misprediction: the heuristic "merely
happens to pick block_size=16 for some shapes and thus inherits the wrong
result". Also note the aggregation caveat: the `qk_norm_rope_gen` row is a
geomean over the correct-seed shapes only (the 114-cell audit's 9
`fused_qk_norm_rope` cells all passed accuracy — different shape roster).

### M9 — Fully-FP8 contractions are declined outright

(i) Both matmul front ends refuse any dot whose *both* operands are 1-byte
floats, so FP8 GEMMs get no heuristic config at all (they fall back to
`_base_default_config`'s small tile).

(ii) Code + root cause, `triton.py:3247-3274` (front end 1) and `:3594-3597`
(front end 2), plus the predicate at `:191-195`:
- the budget formula sizes FP8 GEMMs at `block_m=128`;
- at `block_m ≥ 64` Triton lowers `tl.dot` to the native FP8 warp-group MMA
  (QGMMA / `warp_group_dot`) reading raw FP8 from SMEM;
- Helion never passes `max_num_imprecise_acc`, so Triton falls back to its sm90
  default of `2**30` (the "never promote" sentinel) and the fp32 accumulator is
  never flushed across the K loop → **error grows with K: ~0.03% at K=512 up to
  ~5% at K=8192**;
- `block_m ≤ 32` dodges it (Triton upcasts fp8→fp16 and uses HMMA with a real
  fp32 accumulate) — which is exactly what the base default already emits;
- in max-autotune the bitwise accuracy gate rejects the wide-tile config anyway,
  so seeding it only wastes a trial; and because this heuristic sets
  `promote_seed_to_default=True`, an eligible FP8 seed would become the
  effort=none default, which runs **no** accuracy check → silently wrong FP8 GEMMs.

(iii) The stated real fix is to emit `max_num_imprecise_acc=0` on FP8 `tl.dot`
in `_emit_tl_dot`, after which the seed should be re-enabled. The CuTe backend
is unaffected (fp32 accumulation is baked into the MMA op type).

(iv) **Deliberate coverage hole for a correctness reason, not a perf choice.**
Do not describe it as "the formula predicts FP8 badly."

### M10 — Off-corpus fixes traded away on-corpus linear-attention performance

(i) The three stacked follow-ups after the headline measurements moved a lot of
off-corpus performance but *regressed* changed linear-attention cells.

(ii) `MULTI_MATMUL_WORK_MODEL_FOLLOWUP_RESULTS.md` (cold-L2, rotated interleaved
CUDA events, every measured pair passed accuracy):

| Corpus | Measured pairs | Geomean | Median | Wins >1% | Regressions >1% |
|---|---:|---:|---:|---:|---:|
| Off-corpus, strict pairs | 21 | 1.401x | 1.175x | 16 | 3 |
| **Linear attention** | 41 | **0.913x** | 1.082x | 21 | 18 |
| Combined strict pairs | 62 | 1.056x | 1.094x | 37 | 21 |

Worst shipped linear-attention regressions in that step:
`chunk_bwd_dstate_delta_helion#4/#10:B8_T2048_H32_D256` **0.475x / 0.500x**
(`blocks [64] → [16]`, `warps 8 → 4`); `chunk_bwd_wu_kda_helion` ×4 cells
`0.638–0.694x` (`warps 4→2`, `stages 3→1`); `chunk_bwd_wy_dL_delta_helion` ×2
`0.752x / 0.764x`; `chunk_bwd_dqkw_delta_helion` 8 of 12 cells regress
(4 at `0.703–0.717x`); `split_k_matmul m64_k4096_n64` **0.155x**.
Counting unchanged cells as 1.000x: config-impact `1.109x` over 75 off-corpus,
**`0.991x` over all 432 linear-attention**, `1.008x` over all 507.
The next follow-up (`MULTI_MATMUL_GRID_WARP_FOLLOWUP.md`) recovered
`1.006x` over the same 432 (8 changed cells, all KDA decode-body, geomean
1.347x), so the net on-corpus linear-attention drift from the tree used for the
1.20x headline is ≈ `0.991 × 1.006 ≈ 0.997x` — approximately neutral, but the
individual D256 `dstate_delta` halvings are real and shipped.

(iii) Removing the generic TMEM-column floor from *grid* axes (the intended fix
for squeeze-excitation and `_chunk_state`) let genuine grid knobs shrink below
32; on the `dstate_delta` D256 cells that shrink `[64]→[16]` also dropped warps
8→4 and cost 2x. Candidate-resolved dynamic work + preserved dot-dimension
provenance similarly re-ranked tile/warp/stage choices in `dqkw_delta`.

(iv) **Accepted trade, documented.** Worth one sentence in the blog as the
generalisation-vs-specialisation cost, not more.

### M11 — Residency-aware 4→8 warp policy: same facts, opposite outcomes

(i) The grid/warp follow-up scaled every work-driven warp transition by the
lower-/higher-warp residency ratio. Three attention kernels expose *identical*
modeled work (`2^26`) and *identical* residency change (2 resident CTAs at 4
warps vs 1 at 8), and the policy helps one, is neutral on one, and badly hurts
the third.

(ii) `MULTI_MATMUL_GRID_WARP_FOLLOWUP.md` off-corpus table
(`[256,32]/8/3 → [256,32]/4/3` in all three):
Blackwell forward `b4_h32_s2048_d64` 409.600 → 311.392 µs = **1.315x**;
dense attention `b8_h16_m2048_n2048_d64` 397.248 → 396.256 = 1.003x;
causal attention `b8_h16_s2048_d64` 501.792 → **842.736 = 0.595x**.
The doc states it plainly: "the current facts do not distinguish the opposing
outcomes."

(iii) The structural fact set (modeled dot work + estimated resident CTAs) is
not expressive enough to separate these three kernels.

(iv) **Open modeling gap**, honestly labelled in the artifact.

### M12 — Jagged dense BMM: the emitted default is a standing correctness concern

(i) On `jagged_dense_bmm` the generated defaults do not reliably match the
stable fragment default.

(ii) `MULTI_MATMUL_GRID_WARP_FOLLOWUP.md`: "the frozen and new jagged configs
**both** fail against the stable fragment default because this kernel
accumulates in BF16 and is sensitive to reduction tiling. The new configs also
fail the direct frozen/new check and are slightly worse numerically. The saved
full-autotune configs pass by using reduction block 16. These three generated
defaults remain an explicit correctness concern rather than being counted as
successful comparisons." Corroborated independently: in `multi_seed_quality_50`
the **primary** seed for `jagged_bmm_b16_l2522_d256_k512_no_bias` is recorded
`invalid (accuracy)` — the only such cell in 50. Also
`off_corpus_full_autotune_unseeded/FINAL_AUDIT.md`: all five jagged full winners
needed an "independent fragment-default reconciliation", relative L2 differences
`0.004928`–`0.009342`.

(iii) BF16 accumulation makes the answer reduction-tiling dependent, so any
tile the heuristic picks other than reduction block 16 changes the numerics.

(iv) **Open bug** (in the kernel/accumulation contract as much as the heuristic).

### M13 — `tmem_regime_fix` (uncommitted): net win, one 0.816x regression, one rejected variant

(i) A working-tree TMEM-regime change (before and after both recorded at commit
`40151a23f`, i.e. uncommitted) that repairs the two jagged and one
squeeze-excitation configs the previous follow-up had made worse — and makes one
other squeeze cell worse.

(ii) `matmul_heuristic_perf_results/tmem_regime_fix/` (9 rounds, cold L2,
rotated interleaved CUDA events, CUDA graphs captured both arms, 4/4 accuracy
`pass`):

| Cell | before → after | before µs | after µs | speedup |
|---|---|---:|---:|---:|
| `jagged_bmm_b16_l2522_d256_k512_no_bias` | `[1,32,512,32]/8/6 → [1,64,256,32]/8/6` | 354.368 | 145.440 | **2.4365x** |
| `jagged_bmm_b4_l10240_d512_k512_no_bias` | `[1,32,512,32]/8/6 → [1,64,256,32]/8/6` | 2499.648 | 921.600 | **2.7123x** |
| `se_net_m1024_n1024_k256` | `[8,64,256,64]/4/1 → [8,32,128,64]/4/3` | 49.184 | 36.832 | 1.3354x |
| `se_net_m128_n1024_k64` | `[1,32,256,16]/4/3 → [1,32,128,16]/4/3` | 18.464 | 22.624 | **0.8161x** |

overall: 4 changed cells, geomean `1.6382x`, median `1.8859x`, sum-latency
`2.5936x`, 3 wins >3%, 1 regression >3%. **0 of 432 linear-attention configs
change** (`linear_config_diff.json` and `v2_linear_config_diff.json` both report
`changed_default_cells: 0`).
The two jagged "before" configs are exactly the grid/warp follow-up's *new*
(regressed) configs (`[1,32,512,32]/8/6` in both), so this change reverses that
step's `0.562x` / `0.462x` and ends up **1.37x / 1.25x faster than even the
pre-grid/warp frozen configs** (198.624 → 145.440 µs and 1151.968 → 921.600 µs).
Caveat: that last cross-run comparison mixes two harness runs (the grid/warp
"frozen" timing and the tmem "after" timing), so treat it as approximate; the
`2.44x` / `2.71x` figures are within-run paired measurements.

A rejected alternative is also on disk: `relief_first_probe/perf.json` compares
`current_tcgen_first` vs `relief_first` resource-fix-up ordering on
`se_net_m1024_n1024_k256` — relief-first is `0.6011x` (36.896 → 61.376 µs). The
shipped ordering (stages before tiles, tcgen05 first) is the better one on this
cell.

(iii) The uncommitted diff introduces an explicit
`MatmulResourcePolicy = Literal["strict","optimistic"]`, a liveness-based
`_candidate_tmem_columns`, and an `EXPANDED_SEED_POOL` class flag. The change is
about *how many TMEM columns a candidate is charged* (peak-live accumulators +
promoted-LHS scratch, vs a per-dot sum).

(iv) **Work in progress, not yet committed.** The `accuracy: pass` records here
do **not** by themselves retire M12: the artifact does not state which reference
the accuracy comparison used, and the new jagged config uses reduction block 32,
not the 16 the passing full-autotune configs use.

### M14 — Mamba2 chunk scan: config *similarity* does not predict the best seed

(i) On one Mamba2 shape the shipped primary is 57.7% slower than the
full-autotune winner, and the seed that helps most is in the *opposite* regime
from the winner.

(ii) `matmul_heuristic_perf_results/mamba2_chunk_scan_followup/REPORT.md`
(`mamba_scan_b2_h32_g8_t8192_c128_d128_s256`, B200 GPU 0, cold L2, rotated
interleaved, confirmation at 9 rounds / 25 ms per arm-round; all five canonical
seeds compiled and passed accuracy):

| Config | µs | coverage | slowdown | block sizes | warps | stages | L2 | PID |
|---|---:|---:|---:|---|---:|---:|---:|---|
| full-autotune winner | **223.264** | 1.000x | — | `[128,128,64]` | 8 | 3 | 32 | persistent-interleaved |
| primary (shipped) | 352.192 | 0.634x | +57.7% | `[128,128,64]` | 4 | 2 | 1 | flat |
| focal dot 0 | 333.792 | 0.669x | +49.5% | `[128,128,64]` | 4 | 3 | 1 | flat |
| register-MMA moderate | **294.848** | 0.757x | +32.1% | `[64,64,64]` | 2 | 1 | 1 | flat |
| raw-formula (initial pool) | 6326.208 | 0.035x | **+2733.5%** | — | — | — | — | — |

The best seed recovers 44.5% of the primary→reference latency gap (1.194x faster
than primary, −16.3% latency) but is still 32.1% off. Config Hamming distance
does **not** rank it: focal dot 0 is closest (d=9) yet slower; register-MMA
moderate wins at d=12.

(iii) The primary already has the winner's exact tile `[128,128,64]`; what it
misses is the launch/schedule regime (8 warps, L2 grouping 32,
persistent-interleaved PID) — i.e. two of the three missing knobs are outside the
heuristic's four. Note also that the *unmodified* raw formula output is
27x slower here; the correction stages are load-bearing, not cosmetic.

(iv) **Open gap; the diagnosis names the fix** (`PRIMARY_VS_POOL_BREAKDOWN.md`
"Common Implications" 4: "Persistent regimes deserve explicit representation").

### M15 — Pointwise: no pipeline depth, and `torch.compile` wins the small-kernel cohorts

(i) The pointwise heuristic sets only `block_sizes` (plus `num_warps` when it
differs from 4) and never sets `num_stages`; on the two production cohorts
default-mode `torch.compile` beats it.

(ii) `b200-pointwise-audit/results/full/` (36 planned cells, 35 timed;
calibrated cold-L2 CUDA graphs, flush inside the timed graph, 9 round medians;
all values relative to the unseeded default):

| Cohort | n | heuristic | torch.compile | AOT |
|---|---:|---:|---:|---:|
| general (SwiGLU/GEGLU/RoPE) | 17 | 19.351x | 18.508x | 22.145x (5 RoPE cells, **SM90 table on B200**) |
| vLLM `silu_mul_fp8` | 9 | 1.096x | **1.232x** | 1.130x (SM90) |
| SGLang `silu_and_mul_interleaved` | 9 | 1.257x | **1.420x** | 1.036x (SM100) |

Seed `num_stages` histogram over all 36 cells: **`{None: 36}`** (unset → base
default). AOT `num_stages` histogram over the 23 AOT cells:
`{1: 11, 2: 3, 3: 4, 4: 1, 5: 3, 7: 1}` → 12/23 use ≥2. Worst seed cells:
`silu_mul_fp8 (1,2048)` 0.958x and `(256,7688)` 0.979x vs the unseeded default.
Note the *heuristic beats the SM100 AOT table* on SGLang interleaved SiLU
(1.257 vs 1.036), and the large general-pointwise numbers mostly reflect how bad
the tiny unseeded base tiles are. One RoPE cell timed out after 300 s.

(iii) Structural: `POINTWISE_HEURISTIC_HIGH_LEVEL_TRACE.md` — "`num_stages`:
Not selected; retains the base default", "There are no pointwise-specific
alternate seeds, pipeline-stage choices, or iterative resource fix-ups."
The losing cells are 2–24 µs kernels where an unrolled/multi-stage Inductor
schedule matters more than tile size.

(iv) **Accepted scope limit.** These are small absolute latencies; state them,
don't dramatise them.

### M16 — Single-contraction GEMM: small-M is the weak regime

(i) The formula seed beats a Triton-template Inductor GEMM at medium and large
M but loses ~13% at small M.

(ii) `formula_matmul_49_results/{REPORT.md,summary.json}` (49 core BF16 cells,
cold-L2, externally captured CUDA graphs, 5 rotated interleaved rounds):

| Group | cells | Helion default / formula | torch-Triton / formula |
|---|---:|---:|---:|
| all core | 49 | 21.90 | 1.0546 |
| plain GEMM | 16 | 25.33 | 1.1014 |
| with epilogue | 33 | 20.42 | 1.0327 |
| **M small (16–256)** | 11 | 7.45 | **0.8708** |
| M medium | 11 | 28.66 | 1.1526 |
| M large | 27 | 30.46 | 1.0997 |

Worst cells: `bert_base_ffn_up (2048×3072×768) + bias_gelu` **0.7105**;
`token_m64 + gelu` 0.7720; `token_m64 + relu/silu/bias_relu/bias_gelu/silu_mul`
0.833–0.837; `token_m64` plain 0.8316; `shallow_k 4096×4096×512` 0.9419.
By epilogue: `bias_gelu` 0.9299 (n=4), `gelu` 0.9705 (n=3),
`bias_residual_gelu` 0.9955 (n=3) — the transcendental epilogues are where the
formula loses.

(iii) Two candidate mechanisms visible in the data, neither isolated by an
ablation: the accumulator/wave-fill logic at M=16–256 leaves the tile too large
(the formula "generally favors N"), and epilogue SFU work is not part of the
tile budget at all (the trace's Stage-3 budget is operand bytes + accumulator
capacity only).

(iv) **Open gap, small.** Caveat: see Correction C4 — this arm is *not*
default-mode `torch.compile`.

### M17 — Search-side: expanded seeds are not uniformly cheaper, and censoring matters

(i) The seed pool speeds convergence in aggregate but slows the search on some
cells, and one headline sum is built on right-censored lower bounds.

(ii) `multi_seed_full_autotune_ablation/RESULTS.md` (38 paired full-autotune
searches, main vs branch, paired replay for winners):
branch/main geomean **configs 0.8868x, autotune time 0.8817x, time-to-best
0.8813x, paired winner latency 0.9950x**; multi-only autotune time 0.8506x but
**single-only 1.0088x**. Per-cell *treatment regressions* in search cost:
`jagged_bmm_b8_l255` 734 vs 544 configs (359.7 s vs 249.6 s),
`gdn_b8_t4096_h80` 657 vs 492 (1181.1 vs 1004.5 s),
`_chunk_output::packed_h12_t2048` 277 vs 240 (414.2 vs 331.7 s),
`gather_gemv_b64` 425 vs 336, `chunk_fwd_A_diag_anchored#0` 610 vs 477.
Failures main/branch: compile 2822/2576, worker 31/15, accuracy 948/459.
From the ground-truth index: the 38-cell "within 5%" reach counts are
no-seed 28/38, **old-seed 34/38**, expanded-seed 28/38 — old seeds reached the
shared 5% target in more cells than either other arm even though the expanded
sum looks lower, because unreached arms are substituted with their full terminal
attempt count.

(iii) Extra seeds mean extra compiles and extra LFBO surrogate data of uneven
quality; they can also anchor the search away from the eventual winner.

(iv) **Accepted, but the 5% row must be quoted with its reach counts.**

### M18 — Exploratory ("optimistic TMEM") seeds may be unlaunchable by design

(i) The expanded seed pool intentionally permits some invalid exploratory seeds,
and one known undercount can produce them.

(ii) `MATMUL_MULTI_SEED_FULL_AUTOTUNE_OVERNIGHT_LOG.md`, 2026-08-22:
"Follow-up review confirmed a bounded optimistic-TMEM limitation: graph-local
role peaks are selected before candidate dimensions are known, so a nested
parent/child call can undercount a larger parent tile that remains live across
the child. **Strict rank zero is unaffected.** … The optimistic policy
intentionally permits some invalid exploratory seeds."

(iii) Peak-live selection happens before candidate block sizes are resolved.
Fix named: exact candidate-resolved cross-graph liveness.

(iv) **Known-and-accepted, scoped to non-primary seeds.** Important for the
blog's honesty: the compiler *default* (rank 0) never uses the optimistic
policy.

### M19 — Two named "future opportunity" items

**(a) Multiple scalar priors (FE2).** The multi-contraction front end has two
defensible starting points for its kernel-global scalar solve: a whole-kernel
prior conditioned on each dot, and a merged-only prior derived after shared
block sizes are resolved. `MATMUL_HEURISTIC_HIGH_LEVEL_TRACE.md` reports
GPU-1 measurements finding "substantial wins in both directions: the
preconditioned result was up to **1.58x** faster, while the merged-only result
was up to **1.43x** faster. Neither prior dominates." Only the
whole-kernel-conditioned path is implemented, deliberately, to avoid a second
policy/validation surface. **⚠ I could not locate the raw artifact behind
1.58x/1.43x** (see Gaps); the verifiable adjacent measurement is
`KERNEL_SMEM_PRECONDITIONED_PRIOR_RESTORE_PLAN.md`, which restored the
preconditioned prior with 430/432 defaults identical, the two changed
`_chunk_output` cells neutral (`0.99957x`, `1.00002x`, below the 0.108% null
spread) and two removed alternate seeds where the surviving primary is
`1.13323x` / `1.20121x` faster (and two more where the removed seed needed
293,392 B / 291,344 B SMEM against a 232,448 B hardware limit).

**(b) Joint BK/stage/warp selection.** See M2.

---

## 3. Ranking: the 3–4 worth naming in the blog

1. **`num_stages` = deepest feasible, not most profitable (M1/M2).** The best
   entry: it has a clean mechanism, an average-positive effect (stages raised in
   117/149 cells; `post/pre` 1.586x where raised vs 1.436x where not), a
   dominant-failure signature (19 of the 23 >1% regressions raise stages), a
   textbook single-variable case (`chunk_fwd_wy_delta_varlen`, only `1→3`,
   −9.8…−11.7% on all five shapes), and a *published opposing ablation*
   (plain GEMM `0.8908` vs split-K `1.1216`) that proves the naive fix is wrong.
   It also motivates the named future work.
2. **The residual gap is broad, not exotic — and half of it is in knobs the
   heuristic doesn't own (§1.1/§1.4).** Median cell is 1.156x behind AOT;
   53/149 sit in the 1.10–1.25 band; 26/149 within 1%; top-5 cells hold only
   16.4% of the log-gap. And 148/149 AOT configs differ in `indexing` /
   `loop_orders` / `range_*` / `pid_type` — 0/50 random cells even have an
   exact-config seed. This reframes 1.20x from "the formula is 20% wrong" to
   "four knobs get you within ~20% of a 14-knob search."
3. **SGLang KDA decode: slower than the unseeded default (M4).** The single most
   uncomfortable number in the whole set (`heuristic/unseeded = 0.8070x`,
   `heuristic/Triton = 0.8739x`) and the one a skeptical reader will hunt for.
   Name it *and* name its true owner: the reduction heuristic emitting one
   shape-invariant `[4]/2 warps/1 stage` config for all 16 batch sizes.
4. **Reduction never emits a pipeline depth (M5/M6).** `num_stages == 1` in
   114/114 seed configs while 61/78 AOT winners use ≥2, which is most of the
   0.917x/0.918x AOT deficit; and on 17/114 cells the unseeded default is
   actually faster (worst `0.514x`, `fused_qk_norm_rope [8192,64,8]`).

Runners-up if space allows: **M9 (FP8 declined for a correctness reason, not a
perf one)** — cheap to state and it earns credibility; **M11 (identical modeled
facts, opposite 4→8-warp outcomes)** — the sharpest single illustration of
where a static fact set runs out.

---

## 4. Draft paragraph (≈240 words)

> **Where the heuristics mispredict.** The residual gap to full autotuning is
> broad rather than exotic. Across 149 B200 matmul cells the pre-tuned config is
> 1.20x faster in geomean, but the median cell is only 1.16x behind and 53 of
> them sit between 1.10x and 1.25x; no handful of pathological kernels accounts
> for it. Part of that gap is structural: the heuristic chooses four knobs, and
> 148 of 149 tuned configs also differ in indexing, loop order, range pipelining
> or PID schedule, none of which the model predicts. The clearest genuine error
> is pipeline depth: the stage solver picks the deepest depth that fits, not the
> most profitable one. Raising `num_stages` is right on average (117 of 149
> cells, which improve more than the rest), yet it accounts for 19 of the 23
> cells that regress by more than 1%; on one varlen kernel the only config
> change is one stage to three, and all five shapes lose about 10%. We tried the
> obvious fix and it made ordinary GEMMs 11% slower while making split-K 12%
> faster, so we kept the simple policy and wrote the ablation down. Two known
> holes remain: the reduction seeds never emit a pipeline depth at all — they are
> one stage in all 114 audited cells, while 61 of 78 tuned winners use two or
> more, and they end up about 8% behind those tuned configs — and fully-FP8
> contractions are declined outright, for a Triton accumulator-precision bug
> rather than for performance.

Alternate closing sentence if the SGLang decode number should be in-paragraph:

> …and on SGLang's KDA decode body our config is 13% slower than handwritten
> Triton and 19% slower than doing nothing at all, because the reduction
> heuristic emits one batch-independent tile for every decode shape.

---

## 5. Corrections / caveats for `PYTORCH_BLOG_HEURISTICS_RESULTS.md`

**C1 (attribution).** §2 "SGLang KDA" presents the decode loss inside a document
whose surrounding sections are about the matmul/multi-matmul heuristic. Raw data
shows the decode body's config comes from **`triton_reduction_tile_sm100`**;
in all 16 decode records `heuristic_names == ['triton_reduction_tile_sm100']`,
and no matmul heuristic is recorded. Only the four prefill matmul kernels
(`_chunk_output`, `_chunk_state`, `_intra_matrices_wide`,
`_intra_solve_recompute`) are `triton_b200_multi_matmul`; `_gate_cumsum_operands`
and `_l2norm_qk` are also reduction. The blog must not attribute the decode loss
to the multi-matmul heuristic.

**C2 (reference provenance differs between the two vLLM views).** §6 lists
`0.918x` (54-cell exact-key SM100 AOT) and `0.955x/0.956x` (96-cell tuned grid)
without flagging that the broader audit's arm is documented as "vLLM's tuned
**H100** JSON" with "nearest-shape / exact-key lookup" replayed on B200
(`wt-b200-perf-report-repro-plan/perf-repro/results/SUMMARY.md` line 3;
`README.md` line 38), whereas all 78 AOT cells in the 114-cell audit are
`aot_exact_key_or_sweep_shape = True` against checked-in
`_helion_aot_<kernel>_cuda_sm100.py`. These are not the same quality of
reference.

**C3 (the on-corpus/e2e headlines predate the shipped follow-ups).**
`on_corpus_pretuned` was measured at `post_sha ccfcfbdd8…` and `e2e_fla_v3` at
`375363d86…`. Three follow-ups landed after: off-corpus (0/432 linear configs
changed), work-model (**41/432 changed, geomean 0.913x on changed cells,
0.991x config-impact over all 432**) and grid/warp (8/432 changed, 1.347x,
1.006x over all 432). Net on-corpus linear-attention drift ≈ 0.997x — small, but
the 1.5527x / 1.2046x / 1.9816x figures are not from the final tree, and the
shipped tree contains two `chunk_bwd_dstate_delta` D256 cells at 0.475x/0.500x
relative to the tree that produced the headline.

**C4 (`torch.compile` arm definition is not universal).** The Quoting Notes say
"`torch.compile` means default mode, not `max-autotune`". That holds for the
reduction, pointwise, and generalization audits. It does **not** hold for the
49-cell formula-matmul report: `scripts/formula_matmul_49_benchmark.py` sets
`max_autotune=False` but **`max_autotune_gemm=True`,
`max_autotune_gemm_backends="TRITON"`, `max_autotune_gemm_search_space="DEFAULT"`,
`coordinate_descent_tuning=False`**, with generated code audited to reject
`extern_kernels.mm/addmm`, `aten.mm/addmm`, cuBLAS and cuDNN. If the blog cites
`1.0546x` / the small-M `0.8708x`, label that arm "Inductor Triton-template GEMM
autotuning", not "torch.compile default mode". (That report is also absent from
the index; consider adding it.)

**C5 (the 244-cell "1.4547x heuristic/pre-change" is superseded off-corpus).**
§3's broader-corpus row uses quick-autotune as its ceiling proxy. The stronger,
later measurement for the same 75 off-corpus cells is unseeded **full** autotune:
`full / heuristic = 1.4477x` geomean (worst kernel
`squeeze_excitation_forward` 4.3268x, `causal_forward` 2.7731x) and
`heuristic / quick = 0.8521x`. Any statement of the form "the heuristic is close
to autotuning" needs the on-corpus/off-corpus split, because off-corpus the gap
is 1.45x, not 1.20x, and the heuristic is *behind* a plain quick autotune.

**C6 (one reduction cell is excluded for accuracy, and 6 more elsewhere are
miscompiles).** §5 mentions the RMSNorm-backward `[2048, 11008]` exclusion
(both heuristic and unseeded configs fail). It does not mention that in the
455-cell generalization audit **6 of 15 `fused_qk_norm_rope` cells produce
numerically wrong output with the heuristic's block size while the default's is
correct**, and are excluded from the geomean
(`notes/QK_NORM_ROPE_FINDING.md`). The root cause is an upstream Helion codegen
bug, but the exclusion is heuristic-selected and should be stated.

**C7 (pointwise heuristic ≠ 19.351x "win").** §7 already says the large general
gains "primarily reflect how poor the tiny unseeded base configurations are";
worth adding that in both production cohorts default-mode `torch.compile` is
*faster* than the heuristic (1.232 vs 1.096; 1.420 vs 1.257).

---

## 6. Gaps (could not establish from available data)

1. **The 1.58x / 1.43x "multiple scalar priors" numbers.** They appear only as
   prose in `MATMUL_HEURISTIC_HIGH_LEVEL_TRACE.md:378-379`. Greps across
   `/home/dev/local/sm100-linattn/*.md`, `PROGRESS.md`,
   `LINATTN_HEURISTIC_RUN_LOG.md`, `KERNEL_SMEM_*`, and the worktree's `*.md`
   found no per-cell table, JSONL or oracle behind them. Quote them as the
   author's summary, or ask for the artifact.
2. **No measured cell attributed to the register/load undercount (M3).** The
   limitation is documented and unit-tested, but nothing in the perf artifacts
   ties a specific regression to it. Do not claim a performance cost.
3. **What `tmem_regime_fix`'s accuracy `pass` compares against.** The per-cell
   JSON records `{"problems": [], "status": "pass"}` with no reference field, and
   the producing runner is not in `scripts/`. So it cannot be used to close the
   jagged-BMM correctness concern (M12).
4. **No `quick_unseeded` / `full_unseeded` arm in the 49-cell formula-matmul
   run.** The curriculum defines the primary metric as
   `quick_unseeded_us / formula_seed_us` and mandates escalation when the
   formula is >5% slower than the Triton arm (which happens on 11 small-M
   cells), but the executed run has only three arms
   (`helion_default`, `formula_seed`, `torch_compile_triton`). There is no
   retained-quality number for single-contraction GEMM.
5. **Why the SGLang decode config is batch-invariant.** I established *that* it
   is (one config for B∈{1…256}) and *which* knobs the shipped config uses that
   the reduction heuristic cannot emit (`loop_orders`, `pid_type='xyz'`), but not
   whether the batch axis is symbolic/unbacked at compile time, nor a per-knob
   ablation of which difference costs the 1.72x. No ncu/ablation artifact exists.
6. **M11's opposing 4→8-warp outcomes have no root cause.** The artifact
   explicitly says the current facts do not distinguish them; no follow-up
   diagnosis (occupancy trace, spill count, ncu) exists in these directories.
7. **No re-measurement of the 149-cell / 96-cell headlines on the final tree.**
   The drift in C3 is inferred by chaining config-impact geomeans from the
   follow-up reports, not measured directly.
8. **`num_stages` profitability has never been ablated per-cell on the shipped
   tree.** The two ablations available (tunable-BK graded stages; `ns2` ceiling)
   each change a *policy*, not the depth of an individual shipped config, so
   "how much of the 1.20x is stage depth" is unquantified.
