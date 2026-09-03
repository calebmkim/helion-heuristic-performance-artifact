# 14 — The Cost of Full Autotuning (raw-data notes)

Scope: make the blog's economic claim defensible. Everything below is re-derived from
`results.json` / `summary.json` / `events.jsonl` / `THREE_WAY_AUTOTUNE_TABLES.json` /
the shipped config tables themselves — not from prose summaries. All GPU numbers are
NVIDIA B200 (sm100), 148 SMs, unless stated.

**Bottom line up front:** the "~15 minutes per shape" figure is *exactly right* — it is the
measured **mean** of 189 full-effort autotune searches (15.75 min; median 10.70 min).
But **"nearly a day of GPU time" understates the corpus cost by ~2x.** The linear-attention
family's shipped B200 config table is **256 tuned call keys over 26 kernels**, not 150, and at
the measured mean that is **67 GPU-hours ≈ 2.8 days**. Even the author's own "25 × 6 = 150"
arithmetic at 15 min/shape is **37.5 GPU-hours = 1.6 days**, not "nearly a day."

---

## 1. Corpus sizes — exact counts, with provenance

### 1a. Linear-attention: four different "corpus" definitions. Do not mix them.

| Definition | Count | Source |
|---|---|---|
| **End-to-end operation cells** (the headline FLA comparison) | **96** = 7 dense variants × 6 shapes × 2 modes (84) + `kda_fused` × 6 + `kda_varlen` × 6 | `LINEAR_ATTENTION_E2E_BENCHMARK_PLAN.md:56-88`; `matmul_heuristic_perf_results/e2e_fla_v3/results.json` has exactly 96 records |
| **Helion kernel bodies in the engine** | **28** `@helion.kernel` functions; **26** are in the curriculum's primary set + the AOT table; 2 (`recurrent_step_fused`, `recurrent_step_correction_fused`) are "coverage_only" | `grep -c "@helion.kernel" examples/linear/linear_attention_engine.py` = 28; `LINATTN_HEURISTIC_CURRICULUM.json` |
| **Shipped B200 AOT table = tuning units** | **26 kernels / 256 tuned call keys**. Exact per-kernel distribution: **4 keys × 4 kernels + 6 keys × 10 kernels + 12 keys × 6 kernels + 18 keys × 6 kernels = 256** (range **4–18**) | AST-parsed `_KEYS_*` lists in `examples/linear/_helion_aot_linear_attention_engine_cuda_sm100.py` |
| **Shipped SM90 AOT table** | **26 kernels / 252 tuned call keys** | `examples/linear/_helion_aot_linear_attention_engine_cuda_sm90.py` |

**Why 6 shapes → 4-18 keys per kernel.** The AOT table's own docstring: *"A call is keyed on its
tensor arguments' shapes together with its None and bool arguments, which choose a code path."*
Measured fan-out: `chunk_bwd_dh_diag_fused` has 18 keys = 18 distinct shape tuples × 3 distinct
flag tuples; `chunk_fwd_h_delta_helion` 18 keys / 18 shapes / 3 flag combos;
`chunk_bwd_dk_delta_helion` 6 keys / 6 shapes / 1 flag combo. So the *model-level* shape count is 6,
but the *tuning-unit* count is 256.

The 6 production dense shapes (from `benchmarks/run_linattn.py`, per the E2E plan):
`B1_T8192_H96_D128`, `B2_T16384_H16_D128`, `B4_T2048_H16_D128`, `B4_T4096_H64_D128`,
`B8_T2048_H32_D256`, `B8_T1024_H8_D64`.

### 1b. Full heuristic curriculum (the superset the perf run sampled from)

`/home/dev/local/sm100-linattn/LINATTN_HEURISTIC_CURRICULUM.json`:

- **34 primary bodies / 399 cases** — 26 Helion linear-attention bodies (`examples.linear.linear_attention_engine`) + 8 SGLang KDA bodies (`kda_prefill` ×6, `kda_decode` ×1, `kda_replayssm` ×1).
- **+2 coverage-only bodies / 8 cases**, **+4 secondary bodies / 25 cases**.
- **Total 40 bodies / 432 cases.**
- Per-body available cases run 4–28 (`_chunk_state` 26, `packed_decode_body` 28, `chunk_bwd_dh_diag_fused` 18…).

### 1c. The matmul perf corpus actually benchmarked

`matmul_heuristic_perf_results/manifest.json` (schema `matmul-heuristic-perf-manifest/1`,
`pre_sha=9c46dd311…`, `post_sha=ccfcfbdd8…`, `physical_gpu=1`):

- **244 cells / 49 kernel bodies.**
  - `primary_on_corpus`: **169 cells / 34 bodies** (≤5 cases sampled per body; one body has only 4) — 129 Helion linattn + 40 SGLang.
  - `off_corpus`: **75 cells / 15 bodies** × 5 shapes (8 attention, 5 matmul, 2 state-space).
- Reporting scope for the matmul-heuristic headline: **149 cells** where `triton_b200_formula_matmul` (20) or `triton_b200_multi_matmul` (129) fired; 20 reduction-only cells excluded.
- **Sampling ratio: 169 of 399 available on-corpus cases = 42%.** The measured corpus is a sample, not the whole thing.

### 1d. Other corpora (for context)

- Formula-matmul GEMM study: **49 core cells** (79 in the "full" tier) — `FORMULA_MATMUL_PERF_CURRICULUM.json` `expected_expansion.tier_unique_cells`.
- Zero/old/expanded seed search study: **38 frozen cells** (23 on-corpus, 15 off-corpus).
- Reduction audit: **114 cells / 16 kernels**. Pointwise audit: **36 cells / 5 kernels** (verified from their `summary.json`).

---

## 2. Measured full-autotune wall time — the real distribution

### 2a. The combined distribution (the number to quote)

Pooled over every **full-effort** search in the project's artifacts: the 3 arms × 38 cells of
`zero_seed_full_autotune/THREE_WAY_AUTOTUNE_TABLES.json` (`arms.*.autotune_wall_seconds`) plus the
75 cells of `off_corpus_full_autotune_unseeded/results.json`
(`records[].full_autotune.tuning_metadata.duration_seconds`).

**n = 189 searches over 98 distinct kernel/shape cells. Total = 178,646 s = 49.62 GPU-hours.**

| statistic | seconds | minutes |
|---|---:|---:|
| mean | 945.2 | **15.75** |
| median | 642.0 | **10.70** |
| geomean | 671.9 | 11.20 |
| min | 167.2 | 2.79 |
| p10 | 300.0 | 5.00 |
| p25 | 394.7 | 6.58 |
| p75 | 1013.9 | 16.90 |
| p90 | 1721.3 | 28.69 |
| p95 | 3194.1 | 53.23 |
| max | 8435.8 | **140.6** (2 h 21 m, `jagged_hstu_b2_l7168_max4096_h16_d64`) |

- 31.2% of searches took **over 15 min**; 9.5% took **over 30 min**; only 9.5% finished under 5 min.
- The distribution is heavily right-skewed: **mean 15.75 min vs median 10.70 min.** Quote both, or the
  mean will look cherry-picked.

### 2b. Per-run ledger (every measured autotune run, with its own file)

| Run | Effort / seeds | Cells / searches | GPU-hours | Wall clock | Mean per search | Source |
|---|---|---|---:|---|---:|---|
| `zero_seed_full_autotune` | full LFBO, **zero** seeds | 38 | **9.764** (`autotune_time` sum) / 9.845 (`events.jsonl` `arm_completed`) | 5 h 52 m on **2** B200s | 15.42 / 15.54 min | `zero_seed_full_autotune/RESULTS.md`, `.../events.jsonl`, 38 × `cells/*/zero_seed/result.json` |
| `multi_seed_full_autotune_ablation` | full LFBO, old-seed + expanded-seed arms | 72 completed arms (+36 failed arms) | **15.568** completed + 2.110 failed = **17.68** | 11 h 36 m on 2 B200s | 12.97 min | `multi_seed_full_autotune_ablation/events.jsonl` |
| `off_corpus_full_autotune_unseeded` | full LFBO, no seeds, heuristics disabled | 75 | **22.016** | 11 h 59 m on 2 B200s | 17.61 min | `off_corpus_full_autotune_unseeded/FINAL_AUDIT.md` ("Accumulated full-tuning time: 22.016 GPU-hours") + `results.json` (recomputed: 79,256.3 s) |
| **full-autotune subtotal** | | **185 searches** | **≈49.5 GPU-hours** | | | |
| main 244-cell run, **quick** autotune | quick | 178 (stopped at 178/244) | **6.968** | inside a 10 h 27 m span, 1 GPU | 2.35 min | `matmul_heuristic_perf_results/results.json` (`quick_autotune.tuning_metadata.duration_seconds`), `events.jsonl` |
| `off_corpus_main_autotune`, **quick** on stock main | quick | 75 | **2.862** | 1 h 52 m on 2 B200s | 2.29 min | `off_corpus_main_autotune/events.jsonl`, `results.json` |
| **quick-autotune subtotal** | | 253 searches | **9.83 GPU-hours** | | | |

Note the wall-clock/GPU-hour split: every full-autotune run used **both** B200s in independent lanes,
so wall clock ≈ GPU-hours / 2. Quote GPU-hours for cost, wall clock for schedule.

### 2c. Full vs quick, measured on the same 75 cells

`off_corpus_full_autotune_unseeded/results.json` records both durations per cell:

| | total | mean | median | min | max |
|---|---:|---:|---:|---:|---:|
| **full** (unseeded) | 22.016 GPU-h | 17.61 min | 11.40 min | 3.01 min | 140.6 min |
| **quick** | 2.862 GPU-h | 2.29 min | 1.92 min | 1.00 min | 9.85 min |

**Full/quick per-cell cost ratio: geomean 6.03x, arithmetic mean 6.93x.** Config counts agree:
quick averaged **70.5 configs/cell** (median 60, min 1, max 173; 12,692 total, from the `started` rows of
the 180 `autotune_logs/*.csv` files in the main run — 180 log files vs the 178 cells with a recorded
`duration_seconds`, the extra two being the smoke cell and a repair), full averaged **521 configs/cell**.
Independent cross-check of the quick lane's own wall time from its logs (max `[Ns]` stamp per log):
180 logs, **5.17 GPU-hours**, median **85.5 s**, mean 103.4 s, p90 202 s, max 553 s — lower than the
`duration_seconds` mean of 140.9 s because the log stamp excludes process/bind setup.

### 2d. Inside one full search (zero-seed arm, 38 cells)

From the 38 `cells/*/zero_seed/result.json` files:

- **19,796 configs benchmarked** total; mean 521/cell, median 521, min 147, max 992.
- **1.78 s per config** on average (median 1.29 s; range 0.48–8.95 s across cells).
- **3,102 compile failures (15.7% of all configs)**, 450 accuracy failures, 16 worker failures.
  A sixth of the search budget buys nothing but a compile error.
- Generations completed: median 8 (profile cap 20); min 2, max 20.
- **74.7% of total search time elapses before the best incumbent is found** (26,249.5 s of 35,151.5 s);
  per-cell median 69%. So ~25% of the bill is pure confirmation.
- Harness overhead outside the tune call is negligible: `process_wall_seconds` sum 35,254.2 s vs
  `autotune_time` sum 35,151.5 s = **+0.3%**.
- Full-effort profile (recorded in every cell): `LFBOTreeSearch`, `initial_population=100`,
  `copies=5`, `max_generations=20`, `initial_population_strategy=FROM_RANDOM`, `random_seed=0`,
  accuracy checking on, `static_shapes=True`.

### 2e. Which kernels are expensive (mean full-autotune per off-corpus family)

`off.attention.jagged_hstu` **80.8 min**, `flex_forward` 28.8, `causal_forward` 18.7,
`dense_forward` 18.0, `mamba2_chunk_scan` 16.5, `blackwell_backward` 15.6, `gdn_fwd_h` 14.6,
`blackwell_forward` 12.0, `attention.backward` 11.8, `biased_forward` 11.5,
`squeeze_excitation` 9.7, `jagged_dense_bmm` 8.6, `bf16xint16` 7.2, `broadcast` 6.3,
`gather_gemv` 4.1. **The cost is not per-kernel-uniform: 20x spread across families.**

### 2f. Restricting to the linear-attention kernels only

19 of the 38 zero-seed cells are `helion.linear_attention_engine.*` bodies:
**mean 14.06 min, median 8.08 min, min 5.06 min, max 38.5 min, total 4.452 GPU-hours.**
Use this rate if the blog's projection is specifically about the linear-attention corpus.

### 2g. Helion's own documentation agrees with the measurement

- `docs/deployment_autotuning.md:859` — AOT `collect` "runs full autotuning on every distinct shape,
  which is typically **5–15 minutes per shape** on a recent GPU."
- `docs/index.md:31` — "Helion spends more time (**approx 10 min**) autotuning as it evaluates hundreds
  of potential Triton implementations."
- `docs/index.md:157` — worked example: "`[586s] Autotuning complete in 586.6s after searching 1520 configs.`"
- `docs/deployment_autotuning.md:141` — quick "finishes in seconds (e.g. **~3s / ~32 configs** for a
  simple kernel)"; full "often ~10 minutes".

So "~15 minutes per shape" sits at the top of the *documented* range and equals the *measured mean* on
this corpus. Both are citable; prefer the measured one.

---

## 3. Cost of the heuristic itself

### 3a. Best available GPU measurement: `selection_wall_seconds`, heuristics ON vs OFF

`scripts/formula_matmul_49_benchmark.py::collect_config` runs **config selection in an isolated GPU
worker** with `HELION_AUTOTUNE_EFFORT=none`, once with `HELION_DISABLE_AUTOTUNER_HEURISTICS=1`
(`helion_default` arm) and once with `=0` (`formula_seed` arm), and records
`selection_wall_seconds` = build inputs → `matmul.bind(args)` → `spec.default_config()`.
49 BF16 GEMM cells; `formula_matmul_49_results/configs/*/*.json`.

| arm | mean | median | min | max |
|---|---:|---:|---:|---:|
| heuristics **OFF** | 2.375 s | 2.332 s | 2.275 s | 2.694 s |
| heuristics **ON** (formula matmul fires) | 2.433 s | 2.371 s | 2.287 s | 2.698 s |

**Paired delta (ON − OFF), n=49: mean +57.9 ms, median +33.7 ms; ratio geomean 1.0238x
(median 1.0145x); ON was slower on 33/49 cells; range −305 ms to +392 ms.**

Read: **enabling the compiler heuristics adds tens of milliseconds (~1.4–2.4%) to a ~2.4 s
config-selection step**, which is itself dominated by CUDA/torch init and front-end tracing, not by
the heuristic. The ±0.3 s per-cell spread exceeds the mean delta, so the honest bound is
"tens of milliseconds, certainly under half a second."

**Ratio to what it replaces: 945.2 s / 0.058 s ≈ 16,000x.**

### 3b. Same run: compiling and launching the emitted config is unchanged

`compile.<arm>.compile_and_first_call_wall_seconds` in `formula_matmul_49_results/cells/*.json`
(`to_triton_code(config)` → `set_config` → first call → `cuda.synchronize()`; both Helion arms share
one `BoundKernel` in one process, arm order is always `helion_default, formula_seed, torch_compile_triton`):

| arm | mean | median | min | max | total (49 cells) |
|---|---:|---:|---:|---:|---:|
| Helion, heuristic config | 0.514 s | 0.393 s | 0.279 s | 1.213 s | 25.2 s |
| Helion, raw default config | 0.536 s | 0.531 s | 0.490 s | 0.603 s | 26.3 s |
| `torch.compile(max_autotune_gemm=True, backends=TRITON)` | **2.904 s** | 2.612 s | 1.601 s | 4.922 s | **142.3 s** |

Delta heuristic-config vs default-config compile: mean **−22 ms** (heuristic config compiled *faster*
on 36/49). Caveat: `helion_default` always compiled first, so it eats any process warmup — do not
claim "the heuristic makes compiles faster." The safe claim is: **the emitted config's compile time is
indistinguishable from the default's.**

Useful mid-point: **Inductor's Triton-template GEMM autotune costs 2.9 s/cell — 6.08x geomean the
Helion heuristic path — while Helion's own full search costs 15.75 min.** Three orders of magnitude
separate "template autotune" from "search the whole config space."

### 3c. Denominator: what one Triton compile costs here

From 25,474 `compile_time_s` records across the 180 quick-autotune CSVs
(`matmul_heuristic_perf_results/autotune_logs/*.csv`):
**median 0.56 s, mean 0.95 s, p10 0.20 s, p90 1.47 s, p99 7.92 s, max 30.06 s** (the 30 s cap).
Per-cell median compile time ranges 0.10–2.12 s.
So the heuristic's marginal cost is a few percent of **one** config's compile, and full autotuning
compiles ~521 of them.

### 3d. Direct CPU micro-benchmark of the analysis itself (new measurement, made for these notes)

No GPU is visible in the notes-writing sandbox, so this is CPU-only, using the repo's own unit-test
fact/spec stubs (`test/test_matmul_heuristics.py::_matmul_fact`, `_generalized_spec`,
`_block_sizes_stub`) with `helion.runtime.get_num_sm` patched to 148. `/home/dev/helion-env/bin/python`;
scripts kept at `/tmp/hb/bench_heuristic_cpu.py` and `/tmp/hb/bench_multi.py`.

Single-contraction front end, `TritonB200FormulaMatmulHeuristic._ranked_configs` (emits 9 ranked configs):

| shape | best-of-5 per call |
|---|---:|
| M1024 N1024 K1024 | 2.67 ms |
| M4096 N4096 K4096 | 2.69 ms |
| M8192 N8192 K512 | 2.83 ms |
| M256 N512 K32 | 2.03 ms |

Multi-contraction front end, `TritonB200MultiMatmulHeuristic._multi_ranked`:
**2 dots → 4.22 ms, 4 dots → 7.08 ms, 8 dots → 15.60 ms** (≈1.9 ms per additional dot).

cProfile attributes ~14,300 Python calls per single-contraction invocation, all inside
`autotuner_heuristics/triton.py` — it is genuine heuristic arithmetic, not `Config` normalization.
The `matmul_b200.json` table (9 KB) is loaded once behind `@functools.cache`.

**Caveats on 3d:** stubbed facts, not a real kernel; excludes fact collection and codegen; measured on
this box's CPU with no GPU present. It is a cross-check on the order of magnitude of 3a's +34–58 ms
(the extra also covers the other ~12 registered Triton heuristics' eligibility checks, which are all
skipped when `disable_autotuner_heuristics=1`), not a substitute for it.

### 3e. What is NOT measured (name this as a blog gap)

- **No artifact anywhere records the heuristic's own time inside a real compile.** `helion/_compiler/autotuner_heuristics/*` contains **zero** timing instrumentation (`grep perf_counter` → no hits), and no `results.json` in any run has a heuristic-selection duration field. Section 3a is the closest proxy and it only covers the *single-GEMM* formula front end on 49 clean BF16 GEMMs.
- **The multi-matmul front end's added compile latency is unmeasured on GPU** — which matters, because that is the front end the blog headlines (129 of 149 matmul cells) and it is the more expensive one on CPU (4–16 ms vs 2–3 ms).
- **How to measure it properly:** re-run `collect_config`-style isolated workers on the 129 Helion linear-attention cells with `HELION_DISABLE_AUTOTUNER_HEURISTICS` ∈ {0,1}, ≥10 repeats/arm, order-balanced, recording `selection_wall_seconds`; or add a `perf_counter` around `compiler_seed_configs()` in `helion/_compiler/autotuner_heuristics/__init__.py:92` and emit it into the compile log. Either is a one-afternoon job on the B200 box.

---

## 4. How autotuning cost scales in practice

### 4a. Per-shape retuning is the *default*, documented behaviour

- `docs/deployment_autotuning.md:428-431` — "**Default (`static_shapes=True`):** Helion
  shape-specializes on the exact shape/stride signature, **rerunning the selection whenever those
  shapes differ**. This delivers the best per-shape performance but requires all calls to match the
  example shapes exactly." (`static_shapes=True` is confirmed in every autotune `meta.jsonl` in this corpus.)
- `static_shapes=False` buckets dimensions at `{0, 1, ≥2}` only — one compiled kernel for many sizes,
  at a performance cost.
- `docs/deployment_autotuning.md:13-16` — "Helion will autotune on the **first call for each
  specialization key**… the first call pays a large tuning cost."

### 4b. Cache behaviour: the cache never crosses a GPU or a shape

- The on-disk autotune cache key includes **hardware** (`get_device_name(dev)`) plus the
  specialization key and a structural fingerprint —
  `helion/autotuner/local_cache.py:57-58,162`, `helion/autotuner/base_cache.py:118-151`.
- `docs/deployment_autotuning.md:166-168` — warm-start "matches cached configs by **hardware,
  specialization key, and structural fingerprint**, so it only reuses results that are structurally
  compatible."
- `HELION_FORCE_AUTOTUNE=1` re-tunes even on a hit; `HELION_SKIP_CACHE=1` disables read+write.
- A remote/shared cache exists (`RemoteAutotuneCache`, read-through/write-through) so a fleet can
  avoid "independently re-tuning identical kernels on identical hardware" — the same filtering applies.

### 4c. Does a tuned config transfer? Measured answer: to a new GPU, mostly no

`/home/dev/local/sm100-linattn/SM100_CONFIGS.json` (schema `sm100-linattn-configs/1`) holds **251 cells,
each with the SM90 seed and the retuned SM100 config side by side**, `seeded_from: "shipped sm90 AOT
heuristic"`, **`geomean_speedup: 1.1814`** — i.e. retuning on B200 bought **18.1% geomean** over
replaying the H100 config.

From `/home/dev/local/sm100-linattn/PROGRESS.md` + `PATTERNS.md` (both recomputed from that JSON):

- **Only 21 of 251 cells kept the SM90 config byte-identical** (`seed_kept`) → **92% of cells changed.**
- Transfer quality is shape-dependent and the *opposite* of intuition: **D=256 geomean 1.3122x
  (1/38 seed_kept), D=128 1.1820x (5/152), D=64 1.0767x (15/38 = 39% seed_kept)**. By total work:
  >70M elements 1.2400x vs 2–20M 1.1120x. "SM90 configs transfer **well on small/narrow** work and
  **poorly on large/wide** work."
- Systematic arch drift, not noise: `num_stages` moved **down** 77 times vs up 38 ("H100 configs are
  systematically over-staged for B200"); **19 cells moved off a persistent grid, only 2 onto one**
  (`persistent_blocked → flat` geomean **1.773x**).
- **Zero SM90 seeds use `block_ptr`; 46 tuned B200 configs do** (geomean 1.175x) — and
  `valid_indexing_types()` excludes `block_ptr` when TMA is supported, so *"no autotuner run on this
  hardware can produce these 46 configs."*

Corroborating cross-arch evidence from the pointwise audit: the RoPE `AOT` arm is an **SM90 table
replayed on B200** and scores **22.145x** vs default, while the compile-time heuristic scores
**27.943x** — the stale cross-arch table loses to the free heuristic by 26%
(`PYTORCH_BLOG_HEURISTICS_RESULTS.md` §7, flagged there as an SM90 table on B200).

### 4d. Does a tuned config transfer to an unseen *shape*? Documented: no

`docs/deployment_autotuning.md:875-880` — "If you add new shapes to the kernel's benchmark, regenerate
the heuristic the same way — **Helion does not extrapolate cleanly outside the shapes used during
`collect`/`measure`**." Both shipped table styles therefore carry an explicit fallback:

- linear-attention AOT: *"An unseen call falls back to the nearest tuned call with the same None/bool
  arguments, by total element count."*
- vLLM-derived tables: *"closest match on the hidden_size dimension(s), then the smallest tuned
  num_tokens ≥ the input (falling back to the largest)."*

Weak measured signal on that fallback (n=5, treat as anecdote): in
`/home/dev/local/wt-b200-perf-report-repro-plan/perf-repro/results/vllm__*.json`, **5 of the 16
"posted-number" shapes do not exact-key into vLLM's shipped grid**. On those 5 fallback shapes the
heuristic/vLLM ratio is **0.9852x**, vs **0.9659x** on the 11 exactly-tuned shapes — the tuned table's
edge shrinks when it has to interpolate. (Both computed from `arms.seed.us` / `arms.vllm_shipped.us`;
the 96-cell full tuned-grid sweep reproduces the published 0.9558x, and all 96 of those key exactly.)

### 4e. Warm-starting the search: measured savings, on the same 38 cells

`zero_seed_full_autotune/THREE_WAY_AUTOTUNE_TABLES.json` `summaries`, three full-effort arms:

| metric | no seed | old seeds | expanded seeds |
|---|---:|---:|---:|
| total autotune wall | 35,151.5 s (9.764 h) | 35,058.0 s (9.738 h) | **29,180.6 s (8.106 h)** |
| geomean ratio vs no seed | 1.0000x | 0.9572x | **0.8440x** |
| mean per cell | 15.42 min | 15.38 min | **12.80 min** |
| total configs | 19,796 | 18,923 | 16,936 |
| wall to within 5% of shared best (censored) | 19,885.5 s | 15,210.1 s | **13,901.0 s** (geomean 0.6141x) |
| final winner latency geomean (lower=better) | 1.0000x | 1.0005x | 1.0142x |

So seeding an *existing* search buys **~16% off a complete full search** and **~39% off time-to-within-5%**
— real, but it does not change the order of magnitude. The dramatic seed win is at loose tolerance:
within-50% recorded effort drops **97.1%** (2,683 → 78 configs, per `PYTORCH_BLOG_HEURISTICS_RESULTS.md` §4).

### 4f. What everyone actually ships instead of tuning at runtime: big per-shape tables

| Table | Tuned entries | Grid |
|---|---:|---|
| linear-attention AOT, sm100 | **256** | 26 kernels × (shapes × code-path flags) |
| linear-attention AOT, sm90 | **252** | same 26 kernels |
| `SM100_CONFIGS.json` (independent hand-tuned answer key) | **251** | same 26 kernels; differs from the AOT table in **262/262** compared cases (`CORPUSA_RESOLUTION_NOTES.md` D6) |
| vLLM `silu_mul_fp8` h100 | **308** | 7 intermediates × 51 token counts |
| vLLM `silu_mul_fp8` h200 | **308** | **byte-identical to the h100 file** (`md5 aa85b3c4…`) — transfer by assumption, and **there is no b200 file at all** |
| vLLM `dynamic_per_token_scaled_fp8_quant` h100 | 115 | 8 hidden × 15 token counts |
| vLLM `rms_norm_per_block_quant` h100 | 96 | 4 hidden × 2 group × 15 tokens |
| vLLM `rms_norm_dynamic_per_token_quant` h100 | 59 | |
| vLLM `fused_qk_norm_rope` h100 / b200 | 45 / 42 | 3 q_heads × 14–15 token counts |
| vLLM `dynamic_per_token…`, `per_token_group…`, `rms_norm_*` b200 | 42 each | 3 hidden × 14 token counts |
| **vLLM total** | **1,183 entries, 12 files, 6 kernels, 3 GPU SKUs** | `/home/dev/local/vllm-src/vllm/kernels/helion/configs/**/*.json` @ `b790c84c` |
| `pretuned_kernels/` in-tree AOT files | 30 files; **1,074 tuned shape keys** in the 25 files that list them (sm100 269, sm90 805); the remaining 5 files (`grouped_gemm`, `grouped_gemm_deepgemm`, `nvfp4_gemv`, `nvfp4_gemv_cute`, `rope`) use a structure with no key list. 24 of 30 are `decision_tree` heuristics whose leaf-config count equals the tuned-key count. | `pretuned_kernels/*/_helion_aot_*.py`, `pretuned_kernels/README.md` |
| SGLang KDA (contrast) | **14** hardcoded `helion.Config` objects, **one per kernel, no shape table** | `sglang-pr32593/python/sglang/kernels/ops/attention/helion/{kda_prefill,kda_decode,kda_replayssm}.py` |

Also note the *asymmetry* per kernel across SKUs (42 on B200 vs 115 on H100 for the same kernel): the
newer GPU has the *thinner* table. The B200 grids are all exactly 42 = 3 hidden sizes × 14 token counts;
H100's are 42–115. Whoever tuned B200 tuned a smaller grid, and `silu_mul_fp8` never got one.

---

## 5. Two sentences the author can use

### (A) The projection sentence — states its assumptions, corrects the headline

> "Autotuning the linear-attention engine from scratch means one full-effort search per entry in its
> shipped B200 config table: **256 searches across 26 kernels** — six production shapes fan out to
> 4–18 tuning keys per kernel, because each kernel is also keyed on the flags that pick its code path.
> At the cost we actually measured for such a search on this hardware — a **mean of 15.8 minutes**
> across 189 searches — that table is **67 GPU-hours, about 2.8 days**; at the median (10.7 min) it is
> **46 GPU-hours, about 1.9 days**. Either way it is days, not hours, and it has to be redone for every
> GPU generation."

Assumptions to state: (i) one full-effort search per key, no warm start, no cache hits;
(ii) the 189-search cost distribution — 19 of these same linear-attention kernels plus off-corpus
attention/matmul/state-space kernels — carries over to all 256 keys; (iii) one GPU (the runs used two,
which halves wall clock but not GPU-hours); (iv) the measured mean is inflated by a 2-hour
`jagged_hstu` outlier, hence the median given alongside.

Smaller variants if the author wants a tighter claim:

| corpus | at mean 15.75 min | at median 10.70 min | at linattn-only mean 14.06 min |
|---|---:|---:|---:|
| 26 kernels × 6 shapes = 156 | 41.0 GPU-h (1.71 d) | 27.8 GPU-h (1.16 d) | 36.6 GPU-h (1.52 d) |
| **256 shipped sm100 keys** | **67.2 GPU-h (2.80 d)** | 45.7 GPU-h (1.90 d) | 60.0 GPU-h (2.50 d) |
| 129 Helion linattn perf cells | 33.9 GPU-h (1.41 d) | 23.0 GPU-h (0.96 d) | 30.2 GPU-h (1.26 d) |
| 244-cell matmul perf corpus | 64.1 GPU-h (2.67 d) | 43.5 GPU-h (1.81 d) | 57.2 GPU-h (2.38 d) |
| 399 on-corpus curriculum cases | 104.8 GPU-h (4.36 d) | 71.2 GPU-h (2.96 d) | 93.5 GPU-h (3.90 d) |

(Do **not** apply the matmul-corpus rate to the 1,183 vLLM pointwise/reduction entries — those kernels
have far smaller search spaces and no measurement of their full-autotune cost exists here.)

### (B) The strictly-measured sentence — no assumptions at all

> "Across every full-effort autotuning search this work actually ran on B200 — **189 searches over 98
> distinct kernel/shape cells** — the bill was **49.6 GPU-hours**: a mean of **15.8 minutes** and a
> median of **10.7 minutes** per search, a 90th percentile of **29 minutes**, and a single worst cell at
> **2 h 21 m**. One run alone — 75 off-corpus kernel/shape cells — cost **22.0 GPU-hours** and 12 hours
> of wall clock on two B200s. The heuristic replaces each of those searches with **tens of milliseconds**
> of compile-time analysis: turning the compiler heuristics on raised Helion's config-selection step from
> **2.375 s to 2.433 s** (median delta **+34 ms**) across 49 GEMM cells."

Two more one-liners, both measured:

- "Sixteen percent of the configs a full search compiles **fail to compile** (3,102 of 19,796), and
  **75% of the search's wall clock elapses after the eventual winner has already been found**."
- "Dropping from full to quick autotuning cuts the cost **6x** (17.6 → 2.3 min/cell on the same 75
  cells) — and gives up **23%** of the performance (full/quick geomean **1.2336x**)."

---

## 6. Corrections to `/home/dev/PYTORCH_BLOG_HEURISTICS_RESULTS.md` and to the draft claim

1. **"25 kernels × 6 shapes" is wrong on both factors.** It is **26** Helion linear-attention kernel
   bodies (28 `@helion.kernel` in the engine, 26 in the curriculum/AOT table), and the tuning unit is
   the **call key** (shapes + code-path flags), of which the shipped sm100 table has **256**, i.e.
   **4–18 per kernel, not 6**. 150 undercounts by 41%.
2. **"nearly a day of GPU time" understates it.** 150 cells × 15 min = **37.5 GPU-h = 1.6 days**;
   256 keys × measured mean = **67 GPU-h = 2.8 days**. "Nearly a day" is only reachable at ~9.6
   min/cell, i.e. by using the *median* on the *undercounted* corpus. Say "days," or say
   "one to three days depending on whether you count shapes or call keys."
3. **`AOT`/`pre-tuned` is NOT full-autotune output in the 149-cell matmul table.** Verified: the
   manifest's `reference_config` for the 124–129 Helion linear-attention cells is verbatim
   `SM100_CONFIGS.json[cell]["sm100_config"]` (checked byte-for-byte for
   `chunk_bwd_dh_diag_fused#1`: `block_sizes=[32,64], l2_groupings=[8], num_stages=8, num_warps=8`),
   which `PROGRESS.md` describes as *"Seeded from sm90 and hand-tuned per cell"* — and which contains
   46 `block_ptr` configs that *no autotuner on this hardware can emit*. The other 25/40 cells use
   `reference_kind: shipped_helion_config` = SGLang's 14 hardcoded per-kernel `helion.Config`s.
   `on_corpus_pretuned/RESULTS.md` correctly calls the arm "Pre-Tuned"; the index doc's column heading
   "Full-autotuned / Pre" and its Quoting Note ("`AOT` and `pre-tuned` are configurations produced by
   full autotuning") are wrong for that table. **The e2e linear-attention `AOT` arm is fine** — its
   records show `autotune_cache: "AOTAutotuneCache"`, i.e. the 256-config table that the overnight log
   confirms "was produced with full autotuning."
4. **Two independent tuning campaigns, not one.** `CORPUSA_RESOLUTION_NOTES.md` D6: the 256-config AOT
   table and the 251-cell `SM100_CONFIGS.json` cover the same 26 bodies and key-match on all 262
   corpus-A cases, **yet differ in 262/262**. The blog should not imply a single canonical "tuned"
   answer exists.
5. **Cross-lane inconsistency in the quick-autotune arm.** The same 75 off-corpus cells give
   quick/heuristic = **1.5810x** in `off_corpus_main_autotune/results.json` (1.3205x excluding
   `jagged_hstu`) but **1.1736x** in `off_corpus_full_autotune_unseeded/results.json` (1.1797x excluding
   `jagged_hstu`) — different replay trees/timing lanes. Pick one lane and name it; do not average them.
6. **Minor, outside this dataset:** the index doc calls the broader reduction audit "455-cell", but
   `wt-b200-perf-report-repro-plan/.../summary.json` `headline` reports 188 reproduction + 257
   generalization = 445 attempted (252 of 257 valid). Worth a re-check before printing 455.
7. Every full-autotune run used **both** B200s. Do not present the recorded totals as wall clock —
   9.764 GPU-h landed in 5 h 52 m; 22.016 GPU-h landed in 11 h 59 m.

---

## 7. Gaps (name these; do not fabricate)

- **No log records the cost of building the shipped 256-config sm100 AOT table.** It predates this work;
  its cost can only be projected from the 189 searches measured here.
- **No measurement of the heuristic's time inside a real compile** beyond the 49 single-GEMM cells of
  §3a — in particular **nothing for the multi-matmul front end on GPU**, which is the blog's headline
  path. §3d's 4–16 ms is a CPU micro-benchmark on stubbed facts, made for these notes.
- **No full-autotune cost data for the reduction, pointwise, or vLLM families.** Their audits
  (114 / 36 / 445-ish cells) replay checked-in tables and record no tuning duration
  (`grep autotune|duration|seconds` over their `summary.json` → 0 hits). So the "1,183 vLLM configs
  cost N GPU-hours" line cannot be made with a measured rate.
- **No dtype-transfer measurement in this corpus.** The linear-attention corpus is BF16 only
  (chunk 64); the formula-matmul study has 6 FP16 cells but only in the unmeasured "full" tier
  (49 core cells measured are all BF16). Whether a bf16-tuned config serves fp16 is untested here.
- **No measured "nearest-key fallback penalty"** for the linear-attention AOT table (only the n=5
  vLLM anecdote in §4d). Every cell in the on-corpus report deliberately used an **exact** reference
  and forbade nearest-shape fallback (`MATMUL_HEURISTIC_PERF_CURRICULUM.md:14`).
- **No end-to-end wall clock for `python -m helion.autotuner.aot_runner`** (`collect`/`measure`/`emit`),
  which is the actual user-facing cost of producing an AOT table.
- **Quick-autotune's true "hours" story is incomplete:** only 178 of 244 cells were quick-tuned before
  the user stopped the run (`MATMUL_HEURISTIC_PERF_OVERNIGHT_LOG.md`, 2026-08-17 16:58 UTC), so the
  quick ledger is a partial corpus.
- **Sandbox limitation for anything new:** no GPU is visible from the notes-writing environment
  (`torch.cuda.is_available() == False`, no `/dev/nvidia*`, no `nvidia-smi`), so §3a/§3b/§2 could only be
  re-derived from stored artifacts and §3d could only be run on CPU.

---

## 8. File index (everything cited above)

Autotune cost:
- `/home/dev/local/wt-sm100-linattn/matmul_heuristic_perf_results/zero_seed_full_autotune/THREE_WAY_AUTOTUNE_TABLES.json` (`summaries`, `records[].arms.*.autotune_wall_seconds`)
- `/home/dev/local/wt-sm100-linattn/matmul_heuristic_perf_results/zero_seed_full_autotune/RESULTS.md` and `events.jsonl` and `cells/*/zero_seed/result.json`
- `/home/dev/local/wt-sm100-linattn/matmul_heuristic_perf_results/off_corpus_full_autotune_unseeded/{results.json,FINAL_AUDIT.md,RESULTS.md,events.jsonl}`
- `/home/dev/local/wt-sm100-linattn/matmul_heuristic_perf_results/multi_seed_full_autotune_ablation/events.jsonl`
- `/home/dev/local/wt-sm100-linattn/matmul_heuristic_perf_results/{results.json,events.jsonl}` (quick arm) and `autotune_logs/*.{log,csv,meta.jsonl}`
- `/home/dev/local/wt-sm100-linattn/matmul_heuristic_perf_results/off_corpus_main_autotune/{results.json,RESULTS.md,events.jsonl}`
- `/home/dev/local/wt-sm100-linattn/{MATMUL_ZERO_SEED_FULL_AUTOTUNE_OVERNIGHT_LOG.md,OFF_CORPUS_FULL_AUTOTUNE_OVERNIGHT_LOG.md,MATMUL_HEURISTIC_PERF_OVERNIGHT_LOG.md,MATMUL_HEURISTIC_OFF_CORPUS_MAIN_AUTOTUNE_OVERNIGHT_LOG.md}`

Heuristic compile cost:
- `/home/dev/local/wt-sm100-linattn/formula_matmul_49_results/configs/*/*.json` (`selection_wall_seconds`)
- `/home/dev/local/wt-sm100-linattn/formula_matmul_49_results/cells/*.json` (`compile.*.compile_and_first_call_wall_seconds`), `summary.json`
- `/home/dev/local/wt-sm100-linattn/scripts/formula_matmul_49_benchmark.py` (`collect_config`, `prepare_helion_runner`)
- `/home/dev/local/wt-sm100-linattn/helion/_compiler/autotuner_heuristics/{__init__.py,triton.py}`

Corpus definitions:
- `/home/dev/local/wt-sm100-linattn/{MATMUL_HEURISTIC_PERF_CURRICULUM.json,MATMUL_HEURISTIC_PERF_CURRICULUM.md,LINEAR_ATTENTION_E2E_BENCHMARK_PLAN.md,FORMULA_MATMUL_PERF_CURRICULUM.json,FORMULA_MATMUL_49_CELL_BENCHMARK_PLAN.md}`
- `/home/dev/local/wt-sm100-linattn/matmul_heuristic_perf_results/{manifest.json,on_corpus_pretuned/manifest.json,e2e_fla_v3/results.json}`
- `/home/dev/local/sm100-linattn/{LINATTN_HEURISTIC_CURRICULUM.json,SM100_CONFIGS.json,PROGRESS.md,PATTERNS.md,CORPUSA_RESOLUTION_NOTES.md}`

Shipped tables / transfer:
- `/home/dev/local/wt-sm100-linattn/examples/linear/_helion_aot_linear_attention_engine_cuda_sm{90,100}.py`
- `/home/dev/local/wt-sm100-linattn/pretuned_kernels/*/_helion_aot_*.py`, `pretuned_kernels/README.md`
- `/home/dev/local/vllm-src/vllm/kernels/helion/configs/**/*.json` (vllm-src @ `b790c84c`)
- `/home/dev/local/sglang-pr32593/python/sglang/kernels/ops/attention/helion/*.py`
- `/home/dev/local/wt-b200-perf-report-repro-plan/perf-repro/results/vllm*__*.json`

Documented cost/behaviour:
- `/home/dev/local/wt-sm100-linattn/docs/deployment_autotuning.md` (lines 13-21, 141, 166-168, 428-431, 859, 875-880)
- `/home/dev/local/wt-sm100-linattn/docs/index.md` (lines 31, 157, 162)
- `/home/dev/local/wt-sm100-linattn/helion/autotuner/{local_cache.py,base_cache.py}`
