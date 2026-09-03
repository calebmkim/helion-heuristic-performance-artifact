# Headline dataset #1 — End-to-end linear attention on B200 (multi-matmul heuristic)

Research notes for the PyTorch blog post. Everything below was re-derived from the raw
`results.json` (17 MB, 96 records) with `python3`, not from prose. Where a number came from
a prose report or a second dataset, that is said explicitly.

**Primary artifacts**

| What | Path |
|---|---|
| Human-readable report | `/home/dev/local/wt-sm100-linattn/matmul_heuristic_perf_results/e2e_fla_v3/RESULTS.md` |
| Raw per-cell data (rawest available) | `/home/dev/local/wt-sm100-linattn/matmul_heuristic_perf_results/e2e_fla_v3/results.json` |
| Per-arm raw samples (288 files) | `.../e2e_fla_v3/arms/<variant>__<mode>__<shape>/{pre_change,post_change,pretuned}.json` |
| Methodology manifest | `.../e2e_fla_v3/manifest.json` |
| Stability audit | `.../e2e_fla_v3/stability_audit.json` |
| Rerun aggregation (14 cells × 3 processes) | `.../e2e_fla_v3/process_repeat_aggregation.json` |
| Benchmark plan / protocol history | `/home/dev/local/wt-sm100-linattn/LINEAR_ATTENTION_E2E_BENCHMARK_PLAN.md` |
| Runner source | `/home/dev/local/wt-sm100-linattn/scripts/linear_attention_e2e_fla_v3.py` |
| Plot source | `/home/dev/local/wt-sm100-linattn/scripts/plot_linear_attention_e2e.py` |
| Shape/variant registry | `/home/dev/local/wt-sm100-linattn/benchmarks/run_linattn.py` |
| Shipped sm100 AOT table | `/home/dev/local/wt-sm100-linattn/examples/linear/_helion_aot_linear_attention_engine_cuda_sm100.py` |

`e2e_fla/` and `e2e_fla_v2/` in the same directory are **explicitly invalidated** by the plan
(v1: dispatch machinery inside the timed region; v2: cross-NUMA CPU migration). Never quote them.

---

## 1. Re-derivation of the four-arm geomeans — MATCHES the summary doc to 4 dp

Aggregation prescribed by the plan ("Aggregation And Deliverables"): per-variant geometric mean
across its six shapes, then geometric mean across the variant-level geomeans, forward and
forward+backward kept separate. I reproduced that exactly.

| Scope | Cells | Pre | FLA Triton | New heuristic | Full-autotuned AOT |
|---|---:|---:|---:|---:|---:|
| Forward | 54 | 1.0000x | **1.521680x** | **1.981624x** | **2.259873x** |
| Forward + backward | 42 | 1.0000x | **1.255921x** | **1.810769x** | **2.120399x** |
| All measured cells | 96 | 1.0000x | **1.399113x** | **1.904976x** | **2.197759x** |

Summary doc section 1 states 1.5217 / 1.9816 / 2.2599, 1.2559 / 1.8108 / 2.1204, and
1.3991 / 1.9050 / 2.1978. **All nine values match my re-derivation to 4 decimal places. No mismatch.**

Direct comparisons (also re-derived, also matching to 4 dp):

| Scope | Heuristic / Pre | Heuristic / FLA | AOT / Pre | AOT / heuristic | AOT / FLA |
|---|---:|---:|---:|---:|---:|
| Forward | 1.981624x | **1.302261x** (doc 1.3023) | 2.259873x | **1.140415x** (doc 1.1404) | 1.485117x |
| Forward + backward | 1.810769x | **1.441786x** (doc 1.4418) | 2.120399x | **1.170994x** (doc 1.1710) | 1.688322x |
| All 96 | 1.904976x | 1.361560x | 2.197759x | 1.153694x | 1.570823x |

**Aggregation is not load-bearing here.** Because the design is perfectly balanced (every
variant-mode group has exactly 6 shapes), the nested per-variant geomean equals a flat geomean
over all cells to machine precision: flat forward = 1.5217 / 1.9816 / 2.2599, flat fwd+bwd =
1.2559 / 1.8108 / 2.1204, flat 96 = 1.3991 / 1.9050 / 2.1978. This is a good sentence for the
blog: the headline does not depend on how you nest the mean.

I also recomputed every ratio from the recorded latencies rather than the recorded ratio fields;
max relative discrepancy = **0** (the ratio fields are exactly `pre_latency / arm_latency`).

**Total-time (sum of latency) view**, offered as an alternative to geomean because geomean over
shapes hides that the big shapes dominate wall-clock:

| Scope | Pre | Heuristic | AOT | FLA | Heur/Pre | Heur/FLA | AOT/Heur |
|---|---:|---:|---:|---:|---:|---:|---:|
| Forward (54 cells) | 118.21 ms | 60.11 ms | 49.91 ms | 69.31 ms | 1.966x | 1.153x | 1.204x |
| Fwd+bwd (42 cells) | 291.33 ms | 164.80 ms | 129.08 ms | 279.73 ms | 1.768x | 1.697x | 1.277x |
| All 96 | 409.53 ms | 224.91 ms | 178.98 ms | 349.04 ms | 1.821x | 1.552x | 1.257x |

Sum-of-time excluding the D256 shape: forward Heur/FLA 1.191x, fwd+bwd Heur/FLA 1.157x.

**Win counts (per cell, out of 96):** heuristic ≥ FLA in **77/96**; pre-change ≥ FLA in only
**20/96**; AOT ≥ FLA in **93/96**. Heuristic ≥ pre in **87/96**; heuristic ≥ 2.0x pre in **44/96**.
The heuristic beats the full-autotuned AOT config on **14/96** cells.

---

## 2. Exact benchmark setup (what a blog reader needs)

### Variants — real names as they appear in the data

Seven **dense** variants (each in `forward` and `forward_backward`, 6 shapes → 84 cells):
`vanilla_linear_attn`, `simple_gla`, `retention`, `full_gla`, `delta_rule`,
`gated_delta_rule`, `kda`.

Two **forward-only** variants (6 shapes each → 12 cells):
- `kda_fused` — the `kda` module re-run with `fused_preamble=True` (takes pre-activation inputs,
  so there is no backward row), on the six dense shapes.
- `kda_varlen` — the `kda` module re-run with `varlen=True` + `cu_seqlens`, on six **varlen** shapes.

Total **96 cells** = 84 + 6 + 6. Plot labels used in the figures: "Vanilla linear attn",
"Simple GLA", "Retention", "Full GLA", "Delta rule", "Gated delta rule", "KDA", "KDA fused",
"KDA varlen".

### Shapes

Six dense "production shapes", taken verbatim from
`flash-linear-attention/benchmarks/ops/registry` (per the comment in `benchmarks/run_linattn.py`).
Recorded shape tuple in the JSON is `[B, H, T, D, DV]`, with `D == DV`:

| Name | B | H | T | D (=DV) |
|---|---:|---:|---:|---:|
| `B1_T8192_H96_D128` | 1 | 96 | 8192 | 128 |
| `B2_T16384_H16_D128` | 2 | 16 | 16384 | 128 |
| `B4_T2048_H16_D128` | 4 | 16 | 2048 | 128 |
| `B4_T4096_H64_D128` | 4 | 64 | 4096 | 128 |
| `B8_T2048_H32_D256` | 8 | 32 | 2048 | 256 |
| `B8_T1024_H8_D64` | 8 | 8 | 1024 | 64 |

Six varlen shapes, lengths taken verbatim from FlashKDA's `benchmarks/bench_fwd.py`, each
totalling 8192 tokens:

| Name | Sequence lengths | H | D |
|---|---|---:|---:|
| `fixed_T8192_H96_D128` | `[8192]` | 96 | 128 |
| `fixed_T8192_H64_D128` | `[8192]` | 64 | 128 |
| `ragged_T8192_H96_D128` | `[1300, 547, 2048, 963, 271, 3063]` | 96 | 128 |
| `ragged_T8192_H64_D128` | `[1300, 547, 2048, 963, 271, 3063]` | 64 | 128 |
| `uniform_T8192_H96_D128` | `[1024] * 8` | 96 | 128 |
| `uniform_T8192_H64_D128` | `[1024] * 8` | 64 | 128 |

### Numerics, kernel, hardware

- **dtype: BF16** (`DTYPE = torch.bfloat16` in `examples/linear/linear_attention_harness.py`).
- **chunk size: 64** (`CHUNK_SIZE = 64` in `scripts/linear_attention_e2e_fla_v3.py:47`).
- **GPU: NVIDIA B200** (`measurement.device_name == "NVIDIA B200"` in every arm file), physical
  GPU 1 of a two-GPU box, forced with `CUDA_VISIBLE_DEVICES=1`, so each child sees it as `cuda:0`.
- **torch 2.12.0+cu130, triton 3.7.0, flash-linear-attention / fla_core 0.5.2** (recorded per arm).
- One end-to-end Helion call fans out into **2–10 constituent Helion kernels**:
  retention/simple_gla/vanilla fwd = 2, delta/gated_delta fwd = 3, full_gla fwd = 4, kda fwd = 5,
  kda_fused/kda_varlen fwd = 6, retention/simple_gla/vanilla fwd+bwd = 5,
  delta/gated_delta/full_gla fwd+bwd = 7, **kda fwd+bwd = 10**. Across the 96 cells there are
  **474 (kernel, call-key) constituent instances** spanning **26 distinct Helion kernels**
  (most frequent: `chunk_fwd_o_helion` 60, `chunk_fwd_h_diag_fused` 48,
  `chunk_fwd_h_delta_helion` 42, `chunk_fwd_wy_delta_helion` 42).

### Dispatch mode and timing methodology

- **Normal warmed eager dispatch. No CUDA graphs** (`measurement.cuda_graph == False`,
  `normal_eager_dispatch == True`, `HELION_BENCHMARK_CUDAGRAPH=0`) — chosen deliberately because
  the author's production benchmark uses normal dispatch. A dispatch probe wraps `Kernel.bind`
  and `BoundKernel.set_config` and recorded **0 calls to both** in every timed arm, i.e. no
  config-selection machinery inside the timed region.
- **Cold L2** before each timed repetition (`cold_l2 == True`).
- **3 rounds**; per round: 100 ms warmup then repetitions targeting 300 ms of timed work,
  min 5, max 2000. Adaptive in practice: the 75 µs vanilla/D64 cell runs 712 warmup + 2000 timed
  reps per round; the 4 ms retention/D256 backward cell runs 25 + 74.
- **Headline statistic = median of the three round arithmetic means**
  (`headline_statistic == "median_of_three_cold_l2_round_means"`). The arithmetic mean preserves
  the author's statistic; the median-across-rounds guards against one transient round.
- Every arm runs in a **fresh child process**; compilation is allowed on
  CPUs `0-23,25-47,96-119,121-143` (NUMA-node-0, GPU-1-local) and then **all ~131 process threads
  are pinned to reserved physical CPU 24** (SMT sibling 120 also excluded from compilation).
  This was added after v2 showed 46.7% cross-process FLA variation from CPU migration.
- **FLA is measured inside every Helion arm's process** (3 paired FLA measurements per cell); the
  cell's FLA headline is the median of those. FLA inputs are built as **independent leaf tensors in
  FLA's native time-first layout**, so no layout-conversion autograd nodes and no cross-repetition
  gradient accumulation contaminate FLA backward (a real bug that was found in the author harness
  and fixed for this run). Every differentiable leaf's `.grad` is set to `None` before each
  forward+backward call.
- Shared Triton cache directory across arms; `PYTHONHASHSEED=0`, `HELION_AUTOTUNE_RANDOM_SEED=0`.
- **Fairness to FLA:** the protocol explicitly warms "compilation, FLA autotuning, dispatch caches,
  and pure shape-derived caches before timing" (plan, Timing Protocol item 6), so FLA's own Triton
  autotuner has already converged when the timed window opens — FLA is not charged first-call
  autotune cost. FLA also gets inputs in its own native time-first layout (no transpose in the
  timed path).

### Arm definitions — precise

All four arms run the **same kernel source**. `_prepare_imports()` imports Helion from the
selected tree (asserting the import path), then forces the PR worktree to the front of
`sys.path` "so every arm benchmarks identical example and FLA adapter code". Only the compiler
differs between Pre and Post.

1. **`Pre` ("pre-change compiler heuristic" / plotted as "Helion default (pre-change)")** —
   Helion imported from a pristine copy of the pre-PR tree
   (`/home/dev/local/sm100-linattn/_pristine_9c46dd311`, intended upstream commit
   `9c46dd311` = `[pallas] compose narrowing views on resident VMEM refs (#3273)`, 2026-08-12),
   run with `HELION_AUTOTUNE_CACHE=LocalAutotuneCache`, `HELION_AUTOTUNE_EFFORT=none`,
   `HELION_AOT_MODE=disabled`. This is the **old compiler-selected default**, *not* the raw
   unseeded `_base_default_config()`. Do not describe it as "unseeded".
2. **`Post` ("new heuristic", plotted "Heuristic (ours)")** — same env
   (`LocalAutotuneCache`, effort `none`, AOT disabled) on the PR tree at commit `375363d8`
   = `[autotuner] extend the B200 matmul heuristic to multi-matmul kernels` (2026-08-17).
   Zero tuning time: the config is chosen by static analysis at compile time.
3. **`AOT` ("full-autotuned / pre-tuned")** — same PR tree, but
   `HELION_AUTOTUNE_CACHE=AOTAutotuneCache`, `HELION_AUTOTUNE_EFFORT=full`,
   `HELION_AOT_MODE=evaluate`, which replays the **shipped B200 configs produced by full
   autotuning**. Tuning time is not counted. This is the ceiling arm.
4. **`FLA`** — the handwritten Triton implementation from
   flash-linear-attention 0.5.2 (`fla_core` 0.5.2), called through
   `examples/linear/linear_attention_fla.py`. Note: for `kda_varlen` the reference is FlashKDA-style
   varlen FLA. These are real handwritten Triton kernels, not a torch.compile arm.

### "How many autotuned cells exist"

- **All 96 cells have a full-autotuned AOT arm.** Every one of the **474** constituent
  (kernel, call-key) instances resolved through `AOTAutotuneCache` at effort `full`, and all 474
  recorded `expected_subset_match == True` (the runner raises if the active config does not match
  the expected AOT/default selection, so this is an enforced gate, not a soft check).
- The shipped table `_helion_aot_linear_attention_engine_cuda_sm100.py` holds **256 tuned configs
  across 26 kernels**, keyed on tensor shapes plus the None/bool flags that pick a code path, with
  a nearest-same-flags fallback by element count. Its docstring: "Tuned with
  `HELION_AUTOTUNE_EFFORT=full` over the `benchmarks/run_linattn.py` SHAPES and VARLEN_SHAPES
  sweeps; each call keeps its own best config."
- **Caveat to state in the blog:** the AOT arm is therefore **in-distribution / on-corpus** — it was
  tuned on exactly these shapes and is measured on them. It is a fair "ceiling", not a held-out
  generalization result.

### Correctness gating

- Correctness ran before performance for every cell. **96 cells × 3 Helion arms = 288 arms, 0
  forward failures** (forward tolerance 0.02 relative error vs FLA; worst observed 0.0061 on
  gated_delta_rule B1_T8192_H96_D128).
- For the 42 fwd+bwd cells × 3 arms = **126 backward checks, 0 failures**, but the gate is
  **`backward_gate == "q"`**: only the **dq** gradient is gated, at tolerance 0.05. Other
  gradients are recorded but not gated, and the worst recorded is **dv rel error 0.168 on
  `kda::forward_backward::B8_T1024_H8_D64` (pretuned; 0.132 post, 0.037 pre)**. Almost certainly
  BF16 noise on the smallest shape, but "all gradients verified" would be an overclaim.

---

## 3. Per-variant table (re-derived; verified against RESULTS.md and the summary doc)

Every value below was recomputed from `results.json`; all 16 rows of the summary doc's
per-variant table match to 4 dp, and all columns of `RESULTS.md`'s per-variant table match.

| Variant | Mode | Heur/Pre | AOT/Pre | Heur/FLA | AOT/FLA | Pre/FLA | FLA/Pre | AOT/Heur |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `vanilla_linear_attn` | forward | 3.2689 | 3.5828 | 1.2728 | 1.3951 | 0.3894 | 2.5682 | 1.0961 |
| `vanilla_linear_attn` | fwd+bwd | 2.4520 | 2.8362 | 2.0281 | 2.3459 | 0.8271 | 1.2090 | 1.1567 |
| `simple_gla` | forward | 3.0176 | 3.3084 | 1.3967 | 1.5313 | 0.4629 | 2.1605 | 1.0964 |
| `simple_gla` | fwd+bwd | 2.3384 | 2.6389 | 1.6744 | 1.8895 | 0.7160 | 1.3966 | 1.1285 |
| `retention` | forward | 3.0006 | 3.2558 | 1.7975 | 1.9504 | 0.5990 | 1.6693 | 1.0851 |
| `retention` | fwd+bwd | 2.3447 | 2.6339 | 2.1690 | 2.4366 | 0.9251 | 1.0810 | 1.1234 |
| `full_gla` | forward | 1.9745 | 2.2174 | 1.5132 | 1.6993 | 0.7663 | 1.3049 | 1.1230 |
| `full_gla` | fwd+bwd | 1.9000 | 2.1040 | 2.0353 | 2.2539 | 1.0712 | 0.9335 | 1.1074 |
| `delta_rule` | forward | 1.8613 | 2.1409 | 1.3070 | 1.5033 | 0.7022 | 1.4241 | 1.1502 |
| `delta_rule` | fwd+bwd | 1.5272 | 1.7665 | **0.9624** | 1.1131 | 0.6301 | 1.5870 | 1.1567 |
| `gated_delta_rule` | forward | 1.7357 | 2.1537 | 1.0811 | 1.3415 | 0.6229 | 1.6055 | 1.2408 |
| `gated_delta_rule` | fwd+bwd | 1.3251 | 1.8664 | **0.8245** | 1.1613 | 0.6222 | 1.6071 | 1.4085 |
| `kda` | forward | 1.5158 | 1.7155 | 1.2367 | 1.3997 | 0.8159 | 1.2257 | 1.1318 |
| `kda` | fwd+bwd | 1.2348 | 1.4092 | 1.0888 | 1.2425 | 0.8817 | 1.1342 | 1.1412 |
| `kda_fused` | forward | 1.4501 | 1.6492 | 1.2674 | 1.4413 | 0.8740 | 1.1442 | 1.1372 |
| `kda_varlen` | forward | 1.1353 | 1.3771 | 1.0059 | 1.2202 | 0.8860 | 1.1286 | 1.2130 |

Note the `Pre/FLA` column: **pre-change Helion lost to handwritten FLA Triton in 15 of the 16
variant-mode groups** (only `full_gla` fwd+bwd at 1.0712x was above parity).
Overall pre/FLA = 0.66x forward, 0.80x fwd+bwd. That is the "before" picture: Helion's compiler
default was 1.2–2.6x *slower* than FLA on most variants, and the heuristic flips it to 1.30x
(forward) / 1.44x (fwd+bwd) *faster*, with no autotuning.

### Per-shape aggregates (re-derived; not in the summary doc)

Dense shapes, forward (8 variant-cells each; includes kda_fused):

| Shape | Heur/Pre | AOT/Pre | FLA/Pre | Heur/FLA | AOT/Heur |
|---|---:|---:|---:|---:|---:|
| `B1_T8192_H96_D128` | 2.4254 | 2.8340 | 1.9481 | 1.2450 | 1.1685 |
| `B2_T16384_H16_D128` | 2.5880 | 2.9609 | 1.8985 | 1.3632 | 1.1441 |
| `B4_T2048_H16_D128` | 2.0545 | 2.1843 | 1.4222 | 1.4447 | 1.0632 |
| `B4_T4096_H64_D128` | 2.5292 | 2.7970 | 2.0006 | 1.2642 | 1.1059 |
| `B8_T2048_H32_D256` | 2.8046 | 3.7610 | 2.5736 | 1.0898 | 1.3410 |
| `B8_T1024_H8_D64` | **1.0052** | **1.0016** | 0.5736 | 1.7525 | 0.9964 |

Dense shapes, forward+backward (7 variant-cells each):

| Shape | Heur/Pre | AOT/Pre | FLA/Pre | Heur/FLA | AOT/Heur |
|---|---:|---:|---:|---:|---:|
| `B1_T8192_H96_D128` | 2.0749 | 2.5530 | 1.7492 | 1.1862 | 1.2304 |
| `B2_T16384_H16_D128` | 2.0949 | 2.4468 | 1.6244 | 1.2897 | 1.1680 |
| `B4_T2048_H16_D128` | 1.8206 | 1.9666 | 1.3405 | 1.3581 | 1.0802 |
| `B4_T4096_H64_D128` | 2.0859 | 2.4863 | 1.7468 | 1.1941 | 1.1920 |
| `B8_T2048_H32_D256` | 2.1665 | 3.0440 | 0.9323 | 2.3238 | 1.4050 |
| `B8_T1024_H8_D64` | **0.9857** | **0.9776** | 0.6327 | 1.5580 | 0.9917 |

`kda_varlen` per-shape (1 cell each): heur/pre 1.2141 (fixed H64), 1.1484 (fixed H96),
1.0900 (ragged H64), 1.1017 (ragged H96), 1.1276 (uniform H64), 1.1339 (uniform H96);
heur/FLA 1.0365, 1.0348, 0.9665, 0.9691, 1.0184, 1.0125.

**The honest null: `B8_T1024_H8_D64`.** On the smallest shape (8×8×1024×64) the heuristic is a
no-op — heur/pre 1.0052 forward and 0.9857 fwd+bwd — and *the full-autotuned AOT arm is also a
no-op* (1.0016 / 0.9776). All three Helion arms tie inside a few percent, because these cells are
launch/dispatch-bound (75–780 µs end-to-end over 2–10 kernel launches), not config-bound. Helion
still beats FLA 1.75x / 1.56x there, but that is inherited, not earned by the heuristic. Excluding
this shape raises the heuristic's aggregate to heur/pre 2.2636x forward and 2.0450x fwd+bwd.

---

## 4. Best and worst cells; the two variants that lose to FLA; the D256 cliff

### Best

- Largest heuristic-vs-pre cell gains, all forward: `vanilla_linear_attn` B8_T2048_H32_D256
  **4.4767x**, `retention` B8_T2048_H32_D256 4.4752x, `simple_gla` B8_T2048_H32_D256 4.4746x,
  `vanilla_linear_attn` B2_T16384_H16_D128 4.3893x, `vanilla_linear_attn` B1_T8192_H96_D128 4.3422x.
- Best variant rows: `vanilla_linear_attn` forward 3.2689x heur/pre; `retention` fwd+bwd 2.1690x
  heur/FLA (excluding the two cliff cells below).
- Largest heuristic-vs-FLA cells are the two D256 backward cliff cells:
  `vanilla_linear_attn` fwd+bwd D256 **13.3229x** and `retention` fwd+bwd D256 **12.9399x**
  (see cliff discussion). Next best legitimate cells: `retention` forward B4_T2048_H16_D128
  2.3959x, `full_gla` fwd+bwd B2_T16384_H16_D128 2.3276x.

### Worst

- Worst heuristic-vs-pre cells are all on `B8_T1024_H8_D64` (the launch-bound shape):
  `simple_gla` fwd+bwd **0.9475x**, `kda` forward 0.9699x, `gated_delta_rule` fwd+bwd 0.9798x,
  `retention` forward 0.9842x, `delta_rule` forward 0.9849x. Nothing worse than −5.3% anywhere.
- Worst variant row vs FLA: `gated_delta_rule` fwd+bwd 0.8245x, then `delta_rule` fwd+bwd 0.9624x.
- Worst AOT/heuristic gap: `gated_delta_rule` fwd+bwd D256 **2.1796x** (AOT is 2.18x faster than
  the heuristic on that single cell) — the single biggest missed opportunity in the dataset.
- The heuristic **beats** AOT on 14 cells, best being `retention` forward B4_T2048_H16_D128
  (heuristic 108.7 µs vs AOT 129.3 µs → AOT/heur 0.8405x) and `vanilla_linear_attn` forward
  B4_T2048_H16_D128 (77.0 vs 83.4 µs, 0.9238x).

### The 19 cells where the heuristic loses to FLA Triton (of 96)

Sorted worst-first; latencies in µs.

| Cell | Heur/FLA | AOT/FLA | Heur | AOT | FLA | Heur/Pre |
|---|---:|---:|---:|---:|---:|---:|
| `gated_delta_rule` fwd+bwd `B8_T2048_H32_D256` | 0.5456 | 1.1892 | 16254.7 | 7457.7 | 8868.9 | 1.1795 |
| `gated_delta_rule` fwd+bwd `B1_T8192_H96_D128` | 0.7607 | 1.0673 | 6366.1 | 4537.4 | 4842.6 | 1.4692 |
| `delta_rule` fwd+bwd `B4_T4096_H64_D128` | 0.7882 | **0.9240** | 5978.8 | 5099.8 | 4712.5 | 1.6440 |
| `gated_delta_rule` fwd+bwd `B4_T4096_H64_D128` | 0.7918 | 1.0963 | 8083.4 | 5838.0 | 6400.2 | 1.5022 |
| `delta_rule` fwd+bwd `B1_T8192_H96_D128` | 0.7949 | **0.9416** | 4740.7 | 4002.3 | 3768.4 | 1.6199 |
| `gated_delta_rule` fwd+bwd `B4_T2048_H16_D128` | 0.8389 | 1.1548 | 1219.2 | 885.7 | 1022.8 | 1.3594 |
| `gated_delta_rule` fwd+bwd `B2_T16384_H16_D128` | 0.8513 | 1.1273 | 4442.5 | 3355.1 | 3782.0 | 1.5612 |
| `delta_rule` fwd+bwd `B2_T16384_H16_D128` | 0.8584 | **0.9709** | 3302.9 | 2920.1 | 2835.0 | 1.7122 |
| `delta_rule` fwd+bwd `B8_T2048_H32_D256` | 0.9072 | 1.2732 | 8867.5 | 6318.5 | 8044.7 | 1.7426 |
| `gated_delta_rule` forward `B1_T8192_H96_D128` | 0.9254 | 1.1594 | 1333.4 | 1064.2 | 1233.9 | 1.7943 |
| `gated_delta_rule` forward `B8_T2048_H32_D256` | 0.9378 | 1.3803 | 2327.3 | 1581.1 | 2182.5 | 2.2671 |
| `gated_delta_rule` forward `B4_T4096_H64_D128` | 0.9589 | 1.1042 | 1625.1 | 1411.4 | 1558.4 | 2.0240 |
| `kda_fused` forward `B8_T2048_H32_D256` | 0.9649 | 1.3205 | 4025.5 | 2941.3 | 3884.1 | 1.6831 |
| `kda_varlen` forward `ragged_T8192_H64_D128` | 0.9665 | 1.1908 | 1866.6 | 1515.0 | 1804.2 | 1.0900 |
| `kda_varlen` forward `ragged_T8192_H96_D128` | 0.9691 | 1.1930 | 2630.2 | 2136.6 | 2548.9 | 1.1017 |
| `kda` fwd+bwd `B4_T4096_H64_D128` | 0.9885 | 1.1622 | 14630.0 | 12443.2 | 14461.2 | 1.2575 |
| `delta_rule` forward `B8_T2048_H32_D256` | 0.9900 | 1.4220 | 2135.7 | 1486.9 | 2114.4 | 2.2766 |
| `kda` forward `B8_T2048_H32_D256` | 0.9932 | 1.3529 | 3718.7 | 2729.8 | 3693.3 | 1.7605 |
| `vanilla_linear_attn` forward `B8_T2048_H32_D256` | 0.9944 | 1.2960 | 831.6 | 638.1 | 827.0 | 4.4767 |

Note the last row: that cell is simultaneously the **largest heuristic-over-pre win in the dataset
(4.4767x)** and a 0.9944x tie/loss against FLA — a useful illustration that "beat the old default
by 4.5x" and "match handwritten Triton" are different claims.

Only **3 cells** are losses for the *AOT* arm too, and all three are `delta_rule` fwd+bwd
(0.9240x, 0.9416x, 0.9709x). So `delta_rule` backward is the one family where FLA's handwritten
Triton genuinely beats even fully-autotuned Helion; everywhere else a loss is a *heuristic*
shortfall, not a Helion shortfall.

### What drives the two losing variant rows

**`gated_delta_rule` fwd+bwd = 0.8245x heur/FLA.** All six shapes lose (0.5456–0.8513x); the
geomean is not one outlier. But the depth is set by `B8_T2048_H32_D256` (0.5456x) and, in log
terms, the whole row is a *heuristic* problem: AOT/FLA for the same row is 1.1613x and
AOT/heuristic is **1.4085x** — the largest AOT-over-heuristic gap in the dataset. The heuristic
captures only **45.1%** of AOT's log-space gain here.
The most likely single culprit, visible in the recorded configs, is
`chunk_bwd_dqkw_delta_helion` (call key `...:3d27556e5242fd46`, 5 block-size knobs): pre picks
`[16,16,16,16,16]`, the heuristic grows only the first three to `[64,64,64,16,16]` with
`num_warps=4, num_stages=3`, while the AOT winner is `[16,32,32,32,128]` with
`num_warps=2, num_stages=4` and a `[[0,1]]` loop order — i.e. the heuristic grows the wrong three
axes. **Corroborating (but methodologically separate) evidence:** the constituent-kernel report
`matmul_heuristic_perf_results/on_corpus_pretuned/RESULTS.md` isolates that exact call at
B8_T2048_H32_D256 as pre 8213.6 µs / post 9059.2 µs / pre-tuned 2232.2 µs — post is a **0.9067x
regression vs pre** and AOT is **4.06x faster than post** on that one kernel. That single kernel is
of the same order as the whole-cell 8.8 ms heuristic→AOT gap. Caveat: that report was measured at
a *different* post commit (`ccfcfbdd`) with CUDA-graph timing and records a different AOT config
(`[128,128,64,256,128]`, 8 warps) for the same call, so treat it as directionally corroborating,
not as an exact decomposition of the e2e cell.

**`delta_rule` fwd+bwd = 0.9624x heur/FLA.** Four of six shapes lose (0.7882–0.9072x) and two win
(`B4_T2048_H16_D128` 1.0113x, `B8_T1024_H8_D64` 1.6100x). This row is *not* mostly a heuristic
shortfall: AOT/heuristic is only 1.1567x and AOT/FLA is 1.1131x, with three cells where even AOT
loses to FLA. FLA's chunked delta-rule backward is simply strong on B200. The heuristic still
improves delta_rule fwd+bwd by 1.5272x over pre-change Helion.

### The "D256 FLA backward cliff" — what it is, and it HELPS Helion's aggregate

**What it is.** On the `B8_T2048_H32_D256` shape (head dim 256), FLA's Triton *backward* falls off
a cliff for two variants. FLA fwd+bwd / FLA fwd latency ratio, per variant × shape (re-derived):

| Variant | B1_T8192_H96 | B2_T16384_H16 | B4_T2048_H16 | B4_T4096_H64 | **B8_T2048_H32_D256** | B8_T1024_H8_D64 |
|---|---:|---:|---:|---:|---:|---:|
| `vanilla_linear_attn` | 4.62x | 4.60x | 4.17x | 4.52x | **63.04x** | 4.12x |
| `simple_gla` | 4.70x | 4.04x | 3.71x | 4.77x | **9.02x** | 3.67x |
| `retention` | 3.30x | 3.40x | 2.87x | 3.46x | **38.24x** | 2.71x |
| `full_gla` | 4.21x | 4.31x | 4.23x | 4.70x | 4.40x | 3.14x |
| `delta_rule` | 2.72x | 2.76x | 3.01x | 2.65x | 3.80x | 2.93x |
| `gated_delta_rule` | 3.92x | 3.95x | 3.15x | 4.11x | 4.06x | 3.14x |
| `kda` | 4.78x | 4.38x | 4.45x | 4.81x | 5.10x | 2.89x |

Helion's own fwd+bwd/fwd ratio at D256 stays in family (vanilla 4.71x, retention 4.41x). So the
cliff is FLA-specific: FLA vanilla backward at D256 takes **52.14 ms** (vs 3.91 ms Helion
heuristic, 2.81 ms AOT) and FLA retention backward takes **51.70 ms** (vs 4.00 / 3.08 ms). It is
reproducible, not a measurement artifact: the plan's progress log records a dedicated
investigation — author-style non-leaf FLA inputs gave ~52.72 ms and independent time-first leaves
~51.49 ms, and round spread on that cell was **below 0.3% for every arm**. It is not a bug in the
harness; it is FLA's own D256 backward path.

**Direction of effect: the cliff HELPS Helion.** Sensitivity of the forward+backward aggregate:

| fwd+bwd variant | Cells | FLA/Pre | Heur/Pre | AOT/Pre | **Heur/FLA** | AOT/FLA |
|---|---:|---:|---:|---:|---:|---:|
| As published (all) | 42 | 1.2559 | 1.8108 | 2.1204 | **1.4418** | 1.6883 |
| Drop the 2 cliff cells only | 40 | 1.3753 | 1.7856 | 2.0714 | **1.2983** | 1.5062 |
| Drop the whole D256 shape | 35 | 1.3330 | 1.7470 | 1.9725 | **1.3105** | 1.4797 |

So the two cliff cells are worth about **+11%** on the published fwd+bwd heuristic-vs-FLA headline
(1.4418x → 1.2983x without them). **Recommended blog framing:** quote 1.44x with the caveat, or
quote "≈1.30x forward and ≈1.30x forward+backward once FLA's two anomalous D256 backward cells are
removed" — the conservative statement is remarkably clean because forward (1.3023x) and
cliff-free backward (1.2983x) agree. Note the cliff *hurts* the heuristic's heur/pre number
slightly (1.8108 → 1.7856 without it) since Pre is a Helion arm; the cliff only moves FLA-relative
numbers.

Symmetrically, the forward aggregate is *helped* by dropping the tiny shape and *hurt* by dropping
D256: forward heur/FLA is 1.3023x published, 1.3519x without D256, 1.2424x without B8_T1024_H8_D64.

---

## 5. How much of the full-autotune ceiling the heuristic captures

Three equivalent framings (all re-derived):

- **Ratio framing:** AOT/heuristic = **1.1404x forward**, **1.1710x fwd+bwd**, 1.1537x over all 96.
  Equivalently the heuristic's latency is **14.0% / 17.1% / 15.4% above** the fully-autotuned
  configuration.
- **Log-space capture** (`log(heur/pre) / log(AOT/pre)`, i.e. share of the achievable speedup
  captured): **83.9% forward**, **79.0% fwd+bwd**, 81.8% overall.
- **Total-time framing:** summed over cells, AOT is 1.204x (forward) / 1.277x (fwd+bwd) /
  1.257x (all) faster than the heuristic.

Per variant (AOT/heuristic, and log-space capture):

| Variant | Mode | Heur/Pre | AOT/Pre | AOT/Heur | Log-capture |
|---|---|---:|---:|---:|---:|
| `retention` | forward | 3.0006 | 3.2558 | **1.0851** | **93.1%** |
| `vanilla_linear_attn` | forward | 3.2689 | 3.5828 | 1.0961 | 92.8% |
| `simple_gla` | forward | 3.0176 | 3.3084 | 1.0964 | 92.3% |
| `full_gla` | fwd+bwd | 1.9000 | 2.1040 | 1.1074 | 86.3% |
| `full_gla` | forward | 1.9745 | 2.2174 | 1.1230 | 85.4% |
| `retention` | fwd+bwd | 2.3447 | 2.6339 | 1.1234 | 88.0% |
| `simple_gla` | fwd+bwd | 2.3384 | 2.6389 | 1.1285 | 87.5% |
| `kda` | forward | 1.5158 | 1.7155 | 1.1318 | 77.1% |
| `kda_fused` | forward | 1.4501 | 1.6492 | 1.1372 | 74.3% |
| `kda` | fwd+bwd | 1.2348 | 1.4092 | 1.1412 | 61.5% |
| `delta_rule` | forward | 1.8613 | 2.1409 | 1.1502 | 81.6% |
| `delta_rule` | fwd+bwd | 1.5272 | 1.7665 | 1.1567 | 74.4% |
| `vanilla_linear_attn` | fwd+bwd | 2.4520 | 2.8362 | 1.1567 | 86.0% |
| `kda_varlen` | forward | 1.1353 | 1.3771 | 1.2130 | 39.7% |
| `gated_delta_rule` | forward | 1.7357 | 2.1537 | 1.2408 | 71.9% |
| `gated_delta_rule` | fwd+bwd | 1.3251 | 1.8664 | **1.4085** | **45.1%** |

Pattern worth writing up: the heuristic is near-ceiling (91–93% capture, ≤10% behind AOT) on the
**simple, few-kernel gated variants** (retention, vanilla, simple_gla — 2 kernels forward, 5 with
backward), and furthest from ceiling on the **many-knob, many-kernel delta family and varlen**
(`gated_delta_rule` fwd+bwd 7 kernels, `kda_varlen` 6 kernels, 5-block-size call keys).

**Mechanistic reason for the residual gap** (from
`/home/dev/local/sm100-linattn/MATMUL_HEURISTIC_HIGH_LEVEL_TRACE.md`): the heuristic actively
chooses only `block_sizes`, `num_warps`, `num_stages`, and (single-contraction path only)
`l2_groupings`; "other config fields retain their normal base defaults". The autotuner also owns
`indexing` (pointer vs `tensor_descriptor`), `pid_type` (`flat` vs `persistent_blocked` /
`persistent_interleaved`), `loop_orders`, `range_unroll_factors`, `range_num_stages`,
`range_warp_specializes`, `range_flattens`, `load_eviction_policies`, `maxnreg`,
`num_sm_multiplier`. Measured in the raw configs across all **474** constituent instances:

- The heuristic changes the config vs pre-change on **468/474** instances (only 6 unchanged), and
  changes `block_sizes` on **356/474 (75.1%)**.
- The heuristic's config is **never byte-identical** to the AOT config (0/474) — but it **agrees
  with AOT on `block_sizes` for 190/474 (40.1%)** of constituent kernels. The keys that differ
  most are exactly the ones the heuristic does not set: `indexing` 450/474,
  `load_eviction_policies` 434, `num_stages` 343, `range_unroll_factors` 308.

A concrete quotable example (`vanilla_linear_attn` forward `B1_T8192_H96_D128`, kernel
`chunk_fwd_h_diag_fused`):

- Pre: `block_sizes=[32,32]`, `num_warps=4`, `num_stages=1`, `pid_type=flat`, pointer indexing.
- Heuristic: `block_sizes=[128,128]`, `num_warps=8`, `num_stages=6` — same tile as AOT.
- AOT: `block_sizes=[128,128]` **plus** `indexing=[tensor_descriptor, pointer, tensor_descriptor]`,
  `pid_type=persistent_interleaved`, `l2_groupings=[64]`, `loop_orders=[[2,1,0]]`,
  `range_num_stages=[3,0]`, `range_warp_specializes=[True,None]`, `maxnreg=256`, `num_warps=2`.

That cell: pre 1588.9 µs → heuristic 365.9 µs (4.34x) → AOT 332.5 µs (4.78x). The tile size is
most of the win; the remaining ~10% is the knobs the heuristic deliberately leaves alone.

---

## 6. Figures in `.../e2e_fla_v3/` (all 9, with what each shows)

All generated by `scripts/plot_linear_attention_e2e.py` from `results.json` (schema-gated: the
script refuses anything but `linear-attention-e2e-fla-results/3` and requires exactly six shapes
per variant). Every figure has 9 or 7 variant groups plus a shaded bold **"Overall geomean"** group,
value labels on every bar, and the footer "NVIDIA B200 | BF16 | Corrected native-input FLA |
Pinned eager timing". Bar colors: pre-change `#59636E` grey, heuristic `#087F8C` teal,
full-autotuned `#7C3E8C` purple, FLA `#D97706` orange.

**Recommended for the blog (three-arm-vs-FLA family — shows the flip from below to above parity):**

1. `linear_attention_e2e_fla_baseline_with_default_forward.png` (154 KB) — Forward, baseline =
   FLA Triton dotted at 1.000x, three bars per variant (Helion default pre-change / Helion
   heuristic / Helion full-autotuned) over the 9 forward variants. **Verified label values:**
   overall geomean 0.66 / 1.30 / 1.49; pre-change is below the FLA line for all 9 variants
   (0.39–0.89); the heuristic is at or above 1.0 for **all 9** (closest: gated delta rule 1.08,
   KDA varlen 1.01), so no forward variant sits below FLA after the change. Best group:
   retention 0.60 / 1.80 / 1.95.
2. `linear_attention_e2e_fla_baseline_with_default_forward_backward.png` (174 KB) — same, forward +
   backward, 7 dense variants. **Verified labels:** overall 0.80 / 1.44 / 1.69; the two losing
   groups are visible as sub-1.0 teal bars (delta rule 0.96, gated delta rule 0.82); footer adds
   "F+B includes both forward and backward".
3. `linear_attention_e2e_fla_baseline_with_default_combined.png` (284 KB) — the two panels above
   stacked in one figure (forward on top, forward+backward below). Best single figure if the blog
   only has room for one.

**Pre-change-baseline family (headline "what the heuristic bought us", FLA shown as a bar):**

4. `linear_attention_e2e_prechange_baseline_forward.png` (185 KB) — Forward, baseline = pre-change
   Helion dotted at 1.000x, bars = FLA Triton / Heuristic (ours) / Full-autotuned.
5. `linear_attention_e2e_prechange_baseline_forward_backward.png` (175 KB) — same for fwd+bwd.
6. `linear_attention_e2e_prechange_baseline_combined.png` (294 KB) — both panels stacked.
   **Verified labels:** forward overall 1.52 / 1.98 / 2.26 with per-variant heuristic bars
   3.27 / 3.02 / 3.00 / 1.97 / 1.86 / 1.74 / 1.52 / 1.45 / 1.14; fwd+bwd overall 1.26 / 1.81 / 2.12
   with heuristic bars 2.45 / 2.34 / 2.34 / 1.90 / 1.53 / 1.33 / 1.23. Note in the fwd+bwd panel
   FLA's bar exceeds the heuristic's for delta rule (1.59 vs 1.53) and gated delta rule
   (1.61 vs 1.33) — the same two losses, seen from the pre-change baseline.

**Two-arm-vs-FLA family (drops the pre-change bar; cleaner but hides the "before"):**

7. `linear_attention_e2e_fla_baseline_forward.png` (154 KB) — forward, FLA = 1.000x, only heuristic
   + full-autotuned bars.
8. `linear_attention_e2e_fla_baseline_forward_backward.png` (155 KB) — same, fwd+bwd.
9. `linear_attention_e2e_fla_baseline_combined.png` (248 KB) — both panels stacked.

All figures are 200 dpi PNG with a white facecolor. The plotted "Overall geomean" bar is the
geomean over the plotted per-variant values, which (because groups are balanced) is numerically
identical to the flat per-cell geomean in section 1 — I verified the printed labels against my
re-derivation for figures 1, 2 and 6.

---

## 7. Stability, and the caveats a careful reviewer will raise

**Stability process (strong).** First pass: 96 cells × 3 Helion arms = 288 arm processes, 0 hard
errors, 0 correctness failures, 0 dispatch-probe or expected-config failures. The audit
(thresholds: 5% within-process round spread, 3% cross-process FLA spread) flagged 14 cells; each
was rerun in **three additional fresh full processes**, preserved under `e2e_fla_v3_first_pass`,
`_second_pass`, `_third_pass`. Final Helion latency for a reran cell = median of the three process
headlines; final FLA latency = median of all nine arm-paired process measurements. Nothing was
discarded. For the reran cells the method cross-process spreads are tiny (e.g. 0.08% on
`delta_rule::forward_backward::B4_T2048_H16_D128`).

**In the consolidated dataset (384 arm-timings):** median relative spread **0.065%**;
9 timings above 5%; 23 above 3%. The worst is a **48.3% FLA process spread** on
`delta_rule::forward_backward::B4_T2048_H16_D128`, driven by exactly one outlier of nine
measurements (1246.3 µs vs the other eight at 834.5–864.8 µs); the median 852.6 µs is robust.
That cell is one of the near-parity cells (heur/FLA 1.0113x), so its ratio should be treated as
"parity, unresolved" rather than a win.

**Caveats / limits (state at least the starred ones in the blog):**

- \* **The tiny shape is a null.** `B8_T1024_H8_D64` gets nothing from the heuristic *or* from full
  autotuning (≈1.00x both). Small launch-bound linear attention is not a config problem.
- \* **The FLA D256 backward cliff inflates the fwd+bwd FLA comparison by ~11%** (section 4).
- \* **The AOT arm is on-corpus**: those configs were autotuned on exactly these shapes.
- \* **`gated_delta_rule` and `delta_rule` backward still lose to FLA** (0.82x, 0.96x).
- **"Pre" is a 14-commit window, not a one-commit A/B.** The pre tree is upstream `9c46dd311`
  (2026-08-12) and the post tree is `375363d8` (2026-08-17); between them sit 14 upstream commits,
  including `runtime: restore late-specialized eager dispatch performance (#3383)` and
  `[examples] Cache the varlen chunk tables and fuse q's l2norm in kda_varlen (#3361)`. Kernel
  source is neutralized (both arms import examples from the PR worktree by explicit `sys.path`
  ordering), but the *runtime* differs. Counter-evidence that dispatch-overhead drift matters:
  on the launch-bound shape where a dispatch fix would show up most, heur/pre is 1.0052x /
  0.9857x — i.e. no visible dispatch-side gift.
- **Provenance nit in the raw data:** each record's `versions.pre_change.helion_sha` is
  `7f2ee6d80…`, which is the *enclosing* repo's HEAD, not the pristine copy's — `_pristine_9c46dd311`
  has no `.git`, so `git rev-parse HEAD` walked up one directory. The intended pre-change SHA is
  `manifest.json → pre_change_sha = 9c46dd311bcba596324fe5cba996495bda3e65fb`. Use the manifest.
- **The measured "post" tree is not today's HEAD.** The benchmark ran at `375363d8`; on branch
  `calebmkim/stack/50` that commit was rebased into `9d835ee5f`, and HEAD (`40151a23`,
  "[autotuner] harden multi-matmul work and grid modeling") rewrites
  `helion/_compiler/autotuner_heuristics/triton.py` by ~778 lines, with further uncommitted
  working-tree edits on top. These numbers are a snapshot; they are not guaranteed to reproduce at
  HEAD.
- **Backward correctness gate is dq-only** at tol 0.05; worst ungated gradient error is dv 0.168 on
  `kda` fwd+bwd `B8_T1024_H8_D64` (post and AOT).
- **Headline statistic is a mean, so sub-100 µs cells carry launch-tail inflation.** On
  `B8_T1024_H8_D64` forward, the headline (median of round means) sits 4.8–7.8% above the median of
  round medians — but uniformly across all four arms, so the ratios are essentially unaffected.
- **Do not merge these numbers with the SGLang KDA table** in the summary doc: that one uses
  CUDA-graph operation timing, while this one is deliberately eager.
- GPU clocks were not recorded as locked in the manifest (no `nvidia-smi -lgc` evidence). All arms
  for a cell are measured in interleaved fresh processes on the same GPU, so drift affects arms
  symmetrically, but absolute latencies are not clock-guaranteed.

---

## 8. Quick "best sentences" candidates

- "Across 96 end-to-end linear-attention cells on a B200, the compile-time heuristic makes Helion
  **1.98x faster on forward and 1.81x on forward+backward** than the previous compiler default,
  with **no autotuning at all**."
- "It flips the comparison against handwritten Triton: Helion's old default was **0.66x FLA**
  forward and **0.80x** with backward; the heuristic is **1.30x** and **1.44x**."
- "That is **84% (forward) / 79% (forward+backward)** of the log-space speedup that a full
  autotune achieves — the heuristic ends up **14–17% off** the pre-tuned ceiling, from a compile-time
  static analysis instead of a search."
- "Pre-change Helion lost to FLA in 15 of 16 variant-mode groups, and in 76 of 96 individual cells.
  After the heuristic it wins **77 of 96 cells**, and every one of the nine forward variant groups."
- "The heuristic sets four knobs — `block_sizes`, `num_warps`, `num_stages`, `l2_groupings` — and it
  agrees with the fully autotuned tile shape on **40% of 474 constituent kernels**. On
  `chunk_fwd_h_diag_fused` at B1/T8192/H96/D128 it moves `[32,32]`→`[128,128]`, taking 1589 µs to
  366 µs; the autotuner's extra 10% comes from `indexing`, `pid_type` and range scheduling, which
  the heuristic deliberately doesn't touch."
