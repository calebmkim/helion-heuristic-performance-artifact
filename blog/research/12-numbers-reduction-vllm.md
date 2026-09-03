# Headline dataset #3 — the reduction heuristic, with vLLM as the headliner

**Status of this file:** every number below was re-derived by me from the rawest available artifact
(`summary.json` / the per-cell `raw/*.json` / the per-corpus `<corpus>__<kernel>.json`), not copied
from a prose summary. Where a claim in `/home/dev/PYTORCH_BLOG_HEURISTICS_RESULTS.md` does not
survive that check, it is called out in **CORRECTIONS**. Where a number could not be established it
is in **GAPS**.

There are **two different reduction audits** and they must never be merged. Read the "two audits"
table first.

---

## 0. The two audits at a glance (DO NOT MERGE)

| | **Audit A — "definitive 114-cell four-arm audit"** | **Audit B — "broader reproduction/generalization audit"** |
|---|---|---|
| Root | `/home/dev/local/wt-b200-reduction-audit-run/perf-repro/b200-reduction-audit/results/full/` | `/home/dev/local/wt-b200-perf-report-repro-plan/perf-repro/results/` |
| Cells | **114 planned, 114 recorded, 113 valid** | **455 cells** (all recorded) |
| Helion revision | `2d753a5c6e855638a8947a296484e187b4ea0bbf` (branch `b200-reduction-audit-run`) | the PR **#2996** head rebased on `upstream/main` (branch `perf-report-repro`) |
| Heuristic that fired | `triton_reduction_tile_sm100` (69 cells) + `triton_reduction_user_tile_sm100` (45) — **the B200-retuned split** | `triton_reduction_tile` (227) + `triton_reduction_user_tile` (228) — **pre-sm100-split version** |
| Fire rate | 114/114 (`heuristic_no_seed` is an empty list) | 455/455 |
| 4th arm | `aot_sm100` = the checked-in `_helion_aot_<k>_cuda_sm100.py` selector | `vllm_shipped` = a live read of vLLM's tuned JSON |
| For vLLM kernels the 4th arm is | vLLM's **B200** table (`nvidia_b200.json`, ported to `_helion_aot_*_cuda_sm100.py`) | vLLM's **H100** table (`nvidia_h100.json`) — **verified from the raw configs, see §4.4** |
| Metric | cold-L2 **CUDA-graph device time** for every arm and every kernel | eager **cold-L2 interleaved `do_bench`** for functional kernels; **cold-L2 cudagraph device time** for the vLLM-family corpora (`vllm`, `vllm_gen`, `qk_norm_rope_gen`) |
| Rounds | median of 9 round medians, escalate to 15 rounds when relative spread > 5%; ~100 ms/round, reps clamped `[5,1000]` | median-of-9 interleaved rounds, escalate to 15 on >5% spread; 100 ms/round budget + 25 ms warmup, reps clamped `[5,1000]` |
| Accuracy-fail policy | **exclude** the cell from all ratios (n drops) | **"close-enough" rule**: include if the *default* makes the identical mistake; exclude only if the seed is wrong while the default is right |
| Env | PyTorch `2.12.0+cu130` (`7661cd9c6`), Triton `3.7.0`, CUDA rt `13.0` / driver `595.71.05`, `NVIDIA B200` SM100, physical GPU 1, `CUDA_VISIBLE_DEVICES=1`, `environment_consistent: true` across all 114 cells | same box/venv (`/home/dev/helion/.venv`, Triton 3.7.0); run wall time ~95 min (5688 s) |
| Charts | `results/charts/*.png` (5 files) | **none** (no PNGs exist anywhere under `perf-repro/`) |

Both audits state the ratio convention identically: **ratio = baseline latency ÷ heuristic latency,
so >1 means the heuristic is faster.** Audit A's PLAN spells it out: `G_default = default_us /
seed_us`, `G_tc = torch_compile_us / seed_us`, `G_aot = aot_sm100_us / seed_us`.

Arm definitions to reuse verbatim in the blog:

- **heuristic / "seed"** = `config_spec.compiler_seed_configs[0]`, the first config the compile-time
  reduction heuristic emits. It is *also* the no-autotune compiler default
  (`promote_seed_to_default = True` on `_TritonReductionSeedBase`). No online autotuning anywhere in
  either audit.
- **default** = `config_spec._base_default_config()` — the **raw unseeded** base default. Both
  harnesses explicitly refuse to use `default_config()`, because `promote_seed_to_default` would make
  `default_config()` return the seed and you'd be timing seed-vs-seed.
- **torch.compile** = `torch.compile(reference_fn)` in **default mode**, never `max-autotune`.
  Compilation excluded; Dynamo reset between shapes; `torch._dynamo.explain` used to confirm no
  accidental graph breaks (Audit A: "All 16 smoke references compiled as one Dynamo graph with zero
  graph breaks").
- **AOT / vLLM-tuned** = a *config* produced by full autotuning, replayed on **the same Helion kernel
  body**, with zero tuning time counted. For the vLLM kernels this is **vLLM's own shipped Helion
  config table**. It is **NOT vLLM's native CUDA operator.** Audit A's PLAN says it in the plan text:
  "Each vLLM kernel has all four arms. This is a config comparison on the Helion body, not a
  comparison with vLLM's native CUDA operator." Audit B's aggregator repeats it under every vLLM
  table: "NB: this runs the HELION kernel with vLLM's config — it is NOT vLLM's native CUDA kernel."

---

## 1. Audit A verified: the definitive 114-cell four-arm audit

Re-derived from `results/full/summary.json` (`rows` array, 114 entries), by taking
`row["ratios_vs_seed"][arm]` for every row whose `arms.seed.status == "ok"` and geomeaning.
**Every figure below reproduces the stored table and `SUMMARY.md` to 4 decimal places.**

| Cohort | Cells recorded | Valid cells | Heuristic / default | Heuristic / torch.compile | Heuristic / SM100 AOT |
|---|---:|---:|---:|---:|---:|
| `general_aot` | 24 | 24 | **1.0294** [0.7655, 1.3661] | **1.1242** [0.8100, 1.8927] | **0.9171** [0.6469, 1.4089] |
| `original` | 36 | **35** | **6.0278** [1.1786, 40.3160] | **1.0130** [0.6445, 1.7663] | n/a (n=0) |
| `vllm` | 54 | 54 | **1.9178** [0.5138, 16.6063] | **1.3880** [0.7624, 4.3139] | **0.9181** [0.4343, 1.0477] |

- 24 + 36 + 54 = 114 recorded; 113 **valid** (one `original` cell is dropped, see §5.6).
- The blog doc's "Valid cells 24 / 35 / 54" is correct. It is worth saying "114 recorded, 113 valid"
  rather than "114 valid".
- Important n subtlety I verified: the dropped cell (`rms_norm_bwd [2048, 11008]`) *does* have a
  stored `torch_compile` ratio of 1.28292 in the raw JSON, but `SUMMARY.md` and the stored cohort
  block correctly exclude it from **both** the default and the torch.compile geomean (n=35 for each).
  Including it would move `original` vs torch.compile from 1.0130 to 1.0197. Use **1.013**.
- Precision note: `rows[*].ratios_vs_seed` is stored rounded to 5 significant digits (e.g. `grpo
  [4,1024,256000]` torch.compile = `0.6445`), while the `kernels`/`cohorts` blocks keep full
  precision (`0.6445042678923177`). Geomeans computed from either agree to 4 decimals. This is why
  `SUMMARY.md` prints the `grpo` torch.compile min as 0.645 and a naive `%.3f` of the stored row value
  gives 0.644 — same number.

### 1.1 The three cohorts, defined

- **`general_aot` (24 cells)** — 4 general reduction kernels × 6 shapes, run on the
  `pretuned_kernels/` bodies (deliberately *not* the `examples/` bodies), so the seed/default/AOT arms
  compile the identical body the AOT table was tuned on. Kernels + dtypes: `rms_norm` bf16,
  `layer_norm` fp16, `softmax` fp16, `cross_entropy` bf16. Their AOT arm is Helion's own checked-in
  `decision_tree`-backend AOT heuristic (`"Auto-generated heuristic for kernel: rms_norm / Backend:
  decision_tree"`) — i.e. Helion's own full-autotune sweep, **not** vLLM.
- **`original` (36 cells, 35 valid)** — the 6 reduction kernels from the original reduction
  experiment, all bf16, all on the historical `examples/` bodies (`kl_div`, `jsd`,
  `fused_linear_jsd`, `grpo`, `rms_norm_bwd`, `layer_norm_bwd`), 6 shapes each. **No AOT table
  exists for these**, hence three arms only.
- **`vllm` (54 cells)** — 6 reduction-containing vLLM kernels × 9 shapes, four arms. See §4.

### 1.2 Per-kernel table (all 16 kernels), re-derived

Entries are `geomean [min, max] (n)`. All 16 rows reproduce `SUMMARY.md` exactly.

| Kernel | Cohort | Dtype | Which heuristic fired | vs default | vs torch.compile | vs SM100 AOT |
|---|---|---|---|---:|---:|---:|
| `rms_norm` | general_aot | bf16 | standard | 1.075 [0.991, 1.366] (6) | 1.030 [0.992, 1.154] (6) | 1.008 [0.997, 1.048] (6) |
| `layer_norm` | general_aot | fp16 | standard | 0.978 [0.766, 1.115] (6) | 0.961 [0.810, 1.127] (6) | 0.945 [0.647, 1.409] (6) |
| `softmax` | general_aot | fp16 | standard | 1.096 [0.997, 1.256] (6) | 1.259 [0.997, 1.893] (6) | 0.975 [0.800, 1.253] (6) |
| `cross_entropy` | general_aot | bf16 | standard | 0.974 [0.854, 1.083] (6) | 1.282 [1.165, 1.360] (6) | 0.762 [0.682, 0.967] (6) |
| `kl_div` | original | bf16 | user-tiled | 9.509 [2.296, 28.952] (6) | 1.183 [0.826, 1.766] (6) | n/a |
| `jsd` | original | bf16 | user-tiled | 5.237 [2.122, 15.512] (6) | 0.936 [0.856, 1.064] (6) | n/a |
| `fused_linear_jsd` | original | bf16 | standard | 1.571 [1.179, 2.210] (6) | 0.927 [0.805, 1.010] (6) | n/a |
| `grpo` | original | bf16 | user-tiled | 3.654 [1.339, 8.035] (6) | 0.898 [0.644, 1.120] (6) | n/a |
| `rms_norm_bwd` | original | bf16 | standard | 12.490 [8.360, 20.644] (**5**) | 0.989 [0.889, 1.067] (**5**) | n/a |
| `layer_norm_bwd` | original | bf16 | standard | 15.173 [6.778, 40.316] (6) | 1.181 [0.975, 1.400] (6) | n/a |
| `dynamic_per_token_scaled_fp8_quant` | vllm | native | user-tiled | 2.850 [1.750, 6.163] (9) | 1.030 [0.851, 1.270] (9) | 0.940 [0.630, 1.012] (9) |
| `per_token_group_fp8_quant` | vllm | native | standard | 1.088 [0.996, 1.649] (9) | 1.343 [1.242, 1.565] (9) | 0.995 [0.950, 1.004] (9) |
| `rms_norm_dynamic_per_token_quant` | vllm | native | user-tiled | 7.434 [3.504, 16.606] (9) | 1.015 [0.762, 1.290] (9) | 0.940 [0.643, 1.048] (9) |
| `rms_norm_per_block_quant` | vllm | native | user-tiled | 2.204 [1.500, 3.216] (9) | 1.387 [0.973, 1.753] (9) | 0.925 [0.709, 1.000] (9) |
| `silu_and_mul_per_block_quant` | vllm | native | standard | 1.025 [0.558, 1.252] (9) | 1.483 [1.250, 2.291] (9) | 0.843 [0.434, 1.000] (9) |
| `fused_qk_norm_rope` | vllm | native | standard | 0.956 [0.514, 1.247] (9) | 2.475 [1.996, 4.314] (9) | 0.873 [0.441, 1.001] (9) |

("standard" = `triton_reduction_tile_sm100`, the Helion-rolled-rdim track; "user-tiled" =
`triton_reduction_user_tile_sm100`, the track for kernels where the author writes the `hl.tile` loop
over the reduction axis. 69 standard cells + 45 user-tiled cells = 114.)

### 1.3 Useful cell-count framings (Audit A, 113 valid cells)

- vs the unseeded default: heuristic faster on **93/113** cells; ≥2× faster on **52/113**; ≥10×
  faster on **13/113**.
- vs `torch.compile` default mode: heuristic faster on **74/113** cells.
- vs the checked-in AOT config (78 cells that have one): heuristic within 5% on **54/78**, ties or
  beats it (≥0.995) on **51/78**, is strictly faster on **6/78**.
- **PLAN-discouraged rollups** (compute them only if you also show the cohorts — the PLAN says "Do
  not combine the three cohorts into a single headline geomean without also showing the separate
  cohort and per-kernel results"): all-113 vs default **2.396**, vs torch.compile **1.204**;
  all-78-AOT-cells **0.918**.

### 1.4 The charts (Audit A) — what they actually plot

Directory: `/home/dev/local/wt-b200-reduction-audit-run/perf-repro/b200-reduction-audit/results/charts/`

- `all_cohorts_relative_performance.png` (355 KB) — combined 3-panel figure
- `general_aot_relative_performance.png`
- `original_relative_performance.png`
- `vllm_relative_performance.png`
- `examples_general_relative_performance.png` — the 10 non-vLLM kernels in one panel, **seed and
  torch.compile arms only**, and its "Overall" bar is a **kernel-weighted** geomean (geomean of the
  per-kernel geomeans), not the cell-weighted geomean in the tables. Do not quote its Overall next to
  a cohort table number.

Grouped bars are **performance relative to the unseeded default** (`default_us / arm_us`, default =
1.00×), arms = Reduction seed (blue `#0072B2`), torch.compile (orange `#E69F00`), SM100 AOT (green
`#009E73`). I recomputed the chart values from `results/full/raw/*.json`:

| Chart panel | seed | torch.compile | SM100 AOT |
|---|---:|---:|---:|
| General AOT — Overall | **1.029×** | 0.916× | 1.122× |
| Original Kernels — Overall | **6.028×** | 5.950× | n/a |
| vLLM — Overall | **1.918×** | 1.382× | 2.089× |
| Examples/General (kernel-weighted) | **3.009×** | 2.850× | (arm not drawn) |

These are internally consistent with the tables because geomeans are multiplicative:
1.918 / 1.382 = 1.388 (= heuristic/torch.compile) and 1.918 / 2.089 = 0.918 (= heuristic/AOT).
The vLLM panel is the single most blog-friendly picture: **default 1.00 → torch.compile 1.38 →
heuristic 1.92 → vLLM's own tuned table 2.09.**

---

## 2. THE vLLM STORY (Audit A, exact-key B200 tables)

### 2.1 The six kernels, named precisely

All six are **byte-identical ports of vLLM's own Helion kernels** (`vllm/kernels/helion/ops/*`); each
`pretuned_kernels/<k>/<k>.py` docstring says "Ported from vLLM's Helion `<k>` kernel (the checked-in
heuristic is converted from vLLM's per-hardware config JSON)". All six take **BF16 primary inputs**;
the five quant kernels emit FP8-e4m3 values plus FP32 scales; QK-norm+RoPE mutates the BF16 QKV
tensor in place.

| # | Kernel (exact symbol) | Blog-friendly name | Shape convention | Structural sizes swept | Track |
|---|---|---|---|---|---|
| 1 | `dynamic_per_token_scaled_fp8_quant` | dynamic per-token FP8 quantization | `(tokens, hidden)` | hidden ∈ {2048, 4096, 5120} | user-tiled |
| 2 | `per_token_group_fp8_quant` | per-token-group FP8 quantization | `(tokens, hidden, group)` | hidden ∈ {2048, 4096, 5120}, group = 128 | standard |
| 3 | `rms_norm_dynamic_per_token_quant` | RMSNorm + dynamic per-token quant | `(tokens, hidden)` | hidden ∈ {2048, 4096, 5120} | user-tiled |
| 4 | `rms_norm_per_block_quant` | RMSNorm + per-block (group) quant | `(tokens, hidden, group)` | hidden ∈ {2048, 4096, 5120}, group = 128 | user-tiled |
| 5 | `silu_and_mul_per_block_quant` | SiLU-and-multiply + per-block quant | `(tokens, intermediate, group)` | intermediate ∈ {6144, 12288, 25600}, group = 128 | standard |
| 6 | `fused_qk_norm_rope` | fused QK-RMSNorm + RoPE | `(tokens, q_heads, kv_heads)` | q_heads ∈ {16, 32, 64}, kv_heads = 8 | standard |

### 2.2 The nine shapes

Nine shapes per kernel = **{1, 128, 8192} token counts × the 3 structural sizes**. The PLAN's words:
"This samples decode, an intermediate batch, and large prefill while avoiding the AOT selector's
fallback path." Every one of the 54 cells recorded `aot_exact_key_or_sweep_shape: true` and
`has_aot: true` — i.e. **all 54 comparisons are against an exact tuned key in vLLM's B200 table, no
nearest-neighbour fallback**. Audit A's README: "all 78 exact-key SM100 AOT arms passed accuracy".

The nine shapes per kernel, literally:

- kernels 1 & 3: `[1|128|8192, 2048]`, `[…, 4096]`, `[…, 5120]`
- kernels 2 & 4: `[1|128|8192, 2048, 128]`, `[…, 4096, 128]`, `[…, 5120, 128]`
- kernel 5: `[1|128|8192, 6144, 128]`, `[…, 12288, 128]`, `[…, 25600, 128]`
- kernel 6: `[1|128|8192, 16, 8]`, `[…, 32, 8]`, `[…, 64, 8]`

### 2.3 Per-kernel vLLM numbers (all three baselines)

| Kernel | vs unseeded default | vs torch.compile (default mode) | vs **vLLM's shipped B200 config** |
|---|---:|---:|---:|
| dynamic per-token FP8 quant | **2.850** [1.750, 6.163] | 1.030 [0.851, 1.270] | 0.940 [0.630, 1.012] |
| per-token-group FP8 quant | 1.088 [0.996, 1.649] | **1.343** [1.242, 1.565] | **0.995** [0.950, 1.004] |
| RMSNorm + dynamic per-token quant | **7.434** [3.504, 16.606] | 1.015 [0.762, 1.290] | 0.940 [0.643, 1.048] |
| RMSNorm + per-block quant | 2.204 [1.500, 3.216] | **1.387** [0.973, 1.753] | 0.925 [0.709, 1.000] |
| SiLU-and-mul + per-block quant | 1.025 [0.558, 1.252] | **1.483** [1.250, 2.291] | 0.843 [0.434, 1.000] |
| fused QK-norm + RoPE | 0.956 [0.514, 1.247] | **2.475** [1.996, 4.314] | 0.873 [0.441, 1.001] |
| **Cohort (54 cells)** | **1.918** | **1.388** | **0.918** |

Absolute latency range on the seed arm across the 54 cells: **7.968 µs → 561.120 µs**.

### 2.4 The decode-vs-prefill structure of the vLLM result (strong blog angle, my derivation)

Splitting the 54 cells by token count (18 cells each):

| token count | vs default | vs torch.compile | vs vLLM's B200 config |
|---|---:|---:|---:|
| 1 (decode) | 1.749 | 1.264 | **0.988** |
| 128 | 1.775 | 1.366 | **0.988** |
| 8192 (large prefill) | 2.272 | 1.549 | **0.793** |

And restricted to the four "main" quantization kernels (36 cells; excludes SiLU-and-mul and
QK-norm+RoPE):

| token count | vs default | vs torch.compile | vs vLLM's B200 config |
|---|---:|---:|---:|
| 1 | 2.187 | 1.128 | 0.983 |
| 128 | 2.247 | 1.173 | **1.000** |
| 8192 | 3.870 | 1.246 | 0.871 |
| all 36 | **2.669** | **1.181** | **0.950** |

**Reading:** at decode and small-batch token counts — the regime that dominates serving — a formula
that ran in microseconds at compile time is *within ~1–2%* of vLLM's per-shape hand-tuned B200 table
(0.988× / 0.988×, and exactly 1.000× on the 4 quant kernels at 128 tokens). Essentially the whole
0.918× cohort gap is the large-prefill (8192-token) row at 0.793×.

### 2.5 Concrete config-level illustrations from the vLLM cells

These make the "formula, not a table" point concrete (all read from `summary.json` seed/default/AOT
configs):

- `rms_norm_dynamic_per_token_quant` at hidden 2048/4096: the seed is `block_sizes=[2048,2048,2048]`
  (all three user-tiled reduction axes at **full row extent = one persistent pass**), `num_warps=8`;
  the unseeded default is `block_sizes=[16,16,16], num_warps=4`. That single sizing difference is the
  **3.5×–16.6×** speedup on this kernel. For hidden 5120 the seed becomes
  `[8192, 8192, 4096]`/`num_warps=16` — the first two axes hold the full padded row, the third one
  chunks because the byte budget ran out.
- `per_token_group_fp8_quant`: the seed's single `block_sizes` entry is *rows per program*, and it
  scales cleanly with token count: 1 → 1 → 16 (hidden 2048), 1 → 2 → 32 (hidden 4096),
  1 → 4 → 64 (hidden 5120), always `num_warps=2`. The default is a flat `[16]` or `[32]`,
  `num_warps=4`, for every shape.
- `fused_qk_norm_rope`: seed rows/program go 1, 2, 32 / 1, 4, 64 / 2, 4, 128 across the nine shapes,
  always `num_warps=2`; the default is always `[32], num_warps=4`. This is why torch.compile loses
  **2.0×–4.3×** here (the Helion kernel is one fused kernel; Inductor lowers the reference to two)
  while the default is roughly a wash.

### 2.6 Cells with >5% timing spread (Audit A's own caution list)

Eight arm-level entries exceeded 5% spread even after the 15-round escalation, and six of the eight
are vLLM cells at ~8–14 µs. Worst: `dynamic_per_token_scaled_fp8_quant [1,2048]` seed arm 7.968 µs
at **23.3%** spread; `silu_and_mul_per_block_quant [1,12288,128]` default 10.112 µs at 18.7%;
`fused_qk_norm_rope [128,64,8]` seed 10.080 µs at 18.1%; `per_token_group_fp8_quant [1,2048,128]`
torch.compile 10.080 µs at 17.0%; `dynamic_per_token_scaled_fp8_quant [1,2048]` default 14.144 µs at
13.6%; `silu_and_mul_per_block_quant [128,12288,128]` AOT 10.048 µs at 9.9%. The other two are
`layer_norm [4096,1024]` default (20.9%) and `softmax [4096,384]` torch.compile (18.1%). **Do not
quote a single ~8–10 µs vLLM cell as a precise result** — quote the geomeans.

---

## 3. Audit B verified: the broader tuned-grid / generalization audit

Re-derived from the 35 per-corpus raw JSON files by re-implementing the aggregator's rules
(per-corpus metric choice + close-enough inclusion) myself. **All aggregates reproduce exactly.**

### 3.1 Headline

| scope | Heuristic / torch.compile | Heuristic / default | Heuristic / vLLM tuned | n(tc) | n(def) | n(vllm) |
|---|---:|---:|---:|---:|---:|---:|
| **reproduction** | **1.042** | **2.589** | **0.987** | 188 | 188 | 16 |
| **generalization** | **1.102** | **2.899** | **0.955** | 252 | 257 | 106 |
| reproduction bf16 | 1.034 | 2.495 | — | 86 | 86 | 0 |
| reproduction fp32 | 1.015 | 2.499 | — | 86 | 86 | 0 |
| reproduction native | 1.249 | 3.816 | 0.987 | 16 | 16 | 16 |
| generalization bf16 | 1.018 | 2.670 | — | 72 | 74 | 0 |
| generalization fp32 | 0.997 | 2.827 | — | 74 | 77 | 0 |
| generalization native | 1.247 | 3.127 | 0.955 | 106 | 106 | 106 |

Per corpus (also all re-derived exactly):

| corpus | tc | def | vllm | n | metric |
|---|---:|---:|---:|---:|---|
| `curriculum` | 1.010 | 2.144 | — | 132 | eager cold-L2 |
| `transfer` | 1.009 | 2.373 | — | 28 | eager cold-L2 |
| `mreduction` | 1.245 | **15.029** | — | 12 | eager cold-L2 |
| `vllm` | **1.249** | **3.816** | **0.987** | 16 | cudagraph device |
| `curriculum_gen` | 0.987 | 2.328 | — | 108 | eager cold-L2 |
| `transfer_gen` | 1.025 | 1.728 | — | 24 | eager cold-L2 |
| `mreduction_gen` | 1.141 | **12.687** | — | 14 / 19 | eager cold-L2 |
| `vllm_gen` | **1.192** | **3.523** | **0.956** | 96 | cudagraph device |
| `qk_norm_rope_gen` | **1.919** | 0.996 | 0.942 | 10 | cudagraph device |

### 3.2 The two vLLM numbers the blog quotes

| Scope | Cells | vs torch.compile | vs default | vs **vLLM tuned** |
|---|---:|---:|---:|---:|
| "Posted-number shapes" = corpus `vllm` (4 kernels × 4 shapes) | **16** | **1.249** | **3.816** | **0.987** |
| "Full tuned-grid sweep" = corpus `vllm_gen` | **96** | **1.192** | **3.523** | **0.956** |

Both verified exactly. Per-kernel `G_vllm` inside them:

| Kernel | `vllm` (4 shapes) | `vllm_gen` (n shapes) |
|---|---:|---:|
| dynamic per-token FP8 quant | **1.004** | 0.938 (35) |
| per-token-group FP8 quant | 0.977 | 0.977 (12) |
| RMSNorm + dynamic per-token quant | 0.992 | 0.948 (19) |
| RMSNorm + per-block quant | 0.976 | 0.975 (30) |
| fused QK-norm + RoPE (`qk_norm_rope_gen`) | — | 0.942 (10) |

The blog is right that this audit covers **the four main quantization kernels** in `vllm`/`vllm_gen`;
`fused_qk_norm_rope` is a *separate* out-of-sample corpus of its own (15 shapes, 10 valid).

Same decode→prefill drift as Audit A, in `vllm_gen` (my derivation):

| token count | n | vs tc | vs default | vs vLLM tuned |
|---|---:|---:|---:|---:|
| 1 | 22 | 1.142 | 3.531 | **0.978** |
| 64 | 22 | 1.145 | 3.399 | 0.971 |
| 512 | 22 | 1.200 | 3.015 | 0.959 |
| 4096 | 22 | 1.247 | 3.538 | 0.928 |
| 16384 | 8 | 1.296 | 5.857 | 0.927 |

### 3.3 `PERF_TABLES.md` verified row-by-row

I recomputed all 29 rows. The 24 non-vLLM rows are the **fused** (reproduction + generalization)
geomean per (kernel, dtype); the 5 vLLM rows are **`vllm_gen` only** (the full tuned-grid sweep, as
the file states). Spot-checks all match to the printed 2 decimals: `rms_norm` bf16 1.043/1.109 →
"1.04 / 1.11"; `layer_norm` bf16 0.925/0.996 → "0.93 / 1.00"; `softmax` bf16 4.478/1.375 →
"4.48 / 1.38"; `layer_norm_bwd` bf16 17.404/1.208 → "17.40 / 1.21"; `long_sum` bf16 2.775/0.832 →
"2.78 / 0.83"; `dynamic_per_token` 4.068/1.018/0.938 → "4.07 / 1.02 / 0.94". **One omission:**
`welford` exists in the data (bf16 2.889 vs default / 1.021 vs tc; fp32 2.658 / 0.939) and in
`SUMMARY.md`, but has **no row in `PERF_TABLES.md`**. That is why the file has 29 rows and the data
has 31 distinct (kernel, dtype) pairs.

---

## 4. Reproduction vs generalization — what "unseen shapes" actually means

This is the blog's most important methodological point, so here is the exact mechanism, from
`perf-repro/README.md` §5 and `shapes.json`.

### 4.1 The split

The PR's heuristic constants were hand-tuned against a **curriculum** with `train` / `val` /
`robustness` / `test` splits per kernel (e.g. `rms_norm`: 16 train, 8 val, 8 robustness, 8 test
shapes).

- **Reproduction corpora** re-run only the shapes behind the PR's posted numbers: `curriculum`'s
  **`test` split only** (7–8 shapes per kernel per dtype, 9 kernels × 2 dtypes = 132 cells),
  `transfer` (`fused_linear_jsd`, `grpo`, 7 shapes × 2 dtypes = 28), `mreduction`
  (`rms_norm_bwd`, `layer_norm_bwd`, 3 shapes × 2 dtypes = 12), and `vllm` (4 quant kernels × 4
  shapes = 16). Total 188.
- **Generalization corpora (`*_gen`)** are shapes/kernels the heuristic **never saw in any split**.
  README's exact wording: "**verified absent from *every* train/val/test/robustness split** of the
  PR's curriculum, chosen to be realistic (real model dims + interpolations between seen bands) and
  to have a **novel reduction width**."

### 4.2 The three flavours of "unseen"

1. **Unseen shapes, same kernels** (`curriculum_gen` 108 cells, `transfer_gen` 24,
   `mreduction_gen` 24 recorded): deliberately awkward, non-power-of-two, real-model widths —
   e.g. `rms_norm`/`layer_norm` at `[6144,3200]`, `[12288,4544]`, `[3072,5632]`, `[6144,11008]`,
   `[1024,9216]`, `[24576,2816]`; `softmax` at `[4096,1792]`, `[8192,6400]`, `[65536,1600]`;
   `cross_entropy`/`kl_div`/`jsd` at vocab 49408, 152064, 131072, 201088, 262144, 100000;
   `long_sum` at `[24,1572864]`, `[160,3145728]`.
2. **Unseen shapes on a tuned grid you can compare against** (`vllm_gen`, 96 cells): a stratified
   sweep of vLLM's **exact tuned-grid keys** — every hidden/intermediate dim × log-spaced token
   counts {1, 64, 512, 4096, 16384} — so `G_vllm` stays an *exact-key* config comparison while
   covering the whole decode→prefill range vLLM serves. README calls the PR's posted 4 shapes/kernel
   "a favorable subset".
3. **An unseen KERNEL** (`qk_norm_rope_gen`, 15 shapes, 10 valid): `fused_qk_norm_rope` — 3D grid +
   inner RMS reduction over `head_dim` + a RoPE epilogue that reads and writes `qkv` in place.
   README: "The strongest overfit test: does the heuristic even fire correctly on a kernel it was
   never designed around?" It did — 455/455 cells fired a reduction heuristic — and it surfaced a
   latent Helion codegen bug (§5.7).

### 4.3 The result, and why it matters for the "formula" framing

| | vs torch.compile | vs default | vs vLLM tuned |
|---|---:|---:|---:|
| reproduction | 1.042 | 2.589 | 0.987 |
| generalization | **1.102** | **2.899** | 0.955 |

Generalization is **not worse** than reproduction against either free baseline — it is slightly
*better* (1.102 vs 1.042; 2.899 vs 2.589). The overfit hypothesis fails. This is exactly the payoff
of a formula over a lookup table: the heuristic computes block sizes from a byte/occupancy budget
evaluated on *this* shape, so a width nobody ever benchmarked (3200, 4544, 5632, 11008, 201088,
1572864) is not a cache miss — there is no cache. The only place generalization costs anything is
`G_vllm` (0.987 → 0.955), and §3.2's token-count breakdown shows that drift is concentrated at
prefill-sized token counts, not at unfamiliar hidden sizes.

### 4.4 CAVEAT the blog currently omits: Audit B's vLLM arm is the **H100** table

Audit B's harness docstring says `vllm_shipped = vLLM's nvidia_h100.json config` and
`deps/bench_arms.py` loads `<VLLM_CONFIG_DIR>/<kernel>/<PLATFORM>.json`. I proved from the raw data
that the H100 table is what was actually used:

- Cells recorded `vllm_exact_dims: true` at keys that **only exist in `nvidia_h100.json`** —
  `hidden_size` ∈ {512, 6144, 8192, 12288, 28672}, `group_size` = 64, `num_tokens` = 16384. The B200
  table only has `hidden_size` ∈ {2048, 4096, 5120}, `group_size` = 128, `num_tokens` ≤ 8192.
- Byte-level check on a key present in **both** tables
  (`dynamic_per_token_scaled_fp8_quant`, hidden 4096 / tokens 8192): the recorded
  `vllm_shipped_config` is `block_sizes=[4096,4096], num_warps=8, num_stages=3, indexing=[tensor_descriptor,…]`
  — that is the **H100** entry verbatim. The B200 entry for the same key is
  `block_sizes=[512,4096], num_warps=1, num_stages=4`.

So: **Audit A's 0.918× is against vLLM's real B200 table; Audit B's 0.987×/0.955× are against
vLLM's H100 table replayed on a B200.** The H100 table is the *weaker* baseline on B200, which is
consistent with the heuristic looking closer to it. If the blog wants a single "as good as vLLM's
tuning?" number on B200 hardware, **use Audit A (0.918×, or 0.988× at decode)**, and describe Audit
B's `G_vllm` as "vs vLLM's shipped H100 config table".

### 4.5 Other Audit B methodology facts worth carrying

- **Per-corpus metric switch**: `aggregate_report.py` derives vLLM-family ratios from
  `coldgraph_us` (cudagraph device time) because "vLLM deploys these under CUDA graphs, so launch is
  amortized in production"; all other corpora use eager cold-L2 `us`. `PERF_TABLES.md` states this;
  `SUMMARY.md`'s opening line ("no CUDA graphs in the headline number") is a stale blanket
  statement — treat `PERF_TABLES.md` / `aggregate_report.py` as authoritative.
- Two harness corrections were applied to the vLLM-family cells *after* the overnight run and are
  reflected in the final numbers: (1) the timing thunk no longer clones read-only args every rep
  (cloning qk's 10.5 MB cos/sin cache added ~31 µs/rep); (2) the metric switch above. Documented in
  `MORNING_SUMMARY.md`.
- A prior version of this audit reported `G_vllm` ≈ 0.90–0.93 and called `per_token_group` a ~4×
  regression. That was a **harness bug** (`_extract_configs` reused one bound kernel across a shape
  sweep without `kfn.reset()`; `per_token_group` is `static_shapes=False` and doesn't
  `hl.specialize(num_tokens)`, so later shapes inherited the tokens=1 seed `block_sizes=[1]`). Fixed
  and re-run; blast radius 8 cells of 150, all `per_token_group`. **Do not quote the pre-fix
  numbers.**
- torch.compile references were deliberately *repaired* so the baseline is fair: `.item()` graph
  breaks removed from the vLLM quant references, and `index_put`/scatter in the QK-norm+RoPE
  reference replaced by a single contiguous slice-write, which took Inductor from **6 kernels to 2**
  and dropped `qk_norm_rope_gen` `G_tc` from 3.44 → **1.919**. The 1.919 is the honest number;
  the stated principle is "we wrote the torch code, so we make it fuse."

### 4.6 INVERSION TRAP: the legacy H100 vLLM numbers in blog §6 use the opposite convention

`/home/dev/local/prompts-lab/vllm-bench/_bench_results/REPORT.md` (2026-06-25, **H100 80GB sm90**,
helion `d8255bae`, 5 kernels / 30 cells, `do_bench` median cold-L2) reports **latency ratios where
LOWER is better**, not speedups:

- post-fix `seed / default` = **0.754** → the seed is ~25% *faster* than the unseeded default
  (pre-fix 0.774, "~23% faster on average, up to 5.2× faster on prefill reductions")
- post-fix `seed / vLLM-shipped` = **1.013** → the seed is ~1.3% *slower* than vLLM's tuned H100
  config (pre-fix 1.046 with a worst case of 2.05, improved to a 1.10 worst case by the fix)

The blog doc labels them "latency ratio", which is correct, but every other ratio in the document is
a speedup. If a draft says "0.754× vs default" next to "6.03× vs default" the reader will conclude
the heuristic *lost*. Either invert them (1.33× vs default, 0.99× vs vLLM tuned) or state the
convention loudly. Two more facts from that report worth knowing: at that revision the reduction
heuristics had **`promote_seed_to_default = False`** (so `effort=none` would have made arms A and B
identical, which is why configs were replayed explicitly) — today they are promoted; and its
`vllm_shipped` arm is `nvidia_h100.json`, which on H100 hardware is the *right* table, unlike Audit
B's use of it on a B200.

---

## 5. Where the reduction heuristic LOSES (complete enumeration)

### 5.1 Audit A — cohort level, below 1.0

- `general_aot` vs **SM100 AOT**: **0.917** (min 0.647).
- `vllm` vs **SM100 AOT**: **0.918** (min 0.434).
- (No cohort is below 1.0 against the unseeded default or against torch.compile.)

### 5.2 Audit A — per-kernel rows below 1.0, by arm

**vs the unseeded default (2 kernels):**
- `fused_qk_norm_rope` **0.956** (min 0.514)
- `layer_norm` **0.978** (min 0.766)
- `cross_entropy` **0.974** (min 0.854)
- (`silu_and_mul_per_block_quant` is 1.025 but has cells at 0.558/0.660.)

**vs torch.compile, default mode (4 kernels):**
- `grpo` **0.898** (min 0.644)
- `fused_linear_jsd` **0.927** (min 0.805)
- `jsd` **0.936** (min 0.856)
- `layer_norm` **0.961** (min 0.810)
- `rms_norm_bwd` **0.989** (min 0.889)

**vs the checked-in SM100 AOT config (9 of the 10 kernels that have one):**
- `cross_entropy` **0.762** (min 0.682) — the worst per-kernel AOT gap
- `silu_and_mul_per_block_quant` **0.843** (min 0.434)
- `fused_qk_norm_rope` **0.873** (min 0.441)
- `rms_norm_per_block_quant` **0.925**
- `dynamic_per_token_scaled_fp8_quant` **0.940**
- `rms_norm_dynamic_per_token_quant` **0.940**
- `layer_norm` **0.945**
- `softmax` **0.975**
- `per_token_group_fp8_quant` **0.995**
- (only `rms_norm` is ≥1.0, at 1.008)

### 5.3 Audit A — the worst individual cells

| Arm | ratio | kernel | shape |
|---|---:|---|---|
| vs default | **0.514** | `fused_qk_norm_rope` | `[8192, 64, 8]` |
| vs default | 0.558 | `silu_and_mul_per_block_quant` | `[8192, 25600, 128]` |
| vs default | 0.660 | `silu_and_mul_per_block_quant` | `[8192, 12288, 128]` |
| vs default | 0.766 | `layer_norm` | `[1024, 36864]` |
| vs torch.compile | **0.644** | `grpo` | `[4, 1024, 256000]` |
| vs torch.compile | 0.762 | `rms_norm_dynamic_per_token_quant` | `[8192, 5120]` |
| vs torch.compile | 0.805 | `fused_linear_jsd` | `[4096, 50257]` |
| vs torch.compile | 0.810 | `layer_norm` | `[8192, 5120]` |
| vs SM100 AOT | **0.434** | `silu_and_mul_per_block_quant` | `[8192, 25600, 128]` |
| vs SM100 AOT | 0.441 | `fused_qk_norm_rope` | `[8192, 64, 8]` |
| vs SM100 AOT | 0.599 | `silu_and_mul_per_block_quant` | `[8192, 12288, 128]` |
| vs SM100 AOT | 0.630 | `dynamic_per_token_scaled_fp8_quant` | `[8192, 5120]` |
| vs SM100 AOT | 0.643 | `rms_norm_dynamic_per_token_quant` | `[8192, 5120]` |
| vs SM100 AOT | 0.647 | `layer_norm` | `[1024, 36864]` |

Counts of sub-1.0 cells among the 113 valid: **17 vs default, 34 vs torch.compile, 33 of 78 vs AOT**.
Note the pattern: nearly every large AOT/default loss is at **tokens = 8192** or at an unusually wide
row (36864, 256000, 25600) — the heuristic's persistent/full-extent instinct is too aggressive when
both the row and the batch are large, exactly the regime a per-shape tuned table exploits by dropping
to `num_warps=1` and a much smaller tile (see the AOT configs: `[8192, 8] w=1 s=3` for
`rms_norm_per_block_quant [8192,5120,128]` vs the seed's `[8192, 64] w=16`).

### 5.4 Audit B — per-(kernel, dtype) rows below 1.0 vs torch.compile

From `PERF_TABLES.md`, verified: `cross_entropy` fp32 **0.67**, `jsd` bf16 **0.81**, `long_sum` bf16
**0.83**, `fused_linear_jsd` bf16 **0.88**, `long_sum` fp32 **0.91**, `grpo` fp32 **0.93**,
`welford` fp32 **0.939** (row missing from PERF_TABLES), `sum` fp32 **0.97**, `layer_norm` bf16
**0.996**. Below 1.0 vs default: `layer_norm` bf16 **0.93**; and `fused_qk_norm_rope` **1.00**
(0.996, i.e. no win over the default on that kernel).

Audit B's own "disasters" list (a realistic shape with `G_tc < 0.75`) has **18 cells**, all in
`curriculum`/`curriculum_gen`, dominated by two families:
- `long_sum`: `[24,1572864]` bf16 **0.415**, `[8,2097152]` fp32 0.479, `[32,1000000]` bf16 0.519,
  `[8,2097152]` bf16 0.538, `[24,1572864]` fp32 0.645, `[40,1200000]` bf16 0.706, `[48,786432]`
  bf16 0.707
- `cross_entropy` **fp32**: `[8192,152064]` **0.516**, `[4096,201088]` 0.524, `[4096,151936]` 0.526,
  `[2048,100000]` 0.546, `[1024,250000]` 0.554, `[1024,128256]` 0.556, `[4096,114688]` 0.560,
  `[1024,262144]` 0.571, `[2048,131072]` 0.571; plus `[4096,151936]` bf16 0.708 and `[8192,152064]`
  bf16 0.713.

Audit B's README also gives the *explanation* for one of them, which is quotable: `fused_linear_jsd`
bf16 loses to torch.compile (~0.76–0.87) "because Inductor fuses the shared softmax stats across its
4 softmax-family ops (fewer HBM re-reads); the seed's config is fine, it's a below-the-config
codegen-fusion gap."

### 5.5 Honest framing the audit itself insists on

Audit B's README, verbatim: "G_def is large (up to 15× on backward kernels) mostly because the
**untuned default is pathological** on those kernels (register-spilling tile), not because the seed
is superhuman — the seed does the *obvious correct thing* the default fails to. Honest framing:
'seed fixes a bad default.'" The blog should use that sentence or something like it next to the
6.03× / 15× / 12.5× numbers.

### 5.6 The excluded RMSNorm-backward accuracy-gate cell (Audit A)

- Cell: **`rms_norm_bwd`, shape `[2048, 11008]`, bf16**, cohort `original`.
- **Both** the heuristic seed and the unseeded default failed the shared accuracy gate with the
  *identical* detail: `0:ok=False,maxabs=0.0625,maxrel=1.067,rtol=0.03,atol=0.03;1:ok=True,maxabs=0.125,maxrel=0.007353,...`
  — i.e. output 0 (`grad_x`) misses; output 1 passes. It is a BF16 accumulator-margin fact shared by
  the whole family, not a seed-specific wrong answer.
- Recorded timings exist (seed 57.344 µs, default 2890.72 µs, torch.compile 73.568 µs) but the cell is
  **excluded from every Audit A aggregate**: `rms_norm_bwd` is `6/6` cells recorded with `n=5` in
  both ratio columns, `original` drops from 36 cells to 35 valid, and the audit total is 113 valid of
  114 recorded.
- Audit A's README: "That finding reproduced in the initial full pass, a targeted repeat, and the
  definitive pass, and is excluded from performance aggregates."
- **Methodological contrast worth one sentence in the blog:** Audit B treats exactly this class of
  cell the *opposite* way — its "close-enough" rule *includes* a cell when the default makes the
  identical mistake (14 such cells: `rms_norm_bwd` bf16/fp32 margin + `per_token_group` fp8
  ~1-ULP tie-rounding on ~3% of elements), marking them with a †. Same underlying phenomenon, two
  defensible policies. Do not describe the two audits' n's as if they used one rule.
- Note the default arm would have been *hugely* penalised had it been counted (2890.72 / 57.344 =
  50.4×), so excluding it is the conservative choice against the heuristic.

### 5.7 The other accuracy story (Audit B only): the QK-norm+RoPE tile-size miscompile

- 5 of the 15 `fused_qk_norm_rope` cells are **excluded from every ratio** because the seed was
  genuinely wrong while the default was correct: `[16,8,512]` maxabs 1.467, `[32,8,512]` 2.294,
  `[32,8,4096]` 1.897, `[32,8,16384]` 1.861, `[64,8,64]` 2.077. Hence "10 shapes where the seed
  produces a correct config" in `PERF_TABLES.md`.
- `notes/QK_NORM_ROPE_FINDING.md` shows by **hand-set** configs (no heuristic involved) at
  (q=32, kv=8, tok=512) that correctness *alternates* with the token-axis block size: 8 ✓ (0.0156),
  16 ✗ (1.90), 32 ✓, 64 ✗ (1.59), 128 ✓. So it is an upstream **Helion codegen bug at specific tile
  sizes** on this kernel (suspected in-place read-after-write/aliasing in the RoPE epilogue), not a
  heuristic bug — the heuristic merely sometimes picks 16. The vendored body is byte-identical to
  upstream (72/72 lines).
- **In Audit A (the later revision), all 9 `fused_qk_norm_rope` cells passed accuracy** on every arm
  (seed maxabs 0–0.0156 against rtol=atol=0.05), including `[8192,32,8]` whose seed block size is 64.
  Audit A's whole `failures` list contains only the two `rms_norm_bwd` entries. So the miscompile did not
  manifest on Audit A's nine shapes. (Audit A's tolerance for this kernel is looser — rtol=atol=0.05
  vs Audit B's 1e-2 — but that cannot explain it: Audit B's failing cells were off by maxabs
  1.467–2.294, orders of magnitude past either gate.) I could not determine whether the codegen bug
  was fixed between the two revisions or merely not hit by Audit A's nine shapes (see GAPS).
- 2 more Audit B cells have no `G_def` at all because the **default** config hit a ptxas compile
  timeout while the seed compiled and ran (`rms_norm_bwd [2048,11008]` bf16, `rms_norm_bwd
  [1024,10240]` bf16) — "a point in the seed's favor, noted not counted." Separately, 9
  `layer_norm_bwd`/`rms_norm_bwd` `tc` arms failed to compile (InductorError /
  BackendCompilerFailed / timeout), which is why `mreduction_gen` has n(tc)=14 but n(def)=19.

---

## 6. What the reduction heuristic actually decides (plain language)

Source read: `helion/_compiler/autotuner_heuristics/triton.py` in
`/home/dev/local/wt-sm100-linattn` (branch `calebmkim/stack/50`), classes `_TritonReductionSeedBase`,
`TritonStandardReductionHeuristicSM90/SM100`, `TritonUserTiledReductionHeuristicSM90/SM100`,
`TritonNarrowReductionHeuristic`. This is the *current* tree, and the constants and heuristic names
match what Audit A recorded, so I treat it as the description of Audit A's arm. Wherever a claim is
inference rather than something I read, I say so.

**One budget, two emission paths.** The heuristic first builds structural facts about the kernel (a
`ReductionKernelFact`: which axes are reductions, which are grid axes, which loops are non-reduction
"apply" passes, and per co-residency group the *actual live tiles*). Then a single allocator —
`size_reduction_tiles` — hands out a size to **every** axis from one hardware budget. The only thing
that differs between the two registered heuristics is where the answer is *written*: if the author
wrote `x.sum(-1)` and Helion rolled the reduction axis itself, the size lands on the
`reduction_loops` knob (`triton_reduction_tile_sm100`, 69 of Audit A's 114 cells); if the author
hand-wrote `hl.tile(n, block_size=R)` over the reduction axis, every reduction axis is an ordinary
`block_sizes` entry (`triton_reduction_user_tile_sm100`, 45 cells). The source is explicit that this
is "EMISSION routing, not a different way to compute." A third class,
`TritonNarrowReductionHeuristic`, is the conservative pre-existing fallback (one row per program, one
persistent pass) and fires only on hardware with no tuned track — i.e. neither sm90 nor sm100, so it
never fires in these audits.

**The budget is bytes of resident working set, and the decision it forces is
persistent-vs-looped.** The ceiling is a per-program byte number: `ROW_PERSIST_MAX_BYTES = 245760`
(~240 KiB, "just over H100 SMEM"), halved to `CARRIED_PERSIST_MAX_BYTES = 122880` when the reduction
carries a ≥2-D accumulator whose last dim is the reduction axis (the `kl_div`/`jsd`/norm-backward
shape), and raised to a *persistence-hold* watermark (`3 × 245760` for register-resident reuse like
`cross_entropy`/`sum`, `294912` for rows that get re-swept from L2 like
`softmax`/`rms_norm`/`layer_norm`/`welford`) when a re-read row may keep its full extent. Pass 1
seats the reductions with the grid axes pinned at their floor: if the row's resident tile fits, the
reduction axis is given its **full padded extent — one persistent pass, no inner loop**; if it
doesn't fit, it chunks to `min(LOOPED_CHUNK = 16384, byte budget, extent)`. You can see this decision
flip in Audit A's own configs: `cross_entropy` emits `reduction_loops=[None]` (persistent) at vocab
32000/128000/128256/152064 and `[16384]` (looped) at vocab 256000; `layer_norm` is persistent up to
N = 16384 and looped `[16384]` at N = 36864; `fused_linear_jsd` is always looped at `[8192]` (it is a
*carried* reduction, so it gets the halved budget); `rms_norm` and `softmax` are persistent at every
tested width. The source is candid that the hold ceiling is "an ADMITTED PROXY… not a faithful
measure of the underlying cache-tier question and can be fooled."

**Pass 2 spends whatever is left on rows per program, in one of two opposite directions.** A grid-M
axis that is *resident* (its row co-occupies the working set) **widens** into the byte remainder,
capped by an occupancy floor (post-tile grid ≥ `num_sm × MIN_WAVES`, `MIN_WAVES = 8`), by
`WIDEN_MAX_ROWS = 8`, and by the extent — that is the `block_sizes=[2]` you see on `rms_norm
[4096,7168]` / `softmax [4096,4096]`. A grid axis that is in **no** live tile has been *reduced away*
(the grad-parameter `.sum(0)` idiom in `rms_norm_bwd`/`layer_norm_bwd`) and does the opposite: its
floor is raised to about `grid_rows / num_sm` so the cross-grid finalize collapses to ~one SM wave.
That rule is directly visible and arithmetically checkable in Audit A: for both `rms_norm_bwd` and
`layer_norm_bwd`, M = 2048 → `block_sizes[0] = 16`, M = 4096 → 32, M = 8192 → 64, M = 16384 → 128 —
i.e. `M / num_sm` rounded up to a power of two, which for a B200 (148 SMs) gives exactly the recorded
seeds (the recorded values pin `num_sm` only to `[128, 256)`; the raw JSON does not store the SM
count, so 148 is the hardware fact, not something I read out of the data). Non-reduction "apply" loops
(welford's normalize, `rms_norm_per_block`'s groups-per-row) are sized last against their own
headroom, capped on B200 at `NON_REDUCTION_LOOP_MAX_ELEMS = 4096`.

**`num_warps` is a ramp on the reduction extent, with a B200 override that only ever lowers it.** The
base ramp: extent ≤ 1024 → 4 warps, ≤ 4096 → 8, ≤ 16384 → 16, else 32. On top of that, the sm100
carrier recomputes a warp count from *load traffic* (`extent × load itemsize × number of loads`): if
traffic ≤ `NW8_MAX_ROW_TRAFFIC = 64 KiB` the row is "light", and gets **2 warps** when the extent is
≤ `NARROW_ROW_MAX_ELEMS = 1024` or **4 warps** above that; if traffic is heavy but the extent is
≤ `HEAVY_ROW_MAX_ELEMS = 16384` it gets **8**; genuinely huge rows keep the base ramp's 32. The
override is skipped for M-collapse kernels, reduce-then-apply kernels, kernels with more than one
sized reduction, and looped reductions, and the comment says it is "purely additive: only lowers,
never raises." There is also a floor in the other direction: a kernel that reduces its grid-M axis
away gets at least `M_COLLAPSE_MIN_NUM_WARPS` warps (8 on H100, retuned **down to 4** on B200
because "8 overshoots on B200"). I reconstructed this ramp against Audit A's recorded seeds and it
matches every case I checked: `rms_norm [2048,48]` → 2 warps, `[2048,4096]` → 4, `[16384,8192]` → 4;
`layer_norm [4096,1024]` → 2, `[4096,3072]` → 4, `[4096,12288]` → 8, `[1024,36864]` → 32;
`cross_entropy` → 32 everywhere; the four backward-kernel cells → 8 and the two widest → 16. Note
how strongly the B200 retune moved things: the *default* config uses `num_warps=4` on **all 114**
cells, whereas the heuristic's warp counts are spread right across the range — **2 warps on 34
cells, 4 on 11, 8 on 28, 16 on 13, 32 on 28**.

**Two smaller knobs.** `num_stages` is pinned to **1** on every seed in all 114 cells (as it is on
every `default` config too; contrast the 78 AOT configs, which span `num_stages` 1–8 and `num_warps`
1–32) and `pid_type` is pinned to `'flat'` on all 114 because "these reductions are
grid-saturated at the M-grid" — then run through a shared guard that repairs an illegal `pid_type`
rather than shipping it. `load_eviction_policies` gets a structural rule: a single streamed input
becomes `'first'` everywhere (free the L2 lines), while a row that is genuinely re-read across a
grid-collapse loop pins its first load `'last'` (keep it L2-resident) and the rest `'first'`.

**What I cannot establish from these artifacts** (see GAPS): the *ordering* of the alternate seeds
the heuristic offers the autotuner (only `compiler_seed_configs[0]` was benchmarked), what the
heuristic decides for non-reduction knobs like `indexing`/`range_*` (they retain base defaults in
every recorded seed), and how much of the B200 constant set was fitted on which shapes.

---

## 7. Quoting checklist for this dataset

1. Say **"114 recorded, 113 valid"**, and give the three cohorts separately (the PLAN forbids a bare
   combined geomean).
2. Never write "vs vLLM" without "the same Helion kernel body with vLLM's shipped config — not
   vLLM's native CUDA kernel."
3. Distinguish the two vLLM tables: Audit A = vLLM's **B200** table (0.918× cohort, 0.988× at
   decode); Audit B = vLLM's **H100** table (0.987×/0.955×).
4. `torch.compile` = **default mode**, not `max-autotune`, in both audits.
5. `default` = `_base_default_config()` — the **raw unseeded** base default, explicitly not
   `default_config()`.
6. Audit A metric = cold-L2 CUDA-graph device time for everything. Audit B metric = eager cold-L2 for
   functional kernels, cudagraph device time for the vLLM-family corpora.
7. Do not merge Audit A and Audit B aggregates: different Helion revisions, different heuristic
   versions (`*_sm100` vs not), different kernel sets, different accuracy-inclusion rules, different
   vLLM tables.
8. Don't quote an individual ~8–10 µs vLLM cell; six of Audit A's eight >5%-spread arms are exactly
   those cells.
9. Pair the 6.03×/15×/12.5× default-relative numbers with the audit's own framing: "seed fixes a bad
   default."
10. `welford` is missing from `PERF_TABLES.md`; if you want it, the values are bf16 2.889 vs default /
    1.021 vs tc, fp32 2.658 / 0.939 (my derivation from the raw JSON).

---

## Appendix A — file inventory

**Audit A** (`/home/dev/local/wt-b200-reduction-audit-run/perf-repro/b200-reduction-audit/`)
- `PLAN.md` — the pre-registered protocol (arms, cohorts, shapes, timing, accuracy gate, reporting
  rules). Best single source for arm definitions.
- `matrix.py` — the frozen kernel/dtype/shape matrix (all 114 cells).
- `bench.py` — config extraction (`compiler_seed_configs[0]`, `_base_default_config()`), AOT
  selection, cold-L2 cudagraph timing.
- `aggregate.py`, `plot_performance.py`
- `results/full/SUMMARY.md`, `results/full/summary.json` (1.5 MB, 114 rows), `results/full/raw/*.json`
  (one file per cell), `results/full/logs/`, `results/full/helion-cache/`
- `results/smoke/`, `results/README.md`, `results/charts/` (5 PNGs)
- AOT tables read by the harness: `pretuned_kernels/<k>/_helion_aot_<k>_cuda_sm100.py`

**Audit B** (`/home/dev/local/wt-b200-perf-report-repro-plan/perf-repro/`)
- `README.md` — arms, corpus design (§5), benchmarking technique (§6), results (§7), gotchas (§8)
- `shapes.json` — every shape, with the curriculum's train/val/test/robustness splits
- `perf_report_bench.py`, `aggregate_report.py`, `deps/bench_arms.py` (vLLM config lookup)
- `notes/ACCURACY_FIXES.md`, `notes/LAUNCH_OVERHEAD_NOTE.md`, `notes/QK_NORM_ROPE_FINDING.md`
- `results/PERF_TABLES.md`, `results/SUMMARY.md`, `results/MORNING_SUMMARY.md`,
  `results/summary.json`, `results/<corpus>__<kernel>.json` (35 files, 455 rows),
  `results/run_manifest.json`
- **No charts directory / no PNGs.**

**vLLM's own config tables** (read live by Audit B, ported into Audit A's AOT files):
`/home/dev/local/vllm-src/vllm/kernels/helion/configs/<kernel>/{nvidia_b200.json,nvidia_h100.json}`
— entry counts: `dynamic_per_token` 42 (b200) / 115 (h100); `rms_norm_per_block` 42 / 96;
`per_token_group` 42 / 42; `rms_norm_dynamic_per_token` 42 / 59; `fused_qk_norm_rope` 42 / 45.

## Appendix B — exact commands I used to re-derive

Audit A cohort/per-kernel geomeans:
```python
d = json.load(open(".../results/full/summary.json"))
rows = [r for r in d["rows"] if r["arms"]["seed"]["status"] == "ok"]   # drops 1 cell
geomean(r["ratios_vs_seed"][arm] for r in rows if <cohort/kernel>)
```
Audit B headline (must re-implement two rules):
```python
# 1. metric: coldgraph_us for corpus in {vllm, vllm_gen, qk_norm_rope_gen}, else us
# 2. inclusion: seed status == "ok"  OR  (seed acc-fail AND default acc-fail AND
#    seed.acc_detail == default.acc_detail); other arm must have status in {ok, acc-fail}
```
Both reproduce the stored aggregates to 4 decimal places.
