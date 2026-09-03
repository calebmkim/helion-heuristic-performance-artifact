# Mechanism deep-read #1: the resource model of the matmul / multi-matmul heuristic

**Scope.** How the Helion compile-time matmul heuristic (front end 1, single contraction) and
multi-matmul heuristic (front end 2, several contractions) estimate each hardware resource on
Blackwell/B200 (`sm100`), what compiler facts feed each estimate, and what each deliberately
approximates.

**Primary sources (all paths absolute, branch `calebmkim/stack/50`):**

| What | Path |
|---|---|
| The heuristic itself (5827 lines) | `/home/dev/local/wt-sm100-linattn/helion/_compiler/autotuner_heuristics/triton.py` |
| Structural-fact dataclasses | `/home/dev/local/wt-sm100-linattn/helion/autotuner/config_spec.py` (lines 151–467) |
| The compiler analysis that BUILDS those facts | `/home/dev/local/wt-sm100-linattn/helion/_compiler/device_ir_analysis.py` |
| Registry / loader | `.../autotuner_heuristics/registry.py`, `.../autotuner_heuristics/__init__.py` |
| Unit tests carrying the measured ground truth | `/home/dev/local/wt-sm100-linattn/test/test_matmul_heuristics.py` |
| Author's own high-level trace | `/home/dev/local/sm100-linattn/MATMUL_HEURISTIC_HIGH_LEVEL_TRACE.md` |

Everything below marked **[verified]** I re-derived by executing the real code in
`/home/dev/helion-env/bin/python` (CPU only — the classmethods that matter are pure arithmetic and
take `num_sm` as an argument) or by recomputing from raw `results.json`. Items marked
**[documented]** are constants/claims I could read but not independently measure (no GPU visible
from this sandbox).

---

## 0. What the heuristic is allowed to decide, and how it ships

Two classes, both on `sm100`, both `promote_seed_to_default = True`:

- `TritonB200FormulaMatmulHeuristic` (`triton.py:3299`), a subclass of the sm90
  `TritonH100MatmulHeuristic` (`triton.py:536`) that overrides only the hardware gate and the
  budget constants. Registry name `triton_b200_formula_matmul`.
- `TritonB200MultiMatmulHeuristic` (`triton.py:3508`), a subclass of the above. Registry name
  `triton_b200_multi_matmul`. It *declines whenever front end 1 fires* (`triton.py:3578`), so
  exactly one of the two owns any kernel.

The emitted fields are only `block_sizes`, `num_warps`, `num_stages`, `l2_groupings`
(`_emit_seed_draft`, `triton.py:2206`; `_h100_config`, `triton.py:510`). `_extra_config_fields`
(`triton.py:729`) is a hook for arch-specific extras and **is never overridden** — so no
`epilogue_subtile` / `indexing` / `range_*` field is chosen by this heuristic. **[verified by grep:
only the base definition and one call site exist.]**

The loader (`autotuner_heuristics/__init__.py:91` `compiler_seed_configs`) plants the whole ranked
list as autotuner seeds and promotes `ranked[0]` to `config_spec.compiler_default_config` — the
config used when autotuning is off. So one artifact does two jobs: **the no-autotune default and
the rank-0 search seed.** Multi-matmul caps its pool at `COMPILER_SEED_CAP = 20` with
`FOCAL_SEED_CAP = 10` (`triton.py:3552-3553`).

`num_sm` is not a constant: it comes from `helion.runtime.get_num_sm` →
`torch.cuda.get_device_properties(...).multi_processor_count` (`helion/runtime/triton/launcher.py:66`).
On B200 that is 148, and every wave/saturation threshold below is arithmetic in it.

---

## (a) SHARED MEMORY — the operand pipeline ring

### The dot-local model: `max(ring, epilogue) + slack`

`_smem_bytes` (`triton.py:775-807`) — quote:

```python
ring = (bm * bk + bk * bn) * itemsize * num_stages
epilogue = bm * bn * cls.EPILOGUE_ACC_ITEMSIZE
return max(ring, epilogue) + cls.SMEM_SLACK
```

- **Ring = bytes per stage × num_stages.** Bytes per stage is `(bm·bk + bk·bn) · itemsize` — the A
  and B operand tiles at their *input* dtype. `num_stages` multiplies it linearly. The docstring is
  explicit that charging the full `num_stages` is "exact for fp32 operands and over-strict (safe)
  for 16-bit ones, which cap at double-buffering."
- **`max`, not a sum**, because Triton's reported `shared` is a *liveness-packed peak*: "the operand
  ring is dead by the time the epilogue conversion runs, so the two reuse the same bytes."
- **The epilogue term is the one people don't expect.** On tcgen05 the fp32 accumulator lives in
  TMEM and must be *staged through SMEM* to reach registers: `bm·bn·4`. It is independent of `bk`
  and of `num_stages`, so **a tile can fit the ring and still OOM**, and no amount of shallower
  pipelining rescues it. It also *saturates* (one epilogue tile costs the same as six), which is
  what makes it a hard ceiling rather than a guess.
- `EPILOGUE_ACC_ITEMSIZE = 4` is set on the **sm90 base**, not on Blackwell: the docstring records
  that an sm90 `[128,256,64]` bf16 dot with fp32 output also reports `shared = 131072 == bm·bn·4`.
  What is Blackwell-specific is *enforcing* it (`ENFORCE_SMEM_BUDGET = True` only on sm100,
  `triton.py:3350`), because only there can the tile reach `bm·bn = 65536` where the term exceeds
  the cap.

### Device limits used

| Constant | Value | Where |
|---|---|---|
| `SMEM_BUDGET` (sm100) | `232448` bytes = 227 KiB, documented as B200 `shared_memory_per_block_optin` | `triton.py:3312` |
| `SMEM_BUDGET` (sm90 base) | `228 * 1024 = 233472` | `triton.py:561` |
| `SMEM_SLACK` (sm100) | `1024` bytes | `triton.py:3353` |
| `SMEM_SLACK` (sm90) | `0` | `triton.py:607` |

The slack is load-bearing and the comment says why: "an otherwise byte-exact bound is violated by
exactly 16 B" (the 8-byte mbarriers). `test_matmul_heuristics.py:323` pins `SMEM_SLACK >= 16`.

### The whole-kernel model (this is the one multi-matmul uses)

`_smem_region_demands` (`triton.py:1238-1302`) does **not** use `_smem_bytes` at all. It charges
*per pipelined region*, from the compiler's `PipelinedRegion` facts:

```
region_bytes = (sum of stageable loads) * max(1, stages)
             + (sum of non-stageable loads)
             + (sum of stores)
```

Three deliberate structural choices:

1. **Every conservatively stageable load in the loop body is charged, not just the dot's A and B** —
   "the pipeliner gives each load in the body its own multi-stage buffer, so within a stage they are
   all resident."
2. **A region that stores from inside itself adds its store staging to the ring** (they are live
   simultaneously — a state recurrence publishes each chunk from inside the sequential loop),
   whereas a plain K-loop is the liveness-packed `max` case.
3. **Separate loops stay separate list entries** (a `max` across regions, `_smem_from_map`
   `triton.py:1304`), "because resource fixup must be able to relieve tied peaks independently."
   `test_matmul_heuristics.py:1569` (`..._relieves_tied_smem_regions_independently`) pins that:
   with two tied regions the fixup halves *both* knobs, `[128,128] -> [64,64]`.
4. Non-loop graphs become `resident_regions` and are charged **once, with no stage multiplier**
   (`triton.py:1300-1301`) — "charging it per stage over-states shared memory badly enough to shrink
   a tile to the dot minimum for a phantom overflow."

`_kernel_smem_demands` (`triton.py:1690`) appends the whole-kernel epilogue term
(`_epilogue_smem`, `triton.py:1664` = max over *every* dot's `rows·cols·4`) as its own independent
demand entry and adds `SMEM_SLACK` to each entry.

There is a second, subtler split: **useful depth vs hard allocation.** With
`hard_allocation=False` the region's stage multiplier is capped by the region's *candidate-real loop
trip count* (`triton.py:1291-1298`); with `hard_allocation=True` (used only by
`_candidate_hard_resources`) the emitted global depth is charged, "because Triton can reserve that
allocation even for a shorter loop." (`test_matmul_heuristics.py:1039` pins this.)

### "Reduce stages before shrinking tiles"

`_fixup_candidate_resources` (`triton.py:2055-2204`) is the ordering. Inside a `while True`:

1. Recompute `_candidate_hard_resources`; break if everything fits (`triton.py:2130-2138`).
2. **First** try `num_stages - 1` if `num_stages > stage_floor` (`stage_floor = MIN_NUM_STAGES = 1`);
   if that produces *relief* on a currently-overflowing resource, accept and `continue`
   (`triton.py:2140-2160`).
3. **Only then** consider halving a shrinkable block knob (`triton.py:2162-2196`), scored by
   `relief()` = `(how many overflowing resources it brings under budget, total fractional
   reduction)` with a tie-break.
4. A knob is only shrinkable while `block_sizes[slot] // 2 >= max(legal_floor, DOT_MIN=16)`.
5. `num_warps` is re-solved after every accepted tile change and once at the end
   (`solve_final_warps`).

Fixed axes never appear in `shrinkable`: front end 1 passes only `TUNABLE_TILED` M/N block ids
(`triton.py:2981-2988`), front end 2 passes `[bid for bid, _users in mm.knob_users]`
(`triton.py:3830`), and `knob_users` by construction contains only ids in
`valid_block_ids()` (`device_ir_analysis.py:1372`).

The two front ends differ in shrink *order*: single-contraction prefers N then M and uses
`largest_first=False` (`triton.py:2996`); multi-matmul uses `largest_first=True`, i.e. shrink the
largest dot-claimed knob, which may be shared by several contractions (`triton.py:3831`).

`MIN_NUM_STAGES = 1` (`triton.py:661`) is itself an argued decision: the comment records that 53 of
251 hand-tuned B200 cells (21.1%), spanning 13 of 26 bodies, use `num_stages=1`, and that a kernel
whose ring needs 152576 B/stage was pinned to its *floor tile* because the fixup refused to go
below two stages.

---

## (b) TENSOR MEMORY (TMEM) — columns, not bytes

There are **two** TMEM models in the file, and only one of them actually binds.

### Model 1 (bytes, dot-local) — `_tmem_bytes`, `triton.py:748-773`

```python
if cls.TMEM_BUDGET is None: return 0          # sm90: no tensor memory
if bm < cls.TCGEN05_MIN_BM: return 0          # below 64 rows: measured tmem_size == 0
return bm * bn * 4 + bm * bk * itemsize       # fp32 accumulator + A operand
```

A is charged **unconditionally** because Triton's `tritongpu-promote-lhs-to-tmem` copies A into
tensor memory (while *also* keeping it in SMEM) for any A that reaches the MMA as a register value —
a dtype cast, *any* same-dtype elementwise op, or a strided load all qualify. "Trying to predict
that pass is a losing game, so we always pay for it. B is never promoted."
`TMEM_BUDGET = 128 * 512 * 4 = 262144` (`triton.py:3331`). Docstring claims validation over
15210 emitted configs against compiled `tmem_size` metadata.

### Model 2 (columns, whole-kernel) — `_tmem_columns`, `triton.py:809-879`

**This is the correct unit and the one the task asked about.** tcgen05 TMEM is
`TMEM_LANES = 128` lanes × `TMEM_COLUMN_BUDGET = 512` columns of 32 bits, and *a request is
denominated in columns*:

```python
def _tmem_accumulator_columns(cls, bm, bn):
    if cls.TMEM_COLUMN_BUDGET is None or bm < cls.TCGEN05_MIN_BM: return 0
    return max(1, -(-bm // cls.TMEM_LANES)) * bn          # ceil(bm/128) * bn
```

So **a `bm < 128` accumulator costs exactly as many columns as a full-lane one** — a byte model
divides by the lanes it does not use and under-charges every narrow accumulator. Measured
`tmem_size` equals the accumulator's N extent exactly. **[verified]** by execution:
`[64,64]→64`, `[128,128]→128`, `[128,256]→256`, `[64,256]→256`, `[256,256]→512`, `[32,256]→0`.

**Columns from separate live accumulators ADD** (`triton.py:847-862`). This is the term whose
absence is a *correctness* bug, not a missed optimisation: a three-dot chunked kernel at chunk 256
emitted a config that died with `OutOfResources: tensor memory, Required: 768, limit 512` — exactly
`3 × 256`. **[verified]** `_tmem_columns([(128,256,2)]*3) == 768 > 512`.

**The allocation floor.** `_tmem_lhs_scratch_columns` (`triton.py:870-879`) models Triton's
power-of-two LHS promotion allocation:

```python
raw_columns = max(1, ceil(bm*bk*itemsize / (TMEM_LANES*4)))
allocated_columns = 1 << (raw_columns - 1).bit_length()      # round up to pow2
return max(cls.TMEM_ALLOC_COLUMNS, allocated_columns)         # TMEM_ALLOC_COLUMNS = 32
```

`TMEM_ALLOC_COLUMNS = 32` (`triton.py:3378`) is "tcgen05 tensor memory allocates at least 32
columns", and it is reused as the *shrink floor* for a reuse-free N knob in `_apply_knob_roles`
(see (f)/role correction) — below 32 a tile "reserves the same tensor memory while issuing less MMA
work." The scratch-inclusive count is a **hard launchability check only**; the residency policy uses
accumulator columns alone, because the all-dot bound is not a reliable peak-residency estimate
(`triton.py:832-838`). `test_matmul_heuristics.py:574` records the measured failure it fixes: five
backward-attention accumulators come to exactly 512 columns and the compiler needed ≥64 more —
`Required: 576, limit: 512`; with LHS scratch the model says 736.

**Strict vs optimistic policy** (`_candidate_tmem_columns`, `triton.py:1416-1488`):
`"strict"` sums over **every** dot in `_all_dot_acc_tiles` (`triton.py:1379`), justified by a
measured 5-dot kernel whose *peak-live* outputs came to 512 columns yet raised
`Required: 704, limit 512` — the compiler does not always reuse TMEM across a kernel's dots.
`"optimistic"` instead takes the max over `live_tile_steps` / `live_dot_outputs`, and is used only
for extra *search seeds*, never for the promoted default.

### What makes a dot tcgen05-eligible

Two independent conditions, both in the code:

1. **Accumulator rows ≥ `TCGEN05_MIN_BM = 64`** (`triton.py:611`). Below that "the dot lowers to a
   non-tcgen05 path that uses NO tensor memory at all (measured `tmem_size == 0`)", so it is charged
   zero TMEM and its accumulator lands on the **register file** instead
   (`_register_live_bytes`, `triton.py:1618-1621`).
2. **`num_warps >= TCGEN05_WARPGROUP_WARPS = 4`** — a full warpgroup. "Confirmed in PTX:
   `num_warps` 1 or 2 emits zero `tcgen05.mma` and `tmem_size = 0`" (`triton.py:1592-1594`). Both
   `_candidate_hard_resources` (`triton.py:2028`) and `_estimated_resident_ctas`
   (`triton.py:1901`) charge TMEM only at ≥4 warps.

`_candidate_dot_work` (`triton.py:1555`) uses condition 1 alone to split work into
`tcgen05_eligible` vs not, which is what the warp regime selector consumes.

### ⚠️ Finding: the dot-local TMEM *byte* budget is inert on B200

The trace doc describes step (2.7) as "growth into the TMEM budget" and "grow the tile while
tensor-memory capacity and launch-wave occupancy allow it". **[verified]** that with the shipped
constants this step never changes the emitted tile:

- I A/B'd `_matmul_tile` against a subclass with `TMEM_BUDGET = None` (which disables both the
  (2.7) growth loop and the (5') TMEM shrink) over **12,000 cells**
  (M,N ∈ 10 powers of two, K ∈ {16,64,256,1024,4096}, itemsize ∈ {2,4},
  `pinned_grid` ∈ {1,2,4,16,64,148,200,296,400,591,592,1000}, num_sm=148): **0 differing cells.**
- The reason is algebraic and launch-grid-independent: for `bm >= 64`,
  `bm·bn·4 + bm·256·itemsize <= 262144` forces `bm·bn <= (262144 − 64·256)/4 = 61440 < 65536`, so
  the doubling from 32768 is always rejected — while `ACC_BUDGET = 32768` already caps the tile
  there. For `bm < 64` TMEM is charged **zero**, so the growth loop's only real gate is wave
  occupancy, and its result is then clawed back by the **SMEM epilogue term**: e.g.
  `_matmul_tile(32, 4096, 4096, 2, 148, 400)` returns `(32, 1024, 16, 8, 6, 1)`; with
  `ENFORCE_SMEM_BUDGET=False` it returns `(32, 4096, ...)`. Growth fired, then (4') undid it.
- By contrast the **(4') SMEM enforcement is real**: it changes the tile in **370 of the same
  12,000 cells** (e.g. `(16,2048,16,fp32)`: `bn 2048 → 1024`).

**So the honest headline is: on B200 the tile area is bounded by `ACC_BUDGET = 32768` elements and
by the fp32 *epilogue staging buffer in shared memory* (`bm·bn·4 ≤ 232448 − 1024` → 57856 elements
→ 32768 as a power of two). Tensor memory binds only through the whole-kernel COLUMN model — which
is exactly the multi-contraction case the blog headlines.** That is a *better* story than the one
the trace doc tells, not a worse one: TMEM's real job here is catching several simultaneously live
accumulators, which is precisely what a per-shape lookup table cannot see.

---

## (c) tcgen05 / warpgroup REGIME SELECTION

`num_warps` in Triton *is* the CTA size: `threads = 32 * num_warps`. A tcgen05 MMA needs a full
warpgroup (4 warps / 128 threads), so choosing `num_warps` is simultaneously choosing the MMA
regime. There are two selectors.

### Legacy (sm90, and the fallback when there is no dot work): raise-only register ladder

`_warps_for_live_set` (`triton.py:881-974`) — the whole body is 7 lines:

```python
warps = max(1, num_warps)
while warps < cls.REG_CLIMB_MAX_WARPS:
    need = cls._register_live_bytes(env, block_sizes, warps)
    if need <= warps * 32 * cls.REG_BYTES_PER_THREAD:
        return warps
    warps *= 2
return warps
```

Climbs 1→2→4→8 while the register-resident live set overshoots the file; **never lowers**. A fixed
point rather than one shot, because raising both enlarges the file *and* can move the accumulators
out of it entirely (by making tcgen05 available), so the estimate is re-asked at each rung.

**Where it stops is `REG_CLIMB_MAX_WARPS`, and on B200 that is 4, not 8** (`triton.py:3347`,
written as `TritonH100MatmulHeuristic.TCGEN05_WARPGROUP_WARPS` by reference so the stop and the
absorption boundary cannot drift apart). The argument in the docstring is the CTA-residency one:
registers are a *soft* budget (overshoot ⇒ ptxas spills, which degrades) whereas TMEM/SMEM
overshoot is a hard `OutOfResources` at launch, so relieving pressure past the point where it is
relieved has a real cost — each warp doubling doubles registers per CTA and halves how many CTAs
stay resident. The measured case: `chunk_fwd_A_diag_anchored_varlen` at its emitted tile — **two
warps spills 38 B and holds 4 CTAs/SM; eight warps spills nothing and holds 1; two warps is 1.75×
faster (325 µs vs 570 µs)**, with thread occupancy flat at 256 threads/SM across both. Scored
against per-cell measured optima over the 53 (of 75) curriculum cells where `num_warps` moves time
≥10%: **climb-to-fit ladder to eight = 0.8959; climb-to-fit stopping at a warpgroup = 0.9591; the
hand-tuned answer key = 0.9363; a flat 1.35 overshoot tolerance = 0.9433.** [documented]

### B200: `_select_num_warps` solves from scratch (`triton.py:976-1077`)

Gated by `REGIME_AWARE_WARPS = True` / `WORK_AWARE_WARPS = True` (`triton.py:3362-3366`). Unlike the
ladder it can move **either direction**, so a final projected or resource-shrunk tile gets a fresh
answer. The work thresholds:

```python
if work.tcgen05_eligible >= EIGHT_WARP_DOT_WORK * penalty(4, 8):      warps = 8
elif work.tcgen05_eligible >  TCGEN05_DOT_WORK  * penalty(2, 4) \
     or (not work.tcgen05_eligible and wide_register_accumulator):    warps = 4
elif work.total          >= SUBSTANTIAL_DOT_WORK * penalty(1, 2):     warps = 2
else:                                                                 warps = 1 if p1 <= 1.2 else 2
```

| Constant | Value | Meaning |
|---|---|---|
| `SUBSTANTIAL_DOT_WORK` | `1 << 20` | enter 2 warps |
| `TCGEN05_DOT_WORK` | `1 << 20` | enter the 4-warp warpgroup |
| `EIGHT_WARP_DOT_WORK` | `1 << 26` | **64× more work** to justify 8 |
| `NON_TCGEN_WIDE_N` | `128` | a `rows<64, cols>=128` accumulator forces a warpgroup even with no tcgen05 work |
| `WARP1_SOFT_PRESSURE` | `1.2` | 1 warp only if register pressure ≤ 1.2× the 1-warp file |
| `FORCED_MMA_SOFT_PRESSURE` | `1.2` | relief threshold when *no* dot can reach tcgen05 |
| `TCGEN_CATASTROPHIC_PRESSURE` | `1.75` | relief threshold when a dot *can* |

`work` is `CandidateDotWork` (`triton.py:60`) = `Σ_dots m·n·k·trips` under the candidate block
sizes, split into `total`, `tcgen05_eligible`, `..._independent`, `..._carried`, `uncertain`
(`_candidate_dot_work`, `triton.py:1490-1567`). This is *dynamic* work: per-invocation dims come
from the candidate tile, enclosing loop axes contribute the matching candidate trip count, and an
axis represented by both terms is counted once as `block · ceil(extent/block)`.

### Why 2→4 is a bigger decision than 4→8

Three separate mechanisms, all in the code:

1. **Threshold ratio.** `EIGHT_WARP_DOT_WORK / TCGEN05_DOT_WORK = 2^26 / 2^20 = 64`. Entering the
   warpgroup needs 1 Mi MAC-equivalents; going to 8 warps needs 64 Mi.
2. **The pressure guardrail is asymmetric by regime** (`triton.py:1045-1071`). If *no* dot is
   tcgen05-eligible, "adding warps cannot accidentally cross into tcgen05, so relieve even a modest
   overshoot" — the climb triggers at `pressure > 1.2`. If a dot *is* eligible, crossing into the
   warpgroup from below requires `p2 > 1.75` (**genuinely catastrophic**, not merely soft), because
   that crossing changes the MMA regime and costs residency. So the same amount of spill buys a
   warp doubling in the register-MMA regime and does not in the tcgen05 regime.
3. **The residency penalty multiplies every threshold**
   (`_warp_transition_occupancy_penalty`, `triton.py:1079-1092`):
   ```python
   ratio = max(1.0, max(1, lower_warp_ctas) / max(1, higher_warp_ctas))
   return min(cls.WARP_TRANSITION_OCCUPANCY_PENALTY_MAX, ratio)     # cap 4.0
   ```
   If moving up a rung halves resident CTAs, that rung's work threshold **doubles**. Both inputs
   already include launch demand, so *queued* waves after residency saturates cannot inflate it.
   **[verified]** `penalty(1,1)=1.0`, `penalty(4,4)=1.0`, `penalty(4,2)=2.0`, `penalty(32,1)=4.0`
   (the cap).
   `test_matmul_heuristics.py:799` pins the end-to-end effect: with equal residency,
   `SUBSTANTIAL_DOT_WORK`→2 warps, `TCGEN05_DOT_WORK+1`→4, `EIGHT_WARP_DOT_WORK`→8; with a
   one-step residency loss at each transition the same work selects 1, 2 and 4 respectively.

Two more rules worth quoting:

- **4→8 under pressure requires residency to be non-decreasing**: `pressure(4) > 1.75 and
  resident_ctas(8) >= resident_ctas(4)` (`triton.py:1066-1071`). The 8th warp is only bought when
  it is free in occupancy terms.
- **Uncertainty can raise but never lower.** `if work.uncertain: warps = max(warps, initial)`
  (`triton.py:1075-1076`). "An unresolved dynamic loop contributes only a proven one-invocation
  lower bound to `work`. That is sufficient evidence to raise a provisional choice, never to lower
  one derived from the tile shape."

### The CTA-residency estimate itself

`_estimated_resident_ctas` (`triton.py:1855-1912`) = `max(1, min(...))` over six limits:

| Limit | Expression | Constant |
|---|---|---|
| launch demand | `ceil(grid / num_sm)` | — |
| hardware CTA cap | `MAX_CTAS_PER_SM` | 32 |
| thread cap | `MAX_THREADS_PER_SM // (32*warps)` | 2048 |
| shared memory | `SMEM_BUDGET // smem_bytes` | 232448 |
| registers | `REGISTER_FILE_BYTES_PER_SM // min(logical_live_bytes, warps*32*REG_BYTES_PER_THREAD)` | 65536·4 = 262144 |
| tensor memory (only at ≥4 warps) | `TMEM_COLUMNS_PER_SM // columns` | 512 |

The register clamp is the interesting one: "excess logical liveness *spills* rather than allocating
an impossible register file", so the estimate is clamped to the physical per-CTA allocation before
dividing.

---

## (d) REGISTER PRESSURE / SPILLING

`_register_live_bytes` (`triton.py:1569-1624`) — peak register-resident bytes, **selected by
resolved bytes at the candidate config**:

```python
for step in mm.live_tile_steps:           # EVERY recorded liveness step
    total = 0
    for tile in step:
        if tile.kind == "load":  continue                 # charged to the SMEM ring instead
        nbytes = cls._resolve_tile_bytes(tile, block_of)
        if nbytes > ceiling:     continue                 # 8*32*1020 = 261120 B; bigger => it's in HBM
        if tile.kind == "dot_out" and num_warps >= 4 and rows >= 64: continue   # TMEM absorbs it
        total += nbytes
    peak = max(peak, total)
```

- **Capacity** is `warps * 32 * REG_BYTES_PER_THREAD` with `REG_BYTES_PER_THREAD = 255 * 4 = 1020`
  (`triton.py:624`) — the **architectural** 255-register ceiling, deliberately, because "this budget
  answers 'will this spill catastrophically', not 'is occupancy ideal'". **[verified]** the per-warp
  capacities are `{1: 32640, 2: 65280, 4: 130560, 8: 261120}`; one warp = 31.875 KiB, which the
  docstring says is "the bound the measured failures sit on" (two live 64×64 fp32 accumulators =
  32 KiB, over budget before a single operand tile).
- **Warp-count dependent**, so it must be re-asked at each rung of the ladder.
- The peak-*selector* is itself a fixed bug: the previous version picked one step by `LiveTile`
  **rank profile** (right for a reduction's block-size-free working set, wrong here where block
  sizes are known). It "UNDER-counted a kernel that spills 540 registers at one warp (43520 B
  against a 32640 B one-warp file) while OVER-counting one that spills none (49152 B)" — the
  ordering inverted, so no threshold could separate them, and two patches (a calibration divisor,
  then an unconditional two-warp floor) were layered over a defect that was in the *selector*.
  With byte selection: "zero-spill cells at most 1.506× the one-warp file, spilling cells at least
  2.298×, measured over 12 curriculum cells." `test_matmul_heuristics.py:675` and `:1911` pin it.

### The guardrail

Two uses. (1) In `_select_num_warps`, `pressure(w) = live_bytes(w) / capacity(w)` gates the
1-vs-2 choice at 1.2 and drives the relief climbs at 1.2 / 1.75 described in (c). (2) In
`_one_trip_stage2_allowed` (`triton.py:1807-1853`): a one-trip enclosing loop is allowed a second
top-level stage only if `smem_of(2) <= SMEM_BUDGET`, `pressure <=
ONE_TRIP_STAGE2_MAX_REGISTER_PRESSURE = 1.0`, and stage 2 does not collapse residency from ≥2 CTAs
to 1. "At higher pressure the measured compiler schedule turns the additional in-flight state into
severe spilling." Note it deliberately excludes TMEM from that residency check, because the
all-dot TMEM bound is a hard-safety upper bound and would falsely claim residency is already 1.

### The documented limitation: loads are charged to SMEM, not registers

The exclusion is one line (`triton.py:1613-1614`: `if tile.kind == "load": continue`) and the stated
reason is "charging them here would put the same bytes in two budgets." The author documents the
hole in `MATMUL_HEURISTIC_HIGH_LEVEL_TRACE.md:418-443`:

> `LiveTile(kind="load")` describes **every** load, not only promoted dot operands. […]
> ```python
> value = y[0, tile_m, tile_n]
> x[tile_m, tile_n] = value
> ```
> `x` remains an HBM destination, but `value` must pass through registers on the way from the load
> to the store. Because it is classified only as a load, the current model assigns it **zero**
> register bytes.

And why the naive fix is rejected: "Simply charging every load to both budgets would also be too
crude: promoted operands need different accounting, and compiler scheduling may stream or reuse
registers rather than materialize the full logical tile at once." The underlying limitation is that
`LiveTile` records that an op *is a load* but not its eventual *storage role*.

Two smaller approximations in the same function: a value bigger than the largest possible CTA
register file is dropped entirely (a varlen packed `[T,C,D]` buffer measured 256 MiB —
`test_matmul_heuristics.py:719-722`), and dot outputs are handed to TMEM at ≥4 warps
**unconditionally of `TMEM_COLUMN_BUDGET`**, which `triton.py:641-646` flags as a pre-existing
inconsistency on an arch with no tensor memory, retained only to hold the frozen sm90 emit still.

---

## (e) OCCUPANCY / CTA-per-SM and LAUNCH WAVES

### The wave-utilization formula

Two implementations of the same quantity, in two places, with **different policies** — worth
getting right in the blog.

**Dot-local, in `_matmul_tile`** (`triton.py:2634-2638`):

```python
def _wave_eff(_bm, _bn):
    g = launch_grid(_bm, _bn)
    waves = (g + num_sm - 1) // num_sm
    return g / (waves * num_sm)
```

used by the shrink loop (`triton.py:2675-2685`), which halves the *larger* tile axis while
`_wave_eff < WAVE_FULL = 0.8` and the halving `_better`s utilization — on sm100 `_better` is
**strict** (`WAVE_FILL_STRICT = True`, `triton.py:3327`), on sm90 it is `>=` to preserve a
byte-identical freeze. The shrink floors at `WAVE_FILL_FLOOR = 64` unless M ≤ `DOT_MIN`
(tiny-M decode keeps a floor of 16).

**Grid-knob role correction, in `_apply_knob_roles`** (`triton.py:3474-3505`), the one the trace doc
describes. It compares `g / ceil(g/num_sm)` by **integer cross-multiplication** so there is no
float:

```python
want_programs = max(1, int(num_sm * cls.WAVE_FULL))          # 148 * 0.8 -> 118
while guard < 32:
    current_grid = cls._launch_grid(env, block_sizes)
    if current_grid >= want_programs: break
    current_waves = max(1, ceil(current_grid / num_sm))
    for bid in grid_knobs:
        trial = block_sizes with bid halved
        trial_grid  = cls._launch_grid(env, trial)
        trial_waves = max(1, ceil(trial_grid / num_sm))
        if trial_grid * current_waves > current_grid * trial_waves:   # STRICT improvement
            candidates.append(bid)
    if not candidates: break
    victim = the largest candidate knob;  block_sizes[victim] //= 2
```

### The worked examples — **[verified, they are exactly what the code does]**

With `num_sm = 148`, `want_programs = 118`:

| transition | `current_waves` | `trial_waves` | `trial_grid*cw` vs `current_grid*tw` | verdict |
|---|---:|---:|---|---|
| 64 → 128 | 1 | 1 | 128 > 64 | **accept** (both in one wave, utilization 0.432 → 0.865) |
| 86 → 172 | 1 | 2 | 172 = 172 | **reject** (utilization 0.581 both sides) |
| 96 → 192 | 1 | 2 | 192 = 192 | **reject** (0.649 both sides) |
| 32 → 64 | 1 | 1 | 64 > 32 | accept |
| 74 → 148 | 1 | 1 | 148 > 74 | accept |
| 118 → 236 | 1 | 2 | 236 = 236 | reject (and the loop has already broken at `current_grid >= 118`) |

The 86 → 172 case is not synthetic: `test_matmul_heuristics.py:1505-1518` uses extent **11008**
(the Llama-family MLP width), where `11008/128 = 86` programs in one wave and `11008/64 = 172` in
two. The heuristic leaves the knob at 128. The same test pins the other three behaviours:
extent 8192 shrinks `8192 → 64` (giving 128 programs ≥ 118, then stops rather than running to the
floor); extent 1024 shrinks all the way to the *legal block minimum* 8 because even a tiny tile
cannot fill a wave ("tensor-memory column granularity is not a launch-grid constraint" — the grid
role does **not** use the 32-column floor); extent 65536 at block 128 (512 programs) is left alone.

**Caveat the blog should not skip:** the two policies disagree above one wave. At `grid = 200`
(waves = 2, utilization 0.676) the *dot-local* `_matmul_tile` loop keeps shrinking — halving gives
`grid = 400`, utilization 0.901, a strict gain — while `_apply_knob_roles` **stops**, because
`200 >= want_programs = 118`. **[verified]** So "shrink only below one wave" describes the grid-knob
role correction; the tile formula's own wave-fill loop has no such bound.

### Launch grid derivation

`_launch_grid_for_graphs` (`triton.py:1202-1236`) + `_candidate_launch_grid_size`
(`triton.py:91-113`): for each root grid group, `Π_axes ceil(extent / candidate_block)`, summed over
groups (independent top-level grids add), floored at 1, **and axes whose extent is unknown
contribute no proven parallelism** (`triton.py:107-109`). It uses only the root groups belonging to
the *dot-bearing* graphs when it can (`triton.py:1216-1223`), falling back to all roots.

The per-axis extent comes from `_full_block_map` (`triton.py:1094-1165`), which is the single most
under-appreciated piece of the whole file. It asks the compiler's own `BlockSizeSource` because the
three kinds of axis resolve completely differently:

- a tunable loop axis → the config's entry for it;
- an `hl.grid` axis → a fixed source of **1** (one row per program) *even though the axis's extent
  is the whole grid*;
- a specialized axis → **its full extent**.

The docstring records the measured consequence of guessing: reading a pinned outer grid axis at its
full extent (8192 rather than 1) made every accumulator look astronomical, so the resource fix-up
shrank everything to the dot minimum — "a 6-dot kernel emitted `[16,16,16,16,16]` against a
hand-tuned `[128,128,64,256,128]` while sitting at 22 KiB of shared memory and 64 tensor-memory
columns, i.e. nowhere near any limit."

### Graded stage depth: occupancy as the gradient

`_graded_stage_depth` (`triton.py:1725-1805`), enabled by `GRADED_STAGES = True`
(`triton.py:3360`), is the occupancy model for `num_stages` when K is a fixed full extent and the
pipelinable loop is the enclosing sequential (chunk) loop:

```
share  = SMEM_BUDGET // ctas_per_sm
depth  = deepest ring that fits `share`, capped by min(HW_MAX_STAGES, max(one_trip_floor, loop_trips))
```

- `ctas_per_sm` is `ceil(grid/num_sm)` **clamped at `GRADED_MAX_CTAS_PER_SM = 4`**
  (`triton.py:690`), because "a grid far above the machine size does NOT demand a matching number of
  simultaneously-resident CTAs — the excess queues" (measured: outer grid 8192 on 148 SMs gives 56
  waves and a 4 KiB per-CTA share, which nothing fits).
- `HW_MAX_STAGES = 6` (`triton.py:653`), raised above `MAX_STAGES = 6`… identical here; the comment
  records that 4/6/12 landed within 0.002 geomean of each other over 34 scored bodies.
- `OCCUPANCY_RELAXED_MAX_STAGES = 3` (`triton.py:694`) lets an occupancy proof recover triple
  buffering: "occupancy facts can prove that a deeper pipeline costs no additional CTA, but not that
  pipeline/barrier overhead is free."
- The measured gradient it reproduces: "outer grid 32 → 8-11 stages, 64 → 6-8, 96 → 3-4, 256 → 2-4,
  and ≥1024 → 2" — and crucially **not** a function of loop length ("a 16-iteration walk wants 8
  while a 128-iteration walk wants 3-4"). `test_matmul_heuristics.py:737` pins monotonicity in the
  grid and saturation of the clamp (`grid=1024`, `16384` and `10^6` all give the same depth).
- `GRADED_SHARE_FALLBACK = False` (`triton.py:682`) is a **refuted** switch kept documented: the
  "once one stage can't meet the share, co-residency is already lost" argument is sound-sounding and
  loses twice on two populations (0.8632 with 17 bodies ≥0.90 vs 0.8678 with 19 for the floor;
  median null-arm spread 0.03%). The comment calls the residual "a MISSING PROPERTY, not a missing
  constant" — the largest single characterized residual in the B200 linear-attention curriculum.

---

## (f) L2 grouping — the condition

One line, `triton.py:2769-2779`:

```python
grid_m = ceil(m / bm);  grid_n = ceil(n / bn)
l2_grouping = 2 if grid_m > 1 and grid_m >= cls.L2_TALL_RATIO * grid_n else 1
```

`L2_TALL_RATIO = 3` (`triton.py:725`): "reorder PIDs so a group of M-tiles shares an L2-resident B
operand. Helps a tall tile-grid (many M-tiles reuse one B) but hurts a wide/square grid, so gate on
the measured crossover." Only ever 1 or 2 — never deeper.

It is additionally gated at the projection layer (`allow_l2_grouping`, `triton.py:2829-2835`):
**both** M and N must have a block id that is (i) a real config slot, (ii) in *this dot's root grid
group*, and (iii) `TUNABLE_TILED`. When emitted, `_l2_groupings_for_site` (`triton.py:2227-2259`)
writes the value into only that root's slot and preserves every other root's setting.

The multi-matmul front end **never emits `l2_groupings` for its primary** — `_multi_ranked` builds
`Config(block_sizes=…, num_warps=…, num_stages=…)` only (`triton.py:4189-4194`), so the base
setting survives. Only the *focal* alternate seeds carry an L2 choice.
**[verified from raw data]** across the 149 on-corpus cells, **0** post-change configs have any
`l2_groupings` entry ≠ 1 — consistent with 129/149 being multi-matmul and the tall-grid condition
not firing for the 20 formula cells.

---

## The compiler analyses the heuristic depends on

This is the blog's thesis. Below, each fact, where it is *produced*, where it is *consumed*, and
why an out-of-compiler autotuner or a shape→config lookup table cannot supply it. All producers are
in `/home/dev/local/wt-sm100-linattn/helion/_compiler/device_ir_analysis.py`; the fact types are in
`/home/dev/local/wt-sm100-linattn/helion/autotuner/config_spec.py:151-467`.

| # | Fact | Produced at | Consumed by | Why it is not available outside a compiler |
|---|---|---|---|---|
| 1 | **Per-step tile liveness** (`live_tile_steps`, `live_tiles`, `live_dot_outputs`, `live_promoted_lhs`) | `GraphAnalysis.build`, last-use sweep `device_ir_analysis.py:374-503` | `_register_live_bytes` (`triton.py:1569`), optimistic TMEM (`triton.py:1438-1487`) | Requires a last-use walk over the traced FX graph *and* each value's producer kind (`dot_out`/`load`/`carry`/`other`) — the same bytes must be charged to registers or to the SMEM ring but never both. A tuner outside the compiler sees only a launch signature; a lookup table keyed on (M,N,K,dtype) has no liveness axis at all. |
| 2 | **Per-dot M/N/K identity and extents**, including *propagated* block identity | `kernel_matmul_fact` identity pass `device_ir_analysis.py:1216-1331` | every sizing decision; `_all_dot_acc_tiles` (`triton.py:1379`), `_candidate_dot_work` (`triton.py:1490`) | The pass establishes that a dot's output M/N dimensions have a block identity, then **carries that identity through shape-preserving pointwise ops** (`torch.Tag.pointwise`, `:1310-1327`) until a *later* dot consumes the tensor. That is a dataflow fact about the fused body. Outside the compiler you cannot even name the second dot's K axis, let alone know it is the first dot's N. |
| 3 | **Tunable vs fixed-full-extent vs unknown axis classification** (`DotAxisKind`) | `classify_axis` `:1335-1347` + `_immovable_extent` `:82-105` | front-end eligibility (`_generalized_static_matmul_fact`, `triton.py:277`), `project()` (`triton.py:2876-2886`), `shrinkable` lists | The distinction is between *the shape* and *what the config can move*: an `hl.specialize`d axis has either no block id or one absent from `valid_block_ids()`, and `_immovable_extent` must consult `FixedBlockSizeSource.from_config(_base_default_config())` to resolve it. It is a property of the **knob surface**, which does not exist outside the compiler. This is also what lets the heuristic serve a kernel with **zero** tunable dot axes (14 of the 149 on-corpus cells have an empty `block_sizes` list — **[verified]**) where only `num_warps`/`num_stages` remain. |
| 4 | **Shared-knob aliasing** (`knob_users`: for each tunable block id, the `(dot_index, axis)` pairs competing for it) | `:1365-1373` | `_draft` knob resolution (`triton.py:3737-3764`), `_rank_key` (`triton.py:3599`), `_fixup` shrink set (`triton.py:3830`) | This is the *reason front end 2 exists*: "an intra-chunk kernel builds `A = q @ k.T` and then consumes `A @ v`, so one knob is dot 1's N and dot 2's K" (`triton.py:3515-3519`). Without the aliasing map "whichever dot the code happens to size first wins by accident." No external tuner can know two logical axes are the same slider; a lookup table would need one entry per (kernel, aliasing pattern). |
| 5 | **Loop trip counts, candidate-resolved** | facts: `loop_axes_for` `:1453`, `max_trips_for` `:1481`, `prefix_outer_block_id` `:1378` (a *sympy* match on the `(outer_tile.id+1)*outer_block` bound), `is_single_segment_loop` `:1515` + `direct_operand_trip_count` `:1578`; resolution: `_resolved_loop_trips` (`triton.py:1314`) | useful pipeline depth (`triton.py:1291-1298`, `_pipelined_loop_trips` `:1354`), work estimation (`triton.py:1536-1552`), the graded stage ceiling | Trip counts depend on the *candidate block sizes*, so the fact deliberately records the axis and defers the division ("the per-program block is … a candidate property and must be resolved from the emitted `block_sizes`", `config_spec.py:273-276`). Two cases are outright theorem-proving: a triangular/prefix loop resolved at the **lower-median outer tile** (`triton.py:1329-1336`), and a **varlen** loop whose exact trip count is proven from operand `numel` ratios when two independent operands agree (`:1631-1642`). Neither is derivable from a shape signature. |
| 6 | **Loop-carried state detection** (`DotSite.updates_carry`) | `:1692-1695`, `updates_carry = reaches_output(node) and graph_id in loop_block_ids`, with a bounded users-walk `reaches_output` `:522-536` | the leading term of `_rank_key` (`triton.py:3615`) | "A dot writing a loop-carried accumulator keeps that accumulator resident for the whole loop, so its tile sets the kernel's whole-loop footprint." That is a graph-reachability fact about `acc=` flowing to the loop output. `test_matmul_heuristics.py:855` pins that it is a *preference*: with no carry anywhere, work decides; execution count is part of work so a small dot run 4096× outranks a big one. |
| 7 | **Pipelined-region detection + per-load stageability** | `:1722-1739` (loop graphs → `PipelinedRegion`, non-loop graphs → `ResidentRegion`); `memory_tiles_for_loop_axes` `:538-560` with `_index_depends_on_loop` `:243-305` | the entire whole-kernel SMEM model (`_smem_region_demands`, `triton.py:1238`), and `_knob_amortizes` (`triton.py:3380`) | Stageability is "is this load's **index** proven to vary with the enclosing loop axis" — a tri-state (True/False/**None**=uncertain) walk over the index expression's FX subgraph. A load whose index is loop-invariant is resident, not multi-buffered, and charging it per stage over-states SMEM enough to shrink a tile to the dot minimum. This is the *index dataflow*, invisible from outside. |
| 8 | **Launch-grid topology** (`KernelGridFact`: independent roots in source order + `graph_to_root`) | `:663-681` | `_launch_grid_for_graphs` (`triton.py:1202`), `_grid_group_for_site` (`triton.py:264`), `_l2_groupings_for_site` (`triton.py:2227`) | "Concrete program counts are intentionally absent: they depend on the candidate block sizes and must be recomputed while a heuristic edits its draft" (`config_spec.py:317-319`). And the per-axis *per-program extent* needs `BlockSizeSource` (see `_full_block_map`, fact 3) to tell an `hl.grid` axis (1 row/program) from a specialized axis (full extent). A kernel with several independent top-level grids has no single "the grid" a table could key on. |
| 9 | **Attribution post-condition** (`attribution_complete`) | `:1207-1211` + an extent cross-check `:1679-1691`; on failure every site collapses to `DotSite(-1, False, (), None, None)` `:1711-1712` | `_rank_key` carry term (`triton.py:3615`), every consumer of per-dot placement | The heuristic *knows when it does not know*: a computed pairing of trace-order `MatmulFact`s to graph nodes, validated by a post-condition, so consumers degrade to ranking on pure work rather than trusting a wrong pairing. A lookup table has no way to express "this decision is unattributed." |
| 10 | **`_knob_amortizes`: does growing this knob buy arithmetic intensity?** | reads the per-region load tiles (fact 7) | role correction (`triton.py:3462-3472`) | The predicate is: *some pipelined region the knob appears in also stages a load the knob does not span*. If every load spans the knob, bytes moved and MMA work both scale linearly and arithmetic intensity is **constant** — growth is pure cost. Measured on both sides of the same axis position and dtype: reuse-free (`chunk_fwd_wy_delta`'s `hl.tile(D)` body) shrinking 128→32 is **1.18–1.74× faster**; reuse-bearing (`chunk_bwd_dqkw_delta`'s inner DV loop) the same 64→32 shrink is **0.79–0.96×**, i.e. slower, and 64→128 is 1.05× faster. "Same axis position, same dtype, opposite sign — so the discriminator has to be the reuse, not the axis." **No shape-keyed table can distinguish those two kernels: they have the same shapes.** This is the strongest single instance of the thesis. |

### Where the thesis is weaker than it looks — say this out loud

1. **Some of the inputs are just shapes.** `static_m/n/k`, `lhs_dtype.itemsize`, `num_sm` are
   available to any external tool with the same call signature. The compiler-only content is the
   *structural* half: aliasing, liveness, stageability, carry, trip proofs, launch topology.
2. **The compiler analysis sometimes cannot deliver.** `attribution_complete` can be False;
   `_index_depends_on_loop` returns `None` (uncertain, charged conservatively as stageable);
   `_full_block_map` swallows exceptions and falls back to `1` (`triton.py:1157-1164`), which
   silently reads a fixed axis as one row. The facts are *best-effort*, and the heuristic's
   degradation paths are as important as its estimates.
3. **The register model has a documented hole** (§(d)): every load is charged zero register bytes,
   so the "the compiler knows liveness" claim is at its weakest exactly where the values are
   register-resident-but-load-produced.
4. **Being in the compiler does not close the gap to autotuning.** Full autotuning still beats the
   heuristic **1.2046× geomean over the same 149 on-corpus cells** and 1.14×/1.17× end-to-end
   (fwd / fwd+bwd). The argument is about the *no-autotune default* and about *seeding search*, not
   about reaching the ceiling.
5. **The biggest residual is a missing model, not a missing fact.** The stage solver is explicitly a
   *feasibility* solver ("use as many stages as possible subject to useful-loop-depth, occupancy and
   hard-resource limits") with no *profitability* term. **[verified from raw data]** of the 23
   on-corpus cells that regress by >1%, **19 raise `num_stages`** (9 change block sizes, 9 change
   `num_warps`) — i.e. the single dominant failure mode of the shipped heuristic is precisely the
   limitation its own docstrings admit. On `chunk_fwd_wy_delta_varlen_helion` the *only* pre→post
   config difference across all five cells is `num_stages: 1 → 3`, and all five lose 9.8–11.7%.

---

## Verification log (what I re-derived, and how)

Executed against the real code, `num_sm = 148`:

1. **Wave-utilization worked examples** — reproduced the exact integer cross-multiplication
   comparison; 64→128 accept, 86→172 reject, 96→192 reject, 118→236 reject. `want_programs = 118`.
2. **Two different wave policies** — `_matmul_tile`'s shrink loop continues above one wave
   (grid 200 → shrink, since eff(200)=0.676 < 0.8 and eff(400)=0.901 improves), while
   `_apply_knob_roles` breaks at `grid >= 118`.
3. **TMEM columns** — `[64,64]→64`, `[128,128]→128`, `[128,256]→256`, `[64,256]→256`,
   `[256,256]→512`, `[32,256]→0`; three live `[128,256]` accumulators = 768 > 512.
   LHS scratch: `[128,64]`bf16→32, `[128,128]`bf16→64.
4. **Register capacities** — `{1: 32640, 2: 65280, 4: 130560, 8: 261120}` bytes.
5. **`_smem_bytes(128,256,64,bf16,ns)`** = `{1: 132096, 2: 132096, 3: 148480, 4: 197632, 6: 295936}`
   against budget 232448 — i.e. the epilogue term (131072) dominates the ring up to ns=2, and ns=4
   is the deepest that fits, which is exactly what the formula emits.
6. **Formula outputs** (`bm,bn,bk,nw,ns,l2`), B200 and H100 identical:
   `(4096,4096,4096,bf16) → (128,256,64,8,4,1)`; `(8192³) → (128,256,64,8,4,1)`;
   `(1024³) → (128,64,128,4,4,1)`; `(128³) → (64,64,32,4,4,1)`; `(16,4096,4096) → (16,32,256,4,6,1)`.
   Worked trace for 4096³: `bk` starts at `p2le(4096/PIPE=1024)=256`, halves twice because the ring
   at `PIPE=4` stages exceeds 232448, lands at 64; then `ns` counts down from `min(6, k/bk=64)` to
   the first depth that fits → 4; `bm·bn = 32768 ≥ WARPS_HI_ELEMS = 16384` → 8 warps.
7. **Dot-local TMEM is inert** — 0/12000 cells differ with `TMEM_BUDGET=None`; the SMEM enforcement
   differs in 370/12000. Max emitted `bm·bn` over the sweep = 32768 = `ACC_BUDGET`, including for
   `bm ≥ 64`.
8. **On-corpus raw JSON** (`/home/dev/local/wt-sm100-linattn/matmul_heuristic_perf_results/on_corpus_pretuned/results.json`)
   fully reproduces its `RESULTS.md`: 149 cells; post/pre 1.5527×, pretuned/pre 1.8704×,
   pretuned/post 1.2046×; front-end split `triton_b200_multi_matmul` 129 cells (1.4935× post/pre,
   1.2173× pretuned/post) and `triton_b200_formula_matmul` 20 cells (1.9948×, 1.1257×);
   23 regressing cells, 19 of which raise `num_stages`.
9. **Knob distributions, on-corpus 149 cells** — pre: `num_stages` 1 on **all** 149, `num_warps` 4
   on all 149, block values only `{1, 16, 32}`. post: `num_stages` `{1:32, 2:34, 3:49, 4:22, 5:4,
   6:8}`, `num_warps` `{1:6, 2:37, 4:83, 8:23}`, block values `{1, 16, 32, 64, 128, 256}`.
   pretuned: `num_stages` up to **8** (`{…, 7:5, 8:5}`), `num_warps` `{1:21, 2:18, 4:78, 8:32}`.
   Post changed *only scalars* (block sizes identical to pre) on 51 of 149 cells.
   `pid_type` is `flat` on all 149 post configs; max post block value is 256 = `BASE_BN_CAP`.

---

## Constants cheat-sheet (B200, `TritonB200FormulaMatmulHeuristic` + inherited)

| Constant | Value | Role |
|---|---:|---|
| `ACC_BUDGET` | 32768 | fp32 `[bm,bn]` accumulator elements — the first-order tile-area budget |
| `BASE_BM_CAP` / `BASE_BN_CAP` | 128 / 256 | base clamps; wide-N because N is the coalesced store / B-reuse axis |
| `DOT_MIN` | 16 | `tl.dot` min M/N, also the fixup shrink floor |
| `SMEM_BUDGET` | 232448 B | per-CTA shared memory ceiling (B200 opt-in) |
| `SMEM_SLACK` | 1024 B | unmodelled small allocations (8-byte mbarriers) |
| `EPILOGUE_ACC_ITEMSIZE` | 4 | bytes/elem of the accumulator staging buffer |
| `ENFORCE_SMEM_BUDGET` | True (sm100 only) | shrink the tile to satisfy SMEM |
| `TMEM_BUDGET` | 262144 B (= 128 lanes × 512 cols × 4 B) | dot-local byte model (measurably inert — see §(b)) |
| `TMEM_COLUMN_BUDGET` / `TMEM_LANES` / `TMEM_COLUMNS_PER_SM` | 512 / 128 / 512 | the faithful whole-kernel unit |
| `TMEM_ALLOC_COLUMNS` | 32 | tcgen05 minimum allocation; reuse-free-N shrink floor |
| `TCGEN05_MIN_BM` | 64 | min accumulator rows that lower to tcgen05 |
| `TCGEN05_WARPGROUP_WARPS` | 4 | warps in a warpgroup; below it no tcgen05 at all |
| `MAX_NUM_WARPS` / `REG_CLIMB_MAX_WARPS` | 8 / **4** | overall cap vs where the *register* ladder stops |
| `REG_BYTES_PER_THREAD` | 1020 (255×4) | architectural register ceiling per thread |
| `REGISTER_FILE_BYTES_PER_SM` | 262144 | 65536 × 4 B |
| `MAX_THREADS_PER_SM` / `MAX_CTAS_PER_SM` | 2048 / 32 | residency limits |
| `SUBSTANTIAL_DOT_WORK` / `TCGEN05_DOT_WORK` / `EIGHT_WARP_DOT_WORK` | 2²⁰ / 2²⁰ / 2²⁶ | warp-regime work thresholds |
| `WARP1_SOFT_PRESSURE` / `FORCED_MMA_SOFT_PRESSURE` / `TCGEN_CATASTROPHIC_PRESSURE` | 1.2 / 1.2 / 1.75 | register-pressure gates |
| `WARP_TRANSITION_OCCUPANCY_PENALTY_MAX` | 4.0 | cap on the residency multiplier |
| `ONE_TRIP_STAGE2_MAX_REGISTER_PRESSURE` | 1.0 | headroom needed for ns=2 on a one-trip loop |
| `WARPS_HI_ELEMS` | 16384 | tile elems at which the *initial* warp ramp goes 4→8 |
| `MIN_NUM_STAGES` / `MAX_STAGES` / `HW_MAX_STAGES` | 1 / 6 / 6 | stage floor and ceilings |
| `GRADED_MAX_CTAS_PER_SM` / `OCCUPANCY_RELAXED_MAX_STAGES` | 4 / 3 | graded-depth clamp and relaxation |
| `BK_CAP` / `PIPE` | 256 / 4 | max `block_k`; baseline K-loop depth `bk` is sized for |
| `WAVE_FULL` / `WAVE_FILL_FLOOR` / `WAVE_FILL_STRICT` | 0.8 / 64 / True | wave-fill target, shrink floor, strict-gain requirement |
| `SAT_WAVES` / `SAT_TILE_BM` / `SAT_TILE_BN` / `SAT_NUM_WARPS` / `SAT_MAX_STAGES` | 4 / 128 / 128 / 1 / 2 | saturated-batched-dot occupancy caps |
| `SAT_PARTITIONED_K_BM` / `_BN` | 32 / 64 | tighter ceiling for a grid-partitioned K loop |
| `L2_TALL_RATIO` | 3 | `grid_m >= 3 * grid_n` ⇒ `l2_grouping = 2` |
| `COMPILER_SEED_CAP` / `FOCAL_SEED_CAP` (multi) | 20 / 10 | seed-pool caps |

## Blog-usable framings

- **"Two budgets, checked in different units, for the same physical thing."** Shared memory is
  charged in bytes and multiplied by pipeline depth; tensor memory is charged in **columns of 128
  lanes**, where a 64-row accumulator costs the same as a 128-row one, and where separate live
  accumulators **add**. Getting the unit wrong is not a lost percent — it is
  `OutOfResources: tensor memory, Required: 768, limit 512` at launch.
- **"Spend the cheap knob first."** Pipeline depth is surrendered before tile area, and a
  kernel-author-fixed axis is never touched.
- **"Occupancy is a lattice, so shrink only across a favourable wave boundary."** 64→128 programs on
  148 SMs doubles utilization inside one wave; 86→172 buys nothing at all (0.581 either side) and
  costs operand reuse plus CTA overhead. The threshold is arithmetic in `num_sm`, not a magic number.
- **"Registers are soft, tensor memory is hard."** Overshooting registers makes ptxas spill (a
  gradient); overshooting TMEM/SMEM fails at launch (a cliff). That asymmetry is why the register
  ladder is allowed to be a rough over-estimate but stops at a warpgroup, and why the TMEM/SMEM
  models are unconditionally pessimistic — "the accounting can only err toward being too strict."
- **"The discriminator has to be the reuse, not the axis."** Two kernels with the *same* axis
  position, dtype and shapes want opposite tile sizes (1.18–1.74× faster shrinking one, 0.79–0.96×
  slower shrinking the other). Only a dataflow fact separates them. That is the thesis in one
  example.
