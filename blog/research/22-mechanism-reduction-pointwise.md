# Mechanism deep-read #3 — the REDUCTION and POINTWISE heuristics

Source of truth for everything below: the working tree at `/home/dev/local/wt-sm100-linattn`
(branch `calebmkim/stack/50`, HEAD `40151a23f "[autotuner] harden multi-matmul work and grid
modeling"`, **with uncommitted local modifications** to `autotuner_heuristics/triton.py`,
`device_ir_analysis.py`, `autotuner/config_spec.py`, `test/test_matmul_heuristics.py`).
Line numbers are from that working tree.

Everything numeric here was re-derived from either (a) the source, or (b) the rawest result
file available (`summary.json`), never from a prose summary. Where I recomputed a config from
the recorded facts by hand, I say so.

---

## 0. One-paragraph orientation

Helion has **one** compile-time heuristic layer, `helion/_compiler/autotuner_heuristics/`,
registered per backend. Every heuristic is a class with two class methods: `is_eligible(env,
device_ir)` and `get_seed_config(env, device_ir)` (plus an optional ranked
`get_seed_configs`). The heuristics do **not** re-walk the kernel: a **Stage-1 fact layer**
in `device_ir.py` / `device_ir_analysis.py` runs on every live compile and records structural
+ liveness facts onto `ConfigSpec`; each heuristic is then a pure function from facts to a
`Config`. The reduction family reads `ConfigSpec.reduction_kernel_fact`; the pointwise family
reads `ConfigSpec.pointwise_facts[0]`. Both are Triton-only *policies* over a
backend-agnostic *fact layer*.

Files:

| Path | Lines | Role |
|---|---|---|
| `helion/_compiler/autotuner_heuristics/registry.py` | 66 | `AutotunerHeuristic` base class, `PROMOTE_TARGETS`, `should_promote` |
| `helion/_compiler/autotuner_heuristics/common.py` | 123 | shared helpers: `is_canonical_row_reduction`, `matches_hardware`, `clamp_block_size_targets`, `dedupe_configs` |
| `helion/_compiler/autotuner_heuristics/triton.py` | 5827 | all 12 Triton heuristics (matmul, multi-matmul, reduction ×5, pointwise, matmul+reduction-epilogue) |
| `helion/_compiler/autotuner_heuristics/cute.py` | 1179 | 12 CuTe heuristics |
| `helion/_compiler/autotuner_heuristics/pallas.py` | 152 | 2 Pallas heuristics |
| `helion/_compiler/autotuner_heuristics/__init__.py` | 131 | `HEURISTICS_BY_BACKEND` registry + `compiler_seed_configs()` driver |
| `helion/_compiler/autotuner_heuristics/matmul_b200.json` | — | the *legacy* B200 matmul lookup table (demoted; formula subsumes it) |

---

## 1. THE REDUCTION HEURISTIC

### 1.1 Five classes, one allocator

`triton.py` has five reduction classes and one shared base:

```
_TritonReductionSeedBase                       (triton.py:4715)  — THE budget allocator
├── TritonStandardReductionHeuristicSM90       (:5358)  name="triton_reduction_tile"
│   └── TritonStandardReductionHeuristicSM100  (:5634)  name="triton_reduction_tile_sm100"
├── TritonUserTiledReductionHeuristicSM90      (:5469)  name="triton_reduction_user_tile"
│   └── TritonUserTiledReductionHeuristicSM100 (:5644)  name="triton_reduction_user_tile_sm100"
└── (sm100 constants carrier) _TritonReductionSeedSM100 (:5544)

TritonNarrowReductionHeuristic                 (:5661)  name="triton_reduction_narrow"  — NOT a subclass
```

- **standard track** = Helion *rolls* the reduction axis itself, so the primary reduction's
  size lands on the `reduction_loops` knob (`sum`, `rms_norm`, `layer_norm`, row `softmax`,
  `cross_entropy`).
- **user-tiled track** = the author hand-wrote `hl.tile(n, block_size=R)` over the reduction
  axis, so the rdim is an ordinary `block_sizes` entry (`softmax_two_pass`, `kl_div`, `jsd`,
  welford, grad-parameter kernels). Upstream's canonical gate rejected this shape entirely.
- **The two tracks share `size_reduction_tiles()` verbatim.** The class docstring is explicit
  that emission routing is the *only* difference (`triton.py:5007-5010`):

  > "EMISSION is the ONLY standard-vs-user difference: a reduction's computed size is WRITTEN
  > to ``reduction_loops`` (rolled/standard) or a ``block_sizes`` slot (user-tiled). Every
  > reduction gets a size from the SAME budget; the split is codegen routing, not a different
  > way to compute."

- **Exactly one track is eligible per GPU**: sm90 → `*SM90`, sm100 → `*SM100`, everything else
  → `TritonNarrowReductionHeuristic` (the verbatim pre-work upstream seed: `block_sizes=[1]`,
  `reduction_loops=[None]`, `['last']` eviction; **not** promoted to default). This invariant
  is codified by `test_exactly_one_reduction_track_eligible_per_hardware` (added in PR #3035).

### 1.2 The Stage-1 fact it consumes

`ReductionKernelFact` (`autotuner/config_spec.py:558`), built by
`DeviceIR.build_reduction_kernel_fact` (`device_ir.py:1071`) on **every** live compile:

```python
class ReductionKernelFact(NamedTuple):
    reductions: tuple[ReductionDescriptor, ...]
    coresidency_groups: tuple[CoResidencyGroup, ...]
    non_reduction_loop_block_ids: tuple[int, ...] = ()
    grid_axis_block_ids: tuple[int, ...] = ()
```

One `ReductionDescriptor` per reduction *occurrence* (`(graph_id, block_id)` on the original
pre-roll graphs), `config_spec.py:504`:

```python
category: ReductionCategory        # FULL_SLICE | FULL_GRID | GRID_TILE | USER_TILE | DECLINED
block_id: int
graph_id: int                      # the co-residency key
size_hint: int                     # reduction extent, element count
itemsize: int                      # fp32-PROMOTED accumulator itemsize
input_load_itemsize: int = 0       # HBM load element width feeding it
carried_2d_count: int = 0          # # of >=2-D [M_BLOCK, R_BLOCK] loop-carried accumulators
row_reread: bool = False
reread_eviction_index: int | None = None
num_load: int = 0                  # loads in THIS reduction's graph
```

Category taxonomy (`config_spec.py:470`): `FULL_SLICE` (whole axis reduced in one program),
`FULL_GRID` (grid axis tiled at full extent, `cdiv==1`), `GRID_TILE` (grid axis reduced over
but *not* full extent — stays a grid row, never sized as a reduction), `USER_TILE`,
`DECLINED` (no static extent). `SIZED_REDUCTION_CATEGORIES = {FULL_SLICE, FULL_GRID,
USER_TILE}`; `FULL_EXTENT_CATEGORIES = {FULL_SLICE, FULL_GRID}`.

`CoResidencyGroup` (`config_spec.py:540`) is a `graph_id` equivalence class carrying
`live_tiles: tuple[tuple[int | None, ...], ...]` — the group's peak simultaneously-live
register-resident tile set, one `dim_block_ids` tuple per tile, `None` for a static/broadcast
dim. Loop-carried accumulators are **captured inline at their real shape**, which is what
makes the footprint a plain sum rather than a reconstruction.

`row_reread` and `num_load` (`device_ir.py:1180-1192`): `num_load` is the load count *scoped to
this reduction's graph*; `row_reread` is true iff some load either feeds this reduction ≥2
times **or** feeds the reduction *and* also feeds a store. (Softmax is the canonical
both-at-once case: `num_load==1` yet `row_reread==True`, because `x` feeds both the row-max
and the sum.)

### 1.3 The budget — a register/byte capacity, not a table

`triton.py:4744-4778`, verbatim:

```python
# ----- THE BUDGET (a register/byte capacity; everything else is a per-axis desire) -----
ROW_PERSIST_MAX_BYTES = 245760                  # 240 KiB, "just over H100 SMEM"
CARRIED_PERSIST_MAX_BYTES = 245760 // 2         # 120 KiB — a CARRIED [grid_M, R] accumulator
PERSIST_HOLD_MAX_BYTES = 3 * 245760             # 720 KiB — persistence-HOLD watermark (big bucket)
USER_TILE_PERSIST_HOLD_MAX_BYTES = 294912       # 288 KiB (~1.2x ROW) — persistence-HOLD (small bucket)
LOOPED_CHUNK = 16384                            # looped-fallback chunk when the row does not fit
MIN_WAVES = 8                                   # grid >= num_sm * MIN_WAVES occupancy floor
WIDEN_MAX_ROWS = 8                              # diminishing-returns rows/program ceiling
```

The **footprint is faithful** and is the single formula everywhere; only *which ceiling* it is
compared against changes. `triton.py:4981-4987`:

> "The footprint is faithful: ``resident_bytes = itemsize × Σ over the group's live tiles of
> ∏(tile dim widths)``. Sizing an axis A splits that sum into ``(scale, flat)`` — tiles
> CONTAINING A scale with ``block(A)``, tiles WITHOUT A are constant — and the budget test is
> ``itemsize × (scale × block(A) + flat) <= budget`` (the constant term SUBTRACTED, never
> divided). No ``num_live`` multiplier, no separate accumulator sum, no feature-extent
> reconstruction: the live tiles ARE the resident set."

`footprint_terms`, `triton.py:5113-5138`:

```python
def footprint_terms(tiles, axis) -> tuple[int, int]:
    scale = 0
    flat = 0
    for tile in tiles:
        contains_axis = axis in tile
        prod = 1
        for d in tile:
            if d is None or d == axis:
                continue
            prod *= conservatively_large_tile_width(d)
        if contains_axis:
            scale += prod
        else:
            flat += prod
    return max(1, scale), flat
```

`conservatively_large_tile_width` (`:5140-5150`) returns an axis's **seated** width if already
chosen, else its **full extent** — safe *by seating order*, and the code flags that the
footprint is therefore order-dependent and the `order` sort is load-bearing for correctness,
not cosmetic.

### 1.4 Two passes

`size_reduction_tiles` (`triton.py:4970-5355`) runs per co-residency group:

**PASS 1 — seat the reductions** with the grid axes pinned at their floor, in priority order
`FULL_EXTENT → USER_TILE → GRID_TILE`, ties broken by descending extent (`:5171-5179`).
- `FULL_GRID`, and a *materialized* full-width `FULL_SLICE` the roller declined (no tunable
  slot at all — e.g. a grad-parameter `grad_weight[N]` axis or a specialized `group_size`),
  are seated at the **full extent** and never chunked: they cannot be split across programs and
  have nowhere to emit a chunk. Seating them full-width is what lets a co-resident inner tile
  see the real `N` instead of reading it as 1 and growing into a spill (`:5192-5209`).
- Everything else is **first sized as a streamed chunk from the byte budget** (`:5218-5225`):

```python
avail = persist_budget_for(d) // itemsize - flat
byte_budget = _pp2(max(1, avail // scale))
r = max(1, min(cls.LOOPED_CHUNK, byte_budget, ext))
```

  where `persist_budget_for(d)` is `CARRIED_PERSIST_MAX_BYTES` if `d.carried_2d_count > 0`
  else `ROW_PERSIST_MAX_BYTES` (`:5059-5064`). This is the **only** place the
  carried-vs-streamed distinction lives.

**PASS 2 — the grid-M rows take the remainder** (`:5265-5315`). Three outcomes, all pure
per-axis **membership** results, no `cdiv` branch and no kernel recognizer:
- grid axis **not** in any live tile → *reduced away* (a sequential cross-grid `.sum(0)`
  finalize, the grad-parameter idiom) → raise its floor to `next_pow2(grid_rows // num_sm)` to
  collapse the finalize to ~1 SM wave.
- kernel has any `carried_2d_count > 0` → the resident grid stays at **floor** (widening
  multiplies a register-pinned `[grid_M, R]` accumulator and trips the CTA/SM occupancy cliff
  that neither the byte widen nor the program-count widen can see).
- otherwise **widen** into the byte remainder:
  `blk = max(floor, min(byte_widen, occ_widen, rows_ceiling, ext))` with
  `occ_widen = prev_pow2(grid_rows // (num_sm * MIN_WAVES))` and
  `rows_ceiling = ext if pd.category is FULL_GRID else WIDEN_MAX_ROWS`.

**Then the non-reduction loops last** (`:5317-5335`) — welford's normalize pass,
`rms_norm_per_block`'s `groups_per_row` — each against a **fresh** budget
`prev_pow2(ROW_PERSIST_MAX_BYTES // itemsize)`, optionally tightened by a subclass hook
(`non_reduction_loop_block_cap`; sm100 returns 4096 elements).

### 1.5 Persistent vs looped (`reduction_loops`)

Crucially, **there is no separate persistence branch.** The chunk is sized first, then *lifted*
to full extent iff four analytical conditions hold (`triton.py:5226-5240`):

```python
element_cap = env.backend.max_tensor_numel          # Triton: 2**20 = 1048576
expand_to_persist = (
    d.row_reread                                    # a persistent pass fuses reduce+apply -> 1 HBM load
    and d.carried_2d_count == 0                     # a carried tile is held the whole loop: it chunks
    and (element_cap is None or raw_ext <= element_cap)
    and itemsize * (scale * raw_ext + flat) <= hold_ceiling
)
if expand_to_persist:
    r = ext
...
persistent = r >= ext and d.category in FULL_EXTENT_CATEGORIES
```

The byte test uses the **raw** (non-pow2-padded) extent — the true resident element count.

`hold_ceiling` is one of **two calibrated buckets**, selected by
`_has_store_only_row_reread(spec, pd)` (`:4852-4890` / `:5156-5160`) — is the row *also* loaded
by a pass that feeds a store and no reduction?
- **no** store-only re-read (cross_entropy, sum) → reuse is register-resident → `720 KiB`.
- **yes** (softmax, rms_norm, layer_norm, welford) → the row is re-swept from L2 → `288 KiB`.

The code labels this an **ADMITTED PROXY** and names itself as the first suspect on a
regression (`:4862-4867`):

> "That quantity is not cleanly recoverable from any seed-time signal — kernels with the same
> byte footprint, load count, and output width can flip persist->chunk at ~2x-different points
> — so this is an ADMITTED PROXY: it classifies the tested kernels correctly but is not a
> faithful measure of the underlying cache-tier question and can be fooled ... If a kernel
> regresses on the persist ceiling, this proxy is the first suspect."

The comment on `PERSIST_HOLD_MAX_BYTES` (`:4755-4766`) states the *measured* reason two buckets
exist: "the true cutoff is not a single faithful byte budget (e.g. softmax flips at ~128-160
KiB, cross_entropy at ~256-384 KiB with the same footprint)".

Emission (`:5412-5436`, standard track): a *materialized* rdim emits `reduction_loops = []`
(already full-width persistent; a length-1 list would fail normalize against the 0-length
spec); a single rolled reduction emits `[None]` if persistent else `[r_block]`; a
multi-rolled kernel emits one entry per spec, using `alloc.rolled_loop_sizes` for the
non-primary axes.

### 1.6 Register-pressure / spill estimation

There is **no register-count or spill-count model.** The register-pressure estimate *is* the
byte footprint above, with `itemsize` = the **fp32-promoted accumulator** itemsize, compared
against calibrated byte ceilings. Reference points: `ROW_PERSIST_MAX_BYTES = 240 KiB` ≈ one
SM's 256 KiB register file; `PERSIST_HOLD = 720 KiB` deliberately over-subscribes it (trading
some spill for an avoided HBM re-read); `CARRIED_PERSIST = 120 KiB` halves it for a tile held
resident across the whole inner loop.

Historically there *was* an explicit spill proxy: PR #2828 added a walker liveness fact
`ReductionFact.body_live_tiles` ("peak simultaneously-live rdim-shaped tiles") and a
multi-tile spill ceiling, motivated by `fused_linear_jsd`, "whose softmax->log_softmax->KL->grad
chain holds ~7 live full-width fp32 tiles; persisting spilled (n_spills 480 -> 0 after
looping)" — worth **0.609 → 0.835** (bf16) and **0.857 → 1.141** (fp32) vs torch.compile on
that kernel, with the other 16 of 17 kernels byte-identical. PR #2996 then replaced the
per-lever recognizers (including that one) with the single Σ-over-live-tiles budget.

### 1.7 The num_warps ramp — what it is a function of

**Base ramp (sm90, and the sm100 fallback), `triton.py:4901-4913` — keyed on the primary
reduction EXTENT only:**

```python
@classmethod
def _num_warps(cls, pd: ReductionDescriptor) -> int:
    """Scale num_warps with the reduction extent (pow2): rnumel <= 1024 -> 4, <= 4096 -> 8,
    <= 16384 -> 16, > 16384 -> 32."""
```

Plus one floor: if `_has_reduced_away_grid(spec)` (the grad-parameter M-collapse idiom), take
`max(M_COLLAPSE_MIN_NUM_WARPS, num_warps)` — 8 on sm90, **4** on sm100 (`:5409-5410`,
`:5377`, `:5569`).

**sm100 re-tune, `_b200_num_warps` (`triton.py:5594-5631`) — keyed on LOAD TRAFFIC, with extent
as a sub-key.** This is the live path for every non-M-collapse reduction on B200 and *shadows*
the base ramp. It is purely subtractive: "only lowers, never raises", and returns `None`
(keep the base ramp) for five excluded regimes.

```python
# Load traffic per row = elems × load-width × #loads.
traffic = pd.size_hint * max(1, pd.input_load_itemsize) * max(1, pd.num_load)
if traffic <= cls.NW8_MAX_ROW_TRAFFIC:            # 64 KiB
    if pd.size_hint <= cls.NARROW_ROW_MAX_ELEMS:  # 1024
        return cls.NARROW_ROW_NUM_WARPS           # 2
    return cls.WIDE_ROW_NUM_WARPS                 # 4
if pd.size_hint <= cls.HEAVY_ROW_MAX_ELEMS:       # 16384
    return cls.HEAVY_ROW_NUM_WARPS                # 8
return None                                       # genuinely huge rows keep the base ramp
```

Guards that return `None` first, in order: `_has_reduced_away_grid` (M-collapse is a cross-warp
accumulate, not a streamed row); a non-empty `non_reduction_loop_ids` (reduce-then-apply, whose
re-read is invisible to the traffic key); more than one sized reduction (a second one adds
cross-warp compute with a non-monotonic warp optimum); and a **looped** reduction (positive
`reduction_loops` chunk keeps the base ramp).

The three-rung ladder is the content of commit `a9d12c1e9` / PR **#3338**, "[autotuner] retune
the sm100 reduction num_warps ramp". Its message (unsquashed `ffc2db758`) gives the transition
table and the measurement protocol:

```
rnumel <= 1024, light traffic  ->  2  (was 4)
rnumel  > 1024, light traffic  ->  4  (was 8)
heavy traffic, rnumel <= 16384 ->  8  (was the base ramp's 16/32)
M-collapse floor               ->  4  (was 8)
```

- "Measured on a B200 with the kernel captured K times inside one CUDA graph, so the ~1us
  per-replay CPU cost is amortized ... at K=1 every warp count reads identically; the arms only
  separate at K>=8."
- Held-out shapes only (tuning used M=4096 / pow2 extents): **n=59, geomean 1.1158x, 31 wins
  >2%, 24 neutral, 4 regressions >2%.** examples/ reductions n=29 geomean **1.0938x**; vLLM
  reductions n=30 geomean **1.1376x**.
- Largest single movers: `softmax M=32768 N=3072` nw 8→4, **120.62 → 77.02 us (1.566x)`;
  `per_token_group_fp8 h=2048 t=512` nw 8→2, **4.76 → 2.71 us (1.755x)**;
  `layer_norm_fwd M=1024 N=3072` nw 8→4, 4.75 → 3.51 us (1.351x).
- Independent corroboration: "The vLLM configs independently corroborate the narrow rung —
  `per_token_group_fp8_quant` ships 15 tuned cells at nw2 vs 3 at nw1 over a group_size of 128,
  values the old ramp could not express."
- **Disclosed regression, not worked around:** `silu_and_mul_per_block_quant` (extent PINNED at
  128 by `group_size`) wants nw2 only at small token counts and nw4/nw8 above:
  `i=25600 t=512 → nw1 88.2 / nw2 31.2 / nw4 23.5 / nw8 23.9 us`. "The driver is total work
  (groups = i * t / group_size), not the reduction extent... No gate for this is included
  because the signal is not in the reduction descriptor: extent, num_load, traffic and the
  non-reduction-loop count are all identical between the nw2-wanting and nw4-wanting cases."
  Costs ~1.5 points of the held-out geomean.
- Also fixed a `_has_reduced_away_grid` false positive (a `hl.tile(B, block_size=1)` axis used
  as a scalar index is non-resident but batches nothing): it now additionally requires the axis
  to batch >1 row/program. "On the decode kernel the old form fired on two `block_size=1` axes
  and cost up to 1.66x."

A **refuted** rung is worth quoting too (commit `d017fc901`): a `num_warps=1` narrow-row
refinement was removed because it "fired num_warps=1 on 768/896/1024-wide bf16 rows at LOW
occupancy, where w1 is exactly wrong -- worst was fused_add_layernorm (16384,1024) bf16 at
**4.3x slower** than the ramp." PR #3338 also declines nw1 for the same reason:
"`layer_norm_bwd` at 32768x1024 is 16x slower at nw2 than nw4, so the low end is a cliff."

### 1.8 Block-size selection on the non-reduced (grid-M) axis

Covered in §1.4 PASS 2. The three outcomes are **widen / floor / collapse**, and all three are
budget/membership outcomes rather than branches. Two details a blog paragraph should get right:

- `_m_axis_block_size` (`:4921-4943`) handles a **grid-pinned** axis (`hl.tile(M,
  block_size=1)` — the idiom every vLLM quant kernel uses), which has *no tunable slot at all*
  and must be read off `env.block_sizes`.
- The `WIDEN_MAX_ROWS = 8` ceiling explicitly does *not* bound (a) the grad-param collapse
  branch, nor (b) a raised `autotuner_min` floor, because `max(floor, ...)` still wins. This is
  visible in the data: `rms_norm [589824, 256]` emits `block_sizes=[16]` even though
  `WIDEN_MAX_ROWS` is 8, because the huge-M `autotuner_min` floor is 16.

### 1.9 Bytes-in-flight / bandwidth reasoning in the reduction seed

The reduction seed's budgets are **residency** budgets (register/SMEM bytes), *not* a bandwidth
budget — unlike the pointwise seed, it has no `TILE_BYTES` HBM-saturation term. Its memory
reasoning enters in exactly three places:

1. **The load-traffic warp key** (§1.7): `traffic = extent × input_load_itemsize × num_load`,
   thresholded at 64 KiB. Commit `7a98cd536` gives the physical argument: "A light-traffic
   (<=64 KiB) PERSISTENT, single-pass streamed row reduction is memory-bound on B200's wider
   memory system (HBM ~2.3x, L2 ~2.5x), and the base extent-ramp's 16 warps over-subscribe it".
   Note "dtype enters only as the itemsize multiplier inside the traffic budget (a faithful
   workload quantity, not a dtype fence)".
2. **L2 eviction policies** (`_eviction_policies`, `:4945-4967`): a single streamed input
   (`pd.num_load == 1`) sets every load to `'first'` (free L2 immediately); a re-read row
   reloaded across a grid-**collapse** loop pins its first load `'last'` (L2-resident) and the
   rest `'first'`, with the slot index read straight off the descriptor
   (`reread_eviction_index`), not re-walked per config. The standard track gates the `reread`
   policy on `_has_reduced_away_grid`, *not* on `not persistent`: "a single fused persistent row
   does not reload from L2, so pinning there only oversubscribes L2 and evicts store lines".
   The user-tiled track applies `reread` whenever `pd.row_reread` **or** a non-reduction loop
   exists, "even when PERSISTENT: the second pass still re-fetches x from HBM
   (profiler-confirmed)".
3. **Persistence itself** is justified as an HBM-traffic argument — "a persistent pass fuses
   reduce+apply to one HBM load" (`:5227`).

One more B200 lever with a *measured mechanism*, `NON_REDUCTION_LOOP_MAX_ELEMS = 4096`
(`:5559`, `non_reduction_loop_block_cap` `:5571-5576`). Commit `faffeb7ef` records that the
win was mis-attributed at first: "Gate F (ncu) established the win is a register-pressure ->
occupancy cliff, and a NON-re-reading apply loop hits the identical cliff (~0% L2 reuse), so
`row_reread` was only a COINCIDENTAL proxy for register-footprint, not the faithful key." The
key was widened to every non-reduction apply loop, verified corpus no-op (0/874 config diffs).

### 1.10 Reduction: safety work that made promotion legal

`promote_seed_to_default = True` on `_TritonReductionSeedBase` (`:4742`) — so the reduction
seed is the **autotune-off compiler default**, not just a search seed, on sm90 *and* sm100.
PR #3036 gated that on three seed-vs-default correctness fixes found by a B200 hunt:
1. hard-coded `pid_type='flat'` crashed under `hl.barrier()` / data-dependent grid bound; both
   seeds now route through `_materialize_config` (`:403-426`), which *replaces* an illegal
   `pid_type` with `allowed_pid_types[0]` rather than popping it (a pop let `normalize` refill
   `'flat'`).
2. the byte budget could collapse a rolled chunk to `reduction_loops=[1]`, which
   `LoopedReductionStrategy` rejects; `ReductionLoopSpec._normalize` now floors a degenerate 1.
3. a `cumsum`/`hl.associative_scan` inside a rolled reduction re-ran the scan per chunk with no
   cross-chunk prefix carry (**silently wrong ~55%**); the roller now refuses to roll a
   scan-containing graph.

---

## 2. THE POINTWISE HEURISTIC

One class: `TritonPointwiseSeedHeuristic` (`triton.py:4322-4620`), `name="triton_pointwise"`,
`promote_seed_to_default = True`, `PROMOTE_TARGETS = (("cuda","sm90"), ("cuda","sm100"))`,
`TUNED_TARGETS = (("cuda","sm100"),)`. Note the deliberate split: the heuristic **fires
arch-agnostically** (`is_eligible` is just `bool(env.config_spec.pointwise_facts)`), and only
*promotion to default* and the *sm100 constant set* are arch-gated.

### 2.1 Why it exists (the number the blog will want)

Docstring (`:4323-4330`): "A pointwise kernel (no reduction / matmul / accumulator) is
BANDWIDTH-bound, but the compiler defaults it to `block_size=32` (~10% of HBM)." That default
is real and verifiable: `BlockSizeSpec._fragment` (`config_spec.py:3236-3267`) returns
`default = 32` when `total_ndim <= 2 and reduction_numel <= 128`. PR #2866: "The compiler
default tiles a flattened pointwise op at `block_size=32`, moving only ~10% of HBM (~0.32 TB/s
on H100). The seed picks a byte-budgeted, size_hint-aware, coalesced tile (~2.2 TB/s)."

### 2.2 The Stage-1 fact

`PointwiseElementwiseFact` (`config_spec.py:675`), built by
`DeviceIRAnalysis.pointwise_fact` (`device_ir_analysis.py:1006-1171`):

```python
total_numel: int            # ∏ tiled block dims' size_hints — the problem element count
slab_numel: int             # UNTILED inner slab in ELEMENTS that full-extent ops drag per tiled
                            # element = Σ over ops with accessed_numel >= total_numel of
                            # (accessed_numel // total_numel).  flat kernel -> 1/op; rope -> heads*head_dim
storage_itemsize: int       # HBM byte width (max over ops)  -> bandwidth traffic
compute_itemsize: int       # widest COMPUTE (fp32-promoted) byte width -> register cap
contig_block_ids: tuple     # TILED block-ids that are the stride-1 axis of some full-extent op
sfu_ops: int = 0            # count of transcendental (SFU) ops
gather_stride: int = 1      # widest GATHER stride (tile-index multiplier in the ADDRESS expr)
max_op_slab_numel: int = 1  # LARGEST single op's untiled fan-out (vs slab_numel's SUM)
```

A **broadcast** operand (`bias[N]`, `[M,1]`, a stride-0 `.expand()`) has
`accessed_numel < total_numel`, is amortized, and is **excluded**; an *oversized* operand still
touches the full problem and is counted (hence the `>=`).

### 2.3 The three caps and the floor

`get_seed_config`, `triton.py:4416-4468` (quoting the load-bearing lines):

```python
slab_bytes = max(1, fact.slab_numel * fact.storage_itemsize)
reg_bytes  = max(1, fact.slab_numel * fact.compute_itemsize)
# Charge bytes FETCHED from HBM, not bytes the kernel finds useful.
fetch_bytes = slab_bytes * max(1, fact.gather_stride) if tuned_arch else slab_bytes
budget_target = max(1, cls.tile_bytes_for(env) // fetch_bytes)   # TILE_BYTES: 8192 sm90 / 16384 sm100
reg_cap       = max(1, cls.REGISTER_BYTES // reg_bytes)          # REGISTER_BYTES = 65536
occ_cap       = max(1, fact.total_numel // (num_sm * cls.min_waves_for(env)))  # MIN_WAVES 8 sm90 / 4 sm100
target        = max(1, min(budget_target, reg_cap, occ_cap))
inner_floor   = min(cls.BLOCK_FLOOR, cls._pow2_floor(reg_cap))   # BLOCK_FLOOR = 256
balance_cap   = max(1, reg_cap)
```

Three important asymmetries, each documented in the code:
- The anti-undershoot floor (`BLOCK_FLOOR = 256`) is capped by the **register** budget **only**
  — not by `budget_target`, not by `occ_cap`: "keep a coalesced per-operand run, lowering it
  only on a genuine register overflow (a heavy rope slab → reg_cap≈1). Low occupancy or a
  fan-in kernel's small byte budget is not worth it."
- `reg_cap` is admitted as a **coarse proxy** (blind to compute temporaries) but "benign —
  pointwise is memory-bound; its only jobs are relaxing the floor for a heavy slab and capping
  the transpose-conflict tile."
- Occupancy is handled by shrinking the tile (`occ_cap`), **not** by the warp count — and the
  warp comment explicitly says the reverse trade "measured worse on every cell tried".

### 2.4 bytes-FETCHED vs bytes-USEFUL (the gather-stride term)

Present, and it is the headline of PR **#3297** ("tune the pointwise seed heuristic for
sm100"). Code comment, `:4429-4436`:

> "Charge bytes FETCHED from HBM, not bytes the kernel finds useful: a stride-k gather pulls k
> 32B sectors per useful sector, so a W-element tile costs `k*W*slab_bytes` of real traffic.
> Charging useful bytes over-sizes a strided tile by k, and makes two kernels with the same
> fact but different gather strides get the same width when their optima differ by exactly that
> factor."

`gather_stride` is a **new fact field** because the pre-existing signal provably cannot express
it: `MemoryOpFact.subscript_strides` is the accessed tensor's **layout** stride, and both
sglang `silu_and_mul` variants index the *same row-major tensor* — the interleave lives in the
index expression (`2*i` / `2*i+1`). It is populated by a `subscript_index_scale` walk that also
recovers the block-id a scaled subscript loses (the lowering records tile provenance for
`tile + const` but not `tile * const`).

The measured invariant (`/home/dev/local/pointwise-curriculum/WORKLOG.md`, Entry 15): **fetched
bytes are the conserved quantity at peak bandwidth, ~10 KiB per program, and useful bytes are
not.** Stride-1 `silu_and_mul` peaks at width 1024, stride-2 `silu_and_mul_interleaved` at 512,
on every cell — exactly the 2x the fetch model predicts. Geomean fetched bytes **9569 (stride 1,
n=7) vs 11763 (stride 2, n=5)** (equal within spread); geomean *useful* bytes **9569 vs 5881**
(a clean 1.6x apart).

Direct A/B on the constant (`WORKLOG.md` Entry 17): with the fetch charge, `TILE_BYTES=8192`
collapses the stride-2 family's tile to `[1,256]` and **breaches −20%** (`F1 interleaved`
geomean 0.9917 → 0.7966), so sm100 keeps 16384 — "decided by measurement, not by argument".
Honest caveat recorded there: the optimal fetched-byte budget is **not** constant across
families (slab-10 families want ~10 KiB, slab-6 families 12–24 KiB); 16384 is the compromise.

### 2.5 The warp-count law

Two paths. **sm90 = the SFU ramp only** (`_warps_for`, `:4538-4549`): `≥9 SFU → 16`, `≥3 SFU →
8`, else 4, capped by `pow2_floor(tile_numel // ELEMS_PER_WARP)` and floored at `DEFAULT_WARPS`.

**sm100 = lanes + residency, with the SFU ramp demoted to a floor** (`_warps_for_sm100`,
`:4490-4536`):

```python
per_thread    = cls._elems_per_thread(fact.gather_stride)        # 16 / 4 / 2 banded ladder
work          = cls._pow2_floor(max(1, tile_numel // (32 * per_thread)))
programs      = max(1, fact.total_numel // max(1, tile_numel))
wave_slots    = warp_slots * cls.WAVE_TARGET_NUMERATOR // cls.WAVE_TARGET_DENOMINATOR   # 64 * 3//4 = 48
resident_wave = cls._pow2_floor(max(1, (num_sm * wave_slots) // programs))
target        = max(work, resident_wave)
if fact.sfu_ops >= cls.SFU_W16:  target = max(target, cls.MAX_WARPS)   # 9 -> 16
elif fact.sfu_ops >= cls.SFU_W8: target = max(target, 8)               # 3 -> 8
lanes = tile_numel * max(1, fact.max_op_slab_numel)
cap   = cls._pow2_floor(max(1, lanes // cls.ELEMS_PER_WARP))           # ELEMS_PER_WARP = 64
return max(1, min(cls.MAX_WARPS, target, cap))                         # MAX_WARPS = 16
```

Rationale in the docstring: "`sfu_ops` alone is the wrong signal for bandwidth-bound work -- a
1-op activation sits below `SFU_W8`, so every such kernel fell to the default warp count
regardless of tile size." The two replacement terms are combined with `max` because "both answer
'how many warps does this program want'": lanes the tile can keep busy, and "the largest warp
count whose CTAs still fit resident at once -- **warps are free until they cost a second wave**.
Inert on a saturated grid."

The starvation `cap` uses `max_op_slab_numel` (**max** per-op fan-out) and **not** the
`slab_numel` **sum** the byte budgets use, "since a CTA's separate vector instructions run over
the same threads" — bytes add across ops, lanes do not. This is what lets a `[1,1]` rope tile
still earn 2 warps (it materializes an 8192-element untiled load).

Fit quality (WORKLOG Entry 14 / final verdict): the law is
`clamp(max(tile_numel/(32*E), num_sm*48/programs), 1, 16)`, grid-searched over
`E1 ∈ {8,16,32}`, `E2 ∈ {2,4,8}`, `WARPS_PER_SM ∈ {16,32,48,64,96,128}`, `MAX_WARPS ∈
{8,16,32}`, starvation divisor `∈ {64,128,256,512}` against an **81-group / 702-point** set.
Winner `E1=16, E2=4, W/SM=48`, geomean **0.9870** of the per-cell oracle (next best 0.9856,
W/SM=16 gives 0.9746, W/SM=64 gives 0.9774). Versus the fixed `w4` it replaces: **0.986 vs
0.80–0.90**.

The `ELEMS_PER_THREAD_BY_STRIDE = ((1,16),(4,4))` + `ELEMS_PER_THREAD_WIDE_GATHER = 2` bands are
also measured, and an earlier "saturates past stride 2" claim was **retracted as an overclaim**
after sweeping strides 8 and 16 (modal E: stride 1→8–16, 2→4, 4→4, 8→2, 16→1–2). Ladder fit over
61 groups: 3-band (16/4/2) **48/61 exact, geomean 0.986**; 2-band (16/4) 42/61, 0.964; the
continuous `max(1, 16 // stride)` **38/61, 0.952** — "A continuous formula is *worse* — it
collapses stride 2, where the measured optimum is 4 and not 8."

### 2.6 The tile/warps RIDGE

This is the sharpest single finding in the pointwise work and it belongs in the blog.
`WORKLOG.md` Entry 13, "the byte budget is NOT the lever; warps are":

> "**At the best warp count, the tile barely matters.** swiglu 4096×14336: `[1024]`w4 = 6643
> GB/s, `[2048]`w4 = 6607, `[4096]`w8 = 6643 — a **0.5% spread across a 4× range of tile
> size**. fp8 512×14336: `[1,512]`w1 = 3252, `[1,1024]`w2 = 3161, `[1,4096]`w8 = 3374 — 6% over
> an 8× range. **But at a FIXED wrong warp count the same tiles spread 2–3×** (fp8 512×14336 at
> w1: 2143 → 3252 → 3161 → 2898 → 1026 as the tile grows; swiglu at w1: 2995 → 5889 → 6592 →
> 5877 → 4490).
>
> So the surface is a RIDGE along `tile_numel / num_warps ≈ const`, and both fixes were looking
> at the wrong axis: `TILE_BYTES` moves you *along* the ridge (nearly free), `num_warps` moves
> you *off* it (expensive)."

Entry 15 **partially supersedes** this and the blog must carry both halves: "Half right: at a
*fixed* stride the tile is second-order (the ridge is real). But ACROSS strides the budget is
first-order, because charging useful bytes puts the stride-2 kernel one full pow2 step off its
peak. Both fixes were needed."

Practical corollary the code encodes: **tune warps first, tile second**; and the *occupancy*
lever is applied to the tile (`occ_cap`), never to the warp count.

### 2.7 Tile *shape*: coalescing-aware distribution

`_seed_block_sizes` (`:4565-4601`) does not blindly widen the last dim:
- exactly one contiguous axis → fill it innermost-first and spill leftover budget outward. A
  transposed view roots the wide tile on dim 0 (`[1024,1]`, not the uncoalesced `[1,1024]`).
- **≥2 contiguous axes = a coalescing CONFLICT** (e.g. transposed load + contiguous store) → no
  single wide axis coalesces every operand, so emit a **balanced square-ish** pow2 tile
  (`_balanced_block_sizes`, `:4603-4620`) filled up to `balance_cap = reg_cap` only, because
  "the bandwidth budget is wasted on the strided operand, and a long coalescing run beats more
  programs".
- The register-capped floor applies **only** to the primary contiguous axis.
- `_clamp_dim` (`:4555-4563`) rounds **down** to a pow2 in `[floor ∪ min_size, max_size]` and
  deliberately does **not** apply `autotuner_min` ("the autotuner's search floor, not a seed
  constraint").

Measured motivation (PR #2866): `transposed_out_add [2048,512]` gets a balanced `[64,64]`
(1.14x default, 0.96 tc-parity), whereas "a coalescing-blind innermost tile `[1,512]` strides the
transposed operand and runs **2.6x slower** than the seed (0.44x default)".

### 2.8 `num_warps` is emitted only when it differs from the default

`:4465-4468`: `num_warps=num_warps if num_warps != cls.DEFAULT_WARPS else None` — "so a shape
landing on the default stays block_sizes-only (no dead knob)". This is why most pointwise seeds
in the audit carry only `block_sizes`.

---

## 3. "Analytical model, not a table" — the one-sentence versions

**Reduction:** *The reduction heuristic never matches a kernel; it solves one inequality —
`itemsize × (Σ over the co-residency group's actually-live tiles of ∏ tile widths) ≤ budget` —
for each axis in priority order, so persistent-vs-looped, floor-vs-widen and collapse are
**outcomes of one byte budget**, and the only kernel-specific numbers are five calibrated byte
ceilings and a three-rung load-traffic warp ladder.*

**Pointwise:** *The pointwise heuristic computes a tile from three closed-form per-program caps
(HBM bytes **fetched** = `TILE_BYTES / (slab_bytes × gather_stride)`, an fp32 working-set cap,
and a grid-occupancy cap), then a warp count from `max(lanes-the-tile-can-fill,
warps-that-still-fit-one-resident-wave)` — a formula whose constants were fit against 702
measured points, not a per-kernel table.*

Both are honest about being *calibrated* rather than derived: the reduction ceilings (240/120/720/288
KiB, 16384, 8 waves, 8 rows) and the pointwise constants (16384/65536/256/4 waves/E-bands/48
warp-slot wave target) are hill-climbed numbers plugged into an analytical form. Notably
`WAVE_TARGET_NUMERATOR/DENOMINATOR = 3/4` over a 64-slot hardware limit is a re-parameterization
of an empirically fit `WARPS_PER_SM = 48`: the WORKLOG says outright "It is **not** a hardware
quantity (B200's limit is 64 warps/SM). It is the argmax of a grid search over
{16,32,48,64,96,128} against 116 measured groups."

### Main modeling gap — reduction

**The persistence ceiling is a proxy for a cache-tier question the compiler cannot see.**
`_has_store_only_row_reread` decides between a 288 KiB and a 720 KiB hold ceiling by asking
whether the row tensor is also loaded by a store-feeding pass. The code states the real
question is whether persistence's avoided HBM re-read is served from L2 or the register file,
that "kernels with the same byte footprint, load count, and output width can flip
persist->chunk at ~2x-different points", and that the proxy "can be fooled (e.g. a 2-pass
kernel whose 2nd pass reduces instead of storing re-reads the row identically but reads as
False)". Runner-up gap: **there is no work-per-program term**, which is exactly why
`silu_and_mul_per_block_quant`'s nw2-vs-nw4 need is inexpressible (PR #3338 disclosed it rather
than fitting six datapoints).

### Main modeling gap — pointwise

**`reg_cap` is blind to compute temporaries**, and `storage_itemsize` is a max-over-ops
approximation rather than a per-op sum, so the byte charge is wrong for mixed-width kernels
("F3 truly moves 5 B/elem, the fact says 6. Real, 1.2×, wrong direction for the F3 symptom").
The visible consequence is the **non-monotonic bandwidth-vs-width surface** on
`silu_mul_fp8`: "a local peak, a dip, then the global peak, so the seed lands in the dip and no
smooth `f(shape)` rule hits both peaks. Three candidate fixes were tried and each cost more
elsewhere than it recovered." (PR #3297; worst cells 0.87x / 0.94x, ~1.7% geomean on that
family.)

---

## 4. HOW THE FOUR FAMILIES ARE DISPATCHED

### 4.1 It is fact presence, not kernel matching — and the fact layer is disjoint by construction

`DeviceIR.lower_to_device_ir` builds the facts in a **fixed order** (`device_ir.py:3002-3050`),
and later phases are defined by the **absence** of earlier ones:

```
analysis  = DeviceIRAnalysis.build(device_ir, env)              # one liveness/structure sweep
memory_op_facts                                                 # per load/store metadata
config_spec.accumulator_facts = build_accumulator_facts(...)     # loop-carried accumulators
build_reduction_kernel_fact(memory_op_facts, accumulator_facts, analysis)   # Phase 3
build_kernel_matmul_fact(analysis)                              # Phase 3b (any kernel with a dot)
build_matmul_reduction_epilogue_facts()                         # Phase 4 (composed)
build_pointwise_facts(analysis)                                 # Phase 5 (defined by ABSENCE)
```

The **disjointness rule** lives in `DeviceIRAnalysis.pointwise_fact`
(`device_ir_analysis.py:1017-1027`):

```python
reduction_fact = spec.reduction_kernel_fact
has_sized_reduction = reduction_fact is not None and any(
    descriptor.category in SIZED_REDUCTION_CATEGORIES for descriptor in reduction_fact.reductions
)
if has_sized_reduction or spec.matmul_facts or spec.accumulator_facts:
    return None
if not spec.memory_op_facts or not spec.block_sizes:
    return None
```

Plus a safety allow-list that *also* declines: any `associative_scan` / `_reduce` HOP, an opaque
`inline_triton` / `triton_kernel` / `inline_asm_elementwise` body, or **any atomic**
(`device_ir_analysis.py:1030-1039`). PR #3053's reason: "Without this, such kernels fell through
to the promoted large tile and produced wrong results (e.g. a segmented associative_scan split
across tiles)."

And symmetrically, the reduction gate declines any kernel with a matmul
(`_triton_reduction_eligible`, `triton.py:4623-4638`):

```python
spec = env.config_spec
if spec.matmul_facts:
    return False                      # GEMMs route to the matmul seeds
kf = spec.reduction_kernel_fact
if kf is None:
    return False
return any(d.category in SIZED_REDUCTION_CATEGORIES for d in kf.reductions)
```

### 4.2 The per-family gate, in one table

| Family | Heuristic(s) | `is_eligible` predicate | Source |
|---|---|---|---|
| single matmul | `TritonH100MatmulHeuristic` (sm90), `TritonB200FormulaMatmulHeuristic` (sm100), `TritonSkinnyGemmHeuristic`, legacy `TritonB200MatmulHeuristic` table | hardware match **AND** `_generalized_static_matmul_fact(spec) is not None` (one static contraction, M/N/K each tunable-tiled or fixed-full-extent) **AND** not fp8 | `triton.py:3232-3274` |
| **multi-matmul** | `TritonB200MultiMatmulHeuristic` | sm100 **AND** `kernel_matmul_fact.matmuls` non-empty **AND** `_generalized_static_matmul_fact(...) is None` (front end 1 declines) **AND** ≥1 dot has a non-`UNKNOWN` sizable extent on all of m/n/k **AND** no fp8 | `triton.py:3569-3597` |
| reduction | 4 tuned classes + narrow fallback | hardware match **AND** `_triton_reduction_eligible` **AND** `_is_standard_reduction(pd)` (standard track) or its negation (user-tiled) | `triton.py:5379-5386`, `5486-5493`, `5673-5681` |
| pointwise | `TritonPointwiseSeedHeuristic` | `bool(env.config_spec.pointwise_facts)` — nothing else | `triton.py:4411-4413` |
| (5th, composed) matmul + reduction epilogue | `TritonMatmulReductionEpilogueHeuristic` | sm90 **AND** exactly one `matmul_reduction_epilogue_facts` **AND** its N axis is `hl.specialize`'d (`n_block_id is None`); **not promoted** | `triton.py:5742-5749` |

The "primary reduction" that every scalar lever is read off is selected by
`_primary_descriptor_selected` (`triton.py:4641-4672`): **max ROW-BYTES**
(`size_hint * input_load_itemsize`) over the *backed* sized descriptors — explicitly **not**
category tier-order, "which would mis-rank the group-quant kernels". The standard/user-tiled
discriminator is then just `pd.category in FULL_EXTENT_CATEGORIES` (`:4675-4680`).

### 4.3 The driver, and last-promote-wins

`compiler_seed_configs(env, device_ir)` (`autotuner_heuristics/__init__.py:93-131`) walks
`HEURISTICS_BY_BACKEND[env.backend_name]` **in registration order**, wraps each heuristic in a
`try/except Exception` that logs at debug and continues (a heuristic can never fail a compile),
collects `get_seed_configs()` (a ranked list) or `[get_seed_config()]`, appends the names to
`spec.autotuner_heuristics`, and sets `spec.compiler_default_config = ranked[0]` for every
heuristic whose `should_promote(env)` is true. **The last promoter wins**, which is why the
registration order in `__init__.py:56-79` is commented as load-bearing:

```python
"triton": (
    TritonH100MatmulHeuristic,          # H100 dense matmul seed FIRST (rank-0 Product-A seed)
    TritonSkinnyGemmHeuristic,
    TritonB200MatmulHeuristic,          # the demoted legacy table
    TritonB200FormulaMatmulHeuristic,   # registered after the table so it wins last-promote-wins
    TritonB200MultiMatmulHeuristic,     # front end 2, after front end 1
    TritonMatmulReductionEpilogueHeuristic,
    TritonStandardReductionHeuristicSM90,
    TritonStandardReductionHeuristicSM100,
    TritonUserTiledReductionHeuristicSM90,
    TritonUserTiledReductionHeuristicSM100,
    TritonNarrowReductionHeuristic,
    TritonPointwiseSeedHeuristic,
),
```

In practice the disjointness rule makes last-promote-wins moot **across families** — a kernel
can only be in one family — and the per-hardware gates make it moot *within* the reduction
family. It is only actually exercised between the sm100 matmul table and the sm100 matmul
formula, and between matmul front ends 1 and 2.

`should_promote` (`registry.py:41-51`) ANDs `promote_seed_to_default` with
`matches_hardware(env, PROMOTE_TARGETS)`; `PROMOTE_TARGETS = None` means "promote wherever the
heuristic fires". This is the mechanism that lets an arch-agnostic pointwise seed be offered
everywhere as a *search candidate* while defaulting only on sm90/sm100.

### 4.4 The fallback when nothing matches

`ConfigSpec.default_config()` (`config_spec.py:2740-2756`):

```python
def default_config(self) -> helion.Config:
    if self.compiler_default_config is None:
        return self._base_default_config()
    merged = dict(self._base_default_config().config)
    merged.update(self.compiler_default_config.config)     # a seed specifies only knobs it cares about
    config = helion.Config.from_dict(merged)
    self.normalize(config, _fix_invalid=True)
    self._shrink_for_numel_constraints(config)
    return config
```

`_base_default_config()` = `flat_config(lambda x: x.default())`, i.e. every fragment's own
default, then numel-constraint shrinking. For block sizes that is `BlockSizeSpec._fragment`
(`config_spec.py:3236-3267`): **32** if `total_ndim <= 2 and reduction_numel <= 128`; a
geometric `(32768/reduction_numel)^(1/ndim)` for `ndim>=3` with a nontrivial reduction; **16** if
`reduction_numel <= 256`; else **1**. `DEFAULT_NUM_WARPS = 4`, `DEFAULT_NUM_STAGES = 1`
(`config_spec.py` tail). Rolled reductions default to `min(next_pow2(size_hint), 4096)`, capped
at the backend's `max_reduction_loop` past `reduction_loop_force_threshold`
(`ReductionLoopSpec._flat_fragment`, `config_spec.py:3314-3327`).

**This `_base_default_config()` is the "raw unseeded default" arm in the benchmarks.** The
merge (rather than replace) behavior was a PR #3053 bug fix: applying a seed verbatim "dropped
every other key — including user `register_tunable` defaults".

Two more escape hatches: `Settings.disable_autotuner_heuristics` (`runtime/settings.py:541`)
makes `compiler_seed_configs` return immediately; and every heuristic's exception is swallowed,
so a fact-shape surprise degrades to the base default rather than failing the compile.

### 4.5 How a seed reaches the autotuner (not just the default)

Even when a heuristic does not promote, its config is planted as a search seed:
`ConfigGeneration.seed_flat_config_pairs` flattens `ConfigSpec.compiler_seed_configs`
(`autotuner/config_generation.py:474-490`), and `BaseSearch._generate_best_available_population_flat`
tries user seeds → compiler seeds → the raw default → cached configs
(`autotuner/base_search.py:1167-1185`). `default_flat()` returns the *promoted* default when one
exists (`config_generation.py:463-472`).

---

## 5. BACKEND GENERICITY AND THE REGISTRY (the portability angle)

### 5.1 What is generic and what is not

| Layer | Generic? |
|---|---|
| Stage-1 fact layer (`MemoryOpFact`, `AccumulatorFact`, `ReductionKernelFact`, `KernelMatmulFact`, `PointwiseElementwiseFact`) | **Backend-generic.** Built unconditionally in `device_ir.lower_to_device_ir` for every backend; the only backend queries are `env.backend.max_tensor_numel` and the reduction thread caps. |
| `AutotunerHeuristic` base + registry + promotion gating | **Backend-generic** (`registry.py`, `__init__.py`). |
| Shared structural gates in `common.py` (`is_canonical_row_reduction`, `matches_hardware`, `clamp_block_size_targets`) | **Generic**, and deliberately so: `is_canonical_row_reduction`'s docstring says it is "Shared by the CuTe and Triton reduction heuristics so the structural gate cannot drift between backends." |
| The reduction *budget allocator* and the pointwise *tile/warp model* | **Triton-only, and arch-fenced within Triton.** `backend = "triton"`; every concrete class carries `HARDWARE_TARGETS` / `TUNED_TARGETS` / `PROMOTE_TARGETS`. |

Concretely: `grep` shows `cute.py` reads **none** of `reduction_kernel_fact`, `pointwise_facts`,
`memory_op_facts` or `accumulator_facts`. The CuTe reduction heuristics key on `ConfigSpec`
*structure* via `is_canonical_row_reduction` instead. There is **no pointwise heuristic for CuTe
or Pallas at all** — `pointwise_facts` is consumed only at `triton.py:4413` and `:4420`.

### 5.2 How the registry organizes it

`HEURISTICS_BY_BACKEND: dict[str, tuple[AutotunerHeuristicType, ...]]`
(`__init__.py:42-84`), keyed by `env.backend_name`, resolved by `get_heuristics(backend)`:

- **`"triton"` — 12 heuristics**: 4 matmul (H100 formula, skinny GEMM, B200 legacy table, B200
  formula) + 1 multi-matmul + 1 composed matmul+reduction-epilogue + 5 reduction (2 tracks × 2
  arches + narrow fallback) + 1 pointwise.
- **`"cute"` — 12 heuristics**, all much thinner and mostly *shape/schedule* seeds:
  `CuteFp8GemmSkinnyMHeuristic`, `CuteFlashAttentionHeuristic`,
  `CuteFlashAttentionCausalLptHeuristic`, `CuteTcgen05ClusterM2{,Ffi}Heuristic`,
  `CuteTcgen05Grouped{StaticCommonK,DynamicBk64}Heuristic`, `CuteReductionTileHeuristic`,
  `CuteReductionWideChunkHeuristic`, `CuteTileVec{,WarpReduce,WarpPerRow}Heuristic`.
  For scale: `CuteReductionTileHeuristic.get_seed_config` is ~20 lines — it seeds
  `block_sizes=[1], num_threads=[1]` and `reduction_loops=[None]` when
  `size_hint <= spec.max_reduction_threads` (1024 for CuTe) else `[max_threads]`, plus a
  dtype-keyed vector width (4 for fp32, 8 for fp16/bf16). There is **no byte budget, no
  co-residency footprint, no warp ramp**.
- **`"pallas"` — 2 heuristics**, both narrow lookup-ish rules:
  `PallasMatmul{,F32}NoTilingSeedHeuristic` seed `[N,N,N]` for square bf16/fp16 cubes with
  `N ∈ {1024, 2048, 4096}` (f32: `{1024}` only) so the backend lowers through
  `lax.dot_general` and XLA `cross_program_prefetch` applies (~17% on bf16 1024³).
- `get_heuristics` returns `()` for any unregistered backend (e.g. `"metal"`), so those kernels
  get `_base_default_config()`.

### 5.3 Honest portability statement for the blog

The **framework** (fact layer, registry, promotion targets, seed-into-search plumbing) is
backend-generic and shared; the **models** are not. On Triton, the reduction and pointwise
heuristics are analytical and cover two arches (sm90 + sm100) with the arch entering as
constants on a subclass, not as a new code path — that is genuine, and PR #2997 / #3297 both
freeze-verified the sm90 emit byte-identical (0/40 seed rows changed under an sm90 spoof) while
re-tuning sm100. But there is currently **no pointwise heuristic on CuTe or Pallas**, and the
CuTe reduction seeds are a separate, far simpler family. The transferable claim is the *shape*
of the design (Stage-1 facts + one budget), not that the constants or the code carry across
backends.

---

## 6. VERIFIED WORKED EXAMPLES (recomputed by hand from recorded facts)

### 6.1 Pointwise — the model reproduces the emitted config exactly

Facts and seeds below are read from
`/home/dev/local/wt-b200-pointwise-audit/perf-repro/b200-pointwise-audit/results/full/summary.json`
(`rows[*].pointwise_fact`, `rows[*].seed_config`). I recomputed the seed from the fact using
`num_sm=148`, `warp_slots=64`, `TILE_BYTES_SM100=16384`, `REGISTER_BYTES=65536`,
`MIN_WAVES_SM100=4`, `BLOCK_FLOOR=256`. All five match.

| kernel / shape | fact (`slab`,`store`,`comp`,`gather`,`sfu`,`max_op_slab`) | derived caps | emitted seed | matches |
|---|---|---|---|---|
| `swiglu [32768,1536]` | 3, 2, 4, 1, 1, 1 | budget 2730, reg 5461, occ 85019 → 2730 | `block_sizes=[2048]`, no `num_warps` | ✔ (`work=4`, `resident_wave=1` → 4 = DEFAULT_WARPS → key omitted) |
| `silu_and_mul_interleaved [192,4096,True]` | 5, 2, 4, **2**, 1, 1 | fetch=20 → budget **819**, reg 3276, occ 664 → 664 | `block_sizes=[1,512]`, **`num_warps=8`** | ✔ (`E=4` band → `work=4`; `programs=768`, `148*48//768=9 → 8`; `cap=8`) |
| `silu_and_mul_interleaved [98304,6144,False]` | same | budget 819, occ 510118 → 819 | `[1,512]`, no `num_warps` | ✔ (`programs=589824` → `resident_wave=1` → 4) |
| `silu_mul_fp8 [1,2048]` | 3, 2, 4, 1, 2, 1 | budget 2730, reg 5461, **occ 3** | `[1,256]` | ✔ — the **`BLOCK_FLOOR=256` anti-undershoot floor overrides `occ_cap`**, exactly as the comment says |
| `rope [1,32,2048,256]` | **33280**, 2, 4, 1, 0, **8192** | fetch 66560 → budget **1**, reg **1**, occ 3 → 1 | `[1,1]`, **`num_warps=2`** | ✔ — heavy slab drives both caps to 1; `lanes = 1 × 8192` keeps 2 warps alive via `max_op_slab_numel` |
| `rope [1,32,8192,256]` | same slab | → target 1 | `[1,1]`, **`num_warps=1`** | ✔ — `programs=8192` → `resident_wave=1`, `work=1` |

The `rope` pair is the cleanest single demonstration that this is a model and not a table: the
*same kernel* at two sequence lengths gets two different warp counts, purely from the residency
term.

### 6.2 Reduction — the emitted configs trace to the formula

From
`/home/dev/local/wt-b200-reduction-audit-run/perf-repro/b200-reduction-audit/results/full/summary.json`
(`rows[*].arms.seed.config`). 114/114 cells recorded; two heuristics fired:
`triton_reduction_tile_sm100` 69 cells, `triton_reduction_user_tile_sm100` 45; `heuristic_no_seed: []`.

Emitted-config distribution across all 114 cells (computed from the raw JSON):

- `num_warps`: **2 → 34 cells, 4 → 11, 8 → 28, 16 → 13, 32 → 28**.
- persistence: `reduction_loops` **absent** (materialized rdim / user-tiled track) 84,
  `[None]` (persistent) 21, `[8192]` 6, `[16384]` 3.
- eviction: spec-default 93, `'first'` (stream) 15, `'last'` present 6.

Selected traces I checked by hand:

| cell | seed | check |
|---|---|---|
| `softmax [4096,256] fp16` | `bs=[2] rl=[None] nw=2 ev=all 'first'` | extent 256 ≤ 1024 → narrow rung 2; `num_load==1` → stream eviction ✔ |
| `softmax [2048,32768] fp16` | `bs=[1] rl=[None] nw=4` | traffic `32768×2×1 = 65536` = exactly `NW8_MAX_ROW_TRAFFIC` (`<=`) → light → wide rung 4; persistent at 32768 elements ✔ |
| `rms_norm [2048,4096] bf16` | `bs=[1] nw=4` | traffic `4096×2×2 = 16384` light, extent > 1024 → 4; `grid_rows=2048`, `occ_widen = 2048 // (148×8) = 1` → M stays 1 ✔ |
| `rms_norm [4096,7168] bf16` | `bs=[2] nw=4` | `4096 // 1184 = 3 → prev_pow2 = 2` → M widens to 2 ✔ |
| `rms_norm [589824,256] bf16` | `bs=[16] nw=2` | `WIDEN_MAX_ROWS=8` is *exceeded* — the raised `autotuner_min` floor wins via `max(floor, ...)`, exactly as documented ✔ |
| `layer_norm [4096,12288] fp16` | `bs=[1] rl=[None] nw=8` | traffic `12288×2×3 = 73728 > 64 KiB` → heavy, extent ≤ 16384 → 8 ✔ |
| `layer_norm [1024,36864] fp16` | `bs=[1] rl=[16384] nw=32` | extent 36864 exceeds the hold ceiling → chunk capped at `LOOPED_CHUNK`; looped ⇒ `_b200_num_warps` returns `None` ⇒ base ramp gives 32 ✔ |
| `cross_entropy [2048,32000] bf16` | `bs=[1] rl=[None] nw=32` | persistent at **32000** elements (big 720 KiB bucket, since cross_entropy has no store-only re-read); extent > 16384 ⇒ base ramp 32 ✔ |
| `cross_entropy [2048,256000] bf16` | `rl=[16384] nw=32` | 256000 exceeds every ceiling → looped at the cap ✔ |
| `fused_linear_jsd [8192,32000] bf16` | `rl=[8192] nw=32` | chunk **8192**, i.e. the *byte* solve bit below `LOOPED_CHUNK` — the multi-live-tile footprint binding ✔ |
| `kl_div [8192,32768] bf16` | `bs=[4096,1]` | user-tiled, `carried_2d_count=1` → tighter 120 KiB carried budget sizes the chunk **and** `carried_kernel` pins the grid row at floor 1 ✔ |
| `jsd [8192,32768] bf16` | `bs=[2048,1]` | same, `carried_2d_count=2` → half again ✔ |
| `rms_norm_bwd / layer_norm_bwd [2048,4096] bf16` | `bs=[16,2] nw=8` | M-collapse: `next_pow2(2048 // 148) = next_pow2(13) = 16` ✔ (an exact match to the collapse formula) |
| `per_token_group_fp8_quant [1,2048,128]` | `bs=[1] nw=2 ev=['first']` | extent pinned at `group_size=128` → narrow rung 2, matching PR #3338's claim and vLLM's own 15-cells-at-nw2 table ✔ |

---

## 7. NUMBERS RE-DERIVED FROM RAW DATA (for the blog's reduction/pointwise paragraphs)

### 7.1 Definitive 114-cell B200 reduction audit
`/home/dev/local/wt-b200-reduction-audit-run/perf-repro/b200-reduction-audit/results/full/summary.json`.
`expected_cells = recorded_cells = 114`, `environment_consistent = true`, GPU `NVIDIA B200`,
`compute_capability [10,0]`, torch `2.12.0+cu130`, triton `3.7.0`, helion commit
`2d753a5c6e855638a8947a296484e187b4ea0bbf` (branch `b200-reduction-audit-run`).

Cohort geomeans (heuristic latency in the denominator; >1 = heuristic faster) — **exact match to
`PYTORCH_BLOG_HEURISTICS_RESULTS.md`**:

| Cohort | n | / default | / torch.compile | / SM100 AOT |
|---|---:|---:|---:|---:|
| general_aot | 24 | 1.0294 (min 0.766, max 1.366) | 1.1242 (0.810–1.893) | 0.9171 (0.647–1.409) |
| original | 35 | 6.0278 (1.179–40.316) | 1.0130 (0.645–1.766) | n/a |
| vllm | 54 | 1.9178 (0.514–16.606) | 1.3880 (0.762–4.314) | 0.9181 (0.434–1.048) |

Per-kernel (all 16 kernels; the six vLLM rows are **not** in the results doc and are new
material):

| kernel | cohort | n | /default | /tc | /AOT |
|---|---|---:|---:|---:|---:|
| rms_norm | general_aot | 6 | 1.075 | 1.030 | 1.008 |
| layer_norm | general_aot | 6 | 0.978 | 0.961 | 0.945 |
| softmax | general_aot | 6 | 1.096 | 1.259 | 0.975 |
| cross_entropy | general_aot | 6 | 0.974 | 1.282 | 0.762 |
| kl_div | original | 6 | 9.509 | 1.183 | n/a |
| jsd | original | 6 | 5.237 | 0.936 | n/a |
| fused_linear_jsd | original | 6 | 1.571 | 0.927 | n/a |
| grpo | original | 6 | 3.654 | 0.898 | n/a |
| rms_norm_bwd | original | 5 (1 excluded) | 12.490 | 0.989 | n/a |
| layer_norm_bwd | original | 6 | 15.173 | 1.181 | n/a |
| dynamic_per_token_scaled_fp8_quant | vllm | 9 | **2.850** | 1.030 | 0.940 |
| per_token_group_fp8_quant | vllm | 9 | **1.088** | 1.343 | 0.995 |
| rms_norm_dynamic_per_token_quant | vllm | 9 | **7.434** | 1.015 | 0.940 |
| rms_norm_per_block_quant | vllm | 9 | **2.204** | 1.387 | 0.925 |
| silu_and_mul_per_block_quant | vllm | 9 | **1.025** | 1.483 | 0.843 |
| fused_qk_norm_rope | vllm | 9 | **0.956** | 2.475 | 0.873 |

The one excluded cell: `rms_norm_bwd [2048,11008]` fails accuracy **identically** for the seed
and the default (`maxabs=0.0625, maxrel=1.067` at `rtol=atol=0.03` on output 0) — a bf16
accumulator margin, not a seed defect.

### 7.2 Broader 455-cell reduction/generalization audit
`/home/dev/local/wt-b200-perf-report-repro-plan/perf-repro/results/summary.json`. `455` verified
as `Σ n_shapes` over 61 `(corpus, kernel, dtype)` rows; 434 accuracy-pass.

**Correction to the results doc:** it says this audit spans "35 kernel/dtype rows".
`PERF_TABLES.md` has **29** `(kernel, dtype)` data rows (24 `/examples` rows = 12 kernels × 2
dtypes, plus 5 native vLLM-family rows); `summary.json`'s `per_cell` has **31** unique
`(kernel, dtype)` pairs over **18** distinct kernels and **61** `(corpus, kernel, dtype)` rows.
`welford` is in `summary.json` but absent from `PERF_TABLES.md`. 35 is not reproducible from
either file.

`headline` (order: `[G_tc, G_def, n_tc, n_def, G_vllm, n_vllm]`):
- reproduction: **1.0421 / 2.5887**, n=188/188, vLLM-tuned **0.9872** (n=16)
- generalization: **1.1019 / 2.8989**, n=252/257, vLLM-tuned **0.9549** (n=106)
- reproduction native: 1.2491 / 3.8159 (n=16), vLLM 0.9872 — the "posted-number shapes" row
- generalization native: 1.2467 / 3.1271 (n=106), vLLM 0.9562
- per_corpus `vllm_gen`: **1.1919 / 3.5227** (n=96), vLLM-tuned **0.9562** — the "full tuned-grid
  sweep" row

The per-`(kernel, dtype)` table itself (`PERF_TABLES.md`, all 29 rows, repro + out-of-sample
fused into one geomean per row; > 1 = heuristic faster):

| kernel | dtype | /default | /torch.compile | /vLLM tuned |
|---|---|---:|---:|---:|
| rms_norm | fp32 / bf16 | 1.12 / 1.04 | 1.05 / 1.11 | n/a |
| layer_norm | fp32 / bf16 | 1.12 / **0.93** | 1.05 / 1.00 | n/a |
| softmax | fp32 / bf16 | 3.84 / 4.48 | 1.17 / 1.38 | n/a |
| sum | fp32 / bf16 | 1.06 / 1.11 | 0.97 / 1.02 | n/a |
| long_sum | fp32 / bf16 | 2.79 / 2.78 | **0.91 / 0.83** | n/a |
| cross_entropy | fp32 / bf16 | 1.25 / 1.19 | **0.67** / 1.06 | n/a |
| kl_div | fp32 / bf16 | 6.17 / 7.94 | 1.07 / 1.05 | n/a |
| jsd | fp32 / bf16 | 4.31 / 4.45 | 1.04 / **0.81** | n/a |
| fused_linear_jsd | fp32 / bf16 | 1.65 / 1.33 | 1.18 / **0.88** | n/a |
| grpo | fp32 / bf16 | 3.28 / 2.45 | **0.93** / 1.11 | n/a |
| rms_norm_bwd | fp32 / bf16 | 10.63 / 13.85 | 1.15 / 1.11 | n/a |
| layer_norm_bwd | fp32 / bf16 | 14.03 / 17.40 | 1.31 / 1.21 | n/a |
| dynamic_per_token_scaled_fp8_quant | native | 4.07 | 1.02 | 0.94 |
| rms_norm_dynamic_per_token_quant | native | 13.27 | 0.98 | 0.95 |
| per_token_group_fp8_quant | native | 1.04 | 1.11 | 0.98 |
| rms_norm_per_block_quant | native | 2.09 | 1.67 | 0.98 |
| fused_qk_norm_rope | native | 1.00 | 1.92 | 0.94 |

**Two timing regimes are mixed in that table** and the report says so: "vLLM kernels are timed
as cudagraph device time (their deployment regime); the rest are eager cold-L2." Do not merge
the two halves into one number. Also `long_sum` and `cross_entropy fp32` are the audit's worst
torch.compile losses (0.83–0.91 and 0.67) and should not be omitted from a per-kernel table.

### 7.3 Definitive 36-cell B200 pointwise audit
`/home/dev/local/wt-b200-pointwise-audit/perf-repro/b200-pointwise-audit/results/full/summary.json`.
`expected 36 / recorded 36`, 35 measured, `heuristic_counts = {"triton_pointwise": 35}`,
`heuristic_no_seed = []`, `high_default_null_delta = []`. Helion commit
`61f4058f3610e1b2bbabc82df8267ac450f591df` (branch `b200-pointwise-audit-run`), torch
`2.12.0+cu132`.

Timing method (recorded per row, worth quoting for methodology credibility):
`"calibrated cold-L2 CUDA graph"`, formula
`(flush_plus_operation_graph_ms - flush_only_graph_ms) / batch`, `flush_inside_graph: true`,
`interleaved: true`, 9 rounds (15 on high spread), spread gate 0.05, batch 16, 20 iterations per
round, and the flush-only medians are recorded alongside.

Relative-to-default geomeans — **exact match to the results doc**:

| cohort | cells | heuristic | torch.compile | AOT |
|---|---:|---:|---:|---:|
| general | 17 | **19.351** (14.284–40.623) | 18.508 (14.221–44.731) | 22.145 (n=5, RoPE only) |
| vllm `silu_mul_fp8` | 9 | **1.096** (0.958–1.251) | 1.232 (0.988–1.487) | 1.130 (0.847–1.388) |
| sglang interleaved SiLU | 9 | **1.257** (1.005–1.669) | 1.420 | 1.036 |
| **overall** | 35 | 4.579 | 4.764 | 2.085 (n=23) |

The single failure is `rope [2,32,2048,256]`: `"timeout after 300s"` at the **cell** level — the
whole cell, not one arm, so no seed config was even recorded for it.

### 7.4 PR-message numbers for the reduction/pointwise stack (from `git log`, unsquashed commits)

- **#3338 / `a9d12c1e9`** (sm100 warp ramp): held-out n=59 geomean **1.1158x**; examples n=29
  **1.0938x**; vLLM n=30 **1.1376x**; 31 wins >2%, 24 neutral, 4 regressions >2%.
- **#3297 / `7956219f3`** (pointwise sm100): new-vs-landed-heuristic, B200, single process,
  interleaved, cold-L2 with flush inside the timed cudagraph, per-shape null arm:
  `silu_and_mul` 15 shapes **1.133x**, `rope` 10 **1.076x**, `swiglu` 10 **1.004x**,
  `silu_and_mul` interleaved 15 **1.002x**, `silu_mul_fp8` 10 **0.983x**; **all 60 → 1.043x**;
  19 faster beyond their noise floor, 8 slower, 33 in-noise. sm90 byte-identical.
- **#2997 / `77785db2c`** (B200 reduction subclasses, vs earlier baselines): vs torch.compile
  `softmax` bf16 1.271, `cross_entropy` bf16 1.324, `kl_div` bf16 1.199; vs Helion default
  `kl_div` bf16 **8.891**, `softmax` bf16 5.271, `jsd` bf16 4.659; vs vLLM's configs
  `rms_norm_dynamic_per_token_quant` 1.019 / vs default **7.310**. sm90 freeze verified 0/40
  seed rows.
- **#2866 / `4af3c26a3`** (pointwise seed, H100 sm90): `seeded_vs_default` `swiglu` **6.83x** (23
  shapes), `geglu` 6.84x (17), `relu_squared` 9.99x (12), `residual_add` 1.29x (16), `bias_gelu`
  1.21x (15), at torch.compile parity `G = 0.968–0.998`. Lever-specific:
  RoPE fwd `[1,32,2048,256]` default `[1,32]` = 100 GB/s (858 us) → seed `[1,1]` = 1900 GB/s
  (45 us), **19x**; `heavy_transcendental_1d` 23 SFU ops → w16 is **1.49x** over w4 at the same
  tile (0.276 vs 0.411 ms), 3.99x vs default. Held-out: 26 never-fitted shapes stay at G
  0.977–0.998, and a never-fitted kernel `dyt` (18 shapes) beats default 1.16x at parity.
- **#2828 / `81e49d555`** (liveness-driven looping): `fused_linear_jsd` vs torch.compile
  0.609 → **0.835** (bf16) and 0.857 → **1.141** (fp32); 16 of 17 kernels byte-identical.
  `n_spills 480 → 0`.

---

## 8. CAVEATS THE BLOG MUST NOT DROP

1. **The reduction/pointwise audits were run on different trees than the linear-attention
   tree.** Reduction audit: helion `2d753a5c6…` / branch `b200-reduction-audit-run`, torch
   `2.12.0+cu130`. Pointwise audit: `61f4058f3…` / `b200-pointwise-audit-run`, torch
   `2.12.0+cu132`. The code I read is `40151a23f` + uncommitted edits on
   `calebmkim/stack/50`. No behavioural drift was verified across these; do not present the
   code and the numbers as coming from one build.
2. **The `general_aot` reduction cohort is a LOSS vs AOT (0.917) and two of its four kernels are
   a loss vs everything**: `layer_norm` 0.978/0.961/0.945 and `cross_entropy` 0.974 vs default,
   0.762 vs AOT. The blog's reduction headline should be the vLLM cohort (1.918 / 1.388 / 0.918)
   and the `original` cohort (6.028 / 1.013), not "reductions win everywhere".
3. **The big `original`-cohort multiples are default-quality artifacts, not heuristic quality.**
   `rms_norm_bwd` 12.49x and `layer_norm_bwd` 15.17x over default land at 0.989 / 1.181 vs
   torch.compile. The same caveat the results doc already makes for pointwise ("The large
   general-pointwise gains primarily reflect how poor the tiny unseeded base configurations
   are") applies verbatim to the backward reductions.
4. **The pointwise `general` cohort's 19.4x is a `block_size=32` artifact.** All 17 measured
   cells are either `swiglu`/`geglu` (12 cells, every one emitting `[2048]` against a default
   `[32]` — a 64x wider tile) or RoPE (5 cells, emitting `[1,1]` or `[1,2]` against a default
   `[1,32]`). On the two cohorts where the default is already
   reasonable, the heuristic is 1.096x and 1.257x, and **torch.compile beats it on both**
   (1.232x and 1.420x).
5. **The broader 455-cell audit has 5 cells where the seed's output is WRONG while the default's
   is correct**, all `fused_qk_norm_rope` in the generalization corpus (`[16,8,512]`,
   `[32,8,512]`, `[32,8,4096]`, `[32,8,16384]`, `[64,8,64]`; "seed maxabs~2, default fine"),
   excluded from every ratio. The definitive 114-cell audit's `fused_qk_norm_rope` shapes (third
   dim 8) all pass. **Attribution matters here and the audit gets it right:**
   `perf-repro/notes/QK_NORM_ROPE_FINDING.md` shows it is a **Helion codegen bug, not a
   heuristic bug** — hand-set `block_sizes` at (q=32, kv=8, tok=512) give maxabs 0.0156 at 8/32/128
   and **1.90 / 1.59 at 16 / 64**, an alternating power-of-two miscompile that reproduces with no
   heuristic involved (suspected in-place read-after-write aliasing in the RoPE epilogue). "The
   seed heuristic merely *happens to pick block_size=16* for some shapes and thus inherits the
   wrong result." The vendored kernel body is byte-identical to upstream vLLM (72/72 lines
   verified), so it is an upstream Helion issue. None of this is mentioned in
   `PYTORCH_BLOG_HEURISTICS_RESULTS.md`; a blog that claims "the heuristic never makes a kernel
   wrong" needs this footnote, and the honest framing is "a wider tile exposed a latent codegen
   bug".
6. `fused_qk_norm_rope` is **0.956x vs the raw default** in the definitive vLLM cohort — the one
   reduction kernel where the heuristic loses to doing nothing (it wins 2.475x vs torch.compile).
7. **The pointwise AOT arm is not uniformly sm100.** `rows[*].aot_arch` in the pointwise
   `summary.json` is `sm90` for **both** `rope` (5 cells) *and* vLLM `silu_mul_fp8` (9 cells),
   and `sm100` only for the 9 sglang `silu_and_mul_interleaved` cells; `swiglu`/`geglu` have no
   AOT arm at all. `PYTORCH_BLOG_HEURISTICS_RESULTS.md` notes the RoPE/SM90 replay but omits
   that `silu_mul_fp8`'s 1.130x AOT figure is also an SM90 table replayed on B200 — so the
   "heuristic 1.096x vs AOT 1.130x" comparison for that kernel is against an off-arch config.
   (The reduction audit has no such issue: its arm is uniformly `aot_sm100`.)
8. `MAX_WARPS = 16` for pointwise: the seed will never emit 32 warps on a pointwise kernel, by
   construction. `MAX_WARPS ∈ {8,16,32}` was in the grid search and 16 won.
9. `WAVE_TARGET_NUMERATOR/DENOMINATOR = 3/4` reads like a hardware derivation in the code
   comment ("Target a fraction just under a full wave") but the WORKLOG is explicit that the
   effective `48` is a fitted argmax, not a hardware quantity.
10. **Two constants are explicitly non-portable across families and were left as compromises:**
   the pointwise `TILE_BYTES` (slab-10 families want ~10 KiB fetched, slab-6 families 12–24 KiB;
   16384 is the compromise) and the reduction `LOOPED_CHUNK` (a 16384→32768 sm100 retune is
   "non-monotonic in grid occupancy and only a 2-sided occupancy band would capture it
   (overfit-smelling)", so it was left to the autotuner — commit `2ea223f29`).

---

## 9. WHAT I COULD NOT ESTABLISH

- No pointwise or reduction result was measurable **on this box** (no NVIDIA driver in the
  sandbox), so every number here is a re-read of stored JSON, not a re-run. `num_sm=148` and
  `max_threads_per_multi_processor=2048` for B200 are asserted, not queried — though my hand
  recomputation of six pointwise cells is only self-consistent at those values, which is
  indirect confirmation.
- I could not find a `_lab/pointwise/NOTEBOOK.md` in the tree (the `TILE_BYTES` comment cites
  it); the surviving hill-climb record is `/home/dev/local/pointwise-curriculum/WORKLOG.md`.
- No equivalent hill-climb worklog was located for the **reduction** constants
  (240/120/720/288 KiB, 16384, `MIN_WAVES=8`, `WIDEN_MAX_ROWS=8`); their justification is only
  the code comments and PR messages. In particular I could not find the measurement that fixes
  `ROW_PERSIST_MAX_BYTES` at exactly 245760 or `WIDEN_MAX_ROWS` at exactly 8.
- The reduction audits report no ncu/profiler counters, so the claimed mechanisms
  ("register-pressure → occupancy cliff" for the apply-loop cap, "over-subscribed memory system"
  for the warp lever) are supported by commit prose (`faffeb7ef` cites an "ncu Gate F") but the
  underlying profiles are not in these result trees.
- `TritonMatmulReductionEpilogueHeuristic` is sm90-only and appears in **no** result file I
  read, so I cannot say whether it fires on any benchmarked kernel today.
