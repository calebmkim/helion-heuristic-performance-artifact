# Mechanism deep-read #2: control flow / staging of the matmul + multi-matmul heuristic

Research notes for the PyTorch blog post. Everything below was re-derived from source, from
recorded fact/config oracles, and from a CPU replay of the heuristic on a real linear-attention
kernel. Verdicts are marked **VERIFIED**, **VERIFIED WITH CAVEAT**, or **STALE/REFUTED**.

## 0. Provenance, and one important tree caveat

| Thing | Path |
|---|---|
| Implementation (the two front ends) | `/home/dev/local/wt-sm100-linattn/helion/_compiler/autotuner_heuristics/triton.py` (5,827 lines) |
| Registration / promotion loop | `/home/dev/local/wt-sm100-linattn/helion/_compiler/autotuner_heuristics/__init__.py` (131 lines) |
| Structural facts consumed | `/home/dev/local/wt-sm100-linattn/helion/autotuner/config_spec.py` (`MatmulFact` L159, `DotAxisKind` L174, `DotAxes` L195, `LiveTile` L232, `LoopAxisFact` L263, `PipelinedRegion` L286, `KernelGridFact` L312, `DotSite` L352, `ResolvedMatmulFact` L385, `KernelMatmulFact` L398) |
| Old B200 shape table | `/home/dev/local/wt-sm100-linattn/helion/_compiler/autotuner_heuristics/matmul_b200.json` (9,055 bytes) |
| Trace doc under review | `/home/dev/local/sm100-linattn/MATMUL_HEURISTIC_HIGH_LEVEL_TRACE.md` |
| Unit tests (all 55 pass on CPU, no GPU needed) | `/home/dev/local/wt-sm100-linattn/test/test_matmul_heuristics.py` |
| Recorded per-kernel fact + config oracle, 432 curriculum cases | `/home/dev/local/sm100-linattn/CANDIDATE_FINALIZER_DEDUP_CANDIDATE_ORACLE.json` (helion sha `4825b14c41`) |
| Wave-guard ablations (origin of the strict guard) | `/home/dev/local/sm100-linattn/FE1_ROLE_CORRECTION_ABLATION.md`, `FE_ROLE_WAVE_GUARD_ABLATION.md` |
| Coverage / vs-default per body | `/home/dev/local/sm100-linattn/SECTION3_VS_DEFAULT.md` |
| Measured post/pre/pre-tuned latencies | `/home/dev/local/wt-sm100-linattn/matmul_heuristic_perf_results/on_corpus_pretuned/results.json` |

**Tree caveat (matters for the blog).** `/home/dev/local/wt-sm100-linattn` is on branch
`calebmkim/stack/50` at HEAD `40151a23f`, **with uncommitted modifications**:
`git diff --stat` = `triton.py +1365/-…`, `device_ir_analysis.py`, `config_spec.py`,
`test_matmul_heuristics.py` (1,713 insertions total). The uncommitted part is exactly the
**expanded multi-seed pool** (`EXPANDED_SEED_POOL`, `_optimistic_*_seed_drafts`,
`_focal_seed_configs`, `FOCAL_SEED_CAP`, `COMPILER_SEED_CAP`): none of those symbols exist at
`40151a23f` or at `ccfcfbdd8` (the sha the on-corpus "post" arm was measured at). Consequences:

* The trace doc's Stage 6 describes the **committed** seed pool and is accurate for it; it is
  **stale for the working tree**. See §3.7.
* The rank-0 (promoted, autotune-off) config is *unchanged* by the expanded pool: HEAD's `_draft`
  body is logically identical to the worktree's (`resource_policy="strict"` reproduces the old
  accounting; the only additions are the returned `_knob_roles` map and proposals being passed in).
  I verified the worktree's rank-0 reproduces the **recorded** promoted configs exactly on 5/5
  linear-attention cells (§2.6).
* `ccfcfbdd8` is *not* an ancestor of HEAD (the branch was relinearized: HEAD = 2 commits on top of
  upstream `a9d12c1e9`). So "the measured post tree" and "the current tree" are siblings, not
  ancestor/descendant. I could not prove textual identity — see Gaps.

---

## 1. The verified pipeline (blog-length, ~800 words)

**Stage 0 — registration decides who owns the kernel.** `HEURISTICS_BY_BACKEND["triton"]`
(`__init__.py:56-77`) lists the heuristics in a deliberate order, and `compiler_seed_configs`
(`__init__.py:91`) walks it: for each eligible heuristic it takes `get_seed_configs()` (a *ranked
list*) or falls back to the single `get_seed_config()`, appends the whole list to the compiler seed
pool, and — if `heuristic.should_promote(env)` (`registry.py:39`) — sets
`env.config_spec.compiler_default_config = ranked[0]`. It is a **last-promote-wins** loop, which is
why the old B200 table is registered *before* the formula and the multi-contraction front end
*after* it. The rank-0 config of the last promoting heuristic is what runs when autotuning is off.

**Stage 1 — structural facts.** Before any heuristic runs the compiler builds one
`KernelMatmulFact` (`config_spec.py:398`) for any kernel with ≥1 contraction: per-dot
`MatmulFact` (M/N/K block ids, static extents, operand dtypes), `DotAxes` (per axis:
`TUNABLE_TILED` / `FIXED_FULL_EXTENT` / `UNKNOWN`, plus the real per-program extent), `DotSite`
(graph id, `updates_carry`, enclosing `loop_axes`, exact/max trip counts), `knob_users`
(*for each tunable block id, the `(dot_index, axis)` pairs competing for it*), `outer_grid`,
`sequential_loop_trips`, four liveness views (`live_tiles`, `live_dot_outputs`,
`live_promoted_lhs`, `live_tile_steps`), `pipelined_regions` / `resident_regions`, `n_dot_nodes`,
and an `attribution_complete` post-condition flag. Nothing here is a kernel name or a pattern
match.

**Stage 2 — pick a front end.** Front end 1 (`TritonB200FormulaMatmulHeuristic`, L3299) gates on
`_generalized_static_matmul_fact` (L277): exactly one `MatmulFact`, each of M/N/K either
`TUNABLE_TILED` or `FIXED_FULL_EXTENT` with a known extent, tunable ones on *distinct* knobs, and
every remaining tunable axis an outer-grid axis. Zero tunable dot axes is still admitted (warps and
stages remain meaningful). Front end 2 (`TritonB200MultiMatmulHeuristic`, L3508) fires when FE1
does not (`is_eligible`, L3570 explicitly declines if `_generalized_static_matmul_fact` returns
non-None) and at least one dot has known M/N/K. Both decline all-fp8 dots (§3.1).

**Stage 3 — draft a tile per dot from the budget formula.** `_matmul_tile` (L2553) is the pure
`(M, N, K, itemsize, num_sm, pinned_grid) -> (bm, bn, bk, num_warps, num_stages, l2_grouping)`
roofline: power-of-two shape-clamped tile under `ACC_BUDGET`, spill-outward onto the other axis,
`SAT_TILE_BM/BN` cap once `pinned_grid >= SAT_WAVES*num_sm`, growth into `TMEM_BUDGET` (N first —
coalesced stores, B reuse), a wave-quantization shrink loop that on sm100 requires a *strict*
wave-efficiency gain, the warp ramp, then a joint `bk`/`num_stages` choice under `_smem_bytes`, then
shrink-to-fit against SMEM (4′) and TMEM (5′), then `l2_grouping = 2` iff `grid_m > 1 and
grid_m >= 3*grid_n`.

**Stage 4 — project onto the kernel's real axes.** `_projected_tile_for_dot` (L2784) replaces the
formula's private launch model with the DeviceIR one: `pinned_grid` is
`_launch_grid_for_graphs(..., collapsed_block_ids=(m_block_id, n_block_id))`, and a
`candidate_launch_grid(bm, bn)` callback is handed to `_matmul_tile` so every wave test uses the
real grid. A `FIXED_FULL_EXTENT` axis takes its extent; a tunable axis is clamped to
`[max(min_size, autotuner_min), max_size]`; a fixed-K dot gets `k_logical = k_extent *
sequential_loop_trips` so a chunked recurrence is sized against its *logical* contraction length. A
split-K site (`loop_axes[i].bounded_by_block_id is not None`) then takes the tighter
`SAT_PARTITIONED_K_BM/BN = 32/64` ceiling.

**Stage 5 — finalize one dot against the whole kernel.** `_tile_for_dot` (L2906) builds the full
block vector (`_h100_build_block_sizes`, L480), optionally applies role correction (FE1 only),
re-resolves loop trips from the *emitted* tile, runs the graded stage model when K is fixed
(`_solve_candidate_stages`, L1915 → `_graded_stage_depth`, L1726), re-solves warps
(`_solve_candidate_warps`, L1964 → `_select_num_warps`, L977), then enforces hard resources
(`_fixup_candidate_resources`, L2056).

**Stage 6 — FE2 only: rank dots and resolve shared knobs.** `_draft` (L3690) proposes for every
understandable dot (`_proposals`, L3666 → `_proposal_for`, L3627), ranks them by `_rank_key`
(L3600), then for each block-size slot: if `knob_users` is non-empty take the **highest-ranked
user's** corresponding M/N/K value (L3744); else if it is a grid axis that is no dot's M or N, pin
it to its floor (L3754); else keep the base default (L3727/L3758).

**Stage 7 — role-correct each knob.** `_apply_knob_roles` (L3416, base no-op hook at L1716) fixes
the two places where the GEMM assumption fails: a **reuse-free output knob** (`_knob_amortizes`,
L3381, is False) drops to the tensor-memory allocation floor (`TMEM_ALLOC_COLUMNS=32` for N,
`TCGEN05_MIN_BM=64` for M); a **grid knob** halves while the launch is under `0.8*num_sm` *and* the
halving strictly improves `g / ceil(g/num_sm)`.

**Stage 8 — recompute kernel-global scalars, then fix up resources.** FE2 seeds `num_warps` /
`num_stages` as the max over per-dot proposals (L3777-3778), then re-solves both from the
role-corrected vector. `_fixup_candidate_resources` loops: try `num_stages - 1` first (accepted only
if it actually reduces an over-budget demand), else halve the best shrinkable knob scored by
`(#demands brought under budget, fractional relief, tiebreak)`, re-solving warps at every trial and
once at the end.

**Stage 9 — emit.** Rank-0 is both the autotune-off default and the first autotuner seed; the rest
of the ranked list is search seeds only (`_ranked_configs`, L3002; `_multi_ranked`, L4166).

One detail worth a sentence in the blog because it is the crux of the SMEM model: `_smem_region_demands`
(L1239) charges **one demand per independently executing region** — `sum(stageable loads) * stages +
resident loads + stores` per pipelined loop body, and `stages = 1` for non-loop graphs — and keeps them
as *separate* entries so the fix-up can relieve tied peaks independently. It also distinguishes
**useful depth** from **hard launchability**: for candidate ranking, `stages` is capped by the region's
resolved loop trips (a 4-trip loop cannot use 6 buffers), while `hard_allocation=True` charges the
emitted global depth because Triton can reserve that allocation even for a shorter loop.

---

## 2. THE KEY CONCEPTUAL POINT + a fully worked example

### 2.1 Why one GEMM formula is not enough

A single-GEMM formula answers "given M, N, K and a hardware budget, what tile?". A
multi-contraction kernel breaks four of its premises at once:

1. **Knobs are shared.** One `block_size` entry is dot A's N *and* dot B's K. Whichever dot the
   code happens to size first wins by accident, so the choice has to be **ranked**
   (`TritonB200MultiMatmulHeuristic` docstring, L3508-3524).
2. **Resources are shared.** Several fp32 accumulators are live simultaneously. Sizing the tile off
   one of them under-counts tensor memory by a factor of the accumulator count. Measured, quoted
   verbatim from the class docstring: *"a three-dot chunked kernel at chunk extent 256 emits a
   config that dies with `OutOfResources: tensor memory, Required: 768, limit 512` — exactly three
   256-column accumulators against a 512-column budget. That is a correctness defect, not a missed
   optimisation."* This is why `_all_dot_acc_tiles` (L1380) reserves for **every** dot, not the
   peak-live set.
3. **A knob can be both a dot axis and the grid.** Growing it improves arithmetic intensity *and*
   shrinks occupancy. The GEMM formula only knows the first effect.
4. **State is loop-carried, and dots have very different sizes.** A dot writing a loop-carried
   accumulator holds it resident for the whole chunk walk, so *its* tile sets the kernel's
   whole-loop footprint; a small dot executed 4096 times can outweigh a large one executed once.

### 2.2 The worked example: `chunk_fwd_o_helion`

Source: `/home/dev/local/wt-sm100-linattn/examples/linear/linear_attention_engine.py:846-894`
(the intra+inter chunk output stage of chunked linear attention; used by delta-rule / GLA /
retention / vanilla linear attention).

```python
@helion.kernel()
def chunk_fwd_o_helion(q, k, v, g_cs, h, use_g=True, scale=1.0):
    BHN = q.size(0)
    C   = hl.specialize(q.size(1))       # chunk length, 64 -> FIXED, no knob
    D   = q.size(2)
    DV  = v.size(2)
    hl.specialize(D)

    out = torch.empty([BHN, C, DV], dtype=q.dtype, device=q.device)

    for tile_bhn, tile_dv in hl.tile([BHN, DV]):        # block ids 0 and 1 = the launch grid
        o_cross = hl.zeros([tile_bhn, C, tile_dv], dtype=torch.float32)   # loop-carried
        attn    = hl.zeros([tile_bhn, C, C],       dtype=torch.float32)   # loop-carried

        for tile_d in hl.tile(D):                       # block id 2 = sequential inner loop
            qt = q[tile_bhn, :, tile_d]
            kt = k[tile_bhn, :, tile_d]
            ht = h[tile_bhn, tile_d, tile_dv]
            o_cross = hl.dot(qt, ht,                     acc=o_cross)   # dot 0
            attn    = hl.dot(qt, kt.transpose(-2, -1),   acc=attn)      # dot 1
        ...                                              # causal mask + exp2 decay
        vt = v[tile_bhn, :, tile_dv]
        o = hl.dot(attn.to(vt.dtype), vt, acc=o_cross)                   # dot 2
        out[tile_bhn, :, tile_dv] = (o * scale).to(out.dtype)
```

Three contractions, **three** tunable knobs, and every pathology in one 45-line kernel.

### 2.3 What the compiler recorded (not inferred — read from the oracle)

For `chunk_fwd_o_helion#0:B8_T1024_H8_D64` (BHN=1024, C=64, D=64, DV=64, bf16):

```
knob_users            = [(1, [(dot0,'n'), (dot2,'n')]), (2, [(dot0,'k'), (dot1,'k')])]
grid_groups           = [[0, 1]]        outer_grid = 1024     sequential_loop_trips = 64
n_dot_nodes = 3       attribution_complete = True
dot0  M=C  FIXED(64)   N=blk1 TUNABLE(64)   K=blk2 TUNABLE(64)   graph 0  updates_carry=True
dot1  M=C  FIXED(64)   N=C    FIXED(64)     K=blk2 TUNABLE(64)   graph 0  updates_carry=True
dot2  M=C  FIXED(64)   N=blk1 TUNABLE(64)   K=C    FIXED(64)     graph 1  updates_carry=False
live_dot_outputs      = 3 fp32 dot_out tiles: [blk0,C,blk1], [blk0,C,blk1], [blk0,C,C]
pipelined_regions     = 1 region, loop axis blk2 (extent 64), 3 stageable loads
```

Read off that fact table, every pathology is explicit:

* **block id 2 (`tile_d`) is one knob serving two dots' K** — dot0's and dot1's.
* **block id 1 (`tile_dv`) is one knob serving two dots' N *and* a launch-grid axis.**
* **block id 0 (`tile_bhn`) is claimed by no dot** — it appears in no `knob_users` entry, only in
  `grid_groups`, so it is the "outer-parallel-only" case.
* **Two of three dots are loop-carried** (`updates_carry=True`), and the third lives in a different
  graph (outside the pipelined loop).
* **Three fp32 accumulators are simultaneously live**, one of them `[.,C,C]` — a shape no single-dot
  budget would ever charge for.

### 2.4 What the heuristic decides, step by step (CPU replay, script `/tmp/replay_fe2.py`)

I rebuilt the recorded `KernelMatmulFact` into a stub `CompileEnvironment` and ran the *current
worktree* `_proposals` / `_draft` / `_fixup` / `_multi_ranked` with `num_sm=148`. For the D=64 cell:

| Step | Value |
|---|---|
| Raw formula proposals `(bm,bn,bk,nw,ns,l2)` | dot0 `(64,64,16,1,2,1)`, dot1 `(64,64,16,1,2,1)`, dot2 `(64,64,64,1,2,1)` |
| Preconditioned (projected + candidate-real) | dot0 `(64,64,16,2,2,1)`, dot1 `(64,64,16,1,2,1)`, dot2 `(64,64,64,2,6,1)` |
| `_rank_key` = (carry, work, area) | dot0 `(1, 262144, 4096)`, dot1 `(1, 262144, 4096)`, dot2 `(0, 262144, 4096)` |
| Knob 1 (`tile_dv`) | winner is a carried dot (dot0), value **64** |
| Knob 2 (`tile_d`) | both users carried and tied → **16** either way |
| Knob 0 (`tile_bhn`) | no users, grid axis → pinned to floor **1** |
| Role correction | **no-op here** (knob 1 is a grid knob and grid=1024 ≥ `0.8*148=118`, so the shrink loop breaks immediately; knob 1 is skipped by the reuse-free rule *because* it is a grid axis) |
| `_pipelined_loop_trips` | **4** ( = D 64 / bk 16 ) |
| Stage ceiling | `min(HW_MAX_STAGES=6, max(2, 4)) = 4` |
| SMEM demands per stage count | ns1 `(7168, 17408, 17408)` … ns4 `(25600, 17408, 17408)`; all fit the per-CTA share → **ns = 4** |
| `_candidate_dot_work` | total 786,432; tcgen05-eligible 786,432 (carried 524,288 / independent 262,144) |
| Warp regime | work < `TCGEN05_DOT_WORK`(2²⁰) and < `SUBSTANTIAL_DOT_WORK`(2²⁰) → pressure branch: `_register_live_bytes(nw=1) = 49152` vs one-warp file `32640` = **1.506× > WARP1_SOFT_PRESSURE 1.2** → **nw = 2** |
| TMEM | 288 of 512 columns → fits, no shrink |
| Emitted rank-0 | **`block_sizes=[1, 64, 16], num_warps=2, num_stages=4`** |
| Recorded promoted config | **`[1, 64, 16], nw 2, ns 4`** — exact match |

The 49,152 B register figure *is* the three-accumulator problem: 2×(1·64·64·4) + 1·64·64·4 =
48 KiB of fp32 accumulator against a 31.9 KiB one-warp register file. A single-dot model would have
charged 16 KiB, measured pressure 0.5, and emitted one warp.

### 2.5 The same kernel at a bigger shape shows the occupancy-relaxed stage rule firing

`chunk_fwd_o_helion#9:B8_T2048_H32_D256` (BHN=8192, D=DV=256): the wave-saturation cap holds
`bn` at `SAT_TILE_BN=128` even though the shape would allow 256; `bk` becomes 64 (= D/`PIPE`).
Launch grid 16,384. Stage solve: `resident_ctas(nw=4) = 1 < grid_ctas_per_sm = 4`, so the
`OCCUPANCY_RELAXED_MAX_STAGES` branch computes `grid_share = 232448//4 = 58112`, finds
`grid_depth = 1` (ns2 needs 66,560 B), and sets `ceiling = min(4, max(1, 3)) = 3`; the actual share
(232,448 at 1 resident CTA) then admits **ns = 3**. Emitted `[1,128,64], nw 4, ns 3` — again an
exact match to the recorded promoted config.

### 2.6 Replay accuracy across all five measured shapes

| Cell | Replay rank-0 | Recorded promoted | post/pre | pre-tuned/post |
|---|---|---|---:|---:|
| `#0 B8_T1024_H8_D64` | `[1,64,16] nw2 ns4` | `[1,64,16] nw2 ns4` | 1.804x | 1.251x |
| `#7 B4_T2048_H16_D128` | `[1,128,32] nw2 ns3` | `[1,128,32] nw4 ns3` | 3.935x | 1.120x |
| `#2 B2_T16384_H16_D128` | `[1,128,32] nw2 ns3` | `[1,128,32] nw4 ns3` | 4.667x | 1.040x |
| `#9 B8_T2048_H32_D256` | `[1,128,64] nw4 ns3` | `[1,128,64] nw4 ns3` | 5.528x | 1.345x |
| `#4 B1_T8192_H96_D128` | `[1,128,32] nw2 ns3` | `[1,128,32] nw4 ns3` | 4.837x | 1.170x |

`block_sizes` and `num_stages` reproduce **5/5**. `num_warps` reproduces 2/5; the three mismatches
are all `2` vs `4` and are attributable to my stub, not to the heuristic: the oracle serialized
`live_tile_steps` as *key lists* (a dumper defect present in all ten oracle files) and did not
record `live_promoted_lhs` at all, so I reconstructed the per-step liveness from `live_tiles` +
`live_dot_outputs`. That reconstruction changes `_estimated_resident_ctas`, which changes the
`_warp_transition_occupancy_penalty` on the 2→4 transition, which is exactly the term that decides
2 vs 4 warps at this work level. **Do not blog the num_warps column from the replay; blog the
recorded configs.**

The `pre` arm on all five cells is `[1,16,16], nw 4, ns 1` — the raw fragment default,
`config_origin: "default_config"`, `heuristic_names: []`. That is the honest "no heuristic fired"
baseline for this kernel family, and it is why the wins are 1.8-5.5x.

### 2.7 Corpus-level version of the same point

`SECTION3_VS_DEFAULT.md` scores each body against its own hand-tuned B200 config
(`1.0` = matching hand-tuning), same-process interleaved rotated arms with a null arm, null spread
median 0.040%:

* Seed coverage went from **93 of 432 curriculum cases to 406 of 432** when the two front ends
  landed (26 `_chunk_state` cases still gated out at that commit, fixed later).
* `chunk_fwd_o_helion`, 12 cases: default **0.2065** → front end 2 **0.6689** = **3.24x**.
* Other FE2 bodies: `chunk_fwd_o_diag_anchored` 3.10x, `chunk_bwd_dv` 3.07x, `chunk_bwd_dqk` 2.15x,
  `_intra_matrices_wide` 2.10x.
* It is not uniformly positive: `chunk_fwd_A_diag_anchored_varlen` 0.64x, `chunk_bwd_dk_delta`
  0.69x, `chunk_bwd_state_du_kda` 0.73x, `chunk_fwd_wy_delta_varlen` 0.80x at that commit.
* And the split by front end in the shipped on-corpus measurement
  (`on_corpus_pretuned/RESULTS.md`, 149 cells, cold-L2 rotated interleaved CUDA events, median of
  five round medians, physical GPU 1): `triton_b200_formula_matmul` **20 cells, post/pre 1.9948x**,
  20 wins >1%, 0 regressions >1%, pre-tuned/post 1.1257x; `triton_b200_multi_matmul`
  **129 cells, post/pre 1.4935x**, 91 wins >1%, **23 regressions >1%**, pre-tuned/post 1.2173x.
  The multi path is the volume case *and* the one with residual regressions — do not present it as
  uniformly better.
* The four "must not move" single-GEMM bodies — `matmul` (7 cases), `bmm` (6),
  `helion_mamba2_chunk_state_kernel` (6), `split_k_matmul` (6) — all score **1.00x**
  (default 0.9974/0.9981/0.9986/1.0007 vs Section-3 0.9998/0.9995/0.9985/0.9999): the
  multi-contraction work did not disturb the single-contraction path.

---

## 3. Claim-by-claim verification of the trace doc

### 3.1 "Fully-FP8 contractions are declined by both front ends, and why" — **VERIFIED**

`_is_fp8_matmul_fact` (L191) = *both* operands are 1-byte floating dtypes. FE1 declines at
`is_eligible` L3241 (`return not _is_fp8_matmul_fact(fact)`); FE2 declines at L3570
(`return not any(_is_fp8_matmul_fact(resolved.fact) for resolved in mm.matmuls)`).

The reason, from the FE1 comment (L3247-3277), is a **precision** issue, not perf: the budget
formula sizes fp8 GEMMs at `block_m=128`; at `block_m >= 64` Triton lowers `tl.dot` to the native
fp8 warp-group MMA reading raw fp8 from SMEM, and because Helion never passes
`max_num_imprecise_acc`, Triton uses its sm90 default of `2**30` (the "never promote" sentinel), so
the fp32 accumulator is never flushed across the K loop — *"results wrong by an error that grows
with K (~0.03% at K=512 up to ~5% at K=8192)"*. `block_m <= 32` dodges it (Triton upcasts fp8→fp16
and uses HMMA), and that is what `_base_default_config` already emits. Crucially: because these
heuristics **promote**, an eligible fp8 seed would become the `effort=none` default, which runs no
accuracy check → *silently wrong* fp8 GEMMs. In max-autotune the accuracy gate rejects the wide
tile anyway, so planting it would only waste trials. The named follow-up fix is emitting
`max_num_imprecise_acc=0` on fp8 `tl.dot`. The CuTe backend is unaffected (fp32 accumulation is
baked into the MMA op type).

*Asymmetry worth a footnote:* FE1 declines only if *the* dot is fp8; FE2 declines if **any** dot in
the kernel is fp8, so a mixed kernel with one fp8 contraction loses its seed entirely.

### 3.2 "Dots are ranked by (loop-carried state, dynamic dot work, output area)" — **VERIFIED**

`_rank_key` (L3600) returns exactly `(carry, work, area)`, higher wins:
`carry = 1 if (mm.attribution_complete and resolved.site.updates_carry) else 0`;
`work = _candidate_dot_work(env, block_sizes, indices=(index,)).total`;
`area = (static_m or m_extent or 1) * (static_n or n_extent or 1)`. Sorted `reverse=True` in
`_draft` (L3735). `_candidate_dot_work` (L1491) is `m*n*k*trips` under the candidate block map, with
trips resolved from the candidate (`exact_loop_trips` if proven, else `_resolved_loop_trips`, else
1 with an `uncertain` flag).

`test_multi_matmul_ranking_prefers_a_carried_accumulator_then_work`
(`test_matmul_heuristics.py:855`) pins all four behaviours: a carried dot beats a 64× bigger
uncarried one; with no carry anywhere work decides; a small dot with 4096 trips outranks a big
one-trip dot; and `attribution_complete=False` collapses the carry term for *every* dot equally so
ranking degrades to pure work rather than to an arbitrary order.

Worked example: dot2 (`attn @ v`, `updates_carry=False`) loses knob 1 to dot0 (`q @ h`,
`updates_carry=True`) even though both have the same 262,144 work and the same 4,096 output area —
the carry term is the only thing separating them, and it is what makes the emitted `tile_dv` come
from the dot whose accumulator lives across the whole chunk walk.

### 3.3 "An outer-parallel-only knob is pinned to its floor" — **VERIFIED**

`_draft` L3754-3757: `elif bid in grid_ids and bid not in mn_ids: value = lo` where
`lo = max(1, bs.min_size, bs.autotuner_min)` and `mn_ids` is the union of every dot's
`m_block_id`/`n_block_id`. FE1 does the same thing structurally: `_h100_build_block_sizes` (L480)
floors every axis that is not the fact's M/N/K.

On `chunk_fwd_o_helion` block id 0 (`tile_bhn`) has no `knob_users` entry and is in
`grid_groups[0]`, and the emitted value is `1` on all five cells. **Caveat:** on this kernel the
base fragment default for that slot is *also* 1, so the pin is observationally a no-op here; I
found **no** cell among the 149 on-corpus records where a grid-only knob moved from a larger base
default down to its floor. So cite the rule from code + `_draft`'s comment ("so the per-program
budget the proposals were sized under holds"), not from a measured delta.

### 3.4 "A knob claimed by no dot keeps the base default" — **VERIFIED (code)**

`_draft` L3727-3729 starts from `spec._base_default_config()`'s `block_sizes` precisely so that
*"an axis that is NO dot's M/N/K and no grid axis keeps exactly the size it has today, rather than
being pinned by a rule that was never measured on it"*, and the final `else` (L3758) leaves that
value untouched (only clamped to `[lo, max_size]`). The nearest test coverage is
`test_multi_focal_seeds_dedupe_before_the_ten_config_cap` (L1772), which asserts
`all(config["block_sizes"][3] == 128 …)` — slot 3 keeps its base default of 128 through the focal
seed builder. I found no curriculum cell that exercises the `_draft` branch itself (all 432 cases'
knobs are either dot-claimed or grid axes), so this is a code-verified, not measurement-verified,
rule.

### 3.5 "Role correction is deliberately SKIPPED in the per-dot phase of the multi path" — **VERIFIED**

Two class attributes carry it: `TritonB200FormulaMatmulHeuristic.SINGLE_ROLE_AWARE_KNOBS = True`
(L3370) and `TritonB200MultiMatmulHeuristic.SINGLE_ROLE_AWARE_KNOBS = False` (L3551). The per-dot
call site is `_tile_for_dot` L2938 (`if cls.SINGLE_ROLE_AWARE_KNOBS and mm is not None:
cls._apply_knob_roles(...)`) — so for FE2 it does not run. FE2's single call is in `_draft` L3770-3774,
gated on `ROLE_AWARE_KNOBS = True` (L3556), applied to the merged draft *before* anything derived
from the emitted tile (launch grid, stage depth, warp count) is computed;
`ROLE_KEEP_STAGES = False` (L3561) means the stage model sees the post-role tile, not the pre-role
snapshot. `test_role_correction_runs_once_per_front_end` (L1881) asserts both flags.

The stated reason is exactly the doc's: shared knobs are not yet resolved during the per-dot phase,
so correcting each dot independently would apply the same kernel-level decision more than once.

### 3.6 "FE1 shrinks N then M; FE2 shrinks the largest dot-claimed knob" — **VERIFIED WITH CAVEAT**

FE1 (`_tile_for_dot` L2963-2970) builds `shrinkable = [n_block_id, m_block_id]` filtered to
`TUNABLE_TILED` and calls `_fixup_candidate_resources(..., largest_first=False)`. FE2 (`_fixup`
L3814) passes `shrinkable=[bid for bid, _users in mm.knob_users]` with `largest_first=True`
(tiebreak key `block_sizes[slot_of[bid]]`, i.e. the largest current block).

**Caveat the doc omits:** N-then-M / largest-first is only the **tiebreak**. Every candidate halving
is scored by `relief()` = `(# demands brought under budget, summed fractional reduction)` and the
`max` of `(*impact, legacy_tiebreak)` wins (L2160-2192), so a knob that actually relieves the
binding resource beats a bigger knob that does not. Two more precise points:

* FE1's shrinkable list contains **only N and M** — the fix-up can never shrink `block_k` (that was
  already spent inside `_matmul_tile`'s `_bk_and_stages`). FE2's list contains every dot-claimed
  knob, K included.
* Stages are surrendered first, down to `MIN_NUM_STAGES = 1` (not 2), and only when the reduction
  actually relieves an over-budget demand. A TMEM overflow is *not* relievable by depth (TMEM
  columns are independent of `num_stages`, except through the `num_warps >= 4` gate at L2020), so a
  TMEM overflow always costs tile area (the only coupling is the `num_warps >= 4` gate at L2028,
  which charges 0 columns below a warpgroup).
* "Never shrink a fixed axis" holds structurally: a `FIXED_FULL_EXTENT` axis has no entry in
  `config_spec.block_sizes`, so it can never appear in `shrinkable`.
* `_finalize_seed_draft` (the seed-drafting path, L2323) uses order **N, K, M** with
  `largest_first=False`.

### 3.7 "The alternate seeds emitted (transposed M/N aspect, one-stage-shallower)" — **STALE for the working tree; VERIFIED for the committed state**

At `40151a23f` (and `ccfcfbdd8`) this is exactly right:

* FE1 `_ranked_configs` emitted rank-0 + *alt-1* transposed aspect (`bm2 = min(cap_m, 2*bm)`,
  `bn2 = max(16, bn//2)`, gated on `bm2*bn2 >= 4096` and re-checked against TMEM and kernel SMEM)
  + *alt-2* `ns - 1`, skipped at the floor `ns == 2`.
* FE2 `_multi_ranked` emitted rank-0 + one `num_stages - 1` neighbour, *"the same neighbour front
  end 1 plants: stage depth is the knob whose optimum the budget model resolves least sharply."*

Recorded oracle evidence for that state (`matmul::m8192_k4096_n11008`): seeds are
`[128,256,64] nw8 ns4` (primary), `[256,128,64] nw8 ns4` (transposed aspect), `[128,256,64] nw8
ns3` (shallower). And `chunk_fwd_o_helion#0` has exactly 2 seeds: `[1,64,16] ns4` and
`[1,64,16] ns3`.

In the **current working tree** the alt-1/alt-2 code is dead on sm100 (`EXPANDED_SEED_POOL = True`
at L3371 makes `_ranked_configs` `return dedupe_configs(ranked)` at **L3167**, before the alt-1/alt-2
block at L3169-3227; those survive only for sm90, where the flag is False). The sm100 FE1 pool is now: rank-0; the **untouched formula prior**
(projection + `ConfigSpec` clamping only, *no* role/stage/warp/resource correction, L3092-3110); two
**register-MMA** drafts (`_register_mma_seed_drafts`, L2438: M→`TCGEN05_MIN_BM`, N→
`TMEM_ALLOC_COLUMNS`, K→`DOT_MIN`, pinned at 1 and 2 warps); two **aspect** drafts (`m_heavy`
True *and* False — L2480, i.e. both transposes, not one); and three **bk/stage** drafts
(`_bk_stage_seed_draft`, L2512, with `(bk_steps, stage_steps, tile_steps, keep_tcgen05)` =
`(-1,+1,0,False)`, `(+1,-1,0,False)`, `(-1,-1,-1,True)`).
`test_b200_single_matmul_emits_the_compact_seed_families` (L1690) asserts `len(ranked) == 9` for a
4096³ bf16 GEMM.

FE2's pool is now up to `COMPILER_SEED_CAP = 20`: rank-0; a **merge-first** draft built from
*un*-preconditioned proposals; one raw focal config; two register-MMA drafts; up to two
`optimistic_{m,n}_expanded` drafts (built under `resource_policy="optimistic"`, i.e. peak-live TMEM
instead of the all-dot reservation); up to four frontier recipes
(`lighter_k_deeper_stages`, `capped_large_k_shallow_stages`, `capped_large_k_high_launch`,
`small_tile_high_launch`); a `persistent_blocked` seed (the only place `pid_type` /
`num_sm_multiplier` are ever emitted, and only when the grid exceeds `num_sm`); then per-dot focal
seeds up to `FOCAL_SEED_CAP = 10`. My replay produced 10-12 seeds per `chunk_fwd_o_helion` cell.
**A one-stage-shallower standalone neighbour no longer exists in either front end** — shallower
stages now only appear bundled with a `bk` change.

The 38-cell expanded-seed study in `PYTORCH_BLOG_HEURISTICS_RESULTS.md` §4 is a study *of this
uncommitted change*, with `40151a23f` as its "old seeds" arm. Worth stating plainly in the blog.

---

## 4. Which knobs the heuristic sets vs leaves at base defaults

**Actively set, always:** `block_sizes`, `num_warps`, `num_stages`.
**Set by FE1 only:** `l2_groupings` — and only ever to `2`, from `_matmul_tile` step (6):
`l2_grouping = 2 if grid_m > 1 and grid_m >= L2_TALL_RATIO(3) * grid_n else 1`, and only when
`allow_l2_grouping` (L2827-2833) holds, which requires *both* M and N to be tunable **grid** axes —
false for every chunked kernel whose M is a specialized chunk length. `_h100_config` emits the key
only when the value exceeds 1; the expanded-pool alternates emit the full list via
`_l2_groupings_for_site` (L2228), which preserves other roots' values and sets only the dot's own
root. FE2's rank-0 is literally `Config(block_sizes=…, num_warps=…, num_stages=…)` (L4189-4194) —
no `l2_groupings` at all; FE2's *focal* seeds do carry one, and the persistent seed carries
`pid_type="persistent_blocked", num_sm_multiplier=1`.

**Never set (left to `_base_default_config` / the autotuner):** `indexing`, `atomic_indexing`,
`pid_type` (except the one persistent seed), `loop_orders`, `flatten_loops`, `reduction_loops`,
`static_ranges`, `range_unroll_factors`, `range_warp_specializes`, `range_num_stages`,
`range_multi_buffers`, `range_flattens`, `load_eviction_policies`, `load_cache_modifiers`,
`store_cache_modifiers`, `num_sm_multiplier`, `maxnreg`, `epilogue_subtile`, and any
`register_tunable`. Field list from `ConfigSpec._flat_fields_with_flash_family`
(`config_spec.py:2812-2984`).

**A correction to a docstring:** `_h100_config` (L510) and `_extra_config_fields` (L730) advertise a
hook for *"the sm100 Blackwell levers: epilogue_subtile / indexing / range_warp_specializes"*. No
subclass overrides `_extra_config_fields` — the only definition is the base one returning `{}`
(grep confirms: definition at L730, single call site at L3029). **Nothing extra is emitted on
sm100.** The hook is aspirational.

**What that implies about the remaining search space.** Concretely, `chunk_fwd_o_helion`'s
normalized config has 14 keys; the heuristic pins 3 of them. So the heuristic is not "autotuning
with extra steps": it fixes the fields whose wrong value is catastrophic (a `[16,16,16]` tile at
`ns=1` is 1.8-5.5x slower on this kernel) and leaves the fields that are worth a few percent to the
search. That is also the honest argument for keeping the table registered: it is the only source
that plants `indexing`, `loop_orders`, `load_eviction_policies` and `l2_groupings > 2` values at all
(see §5), which is why the on-corpus `pre-tuned/post` gap of **1.2046x** (149 cells) does not close
— the full-autotune winner for `chunk_fwd_o_helion#0` is `[1,64,64] ns8 nw4` with three
`tensor_descriptor` indexing choices, `range_num_stages=[0,4]`, `range_unroll_factors=[0,1]` and
`loop_orders=[[1,0]]`, none of which the heuristic can express.

---

## 5. Status of the old B200 shape lookup table (`matmul_b200.json`)

**What it is.** A JSON rule file, **9,055 bytes**, `{"rules": [...]}` with **10 rules** carrying
**16 templates** total. Loaded once and cached (`_heuristic_rules`, L347). Consumed by
`TritonB200MatmulHeuristic` (L454), which is `promote_seed_to_default = False` and comment-labelled
**DEMOTED**: *"the general TritonB200FormulaMatmulHeuristic subsumes this table (faster on every
shape the table fires on, and covers the shapes it declines). Kept as an unpromoted search seed."*
It is registered *before* the formula in `__init__.py:62` precisely so the last-promote-wins loop
gives the compiler default to the formula.

**What it keys on.** `_shape_bucket_from_fact` (L335) builds
`{dtype: <family>, m_value, n_value, k_value}` from the fact, and `_shape_bucket_matches` (L364)
tests a rule's `m_bucket`/`n_bucket`/`k_bucket` intervals plus exact-equality on any other key.
All ten rules require `dtype == "fp16_bf16"`, so fp32 and fp8 can never match. **7 of the 10 rules
additionally pin exact `m_value`/`n_value`/`k_value`** — they are memorized single shapes
(`4096×1024×1024`, `4096×2048×2048`, `2048×4096×2048`, `1024×8192×1024`, `8192×2048×2048`,
`12288×1024×1024`, `1024×12288×1024`). The remaining three are near-diagonal buckets:
`(256,512]³`, `(512,1024]³`, `(1024,4096]³`. `k_bucket` never exceeds `(1024,4096]`, so **K ≤ 256 or
K > 4096 matches nothing**, and a non-diagonal shape (e.g. M=4096, N=512, K=512) matches nothing.

**How much of it is even reachable.** `_rules_for_bucket` (L385) sorts matches by number of
`shape_bucket` keys descending (most specific first) and `_seed_config_for_bucket` (L429) returns
**the first template of the first matching rule** and immediately `return`s. So at most one template
per rule is reachable — **10 of 16** — and any one query yields exactly one config. The gate is
`_single_2d_static_matmul_fact` (L202): exactly one `MatmulFact`, exactly 3 block sizes, both
operands 2-D, static M/N/K, and block ids exactly `(0, 1, 2)`. Batched, multi-dot, jagged, dynamic,
and any kernel with an extra axis are all declined.

**The measured argument that a formula beats a table.** Over the 432-case B200 curriculum recorded
in the oracle, the heuristics that produced a seed were:

| heuristic | cases seeded |
|---|---:|
| `triton_b200_multi_matmul` (FE2) | 279 |
| `triton_b200_formula_matmul` (FE1) | 85 |
| `triton_reduction_tile_sm100` | 68 |
| `triton_b200_matmul` (**the table**) | **1** |

The single case is `matmul::m4096_k4096_n4096`. Its recorded seed list is the clearest possible
picture of the division of labour: seed[0] is the table's rich config
`{block_sizes:[128,128,64], indexing:['pointer','pointer','pointer'], l2_groupings:[4],
load_eviction_policies:['first','first'], loop_orders:[[1,0]], num_stages:7, num_warps:4,
pid_type:'flat'}`, while the **promoted** default is the formula's `[128,256,64] nw8 ns4`. Even for
the one shape the table memorizes, it contributes search diversity, not the default.

The three arguments, each now backed:

1. **Unseen shapes.** 6 of the 7 `matmul` curriculum shapes (e.g. `m32_k4096_n4096`,
   `m128_k4096_n11008`, `m8192_k11008_n4096`) miss every rule; the formula configures all of them
   (`[32,32,256] nw2 ns4`, `[128,128,64] nw8 ns6`, `[128,256,64] l2=[2] nw8 ns4`).
2. **Unseen kernels.** The table's gate is structurally incapable of seeing a batched dot, a
   specialized axis, or a second contraction — i.e. every kernel in the linear-attention corpus. The
   formula + FE2 took seed coverage from 93/432 to 406/432 cases (`SECTION3_VS_DEFAULT.md`).
3. **No maintenance.** The table is a frozen sweep: 10 rules that only answer fp16/bf16
   near-diagonal GEMMs, 7 of them memorized single shapes, and every new dtype, arch or kernel
   family needs another sweep to extend it. The formula's re-tune surface is instead a block of
   class attributes (`SMEM_BUDGET`, `TMEM_BUDGET`, `SAT_TILE_BM/BN`, `HW_MAX_STAGES`,
   `TMEM_ALLOC_COLUMNS`, `REG_CLIMB_MAX_WARPS`, and the capability switches) — which is why
   `TritonB200FormulaMatmulHeuristic` is a subclass of the sm90 formula whose entire re-homing to
   Blackwell is ~75 lines of constants plus two method overrides (`_knob_amortizes`,
   `_apply_knob_roles`), and `TritonB200MultiMatmulHeuristic` is a further subclass of that.

---

## 6. Corroborating measurements worth quoting for the mechanism sections

* **The strict wave-utilization guard has a measured origin.** `FE1_ROLE_CORRECTION_ABLATION.md`
  applied FE2's role correction to FE1 universally: 15/432 configs changed, changed-cell geomean
  **0.9691**, and the failure mode splits cleanly by whether halving the tile actually improves
  `g / ceil(g/num_sm)` on 148 SMs — `64 → 128` programs (utilization `0.432 → 0.865`, 8 cells,
  geomean **1.0532**) versus `86 → 172` (`0.581 → 0.581`, 1 cell, **0.5805**) and `96 → 192`
  (`0.649 → 0.649`, 6 cells, **0.9446**). The worst cell, `matmul m128_k4096_n11008` at 0.5805x, is
  a case where `BM=128` already covers the whole M extent, so halving it only duplicates the B
  fetch and drops `nw8 → nw4`.
* **The guard was then implemented and retained.** `FE_ROLE_WAVE_GUARD_ABLATION.md`: 15/432 configs
  move, changed-cell geomean **1.0472** (FE1 8 cells 1.0573, FE2 7 cells 1.0356), 7 improve >1%,
  7 within 1%, 1 regresses (`chunk_fwd_h_delta_varlen` at 0.8722x — equal wave utilization is not
  sufficient to prove a bigger tile wins under varlen bounds). All-432 multiplier 1.0016, no
  `num_stages` changed as a side effect, 15/15 changed configs compile and pass correctness.
  Confirmation null spread 0.062% median / 0.262% max.
* **`_knob_amortizes` is measured on both sides** (docstring L3381-3415): reuse-free
  (`chunk_fwd_wy_delta`'s `hl.tile(D)` body) shrinking 128→32 is **1.18-1.74x faster**;
  reuse-bearing (`chunk_bwd_dqkw_delta`'s inner DV loop, which re-loads `do`/`v_new`/`dvni` per D
  tile) the same 64→32 shrink is **0.79-0.96x** and growing 64→128 is **1.05x**. Same axis position,
  same dtype, opposite sign — so the discriminator has to be the reuse, not the axis.
* **The warp-ladder stop is measured** (`_warps_for_live_set` docstring L882-970): scored against
  each cell's own measured optimum over the 53 of 75 curriculum cells where `num_warps` moves time
  ≥10% — climb-to-fit laddering to 8 warps **0.8959**, climb-to-fit stopping at a warpgroup
  **0.9591**, the hand-tuned answer key **0.9363**, a flat 1.35 overshoot tolerance **0.9433**.
  Stopping at a warpgroup scores *above* the hand-tuned configs. Also: on
  `chunk_fwd_A_diag_anchored_varlen`, 2 warps spills 38 B and holds 4 CTAs/SM while 8 warps spills
  nothing and holds 1, and 2 warps is **1.75x faster** (325 us vs 570) with thread occupancy flat.
* **Known limitation to state honestly.** The stage solver is a *feasibility* solver
  ("deepest feasible depth subject to loop length, occupancy and hard resources"), not a
  profitability model; `GRADED_SHARE_FALLBACK` documents a refuted alternative measured twice
  (0.8632 with 17 bodies ≥0.90 versus 0.8678 with 19 for the floor). 11 of the 15 bodies still
  below the 0.80 bar emit `ns=1` where hand-tuning uses `ns=2..8`, costing 0.43-0.64x per cell on
  two 18-case bodies. This is called *"the largest single characterized residual in the B200
  linear-attention curriculum, and it is a MISSING PROPERTY, not a missing constant."*

---

## 7. Errata / deltas against the trace doc (for the blog author)

1. **Stage 6 is stale** for the current tree (see §3.7). If the blog describes the seed pool, it
   must describe the expanded pool, and it should say the 38-cell expanded-seed study is a study of
   *that* change with `40151a23f` as the "old seeds" arm.
2. **"Shrinks N then M" / "shrinks the largest dot-claimed knob" are tiebreaks**, not the primary
   rule; the primary rule is measured resource relief (§3.6).
3. **`MIN_NUM_STAGES = 1`**, so "reduce `num_stages` first" can go all the way to an unpipelined
   loop. The doc's alt-2 text ("skipped at the floor `ns == 2`") refers to the *seed* perturbation
   floor, not the fix-up floor.
4. **`l2_groupings` is only ever `1` or `2`** from the formula (`L2_TALL_RATIO = 3`), and requires
   both M and N to be tunable grid axes — so it is effectively GEMM-only. The doc's knob table says
   "no later correction", which is true, but should not be read as "the formula explores L2
   grouping": the table's templates use 2/4/8/16 and the formula cannot reach past 2.
5. **The `_extra_config_fields` hook is unimplemented** on every arch (§4), so the doc's
   "Other config fields retain their normal base defaults" is exactly right and the docstring
   examples in `_h100_config` are not.
6. **FE2 declines a kernel if *any* dot is all-fp8**, which is stricter than FE1's per-dot test
   (§3.1).
7. The doc says the multi path's `num_stages`/`num_warps` come from "aggregate pipeline regions …
   and estimated resident CTAs". Precisely: the *seed* values are `max` over per-dot proposals
   (L3777-3778), and only then are both re-solved from the merged, role-corrected vector.
8. Trace-doc appendix numbers I could **not** verify from raw data: the "preconditioned result was
   up to 1.58x faster / merged-only up to 1.43x faster" GPU-1 comparison of the two FE2 scalar
   priors. The corresponding plan file is
   `/home/dev/local/sm100-linattn/KERNEL_SMEM_PRECONDITIONED_PRIOR_RESTORE_PLAN.md`; I did not open
   the per-cell JSONL behind it. Treat as unverified.

---

## 8. Reproduction

```bash
# unit tests (CPU only, no GPU in this sandbox)
PYTHONPATH=/home/dev/local/wt-sm100-linattn /home/dev/helion-env/bin/python \
  -m pytest /home/dev/local/wt-sm100-linattn/test/test_matmul_heuristics.py -q   # 55 passed

# the pure budget formula, no env needed
PYTHONPATH=/home/dev/local/wt-sm100-linattn /home/dev/helion-env/bin/python -c "
from helion._compiler.autotuner_heuristics.triton import TritonB200FormulaMatmulHeuristic as B
print(B._matmul_tile(64,64,64,2,148,1))     # (64, 64, 16, 4, 4, 1)
print(B._matmul_tile(64,128,128,2,148,1))   # (64, 64, 32, 4, 4, 1)
print(B._matmul_tile(64,256,256,2,148,1))"  # (64, 64, 64, 4, 4, 1)

# the FE2 replay on the recorded chunk_fwd_o_helion facts (script kept next to these notes)
PYTHONPATH=/home/dev/local/wt-sm100-linattn /home/dev/helion-env/bin/python \
  /home/dev/local/pytorch-blog-heuristics/research/21-replay_fe2_chunk_fwd_o.py
```

Note: **no GPU is visible in this sandbox** (`torch.cuda.is_available() == False`, no `/dev/nvidia*`,
no `nvidia-smi`), so every number above is either read from a recorded artifact or computed by a
pure classmethod with `num_sm` patched to 148.
