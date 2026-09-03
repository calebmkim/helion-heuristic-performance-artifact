# Triton Reduction Heuristic: Candidate-Resolved Liveness Trace

This document traces the promoted Triton reduction heuristic after the
candidate-resolved liveness rewrite. It follows the implementation's decision
order: build structural facts, construct one complete tile candidate, resolve
that candidate against every live step, estimate resources and useful
occupancy, adjust blocks, test narrow full-row persistence, and emit one Helion
config.

The rewrite started from local merge commit `4bf86939`, which includes
`pytorch/helion:main` commit `792b7106`. Historical timings in
`results_baseline_v1` remain pinned to their recorded reproduction commit; they
were not used as source code for this implementation.

The warp-descent revision documented here is source commit `3bf9f342`,
evaluated against frozen pre-descent commit `728fdece`.

The relevant implementation is primarily in:

- `helion/_compiler/device_ir_analysis.py`
- `helion/_compiler/device_ir.py`
- `helion/autotuner/config_spec.py`
- `helion/_compiler/autotuner_heuristics/triton.py`
- `helion/_compiler/autotuner_heuristics/__init__.py`

The tuned implementation is `TritonReductionHeuristic`, registered once for
both H100 and B200. Standard and user-tiled reductions are internal emission
branches of this class rather than separate heuristics.

## Scope

This trace covers pure Triton reduction kernels on H100 (`sm90`) and B200
(`sm100`). Both architectures currently use one shared sizing and warp policy.
Their actual SM count still enters launch-wave calculations.

The following are separate paths:

- Fused matmul plus reduction epilogues use
  `TritonMatmulReductionEpilogueHeuristic`.
- Pure matmuls are rejected by the reduction eligibility gate.
- Other GPU targets can use `TritonNarrowReductionHeuristic`, a conservative,
  unpromoted standard-reduction fallback.
- CuTe reduction heuristics are independent.

The rewrite emits exactly one seed. Internal "candidates" are temporary states
considered while building that seed, not extra autotuner seeds.

## Executive Summary

The heuristic is best understood as:

1. Record each original reduction occurrence, every relevant graph-local live
   step, surrounding loop/grid axes, and memory-coalescing-sensitive axes.
2. Reject matmuls and kernels without a statically sizeable reduction.
3. Select a primary descriptor for emission routing and the few policies that
   intrinsically need one reduction. `num_warps` is kernel-wide and is not
   owned by that descriptor.
4. Start every mutable axis at its legal floor.
5. Grow reductions, grid axes, and non-reduction loops in the existing
   priority order while the complete candidate fits a 240 KiB peak live-state
   budget. This growth phase has no fixed launch-occupancy target.
6. Draft `num_warps` from selected reduction parallelism, climb only to
   relieve excessive spill pressure, then estimate registers/thread,
   registers/CTA, resource residency, root-wise launch supply, effective
   CTAs/SM, and effective resident warps/SM.
7. Protect every already-full reduction. Halve a grid axis only for sufficient
   effective-CTA gain, and halve a serial axis only when effective-warp gain
   pays for its extra loop iterations. A stricter rule applies to the first
   split of a full grid tile.
8. For a proven re-read row, narrowly test full persistence at up to 1.5 times
   the normal live budget. Keep it only if spill risk and effective residency
   do not worsen.
9. On the stable final blocks, evaluate lower warp rungs against launch-capped
   resident warps and spill confidence. Route reduction widths to either
   `block_sizes` or `reduction_loops`, set remaining scalar/cache policies,
   and normalize the config.

The key distinction is workload structure, not kernel name:

> Wide norm-style rows retain full persistence and broad reduction
> parallelism when the model permits it. Small vLLM-style reductions can trade
> tile width for real register residency because their useful reduction
> parallelism is already limited.

## Config Knobs

The promoted heuristic actively chooses:

| Knob | Purpose |
|---|---|
| `block_sizes` | Tiles for user-written reductions, grid axes, and all tunable non-reduction loops |
| `reduction_loops` | Chunks for compiler-rolled reductions; `None` means persistent |
| `num_warps` | One kernel-wide threads/CTA choice derived from selected reduction work and live state |
| `num_stages` | Fixed to one |
| `pid_type` | Initially `flat`, then repaired if that value is illegal |
| `load_eviction_policies` | Optional per-load L2 hints for proven re-read inputs |

Fields not set by the heuristic retain their base defaults when the promoted
config is materialized.

## Core Facts and Temporary State

### Reduction Descriptor

A `ReductionDescriptor` represents one
`(original_graph_id, reduction_block_id)` occurrence. It records:

| Field | Meaning |
|---|---|
| `category` | How the reduction axis is distributed and whether it can be sized |
| `block_id`, `graph_id` | Reduction-axis identity and original graph identity |
| `size_hint` | Raw static reduction extent |
| `input_load_itemsize` | Narrowest input element width feeding this reduction |
| `row_reread` | Whether the row is consumed again by a reduction or later store |
| `reread_eviction_index` | Load-policy slot for the re-read input |
| `fixed_tile_size_hint` | Source-fixed reduction width for `FIXED_TILE` |

The descriptor no longer contains `carried_2d_count`. A carried tensor's real
shape, item width, multiplicity, and lifetime are represented directly by
`LiveTile` values.

### Live Tile

A `LiveTile` describes one logical tensor value resident at one graph step:

| Field | Meaning |
|---|---|
| `dim_block_ids` | Block ID controlling each dynamic tile dimension; `None` for a static dimension |
| `static_dims` | Known extent for each `None` dimension |
| `itemsize` | This value's actual element width |
| `kind` | `load`, `carry`, `dot_out`, `other`, or `global` |
| `stageable` | Whether a loop-body load varies with its enclosing loop |

`global` denotes a host/global tensor handle, not the tensor's entire HBM
payload. It remains useful structural information but contributes zero
register-resident bytes.

### Reduction Candidate

`_ReductionCandidate` is one complete temporary seed state:

```text
block_sizes:
    every ordinary Config.block_sizes slot

reduction_widths:
    block_id -> selected per-program reduction width
    includes rolled, user-tiled, fixed, and materialized reductions
```

Keeping reduction widths separately matters because a compiler-rolled
reduction has no ordinary `block_sizes` slot. Candidate widths remain concrete
integers even after the persistence probe: selecting the normalized full extent
is itself the persistent state. Final emission converts that width to `None`
for a compiler-rolled reduction and leaves it as the full integer tile for a
user-tiled reduction.

### Resource Estimate

`_ReductionResources` stores only the independent values resolved for one
candidate and warp count:

```text
peak_live_bytes
num_warps
launch_waves per independent root
```

Named helpers derive estimated registers/thread, spill pressure,
resource-resident CTAs/SM, and effective resident CTAs/SM whenever policy needs
them. They are not stored, so they cannot become inconsistent with the
inputs above. Intermediate thread and allocated-register counts remain local
to those calculations. The current reduction model charges logical live state
to registers and assumes zero shared-memory usage; reductions do not use a
TMEM term. Backend tensor-numel legality remains owned by the generic config
constraints rather than the reduction resource estimate.

## Stage 1: Build Structural Facts

`DeviceIR.build_reduction_kernel_fact` runs before a reduction seed is chosen.
It produces one config-independent `ReductionKernelFact`.

### 1.1 Record Original Reduction Occurrences

The analysis walks original pre-rolling device graphs. It:

- excludes `ReductionLoopGraphInfo` copies created by candidate-dependent
  reduction rolling;
- creates a descriptor for each distinct `(graph_id, block_id)` occurrence;
- keeps the same reduction axis in sequential graphs as separate occurrences;
- deduplicates repeated uses of one axis inside one original graph.

This preserves semantic routing information without pretending that all
occurrences are simultaneously live.

### 1.2 Categorize Each Reduction

Each occurrence receives one `ReductionCategory`:

| Category | Structural meaning | Sizing treatment |
|---|---|---|
| `FULL_SLICE` | One program sees the full reduction axis, usually through a full slice | sizeable |
| `FULL_GRID` | The axis is on the grid, but its fixed block equals the full extent | sizeable and fixed |
| `USER_TILE` | A tunable sequential `hl.tile` loop owns the reduction width | sizeable and mutable |
| `FIXED_TILE` | A sequential `hl.tile` has a source-fixed block smaller than the logical extent | sizeable but immutable |
| `GRID_TILE` | Different programs own different pieces of the reduction axis | not sized by this heuristic |
| `DECLINED` | The logical extent cannot be sized statically | not sizeable |

`FULL_SLICE`, `FULL_GRID`, `USER_TILE`, and `FIXED_TILE` are in
`SIZED_REDUCTION_CATEGORIES`.

`FULL_GRID` is not another name for "dynamic." The classifier first proves
that the extent is statically sizeable, then proves that the fixed grid block
covers that extent. `DECLINED` covers unresolved shapes such as `AutoSize`,
`None`, or genuinely unbacked symbolic extents.

For `FIXED_TILE`, `size_hint` remains the logical iteration extent while
`fixed_tile_size_hint` records the actual per-iteration width. The fixed width
participates in liveness accounting but does not create an emitted tuning
knob.

### 1.3 Build the Complete Live-Step Timeline

`GraphAnalysis` computes ordinary dataflow liveness at each original graph
step. For each tensor-producing node it records a `LiveTile`, tracks the last
use of every value, and emits the set of values simultaneously live at each
step. Duplicate live sets are removed.

The kernel timeline then flattens these graph-local timelines while keeping
the steps distinct:

```text
kernel_live_tile_steps =
    graph_0_step_0
    graph_0_step_1
    ...
    graph_1_step_0
    ...
```

No bytes are summed across those separate entries. Candidate peak bytes are
computed later with a `max`.

This gives the intended control-flow semantics:

- sequential graphs are compared, not added;
- mutually exclusive branch graphs are compared, not added;
- loop-body placeholders carry values from enclosing loops into every body
  step where those values remain live;
- a loop-carried accumulator appears with its real dimensions and item width;
- multiple simultaneously live values remain separate and are summed only
  within their actual step.

There is no structural "peak tile" chosen before block sizes are known.
Consequently there is no reduction `CoResidencyGroup`, no rank-profile proxy,
and no rule approximating one group per graph. The complete candidate decides
which step is largest.

### 1.4 Identify Coalescing-Sensitive Axes

The fact builder inspects memory operations. An axis is marked
coalescing-sensitive when it supplies a unit-stride, unit-scale affine
subscript dimension.

Conceptually:

```text
for memory_op in kernel:
    for indexed_axis in memory_op:
        if stride == 1 and gather_scale == 1:
            mark axis as coalescing_sensitive
```

A plain `:` slice may not carry an explicit subscript block ID. When that
slice feeds exactly one reduction axis, reduction dataflow supplies the
otherwise missing identity. This lets ordinary contiguous row reductions
receive the same protection as explicitly tiled accesses.

This classification is structural. It does not claim that 128 elements is a
universal mathematical condition for coalescing.

### 1.5 Record Remaining Loop and Grid Axes

The kernel fact also records:

- `non_reduction_loop_block_ids`: every tunable non-grid loop that is not a
  sized reduction, including normalize/apply and independent loops;
- `grid_axis_block_ids`: parallel grid axes that are not themselves sized
  reduction axes.

Every tunable axis can therefore participate in one candidate model. Extent
equality with a reduction is no longer used to decide whether an ordinary loop
receives "apply" treatment.

## Stage 2: Check Eligibility and Select Emission Routing

### 2.1 Common Eligibility

The tuned reduction path requires:

- no `matmul_facts`;
- a `ReductionKernelFact`;
- at least one descriptor in `SIZED_REDUCTION_CATEGORIES`.

A kernel with only `GRID_TILE` or `DECLINED` occurrences declines. The tuned
class additionally requires either the SM90 or SM100 hardware target.

### 2.2 Select the Primary Descriptor

The heuristic first considers statically backed sized descriptors. If none are
backed, it falls back to all sized descriptors. It chooses the maximum key:

```text
(
    size_hint * max(1, input_load_itemsize),
    size_hint,
)
```

The primary descriptor owns:

- standard versus user-tiled emission routing;
- the identity of the one reduction eligible for the late persistence probe;
- row-reread and load-eviction scalar policy.

It does not own `num_warps`. Warps are selected from the unified candidate's
reduction widths and complete live state.

All sized reductions are still allocated. "Primary" does not mean "only."

### 2.3 Select the Emission Track

| Primary category | Track | Reduction-width destination |
|---|---|---|
| `FULL_SLICE`, `FULL_GRID` | standard | `reduction_loops` when rolled; otherwise implicit |
| `USER_TILE`, `FIXED_TILE` | user-tiled | `block_sizes` when mutable; otherwise source-fixed |

The two emission branches share the complete allocator and warp selector. They
differ only in final knob routing and load-eviction policy.

## Stage 3: Establish the Legal Candidate Space

### 3.1 Normalize Extents

The heuristic retains two forms of an extent:

- raw extent, used for semantic comparisons and backend limits;
- next-power-of-two extent, used as the candidate maximum.

For example:

```text
raw extent        = 11008
candidate maximum = 16384
```

Actual bytes are computed with each live value's own `itemsize`; there is no
kernel-wide byte scale copied from the primary descriptor.

### 3.2 Establish Floors

Each ordinary tunable block-size axis starts at:

```text
max(1, min_size, autotuner_min)
```

Reduction widths start as follows:

| Reduction form | Initial width | Maximum |
|---|---:|---:|
| `FIXED_TILE` | next power of two of `fixed_tile_size_hint` | same fixed width |
| Mutable `block_sizes` reduction | its legal block floor | normalized logical extent |
| Compiler-rolled reduction | `min(8, normalized extent)` | normalized logical extent |
| Materialized/fixed full reduction | normalized logical extent | same full width |

All ordinary `block_sizes` entries begin at their legal floors. A pinned
`hl.grid(..., block_size=1)` axis has no mutable slot and resolves through its
fixed compiler source.

### 3.3 Resolve a Candidate Block Map

Every candidate evaluation constructs:

```text
block_id -> resident tile/chunk width used by one program
```

Mutable slots come from `candidate.block_sizes`. Reduction-axis entries are
overridden by `candidate.reduction_widths`, including compiler-rolled axes.
Remaining fixed axes are resolved through the compiler's own
`BlockSizeSource`.

For a grid or ordinary tiled axis, this width is the number of elements one
program handles at once. For a compiler-rolled reduction, it is the resident
chunk for one loop iteration; the program may process several such chunks
before completing the full reduction.

This is important for correctness:

- a tunable loop uses its selected config entry;
- a fixed `hl.grid` axis can remain one row per CTA even when its global extent
  is thousands of rows;
- a rolled reduction uses its candidate chunk rather than its global extent.

## Stage 4: Build a Maximal Normal Candidate

### 4.1 Normal Capacity

The shared normal capacity is:

```text
NORMAL_LIVE_BUDGET = 240 * 1024 bytes
```

For one candidate:

```text
resolved_tile_bytes(tile, candidate) =
    tile.itemsize
    * product(
        candidate_width(block_id) if block_id is not None
        else static_extent
      )

step_live_bytes(candidate) =
    sum(
        resolved_tile_bytes(tile, candidate)
        for tile in step
        if tile.kind != "global"
    )

peak_live_bytes(candidate) =
    max(step_live_bytes(candidate) for step in live_steps)
```

### 4.2 Growth Order

The first implementation retains the prior high-level priority:

1. reduction axes;
2. grid axes;
3. non-reduction loops;
4. any otherwise unclassified mutable block-size axes.

Reduction order is:

1. fixed reductions, which are already seated;
2. full-extent categories;
3. user-tiled reductions;
4. larger raw extents before smaller extents within a tier.

For each mutable axis, `grow` repeatedly doubles its width up to the normalized
maximum.

### 4.3 Growth Acceptance

A doubled candidate is accepted when:

```text
trial_peak_live_bytes <= 240 KiB
```

There is one important neutral-growth allowance:

```text
if current candidate is already over 240 KiB:
    accept growth when trial_peak <= current_peak
```

Some immutable full-width state can put the initial candidate over the
calibrated normal budget. An axis absent from the peak live step should still
be allowed to grow because it adds useful work without worsening that peak.
This is particularly relevant to reduced-away grid axes and separate
sequential regions.

### 4.4 Record Grid Launch Supply Without Capping Growth

Grid-axis growth changes CTA count. The allocator records that launch supply
for later resource comparisons:

For every independently executing root:

```text
root_ctas =
    product(ceil(global_extent(axis) / candidate_width(axis)))

launch_waves =
    root_ctas / num_sms
```

The value remains fractional. `1.2` waves does not mean that every SM can hold
two CTAs simultaneously.

Stage 4 does not stop growth at a fixed number of CTAs/SM. It forms the
maximal budget-fitting candidate first. Stage 6 may then shrink a grid axis
when the resulting increase in effective CTAs/SM is large enough to justify
the smaller tile.

If a root extent is dynamic, its launch supply is unknown. The allocator does
not reject growth or invent a static launch count.

## Stage 5: Draft Warps and Estimate Resources

Every candidate considered during block sizing receives the same deterministic
width draft and upward spill-relief adjustment. Downward warp selection is
deferred until blocks are stable in Stage 8 so a warp-only proposal cannot
manufacture a reason to retile the grid.

### 5.1 Draft Warps From Selected Reduction Work

The draft uses the maximum selected reduction width:

```text
parallel_width =
    max(candidate reduction widths)
```

The calibrated ladder is:

| Selected reduction width | Draft `num_warps` |
|---:|---:|
| `<= 1,024` | 4 |
| `1,025 .. 4,096` | 8 |
| `4,097 .. 16,384` | 16 |
| `> 16,384` | 32 |

The draft deliberately does not use the largest whole live tile. In the
in-sample diagnosis, a large outer-row tile around a 2,048-element reduction
incorrectly looked equivalent to an 8,192-element lane-parallel reduction.
Those kernels need different warp counts.

There is one structural wide-reduction floor:

```text
if primary raw extent > 16384
and selected parallel width >= 8192:
    draft_num_warps = 32
```

This protects broad reduction parallelism for very wide rows. It is one reason
wide RMSNorm/LayerNorm-style work is treated differently from small vLLM token
reductions.

### 5.2 Convert Peak Live Bytes to Register Pressure

For `W` warps:

```text
threads_per_cta = 32 * W

logical_registers_per_thread =
    ceil(peak_live_bytes / (threads_per_cta * 4))

estimated_registers_per_thread =
    logical_registers_per_thread + 10
```

The estimate then rounds up to an eight-register allocation granularity and
caps the allocated value at 255 registers/thread for the resource calculation:

```text
allocated_registers_per_thread =
    min(
        255,
        round_up(estimated_registers_per_thread, 8),
    )

allocated_register_bytes_per_cta =
    allocated_registers_per_thread
    * threads_per_cta
    * 4
```

The uncapped estimate defines spill pressure:

```text
spill_pressure =
    estimated_registers_per_thread / 255
```

Logical liveness overpredicted ptxas registers on several inspected kernels.
The tuned excessive-spill threshold is:

```text
SPILL_PRESSURE_THRESHOLD = 1.0
```

This is a confidence threshold, not a claim that hardware permits more than
255 registers/thread.

### 5.3 Spill-Relief Warp Climb

Starting from the draft:

```text
while num_warps < 32
and spill_pressure > 1.0:
    num_warps *= 2
```

The draft/climb ladder is `4, 8, 16, 32`. More warps split logical live
elements across more threads and can reduce estimated registers/thread. The
climb stops at the first acceptable rung because more warps also increase
threads/CTA and register allocation per CTA. Stage 8 may later descend through
the complete legal ladder `1, 2, 4, 8, 16, 32`.

### 5.4 Resource-Resident CTAs/SM

The current shared hardware constants are:

```text
register file       = 65,536 32-bit registers/SM
max threads         = 2,048 threads/SM
max CTAs            = 32 CTAs/SM
shared memory term  = 0 bytes/CTA for this estimate
```

Then:

```text
register_limit =
    floor(register_file_bytes / allocated_register_bytes_per_cta)

resource_resident_ctas_per_sm =
    max(
        1,
        min(
            32,
            floor(2048 / threads_per_cta),
            register_limit,
        ),
    )
```

This is resource capacity assuming the launch supplies enough independent
CTAs.

### 5.5 Launch and Effective Residency

For each independent root:

```text
effective_resident_ctas_per_sm =
    resource_resident_ctas_per_sm              if launch waves are unknown
    min(resource_resident_ctas_per_sm, waves)  otherwise
```

The allocator's CTA score is uncapped and conservative across roots:

```text
effective_cta_score =
    min(root_effective_ctas for each root)
```

Serial-axis decisions use effective resident warps because changing a serial
chunk can also change `num_warps`:

```text
effective_warp_score =
    min(
        min(64, root_effective_ctas * num_warps)
        for each root
    )
```

Queued launch waves do not increase the score after resource residency is
filled. The `64` cap is the hardware thread-slot limit expressed in warps.

## Stage 6: Adjust Blocks for Useful Occupancy

After maximal growth, the allocator considers smaller blocks. This is the
replacement for the old nonresident-grid, carried-2D, and ordinary-grid
branches.

### 6.1 Protect Every Full Reduction

A reduction axis is protected from occupancy shrinking when:

```text
selected width == normalized full extent
```

This prevents a modeled occupancy gain from introducing a new serial reduction
loop after Stage 4 has already selected the full row. The protection does not
depend on whether immutable state placed the complete candidate above the
normal 240 KiB budget.

Fixed reductions are immutable. A partial compiler-rolled reduction is
adjustable like any other partial serial reduction; if it reaches full width,
it receives the same protection.

### 6.2 Adjustment Order and Floors

Mutable neighbors are considered in this order:

1. grid axes;
2. non-grid axes that are not coalescing-sensitive;
3. coalescing-sensitive axes.

Each neighbor normally halves one axis:

```text
trial_width = max(adjustment_floor, current_width / 2)
```

For a serial non-grid loop whose current width exceeds its raw extent, the
allocator skips intermediate widths that do not reduce padded element work:

```text
scheduled(width) = ceil(raw_extent / width) * width

while trial_width > adjustment_floor
and scheduled(trial_width) >= scheduled(current_width):
    trial_width = max(adjustment_floor, trial_width / 2)
```

For example, a 5,120-element apply loop jumps from 8,192 past 4,096, because
both schedule 8,192 element slots, and first tests 2,048, which schedules
6,144. Full reductions remain protected by Section 6.1; this rule targets
separate tunable serial loops.

For a coalescing-sensitive non-grid axis:

```text
adjustment_floor =
    max(legal_floor, min(normalized_extent, 128))
```

Other axes use their legal floor. The 128-element floor is a calibrated
protection against destroying contiguous/vectorized work; it is not a hardware
coalescing theorem.

### 6.3 Grid-Axis Acceptance

A grid-axis shrink creates more CTAs. Both candidates are resolved from
scratch, including their width-drafted and spill-adjusted warp counts, and the
trial is accepted when:

```text
cta_gain =
    trial_effective_cta_score / current_effective_cta_score

required_gain = 1.5

if current_width == normalized full grid extent:
    required_gain = 2.0

accept when cta_gain >= required_gain
```

The full-to-half transition is deliberately stickier because it introduces
the first duplicated CTA for each other grid coordinate. `2.0` is also the
largest viable threshold for a simple halving: launch supply cannot increase
by more than two. Later splits use `1.5`.

### 6.4 Serial-Axis Acceptance

Reduction chunks and ordinary loop chunks do not create grid CTAs. Their
comparison therefore uses effective resident warps and the actual increase in
serial loop steps:

```text
current_steps = ceil(raw_extent / current_width)
trial_steps = ceil(raw_extent / trial_width)

warp_gain =
    trial_effective_warp_score / current_effective_warp_score

serial_cost =
    1 + 1.1 * (trial_steps / current_steps - 1)

accept when:
    trial_effective_warp_score > current_effective_warp_score
    and warp_gain >= serial_cost
```

For a simple doubling of loop iterations, the required effective-warp gain is
`2.1x`. This charges extra iterations, synchronization, rereads, and lost
reduction parallelism without pretending to predict exact latency.

A narrow padding-relief path also accepts the jumped serial-loop candidate
when:

```text
current_width > raw_extent
trial scheduled elements < current scheduled elements
trial effective resident warps >= current effective resident warps
```

This permits removing masked power-of-two tail work without claiming a
residency gain. Exact-fit widths retain the normal serial exchange.

### 6.5 Spill-Relief Exception

A grid or serial shrink may bypass its normal exchange only when it crosses
the tuned spill-confidence threshold without reducing effective resident
warps:

```text
current spill pressure > 1.0
trial spill pressure <= 1.0
trial effective warp score >= current effective warp score
```

Merely reducing an estimate that remains above the threshold is insufficient.

### 6.6 Iterate to Stability

Accepted changes are strict block halvings, so the candidate is monotone and
cannot cycle. The allocator stops when no axis changes in a full pass. As a
defensive bound, it runs at most 20 passes; this permits a `2^20` reduction in
any one axis before retaining the current candidate. The late persistence
probe runs only after this normal adjustment finishes.

## Stage 7: Narrow Full-Row Persistence Probe

Persistence is not a second broad candidate search. It is a one-axis probe for
the primary reduction.

### 7.1 Eligibility

The probe requires:

- `primary.row_reread` is true;
- the primary reduction has a legal rolled or user-tiled width knob;
- the stable normal width is smaller than the normalized full extent.

All non-primary block sizes remain unchanged.

### 7.2 Construct and Re-evaluate the Trial

```text
persistent_trial =
    copy(normal_candidate)
    replace primary reduction width with normalized full extent

persistent_num_warps =
    draft_and_raise_num_warps(persistent_trial)
```

Both normal and persistent candidates are resolved again against every live
step.

### 7.3 Acceptance

The relaxed persistence budget is:

```text
1.5 * 240 KiB = 360 KiB
```

The persistent candidate is kept only when all conditions hold:

```text
persistent peak live bytes <= 360 KiB
persistent spill pressure <= 1.0
effective resident CTAs do not decrease on any root
```

There is no kernel-name branch and no `carried_2d_count` exclusion. A carried
row naturally fails or passes based on its resolved live values.

The rule encodes the intended tradeoff: a proven reread justifies spending up
to 1.5 times the normal register budget when persistence does not lose modeled
occupancy. It does not shrink unrelated axes to rescue persistence.

Wide norm-style rows benefit in two ways:

- a full normal reduction that already fits is protected before this probe;
- a re-read full-row trial can use the relaxed budget while retaining wide
  reduction parallelism.

## Stage 8: Finalize Scalar and Cache Policy

### 8.1 Select Final Warps Bidirectionally

The stable final candidate first repeats the Stage 5 width draft and upward
spill-relief climb. It then evaluates every lower rung in:

```text
1, 2, 4, 8, 16, 32
```

The draft's launch-capped effective resident-warp score establishes the useful
target:

```text
if peak_live_bytes >= 8 KiB:
    target = min(32, draft_effective_warps)
else:
    target = draft_effective_warps
```

Light CTAs therefore must preserve the full modeled warp supply; multiplying
tiny CTAs is not treated as a benefit by itself. Candidates with meaningful
live state may trade excess modeled supply above 32 resident warps for fewer
threads/CTA.

Each lower rung must satisfy:

```text
trial spill pressure <= 1.0
trial effective resident warps >= target
```

The scan proceeds from the lowest rung upward and keeps the first passing
value. Launch supply remains part of effective residency, so a root cannot
claim resource capacity that its grid cannot fill.

One bounded correction accounts for measured overprediction of lower-warp
register use:

```text
if draft warps >= 16
and selected parallel width <= 8192
and trial spill pressure <= 0.50:
    residency-only register estimate *= 0.70
```

The unscaled estimate still controls spill rejection. The correction is not
used during block sizing, and it does not apply to wider reductions or
low-confidence spill estimates. This was sufficient to distinguish KL's
profitable `32 -> 16` transition from JSD, which remains one-resource-CTA
resident at 16 warps.

The returned `num_warps` is shared by every reduction occurrence and both
emission tracks.

### 8.2 Fixed Scalar Policy

Both tuned tracks request:

```text
num_stages = 1
pid_type = "flat"
```

If `flat` is illegal for the live config spec, materialization substitutes the
first legal persistent PID type.

### 8.3 Standard-Track Load Eviction

For the primary descriptor:

- a proven reread gets `"last"` on its recorded slot only when a
  nonresident grid axis is already batched above one at its legal/fixed floor;
- otherwise, the base eviction defaults remain.

The floor-batched condition intentionally preserves the existing cache policy.
Heuristic growth of a reduced-away grid axis does not by itself newly enable
`"last"`. Total load count is not part of the descriptor or cache policy.

### 8.4 User-Tiled Load Eviction

If the primary row is re-read, or the kernel has a separate non-reduction
loop, the recorded re-read slot becomes `"last"` and other slots become
`"first"`. If no valid re-read slot exists, the policy is left unset.

## Stage 9A: Emit the Standard Config

The standard path writes:

```text
block_sizes
reduction_loops
num_warps
num_stages
pid_type
optional load_eviction_policies
```

For reduction loops:

- a materialized full reduction has no `reduction_loops` entry;
- a rolled persistent reduction emits `None`;
- a rolled chunked reduction emits its selected integer width;
- multiple rolled reductions emit one value in spec order.

A selected chunk at least as large as the logical extent is normalized to
`None`, even when it reached full width during normal growth rather than the
explicit late persistence probe.

## Stage 9B: Emit the User-Tiled Config

The user-tiled path writes:

```text
block_sizes
num_warps
num_stages
pid_type
optional load_eviction_policies
```

Mutable user-written reductions receive their selected widths in ordinary
`block_sizes` slots. A `FIXED_TILE` already owns its source width and emits no
new reduction-size field. This track does not emit `reduction_loops`.

## Stage 10: Normalize, Validate, and Promote

Both tracks pass their dictionaries through `_materialize_config`:

1. Drop keys unsupported by the live `ConfigSpec`.
2. Repair an illegal `pid_type`.
3. Normalize with invalid-value repair enabled.
4. Enforce backend tensor-numel constraints.

Reduction-loop normalization also:

- raises a degenerate chunk of one to the legal loop floor;
- converts a chunk covering the full logical extent to `None`.

The reduction roller independently refuses to roll a graph containing an
associative scan because a chunked scan lacks a cross-chunk prefix carry.

The unified tuned class sets `promote_seed_to_default = True`. Therefore the
one materialized config is:

- a compiler seed when autotuning is enabled;
- the compiler default when runtime autotuning is not run.

## Representative Family Traces

### Wide RMSNorm and LayerNorm-Style Rows

1. The row reduction is generally a full-extent category.
2. Reduction growth runs before row batching and reaches as much of the
   normalized width as the live budget permits.
3. Any reduction that reaches full width is protected from occupancy
   shrinking.
4. Very wide primary extents retain the 32-warp floor once at least 8,192
   reduction elements are selected.
5. If the row is reread and normal growth did not reach full width, the narrow
   1.5x persistence probe tests the full row without changing other blocks.
6. Persistence is kept only when effective CTA residency does not fall.

The policy is intentionally willing to retain a large block because these
kernels need both persistence and parallel reduction work. Treating them like
small-token reductions would add serial passes and discard useful lanes.

### Small vLLM-Style Reductions

1. The selected reduction width often falls in the 4- or 8-warp bands.
2. Full-width protection applies whenever Stage 4 reaches the full reduction.
3. Otherwise a user-tiled or ordinary loop width may be halved only when its
   effective-warp gain pays the serial-step cost.
4. Coalescing-sensitive widths are not reduced below 128 elements by the
   adjustment stage.
5. A grid tile may be split when effective CTAs/SM double on the first split
   or improve by at least `1.5x` on a later split.
6. Once blocks are stable, final warp descent can retain 32 useful resident
   warps with fewer threads/CTA; light candidates must preserve their full
   draft score.

These kernels can prioritize register residency because increasing an already
small reduction width supplies less additional lane-parallel work.

### KL Divergence and JSD-Style Carried Reductions

1. The primary reduction is usually `USER_TILE`.
2. Carried `[M_BLOCK, R_BLOCK]` values appear directly in the body live steps.
3. Reduction growth stops when any resolved step exceeds the shared 240 KiB
   budget.
4. A mutable chunk can then shrink if the effective resident-warp gain pays
   for its additional serial iterations.
5. Grid M receives no special carried-state branch; widening it naturally
   multiplies every live value containing M and is rejected when that worsens
   capacity or a later split fails the CTA-gain threshold.
6. The final reduction width is emitted in `block_sizes`.
7. Final warp descent may use the bounded lower-warp residency correction for
   an at-most-8,192-wide chunk. KL can therefore select 16 warps when two
   calibrated CTAs retain the 32-warp target, while JSD remains at 32 when the
   same trial still predicts one CTA.

### RMSNorm and LayerNorm Backward

1. Full-width materialized feature state is immutable and contributes its real
   live bytes.
2. A grid axis absent from register-live state can grow without increasing
   peak live bytes.
3. Normal growth forms the maximal budget-fitting tile without a launch
   target.
4. Occupancy adjustment splits that tile only when effective CTAs/SM gain by
   the required ratio; extra queued CTAs receive no credit once resource
   residency is full.
5. Warps follow actual selected reduction width, not the outer live tile's
   total element count.

This directly models the previously observed case where two launch waves did
not improve one-resource-resident-CTA occupancy.

### Separate Normalize, Apply, or Independent Loops

1. Every tunable non-grid, non-reduction loop is in
   `non_reduction_loop_block_ids`, regardless of extent.
2. It grows through the same live-step capacity function as reductions and
   grid axes.
3. A separate sequential graph can use the normal capacity because its live
   step is compared by `max`, not summed with an earlier reduction region.
4. During occupancy adjustment it is ordered by actual coalescing sensitivity,
   not by whether its extent happens to match the reduction.

There is no separate loop-budget allocator.

## Architecture and Fallback Boundaries

### H100 and B200

The unified class currently applies the same policy to SM90 and SM100:

- the 240 KiB normal live budget;
- the 1.5 persistence multiplier;
- the register estimate and thresholds;
- the coalescing floor;
- the `2.0` first-grid-split and `1.5` later-grid-split thresholds;
- the serial effective-warp exchange;
- the 4/8/16/32 width draft and 1/2/4/8/16/32 final legal ladder;
- the same block growth and adjustment policy.

The architecture's actual SM count changes root launch waves. There are no
B200-specific apply-loop caps or warp overrides after the rewrite. Separate
B200 performance validation remains required; broad B200 config changes were
accepted as part of this design.

### Other Hardware

`TritonNarrowReductionHeuristic` owns standard reductions only when neither
tuned target owns the device. It proposes a conservative persistent row seed
and remains unpromoted.

It does not:

- cover user-tiled reductions;
- use the candidate-resolved allocator;
- become the no-autotune compiler default.

## Decision Trace at a Glance

| Decision | Initial state | Resource-aware correction | Emission |
|---|---|---|---|
| Reduction width | Legal floor, fixed width, or full immutable width | Grow under all-step 240 KiB capacity; protect every full width; halve partial widths only for paid effective-warp gain | Rolled loop or `block_sizes` |
| Grid width | Legal/fixed floor | Grow under live capacity; require `2.0x` effective-CTA gain for the first split and `1.5x` thereafter | `block_sizes` when mutable |
| Non-reduction loop | Legal floor | Grow under its actual live steps; skip padding-neutral halves above the raw extent, then use padding relief or the serial-step/effective-warp exchange | `block_sizes` |
| `num_warps` | 4/8/16/32 from selected reduction width | Wide-primary floor; climb for excessive modeled spill during sizing; on final blocks choose the lowest 1/2/4/8/16/32 rung retaining the useful resident-warp target | One kernel-wide value |
| Persistence | Normal grown/chunked state | Primary reread row may use 1.5x budget if spill and occupancy pass | `None` or full user tile |
| `num_stages` | One | None | One |
| `pid_type` | `flat` | Repair if illegal | Normalized value |
| Eviction | Base default | Track-specific proven-reread rules | Only when a valid slot exists |

## Constants

| Constant | Value |
|---|---:|
| Normal live budget | 240 KiB |
| Persistence budget multiplier | 1.5 |
| First full-grid split gain | 2.0 |
| Later grid split gain | 1.5 |
| Serial-loop cost alpha | 1.1 |
| Spill-confidence threshold | 1.0 |
| Heavy-candidate warp target floor | 32 resident warps |
| Light/heavy live-state boundary | 8 KiB |
| Maximum block-adjustment passes | 20 |
| High-warp residency scale | 0.70 |
| Residency-scale spill ceiling | 0.50 |
| Coalescing-sensitive floor | 128 elements |
| Register overhead estimate | 10 registers/thread |
| Register allocation granularity | 8 registers/thread |
| Width-draft ladder | 4, 8, 16, 32 |
| Final legal warp ladder | 1, 2, 4, 8, 16, 32 |

These constants were tuned only on the 114-cell in-sample curriculum before
the holdout split was opened.

## Mental Model

The shortest accurate mental model is:

> Resolve one complete tile candidate against every real live step, derive
> warps and register-limited occupancy from that candidate, and shrink work
> only when the useful residency gain pays for it. Then narrowly spend extra
> register budget to persist a proven reread row without losing occupancy,
> and finally remove excess threads while preserving useful resident warps.

No kernel-family name enters the allocator. Family behavior emerges from
reduction semantics, actual live shapes and lifetimes, memory access, grid
supply, and selected reduction parallelism.

## Known Approximations

The model is more direct than the removed structural proxies, but it remains a
heuristic:

- Logical live bytes are not exact ptxas register allocation.
- The `+10` overhead, eight-register granularity, and 1.0 spill threshold are
  calibrated approximations.
- The final high-warp `0.70` residency correction is intentionally bounded by
  selected width and spill confidence. It is not an exact ptxas predictor and
  is not used to resize blocks.
- The current resource model uses shared H100-like register/thread constants
  for both SM90 and SM100 rather than querying every architectural limit.
- Shared-memory usage is zero in the reduction estimate; a future reduction
  lowering that materially stages through shared memory would need that term.
- Grid decisions use effective CTAs while serial decisions use effective
  resident warps; both remain estimates derived from logical live bytes.
- Root launch supply is unknown for dynamic extents, so no grid-fill shrink is
  justified from a guessed count.
- The 128-element coalescing floor is element-based rather than byte-, dtype-,
  alignment-, or lane-mapping-aware.
- The maximal candidate and occupancy adjustment are greedy. Axis order can
  matter when several dimensions compete.
- Grid and serial-loop work costs are simple calibrated ratios. They do not
  predict exact reread traffic, vector instruction count, partial-output
  bytes, or finalization latency.
- Serial padding relief counts scheduled element slots and loop steps, not
  exact masked-instruction or per-iteration overhead.
- The full-row persistence probe values a proven reread only through a 1.5x
  capacity allowance and no-occupancy-loss rule; it does not estimate cache
  hit rate directly.
- `num_stages` remains fixed at one.
- The heuristic emits one seed and does not compile candidate neighbors to
  obtain exact ptxas resources during seed construction.

These approximations are the appropriate places to investigate future
general improvements. They should not be replaced with distinctions that only
name individual kernels or curriculum cells.
