"""Replay FE2 (_draft + _fixup) on the recorded chunk_fwd_o_helion facts, on CPU."""

from __future__ import annotations

import json
import sys
from types import SimpleNamespace
from unittest.mock import patch

import torch

import helion
from helion._compiler.autotuner_heuristics.triton import (
    TritonB200MultiMatmulHeuristic as MULTI,
)
from helion.autotuner.config_spec import (
    DotAxes,
    DotAxisKind,
    DotSite,
    KernelGridFact,
    KernelMatmulFact,
    LiveTile,
    LoopAxisFact,
    MatmulFact,
    PipelinedRegion,
    ResidentRegion,
    ResolvedMatmulFact,
    RootGridFact,
)

ORACLE = "/home/dev/local/sm100-linattn/CANDIDATE_FINALIZER_DEDUP_CANDIDATE_ORACLE.json"


class BS(list):
    def __init__(self, specs):
        super().__init__(
            SimpleNamespace(
                block_id=b, min_size=lo, max_size=hi, autotuner_min=lo
            )
            for b, lo, hi in specs
        )
        self._ids = [b for b, _lo, _hi in specs]

    def valid_block_ids(self):
        return list(self._ids)

    def block_id_to_index(self, block_id):
        return self._ids.index(block_id)


def dtype_of(d):
    return getattr(torch, d["__repr__"].split(".")[-1])


def mk_fact(f):
    return MatmulFact(
        lhs_ndim=f["lhs_ndim"],
        rhs_ndim=f["rhs_ndim"],
        m_block_id=f["m_block_id"],
        n_block_id=f["n_block_id"],
        k_block_id=f["k_block_id"],
        static_m=f["static_m"],
        static_n=f["static_n"],
        static_k=f["static_k"],
        lhs_dtype=dtype_of(f["lhs_dtype"]),
        rhs_dtype=dtype_of(f["rhs_dtype"]),
    )


def mk_axes(a):
    kind = lambda x: DotAxisKind(x["value"])  # noqa: E731
    return DotAxes(
        kind(a["m_kind"]),
        kind(a["n_kind"]),
        kind(a["k_kind"]),
        a["m_extent"],
        a["n_extent"],
        a["k_extent"],
    )


def mk_loop_axes(la):
    return tuple(
        LoopAxisFact(
            x["block_id"],
            x["extent"],
            x.get("bounded_by_block_id"),
            x.get("bounded_extent"),
            x.get("prefix_outer_block_id"),
        )
        for x in la
    )


def mk_site(s):
    return DotSite(
        graph_id=s["graph_id"],
        updates_carry=s["updates_carry"],
        loop_axes=mk_loop_axes(s["loop_axes"]),
        exact_loop_trips=s.get("exact_loop_trips"),
        max_loop_trips=s.get("loop_trips"),
    )


def mk_tile(t):
    return LiveTile(
        dim_block_ids=tuple(t["dim_block_ids"]),
        static_dims=tuple(t["static_dims"]),
        itemsize=t["itemsize"],
        kind=t["kind"],
        stageable=t.get("stageable"),
    )


def build(case, extents, bounds):
    mf = case["matmul_fact"]
    facts = [mk_fact(f) for f in case["local_matmul_facts"]]
    axes = [mk_axes(a) for a in mf["axes"]]
    sites = [mk_site(s) for s in mf["sites"]]
    mm = KernelMatmulFact(
        matmuls=tuple(
            ResolvedMatmulFact(f, a, s) for f, a, s in zip(facts, axes, sites)
        ),
        knob_users=tuple(
            (b, tuple((i, ax) for i, ax in users)) for b, users in mf["knob_users"]
        ),
        outer_grid=mf["outer_grid"],
        sequential_loop_trips=mf["sequential_loop_trips"],
        live_tiles=tuple(mk_tile(t) for t in mf["live_tiles"]),
        live_dot_outputs=tuple(mk_tile(t) for t in mf["live_dot_outputs"]),
        live_promoted_lhs=tuple(mk_tile(t) for t in mf.get("live_promoted_lhs", ())),
        # The oracle serialized live_tile_steps as key-lists (dumper defect), so
        # reconstruct an approximation from the two usable recorded live sets.
        live_tile_steps=(
            tuple(mk_tile(t) for t in mf["live_tiles"]),
            tuple(mk_tile(t) for t in mf["live_dot_outputs"]),
        ),
        pipelined_regions=tuple(
            PipelinedRegion(
                mk_loop_axes(r["loop_axes"]), tuple(mk_tile(t) for t in r["tiles"])
            )
            for r in mf["pipelined_regions"]
        ),
        resident_regions=tuple(
            ResidentRegion(tuple(mk_tile(t) for t in r["tiles"]))
            for r in mf["resident_regions"]
        ),
        n_dot_nodes=mf["n_dot_nodes"],
        attribution_complete=mf["attribution_complete"],
    )
    grid_fact = KernelGridFact(
        roots=(RootGridFact(0, tuple(mf["grid_groups"][0])),),
        graph_to_root=tuple((g, 0) for g in sorted({s.graph_id for s in sites})),
    )
    spec = SimpleNamespace(
        matmul_facts=facts,
        kernel_matmul_fact=mm,
        kernel_grid_fact=grid_fact,
        block_sizes=BS(bounds),
        grid_block_ids=tuple(mf["grid_groups"][0]),
        l2_groupings=BS([(0, 1, 64)]),
        allowed_pid_types=("flat",),
        _base_default_config=lambda: helion.Config(
            block_sizes=[1, 16, 16], num_warps=4, num_stages=1
        ),
        _flat_fields=lambda: {
            "block_sizes": None,
            "l2_groupings": None,
            "num_warps": None,
            "num_stages": None,
            "pid_type": None,
            "num_sm_multiplier": None,
        },
        normalize=lambda raw, _fix_invalid=False: None,
        _shrink_for_numel_constraints=lambda c: None,
    )
    raiser = SimpleNamespace(
        from_config=lambda *a, **k: (_ for _ in ()).throw(RuntimeError())
    )
    env = SimpleNamespace(
        config_spec=spec,
        block_sizes=[
            SimpleNamespace(size=extents[b], block_size_source=raiser)
            for b in range(len(extents))
        ],
        device=None,
        size_hint=int,
        settings=SimpleNamespace(),
    )
    return env, mm


def main():
    cases = json.load(open(ORACLE))["cases"]
    # extents: bid0 = BHN, bid1 = DV, bid2 = D, bid3 = C
    specs = {
        "chunk_fwd_o_helion#0:B8_T1024_H8_D64": ([1024, 64, 64, 64],),
        "chunk_fwd_o_helion#7:B4_T2048_H16_D128": ([2048, 128, 128, 64],),
        "chunk_fwd_o_helion#2:B2_T16384_H16_D128": ([4096, 128, 128, 64],),
        "chunk_fwd_o_helion#9:B8_T2048_H32_D256": ([8192, 256, 256, 64],),
        "chunk_fwd_o_helion#4:B1_T8192_H96_D128": ([12288, 128, 128, 64],),
    }
    for cid, (extents,) in specs.items():
        key = f"chunk_fwd_o_helion::{cid}"
        case = cases[key]
        bounds = [
            (0, 1, extents[0]),
            (1, 16, extents[1]),
            (2, 16, extents[2]),
        ]
        env, mm = build(case, extents, bounds)
        with patch("helion.runtime.get_num_sm", return_value=148):
            props = MULTI._proposals(env, mm, num_sm=148, precondition=True)
            raw = MULTI._proposals(env, mm, num_sm=148, precondition=False)
            draft = MULTI._draft(env, mm, props, 148)
            pre_fixup = dict(draft)
            pre_fixup["block_sizes"] = list(draft["block_sizes"])
            MULTI._fixup(env, mm, draft)
            ranked = MULTI._multi_ranked(env)
        rec = {
            k: v
            for k, v in case["promoted"].items()
            if k in ("block_sizes", "num_warps", "num_stages")
        }
        print(f"--- {cid}  extents BHN/DV/D/C={extents}")
        print("  raw per-dot proposals (bm,bn,bk,nw,ns,l2):", raw)
        print("  preconditioned per-dot proposals         :", props)
        print(
            "  ranking:",
            [
                (i, MULTI._rank_key(env, mm, i, [1, extents[1], 16]))
                for i in range(len(mm.matmuls))
            ],
        )
        print("  draft pre-fixup :", {k: pre_fixup[k] for k in ("block_sizes", "num_warps", "num_stages")})
        print("  draft post-fixup:", {k: draft[k] for k in ("block_sizes", "num_warps", "num_stages")})
        print("  RECORDED promoted:", rec)
        bsz = list(draft["block_sizes"])
        with patch("helion.runtime.get_num_sm", return_value=148):
            print("  diag: launch_grid =", MULTI._launch_grid(env, bsz))
            print(
                "  diag: pipelined_loop_trips =",
                MULTI._pipelined_loop_trips(
                    env, bsz, max_loop_trips=max(1, mm.sequential_loop_trips)
                ),
            )
            print("  diag: dot work =", MULTI._candidate_dot_work(env, bsz))
            for w in (1, 2, 4):
                print(
                    f"  diag: reg_live_bytes(nw={w}) = {MULTI._register_live_bytes(env, bsz, w)}"
                    f"  capacity = {w * 32 * MULTI.REG_BYTES_PER_THREAD}"
                    f"  resident_ctas = {MULTI._estimated_resident_ctas(env, bsz, num_warps=w, smem_bytes=MULTI._kernel_smem_bytes(env, bsz, 1), grid=MULTI._launch_grid(env, bsz), num_sm=148)}"
                )
            for s in (1, 2, 3, 4, 5, 6):
                print(
                    f"  diag: smem_demands(ns={s}) = {MULTI._kernel_smem_demands(env, bsz, s)}"
                )
            print(
                "  diag: tmem_columns(strict) =",
                MULTI._candidate_tmem_columns(
                    env, bsz, include_lhs_scratch=True, resource_policy="strict"
                ),
                "budget",
                MULTI.TMEM_COLUMN_BUDGET,
            )
            print("  diag: all_dot_acc_tiles =", MULTI._all_dot_acc_tiles(env, bsz))
        print("  seeds emitted now:", len(ranked))
        for c in ranked:
            print("     ", {k: v for k, v in c.config.items() if k in ("block_sizes", "num_warps", "num_stages", "pid_type", "num_sm_multiplier")})
        print()


main()
