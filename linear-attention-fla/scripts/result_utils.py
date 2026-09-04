"""Shared result helpers for the linear-attention artifact scripts."""

from __future__ import annotations

import gzip
import json
import math
import statistics
from collections.abc import Iterable
from pathlib import Path
from typing import Any

ARM_ALIASES = {
    "default": ("default",),
    "fla_triton": ("fla_triton", "fla"),
    "seed": ("seed", "heuristic"),
    "aot_tuned": ("aot_tuned", "pre_tuned", "tuned"),
}

ARM_LABELS = {
    "default": "Default",
    "fla_triton": "FLA Triton",
    "seed": "Seed",
    "aot_tuned": "AOT-tuned",
}


def load_result(path: Path) -> dict[str, Any]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as handle:
        value = json.load(handle)
    if not isinstance(value, dict) or not isinstance(value.get("cells"), list):
        raise ValueError(f"{path} is not a combined linear-attention result")
    return value


def arm_record(cell: dict[str, Any], arm: str) -> dict[str, Any] | None:
    arms = cell.get("arms")
    if not isinstance(arms, dict):
        return None
    for name in ARM_ALIASES[arm]:
        record = arms.get(name)
        if isinstance(record, dict):
            return record
    return None


def latency_ms(cell: dict[str, Any], arm: str) -> float | None:
    record = arm_record(cell, arm)
    if record is None or record.get("status") != "ok":
        return None
    if record.get("correct") is not True:
        return None
    value = record.get("latency_ms")
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or value <= 0
    ):
        return None
    return float(value)


def geometric_mean(values: Iterable[float]) -> float | None:
    usable = [value for value in values if math.isfinite(value) and value > 0]
    return statistics.geometric_mean(usable) if usable else None


def performance_vs_fla(cell: dict[str, Any], arm: str) -> float | None:
    record = arm_record(cell, arm)
    candidate = latency_ms(cell, arm)
    if record is None or candidate is None:
        return None
    paired_fla = record.get("paired_fla_latency_ms")
    if (
        isinstance(paired_fla, (int, float))
        and not isinstance(paired_fla, bool)
        and math.isfinite(float(paired_fla))
        and paired_fla > 0
    ):
        return float(paired_fla) / candidate
    fla = latency_ms(cell, "fla_triton")
    return fla / candidate if fla is not None else None


def relative_to_fla(
    cells: Iterable[dict[str, Any]], arm: str
) -> tuple[float | None, int]:
    ratios = [
        value
        for cell in cells
        if (value := performance_vs_fla(cell, arm)) is not None
    ]
    return geometric_mean(ratios), len(ratios)


def seed_speedup(
    cells: Iterable[dict[str, Any]], baseline: str
) -> tuple[float | None, int]:
    """Compare seed with an arm after normalizing to the cell's FLA timing."""
    ratios: list[float] = []
    for cell in cells:
        seed = performance_vs_fla(cell, "seed")
        base = (
            1.0
            if baseline == "fla_triton"
            else performance_vs_fla(cell, baseline)
        )
        if seed is not None and base is not None:
            ratios.append(seed / base)
    return geometric_mean(ratios), len(ratios)


def variants(cells: Iterable[dict[str, Any]]) -> list[str]:
    return list(
        dict.fromkeys(
            str(cell["variant"])
            for cell in cells
            if isinstance(cell.get("variant"), str)
        )
    )


def display_name(value: str) -> str:
    names = {
        "forward": "Forward",
        "forward_backward": "Forward + backward",
        "full_gla": "Full GLA",
        "gated_delta_rule": "Gated delta rule",
        "kda": "KDA",
        "kda_fused": "KDA fused",
        "kda_varlen": "KDA variable length",
        "simple_gla": "Simple GLA",
        "vanilla_linear_attn": "Vanilla linear attention",
    }
    return names.get(value, value.replace("_", " ").title())
