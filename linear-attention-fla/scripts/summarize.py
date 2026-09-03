"""Validate and summarize a linear-attention four-arm result."""

from __future__ import annotations

import argparse
import gzip
import json
import math
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ARMS = ("default", "fla_triton", "heuristic", "pre_tuned")
BASELINES = ("default", "fla_triton", "pre_tuned")
STATUSES = ("ok", "unsupported", "missing_exact_config", "error")


def _positive_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) > 0
    )


def _validate(data: object) -> list[str]:
    errors: list[str] = []
    if not isinstance(data, dict):
        return ["top-level value must be an object"]
    if data.get("schema_version") != 1:
        errors.append("schema_version must be 1")
    if data.get("benchmark") != "linear-attention-fla":
        errors.append("benchmark must be 'linear-attention-fla'")
    if not isinstance(data.get("provenance"), dict):
        errors.append("provenance must be an object")

    cells = data.get("cells")
    if not isinstance(cells, list) or not cells:
        errors.append("cells must be a non-empty array")
        return errors

    seen: set[str] = set()
    outer_rounds = data.get("provenance", {}).get("outer_rounds")
    for index, cell in enumerate(cells):
        prefix = f"cells[{index}]"
        if not isinstance(cell, dict):
            errors.append(f"{prefix} must be an object")
            continue
        cell_id = cell.get("id")
        if not isinstance(cell_id, str) or not cell_id:
            errors.append(f"{prefix}.id must be a non-empty string")
        elif cell_id in seen:
            errors.append(f"{prefix}.id duplicates {cell_id!r}")
        else:
            seen.add(cell_id)
        for field in ("variant", "mode"):
            if not isinstance(cell.get(field), str) or not cell[field]:
                errors.append(f"{prefix}.{field} must be a non-empty string")
        if not isinstance(cell.get("shape"), dict):
            errors.append(f"{prefix}.shape must be an object")

        arms = cell.get("arms")
        if not isinstance(arms, dict):
            errors.append(f"{prefix}.arms must be an object")
            continue
        for arm_name in ARMS:
            arm_prefix = f"{prefix}.arms.{arm_name}"
            arm = arms.get(arm_name)
            if not isinstance(arm, dict):
                errors.append(f"{arm_prefix} must be an object")
                continue
            status = arm.get("status")
            if status not in STATUSES:
                errors.append(
                    f"{arm_prefix}.status must be one of {', '.join(STATUSES)}"
                )
                continue
            if status != "ok":
                if arm.get("latency_ms") is not None:
                    errors.append(
                        f"{arm_prefix} has reportable latency with status {status!r}"
                    )
                continue
            if not _positive_number(arm.get("latency_ms")):
                errors.append(f"{arm_prefix}.latency_ms must be finite and positive")
            samples = arm.get("samples_ms")
            if not isinstance(samples, list) or not samples:
                errors.append(f"{arm_prefix}.samples_ms must be a non-empty array")
            elif not all(_positive_number(sample) for sample in samples):
                errors.append(
                    f"{arm_prefix}.samples_ms must contain positive finite numbers"
                )
            elif isinstance(outer_rounds, int) and len(samples) != outer_rounds:
                errors.append(
                    f"{arm_prefix}.samples_ms has {len(samples)} values, "
                    f"expected {outer_rounds}"
                )
            if arm.get("correct") is not True:
                errors.append(f"{arm_prefix}.correct must be true for status 'ok'")
            if arm_name == "pre_tuned" and arm.get("exact") is not True:
                errors.append(f"{arm_prefix}.exact must be true for status 'ok'")
    return errors


def _usable(cell: dict[str, Any], arm_name: str) -> bool:
    arm = cell["arms"][arm_name]
    return (
        arm["status"] == "ok"
        and arm.get("correct") is True
        and _positive_number(arm.get("latency_ms"))
        and (arm_name != "pre_tuned" or arm.get("exact") is True)
    )


def _geomean(values: list[float]) -> float | None:
    return statistics.geometric_mean(values) if values else None


def _comparison(
    cells: list[dict[str, Any]], baseline: str
) -> dict[str, int | float | None]:
    common = [
        cell for cell in cells if _usable(cell, baseline) and _usable(cell, "heuristic")
    ]
    return {
        "cells": len(common),
        "heuristic_speedup": _geomean(
            [
                float(cell["arms"][baseline]["latency_ms"])
                / float(cell["arms"]["heuristic"]["latency_ms"])
                for cell in common
            ]
        ),
    }


def _four_way(cells: list[dict[str, Any]]) -> dict[str, Any]:
    common = [cell for cell in cells if all(_usable(cell, arm) for arm in ARMS)]
    return {
        "cells": len(common),
        "normalized_performance": {
            arm: _geomean(
                [
                    float(cell["arms"]["heuristic"]["latency_ms"])
                    / float(cell["arms"][arm]["latency_ms"])
                    for cell in common
                ]
            )
            for arm in ARMS
        },
    }


def _group(cells: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "discovered_cells": len(cells),
        "pairwise": {baseline: _comparison(cells, baseline) for baseline in BASELINES},
        "four_way": _four_way(cells),
    }


def _summarize(data: dict[str, Any], source: Path) -> dict[str, Any]:
    cells = data["cells"]
    modes = sorted({str(cell["mode"]) for cell in cells})
    variants = sorted({str(cell["variant"]) for cell in cells})
    status_counts = {
        arm: dict(Counter(cell["arms"][arm]["status"] for cell in cells))
        for arm in ARMS
    }
    return {
        "schema_version": 1,
        "benchmark": data["benchmark"],
        "source": str(source),
        "provenance": data["provenance"],
        "discovered_cells": len(cells),
        "status_counts": status_counts,
        "overall": _group(cells),
        "by_mode": {
            mode: _group([cell for cell in cells if cell["mode"] == mode])
            for mode in modes
        },
        "by_variant": {
            variant: _group([cell for cell in cells if cell["variant"] == variant])
            for variant in variants
        },
    }


def _fmt(value: float | None, suffix: str = "") -> str:
    return "NA" if value is None else f"{value:.4f}{suffix}"


def _pairwise_cell(group: dict[str, Any], baseline: str) -> str:
    item = group["pairwise"][baseline]
    if item["cells"] == 0:
        return "NA (0)"
    return f"{_fmt(item['heuristic_speedup'], 'x')} ({item['cells']})"


def _display_name(value: str) -> str:
    names = {
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


def _group_rows(summary: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    return [
        ("Overall", summary["overall"]),
        *[(_display_name(mode), group) for mode, group in summary["by_mode"].items()],
        *[
            (_display_name(variant), group)
            for variant, group in summary["by_variant"].items()
        ],
    ]


def _markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Linear-Attention Performance",
        "",
        (
            "Pairwise values are `baseline latency / heuristic latency`; higher "
            "favors the heuristic. Parentheses contain the correct common-cell count."
        ),
        "",
        (
            "| Population | Discovered | Default / heuristic | "
            "FLA Triton / heuristic | Exact pre-tuned / heuristic |"
        ),
        "|---|---:|---:|---:|---:|",
    ]
    for label, group in _group_rows(summary):
        lines.append(
            f"| {label} | {group['discovered_cells']} | "
            f"{_pairwise_cell(group, 'default')} | "
            f"{_pairwise_cell(group, 'fla_triton')} | "
            f"{_pairwise_cell(group, 'pre_tuned')} |"
        )

    lines.extend(
        [
            "",
            "## Common Four-Way Population",
            "",
            "Performance is normalized to the heuristic at `1.0000`; higher is faster.",
            "",
            "| Population | Cells | Default | FLA Triton | Heuristic | Pre-tuned |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    four_way_groups = [
        ("Overall", summary["overall"]["four_way"]),
        *[
            (_display_name(mode), group["four_way"])
            for mode, group in summary["by_mode"].items()
        ],
    ]
    for label, item in four_way_groups:
        normalized = item["normalized_performance"]
        lines.append(
            f"| {label} | {item['cells']} | "
            f"{_fmt(normalized['default'])} | "
            f"{_fmt(normalized['fla_triton'])} | "
            f"{_fmt(normalized['heuristic'])} | "
            f"{_fmt(normalized['pre_tuned'])} |"
        )

    lines.extend(
        [
            "",
            "## Arm Status",
            "",
            "| Arm | OK | Unsupported | Missing exact config | Error |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for arm in ARMS:
        counts = summary["status_counts"][arm]
        lines.append(
            f"| `{arm}` | {counts.get('ok', 0)} | "
            f"{counts.get('unsupported', 0)} | "
            f"{counts.get('missing_exact_config', 0)} | "
            f"{counts.get('error', 0)} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    parser.add_argument("--markdown-out", type=Path)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    try:
        if args.result.suffix == ".gz":
            with gzip.open(args.result, mode="rt") as handle:
                data = json.load(handle)
        else:
            data = json.loads(args.result.read_text())
    except (OSError, json.JSONDecodeError) as error:
        print(f"error: cannot read {args.result}: {error}", file=sys.stderr)
        return 2

    errors = _validate(data)
    if errors:
        print(f"error: invalid result ({len(errors)} issue(s)):", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 2

    summary = _summarize(data, args.result)
    markdown = _markdown(summary)
    print(markdown, end="")
    if args.markdown_out is not None:
        args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_out.write_text(markdown)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
