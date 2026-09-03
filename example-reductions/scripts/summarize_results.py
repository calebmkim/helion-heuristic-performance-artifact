#!/usr/bin/env python3
"""Summarize a three-arm example-reduction benchmark campaign."""

from __future__ import annotations

import argparse
from collections import Counter
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
from typing import Any


ARMS = ("default", "seed", "torch_compile")
LABELS = {
    "default": "Default",
    "seed": "Heuristic seed",
    "torch_compile": "torch.compile",
}


def _geomean(values: list[float]) -> float | None:
    if not values:
        return None
    return math.exp(sum(math.log(value) for value in values) / len(values))


def _latency(arm: dict[str, Any] | None) -> float | None:
    if not arm or arm.get("status") != "ok":
        return None
    if not arm.get("correctness", {}).get("pass"):
        return None
    value = arm.get("latency_us")
    if not isinstance(value, (int, float)) or value <= 0:
        return None
    return float(value)


def _group(cells: list[dict[str, Any]]) -> dict[str, object]:
    valid = [
        cell
        for cell in cells
        if all(cell["latency_us"].get(arm) is not None for arm in ARMS)
    ]
    return {
        "cells": len(cells),
        "valid_cells": len(valid),
        "normalized_performance": {
            arm: _geomean(
                [
                    cell["latency_us"]["torch_compile"] / cell["latency_us"][arm]
                    for cell in valid
                ]
            )
            for arm in ARMS
        },
        "latency_us_geomean": {
            arm: _geomean([cell["latency_us"][arm] for cell in valid])
            for arm in ARMS
        },
    }


def summarize(document: dict[str, Any]) -> dict[str, object]:
    if tuple(document["arms"]) != ARMS:
        raise ValueError(f"expected arms {ARMS}, found {document['arms']}")
    cells: list[dict[str, Any]] = []
    by_kernel: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in document["records"]:
        arm_records = record.get("arms", {})
        cell: dict[str, Any] = {
            "cell_key": record["cell_key"],
            "kernel": record["kernel"],
            "shape_group": record["shape_group"],
            "shape": record["shape"],
            "provenance": record["provenance"],
            "status": {
                arm: arm_records.get(arm, {}).get("status", "missing")
                for arm in ARMS
            },
            "latency_us": {
                arm: _latency(arm_records.get(arm)) for arm in ARMS
            },
        }
        reference = cell["latency_us"]["torch_compile"]
        cell["normalized_performance"] = {
            arm: (
                reference / latency
                if reference is not None and latency is not None
                else None
            )
            for arm, latency in cell["latency_us"].items()
        }
        cells.append(cell)
        by_kernel[cell["kernel"]].append(cell)
        by_group[cell["shape_group"]].append(cell)
    return {
        "schema_version": 1,
        "profile": document["profile"],
        "manifest_sha256": document["manifest_sha256"],
        "helion_git": document["helion_git"],
        "arms": list(ARMS),
        "reference_arm": "torch_compile",
        "normalization": "torch_compile = 1.00; higher is faster",
        "torch_compile_mode": document["torch_compile_mode"],
        "timing": (
            "all arms in one cell process; shared balanced cold-L2 CUDA-event "
            "timing of pre-captured graphs; CPU launch overhead excluded"
        ),
        "overall": _group(cells),
        "kernels": {
            kernel: _group(group) for kernel, group in sorted(by_kernel.items())
        },
        "shape_groups": {
            name: _group(group) for name, group in sorted(by_group.items())
        },
        "status_counts": {
            arm: dict(
                sorted(Counter(cell["status"][arm] for cell in cells).items())
            )
            for arm in ARMS
        },
        "cells": cells,
    }


def _ratio(value: object) -> str:
    return "-" if value is None else f"{float(value):.3f}x"


def _latency_text(value: object) -> str:
    return "-" if value is None else f"{float(value):.2f}"


def _write_markdown(summary: dict[str, Any], path: Path) -> None:
    overall = summary["overall"]
    git = summary["helion_git"]
    lines = [
        "# Example Reduction Performance",
        "",
        f"- Profile: `{summary['profile']}`",
        f"- Helion commit: `{git.get('commit')}`",
        f"- Helion worktree dirty: `{git.get('dirty')}`",
        (
            f"- Torch reference: `torch.compile(mode="
            f"\"{summary['torch_compile_mode']}\")`."
        ),
        (
            "- Timing: all arms are compiled and captured in one process per cell, "
            "then timed together with balanced ordering, cold L2, and CUDA events."
        ),
        "- CPU launch overhead is excluded.",
        "- Normalization: `torch.compile = 1.00x`; higher is faster.",
        (
            f"- Three-arm common population: "
            f"{overall['valid_cells']}/{overall['cells']} cells."
        ),
        "",
        "## Per-kernel performance",
        "",
        "| Kernel | Valid | Default | Heuristic seed | torch.compile |",
        "|---|---:|---:|---:|---:|",
    ]
    for kernel, group in summary["kernels"].items():
        values = group["normalized_performance"]
        lines.append(
            f"| `{kernel}` | {group['valid_cells']}/{group['cells']} | "
            f"{_ratio(values['default'])} | {_ratio(values['seed'])} | "
            f"{_ratio(values['torch_compile'])} |"
        )
    values = overall["normalized_performance"]
    lines.append(
        f"| **Overall** | **{overall['valid_cells']}/{overall['cells']}** | "
        f"**{_ratio(values['default'])}** | **{_ratio(values['seed'])}** | "
        f"**{_ratio(values['torch_compile'])}** |"
    )
    lines.extend(
        [
            "",
            "## Geometric-mean latency",
            "",
            "| Kernel | Default (us) | Heuristic seed (us) | torch.compile (us) |",
            "|---|---:|---:|---:|",
        ]
    )
    for kernel, group in summary["kernels"].items():
        values = group["latency_us_geomean"]
        lines.append(
            f"| `{kernel}` | {_latency_text(values['default'])} | "
            f"{_latency_text(values['seed'])} | "
            f"{_latency_text(values['torch_compile'])} |"
        )
    values = overall["latency_us_geomean"]
    lines.append(
        f"| **Overall** | **{_latency_text(values['default'])}** | "
        f"**{_latency_text(values['seed'])}** | "
        f"**{_latency_text(values['torch_compile'])}** |"
    )
    lines.extend(
        [
            "",
            "## Status",
            "",
            "| Arm | Status counts |",
            "|---|---|",
        ]
    )
    for arm in ARMS:
        encoded = ", ".join(
            f"`{status}`: {count}"
            for status, count in summary["status_counts"][arm].items()
        )
        lines.append(f"| {LABELS[arm]} | {encoded} |")
    lines.extend(
        [
            "",
            "Absolute per-cell latency, shape provenance, outer-round samples, "
            "correctness, selected configs, and environment metadata remain in "
            "`benchmark.json` and `per_cell.csv`.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def _write_csv(summary: dict[str, Any], path: Path) -> None:
    fields = [
        "cell_key",
        "kernel",
        "shape_group",
        "shape_json",
        "source",
        "model",
        "basis",
        "reason",
    ]
    fields.extend(f"{arm}_status" for arm in ARMS)
    fields.extend(f"{arm}_us" for arm in ARMS)
    fields.extend(f"{arm}_vs_torch_compile" for arm in ARMS)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for cell in summary["cells"]:
            provenance = cell["provenance"]
            row: dict[str, object] = {
                "cell_key": cell["cell_key"],
                "kernel": cell["kernel"],
                "shape_group": cell["shape_group"],
                "shape_json": json.dumps(cell["shape"]),
                "source": provenance.get("source"),
                "model": provenance.get("model"),
                "basis": provenance.get("basis"),
                "reason": provenance.get("reason"),
            }
            row.update({f"{arm}_status": cell["status"][arm] for arm in ARMS})
            row.update({f"{arm}_us": cell["latency_us"][arm] for arm in ARMS})
            row.update(
                {
                    f"{arm}_vs_torch_compile": cell[
                        "normalized_performance"
                    ][arm]
                    for arm in ARMS
                }
            )
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path, required=True)
    parser.add_argument("--csv-output", type=Path, required=True)
    args = parser.parse_args()

    document = _read_json(args.input.expanduser().resolve())
    summary = summarize(document)
    json_output = args.json_output.expanduser().resolve()
    json_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(json.dumps(summary, indent=2) + "\n")
    _write_markdown(summary, args.markdown_output.expanduser().resolve())
    _write_csv(summary, args.csv_output.expanduser().resolve())
    print(
        f"Wrote {summary['overall']['valid_cells']}/"
        f"{summary['overall']['cells']} common cells to {json_output}"
    )


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


if __name__ == "__main__":
    main()
