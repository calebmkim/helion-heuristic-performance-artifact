#!/usr/bin/env python3
"""Summarize a combined three- or four-arm vLLM-reduction campaign."""

from __future__ import annotations

import argparse
from collections import Counter
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
from typing import Any


REQUIRED_ARMS = ("default", "seed", "aot_tuned")
OPTIONAL_ARM = "vllm_cuda"
LABELS = {
    "default": "Default",
    "seed": "Seed",
    "aot_tuned": "AOT tuned",
    "vllm_cuda": "vLLM CUDA",
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


def _group(
    cells: list[dict[str, Any]],
    arms: tuple[str, ...],
    reference_arm: str,
) -> dict[str, object]:
    valid = [
        cell
        for cell in cells
        if all(cell["latency_us"].get(arm) is not None for arm in arms)
    ]
    return {
        "cells": len(cells),
        "valid_cells": len(valid),
        "normalized_performance": {
            arm: _geomean(
                [
                    cell["latency_us"][reference_arm] / cell["latency_us"][arm]
                    for cell in valid
                ]
            )
            for arm in arms
        },
        "latency_us_geomean": {
            arm: _geomean([cell["latency_us"][arm] for cell in valid])
            for arm in arms
        },
    }


def _fmt(value: object, digits: int = 3) -> str:
    return "-" if value is None else f"{float(value):.{digits}f}x"


def summarize(document: dict[str, Any]) -> dict[str, object]:
    arms = tuple(document["arms"])
    missing = set(REQUIRED_ARMS) - set(arms)
    unknown = set(arms) - {*REQUIRED_ARMS, OPTIONAL_ARM}
    if missing or unknown:
        raise ValueError(
            f"invalid arm set: missing={sorted(missing)}, unknown={sorted(unknown)}"
        )
    reference_arm = OPTIONAL_ARM if OPTIONAL_ARM in arms else "aot_tuned"
    cells: list[dict[str, Any]] = []
    by_kernel: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for record in document["records"]:
        arm_records = record.get("arms", {})
        cell: dict[str, Any] = {
            "cell_key": record["cell_key"],
            "kernel": record["kernel"],
            "shape": record["shape"],
            "bench_shape": record.get("bench_shape"),
            "status": {
                arm: arm_records.get(arm, {}).get("status", "missing") for arm in arms
            },
            "latency_us": {
                arm: _latency(arm_records.get(arm)) for arm in arms
            },
        }
        reference = cell["latency_us"][reference_arm]
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

    overall = _group(cells, arms, reference_arm)
    return {
        "schema_version": 2,
        "profile": document["profile"],
        "manifest_sha256": document["manifest_sha256"],
        "arms": list(arms),
        "reference_arm": reference_arm,
        "normalization": f"{reference_arm} = 1.00; higher is faster",
        "timing": (
            "all arms in one cell process; shared balanced cold-L2 CUDA-event "
            "timing of pre-captured graphs; CPU launch overhead excluded"
        ),
        "overall": overall,
        "kernels": {
            kernel: _group(group, arms, reference_arm)
            for kernel, group in sorted(by_kernel.items())
        },
        "status_counts": {
            arm: dict(
                sorted(Counter(cell["status"][arm] for cell in cells).items())
            )
            for arm in arms
        },
        "cells": cells,
    }


def _write_markdown(summary: dict[str, Any], path: Path) -> None:
    overall = summary["overall"]
    arms = tuple(summary["arms"])
    reference_arm = summary["reference_arm"]
    header = "| Kernel | Valid | " + " | ".join(LABELS[arm] for arm in arms) + " |"
    separator = "|---|---:|" + "---:|" * len(arms)
    lines = [
        "# vLLM Reduction Performance",
        "",
        f"- Profile: `{summary['profile']}`",
        (
            "- Timing: all arms are compiled and captured in one process per cell, "
            "then timed together with balanced ordering, cold L2, and CUDA events."
        ),
        "- CPU launch overhead is excluded.",
        f"- Normalization: `{reference_arm} = 1.00x`; higher is faster.",
        (
            f"- {len(arms)}-arm common population: "
            f"{overall['valid_cells']}/{overall['cells']} cells."
        ),
    ]
    if reference_arm == "aot_tuned":
        lines.append(
            "- External vLLM CUDA extension unavailable or intentionally omitted."
        )
    lines.extend(["", "## Per-kernel geomeans", "", header, separator])
    for kernel, group in summary["kernels"].items():
        values = " | ".join(
            _fmt(group["normalized_performance"][arm]) for arm in arms
        )
        lines.append(
            f"| `{kernel}` | {group['valid_cells']}/{group['cells']} | {values} |"
        )

    overall_values = " | ".join(
        f"**{_fmt(overall['normalized_performance'][arm])}**" for arm in arms
    )
    lines.extend(
        [
            f"| **Overall** | **{overall['valid_cells']}/{overall['cells']}** | "
            f"{overall_values} |",
            "",
            "## Status",
            "",
            "| Arm | Status counts |",
            "|---|---|",
        ]
    )
    for arm in arms:
        encoded = ", ".join(
            f"`{status}`: {count}"
            for status, count in summary["status_counts"][arm].items()
        )
        lines.append(f"| {LABELS[arm]} | {encoded} |")
    lines.extend(
        [
            "",
            "Absolute latency, outer-round samples, correctness results, selected "
            "configs, and environment metadata remain in `benchmark.json`.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def _write_csv(summary: dict[str, Any], path: Path) -> None:
    arms = tuple(summary["arms"])
    reference_arm = summary["reference_arm"]
    fields = ["cell_key", "kernel", "shape_json"]
    fields.extend(f"{arm}_status" for arm in arms)
    fields.extend(f"{arm}_us" for arm in arms)
    fields.extend(f"{arm}_vs_{reference_arm}" for arm in arms)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for cell in summary["cells"]:
            row: dict[str, object] = {
                "cell_key": cell["cell_key"],
                "kernel": cell["kernel"],
                "shape_json": json.dumps(cell["shape"], sort_keys=True),
            }
            row.update({f"{arm}_status": cell["status"][arm] for arm in arms})
            row.update({f"{arm}_us": cell["latency_us"][arm] for arm in arms})
            row.update(
                {
                    f"{arm}_vs_{reference_arm}": cell[
                        "normalized_performance"
                    ][arm]
                    for arm in arms
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

    document = json.loads(args.input.expanduser().resolve().read_text())
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


if __name__ == "__main__":
    main()
