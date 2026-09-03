#!/usr/bin/env python3
"""Summarize default-versus-seed matmul benchmark results."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

CATEGORY_LABELS = {
    "formula_matmul": "Formula matmul",
    "multi_matmul": "Multi-matmul",
}


def _geomean(values: list[float]) -> float | None:
    if not values:
        return None
    return math.exp(sum(math.log(value) for value in values) / len(values))


def _latency(record: dict[str, Any], arm: str) -> float | None:
    item = record.get("arms", {}).get(arm, {})
    if item.get("status") != "ok" or not item.get("correctness", {}).get("pass"):
        return None
    value = item.get("latency_us")
    return float(value) if isinstance(value, (int, float)) and value > 0 else None


def _base_status(record: dict[str, Any]) -> str:
    if "driver_error" in record:
        return "driver_error"
    for arm in ("default", "seed"):
        item = record.get("arms", {}).get(arm, {})
        if item.get("status") != "ok":
            return f"{arm}_{item.get('status', 'missing')}"
        if not item.get("correctness", {}).get("pass"):
            return f"{arm}_incorrect"
        if _latency(record, arm) is None:
            return f"{arm}_untimed"
    if not record.get("expected_heuristic_fired"):
        return "unexpected_heuristic"
    return "ok"


def _torch_compile_status(record: dict[str, Any]) -> str:
    item = record.get("arms", {}).get("torch_compile")
    if item is None:
        return "not_requested"
    if item.get("status") != "ok":
        return str(item.get("status", "missing"))
    if not item.get("audit", {}).get("pass"):
        return "audit_rejected"
    if not item.get("correctness", {}).get("pass"):
        return "incorrect"
    if _latency(record, "torch_compile") is None:
        return "untimed"
    return "ok"


def _cell(record: dict[str, Any]) -> dict[str, Any]:
    default = _latency(record, "default")
    seed = _latency(record, "seed")
    torch_compile = _latency(record, "torch_compile")
    base_status = _base_status(record)
    torch_compile_status = _torch_compile_status(record)
    return {
        "cell_key": record["cell_key"],
        "family": record["family"],
        "family_label": record["family_label"],
        "category": record["category"],
        "shape_id": record["shape_id"],
        "shape_convention": record["shape_convention"],
        "values": record["values"],
        "dtype": record["dtype"],
        "source": record["source"],
        "status": (
            base_status
            if base_status != "ok" or torch_compile_status == "not_requested"
            else (
                "ok"
                if torch_compile_status == "ok"
                else f"torch_compile_{torch_compile_status}"
            )
        ),
        "base_status": base_status,
        "torch_compile_status": torch_compile_status,
        "expected_heuristic": record["expected_heuristic"],
        "heuristics_fired": record.get("heuristics_fired", []),
        "compiler_seed_count": record.get("compiler_seed_count"),
        "default_latency_us": default,
        "seed_latency_us": seed,
        "torch_compile_latency_us": torch_compile,
        "seed_speedup": (
            default / seed if default is not None and seed is not None else None
        ),
        "torch_compile_speedup_vs_default": (
            default / torch_compile
            if default is not None and torch_compile is not None
            else None
        ),
        "default_speedup_vs_torch_compile": (
            torch_compile / default
            if default is not None and torch_compile is not None
            else None
        ),
        "seed_speedup_vs_torch_compile": (
            torch_compile / seed
            if seed is not None and torch_compile is not None
            else None
        ),
        "default_config": record.get("arms", {}).get("default", {}).get("config"),
        "seed_config": record.get("arms", {}).get("seed", {}).get("config"),
        "error": record.get("driver_error")
        or record.get("arms", {}).get("default", {}).get("error")
        or record.get("arms", {}).get("seed", {}).get("error")
        or record.get("arms", {}).get("torch_compile", {}).get("error"),
    }


def _group(cells: list[dict[str, Any]]) -> dict[str, object]:
    valid = [cell for cell in cells if cell["base_status"] == "ok"]
    compile_valid = [
        cell
        for cell in valid
        if cell["torch_compile_status"] == "ok"
    ]
    return {
        "cells": len(cells),
        "valid_cells": len(valid),
        "torch_compile_valid_cells": len(compile_valid),
        "default_normalized": 1.0 if valid else None,
        "seed_speedup": _geomean([cell["seed_speedup"] for cell in valid]),
        "torch_compile_normalized": 1.0 if compile_valid else None,
        "default_speedup_vs_torch_compile": _geomean(
            [cell["default_speedup_vs_torch_compile"] for cell in compile_valid]
        ),
        "seed_speedup_vs_torch_compile": _geomean(
            [cell["seed_speedup_vs_torch_compile"] for cell in compile_valid]
        ),
        "torch_compile_speedup_vs_default": _geomean(
            [cell["torch_compile_speedup_vs_default"] for cell in compile_valid]
        ),
        "default_latency_us_geomean": _geomean(
            [cell["default_latency_us"] for cell in valid]
        ),
        "seed_latency_us_geomean": _geomean(
            [cell["seed_latency_us"] for cell in valid]
        ),
        "torch_compile_latency_us_geomean": _geomean(
            [cell["torch_compile_latency_us"] for cell in compile_valid]
        ),
    }


def summarize(raw: dict[str, Any]) -> dict[str, object]:
    cells = [_cell(record) for record in raw["records"]]
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for cell in cells:
        by_family[cell["family"]].append(cell)
        by_category[cell["category"]].append(cell)
    families = {
        family: {
            "label": group[0]["family_label"],
            "category": group[0]["category"],
            **_group(group),
        }
        for family, group in by_family.items()
    }
    categories: dict[str, dict[str, object]] = {}
    for category, group in by_category.items():
        family_groups = [
            family
            for family in families.values()
            if family["category"] == category and family["seed_speedup"] is not None
        ]
        categories[category] = {
            **_group(group),
            "family_macro_geomean": _geomean(
                [float(family["seed_speedup"]) for family in family_groups]
            ),
            "valid_families": len(family_groups),
            "families": sum(
                family["category"] == category for family in families.values()
            ),
            "default_vs_torch_compile_family_macro_geomean": _geomean(
                [
                    float(family["default_speedup_vs_torch_compile"])
                    for family in family_groups
                    if family["default_speedup_vs_torch_compile"] is not None
                ]
            ),
            "seed_vs_torch_compile_family_macro_geomean": _geomean(
                [
                    float(family["seed_speedup_vs_torch_compile"])
                    for family in family_groups
                    if family["seed_speedup_vs_torch_compile"] is not None
                ]
            ),
        }
    valid_family_values = [
        float(family["seed_speedup"])
        for family in families.values()
        if family["seed_speedup"] is not None
    ]
    compile_family_groups = [
        family
        for family in families.values()
        if family["seed_speedup_vs_torch_compile"] is not None
    ]
    torch_compile_requested = "torch_compile" in raw.get("arms", [])
    return {
        "schema_version": 2,
        "profile": raw["profile"],
        "manifest": raw["manifest"],
        "manifest_sha256": raw["manifest_sha256"],
        "helion_root": raw["helion_root"],
        "helion_git": raw["helion_git"],
        "environment": raw["environment"],
        "timing": raw["timing"],
        "normalization": "default = 1.00x; higher is faster",
        "ratio": "default latency / heuristic-seed latency",
        "torch_compile_requested": torch_compile_requested,
        "torch_compile_normalization": (
            "torch.compile = 1.00x; higher is faster"
            if torch_compile_requested
            else None
        ),
        "overall": {
            **_group(cells),
            "family_macro_geomean": _geomean(valid_family_values),
            "valid_families": len(valid_family_values),
            "families": len(families),
            "torch_compile_valid_families": len(compile_family_groups),
            "default_vs_torch_compile_family_macro_geomean": _geomean(
                [
                    float(family["default_speedup_vs_torch_compile"])
                    for family in compile_family_groups
                ]
            ),
            "seed_vs_torch_compile_family_macro_geomean": _geomean(
                [
                    float(family["seed_speedup_vs_torch_compile"])
                    for family in compile_family_groups
                ]
            ),
        },
        "categories": categories,
        "families": families,
        "status_counts": dict(sorted(Counter(cell["status"] for cell in cells).items())),
        "base_status_counts": dict(
            sorted(Counter(cell["base_status"] for cell in cells).items())
        ),
        "torch_compile_status_counts": dict(
            sorted(Counter(cell["torch_compile_status"] for cell in cells).items())
        ),
        "secondary_heuristics": dict(
            sorted(
                Counter(
                    heuristic
                    for cell in cells
                    for heuristic in cell["heuristics_fired"]
                    if heuristic != cell["expected_heuristic"]
                ).items()
            )
        ),
        "cells": cells,
    }


def _ratio(value: object) -> str:
    return "-" if value is None else f"{float(value):.3f}x"


def _latency_text(value: object) -> str:
    return "-" if value is None else f"{float(value):.2f}"


def _write_markdown(summary: dict[str, Any], path: Path) -> None:
    git = summary["helion_git"]
    environment = summary["environment"]
    gpu = environment.get("gpu", {})
    timing = summary["timing"]
    compile_requested = summary["torch_compile_requested"]
    arms_text = (
        "- Arms: raw `ConfigSpec._base_default_config()`, the promoted rank-0 "
        "compiler heuristic seed, and each example's natural PyTorch reference "
        "under `torch.compile(mode=\"max-autotune-no-cudagraphs\")`. Helion "
        "autotuning and tuned-cache replay were disabled; TorchInductor GEMMs "
        "were max-autotuned with Triton as the only permitted backend."
        if compile_requested
        else (
            "- Arms: raw `ConfigSpec._base_default_config()` and the promoted "
            "rank-0 compiler heuristic seed; autotuning and tuned-cache replay "
            "were disabled."
        )
    )
    lines = [
        "# Other Matmul Kernel Performance",
        "",
        f"- Profile: `{summary['profile']}`",
        f"- GPU: `{gpu.get('name')}` (compute capability `{gpu.get('capability')}`)",
        f"- Helion commit: `{git.get('commit')}`",
        f"- Helion worktree dirty: `{git.get('dirty')}`",
        arms_text,
        f"- Timing: {timing['method']}; {timing['outer_rounds']} outer rounds.",
        "- Normalization: `default = 1.00x`; higher is faster.",
        (
            "- This is a seed-quality comparison, not an autotuned-quality "
            "comparison. A rigorous extension should autotune every shape."
        ),
        "",
        "## Aggregate performance",
        "",
        "| Population | Valid shapes | Families | Default | Heuristic seed (cell geomean) | Heuristic seed (family macro-geomean) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for category in ("formula_matmul", "multi_matmul"):
        if category not in summary["categories"]:
            continue
        group = summary["categories"][category]
        lines.append(
            f"| {CATEGORY_LABELS[category]} | "
            f"{group['valid_cells']}/{group['cells']} | "
            f"{group['valid_families']}/{group['families']} | 1.000x | "
            f"{_ratio(group['seed_speedup'])} | "
            f"{_ratio(group['family_macro_geomean'])} |"
        )
    overall = summary["overall"]
    lines.append(
        f"| **Overall** | **{overall['valid_cells']}/{overall['cells']}** | "
        f"**{overall['valid_families']}/{overall['families']}** | **1.000x** | "
        f"**{_ratio(overall['seed_speedup'])}** | "
        f"**{_ratio(overall['family_macro_geomean'])}** |"
    )
    if compile_requested:
        lines.extend(
            [
                "",
                "## Torch compile comparison",
                "",
                (
                    "These ratios use `torch.compile = 1.00x`; higher is faster. "
                    "Generated TorchInductor source must pass the Triton-only "
                    "dispatch audit."
                ),
                "",
                "| Population | Valid shapes | Torch compile | Helion default (cell geomean) | Helion seed (cell geomean) | Helion default (family macro-geomean) | Helion seed (family macro-geomean) |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for category in ("formula_matmul", "multi_matmul"):
            if category not in summary["categories"]:
                continue
            group = summary["categories"][category]
            lines.append(
                f"| {CATEGORY_LABELS[category]} | "
                f"{group['torch_compile_valid_cells']}/{group['cells']} | "
                f"1.000x | "
                f"{_ratio(group['default_speedup_vs_torch_compile'])} | "
                f"{_ratio(group['seed_speedup_vs_torch_compile'])} | "
                f"{_ratio(group['default_vs_torch_compile_family_macro_geomean'])} | "
                f"{_ratio(group['seed_vs_torch_compile_family_macro_geomean'])} |"
            )
        lines.append(
            f"| **Overall** | "
            f"**{overall['torch_compile_valid_cells']}/{overall['cells']}** | "
            f"**1.000x** | "
            f"**{_ratio(overall['default_speedup_vs_torch_compile'])}** | "
            f"**{_ratio(overall['seed_speedup_vs_torch_compile'])}** | "
            f"**{_ratio(overall['default_vs_torch_compile_family_macro_geomean'])}** | "
            f"**{_ratio(overall['seed_vs_torch_compile_family_macro_geomean'])}** |"
        )
    lines.extend(
        [
            "",
            (
                "The family macro-geomean gives every kernel family equal weight; "
                "the cell geomean gives every shape equal weight."
            ),
            "",
            "## Per-family performance",
            "",
            "| Kernel family | Family ID | Valid shapes | Default | Heuristic seed | Default latency (us) | Seed latency (us) |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for family, group in summary["families"].items():
        lines.append(
            f"| {group['label']} | `{family}` | "
            f"{group['valid_cells']}/{group['cells']} | 1.000x | "
            f"{_ratio(group['seed_speedup'])} | "
            f"{_latency_text(group['default_latency_us_geomean'])} | "
            f"{_latency_text(group['seed_latency_us_geomean'])} |"
        )
    if compile_requested:
        lines.extend(
            [
                "",
                "## Per-family torch compile comparison",
                "",
                "| Kernel family | Valid shapes | Torch compile | Helion default | Helion seed | Torch compile latency (us) |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for group in summary["families"].values():
            lines.append(
                f"| {group['label']} | "
                f"{group['torch_compile_valid_cells']}/{group['cells']} | "
                f"{_ratio(group['torch_compile_normalized'])} | "
                f"{_ratio(group['default_speedup_vs_torch_compile'])} | "
                f"{_ratio(group['seed_speedup_vs_torch_compile'])} | "
                f"{_latency_text(group['torch_compile_latency_us_geomean'])} |"
            )
    lines.extend(
        [
            "",
            "## Per-shape performance",
            "",
            (
                "| Family | Shape | Dtype | Fired heuristic(s) | Default (us) | "
                "Seed (us) | Torch compile (us) | Seed/default | Default/torch "
                "compile | Seed/torch compile | Status |"
                if compile_requested
                else (
                    "| Family | Shape | Dtype | Fired heuristic(s) | Default "
                    "(us) | Seed (us) | Seed speedup | Status |"
                )
            ),
            (
                "|---|---|---|---|---:|---:|---:|---:|---:|---:|---|"
                if compile_requested
                else "|---|---|---|---|---:|---:|---:|---|"
            ),
        ]
    )
    for cell in summary["cells"]:
        fired = ", ".join(f"`{name}`" for name in cell["heuristics_fired"]) or "-"
        prefix = (
            f"| {cell['family_label']} | `{cell['shape_id']}` | "
            f"`{cell['dtype']}` | {fired} | "
            f"{_latency_text(cell['default_latency_us'])} | "
            f"{_latency_text(cell['seed_latency_us'])} | "
        )
        if compile_requested:
            lines.append(
                prefix
                + f"{_latency_text(cell['torch_compile_latency_us'])} | "
                f"{_ratio(cell['seed_speedup'])} | "
                f"{_ratio(cell['default_speedup_vs_torch_compile'])} | "
                f"{_ratio(cell['seed_speedup_vs_torch_compile'])} | "
                f"`{cell['status']}` |"
            )
        else:
            lines.append(
                prefix
                + f"{_ratio(cell['seed_speedup'])} | `{cell['status']}` |"
            )
    lines.extend(
        [
            "",
            "## Audit notes",
            "",
            (
                "All compiler-generated seeds, selected effective configs, and "
                "absolute per-round latencies are retained in `raw-results.json`."
            ),
        ]
    )
    if compile_requested:
        lines.append(
            "TorchInductor options, generated-source hashes, Triton kernel counts, "
            "and forbidden-dispatch matches are also retained per shape."
        )
        lines.append(
            "Attention, recurrent, and state-space references are natural PyTorch "
            "formulations, not tiled replicas of the Helion algorithms; their "
            "ratios therefore include fusion and algorithmic differences."
        )
    if summary["secondary_heuristics"]:
        encoded = ", ".join(
            f"`{name}` on {count} shapes"
            for name, count in summary["secondary_heuristics"].items()
        )
        lines.append(
            "Secondary compiler heuristics also fired after the expected "
            f"formula/multi-matmul heuristic: {encoded}."
        )
    failures = [cell for cell in summary["cells"] if cell["status"] != "ok"]
    if failures:
        lines.extend(["", "## Failures", ""])
        for cell in failures:
            message = str(cell.get("error") or "see results.json").replace("\n", " ")
            lines.append(
                f"- `{cell['cell_key']}`: `{cell['status']}`: {message[:500]}"
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def _write_csv(summary: dict[str, Any], path: Path) -> None:
    fields = (
        "cell_key",
        "category",
        "family",
        "family_label",
        "shape_id",
        "dtype",
        "source",
        "status",
        "base_status",
        "torch_compile_status",
        "expected_heuristic",
        "heuristics_fired",
        "compiler_seed_count",
        "default_latency_us",
        "seed_latency_us",
        "torch_compile_latency_us",
        "seed_speedup",
        "torch_compile_speedup_vs_default",
        "default_speedup_vs_torch_compile",
        "seed_speedup_vs_torch_compile",
        "default_config",
        "seed_config",
        "error",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for cell in summary["cells"]:
            row = {field: cell.get(field) for field in fields}
            for field in (
                "heuristics_fired",
                "default_config",
                "seed_config",
            ):
                row[field] = json.dumps(row[field], sort_keys=True)
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--markdown-out", type=Path, required=True)
    parser.add_argument("--csv-out", type=Path, required=True)
    args = parser.parse_args()
    raw = json.loads(args.results.expanduser().resolve().read_text())
    summary = summarize(raw)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(summary, indent=2, default=str) + "\n")
    _write_markdown(summary, args.markdown_out)
    _write_csv(summary, args.csv_out)
    print(
        f"Wrote summary for {summary['overall']['valid_cells']}/"
        f"{summary['overall']['cells']} valid shapes"
    )


if __name__ == "__main__":
    main()
