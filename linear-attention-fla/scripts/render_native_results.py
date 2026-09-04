#!/usr/bin/env python3
"""Summarize and plot Helion's native linear-attention dashboard output."""

from __future__ import annotations

import argparse
import importlib
import json
import math
import statistics
import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from aot_adapter import repair_metadata
from result_utils import display_name


def _geomean(values: Iterable[float]) -> float | None:
    usable = [value for value in values if math.isfinite(value) and value > 0]
    return statistics.geometric_mean(usable) if usable else None


def _git_revision(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _fla_version(path: Path | None) -> str | None:
    if path is not None:
        sys.path.insert(0, str(path))
    try:
        module = importlib.import_module("fla")
        version = getattr(module, "__version__", None)
        return str(version) if version is not None else None
    except ImportError:
        return None
    finally:
        if path is not None:
            try:
                sys.path.remove(str(path))
            except ValueError:
                pass


def _load_cells(path: Path) -> tuple[list[dict[str, Any]], str | None]:
    records = json.loads(path.read_text())
    if not isinstance(records, list):
        raise ValueError(f"{path} is not a helionbench result list")

    metrics: dict[tuple[str, str], tuple[list[str], list[float]]] = {}
    device: str | None = None
    for record in records:
        if not isinstance(record, dict):
            continue
        benchmark = record.get("benchmark", {})
        extra = benchmark.get("extra_info", {}) if isinstance(benchmark, dict) else {}
        if device is None and isinstance(extra, dict) and extra.get("device"):
            device = str(extra["device"])
        model = record.get("model", {})
        metric = record.get("metric", {})
        name = model.get("name") if isinstance(model, dict) else None
        metric_name = metric.get("name") if isinstance(metric, dict) else None
        shapes = record.get("shape")
        values = metric.get("benchmark_values") if isinstance(metric, dict) else None
        if not isinstance(name, str) or not isinstance(metric_name, str):
            continue
        if not isinstance(shapes, list) or not isinstance(values, list):
            continue
        if len(shapes) != len(values):
            raise ValueError(
                f"{name}/{metric_name} has {len(shapes)} shapes and "
                f"{len(values)} values"
            )
        metrics[(name, metric_name)] = (
            [str(shape) for shape in shapes],
            [float(value) for value in values],
        )

    cells: list[dict[str, Any]] = []
    models = list(dict.fromkeys(name for name, _metric in metrics))
    for model in models:
        latency_record = metrics.get((model, "helion_latency_ms"))
        speedup_record = metrics.get((model, "helion_speedup"))
        accuracy_record = metrics.get((model, "helion_accuracy"))
        if latency_record is None or speedup_record is None:
            continue
        shapes, latencies = latency_record
        speedup_shapes, speedups = speedup_record
        if speedup_shapes != shapes:
            raise ValueError(f"{model} latency and speedup shapes differ")
        if accuracy_record is None:
            accuracies = [math.nan] * len(shapes)
        else:
            accuracy_shapes, accuracies = accuracy_record
            if accuracy_shapes != shapes:
                raise ValueError(f"{model} latency and accuracy shapes differ")

        flash_record = metrics.get((model, "flashkda_speedup"))
        if flash_record is None:
            flash_speedups: list[float | None] = [None] * len(shapes)
        else:
            flash_shapes, raw_flash_speedups = flash_record
            if flash_shapes != shapes:
                raise ValueError(f"{model} Helion and FlashKDA shapes differ")
            flash_speedups = raw_flash_speedups

        backward = model.endswith("-bwd")
        variant = model.removesuffix("-bwd")
        for shape, latency, speedup, accuracy, flash_speedup in zip(
            shapes,
            latencies,
            speedups,
            accuracies,
            flash_speedups,
            strict=True,
        ):
            cells.append(
                {
                    "model": model,
                    "variant": variant,
                    "mode": "forward_backward" if backward else "forward",
                    "shape": shape,
                    "helion_latency_ms": latency,
                    "helion_speedup": speedup,
                    "accuracy": accuracy,
                    "flashkda_speedup": flash_speedup,
                }
            )
    if not cells:
        raise ValueError(f"{path} has no native Helion latency/speedup cells")
    return cells, device


def _valid_ratio(cell: dict[str, Any], field: str = "helion_speedup") -> float | None:
    accuracy = cell.get("accuracy")
    value = cell.get(field)
    if field == "helion_speedup" and accuracy != 1.0:
        return None
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or value <= 0
    ):
        return None
    return float(value)


def _fmt(value: float | None, count: int) -> str:
    return "NA (0)" if value is None else f"{value:.4f}x ({count})"


def _row(label: str, cells: list[dict[str, Any]]) -> str:
    ratios = [
        ratio
        for cell in cells
        if (ratio := _valid_ratio(cell)) is not None
    ]
    correct = sum(cell.get("accuracy") == 1.0 for cell in cells)
    positive = sum(
        isinstance(cell.get("helion_latency_ms"), (int, float))
        and float(cell["helion_latency_ms"]) > 0
        for cell in cells
    )
    return (
        f"| {label} | {len(cells)} | {correct} | {positive} | "
        f"{_fmt(_geomean(ratios), len(ratios))} |"
    )


def _render_summary(
    cells: list[dict[str, Any]],
    *,
    device: str | None,
    config_label: str,
    helion_revision: str | None,
    fla_version: str | None,
    fla_revision: str | None,
    aot_compat_repair: bool,
) -> str:
    lines = [
        "# Native Linear-Attention Performance",
        "",
        (
            "This is Helion's native block-ordered timing path, not the "
            "controlled sample-interleaved four-arm measurement. It compares "
            f"FLA with the single active Helion configuration: **{config_label}**."
        ),
        "",
    ]
    details = []
    if device:
        details.append(device)
    if helion_revision:
        details.append(f"Helion `{helion_revision[:12]}`")
    if fla_version:
        fla = f"FLA `{fla_version}`"
        if fla_revision:
            fla += f" (`{fla_revision[:12]}`)"
        details.append(fla)
    elif fla_revision:
        details.append(f"FLA `{fla_revision[:12]}`")
    if details:
        lines.extend(["Environment: " + ", ".join(details) + ".", ""])
    if aot_compat_repair:
        repair = repair_metadata()[0]
        lines.extend(
            [
                (
                    "AOT compatibility repair: "
                    f"`{repair['kernel']}` changes "
                    f"`{repair['field']}: {repair['from']}` to "
                    f"`{repair['to']}`. This removes a stale zero-valued "
                    "setting for a range that no longer accepts it; all "
                    "performance-bearing AOT fields remain unchanged."
                ),
                "",
            ]
        )

    correct = sum(cell.get("accuracy") == 1.0 for cell in cells)
    positive = sum(
        isinstance(cell.get("helion_latency_ms"), (int, float))
        and float(cell["helion_latency_ms"]) > 0
        for cell in cells
    )
    lines.extend(
        [
            (
                f"Validation: {correct}/{len(cells)} cells were accepted by the "
                f"native accuracy output; {positive}/{len(cells)} Helion "
                "latencies were positive."
            ),
            "",
            (
                "Values are higher-is-better geometric means of the native "
                "`helion_speedup` metric (`FLA latency / Helion latency`). "
                "Counts are accepted common cells."
            ),
            "",
            "| Population | Cells | Accuracy accepted | Positive latency | Helion / FLA |",
            "|---|---:|---:|---:|---:|",
            _row("Overall", cells),
        ]
    )
    for mode in ("forward", "forward_backward"):
        selected = [cell for cell in cells if cell["mode"] == mode]
        if selected:
            lines.append(_row(display_name(mode), selected))

    lines.extend(
        [
            "",
            "## Per Kernel",
            "",
            "| Kernel | Mode | Cells | Accuracy accepted | Positive latency | Helion / FLA |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    variants = list(dict.fromkeys(str(cell["variant"]) for cell in cells))
    for variant in variants:
        for mode in ("forward", "forward_backward"):
            selected = [
                cell
                for cell in cells
                if cell["variant"] == variant and cell["mode"] == mode
            ]
            if selected:
                row = _row(display_name(variant), selected)
                lines.append(
                    row.replace(
                        f"| {display_name(variant)} |",
                        f"| {display_name(variant)} | {display_name(mode)} |",
                        1,
                    )
                )

    flash = [
        ratio
        for cell in cells
        if (ratio := _valid_ratio(cell, "flashkda_speedup")) is not None
    ]
    if flash:
        lines.extend(
            [
                "",
                (
                    "FlashKDA is available on "
                    f"{len(flash)} cells and has a "
                    f"`{_geomean(flash):.4f}x` geometric-mean performance "
                    "ratio relative to FLA."
                ),
            ]
        )
    return "\n".join(lines) + "\n"


def _mode_values(
    cells: list[dict[str, Any]], mode: str
) -> tuple[list[str], list[float | None], list[int], list[float | None], list[int]]:
    labels: list[str] = []
    helion_values: list[float | None] = []
    helion_counts: list[int] = []
    flash_values: list[float | None] = []
    flash_counts: list[int] = []
    variants = list(
        dict.fromkeys(
            str(cell["variant"]) for cell in cells if cell["mode"] == mode
        )
    )
    groups = [
        (
            display_name(variant),
            [
                cell
                for cell in cells
                if cell["mode"] == mode and cell["variant"] == variant
            ],
        )
        for variant in variants
    ]
    mode_cells = [cell for cell in cells if cell["mode"] == mode]
    if mode_cells:
        groups.append(("Geomean", mode_cells))
    for label, selected in groups:
        helion = [
            ratio
            for cell in selected
            if (ratio := _valid_ratio(cell)) is not None
        ]
        flash = [
            ratio
            for cell in selected
            if (ratio := _valid_ratio(cell, "flashkda_speedup")) is not None
        ]
        labels.append(label)
        helion_values.append(_geomean(helion))
        helion_counts.append(len(helion))
        flash_values.append(_geomean(flash))
        flash_counts.append(len(flash))
    return labels, helion_values, helion_counts, flash_values, flash_counts


def _plot(
    cells: list[dict[str, Any]],
    output: Path,
    *,
    config_label: str,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise SystemExit(
            "render_native_results.py requires matplotlib to create the graph"
        ) from error

    mode_data = {
        mode: _mode_values(cells, mode)
        for mode in ("forward", "forward_backward")
    }
    max_clusters = max((len(item[0]) for item in mode_data.values()), default=1)
    figure, axes = plt.subplots(
        2,
        1,
        figsize=(max(11.0, max_clusters * 1.55), 9.5),
    )
    figure.subplots_adjust(top=0.86, bottom=0.09, hspace=0.78)
    all_values = [
        value
        for _labels, helion, _hn, flash, _fn in mode_data.values()
        for value in [*helion, *flash]
        if value is not None
    ]
    ymax = max(1.35, max(all_values, default=1.0) * 1.22)

    for axis, mode in zip(
        axes, ("forward", "forward_backward"), strict=True
    ):
        labels, helion, helion_n, flash, flash_n = mode_data[mode]
        positions = list(range(len(labels)))
        has_flash = any(value is not None for value in flash)
        width = 0.28 if has_flash else 0.46
        helion_positions = [
            position - width / 2 if has_flash else position
            for position in positions
        ]
        axis.axhline(
            1.0,
            color="#B53A3A",
            linestyle="--",
            linewidth=1.5,
            label="FLA Triton (1.00x)",
            zorder=1,
        )
        bars = axis.bar(
            helion_positions,
            [value if value is not None else math.nan for value in helion],
            width,
            color="#D09A38",
            edgecolor="white",
            linewidth=0.7,
            label=f"Helion ({config_label})",
            zorder=2,
        )
        for bar, value, count in zip(bars, helion, helion_n, strict=True):
            if value is not None:
                axis.text(
                    bar.get_x() + bar.get_width() / 2,
                    value + ymax * 0.025,
                    f"{value:.2f}x\nn={count}",
                    ha="center",
                    va="bottom",
                    fontsize=7.5,
                )
        if has_flash:
            flash_bars = axis.bar(
                [position + width / 2 for position in positions],
                [value if value is not None else math.nan for value in flash],
                width,
                color="#557AA6",
                edgecolor="white",
                linewidth=0.7,
                label="FlashKDA",
                zorder=2,
            )
            for bar, value, count in zip(
                flash_bars, flash, flash_n, strict=True
            ):
                if value is not None:
                    axis.text(
                        bar.get_x() + bar.get_width() / 2,
                        value + ymax * 0.025,
                        f"{value:.2f}x\nn={count}",
                        ha="center",
                        va="bottom",
                        fontsize=7.5,
                    )
        axis.set_title(display_name(mode), fontsize=12, fontweight="bold")
        axis.set_xticks(positions, labels, rotation=25, ha="right")
        axis.set_ylim(0, ymax)
        axis.set_ylabel("Performance relative to FLA")
        axis.grid(axis="y", color="#D9DDE1", linewidth=0.7, alpha=0.8)
        axis.set_axisbelow(True)

    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.945),
        ncol=3,
        frameon=False,
    )
    figure.suptitle(
        "Native Linear-Attention Performance vs FLA Triton",
        fontsize=15,
        fontweight="bold",
        y=0.99,
    )
    figure.text(
        0.5,
        0.002,
        "Native block-ordered timing; geometric mean across accepted common shapes.",
        ha="center",
        fontsize=9,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--plot", type=Path, required=True)
    parser.add_argument("--provenance", type=Path)
    parser.add_argument("--helion-root", type=Path)
    parser.add_argument("--fla-root", type=Path)
    parser.add_argument(
        "--config-label",
        default="environment-selected Helion configuration",
    )
    parser.add_argument("--aot-compat-repair", action="store_true")
    args = parser.parse_args()

    cells, device = _load_cells(args.input)
    helion_revision = _git_revision(args.helion_root)
    fla_version = _fla_version(args.fla_root)
    fla_revision = _git_revision(args.fla_root)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        _render_summary(
            cells,
            device=device,
            config_label=args.config_label,
            helion_revision=helion_revision,
            fla_version=fla_version,
            fla_revision=fla_revision,
            aot_compat_repair=args.aot_compat_repair,
        )
    )
    _plot(cells, args.plot, config_label=args.config_label)

    if args.provenance is not None:
        import torch
        import triton

        provenance = {
            "source": str(args.input.resolve()),
            "measurement": "Helion native block-ordered linear-attention harness",
            "helion_configuration": args.config_label,
            "aot_compatibility_repairs": (
                repair_metadata() if args.aot_compat_repair else []
            ),
            "helion_revision": helion_revision,
            "fla_version": fla_version,
            "fla_revision": fla_revision,
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "triton": triton.__version__,
            "gpu": device,
            "cells": len(cells),
        }
        args.provenance.parent.mkdir(parents=True, exist_ok=True)
        args.provenance.write_text(json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    main()
