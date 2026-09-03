"""Plot per-kernel performance normalized to the FLA Triton baseline."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

from result_utils import ARM_LABELS
from result_utils import display_name
from result_utils import latency_ms
from result_utils import load_result
from result_utils import relative_to_fla
from result_utils import variants

PLOTTED_ARMS = ("default", "seed", "aot_tuned")
COLORS = {
    "default": "#8A9199",
    "seed": "#2F7D69",
    "aot_tuned": "#D09A38",
}
LABEL_LIFTS = {
    "default": 0.015,
    "seed": 0.055,
    "aot_tuned": 0.095,
}


def _mode_values(
    cells: list[dict[str, Any]], mode: str
) -> tuple[
    list[str],
    dict[str, list[float | None]],
    dict[str, list[int]],
    list[int],
]:
    labels: list[str] = []
    values = {arm: [] for arm in PLOTTED_ARMS}
    counts = {arm: [] for arm in PLOTTED_ARMS}
    fla_counts: list[int] = []
    for variant in variants(cells):
        selected = [
            cell
            for cell in cells
            if cell.get("variant") == variant and cell.get("mode") == mode
        ]
        if not selected:
            continue
        computed = {arm: relative_to_fla(selected, arm) for arm in PLOTTED_ARMS}
        labels.append(display_name(variant))
        fla_counts.append(
            sum(latency_ms(cell, "fla_triton") is not None for cell in selected)
        )
        for arm, (value, count) in computed.items():
            values[arm].append(value)
            counts[arm].append(count)

    mode_cells = [cell for cell in cells if cell.get("mode") == mode]
    if mode_cells:
        computed = {arm: relative_to_fla(mode_cells, arm) for arm in PLOTTED_ARMS}
        labels.append("Geomean")
        fla_counts.append(
            sum(latency_ms(cell, "fla_triton") is not None for cell in mode_cells)
        )
        for arm, (value, count) in computed.items():
            values[arm].append(value)
            counts[arm].append(count)
    return labels, values, counts, fla_counts


def plot(data: dict[str, Any], output: Path, title: str | None) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise SystemExit(
            "plot_results.py requires matplotlib; use an environment that provides it "
            "or adapt the plotting script."
        ) from error

    cells = data["cells"]
    modes = ("forward", "forward_backward")
    mode_data = {mode: _mode_values(cells, mode) for mode in modes}
    max_clusters = max((len(item[0]) for item in mode_data.values()), default=1)
    figure, axes = plt.subplots(
        2,
        1,
        figsize=(max(11.0, max_clusters * 1.55), 9.5),
    )
    figure.subplots_adjust(top=0.86, bottom=0.09, hspace=0.78)
    width = 0.24
    offsets = (-width, 0.0, width)
    all_values = [
        value
        for _labels, values, _counts, _fla_counts in mode_data.values()
        for arm_values in values.values()
        for value in arm_values
        if value is not None
    ]
    ymax = max(1.35, max(all_values, default=1.0) * 1.20)

    for axis, mode in zip(axes, modes, strict=True):
        labels, values, counts, fla_counts = mode_data[mode]
        positions = list(range(len(labels)))
        axis.axhline(
            1.0,
            color="#B53A3A",
            linestyle="--",
            linewidth=1.5,
            label="FLA Triton (1.00x)",
            zorder=1,
        )
        for arm, offset in zip(PLOTTED_ARMS, offsets, strict=True):
            heights = [
                value if value is not None else math.nan for value in values[arm]
            ]
            bars = axis.bar(
                [position + offset for position in positions],
                heights,
                width,
                label=ARM_LABELS[arm],
                color=COLORS[arm],
                edgecolor="white",
                linewidth=0.7,
                zorder=2,
            )
            for bar, value, count, fla_count in zip(
                bars, values[arm], counts[arm], fla_counts, strict=True
            ):
                if value is None:
                    continue
                label = f"{value:.2f}x"
                if count != fla_count:
                    label += f"\nn={count}"
                axis.text(
                    bar.get_x() + bar.get_width() / 2,
                    value + ymax * LABEL_LIFTS[arm],
                    label,
                    ha="center",
                    va="bottom",
                    fontsize=7.5,
                )
        for position, fla_count in zip(positions, fla_counts, strict=True):
            if fla_count == 0:
                axis.text(
                    position,
                    ymax * 0.04,
                    "FLA unavailable",
                    ha="center",
                    va="bottom",
                    fontsize=7.5,
                    color="#6B7075",
                )
        axis.set_title(display_name(mode), fontsize=12, fontweight="bold")
        axis.set_xticks(positions, labels, rotation=25, ha="right")
        axis.set_ylim(0, ymax)
        axis.set_ylabel("Performance relative to FLA")
        axis.grid(axis="y", color="#D9DDE1", linewidth=0.7, alpha=0.8)
        axis.set_axisbelow(True)
        if not labels:
            axis.text(
                0.5,
                0.5,
                "No common FLA measurements",
                transform=axis.transAxes,
                ha="center",
                va="center",
            )

    handles, legend_labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.945),
        ncol=4,
        frameon=False,
    )
    figure.suptitle(
        title or "Linear-Attention Performance vs FLA Triton",
        fontsize=15,
        fontweight="bold",
        y=0.99,
    )
    figure.text(
        0.5,
        0.002,
        "Geometric mean across correct common shapes; higher is better.",
        ha="center",
        fontsize=9,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title")
    args = parser.parse_args()
    plot(load_result(args.input), args.output, args.title)


if __name__ == "__main__":
    main()
