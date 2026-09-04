"""Reproduce the example-reduction figure for the heuristic blog post.

This follows the visual conventions used by the linear-attention and vLLM
reduction blog figures: Helion arms are labeled by autotuning cost, the external
baseline is named on the line, legend, and axis, and the final group is the
flat per-cell geomean.

Usage:
    python plot_blog_figures.py \
        --summary ../generated/h100-main-torch213-triton371-liger-mixed/summary.json \
        --gpu H100 --outdir <where>
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402


KEYS = ("default", "seed")
ORDER = (
    "rms_norm",
    "layer_norm",
    "softmax",
    "cross_entropy",
    "kl_div",
    "jsd",
    "fused_linear_jsd",
    "grpo",
    "rms_norm_bwd",
    "layer_norm_bwd",
)
NICE = {
    "rms_norm": "RMSNorm",
    "layer_norm": "LayerNorm",
    "softmax": "Softmax",
    "cross_entropy": "Cross\nentropy",
    "kl_div": "KL\ndivergence",
    "jsd": "JSD",
    "fused_linear_jsd": "Fused-example\nJSD",
    "grpo": "GRPO",
    "rms_norm_bwd": "RMSNorm\nbackward",
    "layer_norm_bwd": "LayerNorm\nbackward",
}

GREY, BLUE, EDGE = "#9CA3AF", "#2563EB", "#1F2937"
STYLES = {"default": GREY, "seed": BLUE}
LEGEND = [
    Patch(
        facecolor=GREY,
        edgecolor=EDGE,
        label="Helion default - no autotuning",
    ),
    Patch(
        facecolor=BLUE,
        edgecolor=EDGE,
        label="Helion heuristic - no autotuning (this post)",
    ),
    Line2D(
        [0],
        [0],
        color=EDGE,
        lw=1.6,
        ls="--",
        label="baseline: torch.compile max-autotune = 1.00x",
    ),
]

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})


def _geomean(values: list[float]) -> float | None:
    valid = [value for value in values if value > 0]
    if not valid:
        return None
    return math.exp(sum(math.log(value) for value in valid) / len(valid))


def build(summary_path: Path, gpu: str, output: Path) -> None:
    data: dict[str, Any] = json.loads(summary_path.read_text())
    environment = data.get("environment", {})
    torch_version = environment.get("torch", "unknown")
    triton_version = environment.get("triton", "unknown")
    cells = [
        cell
        for cell in data["cells"]
        if all(
            cell["status"].get(arm) == "ok"
            for arm in (*KEYS, "torch_compile")
        )
    ]

    by_kernel: dict[str, list[dict[str, float]]] = {}
    for cell in cells:
        by_kernel.setdefault(cell["kernel"], []).append(
            cell["normalized_performance"]
        )
    kernels = [kernel for kernel in ORDER if kernel in by_kernel]
    per = {
        kernel: {
            arm: _geomean(
                [values[arm] for values in by_kernel[kernel]]
            )
            for arm in KEYS
        }
        for kernel in kernels
    }
    per["__overall__"] = {
        arm: _geomean(
            [cell["normalized_performance"][arm] for cell in cells]
        )
        for arm in KEYS
    }
    counts = {kernel: len(by_kernel[kernel]) for kernel in kernels}
    groups = [*kernels, "__overall__"]

    figure, axis = plt.subplots(figsize=(13.0, 6.0))
    figure.suptitle(
        f"Example reduction kernels on {gpu}: a config computed at compile "
        "time matches torch.compile max-autotune",
        fontsize=13,
        fontweight="bold",
        y=0.985,
    )
    figure.text(
        0.5,
        0.93,
        f"Torch {torch_version} / Triton {triton_version}",
        ha="center",
        fontsize=11,
        fontweight="bold",
        color=EDGE,
    )

    width = 0.30
    for group_index, group in enumerate(groups):
        if group == "__overall__":
            axis.axvspan(
                group_index - 0.5,
                group_index + 0.5,
                color="#F3F4F6",
                zorder=0,
            )
        for arm_index, arm in enumerate(KEYS):
            value = per[group][arm]
            if value is None:
                continue
            x = group_index + (arm_index - 0.5) * width
            axis.bar(
                x,
                value,
                width * 0.93,
                color=STYLES[arm],
                edgecolor=EDGE,
                linewidth=1.0,
                zorder=3,
            )
            axis.text(
                x,
                value + 0.03,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=8.0,
                color=EDGE,
                zorder=4,
                fontweight="bold" if group == "__overall__" else "normal",
            )

    axis.axhline(1.0, color=EDGE, lw=1.4, ls="--", zorder=2)
    axis.set_xticks(range(len(groups)))
    axis.set_xticklabels(
        [
            (
                f"{NICE[group]}\n({counts[group]})"
                if group != "__overall__"
                else f"Overall\ngeomean\n({len(cells)} cells)"
            )
            for group in groups
        ],
        fontsize=8.4,
    )
    axis.get_xticklabels()[-1].set_fontweight("bold")
    axis.set_ylabel(
        "speedup vs torch.compile max-autotune\n"
        "(higher is faster; torch.compile = 1.00x)",
        fontsize=9.5,
    )
    maximum = max(
        value
        for group in groups
        for value in per[group].values()
        if value is not None
    )
    axis.set_ylim(0, maximum * 1.20)
    axis.set_xlim(-0.6, len(groups) + 0.70)
    axis.text(
        len(groups) - 0.28,
        1.0,
        "  torch.compile\n  max-autotune\n  = 1.00x (baseline)",
        ha="left",
        va="center",
        fontsize=8.5,
        fontweight="bold",
        color=EDGE,
        zorder=6,
        bbox={
            "boxstyle": "round,pad=0.28",
            "facecolor": "white",
            "edgecolor": "#D1D5DB",
            "linewidth": 0.9,
        },
    )
    axis.grid(axis="y", color="#E5E7EB", lw=0.8, zorder=1)
    axis.set_axisbelow(True)
    for spine in ("top", "right"):
        axis.spines[spine].set_visible(False)

    axis.legend(
        handles=LEGEND,
        loc="upper right",
        bbox_to_anchor=(0.96, 0.98),
        ncol=1,
        frameon=False,
        fontsize=8.6,
    )
    figure.text(
        0.5,
        0.005,
        "CUDA device time; geometric mean across correct common shapes. "
        "torch.compile mode: max-autotune-no-cudagraphs. "
        f"Torch {torch_version}; Triton {triton_version}.",
        ha="center",
        fontsize=8.4,
        style="italic",
        color="#4B5563",
    )
    figure.tight_layout(rect=(0, 0.035, 1, 0.84))
    figure.savefig(
        output,
        dpi=165,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(figure)

    overall = {
        arm: round(value, 4)
        for arm, value in per["__overall__"].items()
        if value is not None
    }
    print(f"{gpu} overall (n={len(cells)}): {overall}")
    print(f"wrote {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--gpu", default="H100")
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    build(
        args.summary,
        args.gpu,
        args.outdir / f"results-example-reductions-{args.gpu.lower()}.png",
    )


if __name__ == "__main__":
    main()
