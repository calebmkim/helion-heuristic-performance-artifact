"""Reproduce the matmul figure for the compile-time-heuristics blog post.

This follows the visual conventions used by the linear-attention and reduction
blog figures: Helion arms are labeled by autotuning cost, the baseline is named
on the line, legend, and axis, and the final group is the family
macro-geometric mean.

Usage:
    python plot_blog_figures.py \
        --summary ../generated/h100-eacfee67/results.json \
        --gpu H100 --outdir <where>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

KEYS = ("default", "seed")
NICE = {
    "plain_matmul": "Plain BF16\nmatmul",
    "bf16xint16_gemm": "BF16 x INT16\nGEMM",
    "broadcast_matmul": "Broadcast\nmatmul",
    "gather_gemv": "Gather\nGEMV",
    "mamba2_chunk_state": "Mamba-2\nchunk state",
    "dense_attention": "Dense attention\nforward",
    "causal_attention": "Causal attention\nforward",
    "biased_attention": "Biased attention\nforward",
    "attention_backward": "Attention\nbackward",
    "jagged_hstu": "Jagged HSTU\nattention",
    "squeeze_excitation": "Squeeze-and-\nexcitation",
    "gdn_forward_h": "GDN\nforward-H",
    "mamba2_chunk_scan": "Mamba-2\nchunk scan",
}

GREY, BLUE, EDGE = "#9CA3AF", "#2563EB", "#1F2937"
STYLES = {"default": GREY, "seed": BLUE}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})


def build(summary_path: Path, gpu: str, output: Path, baseline: str) -> None:
    data: dict[str, Any] = json.loads(summary_path.read_text())
    seed_key = (
        "seed_speedup"
        if baseline == "default"
        else "seed_speedup_vs_torch_compile"
    )
    families = [
        (family, group)
        for family, group in data["families"].items()
        if group[seed_key] is not None
    ]
    if baseline == "default":
        per = {
            family: {
                "default": 1.0,
                "seed": float(group["seed_speedup"]),
            }
            for family, group in families
        }
        per["__overall__"] = {
            "default": 1.0,
            "seed": float(data["overall"]["family_macro_geomean"]),
        }
        count_key = "valid_cells"
        valid_families = data["overall"]["valid_families"]
        baseline_name = "Helion raw default"
        title = (
            f"Matmul kernels on {gpu}: a config computed at compile time is "
            f"{per['__overall__']['seed']:.2f}x faster than the raw default"
        )
        ylabel = "speedup vs raw Helion default"
    else:
        per = {
            family: {
                "default": float(group["default_speedup_vs_torch_compile"]),
                "seed": float(group["seed_speedup_vs_torch_compile"]),
            }
            for family, group in families
        }
        per["__overall__"] = {
            "default": float(
                data["overall"][
                    "default_vs_torch_compile_family_macro_geomean"
                ]
            ),
            "seed": float(
                data["overall"]["seed_vs_torch_compile_family_macro_geomean"]
            ),
        }
        count_key = "torch_compile_valid_cells"
        valid_families = data["overall"]["torch_compile_valid_families"]
        baseline_name = "Triton-only torch.compile max-autotune"
        title = (
            f"Matmul kernels on {gpu}: the compile-time heuristic runs at "
            f"{per['__overall__']['seed']:.2f}x the PyTorch reference performance"
        )
        ylabel = "speedup vs Triton-only torch.compile"
    counts = {
        family: (group[count_key], group["cells"])
        for family, group in families
    }
    groups = [*(family for family, _ in families), "__overall__"]

    figure, axis = plt.subplots(figsize=(18.0, 6.2))
    figure.suptitle(
        title,
        fontsize=14,
        fontweight="bold",
        y=0.985,
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
                value + 0.20,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=7.7,
                color=EDGE,
                zorder=4,
                fontweight="bold" if group == "__overall__" else "normal",
            )

    axis.axhline(1.0, color=EDGE, lw=1.4, ls="--", zorder=2)
    axis.set_xticks(range(len(groups)))
    axis.set_xticklabels(
        [
            (
                f"{NICE[group]}\n({counts[group][0]}/{counts[group][1]})"
                if group != "__overall__"
                else "Overall family\ngeomean\n"
                f"({data['overall']['valid_families']} families)"
            )
            for group in groups
        ],
        fontsize=8.2,
    )
    axis.get_xticklabels()[-1].set_fontweight("bold")
    axis.set_ylabel(
        f"{ylabel}\n"
        f"(higher is faster; {baseline_name} = 1.00x)",
        fontsize=9.5,
    )
    maximum = max(value for group in groups for value in per[group].values())
    axis.set_ylim(0, maximum * 1.18)
    axis.set_xlim(-0.6, len(groups) + 0.82)
    axis.text(
        len(groups) - 0.28,
        1.0,
        f"  {baseline_name}\n  = 1.00x (baseline)",
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

    figure.legend(
        handles=[
            Patch(
                facecolor=GREY,
                edgecolor=EDGE,
                label="Helion raw default - no heuristic, no autotuning",
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
                label=f"baseline: {baseline_name} = 1.00x",
            ),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        ncol=2,
        frameon=False,
        fontsize=9.4,
    )
    figure.text(
        0.5,
        0.005,
        "CUDA device time; cold L2; no CUDA Graph. Family bars are geometric "
        "means across correct shapes; overall gives each family equal weight."
        + (
            " PyTorch references may differ algorithmically from Helion."
            if baseline == "torch_compile"
            else ""
        ),
        ha="center",
        fontsize=8.4,
        style="italic",
        color="#4B5563",
    )
    figure.tight_layout(rect=(0, 0.035, 1, 0.885))
    figure.savefig(
        output,
        dpi=165,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(figure)

    print(
        f"{gpu} overall family geomean "
        f"(n={valid_families}): "
        f"{{'default': {per['__overall__']['default']:.4f}, "
        f"'seed': {per['__overall__']['seed']:.4f}}}"
    )
    print(f"wrote {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--gpu", default="H100")
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument(
        "--baseline",
        choices=("default", "torch_compile"),
        default="default",
    )
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    build(
        args.summary,
        args.gpu,
        args.outdir
        / (
            f"results-matmul-{args.gpu.lower()}.png"
            if args.baseline == "default"
            else f"results-matmul-vs-torch-compile-{args.gpu.lower()}.png"
        ),
        args.baseline,
    )


if __name__ == "__main__":
    main()
