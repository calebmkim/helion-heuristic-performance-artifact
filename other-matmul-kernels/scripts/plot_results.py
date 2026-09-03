#!/usr/bin/env python3
"""Plot per-family performance against raw default or torch.compile."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

FAMILY_LABELS = {
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--baseline",
        choices=("default", "torch_compile"),
        default="default",
    )
    args = parser.parse_args()

    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    summary = json.loads(args.summary.expanduser().resolve().read_text())
    ratio_key = (
        "seed_speedup"
        if args.baseline == "default"
        else "seed_speedup_vs_torch_compile"
    )
    families = [
        (family, group)
        for family, group in summary["families"].items()
        if group[ratio_key] is not None
    ]
    count_key = (
        "valid_cells"
        if args.baseline == "default"
        else "torch_compile_valid_cells"
    )
    labels = [
        f"{FAMILY_LABELS[family]}\n"
        f"{group[count_key]}/{group['cells']} shapes"
        for family, group in families
    ]
    overall = summary["overall"]
    labels.append(
        "Family geomean\n"
        f"{overall['valid_families'] if args.baseline == 'default' else overall['torch_compile_valid_families']}/"
        f"{overall['families']} families"
    )
    if args.baseline == "default":
        series = {
            "Heuristic seed": [
                *[float(group["seed_speedup"]) for _, group in families],
                float(overall["family_macro_geomean"]),
            ]
        }
        ylabel = "Performance relative to raw Helion default"
        title = "H100 Matmul Heuristic Seed Performance (raw Helion default = 1.00x)"
        baseline_label = "Raw Helion default = 1.00x"
    else:
        series = {
            "Helion raw default": [
                *[
                    float(group["default_speedup_vs_torch_compile"])
                    for _, group in families
                ],
                float(overall["default_vs_torch_compile_family_macro_geomean"]),
            ],
            "Helion heuristic seed": [
                *[
                    float(group["seed_speedup_vs_torch_compile"])
                    for _, group in families
                ],
                float(overall["seed_vs_torch_compile_family_macro_geomean"]),
            ],
        }
        ylabel = "Performance relative to Triton-only torch.compile"
        title = (
            "H100 Matmul Performance "
            "(Triton-only torch.compile max-autotune = 1.00x)"
        )
        baseline_label = "Triton-only torch.compile = 1.00x"
    x = np.arange(len(labels))

    figure, axis = plt.subplots(
        figsize=(19.0, 7.0),
        constrained_layout=True,
    )
    axis.axvspan(
        len(labels) - 1.5,
        len(labels) - 0.5,
        color="#F1F3F5",
        zorder=0,
    )
    axis.axhline(
        1.0,
        color="#333333",
        linewidth=1.2,
        linestyle="--",
        zorder=1,
    )
    colors = ("#9CA3AF", "#2878B5")
    width = 0.62 / len(series)
    for index, (name, values) in enumerate(series.items()):
        positions = x + (index - (len(series) - 1) / 2) * width
        bars = axis.bar(
            positions,
            values,
            width=width * 0.92,
            color=colors[index + (1 if len(series) == 1 else 0)],
            edgecolor="white",
            linewidth=0.6,
            label=name,
            zorder=2,
        )
        axis.bar_label(
            bars,
            labels=[
                f"{value:.2f}x" if math.isfinite(value) else "n/a"
                for value in values
            ],
            padding=3,
            fontsize=8.0,
        )
    axis.set_xticks(x, labels)
    axis.tick_params(axis="x", labelsize=8.5)
    axis.get_xticklabels()[-1].set_fontweight("bold")
    axis.set_ylabel(ylabel)
    axis.grid(axis="y", color="#D9DDE3", linewidth=0.8, alpha=0.8)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    finite = [
        value
        for values in series.values()
        for value in values
        if math.isfinite(value)
    ]
    axis.set_ylim(0, max(1.18, max(finite, default=1.0) * 1.16))
    axis.set_xlim(-0.65, len(labels) - 0.35)

    figure.suptitle(title, fontsize=15)
    figure.legend(
        handles=[
            *[
                Patch(facecolor=colors[index + (1 if len(series) == 1 else 0)], label=name)
                for index, name in enumerate(series)
            ],
            Line2D(
                [0],
                [0],
                color="#333333",
                linestyle="--",
                label=baseline_label,
            ),
        ],
        ncol=len(series) + 1,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
        frameon=False,
    )
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
