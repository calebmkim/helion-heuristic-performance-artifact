#!/usr/bin/env python3
"""Plot example-reduction geomeans relative to Torch compile."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


ARMS = ("default", "seed")
LABELS = {
    "default": "Default",
    "seed": "Seed",
}
COLORS = {
    "default": "#8A8F98",
    "seed": "#2878B5",
}
KERNEL_LABELS = {
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import matplotlib.pyplot as plt
    import numpy as np

    summary = json.loads(args.summary.expanduser().resolve().read_text())
    environment = summary.get("environment", {})
    torch_version = environment.get("torch", "unknown")
    triton_version = environment.get("triton", "unknown")
    kernels = list(summary["kernels"])
    groups = [*kernels, "__overall__"]
    x = np.arange(len(groups))
    width = 0.34
    figure, axis = plt.subplots(figsize=(15.5, 6.5), constrained_layout=True)
    axis.axvspan(
        len(groups) - 1.5,
        len(groups) - 0.5,
        color="#F3F4F6",
        zorder=0,
    )
    axis.axhline(
        1.0,
        color="#333333",
        linewidth=1.0,
        linestyle="--",
        label="torch.compile max-autotune = 1.00x",
        zorder=1,
    )
    observed: list[float] = []
    for arm_index, arm in enumerate(ARMS):
        values = [
            (
                summary["overall"]["normalized_performance"][arm]
                if group == "__overall__"
                else summary["kernels"][group]["normalized_performance"][arm]
            )
            for group in groups
        ]
        plotted = [float(value) if value is not None else math.nan for value in values]
        observed.extend(value for value in plotted if math.isfinite(value))
        offset = (arm_index - 0.5) * width
        bars = axis.bar(
            x + offset,
            plotted,
            width,
            label=LABELS[arm],
            color=COLORS[arm],
            edgecolor="white",
            linewidth=0.5,
        )
        axis.bar_label(
            bars,
            labels=[
                f"{value:.2f}" if math.isfinite(value) else "n/a"
                for value in plotted
            ],
            padding=2,
            fontsize=8,
            rotation=90,
        )

    labels = []
    for group_name in groups:
        if group_name == "__overall__":
            group = summary["overall"]
            labels.append(
                f"Geomean\n{group['valid_cells']}/{group['cells']} cells"
            )
        else:
            group = summary["kernels"][group_name]
            labels.append(
                f"{KERNEL_LABELS.get(group_name, group_name)}\n"
                f"{group['valid_cells']}/{group['cells']} cells"
            )
    axis.set_xticks(x, labels)
    axis.get_xticklabels()[-1].set_fontweight("bold")
    axis.set_ylabel("Normalized performance (higher is faster)")
    axis.set_title(
        "Example Reduction Kernels - Liger-Mixed Profile "
        "(torch.compile max-autotune = 1.00x)\n"
        f"Torch {torch_version} / Triton {triton_version}"
    )
    axis.grid(axis="y", color="#D9DDE3", linewidth=0.8, alpha=0.8)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.0))
    axis.set_ylim(0, max(1.15, max(observed, default=1.0) * 1.22))
    axis.set_xlim(-0.6, len(groups) - 0.4)

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
