#!/usr/bin/env python3
"""Plot per-kernel geomeans normalized to the report's reference arm."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


LABELS = {
    "default": "Default",
    "seed": "Seed",
    "aot_tuned": "AOT tuned",
    "vllm_cuda": "vLLM CUDA",
}
COLORS = {
    "default": "#8A8F98",
    "seed": "#2878B5",
    "aot_tuned": "#D98C2B",
    "vllm_cuda": "#3B9B6D",
}
KERNEL_LABELS = {
    "dynamic_per_token_scaled_fp8_quant": "Dynamic per-token\nFP8 quant",
    "per_token_group_fp8_quant": "Per-token group\nFP8 quant",
    "rms_norm_dynamic_per_token_quant": "RMSNorm dynamic\nFP8 quant",
    "rms_norm_per_block_quant": "RMSNorm block\nFP8 quant",
    "silu_and_mul_per_block_quant": "SiLU-and-mul\nblock quant",
    "fused_qk_norm_rope": "Fused QK norm\nRoPE",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import matplotlib.pyplot as plt
    import numpy as np

    summary = json.loads(args.summary.expanduser().resolve().read_text())
    arms = tuple(summary["arms"])
    plotted_arms = tuple(arm for arm in arms if arm != "vllm_cuda")
    reference_arm = summary["reference_arm"]
    kernels = list(summary["kernels"])
    x = np.arange(len(kernels))
    width = 0.76 / len(plotted_arms)

    figure, axis = plt.subplots(figsize=(13.5, 6.5), constrained_layout=True)
    observed: list[float] = []
    for arm_index, arm in enumerate(plotted_arms):
        values = [
            summary["kernels"][kernel]["normalized_performance"][arm]
            for kernel in kernels
        ]
        plotted = [float(value) if value is not None else math.nan for value in values]
        observed.extend(value for value in plotted if math.isfinite(value))
        offset = (arm_index - (len(plotted_arms) - 1) / 2) * width
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
    for kernel in kernels:
        group = summary["kernels"][kernel]
        name = KERNEL_LABELS.get(kernel, kernel.replace("_", " "))
        labels.append(f"{name}\n{group['valid_cells']}/{group['cells']} cells")
    axis.set_xticks(x, labels)
    axis.set_ylabel("Normalized performance (higher is faster)")
    axis.set_title(
        f"vLLM Reduction Kernels - {summary['profile']} profile "
        f"({LABELS[reference_arm]} = 1.00x)"
    )
    axis.axhline(1.0, color="#333333", linewidth=1.0, linestyle="--", zorder=0)
    axis.grid(axis="y", color="#D9DDE3", linewidth=0.8, alpha=0.8)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.legend(
        ncol=len(plotted_arms),
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
    )
    axis.set_ylim(0, max(1.15, max(observed, default=1.0) * 1.22))

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
