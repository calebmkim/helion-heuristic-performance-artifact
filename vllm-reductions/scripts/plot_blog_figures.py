"""Reproduce the vLLM-reduction figure used in the compile-time-heuristics blog post.

Same conventions as the linear-attention blog figure: arms are labelled by whether they
required autotuning, the autotuned arm is hatched, and the baseline is named on the line,
in the legend and on the axis rather than only in prose.

The baseline here is vLLM's own handwritten CUDA/C++ kernel, not a Helion config, so this
is a comparison against production code rather than against another Helion arm.

Usage:
    python plot_blog_figures.py \
        --summary ../generated/h100-pr3551-curated/summary.json \
        --gpu H100 --outdir <where>
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

KEYS = ("default", "seed", "aot_tuned")
ORDER = ("dynamic_per_token_scaled_fp8_quant", "per_token_group_fp8_quant",
         "rms_norm_dynamic_per_token_quant", "rms_norm_per_block_quant",
         "silu_and_mul_per_block_quant", "fused_qk_norm_rope")
NICE = {
    "dynamic_per_token_scaled_fp8_quant": "Dynamic\nper-token\nFP8 quant",
    "per_token_group_fp8_quant": "Per-token-group\nFP8 quant",
    "rms_norm_dynamic_per_token_quant": "RMSNorm +\ndynamic\nper-token quant",
    "rms_norm_per_block_quant": "RMSNorm +\nper-block quant",
    "silu_and_mul_per_block_quant": "SiLU-and-mul +\nper-block quant",
    "fused_qk_norm_rope": "Fused QK-norm\n+ RoPE",
}
GREY, BLUE, PURPLE, EDGE = "#9CA3AF", "#2563EB", "#7C3AED", "#1F2937"
STYLES = {"default": (GREY, None), "seed": (BLUE, None), "aot_tuned": (PURPLE, "///")}

LEGEND = [
    Patch(facecolor=GREY, edgecolor=EDGE, label="Helion default  —  no autotuning"),
    Patch(facecolor=BLUE, edgecolor=EDGE,
          label="Helion heuristic  —  no autotuning  (this post)"),
    Patch(facecolor=PURPLE, edgecolor=EDGE, hatch="///",
          label="Helion AOT config  —  full autotuning (hours of GPU time)"),
    Line2D([0], [0], color=EDGE, lw=1.6, ls="--",
           label="baseline: vLLM's own CUDA/C++ kernel = 1.00x"),
]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})


def geo(values):
    vals = [v for v in values if v and v > 0]
    return math.exp(sum(math.log(v) for v in vals) / len(vals)) if vals else None


def build(summary_path: Path, gpu: str, out: Path) -> None:
    data = json.loads(summary_path.read_text())
    cells = [c for c in data["cells"]
             if all(c["status"].get(k) == "ok" for k in (*KEYS, "vllm_cuda"))]

    by_kernel: dict[str, list[dict]] = {}
    for c in cells:
        by_kernel.setdefault(c["kernel"], []).append(c["normalized_performance"])
    kernels = [k for k in ORDER if k in by_kernel]
    per = {k: {a: geo([n[a] for n in by_kernel[k]]) for a in KEYS} for k in kernels}
    # Overall is the flat per-cell geomean, matching the artifact's own reported number:
    # the kernels have unequal cell counts, so a nested mean would not reproduce it.
    per["__overall__"] = {a: geo([c["normalized_performance"][a] for c in cells]) for a in KEYS}
    counts = {k: len(by_kernel[k]) for k in kernels}

    groups = [*kernels, "__overall__"]
    fig, ax = plt.subplots(figsize=(13.0, 5.6))
    fig.suptitle(f"vLLM's reduction kernels on {gpu}: a config computed at compile time "
                 "beats vLLM's handwritten CUDA",
                 fontsize=14, fontweight="bold", y=0.985)
    width = 0.26
    for gi, group in enumerate(groups):
        if group == "__overall__":
            ax.axvspan(gi - 0.5, gi + 0.5, color="#F3F4F6", zorder=0)
        for ki, key in enumerate(KEYS):
            value = per[group][key]
            face, hatch = STYLES[key]
            ax.bar(gi + (ki - 1) * width, value, width * 0.93, color=face, edgecolor=EDGE,
                   linewidth=1.0, hatch=hatch, zorder=3)
            ax.text(gi + (ki - 1) * width, value + 0.03, f"{value:.2f}", ha="center",
                    va="bottom", fontsize=8.2, color=EDGE, zorder=4,
                    fontweight="bold" if group == "__overall__" else "normal")
    ax.axhline(1.0, color=EDGE, lw=1.4, ls="--", zorder=2)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(
        [f"{NICE[g]}\n({counts[g]})" if g != "__overall__"
         else f"Overall\ngeomean\n({len(cells)} cells)" for g in groups], fontsize=8.4)
    for label, group in zip(ax.get_xticklabels(), groups):
        if group == "__overall__":
            label.set_fontweight("bold")
    ax.set_ylabel("speedup vs vLLM's own CUDA kernel\n(higher is faster; vLLM CUDA = 1.00x)",
                  fontsize=9.5)
    ax.set_ylim(0, max(max(per[g].values()) for g in groups) * 1.18)
    ax.set_xlim(-0.6, len(groups) + 0.55)
    ax.text(len(groups) - 0.32, 1.0, "  vLLM's own\n  CUDA/C++ kernel\n  = 1.00x  (baseline)",
            ha="left", va="center", fontsize=8.6, fontweight="bold", color=EDGE, zorder=6,
            bbox=dict(boxstyle="round,pad=0.28", facecolor="white",
                      edgecolor="#D1D5DB", lw=0.9))
    ax.grid(axis="y", color="#E5E7EB", lw=0.8, zorder=1)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.legend(handles=LEGEND, loc="upper center", bbox_to_anchor=(0.5, 0.955),
               ncol=2, frameon=False, fontsize=9.4)
    fig.text(0.5, 0.005, data["timing"], ha="center", fontsize=8.4, style="italic",
             color="#4B5563")
    fig.tight_layout(rect=(0, 0.035, 1, 0.885))
    fig.savefig(out, dpi=165, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"{gpu} overall (n={len(cells)}): " +
          str({k: round(v, 4) for k, v in per['__overall__'].items()}))
    print(f"wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--summary", type=Path, required=True)
    ap.add_argument("--gpu", default="H100")
    ap.add_argument("--outdir", type=Path, required=True)
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    build(args.summary, args.gpu,
          args.outdir / f"results-vllm-{args.gpu.lower()}.png")


if __name__ == "__main__":
    main()
