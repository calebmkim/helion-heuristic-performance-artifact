"""Reproduce the linear-attention figures used in the compile-time-heuristics blog post.

Differs from ``plot_results.py`` in three ways, all of them about the blog's argument
rather than about the data:

* arms are labelled by whether they required autotuning, and the autotuned arm is
  hatched, so a reader can see at a glance which bars cost zero GPU seconds;
* everything is normalized to handwritten FLA Triton at 1.00x, and the baseline is
  named on the line, in the legend and on the axis rather than only in prose;
* H100 and B200 go through one code path, so the two figures are directly comparable.

The B200 end-to-end run uses a different result schema (``records`` with ``timings``
and arms ``pre_change`` / ``post_change`` / ``pretuned``) than this artifact's
``cells`` / ``arms`` layout, so both are normalized to a common shape first.

Usage:
    python plot_blog_figures.py \
        --h100 ../generated/h100-same-process/results.json \
        --b200 /path/to/e2e_fla_v3/results.json \
        --outdir ../generated/blog-figures

Either --h100 or --b200 may be omitted; the summary figure needs both.
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

# Blog arm names -> (this artifact's arm key, the B200 e2e run's arm key)
ARMS = {"default": ("default", "pre_change"),
        "heuristic": ("seed", "post_change"),
        "aot": ("aot_tuned", "pretuned")}
KEYS = ("default", "heuristic", "aot")

ORDER = ("vanilla_linear_attn", "simple_gla", "retention", "full_gla",
         "delta_rule", "gated_delta_rule", "kda", "kda_fused", "kda_varlen")
NICE = {"vanilla_linear_attn": "Vanilla\nlinear attn", "simple_gla": "Simple\nGLA",
        "retention": "Retention", "full_gla": "Full\nGLA", "delta_rule": "Delta\nrule",
        "gated_delta_rule": "Gated\ndelta rule", "kda": "KDA",
        "kda_fused": "KDA\nfused", "kda_varlen": "KDA\nvarlen"}

GREY, BLUE, PURPLE, EDGE = "#9CA3AF", "#2563EB", "#7C3AED", "#1F2937"
STYLES = {"default": (GREY, None), "heuristic": (BLUE, None), "aot": (PURPLE, "///")}
FLA_VERSION = "0.5.2"

LEGEND = [
    Patch(facecolor=GREY, edgecolor=EDGE, label="Helion default  —  no autotuning"),
    Patch(facecolor=BLUE, edgecolor=EDGE,
          label="Helion heuristic  —  no autotuning  (this post)"),
    Patch(facecolor=PURPLE, edgecolor=EDGE, hatch="///",
          label="Helion AOT config  —  full autotuning (hours of GPU time)"),
    Line2D([0], [0], color=EDGE, lw=1.6, ls="--",
           label=f"baseline: handwritten FLA Triton {FLA_VERSION} = 1.00x"),
]

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})


def geo(values: list[float]) -> float | None:
    vals = [v for v in values if v and v > 0]
    if not vals:
        return None
    return math.exp(sum(math.log(v) for v in vals) / len(vals))


def _load(path: Path) -> dict[str, Any]:
    with open(path) as handle:
        return json.load(handle)


def load_cells_schema(path: Path) -> dict[tuple[str, str], list[dict[str, float]]]:
    """This artifact's layout: ``cells`` each carrying an ``arms`` map of latencies."""
    out: dict[tuple[str, str], list[dict[str, float]]] = {}
    for cell in _load(path)["cells"]:
        arms = cell["arms"]
        needed = ["fla_triton"] + [a for a, _ in ARMS.values()]
        if any(arms.get(k, {}).get("status") != "ok" for k in needed):
            continue
        fla = arms["fla_triton"]["latency_ms"]
        out.setdefault((cell["mode"], cell["variant"]), []).append(
            {k: fla / arms[ARMS[k][0]]["latency_ms"] for k in KEYS})
    return out


def load_records_schema(path: Path) -> dict[tuple[str, str], list[dict[str, float]]]:
    """The B200 e2e layout: ``records`` carrying a precomputed ``speedup_vs_fla`` map."""
    out: dict[tuple[str, str], list[dict[str, float]]] = {}
    for rec in _load(path)["records"]:
        speedup = rec.get("speedup_vs_fla") or {}
        if not speedup:
            continue
        cell = rec["cell"]
        out.setdefault((cell["mode"], cell["variant"]), []).append(
            {k: speedup[ARMS[k][1]] for k in KEYS})
    return out


def load_any(path: Path) -> dict[tuple[str, str], list[dict[str, float]]]:
    data = _load(path)
    return load_cells_schema(path) if "cells" in data else load_records_schema(path)


def _panel(ax, data, mode: str, title: str, note: str):
    variants = [v for v in ORDER if (mode, v) in data]
    per = {v: {k: geo([c[k] for c in data[(mode, v)]]) for k in KEYS} for v in variants}
    per["__overall__"] = {k: geo([per[v][k] for v in variants]) for k in KEYS}
    groups = [*variants, "__overall__"]
    ncells = sum(len(data[(mode, v)]) for v in variants)

    width = 0.26
    for gi, group in enumerate(groups):
        if group == "__overall__":
            ax.axvspan(gi - 0.5, gi + 0.5, color="#F3F4F6", zorder=0)
        for ki, key in enumerate(KEYS):
            value = per[group][key]
            face, hatch = STYLES[key]
            ax.bar(gi + (ki - 1) * width, value, width * 0.93, color=face, edgecolor=EDGE,
                   linewidth=1.0, hatch=hatch, zorder=3)
            ax.text(gi + (ki - 1) * width, value + 0.035, f"{value:.2f}", ha="center",
                    va="bottom", fontsize=7.6, color=EDGE, zorder=4,
                    fontweight="bold" if group == "__overall__" else "normal")

    ax.axhline(1.0, color=EDGE, lw=1.4, ls="--", zorder=2)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([NICE.get(g, "Overall\ngeomean") for g in groups], fontsize=8.4)
    for label, group in zip(ax.get_xticklabels(), groups):
        if group == "__overall__":
            label.set_fontweight("bold")
    ax.set_ylabel("speedup vs handwritten FLA Triton\n(higher is faster; FLA = 1.00x)",
                  fontsize=9)
    ax.set_title(f"{title}   —   {ncells} cells", fontsize=11.5, fontweight="bold", pad=8)
    ax.set_ylim(0, max(max(per[g].values()) for g in groups) * 1.20)
    ax.set_xlim(-0.6, len(groups) + 0.42)
    ax.text(len(groups) - 0.30, 1.0,
            f"  handwritten FLA Triton {FLA_VERSION}\n  = 1.00x  (baseline)",
            ha="left", va="center", fontsize=8.6, fontweight="bold", color=EDGE, zorder=6,
            bbox=dict(boxstyle="round,pad=0.28", facecolor="white",
                      edgecolor="#D1D5DB", lw=0.9))
    ax.grid(axis="y", color="#E5E7EB", lw=0.8, zorder=1)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    if note:
        ax.text(0.5, -0.30, note, transform=ax.transAxes, ha="center", va="top",
                fontsize=8.4, style="italic", color="#4B5563")
    return per["__overall__"], ncells


def per_gpu_figure(data, gpu: str, path: Path, note: str = ""):
    fig, axes = plt.subplots(2, 1, figsize=(13.0, 9.2))
    fig.suptitle(f"End-to-end linear attention on {gpu}: a config computed at compile "
                 "time beats handwritten Triton",
                 fontsize=14, fontweight="bold", y=0.985)
    fwd = _panel(axes[0], data, "forward", "Forward", "")
    bwd = _panel(axes[1], data, "forward_backward", "Forward + backward", note)
    fig.legend(handles=LEGEND, loc="upper center", bbox_to_anchor=(0.5, 0.955),
               ncol=2, frameon=False, fontsize=9.4)
    fig.tight_layout(rect=(0, 0.02, 1, 0.905))
    fig.subplots_adjust(hspace=0.44)
    fig.savefig(path, dpi=165, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"{gpu} forward (n={fwd[1]}): " +
          str({k: round(v, 4) for k, v in fwd[0].items()}))
    print(f"{gpu} fwd+bwd (n={bwd[1]}): " +
          str({k: round(v, 4) for k, v in bwd[0].items()}))
    return fwd, bwd


def summary_figure(h100, b200, path: Path):
    fig, ax = plt.subplots(figsize=(11.6, 5.0))
    fig.suptitle("A config computed at compile time, on both parts",
                 fontsize=14, fontweight="bold", y=0.98)
    groups = [("H100\nforward", *h100[0]), ("H100\nfwd + bwd", *h100[1]),
              ("B200\nforward", *b200[0]), ("B200\nfwd + bwd", *b200[1])]
    width = 0.26
    for gi, (_, values, _n) in enumerate(groups):
        for ki, key in enumerate(KEYS):
            value = values[key]
            face, hatch = STYLES[key]
            ax.bar(gi + (ki - 1) * width, value, width * 0.93, color=face, edgecolor=EDGE,
                   linewidth=1.1, hatch=hatch, zorder=3)
            ax.text(gi + (ki - 1) * width, value + 0.03, f"{value:.2f}x", ha="center",
                    va="bottom", fontsize=9.6, fontweight="bold", color=EDGE, zorder=4)
    ax.axhline(1.0, color=EDGE, lw=1.5, ls="--", zorder=2)
    ax.text(3.66, 1.0, f"  handwritten FLA Triton {FLA_VERSION}\n  = 1.00x  (baseline)",
            ha="left", va="center", fontsize=8.4, fontweight="bold", color=EDGE, zorder=6,
            bbox=dict(boxstyle="round,pad=0.26", facecolor="white",
                      edgecolor="#D1D5DB", lw=0.9))
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([f"{g[0]}\n({g[2]} cells)" for g in groups], fontsize=10)
    ax.set_ylabel("speedup vs handwritten FLA Triton\n(higher is faster; FLA = 1.00x)",
                  fontsize=10)
    ax.set_ylim(0, 1.95)
    ax.set_xlim(-0.6, 4.62)
    ax.grid(axis="y", color="#E5E7EB", lw=0.8, zorder=1)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.legend(handles=LEGEND, loc="upper center", bbox_to_anchor=(0.5, 0.935),
               ncol=1, frameon=False, fontsize=9.4)
    fig.text(0.5, 0.005, "Same kernel source in all three Helion arms; only the config "
             "differs. Geomean over per-variant geomeans.",
             ha="center", fontsize=8.8, style="italic", color="#4B5563")
    fig.tight_layout(rect=(0, 0.03, 1, 0.80))
    fig.savefig(path, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)


H100_NOTE = ("Helion arms all run the same kernel source; only the config differs. "
             "The default arm is Helion's unseeded base config -- no heuristic fired.")
B200_NOTE = ("The default arm here is Helion's previous compiler-selected default, not the "
             "unseeded base config. Delta-rule and gated-delta-rule backward remain FLA wins.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h100", type=Path, help="results.json from an H100 run")
    parser.add_argument("--b200", type=Path, help="results.json from a B200 run")
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()
    if not args.h100 and not args.b200:
        parser.error("pass at least one of --h100 / --b200")
    args.outdir.mkdir(parents=True, exist_ok=True)

    results = {}
    for gpu, path, note in (("H100", args.h100, H100_NOTE), ("B200", args.b200, B200_NOTE)):
        if not path:
            continue
        out = args.outdir / f"results-linattn-{gpu.lower()}.png"
        results[gpu] = per_gpu_figure(load_any(path), gpu, out, note)
        print(f"wrote {out}")
    if len(results) == 2:
        out = args.outdir / "results-linattn-summary.png"
        summary_figure(results["H100"], results["B200"], out)
        print(f"wrote {out}")
    else:
        print("summary figure skipped: needs both --h100 and --b200")


if __name__ == "__main__":
    main()
