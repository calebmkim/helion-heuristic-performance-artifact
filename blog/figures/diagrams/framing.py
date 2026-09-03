"""Two framing diagrams for the front of the post.

D: the tradeoff space -- schematic, deliberately not to scale.
E: what one heuristic run produces, and the two things that consume it.

Colours match the arms in the results charts (grey = default, blue = heuristic,
purple = autotuned) so a reader connects the diagrams to the bar charts.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

GREY, GREY_D, EDGE = "#9CA3AF", "#6B7280", "#1F2937"
BLUE, BLUE_L = "#2563EB", "#DBEAFE"
PURPLE = "#7C3AED"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})


def tradeoff(path):
    fig, ax = plt.subplots(figsize=(10.2, 5.6))
    fig.suptitle("The tradeoff this changes", fontsize=15, fontweight="bold", y=0.965)

    # the corner that used to be empty
    ax.add_patch(Rectangle((0.04, 0.54), 0.38, 0.40, facecolor=BLUE_L, alpha=0.55,
                           edgecolor=BLUE, lw=1.2, ls=(0, (5, 3)), zorder=1))
    ax.text(0.23, 0.885, "nothing used to live here", ha="center", fontsize=11.5,
            fontweight="bold", color=BLUE, zorder=4)

    # the old frontier
    ax.add_patch(FancyArrowPatch((0.115, 0.28), (0.775, 0.775), arrowstyle="-",
                                 color=GREY, lw=1.6, ls=(0, (4, 4)), zorder=2))
    ax.text(0.52, 0.44, "the old choice", ha="center", fontsize=10.5, style="italic",
            color=GREY_D, zorder=5)

    # the move
    ax.add_patch(FancyArrowPatch((0.10, 0.30), (0.10, 0.645), arrowstyle="-|>",
                                 mutation_scale=22, color=BLUE, lw=2.8, zorder=5))

    for x, y, c, name, place in [
        (0.10, 0.26, GREY, "Helion default", "right"),
        (0.10, 0.68, BLUE, "Helion heuristic  (this post)", "right"),
        (0.80, 0.80, PURPLE, "Helion + full autotuning", "above"),
    ]:
        ax.scatter([x], [y], s=340, color=c, edgecolor=EDGE, linewidth=1.6, zorder=6)
        if place == "right":
            ax.text(x + 0.038, y, name, ha="left", va="center", fontsize=12,
                    fontweight="bold", color=EDGE, zorder=6)
        else:
            ax.text(x, y + 0.075, name, ha="center", va="center", fontsize=12,
                    fontweight="bold", color=EDGE, zorder=6)

    ax.set_xlabel("GPU time spent tuning  \u2192", fontsize=12, labelpad=8)
    ax.set_ylabel("kernel performance  \u2192", fontsize=12, labelpad=8)
    ax.set_xticks([0.10, 0.45, 0.80])
    ax.set_xticklabels(["none", "minutes per shape", "hours to days per library"], fontsize=10)
    ax.set_yticks([])
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.10, 1.0)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GREY_D)
    ax.text(0.995, 0.115, "schematic", ha="right", va="bottom", fontsize=8,
            style="italic", color="#D1D5DB")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(path, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fork(path):
    fig, ax = plt.subplots(figsize=(12.4, 4.5))
    fig.suptitle("One analysis, two things that consume it", fontsize=14.5,
                 fontweight="bold", y=0.96)

    def box(x, y, w, h, title, sub, fc="white", ec=EDGE, tc=EDGE, lw=1.5, ts=11):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                                    boxstyle="round,pad=0,rounding_size=0.014",
                                    facecolor=fc, edgecolor=ec, linewidth=lw))
        ax.text(x + w / 2, y + h * (0.63 if sub else 0.5), title, ha="center", va="center",
                fontsize=ts, fontweight="bold", color=tc)
        if sub:
            ax.text(x + w / 2, y + h * 0.28, sub, ha="center", va="center",
                    fontsize=8.8, color=GREY_D)

    box(0.005, 0.40, 0.175, 0.26, "compiler facts", "liveness, extents,\nloops, shared knobs",
        fc="#F3F4F6", ec=GREY_D)
    box(0.215, 0.40, 0.155, 0.26, "heuristic", "closed-form\nhardware budgets", fc="#F9FAFB")

    box(0.405, 0.68, 0.175, 0.24, "one answer", None, fc=BLUE, tc="white", ec=EDGE)
    box(0.405, 0.14, 0.175, 0.24, "+ 5-20 variations", None, fc="#C7D2FE", ec=EDGE)

    box(0.635, 0.66, 0.36, 0.28, "what Helion emits with autotuning OFF",
        "the config you get for free", fc="#EFF6FF", ec=BLUE, tc=BLUE, lw=1.8, ts=11.5)
    box(0.635, 0.12, 0.36, 0.28, "the autotuner's initial population",
        "when autotuning is ON", fc="#F5F3FF", ec=PURPLE, tc=PURPLE, lw=1.8, ts=11.5)

    for a, b in (((0.185, 0.53), (0.208, 0.53)),
                 ((0.375, 0.53), (0.400, 0.80)),
                 ((0.375, 0.53), (0.400, 0.26)),
                 ((0.585, 0.80), (0.630, 0.80)),
                 ((0.585, 0.26), (0.630, 0.26)),
                 ((0.4925, 0.675), (0.4925, 0.385))):
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=15,
                                     color=EDGE, lw=1.5,
                                     connectionstyle="arc3,rad=0"))
    ax.text(0.503, 0.53, "the answer goes in\nthe population too", ha="left", va="center",
            fontsize=8.4, style="italic", color=GREY_D)
    ax.text(1.007, 0.80, "Result 1", ha="left", va="center", fontsize=10.5,
            fontweight="bold", color=BLUE)
    ax.text(1.007, 0.26, "Result 2", ha="left", va="center", fontsize=10.5,
            fontweight="bold", color=PURPLE)
    ax.set_xlim(0, 1.10)
    ax.set_ylim(0.05, 1.0)
    ax.axis("off")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)


tradeoff("diagram-D-tradeoff-space.png")
fork("diagram-E-two-consumers.png")
print("wrote D + E")
