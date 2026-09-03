"""Mechanism diagrams A1, A2, B, C -- in the same visual idiom as D and E.

Shared conventions: one bold title and nothing else at the top, a restrained
grey/blue/violet palette with light tints for containers, rounded boxes with thin
dark strokes, short bold labels instead of sentences, and no footer strips.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch

GREY, GREY_D, EDGE = "#9CA3AF", "#6B7280", "#1F2937"
BLUE, BLUE_L, BLUE_XL = "#2563EB", "#93C5FD", "#EFF6FF"
VIO, VIO_L, VIO_XL = "#7C3AED", "#C4B5FD", "#F5F3FF"
GREY_XL, GREY_L = "#F3F4F6", "#F9FAFB"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})


def rbox(ax, x, y, w, h, fc="white", ec=EDGE, lw=1.5, r=0.014, z=3):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle=f"round,pad=0,rounding_size={r}",
                                facecolor=fc, edgecolor=ec, linewidth=lw, zorder=z))


def title(fig, text, y=0.965, size=14.5):
    fig.suptitle(text, fontsize=size, fontweight="bold", y=y)


def save(fig, path, rect=(0, 0, 1, 0.92)):
    fig.tight_layout(rect=rect)
    fig.savefig(path, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ----------------------------------------------------------------- A1: liveness
CODE = ["for tile_m in hl.tile(M):",
        "    acc = hl.zeros([tile_m, D])",
        "    for tile_n in hl.tile(N):",
        "        q  = q_in[tile_m, :]",
        "        k  = k_in[tile_n, :]",
        "        qk = hl.dot(q, k.T)",
        "        p  = qk * scale",
        "        v  = v_in[tile_n, :]",
        "        acc = hl.dot(p, v, acc=acc)",
        "    out[tile_m, :] = acc"]
VALUES = [("acc", 1, 9, VIO_L), ("q", 3, 5, BLUE_L), ("k", 4, 5, BLUE_L),
          ("qk", 5, 6, VIO_L), ("p", 6, 8, VIO_L), ("v", 7, 8, BLUE_L)]
PEAK = 5


def fig_a1(path):
    n = len(CODE)
    fig, ax = plt.subplots(figsize=(12.4, 5.2))
    title(fig, "The compiler knows what is live at every step")

    def yof(i):
        return n - i

    ax.add_patch(Rectangle((0.005, yof(PEAK) - 0.36), 1.02, 0.72,
                           facecolor=BLUE_XL, edgecolor="none", zorder=0))
    for i, line in enumerate(CODE):
        ax.text(0.015, yof(i), line, fontsize=10.5, family="monospace",
                va="center", color="#111827", zorder=4)

    x0, w, gap = 0.505, 0.062, 0.017
    for j, (name, d, l, fill) in enumerate(VALUES):
        x = x0 + j * (w + gap)
        ax.text(x + w / 2, yof(-1) - 0.30, name, ha="center", fontsize=10,
                family="monospace", fontweight="bold", color=EDGE)
        rbox(ax, x, yof(l) - 0.28, w, yof(d) - yof(l) + 0.56, fc=fill, lw=1.1, r=0.012)

    ax.annotate("", xy=(1.03, yof(PEAK)), xytext=(1.10, yof(PEAK)),
                arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=2.2))
    ax.text(1.115, yof(PEAK) + 0.26, "peak live step", fontsize=11,
            fontweight="bold", color=BLUE, va="center")
    ax.text(1.115, yof(PEAK) - 0.32, "4 tiles at once", fontsize=9.5, color=BLUE, va="center")

    for k, (lbl, fill) in enumerate((("load", BLUE_L), ("dot output / carry", VIO_L))):
        y = 2.05 - k * 0.62
        rbox(ax, 1.115, y, 0.042, 0.34, fc=fill, lw=1.0, r=0.01)
        ax.text(1.172, y + 0.17, lbl, fontsize=9.5, va="center", color=GREY_D)

    ax.text(0.015, 0.62, "illustrative kernel: attention-shaped, softmax elided",
            fontsize=8.6, style="italic", color="#D1D5DB", va="center")
    ax.set_xlim(0.0, 1.40)
    ax.set_ylim(0.30, n + 1.20)
    ax.axis("off")
    save(fig, path)


# ------------------------------------------------------------------ A2: budgets
def fig_a2(path):
    fig, ax = plt.subplots(figsize=(11.8, 4.9))
    title(fig, "The scarcest budget decides how many CTAs fit")

    spaces = [("registers", 0.10, GREY_XL, GREY_D, [("p", VIO_L)]),
              ("shared memory", 0.10, GREY_XL, GREY_D, [("q tile", BLUE_L), ("k tile", BLUE_L)]),
              ("tensor memory", 0.60, VIO_L, VIO, [("qk accumulator", VIO_L),
                                                   ("output accumulator", VIO_L)])]
    bw = 0.155
    for i, (name, frac, fc, ec, holds) in enumerate(spaces):
        x = 0.05 + i * 0.225
        rbox(ax, x, 0.40, bw, 0.46, fc="white", ec=GREY, lw=1.2, r=0.012, z=2)
        rbox(ax, x, 0.40, bw, 0.46 * frac, fc=fc, ec=ec, lw=1.5, r=0.012, z=3)
        ax.text(x + bw / 2, 0.40 + 0.46 * frac + 0.025, f"{int(frac * 100)}%", ha="center",
                va="bottom", fontsize=12, fontweight="bold",
                color=VIO if frac > 0.5 else GREY_D)
        ax.text(x + bw / 2, 0.355, name, ha="center", va="top", fontsize=11.5,
                fontweight="bold", color=VIO if frac > 0.5 else EDGE)
        for j, (lbl, chip) in enumerate(holds):
            y = 0.265 - j * 0.072
            rbox(ax, x + 0.004, y, 0.026, 0.044, fc=chip, lw=0.9, r=0.008)
            ax.text(x + 0.040, y + 0.022, lbl, va="center", fontsize=9, color=GREY_D)

    ax.plot([0.03, 0.665], [0.865, 0.865], color=GREY, lw=1.2, ls=(0, (4, 3)), zorder=1)
    ax.text(0.03, 0.895, "budget", fontsize=9, color=GREY_D, va="bottom")

    ax.add_patch(FancyArrowPatch((0.675, 0.62), (0.735, 0.62), arrowstyle="-|>",
                                mutation_scale=16, color=EDGE, lw=1.6))
    rbox(ax, 0.748, 0.44, 0.235, 0.36, fc=VIO_XL, ec=VIO, lw=1.8, r=0.018)
    ax.text(0.8655, 0.705, "1 CTA per SM", ha="center", fontsize=14, fontweight="bold",
            color=VIO)
    ax.text(0.8655, 0.585, "tensor memory is the\nonly reason why", ha="center",
            fontsize=9.5, color=GREY_D, va="center")
    ax.text(0.8655, 0.485, "the other two would\nhave allowed ten", ha="center",
            fontsize=9.5, color=GREY_D, va="center")

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.10, 0.95)
    ax.axis("off")
    save(fig, path)


# ------------------------------------------------------------------- B: stages
def fig_b(path):
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.4))
    title(fig, "More pipeline stages fit -- by spending what a second CTA needed", y=0.975)
    panels = [("num_stages = 6", 1, 6, GREY, GREY_D, "1 CTA resident"),
              ("num_stages = 3", 2, 3, BLUE_L, BLUE, "2 CTAs resident")]
    for ax, (ttl, nctas, nstages, fc, ec, cap) in zip(axes, panels):
        ax.text(0.5, 1.09, ttl, ha="center", va="center", fontsize=12.5,
                fontweight="bold", color=EDGE)
        rbox(ax, 0.03, 0.20, 0.94, 0.76, fc="white", ec=EDGE, lw=1.6, r=0.022, z=2)
        ax.text(0.055, 0.915, "SM shared memory", fontsize=9, color=GREY_D, va="center")
        pad, gap = 0.028, 0.024
        cw = (0.94 - 2 * pad - gap * (nctas - 1)) / nctas
        for c in range(nctas):
            cx = 0.03 + pad + c * (cw + gap)
            rbox(ax, cx, 0.26, cw, 0.60, fc=GREY_L, ec=ec, lw=1.4, r=0.014, z=3)
            ax.text(cx + cw / 2, 0.815, f"CTA {c}", ha="center", va="center",
                    fontsize=10, fontweight="bold", color=ec, zorder=5)
            sh = 0.50 / nstages
            for s in range(nstages):
                rbox(ax, cx + 0.012, 0.275 + s * sh, cw - 0.024, sh * 0.84,
                     fc=fc, ec=EDGE, lw=0.9, r=0.006, z=4)
        rbox(ax, 0.03, -0.02, 0.94, 0.17, fc=GREY_XL, ec=EDGE, lw=1.3, r=0.016, z=2)
        ax.text(0.5, 0.065, cap, ha="center", va="center", fontsize=12,
                fontweight="bold", color=ec, zorder=5)
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.08, 1.18)
        ax.axis("off")
    save(fig, path, rect=(0, 0.01, 1, 0.90))


# -------------------------------------------------------------- C: shared knob
def fig_c(path):
    fig, ax = plt.subplots(figsize=(11.2, 4.3))
    title(fig, "In a multi-contraction kernel, one knob is two different axes")

    for x, tag, expr, role in ((0.045, "dot 0", "attn = q @ k.T", "this knob is its  N"),
                               (0.575, "dot 1", "out = attn @ v", "this knob is its  K")):
        rbox(ax, x, 0.60, 0.38, 0.27, fc="white", ec=EDGE, lw=1.5, r=0.016)
        ax.text(x + 0.19, 0.815, tag, ha="center", fontsize=10.5, fontweight="bold",
                color=GREY_D)
        ax.text(x + 0.19, 0.725, expr, ha="center", fontsize=11.5, family="monospace",
                color="#111827")
        ax.text(x + 0.19, 0.648, role, ha="center", fontsize=9.5, color=GREY_D)

    rbox(ax, 0.315, 0.345, 0.37, 0.135, fc=GREY_XL, ec=EDGE, lw=1.5, r=0.016)
    ax.text(0.5, 0.4125, "one block-size knob", ha="center", va="center",
            fontsize=12, fontweight="bold", color=EDGE)

    for x0, rad in ((0.235, 0.16), (0.765, -0.16)):
        ax.add_patch(FancyArrowPatch((x0, 0.595), (0.5, 0.487), arrowstyle="-|>",
                                     mutation_scale=14, color=GREY_D, lw=1.4,
                                     connectionstyle=f"arc3,rad={rad}", zorder=2))
    ax.text(0.115, 0.520, "wants 128", ha="center", fontsize=11, fontweight="bold", color=BLUE)
    ax.text(0.885, 0.520, "wants 32", ha="center", fontsize=11, fontweight="bold", color=BLUE)

    rbox(ax, 0.045, 0.05, 0.91, 0.215, fc=BLUE_XL, ec=BLUE, lw=1.7, r=0.018)
    ax.text(0.5, 0.202, "They cannot both win, so the higher-ranked dot sets it",
            ha="center", fontsize=11.5, fontweight="bold", color=BLUE)
    ax.text(0.5, 0.108, "loop-carried accumulator     →     dynamic matmul work     →     output area",
            ha="center", fontsize=10.5, family="monospace", color=GREY_D)

    ax.set_xlim(0, 1)
    ax.set_ylim(0.0, 0.90)
    ax.axis("off")
    save(fig, path)


fig_a1("diagram-A1-liveness.png")
fig_a2("diagram-A2-budgets.png")
fig_b("diagram-B-stages-vs-residency.png")
fig_c("diagram-C-shared-knob.png")
print("restyled A1, A2, B, C")
