"""Result 2 figure: one cell's autotuning trajectory, no seeds vs compiler seeds.

Deliberately spare: two lines, two reference levels, two annotations. The prose
defines "within X%" and says what the arms are, so the figure does not repeat it.

Reads the committed trajectories in ../data/seeded-search/, so this reproduces
without access to the machine the search ran on. Each CSV is one autotuner run:
a row per attempt, with `perf_ms` empty when the attempt failed to compile or to
produce a timing. Best-so-far is the running minimum over `perf_ms`.

Cell: bf16 x int16 matmul, m=1 k=4096 n=4096 (off-corpus, not linear attention).
"""
import csv
import gzip
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

DATA = Path(__file__).resolve().parent.parent / "data" / "seeded-search"
CELL = "off__off_matmul_bf16xint16__bf16xint16_m1_k4096_n4096_754c795cfa"
ARMS = {"no seeds": DATA / "no-seed" / f"{CELL}.csv.gz",
        "compiler seeds": DATA / "expanded" / f"{CELL}.csv.gz"}

GREY, GREY_D, BLUE, EDGE = "#9CA3AF", "#6B7280", "#2563EB", "#1F2937"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})


def curve(path):
    best, out = None, []
    with gzip.open(path, "rt") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        if row["status"] == "started":
            continue
        try:
            v = float(row["perf_ms"]) if row["perf_ms"] else None
        except ValueError:
            v = None
        if v and v > 0 and (best is None or v < best):
            best = v
        out.append(best)
    return out


ns, ex = curve(ARMS["no seeds"]), curve(ARMS["compiler seeds"])
shared = min(b for b in ns + ex if b)
ns_y = [shared / b if b else 0 for b in ns]
ex_y = [shared / b if b else 0 for b in ex]

fig, ax = plt.subplots(figsize=(10.8, 4.9))
fig.suptitle("Seeding reaches a good config immediately, and the advantage is gone by the finish",
             fontsize=13, fontweight="bold", y=0.965)

# two reference levels, labelled short and inside
for frac, label in ((1 / 1.5, "within 50%"), (1 / 1.05, "within 5%")):
    ax.axhline(frac, color="#E5E7EB", lw=1.0, zorder=1)
    ax.text(len(ns) + 4, frac, f" {label}", fontsize=9, va="center", color=GREY_D)

ax.plot(range(1, len(ns) + 1), ns_y, lw=2.0, color=GREY, zorder=3, label="no seeds")
ax.plot(range(1, len(ex) + 1), ex_y, lw=2.4, color=BLUE, zorder=4, label="compiler seeds")

# one annotation for the early gap, in the colour of the arm it belongs to
ax.annotate("", xy=(1, 1 / 1.5), xytext=(102, 1 / 1.5),
            arrowprops=dict(arrowstyle="<|-|>", color=BLUE, lw=1.6, shrinkA=0, shrinkB=0))
ax.text(52, 0.615, "1 config vs 102", ha="center", va="top",
        fontsize=11, fontweight="bold", color=BLUE)

# one annotation for where it evaporates
ax.annotate("both within 5% by ~130 configs",
            xy=(133, 1 / 1.05), xytext=(215, 0.80),
            arrowprops=dict(arrowstyle="-|>", color=EDGE, lw=1.3,
                            connectionstyle="arc3,rad=-0.25"),
            fontsize=11, fontweight="bold", color=EDGE, ha="left", va="center")

ax.text(len(ns) * 0.62, 0.40, "then several hundred more configs\nchange nothing for either arm",
        ha="center", va="center", fontsize=10, style="italic", color=GREY_D)

ax.set_xlabel("configs benchmarked", fontsize=10.5)
ax.set_ylabel("best config so far\n(1.00 = best found)", fontsize=10.5)
ax.set_xlim(-8, len(ns) + 62)
ax.set_ylim(0, 1.06)
ax.legend(loc="lower right", fontsize=10, framealpha=1.0, edgecolor="#E5E7EB")
ax.set_axisbelow(True)
for spine in ("top", "right"):
    ax.spines[spine].set_visible(False)
ax.text(0.012, 0.025, "bf16 x int16 matmul, m=1 k=4096 n=4096, B200",
        transform=ax.transAxes, ha="left", fontsize=8, color="#C7CBD1")
fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.savefig("results-seed-trajectory.png", dpi=170, bbox_inches="tight", facecolor="white")
print("wrote results-seed-trajectory.png")
