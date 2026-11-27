from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import (
    FancyBboxPatch, Polygon, FancyArrowPatch
)

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "manuscript/figures"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "pdf.fonttype": 42,
    "svg.fonttype": "none"
})

fig, ax = plt.subplots(figsize=(11, 13))
ax.set_xlim(0, 12)
ax.set_ylim(0, 15)
ax.axis("off")

TEAL = "#009688"
BLUE = "#2166AC"
GRAY = "#64748B"
DARK = "#243447"

def box(x, y, w, h, text, color=GRAY,
        fs=9, bold=False, alpha=0.07):

    patch = FancyBboxPatch(
        (x-w/2, y-h/2), w, h,
        boxstyle="round,pad=0.04,rounding_size=0.09",
        facecolor=color,
        edgecolor=color,
        alpha=alpha,
        linewidth=1.2
    )
    ax.add_patch(patch)

    border = FancyBboxPatch(
        (x-w/2, y-h/2), w, h,
        boxstyle="round,pad=0.04,rounding_size=0.09",
        facecolor="none",
        edgecolor=color,
        linewidth=1.1
    )
    ax.add_patch(border)

    ax.text(
        x, y, text,
        ha="center", va="center",
        fontsize=fs, color=DARK,
        fontweight="bold" if bold else "normal"
    )

def diamond(x, y, w, h, text, color=TEAL):

    points = [
        (x, y+h/2),
        (x+w/2, y),
        (x, y-h/2),
        (x-w/2, y)
    ]

    ax.add_patch(Polygon(
        points,
        closed=True,
        facecolor="white",
        edgecolor=color,
        linewidth=1.2
    ))

    ax.text(
        x, y, text,
        ha="center", va="center",
        fontsize=9, color=DARK
    )

def arrow(x1, y1, x2, y2,
          label=None, dashed=False):

    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle="-|>",
        mutation_scale=11,
        linewidth=1.05,
        color=GRAY,
        linestyle="--" if dashed else "-",
        shrinkA=0, shrinkB=0
    ))

    if label:
        ax.text(
            (x1+x2)/2+0.08,
            (y1+y2)/2+0.08,
            label, fontsize=8,
            color=DARK
        )

def route(points, label=None, dashed=False):

    xs, ys = zip(*points[:-1])

    ax.plot(
        xs, ys,
        color=GRAY,
        linewidth=1.05,
        linestyle="--" if dashed else "-"
    )

    arrow(
        *points[-2], *points[-1],
        dashed=dashed
    )

    if label:
        x, y = points[0]
        ax.text(
            x+0.12, y+0.12,
            label, fontsize=8,
            color=DARK
        )

def section(x, y, w, h, title, color):

    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.03",
        facecolor=color,
        edgecolor=color,
        alpha=0.035
    ))

    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.03",
        facecolor="none",
        edgecolor=color,
        linewidth=0.9
    ))

    ax.text(
        x+0.18, y+h-0.22,
        title, fontsize=10,
        fontweight="bold",
        color=color, va="top"
    )

# I. Initialization
section(
    0.4, 13.45, 11.2, 1.3,
    "I. SCENARIO INITIALIZATION", GRAY
)

box(2.3, 13.95, 3.2, 0.55, "Robot fleet")
box(6.0, 13.95, 3.2, 0.55, "Task set")
box(9.7, 13.95, 3.2, 0.55,
    "Grid map and failure schedule")

for x in [2.3, 6.0, 9.7]:
    ax.plot(
        [x, x], [13.675, 13.30],
        color=GRAY, linewidth=1
    )

ax.plot(
    [2.3, 9.7], [13.30, 13.30],
    color=GRAY, linewidth=1
)

# II. Allocation
section(
    0.4, 3.55, 11.2, 9.55,
    "II. STALL-AWARE CASR ALLOCATION", TEAL
)

box(
    6, 12.35, 5.4, 0.62,
    "Apply active robot/task filtering\n"
    "and retain valid reservations"
)

arrow(6, 13.30, 6, 12.66)

box(
    6, 11.40, 5.4, 0.62,
    "Update unchanged-state counters\n"
    "(task ID and robot position)"
)

arrow(6, 12.04, 6, 11.71)

diamond(
    6, 10.30, 4.6, 1.1,
    "Saturated with at least\n"
    "one stalled owner?"
)

arrow(6, 11.09, 6, 10.85)

# Early intervention branch
box(
    3.1, 8.95, 4.3, 0.8,
    "Compare stalled owners and idle robots\n"
    "using finite A* path costs",
    TEAL
)

route([
    (6, 9.75),
    (6, 9.52),
    (3.1, 9.52),
    (3.1, 9.35)
], "Yes")

diamond(
    3.1, 7.70, 3.7, 0.95,
    "Best gain > 0?"
)

arrow(3.1, 8.55, 3.1, 8.175)

box(
    3.1, 6.55, 4.3, 0.85,
    "Early reservation transfer\n"
    "Relax normal margin; reset\n"
    "saturation and rejection counters",
    TEAL, bold=True
)

arrow(3.1, 7.225, 3.1, 6.975, "Yes")

# Inherited CASR branch
diamond(
    8.8, 8.95, 4.5, 1.1,
    "Saturated and next streak\n"
    "exceeds threshold?"
    , BLUE
)

route([
    (8.3, 10.3),
    (8.8, 10.3),
    (8.8, 9.50)
], "No")

route([
    (4.95, 7.70),
    (5.6, 7.70),
    (5.6, 9.85),
    (7.2, 9.85),
    (7.2, 9.34)
], "No")

# Normal transfer logic
box(
    8.8, 7.65, 4.2, 0.7,
    "Normal selective check:\n"
    "positive gain and gain >= margin",
    BLUE
)

arrow(8.8, 8.40, 8.8, 8.0, "Yes")

diamond(
    8.8, 6.65, 3.7, 0.85,
    "Eligible transfer?",
    BLUE
)

arrow(8.8, 7.30, 8.8, 7.075)

box(
    6.25, 5.55, 2.65, 0.72,
    "Transfer reservation\n"
    "Reset rejection count",
    BLUE, fs=8.5
)

route([
    (6.95, 6.65),
    (6.25, 6.65),
    (6.25, 5.91)
], "Yes")

box(
    9.7, 5.55, 3.0, 0.72,
    "Increment rejection count\n"
    "Global recovery if count >= 2",
    BLUE, fs=8.5
)

route([
    (8.8, 6.225),
    (8.8, 6.10),
    (9.7, 6.10),
    (9.7, 5.91)
], "No")

# PALR behavior, routed separately
box(
    10.0, 11.65, 2.55, 0.95,
    "Inherited PALR:\n"
    "initialization, repair,\n"
    "reservation delay",
    BLUE, fs=8.3
)

route([
    (11.05, 8.95),
    (11.35, 8.95),
    (11.35, 10.95),
    (10.0, 10.95),
    (10.0, 11.175)
], "No")

# Common state update
box(
    6, 4.15, 6.1, 0.70,
    "Store post-allocation reservations\n"
    "and pre-movement positions; reset changed states",
    GRAY
)

route([
    (3.1, 6.125),
    (3.1, 4.85),
    (6, 4.85),
    (6, 4.50)
])

route([
    (6.25, 5.19),
    (6.25, 4.85),
    (6, 4.85),
    (6, 4.50)
])

route([
    (9.7, 5.19),
    (9.7, 4.85),
    (6, 4.85),
    (6, 4.50)
])

route([
    (11.275, 11.65),
    (11.48, 11.65),
    (11.48, 4.85),
    (6, 4.85),
    (6, 4.50)
])

# III. Navigation and execution
section(
    0.4, 1.25, 11.2, 2.0,
    "III. NAVIGATION AND EXECUTION", BLUE
)

box(2.0, 2.42, 2.6, 0.6,
    "A* navigation", BLUE)

box(6.0, 2.42, 3.4, 0.6,
    "Occupancy and destination check",
    BLUE, fs=8.5)

box(10.0, 2.42, 2.6, 0.6,
    "Move or block", BLUE)

route([
    (6, 3.80),
    (6, 3.40),
    (2, 3.40),
    (2, 2.72)
])

arrow(3.3, 2.42, 4.3, 2.42)
arrow(7.7, 2.42, 8.7, 2.42)

box(
    6, 1.78, 4.3, 0.55,
    "Update completed tasks and iteration",
    BLUE, fs=8.5
)

route([
    (10, 2.12),
    (10, 1.78),
    (8.15, 1.78)
])

diamond(
    6, 0.80, 4.2, 0.85,
    "Mission continues?",
    GRAY
)

arrow(6, 1.505, 6, 1.225)

route([
    (3.9, 0.80),
    (0.18, 0.80),
    (0.18, 12.35),
    (3.3, 12.35)
], "Yes", dashed=True)

box(
    9.9, 0.80, 3.0, 0.85,
    "Performance evaluation\n"
    "Blocking, A*, makespan,\n"
    "success, runtime, distance",
    GRAY, fs=8
)

arrow(8.1, 0.80, 8.4, 0.80, "No")

fig.tight_layout(pad=0.5)

for ext in ["pdf", "svg", "png"]:
    fig.savefig(
        OUT / f"figure_1_architecture.{ext}",
        dpi=600,
        bbox_inches="tight",
        facecolor="white"
    )

plt.close(fig)

print("===== FIGURE 1 GENERATED =====")
print("Architecture: Stall-aware CASR")
print("Formats: PDF, SVG, PNG")
print("Figures:", OUT)
