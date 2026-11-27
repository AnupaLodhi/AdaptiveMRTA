from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "manuscript/figures"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "pdf.fonttype": 42,
    "svg.fonttype": "none"
})

fig, ax = plt.subplots(figsize=(12, 8))
ax.set_xlim(0, 16)
ax.set_ylim(0, 11)
ax.axis("off")

GRAY = "#64748B"
BLUE = "#2166AC"
TEAL = "#00897B"
DARK = "#243447"

def box(x, y, w, h, title, body,
        color=GRAY, fill="#F8FAFC"):

    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.10",
        facecolor=fill,
        edgecolor=color,
        linewidth=1.3
    ))

    ax.text(
        x+w/2, y+h-0.32,
        title,
        ha="center", va="center",
        fontsize=11,
        fontweight="bold",
        color=color
    )

    ax.text(
        x+w/2, y+(h-0.5)/2,
        body,
        ha="center", va="center",
        fontsize=9.5,
        linespacing=1.6,
        color=DARK
    )

def arrow(x1, y1, x2, y2, color=GRAY):
    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle="-|>",
            color=color,
            lw=1.3,
            mutation_scale=13,
            shrinkA=0,
            shrinkB=0
        )
    )

# 1. System input
box(
    4.5, 9.5, 7, 1.1,
    "SYSTEM STATE",
    "Active robots | Pending tasks | Grid map"
)

arrow(8, 9.5, 8, 9.0)

# 2. State monitoring
box(
    4.5, 7.5, 7, 1.5,
    "STATE MONITORING",
    "Validate reservations and track unchanged states\n"
    "Identify saturation and eligible stalled owners"
)

arrow(8, 7.5, 8, 7.0)

# 3. Proposed mechanism
box(
    4.5, 5.0, 7, 2.0,
    "PROPOSED STALL-AWARE MECHANISM",
    "Under saturation, compare stalled owners with idle robots\n"
    "Evaluate finite A* path-cost gains\n"
    "Select the best strictly positive gain",
    TEAL, "#E8F5F2"
)

# 4. Early transfer
box(
    0.5, 2.9, 6.5, 1.4,
    "EARLY RESERVATION TRANSFER",
    "Relax normal cost margin\n"
    "Reset saturation and rejection counters",
    TEAL, "#E8F5F2"
)

# 5. Inherited policy
box(
    9.0, 2.9, 6.5, 1.4,
    "INHERITED CASR POLICY",
    "Reservation maintenance and normal transfer\n"
    "Global recovery after two rejected normal checks",
    BLUE, "#EDF4FB"
)

# Dedicated branching lane
ax.plot(
    [8, 8], [5.0, 4.65],
    color=GRAY, lw=1.3
)

ax.plot(
    [3.75, 12.25], [4.65, 4.65],
    color=GRAY, lw=1.3
)

arrow(3.75, 4.65, 3.75, 4.3, TEAL)
arrow(12.25, 4.65, 12.25, 4.3, BLUE)

ax.text(
    3.75, 4.82,
    "Eligible early transfer",
    ha="center", fontsize=9, color=TEAL
)

ax.text(
    12.25, 4.82,
    "Otherwise",
    ha="center", fontsize=9, color=BLUE
)

# Dedicated merging lane
ax.plot(
    [3.75, 3.75], [2.9, 2.55],
    color=GRAY, lw=1.3
)

ax.plot(
    [12.25, 12.25], [2.9, 2.55],
    color=GRAY, lw=1.3
)

ax.plot(
    [3.75, 12.25], [2.55, 2.55],
    color=GRAY, lw=1.3
)

arrow(8, 2.55, 8, 2.2)

# 6. Common output
box(
    4.5, 1.1, 7, 1.1,
    "RESERVATION AND STATE UPDATE",
    "Store assignments and pre-movement positions"
)

arrow(8, 1.1, 8, 0.7)

ax.text(
    8, 0.45,
    "A* navigation  |  Occupancy checks  |  Task completion",
    ha="center", va="center",
    fontsize=10, fontweight="bold",
    color=BLUE
)

ax.text(
    8, 0.05,
    "Repeat until mission termination; record performance metrics.",
    ha="center", va="center",
    fontsize=9, color=GRAY
)

fig.tight_layout(pad=0.6)

for ext in ["pdf", "svg", "png"]:
    fig.savefig(
        OUT / f"figure_1_architecture.{ext}",
        dpi=600,
        bbox_inches="tight",
        facecolor="white"
    )

plt.close(fig)

print("===== FIGURE 1 REDESIGNED =====")
print("Dependencies: Matplotlib only")
print("Formats: PDF, SVG, PNG")
print("Figures:", OUT)
