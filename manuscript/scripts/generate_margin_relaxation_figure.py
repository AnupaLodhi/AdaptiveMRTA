"""Figure 2: Illustrative conditional cost-margin relaxation."""

from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "manuscript/figures"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
})

BLUE = "#2166AC"
TEAL = "#00897B"
GRAY = "#64748B"
DARK = "#243447"
RED = "#B5473D"

fig = plt.figure(figsize=(11.5, 6.8), facecolor="white")


# ---------------- GRID EXAMPLE ----------------

ax = fig.add_axes([0.12, 0.49, 0.76, 0.39])
ax.set_aspect("equal")
ax.set_xlim(-1.0, 7.0)
ax.set_ylim(-1.25, 3.45)
ax.axis("off")

# Grid
for x in range(7):
    for y in range(3):
        ax.add_patch(Rectangle(
            (x - 0.5, y - 0.5), 1, 1,
            facecolor="white",
            edgecolor="#D4DCE5",
            linewidth=0.8
        ))

# Separate visual lanes for the two estimated paths.
# Offsets are for display only.
ax.plot(
    [0, 6], [1.10, 1.10],
    linestyle=(0, (5, 4)),
    color=GRAY, linewidth=2.0, zorder=2
)

ax.plot(
    [2, 6], [0.90, 0.90],
    color=TEAL, linewidth=2.5, zorder=3
)

# Robot and task markers
for x, label, color in [
    (0, "A", GRAY),
    (2, "B", TEAL),
    (6, "T", BLUE),
]:
    ax.scatter(
        [x], [1], s=450, c=color,
        edgecolors="white",
        linewidth=1.5, zorder=5
    )

    ax.text(
        x, 1, label,
        ha="center", va="center",
        fontsize=11, weight="bold",
        color="white", zorder=6
    )

# Compact labels centered inside individual cells
for x, label in [
    (0, "Stalled\nowner"),
    (2, "Idle\ncandidate"),
    (6, "Reserved\ntask"),
]:
    ax.text(
        x, 0, label,
        ha="center", va="center",
        fontsize=8.5,
        linespacing=1.15,
        color=DARK
    )

# Owner distance bracket above grid
ax.plot(
    [0, 0, 6, 6],
    [2.65, 2.85, 2.85, 2.65],
    color=GRAY, linewidth=1.2
)

ax.text(
    3, 3.12,
    "Owner A* cost = 6 steps",
    ha="center", va="center",
    fontsize=10, color=GRAY,
    weight="bold"
)

# Candidate distance bracket below grid
ax.plot(
    [2, 2, 6, 6],
    [-0.65, -0.82, -0.82, -0.65],
    color=TEAL, linewidth=1.2
)

ax.text(
    4, -1.10,
    "Candidate A* cost = 4 steps",
    ha="center", va="center",
    fontsize=10, color=TEAL,
    weight="bold"
)

fig.text(
    0.5, 0.94,
    "Illustrative reservation-reassignment scenario",
    ha="center", fontsize=13,
    color=DARK, weight="bold"
)

fig.text(
    0.5, 0.46,
    "Estimated improvement: gain = 6 - 4 = 2 steps",
    ha="center", fontsize=11.5,
    color=DARK, weight="bold"
)

# ---------------- COMPARISON PANELS ----------------
panel_ax = fig.add_axes([0.04, 0.12, 0.92, 0.31])
panel_ax.set_xlim(0, 10)
panel_ax.set_ylim(0, 3)
panel_ax.axis("off")

def panel(x, title, lines, footer, edge, background):
    width = 4.75
    panel_ax.add_patch(FancyBboxPatch(
        (x, 0.18), width, 2.65,
        boxstyle="round,pad=0.02,rounding_size=0.12",
        facecolor=background, edgecolor=edge, linewidth=1.5
    ))
    panel_ax.text(
        x + width / 2, 2.42, title,
        ha="center", va="center", fontsize=12,
        weight="bold", color=edge
    )
    for i, line in enumerate(lines):
        panel_ax.text(
            x + width / 2, 1.88 - i * 0.39, line,
            ha="center", va="center",
            fontsize=10.4, color=DARK
        )
    panel_ax.text(
        x + width / 2, 0.44, footer,
        ha="center", va="center",
        fontsize=12, weight="bold", color=edge
    )

panel(
    0.10,
    "Conventional CASR",
    [
        "At a normal selective check",
        "Positive gain = 2 steps",
        "Required margin = 3 steps",
    ],
    "REJECT TRANSFER",
    RED, "#FDF1EF"
)

panel(
    5.15,
    "Proposed CASR-V2-S2",
    [
        "Saturated reservations; stalled owner",
        "At least 2 unchanged-state checks",
        "Positive gain = 2 steps > 0",
    ],
    "ACCEPT EARLY TRANSFER",
    TEAL, "#EBF7F3"
)

fig.text(
    0.5, 0.063,
    "Hypothetical example, not an observed experimental trace. "
    "The proposed decision assumes no better eligible candidate.",
    ha="center", fontsize=8.9, color=GRAY
)

for ext in ["pdf", "svg", "png"]:
    fig.savefig(
        OUT / f"figure_2_margin_relaxation.{ext}",
        dpi=600, bbox_inches="tight",
        facecolor="white"
    )

plt.close(fig)

print("===== FIGURE 2 GENERATED =====")
for ext in ["pdf", "svg", "png"]:
    print(OUT / f"figure_2_margin_relaxation.{ext}")
