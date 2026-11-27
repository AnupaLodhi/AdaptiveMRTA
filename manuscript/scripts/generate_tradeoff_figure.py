from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "manuscript/figures"
TABLES = ROOT / "manuscript/tables"

OUT.mkdir(parents=True, exist_ok=True)
TABLES.mkdir(parents=True, exist_ok=True)

holdout = pd.read_csv(
    ROOT / "data/raw/casr_v2_holdout.csv"
)

scalability = pd.read_csv(
    ROOT / "data/raw/casr_v2_scalability.csv"
)

assert len(holdout) == 280
assert len(scalability) == 600
assert holdout["success"].sum() == 280
assert scalability["success"].sum() == 600

assert not holdout.duplicated([
    "seed", "failure_rate", "method"
]).any()

assert not scalability.duplicated([
    "configuration", "seed",
    "failure_rate", "method"
]).any()

styles = {
    "lock_aware": (
        "Lock-Aware", "#555555", "o"
    ),
    "adaptive": (
        "Adaptive", "#2166AC", "s"
    ),
    "palr_5": (
        "PALR-5", "#9467BD", "^"
    ),
    "casr_0": (
        "CASR-0", "#E69F00", "v"
    ),
    "casr_3": (
        "CASR-3", "#D6604D", "D"
    ),
    "casr_v2_s2": (
        "CASR-V2-S2", "#009688", "P"
    ),
    "casr_v2_s3": (
        "CASR-V2-S3", "#56B4E9", "X"
    )
}

def pareto_mask(points):
    result = np.ones(len(points), dtype=bool)

    for i, p in enumerate(points):
        dominated = np.any(
            np.all(points <= p, axis=1) &
            np.any(points < p, axis=1)
        )
        result[i] = not dominated

    return result

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "svg.fonttype": "none"
})

fig, axes = plt.subplots(
    2, 2,
    figsize=(11, 8),
    sharex="row",
    sharey="row"
)

records = []
legend = {}

for row, (study, df) in enumerate([
    ("Holdout", holdout),
    ("Scalability", scalability)
]):
    for col, rate in enumerate([0.0, 0.2]):

        ax = axes[row, col]

        subset = df.loc[
            df["failure_rate"] == rate
        ]

        summary = subset.groupby(
            "method", as_index=False
        ).agg(
            n=("seed", "size"),
            mean_astar=("total_astar", "mean"),
            mean_makespan=("iterations", "mean"),
            mean_blocked=("blocked_moves", "mean"),
            success_rate=("success", "mean")
        )

        expected_n = (
            20 if study == "Holdout" else 60
        )

        assert (summary["n"] == expected_n).all()

        points = summary[
            ["mean_astar", "mean_makespan"]
        ].to_numpy()

        summary["pareto_nondominated"] = (
            pareto_mask(points)
        )

        frontier = summary.loc[
            summary["pareto_nondominated"]
        ].sort_values("mean_astar")

        ax.plot(
            frontier["mean_astar"],
            frontier["mean_makespan"],
            color="#888888",
            linestyle="--",
            linewidth=1.1,
            alpha=0.8,
            zorder=1
        )

        for _, item in summary.iterrows():

            method = item["method"]
            label, color, marker = styles[method]

            proposed = method == "casr_v2_s2"

            handle = ax.scatter(
                item["mean_astar"],
                item["mean_makespan"],
                color=color,
                marker=marker,
                s=130 if proposed else 65,
                edgecolors="black" if proposed else color,
                linewidths=0.8,
                zorder=5 if proposed else 3,
                label=label
            )

            legend[method] = handle

            if proposed:
                ax.annotate(
                    "V2-S2",
                    (
                        item["mean_astar"],
                        item["mean_makespan"]
                    ),
                    xytext=(9, 10),
                    textcoords="offset points",
                    fontsize=9,
                    fontweight="bold",
                    color=color
                )

            records.append({
                "study": study,
                "failure_rate": rate,
                "method": method,
                "n": item["n"],
                "mean_astar": item["mean_astar"],
                "mean_makespan": item["mean_makespan"],
                "mean_blocked": item["mean_blocked"],
                "success_rate": item["success_rate"],
                "pareto_nondominated": bool(
                    item["pareto_nondominated"]
                )
            })

        panel = chr(ord("a") + row*2 + col)

        ax.set_title(
            f"({panel}) {study} — {rate:.0%} failure",
            fontweight="bold"
        )

        ax.set_xlabel("Mean total A* evaluations")
        ax.set_ylabel("Mean makespan (iterations)")

        ax.grid(alpha=0.18)
        ax.margins(x=0.15, y=0.20)

fig.legend(
    [legend[m] for m in styles if m in legend],
    [styles[m][0] for m in styles if m in legend],
    loc="upper center",
    ncol=4,
    frameon=False
)

fig.text(
    0.5, 0.015,
    "Dashed lines connect nondominated sample means. "
    "Lower values are preferred on both axes.",
    ha="center",
    fontsize=9
)

fig.tight_layout(
    rect=[0, 0.04, 1, 0.91]
)

for ext in ["pdf", "svg", "png"]:
    fig.savefig(
        OUT / f"figure_6_tradeoff.{ext}",
        dpi=600,
        bbox_inches="tight"
    )

plt.close(fig)

table = pd.DataFrame(records)

table.to_csv(
    TABLES / "tradeoff_figure6_table.csv",
    index=False
)

print("===== FIGURE 6 GENERATED =====")
print("Holdout executions:", len(holdout))
print("Scalability executions:", len(scalability))
print("Table rows:", len(table))

print("\nCASR-V2-S2 Pareto status:")

print(
    table.loc[
        table["method"] == "casr_v2_s2",
        [
            "study",
            "failure_rate",
            "mean_astar",
            "mean_makespan",
            "pareto_nondominated"
        ]
    ].to_string(index=False)
)

print("\nNondominated methods:")

print(
    table.loc[
        table["pareto_nondominated"],
        [
            "study",
            "failure_rate",
            "method"
        ]
    ].to_string(index=False)
)

print("\nFigures:", OUT)
print("Tables:", TABLES)
