from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import t

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "manuscript/figures"
TABLES = ROOT / "manuscript/tables"

OUT.mkdir(parents=True, exist_ok=True)
TABLES.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(
    ROOT / "data/raw/casr_v2_scalability.csv"
)

configs = [
    "small",
    "medium",
    "large",
    "extra_large",
    "expanded",
    "dense_obstacles"
]

config_labels = [
    "5R/10T",
    "15R/20T",
    "30R/50T",
    "50R/100T",
    "Expanded",
    "Dense"
]

methods = {
    "lock_aware": (
        "Lock-Aware", "#555555", "o", "--"
    ),
    "adaptive": (
        "Adaptive", "#2166AC", "s", "-"
    ),
    "palr_5": (
        "PALR-5", "#9467BD", "^", ":"
    ),
    "casr_3": (
        "CASR-3", "#D6604D", "D", "--"
    ),
    "casr_v2_s2": (
        "CASR-V2-S2", "#009688", "P", "-"
    )
}

metrics = [
    ("blocked_moves", "Blocked movements"),
    ("total_astar", "Total A* evaluations"),
    ("iterations", "Makespan (iterations)")
]

assert len(df) == 600
assert df["success"].sum() == 600

assert set(df["configuration"]) == set(configs)
assert set(df["method"]) == set(methods)

assert not df.duplicated([
    "configuration",
    "seed",
    "failure_rate",
    "method"
]).any()

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "svg.fonttype": "none"
})

fig, axes = plt.subplots(
    3, 2,
    figsize=(12, 10),
    sharey="row"
)

records = []
x = np.arange(len(configs))

for row, (metric, ylabel) in enumerate(metrics):

    for col, rate in enumerate([0.0, 0.2]):

        ax = axes[row, col]

        for method, style in methods.items():

            label, color, marker, linestyle = style

            means = []
            lows = []
            highs = []

            for config in configs:

                group = df.loc[
                    (df["configuration"] == config) &
                    (df["failure_rate"] == rate) &
                    (df["method"] == method)
                ]

                values = group[metric]

                n = len(values)
                assert n == 10

                mean = values.mean()
                sd = values.std(ddof=1)

                half = (
                    t.ppf(0.975, n-1)
                    * sd / np.sqrt(n)
                )

                low = max(0, mean-half)
                high = mean+half

                means.append(mean)
                lows.append(low)
                highs.append(high)

                records.append({
                    "configuration": config,
                    "robots": group["robots"].iloc[0],
                    "tasks": group["tasks"].iloc[0],
                    "grid_width": group["grid_width"].iloc[0],
                    "grid_height": group["grid_height"].iloc[0],
                    "obstacle_density": group["obstacle_density"].iloc[0],
                    "failure_rate": rate,
                    "method": method,
                    "metric": metric,
                    "n": n,
                    "mean": mean,
                    "std": sd,
                    "ci95_lower": low,
                    "ci95_upper": high
                })

            means = np.asarray(means)
            lows = np.asarray(lows)
            highs = np.asarray(highs)

            width = (
                2.2 if method == "casr_v2_s2"
                else 1.2
            )

            # Fixed-grid workload scaling
            ax.errorbar(
                x[:4],
                means[:4],
                yerr=[
                    means[:4]-lows[:4],
                    highs[:4]-means[:4]
                ],
                color=color,
                marker=marker,
                linestyle=linestyle,
                linewidth=width,
                markersize=5,
                capsize=2,
                label=label
            )

            # Separate environmental configurations
            ax.errorbar(
                x[4:],
                means[4:],
                yerr=[
                    means[4:]-lows[4:],
                    highs[4:]-means[4:]
                ],
                color=color,
                marker=marker,
                linestyle="none",
                linewidth=width,
                markersize=5,
                capsize=2
            )

        panel = chr(ord("a") + row*2 + col)

        ax.set_title(
            f"({panel}) {ylabel} — {rate:.0%} failure",
            fontweight="bold"
        )

        ax.set_xticks(x)
        ax.set_xticklabels(
            config_labels,
            rotation=20,
            ha="right"
        )

        ax.axvline(
            3.5,
            color="#888888",
            linestyle=":",
            linewidth=1
        )

        ax.axvspan(
            3.5, 5.4,
            color="#888888",
            alpha=0.05
        )

        ax.set_xlim(-0.3, 5.4)
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", alpha=0.2)

        if col == 0:
            ax.set_ylabel(ylabel)

handles, labels = (
    axes[0, 0].get_legend_handles_labels()
)

fig.legend(
    handles,
    labels,
    loc="upper center",
    ncol=5,
    frameon=False
)

fig.text(
    0.5, 0.015,
    "R = robots; T = tasks. "
    "Expanded and Dense are separate environmental tests.",
    ha="center",
    fontsize=9
)

fig.tight_layout(
    rect=[0, 0.04, 1, 0.95]
)

for ext in ["pdf", "svg", "png"]:
    fig.savefig(
        OUT / f"figure_5_scalability.{ext}",
        dpi=600,
        bbox_inches="tight"
    )

plt.close(fig)

table = pd.DataFrame(records)

table.to_csv(
    TABLES / "scalability_figure5_table.csv",
    index=False
)

print("===== FIGURE 5 GENERATED =====")
print("Executions:", len(df))
print("Successful:", int(df["success"].sum()))
print("Configurations:", df["configuration"].nunique())
print("Methods:", df["method"].nunique())
print("Table rows:", len(table))
print("Figures:", OUT)
print("Tables:", TABLES)
