from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import t, norm

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "manuscript/figures"
TABLES = ROOT / "manuscript/tables"

OUT.mkdir(parents=True, exist_ok=True)
TABLES.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(
    ROOT / "data/raw/casr_v2_ablation.csv"
)

methods = [
    "casr_0",
    "casr_3",
    "v2_fixed_margin",
    "v2_no_recovery",
    "casr_v2_s2"
]

labels = [
    "CASR-0",
    "CASR-3",
    "V2-Fixed-Margin",
    "V2-No-Recovery",
    "CASR-V2-S2"
]

metrics = {
    "blocked_moves": "(a) Blocked movements",
    "total_astar": "(b) Total A* evaluations",
    "iterations": "(c) Executed iterations",
    "success": "(d) Mission success (%)"
}

assert len(df) == 300
assert set(df["method"]) == set(methods)
assert not df.duplicated(
    ["seed", "failure_rate", "method"]
).any()

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "svg.fonttype": "none"
})

fig, axes = plt.subplots(
    2, 2, figsize=(11, 7)
)

records = []

for ax, (metric, title) in zip(
    axes.flat, metrics.items()
):
    for rate, offset, color, marker in [
        (0.0, -0.13, "#2166AC", "o"),
        (0.2, 0.13, "#D6604D", "s")
    ]:
        for i, method in enumerate(methods):

            group = df.loc[
                (df["method"] == method) &
                (df["failure_rate"] == rate)
            ]

            values = group[metric]
            n = len(values)
            assert n == 30

            mean = values.mean()
            sd = values.std(ddof=1)

            if metric == "success":
                # Wilson binomial confidence interval
                z = norm.ppf(0.975)
                p = mean
                den = 1 + z*z/n

                center = (
                    p + z*z/(2*n)
                ) / den

                half = z * np.sqrt(
                    p*(1-p)/n + z*z/(4*n*n)
                ) / den

                low = 100 * max(0, center-half)
                high = 100 * min(1, center+half)
                mean *= 100
                sd *= 100

                ci_method = "Wilson"

            else:
                half = (
                    t.ppf(0.975, n-1)
                    * sd / np.sqrt(n)
                )

                low = max(0, mean-half)
                high = mean+half

                ci_method = "t-based"

            ax.errorbar(
                mean,
                i+offset,
                xerr=[
                    [max(0, mean-low)],
                    [max(0, high-mean)]
                ],
                fmt=marker,
                color=color,
                capsize=3,
                markersize=5,
                label=(
                    f"{rate:.0%} failure"
                    if i == 0 else None
                )
            )

            records.append({
                "method": method,
                "failure_rate": rate,
                "metric": metric,
                "n": n,
                "mean": mean,
                "std": sd,
                "ci95_lower": low,
                "ci95_upper": high,
                "ci_method": ci_method,
                "successful_runs": int(
                    group["success"].sum()
                )
            })

    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()

    if metric == "success":
        ax.set_xlim(0, 103)
    else:
        ax.set_xlim(left=0)

    ax.set_title(title, fontweight="bold")
    ax.grid(axis="x", alpha=0.2)

    ax.axhspan(
        3.6, 4.4,
        color="#009688",
        alpha=0.08
    )

    for tick in ax.get_yticklabels():
        if tick.get_text() == "CASR-V2-S2":
            tick.set_fontweight("bold")

handles, legend_labels = (
    axes[0, 0].get_legend_handles_labels()
)

fig.legend(
    handles,
    legend_labels,
    loc="upper center",
    ncol=2,
    frameon=False
)

fig.tight_layout(
    rect=[0, 0, 1, 0.94]
)

for ext in ["pdf", "svg", "png"]:
    fig.savefig(
        OUT / f"figure_4_ablation.{ext}",
        dpi=600,
        bbox_inches="tight"
    )

plt.close(fig)

pd.DataFrame(records).to_csv(
    TABLES / "ablation_figure4_table.csv",
    index=False
)

print("===== FIGURE 4 GENERATED =====")
print("Executions:", len(df))
print("Successful:", int(df["success"].sum()))

print("\nIncomplete missions:")
print(
    df.loc[
        df["success"] == 0,
        [
            "seed", "method",
            "failure_rate", "iterations",
            "completed_tasks"
        ]
    ].to_string(index=False)
)

print("\nFigures:", OUT)
print("Table:", TABLES / "ablation_figure4_table.csv")
