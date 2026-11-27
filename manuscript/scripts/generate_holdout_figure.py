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

df = pd.read_csv(ROOT / "data/raw/casr_v2_holdout.csv")

methods = [
    "lock_aware", "adaptive", "palr_5",
    "casr_0", "casr_3", "casr_v2_s2", "casr_v2_s3"
]

labels = [
    "Lock-Aware", "Adaptive", "PALR-5",
    "CASR-0", "CASR-3", "CASR-V2-S2", "CASR-V2-S3"
]

metrics = {
    "blocked_moves": "(a) Blocked movements",
    "total_astar": "(b) Total A* evaluations",
    "iterations": "(c) Makespan (iterations)",
    "global_calls": "(d) Global assignment calls"
}

assert len(df) == 280
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

fig, axes = plt.subplots(2, 2, figsize=(11, 7))
records = []

for ax, (metric, title) in zip(axes.flat, metrics.items()):
    for rate, offset, color, marker in [
        (0.0, -0.13, "#2166AC", "o"),
        (0.2, 0.13, "#D6604D", "s")
    ]:
        for i, method in enumerate(methods):
            values = df.loc[
                (df["method"] == method) &
                (df["failure_rate"] == rate),
                metric
            ]

            n = len(values)
            assert n == 20

            mean = values.mean()
            sd = values.std(ddof=1)
            half = t.ppf(0.975, n-1) * sd / np.sqrt(n)

            low = max(0, mean-half)
            high = mean+half

            ax.errorbar(
                mean, i+offset,
                xerr=[[mean-low], [high-mean]],
                fmt=marker,
                color=color,
                capsize=3,
                markersize=5,
                label=f"{rate:.0%} failure" if i == 0 else None
            )

            records.append({
                "method": method,
                "failure_rate": rate,
                "metric": metric,
                "n": n,
                "mean": mean,
                "std": sd,
                "ci95_lower": low,
                "ci95_upper": high
            })

    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlim(left=0)
    ax.set_title(title, fontweight="bold")
    ax.grid(axis="x", alpha=0.2)
    ax.axhspan(4.6, 5.4, color="#009688", alpha=0.08)

    for tick in ax.get_yticklabels():
        if tick.get_text() == "CASR-V2-S2":
            tick.set_fontweight("bold")

handles, legend_labels = axes[0, 0].get_legend_handles_labels()

fig.legend(
    handles, legend_labels,
    loc="upper center",
    ncol=2,
    frameon=False
)

fig.tight_layout(rect=[0, 0, 1, 0.94])

for ext in ["pdf", "svg", "png"]:
    fig.savefig(
        OUT / f"figure_3_holdout.{ext}",
        dpi=600,
        bbox_inches="tight"
    )

plt.close(fig)

pd.DataFrame(records).to_csv(
    TABLES / "holdout_figure3_table.csv",
    index=False
)

print("===== FIGURE 3 GENERATED =====")
print("Executions:", len(df))
print("Successful:", int(df["success"].sum()))
print("Figures:", OUT)
print("Table:", TABLES / "holdout_figure3_table.csv")
