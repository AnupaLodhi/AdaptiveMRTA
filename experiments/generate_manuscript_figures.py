from pathlib import Path
import csv

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/raw/adaptive_development_20_seeds.csv"
FIG = ROOT / "manuscript/figures"
TAB = ROOT / "manuscript/tables"

FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(SOURCE)

methods = ["lock_aware", "palr_2", "palr_5", "adaptive"]
labels = ["Lock-aware", "PALR-2", "PALR-5", "Adaptive"]

assert len(df) == 160
assert not df.duplicated(
    ["seed", "failure_rate", "method"]
).any()

for rate in [0.0, 0.2]:
    subset = df[df.failure_rate == rate]
    assert set(subset.method) == set(methods)
    assert all(
        len(subset[subset.method == method]) == 20
        for method in methods
    )

summary = (
    df.groupby(["failure_rate", "method"])
    .agg(
        runs=("success", "size"),
        successes=("success", "sum"),
        mean_iterations=("iterations", "mean"),
        mean_distance=("total_distance", "mean"),
        mean_astar=("total_astar", "mean"),
        mean_global_calls=("global_calls", "mean"),
        mean_runtime_seconds=("runtime_seconds", "mean"),
    )
    .reset_index()
)

summary.to_csv(
    TAB / "development_summary.csv",
    index=False,
)

metrics = [
    ("total_astar", "Total A* calls", "fig01_astar_calls"),
    ("iterations", "Mission completion iterations", "fig02_iterations"),
    ("total_distance", "Total travel distance (grid steps)", "fig03_distance"),
    ("global_calls", "Global reassignment calls", "fig04_global_calls"),
]

for metric, ylabel, filename in metrics:
    fig, ax = plt.subplots(figsize=(8, 4.8))
    x = np.arange(len(methods))
    width = 0.35

    for offset, rate, label in [
        (-width / 2, 0.0, "0% failures"),
        (width / 2, 0.2, "20% failures"),
    ]:
        means = [
            df[
                (df.method == method)
                & (df.failure_rate == rate)
            ][metric].mean()
            for method in methods
        ]

        ax.bar(
            x + offset,
            means,
            width,
            label=label,
        )

    ax.set_xticks(x, labels)
    ax.set_ylabel(ylabel)
    ax.set_title("Adaptive MRTA development evaluation")
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)
    fig.tight_layout()

    fig.savefig(FIG / f"{filename}.pdf")
    fig.savefig(FIG / f"{filename}.png", dpi=300)
    plt.close(fig)

# Paired adaptive-versus-PALR-5 analysis.
paired = df.pivot(
    index=["seed", "failure_rate"],
    columns="method",
    values=["iterations", "total_astar", "total_distance", "global_calls"],
)

rows = []
for (seed, rate), values in paired.iterrows():
    record = {"seed": seed, "failure_rate": rate}
    for metric in [
        "iterations",
        "total_astar",
        "total_distance",
        "global_calls",
    ]:
        record[f"delta_{metric}"] = (
            values[(metric, "adaptive")]
            - values[(metric, "palr_5")]
        )
    rows.append(record)

comparison = pd.DataFrame(rows)
comparison.to_csv(
    TAB / "adaptive_vs_palr5_paired.csv",
    index=False,
)

fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

for ax, rate in zip(axes, [0.0, 0.2]):
    subset = comparison[comparison.failure_rate == rate]

    ax.scatter(
        subset["delta_total_astar"],
        subset["delta_iterations"],
        alpha=0.8,
    )
    ax.axhline(0, linestyle="--", linewidth=1)
    ax.axvline(0, linestyle="--", linewidth=1)
    ax.set_title(f"{int(rate * 100)}% robot failures")
    ax.set_xlabel("Additional A* calls vs PALR-5")
    ax.set_ylabel("Additional iterations vs PALR-5")
    ax.grid(alpha=0.2)

fig.suptitle("Paired adaptive policy trade-off")
fig.tight_layout()
fig.savefig(FIG / "fig05_adaptive_tradeoff.pdf")
fig.savefig(FIG / "fig05_adaptive_tradeoff.png", dpi=300)
plt.close(fig)

# Preserve a manuscript figure index.
index = """# Adaptive MRTA — Manuscript Figure Index

All figures use the 20-seed development experiment.
These are not final holdout results.

## Figure 1
Mean total A* calls across methods and failure rates.
A* call count is a planning-work proxy, not measured runtime.

## Figure 2
Mean mission completion iterations.

## Figure 3
Mean total robot travel distance.

## Figure 4
Mean global reassignment calls.

## Figure 5
Paired adaptive-versus-PALR-5 differences.
Positive values mean adaptive uses more resources or iterations.

## Tables
- development_summary.csv: aggregated development metrics.
- adaptive_vs_palr5_paired.csv: individual paired differences.

## Limitations
- 20 development seeds per failure condition.
- Single scenario size and obstacle density.
- Runtime values are exploratory.
- Statistical confidence intervals and final holdout evaluation pending.
- Selective reassignment has not yet been implemented.
"""

(FIG / "FIGURE_INDEX.md").write_text(index)

print("Generated 5 figures in PDF and PNG formats.")
print("Generated 2 CSV tables.")
print("Saved manuscript figure index.")
print("Location:", ROOT / "manuscript")
