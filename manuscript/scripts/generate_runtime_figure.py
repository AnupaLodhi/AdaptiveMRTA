from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "manuscript/figures"
TABLES = ROOT / "manuscript/tables"

OUT.mkdir(parents=True, exist_ok=True)
TABLES.mkdir(parents=True, exist_ok=True)

raw = pd.read_csv(
    ROOT / "data/raw/casr_v2_runtime.csv"
)

saved_medians = pd.read_csv(
    ROOT / "data/processed/casr_runtime_scenario_medians.csv"
)

stats = pd.read_csv(
    ROOT / "data/processed/casr_runtime_paired_statistics.csv"
)

keys = [
    "configuration",
    "seed",
    "failure_rate",
    "method"
]

assert len(raw) == 300
assert raw["success"].sum() == 300
assert len(saved_medians) == 100
assert len(stats) == 16

assert not raw.duplicated(
    keys + ["repetition"]
).any()

assert (raw.groupby(keys).size() == 3).all()

# Reconstruct scenario-level medians.
medians = raw.groupby(
    keys, as_index=False
).agg(
    median_runtime=("runtime_seconds", "median")
)

check = medians.merge(
    saved_medians[keys + ["median_runtime"]],
    on=keys,
    suffixes=("_raw", "_saved"),
    validate="one_to_one"
)

assert len(check) == 100

assert np.allclose(
    check["median_runtime_raw"],
    check["median_runtime_saved"]
)

methods = [
    "lock_aware",
    "adaptive",
    "palr_5",
    "casr_3",
    "casr_v2_s2"
]

labels = {
    "lock_aware": "Lock-Aware",
    "adaptive": "Adaptive",
    "palr_5": "PALR-5",
    "casr_3": "CASR-3",
    "casr_v2_s2": "CASR-V2-S2"
}

configs = [
    ("medium", "Medium: 15R/20T"),
    ("extra_large", "Extra-large: 50R/100T")
]

conditions = [
    (0.0, -0.14, "#2166AC", "o"),
    (0.2, 0.14, "#D6604D", "s")
]

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "svg.fonttype": "none"
})

fig, axes = plt.subplots(
    2, 2, figsize=(12, 8)
)

records = []

for col, (config, config_label) in enumerate(configs):

    ax = axes[0, col]

    for rate, offset, color, marker in conditions:

        for i, method in enumerate(methods):

            group = medians.loc[
                (medians["configuration"] == config) &
                (medians["failure_rate"] == rate) &
                (medians["method"] == method)
            ]

            values = group["median_runtime"].to_numpy()

            assert len(values) == 5

            mean = values.mean()

            # Show individual scenario-level medians.
            ax.scatter(
                values,
                np.full(len(values), i+offset),
                color=color,
                s=16,
                alpha=0.30,
                zorder=2
            )

            # Highlight the mean of scenario medians.
            ax.scatter(
                mean,
                i+offset,
                color=color,
                marker=marker,
                s=65,
                edgecolors="black",
                linewidths=0.6,
                zorder=3,
                label=(
                    f"{rate:.0%} failure"
                    if i == 0 else None
                )
            )

            records.append({
                "configuration": config,
                "failure_rate": rate,
                "method": method,
                "n_scenarios": len(values),
                "repetitions_per_scenario": 3,
                "mean_of_medians": mean,
                "std_of_medians": values.std(ddof=1),
                "min_median": values.min(),
                "max_median": values.max()
            })

    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(
        [labels[m] for m in methods]
    )

    ax.invert_yaxis()
    ax.set_xlim(left=0)

    ax.set_xlabel("Runtime (seconds)")

    ax.set_title(
        f"({chr(97+col)}) {config_label}",
        fontweight="bold"
    )

    ax.axhspan(
        3.6, 4.4,
        color="#009688",
        alpha=0.08
    )

    ax.grid(axis="x", alpha=0.2)

    for tick in ax.get_yticklabels():
        if tick.get_text() == "CASR-V2-S2":
            tick.set_fontweight("bold")

    # Paired percentage reduction forest plot.
    ax = axes[1, col]

    baselines = [
        "lock_aware",
        "adaptive",
        "palr_5",
        "casr_3"
    ]

    for rate, offset, color, marker in conditions:

        for i, baseline in enumerate(baselines):

            group = stats.loc[
                (stats["configuration"] == config) &
                (stats["failure_rate"] == rate) &
                (stats["baseline"] == baseline)
            ]

            assert len(group) == 1

            item = group.iloc[0]

            assert item["n_pairs"] == 5

            effect = item["reduction_pct"]
            low = item["ci95_low"]
            high = item["ci95_high"]

            ax.hlines(
                i+offset,
                low,
                high,
                color=color,
                linewidth=1.4
            )

            ax.plot(
                [low, high],
                [i+offset, i+offset],
                linestyle="none",
                marker="|",
                color=color,
                markersize=7
            )

            ax.scatter(
                effect,
                i+offset,
                marker=marker,
                color=color,
                s=45,
                zorder=3
            )

    ax.axvline(
        0,
        color="#555555",
        linestyle="--",
        linewidth=1
    )

    ax.set_yticks(range(len(baselines)))

    ax.set_yticklabels(
        [labels[m] for m in baselines]
    )

    ax.invert_yaxis()
    ax.set_xlim(-15, 80)

    ax.set_xlabel("Runtime reduction (%)")

    ax.set_title(
        f"({chr(99+col)}) Paired reduction: {config_label}",
        fontweight="bold"
    )

    ax.grid(axis="x", alpha=0.2)

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

fig.text(
    0.5, 0.035,
    "Top: faint points = scenario medians; "
    "large markers = mean of five medians.",
    ha="center",
    fontsize=9
)

fig.text(
    0.5, 0.015,
    "Bottom: paired bootstrap 95% intervals; "
    "positive values favor CASR-V2-S2. "
    "All Holm-adjusted p-values = 1.000.",
    ha="center",
    fontsize=9
)

fig.tight_layout(
    rect=[0, 0.07, 1, 0.94]
)

for ext in ["pdf", "svg", "png"]:
    fig.savefig(
        OUT / f"figure_7_runtime.{ext}",
        dpi=600,
        bbox_inches="tight"
    )

plt.close(fig)

table = pd.DataFrame(records)

table.to_csv(
    TABLES / "runtime_figure7_table.csv",
    index=False
)

stats.to_csv(
    TABLES / "runtime_figure7_paired_statistics.csv",
    index=False
)

print("===== FIGURE 7 GENERATED =====")
print("Timed executions:", len(raw))
print("Successful:", int(raw["success"].sum()))
print("Scenario medians:", len(medians))
print("Runtime table rows:", len(table))
print("Paired comparisons:", len(stats))
print("Significant comparisons:",
      int((stats["p_holm"] < 0.05).sum()))

print("\nCASR-V2-S2 runtime:")
print(
    table.loc[
        table["method"] == "casr_v2_s2",
        [
            "configuration",
            "failure_rate",
            "mean_of_medians"
        ]
    ].to_string(index=False)
)

print("\nFigures:", OUT)
print("Tables:", TABLES)
