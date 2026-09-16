from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]

INPUT = (
    ROOT
    / "data"
    / "raw"
    / "hungarian_comparison_30_seeds.csv"
)

OUTPUT_DIR = ROOT / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


df = pd.read_csv(INPUT)


metrics = [
    "iterations",
    "total_distance",
    "productive_agents",
    "blocked_moves",
    "redundant_target_ratio",
    "runtime_seconds",
]


print()
print("=" * 80)
print("30-SEED HUNGARIAN COMPARISON ANALYSIS")
print("=" * 80)


# =========================================================
# DESCRIPTIVE STATISTICS
# =========================================================

summary = (
    df.groupby("algorithm")
    .agg(
        success_rate=("success", "mean"),

        mean_iterations=("iterations", "mean"),
        std_iterations=("iterations", "std"),

        mean_distance=("total_distance", "mean"),
        std_distance=("total_distance", "std"),

        mean_productive_agents=("productive_agents", "mean"),
        std_productive_agents=("productive_agents", "std"),

        mean_blocked_moves=("blocked_moves", "mean"),
        std_blocked_moves=("blocked_moves", "std"),

        mean_redundancy=("redundant_target_ratio", "mean"),
        std_redundancy=("redundant_target_ratio", "std"),

        mean_runtime=("runtime_seconds", "mean"),
        std_runtime=("runtime_seconds", "std"),
    )
    .reset_index()
)

summary["success_rate"] *= 100


print()
print("DESCRIPTIVE SUMMARY")
print("-" * 80)

print(
    summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)


summary.to_csv(
    OUTPUT_DIR
    / "hungarian_comparison_summary.csv",
    index=False,
)


# =========================================================
# PAIRED COMPARISONS AGAINST HUNGARIAN
# =========================================================

wide = df.pivot(
    index="seed",
    columns="algorithm",
    values=metrics,
)


comparators = [
    "manhattan_greedy",
    "path_cost_greedy",
]


paired_rows = []


print()
print("=" * 80)
print("PAIRED COMPARISONS: HUNGARIAN VS BASELINES")
print("=" * 80)


for baseline in comparators:

    print()
    print(f"HUNGARIAN vs {baseline.upper()}")
    print("-" * 80)

    for metric in metrics:

        h = (
            wide[metric]["hungarian_astar"]
            .to_numpy(dtype=float)
        )

        b = (
            wide[metric][baseline]
            .to_numpy(dtype=float)
        )

        # Positive = baseline value is larger than Hungarian.
        difference = b - h

        mean_difference = np.mean(difference)

        median_difference = np.median(difference)

        # Percentage reduction only for metrics where
        # lower values are naturally interpretable as better.
        if metric in [
            "iterations",
            "total_distance",
            "blocked_moves",
            "redundant_target_ratio",
        ]:
            baseline_mean = np.mean(b)
            hungarian_mean = np.mean(h)

            percent_change = (
                (
                    baseline_mean
                    - hungarian_mean
                )
                / baseline_mean
                * 100
                if baseline_mean != 0
                else np.nan
            )
        else:
            percent_change = np.nan

        # Paired Wilcoxon signed-rank test.
        # If every difference is zero, p = 1.
        if np.allclose(difference, 0):

            statistic = 0.0
            p_value = 1.0

        else:

            test = stats.wilcoxon(
                b,
                h,
                alternative="two-sided",
                zero_method="wilcox",
            )

            statistic = test.statistic
            p_value = test.pvalue

        # Paired standardized mean difference (dz).
        sd_difference = np.std(
            difference,
            ddof=1,
        )

        if sd_difference > 0:
            effect_dz = (
                mean_difference
                / sd_difference
            )
        else:
            effect_dz = np.nan

        wins = int(
            np.sum(h < b)
        )

        ties = int(
            np.sum(h == b)
        )

        losses = int(
            np.sum(h > b)
        )

        paired_rows.append(
            {
                "baseline": baseline,
                "metric": metric,
                "baseline_mean": np.mean(b),
                "hungarian_mean": np.mean(h),
                "mean_baseline_minus_hungarian": mean_difference,
                "median_baseline_minus_hungarian": median_difference,
                "percent_reduction": percent_change,
                "wilcoxon_statistic": statistic,
                "p_value": p_value,
                "paired_effect_dz": effect_dz,
                "hungarian_wins": wins,
                "ties": ties,
                "hungarian_losses": losses,
            }
        )

        print(
            f"{metric:25s}"
            f" baseline={np.mean(b):8.3f}"
            f" hungarian={np.mean(h):8.3f}"
            f" diff={mean_difference:8.3f}"
            f" reduction={percent_change:7.2f}%"
            if not np.isnan(percent_change)
            else
            f"{metric:25s}"
            f" baseline={np.mean(b):8.3f}"
            f" hungarian={np.mean(h):8.3f}"
            f" diff={mean_difference:8.3f}"
        )

        print(
            f"{'':25s}"
            f" W={statistic:.3f}"
            f" p={p_value:.6g}"
            f" dz={effect_dz:.3f}"
            f" wins/ties/losses="
            f"{wins}/{ties}/{losses}"
        )


paired = pd.DataFrame(
    paired_rows
)


# =========================================================
# HOLM CORRECTION
# =========================================================

raw_p = paired["p_value"].to_numpy()

order = np.argsort(raw_p)

adjusted = np.empty_like(
    raw_p,
    dtype=float,
)

m = len(raw_p)

running_max = 0.0

for rank, index in enumerate(order):

    value = min(
        1.0,
        raw_p[index]
        * (m - rank),
    )

    running_max = max(
        running_max,
        value,
    )

    adjusted[index] = running_max


paired["holm_adjusted_p"] = adjusted


paired.to_csv(
    OUTPUT_DIR
    / "hungarian_paired_statistics.csv",
    index=False,
)


print()
print("=" * 80)
print("HOLM-CORRECTED P VALUES")
print("=" * 80)

print(
    paired[
        [
            "baseline",
            "metric",
            "p_value",
            "holm_adjusted_p",
            "paired_effect_dz",
            "hungarian_wins",
            "ties",
            "hungarian_losses",
        ]
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}",
    )
)


print()
print("Saved:")
print(
    OUTPUT_DIR
    / "hungarian_comparison_summary.csv"
)

print(
    OUTPUT_DIR
    / "hungarian_paired_statistics.csv"
)
print()
