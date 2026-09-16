from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]

INPUT = (
    ROOT
    / "data"
    / "raw"
    / "coordinated_obstacle_density_30_seeds.csv"
)

PROCESSED = ROOT / "data" / "processed"
TABLES = ROOT / "results" / "tables"

PROCESSED.mkdir(parents=True, exist_ok=True)
TABLES.mkdir(parents=True, exist_ok=True)


df = pd.read_csv(INPUT)

expected_algorithms = {
    "manhattan_greedy",
    "path_cost_greedy",
    "hungarian_astar",
}


# ============================================================
# 1. DATA INTEGRITY
# ============================================================

print()
print("=" * 90)
print("COORDINATED OBSTACLE-DENSITY ANALYSIS")
print("=" * 90)

print("Rows:", len(df))
print(
    "Densities:",
    sorted(df["obstacle_density"].unique()),
)
print(
    "Seeds:",
    df["seed"].nunique(),
)
print(
    "Algorithms:",
    sorted(df["algorithm"].unique()),
)


if len(df) != 450:
    raise RuntimeError(
        f"Expected 450 rows, found {len(df)}."
    )


if set(df["algorithm"].unique()) != expected_algorithms:
    raise RuntimeError(
        "Unexpected algorithm set."
    )


# Every density/seed must contain exactly three algorithms.
counts = (
    df.groupby(
        ["obstacle_density", "seed"]
    )["algorithm"]
    .nunique()
)

if not (counts == 3).all():
    raise RuntimeError(
        "Some scenarios do not contain all 3 algorithms."
    )


# Every density/seed must have ONE fingerprint.
fingerprints = (
    df.groupby(
        ["obstacle_density", "seed"]
    )["scenario_fingerprint"]
    .nunique()
)

if not (fingerprints == 1).all():
    raise RuntimeError(
        "SCENARIO CLONING FAILURE: "
        "algorithms received different scenarios."
    )


print()
print("SCENARIO INTEGRITY: PASS")
print(
    "All 150 density/seed scenarios were shared "
    "identically across the three algorithms."
)


# ============================================================
# 2. DESCRIPTIVE STATISTICS
# ============================================================

summary = (
    df.groupby(
        ["obstacle_density", "algorithm"]
    )
    .agg(
        n=("seed", "count"),

        success_rate=("success", "mean"),

        mean_iterations=("iterations", "mean"),
        sd_iterations=("iterations", "std"),

        mean_distance=("total_distance", "mean"),
        sd_distance=("total_distance", "std"),

        mean_productive_agents=(
            "productive_agents",
            "mean",
        ),

        mean_blocked_moves=(
            "blocked_moves",
            "mean",
        ),

        mean_redundancy=(
            "redundant_target_ratio",
            "mean",
        ),

        mean_runtime=(
            "runtime_seconds",
            "mean",
        ),
    )
    .reset_index()
)

summary["success_rate"] *= 100


print()
print("=" * 90)
print("DESCRIPTIVE SUMMARY")
print("=" * 90)

for density in sorted(
    summary["obstacle_density"].unique()
):

    print()
    print(
        f"OBSTACLE DENSITY = {density:.0%}"
    )
    print("-" * 90)

    block = summary[
        summary["obstacle_density"]
        == density
    ]

    print(
        block[
            [
                "algorithm",
                "success_rate",
                "mean_iterations",
                "mean_distance",
                "mean_productive_agents",
                "mean_blocked_moves",
                "mean_redundancy",
                "mean_runtime",
            ]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )


summary.to_csv(
    PROCESSED
    / "coordinated_obstacle_density_summary.csv",
    index=False,
)


# ============================================================
# 3. HUNGARIAN PAIRED COMPARISONS
# ============================================================

baselines = [
    "manhattan_greedy",
    "path_cost_greedy",
]

primary_metrics = [
    "iterations",
    "total_distance",
    "blocked_moves",
    "redundant_target_ratio",
]

rows = []


def paired_effect_dz(diff):

    sd = np.std(
        diff,
        ddof=1,
    )

    if sd == 0:
        return np.nan

    return np.mean(diff) / sd


for density in sorted(
    df["obstacle_density"].unique()
):

    density_df = df[
        df["obstacle_density"]
        == density
    ]

    for baseline in baselines:

        for metric in primary_metrics:

            base = (
                density_df[
                    density_df["algorithm"]
                    == baseline
                ]
                .sort_values("seed")
                [metric]
                .to_numpy(dtype=float)
            )

            hungarian = (
                density_df[
                    density_df["algorithm"]
                    == "hungarian_astar"
                ]
                .sort_values("seed")
                [metric]
                .to_numpy(dtype=float)
            )

            # Positive means baseline > Hungarian.
            diff = base - hungarian

            baseline_mean = np.mean(base)
            hungarian_mean = np.mean(
                hungarian
            )

            if baseline_mean != 0:

                reduction = (
                    (
                        baseline_mean
                        - hungarian_mean
                    )
                    / baseline_mean
                    * 100
                )

            else:
                reduction = np.nan


            if np.allclose(diff, 0):

                W = 0.0
                p = 1.0

            else:

                result = stats.wilcoxon(
                    base,
                    hungarian,
                    alternative="two-sided",
                    zero_method="wilcox",
                )

                W = result.statistic
                p = result.pvalue


            # Lower is better for all four
            # metrics in this comparison.
            wins = int(
                np.sum(
                    hungarian < base
                )
            )

            ties = int(
                np.sum(
                    hungarian == base
                )
            )

            losses = int(
                np.sum(
                    hungarian > base
                )
            )


            rows.append(
                {
                    "obstacle_density":
                        density,

                    "baseline":
                        baseline,

                    "metric":
                        metric,

                    "baseline_mean":
                        baseline_mean,

                    "hungarian_mean":
                        hungarian_mean,

                    "mean_baseline_minus_hungarian":
                        np.mean(diff),

                    "percent_reduction":
                        reduction,

                    "wilcoxon_W":
                        W,

                    "p_value":
                        p,

                    "paired_effect_dz":
                        paired_effect_dz(
                            diff
                        ),

                    "hungarian_wins":
                        wins,

                    "ties":
                        ties,

                    "hungarian_losses":
                        losses,
                }
            )


paired = pd.DataFrame(rows)


# ============================================================
# 4. HOLM CORRECTION
#
# Correct the family of primary tests:
# 5 densities × 2 baselines × 4 outcomes = 40 tests.
# ============================================================

raw_p = paired["p_value"].to_numpy()

order = np.argsort(raw_p)

adjusted = np.empty(
    len(raw_p),
    dtype=float,
)

running_max = 0.0
m = len(raw_p)

for rank, index in enumerate(order):

    candidate = min(
        1.0,
        raw_p[index]
        * (m - rank),
    )

    running_max = max(
        running_max,
        candidate,
    )

    adjusted[index] = (
        running_max
    )


paired["holm_adjusted_p"] = (
    adjusted
)


paired.to_csv(
    PROCESSED
    / "coordinated_obstacle_density_paired.csv",
    index=False,
)


# ============================================================
# 5. PAPER-FRIENDLY REDUCTION TABLE
# ============================================================

paper_rows = []


for density in sorted(
    df["obstacle_density"].unique()
):

    for baseline in baselines:

        block = paired[
            (
                paired["obstacle_density"]
                == density
            )
            &
            (
                paired["baseline"]
                == baseline
            )
        ]


        def get(metric, column):

            return float(
                block[
                    block["metric"]
                    == metric
                ][column].iloc[0]
            )


        paper_rows.append(
            {
                "obstacle_density":
                    density,

                "baseline":
                    baseline,

                "iteration_reduction_pct":
                    get(
                        "iterations",
                        "percent_reduction",
                    ),

                "distance_reduction_pct":
                    get(
                        "total_distance",
                        "percent_reduction",
                    ),

                "blocked_move_reduction_pct":
                    get(
                        "blocked_moves",
                        "percent_reduction",
                    ),

                "redundancy_reduction_pct":
                    get(
                        "redundant_target_ratio",
                        "percent_reduction",
                    ),

                "iteration_holm_p":
                    get(
                        "iterations",
                        "holm_adjusted_p",
                    ),

                "distance_holm_p":
                    get(
                        "total_distance",
                        "holm_adjusted_p",
                    ),
            }
        )


paper = pd.DataFrame(
    paper_rows
)


paper.to_csv(
    TABLES
    / "obstacle_density_reductions.csv",
    index=False,
)


print()
print("=" * 90)
print("HUNGARIAN REDUCTIONS BY OBSTACLE DENSITY")
print("=" * 90)

print(
    paper.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)


# ============================================================
# 6. WIN / TIE / LOSS
# ============================================================

print()
print("=" * 90)
print("PAIRED WIN / TIE / LOSS")
print("=" * 90)

print(
    paired[
        [
            "obstacle_density",
            "baseline",
            "metric",
            "hungarian_wins",
            "ties",
            "hungarian_losses",
            "paired_effect_dz",
            "holm_adjusted_p",
        ]
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}",
    )
)


print()
print("=" * 90)
print("FILES SAVED")
print("=" * 90)

print(
    PROCESSED
    / "coordinated_obstacle_density_summary.csv"
)

print(
    PROCESSED
    / "coordinated_obstacle_density_paired.csv"
)

print(
    TABLES
    / "obstacle_density_reductions.csv"
)

print("=" * 90)
