from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]

INPUT = ROOT / "data/raw/dynamic_failures_30_seeds.csv"
OUT = ROOT / "data/processed"
TABLES = ROOT / "results/tables"

OUT.mkdir(parents=True, exist_ok=True)
TABLES.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT)

ALGORITHMS = {
    "manhattan_greedy",
    "path_cost_greedy",
    "hungarian_astar",
}

RATES = [0.0, 0.2, 0.4, 0.6]

print("=" * 92)
print("E4: DYNAMIC FAILURE ANALYSIS")
print("=" * 92)
print("Rows:", len(df))

# ============================================================
# 1. INTEGRITY
# ============================================================

if len(df) != 360:
    raise RuntimeError(
        f"Expected 360 rows, found {len(df)}."
    )

if set(df["algorithm"].unique()) != ALGORITHMS:
    raise RuntimeError("Unexpected algorithms.")

groups = df.groupby(["failure_rate", "seed"])

if not (groups["algorithm"].nunique() == 3).all():
    raise RuntimeError(
        "Missing algorithm in paired scenario."
    )

if not (
    groups["scenario_fingerprint"].nunique() == 1
).all():
    raise RuntimeError(
        "Scenario fingerprint mismatch."
    )

if not (
    groups["failed_agent_ids"].nunique(
        dropna=False
    ) == 1
).all():
    raise RuntimeError(
        "Failure-set mismatch between algorithms."
    )

counts = (
    df.groupby(["failure_rate", "algorithm"])
    .size()
)

if not (counts == 30).all():
    raise RuntimeError(
        "Expected 30 runs per rate/algorithm."
    )

print("PAIRING / FAILURE INTEGRITY: PASS")

# ============================================================
# 2. SUMMARY
# ============================================================

summary = (
    df.groupby(["failure_rate", "algorithm"])
    .agg(
        n=("seed", "count"),
        success_rate=("success", "mean"),

        mean_iterations=("iterations", "mean"),
        sd_iterations=("iterations", "std"),

        mean_distance=("total_distance", "mean"),
        sd_distance=("total_distance", "std"),

        mean_productive_agents=(
            "productive_agents", "mean"
        ),

        mean_blocked_moves=(
            "blocked_moves", "mean"
        ),

        mean_redundancy=(
            "redundant_target_ratio", "mean"
        ),

        mean_runtime=(
            "runtime_seconds", "mean"
        ),
    )
    .reset_index()
)

summary["success_rate"] *= 100

summary.to_csv(
    OUT / "dynamic_failures_summary.csv",
    index=False,
)

print()
print("=" * 92)
print("DESCRIPTIVE RESULTS")
print("=" * 92)

print(
    summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)

# ============================================================
# 3. PAIRED HUNGARIAN VS BASELINES
# ============================================================

baselines = [
    "manhattan_greedy",
    "path_cost_greedy",
]

metrics = [
    "iterations",
    "total_distance",
    "blocked_moves",
    "redundant_target_ratio",
    "runtime_seconds",
]

rows = []

for rate in RATES:

    block = df[
        np.isclose(df["failure_rate"], rate)
    ]

    for baseline in baselines:

        for metric in metrics:

            b = (
                block[
                    block["algorithm"] == baseline
                ]
                .sort_values("seed")[metric]
                .to_numpy(float)
            )

            h = (
                block[
                    block["algorithm"]
                    == "hungarian_astar"
                ]
                .sort_values("seed")[metric]
                .to_numpy(float)
            )

            # All selected metrics are lower-is-better.
            # Positive difference favors Hungarian.
            diff = b - h

            if np.allclose(diff, 0):
                W = 0.0
                p = 1.0
            else:
                test = stats.wilcoxon(
                    b,
                    h,
                    alternative="two-sided",
                    zero_method="wilcox",
                )
                W = test.statistic
                p = test.pvalue

            sd = np.std(diff, ddof=1)

            dz = (
                np.mean(diff) / sd
                if sd > 0
                else np.nan
            )

            # Deterministic paired bootstrap CI.
            rng = np.random.default_rng(
                20260916
                + int(rate * 1000)
                + (0 if baseline ==
                   "manhattan_greedy" else 10000)
                + metrics.index(metric)
            )

            boot = np.empty(10000)

            for i in range(10000):
                sample = rng.choice(
                    diff,
                    size=len(diff),
                    replace=True,
                )
                boot[i] = np.mean(sample)

            low, high = np.percentile(
                boot,
                [2.5, 97.5],
            )

            bmean = np.mean(b)
            hmean = np.mean(h)

            reduction = (
                (bmean - hmean)
                / bmean
                * 100
                if bmean != 0
                else np.nan
            )

            rows.append(
                {
                    "failure_rate": rate,
                    "baseline": baseline,
                    "metric": metric,

                    "baseline_mean": bmean,
                    "hungarian_mean": hmean,

                    "mean_improvement":
                        np.mean(diff),

                    "ci95_low": low,
                    "ci95_high": high,

                    "percent_reduction":
                        reduction,

                    "wilcoxon_W": W,
                    "p_value": p,
                    "effect_dz": dz,

                    "hungarian_wins":
                        int(np.sum(h < b)),

                    "ties":
                        int(np.sum(h == b)),

                    "hungarian_losses":
                        int(np.sum(h > b)),
                }
            )

paired = pd.DataFrame(rows)

# ============================================================
# 4. HOLM CORRECTION — PRIMARY OUTCOMES ONLY
#
# 4 rates × 2 baselines ×
# iterations/distance = 16 primary tests.
# ============================================================

mask = paired["metric"].isin(
    ["iterations", "total_distance"]
)

indices = paired.index[mask]
pvals = paired.loc[indices, "p_value"].to_numpy()

order = np.argsort(pvals)
m = len(pvals)

adjusted = np.empty(m)
running = 0.0

for rank, idx in enumerate(order):

    candidate = min(
        1.0,
        pvals[idx] * (m - rank),
    )

    running = max(
        running,
        candidate,
    )

    adjusted[idx] = running

paired["holm_primary_p"] = np.nan

paired.loc[
    indices,
    "holm_primary_p"
] = adjusted

paired.to_csv(
    OUT / "dynamic_failures_paired.csv",
    index=False,
)

# ============================================================
# 5. COMPACT KEY TABLE
# ============================================================

key = paired[
    paired["metric"].isin(
        [
            "iterations",
            "total_distance",
            "runtime_seconds",
        ]
    )
].copy()

key.to_csv(
    TABLES / "dynamic_failures_key_results.csv",
    index=False,
)

print()
print("=" * 92)
print("KEY PAIRED RESULTS")
print("Positive improvement = lower Hungarian value")
print("=" * 92)

print(
    key[
        [
            "failure_rate",
            "baseline",
            "metric",
            "baseline_mean",
            "hungarian_mean",
            "percent_reduction",
            "ci95_low",
            "ci95_high",
            "effect_dz",
            "holm_primary_p",
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
print("=" * 92)
print("FILES SAVED")
print("=" * 92)
print(OUT / "dynamic_failures_summary.csv")
print(OUT / "dynamic_failures_paired.csv")
print(TABLES / "dynamic_failures_key_results.csv")
print("=" * 92)
