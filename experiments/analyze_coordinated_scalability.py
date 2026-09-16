from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]

INPUT = ROOT / "data/raw/coordinated_scalability_30_seeds.csv"
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

print("=" * 88)
print("COORDINATED SCALABILITY ANALYSIS")
print("=" * 88)
print("Rows:", len(df))

# ------------------------------------------------------------
# Integrity checks
# ------------------------------------------------------------

if len(df) != 540:
    raise RuntimeError(f"Expected 540 rows, got {len(df)}")

if set(df["algorithm"].unique()) != ALGORITHMS:
    raise RuntimeError("Unexpected algorithm set.")

groups = df.groupby(["num_agents", "num_tasks", "seed"])

if not (groups["algorithm"].nunique() == 3).all():
    raise RuntimeError("Missing algorithm in one or more paired scenarios.")

if not (groups["scenario_fingerprint"].nunique() == 1).all():
    raise RuntimeError("Scenario fingerprint mismatch.")

counts = (
    df.groupby(["num_agents", "num_tasks", "algorithm"])
    .size()
)

if not (counts == 30).all():
    raise RuntimeError("Expected 30 runs per algorithm/configuration.")

print("PAIRING / SCENARIO INTEGRITY: PASS")
print()

# ------------------------------------------------------------
# Descriptive summary
# ------------------------------------------------------------

summary = (
    df.groupby(["num_agents", "num_tasks", "algorithm"])
    .agg(
        n=("seed", "count"),
        success_rate=("success", "mean"),
        mean_iterations=("iterations", "mean"),
        sd_iterations=("iterations", "std"),
        mean_distance=("total_distance", "mean"),
        sd_distance=("total_distance", "std"),
        mean_productive_agents=("productive_agents", "mean"),
        mean_blocked_moves=("blocked_moves", "mean"),
        mean_redundancy=("redundant_target_ratio", "mean"),
        mean_runtime=("runtime_seconds", "mean"),
        sd_runtime=("runtime_seconds", "std"),
    )
    .reset_index()
)

summary["success_rate"] *= 100

summary.to_csv(
    OUT / "coordinated_scalability_summary.csv",
    index=False,
)

print("=" * 88)
print("MEAN SCALABILITY RESULTS")
print("=" * 88)

print(
    summary[
        [
            "num_agents",
            "num_tasks",
            "algorithm",
            "success_rate",
            "mean_iterations",
            "mean_distance",
            "mean_blocked_moves",
            "mean_redundancy",
            "mean_runtime",
        ]
    ].to_string(index=False, float_format=lambda x: f"{x:.4f}")
)

# ------------------------------------------------------------
# Paired Hungarian comparisons
# ------------------------------------------------------------

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

for (na, nt) in (
    df[["num_agents", "num_tasks"]]
    .drop_duplicates()
    .itertuples(index=False, name=None)
):

    block = df[
        (df["num_agents"] == na)
        & (df["num_tasks"] == nt)
    ]

    for baseline in baselines:

        for metric in metrics:

            b = (
                block[block["algorithm"] == baseline]
                .sort_values("seed")[metric]
                .to_numpy(float)
            )

            h = (
                block[block["algorithm"] == "hungarian_astar"]
                .sort_values("seed")[metric]
                .to_numpy(float)
            )

            # Lower is better for these metrics.
            diff = b - h

            if np.allclose(diff, 0):
                W, p = 0.0, 1.0
            else:
                test = stats.wilcoxon(
                    b,
                    h,
                    alternative="two-sided",
                    zero_method="wilcox",
                )
                W, p = test.statistic, test.pvalue

            sd = np.std(diff, ddof=1)
            dz = np.mean(diff) / sd if sd > 0 else np.nan

            # Deterministic bootstrap CI of paired mean improvement.
            rng = np.random.default_rng(
                20260916 + na * 100 + nt
            )

            boot = np.empty(10000)

            for i in range(10000):
                sample = rng.choice(
                    diff,
                    size=len(diff),
                    replace=True,
                )
                boot[i] = np.mean(sample)

            ci_low, ci_high = np.percentile(
                boot,
                [2.5, 97.5],
            )

            bmean = np.mean(b)
            hmean = np.mean(h)

            reduction = (
                (bmean - hmean) / bmean * 100
                if bmean != 0
                else np.nan
            )

            rows.append(
                {
                    "num_agents": na,
                    "num_tasks": nt,
                    "baseline": baseline,
                    "metric": metric,
                    "baseline_mean": bmean,
                    "hungarian_mean": hmean,
                    "mean_improvement": np.mean(diff),
                    "improvement_ci95_low": ci_low,
                    "improvement_ci95_high": ci_high,
                    "percent_reduction": reduction,
                    "wilcoxon_W": W,
                    "p_value": p,
                    "effect_dz": dz,
                    "hungarian_wins": int(np.sum(h < b)),
                    "ties": int(np.sum(h == b)),
                    "hungarian_losses": int(np.sum(h > b)),
                }
            )

paired = pd.DataFrame(rows)

# Holm correction ONLY for primary outcomes:
# iterations + total_distance
primary_mask = paired["metric"].isin(
    ["iterations", "total_distance"]
)

primary_indices = paired.index[primary_mask]
pvals = paired.loc[primary_indices, "p_value"].to_numpy()

order = np.argsort(pvals)
m = len(pvals)

adjusted = np.empty(m)
running = 0.0

for rank, idx in enumerate(order):
    value = min(1.0, pvals[idx] * (m - rank))
    running = max(running, value)
    adjusted[idx] = running

paired["holm_primary_p"] = np.nan
paired.loc[primary_indices, "holm_primary_p"] = adjusted

paired.to_csv(
    OUT / "coordinated_scalability_paired.csv",
    index=False,
)

# ------------------------------------------------------------
# Compact paper table
# ------------------------------------------------------------

primary = paired[
    paired["metric"].isin(
        ["iterations", "total_distance", "runtime_seconds"]
    )
].copy()

primary.to_csv(
    TABLES / "coordinated_scalability_key_results.csv",
    index=False,
)

print()
print("=" * 88)
print("HUNGARIAN: KEY PAIRED RESULTS")
print("Positive improvement = lower value for Hungarian")
print("=" * 88)

print(
    primary[
        [
            "num_agents",
            "num_tasks",
            "baseline",
            "metric",
            "baseline_mean",
            "hungarian_mean",
            "percent_reduction",
            "improvement_ci95_low",
            "improvement_ci95_high",
            "effect_dz",
            "holm_primary_p",
            "hungarian_wins",
            "ties",
            "hungarian_losses",
        ]
    ].to_string(index=False, float_format=lambda x: f"{x:.6f}")
)

print()
print("=" * 88)
print("FILES SAVED")
print("=" * 88)
print(OUT / "coordinated_scalability_summary.csv")
print(OUT / "coordinated_scalability_paired.csv")
print(TABLES / "coordinated_scalability_key_results.csv")
print("=" * 88)
