from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]

df = pd.read_csv(
    ROOT / "data/raw/casr_v2_holdout.csv"
)

assert len(df) == 280
assert set(df.seed) == set(range(3000, 3020))
assert not df.duplicated(
    ["seed", "failure_rate", "method"]
).any()

assert df.groupby(
    ["seed", "failure_rate"]
).size().eq(7).all()

out = ROOT / "data/processed"
out.mkdir(parents=True, exist_ok=True)

summary = df.groupby(
    ["failure_rate", "method"]
).agg(
    runs=("success", "size"),
    success_pct=("success", lambda x: 100*x.mean()),
    makespan=("iterations", "mean"),
    distance=("total_distance", "mean"),
    blocked=("blocked_moves", "mean"),
    astar=("total_astar", "mean"),
    global_calls=("global_calls", "mean"),
    runtime=("runtime_seconds", "mean"),
    stall_transfers=("stall_transfers", "mean"),
).round(4)

summary.to_csv(
    out / "casr_v2_holdout_summary.csv"
)

print("\n===== HOLDOUT SUMMARY =====\n")
print(summary.to_string())

print("\n===== COMPLETION =====")
print("Total runs:", len(df))
print("Successful:", int(df.success.sum()))
print("Failed:", int((df.success == 0).sum()))

print("\n===== FAILED RUNS =====")
print(df[df.success == 0].to_string(index=False))

if not df.success.eq(1).all():
    raise SystemExit(
        "Failures detected. Review completion rates "
        "before interpreting makespan statistics."
    )

rng = np.random.default_rng(2027)
rows = []

metrics = [
    "iterations",
    "blocked_moves",
    "total_astar",
    "total_distance",
    "global_calls",
    "runtime_seconds",
]

baselines = [
    "casr_3",
    "casr_0",
    "palr_5",
    "adaptive",
    "lock_aware",
]

for rate in [0.0, 0.2]:

    subset = df[df.failure_rate == rate]

    for metric in metrics:

        pivot = subset.pivot(
            index="seed",
            columns="method",
            values=metric,
        )

        proposed = pivot["casr_v2_s2"].to_numpy()

        for baseline in baselines:

            reference = pivot[baseline].to_numpy()
            diff = proposed - reference

            indices = rng.integers(
                0, len(diff),
                size=(10000, len(diff)),
            )

            bootstrap = diff[indices].mean(axis=1)

            low, high = np.percentile(
                bootstrap, [2.5, 97.5]
            )

            if np.all(diff == 0):
                p_value = 1.0
            else:
                p_value = wilcoxon(
                    diff,
                    alternative="two-sided",
                ).pvalue

            reduction = (
                100 * (
                    reference.mean() - proposed.mean()
                ) / reference.mean()
                if reference.mean() != 0
                else np.nan
            )

            rows.append({
                "failure_rate": rate,
                "metric": metric,
                "baseline": baseline,
                "n_pairs": len(diff),
                "proposed_mean": proposed.mean(),
                "baseline_mean": reference.mean(),
                "reduction_pct": reduction,
                "mean_difference": diff.mean(),
                "ci95_low": low,
                "ci95_high": high,
                "wins": int((diff < 0).sum()),
                "ties": int((diff == 0).sum()),
                "losses": int((diff > 0).sum()),
                "p_value": p_value,
            })

stats = pd.DataFrame(rows)

# Holm correction across all statistical comparisons.
p = stats.p_value.to_numpy()
order = np.argsort(p)

adjusted = np.maximum.accumulate(
    p[order] * (len(p) - np.arange(len(p)))
)

corrected = np.empty_like(p)
corrected[order] = np.minimum(adjusted, 1.0)

stats["p_holm"] = corrected
stats["significant"] = stats.p_holm < 0.05

stats.to_csv(
    out / "casr_v2_holdout_statistics.csv",
    index=False,
)

print("\n===== PRIMARY COMPARISONS =====\n")

selected = stats[
    stats.baseline.isin(["casr_3", "palr_5"])
    & stats.metric.isin([
        "iterations",
        "blocked_moves",
        "total_astar",
    ])
]

print(selected.round(5).to_string(index=False))

print("\nSaved summary and statistical results.")
