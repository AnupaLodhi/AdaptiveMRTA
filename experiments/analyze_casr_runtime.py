from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/processed"
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(
    ROOT / "data/raw/casr_v2_runtime.csv"
)

keys = [
    "configuration",
    "seed",
    "failure_rate",
    "method",
]

assert len(df) == 300
assert not df.duplicated(
    keys + ["repetition"]
).any()

assert df.groupby(keys).size().eq(3).all()
assert df.success.eq(1).all()
assert df.runtime_seconds.gt(0).all()

# Check that repetitions reproduce the same
# non-timing outcomes.
metrics = [
    "success",
    "iterations",
    "completed_tasks",
    "blocked_moves",
    "total_astar",
    "scenario_fingerprint",
]

assert (
    df.groupby(keys)[metrics]
    .nunique()
    .le(1)
    .all()
    .all()
)

scenario = df.groupby(keys).agg(
    median_runtime=("runtime_seconds", "median"),
    mean_runtime=("runtime_seconds", "mean"),
    min_runtime=("runtime_seconds", "min"),
    max_runtime=("runtime_seconds", "max"),
    std_runtime=("runtime_seconds", "std"),
).reset_index()

scenario["cv_pct"] = (
    100 * scenario.std_runtime
    / scenario.mean_runtime
)

assert len(scenario) == 100

scenario.to_csv(
    OUT / "casr_runtime_scenario_medians.csv",
    index=False,
)

summary = scenario.groupby([
    "configuration",
    "failure_rate",
    "method",
]).agg(
    scenarios=("seed", "size"),
    mean_runtime=("median_runtime", "mean"),
    median_runtime=("median_runtime", "median"),
    std_runtime=("median_runtime", "std"),
    mean_cv_pct=("cv_pct", "mean"),
)

summary.to_csv(
    OUT / "casr_runtime_summary.csv"
)

print("\n===== RUNTIME SUMMARY =====")
print(summary.round(5).to_string())

rng = np.random.default_rng(2027)
rows = []

for (config, rate), subset in scenario.groupby([
    "configuration",
    "failure_rate",
]):

    pivot = subset.pivot(
        index="seed",
        columns="method",
        values="median_runtime",
    )

    for baseline in [
        "casr_3",
        "palr_5",
        "adaptive",
        "lock_aware",
    ]:

        proposed = pivot["casr_v2_s2"].to_numpy()
        reference = pivot[baseline].to_numpy()

        diff = proposed - reference
        speedup = reference / proposed

        reduction = 100 * (
            1 - proposed.mean() / reference.mean()
        )

        indices = rng.integers(
            0, len(diff),
            size=(10000, len(diff)),
        )

        boot = 100 * (
            1
            - proposed[indices].mean(axis=1)
            / reference[indices].mean(axis=1)
        )

        low, high = np.percentile(
            boot, [2.5, 97.5]
        )

        p = (
            wilcoxon(
                diff,
                alternative="two-sided",
            ).pvalue
            if np.any(diff != 0)
            else 1.0
        )

        rows.append({
            "configuration": config,
            "failure_rate": rate,
            "baseline": baseline,
            "n_pairs": len(diff),
            "proposed_runtime": proposed.mean(),
            "baseline_runtime": reference.mean(),
            "reduction_pct": reduction,
            "ci95_low": low,
            "ci95_high": high,
            "median_speedup": np.median(speedup),
            "wins": int((diff < 0).sum()),
            "ties": int((diff == 0).sum()),
            "losses": int((diff > 0).sum()),
            "p_value": p,
        })

stats = pd.DataFrame(rows)

p = stats.p_value.to_numpy()
order = np.argsort(p)

adjusted = np.maximum.accumulate(
    p[order] * (len(p) - np.arange(len(p)))
)

corrected = np.empty_like(p)
corrected[order] = np.minimum(adjusted, 1)

stats["p_holm"] = corrected
stats["significant"] = stats.p_holm < 0.05

stats.to_csv(
    OUT / "casr_runtime_paired_statistics.csv",
    index=False,
)

print("\n===== PAIRED RUNTIME COMPARISONS =====")
print(stats.round(5).to_string(index=False))

print("\n===== COMPLETION =====")
print("Timed runs:", len(df))
print("Successful:", int(df.success.sum()))
print("Scenario-method groups:", len(scenario))
print("Comparisons:", len(stats))
print("Significant:", int(stats.significant.sum()))

print("\nRuntime analysis complete.")
