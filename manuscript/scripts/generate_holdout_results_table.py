"""Generate Table V from independent holdout experiments."""

from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data/raw/casr_v2_holdout.csv"
SUMMARY = ROOT / "data/processed/casr_v2_holdout_summary.csv"
STATS = ROOT / "data/processed/casr_v2_holdout_statistics.csv"
OUT = ROOT / "manuscript/tables"

OUT.mkdir(parents=True, exist_ok=True)

raw = pd.read_csv(RAW)
summary = pd.read_csv(SUMMARY)
stats = pd.read_csv(STATS)

METHODS = [
    "lock_aware",
    "adaptive",
    "palr_5",
    "casr_0",
    "casr_3",
    "casr_v2_s3",
    "casr_v2_s2",
]

assert len(raw) == 280
assert raw["success"].eq(1).all()
assert set(raw["seed"]) == set(range(3000, 3020))
assert set(raw["failure_rate"]) == {0.0, 0.2}
assert set(raw["method"]) == set(METHODS)
assert not raw.duplicated(
    ["seed", "failure_rate", "method"]
).any()

assert raw.groupby(
    ["seed", "failure_rate"]
).size().eq(7).all()

# Independently aggregate the raw experimental records.
aggregated = raw.groupby(
    ["failure_rate", "method"], as_index=False
).agg(
    runs=("success", "size"),
    success_pct=("success", lambda x: x.mean() * 100),
    makespan=("iterations", "mean"),
    blocked=("blocked_moves", "mean"),
    astar=("total_astar", "mean"),
    global_calls=("global_calls", "mean"),
    distance=("total_distance", "mean"),
)

# Verify against the previously committed summary.
check_cols = [
    "runs", "success_pct", "makespan", "blocked",
    "astar", "global_calls", "distance"
]

merged = aggregated.merge(
    summary,
    on=["failure_rate", "method"],
    suffixes=("_raw", "_summary"),
    validate="one_to_one",
)

assert len(merged) == 14

for column in check_cols:
    assert np.allclose(
        merged[f"{column}_raw"].to_numpy(),
        merged[f"{column}_summary"].to_numpy(),
        rtol=0,
        atol=0.00011,
    ), f"Summary mismatch: {column}"

# Main manuscript table.
main = aggregated.copy()
main["failure_rate_pct"] = (
    main["failure_rate"] * 100
).astype(int)

main["method_order"] = main["method"].map(
    {method: i for i, method in enumerate(METHODS)}
)

main = main.sort_values([
    "failure_rate_pct", "method_order"
])

main = main[[
    "failure_rate_pct",
    "method",
    "runs",
    "success_pct",
    "makespan",
    "blocked",
    "astar",
    "global_calls",
    "distance",
]]

numeric = [
    "success_pct", "makespan", "blocked",
    "astar", "global_calls", "distance"
]

main[numeric] = main[numeric].round(2)

main_path = OUT / "table_5_holdout_results.csv"
main.to_csv(main_path, index=False)

# Supporting comparisons, preserving original Holm results.
selected = stats[
    stats["baseline"].isin(["casr_3", "palr_5"])
    & stats["metric"].isin([
        "blocked_moves", "total_astar", "iterations"
    ])
].copy()

assert len(selected) == 12
assert selected["n_pairs"].eq(20).all()

selected["failure_rate_pct"] = (
    selected["failure_rate"] * 100
).astype(int)

selected = selected[[
    "failure_rate_pct",
    "baseline",
    "metric",
    "n_pairs",
    "proposed_mean",
    "baseline_mean",
    "reduction_pct",
    "mean_difference",
    "ci95_low",
    "ci95_high",
    "wins",
    "ties",
    "losses",
    "p_value",
    "p_holm",
    "significant",
]]

comparisons_path = (
    OUT / "table_5_holdout_primary_comparisons.csv"
)
selected.to_csv(comparisons_path, index=False)

print("===== TABLE V GENERATED =====")
print("Raw executions verified:", len(raw))
print("Successful executions:", int(raw.success.sum()))
print("Main result rows:", len(main))
print("Primary comparison rows:", len(selected))
print("Saved:", main_path)
print("Saved:", comparisons_path)

print("\n===== CASR-V2-S2 VS CASR-3 =====")
print(
    selected[
        (selected.baseline == "casr_3")
        & (selected.metric == "blocked_moves")
    ][[
        "failure_rate_pct",
        "reduction_pct",
        "p_holm",
        "significant"
    ]].round(5).to_string(index=False)
)
