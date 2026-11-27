"""Generate Table VII from verified scalability experiments."""

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "manuscript/tables"
OUT.mkdir(parents=True, exist_ok=True)

raw = pd.read_csv(
    ROOT / "data/raw/casr_v2_scalability.csv"
)
saved = pd.read_csv(
    ROOT / "data/processed/casr_v2_scalability_summary.csv"
)
overall_saved = pd.read_csv(
    ROOT / "data/processed/casr_v2_scalability_overall.csv"
)
statistics = pd.read_csv(
    ROOT / "data/processed/casr_final_blocking_statistics.csv"
)

CONFIGS = [
    "small", "medium", "large",
    "extra_large", "expanded", "dense_obstacles"
]
METHODS = [
    "lock_aware", "adaptive", "palr_5",
    "casr_3", "casr_v2_s2"
]

# Check experimental design and all paired scenarios.
assert len(raw) == 600
assert raw.success.eq(1).all()
assert set(raw.configuration) == set(CONFIGS)
assert set(raw.method) == set(METHODS)
assert set(raw.seed) == set(range(5000, 5010))
assert set(raw.failure_rate) == {0.0, 0.2}

assert not raw.duplicated([
    "configuration", "seed", "failure_rate", "method"
]).any()

assert raw.groupby([
    "configuration", "seed", "failure_rate"
]).size().eq(5).all()

# Independently calculate per-configuration means.
group_cols = [
    "configuration", "failure_rate", "method"
]

main = raw.groupby(group_cols, as_index=False).agg(
    runs=("success", "size"),
    success_pct=("success", lambda x: 100 * x.mean()),
    makespan=("iterations", "mean"),
    blocked=("blocked_moves", "mean"),
    astar=("total_astar", "mean"),
    global_calls=("global_calls", "mean"),
    distance=("total_distance", "mean"),
    stall_transfers=("stall_transfers", "mean"),
)

assert len(main) == 60
assert main.runs.eq(10).all()

check = main.merge(
    saved,
    on=group_cols,
    suffixes=("_raw", "_saved"),
    validate="one_to_one"
)
assert len(check) == 60

for col in [
    "runs", "success_pct", "makespan", "blocked",
    "astar", "global_calls", "distance", "stall_transfers"
]:
    assert np.allclose(
        check[f"{col}_raw"],
        check[f"{col}_saved"],
        rtol=0,
        atol=0.00011,
    ), f"Mismatch in {col}"

# Main per-configuration table.
main["failure_rate_pct"] = (
    100 * main.failure_rate
).astype(int)

main["config_order"] = main.configuration.map(
    {name: i for i, name in enumerate(CONFIGS)}
)
main["method_order"] = main.method.map(
    {name: i for i, name in enumerate(METHODS)}
)

main = main.sort_values([
    "config_order", "failure_rate_pct", "method_order"
])

main = main[[
    "configuration", "failure_rate_pct", "method",
    "runs", "success_pct", "makespan", "blocked",
    "astar", "global_calls", "distance",
    "stall_transfers"
]].round(2)

main_path = OUT / "table_7_scalability_results.csv"
main.to_csv(main_path, index=False)

# Aggregate across the 120 records per method.
overall = raw.groupby("method", as_index=False).agg(
    runs=("success", "size"),
    success_pct=("success", lambda x: 100 * x.mean()),
    makespan=("iterations", "mean"),
    blocked=("blocked_moves", "mean"),
    astar=("total_astar", "mean"),
    distance=("total_distance", "mean"),
)

check_overall = overall.merge(
    overall_saved,
    on="method",
    suffixes=("_raw", "_saved"),
    validate="one_to_one"
)

assert len(check_overall) == 5

for col in [
    "runs", "success_pct", "makespan",
    "blocked", "astar", "distance"
]:
    assert np.allclose(
        check_overall[f"{col}_raw"],
        check_overall[f"{col}_saved"],
        rtol=0,
        atol=0.00011,
    ), f"Overall mismatch: {col}"

overall = overall.round(2)

overall_path = OUT / "table_7_scalability_overall.csv"
overall.to_csv(overall_path, index=False)

# Existing Holm-adjusted blocking comparisons.
paired = statistics[
    statistics.study == "scalability"
].copy()

assert len(paired) == 48
assert paired.n_pairs.eq(10).all()
assert paired.significant.astype(str).str.lower().eq(
    "true"
).sum() == 0

paired["failure_rate_pct"] = (
    100 * paired.failure_rate
).astype(int)

paired["blocking_reduction_pct"] = np.where(
    paired.baseline_mean != 0,
    100 * (
        paired.baseline_mean - paired.proposed_mean
    ) / paired.baseline_mean,
    np.nan
)

paired = paired[[
    "configuration", "failure_rate_pct", "baseline",
    "n_pairs", "proposed_mean", "baseline_mean",
    "blocking_reduction_pct", "mean_difference",
    "ci95_low", "ci95_high", "wins", "ties",
    "losses", "p_value", "p_holm", "significant"
]]

paired_path = (
    OUT / "table_7_scalability_blocking_statistics.csv"
)
paired.to_csv(paired_path, index=False)

print("===== TABLE VII GENERATED =====")
print("Executions verified:", len(raw))
print("Successful:", int(raw.success.sum()))
print("Configuration-level rows:", len(main))
print("Overall comparison rows:", len(overall))
print("Paired blocking comparisons:", len(paired))
print("Holm-significant blocking comparisons: 0")
print("Saved:", main_path)
print("Saved:", overall_path)
print("Saved:", paired_path)

print("\n===== OVERALL MEANS =====")
print(overall.to_string(index=False))
