"""Generate verified Table VI from recorded ablation experiments."""

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "manuscript/tables"
OUT.mkdir(parents=True, exist_ok=True)

raw = pd.read_csv(ROOT / "data/raw/casr_v2_ablation.csv")
summary = pd.read_csv(
    ROOT / "data/processed/casr_v2_ablation_summary.csv"
)
stats = pd.read_csv(
    ROOT / "data/processed/casr_final_blocking_statistics.csv"
)

METHODS = [
    "casr_3",
    "casr_0",
    "v2_fixed_margin",
    "v2_no_recovery",
    "casr_v2_s2",
]

# Verify experimental design and pairing.
assert len(raw) == 300
assert set(raw.seed) == set(range(4000, 4030))
assert set(raw.failure_rate) == {0.0, 0.2}
assert set(raw.method) == set(METHODS)
assert not raw.duplicated(
    ["seed", "failure_rate", "method"]
).any()
assert raw.groupby(
    ["seed", "failure_rate"]
).size().eq(5).all()

failures = raw[raw.success == 0].copy()

assert len(failures) == 2
assert (failures.method == "v2_no_recovery").all()
assert (failures.failure_rate == 0.2).all()
assert failures.iterations.eq(500).all()
assert raw[raw.method == "casr_v2_s2"].success.eq(1).all()

# Aggregate ALL executions, including failures.
main = raw.groupby(
    ["failure_rate", "method"], as_index=False
).agg(
    runs=("success", "size"),
    success_pct=("success", lambda x: 100 * x.mean()),
    executed_iterations=("iterations", "mean"),
    completed_tasks=("completed_tasks", "mean"),
    blocked_moves=("blocked_moves", "mean"),
    total_astar=("total_astar", "mean"),
    global_calls=("global_calls", "mean"),
    stall_checks=("stall_checks", "mean"),
    stall_transfers=("stall_transfers", "mean"),
    recovery_calls=("recovery_calls", "mean"),
    total_distance=("total_distance", "mean"),
)

# Check raw aggregates against the saved summary.
check_map = {
    "runs": "runs",
    "success_pct": "success_pct",
    "executed_iterations": "iterations",
    "completed_tasks": "completed_tasks",
    "blocked_moves": "blocked",
    "total_astar": "astar",
    "global_calls": "global_calls",
    "stall_checks": "stall_checks",
    "stall_transfers": "stall_transfers",
    "recovery_calls": "recovery_calls",
    "total_distance": "distance",
}

check = main.merge(
    summary,
    on=["failure_rate", "method"],
    suffixes=("_raw", "_saved"),
    validate="one_to_one",
)

assert len(check) == 10

for raw_col, saved_col in check_map.items():
    if raw_col == saved_col:
        x = check[f"{raw_col}_raw"]
        y = check[f"{saved_col}_saved"]
    else:
        x = check[raw_col]
        y = check[saved_col]

    assert np.allclose(
        x, y, rtol=0, atol=0.00011
    ), f"Mismatch: {raw_col}"

main["failure_rate_pct"] = (
    main.failure_rate * 100
).astype(int)

main["method_order"] = main.method.map(
    {name: i for i, name in enumerate(METHODS)}
)

main = main.sort_values(
    ["failure_rate_pct", "method_order"]
)

main = main.drop(
    columns=["failure_rate", "method_order"]
)

main = main[[
    "failure_rate_pct",
    "method",
    "runs",
    "success_pct",
    "executed_iterations",
    "completed_tasks",
    "blocked_moves",
    "total_astar",
    "global_calls",
    "stall_checks",
    "stall_transfers",
    "recovery_calls",
    "total_distance",
]]

main = main.round(4)

main_path = OUT / "table_6_ablation_results.csv"
main.to_csv(main_path, index=False)

# Use existing sign tests and Holm-corrected p-values.
comparisons = stats[
    stats.study == "ablation"
].copy()

assert len(comparisons) == 8
assert comparisons.n_pairs.eq(30).all()
assert comparisons.significant.astype(str).str.lower().eq(
    "true"
).sum() == 4

comparisons["failure_rate_pct"] = (
    comparisons.failure_rate * 100
).astype(int)

comparisons["blocking_reduction_pct"] = (
    100 * (
        comparisons.baseline_mean -
        comparisons.proposed_mean
    ) / comparisons.baseline_mean
)

comparisons = comparisons[[
    "failure_rate_pct",
    "baseline",
    "n_pairs",
    "proposed_mean",
    "baseline_mean",
    "blocking_reduction_pct",
    "mean_difference",
    "ci95_low",
    "ci95_high",
    "wins",
    "ties",
    "losses",
    "p_value",
    "p_holm",
    "significant",
]].sort_values(["failure_rate_pct", "baseline"])

comp_path = OUT / "table_6_ablation_blocking_statistics.csv"
comparisons.to_csv(comp_path, index=False)

failure_path = OUT / "table_6_ablation_failed_runs.csv"
failures.to_csv(failure_path, index=False)

print("===== TABLE VI GENERATED =====")
print("Recorded executions:", len(raw))
print("Successful executions:", int(raw.success.sum()))
print("Unsuccessful executions:", len(failures))
print("Main table rows:", len(main))
print("Paired comparisons:", len(comparisons))
print(
    "Holm-significant blocking comparisons:",
    int(comparisons.significant.astype(str).str.lower().eq("true").sum())
)
print("Saved:", main_path)
print("Saved:", comp_path)
print("Saved:", failure_path)
print()
print("Unsuccessful cases:")
print(
    failures[
        ["seed", "failure_rate", "method", "iterations",
         "completed_tasks", "blocked_moves"]
    ].to_string(index=False)
)
