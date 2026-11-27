"""Generate Table VIII: statistical evidence and controlled runtime."""

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "manuscript/tables"
OUT.mkdir(parents=True, exist_ok=True)

raw = pd.read_csv(ROOT / "data/raw/casr_v2_runtime.csv")
scenario_saved = pd.read_csv(
    ROOT / "data/processed/casr_runtime_scenario_medians.csv"
)
summary_saved = pd.read_csv(
    ROOT / "data/processed/casr_runtime_summary.csv"
)
runtime_stats = pd.read_csv(
    ROOT / "data/processed/casr_runtime_paired_statistics.csv"
)
holdout_stats = pd.read_csv(
    ROOT / "data/processed/casr_v2_holdout_statistics.csv"
)
blocking_stats = pd.read_csv(
    ROOT / "data/processed/casr_final_blocking_statistics.csv"
)

METHODS = {
    "lock_aware", "adaptive", "palr_5",
    "casr_3", "casr_v2_s2"
}
CONFIGS = {"medium", "extra_large"}

# Validate experimental design.
assert len(raw) == 300
assert raw.success.eq(1).all()
assert raw.runtime_seconds.gt(0).all()
assert set(raw.seed) == set(range(6000, 6005))
assert set(raw.method) == METHODS
assert set(raw.configuration) == CONFIGS
assert set(raw.failure_rate) == {0.0, 0.2}
assert set(raw.repetition) == {0, 1, 2}

keys = ["configuration", "seed", "failure_rate", "method"]

assert not raw.duplicated(
    keys + ["repetition"]
).any()

assert raw.groupby(keys).size().eq(3).all()

# Three repetitions must reproduce non-timing outcomes.
for metric in [
    "success", "iterations", "completed_tasks",
    "blocked_moves", "total_astar",
    "scenario_fingerprint"
]:
    assert raw.groupby(keys)[metric].nunique().eq(1).all()

# Recalculate scenario-level timing medians.
scenario = raw.groupby(keys, as_index=False).agg(
    median_runtime=("runtime_seconds", "median")
)

assert len(scenario) == 100

check_scenarios = scenario.merge(
    scenario_saved[keys + ["median_runtime"]],
    on=keys,
    suffixes=("_raw", "_saved"),
    validate="one_to_one"
)

assert len(check_scenarios) == 100
assert np.allclose(
    check_scenarios.median_runtime_raw,
    check_scenarios.median_runtime_saved,
    rtol=0,
    atol=1e-9
)

# Summarize means of scenario-level medians.
summary = scenario.groupby(
    ["configuration", "failure_rate", "method"],
    as_index=False
).agg(
    scenarios=("seed", "size"),
    mean_runtime=("median_runtime", "mean"),
    median_runtime=("median_runtime", "median"),
)

assert len(summary) == 20
assert summary.scenarios.eq(5).all()

check = summary.merge(
    summary_saved,
    on=["configuration", "failure_rate", "method"],
    suffixes=("_raw", "_saved"),
    validate="one_to_one"
)

assert len(check) == 20

for col in ["scenarios", "mean_runtime", "median_runtime"]:
    assert np.allclose(
        check[f"{col}_raw"],
        check[f"{col}_saved"],
        rtol=0,
        atol=1e-9
    ), f"Runtime summary mismatch: {col}"

summary["failure_rate_pct"] = (
    summary.failure_rate * 100
).astype(int)

summary = summary[[
    "configuration", "failure_rate_pct", "method",
    "scenarios", "mean_runtime", "median_runtime"
]].round(6)

runtime_path = OUT / "table_8_controlled_runtime.csv"
summary.to_csv(runtime_path, index=False)

# Preserve the original Holm-corrected paired tests.
assert len(runtime_stats) == 16
assert runtime_stats.n_pairs.eq(5).all()
assert runtime_stats.significant.astype(str).str.lower().eq(
    "true"
).sum() == 0

# Verify proposed and baseline means against scenario medians.
lookup = scenario.groupby(
    ["configuration", "failure_rate", "method"]
).median_runtime.mean().to_dict()

for row in runtime_stats.itertuples(index=False):
    key = (row.configuration, row.failure_rate)

    assert np.isclose(
        row.proposed_runtime,
        lookup[key + ("casr_v2_s2",)],
        atol=1e-9
    )
    assert np.isclose(
        row.baseline_runtime,
        lookup[key + (row.baseline,)],
        atol=1e-9
    )

paired = runtime_stats.copy()
paired["failure_rate_pct"] = (
    paired.failure_rate * 100
).astype(int)

paired = paired[[
    "configuration", "failure_rate_pct", "baseline",
    "n_pairs", "proposed_runtime", "baseline_runtime",
    "reduction_pct", "ci95_low", "ci95_high",
    "median_speedup", "wins", "ties", "losses",
    "p_value", "p_holm", "significant"
]]

paired_path = OUT / "table_8_runtime_paired_statistics.csv"
paired.to_csv(paired_path, index=False)

# Reconcile the statistical test families.
ablation = blocking_stats[
    blocking_stats.study == "ablation"
]
scalability = blocking_stats[
    blocking_stats.study == "scalability"
]

def n_sig(frame):
    return int(
        frame.significant.astype(str).str.lower().eq(
            "true"
        ).sum()
    )

assert len(holdout_stats) == 60
assert len(ablation) == 8
assert len(scalability) == 48

assert n_sig(holdout_stats) == 21
assert n_sig(ablation) == 4
assert n_sig(scalability) == 0
assert n_sig(runtime_stats) == 0

evidence = pd.DataFrame([
    {
        "analysis": "Independent holdout",
        "test": "Paired two-sided Wilcoxon",
        "correction_family": "Holdout: 60 comparisons",
        "comparisons": 60,
        "holm_significant": n_sig(holdout_stats),
        "paired_scenarios_per_comparison": 20,
    },
    {
        "analysis": "Ablation blocking",
        "test": "Paired two-sided exact sign test",
        "correction_family": "Blocking: 56 comparisons",
        "comparisons": 8,
        "holm_significant": n_sig(ablation),
        "paired_scenarios_per_comparison": 30,
    },
    {
        "analysis": "Scalability blocking",
        "test": "Paired two-sided exact sign test",
        "correction_family": "Blocking: 56 comparisons",
        "comparisons": 48,
        "holm_significant": n_sig(scalability),
        "paired_scenarios_per_comparison": 10,
    },
    {
        "analysis": "Controlled runtime",
        "test": "Paired two-sided Wilcoxon",
        "correction_family": "Runtime: 16 comparisons",
        "comparisons": 16,
        "holm_significant": n_sig(runtime_stats),
        "paired_scenarios_per_comparison": 5,
    },
])

evidence_path = OUT / "table_8_statistical_evidence.csv"
evidence.to_csv(evidence_path, index=False)

print("===== TABLE VIII GENERATED =====")
print("Timed executions:", len(raw))
print("Successful executions:", int(raw.success.sum()))
print("Scenario-method medians:", len(scenario))
print("Runtime summary rows:", len(summary))
print("Paired runtime comparisons:", len(paired))
print("Runtime Holm-significant:", n_sig(runtime_stats))
print("Statistical evidence rows:", len(evidence))
print("Saved:", runtime_path)
print("Saved:", paired_path)
print("Saved:", evidence_path)

print("\n===== STATISTICAL EVIDENCE =====")
print(evidence.to_string(index=False))

print("\n===== PROPOSED VS CASR-3 RUNTIME =====")
print(
    paired[paired.baseline == "casr_3"][
        ["configuration", "failure_rate_pct",
         "reduction_pct", "p_holm", "significant"]
    ].round(4).to_string(index=False)
)
