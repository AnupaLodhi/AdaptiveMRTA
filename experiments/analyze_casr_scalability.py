from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

df = pd.read_csv(
    ROOT / "data/raw/casr_v2_scalability.csv"
)

keys = [
    "configuration",
    "failure_rate",
    "seed",
    "method",
]

assert len(df) == 600
assert not df.duplicated(keys).any()

assert df.groupby(
    ["configuration", "failure_rate", "method"]
).size().eq(10).all()

assert df.configuration.nunique() == 6
assert df.method.nunique() == 5
assert set(df.seed) == set(range(5000, 5010))

out = ROOT / "data/processed"
out.mkdir(parents=True, exist_ok=True)

df["successful_makespan"] = df.iterations.where(
    df.success == 1
)

df["task_completion_pct"] = (
    100 * df.completed_tasks / df.tasks
)

summary = df.groupby(
    ["configuration", "failure_rate", "method"]
).agg(
    runs=("success", "size"),
    success_pct=("success", lambda x: 100*x.mean()),
    task_completion_pct=("task_completion_pct", "mean"),
    makespan=("successful_makespan", "mean"),
    termination_iterations=("iterations", "mean"),
    distance=("total_distance", "mean"),
    blocked=("blocked_moves", "mean"),
    astar=("total_astar", "mean"),
    global_calls=("global_calls", "mean"),
    runtime=("runtime_seconds", "mean"),
    stall_transfers=("stall_transfers", "mean"),
).round(4)

summary.to_csv(
    out / "casr_v2_scalability_summary.csv"
)

print("\n===== SCALABILITY SUMMARY =====")
print(summary.to_string())

print("\n===== COMPLETION =====")
print("Total runs:", len(df))
print("Successful:", int(df.success.sum()))
print("Failed:", int((df.success == 0).sum()))

failed = df[df.success == 0]

failed.to_csv(
    out / "casr_v2_scalability_failures.csv",
    index=False
)

print("\n===== FAILED RUNS =====")
print(failed.to_string(index=False))

print("\n===== OVERALL METHOD SUMMARY =====")

overall = df.groupby("method").agg(
    runs=("success", "size"),
    success_pct=("success", lambda x: 100*x.mean()),
    makespan=("successful_makespan", "mean"),
    termination_iterations=("iterations", "mean"),
    blocked=("blocked_moves", "mean"),
    astar=("total_astar", "mean"),
    runtime=("runtime_seconds", "mean"),
    distance=("total_distance", "mean"),
).round(4)

print(overall.to_string())

overall.to_csv(
    out / "casr_v2_scalability_overall.csv"
)

rows = []

for (config, rate), subset in df.groupby(
    ["configuration", "failure_rate"]
):
    for metric in [
        "success",
        "iterations",
        "blocked_moves",
        "total_astar",
        "total_distance",
        "global_calls",
        "runtime_seconds",
    ]:
        pivot = subset.pivot(
            index="seed",
            columns="method",
            values=metric,
        )

        for baseline in [
            "casr_3",
            "palr_5",
            "adaptive",
            "lock_aware",
        ]:
            proposed = pivot["casr_v2_s2"]
            reference = pivot[baseline]

            diff = proposed - reference

            if metric == "success":
                diff = -diff

            reduction = (
                100 * (
                    reference.mean() - proposed.mean()
                ) / reference.mean()
                if reference.mean() != 0
                and metric != "success"
                else np.nan
            )

            rows.append({
                "configuration": config,
                "failure_rate": rate,
                "metric": metric,
                "baseline": baseline,
                "proposed_mean": proposed.mean(),
                "baseline_mean": reference.mean(),
                "reduction_pct": reduction,
                "wins": int((diff < 0).sum()),
                "ties": int((diff == 0).sum()),
                "losses": int((diff > 0).sum()),
            })

paired = pd.DataFrame(rows)

paired.to_csv(
    out / "casr_v2_scalability_paired.csv",
    index=False
)

print("\n===== CASR-V2 VS CASR-3 =====")

print(
    paired[
        (paired.baseline == "casr_3")
        & paired.metric.isin([
            "blocked_moves",
            "total_astar",
            "iterations",
        ])
    ].round(4).to_string(index=False)
)

print("\n===== ANALYSIS COMPLETE =====")
