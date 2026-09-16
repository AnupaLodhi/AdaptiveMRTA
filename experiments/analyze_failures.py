from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

input_path = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "failure_trials.csv"
)

output_dir = (
    PROJECT_ROOT
    / "results"
    / "tables"
)

processed_dir = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

output_dir.mkdir(
    parents=True,
    exist_ok=True
)

processed_dir.mkdir(
    parents=True,
    exist_ok=True
)


df = pd.read_csv(input_path)


# ---------------------------------------
# NORMALIZED METRICS
# ---------------------------------------

df["surviving_agents"] = (
    15 - df["failed_agents"]
)

df["distance_per_task"] = (
    df["total_distance"]
    / df["tasks_completed"]
)

df["distance_per_surviving_agent"] = (
    df["total_distance"]
    / df["surviving_agents"]
)

df["tasks_per_surviving_agent"] = (
    df["tasks_completed"]
    / df["surviving_agents"]
)


processed_path = (
    processed_dir
    / "failure_trials_normalized.csv"
)

df.to_csv(
    processed_path,
    index=False
)


# ---------------------------------------
# SUMMARY
# ---------------------------------------

summary = (
    df.groupby(
        [
            "failure_rate",
            "algorithm",
        ]
    )
    .agg(
        success_rate=(
            "success",
            "mean",
        ),
        mean_iterations=(
            "iterations",
            "mean",
        ),
        mean_total_distance=(
            "total_distance",
            "mean",
        ),
        mean_distance_per_task=(
            "distance_per_task",
            "mean",
        ),
        mean_distance_per_surviving_agent=(
            "distance_per_surviving_agent",
            "mean",
        ),
        mean_tasks_per_surviving_agent=(
            "tasks_per_surviving_agent",
            "mean",
        ),
    )
    .reset_index()
)


summary["success_rate"] *= 100


summary_path = (
    output_dir
    / "failure_normalized_summary.csv"
)

summary.to_csv(
    summary_path,
    index=False
)


print("\n==============================================")
print("NORMALIZED FAILURE ANALYSIS")
print("==============================================\n")


for failure_rate in sorted(
    summary["failure_rate"].unique()
):

    print(
        f"Failure rate: "
        f"{failure_rate * 100:.0f}%"
    )

    print("-" * 90)

    subset = summary[
        summary["failure_rate"]
        == failure_rate
    ]

    for _, row in subset.iterrows():

        print(
            f"{row['algorithm']:12s} | "
            f"Iter={row['mean_iterations']:6.2f} | "
            f"TotalDist={row['mean_total_distance']:8.2f} | "
            f"Dist/Task={row['mean_distance_per_task']:6.2f} | "
            f"Dist/Robot={row['mean_distance_per_surviving_agent']:6.2f} | "
            f"Tasks/Robot={row['mean_tasks_per_surviving_agent']:5.2f}"
        )

    print()


print("Processed dataset:")
print(processed_path)

print("\nSummary:")
print(summary_path)
