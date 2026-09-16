from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

INPUT = ROOT / "data/raw/e5_replanning_tradeoff_30_seeds.csv"
OUT = ROOT / "data/processed"
TABLES = ROOT / "results/tables"

OUT.mkdir(parents=True, exist_ok=True)
TABLES.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT)

print("=" * 92)
print("E5: REPLANNING TRADE-OFF ANALYSIS")
print("=" * 92)

# ------------------------------------------------------------
# Integrity
# ------------------------------------------------------------

assert len(df) == 180

groups = df.groupby(
    ["failure_rate", "seed"]
)

assert (
    groups["method"].nunique() == 3
).all()

assert (
    groups["scenario_fingerprint"].nunique()
    == 1
).all()

assert (
    groups["failed_agent_ids"].nunique(
        dropna=False
    )
    == 1
).all()

print("INTEGRITY: PASS")
print("Rows:", len(df))

# ------------------------------------------------------------
# Descriptive summary
# ------------------------------------------------------------

summary = (
    df.groupby(
        ["failure_rate", "method"]
    )
    .agg(
        n=("seed", "count"),
        success_rate=("success", "mean"),
        mean_iterations=("iterations", "mean"),
        sd_iterations=("iterations", "std"),
        mean_distance=("total_distance", "mean"),
        sd_distance=("total_distance", "std"),
        mean_calls=(
            "global_assignment_calls",
            "mean",
        ),
        sd_calls=(
            "global_assignment_calls",
            "std",
        ),
        mean_runtime=(
            "runtime_seconds",
            "mean",
        ),
        sd_runtime=(
            "runtime_seconds",
            "std",
        ),
    )
    .reset_index()
)

summary["success_rate"] *= 100

summary.to_csv(
    OUT / "e5_replanning_summary.csv",
    index=False,
)

print()
print("DESCRIPTIVE RESULTS")
print(
    summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)

# ------------------------------------------------------------
# Event-global vs repeated
# ------------------------------------------------------------

rows = []

for rate in sorted(
    df["failure_rate"].unique()
):

    block = df[
        np.isclose(
            df["failure_rate"],
            rate,
        )
    ]

    repeated = (
        block[
            block["method"] == "repeated"
        ]
        .sort_values("seed")
        .reset_index(drop=True)
    )

    event = (
        block[
            block["method"]
            == "event_global"
        ]
        .sort_values("seed")
        .reset_index(drop=True)
    )

    assert np.array_equal(
        repeated["seed"],
        event["seed"],
    )

    iteration_equal = (
        repeated["iterations"].to_numpy()
        ==
        event["iterations"].to_numpy()
    )

    distance_equal = np.isclose(
        repeated["total_distance"].to_numpy(),
        event["total_distance"].to_numpy(),
    )

    call_reduction = (
        repeated[
            "global_assignment_calls"
        ].to_numpy(float)
        -
        event[
            "global_assignment_calls"
        ].to_numpy(float)
    )

    call_reduction_pct = (
        call_reduction
        /
        repeated[
            "global_assignment_calls"
        ].to_numpy(float)
        * 100
    )

    runtime_reduction = (
        repeated[
            "runtime_seconds"
        ].to_numpy(float)
        -
        event[
            "runtime_seconds"
        ].to_numpy(float)
    )

    runtime_reduction_pct = (
        runtime_reduction
        /
        repeated[
            "runtime_seconds"
        ].to_numpy(float)
        * 100
    )

    rows.append(
        {
            "failure_rate": rate,

            "paired_scenarios":
                len(repeated),

            "identical_iterations":
                int(iteration_equal.sum()),

            "identical_distance":
                int(distance_equal.sum()),

            "repeated_mean_calls":
                repeated[
                    "global_assignment_calls"
                ].mean(),

            "event_mean_calls":
                event[
                    "global_assignment_calls"
                ].mean(),

            "mean_call_reduction":
                call_reduction.mean(),

            "mean_call_reduction_pct":
                call_reduction_pct.mean(),

            "median_call_reduction_pct":
                np.median(
                    call_reduction_pct
                ),

            "repeated_mean_runtime":
                repeated[
                    "runtime_seconds"
                ].mean(),

            "event_mean_runtime":
                event[
                    "runtime_seconds"
                ].mean(),

            "mean_runtime_reduction_pct":
                runtime_reduction_pct.mean(),

            "event_faster_count":
                int(
                    (
                        event[
                            "runtime_seconds"
                        ].to_numpy()
                        <
                        repeated[
                            "runtime_seconds"
                        ].to_numpy()
                    ).sum()
                ),
        }
    )

comparison = pd.DataFrame(rows)

comparison.to_csv(
    TABLES
    / "e5_event_global_tradeoff.csv",
    index=False,
)

print()
print("=" * 92)
print("EVENT-GLOBAL VS REPEATED")
print("=" * 92)

print(
    comparison.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)

# ------------------------------------------------------------
# Local repair reliability
# ------------------------------------------------------------

local = (
    df[
        df["method"] == "local_repair"
    ]
    .groupby("failure_rate")
    .agg(
        runs=("seed", "count"),
        successes=("success", "sum"),
        success_rate=("success", "mean"),
        timeouts=(
            "success",
            lambda x: int(
                (x == 0).sum()
            ),
        ),
        mean_calls=(
            "global_assignment_calls",
            "mean",
        ),
        mean_runtime=(
            "runtime_seconds",
            "mean",
        ),
    )
    .reset_index()
)

local["success_rate"] *= 100

local.to_csv(
    TABLES
    / "e5_local_repair_reliability.csv",
    index=False,
)

print()
print("=" * 92)
print("LOCAL REPAIR RELIABILITY")
print("=" * 92)

print(
    local.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)

# ------------------------------------------------------------
# Overall exact-equivalence check
# ------------------------------------------------------------

rep = (
    df[df["method"] == "repeated"]
    .sort_values(
        ["failure_rate", "seed"]
    )
    .reset_index(drop=True)
)

evt = (
    df[df["method"] == "event_global"]
    .sort_values(
        ["failure_rate", "seed"]
    )
    .reset_index(drop=True)
)

same_iter = np.array_equal(
    rep["iterations"].to_numpy(),
    evt["iterations"].to_numpy(),
)

same_dist = np.allclose(
    rep["total_distance"].to_numpy(),
    evt["total_distance"].to_numpy(),
)

print()
print("=" * 92)
print("OVERALL CHECK")
print("=" * 92)
print(
    "All 60 paired iterations identical:",
    same_iter,
)
print(
    "All 60 paired distances identical:",
    same_dist,
)

print()
print("FILES SAVED:")
print(
    OUT / "e5_replanning_summary.csv"
)
print(
    TABLES
    / "e5_event_global_tradeoff.csv"
)
print(
    TABLES
    / "e5_local_repair_reliability.csv"
)
print("=" * 92)
