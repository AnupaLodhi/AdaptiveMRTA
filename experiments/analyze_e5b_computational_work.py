from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]

RAW = (
    ROOT
    / "data/raw"
    / "e5b_computational_work_30_seeds.csv"
)

PROCESSED = (
    ROOT
    / "data/processed"
    / "e5b_computational_work_summary.csv"
)

TABLE = (
    ROOT
    / "results/tables"
    / "e5b_computational_work_key_results.csv"
)


def safe_wilcoxon(a, b):
    diff = np.asarray(a) - np.asarray(b)

    if np.allclose(diff, 0):
        return np.nan, 1.0

    result = wilcoxon(
        a,
        b,
        zero_method="wilcox",
        alternative="two-sided",
    )

    return (
        float(result.statistic),
        float(result.pvalue),
    )


def pct_reduction(old, new):
    old = np.asarray(old, dtype=float)
    new = np.asarray(new, dtype=float)

    return (
        (old - new) / old
    ) * 100.0


def analyze_condition(df, failure_rate):

    subset = df[
        np.isclose(
            df["failure_rate"],
            failure_rate,
        )
    ].copy()

    repeated = (
        subset[
            subset["method"] == "repeated"
        ]
        .sort_values("seed")
        .reset_index(drop=True)
    )

    event = (
        subset[
            subset["method"] == "event_global"
        ]
        .sort_values("seed")
        .reset_index(drop=True)
    )

    assert len(repeated) == 30
    assert len(event) == 30

    assert (
        repeated["seed"].tolist()
        == event["seed"].tolist()
    )

    assert (
        repeated[
            "scenario_fingerprint"
        ].tolist()
        == event[
            "scenario_fingerprint"
        ].tolist()
    )

    assert (
        repeated[
            "failed_agent_ids"
        ].fillna("").tolist()
        == event[
            "failed_agent_ids"
        ].fillna("").tolist()
    )

    iteration_equal = (
        repeated["iterations"].to_numpy()
        == event["iterations"].to_numpy()
    )

    distance_equal = np.isclose(
        repeated[
            "total_distance"
        ].to_numpy(),
        event[
            "total_distance"
        ].to_numpy(),
    )

    astar_reduction = pct_reduction(
        repeated[
            "path_cost_evaluations"
        ],
        event[
            "path_cost_evaluations"
        ],
    )

    global_reduction = pct_reduction(
        repeated[
            "global_assignment_calls"
        ],
        event[
            "global_assignment_calls"
        ],
    )

    runtime_reduction = pct_reduction(
        repeated["runtime_seconds"],
        event["runtime_seconds"],
    )

    astar_stat, astar_p = (
        safe_wilcoxon(
            repeated[
                "path_cost_evaluations"
            ],
            event[
                "path_cost_evaluations"
            ],
        )
    )

    global_stat, global_p = (
        safe_wilcoxon(
            repeated[
                "global_assignment_calls"
            ],
            event[
                "global_assignment_calls"
            ],
        )
    )

    runtime_stat, runtime_p = (
        safe_wilcoxon(
            repeated[
                "runtime_seconds"
            ],
            event[
                "runtime_seconds"
            ],
        )
    )

    label = (
        "static"
        if np.isclose(failure_rate, 0.0)
        else "20% failure"
    )

    result = {
        "condition":
            label,

        "paired_runs":
            len(repeated),

        "repeated_success_pct":
            repeated["success"].mean() * 100,

        "event_success_pct":
            event["success"].mean() * 100,

        "identical_iterations":
            int(iteration_equal.sum()),

        "identical_distance":
            int(distance_equal.sum()),

        "repeated_mean_iterations":
            repeated["iterations"].mean(),

        "event_mean_iterations":
            event["iterations"].mean(),

        "repeated_mean_distance":
            repeated["total_distance"].mean(),

        "event_mean_distance":
            event["total_distance"].mean(),

        "repeated_mean_global_calls":
            repeated[
                "global_assignment_calls"
            ].mean(),

        "event_mean_global_calls":
            event[
                "global_assignment_calls"
            ].mean(),

        "mean_global_reduction_pct":
            global_reduction.mean(),

        "median_global_reduction_pct":
            np.median(global_reduction),

        "repeated_mean_astar":
            repeated[
                "path_cost_evaluations"
            ].mean(),

        "repeated_sd_astar":
            repeated[
                "path_cost_evaluations"
            ].std(ddof=1),

        "event_mean_astar":
            event[
                "path_cost_evaluations"
            ].mean(),

        "event_sd_astar":
            event[
                "path_cost_evaluations"
            ].std(ddof=1),

        "mean_astar_reduction_pct":
            astar_reduction.mean(),

        "median_astar_reduction_pct":
            np.median(astar_reduction),

        "min_astar_reduction_pct":
            astar_reduction.min(),

        "max_astar_reduction_pct":
            astar_reduction.max(),

        "astar_reduction_wins":
            int(
                (
                    event[
                        "path_cost_evaluations"
                    ].to_numpy()
                    <
                    repeated[
                        "path_cost_evaluations"
                    ].to_numpy()
                ).sum()
            ),

        "astar_wilcoxon_stat":
            astar_stat,

        "astar_wilcoxon_p":
            astar_p,

        "repeated_mean_runtime":
            repeated[
                "runtime_seconds"
            ].mean(),

        "event_mean_runtime":
            event[
                "runtime_seconds"
            ].mean(),

        "mean_runtime_reduction_pct":
            runtime_reduction.mean(),

        "median_runtime_reduction_pct":
            np.median(runtime_reduction),

        "runtime_wins":
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

        "global_wilcoxon_stat":
            global_stat,

        "global_wilcoxon_p":
            global_p,

        "runtime_wilcoxon_stat":
            runtime_stat,

        "runtime_wilcoxon_p":
            runtime_p,
    }

    print()
    print("=" * 78)
    print(label.upper())
    print("=" * 78)

    print(
        "Success:",
        f"Repeated={result['repeated_success_pct']:.1f}%",
        f"Event={result['event_success_pct']:.1f}%",
    )

    print(
        "Identical iterations:",
        f"{result['identical_iterations']}/30",
    )

    print(
        "Identical distance:",
        f"{result['identical_distance']}/30",
    )

    print()
    print(
        "Global calls:",
        f"{result['repeated_mean_global_calls']:.2f}",
        "->",
        f"{result['event_mean_global_calls']:.2f}",
    )

    print(
        "Mean paired global-call reduction:",
        f"{result['mean_global_reduction_pct']:.2f}%",
    )

    print()
    print(
        "A* evaluations:",
        f"{result['repeated_mean_astar']:.2f}",
        "->",
        f"{result['event_mean_astar']:.2f}",
    )

    print(
        "A* SD:",
        f"{result['repeated_sd_astar']:.2f}",
        "->",
        f"{result['event_sd_astar']:.2f}",
    )

    print(
        "Mean paired A* reduction:",
        f"{result['mean_astar_reduction_pct']:.2f}%",
    )

    print(
        "Median paired A* reduction:",
        f"{result['median_astar_reduction_pct']:.2f}%",
    )

    print(
        "A* reduction range:",
        f"{result['min_astar_reduction_pct']:.2f}%",
        "to",
        f"{result['max_astar_reduction_pct']:.2f}%",
    )

    print(
        "A* reduction wins:",
        f"{result['astar_reduction_wins']}/30",
    )

    print(
        "A* Wilcoxon p:",
        f"{result['astar_wilcoxon_p']:.6g}",
    )

    print()
    print(
        "Runtime:",
        f"{result['repeated_mean_runtime']:.3f}s",
        "->",
        f"{result['event_mean_runtime']:.3f}s",
    )

    print(
        "Mean paired runtime reduction:",
        f"{result['mean_runtime_reduction_pct']:.2f}%",
    )

    print(
        "Median paired runtime reduction:",
        f"{result['median_runtime_reduction_pct']:.2f}%",
    )

    print(
        "Runtime wins:",
        f"{result['runtime_wins']}/30",
    )

    print(
        "Runtime Wilcoxon p:",
        f"{result['runtime_wilcoxon_p']:.6g}",
    )

    return result


def main():

    df = pd.read_csv(RAW)

    print()
    print("=" * 78)
    print("E5B COMPUTATIONAL-WORK ANALYSIS")
    print("=" * 78)

    print("Rows:", len(df))

    assert len(df) == 120

    assert set(
        df["method"].unique()
    ) == {
        "repeated",
        "event_global",
    }

    assert set(
        np.round(
            df["failure_rate"].unique(),
            2,
        )
    ) == {
        0.0,
        0.2,
    }

    # Counter integrity.
    assert (
        df["path_cost_evaluations"]
        ==
        (
            df["successful_astar_queries"]
            +
            df["unreachable_astar_queries"]
        )
    ).all()

    print(
        "Counter integrity: PASS"
    )

    results = []

    for rate in [0.0, 0.2]:
        results.append(
            analyze_condition(
                df,
                rate,
            )
        )

    summary = pd.DataFrame(results)

    PROCESSED.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    TABLE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary.to_csv(
        PROCESSED,
        index=False,
    )

    key_columns = [
        "condition",
        "paired_runs",
        "identical_iterations",
        "identical_distance",
        "repeated_mean_global_calls",
        "event_mean_global_calls",
        "mean_global_reduction_pct",
        "repeated_mean_astar",
        "event_mean_astar",
        "mean_astar_reduction_pct",
        "median_astar_reduction_pct",
        "min_astar_reduction_pct",
        "max_astar_reduction_pct",
        "astar_reduction_wins",
        "astar_wilcoxon_p",
        "repeated_mean_runtime",
        "event_mean_runtime",
        "mean_runtime_reduction_pct",
        "runtime_wins",
        "runtime_wilcoxon_p",
    ]

    summary[
        key_columns
    ].to_csv(
        TABLE,
        index=False,
    )

    static = results[0]
    failure = results[1]

    total_identical_iterations = (
        static["identical_iterations"]
        +
        failure["identical_iterations"]
    )

    total_identical_distance = (
        static["identical_distance"]
        +
        failure["identical_distance"]
    )

    total_astar_wins = (
        static["astar_reduction_wins"]
        +
        failure["astar_reduction_wins"]
    )

    total_runtime_wins = (
        static["runtime_wins"]
        +
        failure["runtime_wins"]
    )

    print()
    print("=" * 78)
    print("OVERALL VALIDATION")
    print("=" * 78)

    print(
        "Identical iterations:",
        f"{total_identical_iterations}/60",
    )

    print(
        "Identical distance:",
        f"{total_identical_distance}/60",
    )

    print(
        "Fewer A* evaluations:",
        f"{total_astar_wins}/60",
    )

    print(
        "Faster runtime:",
        f"{total_runtime_wins}/60",
    )

    print()
    print(
        "Saved:",
        PROCESSED,
    )

    print(
        "Saved:",
        TABLE,
    )

    print("=" * 78)


if __name__ == "__main__":
    main()
