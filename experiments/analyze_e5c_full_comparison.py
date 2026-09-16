from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]

RAW = (
    ROOT
    / "data/raw"
    / "e5c_full_comparison_30_seeds.csv"
)

PROCESSED = (
    ROOT
    / "data/processed"
    / "e5c_full_comparison_summary.csv"
)

PAIRED_OUT = (
    ROOT
    / "data/processed"
    / "e5c_lock_vs_event_paired.csv"
)

KEY_OUT = (
    ROOT
    / "results/tables"
    / "e5c_lock_aware_key_results.csv"
)

df = pd.read_csv(RAW)

assert len(df) == 240
assert set(df["method"]) == {
    "repeated",
    "event_global",
    "local_repair",
    "lock_aware",
}

# --------------------------------------------------
# Pairing integrity
# --------------------------------------------------

for rate in sorted(df["failure_rate"].unique()):

    part = df[df["failure_rate"] == rate]

    for seed in sorted(part["seed"].unique()):

        pair = part[part["seed"] == seed]

        assert len(pair) == 4

        assert (
            pair["scenario_fingerprint"].nunique()
            == 1
        )

        assert (
            pair["failed_agent_ids"]
            .fillna("")
            .nunique()
            == 1
        )

print("Pairing integrity: PASS")


# --------------------------------------------------
# Descriptive summary
# --------------------------------------------------

summary_rows = []

for rate in sorted(df["failure_rate"].unique()):

    for method in [
        "repeated",
        "event_global",
        "local_repair",
        "lock_aware",
    ]:

        g = df[
            (df["failure_rate"] == rate)
            & (df["method"] == method)
        ]

        successful = g[g["success"] == 1]

        summary_rows.append({
            "failure_rate":
                rate,

            "method":
                method,

            "success_count":
                int(g["success"].sum()),

            "success_rate_percent":
                100 * g["success"].mean(),

            "mean_iterations_all":
                g["iterations"].mean(),

            "mean_iterations_success_only":
                (
                    successful["iterations"].mean()
                    if len(successful)
                    else np.nan
                ),

            "mean_distance_all":
                g["total_distance"].mean(),

            "mean_distance_success_only":
                (
                    successful[
                        "total_distance"
                    ].mean()
                    if len(successful)
                    else np.nan
                ),

            "mean_global_calls":
                g[
                    "global_assignment_calls"
                ].mean(),

            "mean_local_repairs":
                g[
                    "local_repair_calls"
                ].mean(),

            "mean_lock_escalations":
                g[
                    "lock_escalations"
                ].mean(),

            "mean_astar":
                g[
                    "path_cost_evaluations"
                ].mean(),

            "sd_astar":
                g[
                    "path_cost_evaluations"
                ].std(ddof=1),

            "median_astar":
                g[
                    "path_cost_evaluations"
                ].median(),

            "mean_runtime_seconds":
                g[
                    "runtime_seconds"
                ].mean(),

            "median_runtime_seconds":
                g[
                    "runtime_seconds"
                ].median(),
        })


summary = pd.DataFrame(
    summary_rows
)

PROCESSED.parent.mkdir(
    parents=True,
    exist_ok=True,
)

summary.to_csv(
    PROCESSED,
    index=False,
)


# --------------------------------------------------
# Helpers
# --------------------------------------------------

def bootstrap_mean_ci(
    values,
    seed=12345,
    n_boot=20000,
):
    values = np.asarray(
        values,
        dtype=float,
    )

    rng = np.random.default_rng(seed)

    means = np.empty(
        n_boot,
        dtype=float,
    )

    n = len(values)

    for i in range(n_boot):
        sample = rng.choice(
            values,
            size=n,
            replace=True,
        )
        means[i] = sample.mean()

    return np.percentile(
        means,
        [2.5, 97.5],
    )


def rank_biserial_paired(diff):
    """
    Rank-biserial correlation for paired differences.

    Positive means the differences tend to be positive.
    Zeros are removed.
    """

    diff = np.asarray(
        diff,
        dtype=float,
    )

    diff = diff[diff != 0]

    if len(diff) == 0:
        return 0.0

    abs_diff = np.abs(diff)

    ranks = pd.Series(
        abs_diff
    ).rank(
        method="average"
    ).to_numpy()

    positive = ranks[
        diff > 0
    ].sum()

    negative = ranks[
        diff < 0
    ].sum()

    total = positive + negative

    if total == 0:
        return 0.0

    return (
        positive - negative
    ) / total


def safe_wilcoxon(diff):
    diff = np.asarray(
        diff,
        dtype=float,
    )

    if np.all(diff == 0):
        return np.nan

    return wilcoxon(
        diff,
        zero_method="wilcox",
        alternative="two-sided",
    ).pvalue


def counts_lower_better(
    lock,
    event,
):
    lock = np.asarray(lock)
    event = np.asarray(event)

    return (
        int(np.sum(lock < event)),
        int(np.sum(lock == event)),
        int(np.sum(lock > event)),
    )


# --------------------------------------------------
# Lock-Aware vs Event-Global paired analysis
# --------------------------------------------------

paired_rows = []
key_rows = []

for rate in sorted(df["failure_rate"].unique()):

    event = (
        df[
            (df["failure_rate"] == rate)
            & (df["method"] == "event_global")
        ]
        .sort_values("seed")
        .reset_index(drop=True)
    )

    lock = (
        df[
            (df["failure_rate"] == rate)
            & (df["method"] == "lock_aware")
        ]
        .sort_values("seed")
        .reset_index(drop=True)
    )

    assert np.array_equal(
        event["seed"].to_numpy(),
        lock["seed"].to_numpy(),
    )

    assert np.array_equal(
        event[
            "scenario_fingerprint"
        ].to_numpy(),
        lock[
            "scenario_fingerprint"
        ].to_numpy(),
    )

    # Difference convention:
    # Lock-Aware - Event-Global
    iter_diff = (
        lock["iterations"].to_numpy()
        - event["iterations"].to_numpy()
    )

    dist_diff = (
        lock["total_distance"].to_numpy()
        - event["total_distance"].to_numpy()
    )

    astar_diff = (
        lock[
            "path_cost_evaluations"
        ].to_numpy()
        - event[
            "path_cost_evaluations"
        ].to_numpy()
    )

    runtime_diff = (
        lock[
            "runtime_seconds"
        ].to_numpy()
        - event[
            "runtime_seconds"
        ].to_numpy()
    )

    global_diff = (
        lock[
            "global_assignment_calls"
        ].to_numpy()
        - event[
            "global_assignment_calls"
        ].to_numpy()
    )

    astar_reduction_percent = (
        (
            event[
                "path_cost_evaluations"
            ].to_numpy()
            - lock[
                "path_cost_evaluations"
            ].to_numpy()
        )
        /
        event[
            "path_cost_evaluations"
        ].to_numpy()
        * 100
    )

    runtime_reduction_percent = (
        (
            event[
                "runtime_seconds"
            ].to_numpy()
            - lock[
                "runtime_seconds"
            ].to_numpy()
        )
        /
        event[
            "runtime_seconds"
        ].to_numpy()
        * 100
    )

    iter_ci = bootstrap_mean_ci(
        iter_diff,
        seed=1000 + int(rate * 100),
    )

    dist_ci = bootstrap_mean_ci(
        dist_diff,
        seed=2000 + int(rate * 100),
    )

    astar_ci = bootstrap_mean_ci(
        astar_diff,
        seed=3000 + int(rate * 100),
    )

    astar_reduction_ci = (
        bootstrap_mean_ci(
            astar_reduction_percent,
            seed=4000 + int(rate * 100),
        )
    )

    iter_wtl = counts_lower_better(
        lock["iterations"],
        event["iterations"],
    )

    dist_wtl = counts_lower_better(
        lock["total_distance"],
        event["total_distance"],
    )

    astar_wtl = counts_lower_better(
        lock["path_cost_evaluations"],
        event["path_cost_evaluations"],
    )

    runtime_wtl = counts_lower_better(
        lock["runtime_seconds"],
        event["runtime_seconds"],
    )

    condition = (
        "static"
        if rate == 0
        else f"{int(rate * 100)}pct_failure"
    )

    row = {
        "condition":
            condition,

        "pairs":
            len(lock),

        "event_success":
            int(event["success"].sum()),

        "lock_success":
            int(lock["success"].sum()),

        "event_mean_iterations":
            event["iterations"].mean(),

        "lock_mean_iterations":
            lock["iterations"].mean(),

        "mean_iteration_difference_lock_minus_event":
            iter_diff.mean(),

        "iteration_diff_ci95_low":
            iter_ci[0],

        "iteration_diff_ci95_high":
            iter_ci[1],

        "iteration_wilcoxon_p":
            safe_wilcoxon(iter_diff),

        "iteration_rank_biserial":
            rank_biserial_paired(
                iter_diff
            ),

        "iteration_lock_better":
            iter_wtl[0],

        "iteration_ties":
            iter_wtl[1],

        "iteration_event_better":
            iter_wtl[2],

        "event_mean_distance":
            event[
                "total_distance"
            ].mean(),

        "lock_mean_distance":
            lock[
                "total_distance"
            ].mean(),

        "mean_distance_difference_lock_minus_event":
            dist_diff.mean(),

        "distance_diff_ci95_low":
            dist_ci[0],

        "distance_diff_ci95_high":
            dist_ci[1],

        "distance_wilcoxon_p":
            safe_wilcoxon(dist_diff),

        "distance_rank_biserial":
            rank_biserial_paired(
                dist_diff
            ),

        "distance_lock_better":
            dist_wtl[0],

        "distance_ties":
            dist_wtl[1],

        "distance_event_better":
            dist_wtl[2],

        "event_mean_astar":
            event[
                "path_cost_evaluations"
            ].mean(),

        "lock_mean_astar":
            lock[
                "path_cost_evaluations"
            ].mean(),

        "mean_astar_difference_lock_minus_event":
            astar_diff.mean(),

        "astar_diff_ci95_low":
            astar_ci[0],

        "astar_diff_ci95_high":
            astar_ci[1],

        "mean_paired_astar_reduction_percent":
            astar_reduction_percent.mean(),

        "median_paired_astar_reduction_percent":
            np.median(
                astar_reduction_percent
            ),

        "astar_reduction_ci95_low":
            astar_reduction_ci[0],

        "astar_reduction_ci95_high":
            astar_reduction_ci[1],

        "astar_wilcoxon_p":
            safe_wilcoxon(astar_diff),

        "astar_rank_biserial":
            rank_biserial_paired(
                astar_diff
            ),

        "astar_lock_better":
            astar_wtl[0],

        "astar_ties":
            astar_wtl[1],

        "astar_event_better":
            astar_wtl[2],

        "event_mean_global_calls":
            event[
                "global_assignment_calls"
            ].mean(),

        "lock_mean_global_calls":
            lock[
                "global_assignment_calls"
            ].mean(),

        "mean_global_call_difference_lock_minus_event":
            global_diff.mean(),

        "lock_mean_local_repairs":
            lock[
                "local_repair_calls"
            ].mean(),

        "lock_mean_escalations":
            lock[
                "lock_escalations"
            ].mean(),

        "event_mean_runtime":
            event[
                "runtime_seconds"
            ].mean(),

        "lock_mean_runtime":
            lock[
                "runtime_seconds"
            ].mean(),

        "mean_paired_runtime_reduction_percent":
            runtime_reduction_percent.mean(),

        "median_paired_runtime_reduction_percent":
            np.median(
                runtime_reduction_percent
            ),

        "runtime_wilcoxon_p":
            safe_wilcoxon(runtime_diff),

        "runtime_lock_faster":
            runtime_wtl[0],

        "runtime_ties":
            runtime_wtl[1],

        "runtime_event_faster":
            runtime_wtl[2],
    }

    paired_rows.append(row)

    key_rows.append({
        "condition":
            condition,

        "lock_success_rate_percent":
            100 * lock["success"].mean(),

        "event_success_rate_percent":
            100 * event["success"].mean(),

        "lock_mean_iterations":
            lock["iterations"].mean(),

        "event_mean_iterations":
            event["iterations"].mean(),

        "lock_mean_distance":
            lock["total_distance"].mean(),

        "event_mean_distance":
            event["total_distance"].mean(),

        "lock_mean_astar":
            lock[
                "path_cost_evaluations"
            ].mean(),

        "event_mean_astar":
            event[
                "path_cost_evaluations"
            ].mean(),

        "paired_astar_reduction_percent":
            astar_reduction_percent.mean(),

        "astar_wins_out_of_30":
            astar_wtl[0],

        "lock_mean_global_calls":
            lock[
                "global_assignment_calls"
            ].mean(),

        "event_mean_global_calls":
            event[
                "global_assignment_calls"
            ].mean(),

        "lock_mean_escalations":
            lock[
                "lock_escalations"
            ].mean(),
    })


paired = pd.DataFrame(
    paired_rows
)

key = pd.DataFrame(
    key_rows
)

PAIRED_OUT.parent.mkdir(
    parents=True,
    exist_ok=True,
)

KEY_OUT.parent.mkdir(
    parents=True,
    exist_ok=True,
)

paired.to_csv(
    PAIRED_OUT,
    index=False,
)

key.to_csv(
    KEY_OUT,
    index=False,
)


# --------------------------------------------------
# Console report
# --------------------------------------------------

print()
print("=" * 100)
print("E5C ANALYSIS")
print("=" * 100)

for _, r in paired.iterrows():

    print()
    print("#" * 100)
    print(r["condition"].upper())
    print("#" * 100)

    print(
        f"Success: Event={int(r['event_success'])}/30"
        f" | Lock={int(r['lock_success'])}/30"
    )

    print()
    print("SOLUTION QUALITY")

    print(
        "Iterations:",
        f"{r['event_mean_iterations']:.2f}",
        "->",
        f"{r['lock_mean_iterations']:.2f}",
        "| mean Δ=",
        f"{r['mean_iteration_difference_lock_minus_event']:+.2f}",
        "| 95% CI",
        f"[{r['iteration_diff_ci95_low']:.2f},"
        f" {r['iteration_diff_ci95_high']:.2f}]",
        "| W/T/L",
        f"{int(r['iteration_lock_better'])}/"
        f"{int(r['iteration_ties'])}/"
        f"{int(r['iteration_event_better'])}",
        "| p=",
        f"{r['iteration_wilcoxon_p']:.6g}",
    )

    print(
        "Distance:",
        f"{r['event_mean_distance']:.2f}",
        "->",
        f"{r['lock_mean_distance']:.2f}",
        "| mean Δ=",
        f"{r['mean_distance_difference_lock_minus_event']:+.2f}",
        "| 95% CI",
        f"[{r['distance_diff_ci95_low']:.2f},"
        f" {r['distance_diff_ci95_high']:.2f}]",
        "| W/T/L",
        f"{int(r['distance_lock_better'])}/"
        f"{int(r['distance_ties'])}/"
        f"{int(r['distance_event_better'])}",
        "| p=",
        f"{r['distance_wilcoxon_p']:.6g}",
    )

    print()
    print("COMPUTATIONAL WORK")

    print(
        "Allocation A*:",
        f"{r['event_mean_astar']:.2f}",
        "->",
        f"{r['lock_mean_astar']:.2f}",
    )

    print(
        "Mean paired A* reduction:",
        f"{r['mean_paired_astar_reduction_percent']:.2f}%",
        "| median:",
        f"{r['median_paired_astar_reduction_percent']:.2f}%",
        "| 95% CI:",
        f"[{r['astar_reduction_ci95_low']:.2f}%,"
        f" {r['astar_reduction_ci95_high']:.2f}%]",
    )

    print(
        "A* W/T/L:",
        f"{int(r['astar_lock_better'])}/"
        f"{int(r['astar_ties'])}/"
        f"{int(r['astar_event_better'])}",
        "| p=",
        f"{r['astar_wilcoxon_p']:.6g}",
        "| rank-biserial=",
        f"{r['astar_rank_biserial']:.3f}",
    )

    print()
    print("TRIGGER BEHAVIOUR")

    print(
        "Global calls:",
        f"{r['event_mean_global_calls']:.2f}",
        "Event ->",
        f"{r['lock_mean_global_calls']:.2f}",
        "Lock",
    )

    print(
        "Lock local repairs:",
        f"{r['lock_mean_local_repairs']:.2f}",
    )

    print(
        "Lock escalations:",
        f"{r['lock_mean_escalations']:.2f}",
    )

    print()
    print("RUNTIME (SECONDARY / MACHINE-DEPENDENT)")

    print(
        "Mean runtime:",
        f"{r['event_mean_runtime']:.3f}",
        "->",
        f"{r['lock_mean_runtime']:.3f}",
    )

    print(
        "Runtime W/T/L:",
        f"{int(r['runtime_lock_faster'])}/"
        f"{int(r['runtime_ties'])}/"
        f"{int(r['runtime_event_faster'])}",
        "| p=",
        f"{r['runtime_wilcoxon_p']:.6g}",
    )


print()
print("=" * 100)
print("FILES")
print("=" * 100)
print("Summary:", PROCESSED)
print("Paired:", PAIRED_OUT)
print("Key:", KEY_OUT)
print("=" * 100)
