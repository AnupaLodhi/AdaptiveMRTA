from pathlib import Path
import sys
import time

import pandas as pd


PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

SRC_PATH = (
    PROJECT_ROOT
    / "src"
)

sys.path.insert(
    0,
    str(SRC_PATH),
)


from mrta.config import SimulationConfig
from mrta.obstacle_allocators import (
    ManhattanGreedyAllocator,
    PathCostGreedyAllocator,
)
from mrta.obstacle_simulator import (
    ObstacleSimulator,
)


NUM_TRIALS = 30

NUM_AGENTS = 15
NUM_TASKS = 20

OBSTACLE_DENSITIES = [
    0.00,
    0.05,
    0.10,
    0.15,
    0.20,
]


def run_trial(
    algorithm_name,
    density,
    seed,
):

    config = SimulationConfig(
        num_agents=NUM_AGENTS,
        num_tasks=NUM_TASKS,
        seed=seed,
        max_iterations=500,
    )

    if algorithm_name == "manhattan_greedy":
        allocator = (
            ManhattanGreedyAllocator()
        )

    elif algorithm_name == "path_cost_greedy":
        allocator = (
            PathCostGreedyAllocator()
        )

    else:
        raise ValueError(
            algorithm_name
        )

    simulator = ObstacleSimulator(
        config=config,
        allocator=allocator,
        obstacle_density=density,
    )

    result = simulator.run()

    return {
        "algorithm": algorithm_name,
        "seed": seed,
        "obstacle_density": density,
        "success": result["success"],
        "iterations": result["iterations"],
        "completed_tasks":
            result["completed_tasks"],
        "total_distance":
            result["total_distance"],
        "productive_agents":
            result["productive_agents"],
        "blocked_moves":
            result["blocked_moves"],
        "unreachable_plans":
            result["unreachable_plans"],
        "redundant_target_events":
            result[
                "redundant_target_events"
            ],
        "redundant_target_ratio":
            result[
                "redundant_target_ratio"
            ],
    }


def main():

    algorithms = [
        "manhattan_greedy",
        "path_cost_greedy",
    ]

    rows = []

    print(
        "\n=============================================="
    )

    print(
        "OBSTACLE DENSITY EXPERIMENT"
    )

    print(
        "=============================================="
    )

    print(
        f"\nTrials per setting: {NUM_TRIALS}"
    )

    print(
        f"Agents: {NUM_AGENTS}"
    )

    print(
        f"Tasks: {NUM_TASKS}"
    )

    start_time = time.time()

    for density in OBSTACLE_DENSITIES:

        print(
            f"\nObstacle density: "
            f"{density * 100:.0f}%"
        )

        print(
            "-" * 90
        )

        for algorithm_name in algorithms:

            temp_rows = []

            for seed in range(
                NUM_TRIALS
            ):

                result = run_trial(
                    algorithm_name,
                    density,
                    seed,
                )

                rows.append(result)
                temp_rows.append(result)

            temp = pd.DataFrame(
                temp_rows
            )

            success_rate = (
                temp["success"].mean()
                * 100
            )

            successful = temp[
                temp["success"]
            ]

            if len(successful) > 0:

                iterations = (
                    successful[
                        "iterations"
                    ].mean()
                )

                distance = (
                    successful[
                        "total_distance"
                    ].mean()
                )

                redundancy = (
                    successful[
                        "redundant_target_ratio"
                    ].mean()
                )

                blocked = (
                    successful[
                        "blocked_moves"
                    ].mean()
                )

            else:

                iterations = float("nan")
                distance = float("nan")
                redundancy = float("nan")
                blocked = float("nan")

            print(
                f"{algorithm_name:20s} | "
                f"Success={success_rate:6.1f}% | "
                f"Iter={iterations:7.2f} | "
                f"Dist={distance:8.2f} | "
                f"Redundancy={redundancy:6.3f} | "
                f"Blocked={blocked:7.2f}"
            )

    df = pd.DataFrame(
        rows
    )

    raw_dir = (
        PROJECT_ROOT
        / "data"
        / "raw"
    )

    processed_dir = (
        PROJECT_ROOT
        / "data"
        / "processed"
    )

    table_dir = (
        PROJECT_ROOT
        / "results"
        / "tables"
    )

    raw_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    processed_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    table_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    raw_path = (
        raw_dir
        / "obstacle_density_trials.csv"
    )

    df.to_csv(
        raw_path,
        index=False,
    )

    summary = (
        df.groupby(
            [
                "obstacle_density",
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
            std_iterations=(
                "iterations",
                "std",
            ),
            mean_distance=(
                "total_distance",
                "mean",
            ),
            std_distance=(
                "total_distance",
                "std",
            ),
            mean_redundancy=(
                "redundant_target_ratio",
                "mean",
            ),
            mean_blocked_moves=(
                "blocked_moves",
                "mean",
            ),
            mean_productive_agents=(
                "productive_agents",
                "mean",
            ),
        )
        .reset_index()
    )

    summary[
        "success_rate"
    ] *= 100

    summary_path = (
        table_dir
        / "obstacle_density_summary.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    # ---------------------------------
    # PAIRED COMPARISON
    # ---------------------------------

    manhattan = df[
        df["algorithm"]
        == "manhattan_greedy"
    ].copy()

    pathcost = df[
        df["algorithm"]
        == "path_cost_greedy"
    ].copy()

    paired = manhattan.merge(
        pathcost,
        on=[
            "seed",
            "obstacle_density",
        ],
        suffixes=(
            "_manhattan",
            "_path",
        ),
    )

    paired[
        "iteration_difference"
    ] = (
        paired[
            "iterations_path"
        ]
        -
        paired[
            "iterations_manhattan"
        ]
    )

    paired[
        "distance_difference"
    ] = (
        paired[
            "total_distance_path"
        ]
        -
        paired[
            "total_distance_manhattan"
        ]
    )

    paired[
        "redundancy_difference"
    ] = (
        paired[
            "redundant_target_ratio_path"
        ]
        -
        paired[
            "redundant_target_ratio_manhattan"
        ]
    )

    paired_path = (
        processed_dir
        / "obstacle_density_paired.csv"
    )

    paired.to_csv(
        paired_path,
        index=False,
    )

    print(
        "\n=============================================="
    )

    print(
        "PAIRED DIFFERENCES"
    )

    print(
        "Path-Cost minus Manhattan"
    )

    print(
        "Negative = Path-Cost is better"
    )

    print(
        "==============================================\n"
    )

    for density in OBSTACLE_DENSITIES:

        subset = paired[
            paired[
                "obstacle_density"
            ]
            == density
        ]

        print(
            f"{density*100:4.0f}% obstacles | "
            f"ΔIter="
            f"{subset['iteration_difference'].mean():7.2f} | "
            f"ΔDist="
            f"{subset['distance_difference'].mean():8.2f} | "
            f"ΔRedundancy="
            f"{subset['redundancy_difference'].mean():7.4f}"
        )

    elapsed = (
        time.time()
        - start_time
    )

    print(
        "\n=============================================="
    )

    print(
        "FILES SAVED"
    )

    print(
        "=============================================="
    )

    print(raw_path)
    print(summary_path)
    print(paired_path)

    print(
        f"\nRuntime: "
        f"{elapsed:.1f} seconds"
    )


if __name__ == "__main__":
    main()
