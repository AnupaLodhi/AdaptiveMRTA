from pathlib import Path
import sys
import csv
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mrta.config import SimulationConfig
from mrta.scenario import Scenario
from mrta.obstacle_simulator import ObstacleSimulator

from mrta.obstacle_allocators import (
    ManhattanGreedyAllocator,
    PathCostGreedyAllocator,
)

from mrta.coordinated_allocators import (
    HungarianPathAllocator,
)


NUM_AGENTS = 15
NUM_TASKS = 20
NUM_SEEDS = 30

OBSTACLE_DENSITY = 0.20
MAX_ITERATIONS = 500
FAILURE_ITERATION = 5

FAILURE_RATES = [
    0.0,
    0.2,
    0.4,
    0.6,
]

METHODS = [
    ("manhattan_greedy", ManhattanGreedyAllocator),
    ("path_cost_greedy", PathCostGreedyAllocator),
    ("hungarian_astar", HungarianPathAllocator),
]


def choose_failed_agents(
    seed,
    failure_rate,
):
    count = int(
        NUM_AGENTS
        * failure_rate
    )

    if count == 0:
        return []

    # Separate deterministic RNG stream used only
    # for failure selection.
    rng = np.random.default_rng(
        seed + 50000
    )

    selected = rng.choice(
        NUM_AGENTS,
        size=count,
        replace=False,
    )

    return sorted(
        int(x)
        for x in selected
    )


def main():

    output_dir = ROOT / "data/raw"
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file = (
        output_dir
        / "dynamic_failures_30_seeds.csv"
    )

    rows = []

    expected = (
        len(FAILURE_RATES)
        * NUM_SEEDS
        * len(METHODS)
    )

    experiment_start = (
        time.perf_counter()
    )

    print()
    print("=" * 84)
    print("E4: DYNAMIC ROBOT-FAILURE ROBUSTNESS")
    print("=" * 84)
    print(
        f"Agents / Tasks: "
        f"{NUM_AGENTS}/{NUM_TASKS}"
    )
    print(
        f"Obstacle density: "
        f"{OBSTACLE_DENSITY:.0%}"
    )
    print(
        "Failure iteration:",
        FAILURE_ITERATION,
    )
    print(
        "Failure rates:",
        FAILURE_RATES,
    )
    print(
        "Seeds:",
        f"0-{NUM_SEEDS - 1}",
    )
    print(
        "Expected runs:",
        expected,
    )
    print("=" * 84)


    for failure_rate in FAILURE_RATES:

        print()
        print("#" * 84)
        print(
            f"FAILURE RATE: "
            f"{failure_rate:.0%}"
        )
        print("#" * 84)


        for seed in range(NUM_SEEDS):

            config = SimulationConfig(
                num_agents=NUM_AGENTS,
                num_tasks=NUM_TASKS,
                seed=seed,
                max_iterations=MAX_ITERATIONS,
            )

            scenario = Scenario.generate(
                config=config,
                obstacle_density=OBSTACLE_DENSITY,
            )

            fingerprint = (
                scenario.fingerprint()
            )

            failed_ids = (
                choose_failed_agents(
                    seed,
                    failure_rate,
                )
            )

            if failed_ids:
                failure_schedule = {
                    FAILURE_ITERATION:
                        failed_ids
                }
            else:
                failure_schedule = {}


            print()
            print(
                f"rate={failure_rate:.0%}"
                f" seed={seed:02d}"
                f" failed={failed_ids}"
            )


            for name, allocator_class in METHODS:

                simulator = ObstacleSimulator(
                    config=config,
                    allocator=allocator_class(),
                    obstacle_density=OBSTACLE_DENSITY,
                    scenario=scenario,
                    failure_schedule=failure_schedule,
                )

                start = (
                    time.perf_counter()
                )

                result = simulator.run()

                runtime = (
                    time.perf_counter()
                    - start
                )

                row = {
                    "seed":
                        seed,

                    "scenario_fingerprint":
                        fingerprint,

                    "algorithm":
                        name,

                    "num_agents":
                        NUM_AGENTS,

                    "num_tasks":
                        NUM_TASKS,

                    "obstacle_density":
                        OBSTACLE_DENSITY,

                    "failure_rate":
                        failure_rate,

                    "failure_iteration":
                        FAILURE_ITERATION,

                    "failed_agent_count":
                        len(failed_ids),

                    "failed_agent_ids":
                        ";".join(
                            map(
                                str,
                                failed_ids,
                            )
                        ),

                    "success":
                        int(
                            bool(
                                result["success"]
                            )
                        ),

                    "iterations":
                        result["iterations"],

                    "completed_tasks":
                        result[
                            "completed_tasks"
                        ],

                    "total_distance":
                        result[
                            "total_distance"
                        ],

                    "productive_agents":
                        result[
                            "productive_agents"
                        ],

                    "blocked_moves":
                        result[
                            "blocked_moves"
                        ],

                    "unreachable_plans":
                        result[
                            "unreachable_plans"
                        ],

                    "redundant_target_ratio":
                        result[
                            "redundant_target_ratio"
                        ],

                    "runtime_seconds":
                        runtime,
                }

                rows.append(row)

                print(
                    f"  {name:22s}"
                    f" success={row['success']}"
                    f" iter={row['iterations']:3d}"
                    f" completed={row['completed_tasks']:2d}"
                    f" dist={row['total_distance']:7.1f}"
                    f" blocked={row['blocked_moves']:3d}"
                    f" redund={row['redundant_target_ratio']:.4f}"
                    f" time={runtime:.3f}s"
                )


    fields = [
        "seed",
        "scenario_fingerprint",
        "algorithm",
        "num_agents",
        "num_tasks",
        "obstacle_density",
        "failure_rate",
        "failure_iteration",
        "failed_agent_count",
        "failed_agent_ids",
        "success",
        "iterations",
        "completed_tasks",
        "total_distance",
        "productive_agents",
        "blocked_moves",
        "unreachable_plans",
        "redundant_target_ratio",
        "runtime_seconds",
    ]


    with output_file.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )

        writer.writeheader()
        writer.writerows(rows)


    elapsed = (
        time.perf_counter()
        - experiment_start
    )

    print()
    print("=" * 84)
    print("E4 EXPERIMENT COMPLETE")
    print("=" * 84)
    print(
        "Runs:",
        len(rows),
    )
    print(
        "Expected:",
        expected,
    )
    print(
        "Total runtime:",
        round(elapsed, 2),
        "seconds",
    )
    print()
    print("Saved:")
    print(output_file)
    print("=" * 84)


if __name__ == "__main__":
    main()
