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
from mrta.coordinated_allocators import (
    HungarianPathAllocator,
    PeriodicHungarianAllocator,
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
]

METHODS = [
    "repeated",
    "event_global",
]


def choose_failed_agents(
    seed,
    failure_rate,
):
    count = int(
        NUM_AGENTS * failure_rate
    )

    if count == 0:
        return []

    # Keep exactly the same deterministic
    # failure-generation rule used in E5A.
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


def create_allocator(method):

    if method == "repeated":
        return HungarianPathAllocator()

    if method == "event_global":
        return PeriodicHungarianAllocator(
            replan_interval=1000
        )

    raise ValueError(method)


def main():

    output = (
        ROOT
        / "data/raw"
        / "e5b_computational_work_30_seeds.csv"
    )

    rows = []

    expected = (
        NUM_SEEDS
        * len(FAILURE_RATES)
        * len(METHODS)
    )

    total_start = time.perf_counter()

    print()
    print("=" * 94)
    print(
        "E5B: COMPUTATIONAL-WORK EXPERIMENT"
    )
    print("=" * 94)
    print(
        "Agents / Tasks:",
        NUM_AGENTS,
        "/",
        NUM_TASKS,
    )
    print(
        "Obstacle density:",
        OBSTACLE_DENSITY,
    )
    print(
        "Seeds:",
        NUM_SEEDS,
    )
    print(
        "Failure rates:",
        FAILURE_RATES,
    )
    print(
        "Expected runs:",
        expected,
    )
    print("=" * 94)


    for failure_rate in FAILURE_RATES:

        print()
        print("#" * 94)
        print(
            "FAILURE RATE:",
            f"{failure_rate:.0%}",
        )
        print("#" * 94)


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

            failure_schedule = (
                {
                    FAILURE_ITERATION:
                    failed_ids
                }
                if failed_ids
                else {}
            )

            print(
                f"\nrate={failure_rate:.0%}"
                f" seed={seed:02d}"
                f" failed={failed_ids}"
            )


            for method in METHODS:

                allocator = (
                    create_allocator(
                        method
                    )
                )

                simulator = (
                    ObstacleSimulator(
                        config=config,
                        allocator=allocator,
                        obstacle_density=OBSTACLE_DENSITY,
                        scenario=scenario,
                        failure_schedule=failure_schedule,
                    )
                )

                start = (
                    time.perf_counter()
                )

                result = simulator.run()

                runtime = (
                    time.perf_counter()
                    - start
                )

                total_astar = (
                    allocator
                    .path_cost_evaluations
                )

                successful_astar = (
                    allocator
                    .successful_path_cost_evaluations
                )

                unreachable_astar = (
                    allocator
                    .unreachable_path_cost_evaluations
                )

                # Counter integrity.
                assert (
                    total_astar
                    ==
                    successful_astar
                    + unreachable_astar
                )

                rows.append(
                    {
                        "seed":
                            seed,

                        "scenario_fingerprint":
                            fingerprint,

                        "failure_rate":
                            failure_rate,

                        "failed_agent_ids":
                            ";".join(
                                map(
                                    str,
                                    failed_ids,
                                )
                            ),

                        "method":
                            method,

                        "success":
                            int(
                                result["success"]
                            ),

                        "iterations":
                            result["iterations"],

                        "total_distance":
                            result[
                                "total_distance"
                            ],

                        "blocked_moves":
                            result[
                                "blocked_moves"
                            ],

                        "redundancy":
                            result[
                                "redundant_target_ratio"
                            ],

                        "runtime_seconds":
                            runtime,

                        "assignment_requests":
                            allocator
                            .assignment_requests,

                        "global_assignment_calls":
                            allocator
                            .global_assignment_calls,

                        "path_cost_evaluations":
                            total_astar,

                        "successful_astar_queries":
                            successful_astar,

                        "unreachable_astar_queries":
                            unreachable_astar,
                    }
                )

                print(
                    f"  {method:12s}"
                    f" success={int(result['success'])}"
                    f" iter={result['iterations']:3d}"
                    f" dist={result['total_distance']:7.1f}"
                    f" global={allocator.global_assignment_calls:3d}"
                    f" astar={total_astar:5d}"
                    f" time={runtime:.3f}s"
                )

        # Save after each complete failure-rate
        # block as an extra safety measure.
        fields = list(
            rows[0].keys()
        )

        with output.open(
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

        print()
        print(
            "Checkpoint saved:",
            len(rows),
            "runs"
        )


    elapsed = (
        time.perf_counter()
        - total_start
    )

    print()
    print("=" * 94)
    print(
        "E5B COMPUTATIONAL-WORK "
        "EXPERIMENT COMPLETE"
    )
    print("=" * 94)
    print(
        "Runs:",
        len(rows),
    )
    print(
        "Expected:",
        expected,
    )
    print(
        "Runtime:",
        round(elapsed, 2),
        "seconds",
    )
    print(
        "Saved:",
        output,
    )
    print("=" * 94)


if __name__ == "__main__":
    main()
