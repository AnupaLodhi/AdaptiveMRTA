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
    StatefulHungarianAllocator,
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


def choose_failed_agents(
    seed,
    failure_rate,
):
    count = int(
        NUM_AGENTS * failure_rate
    )

    if count == 0:
        return []

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


def create_allocator(name):

    if name == "repeated":
        return HungarianPathAllocator()

    if name == "event_global":
        # Large interval means that in these short
        # experiments replanning is driven primarily
        # by invalid-assignment events.
        return PeriodicHungarianAllocator(
            replan_interval=1000
        )

    if name == "local_repair":
        return StatefulHungarianAllocator()

    raise ValueError(name)


METHODS = [
    "repeated",
    "event_global",
    "local_repair",
]


def main():

    output = (
        ROOT
        / "data/raw"
        / "e5_replanning_tradeoff_30_seeds.csv"
    )

    rows = []

    expected = (
        len(FAILURE_RATES)
        * NUM_SEEDS
        * len(METHODS)
    )

    total_start = time.perf_counter()

    print()
    print("=" * 88)
    print("E5: GLOBAL REPLANNING TRADE-OFF")
    print("=" * 88)
    print("Agents / Tasks: 15 / 20")
    print("Obstacle density: 20%")
    print("Seeds: 30")
    print("Conditions: static + 20% failure")
    print("Expected runs:", expected)
    print("=" * 88)


    for failure_rate in FAILURE_RATES:

        print()
        print("#" * 88)
        print(
            "FAILURE RATE:",
            f"{failure_rate:.0%}",
        )
        print("#" * 88)


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

                # Repeated Hungarian performs one
                # global assignment per active step.
                if method == "repeated":
                    global_calls = (
                        result["iterations"]
                    )

                    assignment_requests = (
                        result["iterations"]
                    )

                    event_replans = np.nan
                    periodic_replans = np.nan

                else:
                    global_calls = (
                        allocator
                        .global_assignment_calls
                    )

                    assignment_requests = (
                        allocator
                        .assignment_requests
                    )

                    event_replans = getattr(
                        allocator,
                        "event_replans",
                        np.nan,
                    )

                    periodic_replans = getattr(
                        allocator,
                        "periodic_replans",
                        np.nan,
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
                            assignment_requests,

                        "global_assignment_calls":
                            global_calls,

                        "event_replans":
                            event_replans,

                        "periodic_replans":
                            periodic_replans,
                    }
                )

                print(
                    f"  {method:14s}"
                    f" success={int(result['success'])}"
                    f" iter={result['iterations']:3d}"
                    f" dist={result['total_distance']:7.1f}"
                    f" calls={global_calls:3d}"
                    f" time={runtime:.3f}s"
                )


    df_rows = rows

    fields = list(
        df_rows[0].keys()
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
        writer.writerows(df_rows)

    elapsed = (
        time.perf_counter()
        - total_start
    )

    print()
    print("=" * 88)
    print("E5 DIAGNOSTIC COMPLETE")
    print("=" * 88)
    print("Runs:", len(rows))
    print("Expected:", expected)
    print(
        "Runtime:",
        round(elapsed, 2),
        "seconds",
    )
    print("Saved:")
    print(output)
    print("=" * 88)


if __name__ == "__main__":
    main()
