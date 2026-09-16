from pathlib import Path
import sys
import csv
import time


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


from mrta.config import SimulationConfig
from mrta.scenario import Scenario

from mrta.obstacle_allocators import (
    ManhattanGreedyAllocator,
    PathCostGreedyAllocator,
)

from mrta.coordinated_allocators import (
    HungarianPathAllocator,
)

from mrta.obstacle_simulator import (
    ObstacleSimulator,
)


NUM_SEEDS = 30
NUM_AGENTS = 15
NUM_TASKS = 20
MAX_ITERATIONS = 500

DENSITIES = [
    0.00,
    0.05,
    0.10,
    0.15,
    0.20,
]

METHODS = [
    (
        "manhattan_greedy",
        ManhattanGreedyAllocator,
    ),
    (
        "path_cost_greedy",
        PathCostGreedyAllocator,
    ),
    (
        "hungarian_astar",
        HungarianPathAllocator,
    ),
]


def main():

    output_dir = ROOT / "data" / "raw"

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file = (
        output_dir
        / "coordinated_obstacle_density_30_seeds.csv"
    )

    rows = []

    experiment_start = time.perf_counter()

    print()
    print("=" * 76)
    print("COORDINATED OBSTACLE-DENSITY EXPERIMENT")
    print("=" * 76)

    print(
        f"Agents: {NUM_AGENTS} | "
        f"Tasks: {NUM_TASKS} | "
        f"Seeds: 0-{NUM_SEEDS - 1}"
    )

    print(
        "Densities:",
        ", ".join(
            f"{d:.0%}"
            for d in DENSITIES
        ),
    )

    print(
        "Total planned runs:",
        len(DENSITIES)
        * NUM_SEEDS
        * len(METHODS),
    )

    print("=" * 76)


    for density in DENSITIES:

        print()
        print("#" * 76)
        print(
            f"OBSTACLE DENSITY: {density:.0%}"
        )
        print("#" * 76)


        for seed in range(NUM_SEEDS):

            config = SimulationConfig(
                num_agents=NUM_AGENTS,
                num_tasks=NUM_TASKS,
                seed=seed,
                max_iterations=MAX_ITERATIONS,
            )


            # -------------------------------------------------
            # Generate ONCE.
            # Every algorithm receives clones of this scenario.
            # -------------------------------------------------

            scenario = Scenario.generate(
                config=config,
                obstacle_density=density,
            )

            fingerprint = (
                scenario.fingerprint()
            )


            print(
                f"\nDensity {density:.0%} "
                f"| Seed {seed:02d}"
            )


            for (
                method_name,
                allocator_class,
            ) in METHODS:

                simulator = ObstacleSimulator(
                    config=config,
                    allocator=allocator_class(),
                    obstacle_density=density,
                    scenario=scenario,
                )


                start = time.perf_counter()

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
                        method_name,

                    "num_agents":
                        NUM_AGENTS,

                    "num_tasks":
                        NUM_TASKS,

                    "obstacle_density":
                        density,

                    "success":
                        int(
                            bool(
                                result[
                                    "success"
                                ]
                            )
                        ),

                    "iterations":
                        result[
                            "iterations"
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

                    "redundant_target_ratio":
                        result[
                            "redundant_target_ratio"
                        ],

                    "runtime_seconds":
                        runtime,
                }


                rows.append(
                    row
                )


                print(
                    f"  {method_name:22s}"
                    f" success={row['success']}"
                    f" iter={row['iterations']:3d}"
                    f" dist={row['total_distance']:7.1f}"
                    f" blocked={row['blocked_moves']:3d}"
                    f" redund={row['redundant_target_ratio']:.4f}"
                    f" time={runtime:.3f}s"
                )


    fieldnames = [
        "seed",
        "scenario_fingerprint",
        "algorithm",
        "num_agents",
        "num_tasks",
        "obstacle_density",
        "success",
        "iterations",
        "total_distance",
        "productive_agents",
        "blocked_moves",
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
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(
            rows
        )


    elapsed = (
        time.perf_counter()
        - experiment_start
    )


    print()
    print("=" * 76)
    print("EXPERIMENT COMPLETE")
    print("=" * 76)

    print(
        "Runs:",
        len(rows),
    )

    print(
        "Expected:",
        len(DENSITIES)
        * NUM_SEEDS
        * len(METHODS),
    )

    print(
        "Total runtime:",
        round(
            elapsed,
            2,
        ),
        "seconds",
    )

    print()
    print(
        "Saved:"
    )

    print(
        output_file
    )

    print("=" * 76)


if __name__ == "__main__":
    main()
