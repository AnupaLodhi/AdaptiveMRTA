from pathlib import Path
import sys
import csv
import time


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


SCENARIOS = [
    (5, 10),
    (10, 10),
    (15, 20),
    (20, 20),
    (20, 30),
    (30, 30),
]

METHODS = [
    ("manhattan_greedy", ManhattanGreedyAllocator),
    ("path_cost_greedy", PathCostGreedyAllocator),
    ("hungarian_astar", HungarianPathAllocator),
]

NUM_SEEDS = 30
OBSTACLE_DENSITY = 0.20
MAX_ITERATIONS = 500


def main():

    output_dir = ROOT / "data" / "raw"
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file = (
        output_dir
        / "coordinated_scalability_30_seeds.csv"
    )

    rows = []

    experiment_start = time.perf_counter()

    total_expected = (
        len(SCENARIOS)
        * NUM_SEEDS
        * len(METHODS)
    )

    print()
    print("=" * 80)
    print("COORDINATED SCALABILITY EXPERIMENT")
    print("=" * 80)
    print("Obstacle density: 20%")
    print(f"Seeds: 0-{NUM_SEEDS - 1}")
    print("Problem sizes:", SCENARIOS)
    print("Total planned runs:", total_expected)
    print("=" * 80)


    for num_agents, num_tasks in SCENARIOS:

        print()
        print("#" * 80)
        print(
            f"{num_agents} AGENTS / "
            f"{num_tasks} TASKS"
        )
        print("#" * 80)


        for seed in range(NUM_SEEDS):

            config = SimulationConfig(
                num_agents=num_agents,
                num_tasks=num_tasks,
                seed=seed,
                max_iterations=MAX_ITERATIONS,
            )

            # Generate the scenario exactly once.
            scenario = Scenario.generate(
                config=config,
                obstacle_density=OBSTACLE_DENSITY,
            )

            fingerprint = scenario.fingerprint()

            print(
                f"\n"
                f"{num_agents:02d}/{num_tasks:02d} "
                f"| seed {seed:02d}"
            )


            for name, allocator_class in METHODS:

                simulator = ObstacleSimulator(
                    config=config,
                    allocator=allocator_class(),
                    obstacle_density=OBSTACLE_DENSITY,
                    scenario=scenario,
                )

                start = time.perf_counter()

                result = simulator.run()

                runtime = (
                    time.perf_counter()
                    - start
                )

                row = {
                    "seed": seed,
                    "scenario_fingerprint":
                        fingerprint,
                    "algorithm": name,
                    "num_agents":
                        num_agents,
                    "num_tasks":
                        num_tasks,
                    "obstacle_density":
                        OBSTACLE_DENSITY,
                    "success":
                        int(
                            bool(
                                result["success"]
                            )
                        ),
                    "iterations":
                        result["iterations"],
                    "total_distance":
                        result["total_distance"],
                    "productive_agents":
                        result["productive_agents"],
                    "blocked_moves":
                        result["blocked_moves"],
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
                    f" dist={row['total_distance']:8.1f}"
                    f" blocked={row['blocked_moves']:4d}"
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
        writer.writerows(rows)


    elapsed = (
        time.perf_counter()
        - experiment_start
    )


    print()
    print("=" * 80)
    print("SCALABILITY EXPERIMENT COMPLETE")
    print("=" * 80)
    print("Runs:", len(rows))
    print("Expected:", total_expected)

    print(
        "Total runtime:",
        round(elapsed, 2),
        "seconds",
    )

    print()
    print("Saved:")
    print(output_file)
    print("=" * 80)


if __name__ == "__main__":
    main()
