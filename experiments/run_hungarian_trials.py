from pathlib import Path
import sys
import csv
import time


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"

sys.path.insert(0, str(SRC_PATH))


from mrta.config import SimulationConfig

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
OBSTACLE_DENSITY = 0.20
MAX_ITERATIONS = 500


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


def run_one(seed, allocator_class):

    config = SimulationConfig(
        num_agents=NUM_AGENTS,
        num_tasks=NUM_TASKS,
        seed=seed,
        max_iterations=MAX_ITERATIONS,
    )

    simulator = ObstacleSimulator(
        config=config,
        allocator=allocator_class(),
        obstacle_density=OBSTACLE_DENSITY,
    )

    start = time.perf_counter()

    result = simulator.run()

    runtime = time.perf_counter() - start

    return result, runtime


def main():

    output_dir = (
        PROJECT_ROOT
        / "data"
        / "raw"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file = (
        output_dir
        / "hungarian_comparison_30_seeds.csv"
    )

    rows = []

    experiment_start = time.perf_counter()

    print()
    print("=" * 70)
    print("PAIRED COORDINATED ASSIGNMENT EXPERIMENT")
    print("=" * 70)
    print(f"Seeds:             0-{NUM_SEEDS - 1}")
    print(f"Agents:            {NUM_AGENTS}")
    print(f"Tasks:             {NUM_TASKS}")
    print(f"Obstacle density:  {OBSTACLE_DENSITY:.0%}")
    print(f"Maximum iterations:{MAX_ITERATIONS}")
    print("=" * 70)
    print()

    for seed in range(NUM_SEEDS):

        print(
            f"\nSEED {seed:02d}/{NUM_SEEDS - 1:02d}"
        )

        print("-" * 70)

        for method_name, allocator_class in METHODS:

            result, runtime = run_one(
                seed,
                allocator_class,
            )

            row = {
                "seed": seed,
                "algorithm": method_name,
                "num_agents": NUM_AGENTS,
                "num_tasks": NUM_TASKS,
                "obstacle_density": OBSTACLE_DENSITY,
                "success": int(
                    bool(
                        result["success"]
                    )
                ),
                "iterations": result[
                    "iterations"
                ],
                "total_distance": result[
                    "total_distance"
                ],
                "productive_agents": result[
                    "productive_agents"
                ],
                "blocked_moves": result[
                    "blocked_moves"
                ],
                "redundant_target_ratio": result[
                    "redundant_target_ratio"
                ],
                "runtime_seconds": runtime,
            }

            rows.append(row)

            print(
                f"{method_name:22s} "
                f"| success={row['success']} "
                f"| iter={row['iterations']:3d} "
                f"| dist={row['total_distance']:7.1f} "
                f"| productive={row['productive_agents']:2d} "
                f"| blocked={row['blocked_moves']:3d} "
                f"| redundancy={row['redundant_target_ratio']:.4f} "
                f"| runtime={runtime:.3f}s"
            )

    fieldnames = [
        "seed",
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

    total_runtime = (
        time.perf_counter()
        - experiment_start
    )

    print()
    print("=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)
    print("Runs:", len(rows))
    print(
        "Total runtime:",
        round(total_runtime, 2),
        "seconds",
    )
    print()
    print("Raw results saved to:")
    print(output_file)
    print("=" * 70)


if __name__ == "__main__":
    main()
