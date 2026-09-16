from pathlib import Path
import sys


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
    EuclideanGreedyAllocator,
    PathCostGreedyAllocator,
)
from mrta.obstacle_simulator import (
    ObstacleSimulator,
)


def run_algorithm(
    allocator,
):
    config = SimulationConfig(
        num_agents=15,
        num_tasks=20,
        seed=42,
        max_iterations=1000,
    )

    simulator = ObstacleSimulator(
        config=config,
        allocator=allocator,
        obstacle_density=0.12,
    )

    return simulator.run()


def print_result(
    name,
    result,
):
    print(
        f"\n{name}"
    )

    print("-" * 45)

    print(
        "Success:",
        result["success"]
    )

    print(
        "Tasks:",
        f"{result['completed_tasks']}/"
        f"{result['total_tasks']}"
    )

    print(
        "Iterations:",
        result["iterations"]
    )

    print(
        "Total distance:",
        round(
            result["total_distance"],
            2,
        )
    )

    print(
        "Productive agents:",
        result[
            "productive_agents"
        ]
    )

    print(
        "Blocked moves:",
        result[
            "blocked_moves"
        ]
    )

    print(
        "Unreachable plans:",
        result[
            "unreachable_plans"
        ]
    )

    print(
        "Redundant target events:",
        result[
            "redundant_target_events"
        ]
    )

    print(
        "Redundant target ratio:",
        round(
            result[
                "redundant_target_ratio"
            ],
            4,
        )
    )


def main():

    print(
        "\n======================================"
    )

    print(
        "OBSTACLE-AWARE ALLOCATION TEST"
    )

    print(
        "======================================"
    )

    euclidean_result = (
        run_algorithm(
            EuclideanGreedyAllocator()
        )
    )

    path_result = (
        run_algorithm(
            PathCostGreedyAllocator()
        )
    )

    print_result(
        "EUCLIDEAN GREEDY + A*",
        euclidean_result,
    )

    print_result(
        "PATH-COST GREEDY + A*",
        path_result,
    )

    print(
        "\n======================================"
    )

    print(
        "PAIRWISE DIFFERENCE"
    )

    print(
        "======================================"
    )

    print(
        "Iteration difference "
        "(Path - Euclidean):",
        path_result["iterations"]
        -
        euclidean_result["iterations"],
    )

    print(
        "Distance difference "
        "(Path - Euclidean):",
        round(
            path_result[
                "total_distance"
            ]
            -
            euclidean_result[
                "total_distance"
            ],
            2,
        )
    )

    print(
        "Redundancy difference "
        "(Path - Euclidean):",
        round(
            path_result[
                "redundant_target_ratio"
            ]
            -
            euclidean_result[
                "redundant_target_ratio"
            ],
            4,
        )
    )


if __name__ == "__main__":
    main()
