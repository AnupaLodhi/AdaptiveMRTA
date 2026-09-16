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
    ManhattanGreedyAllocator,
    PathCostGreedyAllocator,
)

from mrta.coordinated_allocators import (
    HungarianPathAllocator,
)

from mrta.obstacle_simulator import (
    ObstacleSimulator,
)


def run(
    allocator,
):
    config = SimulationConfig(
        num_agents=15,
        num_tasks=20,
        seed=42,
        max_iterations=500,
    )

    simulator = (
        ObstacleSimulator(
            config=config,
            allocator=allocator,
            obstacle_density=0.20,
        )
    )

    return simulator.run()


def show(
    name,
    result,
):
    print(
        f"\n{name}"
    )

    print(
        "-" * 60
    )

    print(
        "Success:",
        result["success"]
    )

    print(
        "Iterations:",
        result["iterations"]
    )

    print(
        "Distance:",
        round(
            result[
                "total_distance"
            ],
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
        "Redundant targeting:",
        round(
            result[
                "redundant_target_ratio"
            ],
            4,
        )
    )


def main():

    print(
        "\n========================================"
    )

    print(
        "COORDINATED ASSIGNMENT TEST"
    )

    print(
        "15 agents / 20 tasks / 20% obstacles"
    )

    print(
        "Seed = 42"
    )

    print(
        "========================================"
    )

    methods = [
        (
            "MANHATTAN GREEDY",
            ManhattanGreedyAllocator(),
        ),
        (
            "PATH-COST GREEDY",
            PathCostGreedyAllocator(),
        ),
        (
            "HUNGARIAN + A*",
            HungarianPathAllocator(),
        ),
    ]

    for name, allocator in methods:

        result = run(
            allocator
        )

        show(
            name,
            result,
        )


if __name__ == "__main__":
    main()
