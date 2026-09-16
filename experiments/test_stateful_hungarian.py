from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mrta.config import SimulationConfig
from mrta.scenario import Scenario
from mrta.obstacle_simulator import ObstacleSimulator

from mrta.coordinated_allocators import (
    HungarianPathAllocator,
    StatefulHungarianAllocator,
)


def run_case(
    label,
    failure_schedule,
):
    config = SimulationConfig(
        num_agents=15,
        num_tasks=20,
        seed=42,
        max_iterations=500,
    )

    scenario = Scenario.generate(
        config=config,
        obstacle_density=0.20,
    )

    print()
    print("=" * 76)
    print(label)
    print("=" * 76)

    # Existing repeated Hungarian
    repeated_allocator = (
        HungarianPathAllocator()
    )

    repeated = ObstacleSimulator(
        config=config,
        allocator=repeated_allocator,
        obstacle_density=0.20,
        scenario=scenario,
        failure_schedule=failure_schedule,
    )

    repeated_result = repeated.run()

    print()
    print("REPEATED HUNGARIAN")
    print(
        "Success:",
        repeated_result["success"],
    )
    print(
        "Iterations:",
        repeated_result["iterations"],
    )
    print(
        "Distance:",
        repeated_result["total_distance"],
    )
    print(
        "Blocked:",
        repeated_result["blocked_moves"],
    )
    print(
        "Redundancy:",
        repeated_result[
            "redundant_target_ratio"
        ],
    )

    # New stateful version
    stateful_allocator = (
        StatefulHungarianAllocator()
    )

    stateful = ObstacleSimulator(
        config=config,
        allocator=stateful_allocator,
        obstacle_density=0.20,
        scenario=scenario,
        failure_schedule=failure_schedule,
    )

    stateful_result = stateful.run()

    print()
    print("STATEFUL HUNGARIAN")
    print(
        "Success:",
        stateful_result["success"],
    )
    print(
        "Iterations:",
        stateful_result["iterations"],
    )
    print(
        "Distance:",
        stateful_result["total_distance"],
    )
    print(
        "Blocked:",
        stateful_result["blocked_moves"],
    )
    print(
        "Redundancy:",
        stateful_result[
            "redundant_target_ratio"
        ],
    )
    print(
        "Assignment requests:",
        stateful_allocator
        .assignment_requests,
    )
    print(
        "Global Hungarian calls:",
        stateful_allocator
        .global_assignment_calls,
    )

    assert stateful_result["success"]

    assert (
        stateful_allocator
        .global_assignment_calls
        <
        stateful_allocator
        .assignment_requests
    )

    print()
    print("CASE: PASS")


run_case(
    "STATIC CASE",
    {},
)

run_case(
    "20% MID-RUN FAILURE CASE",
    {
        5: [0, 1, 2],
    },
)

print()
print("=" * 76)
print("E5 STATEFUL HUNGARIAN VALIDATION: PASS")
print("=" * 76)
