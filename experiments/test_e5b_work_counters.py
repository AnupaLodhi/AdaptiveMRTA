from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mrta.config import SimulationConfig
from mrta.scenario import Scenario
from mrta.obstacle_simulator import ObstacleSimulator

from mrta.coordinated_allocators import (
    HungarianPathAllocator,
    PeriodicHungarianAllocator,
)


def run(
    name,
    allocator,
    scenario,
    config,
):
    sim = ObstacleSimulator(
        config=config,
        allocator=allocator,
        obstacle_density=0.20,
        scenario=scenario,
    )

    result = sim.run()

    total = allocator.path_cost_evaluations
    successful = (
        allocator.successful_path_cost_evaluations
    )
    unreachable = (
        allocator.unreachable_path_cost_evaluations
    )

    print()
    print(name)
    print("-" * 60)
    print("Success:", result["success"])
    print("Iterations:", result["iterations"])
    print("Distance:", result["total_distance"])
    print(
        "Assignment requests:",
        allocator.assignment_requests,
    )
    print(
        "Global assignment calls:",
        allocator.global_assignment_calls,
    )
    print(
        "A* cost evaluations:",
        total,
    )
    print(
        "Successful A*:",
        successful,
    )
    print(
        "Unreachable A*:",
        unreachable,
    )

    assert result["success"]

    assert (
        total
        == successful + unreachable
    )

    assert (
        allocator.global_assignment_calls
        <= allocator.assignment_requests
    )

    return result, allocator


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

repeated_result, repeated = run(
    "REPEATED HUNGARIAN",
    HungarianPathAllocator(),
    scenario,
    config,
)

event_result, event = run(
    "EVENT-GLOBAL",
    PeriodicHungarianAllocator(
        replan_interval=1000
    ),
    scenario,
    config,
)

assert (
    repeated_result["iterations"]
    == event_result["iterations"]
)

assert (
    repeated_result["total_distance"]
    == event_result["total_distance"]
)

saving = (
    1.0
    - (
        event.path_cost_evaluations
        / repeated.path_cost_evaluations
    )
) * 100.0

print()
print("=" * 60)
print(
    "A* EVALUATION REDUCTION:",
    f"{saving:.2f}%"
)
print(
    "SOLUTION PERFORMANCE IDENTICAL: PASS"
)
print(
    "E5B WORK COUNTER VALIDATION: PASS"
)
print("=" * 60)
