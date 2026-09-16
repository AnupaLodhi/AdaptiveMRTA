from pathlib import Path
import sys

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

failure_schedule = {
    5: [0, 1, 2],
}

methods = [
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


print()
print("=" * 78)
print("E4 DYNAMIC FAILURE VALIDATION")
print("=" * 78)
print("Agents: 15")
print("Tasks: 20")
print("Obstacle density: 20%")
print("Failure iteration: 5")
print("Failed robots: [0, 1, 2]")
print(
    "Scenario:",
    scenario.fingerprint()[:16],
)
print("=" * 78)


for name, allocator_class in methods:

    simulator = ObstacleSimulator(
        config=config,
        allocator=allocator_class(),
        obstacle_density=0.20,
        scenario=scenario,
        failure_schedule=failure_schedule,
    )

    result = simulator.run()

    print()
    print(name)
    print(
        "  Success:",
        result["success"],
    )
    print(
        "  Completed:",
        f"{result['completed_tasks']}/"
        f"{result['total_tasks']}",
    )
    print(
        "  Iterations:",
        result["iterations"],
    )
    print(
        "  Distance:",
        result["total_distance"],
    )
    print(
        "  Failed agents:",
        result["failed_agent_ids"],
    )
    print(
        "  Failure events:",
        result["failure_events"],
    )
    print(
        "  Blocked:",
        result["blocked_moves"],
    )
    print(
        "  Redundancy:",
        round(
            result[
                "redundant_target_ratio"
            ],
            4,
        ),
    )

    assert (
        result["failed_agent_ids"]
        == [0, 1, 2]
    )

    assert len(
        result["failure_events"]
    ) == 3

    assert all(
        event["iteration"] == 5
        for event
        in result["failure_events"]
    )

    # Failed robots must remain inactive.
    for agent_id in [0, 1, 2]:
        agent = next(
            a
            for a
            in simulator.environment.agents
            if a.agent_id == agent_id
        )

        assert not agent.active


print()
print("=" * 78)
print("E4 FAILURE MECHANISM: PASS")
print("=" * 78)
print(
    "Same scenario and same mid-run failures "
    "were applied to all three algorithms."
)
