from pathlib import Path
import sys
import numpy as np


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


methods = [
    ManhattanGreedyAllocator(),
    PathCostGreedyAllocator(),
    HungarianPathAllocator(),
]


simulators = [
    ObstacleSimulator(
        config=config,
        allocator=allocator,
        obstacle_density=0.20,
        scenario=scenario,
    )
    for allocator in methods
]


reference = simulators[0].environment


for simulator in simulators[1:]:

    env = simulator.environment

    assert np.array_equal(
        reference.grid_map.obstacles,
        env.grid_map.obstacles,
    )

    assert [
        tuple(a.position)
        for a in reference.agents
    ] == [
        tuple(a.position)
        for a in env.agents
    ]

    assert [
        tuple(t.position)
        for t in reference.tasks
    ] == [
        tuple(t.position)
        for t in env.tasks
    ]


print()
print("=" * 60)
print("SCENARIO CLONING TEST: PASS")
print("=" * 60)

print(
    "Fingerprint:",
    scenario.fingerprint(),
)

print(
    "Agents:",
    len(scenario.agent_positions),
)

print(
    "Tasks:",
    len(scenario.task_positions),
)

print(
    "Obstacle cells:",
    int(
        scenario.obstacle_grid.sum()
    ),
)

print()
print(
    "All algorithms received identical "
    "initial conditions."
)
print("=" * 60)
