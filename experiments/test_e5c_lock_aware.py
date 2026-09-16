from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mrta.config import SimulationConfig
from mrta.scenario import Scenario
from mrta.obstacle_simulator import ObstacleSimulator
from mrta.coordinated_allocators import (
    LockAwareHungarianAllocator,
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

allocator = LockAwareHungarianAllocator()

sim = ObstacleSimulator(
    config=config,
    allocator=allocator,
    obstacle_density=0.20,
    scenario=scenario,
)

result = sim.run()

assert (
    allocator.path_cost_evaluations
    ==
    allocator.successful_path_cost_evaluations
    +
    allocator.unreachable_path_cost_evaluations
)

print("Success:", result["success"])
print("Iterations:", result["iterations"])
print("Distance:", result["total_distance"])
print(
    "Global calls:",
    allocator.global_assignment_calls,
)
print(
    "Local repairs:",
    allocator.local_repair_calls,
)
print(
    "Lock escalations:",
    allocator.lock_escalations,
)
print(
    "A* evaluations:",
    allocator.path_cost_evaluations,
)
print(
    "Unreachable A*:",
    allocator.unreachable_path_cost_evaluations,
)
