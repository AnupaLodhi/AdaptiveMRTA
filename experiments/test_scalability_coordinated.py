from pathlib import Path
import sys
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

OBSTACLE_DENSITY = 0.20
SEED = 42
MAX_ITERATIONS = 500


print()
print("=" * 78)
print("SCALABILITY VALIDATION TEST")
print("=" * 78)
print("Obstacle density: 20%")
print("Seed: 42")
print()


for num_agents, num_tasks in SCENARIOS:

    config = SimulationConfig(
        num_agents=num_agents,
        num_tasks=num_tasks,
        seed=SEED,
        max_iterations=MAX_ITERATIONS,
    )

    # Generate the scenario exactly once.
    scenario = Scenario.generate(
        config=config,
        obstacle_density=OBSTACLE_DENSITY,
    )

    fingerprint = scenario.fingerprint()

    print("-" * 78)
    print(
        f"{num_agents:2d} agents / "
        f"{num_tasks:2d} tasks"
    )

    print(
        "Scenario:",
        fingerprint[:16],
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

        print(
            f"  {name:22s}"
            f" success={int(result['success'])}"
            f" iter={result['iterations']:3d}"
            f" dist={result['total_distance']:7.1f}"
            f" blocked={result['blocked_moves']:3d}"
            f" redund={result['redundant_target_ratio']:.4f}"
            f" runtime={runtime:.3f}s"
        )

        if not result["success"]:
            raise RuntimeError(
                f"{name} failed for "
                f"{num_agents}/{num_tasks}."
            )


print()
print("=" * 78)
print("SCALABILITY VALIDATION: PASS")
print("=" * 78)
print(
    "All six problem sizes completed "
    "with all three algorithms."
)
print()
