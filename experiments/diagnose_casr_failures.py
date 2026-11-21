from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mrta.config import SimulationConfig
from mrta.scenario import Scenario
from mrta.adaptive_simulator import AdaptiveObstacleSimulator
from mrta.cost_aware_selective_allocator import CostAwareSelectiveAllocator

for seed in [2013, 2014]:
    for margin in [0, 3, 5]:
        config = SimulationConfig(
            num_agents=15,
            num_tasks=20,
            seed=seed,
            max_iterations=500,
        )

        scenario = Scenario.generate(
            config=config,
            obstacle_density=0.20,
        )

        allocator = CostAwareSelectiveAllocator(
            margin=margin,
            threshold=5,
        )

        sim = AdaptiveObstacleSimulator(
            config=config,
            allocator=allocator,
            obstacle_density=0.20,
            scenario=scenario,
        )

        result = sim.run()

        print("\n" + "=" * 65)
        print(f"Seed={seed} Margin={margin}")
        print("=" * 65)

        print("Success:", result["success"])
        print("Iterations:", result["iterations"])
        print(
            "Completed tasks:",
            result["completed_tasks"],
            "/",
            result["total_tasks"],
        )
        print("Global calls:", allocator.global_assignment_calls)
        print("Selective checks:", allocator.selective_checks)
        print("Selective transfers:", allocator.selective_transfers)
        print("Rejected checks:", allocator.rejected_checks)
        print("Allocation A* calls:", allocator.path_cost_evaluations)

        print("Final reservations:", allocator.assignment_ids)
