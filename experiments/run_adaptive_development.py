from pathlib import Path
import sys
import csv
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mrta.config import SimulationConfig
from mrta.scenario import Scenario
from mrta.adaptive_simulator import AdaptiveObstacleSimulator
from mrta.coordinated_allocators import (
    LockAwareHungarianAllocator,
)
from mrta.progress_aware_allocator import (
    ProgressAwareLockAllocator,
)
from mrta.adaptive_reassignment_allocator import (
    AdaptiveReassignmentAllocator,
)

METHODS = {
    "lock_aware": LockAwareHungarianAllocator,
    "palr_2": lambda: ProgressAwareLockAllocator(2),
    "palr_5": lambda: ProgressAwareLockAllocator(5),
    "adaptive": AdaptiveReassignmentAllocator,
}

rows = []

for rate in [0.0, 0.2]:
    for seed in range(2000, 2020):
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

        failed_ids = []

        if rate:
            rng = np.random.default_rng(seed + 50000)
            failed_ids = sorted(
                int(x)
                for x in rng.choice(
                    15,
                    size=int(15 * rate),
                    replace=False,
                )
            )

        schedule = {5: failed_ids} if failed_ids else {}

        for name, factory in METHODS.items():
            allocator = factory()

            sim = AdaptiveObstacleSimulator(
                config=config,
                allocator=allocator,
                obstacle_density=0.20,
                scenario=scenario,
                failure_schedule=schedule,
            )

            start = time.perf_counter()
            result = sim.run()
            runtime = time.perf_counter() - start

            allocation_astar = allocator.path_cost_evaluations
            navigation_astar = result["navigation_astar_calls"]

            rows.append({
                "seed": seed,
                "failure_rate": rate,
                "method": name,
                "success": int(result["success"]),
                "iterations": result["iterations"],
                "total_distance": result["total_distance"],
                "allocation_astar": allocation_astar,
                "navigation_astar": navigation_astar,
                "total_astar": allocation_astar + navigation_astar,
                "global_calls": allocator.global_assignment_calls,
                "runtime_seconds": runtime,
                "stall_triggered_requests": getattr(
                    allocator,
                    "stall_triggered_requests",
                    0,
                ),
            })

            print(
                f"seed={seed} rate={rate:.0%} "
                f"{name:10s} "
                f"success={result['success']} "
                f"iter={result['iterations']} "
                f"astar={allocation_astar + navigation_astar}"
            )

output = ROOT / "data/raw/adaptive_development_20_seeds.csv"
output.parent.mkdir(parents=True, exist_ok=True)

with output.open("w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=list(rows[0]),
    )
    writer.writeheader()
    writer.writerows(rows)

print("\nSaved:", output)
print("Total runs:", len(rows))
