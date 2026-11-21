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
from mrta.progress_aware_allocator import ProgressAwareLockAllocator
from mrta.adaptive_reassignment_allocator import AdaptiveReassignmentAllocator
from mrta.cost_aware_selective_allocator import CostAwareSelectiveAllocator


methods = {
    "palr_5": lambda: ProgressAwareLockAllocator(5),
    "adaptive": AdaptiveReassignmentAllocator,
    "casr_0": lambda: CostAwareSelectiveAllocator(margin=0),
    "casr_3": lambda: CostAwareSelectiveAllocator(margin=3),
    "casr_5": lambda: CostAwareSelectiveAllocator(margin=5),
}

cases = [
    (2004, 0.2),
    (2013, 0.0),
    (2014, 0.0),
]

rows = []

for seed, rate in cases:
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

    for name, factory in methods.items():
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

        total_astar = (
            allocator.path_cost_evaluations
            + result["navigation_astar_calls"]
        )

        row = {
            "seed": seed,
            "failure_rate": rate,
            "method": name,
            "success": int(result["success"]),
            "iterations": result["iterations"],
            "total_distance": result["total_distance"],
            "total_astar": total_astar,
            "global_calls": allocator.global_assignment_calls,
            "selective_checks": getattr(
                allocator, "selective_checks", 0
            ),
            "selective_transfers": getattr(
                allocator, "selective_transfers", 0
            ),
            "runtime_seconds": runtime,
        }

        rows.append(row)

        print(
            f"seed={seed} rate={rate:.0%} "
            f"{name:9s} "
            f"success={result['success']} "
            f"iterations={result['iterations']:3d} "
            f"astar={total_astar:5d} "
            f"transfers={row['selective_transfers']}"
        )

output = ROOT / "data/raw/casr_smoke.csv"
output.parent.mkdir(parents=True, exist_ok=True)

with output.open("w", newline="") as f:
    writer = csv.DictWriter(
        f, fieldnames=list(rows[0])
    )
    writer.writeheader()
    writer.writerows(rows)

print("\nSaved:", output)
print("Total runs:", len(rows))
