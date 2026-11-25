from pathlib import Path
import sys
import csv
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mrta.config import SimulationConfig
from mrta.coordinated_allocators import LockAwareHungarianAllocator
from mrta.adaptive_reassignment_allocator import AdaptiveReassignmentAllocator
from mrta.scenario import Scenario
from mrta.adaptive_simulator import AdaptiveObstacleSimulator
from mrta.progress_aware_allocator import ProgressAwareLockAllocator
from mrta.cost_aware_selective_allocator import CostAwareSelectiveAllocator
from mrta.stall_aware_allocator import StallAwareCASRAllocator

methods = {
    "lock_aware": LockAwareHungarianAllocator,
    "adaptive": AdaptiveReassignmentAllocator,
    "palr_5": lambda: ProgressAwareLockAllocator(5),
    "casr_0": lambda: CostAwareSelectiveAllocator(margin=0),
    "casr_3": lambda: CostAwareSelectiveAllocator(margin=3),
    "casr_v2_s2": lambda: StallAwareCASRAllocator(
        stall_threshold=2
    ),
    "casr_v2_s3": lambda: StallAwareCASRAllocator(
        stall_threshold=3
    ),
}

cases = [
    (seed, rate)
    for seed in range(3000, 3020)
    for rate in [0.0, 0.2]
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

    for method, factory in methods.items():

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

        row = {
            "seed": seed,
            "failure_rate": rate,
            "method": method,
            "success": int(result["success"]),
            "iterations": result["iterations"],
            "completed_tasks": result["completed_tasks"],
            "total_distance": result["total_distance"],
            "blocked_moves": result["blocked_moves"],
            "total_astar": (
                allocator.path_cost_evaluations
                + result["navigation_astar_calls"]
            ),
            "global_calls": allocator.global_assignment_calls,
            "selective_transfers": getattr(
                allocator, "selective_transfers", 0
            ),
            "stall_checks": getattr(
                allocator, "stall_checks", 0
            ),
            "stall_transfers": getattr(
                allocator, "stall_transfers", 0
            ),
            "stall_rejections": getattr(
                allocator, "stall_rejections", 0
            ),
            "recovery_calls": getattr(
                allocator, "recovery_global_calls", 0
            ),
            "runtime_seconds": runtime,
        }

        rows.append(row)

        print(
            f"seed={seed} "
            f"{method:12s} "
            f"success={row['success']} "
            f"iterations={row['iterations']:3d} "
            f"blocked={row['blocked_moves']:3d} "
            f"astar={row['total_astar']:5d} "
            f"stall_transfers={row['stall_transfers']}"
        )

output = ROOT / "data/raw/casr_v2_holdout.csv"
output.parent.mkdir(parents=True, exist_ok=True)

with output.open("w", newline="") as f:
    writer = csv.DictWriter(
        f, fieldnames=list(rows[0])
    )
    writer.writeheader()
    writer.writerows(rows)

print("\nSaved:", output)
print("Total runs:", len(rows))
