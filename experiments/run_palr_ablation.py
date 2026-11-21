from pathlib import Path
import sys
import csv
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mrta.config import SimulationConfig
from mrta.scenario import Scenario
from mrta.obstacle_simulator import ObstacleSimulator
from mrta.coordinated_allocators import LockAwareHungarianAllocator
from mrta.progress_aware_allocator import ProgressAwareLockAllocator

OUTPUT = ROOT / "data/raw/palr_ablation_5_seeds.csv"

METHODS = {
    "lock_aware": lambda: LockAwareHungarianAllocator(),
    "palr_0": lambda: ProgressAwareLockAllocator(0),
    "palr_2": lambda: ProgressAwareLockAllocator(2),
    "palr_5": lambda: ProgressAwareLockAllocator(5),
}

rows = []

for failure_rate in [0.0, 0.2]:
    for seed in range(1000, 1005):
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

        count = int(15 * failure_rate)
        failed_ids = []

        if count:
            rng = np.random.default_rng(seed + 50000)
            failed_ids = sorted(
                int(x) for x in rng.choice(
                    15, size=count, replace=False
                )
            )

        failure_schedule = (
            {5: failed_ids} if failed_ids else {}
        )

        for method, factory in METHODS.items():
            allocator = factory()

            simulator = ObstacleSimulator(
                config=config,
                allocator=allocator,
                obstacle_density=0.20,
                scenario=scenario,
                failure_schedule=failure_schedule,
            )

            start = time.perf_counter()
            result = simulator.run()
            runtime = time.perf_counter() - start

            allocation_astar = allocator.path_cost_evaluations
            navigation_astar = result["navigation_astar_calls"]

            rows.append({
                "seed": seed,
                "failure_rate": failure_rate,
                "scenario_fingerprint": scenario.fingerprint(),
                "failed_agent_ids": ";".join(map(str, failed_ids)),
                "method": method,
                "success": int(result["success"]),
                "iterations": result["iterations"],
                "total_distance": result["total_distance"],
                "blocked_moves": result["blocked_moves"],
                "allocation_astar": allocation_astar,
                "navigation_astar": navigation_astar,
                "total_astar": allocation_astar + navigation_astar,
                "global_calls": allocator.global_assignment_calls,
                "local_repairs": allocator.local_repair_calls,
                "lock_escalations": allocator.lock_escalations,
                "saturation_events": getattr(
                    allocator, "saturation_events", 0
                ),
                "delayed_escalations": getattr(
                    allocator, "delayed_escalations", 0
                ),
                "runtime_seconds": runtime,
            })

            print(
                f"failure={failure_rate:.0%} "
                f"seed={seed} {method:11s} "
                f"success={int(result['success'])} "
                f"iter={result['iterations']:3d} "
                f"total_astar={allocation_astar + navigation_astar:5d} "
                f"global={allocator.global_assignment_calls:3d}"
            )

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

with OUTPUT.open("w", newline="") as file:
    writer = csv.DictWriter(
        file, fieldnames=list(rows[0].keys())
    )
    writer.writeheader()
    writer.writerows(rows)

print("\nCompleted:", len(rows), "runs")
print("Saved:", OUTPUT)
