from pathlib import Path
import sys
import csv
import time
import gc
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mrta.config import SimulationConfig
from mrta.scenario import Scenario
from mrta.adaptive_simulator import AdaptiveObstacleSimulator
from mrta.coordinated_allocators import LockAwareHungarianAllocator
from mrta.adaptive_reassignment_allocator import AdaptiveReassignmentAllocator
from mrta.progress_aware_allocator import ProgressAwareLockAllocator
from mrta.cost_aware_selective_allocator import CostAwareSelectiveAllocator
from mrta.stall_aware_allocator import StallAwareCASRAllocator

METHODS = {
    "lock_aware": LockAwareHungarianAllocator,
    "adaptive": AdaptiveReassignmentAllocator,
    "palr_5": lambda: ProgressAwareLockAllocator(5),
    "casr_3": lambda: CostAwareSelectiveAllocator(
        margin=3, threshold=5
    ),
    "casr_v2_s2": lambda: StallAwareCASRAllocator(
        margin=3, threshold=5, stall_threshold=2
    ),
}

CONFIGURATIONS = [
    ("medium", 15, 20, 52, 30),
    ("extra_large", 50, 100, 52, 30),
]

SEEDS = range(6000, 6005)
FAILURE_RATES = (0.0, 0.2)
REPETITIONS = 3

rng = np.random.default_rng(2027)
rows = []

for name, robots, tasks, width, height in CONFIGURATIONS:
    for seed in SEEDS:
        config = SimulationConfig(
            num_agents=robots,
            num_tasks=tasks,
            grid_width=width,
            grid_height=height,
            seed=seed,
            max_iterations=500,
        )

        scenario = Scenario.generate(
            config=config,
            obstacle_density=0.20,
        )

        for failure_rate in FAILURE_RATES:
            failed_ids = []

            if failure_rate:
                failure_rng = np.random.default_rng(
                    seed + 50000
                )
                failed_ids = sorted(
                    int(x)
                    for x in failure_rng.choice(
                        robots,
                        size=int(robots * failure_rate),
                        replace=False,
                    )
                )

            schedule = (
                {5: failed_ids} if failed_ids else {}
            )

            for repetition in range(REPETITIONS):
                method_order = list(METHODS)
                rng.shuffle(method_order)

                for method in method_order:
                    allocator = METHODS[method]()

                    sim = AdaptiveObstacleSimulator(
                        config=config,
                        allocator=allocator,
                        obstacle_density=0.20,
                        scenario=scenario,
                        failure_schedule=schedule,
                    )

                    gc.collect()

                    start = time.perf_counter()
                    result = sim.run()
                    elapsed = (
                        time.perf_counter() - start
                    )

                    row = {
                        "configuration": name,
                        "seed": seed,
                        "failure_rate": failure_rate,
                        "repetition": repetition,
                        "method": method,
                        "runtime_seconds": elapsed,
                        "success": int(result["success"]),
                        "iterations": result["iterations"],
                        "completed_tasks": result["completed_tasks"],
                        "blocked_moves": result["blocked_moves"],
                        "total_astar": (
                            allocator.path_cost_evaluations
                            + result["navigation_astar_calls"]
                        ),
                        "scenario_fingerprint": (
                            scenario.fingerprint()
                        ),
                    }
                    rows.append(row)

                    print(
                        f"{len(rows):3d}/300 "
                        f"{name} seed={seed} "
                        f"failure={failure_rate} "
                        f"repeat={repetition} "
                        f"{method} "
                        f"runtime={elapsed:.4f}s "
                        f"success={row['success']}",
                        flush=True,
                    )

output = ROOT / "data/raw/casr_v2_runtime.csv"
output.parent.mkdir(parents=True, exist_ok=True)

with output.open("w", newline="") as f:
    writer = csv.DictWriter(
        f, fieldnames=list(rows[0])
    )
    writer.writeheader()
    writer.writerows(rows)

assert len(rows) == 300

print("\nSaved:", output)
print("Total timed runs:", len(rows))
print("Successful:", sum(r["success"] for r in rows))
