from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mrta.config import SimulationConfig
from mrta.scenario import Scenario
from mrta.obstacle_simulator import ObstacleSimulator
from mrta.coordinated_allocators import (
    LockAwareHungarianAllocator,
)

TIMEOUT_SEEDS = [
    2, 3, 4, 6, 7,
    8, 9, 10, 11, 12,
    13, 15, 17, 18, 21,
    23, 24, 25, 26, 28,
]

rows = []

print()
print("=" * 86)
print("E5C LOCK-AWARE — PREVIOUS TIMEOUT SEEDS")
print("=" * 86)

for seed in TIMEOUT_SEEDS:

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

    allocator = LockAwareHungarianAllocator()

    simulator = ObstacleSimulator(
        config=config,
        allocator=allocator,
        obstacle_density=0.20,
        scenario=scenario,
    )

    result = simulator.run()

    assert (
        allocator.path_cost_evaluations
        ==
        allocator.successful_path_cost_evaluations
        +
        allocator.unreachable_path_cost_evaluations
    )

    rows.append({
        "seed":
            seed,

        "success":
            int(result["success"]),

        "iterations":
            result["iterations"],

        "total_distance":
            result["total_distance"],

        "global_calls":
            allocator.global_assignment_calls,

        "local_repairs":
            allocator.local_repair_calls,

        "lock_escalations":
            allocator.lock_escalations,

        "path_cost_evaluations":
            allocator.path_cost_evaluations,
    })

    print(
        f"seed={seed:02d}"
        f" success={int(result['success'])}"
        f" iter={result['iterations']:3d}"
        f" dist={result['total_distance']:7.1f}"
        f" global={allocator.global_assignment_calls:3d}"
        f" local={allocator.local_repair_calls:3d}"
        f" locks={allocator.lock_escalations:3d}"
        f" astar={allocator.path_cost_evaluations:5d}"
    )


df = pd.DataFrame(rows)

output = (
    ROOT
    / "data/processed"
    / "e5c_lock_aware_timeout_seeds.csv"
)

output.parent.mkdir(
    parents=True,
    exist_ok=True,
)

df.to_csv(
    output,
    index=False,
)

print()
print("=" * 86)
print("SUMMARY")
print("=" * 86)

print(
    "Recovered:",
    int(df["success"].sum()),
    "/",
    len(df),
)

successful = df[
    df["success"] == 1
]

if len(successful):

    print(
        "Mean iterations:",
        round(
            successful["iterations"].mean(),
            2,
        ),
    )

    print(
        "Mean distance:",
        round(
            successful["total_distance"].mean(),
            2,
        ),
    )

    print(
        "Mean global calls:",
        round(
            successful["global_calls"].mean(),
            2,
        ),
    )

    print(
        "Mean lock escalations:",
        round(
            successful["lock_escalations"].mean(),
            2,
        ),
    )

    print(
        "Mean A* evaluations:",
        round(
            successful[
                "path_cost_evaluations"
            ].mean(),
            2,
        ),
    )

print()
print("Saved:", output)
print("=" * 86)
