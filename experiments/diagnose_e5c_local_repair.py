from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mrta.config import SimulationConfig
from mrta.scenario import Scenario
from mrta.obstacle_simulator import ObstacleSimulator
from mrta.coordinated_allocators import (
    StatefulHungarianAllocator,
)

ROWS = []

print()
print("=" * 80)
print("E5C LOCAL-REPAIR FAILURE DIAGNOSTIC")
print("=" * 80)

for seed in range(30):

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

    allocator = StatefulHungarianAllocator()

    simulator = ObstacleSimulator(
        config=config,
        allocator=allocator,
        obstacle_density=0.20,
        scenario=scenario,
    )

    result = simulator.run()

    row = {
        "seed": seed,
        "success": int(result["success"]),
        "iterations": result["iterations"],
        "total_distance": result["total_distance"],
        "assignment_requests":
            allocator.assignment_requests,
        "repair_calls":
            allocator.repair_calls,
        "path_cost_evaluations":
            allocator.path_cost_evaluations,
        "empty_repair_events":
            allocator.empty_repair_events,
        "max_unassigned_agents":
            allocator.max_unassigned_agents,
        "max_unassigned_tasks":
            allocator.max_unassigned_tasks,
    }

    ROWS.append(row)

    status = (
        "SUCCESS"
        if result["success"]
        else "TIMEOUT"
    )

    print(
        f"seed={seed:02d}"
        f" {status:7s}"
        f" iter={result['iterations']:3d}"
        f" dist={result['total_distance']:7.1f}"
        f" repairs={allocator.repair_calls:3d}"
        f" astar={allocator.path_cost_evaluations:4d}"
        f" empty={allocator.empty_repair_events:3d}"
        f" maxUA={allocator.max_unassigned_agents:2d}"
        f" maxUT={allocator.max_unassigned_tasks:2d}"
    )


df = pd.DataFrame(ROWS)

output = (
    ROOT
    / "data/processed"
    / "e5c_local_repair_diagnostic.csv"
)

output.parent.mkdir(
    parents=True,
    exist_ok=True,
)

df.to_csv(
    output,
    index=False,
)

success = df[df["success"] == 1]
failure = df[df["success"] == 0]

print()
print("=" * 80)
print("SUMMARY")
print("=" * 80)

print(
    "Successful:",
    len(success),
    "/ 30",
)

print(
    "Timed out:",
    len(failure),
    "/ 30",
)

for label, group in [
    ("SUCCESS", success),
    ("TIMEOUT", failure),
]:
    if len(group) == 0:
        continue

    print()
    print(label)

    print(
        "  mean repair calls:",
        round(
            group["repair_calls"].mean(),
            2,
        ),
    )

    print(
        "  mean A* evaluations:",
        round(
            group[
                "path_cost_evaluations"
            ].mean(),
            2,
        ),
    )

    print(
        "  mean empty-repair events:",
        round(
            group[
                "empty_repair_events"
            ].mean(),
            2,
        ),
    )

    print(
        "  mean max unassigned agents:",
        round(
            group[
                "max_unassigned_agents"
            ].mean(),
            2,
        ),
    )

    print(
        "  mean max unassigned tasks:",
        round(
            group[
                "max_unassigned_tasks"
            ].mean(),
            2,
        ),
    )

print()
print("Saved:", output)
print("=" * 80)
