from pathlib import Path
import sys
import csv
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mrta.config import SimulationConfig
from mrta.scenario import Scenario
from mrta.obstacle_simulator import ObstacleSimulator
from mrta.coordinated_allocators import LockAwareHungarianAllocator
from mrta.progress_aware_allocator import ProgressAwareLockAllocator


class TracedSimulator(ObstacleSimulator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.trace = []

    def step(self):
        iteration_before = self.iteration
        global_before = self.allocator.global_assignment_calls
        distance_before = sum(
            agent.total_distance
            for agent in self.environment.agents
        )

        super().step()

        active_agents = self.environment.get_active_agents()
        active_tasks = self.environment.get_active_tasks()

        assignment_ids = getattr(
            self.allocator, "assignment_ids", {}
        )

        assigned = sum(
            agent.agent_id in assignment_ids
            for agent in active_agents
        )

        distance_after = sum(
            agent.total_distance
            for agent in self.environment.agents
        )

        self.trace.append({
            "iteration": iteration_before,
            "active_agents": len(active_agents),
            "remaining_tasks": len(active_tasks),
            "assigned_agents": assigned,
            "unassigned_agents": len(active_agents) - assigned,
            "global_replans": (
                self.allocator.global_assignment_calls
                - global_before
            ),
            "distance_moved": (
                distance_after - distance_before
            ),
            "saturation_streak": getattr(
                self.allocator, "saturation_streak", 0
            ),
        })


def run_case(seed, failure_rate, method):
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
    if failure_rate:
        rng = np.random.default_rng(seed + 50000)
        failed_ids = sorted(
            int(x) for x in rng.choice(
                15, size=int(15 * failure_rate), replace=False
            )
        )

    allocator = (
        LockAwareHungarianAllocator()
        if method == "lock_aware"
        else ProgressAwareLockAllocator(
            threshold=int(method.split("_")[1])
        )
    )

    simulator = TracedSimulator(
        config=config,
        allocator=allocator,
        obstacle_density=0.20,
        scenario=scenario,
        failure_schedule={5: failed_ids} if failed_ids else {},
    )

    result = simulator.run()

    for row in simulator.trace:
        row.update({
            "seed": seed,
            "failure_rate": failure_rate,
            "method": method,
        })

    return result, simulator.trace


def main():
    cases = [
        (2014, 0.2),
        (2009, 0.2),
        (2018, 0.2),
        (2000, 0.0),
    ]

    all_rows = []

    for seed, rate in cases:
        for method in ["lock_aware", "palr_2", "palr_5"]:
            result, trace = run_case(seed, rate, method)
            all_rows.extend(trace)

            print(
                f"seed={seed} failure={rate:.0%} "
                f"{method:10s} "
                f"success={result['success']} "
                f"iterations={result['iterations']} "
                f"trace_rows={len(trace)}"
            )

    output = ROOT / "data/processed/palr_iteration_traces.csv"
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", newline="") as file:
        writer = csv.DictWriter(
            file, fieldnames=list(all_rows[0].keys())
        )
        writer.writeheader()
        writer.writerows(all_rows)

    print("\nSaved:", output)


if __name__ == "__main__":
    main()
