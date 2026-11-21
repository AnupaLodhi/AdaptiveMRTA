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


class DiagnosticLockAware(LockAwareHungarianAllocator):
    def __init__(self):
        super().__init__()
        self.matrix_events = []
        self._inside_global = False

    def _solve_global(self, agents, tasks, grid_map):
        self._inside_global = True
        try:
            return super()._solve_global(
                agents, tasks, grid_map
            )
        finally:
            self._inside_global = False

    def _build_cost_matrix(self, agents, tasks, grid_map):
        rows = len(agents)
        cols = len(tasks)

        self.matrix_events.append({
            "scope": "global" if self._inside_global else "local",
            "robots": rows,
            "tasks": cols,
            "pairs": rows * cols,
        })

        return super()._build_cost_matrix(
            agents, tasks, grid_map
        )


def main():
    output = ROOT / "data/processed/lock_aware_matrix_diagnostics.csv"
    output.parent.mkdir(parents=True, exist_ok=True)

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

            failed = []
            if failure_rate:
                rng = np.random.default_rng(seed + 50000)
                failed = sorted(
                    int(x) for x in rng.choice(
                        15, size=3, replace=False
                    )
                )

            allocator = DiagnosticLockAware()

            simulator = ObstacleSimulator(
                config=config,
                allocator=allocator,
                obstacle_density=0.20,
                scenario=scenario,
                failure_schedule={5: failed} if failed else {},
            )

            result = simulator.run()

            for event in allocator.matrix_events:
                rows.append({
                    "seed": seed,
                    "failure_rate": failure_rate,
                    **event,
                })

            measured_pairs = sum(
                event["pairs"]
                for event in allocator.matrix_events
            )

            assert (
                measured_pairs
                == allocator.path_cost_evaluations
            ), "Matrix-pair accounting mismatch"

            print(
                f"seed={seed} failure={failure_rate:.0%} "
                f"success={result['success']} "
                f"matrices={len(allocator.matrix_events)} "
                f"pairs={measured_pairs}"
            )

    with output.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "seed", "failure_rate",
                "scope", "robots", "tasks", "pairs"
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print("\nSaved:", output)


if __name__ == "__main__":
    main()
