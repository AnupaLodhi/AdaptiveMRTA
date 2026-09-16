from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"

sys.path.insert(0, str(SRC_PATH))


from mrta.config import SimulationConfig
from mrta.algorithms.greedy import GreedyAllocator
from mrta.algorithms.fixed_pso import FixedPSOAllocator
from mrta.algorithms.dynamic_pso import DynamicPSOAllocator
from mrta.simulator import Simulator


NUM_TRIALS = 30

SCENARIOS = [
    (5, 10),
    (10, 10),
    (15, 10),
    (20, 10),
    (15, 20),
    (20, 20),
    (30, 30),
]


class DynamicPSOSimulator(Simulator):
    def step(self):
        active_tasks = self.environment.get_active_tasks()

        if not active_tasks:
            return

        self.algorithm.update_coefficients(
            len(active_tasks)
        )

        for agent in self.environment.get_active_agents():
            target = self.algorithm.select_target(
                agent,
                active_tasks,
            )

            if target is None:
                continue

            agent.target_id = target.task_id

            self.algorithm.move_agent(
                agent,
                target,
            )

        self._resolve_completed_tasks()

        self.iteration += 1


def collect_result(
    algorithm_name,
    num_agents,
    num_tasks,
    seed,
):
    config = SimulationConfig(
        num_agents=num_agents,
        num_tasks=num_tasks,
        seed=seed,
        max_iterations=2000,
    )

    if algorithm_name == "greedy":
        algorithm = GreedyAllocator(config)

        simulator = Simulator(
            config=config,
            algorithm=algorithm,
        )

    elif algorithm_name == "fixed_pso":
        simulator = Simulator(
            config=config,
            algorithm=None,
        )

        algorithm = FixedPSOAllocator(
            config=config,
            rng=simulator.environment.rng,
        )

        simulator.algorithm = algorithm

    elif algorithm_name == "dynamic_pso":
        simulator = DynamicPSOSimulator(
            config=config,
            algorithm=None,
        )

        algorithm = DynamicPSOAllocator(
            config=config,
            rng=simulator.environment.rng,
        )

        simulator.algorithm = algorithm

    else:
        raise ValueError(
            f"Unknown algorithm: {algorithm_name}"
        )

    result = simulator.run()

    productive_agents = sum(
        1
        for agent in simulator.environment.agents
        if agent.tasks_completed > 0
    )

    unproductive_agents = (
        num_agents - productive_agents
    )

    return {
        "algorithm": algorithm_name,
        "num_agents": num_agents,
        "num_tasks": num_tasks,
        "seed": seed,
        "success": result["success"],
        "iterations": result["iterations"],
        "total_distance": result["total_distance"],
        "tasks_completed": result["completed_tasks"],
        "productive_agents": productive_agents,
        "unproductive_agents": unproductive_agents,
    }


def main():
    rows = []

    algorithms = [
        "greedy",
        "fixed_pso",
        "dynamic_pso",
    ]

    print("\n==============================")
    print("SCALABILITY EXPERIMENT")
    print("==============================\n")

    for num_agents, num_tasks in SCENARIOS:
        print(
            f"\nScenario: "
            f"{num_agents} agents / "
            f"{num_tasks} tasks"
        )

        print("-" * 45)

        for algorithm_name in algorithms:
            algorithm_rows = []

            for seed in range(NUM_TRIALS):
                result = collect_result(
                    algorithm_name,
                    num_agents,
                    num_tasks,
                    seed,
                )

                rows.append(result)
                algorithm_rows.append(result)

            temp_df = pd.DataFrame(
                algorithm_rows
            )

            print(
                f"{algorithm_name:12s} | "
                f"Success="
                f"{temp_df['success'].mean()*100:6.1f}% | "
                f"Iterations="
                f"{temp_df['iterations'].mean():7.2f} | "
                f"Distance="
                f"{temp_df['total_distance'].mean():9.2f} | "
                f"Unused="
                f"{temp_df['unproductive_agents'].mean():5.2f}"
            )

    df = pd.DataFrame(rows)

    raw_dir = (
        PROJECT_ROOT
        / "data"
        / "raw"
    )

    table_dir = (
        PROJECT_ROOT
        / "results"
        / "tables"
    )

    raw_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    table_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    raw_path = (
        raw_dir
        / "scalability_trials.csv"
    )

    df.to_csv(
        raw_path,
        index=False,
    )

    summary = (
        df.groupby(
            [
                "algorithm",
                "num_agents",
                "num_tasks",
            ]
        )
        .agg(
            success_rate=(
                "success",
                "mean",
            ),
            mean_iterations=(
                "iterations",
                "mean",
            ),
            std_iterations=(
                "iterations",
                "std",
            ),
            mean_distance=(
                "total_distance",
                "mean",
            ),
            std_distance=(
                "total_distance",
                "std",
            ),
            mean_unused_agents=(
                "unproductive_agents",
                "mean",
            ),
        )
        .reset_index()
    )

    summary["success_rate"] *= 100

    summary_path = (
        table_dir
        / "scalability_summary.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    print("\n==============================")
    print("FILES SAVED")
    print("==============================")

    print(raw_path)
    print(summary_path)


if __name__ == "__main__":
    main()
