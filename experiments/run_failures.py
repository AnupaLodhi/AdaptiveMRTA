from pathlib import Path
import sys

import numpy as np
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

NUM_AGENTS = 15
NUM_TASKS = 20

FAILURE_ITERATION = 3

FAILURE_RATES = [
    0.0,
    0.2,
    0.4,
    0.6,
]


class FailureSimulator(Simulator):
    def __init__(
        self,
        config,
        algorithm,
        algorithm_name,
        failed_agent_ids,
    ):
        super().__init__(
            config=config,
            algorithm=algorithm,
        )

        self.algorithm_name = algorithm_name

        self.failed_agent_ids = set(
            failed_agent_ids
        )

        self.failure_applied = False

    def apply_failures(self):
        if self.failure_applied:
            return

        for agent in self.environment.agents:
            if (
                agent.agent_id
                in self.failed_agent_ids
            ):
                agent.active = False
                agent.velocity[:] = 0.0
                agent.target_id = None

        self.failure_applied = True

    def step(self):
        if (
            self.iteration
            == FAILURE_ITERATION
        ):
            self.apply_failures()

        active_tasks = (
            self.environment.get_active_tasks()
        )

        if not active_tasks:
            return

        active_agents = (
            self.environment.get_active_agents()
        )

        # If every robot has failed,
        # the simulation cannot progress.
        if not active_agents:
            self.iteration = (
                self.config.max_iterations
            )
            return

        if (
            self.algorithm_name
            == "dynamic_pso"
        ):
            self.algorithm.update_coefficients(
                len(active_tasks)
            )

        for agent in active_agents:
            target = (
                self.algorithm.select_target(
                    agent,
                    active_tasks,
                )
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


def choose_failed_agents(
    seed,
    failure_rate,
):
    failure_count = int(
        round(
            NUM_AGENTS
            * failure_rate
        )
    )

    if failure_count == 0:
        return []

    failure_rng = np.random.default_rng(
        100000 + seed
    )

    failed = failure_rng.choice(
        NUM_AGENTS,
        size=failure_count,
        replace=False,
    )

    return sorted(
        int(x)
        for x in failed
    )


def run_trial(
    algorithm_name,
    seed,
    failure_rate,
):
    config = SimulationConfig(
        num_agents=NUM_AGENTS,
        num_tasks=NUM_TASKS,
        seed=seed,
        max_iterations=2000,
    )

    failed_ids = choose_failed_agents(
        seed,
        failure_rate,
    )

    simulator = FailureSimulator(
        config=config,
        algorithm=None,
        algorithm_name=algorithm_name,
        failed_agent_ids=failed_ids,
    )

    # Independent algorithm RNG.
    algorithm_rng = (
        np.random.default_rng(
            200000 + seed
        )
    )

    if algorithm_name == "greedy":
        algorithm = GreedyAllocator(
            config
        )

    elif algorithm_name == "fixed_pso":
        algorithm = FixedPSOAllocator(
            config=config,
            rng=algorithm_rng,
        )

    elif algorithm_name == "dynamic_pso":
        algorithm = DynamicPSOAllocator(
            config=config,
            rng=algorithm_rng,
        )

    else:
        raise ValueError(
            algorithm_name
        )

    simulator.algorithm = algorithm

    result = simulator.run()

    productive_agents = sum(
        1
        for agent
        in simulator.environment.agents
        if agent.tasks_completed > 0
    )

    return {
        "algorithm": algorithm_name,
        "seed": seed,
        "failure_rate": failure_rate,
        "failed_agents": len(failed_ids),
        "success": result["success"],
        "iterations": result["iterations"],
        "tasks_completed": result["completed_tasks"],
        "total_distance": result["total_distance"],
        "productive_agents": productive_agents,
    }


def main():
    algorithms = [
        "greedy",
        "fixed_pso",
        "dynamic_pso",
    ]

    rows = []

    print("\n==============================")
    print("ROBOT FAILURE EXPERIMENT")
    print("==============================")

    for failure_rate in FAILURE_RATES:

        print(
            f"\nFailure rate: "
            f"{failure_rate * 100:.0f}%"
        )

        print("-" * 60)

        for algorithm_name in algorithms:

            temp_rows = []

            for seed in range(NUM_TRIALS):

                result = run_trial(
                    algorithm_name,
                    seed,
                    failure_rate,
                )

                rows.append(result)
                temp_rows.append(result)

            temp = pd.DataFrame(
                temp_rows
            )

            successful = temp[
                temp["success"]
            ]

            success_rate = (
                temp["success"].mean()
                * 100
            )

            if len(successful) > 0:
                mean_iterations = (
                    successful[
                        "iterations"
                    ].mean()
                )

                mean_distance = (
                    successful[
                        "total_distance"
                    ].mean()
                )
            else:
                mean_iterations = float("nan")
                mean_distance = float("nan")

            print(
                f"{algorithm_name:12s} | "
                f"Success={success_rate:6.1f}% | "
                f"Successful-run iterations="
                f"{mean_iterations:7.2f} | "
                f"Successful-run distance="
                f"{mean_distance:9.2f}"
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
        / "failure_trials.csv"
    )

    df.to_csv(
        raw_path,
        index=False,
    )

    summary = (
        df.groupby(
            [
                "algorithm",
                "failure_rate",
            ]
        )
        .agg(
            success_rate=(
                "success",
                "mean",
            ),
            mean_tasks_completed=(
                "tasks_completed",
                "mean",
            ),
            mean_iterations_all=(
                "iterations",
                "mean",
            ),
            mean_distance_all=(
                "total_distance",
                "mean",
            ),
        )
        .reset_index()
    )

    summary[
        "success_rate"
    ] *= 100

    summary_path = (
        table_dir
        / "failure_summary.csv"
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
