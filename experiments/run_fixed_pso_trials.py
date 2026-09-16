from pathlib import Path
import sys

import pandas as pd
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"

sys.path.insert(0, str(SRC_PATH))


from mrta.config import SimulationConfig
from mrta.algorithms.fixed_pso import FixedPSOAllocator
from mrta.simulator import Simulator


NUM_TRIALS = 50


def run_trial(seed):
    config = SimulationConfig(
        num_agents=15,
        num_tasks=10,
        seed=seed,
        max_iterations=2000,
    )

    simulator = Simulator(
        config=config,
        algorithm=None,
    )

    algorithm = FixedPSOAllocator(
        config=config,
        rng=simulator.environment.rng,
    )

    simulator.algorithm = algorithm

    result = simulator.run()

    productive_agents = sum(
        1
        for agent in simulator.environment.agents
        if agent.tasks_completed > 0
    )

    unproductive_agents = (
        config.num_agents - productive_agents
    )

    return {
        "algorithm": "fixed_pso",
        "seed": seed,
        "num_agents": config.num_agents,
        "num_tasks": config.num_tasks,
        "grid_width": config.grid_width,
        "grid_height": config.grid_height,
        "success": result["success"],
        "iterations": result["iterations"],
        "tasks_completed": result["completed_tasks"],
        "total_distance": result["total_distance"],
        "productive_agents": productive_agents,
        "unproductive_agents": unproductive_agents,
    }


def main():
    rows = []

    print("\n==============================")
    print("RUNNING 50 FIXED PSO TRIALS")
    print("==============================\n")

    for seed in range(NUM_TRIALS):
        result = run_trial(seed)
        rows.append(result)

        print(
            f"Seed {seed:02d} | "
            f"Success={result['success']} | "
            f"Iterations={result['iterations']:4d} | "
            f"Distance={result['total_distance']:.2f} | "
            f"Unused agents={result['unproductive_agents']}"
        )

    df = pd.DataFrame(rows)

    raw_dir = PROJECT_ROOT / "data" / "raw"
    figure_dir = PROJECT_ROOT / "results" / "figures"

    raw_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    csv_path = raw_dir / "fixed_pso_50_trials.csv"
    df.to_csv(csv_path, index=False)

    print("\n==============================")
    print("SUMMARY")
    print("==============================")

    print(f"Trials: {len(df)}")

    print(
        "Success rate:",
        f"{df['success'].mean() * 100:.1f}%"
    )

    print(
        "Mean iterations:",
        f"{df['iterations'].mean():.2f}"
    )

    print(
        "Std iterations:",
        f"{df['iterations'].std():.2f}"
    )

    print(
        "Min iterations:",
        df["iterations"].min()
    )

    print(
        "Max iterations:",
        df["iterations"].max()
    )

    print(
        "Mean total distance:",
        f"{df['total_distance'].mean():.2f}"
    )

    print(
        "Std total distance:",
        f"{df['total_distance'].std():.2f}"
    )

    print(
        "Mean unproductive agents:",
        f"{df['unproductive_agents'].mean():.2f}"
    )

    print("\nCSV saved to:")
    print(csv_path)

    plt.figure(figsize=(10, 5))

    plt.plot(
        df["seed"],
        df["iterations"],
        marker="o",
    )

    plt.xlabel("Random seed")
    plt.ylabel("Completion iterations")
    plt.title("Fixed PSO - Completion Time Across 50 Trials")

    plt.tight_layout()

    iteration_plot = (
        figure_dir
        / "fixed_pso_iterations_50_trials.png"
    )

    plt.savefig(
        iteration_plot,
        dpi=300,
    )

    plt.close()

    plt.figure(figsize=(10, 5))

    plt.plot(
        df["seed"],
        df["total_distance"],
        marker="o",
    )

    plt.xlabel("Random seed")
    plt.ylabel("Total distance travelled")
    plt.title("Fixed PSO - Travel Distance Across 50 Trials")

    plt.tight_layout()

    distance_plot = (
        figure_dir
        / "fixed_pso_distance_50_trials.png"
    )

    plt.savefig(
        distance_plot,
        dpi=300,
    )

    plt.close()

    print("\nGraphs saved:")
    print(iteration_plot)
    print(distance_plot)


if __name__ == "__main__":
    main()
