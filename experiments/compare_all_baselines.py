from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]

greedy = pd.read_csv(
    PROJECT_ROOT / "data" / "raw" / "greedy_50_trials.csv"
)

fixed = pd.read_csv(
    PROJECT_ROOT / "data" / "raw" / "fixed_pso_50_trials.csv"
)

dynamic = pd.read_csv(
    PROJECT_ROOT / "data" / "raw" / "dynamic_pso_50_trials.csv"
)


summary = pd.DataFrame(
    {
        "Algorithm": [
            "Greedy",
            "Fixed PSO",
            "Dynamic PSO",
        ],
        "Success Rate (%)": [
            greedy["success"].mean() * 100,
            fixed["success"].mean() * 100,
            dynamic["success"].mean() * 100,
        ],
        "Mean Iterations": [
            greedy["iterations"].mean(),
            fixed["iterations"].mean(),
            dynamic["iterations"].mean(),
        ],
        "Std Iterations": [
            greedy["iterations"].std(),
            fixed["iterations"].std(),
            dynamic["iterations"].std(),
        ],
        "Mean Distance": [
            greedy["total_distance"].mean(),
            fixed["total_distance"].mean(),
            dynamic["total_distance"].mean(),
        ],
        "Std Distance": [
            greedy["total_distance"].std(),
            fixed["total_distance"].std(),
            dynamic["total_distance"].std(),
        ],
        "Mean Unproductive Agents": [
            greedy["unproductive_agents"].mean(),
            fixed["unproductive_agents"].mean(),
            dynamic["unproductive_agents"].mean(),
        ],
    }
)


print("\n==============================")
print("ALL BASELINE COMPARISON")
print("==============================\n")

print(
    summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.2f}",
    )
)


table_dir = PROJECT_ROOT / "results" / "tables"
figure_dir = PROJECT_ROOT / "results" / "figures"

table_dir.mkdir(parents=True, exist_ok=True)
figure_dir.mkdir(parents=True, exist_ok=True)


summary_path = (
    table_dir
    / "baseline_summary.csv"
)

summary.to_csv(
    summary_path,
    index=False,
)


# -----------------------------
# Paired seed-by-seed dataset
# -----------------------------

paired = pd.DataFrame(
    {
        "seed": greedy["seed"],
        "greedy_iterations": greedy["iterations"],
        "fixed_iterations": fixed["iterations"],
        "dynamic_iterations": dynamic["iterations"],
        "greedy_distance": greedy["total_distance"],
        "fixed_distance": fixed["total_distance"],
        "dynamic_distance": dynamic["total_distance"],
        "greedy_unused": greedy["unproductive_agents"],
        "fixed_unused": fixed["unproductive_agents"],
        "dynamic_unused": dynamic["unproductive_agents"],
    }
)

paired_path = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "all_baselines_paired.csv"
)

paired.to_csv(
    paired_path,
    index=False,
)


print("\n==============================")
print("WIN COUNTS - ITERATIONS")
print("==============================")

greedy_wins = 0
fixed_wins = 0
dynamic_wins = 0
ties = 0

for _, row in paired.iterrows():
    values = {
        "greedy": row["greedy_iterations"],
        "fixed": row["fixed_iterations"],
        "dynamic": row["dynamic_iterations"],
    }

    best = min(values.values())

    winners = [
        name
        for name, value in values.items()
        if value == best
    ]

    if len(winners) > 1:
        ties += 1
    elif winners[0] == "greedy":
        greedy_wins += 1
    elif winners[0] == "fixed":
        fixed_wins += 1
    else:
        dynamic_wins += 1


print("Greedy unique wins:", greedy_wins)
print("Fixed PSO unique wins:", fixed_wins)
print("Dynamic PSO unique wins:", dynamic_wins)
print("Tied best seeds:", ties)


# -----------------------------
# Figure 1: mean iterations
# -----------------------------

plt.figure(figsize=(8, 5))

plt.bar(
    summary["Algorithm"],
    summary["Mean Iterations"],
)

plt.ylabel("Mean completion iterations")
plt.title("Baseline Comparison - Completion Time")

plt.tight_layout()

plt.savefig(
    figure_dir
    / "baseline_mean_iterations.png",
    dpi=300,
)

plt.close()


# -----------------------------
# Figure 2: mean distance
# -----------------------------

plt.figure(figsize=(8, 5))

plt.bar(
    summary["Algorithm"],
    summary["Mean Distance"],
)

plt.ylabel("Mean total travel distance")
plt.title("Baseline Comparison - Travel Distance")

plt.tight_layout()

plt.savefig(
    figure_dir
    / "baseline_mean_distance.png",
    dpi=300,
)

plt.close()


# -----------------------------
# Figure 3: unused agents
# -----------------------------

plt.figure(figsize=(8, 5))

plt.bar(
    summary["Algorithm"],
    summary["Mean Unproductive Agents"],
)

plt.ylabel("Mean unproductive agents")
plt.title("Baseline Comparison - Swarm Utilization")

plt.tight_layout()

plt.savefig(
    figure_dir
    / "baseline_unused_agents.png",
    dpi=300,
)

plt.close()


print("\nSaved summary table:")
print(summary_path)

print("\nSaved paired dataset:")
print(paired_path)

print("\nSaved comparison graphs in:")
print(figure_dir)
