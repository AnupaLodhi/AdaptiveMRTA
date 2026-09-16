from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]

greedy_path = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "greedy_50_trials.csv"
)

fixed_path = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "fixed_pso_50_trials.csv"
)


greedy = pd.read_csv(greedy_path)
fixed = pd.read_csv(fixed_path)


merged = greedy.merge(
    fixed,
    on="seed",
    suffixes=("_greedy", "_fixed"),
)


merged["iteration_difference"] = (
    merged["iterations_fixed"]
    - merged["iterations_greedy"]
)

merged["distance_difference"] = (
    merged["total_distance_fixed"]
    - merged["total_distance_greedy"]
)

merged["unused_difference"] = (
    merged["unproductive_agents_fixed"]
    - merged["unproductive_agents_greedy"]
)


print("\n==============================")
print("PAIRED COMPARISON")
print("==============================")

print("\nMEAN RESULTS")
print("------------------------------")

print(
    f"Greedy iterations: "
    f"{merged['iterations_greedy'].mean():.2f}"
)

print(
    f"Fixed PSO iterations: "
    f"{merged['iterations_fixed'].mean():.2f}"
)

print(
    f"Greedy distance: "
    f"{merged['total_distance_greedy'].mean():.2f}"
)

print(
    f"Fixed PSO distance: "
    f"{merged['total_distance_fixed'].mean():.2f}"
)

print(
    f"Greedy unused agents: "
    f"{merged['unproductive_agents_greedy'].mean():.2f}"
)

print(
    f"Fixed PSO unused agents: "
    f"{merged['unproductive_agents_fixed'].mean():.2f}"
)


print("\nPAIRED DIFFERENCES")
print("------------------------------")

print(
    "Mean iteration difference "
    "(Fixed - Greedy):",
    f"{merged['iteration_difference'].mean():.2f}"
)

print(
    "Mean distance difference "
    "(Fixed - Greedy):",
    f"{merged['distance_difference'].mean():.2f}"
)

print(
    "Mean unused-agent difference "
    "(Fixed - Greedy):",
    f"{merged['unused_difference'].mean():.2f}"
)


greedy_iteration_wins = (
    merged["iterations_greedy"]
    < merged["iterations_fixed"]
).sum()

fixed_iteration_wins = (
    merged["iterations_fixed"]
    < merged["iterations_greedy"]
).sum()

iteration_ties = (
    merged["iterations_fixed"]
    == merged["iterations_greedy"]
).sum()


greedy_distance_wins = (
    merged["total_distance_greedy"]
    < merged["total_distance_fixed"]
).sum()

fixed_distance_wins = (
    merged["total_distance_fixed"]
    < merged["total_distance_greedy"]
).sum()


print("\nWIN COUNTS")
print("------------------------------")

print(
    f"Iterations -> "
    f"Greedy: {greedy_iteration_wins}, "
    f"Fixed PSO: {fixed_iteration_wins}, "
    f"Ties: {iteration_ties}"
)

print(
    f"Distance -> "
    f"Greedy: {greedy_distance_wins}, "
    f"Fixed PSO: {fixed_distance_wins}"
)


output_path = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "greedy_vs_fixed_paired.csv"
)

output_path.parent.mkdir(
    parents=True,
    exist_ok=True
)

merged.to_csv(
    output_path,
    index=False
)


figure_dir = (
    PROJECT_ROOT
    / "results"
    / "figures"
)

figure_dir.mkdir(
    parents=True,
    exist_ok=True
)


plt.figure(figsize=(10, 5))

plt.plot(
    merged["seed"],
    merged["iterations_greedy"],
    marker="o",
    label="Greedy",
)

plt.plot(
    merged["seed"],
    merged["iterations_fixed"],
    marker="o",
    label="Fixed PSO",
)

plt.xlabel("Seed")
plt.ylabel("Completion iterations")
plt.title("Greedy vs Fixed PSO - Paired Completion Time")
plt.legend()

plt.tight_layout()

plt.savefig(
    figure_dir / "greedy_vs_fixed_iterations.png",
    dpi=300,
)

plt.close()


plt.figure(figsize=(10, 5))

plt.plot(
    merged["seed"],
    merged["total_distance_greedy"],
    marker="o",
    label="Greedy",
)

plt.plot(
    merged["seed"],
    merged["total_distance_fixed"],
    marker="o",
    label="Fixed PSO",
)

plt.xlabel("Seed")
plt.ylabel("Total distance")
plt.title("Greedy vs Fixed PSO - Paired Travel Distance")
plt.legend()

plt.tight_layout()

plt.savefig(
    figure_dir / "greedy_vs_fixed_distance.png",
    dpi=300,
)

plt.close()


print("\nSaved paired CSV:")
print(output_path)

print("\nSaved graphs:")
print(
    figure_dir
    / "greedy_vs_fixed_iterations.png"
)

print(
    figure_dir
    / "greedy_vs_fixed_distance.png"
)
