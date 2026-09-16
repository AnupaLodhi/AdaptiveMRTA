from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"

sys.path.insert(0, str(SRC_PATH))


from mrta.config import SimulationConfig
from mrta.algorithms.greedy import GreedyAllocator
from mrta.simulator import Simulator


def main():
    config = SimulationConfig(
        num_agents=15,
        num_tasks=10,
        seed=42,
        max_iterations=2000,
    )

    algorithm = GreedyAllocator(config)

    simulator = Simulator(
        config=config,
        algorithm=algorithm,
    )

    result = simulator.run()

    print("\n==============================")
    print("GREEDY BASELINE RESULT")
    print("==============================")

    print("Success:", result["success"])
    print("Iterations:", result["iterations"])
    print(
        "Tasks completed:",
        f'{result["completed_tasks"]}/{result["total_tasks"]}'
    )
    print(
        "Total distance:",
        f'{result["total_distance"]:.2f}'
    )

    print("\nPER-AGENT RESULTS")
    print("------------------------------")

    for agent in simulator.environment.agents:
        print(
            f"Agent {agent.agent_id:02d} | "
            f"Tasks: {agent.tasks_completed} | "
            f"Distance: {agent.total_distance:.2f}"
        )


if __name__ == "__main__":
    main()
