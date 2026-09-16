from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"

sys.path.insert(0, str(SRC_PATH))


from mrta.config import SimulationConfig
from mrta.environment import Environment


def main():
    config = SimulationConfig(
        num_agents=5,
        num_tasks=4,
        seed=42,
    )

    env = Environment(config)

    print("\n=== AGENTS ===")

    for agent in env.agents:
        print(
            f"Agent {agent.agent_id}: "
            f"x={agent.position[0]:.2f}, "
            f"y={agent.position[1]:.2f}"
        )

    print("\n=== TASKS ===")

    for task in env.tasks:
        print(
            f"Task {task.task_id}: "
            f"x={task.position[0]:.2f}, "
            f"y={task.position[1]:.2f}"
        )

    print("\n=== ENVIRONMENT STATUS ===")
    print("Active agents:", len(env.get_active_agents()))
    print("Active tasks:", len(env.get_active_tasks()))
    print("All tasks completed:", env.all_tasks_completed())


if __name__ == "__main__":
    main()
