from pathlib import Path
import sys

import matplotlib.pyplot as plt


PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

SRC_PATH = (
    PROJECT_ROOT
    / "src"
)

sys.path.insert(
    0,
    str(SRC_PATH),
)


from mrta.config import SimulationConfig
from mrta.obstacle_environment import ObstacleEnvironment
from mrta.path_planning import astar


def main():

    config = SimulationConfig(
        num_agents=15,
        num_tasks=10,
        seed=42,
    )

    environment = (
        ObstacleEnvironment(
            config=config,
            obstacle_density=0.12,
        )
    )

    agent = (
        environment.agents[0]
    )

    task = (
        environment.tasks[0]
    )

    start = (
        int(agent.position[0]),
        int(agent.position[1]),
    )

    goal = (
        int(task.position[0]),
        int(task.position[1]),
    )

    path = astar(
        environment.grid_map,
        start,
        goal,
    )

    print("\n==============================")
    print("OBSTACLE ENVIRONMENT TEST")
    print("==============================")

    print(
        "Grid:",
        config.grid_width,
        "x",
        config.grid_height,
    )

    print(
        "Obstacle density:",
        "12%"
    )

    print(
        "Connected free cells:",
        len(
            environment.free_component
        )
    )

    print(
        "Agents:",
        len(environment.agents)
    )

    print(
        "Tasks:",
        len(environment.tasks)
    )

    print(
        "Agent 0 start:",
        start
    )

    print(
        "Task 0 goal:",
        goal
    )

    if path is None:

        print(
            "A* path found: False"
        )

        return

    print(
        "A* path found: True"
    )

    print(
        "Path length:",
        len(path) - 1
    )

    print(
        "Direct Manhattan distance:",
        abs(start[0] - goal[0])
        +
        abs(start[1] - goal[1])
    )

    # -------------------------
    # VISUALIZATION
    # -------------------------

    plt.figure(
        figsize=(12, 7)
    )

    plt.imshow(
        environment.grid_map.obstacles,
        origin="lower",
    )

    path_x = [
        cell[0]
        for cell in path
    ]

    path_y = [
        cell[1]
        for cell in path
    ]

    plt.plot(
        path_x,
        path_y,
        linewidth=2,
        label="A* path",
    )

    plt.scatter(
        [
            agent.position[0]
            for agent
            in environment.agents
        ],
        [
            agent.position[1]
            for agent
            in environment.agents
        ],
        marker="o",
        label="Agents",
    )

    plt.scatter(
        [
            task.position[0]
            for task
            in environment.tasks
        ],
        [
            task.position[1]
            for task
            in environment.tasks
        ],
        marker="x",
        label="Tasks",
    )

    plt.xlabel("X")
    plt.ylabel("Y")

    plt.title(
        "Obstacle-Aware MRTA Environment"
    )

    plt.legend()

    plt.tight_layout()

    figure_dir = (
        PROJECT_ROOT
        / "results"
        / "figures"
    )

    figure_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        figure_dir
        / "obstacle_environment_test.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
    )

    plt.close()

    print(
        "\nFigure saved:"
    )

    print(
        output_path
    )


if __name__ == "__main__":
    main()
