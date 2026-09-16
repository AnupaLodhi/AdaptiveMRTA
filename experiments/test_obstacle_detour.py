from pathlib import Path
import sys

import matplotlib.pyplot as plt


PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

SRC_PATH = PROJECT_ROOT / "src"

sys.path.insert(
    0,
    str(SRC_PATH),
)


from mrta.config import SimulationConfig
from mrta.obstacle_environment import ObstacleEnvironment
from mrta.path_planning import astar


def manhattan(a, b):
    return (
        abs(a[0] - b[0])
        +
        abs(a[1] - b[1])
    )


def main():

    config = SimulationConfig(
        num_agents=15,
        num_tasks=10,
        seed=42,
    )

    environment = ObstacleEnvironment(
        config=config,
        obstacle_density=0.12,
    )

    best_case = None
    best_extra_distance = -1

    # Search every agent-task pair
    # for a route where obstacles cause a detour.
    for agent in environment.agents:

        start = (
            int(agent.position[0]),
            int(agent.position[1]),
        )

        for task in environment.tasks:

            goal = (
                int(task.position[0]),
                int(task.position[1]),
            )

            path = astar(
                environment.grid_map,
                start,
                goal,
            )

            if path is None:
                continue

            path_length = (
                len(path) - 1
            )

            direct_distance = (
                manhattan(
                    start,
                    goal,
                )
            )

            extra_distance = (
                path_length
                - direct_distance
            )

            if (
                extra_distance
                > best_extra_distance
            ):

                best_extra_distance = (
                    extra_distance
                )

                best_case = {
                    "agent": agent,
                    "task": task,
                    "start": start,
                    "goal": goal,
                    "path": path,
                    "path_length": path_length,
                    "direct_distance": direct_distance,
                    "extra_distance": extra_distance,
                }

    print("\n==============================")
    print("OBSTACLE DETOUR TEST")
    print("==============================")

    if best_case is None:

        print(
            "No valid path found."
        )

        return

    print(
        "Agent:",
        best_case["agent"].agent_id
    )

    print(
        "Task:",
        best_case["task"].task_id
    )

    print(
        "Start:",
        best_case["start"]
    )

    print(
        "Goal:",
        best_case["goal"]
    )

    print(
        "Direct Manhattan distance:",
        best_case[
            "direct_distance"
        ]
    )

    print(
        "A* path length:",
        best_case[
            "path_length"
        ]
    )

    print(
        "Obstacle detour:",
        best_case[
            "extra_distance"
        ]
    )

    if (
        best_case["extra_distance"]
        > 0
    ):
        print(
            "\nPASS: Obstacles forced a real detour."
        )

    else:
        print(
            "\nWARNING: No tested pair required a detour."
        )

    # -----------------------------
    # VISUALIZE BEST DETOUR
    # -----------------------------

    plt.figure(
        figsize=(12, 7)
    )

    plt.imshow(
        environment.grid_map.obstacles,
        origin="lower",
    )

    path_x = [
        cell[0]
        for cell
        in best_case["path"]
    ]

    path_y = [
        cell[1]
        for cell
        in best_case["path"]
    ]

    plt.plot(
        path_x,
        path_y,
        linewidth=2,
        label="A* collision-free path",
    )

    start = best_case["start"]
    goal = best_case["goal"]

    plt.scatter(
        [start[0]],
        [start[1]],
        marker="o",
        s=100,
        label="Selected agent",
    )

    plt.scatter(
        [goal[0]],
        [goal[1]],
        marker="x",
        s=120,
        label="Selected task",
    )

    plt.xlabel("X")
    plt.ylabel("Y")

    plt.title(
        "A* Obstacle Detour Verification"
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
        / "obstacle_detour_test.png"
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
