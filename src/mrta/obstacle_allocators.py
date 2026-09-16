import numpy as np

from .path_planning import astar


class EuclideanGreedyAllocator:
    """
    Chooses the active task with minimum
    straight-line Euclidean distance.
    """

    name = "euclidean_greedy"

    def select_target(
        self,
        agent,
        active_tasks,
        grid_map,
    ):
        if not active_tasks:
            return None

        distances = [
            np.linalg.norm(
                task.position
                - agent.position
            )
            for task in active_tasks
        ]

        index = int(
            np.argmin(distances)
        )

        return active_tasks[index]


class ManhattanGreedyAllocator:
    """
    Grid-consistent nearest-task baseline.

    Chooses the task with minimum Manhattan
    distance while ignoring obstacles.
    """

    name = "manhattan_greedy"

    def select_target(
        self,
        agent,
        active_tasks,
        grid_map,
    ):
        if not active_tasks:
            return None

        ax = int(agent.position[0])
        ay = int(agent.position[1])

        best_task = None
        best_cost = float("inf")

        for task in active_tasks:

            tx = int(task.position[0])
            ty = int(task.position[1])

            cost = (
                abs(ax - tx)
                +
                abs(ay - ty)
            )

            if cost < best_cost:
                best_cost = cost
                best_task = task

        return best_task


class PathCostGreedyAllocator:
    """
    Obstacle-aware nearest-task baseline.

    Chooses the task with minimum true A*
    path cost through the occupancy grid.
    """

    name = "path_cost_greedy"

    def select_target(
        self,
        agent,
        active_tasks,
        grid_map,
    ):
        if not active_tasks:
            return None

        start = (
            int(agent.position[0]),
            int(agent.position[1]),
        )

        best_task = None
        best_cost = float("inf")

        for task in active_tasks:

            goal = (
                int(task.position[0]),
                int(task.position[1]),
            )

            path = astar(
                grid_map,
                start,
                goal,
            )

            if path is None:
                continue

            cost = len(path) - 1

            if cost < best_cost:
                best_cost = cost
                best_task = task

        return best_task
