import numpy as np


class GreedyAllocator:
    """
    Simple nearest-task baseline.

    Each active agent selects the nearest active task
    and moves directly toward it with a fixed maximum speed.
    """

    def __init__(self, config):
        self.config = config

    def select_target(self, agent, active_tasks):
        if not active_tasks:
            return None

        distances = [
            np.linalg.norm(task.position - agent.position)
            for task in active_tasks
        ]

        nearest_index = int(np.argmin(distances))
        return active_tasks[nearest_index]

    def move_agent(self, agent, target):
        direction = target.position - agent.position
        distance_to_target = np.linalg.norm(direction)

        if distance_to_target == 0:
            return

        direction = direction / distance_to_target

        step_size = min(
            self.config.max_velocity,
            distance_to_target
        )

        old_position = agent.position.copy()

        agent.position = (
            agent.position
            + direction * step_size
        )

        agent.position[0] = np.clip(
            agent.position[0],
            0,
            self.config.grid_width
        )

        agent.position[1] = np.clip(
            agent.position[1],
            0,
            self.config.grid_height
        )

        travelled = np.linalg.norm(
            agent.position - old_position
        )

        agent.total_distance += travelled
