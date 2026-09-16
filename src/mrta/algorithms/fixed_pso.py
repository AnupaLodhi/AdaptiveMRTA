import numpy as np


class FixedPSOAllocator:
    """
    Fixed-coefficient target-guided PSO baseline.

    Important:
    The target term attracts an agent toward its assigned task.
    It is not a classical global-best particle term.
    """

    def __init__(self, config, rng):
        self.config = config
        self.rng = rng

    def select_target(self, agent, active_tasks):
        if not active_tasks:
            return None

        distances = [
            np.linalg.norm(task.position - agent.position)
            for task in active_tasks
        ]

        nearest_index = int(np.argmin(distances))
        target = active_tasks[nearest_index]

        # Reset personal best when target changes
        if agent.target_id != target.task_id:
            agent.target_id = target.task_id
            agent.personal_best_position = agent.position.copy()
            agent.personal_best_distance = np.linalg.norm(
                agent.position - target.position
            )

        return target

    def move_agent(self, agent, target):
        current_distance = np.linalg.norm(
            agent.position - target.position
        )

        # Update personal best for the current target
        if current_distance < agent.personal_best_distance:
            agent.personal_best_distance = current_distance
            agent.personal_best_position = agent.position.copy()

        r1 = self.rng.random(2)
        r2 = self.rng.random(2)

        inertia = (
            self.config.inertia_weight
            * agent.velocity
        )

        cognitive = (
            self.config.cognitive_coefficient
            * r1
            * (
                agent.personal_best_position
                - agent.position
            )
        )

        target_attraction = (
            self.config.target_coefficient
            * r2
            * (
                target.position
                - agent.position
            )
        )

        new_velocity = (
            inertia
            + cognitive
            + target_attraction
        )

        speed = np.linalg.norm(new_velocity)

        if speed > self.config.max_velocity:
            new_velocity = (
                new_velocity / speed
                * self.config.max_velocity
            )

        old_position = agent.position.copy()

        agent.velocity = new_velocity
        agent.position = (
            agent.position
            + agent.velocity
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
