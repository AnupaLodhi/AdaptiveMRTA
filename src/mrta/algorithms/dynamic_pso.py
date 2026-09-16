import numpy as np


class DynamicPSOAllocator:
    """
    Task-density adaptive PSO baseline.

    Adaptation follows the previous project formulation:

        rho(t) = active_tasks / initial_tasks

        c1(t) = 1.5 + 0.5 * (1 - rho)

        c2(t) = 1.8 - 0.5 * (1 - rho)

    Here c2 is treated as target attraction, not classical
    global-best social attraction.
    """

    def __init__(self, config, rng):
        self.config = config
        self.rng = rng

        self.current_c1 = config.cognitive_coefficient
        self.current_c2 = config.target_coefficient

    def update_coefficients(self, active_task_count):
        rho = (
            active_task_count
            / self.config.num_tasks
        )

        self.current_c1 = (
            1.5
            + 0.5 * (1.0 - rho)
        )

        self.current_c2 = (
            1.8
            - 0.5 * (1.0 - rho)
        )

    def select_target(self, agent, active_tasks):
        if not active_tasks:
            return None

        distances = [
            np.linalg.norm(
                task.position - agent.position
            )
            for task in active_tasks
        ]

        nearest_index = int(
            np.argmin(distances)
        )

        target = active_tasks[
            nearest_index
        ]

        if agent.target_id != target.task_id:
            agent.target_id = target.task_id

            agent.personal_best_position = (
                agent.position.copy()
            )

            agent.personal_best_distance = (
                np.linalg.norm(
                    agent.position
                    - target.position
                )
            )

        return target

    def move_agent(self, agent, target):
        current_distance = np.linalg.norm(
            agent.position
            - target.position
        )

        if (
            current_distance
            < agent.personal_best_distance
        ):
            agent.personal_best_distance = (
                current_distance
            )

            agent.personal_best_position = (
                agent.position.copy()
            )

        r1 = self.rng.random(2)
        r2 = self.rng.random(2)

        inertia = (
            self.config.inertia_weight
            * agent.velocity
        )

        cognitive = (
            self.current_c1
            * r1
            * (
                agent.personal_best_position
                - agent.position
            )
        )

        target_attraction = (
            self.current_c2
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

        speed = np.linalg.norm(
            new_velocity
        )

        if speed > self.config.max_velocity:
            new_velocity = (
                new_velocity
                / speed
                * self.config.max_velocity
            )

        old_position = (
            agent.position.copy()
        )

        agent.velocity = new_velocity

        agent.position = (
            agent.position
            + agent.velocity
        )

        agent.position[0] = np.clip(
            agent.position[0],
            0,
            self.config.grid_width,
        )

        agent.position[1] = np.clip(
            agent.position[1],
            0,
            self.config.grid_height,
        )

        travelled = np.linalg.norm(
            agent.position
            - old_position
        )

        agent.total_distance += travelled
