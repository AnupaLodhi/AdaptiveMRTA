import numpy as np

from .environment import Environment


class Simulator:
    def __init__(self, config, algorithm):
        self.config = config
        self.environment = Environment(config)
        self.algorithm = algorithm

        self.iteration = 0
        self.completed_tasks = 0

    def _resolve_completed_tasks(self):
        """
        Check whether any active agent is close enough
        to an active task to complete it.
        """

        active_tasks = self.environment.get_active_tasks()
        active_agents = self.environment.get_active_agents()

        for task in active_tasks:
            candidates = []

            for agent in active_agents:
                distance = np.linalg.norm(
                    agent.position - task.position
                )

                if distance <= self.config.completion_radius:
                    candidates.append((distance, agent))

            if candidates:
                # The closest agent receives credit.
                candidates.sort(key=lambda item: item[0])
                winning_agent = candidates[0][1]

                task.active = False
                winning_agent.tasks_completed += 1

                self.completed_tasks += 1

    def step(self):
        active_tasks = self.environment.get_active_tasks()

        if not active_tasks:
            return

        # Every active agent observes the same task state
        # at the beginning of the iteration.
        for agent in self.environment.get_active_agents():
            target = self.algorithm.select_target(
                agent,
                active_tasks
            )

            if target is None:
                continue

            agent.target_id = target.task_id

            self.algorithm.move_agent(
                agent,
                target
            )

        # Resolve completion after all agents move.
        self._resolve_completed_tasks()

        self.iteration += 1

    def run(self):
        while (
            self.iteration < self.config.max_iterations
            and not self.environment.all_tasks_completed()
        ):
            self.step()

        success = self.environment.all_tasks_completed()

        total_distance = sum(
            agent.total_distance
            for agent in self.environment.agents
        )

        return {
            "success": success,
            "iterations": self.iteration,
            "completed_tasks": self.completed_tasks,
            "total_tasks": self.config.num_tasks,
            "total_distance": total_distance,
        }
