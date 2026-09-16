import numpy as np

from .config import SimulationConfig
from .entities import Agent, Task


class Environment:
    def __init__(self, config: SimulationConfig):
        self.config = config

        # Dedicated RNG only for scenario generation.
        self.rng = np.random.default_rng(config.seed)

        self.agents = self._create_agents()
        self.tasks = self._create_tasks()

    def _random_position(self) -> np.ndarray:
        x = self.rng.uniform(
            0,
            self.config.grid_width
        )

        y = self.rng.uniform(
            0,
            self.config.grid_height
        )

        return np.array(
            [x, y],
            dtype=float
        )

    def _create_agents(self) -> list[Agent]:
        agents = []

        for agent_id in range(
            self.config.num_agents
        ):
            agents.append(
                Agent(
                    agent_id=agent_id,
                    position=self._random_position(),
                    velocity=np.zeros(
                        2,
                        dtype=float
                    ),
                )
            )

        return agents

    def _create_tasks(self) -> list[Task]:
        tasks = []

        for task_id in range(
            self.config.num_tasks
        ):
            tasks.append(
                Task(
                    task_id=task_id,
                    position=self._random_position(),
                )
            )

        return tasks

    def get_active_agents(self) -> list[Agent]:
        return [
            agent
            for agent in self.agents
            if agent.active
        ]

    def get_active_tasks(self) -> list[Task]:
        return [
            task
            for task in self.tasks
            if task.active
        ]

    def all_tasks_completed(self) -> bool:
        return (
            len(self.get_active_tasks()) == 0
        )
