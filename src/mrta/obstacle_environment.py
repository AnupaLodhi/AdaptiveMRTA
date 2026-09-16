import numpy as np

from .entities import Agent, Task
from .grid_map import GridMap


class ObstacleEnvironment:
    def __init__(
        self,
        config,
        obstacle_density=0.12,
        scenario=None,
    ):
        self.config = config

        if scenario is not None:

            self.grid_map = (
                scenario.create_grid_map()
            )

            self.agents = (
                scenario.create_agents()
            )

            self.tasks = (
                scenario.create_tasks()
            )

            self.free_component = (
                self.grid_map
                .largest_free_component()
            )

            return

        self.grid_map = GridMap(
            width=config.grid_width,
            height=config.grid_height,
            seed=config.seed,
            obstacle_density=obstacle_density,
        )

        self.position_rng = (
            np.random.default_rng(
                config.seed
            )
        )

        self.free_component = (
            self.grid_map
            .largest_free_component()
        )

        required_cells = (
            config.num_agents
            +
            config.num_tasks
        )

        if (
            len(self.free_component)
            <
            required_cells
        ):
            raise RuntimeError(
                "Not enough connected free cells "
                "for agents and tasks."
            )

        selected_indices = (
            self.position_rng.choice(
                len(self.free_component),
                size=required_cells,
                replace=False,
            )
        )

        selected_cells = [
            self.free_component[i]
            for i in selected_indices
        ]

        agent_cells = (
            selected_cells[
                :config.num_agents
            ]
        )

        task_cells = (
            selected_cells[
                config.num_agents:
            ]
        )

        self.agents = (
            self._create_agents(
                agent_cells
            )
        )

        self.tasks = (
            self._create_tasks(
                task_cells
            )
        )

    def _create_agents(
        self,
        cells,
    ):
        agents = []

        for agent_id, cell in enumerate(
            cells
        ):
            x, y = cell

            agents.append(
                Agent(
                    agent_id=agent_id,
                    position=np.array(
                        [x, y],
                        dtype=float,
                    ),
                    velocity=np.zeros(
                        2,
                        dtype=float,
                    ),
                )
            )

        return agents

    def _create_tasks(
        self,
        cells,
    ):
        tasks = []

        for task_id, cell in enumerate(
            cells
        ):
            x, y = cell

            tasks.append(
                Task(
                    task_id=task_id,
                    position=np.array(
                        [x, y],
                        dtype=float,
                    ),
                )
            )

        return tasks

    def get_active_agents(self):
        return [
            agent
            for agent in self.agents
            if agent.active
        ]

    def get_active_tasks(self):
        return [
            task
            for task in self.tasks
            if task.active
        ]

    def all_tasks_completed(self):
        return (
            len(
                self.get_active_tasks()
            )
            == 0
        )
