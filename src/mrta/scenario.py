from dataclasses import dataclass

import numpy as np

from .entities import Agent, Task
from .grid_map import GridMap


@dataclass
class Scenario:
    """
    Immutable-style initial condition for one MRTA trial.

    Stores the exact:
      - obstacle grid
      - robot starting cells
      - task starting cells

    Every algorithm receives a fresh clone of this state.
    """

    seed: int
    width: int
    height: int
    obstacle_density: float
    obstacle_grid: np.ndarray
    agent_positions: list
    task_positions: list

    @classmethod
    def generate(
        cls,
        config,
        obstacle_density,
    ):
        grid_map = GridMap(
            width=config.grid_width,
            height=config.grid_height,
            seed=config.seed,
            obstacle_density=obstacle_density,
        )

        free_component = (
            grid_map.largest_free_component()
        )

        required_cells = (
            config.num_agents
            + config.num_tasks
        )

        if len(free_component) < required_cells:
            raise RuntimeError(
                "Not enough connected free cells "
                "for agents and tasks."
            )

        position_rng = np.random.default_rng(
            config.seed
        )

        selected_indices = position_rng.choice(
            len(free_component),
            size=required_cells,
            replace=False,
        )

        selected_cells = [
            free_component[i]
            for i in selected_indices
        ]

        agent_cells = selected_cells[
            :config.num_agents
        ]

        task_cells = selected_cells[
            config.num_agents:
        ]

        return cls(
            seed=config.seed,
            width=config.grid_width,
            height=config.grid_height,
            obstacle_density=obstacle_density,
            obstacle_grid=grid_map.obstacles.copy(),
            agent_positions=[
                tuple(cell)
                for cell in agent_cells
            ],
            task_positions=[
                tuple(cell)
                for cell in task_cells
            ],
        )

    def create_grid_map(self):
        """
        Create a fresh GridMap containing exactly the
        stored obstacle configuration.
        """

        grid_map = GridMap(
            width=self.width,
            height=self.height,
            seed=self.seed,
            obstacle_density=0.0,
        )

        grid_map.obstacle_density = (
            self.obstacle_density
        )

        grid_map.obstacles = (
            self.obstacle_grid.copy()
        )

        return grid_map

    def create_agents(self):

        agents = []

        for agent_id, cell in enumerate(
            self.agent_positions
        ):
            agents.append(
                Agent(
                    agent_id=agent_id,
                    position=np.array(
                        cell,
                        dtype=float,
                    ),
                    velocity=np.zeros(
                        2,
                        dtype=float,
                    ),
                )
            )

        return agents

    def create_tasks(self):

        tasks = []

        for task_id, cell in enumerate(
            self.task_positions
        ):
            tasks.append(
                Task(
                    task_id=task_id,
                    position=np.array(
                        cell,
                        dtype=float,
                    ),
                )
            )

        return tasks

    def fingerprint(self):
        """
        Compact deterministic fingerprint used for
        experimental sanity checks.
        """

        obstacle_bytes = (
            self.obstacle_grid
            .astype(np.uint8)
            .tobytes()
        )

        import hashlib

        h = hashlib.sha256()

        h.update(obstacle_bytes)

        h.update(
            repr(
                self.agent_positions
            ).encode()
        )

        h.update(
            repr(
                self.task_positions
            ).encode()
        )

        return h.hexdigest()
