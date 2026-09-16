from dataclasses import dataclass
import numpy as np


@dataclass
class Task:
    task_id: int
    position: np.ndarray
    active: bool = True


@dataclass
class Agent:
    agent_id: int
    position: np.ndarray
    velocity: np.ndarray

    active: bool = True
    target_id: int | None = None

    personal_best_position: np.ndarray | None = None
    personal_best_distance: float = float("inf")

    total_distance: float = 0.0
    tasks_completed: int = 0

    def __post_init__(self):
        if self.personal_best_position is None:
            self.personal_best_position = self.position.copy()
