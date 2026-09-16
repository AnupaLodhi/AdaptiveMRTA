from dataclasses import dataclass


@dataclass
class SimulationConfig:
    # Environment
    grid_width: int = 52
    grid_height: int = 30

    # Population
    num_agents: int = 15
    num_tasks: int = 10

    # Experiment
    max_iterations: int = 2000
    seed: int = 42

    # Motion
    max_velocity: float = 2.8
    completion_radius: float = 1.3

    # PSO parameters
    inertia_weight: float = 0.5
    cognitive_coefficient: float = 1.5
    target_coefficient: float = 1.8
