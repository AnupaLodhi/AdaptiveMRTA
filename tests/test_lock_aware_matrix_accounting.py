from mrta.config import SimulationConfig
from mrta.scenario import Scenario
from mrta.obstacle_simulator import ObstacleSimulator
from mrta.coordinated_allocators import LockAwareHungarianAllocator


class CountingLockAware(LockAwareHungarianAllocator):
    def __init__(self):
        super().__init__()
        self.expected_pairs = 0
        self.matrix_calls = 0

    def _build_cost_matrix(self, agents, tasks, grid_map):
        self.expected_pairs += len(agents) * len(tasks)
        self.matrix_calls += 1
        return super()._build_cost_matrix(
            agents, tasks, grid_map
        )


def test_matrix_accounting():
    config = SimulationConfig(
        num_agents=5,
        num_tasks=7,
        seed=42,
        max_iterations=100,
    )

    scenario = Scenario.generate(
        config=config,
        obstacle_density=0.10,
    )

    allocator = CountingLockAware()

    simulator = ObstacleSimulator(
        config=config,
        allocator=allocator,
        obstacle_density=0.10,
        scenario=scenario,
    )

    result = simulator.run()

    assert result["success"]
    assert allocator.matrix_calls > 0
    assert allocator.expected_pairs == (
        allocator.path_cost_evaluations
    )
    assert (
        allocator.successful_path_cost_evaluations
        + allocator.unreachable_path_cost_evaluations
        == allocator.path_cost_evaluations
    )
