from mrta.config import SimulationConfig
from mrta.obstacle_simulator import ObstacleSimulator
from mrta.coordinated_allocators import HungarianPathAllocator


def test_simulation_terminates_when_all_agents_fail():
    """
    Regression test:
    unfinished tasks + zero active robots must terminate safely
    instead of causing an infinite run loop.
    """

    config = SimulationConfig(
        num_agents=2,
        num_tasks=3,
        max_iterations=100,
        seed=42,
    )

    allocator = HungarianPathAllocator()

    # Both robots fail at the beginning of iteration 0,
    # before allocation or movement.
    failure_schedule = {
        0: [0, 1],
    }

    simulator = ObstacleSimulator(
        config=config,
        allocator=allocator,
        obstacle_density=0.0,
        failure_schedule=failure_schedule,
    )

    result = simulator.run()

    assert result["success"] is False
    assert result["completed_tasks"] < result["total_tasks"]
    assert result["failed_agents"] == 2
    assert result["failed_agent_ids"] == [0, 1]

    # Failure occurs before movement at iteration 0.
    # No meaningless empty iterations should be added.
    assert result["iterations"] == 0
