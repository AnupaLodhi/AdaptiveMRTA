from unittest.mock import patch

from mrta.config import SimulationConfig
from mrta.obstacle_simulator import ObstacleSimulator
from mrta.coordinated_allocators import HungarianPathAllocator
import mrta.obstacle_simulator as simulator_module


def test_navigation_counter_matches_actual_calls():
    config = SimulationConfig(
        num_agents=3,
        num_tasks=5,
        seed=42,
        max_iterations=100,
    )

    simulator = ObstacleSimulator(
        config=config,
        allocator=HungarianPathAllocator(),
        obstacle_density=0.0,
    )

    original_astar = simulator_module.astar
    actual_calls = 0

    def counted_astar(*args, **kwargs):
        nonlocal actual_calls
        actual_calls += 1
        return original_astar(*args, **kwargs)

    with patch.object(
        simulator_module,
        "astar",
        side_effect=counted_astar,
    ):
        result = simulator.run()

    assert result["success"]
    assert actual_calls > 0

    assert (
        result["navigation_astar_calls"]
        == actual_calls
    ), (
        f"Counter={result['navigation_astar_calls']}, "
        f"Actual={actual_calls}"
    )

    print("\nVerified navigation calls:", actual_calls)
