import pytest

from mrta.config import SimulationConfig
from mrta.obstacle_simulator import ObstacleSimulator
from mrta.coordinated_allocators import LockAwareHungarianAllocator
from mrta.progress_aware_allocator import ProgressAwareLockAllocator


def run_scenario(allocator, seed):
    config = SimulationConfig(
        num_agents=5,
        num_tasks=7,
        seed=seed,
        max_iterations=100,
    )

    simulator = ObstacleSimulator(
        config=config,
        allocator=allocator,
        obstacle_density=0.10,
    )

    result = simulator.run()
    return result, allocator


@pytest.mark.parametrize("seed", [42, 43, 44])
def test_threshold_zero_matches_original(seed):
    baseline, base_allocator = run_scenario(
        LockAwareHungarianAllocator(), seed
    )

    candidate, palr_allocator = run_scenario(
        ProgressAwareLockAllocator(threshold=0), seed
    )

    for key in [
        "success",
        "iterations",
        "total_distance",
        "navigation_astar_calls",
    ]:
        assert baseline[key] == candidate[key], key

    for key in [
        "path_cost_evaluations",
        "global_assignment_calls",
        "local_repair_calls",
        "lock_escalations",
    ]:
        assert (
            getattr(base_allocator, key)
            == getattr(palr_allocator, key)
        ), key


@pytest.mark.parametrize("threshold", [0, 2, 5])
def test_threshold_variants_have_valid_counters(threshold):
    result, allocator = run_scenario(
        ProgressAwareLockAllocator(threshold=threshold),
        seed=42,
    )

    assert isinstance(result["success"], (bool, int))
    assert allocator.path_cost_evaluations >= 0
    assert allocator.global_assignment_calls >= 0
    assert allocator.delayed_escalations >= 0
    assert allocator.saturation_events >= 0


def test_negative_threshold_rejected():
    with pytest.raises(ValueError):
        ProgressAwareLockAllocator(threshold=-1)
