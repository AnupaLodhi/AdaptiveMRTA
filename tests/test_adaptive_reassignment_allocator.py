import pytest

from mrta.adaptive_reassignment_allocator import (
    AdaptiveReassignmentAllocator,
)


def test_initial_configuration():
    allocator = AdaptiveReassignmentAllocator()

    assert allocator.normal_delay == 5
    assert allocator.stalled_delay == 2
    assert allocator.stall_window == 3


def test_stagnation_detection():
    allocator = AdaptiveReassignmentAllocator()

    for completed in [0, 0, 0, 0]:
        allocator.observe_progress(completed)

    assert allocator.observation_step == 3
    assert (
        allocator.observation_step
        - allocator.last_progress_step
    ) == 3


def test_progress_resets_stagnation():
    allocator = AdaptiveReassignmentAllocator()

    for completed in [0, 0, 0, 0, 1]:
        allocator.observe_progress(completed)

    assert allocator.last_progress_step == 4
    assert allocator.last_completed_tasks == 1


def test_invalid_configuration():
    with pytest.raises(ValueError):
        AdaptiveReassignmentAllocator(
            normal_delay=2,
            stalled_delay=5,
        )


def test_completed_tasks_cannot_decrease():
    allocator = AdaptiveReassignmentAllocator()

    allocator.observe_progress(3)

    with pytest.raises(ValueError):
        allocator.observe_progress(2)


def test_observation_required():
    allocator = AdaptiveReassignmentAllocator()

    with pytest.raises(RuntimeError):
        allocator.assign_tasks([], [], None)
