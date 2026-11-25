from types import SimpleNamespace

import pytest

from mrta.grid_map import GridMap
from mrta.stall_aware_allocator import StallAwareCASRAllocator


def agent(agent_id, x, y):
    return SimpleNamespace(
        agent_id=agent_id,
        position=(x, y),
    )


def task(task_id, x, y):
    return SimpleNamespace(
        task_id=task_id,
        position=(x, y),
    )


def setup_allocator(stall_threshold=2):
    allocator = StallAwareCASRAllocator(
        margin=100,
        threshold=100,
        stall_threshold=stall_threshold,
    )
    allocator.assignment_ids = {0: 0}
    grid = GridMap(
        12, 12,
        seed=42,
        obstacle_density=0.0,
    )
    tasks = [task(0, 9, 9)]
    return allocator, grid, tasks


def test_invalid_stall_threshold():
    for value in [0, -1, True, 2.5]:
        with pytest.raises(ValueError):
            StallAwareCASRAllocator(stall_threshold=value)


def test_stall_transfer_after_two_stationary_observations():
    allocator, grid, tasks = setup_allocator()

    robots = [
        agent(0, 0, 0),
        agent(1, 9, 8),
    ]

    for _ in range(2):
        result = allocator.assign_tasks(
            robots, tasks, grid
        )
        assert set(result) == {0}

    result = allocator.assign_tasks(
        robots, tasks, grid
    )

    assert set(result) == {1}
    assert result[1].task_id == 0
    assert allocator.stall_transfers == 1


def test_movement_resets_stall_counter():
    allocator, grid, tasks = setup_allocator()

    allocator.assign_tasks(
        [agent(0, 0, 0), agent(1, 9, 8)],
        tasks, grid,
    )

    allocator.assign_tasks(
        [agent(0, 1, 0), agent(1, 9, 8)],
        tasks, grid,
    )

    result = allocator.assign_tasks(
        [agent(0, 2, 0), agent(1, 9, 8)],
        tasks, grid,
    )

    assert set(result) == {0}
    assert allocator.stall_transfers == 0


def test_no_transfer_without_idle_robot():
    allocator, grid, tasks = setup_allocator(
        stall_threshold=1
    )

    robots = [agent(0, 0, 0)]

    allocator.assign_tasks(robots, tasks, grid)
    result = allocator.assign_tasks(robots, tasks, grid)

    assert set(result) == {0}
    assert allocator.stall_transfers == 0


def test_no_transfer_without_positive_gain():
    allocator, grid, tasks = setup_allocator(
        stall_threshold=1
    )

    robots = [
        agent(0, 9, 8),
        agent(1, 0, 0),
    ]

    allocator.assign_tasks(robots, tasks, grid)
    result = allocator.assign_tasks(robots, tasks, grid)

    assert set(result) == {0}
    assert allocator.stall_transfers == 0
