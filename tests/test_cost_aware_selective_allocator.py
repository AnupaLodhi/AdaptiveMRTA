from types import SimpleNamespace

from mrta.grid_map import GridMap
import pytest

from mrta.cost_aware_selective_allocator import (
    CostAwareSelectiveAllocator,
)


def agent(agent_id, position):
    return SimpleNamespace(
        agent_id=agent_id,
        position=position,
    )


def task(task_id, position):
    return SimpleNamespace(
        task_id=task_id,
        position=position,
    )


def test_invalid_margin():
    with pytest.raises(ValueError):
        CostAwareSelectiveAllocator(margin=-1)


def test_initial_assignment():
    allocator = CostAwareSelectiveAllocator(
        margin=3,
        threshold=0,
    )

    grid = GridMap(8, 8, seed=42, obstacle_density=0.0)

    result = allocator.assign_tasks(
        [agent(0, (0, 0))],
        [task(0, (1, 0))],
        grid,
    )

    assert result[0].task_id == 0
    assert allocator.global_assignment_calls == 1


def test_selective_transfer():
    allocator = CostAwareSelectiveAllocator(
        margin=3,
        threshold=0,
    )

    grid = GridMap(12, 12, seed=42, obstacle_density=0.0)

    # Establish an existing reservation.
    allocator.assignment_ids = {0: 0}

    result = allocator.assign_tasks(
        [
            agent(0, (0, 0)),
            agent(1, (9, 9)),
        ],
        [task(0, (9, 10))],
        grid,
    )

    assert set(result) == {1}
    assert result[1].task_id == 0
    assert allocator.selective_transfers == 1
    assert allocator.global_assignment_calls == 0
    assert allocator.path_cost_evaluations == 2


def test_margin_prevents_transfer():
    allocator = CostAwareSelectiveAllocator(
        margin=5,
        threshold=0,
    )

    grid = GridMap(12, 12, seed=42, obstacle_density=0.0)
    allocator.assignment_ids = {0: 0}

    result = allocator.assign_tasks(
        [
            agent(0, (5, 5)),
            agent(1, (7, 5)),
        ],
        [task(0, (8, 5))],
        grid,
    )

    assert set(result) == {0}
    assert allocator.selective_transfers == 0
    assert allocator.rejected_checks == 1


def test_recovery_after_two_rejected_checks():
    allocator = CostAwareSelectiveAllocator(
        margin=5,
        threshold=0,
    )

    grid = GridMap(
        12, 12,
        seed=42,
        obstacle_density=0.0,
    )

    allocator.assignment_ids = {0: 0}

    agents = [
        agent(0, (5, 5)),
        agent(1, (7, 5)),
    ]
    tasks = [task(0, (8, 5))]

    allocator.assign_tasks(agents, tasks, grid)

    assert allocator.rejected_checks == 1
    assert allocator.recovery_global_calls == 0

    allocator.assign_tasks(agents, tasks, grid)

    assert allocator.rejected_checks == 2
    assert allocator.recovery_global_calls == 1
    assert allocator.global_assignment_calls == 1
    assert allocator.assignment_ids == {1: 0}
