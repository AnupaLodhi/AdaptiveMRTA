import numpy as np

from scipy.optimize import linear_sum_assignment

from .path_planning import astar


class HungarianPathAllocator:
    """
    Coordinated MRTA baseline.

    Builds a robot-task cost matrix using A* path length
    and solves the minimum-cost one-to-one assignment
    with the Hungarian algorithm.

    When tasks >= robots:
        every robot can receive at most one unique task.

    When robots > tasks:
        only a subset of robots receives an assignment.
    """

    name = "hungarian_path"

    def __init__(self):
        self.assignment_requests = 0
        self.global_assignment_calls = 0

        self.path_cost_evaluations = 0
        self.successful_path_cost_evaluations = 0
        self.unreachable_path_cost_evaluations = 0

    def assign_tasks(
        self,
        active_agents,
        active_tasks,
        grid_map,
    ):
        self.assignment_requests += 1

        if (
            not active_agents
            or not active_tasks
        ):
            return {}

        self.global_assignment_calls += 1

        num_agents = len(
            active_agents
        )

        num_tasks = len(
            active_tasks
        )

        cost_matrix = np.full(
            (
                num_agents,
                num_tasks,
            ),
            1e9,
            dtype=float,
        )

        for i, agent in enumerate(
            active_agents
        ):
            start = (
                int(agent.position[0]),
                int(agent.position[1]),
            )

            for j, task in enumerate(
                active_tasks
            ):
                goal = (
                    int(task.position[0]),
                    int(task.position[1]),
                )

                self.path_cost_evaluations += 1

                path = astar(
                    grid_map,
                    start,
                    goal,
                )

                if path is None:
                    self.unreachable_path_cost_evaluations += 1
                    continue

                self.successful_path_cost_evaluations += 1

                cost_matrix[
                    i,
                    j
                ] = (
                    len(path) - 1
                )

        row_indices, column_indices = (
            linear_sum_assignment(
                cost_matrix
            )
        )

        assignments = {}

        for row, column in zip(
            row_indices,
            column_indices,
        ):
            # Skip impossible assignments.
            if (
                cost_matrix[
                    row,
                    column
                ]
                >= 1e9
            ):
                continue

            agent = active_agents[
                row
            ]

            task = active_tasks[
                column
            ]

            assignments[
                agent.agent_id
            ] = task

        return assignments


class StatefulHungarianAllocator:
    """
    Stateful coordinated MRTA allocator.

    Unlike HungarianPathAllocator, which recomputes the
    complete assignment whenever assign_tasks() is called,
    this allocator preserves valid robot-task assignments.

    A new Hungarian optimization is performed only for
    currently unassigned active robots and currently
    unassigned active tasks.

    This is an experimental E5 allocator, not yet the
    proposed final method.
    """

    name = "stateful_hungarian_path"

    def __init__(self):
        # agent_id -> task_id
        self.assignment_ids = {}

        # Number of actual Hungarian optimization calls.
        self.global_assignment_calls = 0

        # Number of assign_tasks() requests from simulator.
        self.assignment_requests = 0

        # E5C diagnostic counters.
        self.path_cost_evaluations = 0
        self.successful_path_cost_evaluations = 0
        self.unreachable_path_cost_evaluations = 0

        self.repair_calls = 0
        self.empty_repair_events = 0
        self.max_unassigned_agents = 0
        self.max_unassigned_tasks = 0

    def _remove_invalid_assignments(
        self,
        active_agents,
        active_tasks,
    ):
        active_agent_ids = {
            agent.agent_id
            for agent in active_agents
        }

        active_task_ids = {
            task.task_id
            for task in active_tasks
        }

        self.assignment_ids = {
            agent_id: task_id
            for agent_id, task_id
            in self.assignment_ids.items()
            if (
                agent_id in active_agent_ids
                and task_id in active_task_ids
            )
        }

    def assign_tasks(
        self,
        active_agents,
        active_tasks,
        grid_map,
    ):
        self.assignment_requests += 1

        if (
            not active_agents
            or not active_tasks
        ):
            self.assignment_ids = {}
            return {}

        self._remove_invalid_assignments(
            active_agents,
            active_tasks,
        )

        agents_by_id = {
            agent.agent_id: agent
            for agent in active_agents
        }

        tasks_by_id = {
            task.task_id: task
            for task in active_tasks
        }

        # Tasks already reserved by valid assignments.
        reserved_task_ids = set(
            self.assignment_ids.values()
        )

        unassigned_agents = [
            agent
            for agent in active_agents
            if agent.agent_id
            not in self.assignment_ids
        ]

        unassigned_tasks = [
            task
            for task in active_tasks
            if task.task_id
            not in reserved_task_ids
        ]

        # Only solve a new optimization if both sides
        # contain something that can actually be assigned.
        self.max_unassigned_agents = max(
            self.max_unassigned_agents,
            len(unassigned_agents),
        )

        self.max_unassigned_tasks = max(
            self.max_unassigned_tasks,
            len(unassigned_tasks),
        )

        if (
            unassigned_agents
            and not unassigned_tasks
        ):
            self.empty_repair_events += 1

        if (
            unassigned_agents
            and unassigned_tasks
        ):
            self.repair_calls += 1

            num_agents = len(
                unassigned_agents
            )

            num_tasks = len(
                unassigned_tasks
            )

            cost_matrix = np.full(
                (
                    num_agents,
                    num_tasks,
                ),
                1e9,
                dtype=float,
            )

            for i, agent in enumerate(
                unassigned_agents
            ):
                start = (
                    int(agent.position[0]),
                    int(agent.position[1]),
                )

                for j, task in enumerate(
                    unassigned_tasks
                ):
                    goal = (
                        int(task.position[0]),
                        int(task.position[1]),
                    )

                    self.path_cost_evaluations += 1

                    path = astar(
                        grid_map,
                        start,
                        goal,
                    )

                    if path is None:
                        self.unreachable_path_cost_evaluations += 1
                        continue

                    self.successful_path_cost_evaluations += 1

                    cost_matrix[
                        i,
                        j
                    ] = (
                        len(path) - 1
                    )

            row_indices, column_indices = (
                linear_sum_assignment(
                    cost_matrix
                )
            )

            self.global_assignment_calls += 1

            for row, column in zip(
                row_indices,
                column_indices,
            ):
                if (
                    cost_matrix[
                        row,
                        column
                    ]
                    >= 1e9
                ):
                    continue

                agent = (
                    unassigned_agents[
                        row
                    ]
                )

                task = (
                    unassigned_tasks[
                        column
                    ]
                )

                self.assignment_ids[
                    agent.agent_id
                ] = task.task_id

        # Convert stored IDs back to current Task objects.
        assignments = {}

        for agent_id, task_id in (
            self.assignment_ids.items()
        ):
            if (
                agent_id in agents_by_id
                and task_id in tasks_by_id
            ):
                assignments[
                    agent_id
                ] = tasks_by_id[
                    task_id
                ]

        return assignments


class PeriodicHungarianAllocator:
    """
    Periodically replanned global Hungarian allocator.

    Performs a complete A*-cost Hungarian assignment:

    1. On the first allocation request.
    2. Every `replan_interval` allocation requests.
    3. Immediately when a previously stored assignment
       becomes invalid because a robot or task disappears.

    Between replans, valid assignments are preserved.

    Experimental E5 baseline for measuring the
    computation-quality trade-off.
    """

    name = "periodic_hungarian_path"

    def __init__(
        self,
        replan_interval=4,
    ):
        if replan_interval < 1:
            raise ValueError(
                "replan_interval must be >= 1"
            )

        self.replan_interval = (
            replan_interval
        )

        self.assignment_ids = {}

        self.assignment_requests = 0
        self.global_assignment_calls = 0

        self.periodic_replans = 0
        self.event_replans = 0

        self.path_cost_evaluations = 0
        self.successful_path_cost_evaluations = 0
        self.unreachable_path_cost_evaluations = 0

    def _current_ids(
        self,
        active_agents,
        active_tasks,
    ):
        active_agent_ids = {
            agent.agent_id
            for agent in active_agents
        }

        active_task_ids = {
            task.task_id
            for task in active_tasks
        }

        return (
            active_agent_ids,
            active_task_ids,
        )

    def _assignment_invalid(
        self,
        active_agent_ids,
        active_task_ids,
    ):
        if not self.assignment_ids:
            return True

        for agent_id, task_id in (
            self.assignment_ids.items()
        ):
            if (
                agent_id
                not in active_agent_ids
                or task_id
                not in active_task_ids
            ):
                return True

        return False

    def _solve_global_assignment(
        self,
        active_agents,
        active_tasks,
        grid_map,
    ):
        num_agents = len(
            active_agents
        )

        num_tasks = len(
            active_tasks
        )

        cost_matrix = np.full(
            (
                num_agents,
                num_tasks,
            ),
            1e9,
            dtype=float,
        )

        for i, agent in enumerate(
            active_agents
        ):
            start = (
                int(agent.position[0]),
                int(agent.position[1]),
            )

            for j, task in enumerate(
                active_tasks
            ):
                goal = (
                    int(task.position[0]),
                    int(task.position[1]),
                )

                self.path_cost_evaluations += 1

                path = astar(
                    grid_map,
                    start,
                    goal,
                )

                if path is None:
                    self.unreachable_path_cost_evaluations += 1
                    continue

                self.successful_path_cost_evaluations += 1

                cost_matrix[
                    i,
                    j
                ] = (
                    len(path) - 1
                )

        row_indices, column_indices = (
            linear_sum_assignment(
                cost_matrix
            )
        )

        new_assignments = {}

        for row, column in zip(
            row_indices,
            column_indices,
        ):
            if (
                cost_matrix[
                    row,
                    column
                ]
                >= 1e9
            ):
                continue

            agent = active_agents[
                row
            ]

            task = active_tasks[
                column
            ]

            new_assignments[
                agent.agent_id
            ] = task.task_id

        self.assignment_ids = (
            new_assignments
        )

        self.global_assignment_calls += 1

    def assign_tasks(
        self,
        active_agents,
        active_tasks,
        grid_map,
    ):
        self.assignment_requests += 1

        if (
            not active_agents
            or not active_tasks
        ):
            self.assignment_ids = {}
            return {}

        (
            active_agent_ids,
            active_task_ids,
        ) = self._current_ids(
            active_agents,
            active_tasks,
        )

        invalid = (
            self._assignment_invalid(
                active_agent_ids,
                active_task_ids,
            )
        )

        first_request = (
            self.assignment_requests == 1
        )

        periodic_due = (
            (
                self.assignment_requests - 1
            )
            % self.replan_interval
            == 0
        )

        should_replan = (
            first_request
            or invalid
            or periodic_due
        )

        if should_replan:

            # Classify the trigger for diagnostics.
            if (
                invalid
                and not first_request
            ):
                self.event_replans += 1

            elif (
                periodic_due
                and not first_request
            ):
                self.periodic_replans += 1

            self._solve_global_assignment(
                active_agents,
                active_tasks,
                grid_map,
            )

        tasks_by_id = {
            task.task_id: task
            for task in active_tasks
        }

        assignments = {}

        for agent_id, task_id in (
            self.assignment_ids.items()
        ):
            if (
                agent_id
                in active_agent_ids
                and task_id
                in tasks_by_id
            ):
                assignments[
                    agent_id
                ] = tasks_by_id[
                    task_id
                ]

        return assignments


class LockAwareHungarianAllocator:
    """
    Experimental E5C hybrid allocator.

    Preserves valid robot-task assignments and performs
    subset Hungarian repair for newly unassigned robots.

    If local repair becomes allocation-locked -- active
    unassigned robots exist but all remaining tasks are
    reserved -- the allocator releases the preserved
    assignment state and performs a full global Hungarian
    reassignment.

    No numeric escalation threshold is used.
    """

    name = "lock_aware_hungarian"

    def __init__(self):
        self.assignment_ids = {}

        self.assignment_requests = 0

        self.local_repair_calls = 0
        self.global_assignment_calls = 0
        self.lock_escalations = 0

        self.path_cost_evaluations = 0
        self.successful_path_cost_evaluations = 0
        self.unreachable_path_cost_evaluations = 0


    def _remove_invalid_assignments(
        self,
        active_agents,
        active_tasks,
    ):
        active_agent_ids = {
            agent.agent_id
            for agent in active_agents
        }

        active_task_ids = {
            task.task_id
            for task in active_tasks
        }

        self.assignment_ids = {
            agent_id: task_id
            for agent_id, task_id
            in self.assignment_ids.items()
            if (
                agent_id in active_agent_ids
                and task_id in active_task_ids
            )
        }


    def _build_cost_matrix(
        self,
        agents,
        tasks,
        grid_map,
    ):
        cost_matrix = np.full(
            (
                len(agents),
                len(tasks),
            ),
            1e9,
            dtype=float,
        )

        for i, agent in enumerate(agents):

            start = (
                int(agent.position[0]),
                int(agent.position[1]),
            )

            for j, task in enumerate(tasks):

                goal = (
                    int(task.position[0]),
                    int(task.position[1]),
                )

                self.path_cost_evaluations += 1

                path = astar(
                    grid_map,
                    start,
                    goal,
                )

                if path is None:
                    self.unreachable_path_cost_evaluations += 1
                    continue

                self.successful_path_cost_evaluations += 1

                cost_matrix[i, j] = (
                    len(path) - 1
                )

        return cost_matrix


    def _solve_subset(
        self,
        agents,
        tasks,
        grid_map,
    ):
        if not agents or not tasks:
            return {}

        cost_matrix = self._build_cost_matrix(
            agents,
            tasks,
            grid_map,
        )

        rows, columns = (
            linear_sum_assignment(
                cost_matrix
            )
        )

        result = {}

        for row, column in zip(
            rows,
            columns,
        ):
            if (
                cost_matrix[row, column]
                >= 1e9
            ):
                continue

            result[
                agents[row].agent_id
            ] = tasks[column].task_id

        return result


    def _solve_global(
        self,
        active_agents,
        active_tasks,
        grid_map,
    ):
        self.global_assignment_calls += 1

        self.assignment_ids = (
            self._solve_subset(
                active_agents,
                active_tasks,
                grid_map,
            )
        )


    def assign_tasks(
        self,
        active_agents,
        active_tasks,
        grid_map,
    ):
        self.assignment_requests += 1

        if (
            not active_agents
            or not active_tasks
        ):
            self.assignment_ids = {}
            return {}

        self._remove_invalid_assignments(
            active_agents,
            active_tasks,
        )

        agents_by_id = {
            agent.agent_id: agent
            for agent in active_agents
        }

        tasks_by_id = {
            task.task_id: task
            for task in active_tasks
        }

        # Initial request requires a coordinated solution.
        if not self.assignment_ids:

            self._solve_global(
                active_agents,
                active_tasks,
                grid_map,
            )

        else:
            reserved_task_ids = set(
                self.assignment_ids.values()
            )

            unassigned_agents = [
                agent
                for agent in active_agents
                if agent.agent_id
                not in self.assignment_ids
            ]

            unassigned_tasks = [
                task
                for task in active_tasks
                if task.task_id
                not in reserved_task_ids
            ]

            # Structural allocation lock:
            # robots need assignments, tasks still exist,
            # but every remaining task is reserved.
            allocation_lock = (
                bool(unassigned_agents)
                and bool(active_tasks)
                and not unassigned_tasks
            )

            if allocation_lock:

                self.lock_escalations += 1

                self._solve_global(
                    active_agents,
                    active_tasks,
                    grid_map,
                )

            elif (
                unassigned_agents
                and unassigned_tasks
            ):
                self.local_repair_calls += 1

                repaired = self._solve_subset(
                    unassigned_agents,
                    unassigned_tasks,
                    grid_map,
                )

                self.assignment_ids.update(
                    repaired
                )

        # Convert IDs back to current Task objects.
        assignments = {}

        for agent_id, task_id in (
            self.assignment_ids.items()
        ):
            if (
                agent_id in agents_by_id
                and task_id in tasks_by_id
            ):
                assignments[
                    agent_id
                ] = tasks_by_id[
                    task_id
                ]

        return assignments
