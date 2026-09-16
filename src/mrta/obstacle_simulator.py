from collections import Counter

import numpy as np

from .obstacle_environment import ObstacleEnvironment
from .path_planning import astar


class ObstacleSimulator:
    """
    Grid-based MRTA simulator.

    Supports:

    1. Independent allocators
       select_target(...)

    2. Coordinated allocators
       assign_tasks(...)

    Navigation:
        A*

    Motion:
        one grid cell per iteration
    """

    def __init__(
        self,
        config,
        allocator,
        obstacle_density=0.12,
        scenario=None,
        failure_schedule=None,
    ):
        self.config = config
        self.allocator = allocator

        # Optional deterministic mid-run robot failures.
        #
        # Format:
        # {
        #     iteration: [agent_id, ...]
        # }
        #
        # Example:
        # {
        #     5: [0, 3]
        # }
        #
        # Existing experiments are unchanged when omitted.
        self.failure_schedule = (
            failure_schedule
            if failure_schedule is not None
            else {}
        )

        self.failure_events = []
        self.failed_agent_ids = set()

        self.environment = (
            ObstacleEnvironment(
                config=config,
                obstacle_density=obstacle_density,
                scenario=scenario,
            )
        )

        self.iteration = 0
        self.completed_tasks = 0

        self.blocked_moves = 0
        self.unreachable_plans = 0

        self.redundant_target_events = 0
        self.total_target_assignments = 0

    def _cell(
        self,
        position,
    ):
        return (
            int(position[0]),
            int(position[1]),
        )

    def _resolve_completed_tasks(
        self,
    ):
        active_agents = (
            self.environment
            .get_active_agents()
        )

        active_tasks = (
            self.environment
            .get_active_tasks()
        )

        agent_cells = {}

        for agent in active_agents:
            cell = self._cell(
                agent.position
            )

            agent_cells.setdefault(
                cell,
                [],
            ).append(
                agent
            )

        for task in active_tasks:
            task_cell = self._cell(
                task.position
            )

            candidates = (
                agent_cells.get(
                    task_cell,
                    [],
                )
            )

            if not candidates:
                continue

            winner = min(
                candidates,
                key=lambda agent:
                    agent.agent_id,
            )

            task.active = False

            winner.tasks_completed += 1

            self.completed_tasks += 1

    def _count_redundant_targets(
        self,
        target_ids,
    ):
        counts = Counter(
            target_ids
        )

        redundant = sum(
            max(
                0,
                count - 1,
            )
            for count
            in counts.values()
        )

        self.redundant_target_events += (
            redundant
        )

        self.total_target_assignments += (
            len(target_ids)
        )

    def _make_assignments(
        self,
        active_agents,
        active_tasks,
    ):
        # Coordinated allocator.
        if hasattr(
            self.allocator,
            "assign_tasks",
        ):
            return (
                self.allocator
                .assign_tasks(
                    active_agents,
                    active_tasks,
                    self.environment.grid_map,
                )
            )

        # Independent allocator.
        assignments = {}

        for agent in active_agents:
            target = (
                self.allocator
                .select_target(
                    agent,
                    active_tasks,
                    self.environment.grid_map,
                )
            )

            if target is None:
                continue

            assignments[
                agent.agent_id
            ] = target

        return assignments

    def _apply_scheduled_failures(
        self,
    ):
        failed_now = (
            self.failure_schedule.get(
                self.iteration,
                [],
            )
        )

        if not failed_now:
            return

        agents_by_id = {
            agent.agent_id: agent
            for agent
            in self.environment.agents
        }

        for agent_id in failed_now:

            agent = agents_by_id.get(
                agent_id
            )

            if agent is None:
                raise ValueError(
                    f"Failure schedule references "
                    f"unknown agent {agent_id}."
                )

            # Ignore duplicate failure events safely.
            if not agent.active:
                continue

            agent.active = False
            agent.target_id = None

            self.failed_agent_ids.add(
                agent_id
            )

            self.failure_events.append(
                {
                    "iteration":
                        self.iteration,
                    "agent_id":
                        agent_id,
                }
            )

    def step(
        self,
    ):
        # Failure events occur at the beginning
        # of the scheduled iteration, before
        # allocation and movement.
        self._apply_scheduled_failures()

        active_tasks = (
            self.environment
            .get_active_tasks()
        )

        active_agents = (
            self.environment
            .get_active_agents()
        )

        if (
            not active_tasks
            or not active_agents
        ):
            return

        assignments = (
            self._make_assignments(
                active_agents,
                active_tasks,
            )
        )

        target_ids = []

        for agent in active_agents:
            target = assignments.get(
                agent.agent_id
            )

            if target is None:
                agent.target_id = None
                continue

            agent.target_id = (
                target.task_id
            )

            target_ids.append(
                target.task_id
            )

        self._count_redundant_targets(
            target_ids
        )

        occupied_at_start = {
            self._cell(
                agent.position
            )
            for agent
            in active_agents
        }

        reserved_destinations = set()

        for agent in sorted(
            active_agents,
            key=lambda item:
                item.agent_id,
        ):
            target = assignments.get(
                agent.agent_id
            )

            if target is None:
                continue

            start = self._cell(
                agent.position
            )

            goal = self._cell(
                target.position
            )

            if start == goal:
                reserved_destinations.add(
                    start
                )
                continue

            path = astar(
                self.environment.grid_map,
                start,
                goal,
            )

            if (
                path is None
                or len(path) < 2
            ):
                self.unreachable_plans += 1

                reserved_destinations.add(
                    start
                )

                continue

            proposed = (
                path[1]
            )

            blocked_by_occupancy = (
                proposed
                in occupied_at_start
                and proposed != start
            )

            blocked_by_reservation = (
                proposed
                in reserved_destinations
            )

            if (
                blocked_by_occupancy
                or blocked_by_reservation
            ):
                self.blocked_moves += 1

                reserved_destinations.add(
                    start
                )

                continue

            old_position = (
                agent.position.copy()
            )

            agent.position = np.array(
                proposed,
                dtype=float,
            )

            travelled = (
                np.linalg.norm(
                    agent.position
                    - old_position
                )
            )

            agent.total_distance += (
                travelled
            )

            reserved_destinations.add(
                proposed
            )

        self._resolve_completed_tasks()

        self.iteration += 1

    def run(
        self,
    ):
        self._resolve_completed_tasks()

        while (
            self.iteration
            < self.config.max_iterations
            and not self.environment
            .all_tasks_completed()
        ):
            self.step()

        success = (
            self.environment
            .all_tasks_completed()
        )

        total_distance = sum(
            agent.total_distance
            for agent
            in self.environment.agents
        )

        productive_agents = sum(
            1
            for agent
            in self.environment.agents
            if agent.tasks_completed > 0
        )

        if (
            self.total_target_assignments
            > 0
        ):
            redundant_target_ratio = (
                self.redundant_target_events
                /
                self.total_target_assignments
            )

        else:
            redundant_target_ratio = 0.0

        return {
            "success": success,
            "iterations":
                self.iteration,
            "completed_tasks":
                self.completed_tasks,
            "total_tasks":
                self.config.num_tasks,
            "total_distance":
                total_distance,
            "productive_agents":
                productive_agents,
            "blocked_moves":
                self.blocked_moves,
            "unreachable_plans":
                self.unreachable_plans,
            "redundant_target_events":
                self.redundant_target_events,
            "redundant_target_ratio":
                redundant_target_ratio,
            "failed_agents":
                len(self.failed_agent_ids),
            "failed_agent_ids":
                sorted(self.failed_agent_ids),
            "failure_events":
                list(self.failure_events),
        }
