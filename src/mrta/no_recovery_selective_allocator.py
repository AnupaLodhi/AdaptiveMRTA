"""Experimental cost-aware selective reassignment for MRTA."""

from .progress_aware_allocator import ProgressAwareLockAllocator
from .path_planning import astar


class NoRecoverySelectiveAllocator(ProgressAwareLockAllocator):
    """
    PALR-style reservation delay followed by selective
    reassignment instead of a full global reassignment.

    At most one reservation is transferred per trigger.
    All A* evaluations are counted.
    """

    name = "cost_aware_selective"

    def __init__(self, margin=3, threshold=5):
        if (
            not isinstance(margin, int)
            or isinstance(margin, bool)
            or margin < 0
        ):
            raise ValueError("margin must be a nonnegative integer")

        if (
            not isinstance(threshold, int)
            or isinstance(threshold, bool)
            or threshold < 0
        ):
            raise ValueError("threshold must be nonnegative")

        super().__init__(threshold=threshold)

        self.margin = margin
        self.selective_checks = 0
        self.selective_transfers = 0
        self.rejected_checks = 0
        self.consecutive_rejections = 0
        self.recovery_global_calls = 0

    def _path_cost(self, grid_map, start, goal):
        self.path_cost_evaluations += 1

        path = astar(
            grid_map,
            tuple(int(v) for v in start),
            tuple(int(v) for v in goal),
        )

        if path is None:
            self.unreachable_path_cost_evaluations += 1
            return float("inf")

        self.successful_path_cost_evaluations += 1
        return len(path) - 1

    def assign_tasks(self, active_agents, active_tasks, grid_map):
        agent_by_id = {
            agent.agent_id: agent for agent in active_agents
        }
        task_by_id = {
            task.task_id: task for task in active_tasks
        }

        valid = {
            agent_id: task_id
            for agent_id, task_id in self.assignment_ids.items()
            if agent_id in agent_by_id
            and task_id in task_by_id
        }

        reserved = set(valid.values())

        idle = [
            agent for agent in active_agents
            if agent.agent_id not in valid
        ]

        free_tasks = [
            task for task in active_tasks
            if task.task_id not in reserved
        ]

        saturated = (
            bool(valid)
            and bool(idle)
            and bool(active_tasks)
            and not free_tasks
        )

        # Retain existing PALR behavior for initialization,
        # local repair, and the first threshold requests.
        if (
            not saturated
            or self.saturation_streak + 1 <= self.threshold
        ):
            return super().assign_tasks(
                active_agents, active_tasks, grid_map
            )

        # Replace PALR's full global replan with a
        # selective, cost-aware reservation transfer.
        self.assignment_requests += 1
        self.saturation_events += 1
        self.saturation_streak = 0
        self.assignment_ids = valid
        self.selective_checks += 1

        owner_by_task = {
            task_id: agent_id
            for agent_id, task_id in valid.items()
        }

        best = None
        best_gain = float("-inf")

        # Compare idle robots against current task owners.
        for task in active_tasks:
            owner_id = owner_by_task.get(task.task_id)

            if owner_id is None:
                continue

            owner = agent_by_id[owner_id]

            owner_cost = self._path_cost(
                grid_map,
                owner.position,
                task.position,
            )

            # Do not release a reservation when the
            # existing owner cannot be meaningfully compared.
            if owner_cost == float("inf"):
                continue

            for robot in idle:
                candidate_cost = self._path_cost(
                    grid_map,
                    robot.position,
                    task.position,
                )

                if candidate_cost == float("inf"):
                    continue

                gain = owner_cost - candidate_cost

                # Strict improvement is required even
                # when margin=0.
                if gain <= 0 or gain < self.margin:
                    continue

                if gain > best_gain:
                    best_gain = gain
                    best = (
                        owner_id,
                        robot.agent_id,
                        task.task_id,
                    )

        if best is not None:
            old_owner, new_owner, task_id = best

            del self.assignment_ids[old_owner]
            self.assignment_ids[new_owner] = task_id
            self.selective_transfers += 1
            self.consecutive_rejections = 0
        else:
            self.rejected_checks += 1
            self.consecutive_rejections += 1

            if False:  # Global recovery disabled for ablation
                self._solve_global(
                    active_agents,
                    active_tasks,
                    grid_map,
                )
                self.recovery_global_calls += 1
                self.consecutive_rejections = 0

        return {
            agent_id: task_by_id[task_id]
            for agent_id, task_id in self.assignment_ids.items()
        }
