from .coordinated_allocators import LockAwareHungarianAllocator


class ProgressAwareLockAllocator(LockAwareHungarianAllocator):
    """
    Experimental reservation-saturation delay policy.

    threshold=0 reproduces the original lock-aware allocator.

    threshold>0 delays global reassignment for a limited
    number of consecutive saturation requests.

    This is a delay-based baseline, not yet a true
    movement-progress-aware algorithm.
    """

    def __init__(self, threshold=0):
        super().__init__()

        if not isinstance(threshold, int) or threshold < 0:
            raise ValueError("threshold must be a nonnegative integer")

        self.threshold = threshold
        self.saturation_streak = 0
        self.delayed_escalations = 0
        self.saturation_events = 0

    def assign_tasks(self, active_agents, active_tasks, grid_map):
        if self.threshold == 0:
            return super().assign_tasks(
                active_agents, active_tasks, grid_map
            )

        # Count saturation before the parent allocator
        # performs its normal reassignment decision.
        valid_agent_ids = {
            agent.agent_id for agent in active_agents
        }
        valid_task_ids = {
            task.task_id for task in active_tasks
        }

        valid_assignments = {
            agent_id: task_id
            for agent_id, task_id in self.assignment_ids.items()
            if agent_id in valid_agent_ids
            and task_id in valid_task_ids
        }

        reserved = set(valid_assignments.values())

        unassigned_agents = [
            agent for agent in active_agents
            if agent.agent_id not in valid_assignments
        ]

        unassigned_tasks = [
            task for task in active_tasks
            if task.task_id not in reserved
        ]

        saturated = (
            bool(valid_assignments)
            and bool(unassigned_agents)
            and bool(active_tasks)
            and not unassigned_tasks
        )

        if saturated:
            self.saturation_events += 1
            self.saturation_streak += 1
        else:
            self.saturation_streak = 0

        if saturated and self.saturation_streak <= self.threshold:
            # Preserve valid reservations without triggering
            # the parent's expensive global reassignment.
            self.assignment_requests += 1
            self.assignment_ids = valid_assignments
            self.delayed_escalations += 1

            agents_by_id = {
                agent.agent_id: agent for agent in active_agents
            }
            tasks_by_id = {
                task.task_id: task for task in active_tasks
            }

            return {
                agent_id: tasks_by_id[task_id]
                for agent_id, task_id in valid_assignments.items()
                if agent_id in agents_by_id
                and task_id in tasks_by_id
            }

        if saturated:
            self.saturation_streak = 0

        return super().assign_tasks(
            active_agents, active_tasks, grid_map
        )
