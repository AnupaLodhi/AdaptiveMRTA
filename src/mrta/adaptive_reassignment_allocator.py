from .progress_aware_allocator import ProgressAwareLockAllocator


class AdaptiveReassignmentAllocator(ProgressAwareLockAllocator):
    """
    Experimental progress-triggered saturation controller.

    Uses externally observed completed-task counts.
    Shortens the saturation delay when task completion
    has stalled.

    Does not modify the fixed-delay baselines.
    """

    name = "adaptive_progress_triggered_reassignment"

    def __init__(
        self,
        normal_delay=5,
        stalled_delay=2,
        stall_window=3,
    ):
        if not all(
            isinstance(x, int) and not isinstance(x, bool)
            for x in (normal_delay, stalled_delay, stall_window)
        ):
            raise ValueError("Parameters must be integers")

        if not (
            0 <= stalled_delay <= normal_delay
            and stall_window >= 1
        ):
            raise ValueError("Invalid adaptive parameters")

        super().__init__(threshold=normal_delay)

        self.normal_delay = normal_delay
        self.stalled_delay = stalled_delay
        self.stall_window = stall_window

        self.observation_step = -1
        self.last_progress_step = 0
        self.last_completed_tasks = 0

        self.stall_triggered_requests = 0
        self.threshold_history = []

    def observe_progress(self, completed_tasks):
        """Called once per simulation iteration."""

        if (
            not isinstance(completed_tasks, int)
            or completed_tasks < self.last_completed_tasks
        ):
            raise ValueError(
                "Completed-task count must be monotonic"
            )

        self.observation_step += 1

        if completed_tasks > self.last_completed_tasks:
            self.last_progress_step = self.observation_step

        self.last_completed_tasks = completed_tasks

    def assign_tasks(
        self,
        active_agents,
        active_tasks,
        grid_map,
    ):
        if self.observation_step < 0:
            raise RuntimeError(
                "Call observe_progress before assign_tasks"
            )

        stalled_for = (
            self.observation_step
            - self.last_progress_step
        )

        stalled = stalled_for >= self.stall_window

        self.threshold = (
            self.stalled_delay
            if stalled
            else self.normal_delay
        )

        self.threshold_history.append(self.threshold)

        if stalled:
            self.stall_triggered_requests += 1

        return super().assign_tasks(
            active_agents,
            active_tasks,
            grid_map,
        )
