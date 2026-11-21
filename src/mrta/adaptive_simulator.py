from .obstacle_simulator import ObstacleSimulator


class AdaptiveObstacleSimulator(ObstacleSimulator):
    """
    Supplies completed-task observations before allocation.

    Keeps the original ObstacleSimulator unchanged.
    """

    def _make_assignments(
        self,
        active_agents,
        active_tasks,
    ):
        if hasattr(self.allocator, "observe_progress"):
            self.allocator.observe_progress(
                int(self.completed_tasks)
            )

        return super()._make_assignments(
            active_agents,
            active_tasks,
        )
