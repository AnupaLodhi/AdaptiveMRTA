"""Experimental stall-aware extension of CASR."""

from .no_recovery_selective_allocator import (
    NoRecoverySelectiveAllocator,
)


class NoRecoveryCASRAllocator(NoRecoverySelectiveAllocator):

    name = "casr_v2_no_recovery"

    def __init__(
        self,
        margin=3,
        threshold=5,
        stall_threshold=3,
    ):
        super().__init__(
            margin=margin,
            threshold=threshold,
        )

        if (
            type(stall_threshold) is not int
            or stall_threshold < 1
        ):
            raise ValueError(
                "stall_threshold must be a positive integer"
            )

        self.stall_threshold = stall_threshold
        self.previous_states = {}
        self.stall_counts = {}

        self.stall_checks = 0
        self.stall_transfers = 0
        self.stall_rejections = 0

    def assign_tasks(
        self,
        active_agents,
        active_tasks,
        grid_map,
    ):
        agents = {
            a.agent_id: a for a in active_agents
        }

        tasks = {
            t.task_id: t for t in active_tasks
        }

        valid = {
            aid: tid
            for aid, tid in self.assignment_ids.items()
            if aid in agents and tid in tasks
        }

        current_states = {}
        current_counts = {}

        for aid, tid in valid.items():
            position = tuple(
                int(v) for v in agents[aid].position
            )

            state = (tid, position)

            current_states[aid] = state

            if self.previous_states.get(aid) == state:
                current_counts[aid] = (
                    self.stall_counts.get(aid, 0) + 1
                )
            else:
                current_counts[aid] = 0

        self.stall_counts = current_counts

        stalled = [
            aid for aid, count in current_counts.items()
            if count >= self.stall_threshold
        ]

        idle = [
            a for a in active_agents
            if a.agent_id not in valid
        ]

        saturated = (
            bool(valid)
            and bool(idle)
            and bool(tasks)
            and set(valid.values()) == set(tasks)
        )

        best = None
        best_gain = 0

        if saturated and stalled:
            self.stall_checks += 1

            for aid in stalled:
                tid = valid[aid]
                task = tasks[tid]

                owner_cost = self._path_cost(
                    grid_map,
                    agents[aid].position,
                    task.position,
                )

                if owner_cost == float("inf"):
                    continue

                for candidate in idle:
                    candidate_cost = self._path_cost(
                        grid_map,
                        candidate.position,
                        task.position,
                    )

                    if candidate_cost == float("inf"):
                        continue

                    gain = owner_cost - candidate_cost

                    if gain > best_gain:
                        best_gain = gain
                        best = (
                            aid,
                            candidate.agent_id,
                            tid,
                        )

            if best is None:
                self.stall_rejections += 1

        if best is not None:
            old_owner, new_owner, tid = best

            self.assignment_ids = valid.copy()

            del self.assignment_ids[old_owner]
            self.assignment_ids[new_owner] = tid

            self.assignment_requests += 1
            self.saturation_events += 1
            self.saturation_streak = 0

            self.selective_checks += 1
            self.selective_transfers += 1
            self.stall_transfers += 1
            self.consecutive_rejections = 0

            assignments = {
                aid: tasks[tid]
                for aid, tid in self.assignment_ids.items()
            }

        else:
            assignments = super().assign_tasks(
                active_agents,
                active_tasks,
                grid_map,
            )

        # Store post-allocation ownership and pre-movement
        # positions for the next iteration's observation.

        self.previous_states = {
            aid: (
                task.task_id,
                tuple(
                    int(v) for v in agents[aid].position
                ),
            )
            for aid, task in assignments.items()
        }

        self.stall_counts = {
            aid: current_counts.get(aid, 0)
            if current_states.get(aid) == state
            else 0
            for aid, state in self.previous_states.items()
        }

        return assignments
