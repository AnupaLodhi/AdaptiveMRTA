from collections import deque
import numpy as np


class GridMap:
    """
    Reproducible 2D occupancy grid.

    False = free cell
    True  = obstacle
    """

    def __init__(
        self,
        width,
        height,
        seed,
        obstacle_density=0.12,
    ):
        self.width = width
        self.height = height
        self.seed = seed
        self.obstacle_density = obstacle_density

        self.rng = np.random.default_rng(
            seed + 10000
        )

        self.obstacles = np.zeros(
            (height, width),
            dtype=bool,
        )

        self._generate_obstacles()

    def _generate_obstacles(self):
        total_cells = (
            self.width * self.height
        )

        obstacle_count = int(
            total_cells
            * self.obstacle_density
        )

        flat_indices = self.rng.choice(
            total_cells,
            size=obstacle_count,
            replace=False,
        )

        rows = (
            flat_indices
            // self.width
        )

        cols = (
            flat_indices
            % self.width
        )

        self.obstacles[
            rows,
            cols
        ] = True

    def is_inside(self, x, y):
        return (
            0 <= x < self.width
            and
            0 <= y < self.height
        )

    def is_free(self, x, y):
        if not self.is_inside(x, y):
            return False

        return not self.obstacles[
            y,
            x
        ]

    def get_neighbors(self, cell):
        x, y = cell

        candidates = [
            (x + 1, y),
            (x - 1, y),
            (x, y + 1),
            (x, y - 1),
        ]

        return [
            candidate
            for candidate in candidates
            if self.is_free(
                candidate[0],
                candidate[1],
            )
        ]

    def largest_free_component(self):
        """
        Return cells belonging to the largest connected
        free-space region.

        Agents/tasks will be sampled only from this region.
        """

        visited = set()
        largest_component = []

        for y in range(self.height):
            for x in range(self.width):

                start = (x, y)

                if (
                    start in visited
                    or not self.is_free(x, y)
                ):
                    continue

                queue = deque([start])
                visited.add(start)

                component = []

                while queue:
                    current = queue.popleft()

                    component.append(
                        current
                    )

                    for neighbor in (
                        self.get_neighbors(current)
                    ):
                        if neighbor not in visited:
                            visited.add(
                                neighbor
                            )

                            queue.append(
                                neighbor
                            )

                if (
                    len(component)
                    >
                    len(largest_component)
                ):
                    largest_component = (
                        component
                    )

        return largest_component
