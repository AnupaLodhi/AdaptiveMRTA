import heapq


def manhattan_distance(a, b):
    return (
        abs(a[0] - b[0])
        +
        abs(a[1] - b[1])
    )


def reconstruct_path(
    came_from,
    current,
):
    path = [current]

    while current in came_from:
        current = came_from[current]
        path.append(current)

    path.reverse()

    return path


def astar(
    grid_map,
    start,
    goal,
):
    """
    Standard 4-neighbour A* path planning.

    Returns:
        list of (x, y) cells

    If no path exists:
        returns None
    """

    if not grid_map.is_free(
        start[0],
        start[1],
    ):
        return None

    if not grid_map.is_free(
        goal[0],
        goal[1],
    ):
        return None

    open_heap = []

    heapq.heappush(
        open_heap,
        (
            0,
            start,
        )
    )

    came_from = {}

    g_score = {
        start: 0
    }

    closed = set()

    while open_heap:

        _, current = (
            heapq.heappop(
                open_heap
            )
        )

        if current in closed:
            continue

        if current == goal:
            return reconstruct_path(
                came_from,
                current,
            )

        closed.add(current)

        for neighbor in (
            grid_map.get_neighbors(
                current
            )
        ):

            tentative_g = (
                g_score[current]
                + 1
            )

            if (
                tentative_g
                <
                g_score.get(
                    neighbor,
                    float("inf"),
                )
            ):

                came_from[
                    neighbor
                ] = current

                g_score[
                    neighbor
                ] = tentative_g

                f_score = (
                    tentative_g
                    +
                    manhattan_distance(
                        neighbor,
                        goal,
                    )
                )

                heapq.heappush(
                    open_heap,
                    (
                        f_score,
                        neighbor,
                    )
                )

    return None
