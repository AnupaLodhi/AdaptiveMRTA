from pathlib import Path
import json
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"

sys.path.insert(0, str(SRC_PATH))

from mrta.config import SimulationConfig
from mrta.coordinated_allocators import HungarianPathAllocator
from mrta.obstacle_simulator import ObstacleSimulator
from mrta.path_planning import astar


OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "visualizations"
    / "threejs"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


config = SimulationConfig(
    num_agents=15,
    num_tasks=20,
    seed=42,
    max_iterations=500,
)

OBSTACLE_DENSITY = 0.20


simulator = ObstacleSimulator(
    config=config,
    allocator=HungarianPathAllocator(),
    obstacle_density=OBSTACLE_DENSITY,
)


def cell(position):
    return [
        int(position[0]),
        int(position[1]),
    ]


def get_task(task_id):
    for task in simulator.environment.tasks:
        if task.task_id == task_id:
            return task
    return None


def capture_frame():
    robots = []

    for agent in simulator.environment.agents:
        target_id = None
        path = []

        if agent.target_id is not None:
            task = get_task(agent.target_id)

            if task is not None and task.active:
                target_id = task.task_id

                route = astar(
                    simulator.environment.grid_map,
                    tuple(cell(agent.position)),
                    tuple(cell(task.position)),
                )

                if route is not None:
                    path = [
                        [x, y]
                        for x, y in route
                    ]

        robots.append(
            {
                "id": agent.agent_id,
                "position": cell(agent.position),
                "target": target_id,
                "tasks_completed": agent.tasks_completed,
                "path": path,
            }
        )

    tasks = []

    for task in simulator.environment.tasks:
        tasks.append(
            {
                "id": task.task_id,
                "position": cell(task.position),
                "active": bool(task.active),
            }
        )

    total_distance = sum(
        agent.total_distance
        for agent in simulator.environment.agents
    )

    if simulator.total_target_assignments > 0:
        redundancy = (
            simulator.redundant_target_events
            / simulator.total_target_assignments
        )
    else:
        redundancy = 0.0

    return {
        "iteration": simulator.iteration,
        "completed_tasks": simulator.completed_tasks,
        "distance": float(total_distance),
        "blocked_moves": simulator.blocked_moves,
        "redundancy": float(redundancy),
        "robots": robots,
        "tasks": tasks,
    }


frames = [
    capture_frame()
]


while (
    simulator.iteration < config.max_iterations
    and not simulator.environment.all_tasks_completed()
):
    simulator.step()

    frames.append(
        capture_frame()
    )


obstacles = []

grid = simulator.environment.grid_map.obstacles

for y in range(config.grid_height):
    for x in range(config.grid_width):
        if grid[y, x]:
            obstacles.append(
                [x, y]
            )


data = {
    "config": {
        "width": config.grid_width,
        "height": config.grid_height,
        "num_agents": config.num_agents,
        "num_tasks": config.num_tasks,
        "seed": config.seed,
        "obstacle_density": OBSTACLE_DENSITY,
        "algorithm": "Hungarian + A*",
    },
    "obstacles": obstacles,
    "frames": frames,
}


output_path = (
    OUTPUT_DIR
    / "simulation.json"
)


with open(
    output_path,
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        data,
        f,
        indent=2,
    )


print("\n======================================")
print("SIMULATION JSON CREATED")
print("======================================")

print(
    "Frames:",
    len(frames)
)

print(
    "Final iteration:",
    simulator.iteration
)

print(
    "Completed:",
    f"{simulator.completed_tasks}/{config.num_tasks}"
)

print(
    "Distance:",
    sum(
        agent.total_distance
        for agent in simulator.environment.agents
    )
)

print(
    "\nSaved:"
)

print(
    output_path
)
