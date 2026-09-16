from pathlib import Path
import sys

import plotly.graph_objects as go


PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

SRC_PATH = (
    PROJECT_ROOT
    / "src"
)

sys.path.insert(
    0,
    str(SRC_PATH),
)


from mrta.config import SimulationConfig
from mrta.coordinated_allocators import (
    HungarianPathAllocator,
)
from mrta.obstacle_simulator import (
    ObstacleSimulator,
)


# ==========================================================
# CONFIGURATION
# ==========================================================

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


# ==========================================================
# RECORD SIMULATION
# ==========================================================

history = []


def record_state():

    agents = []

    for agent in simulator.environment.agents:

        agents.append(
            {
                "id": agent.agent_id,
                "x": float(
                    agent.position[0]
                ),
                "y": float(
                    agent.position[1]
                ),
                "tasks_completed":
                    agent.tasks_completed,
            }
        )

    tasks = []

    for task in simulator.environment.tasks:

        tasks.append(
            {
                "id": task.task_id,
                "x": float(
                    task.position[0]
                ),
                "y": float(
                    task.position[1]
                ),
                "active": task.active,
            }
        )

    total_distance = sum(
        agent.total_distance
        for agent
        in simulator.environment.agents
    )

    history.append(
        {
            "iteration":
                simulator.iteration,

            "agents":
                agents,

            "tasks":
                tasks,

            "completed":
                simulator.completed_tasks,

            "distance":
                total_distance,

            "blocked":
                simulator.blocked_moves,

            "redundancy":
                (
                    simulator.redundant_target_events
                    /
                    simulator.total_target_assignments
                    if
                    simulator.total_target_assignments > 0
                    else 0.0
                ),
        }
    )


record_state()


while (
    simulator.iteration
    < config.max_iterations
    and not simulator.environment
    .all_tasks_completed()
):

    simulator.step()

    record_state()


# ==========================================================
# OBSTACLE COORDINATES
# ==========================================================

obstacle_x = []
obstacle_y = []
obstacle_z = []

grid = (
    simulator.environment
    .grid_map
    .obstacles
)

for y in range(
    config.grid_height
):
    for x in range(
        config.grid_width
    ):

        if grid[y, x]:

            obstacle_x.append(x)
            obstacle_y.append(y)
            obstacle_z.append(0.15)


# ==========================================================
# FLOOR GRID
# ==========================================================

floor_x = []
floor_y = []
floor_z = []

for y in range(
    config.grid_height
):
    for x in range(
        config.grid_width
    ):

        floor_x.append(x)
        floor_y.append(y)
        floor_z.append(0)


# ==========================================================
# CREATE FRAME
# ==========================================================

def create_frame(
    state,
):

    agents = state[
        "agents"
    ]

    active_tasks = [
        task
        for task
        in state["tasks"]
        if task["active"]
    ]

    completed_tasks = [
        task
        for task
        in state["tasks"]
        if not task["active"]
    ]

    robot_trace = (
        go.Scatter3d(
            x=[
                agent["x"]
                for agent
                in agents
            ],
            y=[
                agent["y"]
                for agent
                in agents
            ],
            z=[
                1.0
                for _
                in agents
            ],

            mode="markers+text",

            text=[
                f"R{agent['id']}"
                for agent
                in agents
            ],

            textposition="top center",

            marker=dict(
                size=7,
                symbol="circle",
            ),

            name="Robots",
        )
    )

    task_trace = (
        go.Scatter3d(
            x=[
                task["x"]
                for task
                in active_tasks
            ],
            y=[
                task["y"]
                for task
                in active_tasks
            ],
            z=[
                0.65
                for _
                in active_tasks
            ],

            mode="markers+text",

            text=[
                f"T{task['id']}"
                for task
                in active_tasks
            ],

            textposition="top center",

            marker=dict(
                size=6,
                symbol="diamond",
            ),

            name="Active Tasks",
        )
    )

    completed_trace = (
        go.Scatter3d(
            x=[
                task["x"]
                for task
                in completed_tasks
            ],
            y=[
                task["y"]
                for task
                in completed_tasks
            ],
            z=[
                0.25
                for _
                in completed_tasks
            ],

            mode="markers",

            marker=dict(
                size=4,
                symbol="x",
            ),

            name="Completed Tasks",
        )
    )

    obstacle_trace = (
        go.Scatter3d(
            x=obstacle_x,
            y=obstacle_y,
            z=obstacle_z,

            mode="markers",

            marker=dict(
                size=6,
                symbol="square",
            ),

            name="Obstacles",
        )
    )

    return [
        obstacle_trace,
        completed_trace,
        task_trace,
        robot_trace,
    ]


# ==========================================================
# INITIAL DATA
# ==========================================================

initial_data = create_frame(
    history[0]
)


# ==========================================================
# ANIMATION FRAMES
# ==========================================================

frames = []

for state in history:

    frame = go.Frame(
        data=create_frame(
            state
        ),

        name=str(
            state["iteration"]
        ),

        layout=go.Layout(
            title=(
                "AdaptiveMRTA — "
                "Hungarian + A*"
                "<br>"
                f"Iteration: "
                f"{state['iteration']} | "
                f"Tasks: "
                f"{state['completed']}/"
                f"{config.num_tasks} | "
                f"Distance: "
                f"{state['distance']:.0f} | "
                f"Blocked: "
                f"{state['blocked']} | "
                f"Redundancy: "
                f"{state['redundancy']:.3f}"
            )
        ),
    )

    frames.append(
        frame
    )


# ==========================================================
# SLIDER
# ==========================================================

slider_steps = []

for state in history:

    slider_steps.append(
        {
            "args": [
                [
                    str(
                        state[
                            "iteration"
                        ]
                    )
                ],
                {
                    "frame": {
                        "duration": 100,
                        "redraw": True,
                    },
                    "mode":
                        "immediate",
                },
            ],

            "label": str(
                state[
                    "iteration"
                ]
            ),

            "method":
                "animate",
        }
    )


# ==========================================================
# FIGURE
# ==========================================================

fig = go.Figure(
    data=initial_data,
    frames=frames,
)


fig.update_layout(

    title=(
        "AdaptiveMRTA — "
        "3D Multi-Robot Task Allocation"
        "<br>"
        "Hungarian Assignment + "
        "A* Navigation"
    ),

    scene=dict(

        xaxis=dict(
            title="X",
            range=[
                -1,
                config.grid_width + 1,
            ],
        ),

        yaxis=dict(
            title="Y",
            range=[
                -1,
                config.grid_height + 1,
            ],
        ),

        zaxis=dict(
            title="Height",
            range=[
                0,
                5,
            ],
        ),

        aspectmode="manual",

        aspectratio=dict(
            x=1.8,
            y=1,
            z=0.35,
        ),

        camera=dict(
            eye=dict(
                x=1.5,
                y=-1.7,
                z=1.4,
            )
        ),
    ),

    legend=dict(
        x=0.01,
        y=0.99,
    ),

    updatemenus=[
        {
            "type":
                "buttons",

            "showactive":
                False,

            "buttons": [

                {
                    "label":
                        "▶ Play",

                    "method":
                        "animate",

                    "args": [
                        None,
                        {
                            "frame": {
                                "duration":
                                    250,
                                "redraw":
                                    True,
                            },

                            "transition": {
                                "duration":
                                    100,
                            },

                            "fromcurrent":
                                True,
                        },
                    ],
                },

                {
                    "label":
                        "⏸ Pause",

                    "method":
                        "animate",

                    "args": [
                        [None],
                        {
                            "frame": {
                                "duration":
                                    0,
                                "redraw":
                                    False,
                            },

                            "mode":
                                "immediate",
                        },
                    ],
                },
            ],
        }
    ],

    sliders=[
        {
            "active": 0,
            "currentvalue": {
                "prefix":
                    "Iteration: "
            },
            "steps":
                slider_steps,
        }
    ],

    margin=dict(
        l=0,
        r=0,
        b=0,
        t=90,
    ),
)


# ==========================================================
# SAVE
# ==========================================================

output_dir = (
    PROJECT_ROOT
    / "results"
    / "visualizations"
)

output_dir.mkdir(
    parents=True,
    exist_ok=True,
)

output_path = (
    output_dir
    / "mrta_3d_hungarian.html"
)


fig.write_html(
    output_path,
    auto_open=True,
)


print(
    "\n========================================"
)

print(
    "3D MRTA VISUALIZATION CREATED"
)

print(
    "========================================"
)

print(
    f"Iterations: "
    f"{simulator.iteration}"
)

print(
    f"Tasks completed: "
    f"{simulator.completed_tasks}/"
    f"{config.num_tasks}"
)

print(
    "Total distance:",
    sum(
        agent.total_distance
        for agent
        in simulator.environment.agents
    ),
)

print(
    "Blocked moves:",
    simulator.blocked_moves,
)

print(
    "\nOpen:"
)

print(
    output_path
)
