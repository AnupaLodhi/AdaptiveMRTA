from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mrta.config import SimulationConfig
from mrta.scenario import Scenario
from mrta.obstacle_simulator import ObstacleSimulator

from mrta.coordinated_allocators import (
    HungarianPathAllocator,
    StatefulHungarianAllocator,
    PeriodicHungarianAllocator,
)


def evaluate(
    label,
    failure_schedule,
):
    config = SimulationConfig(
        num_agents=15,
        num_tasks=20,
        seed=42,
        max_iterations=500,
    )

    scenario = Scenario.generate(
        config=config,
        obstacle_density=0.20,
    )

    print()
    print("=" * 100)
    print(label)
    print("=" * 100)

    results = []

    # --------------------------------------------------------
    # Repeated Hungarian
    # --------------------------------------------------------

    allocator = HungarianPathAllocator()

    sim = ObstacleSimulator(
        config=config,
        allocator=allocator,
        obstacle_density=0.20,
        scenario=scenario,
        failure_schedule=failure_schedule,
    )

    start = time.perf_counter()
    result = sim.run()
    runtime = time.perf_counter() - start

    results.append(
        (
            "repeated",
            result,
            result["iterations"],
            runtime,
            None,
            None,
        )
    )

    # --------------------------------------------------------
    # Periodic intervals
    # --------------------------------------------------------

    for interval in [2, 4, 8]:

        allocator = (
            PeriodicHungarianAllocator(
                replan_interval=interval
            )
        )

        sim = ObstacleSimulator(
            config=config,
            allocator=allocator,
            obstacle_density=0.20,
            scenario=scenario,
            failure_schedule=failure_schedule,
        )

        start = time.perf_counter()
        result = sim.run()
        runtime = (
            time.perf_counter()
            - start
        )

        results.append(
            (
                f"periodic_{interval}",
                result,
                allocator.global_assignment_calls,
                runtime,
                allocator.periodic_replans,
                allocator.event_replans,
            )
        )

    # --------------------------------------------------------
    # Event/stateful only
    # --------------------------------------------------------

    allocator = (
        StatefulHungarianAllocator()
    )

    sim = ObstacleSimulator(
        config=config,
        allocator=allocator,
        obstacle_density=0.20,
        scenario=scenario,
        failure_schedule=failure_schedule,
    )

    start = time.perf_counter()
    result = sim.run()
    runtime = time.perf_counter() - start

    results.append(
        (
            "stateful",
            result,
            allocator.global_assignment_calls,
            runtime,
            None,
            None,
        )
    )

    print()
    print(
        f"{'Method':<14}"
        f"{'Iter':>7}"
        f"{'Dist':>9}"
        f"{'Calls':>8}"
        f"{'Blocked':>10}"
        f"{'Redund':>10}"
        f"{'Time':>9}"
        f"{'P':>6}"
        f"{'E':>6}"
    )

    print("-" * 100)

    for (
        name,
        result,
        calls,
        runtime,
        periodic,
        event,
    ) in results:

        p = (
            "-"
            if periodic is None
            else str(periodic)
        )

        e = (
            "-"
            if event is None
            else str(event)
        )

        print(
            f"{name:<14}"
            f"{result['iterations']:>7}"
            f"{result['total_distance']:>9.1f}"
            f"{calls:>8}"
            f"{result['blocked_moves']:>10}"
            f"{result['redundant_target_ratio']:>10.4f}"
            f"{runtime:>9.3f}"
            f"{p:>6}"
            f"{e:>6}"
        )

        assert result["success"]

        assert (
            result[
                "redundant_target_ratio"
            ]
            == 0.0
        )

    print()
    print("CASE: PASS")


evaluate(
    "STATIC — SEED 42",
    {},
)

evaluate(
    "20% MID-RUN FAILURE — SEED 42",
    {
        5: [0, 1, 2],
    },
)

print()
print("=" * 100)
print(
    "E5 PERIODIC REPLANNING "
    "VALIDATION: PASS"
)
print("=" * 100)
