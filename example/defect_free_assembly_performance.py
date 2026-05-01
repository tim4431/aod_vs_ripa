"""Average defect-free assembly performance over many initial arrangements.

Usage:
    python example/defect_free_assembly_performance.py
"""

from __future__ import annotations

from collections import Counter
from statistics import mean, pstdev

from defect_free_assembly_benchmark import (
    ATOM_COUNT,
    COLLISION_RADIUS_UM,
    GRID_SPACING_UM,
    N,
    SCHEDULERS,
    TARGET_SIDE,
)
from src.atom_config import Grid
from src.benchmark import benchmark_schedulers
from src.routing import (
    RoutingRequest,
    centered_storage_square,
    random_sample_sites,
    storage_subgrid,
)

START_SEED = 0
SEED_COUNT = 100


grid = Grid(N=N, d=GRID_SPACING_UM, rc=COLLISION_RADIUS_UM)
storage = storage_subgrid(N, 1)
dst = centered_storage_square(N, TARGET_SIDE, 1)

records = {name: [] for name in SCHEDULERS}
failures = {name: Counter() for name in SCHEDULERS}
for seed in range(START_SEED, START_SEED + SEED_COUNT):
    src = random_sample_sites(storage, ATOM_COUNT, seed)
    request = RoutingRequest(grid=grid, src=src, dst=dst, labeled=False)
    for result in benchmark_schedulers(request, SCHEDULERS):
        if result.ok:
            records[result.name].append(result)
        else:
            failures[result.name][type(result.error).__name__] += 1

rows = [
    (
        "scheduler",
        "ok/total",
        "avg_us",
        "std_us",
        "avg_plan_ms",
        "avg_steps",
        "avg_cycles",
        "avg_segments",
        "failures",
    )
]
for name in SCHEDULERS:
    ok = records[name]
    total_times = [result.total_time_us for result in ok]
    plan_times = [result.planning_time_s * 1e3 for result in ok]
    steps = [float(result.steps) for result in ok]
    cycles = [
        float(result.clock_cycles)
        for result in ok
        if result.clock_cycles is not None
    ]
    segments = [float(result.segments) for result in ok]
    failure_text = ", ".join(
        f"{error}:{count}" for error, count in failures[name].most_common()
    )
    rows.append(
        (
            name,
            f"{len(ok)}/{SEED_COUNT}",
            f"{mean(total_times):.3f}" if total_times else "-",
            f"{pstdev(total_times):.3f}" if len(total_times) > 1 else (
                "0.000" if total_times else "-"
            ),
            f"{mean(plan_times):.2f}" if plan_times else "-",
            f"{mean(steps):.3f}" if steps else "-",
            f"{mean(cycles):.3f}" if cycles else "-",
            f"{mean(segments):.3f}" if segments else "-",
            failure_text or "-",
        )
    )

print(
    "defect-free assembly performance: "
    f"atoms={ATOM_COUNT}, seeds={START_SEED}..{START_SEED + SEED_COUNT - 1}"
)
print("averages are over successful runs; failures are counted separately")
widths = [max(len(row[col]) for row in rows) for col in range(len(rows[0]))]
for row in rows:
    print("  ".join(value.ljust(widths[col]) for col, value in enumerate(row)))
