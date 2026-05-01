"""C++ CCBS benchmark for a 6x6 RIPA x-inversion.

A fixed 6x6 atom array sits on period-2 storage sites centered in a 24x24 RIPA
grid.  The labeled task is an x-inversion: the atom initially on the left side
of the array must end on the matching right-side site, and vice versa.

The full 36-agent one-shot CCBS search is a useful stress case but usually
times out with the current exact backend. This benchmark compares:

* pairwise C++ CCBS: plan each pair swap, then launch independent columns in
  async parity layers;
* windowed C++ CCBS: group small nested swap windows and translate symmetric
  solutions across identical columns.

Both schedules are validated with the RIPA bang-bang trajectory validator.

Usage:
    python example/ripa_ccbs_c_x_inversion_demo.py        # benchmark -> render/
    python example/ripa_ccbs_c_x_inversion_demo.py --demo # online quality -> demo/
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from example.inversion_6x6_benchmark import x_gradient_colors, x_inverted_targets
from src.atom_config import Grid
from src.moving_sequence import MovingSequence
from src.routing import RoutingRequest, Site, centered_storage_square
from src.scheduler.ccbs_c import (
    CCBSWindowStats,
    RIPACCBSCScheduler,
    RIPACCBSWindowedScheduler,
)
from src.visualization import render_animation

N = 24
TARGET_SIDE = 6
STORAGE_PERIOD = 2
GRID_SPACING_UM = 5.0
COLLISION_RADIUS_UM = 4.0
COLLISION_DT = 5e-6
TIME_LIMIT = 5.0
MAX_HIGH_LEVEL_NODES = 50_000

PREFIX = "ripa_ccbs_x_inversion_benchmark"


@dataclass(frozen=True)
class StageStats:
    layer: int
    high_level_expanded: int
    high_level_generated: int
    segments: int


@dataclass(frozen=True)
class InversionBenchmarkResult:
    name: str
    ok: bool
    total_time_s: float = float("inf")
    planning_time_s: float = 0.0
    steps: int = 0
    segments: int = 0
    stages: int = 0
    grouped: int | None = None
    reused: int | None = None
    high_level_expanded: int = 0
    high_level_generated: int = 0
    sequence: MovingSequence | None = None
    error: Exception | None = None

    @property
    def total_time_us(self) -> float:
        return self.total_time_s * 1e6


def atom_id_for_storage_site(side: int, i: int, j: int) -> int:
    return i * side + j


def assert_x_inversion(sequence: MovingSequence, targets: list[Site]) -> None:
    report = sequence.validate(dt=COLLISION_DT)
    if not report.ok:
        raise RuntimeError(f"collision validation failed: {report}")

    site_by_atom = sequence.final_config().site_of_atom()
    for atom_id, target in enumerate(targets):
        if site_by_atom.get(atom_id) != tuple(target):
            raise RuntimeError(
                f"atom {atom_id} ended at {site_by_atom.get(atom_id)}, "
                f"expected {tuple(target)}"
            )


def shifted_step(step, offset: float):
    """Return a copy of a CCBS plan step shifted to the master timeline."""

    if not hasattr(step, "segments"):
        raise TypeError(f"expected a CCBS plan step with segments, got {type(step)!r}")
    return replace(
        step,
        start_time=float(getattr(step, "start_time", 0.0)) + offset,
        segments=tuple(
            replace(spec, start_time=float(spec.start_time) + offset)
            for spec in step.segments
        ),
    )


def solve_swap_stage(
    *,
    grid: Grid,
    current: list[Site],
    atom_a: int,
    atom_b: int,
) -> tuple[MovingSequence, RIPACCBSCScheduler]:
    dst = list(current)
    dst[atom_a], dst[atom_b] = dst[atom_b], dst[atom_a]
    request = RoutingRequest(grid=grid, src=current, dst=dst, labeled=True)
    scheduler = RIPACCBSCScheduler(
        request,
        collision_dt=COLLISION_DT,
        ccbs_precision=1e-7,
        time_limit=TIME_LIMIT,
        max_high_level_nodes=MAX_HIGH_LEVEL_NODES,
        high_level_order="conflicts",
        freeze_initial_target_atoms=True,
    )
    return scheduler.plan(), scheduler


def build_async_x_inversion(
    grid: Grid,
    src: list[Site],
    dst: list[Site],
) -> tuple[MovingSequence, list[StageStats]]:
    master = MovingSequence(
        grid=grid,
        initial=RoutingRequest(grid, src, dst, labeled=True).initial,
        collision_dt=COLLISION_DT,
    )
    stats: list[StageStats] = []

    layer = 0
    for low in range(TARGET_SIDE // 2):
        high = TARGET_SIDE - 1 - low
        for columns in (range(0, TARGET_SIDE, 2), range(1, TARGET_SIDE, 2)):
            current_by_atom = master.final_config().site_of_atom()
            current = [current_by_atom[atom_id] for atom_id in range(TARGET_SIDE**2)]
            planned: list[tuple[MovingSequence, RIPACCBSCScheduler]] = []

            for col in columns:
                stage_sequence, scheduler = solve_swap_stage(
                    grid=grid,
                    current=current,
                    atom_a=atom_id_for_storage_site(TARGET_SIDE, low, col),
                    atom_b=atom_id_for_storage_site(TARGET_SIDE, high, col),
                )
                planned.append((stage_sequence, scheduler))

            offset = master.next_start_time()
            for stage_sequence, _scheduler in planned:
                for step in stage_sequence.steps:
                    master.append(shifted_step(step, offset))

            for stage_sequence, scheduler in planned:
                solution = scheduler.ccbs_solution
                if solution is None or not solution.found:
                    raise RuntimeError(f"CCBS layer {layer} did not solve")
                stats.append(
                    StageStats(
                        layer=layer,
                        high_level_expanded=solution.high_level_expanded,
                        high_level_generated=solution.high_level_generated,
                        segments=sum(
                            len(atom.segments)
                            for atom in stage_sequence.ensemble.atomtrajs
                        ),
                    )
                )
            layer += 1

    assert_x_inversion(master, dst)
    return master, stats


def build_windowed_x_inversion(
    grid: Grid,
    src: list[Site],
    dst: list[Site],
) -> tuple[MovingSequence, list[CCBSWindowStats]]:
    request = RoutingRequest(grid=grid, src=src, dst=dst, labeled=True)
    scheduler = RIPACCBSWindowedScheduler(
        request,
        collision_dt=COLLISION_DT,
        ccbs_precision=1e-7,
        time_limit=TIME_LIMIT,
        max_high_level_nodes=MAX_HIGH_LEVEL_NODES,
        high_level_order="conflicts",
        grouped_time_limit=15.0,
        grouped_max_high_level_nodes=100_000,
    )
    sequence = scheduler.plan()
    assert_x_inversion(sequence, dst)
    return sequence, list(scheduler.window_stats)


def benchmark_inversion(
    grid: Grid,
    src: list[Site],
    dst: list[Site],
    *,
    only: str,
) -> list[InversionBenchmarkResult]:
    runners = {
        "pairwise": (
            "C++ CCBS pairwise",
            lambda: _benchmark_pairwise(grid, src, dst),
        ),
        "windowed": (
            "C++ CCBS windowed",
            lambda: _benchmark_windowed(grid, src, dst),
        ),
    }
    selected = ["pairwise", "windowed"] if only == "all" else [only]
    results: list[InversionBenchmarkResult] = []
    for key in selected:
        name, runner = runners[key]
        started = time.perf_counter()
        try:
            result = runner()
            results.append(
                replace(
                    result,
                    name=name,
                    planning_time_s=time.perf_counter() - started,
                )
            )
        except Exception as exc:
            results.append(
                InversionBenchmarkResult(
                    name=name,
                    ok=False,
                    planning_time_s=time.perf_counter() - started,
                    error=exc,
                )
            )
    return sorted(results, key=lambda result: (not result.ok, result.total_time_s))


def _benchmark_pairwise(
    grid: Grid,
    src: list[Site],
    dst: list[Site],
) -> InversionBenchmarkResult:
    sequence, stats = build_async_x_inversion(grid, src, dst)
    return InversionBenchmarkResult(
        name="",
        ok=True,
        total_time_s=sequence.total_duration(),
        steps=len(sequence.steps),
        segments=sum(len(atom.segments) for atom in sequence.ensemble.atomtrajs),
        stages=len(stats),
        high_level_expanded=sum(stat.high_level_expanded for stat in stats),
        high_level_generated=sum(stat.high_level_generated for stat in stats),
        sequence=sequence,
    )


def _benchmark_windowed(
    grid: Grid,
    src: list[Site],
    dst: list[Site],
) -> InversionBenchmarkResult:
    sequence, stats = build_windowed_x_inversion(grid, src, dst)
    return InversionBenchmarkResult(
        name="",
        ok=True,
        total_time_s=sequence.total_duration(),
        steps=len(sequence.steps),
        segments=sum(len(atom.segments) for atom in sequence.ensemble.atomtrajs),
        stages=len(stats),
        grouped=sum(stat.grouped for stat in stats),
        reused=sum(stat.reused for stat in stats),
        high_level_expanded=sum(stat.high_level_expanded for stat in stats),
        high_level_generated=sum(stat.high_level_generated for stat in stats),
        sequence=sequence,
    )


def format_inversion_benchmark_table(
    results: list[InversionBenchmarkResult],
) -> str:
    rows = [
        (
            "scheduler",
            "ok",
            "total_us",
            "plan_ms",
            "steps",
            "segments",
            "stages",
            "grouped",
            "reused",
            "expanded",
            "generated",
        ),
    ]
    for result in results:
        rows.append(
            (
                result.name,
                "yes" if result.ok else "no",
                f"{result.total_time_us:.3f}" if result.ok else "-",
                f"{result.planning_time_s * 1e3:.2f}",
                str(result.steps) if result.ok else "-",
                str(result.segments) if result.ok else "-",
                str(result.stages) if result.ok else "-",
                _optional_int(result.grouped) if result.ok else "-",
                _optional_int(result.reused) if result.ok else "-",
                str(result.high_level_expanded) if result.ok else "-",
                str(result.high_level_generated) if result.ok else "-",
            )
        )

    widths = [max(len(row[col]) for row in rows) for col in range(len(rows[0]))]
    return "\n".join(
        "  ".join(value.ljust(widths[col]) for col, value in enumerate(row))
        for row in rows
    )


def _optional_int(value: int | None) -> str:
    return "-" if value is None else str(value)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark C++ CCBS variants on a 6x6 x-inversion."
    )
    parser.add_argument(
        "--demo", action="store_true",
        help="render a high-quality GIF into demo/ instead of a quick render/ check",
    )
    parser.add_argument(
        "--no-render", action="store_true",
        help="print the summary only and skip rendering",
    )
    parser.add_argument(
        "--only",
        choices=("all", "pairwise", "windowed"),
        default="all",
        help="benchmark both schedulers, or only one variant",
    )
    args = parser.parse_args()

    grid = Grid(N=N, d=GRID_SPACING_UM, rc=COLLISION_RADIUS_UM)
    src = centered_storage_square(N, TARGET_SIDE, STORAGE_PERIOD)
    dst = x_inverted_targets(src)
    print(
        f"N={N}, storage_period={STORAGE_PERIOD}, "
        f"atoms={len(src)} ({TARGET_SIDE}x{TARGET_SIDE}), "
        "task=x-inversion"
    )

    results = benchmark_inversion(grid, src, dst, only=args.only)
    print(format_inversion_benchmark_table(results))

    failures = [result for result in results if not result.ok]
    if failures:
        for result in failures:
            print(f"{result.name} error: {type(result.error).__name__}: {result.error}")
        raise SystemExit(1)

    if args.no_render:
        return

    if args.demo:
        out_path = ROOT / "demo" / f"{PREFIX}.gif"
        quality = "quality"
    else:
        out_path = ROOT / "render" / f"{PREFIX}.gif"
        quality = "speed"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    panels = {
        result.name: result.sequence
        for result in results
        if result.ok and result.sequence is not None
    }
    render_animation(
        panels,
        out_path,
        view="benchmark",
        quality=quality,
        time_dilation=1e4,
        hold_seconds=1.5,
        atom_colors=x_gradient_colors(src),
        title=f"C++ CCBS x-inversion benchmark ({TARGET_SIDE}x{TARGET_SIDE})",
    )
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
