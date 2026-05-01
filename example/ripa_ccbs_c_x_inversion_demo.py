"""C++ CCBS x-inversion benchmark/demo for a 6x6 atom array.

The demo places a 6x6 atom array on storage sites inside a larger RIPA grid:
for the default period=2/margin=1 setup, atoms sit on odd sites of a 13x13
grid and the even sites act as transport highways.  The labeled target is an
x-inversion, so the atom at storage coordinate (x, y) ends at
(side - 1 - x, y).

The full 36-agent one-shot CCBS search is a useful stress case but usually
times out with the current exact backend.  The default demo therefore performs
the same labeled inversion as CCBS-planned pair swaps, then launches
geometrically independent swaps in the same master time layer. Adjacent columns
share highway lanes, so the async layers use column parity groups. By default
the GIF compares that async schedule against the serialized version on one
shared physics clock.

Run:
    python example/ripa_ccbs_c_x_inversion_demo.py            # quick check -> render/
    python example/ripa_ccbs_c_x_inversion_demo.py --demo     # online quality -> demo/
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

from src.atom_config import Grid
from src.benchmark import BenchmarkResult, format_benchmark_table
from src.routing import RoutingRequest, Site
from src.scheduler.ripa_ccbs_c import RIPACCBSCScheduler
from src.moving_sequence import MovingSequence
from src.visualization import render_animation

GRID_SPACING_UM = 5.0
COLLISION_RADIUS_UM = 4.0
COLLISION_DT = 5e-6
DEFAULT_SIDE = 6
DEFAULT_PERIOD = 2
DEFAULT_MARGIN = 1
OUT_DIR = ROOT / "render"
PREFIX = "ripa_ccbs_c_x_inversion_6x6"


@dataclass(frozen=True)
class StageStats:
    index: int
    layer: int
    label: str
    atom_a: int
    atom_b: int
    start_time: float
    duration: float
    high_level_expanded: int
    high_level_generated: int
    segments: int


def storage_sites(side: int, period: int, margin: int) -> list[Site]:
    return [
        (margin + period * i, margin + period * j)
        for i in range(side)
        for j in range(side)
    ]


def grid_size(side: int, period: int, margin: int) -> int:
    return 2 * margin + period * (side - 1) + 1


def atom_id_for_storage_site(side: int, i: int, j: int) -> int:
    return i * side + j


def x_inverted_targets(src: list[Site], *, side: int, period: int, margin: int) -> list[Site]:
    lo = margin
    hi = margin + period * (side - 1)
    return [(hi - (i - lo), j) for i, j in src]


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
    time_limit: float,
    max_high_level_nodes: int,
) -> tuple[MovingSequence, RIPACCBSCScheduler]:
    dst = list(current)
    dst[atom_a], dst[atom_b] = dst[atom_b], dst[atom_a]
    request = RoutingRequest(grid=grid, src=current, dst=dst, labeled=True)
    scheduler = RIPACCBSCScheduler(
        request,
        collision_dt=COLLISION_DT,
        ccbs_precision=1e-7,
        time_limit=time_limit,
        max_high_level_nodes=max_high_level_nodes,
        high_level_order="conflicts",
        freeze_initial_target_atoms=True,
    )
    return scheduler.plan(), scheduler


def build_staged_x_inversion(
    *,
    side: int,
    period: int,
    margin: int,
    time_limit: float,
    max_high_level_nodes: int,
    async_layers: bool,
) -> tuple[MovingSequence, list[StageStats], list[Site]]:
    grid = Grid(
        N=grid_size(side, period, margin),
        d=GRID_SPACING_UM,
        rc=COLLISION_RADIUS_UM,
    )
    src = storage_sites(side, period, margin)
    targets = x_inverted_targets(src, side=side, period=period, margin=margin)
    master = MovingSequence(
        grid=grid,
        initial=RoutingRequest(grid, src, targets, True).initial,
        collision_dt=COLLISION_DT,
    )
    stage_stats: list[StageStats] = []

    stage_index = 0
    layer_index = 0
    for low in range(side // 2):
        high = side - 1 - low
        column_groups = (
            [list(range(0, side, 2)), list(range(1, side, 2))]
            if async_layers
            else [[j] for j in range(side)]
        )
        for columns in column_groups:
            current_by_atom = master.final_config().site_of_atom()
            current = [current_by_atom[atom_id] for atom_id in range(side * side)]
            planned: list[tuple[int, int, int, MovingSequence, RIPACCBSCScheduler]] = []
            for j in columns:
                atom_a = atom_id_for_storage_site(side, low, j)
                atom_b = atom_id_for_storage_site(side, high, j)
                stage_sequence, scheduler = solve_swap_stage(
                    grid=grid,
                    current=current,
                    atom_a=atom_a,
                    atom_b=atom_b,
                    time_limit=time_limit,
                    max_high_level_nodes=max_high_level_nodes,
                )
                planned.append((j, atom_a, atom_b, stage_sequence, scheduler))

            # All swaps in this group are independent under the storage/highway
            # geometry, so they share a master start time. Appending each CCBS
            # plan step still validates the combined RIPA timeline.
            offset = master.next_start_time()
            for _j, _atom_a, _atom_b, stage_sequence, _scheduler in planned:
                for step in stage_sequence.steps:
                    master.append(shifted_step(step, offset))

            for j, atom_a, atom_b, stage_sequence, scheduler in planned:
                solution = scheduler.ccbs_solution
                if solution is None or not solution.found:
                    raise RuntimeError(
                        f"stage {stage_index} did not produce a CCBS solution"
                    )
                stage_stats.append(
                    StageStats(
                        index=stage_index,
                        layer=layer_index,
                        label=f"col {j}, rows {low}<->{high}",
                        atom_a=atom_a,
                        atom_b=atom_b,
                        start_time=offset,
                        duration=stage_sequence.total_duration(),
                        high_level_expanded=solution.high_level_expanded,
                        high_level_generated=solution.high_level_generated,
                        segments=sum(
                            len(atom.segments)
                            for atom in stage_sequence.ensemble.atomtrajs
                        ),
                    )
                )
                stage_index += 1
            layer_index += 1

    assert_x_inversion(master, targets)
    return master, stage_stats, targets


def run_inversion_mode(
    *,
    name: str,
    side: int,
    period: int,
    margin: int,
    time_limit: float,
    max_high_level_nodes: int,
    async_layers: bool,
) -> tuple[BenchmarkResult, list[StageStats]]:
    started = time.perf_counter()
    try:
        sequence, stats, _targets = build_staged_x_inversion(
            side=side,
            period=period,
            margin=margin,
            time_limit=time_limit,
            max_high_level_nodes=max_high_level_nodes,
            async_layers=async_layers,
        )
        planning_time = time.perf_counter() - started
        return (
            BenchmarkResult(
                name=name,
                ok=True,
                total_time_s=sequence.total_duration(),
                planning_time_s=planning_time,
                steps=len(sequence.steps),
                segments=sum(len(atom.segments) for atom in sequence.ensemble.atomtrajs),
                clock_cycles=len({stat.layer for stat in stats}),
                sequence=sequence,
            ),
            stats,
        )
    except Exception as exc:
        return (
            BenchmarkResult(
                name=name,
                ok=False,
                planning_time_s=time.perf_counter() - started,
                error=exc,
            ),
            [],
        )


def try_one_shot(
    *,
    side: int,
    period: int,
    margin: int,
    time_limit: float,
    max_high_level_nodes: int,
) -> None:
    grid = Grid(
        N=grid_size(side, period, margin),
        d=GRID_SPACING_UM,
        rc=COLLISION_RADIUS_UM,
    )
    src = storage_sites(side, period, margin)
    dst = x_inverted_targets(src, side=side, period=period, margin=margin)
    request = RoutingRequest(grid=grid, src=src, dst=dst, labeled=True)
    scheduler = RIPACCBSCScheduler(
        request,
        collision_dt=COLLISION_DT,
        ccbs_precision=1e-7,
        time_limit=time_limit,
        max_high_level_nodes=max_high_level_nodes,
        high_level_order="conflicts",
    )
    sequence = scheduler.plan()
    assert_x_inversion(sequence, dst)
    solution = scheduler.ccbs_solution
    assert solution is not None
    print(
        "one-shot solved: "
        f"duration_us={sequence.total_duration() * 1e6:.3f}, "
        f"expanded={solution.high_level_expanded}, "
        f"generated={solution.high_level_generated}"
    )


def print_stage_table(stats: list[StageStats]) -> None:
    print(
        "stage  layer  start_us   swap                 atoms      "
        "duration_us  expanded  generated  segments"
    )
    for stat in stats:
        print(
            f"{stat.index:>5}  {stat.layer:>5}  "
            f"{stat.start_time * 1e6:>8.3f}  "
            f"{stat.label:<20} "
            f"{stat.atom_a:>2}<->{stat.atom_b:<2} "
            f"{stat.duration * 1e6:>11.3f}  "
            f"{stat.high_level_expanded:>8}  "
            f"{stat.high_level_generated:>9}  "
            f"{stat.segments:>8}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="C++ CCBS 6x6 x-inversion demo.")
    parser.add_argument("--side", type=int, default=DEFAULT_SIDE)
    parser.add_argument("--period", type=int, default=DEFAULT_PERIOD)
    parser.add_argument("--margin", type=int, default=DEFAULT_MARGIN)
    parser.add_argument("--time-limit", type=float, default=5.0)
    parser.add_argument("--max-high-level-nodes", type=int, default=50_000)
    parser.add_argument(
        "--serial",
        action="store_true",
        help="run only the serialized pair-swap schedule",
    )
    parser.add_argument(
        "--async-only",
        action="store_true",
        help="run only the async parity-layered schedule",
    )
    parser.add_argument(
        "--try-one-shot",
        action="store_true",
        help="try the full labeled CCBS inversion in one search, then exit",
    )
    parser.add_argument(
        "--strict-one-shot",
        action="store_true",
        help="return a non-zero exit code if --try-one-shot does not solve",
    )
    parser.add_argument(
        "--demo", action="store_true",
        help="render a high-quality GIF into demo/ instead of a quick render/ check",
    )
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="print the benchmark table only and skip rendering",
    )
    parser.add_argument(
        "--show-stages",
        action="store_true",
        help="also print every CCBS-planned pair-swap stage",
    )
    parser.add_argument("--quality", choices=("speed", "quality"), default=None)
    parser.add_argument("--fps", type=int, default=None)
    parser.add_argument("--time-dilation", type=float, default=1_000.0)
    parser.add_argument("--show-atom-ids", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )
    args = parser.parse_args()

    if args.serial and args.async_only:
        raise SystemExit("--serial and --async-only are mutually exclusive")
    if args.side % 2 != 0:
        raise SystemExit("x-inversion pair staging requires an even --side")
    if args.period < 1:
        raise SystemExit("--period must be positive")
    if args.margin < 0:
        raise SystemExit("--margin must be non-negative")

    if args.try_one_shot:
        try:
            try_one_shot(
                side=args.side,
                period=args.period,
                margin=args.margin,
                time_limit=args.time_limit,
                max_high_level_nodes=args.max_high_level_nodes,
            )
        except RuntimeError as exc:
            print(f"one-shot did not solve: {exc}")
            if args.strict_one_shot:
                raise SystemExit(1) from exc
        return

    if args.serial:
        modes = [("serial CCBS swaps", False)]
    elif args.async_only:
        modes = [("async parity CCBS", True)]
    else:
        modes = [
            ("async parity CCBS", True),
            ("serial CCBS swaps", False),
        ]

    print(
        f"N={grid_size(args.side, args.period, args.margin)}, "
        f"storage_period={args.period}, atoms={args.side * args.side} "
        f"({args.side}x{args.side}), task=x-inversion (labeled)"
    )
    runs = [
        run_inversion_mode(
            name=name,
            side=args.side,
            period=args.period,
            margin=args.margin,
            time_limit=args.time_limit,
            max_high_level_nodes=args.max_high_level_nodes,
            async_layers=async_layers,
        )
        for name, async_layers in modes
    ]
    results = [result for result, _stats in runs]
    stats_by_name = {
        result.name: stats
        for result, stats in runs
        if result.ok
    }

    print(format_benchmark_table(results))
    if args.show_stages:
        for result in results:
            if result.ok:
                print(f"\n{result.name} stages:")
                print_stage_table(stats_by_name[result.name])

    failures = [result for result in results if not result.ok]
    if failures:
        for result in failures:
            print(f"{result.name} error: {type(result.error).__name__}: {result.error}")
        raise SystemExit(1)

    if args.no_render:
        return

    by_name = {result.name: result.sequence for result in results if result.sequence is not None}
    panels = {name: by_name[name] for name, _async_layers in modes if name in by_name}
    if not panels:
        raise SystemExit("no schedule produced a valid sequence")

    if args.output is not None:
        out_path = args.output
    else:
        out_dir = ROOT / "demo" if args.demo else OUT_DIR
        suffix = "_serial" if args.serial else "_async" if args.async_only else ""
        out_path = out_dir / f"{PREFIX}{suffix}.gif"

    quality = args.quality or ("quality" if args.demo else "speed")
    fps = args.fps or (20 if args.demo else 8)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    render_animation(
        panels,
        out_path,
        view="benchmark",
        quality=quality,
        fps=fps,
        time_dilation=args.time_dilation,
        hold_seconds=1.5,
        planned_trajectory="next",
        show_atom_ids=args.show_atom_ids,
        show_routing_on_start=True,
        title=f"C++ CCBS x-inversion ({args.side}x{args.side})",
    )
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
