"""C++ CCBS x-inversion benchmark.

A 6x6 atom array on period-2 storage sites in a 24x24 RIPA grid performs an
x-inversion: every atom on the left swaps with its mirror on the right.

Compares two C++ CCBS variants on one shared physics clock:

* pairwise: per-column reciprocal swap stages chained sequentially;
* windowed: groups small nested swap windows for one-shot CCBS.

Usage:
    python example/ripa_ccbs_c_x_inversion_demo.py            # quick check -> render/
    python example/ripa_ccbs_c_x_inversion_demo.py --demo     # online quality -> demo/
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from example.inversion_6x6_benchmark import x_gradient_colors, x_inverted_targets
from src.atom_config import Grid
from src.benchmark import benchmark_schedulers, format_benchmark_table
from src.routing import RoutingRequest, centered_storage_square
from src.scheduler.base import AsyncScheduler
from src.scheduler.ccbs_c import RIPACCBSCScheduler, RIPACCBSWindowedScheduler
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

CCBS_KWARGS = dict(
    collision_dt=COLLISION_DT,
    ccbs_precision=1e-6,
    time_limit=TIME_LIMIT,
    max_high_level_nodes=MAX_HIGH_LEVEL_NODES,
    high_level_order="conflicts",
)


def _shifted_step(step, offset: float):
    return replace(
        step,
        start_time=float(getattr(step, "start_time", 0.0)) + offset,
        segments=tuple(
            replace(spec, start_time=float(spec.start_time) + offset)
            for spec in step.segments
        ),
    )


class PairwiseXInversionScheduler(AsyncScheduler):
    """Plan an x-inversion as parity layers of independent column swaps."""

    def _plan(self) -> None:
        side = int(round(len(self.request.src) ** 0.5))
        for low in range(side // 2):
            high = side - 1 - low
            for columns in (range(0, side, 2), range(1, side, 2)):
                site_by_atom = self.sequence.final_config().site_of_atom()
                current = [site_by_atom[i] for i in range(side * side)]
                stages = [
                    self._solve_swap(current, low * side + col, high * side + col)
                    for col in columns
                ]
                offset = self.sequence.next_start_time()
                for stage in stages:
                    for step in stage.steps:
                        self.sequence.append(_shifted_step(step, offset))

    def _solve_swap(self, current, atom_a, atom_b):
        dst = list(current)
        dst[atom_a], dst[atom_b] = dst[atom_b], dst[atom_a]
        request = RoutingRequest(
            grid=self.request.grid, src=current, dst=dst, labeled=True
        )
        scheduler = RIPACCBSCScheduler(
            request,
            **CCBS_KWARGS,
            freeze_initial_target_atoms=True,
        )
        return scheduler.plan()


# Dict order = panel order. Keys also become panel labels.
SCHEDULERS = {
    "C++ CCBS pairwise": PairwiseXInversionScheduler,
    "C++ CCBS windowed": lambda req: RIPACCBSWindowedScheduler(
        req,
        **CCBS_KWARGS,
        grouped_time_limit=15.0,
        grouped_max_high_level_nodes=100_000,
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="C++ CCBS x-inversion benchmark."
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="render a high-quality GIF into demo/ instead of a quick render/ check",
    )
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="print the benchmark table only and skip rendering",
    )
    args = parser.parse_args()

    grid = Grid(N=N, d=GRID_SPACING_UM, rc=COLLISION_RADIUS_UM)
    src = centered_storage_square(N, TARGET_SIDE, STORAGE_PERIOD)
    dst = x_inverted_targets(src)
    request = RoutingRequest(grid=grid, src=src, dst=dst, labeled=True)
    print(
        f"N={N}, storage_period={STORAGE_PERIOD}, "
        f"atoms={len(src)} ({TARGET_SIDE}x{TARGET_SIDE}), task=x-inversion"
    )

    results = benchmark_schedulers(request, SCHEDULERS, validate_dt=COLLISION_DT)
    print(format_benchmark_table(results))

    failures = [r for r in results if not r.ok]
    if failures:
        for r in failures:
            print(f"{r.name} error: {type(r.error).__name__}: {r.error}")
        raise SystemExit(1)

    if args.no_render:
        return

    by_name = {r.name: r.sequence for r in results}
    panels = {name: by_name[name] for name in SCHEDULERS}

    if args.demo:
        out_path = ROOT / "demo" / f"{PREFIX}.gif"
        quality = "quality"
    else:
        out_path = ROOT / "render" / f"{PREFIX}.gif"
        quality = "speed"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    render_animation(
        panels,
        out_path,
        view="benchmark",
        quality=quality,
        time_dilation=5e3,
        hold_seconds=1.5,
        atom_colors=x_gradient_colors(src),
        show_routing_on_start=False,
        title=f"C++ CCBS x-inversion benchmark ({TARGET_SIDE}x{TARGET_SIDE})",
    )
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
