"""Manual AOD vs C++ CCBS windowed x-inversion benchmark.

Two ways to invert a side=6 array on the same shared physics clock,
rendered for both the 6x6 and the 1x6 (single-row) variants:

* `AOD manual` -- hand-rolled column moves. Phase 1 slides five columns
  (in user pos notation: 5->-1, 0->5, 4->0, 1->4, 3->1) using one
  reserved scratch slot just outside the left edge of the array. After
  phase 1 the columns read 5,4,3,2,_,1,0 across positions -1..5. Phase
  2 is one synchronous AOD step that shifts the left cluster (cols
  5,4,3,2) one slot to the right, filling the gap and landing the array
  at the centered mirror configuration.

  Each phase-1 transport is wrapped in a 1-row j bias (bias up, transport
  in i, drop back down) so the AOD trap clears the central storage rows
  during transit. The 1x6 generalization is automatic: the j set just
  becomes a single row.

* `C++ CCBS windowed` -- the windowed CCBS solver from
  `ripa_ccbs_c_x_inversion_demo.py` on the same labeled centered mirror
  inversion (its native efficient case).

Usage:
    python example/inversion_6x6_ccbs_aod.py            # quick check -> render/
    python example/inversion_6x6_ccbs_aod.py --demo     # online quality -> demo/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from example.inversion_6x6_benchmark import x_gradient_colors, x_inverted_targets
from src.atom_config import Grid
from src.benchmark import benchmark_schedulers, format_benchmark_table
from src.movement import AODStep
from src.routing import RoutingRequest, centered_storage_square
from src.scheduler.base import SyncScheduler
from src.scheduler.ccbs_c import RIPACCBSWindowedScheduler
from src.scheduler.heuristic.ripa_pebroute import RIPAPebRouteScheduler
from src.scheduler.logL_1d.aod_logL_1d import AODLogL1DPerLineScheduler
from src.visualization import render_animation

N = 24
TARGET_SIDE = 6
STORAGE_PERIOD = 2
GRID_SPACING_UM = 5.0
COLLISION_RADIUS_UM = 4.0

CCBS_KWARGS = dict(
    ccbs_precision=1e-6,
    time_limit=5.0,
    max_high_level_nodes=50_000,
    high_level_order="conflicts",
)


class ManualAODXInversionScheduler(SyncScheduler):
    """Hand-rolled AOD inversion: 5 biased column moves + 1 synchronous shift.

    Currently specialized to side = 6 with one reserved slot outside the
    left edge (pos -1).
    """

    def _plan(self) -> None:
        xs = sorted({i for i, _ in self.request.src})
        ys = sorted({j for _, j in self.request.src})
        if len(xs) != 6:
            raise NotImplementedError(
                "ManualAODXInversionScheduler currently assumes side = 6"
            )
        ys_t = tuple(ys)
        ys_biased = tuple(j + 1 for j in ys)
        step = xs[1] - xs[0]
        outside_left = xs[0] - step  # the reserved scratch slot at pos -1

        # Phase 1, in user column-index notation:
        #   (5, -1), (0, 5), (4, 0), (1, 4), (3, 1)
        # Each move uses bias-up / transport / drop so the moving column
        # flies through odd j rows and never collides with stationary atoms.
        phase1 = [
            (xs[5], outside_left),
            (xs[0], xs[5]),
            (xs[4], xs[0]),
            (xs[1], xs[4]),
            (xs[3], xs[1]),
        ]
        for x_old, x_new in phase1:
            self._aod(selected_axis_1=(x_old,), selected_axis_2=ys_t,
                      new_axis_1=(x_old,), new_axis_2=ys_biased)
            self._aod(selected_axis_1=(x_old,), selected_axis_2=ys_biased,
                      new_axis_1=(x_new,), new_axis_2=ys_biased)
            self._aod(selected_axis_1=(x_new,), selected_axis_2=ys_biased,
                      new_axis_1=(x_new,), new_axis_2=ys_t)

        # Phase 2: one synchronous AOD shifting the left cluster
        # (cols at -1, 0, 1, 2) by one storage slot to the right. Cols 1
        # and 0 (at xs[4] and xs[5]) are outside the lattice and stay put.
        self._aod(
            selected_axis_1=(outside_left, xs[0], xs[1], xs[2]),
            selected_axis_2=ys_t,
            new_axis_1=(xs[0], xs[1], xs[2], xs[3]),
            new_axis_2=ys_t,
        )

    def _aod(self, *, selected_axis_1, selected_axis_2, new_axis_1, new_axis_2):
        self.append_step(AODStep(
            start_time=self.sequence.next_start_time(),
            selected_axis_1=selected_axis_1,
            selected_axis_2=selected_axis_2,
            new_axis_1=new_axis_1,
            new_axis_2=new_axis_2,
        ))


# Dict order = panel order. Keys also become panel labels.
SCHEDULERS_6X6 = {
    "RIPA_ccbs": lambda req: RIPACCBSWindowedScheduler(
        req,
        **CCBS_KWARGS,
        grouped_time_limit=15.0,
        grouped_max_high_level_nodes=100_000,
    ),
    "AOD manual": ManualAODXInversionScheduler,
    "AOD logL_1d": AODLogL1DPerLineScheduler,
}

SCHEDULERS_1X6 = {
    "RIPA_pebroute": RIPAPebRouteScheduler,
    "AOD manual": ManualAODXInversionScheduler,
    "AOD logL_1d": AODLogL1DPerLineScheduler,
}


def _centered_row(N: int, side: int, period: int) -> list[tuple[int, int]]:
    """Centered 1xside line on the storage subgrid (mirrors centered_storage_square)."""
    storage = list(range(0, N, period))
    center = (N - 1) / 2
    s = min(
        range(len(storage) - side + 1),
        key=lambda i: abs((storage[i] + storage[i + side - 1]) / 2 - center),
    )
    j = min(storage, key=lambda x: abs(x - center))
    return [(storage[i], j) for i in range(s, s + side)]


def _run(label: str, src, grid: Grid, schedulers, args: argparse.Namespace) -> None:
    dst = x_inverted_targets(src)
    request = RoutingRequest(grid=grid, src=src, dst=dst, labeled=True)
    print(
        f"\n[{label}] N={N}, storage_period={STORAGE_PERIOD}, "
        f"atoms={len(src)}, task=x-inversion"
    )

    results = benchmark_schedulers(request, schedulers)
    print(format_benchmark_table(results))

    failures = [r for r in results if not r.ok]
    if failures:
        for r in failures:
            print(f"{r.name} error: {type(r.error).__name__}: {r.error}")
        raise SystemExit(1)

    if args.no_render:
        return

    by_name = {r.name: r.sequence for r in results}
    panels = {name: by_name[name] for name in schedulers}

    prefix = f"inversion_{label}_ripa_aod"
    if args.demo:
        out_path = ROOT / "demo" / f"{prefix}.gif"
        quality = "quality"
    else:
        out_path = ROOT / "render" / f"{prefix}.gif"
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
        title=f"AOD vs CCBS x-inversion ({label})",
    )
    print(f"wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Manual AOD vs C++ CCBS windowed x-inversion benchmark."
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="render high-quality GIFs into demo/ instead of quick render/ checks",
    )
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="print the benchmark tables only and skip rendering",
    )
    args = parser.parse_args()

    grid = Grid(N=N, d=GRID_SPACING_UM, rc=COLLISION_RADIUS_UM)
    _run("6x6", centered_storage_square(N, TARGET_SIDE, STORAGE_PERIOD), grid,
         SCHEDULERS_6X6, args)
    _run("1x6", _centered_row(N, TARGET_SIDE, STORAGE_PERIOD), grid,
         SCHEDULERS_1X6, args)


if __name__ == "__main__":
    main()
