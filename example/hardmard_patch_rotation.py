"""Hadamard code-patch rotation demo using AOD-style motion.

The paper in arXiv:2412.01391 implements the rotated-surface-code logical H
motion as:

    horizontal reflection + diagonal reflection = 90-degree patch rotation

This example demonstrates that atom moving pattern on a small square code patch.
The horizontal reflection is delegated to the existing projected 1D log-depth AOD
scheduler. The diagonal reflection is expanded into off-diagonal mirror-pair
rearrangements on constant-`i+j` anti-diagonals using the 45-degree AOD support
in `AOD1DProjectedScheduler`.

Usage:
    python example/hardmard_patch_rotation.py            # quick GIF -> render/
    python example/hardmard_patch_rotation.py --demo     # high quality -> demo/
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import matplotlib as mpl

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.atom_config import Grid
from src.benchmark import benchmark_schedulers, format_benchmark_table
from src.routing import RoutingRequest, centered_storage_square
from src.scheduler.base import SyncScheduler
from src.scheduler.logL_1d.aod_logL_1d import AODLogL1DPerLineScheduler
from src.visualization import render_animation

N = 24
PATCH_SIDE = 5
STORAGE_PERIOD = 4
GRID_SPACING_UM = 5.0
COLLISION_RADIUS_UM = 4.0

PREFIX = "hardmard_patch_rotation"


def hadamard_rotation_targets(src):
    """Target positions for horizontal reflection followed by diagonal reflection."""
    coords = sorted({i for i, _ in src})
    pivot_sum = coords[0] + coords[-1]
    return [(j, pivot_sum - i) for i, j in src]


def patch_rotation_colors(src):
    """Color atoms by initial polar angle around the patch center."""
    coords = sorted({i for i, _ in src})
    center = 0.5 * (coords[0] + coords[-1])
    cmap = mpl.colormaps["hsv"]
    colors = {}
    for atom_id, (i, j) in enumerate(src):
        theta = math.atan2(j - center, i - center)
        colors[atom_id] = cmap((theta + math.pi) / (2.0 * math.pi))
    return colors


@dataclass
class AODHadamardPatchRotationScheduler(SyncScheduler):
    """Two-reflection AOD demo for a square code-patch rotation."""

    storage_period: int = STORAGE_PERIOD

    def _plan(self) -> None:
        rows = self._validate_square_patch()
        pivot_sum = rows[0] + rows[-1]

        # Phase 1: native projected-1D AOD reflection along every column.
        mid = [(pivot_sum - i, j) for i, j in self.request.src]
        phase1 = AODLogL1DPerLineScheduler(
            RoutingRequest(
                grid=self.request.grid,
                src=self.request.src,
                dst=mid,
                labeled=True,
            ),
            axis="col",
        ).plan()
        self._append_rebased_steps(phase1.steps)

        # Phase 2: diagonal reflection. Reflection across the main diagonal
        # preserves `i + j` and reverses `i - j`, so each anti-diagonal is a
        # projected 1D AOD problem with a 45-degree lift lane.
        phase2 = AODLogL1DPerLineScheduler(
            RoutingRequest(
                grid=self.request.grid,
                src=mid,
                dst=self.request.dst,
                labeled=True,
            ),
            axis="anti_diag",
        ).plan()
        self._append_rebased_steps(phase2.steps)

    def _validate_square_patch(self) -> list[int]:
        rows = sorted({i for i, _ in self.request.src})
        cols = sorted({j for _, j in self.request.src})
        if rows != cols:
            raise ValueError("Hadamard patch demo expects identical row/col coordinates")
        expected = {(i, j) for i in rows for j in cols}
        if set(self.request.src) != expected:
            raise ValueError("Hadamard patch demo expects a filled square patch")
        if self.request.dst != hadamard_rotation_targets(self.request.src):
            raise ValueError("request dst does not match the 90-degree patch rotation")
        return rows

    def _append_rebased_steps(self, steps) -> None:
        for step in steps:
            self.append_step(replace(step, start_time=self.sequence.next_start_time()))


SCHEDULERS = {
    "AOD H rotation": AODHadamardPatchRotationScheduler,
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="AOD Hadamard code-patch rotation demo."
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
    src = centered_storage_square(N, PATCH_SIDE, STORAGE_PERIOD)
    dst = hadamard_rotation_targets(src)
    request = RoutingRequest(grid=grid, src=src, dst=dst, labeled=True)

    print(
        f"N={N}, storage_period={STORAGE_PERIOD}, atoms={len(src)} "
        f"({PATCH_SIDE}x{PATCH_SIDE}), task=Hadamard patch rotation"
    )
    results = benchmark_schedulers(request, SCHEDULERS, validate_dt=2e-7)
    print(format_benchmark_table(results))
    for result in results:
        if not result.ok:
            print(f"{result.name} error: {type(result.error).__name__}: {result.error}")

    failures = [result for result in results if not result.ok]
    if failures:
        raise SystemExit(1)

    if args.no_render:
        return

    by_name = {r.name: r.sequence for r in results if r.sequence is not None}
    panels = {name: by_name[name] for name in SCHEDULERS}

    if args.demo:
        out_path = ROOT / "demo" / f"{PREFIX}.gif"
        quality = "quality"
    else:
        out_path = ROOT / "render" / f"{PREFIX}.gif"
        quality = "speed"

    render_animation(
        panels,
        out_path,
        view="benchmark",
        quality=quality,
        time_dilation=5e3,
        hold_seconds=1.25,
        atom_colors=patch_rotation_colors(src),
        show_routing_on_start=True,
        title=f"Hadamard patch rotation ({PATCH_SIDE}x{PATCH_SIDE})",
    )
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
