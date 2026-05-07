"""Hadamard code-patch rotation demo using AOD-style motion.

The paper in arXiv:2412.01391 implements the rotated-surface-code logical H
motion as:

    horizontal reflection + diagonal reflection = 90-degree patch rotation

This example demonstrates that atom moving pattern on a small square code patch.
The horizontal reflection is delegated to the existing projected 1D log-depth AOD
scheduler. The diagonal reflection uses one tilted-basis 1D inversion broadcast
across all constant-`i+j` anti-diagonals; empty AOD intersections are allowed
and simply do not move an atom.

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
from typing import Literal

import matplotlib as mpl

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.atom_config import Grid
from src.benchmark import benchmark_schedulers, format_benchmark_table
from src.atom_trajectory import AtomEnsemble
from src.movement import PHYS_A_MAX, Step, grid_accel_from_phys
from src.routing import RoutingRequest, centered_storage_square
from src.scheduler.aod_1d_projected import AODProjection, unit_vector
from src.scheduler.base import AsyncScheduler, SyncScheduler
from src.scheduler.logL_1d.aod_logL_1d import (
    AODLogL1DBroadcastLineScheduler,
    AODLogL1DPerLineScheduler,
)
from src.segments import bang_bang_duration, make_const_acc_segment
from src.visualization import render_animation

N = 24
PATCH_SIDE = 5
STORAGE_PERIOD = 4
GRID_SPACING_UM = 5.0
COLLISION_RADIUS_UM = 4.0

PREFIX = "hardmard_patch_rotation"
RIPAChannel = Literal["row", "col"]
RIPABatchMove = tuple[int, tuple[float, float], RIPAChannel]


def scaled_unit_vector(theta: float, scale: float) -> tuple[float, float]:
    ux, uy = unit_vector(theta)
    return scale * ux, scale * uy


COLUMN_PROJECTION = AODProjection(
    axis_1=unit_vector(0.0),
    axis_2=unit_vector(math.pi / 2.0),
    fixed_axis="axis_2",
    name="columns",
)
ANTI_DIAGONAL_PROJECTION = AODProjection(
    axis_1=scaled_unit_vector(math.pi / 4.0, 1.0 / math.sqrt(2.0)),
    axis_2=scaled_unit_vector(-math.pi / 4.0, 1.0 / math.sqrt(2.0)),
    fixed_axis="axis_1",
    name="anti_diagonals",
)


@dataclass
class RIPABatchStep(Step):
    """Manual same-start batch of independent RIPA moves."""

    start_time: float
    moves: tuple[RIPABatchMove, ...]

    def apply(self, ensemble: AtomEnsemble) -> None:
        accel = grid_accel_from_phys(PHYS_A_MAX, ensemble.grid.d)
        segments = []
        for atom_id, target, channel in self.moves:
            current = ensemble.atomtraj_by_id(atom_id).resting_position_at(
                self.start_time
            )
            self._check_channel(current, target, channel)
            if current == target:
                continue
            segments.append((
                atom_id,
                make_const_acc_segment(
                    current,
                    target,
                    self.start_time,
                    accel=accel,
                    channel=channel,
                ),
            ))
        ensemble.append_segments_batch(segments)

    def end_time(self, ensemble: AtomEnsemble) -> float:
        accel = grid_accel_from_phys(PHYS_A_MAX, ensemble.grid.d)
        duration = 0.0
        for atom_id, target, channel in self.moves:
            current = ensemble.atomtraj_by_id(atom_id).resting_position_at(
                self.start_time
            )
            self._check_channel(current, target, channel)
            duration = max(
                duration,
                bang_bang_duration(
                    math.hypot(target[0] - current[0], target[1] - current[1]),
                    accel,
                ),
            )
        return self.start_time + duration

    @staticmethod
    def _check_channel(
        current: tuple[float, float],
        target: tuple[float, float],
        channel: RIPAChannel,
    ) -> None:
        if channel == "row" and current[1] != target[1]:
            raise ValueError(f"row-channel RIPA move must keep j fixed: {current}->{target}")
        if channel == "col" and current[0] != target[0]:
            raise ValueError(f"col-channel RIPA move must keep i fixed: {current}->{target}")


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
            projection=COLUMN_PROJECTION,
        ).plan()
        self._append_rebased_steps(phase1.steps)

        # Phase 2: diagonal reflection. Reflection across the main diagonal
        # preserves `i + j` and reverses `i - j`. One 1D inversion of the
        # free coordinate is broadcast across all anti-diagonals; many AOD
        # intersections are empty and therefore harmless.
        phase2 = AODLogL1DBroadcastLineScheduler(
            RoutingRequest(
                grid=self.request.grid,
                src=mid,
                dst=self.request.dst,
                labeled=True,
            ),
            projection=ANTI_DIAGONAL_PROJECTION,
            template_fixed=pivot_sum,
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


@dataclass
class ManualRIPAHadamardPatchRotationScheduler(AsyncScheduler):
    """Manual RIPA square-ring rotation for the 5x5 Hadamard patch.

    Each layer is an independent perimeter convoy. One RIPA hop advances every
    atom on that perimeter to the next storage site, and repeating this by the
    side length minus one realizes the 90-degree rotation. Distinct layers use
    independent timelines and can run simultaneously.
    """

    def _plan(self) -> None:
        coords = self._validate_square_patch()
        atom_id_by_site = {
            tuple(site): atom_id for atom_id, site in enumerate(self.request.src)
        }
        hop_duration = self._hop_duration(coords)

        for layer in range(len(coords) // 2):
            ring = self._ring_positions(coords, layer)
            shifts = len(coords) - 1 - 2 * layer
            self._append_ring_rotation(
                ring,
                shifts,
                atom_id_by_site,
                hop_duration,
            )

    def _append_ring_rotation(
        self,
        ring: list[tuple[float, float]],
        shifts: int,
        atom_id_by_site: dict[tuple[float, float], int],
        hop_duration: float,
    ) -> None:
        ring_len = len(ring)
        for shift in range(shifts):
            start_time = shift * hop_duration
            moves: list[RIPABatchMove] = []
            for source_index, source_site in enumerate(ring):
                current = ring[(source_index + shift) % ring_len]
                target = ring[(source_index + shift + 1) % ring_len]
                moves.append((
                    atom_id_by_site[source_site],
                    target,
                    self._channel(current, target),
                ))
            self.append_step(RIPABatchStep(
                start_time=start_time,
                moves=tuple(moves),
            ))

    def _validate_square_patch(self) -> list[float]:
        coords = sorted({i for i, _ in self.request.src})
        cols = sorted({j for _, j in self.request.src})
        if coords != cols:
            raise ValueError("Manual RIPA patch demo expects identical row/col coords")
        if len(coords) != PATCH_SIDE:
            raise ValueError(
                f"Manual RIPA patch demo expects a {PATCH_SIDE}x{PATCH_SIDE} patch"
            )
        expected = {(i, j) for i in coords for j in coords}
        if set(self.request.src) != expected:
            raise ValueError("Manual RIPA patch demo expects a filled square patch")
        if self.request.dst != hadamard_rotation_targets(self.request.src):
            raise ValueError("request dst does not match the 90-degree patch rotation")
        gaps = [b - a for a, b in zip(coords, coords[1:])]
        if not gaps or any(abs(gap - gaps[0]) > 1e-9 for gap in gaps):
            raise ValueError("Manual RIPA patch demo expects evenly spaced storage sites")
        return coords

    @staticmethod
    def _ring_positions(
        coords: list[float],
        layer: int,
    ) -> list[tuple[float, float]]:
        vals = coords[layer: len(coords) - layer]
        if len(vals) <= 1:
            return []
        lo, hi = vals[0], vals[-1]
        return (
            [(lo, j) for j in vals]
            + [(i, hi) for i in vals[1:]]
            + [(hi, j) for j in reversed(vals[:-1])]
            + [(i, lo) for i in reversed(vals[1:-1])]
        )

    def _hop_duration(self, coords: list[float]) -> float:
        spacing = coords[1] - coords[0]
        accel = grid_accel_from_phys(PHYS_A_MAX, self.request.grid.d)
        return bang_bang_duration(spacing, accel)

    @staticmethod
    def _channel(
        current: tuple[float, float],
        target: tuple[float, float],
    ) -> RIPAChannel:
        if current[1] == target[1]:
            return "row"
        if current[0] == target[0]:
            return "col"
        raise ValueError(f"RIPA hop must be axis-aligned: {current} -> {target}")


SCHEDULERS = {
    "AOD H rotation": AODHadamardPatchRotationScheduler,
    "RIPA ring manual": ManualRIPAHadamardPatchRotationScheduler,
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
        hold_seconds=1.5,
        atom_colors=patch_rotation_colors(src),
        show_routing_on_start=True,
        title=f"Hadamard patch rotation ({PATCH_SIDE}x{PATCH_SIDE})",
    )
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
