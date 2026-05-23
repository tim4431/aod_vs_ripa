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
    python example/hadamard_rotation/hadamard_patch_rotation.py            # quick GIF -> render/
    python example/hadamard_rotation/hadamard_patch_rotation.py --demo     # high quality -> demo/
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

import matplotlib as mpl

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.atom_config import Grid
from src.benchmark import benchmark_schedulers, format_benchmark_table
from src.atom_trajectory import AtomEnsemble, CollisionError, CollisionReport
from src.movement import PHYS_A_MAX, Step, grid_accel_from_phys
from src.routing import RoutingRequest, centered_storage_square
from src.scheduler.aod_1d_projected import AODProjection, unit_vector
from src.scheduler.base import AsyncScheduler, SyncScheduler
from src.scheduler.logL_1d.aod_logL_1d import (
    AODLogL1DBroadcastLineScheduler,
    AODLogL1DPerLineScheduler,
)
from src.segments import Segment, bang_bang_duration, make_const_acc_segment
from src.visualization import render_animation

N = 24
PATCH_SIDE = 5
STORAGE_PERIOD = 3
GRID_SPACING_UM = 3.0
COLLISION_RADIUS_UM = 1.0

PREFIX = "hadamard_patch_rotation"
RIPAChannel = Literal["row", "col"]


RIPALeg = tuple[tuple[float, float], tuple[float, float], RIPAChannel, float]
# (start_pos, end_pos, channel, duration) — one straight bang-bang leg.


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


RIPATrajectory = tuple[int, tuple[RIPALeg, ...]]
# (atom_id, legs) — one atom's continuous multi-leg trajectory.


@dataclass
class BatchedRingTrajectoryStep(Step):
    """All-atoms multi-leg trajectories applied as one atomic batch.

    Every atom in the layer crosses up to one corner, so its trajectory
    is one or two bang-bang legs joined back-to-back with no rest at
    intermediate grid points or at corners. The standard incremental
    validator can't see "atom k will move out of slot S before atom j
    arrives" because atoms not yet committed look like obstacles, so this
    step (mirroring `RIPACCBSCScheduler`'s commit-then-check pattern)
    commits every atom's segments and runs a single full-timeline
    pairwise collision pass; it rolls everything back on failure.
    """

    start_time: float
    trajectories: tuple[RIPATrajectory, ...]

    def apply(self, ensemble: AtomEnsemble) -> None:
        by_atom = self._build_segments_by_atom()
        self._check_continuity(ensemble, by_atom)
        old_lengths = {
            atom.atom_id: len(atom.segments) for atom in ensemble.atomtrajs
        }
        try:
            for atom_id, segs in by_atom.items():
                ensemble.atomtraj_by_id(atom_id).segments.extend(segs)
            report = _check_full_timeline(ensemble)
            if not report.ok:
                raise CollisionError(report)
        except Exception:
            for atom in ensemble.atomtrajs:
                del atom.segments[old_lengths[atom.atom_id]:]
            raise

    def end_time(self, ensemble: AtomEnsemble) -> float:
        return self.start_time + max(
            (
                sum(duration for _, _, _, duration in legs)
                for _, legs in self.trajectories
            ),
            default=0.0,
        )

    def _build_segments_by_atom(self) -> dict[int, list[Segment]]:
        by_atom: dict[int, list[Segment]] = {}
        for atom_id, legs in self.trajectories:
            t = self.start_time
            segs: list[Segment] = []
            for start_pos, end_pos, channel, duration in legs:
                self._check_channel(start_pos, end_pos, channel)
                segs.append(make_const_acc_segment(
                    start_pos, end_pos, t,
                    duration=duration, channel=channel,
                ))
                t += duration
            by_atom[atom_id] = segs
        return by_atom

    @staticmethod
    def _check_continuity(
        ensemble: AtomEnsemble,
        by_atom: dict[int, list[Segment]],
    ) -> None:
        for atom_id, segs in by_atom.items():
            atom = ensemble.atomtraj_by_id(atom_id)
            prev_pos = atom.final_pos
            prev_time = atom.final_time
            for seg in segs:
                if seg.start_pos != prev_pos:
                    raise ValueError(
                        f"atom {atom_id}: segment starts at {seg.start_pos} "
                        f"but atom is at {prev_pos}"
                    )
                if seg.start_time + 1e-12 < prev_time:
                    raise ValueError(
                        f"atom {atom_id}: segment starts at t={seg.start_time} "
                        f"before previous segment ends at t={prev_time}"
                    )
                prev_pos = seg.end_pos
                prev_time = seg.end_time

    @staticmethod
    def _check_channel(
        start_pos: tuple[float, float],
        end_pos: tuple[float, float],
        channel: RIPAChannel,
    ) -> None:
        if channel == "row" and start_pos[1] != end_pos[1]:
            raise ValueError(f"row-channel RIPA move must keep j fixed: {start_pos}->{end_pos}")
        if channel == "col" and start_pos[0] != end_pos[0]:
            raise ValueError(f"col-channel RIPA move must keep i fixed: {start_pos}->{end_pos}")


def _check_full_timeline(ensemble: AtomEnsemble) -> CollisionReport:
    """O(N²) pairwise collision pass over the whole ensemble timeline.

    Mirrors the helper inside `RIPACCBSCScheduler`: the incremental
    validator only catches conflicts as they're added, but a batched
    multi-segment commit needs one final cross-atom check across all
    timelines after every atom's segments have been spliced in.
    """
    rc_grid = ensemble.grid.rc / ensemble.grid.d
    horizon = ensemble.total_duration()
    worst_grid = math.inf
    worst_pair: tuple[int, int, float] | None = None
    atoms = ensemble.atomtrajs
    for idx, atom_a in enumerate(atoms):
        for atom_b in atoms[idx + 1:]:
            for seg_a in atom_a.timeline_segments(0.0, horizon):
                if seg_a.duration <= 0:
                    continue
                for seg_b in atom_b.timeline_segments(seg_a.start_time, seg_a.end_time):
                    t0 = max(seg_a.start_time, seg_b.start_time)
                    t1 = min(seg_a.end_time, seg_b.end_time)
                    if t1 < t0 - 1e-12:
                        continue
                    d_grid, t = ensemble._distance_between_segments(
                        seg_a, seg_b, t0, t1, rc_grid,
                    )
                    if d_grid < worst_grid:
                        worst_grid = d_grid
                        worst_pair = (
                            int(atom_a.atom_id),
                            int(atom_b.atom_id),
                            float(t),
                        )
    return CollisionReport(
        ok=worst_grid * ensemble.grid.d + 1e-12 >= ensemble.grid.rc,
        worst_pair=worst_pair,
        worst_distance=worst_grid * ensemble.grid.d,
    )


def hadamard_rotation_targets(src):
    """Target positions for horizontal reflection followed by diagonal reflection."""
    coords = sorted({i for i, _ in src})
    pivot_sum = coords[0] + coords[-1]
    return [(j, pivot_sum - i) for i, j in src]


def patch_rotation_colors(src):
    """Linear gradient along the (1,1) diagonal so a 90-degree rotation
    flips the gradient direction visibly (diagonal -> anti-diagonal)."""
    coords = sorted({i for i, _ in src})
    lo, hi = coords[0] + coords[0], coords[-1] + coords[-1]
    cmap = mpl.colormaps["viridis"]
    colors = {}
    for atom_id, (i, j) in enumerate(src):
        s = (i + j - lo) / (hi - lo) if hi > lo else 0.5
        colors[atom_id] = cmap(0.05 + 0.9 * s)
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
    """Manual RIPA square-ring rotation with continuous corner-to-corner flow.

    Each atom walks its full perimeter share in one go: bang-bang per
    axis-aligned leg, joined back-to-back with no rest at intermediate
    grid points and no rest at corners. Within a ring layer, every atom
    shares one common total time `T_layer` — set by the most-balanced
    two-leg split (the slowest natural trajectory at `a_max`) — by
    spreading sub-max bang-bang acceleration across its legs
    (``a_atom = (2·Σ √Lᵢ / T_layer)²``). Equal total times preserve the
    rigid spatial relationship between atoms throughout the rotation, so
    the only "deceleration to a grid point" happens once at each atom's
    final destination. Outer and inner rings run in parallel.
    """

    def _plan(self) -> None:
        coords = self._validate_square_patch()
        accel_max = grid_accel_from_phys(PHYS_A_MAX, self.request.grid.d)
        atom_id_by_site = {
            tuple(site): atom_id for atom_id, site in enumerate(self.request.src)
        }

        for layer in range(len(coords) // 2):
            ring = self._ring_positions(coords, layer)
            ring_len = len(ring)
            shifts = len(coords) - 1 - 2 * layer
            if ring_len == 0 or shifts == 0:
                continue
            self._append_ring_rotation(ring, shifts, atom_id_by_site, accel_max)

    corner_safety: float = 1.10
    """Multiplier on the geometric corner-clearance time `Δt = 2·√(2·rc/a)`.
    Must be ≥ 1; larger values widen the per-corner gap for safety."""

    def _append_ring_rotation(
        self,
        ring: list[tuple[float, float]],
        shifts: int,
        atom_id_by_site: dict[tuple[float, float], int],
        accel_max: float,
    ) -> None:
        ring_len = len(ring)
        spacing = math.hypot(
            ring[1][0] - ring[0][0], ring[1][1] - ring[0][1]
        )
        T_shift = bang_bang_duration(spacing, accel_max)
        rc_grid = self.request.grid.rc / self.request.grid.d
        # `Δt = 2·√(2·rc/a)`: time for a freshly-arrived atom to clear the
        # corner by `rc` while accelerating from rest in the new
        # direction, plus the symmetric time for the *next* atom to
        # decelerate into the corner from `rc` away. Lockstep used the
        # full `T_shift` here, but the geometric minimum is much smaller
        # (≈0.63·T_shift at rc=0.8·spacing).
        delta_corner = self.corner_safety * 2.0 * math.sqrt(
            2.0 * rc_grid / accel_max
        )

        trajectories: list[RIPATrajectory] = []
        for source_index in range(ring_len):
            atom_id = atom_id_by_site[ring[source_index]]
            path = [
                ring[(source_index + h) % ring_len] for h in range(shifts + 1)
            ]
            legs = self._compress_legs(path)
            n_legs = len(legs)
            step_legs: list[RIPALeg] = []
            for leg_idx, (start_pos, end_pos) in enumerate(legs):
                distance = math.hypot(
                    end_pos[0] - start_pos[0], end_pos[1] - start_pos[1]
                )
                if distance == 0:
                    continue
                slot_count = round(distance / spacing)
                # Two cases for a leg's duration:
                #
                # (a) Leg ends *at a corner* (leg 1 of a multi-leg atom,
                #     or the single leg of a corner-only atom). The end
                #     corner is shared with up to `slot_count` other
                #     atoms threading it from this edge, so the leg's
                #     end time must be at least `T_shift + (slot_count -
                #     1)·Δt_corner` to keep them rc-separated as they
                #     enter the corner one after the other. Lockstep used
                #     `slot_count·T_shift` here — strictly larger.
                #
                # (b) Leg ends *mid-edge* (leg 2 of a multi-leg atom).
                #     Nothing constrains its end time beyond the
                #     hardware ceiling, so just bang-bang at `a_max`.
                ends_at_corner = (n_legs == 1) or (leg_idx < n_legs - 1)
                if ends_at_corner:
                    duration = max(
                        bang_bang_duration(distance, accel_max),
                        T_shift + (slot_count - 1) * delta_corner,
                    )
                else:
                    duration = bang_bang_duration(distance, accel_max)
                step_legs.append((
                    start_pos,
                    end_pos,
                    self._channel(start_pos, end_pos),
                    duration,
                ))
            if step_legs:
                trajectories.append((atom_id, tuple(step_legs)))

        if trajectories:
            self.append_step(BatchedRingTrajectoryStep(
                start_time=0.0,
                trajectories=tuple(trajectories),
            ))

    @staticmethod
    def _compress_legs(
        path: list[tuple[float, float]],
    ) -> list[tuple[tuple[float, float], tuple[float, float]]]:
        """Collapse consecutive same-axis hops in `path` into one leg each."""
        legs: list[tuple[tuple[float, float], tuple[float, float]]] = []
        leg_start = path[0]
        prev_axis: str | None = None
        for k in range(1, len(path)):
            cur, nxt = path[k - 1], path[k]
            if cur == nxt:
                continue
            if cur[0] == nxt[0]:
                axis = "j"
            elif cur[1] == nxt[1]:
                axis = "i"
            else:
                raise ValueError(f"non-axis-aligned hop: {cur}->{nxt}")
            if prev_axis is not None and axis != prev_axis:
                legs.append((leg_start, path[k - 1]))
                leg_start = path[k - 1]
            prev_axis = axis
        legs.append((leg_start, path[-1]))
        return legs

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


RIPALockstepMove = tuple[int, tuple[float, float], RIPAChannel]
# (atom_id, target, channel) — one max-accel one-slot RIPA hop in a lockstep batch.


@dataclass
class LockstepRIPABatchStep(Step):
    """Same-start batch of one-slot RIPA hops at `a_max`.

    The original 1-hop-per-shift behaviour: every atom moves a single
    grid step in lockstep, decelerating to rest at the destination.
    Repeated `shifts` times per ring layer to realize the rotation.
    """

    start_time: float
    moves: tuple[RIPALockstepMove, ...]

    def apply(self, ensemble: AtomEnsemble) -> None:
        accel = grid_accel_from_phys(PHYS_A_MAX, ensemble.grid.d)
        segments = []
        for atom_id, target, channel in self.moves:
            current = ensemble.atomtraj_by_id(atom_id).resting_position_at(
                self.start_time
            )
            BatchedRingTrajectoryStep._check_channel(current, target, channel)
            if current == target:
                continue
            segments.append((
                atom_id,
                make_const_acc_segment(
                    current, target, self.start_time,
                    accel=accel, channel=channel,
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
            BatchedRingTrajectoryStep._check_channel(current, target, channel)
            duration = max(
                duration,
                bang_bang_duration(
                    math.hypot(target[0] - current[0], target[1] - current[1]),
                    accel,
                ),
            )
        return self.start_time + duration


@dataclass
class LockstepRIPAHadamardPatchRotationScheduler(
    ManualRIPAHadamardPatchRotationScheduler
):
    """Per-shift lockstep RIPA rotation — baseline for comparison.

    Each ring layer rotates by `shifts` slot hops; for every shift, all
    atoms move one perimeter slot in a same-start batch at `a_max`,
    decelerating to rest at every grid point. Inherits the patch /
    ring / channel helpers from the continuous-leg scheduler so the two
    classes solve the same `RoutingRequest` and only differ in *how*
    they discretize the rotation.
    """

    def _plan(self) -> None:
        coords = self._validate_square_patch()
        accel_max = grid_accel_from_phys(PHYS_A_MAX, self.request.grid.d)
        spacing = float(coords[1] - coords[0])
        T_shift = bang_bang_duration(spacing, accel_max)
        atom_id_by_site = {
            tuple(site): atom_id for atom_id, site in enumerate(self.request.src)
        }

        for layer in range(len(coords) // 2):
            ring = self._ring_positions(coords, layer)
            shifts = len(coords) - 1 - 2 * layer
            if not ring or shifts == 0:
                continue
            ring_len = len(ring)
            for shift in range(shifts):
                start_time = shift * T_shift
                moves: list[RIPALockstepMove] = []
                for source_index, source_site in enumerate(ring):
                    current = ring[(source_index + shift) % ring_len]
                    target = ring[(source_index + shift + 1) % ring_len]
                    moves.append((
                        atom_id_by_site[source_site],
                        target,
                        self._channel(current, target),
                    ))
                self.append_step(LockstepRIPABatchStep(
                    start_time=start_time,
                    moves=tuple(moves),
                ))


SCHEDULERS = {
    "RIPA synchronous": LockstepRIPAHadamardPatchRotationScheduler,
    "RIPA asynchronous": ManualRIPAHadamardPatchRotationScheduler,
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
    all_panels = {name: by_name[name] for name in SCHEDULERS}
    pair_panels = {
        name: by_name[name] for name in ("RIPA asynchronous", "AOD H rotation")
    }

    if args.demo:
        outputs = [
            (all_panels, ROOT / "demo" / f"{PREFIX}.gif", "quality"),
            (pair_panels, ROOT / "render" / f"{PREFIX}_pair.gif", "quality"),
        ]
    else:
        outputs = [(all_panels, ROOT / "render" / f"{PREFIX}.gif", "speed")]

    for panels, out_path, quality in outputs:
        render_animation(
            panels,
            out_path,
            view="benchmark",
            quality=quality,
            time_dilation=1e4,
            hold_seconds=1.5,
            atom_colors=patch_rotation_colors(src),
            show_routing_on_start=True,
            show_color_code=True,
            panel_speedup={"AOD H rotation": 8.0},
            trap_scale=0.6,
            title=f"Hadamard patch rotation ({PATCH_SIDE}x{PATCH_SIDE})",
        )
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
