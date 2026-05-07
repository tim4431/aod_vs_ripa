"""log(L)-depth AOD scheduler for 1D rearrangement on one projected line.

Wraps `aod_logL_1d_rearrangement` (in `logL_1d.py`) for the
`scheduler.base` Scheduler API. The 1D-on-2D machinery (axis projection,
lift/slide/lower execution, consolidation) lives in
`AOD1DProjectedScheduler`; this module supplies only the planner-specific
pieces (minimum workspace size, the planner call, the natural deposition
pattern) plus wrappers for line-preserving 2D routings. One wrapper plans
each projected line independently; another broadcasts one free-coordinate
plan across many fixed lines, relying on the fact that empty AOD
intersections are allowed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

from ...atom_config import clean_coord
from ...movement import AODStep
from ...routing import RoutingRequest
from ..aod_1d_projected import (
    AOD1DProjectedScheduler,
    AODProjection,
    PlannerMove,
    unit_vector,
)
from ..base import SyncScheduler
from .logL_1d import aod_logL_1d_rearrangement


def _natural_pattern(n: int) -> list[int]:
    """Workspace indices where the planner deposits the n atoms.

    Mirrors the planner's recursive split: each subgroup of size m ends
    at the right edge of its sub-workspace of size 3*m//2.
    """
    if n <= 0:
        return []
    if n == 1:
        return [0]
    n_left = n // 2
    n_right = n - n_left
    ws_left_size = 3 * n_left // 2
    return _natural_pattern(n_left) + [
        p + ws_left_size for p in _natural_pattern(n_right)
    ]


@dataclass
class AODLogL1DScheduler(AOD1DProjectedScheduler):
    """log(L)-depth AOD scheduler for 1D rearrangement.

    Requires every src and dst site to lie on one shared projected line in
    the supplied AOD projection. Works for both labeled (atom k -> dst[k])
    and unlabeled (set -> set) requests.
    """

    def _workspace_size(self, n: int) -> int:
        return (3 * n) // 2

    def _plan_1d(
        self,
        workspace: list[float],
        src_1d: list[float],
        dst_1d: list[float],
        order: list[int],
    ) -> tuple[Sequence[PlannerMove], Sequence[float]]:
        moves = aod_logL_1d_rearrangement(workspace, src_1d, order)
        final_1d = [workspace[i] for i in _natural_pattern(len(src_1d))]
        return moves, final_1d


@dataclass
class AODLogL1DPerLineScheduler(SyncScheduler):
    """Apply `AODLogL1DScheduler` to every projected line simultaneously.

    For a routing that preserves one local AOD coordinate, the rearrangement
    decomposes into independent 1D subproblems along the free coordinate.
    Lines with identical subproblems are planned once with
    `AODLogL1DScheduler` and broadcast together by extending the fixed-axis
    tone tuple. This collapses identical lines into the move count of one
    line while still allowing non-identical line lengths to run separately.
    """

    projection: AODProjection = field(
        default_factory=lambda: AODProjection(
            axis_1=unit_vector(0.0),
            axis_2=unit_vector(math.pi / 2.0),
            fixed_axis="axis_2",
            name="axis_2_lines",
        )
    )

    def _plan(self) -> None:
        projection = self.projection
        by_line: dict[float, list[tuple[tuple[float, float], tuple[float, float]]]] = {}
        for src, dst in zip(self.request.src, self.request.dst):
            src_fixed, _ = projection.project_site(src)
            dst_fixed, _ = projection.project_site(dst)
            if abs(src_fixed - dst_fixed) > 1e-9:
                raise ValueError(
                    f"projection={projection.name!r} requires line-preserving "
                    f"routing; {src} -> {dst} crosses projected lines"
                )
            by_line.setdefault(src_fixed, []).append((src, dst))

        lines = sorted(by_line)
        if not lines:
            return

        groups: dict[tuple[tuple[float, ...], tuple[float, ...]], list[float]] = {}
        for line in lines:
            pairs = by_line[line]
            canon_src = tuple(projection.project_site(s)[1] for s, _ in pairs)
            canon_dst = tuple(projection.project_site(d)[1] for _, d in pairs)
            groups.setdefault((canon_src, canon_dst), []).append(line)

        for group_lines in groups.values():
            first = group_lines[0]
            first_pairs = by_line[first]
            sub = AODLogL1DScheduler(
                request=RoutingRequest(
                    grid=self.request.grid,
                    src=[s for s, _ in first_pairs],
                    dst=[d for _, d in first_pairs],
                    labeled=self.request.labeled,
                ),
                projection=projection,
            )
            for step in sub.plan().steps:
                self.append_step(
                    self._broadcast_step(step, first, group_lines, projection)
                )

    def _broadcast_step(
        self,
        step: AODStep,
        first: float,
        lines: list[float],
        projection: AODProjection,
    ) -> AODStep:
        start_time = self.sequence.next_start_time()
        if projection.fixed_axis == "axis_2":
            d_old = step.selected_axis_2[0] - first
            d_new = step.new_axis_2[0] - first
            return AODStep(
                start_time=start_time,
                selected_axis_1=step.selected_axis_1,
                new_axis_1=step.new_axis_1,
                selected_axis_2=tuple(line + d_old for line in lines),
                new_axis_2=tuple(line + d_new for line in lines),
                axis_1=step.axis_1,
                axis_2=step.axis_2,
                origin=step.origin,
            )
        d_old = step.selected_axis_1[0] - first
        d_new = step.new_axis_1[0] - first
        return AODStep(
            start_time=start_time,
            selected_axis_1=tuple(line + d_old for line in lines),
            new_axis_1=tuple(line + d_new for line in lines),
            selected_axis_2=step.selected_axis_2,
            new_axis_2=step.new_axis_2,
            axis_1=step.axis_1,
            axis_2=step.axis_2,
            origin=step.origin,
        )


@dataclass
class AODLogL1DBroadcastLineScheduler(SyncScheduler):
    """Run one projected 1D plan and broadcast it across many fixed lines.

    This is the AOD-native version for cases like diagonal reflection of a
    square patch: every fixed-coordinate line applies the same free-coordinate
    permutation, but many outer-product intersections are empty. Empty traps are
    harmless; only occupied intersections produce atom trajectories.
    """

    projection: AODProjection = field(default_factory=AODProjection)
    template_fixed: float | None = None

    def _plan(self) -> None:
        projection = self.projection
        fixed_values: list[float] = []
        free_map: dict[float, float] = {}

        for src, dst in zip(self.request.src, self.request.dst):
            src_fixed, src_free = projection.project_site(src)
            dst_fixed, dst_free = projection.project_site(dst)
            if abs(src_fixed - dst_fixed) > 1e-9:
                raise ValueError(
                    f"projection={projection.name!r} requires line-preserving "
                    f"routing; {src} -> {dst} crosses projected lines"
                )
            fixed_values.append(src_fixed)
            if self._has_conflicting_mapping(free_map, src_free, dst_free):
                raise ValueError(
                    f"free coordinate {src_free} maps to multiple destinations"
                )
            free_map[clean_coord(src_free)] = clean_coord(dst_free)

        fixed_lines = self._unique_sorted(fixed_values)
        if not fixed_lines:
            return

        src_free = self._unique_sorted(free_map)
        dst_free = [free_map[free] for free in src_free]
        template_fixed = self._template_fixed(projection, src_free, dst_free, fixed_lines)
        sub = AODLogL1DScheduler(
            request=RoutingRequest(
                grid=self.request.grid,
                src=[
                    projection.site_from_local(
                        *self._fixed_free_to_local(projection, template_fixed, free)
                    )
                    for free in src_free
                ],
                dst=[
                    projection.site_from_local(
                        *self._fixed_free_to_local(projection, template_fixed, free)
                    )
                    for free in dst_free
                ],
                labeled=True,
            ),
            projection=projection,
        )
        for step in sub.plan().steps:
            self.append_step(self._broadcast_step(step, template_fixed, fixed_lines))

    def _template_fixed(
        self,
        projection: AODProjection,
        src_free: Sequence[float],
        dst_free: Sequence[float],
        fixed_lines: Sequence[float],
    ) -> float:
        if self.template_fixed is not None:
            return clean_coord(self.template_fixed)
        free_values = tuple(src_free) + tuple(dst_free)
        candidates = sorted(
            fixed_lines,
            key=lambda x: (abs(x - fixed_lines[len(fixed_lines) // 2]), x),
        )
        for fixed in candidates:
            if all(
                self._free_coord_in_bounds(projection, fixed, free)
                for free in free_values
            ):
                return fixed
        raise ValueError(
            "could not find a fixed-coordinate template line that contains all "
            "broadcast free coordinates; pass template_fixed explicitly"
        )

    def _broadcast_step(
        self,
        step: AODStep,
        template_fixed: float,
        fixed_lines: Sequence[float],
    ) -> AODStep:
        start_time = self.sequence.next_start_time()
        if self.projection.fixed_axis == "axis_2":
            d_old = step.selected_axis_2[0] - template_fixed
            d_new = step.new_axis_2[0] - template_fixed
            return AODStep(
                start_time=start_time,
                selected_axis_1=step.selected_axis_1,
                new_axis_1=step.new_axis_1,
                selected_axis_2=tuple(line + d_old for line in fixed_lines),
                new_axis_2=tuple(line + d_new for line in fixed_lines),
                axis_1=step.axis_1,
                axis_2=step.axis_2,
                origin=step.origin,
            )
        d_old = step.selected_axis_1[0] - template_fixed
        d_new = step.new_axis_1[0] - template_fixed
        return AODStep(
            start_time=start_time,
            selected_axis_1=tuple(line + d_old for line in fixed_lines),
            new_axis_1=tuple(line + d_new for line in fixed_lines),
            selected_axis_2=step.selected_axis_2,
            new_axis_2=step.new_axis_2,
            axis_1=step.axis_1,
            axis_2=step.axis_2,
            origin=step.origin,
        )

    def _free_coord_in_bounds(
        self,
        projection: AODProjection,
        fixed: float,
        free: float,
    ) -> bool:
        i, j = projection.site_from_local(
            *self._fixed_free_to_local(projection, fixed, free)
        )
        N = self.request.grid.N
        return 0 <= i < N and 0 <= j < N

    @staticmethod
    def _fixed_free_to_local(
        projection: AODProjection,
        fixed: float,
        free: float,
    ) -> tuple[float, float]:
        return (
            (fixed, free)
            if projection.fixed_axis == "axis_1"
            else (free, fixed)
        )

    @staticmethod
    def _has_conflicting_mapping(
        free_map: dict[float, float],
        src_free: float,
        dst_free: float,
    ) -> bool:
        key = clean_coord(src_free)
        return key in free_map and abs(free_map[key] - clean_coord(dst_free)) > 1e-9

    @staticmethod
    def _unique_sorted(values: Sequence[float]) -> list[float]:
        out: list[float] = []
        for value in sorted(float(v) for v in values):
            if not out or abs(out[-1] - value) > 1e-9:
                out.append(clean_coord(value))
        return out
