"""log(L)-depth AOD scheduler for 1D rearrangement on one projected line.

Wraps `aod_logL_1d_rearrangement` (in `logL_1d.py`) for the
`scheduler.base` Scheduler API. The 1D-on-2D machinery (axis projection,
lift/slide/lower execution, consolidation) lives in
`AOD1DProjectedScheduler`; this module supplies only the planner-specific
pieces (minimum workspace size, the planner call, the natural deposition
pattern) plus a per-line broadcast wrapper for axis-preserving 2D routings.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from ...movement import AODStep, DiagonalAODStep
from ...routing import RoutingRequest
from ..aod_1d_projected import (
    AOD1DProjectedScheduler,
    Axis,
    PlannerMove,
    project_site,
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

    Requires every src and dst site to lie on one shared projected line: row,
    column, anti-diagonal, or main diagonal. Works for both labeled
    (atom k -> dst[k]) and unlabeled (set -> set) requests.
    """

    def _workspace_size(self, n: int) -> int:
        return (3 * n) // 2

    def _plan_1d(
        self,
        workspace: list[int],
        src_1d: list[int],
        dst_1d: list[int],
        order: list[int],
    ) -> tuple[Sequence[PlannerMove], Sequence[int]]:
        moves = aod_logL_1d_rearrangement(workspace, src_1d, order)
        final_1d = [workspace[i] for i in _natural_pattern(len(src_1d))]
        return moves, final_1d


@dataclass
class AODLogL1DPerLineScheduler(SyncScheduler):
    """Apply `AODLogL1DScheduler` to every row/column simultaneously.

    For a routing that preserves one projected coordinate (row, column,
    anti-diagonal, or main diagonal), the rearrangement decomposes into
    independent 1D subproblems along the free coordinate. Lines with identical
    subproblems are planned once with `AODLogL1DScheduler` and broadcast
    together by extending the fixed-axis tone tuple. This collapses identical
    lines into the move count of one line while still allowing non-identical
    line lengths to run as separate groups.

    `axis` matches the AODLogL1DScheduler convention:
      - "col": each column is preserved; planner moves atoms along rows.
      - "row": each row is preserved; planner moves atoms along cols.
      - "anti_diag": each `i + j` line is preserved; planner moves along `i - j`.
      - "main_diag": each `i - j` line is preserved; planner moves along `i + j`.
    """

    axis: Axis = "col"

    def _plan(self) -> None:
        by_line: dict[int, list[tuple[tuple[int, int], tuple[int, int]]]] = {}
        for src, dst in zip(self.request.src, self.request.dst):
            src_fixed, _ = project_site(self.axis, src)
            dst_fixed, _ = project_site(self.axis, dst)
            if src_fixed != dst_fixed:
                raise ValueError(
                    f"axis={self.axis!r} requires {self.axis}-preserving "
                    f"routing; {src} -> {dst} crosses {self.axis}s"
                )
            by_line.setdefault(src_fixed, []).append((src, dst))

        lines = sorted(by_line)
        if not lines:
            return

        groups: dict[tuple[tuple[int, ...], tuple[int, ...]], list[int]] = {}
        for line in lines:
            pairs = by_line[line]
            canon_src = tuple(project_site(self.axis, s)[1] for s, _ in pairs)
            canon_dst = tuple(project_site(self.axis, d)[1] for _, d in pairs)
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
                axis=self.axis,
            )
            for step in sub.plan().steps:
                self.append_step(self._broadcast_step(step, first, group_lines))

    def _broadcast_step(
        self,
        step: AODStep | DiagonalAODStep,
        first: int,
        lines: list[int],
    ) -> AODStep | DiagonalAODStep:
        start_time = self.sequence.next_start_time()
        if self.axis == "col":
            if not isinstance(step, AODStep):
                raise TypeError("col broadcast expects AODStep")
            d_old = step.selected_cols[0] - first
            d_new = step.new_cols[0] - first
            return AODStep(
                start_time=start_time,
                selected_rows=step.selected_rows,
                new_rows=step.new_rows,
                selected_cols=tuple(line + d_old for line in lines),
                new_cols=tuple(line + d_new for line in lines),
            )
        if self.axis == "row":
            if not isinstance(step, AODStep):
                raise TypeError("row broadcast expects AODStep")
            d_old = step.selected_rows[0] - first
            d_new = step.new_rows[0] - first
            return AODStep(
                start_time=start_time,
                selected_rows=tuple(line + d_old for line in lines),
                new_rows=tuple(line + d_new for line in lines),
                selected_cols=step.selected_cols,
                new_cols=step.new_cols,
            )
        if self.axis == "anti_diag":
            if not isinstance(step, DiagonalAODStep):
                raise TypeError("anti_diag broadcast expects DiagonalAODStep")
            d_old = step.selected_sums[0] - first
            d_new = step.new_sums[0] - first
            return DiagonalAODStep(
                start_time=start_time,
                selected_sums=tuple(line + d_old for line in lines),
                new_sums=tuple(line + d_new for line in lines),
                selected_diffs=step.selected_diffs,
                new_diffs=step.new_diffs,
            )
        if self.axis == "main_diag":
            if not isinstance(step, DiagonalAODStep):
                raise TypeError("main_diag broadcast expects DiagonalAODStep")
            d_old = step.selected_diffs[0] - first
            d_new = step.new_diffs[0] - first
            return DiagonalAODStep(
                start_time=start_time,
                selected_sums=step.selected_sums,
                new_sums=step.new_sums,
                selected_diffs=tuple(line + d_old for line in lines),
                new_diffs=tuple(line + d_new for line in lines),
            )
        raise ValueError(f"unknown AOD projection axis: {self.axis!r}")
