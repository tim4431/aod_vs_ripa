"""log(L)-depth AOD scheduler for 1D rearrangement on a single row/column.

Wraps `aod_logL_1d_rearrangement` (in `logL_1d.py`) for the
`scheduler.base` Scheduler API. The 1D-on-2D machinery (axis projection,
lift/slide/lower execution, consolidation) lives in
`AOD1DProjectedScheduler`; this module supplies only the planner-specific
pieces (minimum workspace size, the planner call, the natural deposition
pattern) plus a per-row/column broadcast wrapper for axis-preserving 2D
routings.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from ...movement import AODStep
from ...routing import RoutingRequest
from ..aod_1d_projected import AOD1DProjectedScheduler, Axis, PlannerMove
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

    Requires every src and dst site to lie on one shared row, or every
    src and dst site to lie on one shared column. Works for both labeled
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

    For a routing that preserves one coordinate (every src and matching
    dst share a row or share a column), the rearrangement decomposes into
    independent 1D subproblems along the perpendicular axis. The per-line
    subproblems must be identical (same sorted src and dst, same labeled
    order); the scheduler plans the first line once with
    `AODLogL1DScheduler` and broadcasts every emitted AODStep across all
    lines simultaneously by extending the fixed-axis tuple to all line
    indices. This collapses the work of N lines into the move count of
    one line.

    `axis` matches the AODLogL1DScheduler convention:
      - "col": each column is preserved; planner moves atoms along rows.
      - "row": each row is preserved; planner moves atoms along cols.
    """

    axis: Axis = "col"

    def _plan(self) -> None:
        fixed_idx = 0 if self.axis == "row" else 1
        free_idx = 1 - fixed_idx

        by_line: dict[int, list[tuple[tuple[int, int], tuple[int, int]]]] = {}
        for src, dst in zip(self.request.src, self.request.dst):
            if src[fixed_idx] != dst[fixed_idx]:
                raise ValueError(
                    f"axis={self.axis!r} requires {self.axis}-preserving "
                    f"routing; {src} -> {dst} crosses {self.axis}s"
                )
            by_line.setdefault(src[fixed_idx], []).append((src, dst))

        lines = sorted(by_line)
        if not lines:
            return

        first = lines[0]
        first_pairs = by_line[first]
        canon_src = [s[free_idx] for s, _ in first_pairs]
        canon_dst = [d[free_idx] for _, d in first_pairs]
        for line in lines[1:]:
            pairs = by_line[line]
            if [s[free_idx] for s, _ in pairs] != canon_src or \
               [d[free_idx] for _, d in pairs] != canon_dst:
                raise ValueError(
                    f"AODLogL1DPerLineScheduler requires identical per-"
                    f"{self.axis} subproblems; {self.axis} {line} differs "
                    f"from {self.axis} {first}"
                )

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
            self.append_step(self._broadcast_step(step, first, lines))

    def _broadcast_step(
        self,
        step: AODStep,
        first: int,
        lines: list[int],
    ) -> AODStep:
        start_time = self.sequence.next_start_time()
        if self.axis == "col":
            d_old = step.selected_cols[0] - first
            d_new = step.new_cols[0] - first
            return AODStep(
                start_time=start_time,
                selected_rows=step.selected_rows,
                new_rows=step.new_rows,
                selected_cols=tuple(line + d_old for line in lines),
                new_cols=tuple(line + d_new for line in lines),
            )
        d_old = step.selected_rows[0] - first
        d_new = step.new_rows[0] - first
        return AODStep(
            start_time=start_time,
            selected_rows=tuple(line + d_old for line in lines),
            new_rows=tuple(line + d_new for line in lines),
            selected_cols=step.selected_cols,
            new_cols=step.new_cols,
        )
