"""log(L)-depth AOD scheduler for 1D rearrangement on a single row/column.

Wraps `aod_logL_1d_rearrangement` (in `logL_1d.py`) for the
`scheduler.base` Scheduler API. The planner emits subset moves that
treat atoms as if they could pass straight through stationary neighbours
on the same line; in this simulation that triggers the physical
collision check. To stay collision-safe each planner move is realised as
three AODSteps: lift the moving subset onto an adjacent free row/col,
slide it along the motion axis (now obstruction-free), then lower it
back. A final consolidation AODStep moves the planner's natural
deposition pattern onto sorted(dst) (no lift needed: every atom moves
together with no stationary obstacles).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ...movement import AODStep
from ..base import SyncScheduler
from .logL_1d import aod_logL_1d_rearrangement

Axis = Literal["row", "col"]


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
class AODLogL1DScheduler(SyncScheduler):
    """log(L)-depth AOD scheduler for 1D rearrangement.

    Requires every src and dst site to lie on one shared row, or every
    src and dst site to lie on one shared column. Works for both labeled
    (atom k -> dst[k]) and unlabeled (set -> set) requests.
    """

    def _plan(self) -> None:
        n = len(self.request.src)
        if n == 0:
            return

        axis, fixed, src_1d, dst_1d = self._project_to_1d()
        workspace = self._build_workspace(n, src_1d)
        order = self._compute_order(src_1d, dst_1d)
        lift_to = self._lift_index(fixed)

        for frm, to in aod_logL_1d_rearrangement(workspace, src_1d, order):
            self._append_aod(axis, fixed, lift_to, frm, frm)
            self._append_aod(axis, lift_to, lift_to, frm, to)
            self._append_aod(axis, lift_to, fixed, to, to)

        nat_positions = [workspace[i] for i in _natural_pattern(n)]
        sorted_dst = sorted(dst_1d)
        if nat_positions != sorted_dst:
            self._append_aod(axis, fixed, fixed, nat_positions, sorted_dst)

    def _append_aod(
        self,
        axis: Axis,
        old_fixed: int,
        new_fixed: int,
        frm: list[int],
        to: list[int],
    ) -> None:
        start_time = self.sequence.next_start_time()
        if axis == "row":
            step = AODStep(
                start_time=start_time,
                selected_rows=(old_fixed,),
                selected_cols=tuple(frm),
                new_rows=(new_fixed,),
                new_cols=tuple(to),
            )
        else:
            step = AODStep(
                start_time=start_time,
                selected_rows=tuple(frm),
                selected_cols=(old_fixed,),
                new_rows=tuple(to),
                new_cols=(new_fixed,),
            )
        self.append_step(step)

    def _project_to_1d(self) -> tuple[Axis, int, list[int], list[int]]:
        src = self.request.src
        dst = self.request.dst
        rows = {p[0] for p in src} | {p[0] for p in dst}
        cols = {p[1] for p in src} | {p[1] for p in dst}
        if len(rows) == 1:
            (r,) = rows
            return "row", r, [p[1] for p in src], [p[1] for p in dst]
        if len(cols) == 1:
            (c,) = cols
            return "col", c, [p[0] for p in src], [p[0] for p in dst]
        raise ValueError(
            "AODLogL1DScheduler requires all src+dst sites on a single row "
            "or single column"
        )

    def _build_workspace(self, n: int, src_1d: list[int]) -> list[int]:
        """Sorted workspace of size 3n//2 containing src plus filler sites."""
        N = self.request.grid.N
        needed = (3 * n) // 2
        sites = list(src_1d)
        next_right = max(sites) + 1
        next_left = min(sites) - 1
        while len(sites) < needed:
            if next_right < N:
                sites.append(next_right)
                next_right += 1
            elif next_left >= 0:
                sites.append(next_left)
                next_left -= 1
            else:
                raise ValueError(
                    f"need {needed} workspace positions but the grid extent "
                    f"along the moving axis is {N}"
                )
        return sorted(sites)

    def _lift_index(self, fixed: int) -> int:
        """An adjacent row/col used as the lifted slide lane during planner
        moves; must lie inside the grid."""
        N = self.request.grid.N
        if fixed + 1 < N:
            return fixed + 1
        if fixed - 1 >= 0:
            return fixed - 1
        raise ValueError(
            "AODLogL1DScheduler needs at least one adjacent row/col to lift "
            f"atoms into; grid extent {N} too small for fixed index {fixed}"
        )

    def _compute_order(self, src_1d: list[int], dst_1d: list[int]) -> list[int]:
        n = len(src_1d)
        if not self.request.labeled:
            return list(range(n))
        sorted_src = sorted(src_1d)
        sorted_dst = sorted(dst_1d)
        order = [0] * n
        for src_pos, dst_pos in zip(src_1d, dst_1d):
            order[sorted_dst.index(dst_pos)] = sorted_src.index(src_pos)
        return order
