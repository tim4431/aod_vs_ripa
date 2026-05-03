"""log(L)-depth AOD scheduler for 1D rearrangement on a single row/column.

Wraps `aod_logL_1d_rearrangement` (in `logL_1d.py`) for the
`scheduler.base` Scheduler API. Each planner move maps directly to one
synchronous AODStep.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ...movement import AODStep
from ..base import SyncScheduler
from .logL_1d import aod_logL_1d_rearrangement

Axis = Literal["row", "col"]


@dataclass
class AODLogL1DScheduler(SyncScheduler):
    """log(L)-depth AOD scheduler for 1D rearrangement.

    The underlying planner packs atoms toward the leftmost end of its
    workspace, so the request must satisfy:

      - every src and dst site lies on the same grid row, or every src
        and dst site lies on the same grid column;
      - sorted(dst) forms the leftmost N positions of the workspace, i.e.
        every src position not also in dst must be strictly greater than
        max(dst). Atoms come in from the right and pack into dst.

    Works for both labeled (atom k -> dst[k]) and unlabeled (set -> set)
    requests.
    """

    def _plan(self) -> None:
        n = len(self.request.src)
        if n == 0:
            return

        axis, fixed, src_1d, dst_1d = self._project_to_1d()
        sites = self._build_workspace(n, src_1d, dst_1d)
        order = self._compute_order(src_1d, dst_1d)

        for frm, to in aod_logL_1d_rearrangement(sites, src_1d, order):
            self.append_step(self._aod_step(axis, fixed, frm, to))

    def _aod_step(
        self,
        axis: Axis,
        fixed: int,
        frm: list[int],
        to: list[int],
    ) -> AODStep:
        start_time = self.sequence.next_start_time()
        if axis == "row":
            return AODStep(
                start_time=start_time,
                selected_rows=(fixed,),
                selected_cols=tuple(frm),
                new_rows=(fixed,),
                new_cols=tuple(to),
            )
        return AODStep(
            start_time=start_time,
            selected_rows=tuple(frm),
            selected_cols=(fixed,),
            new_rows=tuple(to),
            new_cols=(fixed,),
        )

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

    def _build_workspace(
        self,
        n: int,
        src_1d: list[int],
        dst_1d: list[int],
    ) -> list[int]:
        sorted_dst = sorted(dst_1d)
        max_dst = sorted_dst[-1]
        dst_set = set(sorted_dst)
        for s in src_1d:
            if s <= max_dst and s not in dst_set:
                raise ValueError(
                    f"src position {s} is inside the dst range but is not a "
                    "dst; the logL_1d planner only packs atoms in from the right"
                )
        sites = sorted(dst_set | {s for s in src_1d if s > max_dst})
        N = self.request.grid.N
        next_pos = sites[-1] + 1
        while len(sites) < (3 * n) // 2:
            if next_pos >= N:
                raise ValueError(
                    f"need {(3 * n) // 2} workspace positions but the grid "
                    f"extent along the moving axis is {N}"
                )
            sites.append(next_pos)
            next_pos += 1
        return sites

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
