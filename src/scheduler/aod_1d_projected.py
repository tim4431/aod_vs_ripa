"""Base scheduler for AOD planners that operate in 1D and lift to 2D.

Concrete subclasses implement `_plan_1d`, which receives a sorted 1D
workspace and returns the planner's AOD-native moves plus the sorted
positions where atoms end up. This base class:

- pins (or auto-detects) the motion axis and projects request sites to 1D,
- builds the workspace from `request.available_sites` or by auto-padding,
- chooses an adjacent free row/col to use as the lift lane,
- realises every planner move as a lift / slide / lower triple of AODSteps
  so the slide path is obstruction-free,
- emits one final consolidation AODStep mapping the planner's natural
  output positions onto sorted(dst).
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from typing import Literal, Sequence

from ..movement import AODStep
from .base import SyncScheduler

Axis = Literal["row", "col"]

PlannerMove = tuple[list[int], list[int]]

_FIXED_IDX = {"row": 0, "col": 1}


@dataclass
class AOD1DProjectedScheduler(SyncScheduler):
    """Plan in 1D, execute in 2D via a perpendicular lift detour.

    `axis` selects the planning direction:
      - "row": all src+dst share a row; planner moves along cols.
      - "col": all src+dst share a column; planner moves along rows.
      - None: auto-detect (preferring row when both work).
    """

    axis: Axis | None = None

    def _plan(self) -> None:
        n = len(self.request.src)
        if n == 0:
            return

        axis, fixed, src_1d, dst_1d = self._project_to_1d()
        workspace = self._build_workspace(n, src_1d, axis, fixed)
        order = self._compute_order(src_1d, dst_1d)
        lift_to = self._lift_index(fixed)

        moves, final_1d = self._plan_1d(workspace, src_1d, dst_1d, order)

        for frm, to in moves:
            self._append_aod(axis, fixed, lift_to, frm, frm)
            self._append_aod(axis, lift_to, lift_to, frm, to)
            self._append_aod(axis, lift_to, fixed, to, to)

        final_list = list(final_1d)
        sorted_dst = sorted(dst_1d)
        if final_list != sorted_dst:
            self._append_aod(axis, fixed, fixed, final_list, sorted_dst)

    # ---- subclass hooks -----------------------------------------------------

    @abstractmethod
    def _plan_1d(
        self,
        workspace: list[int],
        src_1d: list[int],
        dst_1d: list[int],
        order: list[int],
    ) -> tuple[Sequence[PlannerMove], Sequence[int]]:
        """Return `(moves, final_positions_sorted)` for the 1D subproblem."""

    @abstractmethod
    def _workspace_size(self, n: int) -> int:
        """Minimum workspace size the underlying 1D planner requires."""

    # ---- shared machinery ---------------------------------------------------

    def _project_to_1d(self) -> tuple[Axis, int, list[int], list[int]]:
        src, dst = self.request.src, self.request.dst
        candidates: list[Axis] = [self.axis] if self.axis else ["row", "col"]
        for axis in candidates:
            fi = _FIXED_IDX[axis]
            fixed_values = {p[fi] for p in src} | {p[fi] for p in dst}
            if len(fixed_values) == 1:
                (fixed,) = fixed_values
                fr = 1 - fi
                return axis, fixed, [p[fr] for p in src], [p[fr] for p in dst]

        if self.axis:
            raise ValueError(
                f"axis={self.axis!r} requires all src+dst on one {self.axis}"
            )
        raise ValueError(
            f"{type(self).__name__} requires all src+dst sites on a single "
            "row or single column (set axis='row'/'col' to pin the direction)"
        )

    def _build_workspace(
        self,
        n: int,
        src_1d: list[int],
        axis: Axis,
        fixed: int,
    ) -> list[int]:
        """Sorted workspace along the motion axis.

        Uses `request.available_sites` when supplied; otherwise pads
        `src_1d` with neighbouring grid positions (right of max(src) first,
        then left of min(src)).
        """
        needed = self._workspace_size(n)
        if self.request.available_sites is not None:
            sites = self._project_available_sites(axis, fixed)
            if len(sites) < needed:
                raise ValueError(
                    f"available_sites supplies {len(sites)} positions on "
                    f"the moving axis but the planner needs at least {needed}"
                )
            return sorted(sites)

        N = self.request.grid.N
        sites = list(src_1d)
        pool = list(range(max(sites) + 1, N)) + list(range(min(sites) - 1, -1, -1))
        sites.extend(pool[: needed - len(sites)])
        if len(sites) < needed:
            raise ValueError(
                f"need {needed} workspace positions but the grid extent "
                f"along the moving axis is {N}"
            )
        return sorted(sites)

    def _project_available_sites(self, axis: Axis, fixed: int) -> list[int]:
        fi = _FIXED_IDX[axis]
        fr = 1 - fi
        out = []
        for site in self.request.available_sites:
            if site[fi] != fixed:
                raise ValueError(
                    f"available_site {site} not on the request's shared "
                    f"{axis} {fixed}"
                )
            out.append(site[fr])
        return out

    def _lift_index(self, fixed: int) -> int:
        """An adjacent row/col used as the lifted slide lane."""
        N = self.request.grid.N
        for cand in (fixed + 1, fixed - 1):
            if 0 <= cand < N:
                return cand
        raise ValueError(
            f"{type(self).__name__} needs at least one adjacent row/col to "
            f"lift atoms into; grid extent {N} too small for fixed index "
            f"{fixed}"
        )

    def _compute_order(self, src_1d: list[int], dst_1d: list[int]) -> list[int]:
        """1D permutation: order[final_rank_k] = initial_rank of atom k."""
        if not self.request.labeled:
            return list(range(len(src_1d)))
        sorted_src = sorted(src_1d)
        sorted_dst = sorted(dst_1d)
        order = [0] * len(src_1d)
        for src_pos, dst_pos in zip(src_1d, dst_1d):
            order[sorted_dst.index(dst_pos)] = sorted_src.index(src_pos)
        return order

    def _append_aod(
        self,
        axis: Axis,
        old_fixed: int,
        new_fixed: int,
        frm: list[int],
        to: list[int],
    ) -> None:
        if axis == "row":
            sel_rows, new_rows = (old_fixed,), (new_fixed,)
            sel_cols, new_cols = tuple(frm), tuple(to)
        else:
            sel_rows, new_rows = tuple(frm), tuple(to)
            sel_cols, new_cols = (old_fixed,), (new_fixed,)
        self.append_step(AODStep(
            start_time=self.sequence.next_start_time(),
            selected_rows=sel_rows,
            selected_cols=sel_cols,
            new_rows=new_rows,
            new_cols=new_cols,
        ))
