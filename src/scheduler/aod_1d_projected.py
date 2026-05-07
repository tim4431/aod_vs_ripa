"""Base scheduler for AOD planners that operate in 1D and lift to 2D.

Concrete subclasses implement `_plan_1d`, which receives a sorted 1D
workspace and returns the planner's AOD-native moves plus the sorted
positions where atoms end up. This base class:

- pins (or auto-detects) the motion axis and projects request sites to 1D
  row/column/diagonal coordinates,
- builds the workspace from `request.available_sites` or by auto-padding,
- chooses an adjacent free row/col/diagonal to use as the lift lane,
- realises every planner move as a lift / slide / lower triple of AODSteps
  so the slide path is obstruction-free,
- emits one final consolidation AODStep mapping the planner's natural
  output positions onto sorted(dst).
"""

from __future__ import annotations

import math
from abc import abstractmethod
from dataclasses import dataclass
from typing import Literal, Sequence

from ..movement import AODStep, DiagonalAODStep
from .base import SyncScheduler

Axis = Literal["row", "col", "anti_diag", "main_diag"]

PlannerMove = tuple[list[int], list[int]]

_AUTO_AXES: list[Axis] = ["row", "col", "anti_diag", "main_diag"]


def project_site(axis: Axis, site: tuple[int, int]) -> tuple[int, int]:
    """Return `(fixed, free)` 1D coordinates for a lattice site.

    Axis conventions:
      - "row": fixed `i`, free `j`.
      - "col": fixed `j`, free `i`.
      - "anti_diag": fixed `i + j`, free `i - j`.
      - "main_diag": fixed `i - j`, free `i + j`.
    """
    i, j = site
    if axis == "row":
        return i, j
    if axis == "col":
        return j, i
    if axis == "anti_diag":
        return i + j, i - j
    if axis == "main_diag":
        return i - j, i + j
    raise ValueError(f"unknown AOD projection axis: {axis!r}")


@dataclass
class AOD1DProjectedScheduler(SyncScheduler):
    """Plan in 1D, execute in 2D via a perpendicular lift detour.

    `axis` selects the planning direction:
      - "row": all src+dst share a row; planner moves along cols.
      - "col": all src+dst share a column; planner moves along rows.
      - "anti_diag": all src+dst share `i + j`; planner moves along `i - j`.
      - "main_diag": all src+dst share `i - j`; planner moves along `i + j`.
      - None: auto-detect (preferring row when both work).
    """

    axis: Axis | None = None

    def _plan(self) -> None:
        n = len(self.request.src)
        if n == 0:
            return

        axis, fixed, src_1d, dst_1d = self._project_to_1d()
        lift_to = self._lift_index(axis, fixed, n)
        workspace = self._build_workspace(n, src_1d, axis, fixed, lift_to)
        order = self._compute_order(src_1d, dst_1d)

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
        candidates: list[Axis] = [self.axis] if self.axis else _AUTO_AXES
        for axis in candidates:
            src_coords = [project_site(axis, p) for p in src]
            dst_coords = [project_site(axis, p) for p in dst]
            fixed_values = {fixed for fixed, _ in src_coords + dst_coords}
            if len(fixed_values) == 1:
                (fixed,) = fixed_values
                return (
                    axis,
                    fixed,
                    [free for _, free in src_coords],
                    [free for _, free in dst_coords],
                )

        if self.axis:
            raise ValueError(
                f"axis={self.axis!r} requires all src+dst on one projected line"
            )
        raise ValueError(
            f"{type(self).__name__} requires all src+dst sites on a single "
            "row, column, anti-diagonal, or main diagonal "
            "(set axis='row'/'col'/'anti_diag'/'main_diag' to pin the direction)"
        )

    def _build_workspace(
        self,
        n: int,
        src_1d: list[int],
        axis: Axis,
        fixed: int,
        lift_to: int,
    ) -> list[int]:
        """Sorted workspace along the motion axis.

        Uses `request.available_sites` when supplied; otherwise pads
        `src_1d` with neighbouring grid positions (right of max(src) first,
        then left of min(src)).
        """
        needed = self._workspace_size(n)
        if self.request.available_sites is not None:
            sites = self._project_available_sites(axis, fixed, lift_to)
            if len(sites) < needed:
                raise ValueError(
                    f"available_sites supplies {len(sites)} positions on "
                    f"the moving axis but the planner needs at least {needed}"
                )
            return sorted(sites)

        valid = sorted(
            set(self._line_free_coords(axis, fixed, allow_half=False))
            & set(self._line_free_coords(axis, lift_to, allow_half=True))
        )
        src_set = set(src_1d)
        if not src_set.issubset(valid):
            missing = sorted(src_set - set(valid))
            raise ValueError(
                f"src positions are not valid on both the storage and lift "
                f"lanes for axis={axis!r}: {missing}"
            )

        sites = list(src_1d)
        lo = min(sites)
        hi = max(sites)
        pool = (
            [p for p in valid if p not in src_set and p > hi]
            + [p for p in reversed(valid) if p not in src_set and p < lo]
            + [p for p in valid if p not in src_set and lo <= p <= hi]
        )
        sites.extend(pool[: needed - len(sites)])
        if len(sites) < needed:
            raise ValueError(
                f"need {needed} workspace positions but the grid extent "
                f"along axis={axis!r}, fixed={fixed}, lift={lift_to} "
                f"supplies only {len(valid)}"
            )
        return sorted(sites)

    def _project_available_sites(
        self,
        axis: Axis,
        fixed: int,
        lift_to: int,
    ) -> list[int]:
        valid_on_lift = set(self._line_free_coords(axis, lift_to, allow_half=True))
        out = []
        for site in self.request.available_sites:
            site_fixed, site_free = project_site(axis, site)
            if site_fixed != fixed:
                raise ValueError(
                    f"available_site {site} not on the request's shared "
                    f"axis={axis!r} line {fixed}"
                )
            if site_free not in valid_on_lift:
                raise ValueError(
                    f"available_site {site} does not have a matching lift-lane "
                    f"site for axis={axis!r}, lift={lift_to}"
                )
            out.append(site_free)
        return out

    def _lift_index(self, axis: Axis, fixed: int, n: int) -> int:
        """An adjacent row/col/diagonal used as the lifted slide lane."""
        needed = self._workspace_size(n)
        step = 1
        if axis in ("anti_diag", "main_diag"):
            rc_grid = self.request.grid.rc / self.request.grid.d
            step = max(1, math.floor(rc_grid * math.sqrt(2.0)) + 1)
        current = set(self._line_free_coords(axis, fixed, allow_half=False))
        for cand in (fixed + step, fixed - step):
            candidate = set(self._line_free_coords(axis, cand, allow_half=True))
            if len(current & candidate) >= needed:
                return cand
        raise ValueError(
            f"{type(self).__name__} needs an adjacent lift lane with at least "
            f"{needed} workspace sites for axis={axis!r}, fixed={fixed}"
        )

    def _line_free_coords(
        self,
        axis: Axis,
        fixed: int,
        *,
        allow_half: bool = False,
    ) -> list[int]:
        """All valid free coordinates on one projected line inside the grid."""
        N = self.request.grid.N
        if axis in ("anti_diag", "main_diag"):
            coords = []
            for free in range(-2 * N, 2 * N + 1):
                if self._projected_site_in_bounds(
                    axis,
                    fixed,
                    free,
                    allow_half=allow_half,
                ):
                    coords.append(free)
            return sorted(coords)

        coords = []
        for i in range(N):
            for j in range(N):
                site_fixed, site_free = project_site(axis, (i, j))
                if site_fixed == fixed:
                    coords.append(site_free)
        return sorted(coords)

    def _projected_site_in_bounds(
        self,
        axis: Axis,
        fixed: int,
        free: int,
        *,
        allow_half: bool,
    ) -> bool:
        N = self.request.grid.N
        if axis == "anti_diag":
            i = (fixed + free) / 2.0
            j = (fixed - free) / 2.0
        elif axis == "main_diag":
            i = (free + fixed) / 2.0
            j = (free - fixed) / 2.0
        else:
            return 0 <= fixed < N and 0 <= free < N
        if not (0 <= i < N and 0 <= j < N):
            return False
        if allow_half:
            return True
        return abs(i - round(i)) <= 1e-9 and abs(j - round(j)) <= 1e-9

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
            self.append_step(AODStep(
                start_time=self.sequence.next_start_time(),
                selected_rows=sel_rows,
                selected_cols=sel_cols,
                new_rows=new_rows,
                new_cols=new_cols,
            ))
            return
        if axis == "col":
            sel_rows, new_rows = tuple(frm), tuple(to)
            sel_cols, new_cols = (old_fixed,), (new_fixed,)
            self.append_step(AODStep(
                start_time=self.sequence.next_start_time(),
                selected_rows=sel_rows,
                selected_cols=sel_cols,
                new_rows=new_rows,
                new_cols=new_cols,
            ))
            return
        if axis == "anti_diag":
            self.append_step(DiagonalAODStep(
                start_time=self.sequence.next_start_time(),
                selected_sums=(old_fixed,),
                selected_diffs=tuple(frm),
                new_sums=(new_fixed,),
                new_diffs=tuple(to),
            ))
            return
        if axis == "main_diag":
            self.append_step(DiagonalAODStep(
                start_time=self.sequence.next_start_time(),
                selected_sums=tuple(frm),
                selected_diffs=(old_fixed,),
                new_sums=tuple(to),
                new_diffs=(new_fixed,),
            ))
            return
        raise ValueError(f"unknown AOD projection axis: {axis!r}")
