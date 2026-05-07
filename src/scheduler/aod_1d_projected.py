"""Base scheduler for AOD planners that operate in 1D and lift to 2D.

Concrete subclasses implement `_plan_1d`, which receives a sorted 1D
workspace and returns the planner's AOD-native moves plus the sorted
positions where atoms end up. This base class:

- maps request sites into an explicit AOD projection's local 1D coordinates,
- builds the workspace from `request.available_sites` or by auto-padding,
- chooses an adjacent free local line to use while atoms are lifted,
- realises every planner move as a lift / slide / lower triple of AODSteps
  so the slide path is obstruction-free,
- emits one final consolidation AODStep mapping the planner's natural
  output positions onto sorted(dst).
"""

from __future__ import annotations

import math
from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Literal, Sequence

from ..atom_config import clean_coord
from ..movement import AODStep
from .base import SyncScheduler

LocalAxis = Literal["axis_1", "axis_2"]

PlannerMove = tuple[list[float], list[float]]


def unit_vector(theta: float) -> tuple[float, float]:
    """Unit vector at angle `theta` radians in grid-coordinate space."""
    x = math.cos(theta)
    y = math.sin(theta)
    if abs(x) <= 1e-12:
        x = 0.0
    if abs(y) <= 1e-12:
        y = 0.0
    return x, y


@dataclass(frozen=True)
class AODProjection:
    """AOD local coordinate system used for projected 1D planning.

    Physical position = `origin + coord_1 * axis_1 + coord_2 * axis_2`.
    `fixed_axis` selects which local coordinate is held fixed by a 1D
    subproblem; the other coordinate is the planner's free coordinate.
    The two basis vectors must be independent but need not be perpendicular.
    """

    axis_1: tuple[float, float] = (1.0, 0.0)
    axis_2: tuple[float, float] = (0.0, 1.0)
    origin: tuple[float, float] = (0.0, 0.0)
    fixed_axis: LocalAxis = "axis_1"
    workspace_step: float = 1.0
    name: str = "custom"

    def __post_init__(self):
        object.__setattr__(
            self,
            "axis_1",
            (float(self.axis_1[0]), float(self.axis_1[1])),
        )
        object.__setattr__(
            self,
            "axis_2",
            (float(self.axis_2[0]), float(self.axis_2[1])),
        )
        object.__setattr__(
            self,
            "origin",
            (float(self.origin[0]), float(self.origin[1])),
        )
        object.__setattr__(self, "workspace_step", float(self.workspace_step))
        object.__setattr__(self, "name", str(self.name))
        if self.workspace_step <= 0:
            raise ValueError("AODProjection.workspace_step must be positive")
        if abs(self.det()) <= 1e-12:
            raise ValueError("AODProjection basis vectors must be independent")
        if self.fixed_axis not in ("axis_1", "axis_2"):
            raise ValueError("fixed_axis must be 'axis_1' or 'axis_2'")

    def det(self) -> float:
        return (
            self.axis_1[0] * self.axis_2[1]
            - self.axis_1[1] * self.axis_2[0]
        )

    def site_from_local(self, coord_1: float, coord_2: float) -> tuple[float, float]:
        return (
            self.origin[0] + coord_1 * self.axis_1[0] + coord_2 * self.axis_2[0],
            self.origin[1] + coord_1 * self.axis_1[1] + coord_2 * self.axis_2[1],
        )

    def local_from_site(self, site: tuple[float, float]) -> tuple[float, float]:
        dx = float(site[0]) - self.origin[0]
        dy = float(site[1]) - self.origin[1]
        det = self.det()
        coord_1 = (dx * self.axis_2[1] - dy * self.axis_2[0]) / det
        coord_2 = (self.axis_1[0] * dy - self.axis_1[1] * dx) / det
        return coord_1, coord_2

    def project_site(self, site: tuple[float, float]) -> tuple[float, float]:
        coord_1, coord_2 = self.local_from_site(site)
        coord_1 = clean_coord(coord_1)
        coord_2 = clean_coord(coord_2)
        return (
            (coord_1, coord_2)
            if self.fixed_axis == "axis_1"
            else (coord_2, coord_1)
        )

    def fixed_axis_vector(self) -> tuple[float, float]:
        return self.axis_1 if self.fixed_axis == "axis_1" else self.axis_2

    def free_axis_vector(self) -> tuple[float, float]:
        return self.axis_2 if self.fixed_axis == "axis_1" else self.axis_1

    def fixed_axis_norm(self) -> float:
        vec = self.fixed_axis_vector()
        return math.hypot(vec[0], vec[1])

    def free_axis_norm(self) -> float:
        vec = self.free_axis_vector()
        return math.hypot(vec[0], vec[1])

    def fixed_coord_line_spacing(self) -> float:
        """Physical distance per unit fixed-coordinate change."""
        free_norm = self.free_axis_norm()
        return abs(self.det()) / free_norm

    def free_coord_lift_path_spacing(self) -> float:
        """Distance between unit-separated lift paths."""
        fixed_norm = self.fixed_axis_norm()
        return abs(self.det()) / fixed_norm


@dataclass
class AOD1DProjectedScheduler(SyncScheduler):
    """Plan in 1D, execute in 2D via a lifted local-coordinate detour.

    `projection` is the AOD basis and the local coordinate held fixed by
    the 1D subproblem. There is no automatic row/column/diagonal detection;
    callers pass the actual crossed-AOD geometry they want to use.
    """

    projection: AODProjection = field(
        default_factory=lambda: AODProjection(
            axis_1=unit_vector(0.0),
            axis_2=unit_vector(math.pi / 2.0),
            fixed_axis="axis_1",
            name="axis_1_lines",
        )
    )

    def _plan(self) -> None:
        n = len(self.request.src)
        if n == 0:
            return

        projection, fixed, src_1d, dst_1d = self._project_to_1d()
        lift_to = self._lift_index(projection, fixed, n)
        workspace = self._build_workspace(n, src_1d, projection, fixed, lift_to)
        order = self._compute_order(src_1d, dst_1d)

        moves, final_1d = self._plan_1d(workspace, src_1d, dst_1d, order)

        for frm, to in moves:
            self._append_aod(projection, fixed, lift_to, frm, frm)
            self._append_aod(projection, lift_to, lift_to, frm, to)
            self._append_aod(projection, lift_to, fixed, to, to)

        final_list = list(final_1d)
        sorted_dst = sorted(dst_1d)
        if final_list != sorted_dst:
            self._append_aod(projection, fixed, fixed, final_list, sorted_dst)

    # ---- subclass hooks -----------------------------------------------------

    @abstractmethod
    def _plan_1d(
        self,
        workspace: list[float],
        src_1d: list[float],
        dst_1d: list[float],
        order: list[int],
    ) -> tuple[Sequence[PlannerMove], Sequence[float]]:
        """Return `(moves, final_positions_sorted)` for the 1D subproblem."""

    @abstractmethod
    def _workspace_size(self, n: int) -> int:
        """Minimum workspace size the underlying 1D planner requires."""

    # ---- shared machinery ---------------------------------------------------

    def _project_to_1d(self) -> tuple[AODProjection, float, list[float], list[float]]:
        src, dst = self.request.src, self.request.dst
        projection = self.projection
        src_coords = [projection.project_site(p) for p in src]
        dst_coords = [projection.project_site(p) for p in dst]
        fixed_values = [fixed for fixed, _ in src_coords + dst_coords]
        fixed0 = fixed_values[0]
        if all(abs(fixed - fixed0) <= 1e-9 for fixed in fixed_values):
            return (
                projection,
                fixed0,
                [free for _, free in src_coords],
                [free for _, free in dst_coords],
            )
        raise ValueError(
            "the selected AOD projection requires all src+dst positions "
            "on one projected line"
        )

    def _build_workspace(
        self,
        n: int,
        src_1d: list[float],
        projection: AODProjection,
        fixed: float,
        lift_to: float,
    ) -> list[float]:
        """Sorted workspace along the motion axis.

        Uses `request.available_sites` when supplied; otherwise pads
        `src_1d` with neighbouring grid positions (right of max(src) first,
        then left of min(src)).
        """
        needed = self._workspace_size(n)
        if self.request.available_sites is not None:
            sites = self._project_available_sites(projection, fixed, lift_to)
            if len(sites) < needed:
                raise ValueError(
                    f"available_sites supplies {len(sites)} positions on "
                    f"the moving axis but the planner needs at least {needed}"
                )
            return sorted(sites)

        valid = sorted(
            self._line_free_coords(projection, fixed)
            + [p for p in src_1d if self._free_coord_in_bounds(projection, lift_to, p)]
        )
        valid = self._unique_sorted(valid)
        if not all(self._contains_close(valid, p) for p in src_1d):
            missing = sorted(p for p in src_1d if not self._contains_close(valid, p))
            raise ValueError(
                f"src positions are not valid on both the storage and lift "
                f"lanes for projection={projection.name!r}: {missing}"
            )

        sites = list(src_1d)
        lo = min(sites)
        hi = max(sites)
        pool = (
            [p for p in valid if not self._contains_close(sites, p) and p > hi]
            + [
                p
                for p in reversed(valid)
                if not self._contains_close(sites, p) and p < lo
            ]
            + [
                p
                for p in valid
                if not self._contains_close(sites, p) and lo <= p <= hi
            ]
        )
        sites.extend(pool[: needed - len(sites)])
        if len(sites) < needed:
            raise ValueError(
                f"need {needed} workspace positions but the grid extent "
                f"along projection={projection.name!r}, fixed={fixed}, lift={lift_to} "
                f"supplies only {len(valid)}"
            )
        return sorted(sites)

    def _project_available_sites(
        self,
        projection: AODProjection,
        fixed: float,
        lift_to: float,
    ) -> list[float]:
        out = []
        for site in self.request.available_sites:
            site_fixed, site_free = projection.project_site(site)
            if abs(site_fixed - fixed) > 1e-9:
                raise ValueError(
                    f"available_site {site} not on the request's shared "
                    f"projection={projection.name!r} line {fixed}"
                )
            if not self._free_coord_in_bounds(projection, lift_to, site_free):
                raise ValueError(
                    f"available_site {site} does not have a matching lifted-line "
                    f"site for projection={projection.name!r}, lift={lift_to}"
                )
            out.append(site_free)
        return out

    def _lift_index(self, projection: AODProjection, fixed: float, n: int) -> float:
        """An adjacent local fixed coordinate used for lifted slides."""
        needed = self._workspace_size(n)
        rc_grid = self.request.grid.rc / self.request.grid.d
        line_spacing = projection.fixed_coord_line_spacing()
        raw_offset = rc_grid / line_spacing if line_spacing > 0 else rc_grid
        step = projection.workspace_step
        offset = max(step, math.floor(raw_offset / step) * step + step)
        current = self._line_free_coords(projection, fixed)
        for cand in (fixed + offset, fixed - offset):
            candidate = self._line_free_coords(projection, cand)
            overlap = [
                free for free in current
                if self._contains_close(candidate, free)
            ]
            if len(overlap) >= needed:
                return cand
        raise ValueError(
            f"{type(self).__name__} needs an adjacent lifted line with at least "
            f"{needed} workspace sites for projection={projection.name!r}, "
            f"fixed={fixed}"
        )

    def _line_free_coords(
        self,
        projection: AODProjection,
        fixed: float,
    ) -> list[float]:
        """Discrete candidate free coordinates on one projected line."""
        N = self.request.grid.N
        corners = [(0.0, 0.0), (0.0, N - 1.0), (N - 1.0, 0.0), (N - 1.0, N - 1.0)]
        free_values = [projection.project_site(corner)[1] for corner in corners]
        step = self._free_coord_step(projection)
        lo = math.floor((min(free_values) - step) / step) * step
        hi = math.ceil((max(free_values) + step) / step) * step
        count = int(round((hi - lo) / step)) + 1
        coords = [
            lo + k * step
            for k in range(count)
            if self._free_coord_in_bounds(projection, fixed, lo + k * step)
        ]
        return self._unique_sorted(coords)

    def _free_coord_step(self, projection: AODProjection) -> float:
        free_norm = projection.free_axis_norm()
        lift_path_spacing = projection.free_coord_lift_path_spacing()
        unit_spacing = min(free_norm, lift_path_spacing)
        rc_grid = self.request.grid.rc / self.request.grid.d
        raw_step = rc_grid / unit_spacing if unit_spacing > 0 else rc_grid
        base = projection.workspace_step
        return max(base, math.floor(raw_step / base) * base + base)

    def _fixed_free_to_local(
        self,
        projection: AODProjection,
        fixed: float,
        free: float,
    ) -> tuple[float, float]:
        return (
            (fixed, free)
            if projection.fixed_axis == "axis_1"
            else (free, fixed)
        )

    def _free_coord_in_bounds(
        self,
        projection: AODProjection,
        fixed: float,
        free: float,
    ) -> bool:
        N = self.request.grid.N
        i, j = projection.site_from_local(
            *self._fixed_free_to_local(projection, fixed, free)
        )
        return 0 <= i < N and 0 <= j < N

    @staticmethod
    def _contains_close(values: Sequence[float], target: float) -> bool:
        return any(abs(value - target) <= 1e-9 for value in values)

    @staticmethod
    def _unique_sorted(values: Sequence[float]) -> list[float]:
        out: list[float] = []
        for value in sorted(float(v) for v in values):
            if not out or abs(out[-1] - value) > 1e-9:
                out.append(value)
        return out

    def _compute_order(self, src_1d: list[float], dst_1d: list[float]) -> list[int]:
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
        projection: AODProjection,
        old_fixed: float,
        new_fixed: float,
        frm: list[float],
        to: list[float],
    ) -> None:
        if projection.fixed_axis == "axis_1":
            selected_axis_1, new_axis_1 = (old_fixed,), (new_fixed,)
            selected_axis_2, new_axis_2 = tuple(frm), tuple(to)
        else:
            selected_axis_1, new_axis_1 = tuple(frm), tuple(to)
            selected_axis_2, new_axis_2 = (old_fixed,), (new_fixed,)
        self.append_step(AODStep(
            start_time=self.sequence.next_start_time(),
            selected_axis_1=selected_axis_1,
            selected_axis_2=selected_axis_2,
            new_axis_1=new_axis_1,
            new_axis_2=new_axis_2,
            axis_1=projection.axis_1,
            axis_2=projection.axis_2,
            origin=projection.origin,
        ))
