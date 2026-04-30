"""Per-atom trajectory + the AtomEnsemble that holds the whole timeline.

`AtomEnsemble` is the single mutation entry point: every new segment goes
through `append_segment`, which (a) per-atom continuity, then (b) cross-
atom collision against existing trajectories — both *before* the segment
is committed to the addressed `AtomTrajectory`.

`atom_id` is purely an internal index for tracking and visualization. It
does *not* mark atoms as physically distinguishable; whether atoms are
distinguishable is a property of the routing problem.

Conventions:
- Between segments, an atom rests at an integer site.
- Segments must be contiguous in space (next.start_pos == prev.end_pos)
  and non-overlapping in time (next.start_time >= prev.end_time).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .atom_config import AtomConfig, Grid
from .segments import Segment

_TIME_TOL = 1e-12


@dataclass(frozen=True)
class _TrajectoryPiece:
    start_time: float
    end_time: float
    segment: Segment | None = None
    position: tuple[float, float] | None = None

    def position_at(self, t: float) -> tuple[float, float]:
        if self.segment is not None:
            return self.segment.position_at(t)
        assert self.position is not None
        return self.position


# --- collision report and error --------------------------------------------


@dataclass
class CollisionReport:
    ok: bool
    # (candidate_atom_id, other_atom_id, t) at the closest detected approach.
    worst_pair: tuple[int, int, float] | None = None
    worst_distance: float = float("inf")  # in physical um


class CollisionError(ValueError):
    """Raised by `AtomEnsemble.append_segment` when a candidate segment
    collides with another atom's trajectory; carries the `CollisionReport`
    so callers can inspect which pair / time triggered the failure."""

    def __init__(self, report: CollisionReport):
        self.report = report
        a, b, t = report.worst_pair if report.worst_pair else (-1, -1, 0.0)
        super().__init__(
            f"collision: atoms {a} and {b} at t={t:.3e}s, "
            f"distance={report.worst_distance:.3e}um"
        )


@dataclass
class AtomTrajectory:
    atom_id: int
    initial_pos: tuple[int, int]  # site at t = 0
    segments: list[Segment] = field(default_factory=list)

    # ---- queries ------------------------------------------------------------

    def position_at(self, t: float) -> tuple[float, float]:
        """Position at global time t. Float during motion, integer at rest."""
        if not self.segments or t <= self.segments[0].start_time:
            return float(self.initial_pos[0]), float(self.initial_pos[1])
        last_end_pos = self.initial_pos
        for seg in self.segments:
            if t < seg.start_time:
                return float(last_end_pos[0]), float(last_end_pos[1])
            if t <= seg.end_time:
                return seg.position_at(t)
            last_end_pos = seg.end_pos
        return float(last_end_pos[0]), float(last_end_pos[1])

    def resting_position_at(self, t: float, *, tol: float = 1e-9) -> tuple[int, int]:
        """Integer site at time t. Errors if the atom is in motion at t."""
        for seg in self.segments:
            if seg.start_time + tol < t < seg.end_time - tol:
                raise ValueError(
                    f"atom {self.atom_id} is in motion at t={t} (segment "
                    f"[{seg.start_time}, {seg.end_time}])"
                )
        latest = self.initial_pos
        for seg in self.segments:
            if seg.end_time <= t + tol:
                latest = seg.end_pos
        return latest

    @property
    def final_pos(self) -> tuple[int, int]:
        return self.segments[-1].end_pos if self.segments else self.initial_pos

    @property
    def final_time(self) -> float:
        return self.segments[-1].end_time if self.segments else 0.0

    # ---- internal: continuity check ----------------------------------------

    def _check_continuity(self, segment: Segment) -> None:
        """Raise ValueError unless `segment` chains cleanly (start_pos
        matches the current resting site, start_time at or after the
        previous segment's end_time). External code mutates via
        `AtomEnsemble.append_segment`; this method exists so the ensemble
        can fail-fast on continuity *before* the expensive collision pass.
        """
        if segment.start_pos != self.final_pos:
            raise ValueError(
                f"atom {self.atom_id}: segment starts at {segment.start_pos} "
                f"but atom is at {self.final_pos}"
            )
        if segment.start_time + 1e-12 < self.final_time:
            raise ValueError(
                f"atom {self.atom_id}: segment starts at t={segment.start_time} "
                f"but previous segment ends at t={self.final_time}"
            )


@dataclass
class AtomEnsemble:
    """Collection of `AtomTrajectory`s sharing one Grid.

    Built from an `AtomConfig` (initial state). The only sanctioned way
    to add motion is `append_segment(atom_id, segment)`, which validates
    continuity AND cross-atom collisions before committing.
    """

    grid: Grid
    atomtrajs: list[AtomTrajectory] = field(default_factory=list)
    # Time tolerance scale used by the collision validator's profile search.
    collision_dt: float = 1e-6
    _id_index: dict[int, int] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self):
        self._rebuild_id_index()

    def _rebuild_id_index(self) -> None:
        self._id_index = {a.atom_id: k for k, a in enumerate(self.atomtrajs)}
        if len(self._id_index) != len(self.atomtrajs):
            raise ValueError("AtomEnsemble: atom_ids must be unique")

    @classmethod
    def from_config(cls, grid: Grid, cfg: AtomConfig) -> "AtomEnsemble":
        atoms = [
            AtomTrajectory(atom_id=int(aid), initial_pos=(int(p[0]), int(p[1])))
            for p, aid in zip(cfg.positions, cfg.atom_ids)
        ]
        return cls(grid=grid, atomtrajs=atoms)

    # ---- queries ------------------------------------------------------------

    def atomtraj_by_id(self, atom_id: int) -> AtomTrajectory:
        """Look up an atom by its `atom_id` (not by list index)."""
        return self.atomtrajs[self._id_index[int(atom_id)]]

    def index_of(self, atom_id: int) -> int:
        """List-index of the atom with the given `atom_id`. Useful when
        you need to align with `positions_at` / `xy_at` row order."""
        return self._id_index[int(atom_id)]

    def positions_at(self, t: float) -> np.ndarray:
        """All atom positions at time t, shape (M, 2), in grid units (float)."""
        return np.array([a.position_at(t) for a in self.atomtrajs], dtype=float)

    def xy_at(self, t: float) -> np.ndarray:
        """All atom positions at time t in physical (um) coords."""
        return self.grid.ij_to_xy(self.positions_at(t))

    def total_duration(self) -> float:
        return max((a.final_time for a in self.atomtrajs), default=0.0)

    def occupancy_at_rest(self, t: float) -> dict[tuple[int, int], int]:
        """Map site -> atom_id for atoms at rest at t. Errors on duplicates."""
        occ: dict[tuple[int, int], int] = {}
        for a in self.atomtrajs:
            site = a.resting_position_at(t)
            if site in occ:
                raise ValueError(f"two atoms at site {site} at t={t}")
            occ[site] = a.atom_id
        return occ

    def final_config(self) -> AtomConfig:
        """Snapshot of final resting positions, preserving atom_ids.
        The returned config does not carry the grid — fetch it from
        this `AtomEnsemble` if needed."""
        positions = np.array([a.final_pos for a in self.atomtrajs], dtype=int)
        atom_ids = np.array([a.atom_id for a in self.atomtrajs], dtype=int)
        return AtomConfig(positions=positions, atom_ids=atom_ids)

    # ---- collision check (non-mutating) ------------------------------------

    def _check_collision(self, atom_id: int, segment: Segment) -> CollisionReport:
        """Check the candidate `segment` against every other atom's timeline,
        and report the worst pairwise distance.

        Older segments don't need to be re-checked against each other —
        they were validated when each was appended. So the work per
        appended segment is O(M) timeline passes (one per other atom),
        not O(M^2). The candidate is read directly from `segment`; the
        ensemble is not mutated. Axis-separated path pairs are rejected
        geometrically before checking actual timing profiles.
        """
        return self._check_candidate_collisions({int(atom_id): segment})

    def _check_candidate_collisions(
        self, candidates: dict[int, Segment]
    ) -> CollisionReport:
        rc_grid = self.grid.rc / self.grid.d
        worst_grid = float("inf")
        worst_pair = None

        for atom_id, segment in candidates.items():
            if segment.duration <= 0:
                continue

            for other in self.atomtrajs:
                other_id = int(other.atom_id)
                if other_id == atom_id:
                    continue

                d_grid, t = self._min_distance_to_atom(
                    segment,
                    other,
                    candidates.get(other_id),
                    rc_grid,
                )
                if d_grid < worst_grid:
                    worst_grid = d_grid
                    worst_pair = (int(atom_id), other_id, float(t))

        worst_um = worst_grid * self.grid.d
        return CollisionReport(
            ok=worst_um + 1e-12 >= self.grid.rc,
            worst_pair=worst_pair,
            worst_distance=worst_um,
        )

    def _min_distance_to_atom(
        self,
        segment: Segment,
        other: AtomTrajectory,
        other_candidate: Segment | None,
        rc_grid: float,
    ) -> tuple[float, float]:
        if segment.duration <= 0:
            return float("inf"), segment.start_time

        best_grid = float("inf")
        best_t = segment.start_time
        for piece in self._iter_atom_pieces(
            other, segment.start_time, segment.end_time, other_candidate
        ):
            t0 = max(segment.start_time, piece.start_time)
            t1 = min(segment.end_time, piece.end_time)
            if t1 < t0 - _TIME_TOL:
                continue

            d_grid, t = self._distance_to_piece(segment, piece, t0, t1, rc_grid)
            if d_grid < best_grid:
                best_grid = d_grid
                best_t = t

        return best_grid, best_t

    def _check_batch_collision(self, candidates: dict[int, Segment]) -> CollisionReport:
        """Check simultaneous candidate segments as one mutation."""
        return self._check_candidate_collisions(candidates)

    def _iter_atom_pieces(
        self,
        atomtraj: AtomTrajectory,
        start_time: float,
        end_time: float,
        candidate: Segment | None = None,
    ):
        if candidate is None:
            yield from self._iter_existing_pieces(atomtraj, start_time, end_time)
            return

        before_end = min(end_time, candidate.start_time)
        if start_time < before_end - _TIME_TOL:
            yield from self._iter_existing_pieces(atomtraj, start_time, before_end)

        move_start = max(start_time, candidate.start_time)
        move_end = min(end_time, candidate.end_time)
        if candidate.duration > 0 and move_start < move_end - _TIME_TOL:
            yield _TrajectoryPiece(move_start, move_end, segment=candidate)

        rest_start = max(start_time, candidate.end_time)
        if rest_start < end_time - _TIME_TOL:
            yield _TrajectoryPiece(
                rest_start,
                end_time,
                position=(float(candidate.end_pos[0]), float(candidate.end_pos[1])),
            )

    @staticmethod
    def _iter_existing_pieces(
        atomtraj: AtomTrajectory, start_time: float, end_time: float
    ):
        cursor = start_time
        rest_pos = atomtraj.initial_pos

        for seg in atomtraj.segments:
            if seg.end_time <= start_time + _TIME_TOL:
                rest_pos = seg.end_pos
                continue
            if seg.start_time >= end_time - _TIME_TOL:
                break

            rest_end = min(seg.start_time, end_time)
            if cursor < rest_end - _TIME_TOL:
                yield _TrajectoryPiece(
                    cursor,
                    rest_end,
                    position=(float(rest_pos[0]), float(rest_pos[1])),
                )
                cursor = rest_end

            move_start = max(seg.start_time, cursor, start_time)
            move_end = min(seg.end_time, end_time)
            if move_start < move_end - _TIME_TOL:
                yield _TrajectoryPiece(move_start, move_end, segment=seg)
                cursor = move_end

            if seg.end_time <= cursor + _TIME_TOL:
                rest_pos = seg.end_pos
            else:
                return

        if cursor < end_time - _TIME_TOL:
            yield _TrajectoryPiece(
                cursor,
                end_time,
                position=(float(rest_pos[0]), float(rest_pos[1])),
            )

    def _distance_to_piece(
        self,
        segment: Segment,
        piece: _TrajectoryPiece,
        t0: float,
        t1: float,
        rc_grid: float,
    ) -> tuple[float, float]:
        if t1 <= t0 + _TIME_TOL:
            d = self._distance_grid(segment.position_at(t0), piece.position_at(t0))
            return d, t0

        if piece.segment is None:
            assert piece.position is not None
            return self._moving_point_distance(segment, t0, t1, piece.position)

        lower = self._bbox_distance_grid(
            self._segment_bbox(segment, t0, t1),
            self._segment_bbox(piece.segment, t0, t1),
        )
        if lower >= rc_grid:
            return lower, t0

        return self._moving_segment_distance(segment, piece.segment, t0, t1)

    def _moving_point_distance(
        self,
        segment: Segment,
        t0: float,
        t1: float,
        point: tuple[float, float],
    ) -> tuple[float, float]:
        p0 = segment.position_at(t0)
        p1 = segment.position_at(t1)
        vx = p1[0] - p0[0]
        vy = p1[1] - p0[1]
        length2 = vx * vx + vy * vy
        if length2 <= 0:
            return self._distance_grid(p0, point), t0

        u = ((point[0] - p0[0]) * vx + (point[1] - p0[1]) * vy) / length2
        u = max(0.0, min(1.0, u))
        closest = (p0[0] + u * vx, p0[1] + u * vy)
        t = self._time_for_segment_point(segment, closest, t0, t1, u)
        return self._distance_grid(closest, point), t

    def _moving_segment_distance(
        self,
        a: Segment,
        b: Segment,
        t0: float,
        t1: float,
    ) -> tuple[float, float]:
        breakpoints = self._collision_breakpoints(a, b, t0, t1)
        best_t = breakpoints[0]
        best_d2 = self._distance2_at(a, b, best_t)

        for t in breakpoints[1:]:
            d2 = self._distance2_at(a, b, t)
            if d2 < best_d2:
                best_d2 = d2
                best_t = t

        for lo, hi in zip(breakpoints, breakpoints[1:]):
            if hi <= lo + _TIME_TOL:
                continue
            t, d2 = self._golden_min_distance2(a, b, lo, hi)
            if d2 < best_d2:
                best_d2 = d2
                best_t = t

        return math.sqrt(best_d2), best_t

    def _collision_breakpoints(
        self, a: Segment, b: Segment, t0: float, t1: float
    ) -> list[float]:
        points = {float(t0), float(t1)}
        for seg in (a, b):
            for frac in (0.25, 0.5, 0.75):
                t = seg.start_time + frac * seg.duration
                if t0 < t < t1:
                    points.add(float(t))
            for t in (seg.start_time, seg.end_time):
                if t0 < t < t1:
                    points.add(float(t))

        for t in self._path_crossing_times(a, b, t0, t1):
            if t0 < t < t1:
                points.add(float(t))

        return sorted(points)

    def _path_crossing_times(
        self, a: Segment, b: Segment, t0: float, t1: float
    ) -> list[float]:
        if a.motion_axis == "x" and b.motion_axis == "y":
            return [
                self._time_for_axis_value(a, b.start_pos[0], 0, t0, t1),
                self._time_for_axis_value(b, a.start_pos[1], 1, t0, t1),
            ]
        if a.motion_axis == "y" and b.motion_axis == "x":
            return [
                self._time_for_axis_value(a, b.start_pos[1], 1, t0, t1),
                self._time_for_axis_value(b, a.start_pos[0], 0, t0, t1),
            ]
        return []

    def _golden_min_distance2(
        self, a: Segment, b: Segment, lo: float, hi: float
    ) -> tuple[float, float]:
        phi = (math.sqrt(5.0) - 1.0) / 2.0
        x1 = hi - phi * (hi - lo)
        x2 = lo + phi * (hi - lo)
        f1 = self._distance2_at(a, b, x1)
        f2 = self._distance2_at(a, b, x2)
        tol = max(_TIME_TOL, min(self.collision_dt * 1e-3, (hi - lo) * 1e-9))

        for _ in range(64):
            if hi - lo <= tol:
                break
            if f1 <= f2:
                hi = x2
                x2 = x1
                f2 = f1
                x1 = hi - phi * (hi - lo)
                f1 = self._distance2_at(a, b, x1)
            else:
                lo = x1
                x1 = x2
                f1 = f2
                x2 = lo + phi * (hi - lo)
                f2 = self._distance2_at(a, b, x2)

        candidates = [
            (lo, self._distance2_at(a, b, lo)),
            (hi, self._distance2_at(a, b, hi)),
            (x1, f1),
            (x2, f2),
        ]
        return min(candidates, key=lambda item: item[1])

    @staticmethod
    def _distance2_at(a: Segment, b: Segment, t: float) -> float:
        pa = a.position_at(t)
        pb = b.position_at(t)
        dx = pa[0] - pb[0]
        dy = pa[1] - pb[1]
        return dx * dx + dy * dy

    def _time_for_segment_point(
        self,
        segment: Segment,
        point: tuple[float, float],
        t0: float,
        t1: float,
        fallback_fraction: float,
    ) -> float:
        axis = segment.motion_axis
        if axis == "x":
            return self._time_for_axis_value(segment, point[0], 0, t0, t1)
        if axis == "y":
            return self._time_for_axis_value(segment, point[1], 1, t0, t1)
        return t0 + fallback_fraction * (t1 - t0)

    @staticmethod
    def _time_for_axis_value(
        segment: Segment,
        value: float,
        coord_index: int,
        t0: float,
        t1: float,
    ) -> float:
        p0 = segment.position_at(t0)[coord_index]
        p1 = segment.position_at(t1)[coord_index]
        lo_value = min(p0, p1)
        hi_value = max(p0, p1)
        if value < lo_value - 1e-12 or value > hi_value + 1e-12:
            return t0 if abs(p0 - value) <= abs(p1 - value) else t1
        if abs(p1 - p0) <= 1e-15:
            return t0

        lo = t0
        hi = t1
        increasing = p1 >= p0
        for _ in range(64):
            mid = (lo + hi) / 2.0
            mid_value = segment.position_at(mid)[coord_index]
            if (mid_value < value) == increasing:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2.0

    @staticmethod
    def _segment_bbox(
        segment: Segment, t0: float, t1: float
    ) -> tuple[float, float, float, float]:
        p0 = segment.position_at(t0)
        p1 = segment.position_at(t1)
        return (
            min(p0[0], p1[0]),
            max(p0[0], p1[0]),
            min(p0[1], p1[1]),
            max(p0[1], p1[1]),
        )

    @staticmethod
    def _bbox_distance_grid(
        a: tuple[float, float, float, float],
        b: tuple[float, float, float, float],
    ) -> float:
        dx = max(b[0] - a[1], a[0] - b[1], 0.0)
        dy = max(b[2] - a[3], a[2] - b[3], 0.0)
        return math.hypot(dx, dy)

    @staticmethod
    def _distance_grid(
        a: tuple[float, float], b: tuple[float, float]
    ) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def check_segment(self, atom_id: int, segment: Segment) -> CollisionReport:
        """Non-mutating: would `segment` collide if appended? Also
        verifies continuity (raises ValueError on continuity failure)."""
        self.atomtraj_by_id(atom_id)._check_continuity(segment)
        return self._check_collision(atom_id, segment)

    # ---- mutation -----------------------------------------------------------

    def append_segment(self, atom_id: int, segment: Segment) -> None:
        """Validate then commit a segment to the addressed atom.

        Order:
          1. Per-atom continuity (cheap) raises ValueError on failure,
             so we never run collision checks on a malformed segment.
          2. Cross-atom collision over the candidate's window raises
             `CollisionError` on failure; the ensemble is unchanged.
          3. Commit to `AtomTrajectory.segments`.
        """
        atomtraj = self.atomtraj_by_id(atom_id)
        rep = self.check_segment(atom_id, segment)
        if not rep.ok:
            raise CollisionError(rep)
        atomtraj.segments.append(segment)

    def append_segments_batch(self, segments: list[tuple[int, Segment]]) -> None:
        """Validate then commit multiple same-cycle segments together."""
        if not segments:
            return

        candidates: dict[int, Segment] = {}
        for atom_id, segment in segments:
            atom_id = int(atom_id)
            if atom_id in candidates:
                raise ValueError(f"duplicate candidate segment for atom {atom_id}")
            candidates[atom_id] = segment

        for atom_id, segment in candidates.items():
            self.atomtraj_by_id(atom_id)._check_continuity(segment)

        rep = self._check_batch_collision(candidates)
        if not rep.ok:
            raise CollisionError(rep)

        for atom_id, segment in segments:
            self.atomtraj_by_id(atom_id).segments.append(segment)
