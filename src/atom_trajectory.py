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

from dataclasses import dataclass, field

import numpy as np

from .atom_config import AtomConfig, Grid
from .segments import Segment

# --- collision report and error --------------------------------------------


@dataclass
class CollisionReport:
    ok: bool
    # (candidate_atom_id, other_atom_id, t) at the closest sample.
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
    # Sample spacing (seconds) used by the collision validator.
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
        """Sample the candidate `segment` against every *other* atom's
        existing trajectory, and report the worst pairwise distance.

        Older segments don't need to be re-checked against each other —
        they were validated when each was appended. So the work per
        appended segment is O(M) sample-passes (one per other atom),
        not O(M^2). The candidate is read directly from `segment`; the
        ensemble is not mutated.
        """
        if segment.duration <= 0:
            return CollisionReport(ok=True)

        # dt-spaced samples over the candidate's window, plus boundaries
        # (already at the endpoints of np.linspace).
        n = max(2, int(np.ceil(segment.duration / self.collision_dt)) + 1)
        ts = np.linspace(segment.start_time, segment.end_time, n)

        # Convert grid-unit positions to physical um for the rc check.
        cand_xy = self.grid.ij_to_xy(
            np.array([segment.position_at(float(t)) for t in ts])
        )

        rc = self.grid.rc
        worst_d = float("inf")
        worst_pair = None
        for other in self.atomtrajs:
            if other.atom_id == atom_id:
                continue
            other_xy = self.grid.ij_to_xy(
                np.array([other.position_at(float(t)) for t in ts])
            )
            d = np.linalg.norm(cand_xy - other_xy, axis=1)
            ti = int(np.argmin(d))
            d_min = float(d[ti])
            if d_min < worst_d:
                worst_d = d_min
                worst_pair = (int(atom_id), int(other.atom_id), float(ts[ti]))

        return CollisionReport(
            ok=worst_d >= rc, worst_pair=worst_pair, worst_distance=worst_d
        )

    def check_segment(self, atom_id: int, segment: Segment) -> CollisionReport:
        """Non-mutating: would `segment` collide if appended? Also
        verifies continuity (raises ValueError on continuity failure)."""
        self.atomtraj_by_id(atom_id)._check_continuity(segment)
        return self._check_collision(atom_id, segment)

    # ---- mutation -----------------------------------------------------------

    def append_segment(self, atom_id: int, segment: Segment) -> None:
        """Validate then commit a segment to the addressed atom.

        Order:
          1. Per-atom continuity (cheap) — raises ValueError on failure,
             so we never run the expensive pass on a malformed segment.
          2. Cross-atom collision over the candidate's window — uses
             `collision_dt` for sampling. Raises `CollisionError` on
             failure; the ensemble is unchanged in that case.
          3. Commit to `AtomTrajectory.segments`.

        Pass `check_collisions=False` to skip step 2 (e.g. unit tests
        that intentionally construct overlapping motion).
        """
        atomtraj = self.atomtraj_by_id(atom_id)
        rep = self.check_segment(atom_id, segment)
        if not rep.ok:
            raise CollisionError(rep)
        atomtraj.segments.append(segment)
