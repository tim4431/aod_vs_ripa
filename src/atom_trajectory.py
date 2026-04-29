"""Per-atom trajectory and the AtomEnsemble that holds the whole timeline.

This module replaces the old "AtomConfig snapshot at every step" model.
Atom motion is intrinsically asynchronous (RIPA addresses individual
atoms with independent EOM tones), so the natural representation is one
`AtomTrajectory` per atom, each carrying a list of timed `Trajectory`
segments.

`atom_id` is purely an internal index for tracking and visualization. It
does *not* mark atoms as physically distinguishable; whether atoms are
distinguishable is a property of the routing problem, not of the atoms.

Conventions:
- Between segments the atom rests at an integer site.
- Segments must be contiguous in space (next.start_pos == prev.end_pos)
  and non-overlapping in time (next.start_time >= prev.end_time).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .atoms import AtomConfig, Grid
from .segments import Segment


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
        # Linear scan; segments are short lists in practice.
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

    # ---- mutation -----------------------------------------------------------

    def append(self, segment: Segment) -> None:
        """Append a segment, validating spatial + temporal continuity."""
        prev_pos = self.final_pos
        prev_time = self.final_time
        if segment.start_pos != prev_pos:
            raise ValueError(
                f"atom {self.atom_id}: segment starts at {segment.start_pos} "
                f"but atom is at {prev_pos}"
            )
        if segment.start_time + 1e-12 < prev_time:
            raise ValueError(
                f"atom {self.atom_id}: segment starts at t={segment.start_time} "
                f"but previous segment ends at t={prev_time}"
            )
        self.segments.append(segment)


@dataclass
class AtomEnsemble:
    """Collection of `AtomTrajectory`s sharing one Grid.

    Built from an `AtomConfig` (initial state). Steps mutate this object
    by appending segments to atoms' trajectories.
    """

    grid: Grid
    atoms: list[AtomTrajectory] = field(default_factory=list)

    @classmethod
    def from_config(cls, cfg: AtomConfig) -> "AtomEnsemble":
        atoms = [
            AtomTrajectory(atom_id=k, initial_pos=(int(p[0]), int(p[1])))
            for k, p in enumerate(cfg.positions)
        ]
        return cls(grid=cfg.grid, atoms=atoms)

    # ---- queries ------------------------------------------------------------

    def positions_at(self, t: float) -> np.ndarray:
        """All atom positions at time t, shape (M, 2), in grid units (float)."""
        return np.array([a.position_at(t) for a in self.atoms], dtype=float)

    def xy_at(self, t: float) -> np.ndarray:
        """All atom positions at time t in physical (um) coords."""
        c = (self.grid.N - 1) / 2.0
        return (self.positions_at(t) - c) * self.grid.d

    def total_duration(self) -> float:
        return max((a.final_time for a in self.atoms), default=0.0)

    def occupancy_at_rest(self, t: float) -> dict[tuple[int, int], int]:
        """Map site -> atom_id for atoms at rest at t. Errors on duplicates."""
        occ: dict[tuple[int, int], int] = {}
        for a in self.atoms:
            site = a.resting_position_at(t)
            if site in occ:
                raise ValueError(f"two atoms at site {site} at t={t}")
            occ[site] = a.atom_id
        return occ

    def final_config(self) -> AtomConfig:
        positions = np.array([a.final_pos for a in self.atoms], dtype=int)
        return AtomConfig(grid=self.grid, positions=positions)

    def moving_intervals(self) -> list[tuple[float, float]]:
        """Merged time intervals during which any atom is moving.

        Useful so the validator only samples when something is actually
        in flight rather than across the entire wall-clock duration.
        """
        intervals = sorted(
            (seg.start_time, seg.end_time)
            for a in self.atoms
            for seg in a.segments
            if seg.duration > 0
        )
        merged: list[tuple[float, float]] = []
        for s, e in intervals:
            if merged and s <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], e))
            else:
                merged.append((s, e))
        return merged
