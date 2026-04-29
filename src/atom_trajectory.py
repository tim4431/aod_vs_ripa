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

from .atom_config import AtomConfig, Grid
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
    by appending segments to atoms' trajectories. The list `atoms` is
    kept in the same order as the source `AtomConfig.positions`; lookup
    by `atom_id` is via `atom_by_id` (O(1) through the cached id index).
    """

    grid: Grid
    atoms: list[AtomTrajectory] = field(default_factory=list)
    _id_index: dict[int, int] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self):
        self._rebuild_id_index()

    def _rebuild_id_index(self) -> None:
        self._id_index = {a.atom_id: k for k, a in enumerate(self.atoms)}
        if len(self._id_index) != len(self.atoms):
            raise ValueError("AtomEnsemble: atom_ids must be unique")

    @classmethod
    def from_config(cls, cfg: AtomConfig) -> "AtomEnsemble":
        atoms = [
            AtomTrajectory(atom_id=int(aid), initial_pos=(int(p[0]), int(p[1])))
            for p, aid in zip(cfg.positions, cfg.atom_ids)
        ]
        return cls(grid=cfg.grid, atoms=atoms)

    # ---- queries ------------------------------------------------------------

    def atom_by_id(self, atom_id: int) -> AtomTrajectory:
        """Look up an atom by its `atom_id` (not by list index)."""
        return self.atoms[self._id_index[int(atom_id)]]

    def index_of(self, atom_id: int) -> int:
        """List-index of the atom with the given `atom_id`. Useful when
        you need to align with `positions_at` / `xy_at` row order."""
        return self._id_index[int(atom_id)]

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
        """Snapshot of final resting positions, preserving atom_ids."""
        positions = np.array([a.final_pos for a in self.atoms], dtype=int)
        atom_ids = np.array([a.atom_id for a in self.atoms], dtype=int)
        return AtomConfig(grid=self.grid, positions=positions, atom_ids=atom_ids)
