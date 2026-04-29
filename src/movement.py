"""Step types: how the scheduler tells the hardware to move atoms.

Both step types subclass `Step` and share one effect: they append
`Trajectory` segments to the relevant `AtomTrajectory`s in an
`AtomEnsemble`. They differ in how they pick atoms and shape moves:

* `AODStep` — synchronous lattice operation. Defined by selected
  rows/cols and their new positions; only atoms at the cartesian
  product (selected_rows x selected_cols) are moved, all sharing one
  start_time and one duration (set by the longest move).

* `RIPAStep` — single-atom move, addressed by `atom_id`. Travels along
  one EOM channel ('row' moves along x with j fixed, 'col' moves along
  y with i fixed). Diagonals must be split into multiple RIPASteps.

Steps carry their own absolute `start_time`. Schedulers compute it from
the ensemble's running `total_duration()` plus any inter-step gap.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

from .atom_trajectory import AtomEnsemble
from .trajectories import bang_bang_duration, make_segment


class Step(ABC):
    """Abstract step. Subclasses implement `apply(ensemble)`."""

    start_time: float

    @abstractmethod
    def apply(self, ensemble: AtomEnsemble) -> None:
        """Append trajectory segments to the relevant atoms in `ensemble`."""

    @abstractmethod
    def end_time(self, ensemble: AtomEnsemble) -> float:
        """Latest end_time of any segment this step would emit. May depend on
        the ensemble state at `start_time`. Used by the scheduler to chain
        consecutive steps."""


# --- AOD --------------------------------------------------------------------

@dataclass
class AODStep(Step):
    """Synchronous AOD lattice op.

    The AOD addresses the cartesian product `selected_rows x selected_cols`
    via one RF tone per row and one per col. Atoms sitting at any of those
    intersections move together; their (row, col) indices remap to the
    matching entries in `new_rows`, `new_cols`. Lengths must match.

    Monotonicity (no row/col crossings) is the caller's responsibility —
    the sqrt-time scheduler emits monotone shifts by construction.
    """
    start_time: float
    selected_rows: tuple[int, ...]
    selected_cols: tuple[int, ...]
    new_rows: tuple[int, ...]
    new_cols: tuple[int, ...]
    a_max: float = 1.0

    def __post_init__(self):
        if len(self.selected_rows) != len(self.new_rows):
            raise ValueError("selected_rows and new_rows must have the same length")
        if len(self.selected_cols) != len(self.new_cols):
            raise ValueError("selected_cols and new_cols must have the same length")

    def _affected(self, ensemble: AtomEnsemble):
        """Yield (atom, old_site, new_site) for atoms hit by the lattice op."""
        row_map = dict(zip(self.selected_rows, self.new_rows))
        col_map = dict(zip(self.selected_cols, self.new_cols))
        sel_r = set(self.selected_rows)
        sel_c = set(self.selected_cols)
        for atom in ensemble.atoms:
            i, j = atom.resting_position_at(self.start_time)
            if i in sel_r and j in sel_c:
                yield atom, (i, j), (row_map[i], col_map[j])

    def _shared_duration(self, ensemble: AtomEnsemble) -> float:
        """Duration of the AOD op = bang-bang time of the longest atom move."""
        max_L = 0.0
        for _, (i, j), (ni, nj) in self._affected(ensemble):
            max_L = max(max_L, math.hypot(ni - i, nj - j))
        return bang_bang_duration(max_L, self.a_max)

    def apply(self, ensemble: AtomEnsemble) -> None:
        T = self._shared_duration(ensemble)
        if T == 0:
            return  # nobody moves; nothing to record
        for atom, start, end in self._affected(ensemble):
            seg = make_segment(start, end, self.start_time, self.a_max,
                               duration=T, channel="aod")
            atom.append(seg)

    def end_time(self, ensemble: AtomEnsemble) -> float:
        return self.start_time + self._shared_duration(ensemble)


# --- RIPA -------------------------------------------------------------------

@dataclass
class RIPAStep(Step):
    """Single-atom RIPA move along one EOM channel.

    `channel` selects the row channel ('row', moves along x with j fixed)
    or the col channel ('col', moves along y with i fixed). Diagonal moves
    are not allowed and must be split.
    """
    start_time: float
    atom_id: int
    target: tuple[int, int]
    channel: Literal["row", "col"]
    a_max: float = 1.0

    def _move_for(self, ensemble: AtomEnsemble):
        atom = ensemble.atoms[self.atom_id]
        current = atom.resting_position_at(self.start_time)
        di = self.target[0] - current[0]
        dj = self.target[1] - current[1]
        if self.channel == "row" and dj != 0:
            raise ValueError(f"row-channel move must keep j fixed; got {current}->{self.target}")
        if self.channel == "col" and di != 0:
            raise ValueError(f"col-channel move must keep i fixed; got {current}->{self.target}")
        return atom, current

    def apply(self, ensemble: AtomEnsemble) -> None:
        atom, current = self._move_for(ensemble)
        if current == tuple(self.target):
            return
        seg = make_segment(current, tuple(self.target), self.start_time,
                           self.a_max, channel=self.channel)
        atom.append(seg)

    def end_time(self, ensemble: AtomEnsemble) -> float:
        _, current = self._move_for(ensemble)
        di = self.target[0] - current[0]
        dj = self.target[1] - current[1]
        L = math.hypot(di, dj)
        return self.start_time + bang_bang_duration(L, self.a_max)
