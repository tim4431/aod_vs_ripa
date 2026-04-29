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

Acceleration units. The hardware spec is naturally written in m/s^2
(physical), but `make_const_acc_segment` works in grid_units/s^2
(coordinate). The two constants below are the physical ceilings; each
Step converts them to grid units at apply-time using the ensemble's
site spacing `grid.d`. Override per-step by passing `a_max=...` (in
grid_units/s^2) — useful for tests or for slowing specific moves.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal, Optional

from .atom_config import Grid
from .atom_trajectory import AtomEnsemble
from .segments import bang_bang_duration, make_const_acc_segment


# --- physical-acceleration ceilings -----------------------------------------
#
# Tune these to match the actual hardware. AOD and RIPA are listed
# separately because per-atom heating / loss budgets can differ even
# when the trap optics are similar; in many systems they end up equal.

PHYS_A_MAX_AOD: float = 2750.0    # m/s^2 — synchronous AOD lattice ops
PHYS_A_MAX_RIPA: float = 2750.0   # m/s^2 — RIPA per-atom moves


def grid_accel_from_phys(phys_a_m_s2: float, grid_d_um: float) -> float:
    """Convert a physical acceleration (m/s^2) to grid_units/s^2.

    A grid unit equals `grid_d_um * 1e-6` meters, so

        a [grid/s^2] = phys_a [m/s^2] / (grid_d_um * 1e-6).
    """
    return phys_a_m_s2 / (grid_d_um * 1e-6)


def _resolve_a_max(a_max: Optional[float], grid: Grid, phys_default: float) -> float:
    """Resolve a Step's `a_max` to grid_units/s^2: explicit override
    if given, else convert `phys_default` (m/s^2) via grid.d."""
    if a_max is not None:
        return a_max
    return grid_accel_from_phys(phys_default, grid.d)


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
    # grid_units/s^2 override; None -> convert PHYS_A_MAX_AOD via grid.d.
    a_max: Optional[float] = None

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
        a = _resolve_a_max(self.a_max, ensemble.grid, PHYS_A_MAX_AOD)
        max_L = 0.0
        for _, (i, j), (ni, nj) in self._affected(ensemble):
            max_L = max(max_L, math.hypot(ni - i, nj - j))
        return bang_bang_duration(max_L, a)

    def apply(self, ensemble: AtomEnsemble) -> None:
        T = self._shared_duration(ensemble)
        if T == 0:
            return  # nobody moves; nothing to record
        # All atoms share the longest move's duration. Shorter moves
        # therefore run with implied accel = 4*L/T**2 < a_max.
        for atom, start, end in self._affected(ensemble):
            seg = make_const_acc_segment(start, end, self.start_time,
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
    # grid_units/s^2 override; None -> convert PHYS_A_MAX_RIPA via grid.d.
    a_max: Optional[float] = None

    def _move_for(self, ensemble: AtomEnsemble):
        atom = ensemble.atom_by_id(self.atom_id)
        current = atom.resting_position_at(self.start_time)
        di = self.target[0] - current[0]
        dj = self.target[1] - current[1]
        if self.channel == "row" and dj != 0:
            raise ValueError(
                f"row-channel move must keep j fixed; got {current}->{self.target}"
            )
        if self.channel == "col" and di != 0:
            raise ValueError(
                f"col-channel move must keep i fixed; got {current}->{self.target}"
            )
        return atom, current

    def apply(self, ensemble: AtomEnsemble) -> None:
        atom, current = self._move_for(ensemble)
        if current == tuple(self.target):
            return
        # RIPA single-atom move: run at the step's a_max — schedulers
        # can drop a_max on individual steps to slow specific atoms.
        a = _resolve_a_max(self.a_max, ensemble.grid, PHYS_A_MAX_RIPA)
        seg = make_const_acc_segment(current, tuple(self.target),
                                     self.start_time,
                                     accel=a, channel=self.channel)
        atom.append(seg)

    def end_time(self, ensemble: AtomEnsemble) -> float:
        _, current = self._move_for(ensemble)
        di = self.target[0] - current[0]
        dj = self.target[1] - current[1]
        L = math.hypot(di, dj)
        a = _resolve_a_max(self.a_max, ensemble.grid, PHYS_A_MAX_RIPA)
        return self.start_time + bang_bang_duration(L, a)
