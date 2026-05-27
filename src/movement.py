"""Step types: how the scheduler tells the hardware to move atoms.

Both step types subclass `Step` and share one effect: they append
`Trajectory` segments to the relevant `AtomTrajectory`s in an
`AtomEnsemble`. They differ in how they pick atoms and shape moves:

* `AODStep` — synchronous outer-product lattice operation in an arbitrary
  floating-point AOD basis. Only atoms at the selected local-coordinate
  intersections move, all sharing one start_time and one duration.
* `RIPAStep` — single-atom move, addressed by `atom_id`. Travels along
  one EOM channel ('row' moves along x with j fixed, 'col' moves along
  y with i fixed). Diagonals must be split into multiple RIPASteps.

Steps carry their own absolute `start_time`. Schedulers compute it from
the ensemble's running `total_duration()` plus any inter-step gap.

Acceleration units. The hardware spec is naturally written in m/s^2
(physical), but `make_const_acc_segment` works in grid_units/s^2
(coordinate). The two module-level constants below are the physical
ceilings; each Step converts them to grid units at apply-time using
the ensemble's site spacing `grid.d`. Tune the constants in this file
to retarget different hardware.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

from .atom_config import clean_position, is_integer_position
from .atom_trajectory import AtomEnsemble
from .segments import bang_bang_duration, make_const_acc_segment, make_hold

# --- physical-acceleration ceiling ------------------------------------------
#
# Tune to match the actual hardware. Shared by AOD and RIPA moves.

PHYS_A_MAX: float = 2750.0  # m/s^2


def grid_accel_from_phys(phys_a_m_s2: float, grid_d_um: float) -> float:
    """Convert a physical acceleration (m/s^2) to grid_units/s^2."""
    return phys_a_m_s2 / (grid_d_um * 1e-6)


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
    """Synchronous AOD lattice op with an arbitrary 2D basis.

    The AOD addresses the cartesian product `selected_axis_1 x selected_axis_2`
    in local AOD coordinates. Physical grid coordinates are

        origin + axis_1_coord * axis_1 + axis_2_coord * axis_2

    Axis-aligned AODs use the default basis `(1, 0), (0, 1)`. Crossed AODs
    need only pass independent basis vectors; they do not need to be
    perpendicular.
    """

    start_time: float
    selected_axis_1: tuple[float, ...]
    selected_axis_2: tuple[float, ...]
    new_axis_1: tuple[float, ...]
    new_axis_2: tuple[float, ...]
    axis_1: tuple[float, float] = (1.0, 0.0)
    axis_2: tuple[float, float] = (0.0, 1.0)
    origin: tuple[float, float] = (0.0, 0.0)
    match_tol: float = 1e-9

    def __post_init__(self):
        self.start_time = float(self.start_time)
        if len(self.selected_axis_1) != len(self.new_axis_1):
            raise ValueError(
                "selected_axis_1 and new_axis_1 must have the same length"
            )
        if len(self.selected_axis_2) != len(self.new_axis_2):
            raise ValueError(
                "selected_axis_2 and new_axis_2 must have the same length"
            )

        self.selected_axis_1 = tuple(float(x) for x in self.selected_axis_1)
        self.selected_axis_2 = tuple(float(x) for x in self.selected_axis_2)
        self.new_axis_1 = tuple(float(x) for x in self.new_axis_1)
        self.new_axis_2 = tuple(float(x) for x in self.new_axis_2)
        self.axis_1 = (float(self.axis_1[0]), float(self.axis_1[1]))
        self.axis_2 = (float(self.axis_2[0]), float(self.axis_2[1]))
        self.origin = (float(self.origin[0]), float(self.origin[1]))

        if abs(self._basis_det()) <= self.match_tol:
            raise ValueError("AOD basis vectors must be linearly independent")

        # No-crossing: sorting old tone positions ascending must leave new tone
        # positions strictly ascending. Two RF tones sweeping past each other
        # on the same AOD would heat the atoms.
        for old, new, axis in (
            (self.selected_axis_1, self.new_axis_1, "axis_1"),
            (self.selected_axis_2, self.new_axis_2, "axis_2"),
        ):
            new_sorted_by_old = [n for _, n in sorted(zip(old, new))]
            for a, b in zip(new_sorted_by_old, new_sorted_by_old[1:]):
                if a >= b:
                    raise ValueError(
                        f"AOD {axis} would cross or merge: "
                        f"{list(old)} -> {list(new)}"
                    )

    def _basis_det(self) -> float:
        return (
            self.axis_1[0] * self.axis_2[1]
            - self.axis_1[1] * self.axis_2[0]
        )

    def _site_from_local(self, coord_1: float, coord_2: float) -> tuple[float, float]:
        return (
            self.origin[0] + coord_1 * self.axis_1[0] + coord_2 * self.axis_2[0],
            self.origin[1] + coord_1 * self.axis_1[1] + coord_2 * self.axis_2[1],
        )

    def _local_from_site(self, site: tuple[float, float]) -> tuple[float, float]:
        dx = float(site[0]) - self.origin[0]
        dy = float(site[1]) - self.origin[1]
        det = self._basis_det()
        coord_1 = (dx * self.axis_2[1] - dy * self.axis_2[0]) / det
        coord_2 = (self.axis_1[0] * dy - self.axis_1[1] * dx) / det
        return coord_1, coord_2

    def _match_index(self, value: float, choices: tuple[float, ...]) -> int | None:
        for k, choice in enumerate(choices):
            if abs(value - choice) <= self.match_tol:
                return k
        return None

    def _affected(self, ensemble: AtomEnsemble):
        """Yield (atom_id, old_pos, new_pos) for atoms hit by the lattice op."""
        for atomtraj in ensemble.atomtrajs:
            pos = atomtraj.resting_position_at(self.start_time)
            coord_1, coord_2 = self._local_from_site(pos)
            idx_1 = self._match_index(coord_1, self.selected_axis_1)
            idx_2 = self._match_index(coord_2, self.selected_axis_2)
            if idx_1 is not None and idx_2 is not None:
                yield atomtraj.atom_id, pos, self._site_from_local(
                    self.new_axis_1[idx_1],
                    self.new_axis_2[idx_2],
                )

    def _shared_duration(self, ensemble: AtomEnsemble) -> float:
        """Duration of the AOD op = bang-bang time of the longest atom move."""
        a = grid_accel_from_phys(PHYS_A_MAX, ensemble.grid.d)
        max_L = 0.0
        for _, (i, j), (ni, nj) in self._affected(ensemble):
            max_L = max(max_L, math.hypot(ni - i, nj - j))
        return bang_bang_duration(max_L, a)

    def apply(self, ensemble: AtomEnsemble) -> None:
        affected = list(self._affected(ensemble))
        T = self._shared_duration(ensemble)
        if T == 0:
            return  # nobody moves; nothing to record
        # All atoms share the longest move's duration. Shorter moves
        # therefore run with implied accel = 4*L/T**2 < a_max.
        segments = []
        for atom_id, start, end in affected:
            seg = make_const_acc_segment(
                start, end, self.start_time, duration=T, channel="aod"
            )
            segments.append((atom_id, seg))
        ensemble.append_segments_batch(segments)

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
    target: tuple[float, float]
    channel: Literal["row", "col"]

    def __post_init__(self):
        self.target = clean_position(self.target)
        if not is_integer_position(self.target):
            raise ValueError(f"RIPA target must be an integer grid site: {self.target}")

    def _current_pos(self, ensemble: AtomEnsemble) -> tuple[float, float]:
        """Resting site of the addressed atom at `start_time`, with a
        single-axis check against `channel`."""
        current = ensemble.atomtraj_by_id(self.atom_id).resting_position_at(
            self.start_time
        )
        if not is_integer_position(current):
            raise ValueError(
                f"RIPA can only pick up atoms at integer grid sites; got {current}"
            )
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
        return current

    def apply(self, ensemble: AtomEnsemble) -> None:
        current = self._current_pos(ensemble)
        if current == tuple(self.target):
            return
        a = grid_accel_from_phys(PHYS_A_MAX, ensemble.grid.d)
        seg = make_const_acc_segment(
            current, tuple(self.target), self.start_time, accel=a, channel=self.channel
        )
        ensemble.append_segment(self.atom_id, seg)

    def end_time(self, ensemble: AtomEnsemble) -> float:
        current = self._current_pos(ensemble)
        di = self.target[0] - current[0]
        dj = self.target[1] - current[1]
        L = math.hypot(di, dj)
        a = grid_accel_from_phys(PHYS_A_MAX, ensemble.grid.d)
        return self.start_time + bang_bang_duration(L, a)


# --- gate (stationary pulse on a set of atoms) ------------------------------


@dataclass
class GateStep(Step):
    """A gate pulse: hold a set of atoms parked at their rest sites for
    `duration` while the (global or local) laser does its thing.

    Hardware-agnostic and gate-agnostic — single-qubit rotations, CZ, CCZ,
    or any global Rydberg pulse all boil down to "freeze these atoms here
    for this long." The motion compiler doesn't care which gate it is;
    `gate_type`/`label` are metadata for downstream tooling (visualization,
    scheduling reports). The renderer in
    `src.visualization.draw_gate_overlay` duck-types on `atom_ids` +
    `start_time` + `duration`.
    """

    start_time: float
    atom_ids: tuple[int, ...]
    duration: float
    gate_type: str = "CZ"
    label: str = ""
    channel: str = "aod"

    def apply(self, ensemble: AtomEnsemble) -> None:
        if self.duration <= 0.0:
            return
        segments = []
        for atom_id in self.atom_ids:
            pos = ensemble.atomtraj_by_id(atom_id).resting_position_at(
                self.start_time
            )
            segments.append(
                (atom_id, make_hold(pos, self.start_time, self.duration,
                                    channel=self.channel))
            )
        ensemble.append_segments_batch(segments)

    def end_time(self, ensemble: AtomEnsemble) -> float:
        return self.start_time + max(0.0, float(self.duration))
