"""Small direct-move schedulers.

These are intentionally simple baselines: useful for smoke tests, labeled
requests, and early experimentation before a collision-aware async planner
exists.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ..movement import RIPAStep
from .base import SyncScheduler

AxisOrder = Literal["row_col", "col_row"]


@dataclass
class NaiveRIPAScheduler(SyncScheduler):
    """Move each atom directly with one or two single-axis RIPA steps."""

    axis_order: AxisOrder = "row_col"

    def _plan(self) -> None:
        if self.axis_order not in ("row_col", "col_row"):
            raise ValueError("axis_order must be 'row_col' or 'col_row'")

        for atom_id, target in self.target_assignment().items():
            if self.axis_order == "row_col":
                self._move_atom_row_then_col(atom_id, target)
            else:
                self._move_atom_col_then_row(atom_id, target)

    def _move_atom_row_then_col(self, atom_id: int, target: tuple[int, int]) -> None:
        current = self.ensemble.atomtraj_by_id(atom_id).final_pos
        waypoint = (target[0], current[1])
        self._append_ripa_leg(atom_id, waypoint, "row")
        self._append_ripa_leg(atom_id, target, "col")

    def _move_atom_col_then_row(self, atom_id: int, target: tuple[int, int]) -> None:
        current = self.ensemble.atomtraj_by_id(atom_id).final_pos
        waypoint = (current[0], target[1])
        self._append_ripa_leg(atom_id, waypoint, "col")
        self._append_ripa_leg(atom_id, target, "row")

    def _append_ripa_leg(
        self,
        atom_id: int,
        target: tuple[int, int],
        channel: Literal["row", "col"],
    ) -> None:
        current = self.ensemble.atomtraj_by_id(atom_id).final_pos
        if current == tuple(target):
            return
        self.append_step(
            RIPAStep(
                start_time=self.sequence.next_start_time(),
                atom_id=int(atom_id),
                target=tuple(target),
                channel=channel,
            )
        )
