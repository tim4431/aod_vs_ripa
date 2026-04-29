"""MovementStep: one atomic chunk of motion executed by the hardware.

Two flavors:

* `RIPAStep` — RIPA-SLM addresses each atom independently, so the step is
  *natively* a list of per-atom trajectories. Atoms move along a single
  row OR column at a time (the row/col channel of the EOM); diagonal
  moves are decomposed by the scheduler into multiple steps with
  hand-offs at integer (i, j).

* `AODStep` — Crossed AODs can only stretch / translate whole rows and
  columns without crossing. The native description is therefore two
  monotone permutations: `row_map[r] = r'` and `col_map[c] = c'`. We
  expand that into per-atom trajectories so downstream code (validator,
  visualizer, timing) can treat both backends uniformly.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from .atoms import AtomConfig
from .trajectories import Trajectory, hold, straight_move


# --- RIPA -------------------------------------------------------------------

@dataclass
class RIPAStep:
    """One RIPA move: a set of per-atom trajectories.

    `channel` records which EOM channel carries each atom ('row' moves
    along x, 'col' moves along y). All trajectories in one step start at
    t=0 (relative); absolute timing comes from the parent Sequence.
    """
    trajectories: list[Trajectory]                # one per moved atom
    atom_indices: list[int]                       # which atom in AtomConfig
    channel: list[Literal["row", "col"]]          # 'row' or 'col' per traj

    @property
    def duration(self) -> float:
        return max((tr.duration for tr in self.trajectories), default=0.0)

    def apply(self, cfg: AtomConfig) -> AtomConfig:
        """Return a new AtomConfig with moved atoms at their endpoints."""
        new = cfg.copy()
        for k, tr in zip(self.atom_indices, self.trajectories):
            new.positions[k] = tr.end
        return new


# --- AOD --------------------------------------------------------------------

@dataclass
class AODStep:
    """One AOD move described as monotone row/col remaps.

    `row_map`/`col_map` are *partial* maps: only entries for rows/cols
    that actually move need appear, but the maps must remain *strictly
    monotone* (no crossings) when combined with the identity on the
    untouched indices. Atoms sitting on a moved row inherit the row's
    new index (likewise for columns).
    """
    row_map: dict[int, float] = field(default_factory=dict)
    col_map: dict[int, float] = field(default_factory=dict)
    a_max: float = 1.0  # grid-units / s^2; bang-bang acceleration cap

    def _validate_monotone(self, m: dict[int, float], N: int) -> None:
        # Effective map after filling in identity for untouched indices.
        full = [(r, m.get(r, float(r))) for r in range(N)]
        for (r1, v1), (r2, v2) in zip(full, full[1:]):
            if v1 >= v2:
                raise ValueError(f"AOD map not strictly monotone at {r1}->{v1}, {r2}->{v2}")

    def to_trajectories(self, cfg: AtomConfig) -> tuple[list[Trajectory], list[int]]:
        """Expand the row/col remap into per-atom straight-line moves."""
        self._validate_monotone(self.row_map, cfg.grid.N)
        self._validate_monotone(self.col_map, cfg.grid.N)

        trajs: list[Trajectory] = []
        idxs: list[int] = []
        for k, (i, j) in enumerate(cfg.positions):
            ri, rj = int(round(i)), int(round(j))
            new_i = self.row_map.get(ri, float(i))
            new_j = self.col_map.get(rj, float(j))
            if (new_i, new_j) == (i, j):
                continue
            trajs.append(straight_move((float(i), float(j)),
                                       (float(new_i), float(new_j)),
                                       self.a_max))
            idxs.append(k)
        return trajs, idxs

    def to_ripa_step(self, cfg: AtomConfig) -> RIPAStep:
        """Convenience: lift to a RIPAStep so a uniform validator can run.

        Channel is set to 'row' if the move is purely horizontal, 'col'
        if purely vertical, else 'row' as a default (AOD diagonals don't
        actually use EOM channels, but we keep the field for uniformity).
        """
        trajs, idxs = self.to_trajectories(cfg)
        chans: list[Literal["row", "col"]] = []
        for tr in trajs:
            di = tr.end[0] - tr.start[0]
            dj = tr.end[1] - tr.start[1]
            chans.append("col" if abs(dj) > abs(di) else "row")
        return RIPAStep(trajs, idxs, chans)

    @property
    def duration(self) -> float:
        # Caller can compute via to_trajectories(cfg); kept as 0.0 placeholder.
        return 0.0

    def apply(self, cfg: AtomConfig) -> AtomConfig:
        """Apply the row/col remap directly (no need to integrate trajectories)."""
        new = cfg.copy()
        for k, (i, j) in enumerate(cfg.positions):
            ri, rj = int(round(i)), int(round(j))
            new.positions[k, 0] = self.row_map.get(ri, i)
            new.positions[k, 1] = self.col_map.get(rj, j)
        return new


MovementStep = RIPAStep | AODStep
