"""Atom configuration on the grid.

Stores the (i, j) location of each atom plus optional labels.
- Case 1 (set->set): labels left None; atoms are interchangeable.
- Case 2 (pairwise): labels carry distinguishing info (e.g. qubit id).

Positions are stored as floats so an `AtomConfig` can also describe
mid-flight states sampled along trajectories.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Hashable, Optional

import numpy as np


@dataclass(frozen=True)
class Grid:
    N: int  # grid is N x N
    d: float  # site spacing [um]
    rc: float  # collision radius [um]; rc <= d in normal regimes

    def ij_to_xy(self, i: float, j: float) -> tuple[float, float]:
        """Map (possibly fractional) site index to physical coords, centered."""
        # Center the grid: site (N-1)/2 maps to 0.
        c = (self.N - 1) / 2.0
        return (i - c) * self.d, (j - c) * self.d


@dataclass
class AtomConfig:
    grid: Grid
    # positions[k] = (i, j) of atom k; shape (M, 2). Float to allow mid-move samples.
    positions: np.ndarray
    # labels[k] = identifier of atom k, or None for the whole array (Case 1).
    labels: Optional[list[Hashable]] = None
    # static_traps[s] = (i, j) of an SLM-held trap. Used during AOD hand-offs.
    static_traps: list[tuple[float, float]] = field(default_factory=list)

    def __post_init__(self):
        self.positions = np.asarray(self.positions, dtype=float).reshape(-1, 2)
        if self.labels is not None and len(self.labels) != len(self.positions):
            raise ValueError("labels length must match number of atoms")

    @property
    def n_atoms(self) -> int:
        return len(self.positions)

    @property
    def labeled(self) -> bool:
        return self.labels is not None

    def xy(self) -> np.ndarray:
        """Physical (x, y) coordinates of all atoms, shape (M, 2)."""
        c = (self.grid.N - 1) / 2.0
        return (self.positions - c) * self.grid.d

    def copy(self) -> "AtomConfig":
        return AtomConfig(
            grid=self.grid,
            positions=self.positions.copy(),
            labels=None if self.labels is None else list(self.labels),
            static_traps=list(self.static_traps),
        )

    def occupied_sites(self) -> set[tuple[int, int]]:
        """Sites currently occupied (rounded to nearest integer)."""
        return {(int(round(i)), int(round(j))) for i, j in self.positions}
