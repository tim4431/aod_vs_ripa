"""Grid + AtomConfig.

`AtomConfig` is a *resting-state* snapshot: positions are integer (i, j)
sites on the grid. Atoms in motion are *not* described here — see
`atom_trajectory.AtomTrajectory` for that. AtomConfig is used only as
input to schedulers and to describe the final resting state.

Static traps (e.g. SLM hand-off sites) are a hardware concern, not part
of the atom configuration; pass them separately to the visualizer.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Grid:
    N: int       # grid is N x N
    d: float     # site spacing [um]
    rc: float    # collision radius [um]; rc <= d in normal regimes

    def ij_to_xy(self, i: float, j: float) -> tuple[float, float]:
        """Map (possibly fractional) site index to physical coords, centered."""
        c = (self.N - 1) / 2.0
        return (i - c) * self.d, (j - c) * self.d


@dataclass
class AtomConfig:
    grid: Grid
    # positions[k] = (i, j) of atom k, integer site. Shape (M, 2).
    positions: np.ndarray

    def __post_init__(self):
        arr = np.asarray(self.positions).reshape(-1, 2)
        # Enforce integer sites — this snapshot is a resting state.
        rounded = np.round(arr).astype(int)
        if not np.allclose(arr, rounded, atol=1e-9):
            raise ValueError("AtomConfig positions must be integer grid sites")
        self.positions = rounded
        # Reject duplicates: a site can hold at most one atom.
        seen: set[tuple[int, int]] = set()
        for i, j in self.positions:
            if (int(i), int(j)) in seen:
                raise ValueError(f"duplicate atom at site {(int(i), int(j))}")
            seen.add((int(i), int(j)))

    @property
    def n_atoms(self) -> int:
        return len(self.positions)

    def xy(self) -> np.ndarray:
        """Physical (x, y) coordinates of all atoms, shape (M, 2)."""
        c = (self.grid.N - 1) / 2.0
        return (self.positions.astype(float) - c) * self.grid.d

    def copy(self) -> "AtomConfig":
        return AtomConfig(grid=self.grid, positions=self.positions.copy())

    def occupied_sites(self) -> set[tuple[int, int]]:
        return {(int(i), int(j)) for i, j in self.positions}
