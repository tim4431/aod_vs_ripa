"""Grid + AtomConfig.

`AtomConfig` is a *resting-state snapshot*: it records, for each atom,
its integer (i, j) site **and** its `atom_id`. Atoms in motion are not
described here — see `atom_trajectory.AtomTrajectory` for that.

The `atom_id` is an internal index for tracking and visualization; it
does not by itself mark atoms as physically distinguishable. Whether
atoms are distinguishable is a property of the routing problem.

Static traps (e.g. SLM hand-off sites) are a hardware concern, not part
of the atom configuration; pass them separately to the visualizer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

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
    # atom_ids[k] = identifier of atom k. Shape (M,). Defaults to arange(M).
    atom_ids: Optional[np.ndarray] = None

    def __post_init__(self):
        arr = np.asarray(self.positions).reshape(-1, 2)
        # Enforce integer sites — this snapshot is a resting state.
        rounded = np.round(arr).astype(int)
        if not np.allclose(arr, rounded, atol=1e-9):
            raise ValueError("AtomConfig positions must be integer grid sites")
        self.positions = rounded
        # Reject duplicate sites: a site can hold at most one atom.
        seen: set[tuple[int, int]] = set()
        for i, j in self.positions:
            if (int(i), int(j)) in seen:
                raise ValueError(f"duplicate atom at site {(int(i), int(j))}")
            seen.add((int(i), int(j)))
        # Default atom_ids to 0..M-1 if not provided.
        M = len(self.positions)
        if self.atom_ids is None:
            self.atom_ids = np.arange(M, dtype=int)
        else:
            ids = np.asarray(self.atom_ids, dtype=int).reshape(-1)
            if len(ids) != M:
                raise ValueError(f"atom_ids length {len(ids)} != n_atoms {M}")
            if len(np.unique(ids)) != M:
                raise ValueError("atom_ids must be unique")
            self.atom_ids = ids

    @property
    def n_atoms(self) -> int:
        return len(self.positions)

    def xy(self) -> np.ndarray:
        """Physical (x, y) coordinates of all atoms, shape (M, 2)."""
        c = (self.grid.N - 1) / 2.0
        return (self.positions.astype(float) - c) * self.grid.d

    def copy(self) -> "AtomConfig":
        return AtomConfig(
            grid=self.grid,
            positions=self.positions.copy(),
            atom_ids=self.atom_ids.copy(),
        )

    def occupied_sites(self) -> set[tuple[int, int]]:
        return {(int(i), int(j)) for i, j in self.positions}

    def occupancy(self) -> dict[tuple[int, int], int]:
        """Map site -> atom_id. The two key snapshot views are by site
        (this method) and by id (`site_of_atom`)."""
        return {(int(i), int(j)): int(aid)
                for (i, j), aid in zip(self.positions, self.atom_ids)}

    def site_of_atom(self) -> dict[int, tuple[int, int]]:
        """Inverse of `occupancy`: atom_id -> site."""
        return {int(aid): (int(i), int(j))
                for (i, j), aid in zip(self.positions, self.atom_ids)}
