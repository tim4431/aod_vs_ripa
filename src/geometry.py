"""Grid geometry and collision parameters.

The atom plane is an N x N square grid with spacing `d` (microns).
Integer indices (i, j) label sites; physical coordinates are centered on
the array so that the middle of the grid sits at (x, y) = (0, 0).

Two atoms are considered to collide when their Euclidean distance falls
below `rc`. Typically `rc <= d` so atoms in adjacent rows/cols are safe;
atoms in the same row/col can collide depending on motion profiles.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Grid:
    N: int           # grid is N x N
    d: float         # site spacing [um]
    rc: float        # collision radius [um]; rc <= d in normal regimes

    def ij_to_xy(self, i: float, j: float) -> tuple[float, float]:
        """Map (possibly fractional) site index to physical coords, centered."""
        # Center the grid: site (N-1)/2 maps to 0.
        c = (self.N - 1) / 2.0
        return (i - c) * self.d, (j - c) * self.d

    def xy_to_ij(self, x: float, y: float) -> tuple[float, float]:
        c = (self.N - 1) / 2.0
        return x / self.d + c, y / self.d + c

    def in_bounds(self, i: float, j: float, tol: float = 1e-9) -> bool:
        return -tol <= i <= self.N - 1 + tol and -tol <= j <= self.N - 1 + tol
