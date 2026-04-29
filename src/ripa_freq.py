"""RIPA position <-> EOM frequency tone mapping.

The RIPA-SLM is a 2D spectrometer: each EOM tone `nu` at the input maps
to a unique `(x, y)` spot at the output. With FSR1 along the fast axis
and FSR2 = FSR1 / N along the slow axis, the row and column channels
(rotated 90 deg from each other) read:

    nu_row(i, j) = (j + i / N) * FSR2     (mod FSR1)   # used while moving along x
    nu_col(i, j) = (i / N + j) * FSR2     (mod FSR1)   # used while moving along y

Note both formulas evaluate to the same number — the *channel identity*
(which EOM, row or col) tells the hardware which axis the spot moves
along; the formulas just give the tone you must drive on that channel.

We work in units of FSR1, so frequencies are returned as a number in
[-FSR1/2, +FSR1/2) after wrapping. That matches the visualization
convention where the x-axis is "Δν (GHz, mod FSR1)".
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RIPASpec:
    N: int
    FSR1: float           # Hz (or GHz — units propagate)
    FSR2: float | None = None    # default FSR1 / N

    def fsr2(self) -> float:
        return self.FSR2 if self.FSR2 is not None else self.FSR1 / self.N


def _wrap(nu: float, FSR: float) -> float:
    """Wrap nu into [-FSR/2, +FSR/2)."""
    return ((nu + FSR / 2) % FSR) - FSR / 2


def nu_row(i: float, j: float, spec: RIPASpec) -> float:
    """Row-channel tone (used while moving in x) at site (i, j)."""
    return _wrap((j + i / spec.N) * spec.fsr2(), spec.FSR1)


def nu_col(i: float, j: float, spec: RIPASpec) -> float:
    """Col-channel tone (used while moving in y) at site (i, j)."""
    return _wrap((i / spec.N + j) * spec.fsr2(), spec.FSR1)
