"""RIPA position <-> EOM frequency tone mapping.

The RIPA-SLM is a 2D spectrometer: each EOM tone `nu` at the input maps
to a unique `(x, y)` spot at the output. With FSR1 along the fast axis
and FSR2 = FSR1 / N along the slow axis, the row and column channels
(rotated 90 deg from each other) read:

    nu_row(i, j) = (j + i / N) * FSR2     (mod FSR1)   # moving along x
    nu_col(i, j) = (i + j / N) * FSR2     (mod FSR1)   # moving along y

The channel identity (row EOM or col EOM) tells the hardware which rotated
spectrometer axis is being driven.

`nu_row` and `nu_col` return positive modulo frequencies in [0, FSR1).
The visualization displays these values in normalized FSR1 units,
`nu / FSR1`, on [0, 1).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RIPASpec:
    N: int
    FSR1: float
    FSR2: float | None = None

    def fsr2(self) -> float:
        return self.FSR2 if self.FSR2 is not None else self.FSR1 / self.N


def _wrap(nu: float, FSR: float) -> float:
    """Wrap nu into [0, FSR)."""
    return nu % FSR


def nu_row(i: float, j: float, spec: RIPASpec) -> float:
    """Row-channel tone at site (i, j), used while moving along x."""
    return _wrap((j + i / spec.N) * spec.fsr2(), spec.FSR1)


def nu_col(i: float, j: float, spec: RIPASpec) -> float:
    """Col-channel tone at site (i, j), used while moving along y."""
    return _wrap((i + j / spec.N) * spec.fsr2(), spec.FSR1)
