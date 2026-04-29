"""Per-atom 1D / 2D motion profiles.

A `Trajectory` is just a callable t -> (i, j) over a finite duration. The
default profile is a symmetric trapezoid in acceleration: constant +a_max,
zero, then -a_max. This is the standard "bang-coast-bang" used in atom-
array experiments and is fully determined by the move distance and
`a_max` (no jerk shaping).

Distances and times use grid units for `i, j` and SI for `t`. `a_max` is
expressed in *site-spacings per second^2* so users can plug in physical
acceleration via `Grid.d`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np


# A trajectory: t (seconds, 0 <= t <= duration) -> (i, j) in grid units.
TrajFn = Callable[[float], tuple[float, float]]


@dataclass
class Trajectory:
    start: tuple[float, float]   # (i0, j0) at t=0
    end: tuple[float, float]     # (i1, j1) at t=duration
    duration: float              # [s]
    fn: TrajFn                   # t -> (i, j)

    def __call__(self, t: float) -> tuple[float, float]:
        return self.fn(max(0.0, min(self.duration, t)))

    def sample(self, ts: np.ndarray) -> np.ndarray:
        """Vectorized sample: ts shape (T,) -> positions shape (T, 2)."""
        out = np.empty((len(ts), 2))
        for k, t in enumerate(ts):
            out[k] = self.fn(max(0.0, min(self.duration, float(t))))
        return out


def _bang_bang_duration(distance: float, a_max: float) -> float:
    """Time to traverse `distance` (>=0) under symmetric +a / -a profile.

    No coast phase: accelerate for T/2, decelerate for T/2.
    distance = 0.5 * a_max * (T/2)^2 * 2  =>  T = 2 * sqrt(distance / a_max).
    """
    if distance <= 0:
        return 0.0
    return 2.0 * math.sqrt(distance / a_max)


def _bang_bang_pos(t: float, T: float, distance: float) -> float:
    """Position at time t in [0, T] for a symmetric bang-bang move of length `distance`."""
    if T <= 0:
        return distance
    half = T / 2.0
    a = 4.0 * distance / (T * T)  # = a_max actually used
    if t <= half:
        return 0.5 * a * t * t
    # decel phase: mirror around t = T/2
    tt = t - half
    v_peak = a * half
    return distance / 2.0 + v_peak * tt - 0.5 * a * tt * tt


def straight_move(start: tuple[float, float],
                  end: tuple[float, float],
                  a_max: float,
                  start_time: float = 0.0) -> Trajectory:
    """Straight-line bang-bang move from `start` to `end` in (i, j) space.

    `a_max` is in *grid units / s^2*. Total length is the Euclidean norm in
    grid units; we move along the unit vector toward the target.
    """
    di = end[0] - start[0]
    dj = end[1] - start[1]
    L = math.hypot(di, dj)
    T = _bang_bang_duration(L, a_max)
    if L == 0:
        ux, uy = 0.0, 0.0
    else:
        ux, uy = di / L, dj / L

    def fn(t: float) -> tuple[float, float]:
        s = _bang_bang_pos(t, T, L)
        return start[0] + ux * s, start[1] + uy * s

    return Trajectory(start=start, end=end, duration=T, fn=fn)


def hold(pos: tuple[float, float], duration: float) -> Trajectory:
    """Stay-put trajectory. Useful for atoms that aren't moved this step."""
    return Trajectory(start=pos, end=pos, duration=duration,
                      fn=lambda t, p=pos: p)
