"""Per-segment motion profiles, with absolute timing.

A `Trajectory` is one *timed* segment of a single atom's life. Atoms
rest at integer sites between segments; each segment goes integer ->
integer (start_pos -> end_pos), with float positions during 0 < t_local
< duration. Segments carry their absolute `start_time` so the global
timeline of a Sequence can be reconstructed by looking at any atom's
trajectory list.

The default move profile is a symmetric bang-bang: constant +a_max,
then -a_max, no coast phase. Fully determined by distance and a_max.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Literal, Optional

# Local-time function: t in [0, duration] -> (i, j) in grid units (float).
TrajFn = Callable[[float], tuple[float, float]]
Channel = Literal["row", "col", "aod"]  # 'aod' = AOD synchronous; row/col = RIPA EOM


@dataclass
class Segment:
    start_time: float  # absolute global time [s]
    duration: float  # length of this segment [s]
    start_pos: tuple[int, int]  # integer site at t = start_time
    end_pos: tuple[int, int]  # integer site at t = end_time
    fn: TrajFn  # local-time -> (i, j) float
    channel: Optional[Channel] = None  # which addressing channel drove the move

    @property
    def end_time(self) -> float:
        return self.start_time + self.duration

    def position_at(self, t_global: float) -> tuple[float, float]:
        """Sample at a global time. Outside the segment, returns the endpoint."""
        if t_global <= self.start_time:
            return (float(self.start_pos[0]), float(self.start_pos[1]))
        if t_global >= self.end_time:
            return (float(self.end_pos[0]), float(self.end_pos[1]))
        return self.fn(t_global - self.start_time)


# --- bang-bang helpers -------------------------------------------------------


def bang_bang_duration(distance: float, a_max: float) -> float:
    """Time for symmetric +a/-a profile (no coast) to traverse `distance`.

    distance = 2 * (1/2 a (T/2)^2)  =>  T = 2 sqrt(distance / a_max).
    """
    if distance <= 0:
        return 0.0
    return 2.0 * math.sqrt(distance / a_max)


def _bang_bang_pos(t_local: float, T: float, distance: float) -> float:
    """Scalar position along the move at local time t_local in [0, T]."""
    if T <= 0:
        return distance
    half = T / 2.0
    a = 4.0 * distance / (T * T)  # the actual accel used
    if t_local <= half:
        return 0.5 * a * t_local * t_local
    tt = t_local - half
    v_peak = a * half
    return distance / 2.0 + v_peak * tt - 0.5 * a * tt * tt


def make_segment(
    start: tuple[int, int],
    end: tuple[int, int],
    start_time: float,
    a_max: float,
    *,
    duration: Optional[float] = None,
    channel: Optional[Channel] = None,
) -> Segment:
    """Bang-bang segment from `start` to `end` at given absolute start_time.

    If `duration` is None, use the natural bang-bang time for the move.
    Pass `duration` explicitly to share a window across multiple atoms
    (e.g. an AOD step where every atom's segment shares the longest
    move's duration; shorter moves then run with reduced acceleration).
    """
    di = end[0] - start[0]
    dj = end[1] - start[1]
    L = math.hypot(di, dj)
    T = bang_bang_duration(L, a_max) if duration is None else duration
    ux, uy = (0.0, 0.0) if L == 0 else (di / L, dj / L)

    def fn(t: float, _T=T, _L=L, _s=start, _u=(ux, uy)) -> tuple[float, float]:
        s = _bang_bang_pos(t, _T, _L)
        return _s[0] + _u[0] * s, _s[1] + _u[1] * s

    return Segment(
        start_time=start_time,
        duration=T,
        start_pos=tuple(map(int, start)),
        end_pos=tuple(map(int, end)),
        fn=fn,
        channel=channel,
    )


def make_hold(
    pos: tuple[int, int],
    start_time: float,
    duration: float,
    *,
    channel: Optional[Channel] = None,
) -> Segment:
    """Stay-put segment. Useful for forced waits."""
    p = (int(pos[0]), int(pos[1]))
    return Segment(
        start_time=start_time,
        duration=duration,
        start_pos=p,
        end_pos=p,
        fn=lambda t, _p=p: (float(_p[0]), float(_p[1])),
        channel=channel,
    )
