"""Per-segment motion profiles, with absolute timing.

A `Trajectory` is one *timed* segment of a single atom's life. Resting
positions are float-coordinate trap positions; RIPA callers separately
constrain their hand-off points to integer grid sites. Segments carry their
absolute `start_time` so the global timeline of a Sequence can be reconstructed
by looking at any atom's trajectory list.

The default move profile is a symmetric bang-bang: constant +a_max,
then -a_max, no coast phase. Fully determined by distance and a_max.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Literal, Optional

# Local-time function: t in [0, duration] -> (i, j) in grid units (float).
TrajFn = Callable[[float], tuple[float, float]]
Profile = Callable[[float], float]
Channel = Literal["row", "col", "aod"]  # 'aod' = AOD synchronous; row/col = RIPA EOM
Axis = Literal["x", "y"]  # x == i-coordinate, y == j-coordinate


@dataclass
class Segment:
    start_time: float  # absolute global time [s]
    duration: float  # length of this segment [s]
    start_pos: tuple[float, float]  # resting trap position in grid units
    end_pos: tuple[float, float]  # resting trap position in grid units
    fn: TrajFn  # local-time -> (i, j) float
    channel: Optional[Channel] = None  # which addressing channel drove the move
    profile: Optional[Profile] = None  # local-time -> path fraction, when known

    @property
    def end_time(self) -> float:
        return self.start_time + self.duration

    @property
    def delta(self) -> tuple[float, float]:
        return self.end_pos[0] - self.start_pos[0], self.end_pos[1] - self.start_pos[1]

    @property
    def is_hold(self) -> bool:
        return self.start_pos == self.end_pos

    @property
    def motion_axis(self) -> Axis | None:
        """Physical axis for single-axis motion.

        Code coordinates are `(i, j)`, where `i` is horizontal (`x`) and
        `j` is vertical (`y`). Holds and diagonal/AOD moves return `None`.
        """
        di, dj = self.delta
        if di != 0 and dj == 0:
            return "x"
        if dj != 0 and di == 0:
            return "y"
        return None

    @property
    def direction(self) -> int:
        """Direction along `motion_axis`: -1, 0, or +1."""
        axis = self.motion_axis
        if axis == "x":
            return (self.end_pos[0] > self.start_pos[0]) - (
                self.end_pos[0] < self.start_pos[0]
            )
        if axis == "y":
            return (self.end_pos[1] > self.start_pos[1]) - (
                self.end_pos[1] < self.start_pos[1]
            )
        return 0

    def position_at(self, t_global: float) -> tuple[float, float]:
        """Sample at a global time. Outside the segment, returns the endpoint."""
        if t_global <= self.start_time:
            return (float(self.start_pos[0]), float(self.start_pos[1]))
        if t_global >= self.end_time:
            return (float(self.end_pos[0]), float(self.end_pos[1]))
        return self.fn(t_global - self.start_time)


# --- bang-bang helpers -------------------------------------------------------


def bang_bang_duration(distance: float, accel: float) -> float:
    """Time for a symmetric +a/-a profile (no coast) to traverse `distance`.

    distance = 2 * (1/2 a (T/2)^2)  =>  T = 2 sqrt(distance / accel).

    `accel` is the *actual* acceleration to run at — not necessarily the
    hardware ceiling. Schedulers can slow a move below `a_max` whenever
    they have a reason to (e.g. matching a shared duration across atoms,
    or honoring a heating budget).
    """
    if distance <= 0:
        return 0.0
    return 2.0 * math.sqrt(distance / accel)


# --- general segment wrapper -------------------------------------------------

# Normalized temporal profile: t_local in [0, duration] -> u in [0, 1] giving
# the fraction of the straight-line path from start to end that has been
# covered. profile(0) should be ~0 and profile(duration) should be ~1.

def make_segment(
    start: tuple[float, float],
    end: tuple[float, float],
    start_time: float,
    duration: float,
    profile: Profile,
    *,
    channel: Optional[Channel] = None,
) -> Segment:
    """General segment with an arbitrary temporal profile.

    The atom moves along the straight line from `start` to `end`. The
    *temporal* shape of the move — how the position maps to time — is
    given by `profile(t_local) -> u in [0, 1]`. Acceleration is free
    to vary within the segment (jerk-limited curves, optical-conveyor
    sweeps, multi-stage profiles, etc.) as long as profile(0) ≈ 0 and
    profile(duration) ≈ 1.

    For the common bang-bang case use `make_const_acc_segment`.
    """
    di = end[0] - start[0]
    dj = end[1] - start[1]
    s_pos = (_clean_coord(start[0]), _clean_coord(start[1]))
    e_pos = (_clean_coord(end[0]), _clean_coord(end[1]))

    def fn(t: float, _di=di, _dj=dj, _s=s_pos, _p=profile) -> tuple[float, float]:
        u = _p(t)
        return _s[0] + u * _di, _s[1] + u * _dj

    return Segment(
        start_time=start_time,
        duration=duration,
        start_pos=s_pos,
        end_pos=e_pos,
        fn=fn,
        channel=channel,
        profile=profile,
    )


# --- bang-bang specialization -----------------------------------------------

def _bang_bang_profile(t_local: float, T: float) -> float:
    """Normalized bang-bang profile: 0 at t=0, 1 at t=T, smooth at t=T/2.

    Derived from constant +a then -a with a = 4L/T^2. Independent of L
    because the profile is normalized (returns a fraction).
    """
    if T <= 0:
        return 1.0
    half = T / 2.0
    if t_local <= half:
        return 2.0 * (t_local / T) ** 2
    return 1.0 - 2.0 * ((T - t_local) / T) ** 2


def make_const_acc_segment(
    start: tuple[float, float],
    end: tuple[float, float],
    start_time: float,
    *,
    accel: Optional[float] = None,
    duration: Optional[float] = None,
    channel: Optional[Channel] = None,
) -> Segment:
    """Bang-bang (symmetric +a / -a) segment, the common-case wrapper.

    Provide *exactly one* of:
      * `accel` — the acceleration to run at; duration falls out as
        bang-bang(L, accel). Use this for "as fast as this accel allows".
      * `duration` — the window the move must occupy; implied acceleration
        is 4 * L / duration^2 (used when sharing a window across atoms,
        e.g. AOD lattice ops; shorter moves run at less than a_max).

    What acceleration the hardware allows is the calling Step's concern,
    not the segment's — neither parameter here is intrinsically "max".
    """
    if (accel is None) == (duration is None):
        raise ValueError("provide exactly one of `accel` or `duration`")

    di = end[0] - start[0]
    dj = end[1] - start[1]
    L = math.hypot(di, dj)
    T = bang_bang_duration(L, accel) if duration is None else duration

    return make_segment(
        start, end, start_time, T,
        lambda t, _T=T: _bang_bang_profile(t, _T),
        channel=channel,
    )


def make_hold(
    pos: tuple[float, float],
    start_time: float,
    duration: float,
    *,
    channel: Optional[Channel] = None,
) -> Segment:
    """Stay-put segment. Useful for forced waits."""
    p = (_clean_coord(pos[0]), _clean_coord(pos[1]))
    return Segment(
        start_time=start_time,
        duration=duration,
        start_pos=p,
        end_pos=p,
        fn=lambda t, _p=p: (float(_p[0]), float(_p[1])),
        channel=channel,
    )


def _clean_coord(value: float) -> float:
    """Normalize near-integers while preserving genuine float coordinates."""
    x = float(value)
    rounded = round(x)
    if abs(x - rounded) <= 1e-9:
        return int(rounded)
    return x
