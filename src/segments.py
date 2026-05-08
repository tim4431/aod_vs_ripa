"""Per-segment motion profiles, with absolute timing.

A `Segment` is one *timed* move of a single atom along the straight line
between `start_pos` and `end_pos`. The temporal shape of the move — how
much of the path is covered at any local time — is set by a
`MotionProfile`. That separation mirrors `reference_ramp.RampSequence`:
the segment fields say *what* line to traverse and over what window; the
profile says *how* the position interpolates within the window.

Profiles are normalized on `s = t_local / duration ∈ [0, 1]` and expose
analytic first and second derivatives, so callers can read velocity and
acceleration without finite differences. Bang-bang (symmetric +a / -a)
is one example; quintic minimum-jerk and arbitrary user polynomials are
others. Shorter durations yield higher acceleration; the calling Step
decides whether that is allowed.

There is no row/column assumption baked in here: a segment is defined by
two endpoints in the float-coordinate trap plane, and AOD-style diagonal
motion is just as natural as RIPA axis-aligned hops. The optional
`channel` is metadata (which addressing hardware drove the move) and
does not constrain geometry.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Literal, Optional, Tuple

# 'aod' = AOD synchronous; 'row'/'col' = RIPA EOM channels.
Channel = Literal["row", "col", "aod"]


# --- normalized motion profiles --------------------------------------------


@dataclass(frozen=True)
class MotionProfile:
    """Normalized motion shape `u(s)` on `s ∈ [0, 1]` with analytic derivatives.

    `value(s)` is the fraction in `[0, 1]` of the straight-line path covered
    at normalized time `s = t_local / duration`. `derivative(s)` and
    `second_derivative(s)` are `du/ds` and `d²u/ds²` — the segment scales
    them by `length / duration` and `length / duration²` to recover physical
    velocity and acceleration, so callers do not need finite differences.

    Endpoint requirement: `value(0) ≈ 0` and `value(1) ≈ 1` so the segment
    matches its declared `start_pos` and `end_pos`. `name` is for diagnostics.
    """

    value: Callable[[float], float]
    derivative: Callable[[float], float]
    second_derivative: Callable[[float], float]
    name: str = "profile"


def _bang_bang_value(s: float) -> float:
    if s <= 0.5:
        return 2.0 * s * s
    return 1.0 - 2.0 * (1.0 - s) ** 2


def _bang_bang_derivative(s: float) -> float:
    if s <= 0.5:
        return 4.0 * s
    return 4.0 * (1.0 - s)


def _bang_bang_second_derivative(s: float) -> float:
    return 4.0 if s <= 0.5 else -4.0


BANG_BANG = MotionProfile(
    value=_bang_bang_value,
    derivative=_bang_bang_derivative,
    second_derivative=_bang_bang_second_derivative,
    name="bang_bang",
)


def polynomial_profile(
    coefficients: Tuple[float, ...], name: str = "polynomial"
) -> MotionProfile:
    """Build a `MotionProfile` from polynomial coefficients on `s ∈ [0, 1]`.

    `coefficients = (a_0, a_1, ..., a_n)` defines `u(s) = Σ a_n · s^n`. Pick
    `a_0 = 0` and coefficients summing to `1` so the profile satisfies
    `u(0) = 0` and `u(1) = 1`. `LINEAR` and `QUINTIC_MIN_JERK` below are
    standard members of this family.
    """
    coeffs = tuple(float(c) for c in coefficients)

    def value(s: float, _c=coeffs) -> float:
        result = 0.0
        sn = 1.0
        for a in _c:
            result += a * sn
            sn *= s
        return result

    def derivative(s: float, _c=coeffs) -> float:
        result = 0.0
        for n in range(1, len(_c)):
            result += n * _c[n] * s ** (n - 1)
        return result

    def second_derivative(s: float, _c=coeffs) -> float:
        result = 0.0
        for n in range(2, len(_c)):
            result += n * (n - 1) * _c[n] * s ** (n - 2)
        return result

    return MotionProfile(value, derivative, second_derivative, name)


LINEAR = polynomial_profile((0.0, 1.0), "linear")
QUINTIC_MIN_JERK = polynomial_profile(
    (0.0, 0.0, 0.0, 10.0, -15.0, 6.0), "quintic_min_jerk"
)


# --- bang-bang helpers ------------------------------------------------------


def bang_bang_duration(distance: float, accel: float) -> float:
    """Time for a symmetric +a / -a profile (no coast) to traverse `distance`.

    `distance = 2 · ½ · a · (T/2)²  =>  T = 2 √(distance / a)`. `accel` is
    the *actual* acceleration to run at — not necessarily the hardware
    ceiling. Schedulers can slow a move below `a_max` whenever they have a
    reason to (e.g. matching a shared duration across atoms, honoring a
    heating budget, or pacing a convoy by ∆t).
    """
    if distance <= 0:
        return 0.0
    return 2.0 * math.sqrt(distance / accel)


def min_jerk_duration(distance: float, peak_accel: float) -> float:
    """Time for a quintic minimum-jerk move to traverse `distance` at
    peak acceleration `peak_accel`.

    `QUINTIC_MIN_JERK` has `max|u''(s)| = 10/√3 ≈ 5.7735` on `s ∈ [0, 1]`,
    so peak `|a|` along the path is `(10/√3) · distance / T²`. Solving for
    `T`: `T = √(10·distance / (√3·peak_accel))`. About 1.20× longer than
    `bang_bang_duration` at the same peak acceleration — the price of
    zero velocity and acceleration at both endpoints.
    """
    if distance <= 0:
        return 0.0
    return math.sqrt(10.0 * distance / (math.sqrt(3.0) * peak_accel))


def _profile_peak_accel_coeff(profile: MotionProfile, samples: int = 513) -> float:
    """Numerically locate `max|u''(s)|` on `s ∈ [0, 1]` for any profile.

    Used by `make_smooth_segment(..., accel=...)` to invert the duration
    for an arbitrary user-supplied profile. Closed-form coefficients exist
    for `QUINTIC_MIN_JERK` (`10/√3`) and `BANG_BANG` (`4`), but a generic
    sampler keeps the builder agnostic to the profile family.
    """
    peak = 0.0
    for k in range(samples):
        s = k / (samples - 1)
        v = abs(profile.second_derivative(s))
        if v > peak:
            peak = v
    return peak


# --- the segment -----------------------------------------------------------


@dataclass
class Segment:
    """One timed atom move along the straight line `start_pos -> end_pos`.

    The endpoints are float-coordinate trap positions; AOD segments may
    take any direction, RIPA callers separately constrain their endpoints
    to integer grid sites. `channel` is metadata describing which
    addressing hardware drove the move; it does not constrain geometry.

    `profile` controls the temporal shape: see `MotionProfile`. The default
    is `BANG_BANG`. `make_const_acc_segment` is a convenience for the
    bang-bang case; constructors needing other shapes pass `profile`
    directly.
    """

    start_time: float
    duration: float
    start_pos: tuple[float, float]
    end_pos: tuple[float, float]
    profile: MotionProfile = field(default=BANG_BANG)
    channel: Optional[Channel] = None

    @property
    def end_time(self) -> float:
        return self.start_time + self.duration

    @property
    def length(self) -> float:
        return math.hypot(
            self.end_pos[0] - self.start_pos[0],
            self.end_pos[1] - self.start_pos[1],
        )

    def position_at(self, t_global: float) -> tuple[float, float]:
        """Sample at a global time. Outside the segment, returns the endpoint."""
        u = self.path_fraction_at(t_global)
        return (
            self.start_pos[0] + u * (self.end_pos[0] - self.start_pos[0]),
            self.start_pos[1] + u * (self.end_pos[1] - self.start_pos[1]),
        )

    def velocity_at(self, t_global: float) -> tuple[float, float]:
        """Velocity in grid units / s at global time `t_global`. Zero outside."""
        if (
            self.duration <= 0
            or t_global <= self.start_time
            or t_global >= self.end_time
        ):
            return (0.0, 0.0)
        s = (t_global - self.start_time) / self.duration
        rate = float(self.profile.derivative(s)) / self.duration
        return (
            rate * (self.end_pos[0] - self.start_pos[0]),
            rate * (self.end_pos[1] - self.start_pos[1]),
        )

    def acceleration_at(self, t_global: float) -> tuple[float, float]:
        """Acceleration in grid units / s² at global time `t_global`. Zero outside."""
        if (
            self.duration <= 0
            or t_global <= self.start_time
            or t_global >= self.end_time
        ):
            return (0.0, 0.0)
        s = (t_global - self.start_time) / self.duration
        curvature = float(self.profile.second_derivative(s)) / (self.duration ** 2)
        return (
            curvature * (self.end_pos[0] - self.start_pos[0]),
            curvature * (self.end_pos[1] - self.start_pos[1]),
        )

    def path_fraction_at(self, t_global: float) -> float:
        """Fraction of the path covered at global time `t_global`, clamped to `[0, 1]`."""
        if t_global <= self.start_time:
            return 0.0
        if self.duration <= 0 or t_global >= self.end_time:
            return 1.0
        s = (t_global - self.start_time) / self.duration
        u = float(self.profile.value(s))
        if u < 0.0:
            return 0.0
        if u > 1.0:
            return 1.0
        return u


# --- builders --------------------------------------------------------------


def make_const_acc_segment(
    start: tuple[float, float],
    end: tuple[float, float],
    start_time: float,
    *,
    accel: Optional[float] = None,
    duration: Optional[float] = None,
    channel: Optional[Channel] = None,
) -> Segment:
    """Bang-bang segment, the common-case wrapper.

    Provide *exactly one* of:
      * `accel` — run the move as fast as that acceleration allows; the
        duration falls out as `bang_bang_duration(L, accel)`.
      * `duration` — fix the time window (e.g. matching a shared AOD
        window or pacing a convoy by ∆t); the implied acceleration is
        `4 · L / duration²`, which is below `a_max` for longer windows.

    Whether a given acceleration is hardware-realizable is the calling
    Step's concern, not this builder's; both forms accept any positive
    number.
    """
    if (accel is None) == (duration is None):
        raise ValueError("provide exactly one of `accel` or `duration`")

    L = math.hypot(end[0] - start[0], end[1] - start[1])
    T = bang_bang_duration(L, accel) if duration is None else float(duration)
    return Segment(
        start_time=float(start_time),
        duration=T,
        start_pos=(_clean_coord(start[0]), _clean_coord(start[1])),
        end_pos=(_clean_coord(end[0]), _clean_coord(end[1])),
        profile=BANG_BANG,
        channel=channel,
    )


def make_smooth_segment(
    start: tuple[float, float],
    end: tuple[float, float],
    start_time: float,
    *,
    duration: Optional[float] = None,
    accel: Optional[float] = None,
    profile: MotionProfile = QUINTIC_MIN_JERK,
    channel: Optional[Channel] = None,
) -> Segment:
    """Waypoint-style smooth segment: `x(start_time) = start` and
    `x(start_time + duration) = end`, interpolated by `profile`.

    Mirrors `reference_ramp.RampSequence` semantics — caller specifies
    the endpoints and timing, a named smooth `profile` carries the shape.
    The default `QUINTIC_MIN_JERK` has zero velocity AND zero acceleration
    at both endpoints so successive segments stitch together cleanly,
    unlike `BANG_BANG` which has a velocity kink at every join.

    Provide *exactly one* of:
      * `duration` — fix the time window directly. The implied peak
        acceleration is `max|u''(s)| · L / duration²`.
      * `accel` — run at this peak acceleration. Duration falls out as
        `√(max|u''(s)| · L / accel)`. For `QUINTIC_MIN_JERK` this reduces
        to `min_jerk_duration(L, accel)`.
    """
    if (accel is None) == (duration is None):
        raise ValueError("provide exactly one of `accel` or `duration`")

    L = math.hypot(end[0] - start[0], end[1] - start[1])
    if duration is None:
        peak = _profile_peak_accel_coeff(profile)
        if peak <= 0.0 or L <= 0.0:
            T = 0.0
        else:
            T = math.sqrt(peak * L / accel)
    else:
        T = float(duration)

    return Segment(
        start_time=float(start_time),
        duration=T,
        start_pos=(_clean_coord(start[0]), _clean_coord(start[1])),
        end_pos=(_clean_coord(end[0]), _clean_coord(end[1])),
        profile=profile,
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
        start_time=float(start_time),
        duration=float(duration),
        start_pos=p,
        end_pos=p,
        profile=BANG_BANG,
        channel=channel,
    )


def _clean_coord(value: float) -> float:
    """Normalize near-integers while preserving genuine float coordinates."""
    x = float(value)
    rounded = round(x)
    if abs(x - rounded) <= 1e-9:
        return int(rounded)
    return x
