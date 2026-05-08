"""Smooth time-dependent ramp sequences for moving traps.

`RampSequence` is the time-vs-(center, depth) waypoint table that any
time-dependent trap can consume. Between waypoints, position and depth are
interpolated by a `PolynomialConnector` chosen at construction time. The
default is `LINEAR` so a ramp built without specifying a profile reproduces
plain piecewise-linear interpolation. Callers that want smooth motion pick a
named profile (e.g. `QUINTIC_MIN_JERK`) or build a custom one with
`arb_fifth_poly(beta)`.

Each connector exposes both the value and the derivative, so traps can ask
the ramp for `velocity_at(t)` (used by `AstigmaticAODTrap` for the
velocity-coupled focal shift).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence, Tuple

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class PolynomialConnector:
    """Smooth interpolation kernel between two waypoints.

    Given waypoints `(x0, y0)` and `(x1, y1)`, the connector evaluates
    `y(x) = y0 + (y1 - y0) * sum_n a_n u^n` with `u = (x - x0) / (x1 - x0)`.
    The polynomial coefficients should satisfy `a_0 = 0`, `sum a_n = 1` so the
    connector matches both endpoint values;
    smoothness at the joins comes from extra zero leading coefficients.
    """

    coefficients: Tuple[float, ...]
    name: str = "polynomial"

    def __post_init__(self) -> None:
        """Validate that at least one polynomial coefficient was provided."""

        if len(self.coefficients) < 1:
            raise ValueError("coefficients must have at least one entry.")

    def value(self, u: ArrayLike) -> NDArray[np.float64]:
        """Normalized shape function on `u in [0, 1]`."""

        u_array = np.asarray(u, dtype=float)
        out = np.zeros_like(u_array)
        for n, a in enumerate(self.coefficients):
            if a == 0.0:
                continue
            out = out + a * u_array**n
        return out

    def derivative(self, u: ArrayLike) -> NDArray[np.float64]:
        """Derivative of the shape function with respect to `u`."""

        u_array = np.asarray(u, dtype=float)
        out = np.zeros_like(u_array)
        for n, a in enumerate(self.coefficients):
            if n == 0 or a == 0.0:
                continue
            out = out + a * n * u_array ** (n - 1)
        return out


LINEAR = PolynomialConnector((0.0, 1.0), name="linear")
CONST_JERK = PolynomialConnector((0.0, 0.0, 3.0, -2.0), name="const_jerk")
QUINTIC_MIN_JERK = PolynomialConnector(
    (0.0, 0.0, 0.0, 10.0, -15.0, 6.0), name="quintic_minimum_jerk"
)


def arb_fifth_poly(beta: float) -> PolynomialConnector:
    """Quintic family from `aod_slm_movement_v2.trajectories`.

    `beta = 1` reproduces `QUINTIC_MIN_JERK`; the v2 work uses `beta = 1.5625`.
    """

    return PolynomialConnector(
        (
            0.0,
            0.0,
            15.0 - 8.0 * beta,
            -50.0 + 32.0 * beta,
            60.0 - 40.0 * beta,
            -24.0 + 16.0 * beta,
        ),
        name=f"arb_fifth_poly_beta_{beta:g}",
    )


def const_jerk() -> PolynomialConnector:
    """Cubic with constant jerk (v2's `const_jerk`)."""

    return PolynomialConnector((0.0, 0.0, 3.0, -2.0), name="const_jerk")


@dataclass(frozen=True)
class RampSequence:
    """Time-indexed center-and-depth waypoint table for one trap.

    The table separates **what** values to hit (`centers_m`, `depths_uK` at
    each entry of `times_s`) from **how** to interpolate between them
    (`position_profile`, `depth_profile`). The two are orthogonal: the
    waypoint arrays carry the absolute target values; the profiles are
    normalized shape functions on `u ∈ [0, 1]` that the same kernel applies
    to every segment. You need both — the profile alone has no notion of
    absolute depth, and two waypoints alone don't say whether the transition
    is linear or smooth.

    Fields:
        times_s: 1D, strictly increasing waypoint times. At least two points.
        centers_m: shape `(N, 3)` — trap-center position at each waypoint.
        depths_uK: shape `(N,)` — non-negative trap depth at each waypoint.
        position_profile: per-segment kernel for `centers_m` interpolation.
        depth_profile: per-segment kernel for `depths_uK` interpolation.

    Both profiles default to `LINEAR`, so a plain waypoint list behaves like
    `np.interp`. Switch to `QUINTIC_MIN_JERK` (or another connector) for
    smooth motion — note that `AstigmaticAODTrap` reads `velocity_at(t)` from
    the position profile, so a `LINEAR` position profile gives piecewise-
    constant velocity and step changes in the focal shift at every waypoint.

    Outside `[times_s[0], times_s[-1]]`, `at(t)` clamps to the endpoint
    values and `velocity_at(t)` / `depth_rate_at(t)` return zero.
    """

    times_s: ArrayLike
    centers_m: ArrayLike
    depths_uK: ArrayLike
    position_profile: PolynomialConnector = field(default=LINEAR)
    depth_profile: PolynomialConnector = field(default=LINEAR)

    def __post_init__(self) -> None:
        """Coerce inputs to float arrays and check shape/monotonicity invariants.

        Raises `ValueError` if `times_s` is not strictly increasing, if
        `centers_m` / `depths_uK` don't match its length, or if any depth is
        negative. After validation, the original fields are replaced with
        canonical `np.float64` arrays so downstream code can index without
        re-coercing.
        """

        times = np.asarray(self.times_s, dtype=float)
        centers = np.asarray(self.centers_m, dtype=float)
        depths = np.asarray(self.depths_uK, dtype=float)

        if times.ndim != 1 or len(times) < 2:
            raise ValueError("times_s must be a 1D array with at least two points.")
        if np.any(np.diff(times) <= 0.0):
            raise ValueError("times_s must be strictly increasing.")
        if centers.shape != (len(times), 3):
            raise ValueError("centers_m must have shape (len(times_s), 3).")
        if depths.shape != (len(times),):
            raise ValueError("depths_uK must have shape (len(times_s),).")
        if np.any(depths < 0.0):
            raise ValueError("depths_uK must be non-negative.")

        object.__setattr__(self, "times_s", times)
        object.__setattr__(self, "centers_m", centers)
        object.__setattr__(self, "depths_uK", depths)

    @property
    def start_time_s(self) -> float:
        """Time of the first waypoint (lower bound of the ramp's active range)."""

        return float(self.times_s[0])

    @property
    def end_time_s(self) -> float:
        """Time of the last waypoint (upper bound of the ramp's active range)."""

        return float(self.times_s[-1])

    def at(self, time_s: float) -> tuple[NDArray[np.float64], float]:
        """Return `(center, depth)` interpolated at `time_s`."""

        return self.center_at(time_s), self.depth_at(time_s)

    def center_at(self, time_s: float) -> NDArray[np.float64]:
        """Interpolated trap-center 3-vector (meters) at `time_s`.

        Inside the table, the bracketing waypoints `c0`, `c1` are blended by
        `position_profile.value(u)` where `u` is the segment fraction.
        Outside `[start_time_s, end_time_s]`, returns the nearest endpoint
        center.
        """

        index, u = self._segment_index_and_fraction(time_s)
        if index is None:
            return self._endpoint_center(time_s)
        shape = float(self.position_profile.value(u))
        c0 = self.centers_m[index]
        c1 = self.centers_m[index + 1]
        return c0 + shape * (c1 - c0)

    def depth_at(self, time_s: float) -> float:
        """Interpolated trap depth (microkelvin) at `time_s`.

        Inside the table, blends the bracketing depths via
        `depth_profile.value(u)`. Outside the table, returns the nearest
        endpoint depth.
        """

        index, u = self._segment_index_and_fraction(time_s)
        if index is None:
            return self._endpoint_depth(time_s)
        shape = float(self.depth_profile.value(u))
        d0 = float(self.depths_uK[index])
        d1 = float(self.depths_uK[index + 1])
        return d0 + shape * (d1 - d0)

    def velocity_at(self, time_s: float) -> NDArray[np.float64]:
        """Return `dcenter/dt` at `time_s`. Outside the table this is zero."""

        index, u = self._segment_index_and_fraction(time_s)
        if index is None:
            return np.zeros(3, dtype=float)
        dt = float(self.times_s[index + 1] - self.times_s[index])
        slope = float(self.position_profile.derivative(u)) / dt
        c0 = self.centers_m[index]
        c1 = self.centers_m[index + 1]
        return slope * (c1 - c0)

    def depth_rate_at(self, time_s: float) -> float:
        """Return `ddepth/dt` at `time_s` in uK/s."""

        index, u = self._segment_index_and_fraction(time_s)
        if index is None:
            return 0.0
        dt = float(self.times_s[index + 1] - self.times_s[index])
        slope = float(self.depth_profile.derivative(u)) / dt
        d0 = float(self.depths_uK[index])
        d1 = float(self.depths_uK[index + 1])
        return slope * (d1 - d0)

    def _endpoint_center(self, time_s: float) -> NDArray[np.float64]:
        """Clamped center for times outside the waypoint range.

        Returns the first waypoint's center for `time_s <= start_time_s`,
        otherwise the last waypoint's center. Always returns a fresh array.
        """

        if time_s <= self.start_time_s:
            return np.array(self.centers_m[0], dtype=float)
        return np.array(self.centers_m[-1], dtype=float)

    def _endpoint_depth(self, time_s: float) -> float:
        """Clamped depth for times outside the waypoint range.

        Returns the first waypoint's depth for `time_s <= start_time_s`,
        otherwise the last waypoint's depth.
        """

        if time_s <= self.start_time_s:
            return float(self.depths_uK[0])
        return float(self.depths_uK[-1])

    def _segment_index_and_fraction(self, time_s: float) -> tuple[int | None, float]:
        """Locate the segment containing `time_s` and its normalized fraction.

        Returns `(index, u)` where `index` is the lower waypoint of the
        bracketing segment and `u = (t - times_s[index]) / (times_s[index+1]
        - times_s[index]) ∈ [0, 1]`. Returns `(None, 0.0)` when `time_s` is
        outside `[start_time_s, end_time_s]`, signalling the caller to use
        the endpoint-clamp helpers instead of interpolating.
        """

        t = float(time_s)
        if t <= self.start_time_s or t >= self.end_time_s:
            return None, 0.0
        index = int(np.searchsorted(self.times_s, t, side="right") - 1)
        index = max(0, min(index, len(self.times_s) - 2))
        t0 = float(self.times_s[index])
        t1 = float(self.times_s[index + 1])
        u = (t - t0) / (t1 - t0)
        return index, u
