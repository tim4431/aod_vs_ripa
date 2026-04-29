"""Collision validator over the full ensemble timeline.

Strategy: sample every atom's `position_at(t)` on a fine time grid, only
within intervals when at least one atom is moving, plus the segment
boundaries themselves (so endpoint conflicts are never missed). At each
sample we do an O(M^2) pairwise distance check against `grid.rc`.

For typical M ~ 10^2 atoms and a few hundred dt-samples per move this
is fast enough; tighten/loosen `dt` based on max speed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .atom_trajectory import AtomEnsemble


@dataclass
class CollisionReport:
    ok: bool
    worst_pair: tuple[int, int, float] | None = None  # (atom_a, atom_b, t)
    worst_distance: float = float("inf")              # in physical um


def _sample_times(ensemble: AtomEnsemble, dt: float) -> np.ndarray:
    """Time samples that cover every moving interval at spacing `dt`,
    plus every segment boundary (so we don't miss endpoint collisions)."""
    intervals = ensemble.moving_intervals()
    if not intervals:
        return np.array([0.0])
    pts: list[float] = []
    for (s, e) in intervals:
        n = max(2, int(np.ceil((e - s) / dt)) + 1)
        pts.extend(np.linspace(s, e, n).tolist())
    # Add every segment boundary explicitly.
    for atom in ensemble.atoms:
        for seg in atom.segments:
            pts.append(seg.start_time)
            pts.append(seg.end_time)
    return np.unique(np.asarray(pts))


def validate_ensemble(ensemble: AtomEnsemble, *, dt: float = 1e-6) -> CollisionReport:
    ts = _sample_times(ensemble, dt)
    if ts.size <= 1:
        return CollisionReport(ok=True)

    M = len(ensemble.atoms)
    # Sample (T, M, 2) in grid units, then convert to um.
    samples = np.empty((len(ts), M, 2))
    for k, atom in enumerate(ensemble.atoms):
        for ti, t in enumerate(ts):
            samples[ti, k] = atom.position_at(float(t))
    c = (ensemble.grid.N - 1) / 2.0
    samples = (samples - c) * ensemble.grid.d   # -> um

    worst_d = float("inf")
    worst_pair = None
    rc = ensemble.grid.rc
    # Vectorize over time per pair to keep memory modest.
    for a in range(M):
        for b in range(a + 1, M):
            diff = samples[:, a, :] - samples[:, b, :]
            d = np.linalg.norm(diff, axis=1)
            ti = int(np.argmin(d))
            d_min = float(d[ti])
            if d_min < worst_d:
                worst_d = d_min
                worst_pair = (a, b, float(ts[ti]))
                if worst_d < rc:
                    # Keep scanning to find the very worst case (optional).
                    pass
    return CollisionReport(ok=worst_d >= rc, worst_pair=worst_pair, worst_distance=worst_d)
