"""Incremental collision validation.

Validation runs *per step append*: when new segments are added to the
ensemble, only those new segments need to be checked. Anything older
was already validated when it was appended, so we never re-check
existing-vs-existing pairs.

What we check, for each newly-added segment S of atom A:
  * S against every *other* atom's position function over S's time
    window — this covers (new vs new), (new vs concurrent existing),
    and (new vs at-rest atom that S might fly past).
  * (new vs new) pairs are de-duplicated so each pair is checked once.

What we don't re-check:
  * Pairs of existing segments that don't include any newly-added
    segment — already verified at their own append time.

Time sampling: dt-spaced inside the union of new-segment windows, plus
every new segment's start/end boundary so endpoint conflicts can't be
missed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .atom_trajectory import AtomEnsemble
from .segments import Segment


@dataclass
class CollisionReport:
    ok: bool
    # (atom_id_a, atom_id_b, t) at the closest sample, if any pair was below rc.
    worst_pair: tuple[int, int, float] | None = None
    worst_distance: float = float("inf")   # in physical um


def _sample_times(new_by_id: dict[int, list[Segment]], dt: float) -> np.ndarray:
    pts: list[float] = []
    t_lo = float("inf")
    t_hi = -float("inf")
    for segs in new_by_id.values():
        for s in segs:
            if s.duration <= 0:
                continue
            pts.append(s.start_time)
            pts.append(s.end_time)
            t_lo = min(t_lo, s.start_time)
            t_hi = max(t_hi, s.end_time)
    if t_hi <= t_lo:
        return np.array([])
    n = max(2, int(np.ceil((t_hi - t_lo) / dt)) + 1)
    pts.extend(np.linspace(t_lo, t_hi, n).tolist())
    return np.unique(np.asarray(pts))


def validate_new_segments(
    ensemble: AtomEnsemble,
    new_by_id: dict[int, list[Segment]],
    *,
    dt: float = 1e-6,
) -> CollisionReport:
    """Validate that the just-added segments don't cause collisions.

    Pass `new_by_id` as `{atom_id: [Segment, ...]}` — exactly what
    `Sequence.append` returns. The ensemble passed in must already
    contain the new segments (they need to be sample-able).
    """
    if not new_by_id:
        return CollisionReport(ok=True)

    ts = _sample_times(new_by_id, dt)
    if ts.size == 0:
        return CollisionReport(ok=True)

    # Sample (T, M, 2) for ALL atoms at the candidate times.
    M = len(ensemble.atoms)
    samples = np.empty((len(ts), M, 2))
    for k, atom in enumerate(ensemble.atoms):
        for ti, t in enumerate(ts):
            samples[ti, k] = atom.position_at(float(t))
    c = (ensemble.grid.N - 1) / 2.0
    samples = (samples - c) * ensemble.grid.d   # -> physical um

    new_idx = sorted(ensemble.index_of(aid) for aid in new_by_id)
    new_idx_set = set(new_idx)

    rc = ensemble.grid.rc
    worst_d = float("inf")
    worst_pair: tuple[int, int, float] | None = None

    # Each *new* atom A vs every other atom B. Skip (new, new) where
    # b_idx < a_idx so the pair is checked only once.
    for a_idx in new_idx:
        for b_idx in range(M):
            if b_idx == a_idx:
                continue
            if b_idx in new_idx_set and b_idx < a_idx:
                continue
            diff = samples[:, a_idx, :] - samples[:, b_idx, :]
            d = np.linalg.norm(diff, axis=1)
            ti = int(np.argmin(d))
            d_min = float(d[ti])
            if d_min < worst_d:
                worst_d = d_min
                worst_pair = (
                    int(ensemble.atoms[a_idx].atom_id),
                    int(ensemble.atoms[b_idx].atom_id),
                    float(ts[ti]),
                )

    return CollisionReport(ok=worst_d >= rc, worst_pair=worst_pair, worst_distance=worst_d)
