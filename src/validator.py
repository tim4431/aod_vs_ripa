"""Trajectory collision validator.

Within a single MovementStep we check pairwise distance between every
pair of moving + stationary atoms at a fine time grid. Two atoms are
considered colliding if their physical distance drops below `grid.rc`.

We sample because:
  - Bang-bang trajectories are smooth (no sharp corners), so missing a
    collision is unlikely if dt resolves the fastest expected motion.
  - Non-moving atoms count too: the validator pads the moving list with
    zero-duration `hold(...)` trajectories for the rest of the array.

For step durations < dt the validator effectively checks endpoints only.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .atoms import AtomConfig
from .movement import AODStep, RIPAStep
from .trajectories import Trajectory, hold


@dataclass
class CollisionReport:
    ok: bool
    # (atom_a, atom_b, t) at the worst (closest) sample, if any pair was below rc.
    worst_pair: tuple[int, int, float] | None = None
    worst_distance: float = float("inf")


def _expand_to_full(cfg: AtomConfig, step) -> tuple[list[Trajectory], float]:
    """Return one trajectory per atom in `cfg` and the step duration."""
    if isinstance(step, AODStep):
        step_r = step.to_ripa_step(cfg)
    elif isinstance(step, RIPAStep):
        step_r = step
    else:
        raise TypeError(type(step))

    duration = step_r.duration
    trajs: list[Trajectory] = [hold(tuple(p), duration) for p in cfg.positions]
    for k, tr in zip(step_r.atom_indices, step_r.trajectories):
        trajs[k] = tr
    return trajs, duration


def validate_step(cfg: AtomConfig, step, *, dt: float = 1e-6) -> CollisionReport:
    """Check pairwise collisions over the duration of one step.

    `dt` is the time sample spacing (seconds). 1us is a reasonable
    default for AOD/RIPA moves that finish in tens to hundreds of us.
    """
    trajs, duration = _expand_to_full(cfg, step)
    if duration <= 0:
        return CollisionReport(ok=True)

    n_steps = max(2, int(np.ceil(duration / dt)) + 1)
    ts = np.linspace(0.0, duration, n_steps)

    # Sample all atoms: shape (T, M, 2) in physical units (um).
    M = len(trajs)
    samples = np.empty((n_steps, M, 2))
    for k, tr in enumerate(trajs):
        samples[:, k, :] = tr.sample(ts)
    c = (cfg.grid.N - 1) / 2.0
    samples = (samples - c) * cfg.grid.d  # to physical um

    # Pairwise min distance across all time samples.
    worst_d = float("inf")
    worst_pair = None
    for a in range(M):
        for b in range(a + 1, M):
            diff = samples[:, a, :] - samples[:, b, :]
            d_min = float(np.min(np.linalg.norm(diff, axis=1)))
            if d_min < worst_d:
                worst_d = d_min
                t_idx = int(np.argmin(np.linalg.norm(diff, axis=1)))
                worst_pair = (a, b, float(ts[t_idx]))

    ok = worst_d >= cfg.grid.rc
    return CollisionReport(ok=ok, worst_pair=worst_pair, worst_distance=worst_d)
