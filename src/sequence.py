"""Sequence: ordered list of Steps, building an AtomEnsemble incrementally.

A Sequence owns:
- the initial AtomConfig (resting state at t=0),
- the Grid (lives at the request/sequence level, not on AtomConfig),
- an AtomEnsemble that the steps mutate as they're appended,
- an `inter_step_gap` (e.g. trap settle time) inserted between steps.

Validation is automatic: every segment goes through
`AtomEnsemble.append_segment`, which checks continuity AND cross-atom
collision *before* committing. Step append therefore raises
`CollisionError` (a `ValueError` subclass) if the next step would
collide with what's already there. No separate "append-then-validate"
dance is needed.

Schedulers typically:
  1. ask `seq.next_start_time()` for the next valid step start_time,
  2. construct an AODStep / RIPAStep at that time,
  3. call `seq.append(step)` — which may raise `CollisionError`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .atom_config import AtomConfig, Grid
from .atom_trajectory import AtomEnsemble, CollisionError, CollisionReport
from .movement import Step
from .segments import Segment


@dataclass
class Sequence:
    grid: Grid
    initial: AtomConfig
    inter_step_gap: float = 0.0
    # Forwarded to the underlying AtomEnsemble's collision validator.
    collision_dt: float = 1e-6
    steps: list[Step] = field(default_factory=list)
    _ensemble: Optional[AtomEnsemble] = None

    def __post_init__(self):
        self._ensemble = AtomEnsemble.from_config(self.grid, self.initial)
        self._ensemble.collision_dt = self.collision_dt

    # ---- ensemble access ----------------------------------------------------

    @property
    def ensemble(self) -> AtomEnsemble:
        assert self._ensemble is not None
        return self._ensemble

    def total_duration(self) -> float:
        return self.ensemble.total_duration()

    def next_start_time(self) -> float:
        """Earliest valid start_time for a *new* step appended after the
        current ensemble state, accounting for `inter_step_gap`."""
        T = self.total_duration()
        return T + self.inter_step_gap if T > 0 else 0.0

    def final_config(self) -> AtomConfig:
        return self.ensemble.final_config()

    # ---- synchronous-snapshot view ------------------------------------------

    def current_config(self) -> AtomConfig:
        """Resting state at `next_start_time()` (alias for final_config)."""
        return self.final_config()

    def occupancy_now(self) -> dict[tuple[int, int], int]:
        """Map site -> atom_id at `next_start_time()`."""
        return self.ensemble.occupancy_at_rest(self.next_start_time())

    # ---- mutation -----------------------------------------------------------

    def append(self, step: Step) -> dict[int, list[Segment]]:
        """Apply `step` and return the segments it added, keyed by atom_id.

        Each segment goes through `ensemble.append_segment`, so collision
        validation runs automatically; this raises `CollisionError` if the
        step would put any atom within `grid.rc` of another. The ensemble
        is left unchanged on failure.
        """
        before = {a.atom_id: len(a.segments) for a in self.ensemble.atomtrajs}
        step.apply(self.ensemble)
        new = {
            a.atom_id: a.segments[before[a.atom_id] :]
            for a in self.ensemble.atomtrajs
            if len(a.segments) > before[a.atom_id]
        }
        self.steps.append(step)
        return new

    def append_sync_batch(self, steps: list[Step]) -> dict[int, list[Segment]]:
        """Append a batch of steps that share `next_start_time()`.

        After this call, `next_start_time()` reflects the latest end_time
        across the batch ("wait for everyone" semantics). Any collision
        inside the batch raises `CollisionError` from the offending step.
        """
        merged: dict[int, list[Segment]] = {}
        for step in steps:
            for aid, segs in self.append(step).items():
                merged.setdefault(aid, []).extend(segs)
        return merged

    # ---- re-validation at a different collision tolerance ------------------

    def validate(self, dt: float = 1e-6) -> CollisionReport:
        """Replay the sequence in a fresh ensemble with collision scale `dt`,
        returning the first failing CollisionReport or `ok=True`. Useful for
        re-checking with a tighter validator tolerance than append time.
        """
        fresh = AtomEnsemble.from_config(self.grid, self.initial)
        fresh.collision_dt = dt
        for step in self.steps:
            try:
                step.apply(fresh)
            except CollisionError as e:
                return e.report
        return CollisionReport(ok=True)
