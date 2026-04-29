"""Sequence: ordered list of Steps, building an AtomEnsemble incrementally.

A Sequence owns:
- the initial AtomConfig (resting state at t=0),
- an AtomEnsemble that the steps mutate as they're appended,
- an `inter_step_gap` (e.g. trap settle time) inserted between steps.

Schedulers typically:
  1. ask `seq.next_start_time()` for the next valid step start_time,
  2. construct an AODStep / RIPAStep at that time,
  3. call `seq.append(step)` — which applies the step to the ensemble
     and returns the newly-added segments (per atom_id) so callers can
     run incremental collision validation on them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .atoms import AtomConfig
from .atom_trajectory import AtomEnsemble
from .movement import Step
from .segments import Segment
from .validator import CollisionReport, validate_new_segments


@dataclass
class Sequence:
    initial: AtomConfig
    inter_step_gap: float = 0.0
    steps: list[Step] = field(default_factory=list)
    _ensemble: Optional[AtomEnsemble] = None

    def __post_init__(self):
        self._ensemble = AtomEnsemble.from_config(self.initial)

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
    #
    # Synchronous schedulers (AOD lattice ops, or RIPA used as a baseline
    # sync device) plan one batch at a time: query the current resting
    # state, decide a batch of moves, emit step(s) sharing one start_time,
    # and repeat. `next_start_time()` is a moment when no atom is in
    # flight, so `current_config()` / `occupancy_now()` return the
    # resting snapshot the scheduler should reason about.

    def current_config(self) -> AtomConfig:
        """Resting state at `next_start_time()` (alias for final_config)."""
        return self.final_config()

    def occupancy_now(self) -> dict[tuple[int, int], int]:
        """Map site -> atom_id at `next_start_time()`."""
        return self.ensemble.occupancy_at_rest(self.next_start_time())

    # ---- mutation + incremental validation ---------------------------------

    def append(self, step: Step) -> dict[int, list[Segment]]:
        """Apply `step` and return the segments it added, keyed by atom_id.

        The returned dict is what `validate_new_segments` consumes — older
        segments need not be re-checked since they were validated when
        they were appended.
        """
        before = {a.atom_id: len(a.segments) for a in self.ensemble.atoms}
        step.apply(self.ensemble)
        new = {
            a.atom_id: a.segments[before[a.atom_id]:]
            for a in self.ensemble.atoms
            if len(a.segments) > before[a.atom_id]
        }
        self.steps.append(step)
        return new

    def append_sync_batch(self, steps: list[Step]) -> dict[int, list[Segment]]:
        """Append a batch of steps that share `next_start_time()`.

        After this call, `next_start_time()` reflects the latest end_time
        across the batch ("wait for everyone" semantics). Returns the
        merged dict of segments added by all steps in the batch — pass it
        straight to `validate_new_segments` for a single combined check.
        """
        merged: dict[int, list[Segment]] = {}
        for step in steps:
            for aid, segs in self.append(step).items():
                merged.setdefault(aid, []).extend(segs)
        return merged

    def append_and_validate(self, step: Step, *, dt: float = 1e-6) -> CollisionReport:
        """Append `step`, then run incremental collision validation on the
        segments it added. Returns the report; does NOT roll back on
        failure (the caller decides whether to abort or continue)."""
        new = self.append(step)
        return validate_new_segments(self.ensemble, new, dt=dt)

    def validate(self, dt: float = 1e-6) -> CollisionReport:
        """Re-validate the whole sequence from scratch by replaying every
        step and validating only the segments it adds. Returns the first
        report whose `ok` is False, or `ok=True` if the whole replay is
        clean. Useful for round-trip / sanity testing."""
        fresh = AtomEnsemble.from_config(self.initial)
        for step in self.steps:
            before = {a.atom_id: len(a.segments) for a in fresh.atoms}
            step.apply(fresh)
            new = {
                a.atom_id: a.segments[before[a.atom_id]:]
                for a in fresh.atoms
                if len(a.segments) > before[a.atom_id]
            }
            rep = validate_new_segments(fresh, new, dt=dt)
            if not rep.ok:
                return rep
        return CollisionReport(ok=True)
