"""Sequence: ordered list of Steps, building an AtomEnsemble incrementally.

A Sequence owns:
- the initial AtomConfig (resting state at t=0),
- an AtomEnsemble that the steps mutate as they're appended,
- an `inter_step_gap` (e.g. trap settle time) inserted between steps.

Schedulers typically:
  1. ask `seq.next_start_time()` for the next valid step start_time,
  2. construct an AODStep / RIPAStep at that time,
  3. call `seq.append(step)` — which applies the step to the ensemble.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .atoms import AtomConfig
from .atom_trajectory import AtomEnsemble
from .movement import Step
from .validator import CollisionReport, validate_ensemble


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
    # and repeat. By construction `next_start_time()` is a moment when no
    # atom is mid-flight, so `current_config()` and `occupancy_now()`
    # return the resting snapshot the scheduler should reason about.
    #
    # For a sync RIPA batch, append several RIPASteps with the same
    # start_time = next_start_time() before calling next_start_time()
    # again — the batch ends at the max end_time, which is exactly what
    # `total_duration()` will then report.

    def current_config(self) -> AtomConfig:
        """Resting state at `next_start_time()` (alias for final_config)."""
        return self.final_config()

    def occupancy_now(self) -> dict[tuple[int, int], int]:
        """Map site -> atom_id at `next_start_time()`."""
        return self.ensemble.occupancy_at_rest(self.next_start_time())

    # ---- mutation -----------------------------------------------------------

    def append(self, step: Step) -> None:
        """Apply the step to the running ensemble and remember it."""
        step.apply(self.ensemble)
        self.steps.append(step)

    def append_sync_batch(self, steps: list[Step]) -> None:
        """Append a batch of steps that share `next_start_time()`.

        Convenience for synchronous RIPA scheduling: after this call,
        `next_start_time()` reflects the latest end_time across the batch
        (i.e. the synchronous "wait for everyone" semantics).
        """
        for step in steps:
            self.append(step)

    # ---- validation ---------------------------------------------------------

    def validate(self, dt: float = 1e-6) -> CollisionReport:
        return validate_ensemble(self.ensemble, dt=dt)
