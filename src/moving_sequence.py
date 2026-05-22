"""Sequence: ordered list of Steps, building an AtomEnsemble incrementally.

A Sequence owns:
- the initial AtomConfig (resting state at t=0),
- the Grid (lives at the request/sequence level, not on AtomConfig),
- an AtomEnsemble that the steps mutate as they're appended,

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
class MovingSequence:
    grid: Grid
    initial: AtomConfig
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
        current ensemble state."""
        T = self.total_duration()
        return T if T > 0 else 0.0

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
        """Append a batch of steps that share `next_start_time()`, with
        order-independent validation.

        The naive "apply each step in order" loop has a subtle ordering
        bug: when step A is applied first, the collision validator sees
        every *other* batch member as a static atom at its initial site
        (because their motion hasn't been committed yet), and rejects A
        if A's endpoint sits on top of that static other. The fix here:
        if a step trips `CollisionError` against another not-yet-applied
        batch member, defer it and retry after the rest commit -- by
        then the conflicting member is moving and the validator sees its
        real trajectory. The batch is rolled back atomically if no step
        in a retry round can make progress (a genuine mutual collision).
        Non-collision errors (e.g. continuity) also trigger rollback.
        """
        if not steps:
            return {}

        ensemble = self._ensemble
        assert ensemble is not None
        old_lengths = {
            atom.atom_id: len(atom.segments) for atom in ensemble.atomtrajs
        }

        try:
            pending: list[Step] = list(steps)
            last_error: Exception | None = None
            while pending:
                still_pending: list[Step] = []
                committed_this_round = 0
                for step in pending:
                    try:
                        step.apply(ensemble)
                        committed_this_round += 1
                    except CollisionError as exc:
                        last_error = exc
                        still_pending.append(step)
                if committed_this_round == 0:
                    assert last_error is not None
                    raise last_error
                pending = still_pending
        except Exception:
            for atom in ensemble.atomtrajs:
                del atom.segments[old_lengths[atom.atom_id]:]
            raise

        self.steps.extend(steps)
        added: dict[int, list[Segment]] = {}
        for atom in ensemble.atomtrajs:
            new_segs = atom.segments[old_lengths[atom.atom_id]:]
            if new_segs:
                added[atom.atom_id] = list(new_segs)
        return added

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
