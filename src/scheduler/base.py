"""Base scheduler abstractions.

A scheduler owns the live `Sequence` for one `RoutingRequest`. Concrete
schedulers decide which movement steps to append; the base class handles
initial `AtomEnsemble` creation, final-state checks, and trial helpers for
backtracking / stochastic search.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Iterable

from ..atom_trajectory import AtomEnsemble
from ..movement import Step
from ..routing import RoutingRequest, Site
from ..sequence import Sequence

CostFunction = Callable[[Sequence], float]
CandidateBuilder = Callable[[Sequence], None]


@dataclass(frozen=True)
class ScheduleTrial:
    """Result of evaluating a candidate on cloned scheduler state."""

    ok: bool
    cost: float = math.inf
    sequence: Sequence | None = None
    error: Exception | None = None
    clock_cycles: int = 0


@dataclass
class Scheduler(ABC):
    """Abstract base class for RoutingRequest schedulers."""

    request: RoutingRequest
    inter_step_gap: float = 0.0
    collision_dt: float = 1e-6
    validate_final: bool = True
    cost_function: CostFunction | None = None
    sequence: Sequence = field(init=False)
    last_error: Exception | None = field(default=None, init=False)
    _planned: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        self.sequence = Sequence(
            grid=self.request.grid,
            initial=self.request.initial,
            inter_step_gap=self.inter_step_gap,
            collision_dt=self.collision_dt,
        )

    @property
    def ensemble(self) -> AtomEnsemble:
        return self.sequence.ensemble

    def plan(self) -> Sequence:
        """Build and return the movement sequence."""
        if self._planned:
            return self.sequence

        self._plan()
        if self.validate_final:
            self._check_final_config()
        self._planned = True
        return self.sequence

    schedule = plan

    @abstractmethod
    def _plan(self) -> None:
        """Append the steps needed to satisfy `request`."""

    def append_step(self, step: Step) -> dict[int, list]:
        """Append one step, propagating validation errors to the caller."""
        self.last_error = None
        return self.sequence.append(step)

    def try_append_step(self, step: Step) -> bool:
        """Best-effort append helper for exploratory schedulers."""
        try:
            self.append_step(step)
        except Exception as exc:
            self.last_error = exc
            return False
        return True

    # ---- trial/evaluation helpers -----------------------------------------

    def cost(self, sequence: Sequence | None = None) -> float:
        """Objective value for a candidate sequence.

        The default objective is the physical arrangement duration, matching
        the current project benchmark. A scheduler can supply `cost_function`
        to add penalties or alternate objectives without changing the search
        scaffolding.
        """
        seq = self.sequence if sequence is None else sequence
        if self.cost_function is not None:
            return float(self.cost_function(seq))
        return float(seq.total_duration())

    def clone_sequence(self, sequence: Sequence | None = None) -> Sequence:
        """Replay a sequence into a fresh `Sequence` for speculative edits."""
        source = self.sequence if sequence is None else sequence
        clone = Sequence(
            grid=source.grid,
            initial=source.initial.copy(),
            inter_step_gap=source.inter_step_gap,
            collision_dt=source.collision_dt,
        )
        for step in source.steps:
            clone.append(step)
        return clone

    def evaluate_candidate(
        self,
        build: CandidateBuilder,
        *,
        base: Sequence | None = None,
        clock_cycles: int = 1,
    ) -> ScheduleTrial:
        """Run `build(cloned_sequence)` and score it without live mutation."""
        trial_sequence = self.clone_sequence(base)
        try:
            build(trial_sequence)
        except Exception as exc:
            return ScheduleTrial(
                ok=False,
                error=exc,
                clock_cycles=max(0, int(clock_cycles)),
            )
        return ScheduleTrial(
            ok=True,
            cost=self.cost(trial_sequence),
            sequence=trial_sequence,
            clock_cycles=max(0, int(clock_cycles)),
        )

    def evaluate_steps(
        self,
        steps: Iterable[Step],
        *,
        base: Sequence | None = None,
        clock_cycles: int | None = None,
    ) -> ScheduleTrial:
        """Evaluate appending `steps` one after another on cloned state."""
        batch = tuple(steps)

        def build(seq: Sequence) -> None:
            for step in batch:
                seq.append(step)

        cycles = len(batch) if clock_cycles is None else clock_cycles
        return self.evaluate_candidate(
            build,
            base=base,
            clock_cycles=cycles,
        )

    def commit_trial(self, trial: ScheduleTrial) -> Sequence:
        """Promote a successful trial sequence to live scheduler state."""
        if not trial.ok or trial.sequence is None:
            if trial.error is not None:
                self.last_error = trial.error
            raise ValueError("cannot commit a failed schedule trial")
        self.sequence = trial.sequence
        self.last_error = None
        return self.sequence

    def best_trial(self, trials: Iterable[ScheduleTrial]) -> ScheduleTrial | None:
        """Return the lowest-cost successful trial, or `None` if all failed."""
        ok_trials = [trial for trial in trials if trial.ok]
        if not ok_trials:
            return None
        return min(ok_trials, key=lambda trial: trial.cost)

    def target_assignment(self) -> dict[int, Site]:
        """Return atom_id -> target site for schedulers that move atoms directly."""
        if self.request.labeled:
            return {atom_id: tuple(site) for atom_id, site in enumerate(self.request.dst)}
        return _greedy_nearest_assignment(self.request.src, self.request.dst)

    def _check_final_config(self) -> None:
        final = self.sequence.final_config()
        if self.request.labeled:
            site_by_atom = final.site_of_atom()
            for atom_id, target in enumerate(self.request.dst):
                if site_by_atom.get(atom_id) != tuple(target):
                    raise ValueError(
                        "labeled schedule ended at the wrong site for atom "
                        f"{atom_id}: got {site_by_atom.get(atom_id)}, "
                        f"expected {tuple(target)}"
                    )
            return

        final_sites = final.occupied_sites()
        target_sites = self.request.target_sites()
        if final_sites != target_sites:
            missing = sorted(target_sites - final_sites)
            extra = sorted(final_sites - target_sites)
            raise ValueError(
                "unlabeled schedule ended at the wrong target set: "
                f"missing={missing}, extra={extra}"
            )


@dataclass
class SyncScheduler(Scheduler):
    """Scheduler base for clocked, synchronous movement cycles."""

    clock_cycle: int = field(default=0, init=False)

    def append_step(self, step: Step) -> dict[int, list]:
        added = super().append_step(step)
        self.clock_cycle += 1
        return added

    def append_cycle(self, steps: Iterable[Step]) -> dict[int, list]:
        """Append several same-start steps as one scheduler clock cycle."""
        batch = list(steps)
        if not batch:
            return {}
        self.last_error = None
        added = self.sequence.append_sync_batch(batch)
        self.clock_cycle += 1
        return added

    def try_append_cycle(self, steps: Iterable[Step]) -> bool:
        trial = self.evaluate_cycle(steps)
        if not trial.ok:
            self.last_error = trial.error
            return False
        self.commit_trial(trial)
        return True

    def evaluate_cycle(
        self,
        steps: Iterable[Step],
        *,
        base: Sequence | None = None,
    ) -> ScheduleTrial:
        """Evaluate a same-start batch as one scheduler clock cycle."""
        batch = list(steps)

        def build(seq: Sequence) -> None:
            seq.append_sync_batch(batch)

        return self.evaluate_candidate(build, base=base, clock_cycles=1)

    def commit_trial(self, trial: ScheduleTrial) -> Sequence:
        try:
            sequence = super().commit_trial(trial)
        except Exception as exc:
            self.last_error = exc
            raise
        self.clock_cycle += trial.clock_cycles
        return sequence


@dataclass
class AsyncScheduler(Scheduler):
    """Marker base for future schedulers with independently timed moves."""


@dataclass
class LabeledScheduler(Scheduler):
    """Base class for schedulers that handle labeled (pairwise) requests.

    A labeled scheduler routes the atom currently at `src[k]` to `dst[k]`.
    The request must have `labeled=True`; an unlabeled set→set request must
    first be wrapped by an `UnlabeledScheduler` that picks an atom→target
    assignment.
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.request.labeled:
            raise ValueError(
                f"{type(self).__name__} requires a labeled RoutingRequest "
                "(labeled=True); wrap it in an UnlabeledScheduler to assign "
                "atoms to targets first."
            )


@dataclass
class UnlabeledScheduler(Scheduler):
    """Base class for schedulers that handle unlabeled (set→set) requests.

    An unlabeled scheduler may choose any bijection from `src` to `dst`. A
    common implementation picks an atom→target assignment and delegates the
    resulting labeled problem to a `LabeledScheduler`.
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.request.labeled:
            raise ValueError(
                f"{type(self).__name__} expects an unlabeled RoutingRequest "
                "(labeled=False)."
            )


def _greedy_nearest_assignment(
    sources: Iterable[Site],
    targets: Iterable[Site],
) -> dict[int, Site]:
    remaining = [tuple(site) for site in targets]
    assignment: dict[int, Site] = {}
    for atom_id, src in enumerate(sources):
        if not remaining:
            break
        si, sj = src
        best_index = min(
            range(len(remaining)),
            key=lambda k: (
                math.hypot(remaining[k][0] - si, remaining[k][1] - sj),
                remaining[k],
            ),
        )
        assignment[atom_id] = remaining.pop(best_index)
    return assignment
