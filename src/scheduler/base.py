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
from functools import lru_cache
from typing import Callable, Iterable, Literal

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
class UnlabeledScheduler(Scheduler):
    """Scheduler that requires an unlabeled (set->set) request.

    Concrete subclasses move atoms set-wise: any source may end up at any
    target. Useful as a marker for set-based planners (e.g. AOD) that decide
    pairing dynamically from geometry rather than from an upfront cost-matrix
    assignment.
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.request.labeled:
            raise ValueError(
                f"{type(self).__name__} requires an unlabeled (set->set) request"
            )


@dataclass
class LabeledScheduler(Scheduler):
    """Scheduler that requires a labeled (pairwise) request.

    Atom k must end up at `request.dst[k]`. The trivial identity assignment
    inherited from `Scheduler.target_assignment` is the right default.
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.request.labeled:
            raise ValueError(
                f"{type(self).__name__} requires a labeled (pairwise) request"
            )


@dataclass
class UNFromLabeledScheduler(Scheduler):
    """Solve unlabeled requests by first picking an atom->target assignment.

    A scheduler that is essentially labeled internally but accepts unlabeled
    requests by converting set->set into pairwise. The conversion costs every
    atom against every candidate target via `_estimated_pair_cost`, then runs
    one of two assignment objectives:

      * ``min_sum`` (default): minimize total cost. Uses an exact DP for
        small `n` and SciPy's Hungarian / Jonker-Volgenant solver otherwise,
        falling back to a greedy nearest assignment if SciPy is unavailable.
      * ``min_max``: bottleneck assignment that minimizes the worst single-atom
        cost, breaking ties by total cost. Forces atoms already in target slots
        to also participate in routing instead of self-assigning at zero cost.

    Labeled requests pass through to `Scheduler.target_assignment` unchanged.

    Subclasses override `_estimated_pair_cost` with a cost estimate appropriate
    for their routing primitive (path length, bang-bang duration, etc.). The
    default is Manhattan distance on the grid.
    """

    max_exact_unlabeled_atoms: int = 12
    unlabeled_assignment: Literal["min_sum", "min_max"] = "min_sum"

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.unlabeled_assignment not in ("min_sum", "min_max"):
            raise ValueError("unlabeled_assignment must be 'min_sum' or 'min_max'")

    def target_assignment(self) -> dict[int, Site]:
        if self.request.labeled:
            return super().target_assignment()

        sources = [tuple(site) for site in self.request.src]
        targets = [tuple(site) for site in self.request.dst]
        n = len(sources)
        if n == 0:
            return {}

        costs = [
            [self._estimated_pair_cost(atom_id, src, dst) for dst in targets]
            for atom_id, src in enumerate(sources)
        ]
        if self.unlabeled_assignment == "min_max":
            bottleneck = self._bottleneck_assignment(costs, targets)
            if bottleneck is not None:
                return bottleneck
        if n <= self.max_exact_unlabeled_atoms:
            return self._exact_min_cost_assignment(costs, targets)
        linear = self._linear_sum_assignment(costs, targets)
        if linear is not None:
            return linear
        return self._greedy_min_cost_assignment(costs, targets)

    def _estimated_pair_cost(self, atom_id: int, src: Site, dst: Site) -> float:
        """Cost of routing `atom_id` from `src` to `dst` for the cost matrix.

        Manhattan distance by default. Subclasses with non-uniform leg cost
        (e.g. bang-bang acceleration, blocked corridors) should override.
        """
        return float(abs(src[0] - dst[0]) + abs(src[1] - dst[1]))

    @staticmethod
    def _exact_min_cost_assignment(
        costs: list[list[float]],
        targets: list[Site],
    ) -> dict[int, Site]:
        n = len(costs)

        @lru_cache(maxsize=None)
        def solve(atom_idx: int, mask: int) -> tuple[float, tuple[int, ...]]:
            if atom_idx == n:
                return 0.0, ()

            best_cost = float("inf")
            best_order: tuple[int, ...] = ()
            for target_idx in range(n):
                bit = 1 << target_idx
                if mask & bit:
                    continue
                rest_cost, rest_order = solve(atom_idx + 1, mask | bit)
                total = costs[atom_idx][target_idx] + rest_cost
                if total < best_cost:
                    best_cost = total
                    best_order = (target_idx,) + rest_order
            return best_cost, best_order

        _, order = solve(0, 0)
        return {atom_id: targets[target_idx] for atom_id, target_idx in enumerate(order)}

    @staticmethod
    def _greedy_min_cost_assignment(
        costs: list[list[float]],
        targets: list[Site],
    ) -> dict[int, Site]:
        remaining = set(range(len(targets)))
        assignment: dict[int, Site] = {}
        for atom_id, row in enumerate(costs):
            target_idx = min(remaining, key=lambda idx: (row[idx], targets[idx]))
            remaining.remove(target_idx)
            assignment[atom_id] = targets[target_idx]
        return assignment

    @staticmethod
    def _linear_sum_assignment(
        costs: list[list[float]],
        targets: list[Site],
    ) -> dict[int, Site] | None:
        """SciPy Hungarian / Jonker-Volgenant solver. None if SciPy missing."""
        try:
            from scipy.optimize import linear_sum_assignment
        except Exception:
            return None

        row_ind, col_ind = linear_sum_assignment(costs)
        return {
            int(atom_id): targets[int(target_idx)]
            for atom_id, target_idx in zip(row_ind, col_ind)
        }

    @staticmethod
    def _bottleneck_assignment(
        costs: list[list[float]],
        targets: list[Site],
    ) -> dict[int, Site] | None:
        """Min-max bipartite assignment: minimize the worst single-atom cost,
        breaking ties by min total cost. Returns None if scipy is unavailable.
        """
        try:
            import numpy as np
            from scipy.optimize import linear_sum_assignment
        except Exception:
            return None

        C = np.asarray(costs, dtype=float)
        if C.size == 0:
            return {}
        thresholds = np.unique(C)
        BIG = float(C.max() + 1.0) * (C.shape[0] + 1) + 1.0

        # Binary search for the smallest threshold admitting a perfect matching.
        lo, hi = 0, len(thresholds) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            masked = np.where(C <= thresholds[mid] + 1e-15, C, BIG)
            row_ind, col_ind = linear_sum_assignment(masked)
            if masked[row_ind, col_ind].max() < BIG:
                hi = mid
            else:
                lo = mid + 1

        # Final assignment: min-sum within the threshold-feasible subgraph.
        masked = np.where(C <= thresholds[lo] + 1e-15, C, BIG)
        row_ind, col_ind = linear_sum_assignment(masked)
        return {
            int(atom_id): targets[int(target_idx)]
            for atom_id, target_idx in zip(row_ind, col_ind)
        }


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
