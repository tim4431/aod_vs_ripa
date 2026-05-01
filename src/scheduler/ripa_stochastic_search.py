"""Stochastic-search Manhattan-corridor scheduler.

Wraps `_manhattan_planner.plan_labelled` in a simulated-annealing loop
over three knobs the greedy planner picks deterministically:

  1. **Atom processing order** — which atom reserves first.
  2. **enable_swap_pairs** — whether the swap-pair joint planner runs.
  3. **Bijection** (uncolored only) — Hungarian's *Manhattan*-optimal
     assignment is not always corridor-time-optimal; local 2-target
     swaps explore the alternative.

Bad candidates (occupancy violation, corridor overlap, kinematic
close-approach) are rejected with `+inf` cost so the search treats them
as unreachable. Final result is the best-scored schedule, returned as a
`Schedule` whose `sequence` is the canonical src-side artifact.

Public API:

* function form — `search_labelled`, `search_uncolored`, `validate_schedule`
  (used by the lib-side rendering scenes).
* class form — `RIPASearchScheduler` (a `base.Scheduler` adapter that
  plugs into `src.benchmark.benchmark_schedulers`).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

from ..atom_config import Grid
from ..routing import Site
from ._manhattan_planner import (
    ManhattanParams,
    RoutingMove,
    Schedule,
    plan_labelled,
    validate_in_flight,
    validate_occupancy,
)
from .base import Scheduler
from .ripa_pebroute import (
    Assignment,
    UncoloredRequest,
    assign_uncolored,
)


# ---------------- result dataclass ----------------


@dataclass
class SearchResult:
    schedule: Schedule
    assignment: Optional[Assignment]
    n_iter: int
    accept_count: int
    initial_t_end: float
    best_t_end: float
    enable_swap_pairs_final: bool
    history_best: List[float] = field(default_factory=list)
    history_current: List[float] = field(default_factory=list)


# ---------------- helpers ----------------


def _evaluate(moves: Sequence[RoutingMove], grid: Grid,
              *, params: ManhattanParams,
              enable_swap_pairs: bool,
              t0_global: float = 0.0,
              static_sites: Optional[Sequence[Site]] = None,
              kinematic_check: bool = True,
              kinematic_dt: float = 1e-6,
              collision_dt: float = 1e-6) -> Tuple[float, Schedule]:
    """Plan a candidate schedule and reject invalid ones with `+inf`.

    The kinematic check (off by default in the lib's signature, on here
    because src's MovingSequence is collision-aware natively) catches
    transient cross-corridor overlaps the structural validators miss.
    """
    schedule = plan_labelled(
        list(moves), grid,
        params=params,
        t0_global=t0_global,
        static_sites=static_sites,
        presort=False,
        enable_swap_pairs=enable_swap_pairs,
        collision_dt=collision_dt,
        verbose=False,
    )
    if validate_occupancy(list(moves), schedule):
        return float("inf"), schedule
    if validate_in_flight(schedule, params):
        return float("inf"), schedule
    if kinematic_check:
        report = schedule.sequence.validate(dt=kinematic_dt)
        if not report.ok:
            return float("inf"), schedule
    return schedule.t_end, schedule


def _perturb_order(order: List[int], rng: random.Random) -> List[int]:
    """One of: pair-swap, insertion, 3-cycle. Keeps moves local-ish."""
    n = len(order)
    if n < 2:
        return order[:]
    new = order[:]
    move = rng.random()
    if move < 0.5:
        i, j = rng.sample(range(n), 2)
        new[i], new[j] = new[j], new[i]
    elif move < 0.85:
        i = rng.randrange(n)
        j = rng.randrange(n)
        x = new.pop(i)
        new.insert(j, x)
    else:
        if n >= 3:
            i, j, k = rng.sample(range(n), 3)
            new[i], new[j], new[k] = new[k], new[i], new[j]
    return new


# ---------------- labelled search ----------------


def search_labelled(moves: Sequence[RoutingMove], grid: Grid,
                    *,
                    params: Optional[ManhattanParams] = None,
                    n_iter: int = 400,
                    T0: Optional[float] = None,
                    cooling: float = 0.992,
                    try_disable_pairs: bool = True,
                    seed: int = 0,
                    static_sites: Optional[Sequence[Site]] = None,
                    kinematic_check: bool = True,
                    kinematic_dt: float = 1e-6,
                    collision_dt: float = 1e-6,
                    verbose: bool = False) -> SearchResult:
    """Stochastic-search labelled scheduler.

    Iterates: perturb processing order (and, with low prob., flip the
    swap-pair toggle); replan via `plan_labelled(..., presort=False)`;
    accept via simulated annealing on `Schedule.t_end`.
    """
    params = params or ManhattanParams()
    rng = random.Random(seed)
    moves = list(moves)
    n = len(moves)
    if n == 0:
        sched = plan_labelled([], grid, params=params,
                              collision_dt=collision_dt)
        return SearchResult(sched, None, 0, 0, 0.0, 0.0, True)

    order = list(range(n))
    enable_pairs = True
    cur_moves = [moves[k] for k in order]
    cur_t, cur_sched = _evaluate(
        cur_moves, grid, params=params,
        enable_swap_pairs=enable_pairs,
        static_sites=static_sites,
        kinematic_check=kinematic_check,
        kinematic_dt=kinematic_dt,
        collision_dt=collision_dt,
    )
    initial_t = cur_t

    best_t = cur_t
    best_sched = cur_sched
    best_order = order[:]
    best_pairs = enable_pairs

    if T0 is None:
        T0 = max((initial_t if math.isfinite(initial_t) else 1e-3) * 0.10, 5e-6)
    T = T0
    accept = 0
    history_best = [best_t]
    history_current = [cur_t]

    for it in range(n_iter):
        new_order = _perturb_order(order, rng)
        new_pairs = enable_pairs
        if try_disable_pairs and rng.random() < 0.05:
            new_pairs = not enable_pairs
        new_moves = [moves[k] for k in new_order]
        new_t, new_sched = _evaluate(
            new_moves, grid, params=params,
            enable_swap_pairs=new_pairs,
            static_sites=static_sites,
            kinematic_check=kinematic_check,
            kinematic_dt=kinematic_dt,
            collision_dt=collision_dt,
        )
        delta = new_t - cur_t
        if delta < 0 or rng.random() < math.exp(-delta / max(T, 1e-12)):
            order = new_order
            enable_pairs = new_pairs
            cur_t = new_t
            cur_sched = new_sched
            accept += 1
            if cur_t < best_t:
                best_t = cur_t
                best_sched = cur_sched
                best_order = order[:]
                best_pairs = enable_pairs
        T *= cooling
        history_best.append(best_t)
        history_current.append(cur_t)
        if verbose and (it + 1) % 50 == 0:
            print(f"  iter {it+1:4d}: T={T*1e6:7.2f}us  "
                  f"cur={cur_t*1e6:7.2f}us  best={best_t*1e6:7.2f}us  "
                  f"pairs={enable_pairs}")

    return SearchResult(
        schedule=best_sched, assignment=None,
        n_iter=n_iter, accept_count=accept,
        initial_t_end=initial_t, best_t_end=best_t,
        enable_swap_pairs_final=best_pairs,
        history_best=history_best, history_current=history_current,
    )


# ---------------- uncolored search ----------------


def _build_uncolored_moves(sources, targets, atom_ids, pairs, order):
    labeled = [
        RoutingMove(atom_id=atom_ids[i], src=sources[i], dst=targets[j])
        for i, j in pairs
    ]
    return [labeled[k] for k in order]


def search_uncolored(req: UncoloredRequest, grid: Grid,
                     *,
                     params: Optional[ManhattanParams] = None,
                     n_iter: int = 400,
                     T0: Optional[float] = None,
                     cooling: float = 0.992,
                     bijection_swap_prob: float = 0.15,
                     try_disable_pairs: bool = True,
                     seed: int = 0,
                     static_sites: Optional[Sequence[Site]] = None,
                     kinematic_check: bool = True,
                     kinematic_dt: float = 1e-6,
                     collision_dt: float = 1e-6,
                     verbose: bool = False) -> SearchResult:
    """Stochastic search for uncolored-pebble scheduling.

    Initial state: Hungarian-optimal bijection from `assign_uncolored`,
    natural processing order, swap-pair planning enabled.

    Search moves:
    - With probability `bijection_swap_prob`, swap two target
      assignments (changes the bijection — gains a few steps of
      Manhattan distance, but may shed bigger corridor congestion).
    - Otherwise perturb the processing order.
    - With probability 0.05, also flip the `enable_swap_pairs` toggle.
    """
    params = params or ManhattanParams()
    rng = random.Random(seed)

    for s in req.sources:
        if s[0] % 2 != 0 or s[1] % 2 != 0:
            raise ValueError(f"source {s} not on storage parity site")
    for t in req.targets:
        if t[0] % 2 != 0 or t[1] % 2 != 0:
            raise ValueError(f"target {t} not on storage parity site")

    initial_asg = assign_uncolored(req.sources, req.targets)
    n = len(req.sources)
    if n == 0:
        sched = plan_labelled([], grid, params=params,
                              collision_dt=collision_dt)
        return SearchResult(sched, initial_asg, 0, 0, 0.0, 0.0, True)

    if req.atom_ids is None:
        atom_ids = list(range(n))
    else:
        if len(req.atom_ids) != n:
            raise ValueError("atom_ids length must match sources")
        atom_ids = list(req.atom_ids)

    pairs = list(initial_asg.pairs)
    order = list(range(n))
    enable_pairs = True

    cur_moves = _build_uncolored_moves(req.sources, req.targets,
                                       atom_ids, pairs, order)
    cur_t, cur_sched = _evaluate(
        cur_moves, grid, params=params,
        enable_swap_pairs=enable_pairs,
        static_sites=static_sites,
        kinematic_check=kinematic_check,
        kinematic_dt=kinematic_dt,
        collision_dt=collision_dt,
    )
    initial_t = cur_t
    best_t = cur_t
    best_sched = cur_sched
    best_pairs_state = pairs[:]
    best_order = order[:]
    best_swap_pair_flag = enable_pairs

    if T0 is None:
        T0 = max((initial_t if math.isfinite(initial_t) else 1e-3) * 0.10, 5e-6)
    T = T0
    accept = 0
    history_best = [best_t]
    history_current = [cur_t]

    for it in range(n_iter):
        move = rng.random()
        new_pairs_state = pairs[:]
        new_order = order[:]
        new_swap_flag = enable_pairs
        if move < bijection_swap_prob and n >= 2:
            a, b = rng.sample(range(n), 2)
            ai, aj = new_pairs_state[a]
            bi, bj = new_pairs_state[b]
            new_pairs_state[a] = (ai, bj)
            new_pairs_state[b] = (bi, aj)
        else:
            new_order = _perturb_order(order, rng)
        if try_disable_pairs and rng.random() < 0.05:
            new_swap_flag = not enable_pairs

        new_moves = _build_uncolored_moves(
            req.sources, req.targets, atom_ids, new_pairs_state, new_order)
        new_t, new_sched = _evaluate(
            new_moves, grid, params=params,
            enable_swap_pairs=new_swap_flag,
            static_sites=static_sites,
            kinematic_check=kinematic_check,
            kinematic_dt=kinematic_dt,
            collision_dt=collision_dt,
        )

        delta = new_t - cur_t
        if delta < 0 or rng.random() < math.exp(-delta / max(T, 1e-12)):
            pairs = new_pairs_state
            order = new_order
            enable_pairs = new_swap_flag
            cur_t = new_t
            cur_sched = new_sched
            accept += 1
            if cur_t < best_t:
                best_t = cur_t
                best_sched = cur_sched
                best_pairs_state = pairs[:]
                best_order = order[:]
                best_swap_pair_flag = enable_pairs

        T *= cooling
        history_best.append(best_t)
        history_current.append(cur_t)

        if verbose and (it + 1) % 50 == 0:
            print(f"  iter {it+1:4d}: T={T*1e6:7.2f}us  "
                  f"cur={cur_t*1e6:7.2f}us  best={best_t*1e6:7.2f}us  "
                  f"pairs={enable_pairs}")

    final_assignment = Assignment(
        pairs=best_pairs_state,
        total_cost_sites=float(sum(initial_asg.cost_matrix[i, j]
                                   for i, j in best_pairs_state)),
        cost_matrix=initial_asg.cost_matrix,
    )
    return SearchResult(
        schedule=best_sched, assignment=final_assignment,
        n_iter=n_iter, accept_count=accept,
        initial_t_end=initial_t, best_t_end=best_t,
        enable_swap_pairs_final=best_swap_pair_flag,
        history_best=history_best, history_current=history_current,
    )


def validate_schedule(schedule: Schedule, moves: Sequence[RoutingMove],
                      params: Optional[ManhattanParams] = None,
                      *,
                      label: str = "",
                      kinematic: bool = True,
                      kinematic_dt: float = 1e-6) -> dict:
    """Run the validator suite on a schedule.

    Returns a dict with ``occupancy``, ``in_flight``, and ``kinematic``
    keys. The kinematic value is the `CollisionReport` if `kinematic` is
    True, otherwise an empty list. Mirrors the contract of the lib's
    `stochastic_search.validate_schedule`.
    """
    params = params or ManhattanParams()
    occ = validate_occupancy(list(moves), schedule)
    inf = validate_in_flight(schedule, params)
    kin: object = []
    if kinematic:
        kin = schedule.sequence.validate(dt=kinematic_dt)
    if label:
        kin_ok = (not kinematic) or getattr(kin, "ok", True)
        if not occ and not inf and kin_ok:
            print(f"  validator [{label}]: clean")
        else:
            print(f"  validator [{label}]: "
                  f"{len(occ)} occupancy, {len(inf)} corridor, "
                  f"kinematic={'ok' if kin_ok else 'COLLISION'}")
            for v in occ[:4]:
                print(f"    OCC: {v}")
            for v in inf[:8]:
                print(f"    INF: {v}")
    return {"occupancy": occ, "in_flight": inf, "kinematic": kin}


# ---------------- Scheduler-class wrapper ----------------


@dataclass
class RIPASearchScheduler(Scheduler):
    """`base.Scheduler` adapter for the simulated-annealing search.

    Wraps `search_labelled` (labeled requests) or `search_uncolored`
    (unlabeled requests). The search builds many candidate
    `MovingSequence`s internally; on completion, the best-scoring
    sequence is hoisted onto `self.sequence` so the base class's
    final-config check sees the right state.

    `kinematic_check` defaults to ``False`` here (vs the function form's
    ``True``). The default-on setting is right when callers want a
    *kinematically valid* schedule and are willing to spend iterations
    finding one; the benchmark adapter just wants the best schedule the
    search finds, valid or not, since it needs every scheduler to
    produce a final state. Flip to ``True`` if you need the validity
    guarantee.
    """

    params: Optional[ManhattanParams] = None
    n_iter: int = 400
    cooling: float = 0.992
    seed: int = 0
    kinematic_check: bool = False
    kinematic_dt: float = 1e-6
    bijection_swap_prob: float = 0.15
    try_disable_pairs: bool = True
    static_sites: Optional[Sequence[Site]] = None
    T0: Optional[float] = None

    def _plan(self) -> None:
        params = self.params or ManhattanParams()

        if self.request.labeled:
            moves = [
                RoutingMove(
                    atom_id=k,
                    src=tuple(self.request.src[k]),
                    dst=tuple(self.request.dst[k]),
                )
                for k in range(len(self.request.src))
            ]
            res = search_labelled(
                moves, self.request.grid,
                params=params,
                n_iter=self.n_iter,
                T0=self.T0,
                cooling=self.cooling,
                try_disable_pairs=self.try_disable_pairs,
                seed=self.seed,
                static_sites=self.static_sites,
                kinematic_check=self.kinematic_check,
                kinematic_dt=self.kinematic_dt,
                collision_dt=self.collision_dt,
                verbose=False,
            )
        else:
            req = UncoloredRequest(
                sources=list(self.request.src),
                targets=list(self.request.dst),
            )
            res = search_uncolored(
                req, self.request.grid,
                params=params,
                n_iter=self.n_iter,
                T0=self.T0,
                cooling=self.cooling,
                bijection_swap_prob=self.bijection_swap_prob,
                try_disable_pairs=self.try_disable_pairs,
                seed=self.seed,
                static_sites=self.static_sites,
                kinematic_check=self.kinematic_check,
                kinematic_dt=self.kinematic_dt,
                collision_dt=self.collision_dt,
                verbose=False,
            )

        # Hoist the best schedule's MovingSequence onto self.sequence so the
        # base class's final-config check (and any caller that reads
        # `scheduler.plan()`) sees the right state.
        self.sequence = res.schedule.sequence
