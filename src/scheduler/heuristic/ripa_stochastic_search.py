"""Stochastic-search Manhattan-corridor scheduler.

Wraps `_manhattan_planner.plan_labelled` in a simulated-annealing loop
over three knobs the greedy planner picks deterministically:

  1. **Atom processing order** — which atom reserves first.
  2. **enable_swap_pairs** — whether the swap-pair joint planner runs.
  3. **Bijection** (uncolored only) — Hungarian's *Manhattan*-optimal
     assignment is not always corridor-time-optimal; local 2-target
     swaps explore the alternative.

Final result is the best-scored schedule, returned as a
`MovingSequence` (the canonical src-side artifact).

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

from ...atom_config import Grid
from ...moving_sequence import MovingSequence
from ...routing import Site
from ._manhattan_planner import (
    RoutingMove,
    T_HANDOFF_DEFAULT,
    plan_labelled,
)
from ..base import Scheduler
from .ripa_pebroute import (
    Assignment,
    UncoloredRequest,
    assign_uncolored,
)


# ---------------- result dataclass ----------------


@dataclass
class SearchResult:
    """Best schedule the SA loop produced, plus a few stats for inspection.

    `sequence` is the canonical artefact (a `MovingSequence`); `assignment`
    is set only for uncolored searches that picked a Hungarian-style bijection.
    """

    sequence: MovingSequence
    assignment: Optional[Assignment]
    n_iter: int
    accept_count: int
    initial_t_end: float
    best_t_end: float
    enable_swap_pairs_final: bool
    history_best: List[float] = field(default_factory=list)
    history_current: List[float] = field(default_factory=list)


# ---------------- helpers ----------------


def _is_collision_free(sequence: MovingSequence) -> bool:
    """Pairwise close-approach check on the full committed sequence.

    The Manhattan planner commits via the bypass path (segments appended
    straight to `AtomTrajectory.segments` so the per-step `post_hold`
    check doesn't trip on incremental state). To screen out the lib's
    cross-corridor transient overlaps anyway, we run the framework's own
    pairwise distance helper segment-by-segment: each segment becomes a
    candidate against every other atom's full timeline (segments +
    implicit hold gaps via `AtomTrajectory.timeline_segments`). The
    `min distance >= grid.rc` test inside
    `AtomEnsemble._check_candidate_collisions` is the same one the
    framework uses on `append_segment`.
    """
    ens = sequence.ensemble
    for atom in ens.atomtrajs:
        for seg in atom.segments:
            if not ens._check_candidate_collisions({atom.atom_id: seg}).ok:
                return False
    return True


def _evaluate(moves: Sequence[RoutingMove], grid: Grid,
              *, t_handoff: float,
              enable_swap_pairs: bool,
              static_sites: Optional[Sequence[Site]] = None,
              collision_dt: float = 1e-6) -> Tuple[float, MovingSequence]:
    """Plan a candidate schedule and return its `(t_end, sequence)`.

    Colliding candidates get `t_end = +inf` so the SA loop rejects them.
    """
    sequence = plan_labelled(
        list(moves), grid,
        t_handoff=t_handoff,
        static_sites=static_sites,
        presort=False,
        enable_swap_pairs=enable_swap_pairs,
        collision_dt=collision_dt,
        verbose=False,
    )
    if not _is_collision_free(sequence):
        return float("inf"), sequence
    return sequence.total_duration(), sequence


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
                    t_handoff: float = T_HANDOFF_DEFAULT,
                    n_iter: int = 400,
                    T0: Optional[float] = None,
                    cooling: float = 0.992,
                    try_disable_pairs: bool = True,
                    seed: int = 0,
                    static_sites: Optional[Sequence[Site]] = None,
                    collision_dt: float = 1e-6,
                    verbose: bool = False) -> SearchResult:
    """Stochastic-search labelled scheduler.

    Iterates: perturb processing order (and, with low prob., flip the
    swap-pair toggle); replan via `plan_labelled(..., presort=False)`;
    accept via simulated annealing on `MovingSequence.total_duration()`.
    """
    rng = random.Random(seed)
    moves = list(moves)
    n = len(moves)
    if n == 0:
        seq = plan_labelled([], grid, t_handoff=t_handoff,
                            collision_dt=collision_dt)
        return SearchResult(seq, None, 0, 0, 0.0, 0.0, True)

    order = list(range(n))
    enable_pairs = True
    cur_moves = [moves[k] for k in order]
    cur_t, cur_seq = _evaluate(
        cur_moves, grid, t_handoff=t_handoff,
        enable_swap_pairs=enable_pairs,
        static_sites=static_sites,
        collision_dt=collision_dt,
    )
    initial_t = cur_t

    best_t = cur_t
    best_seq = cur_seq
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
        new_t, new_seq = _evaluate(
            new_moves, grid, t_handoff=t_handoff,
            enable_swap_pairs=new_pairs,
            static_sites=static_sites,
            collision_dt=collision_dt,
        )
        delta = new_t - cur_t
        if delta < 0 or rng.random() < math.exp(-delta / max(T, 1e-12)):
            order = new_order
            enable_pairs = new_pairs
            cur_t = new_t
            cur_seq = new_seq
            accept += 1
            if cur_t < best_t:
                best_t = cur_t
                best_seq = cur_seq
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
        sequence=best_seq, assignment=None,
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
                     t_handoff: float = T_HANDOFF_DEFAULT,
                     n_iter: int = 400,
                     T0: Optional[float] = None,
                     cooling: float = 0.992,
                     bijection_swap_prob: float = 0.15,
                     try_disable_pairs: bool = True,
                     seed: int = 0,
                     static_sites: Optional[Sequence[Site]] = None,
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
    rng = random.Random(seed)

    initial_asg = assign_uncolored(req.sources, req.targets)
    n = len(req.sources)
    if n == 0:
        seq = plan_labelled([], grid, t_handoff=t_handoff,
                            collision_dt=collision_dt)
        return SearchResult(seq, initial_asg, 0, 0, 0.0, 0.0, True)

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
    cur_t, cur_seq = _evaluate(
        cur_moves, grid, t_handoff=t_handoff,
        enable_swap_pairs=enable_pairs,
        static_sites=static_sites,
        collision_dt=collision_dt,
    )
    initial_t = cur_t
    best_t = cur_t
    best_seq = cur_seq
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
        new_t, new_seq = _evaluate(
            new_moves, grid, t_handoff=t_handoff,
            enable_swap_pairs=new_swap_flag,
            static_sites=static_sites,
            collision_dt=collision_dt,
        )

        delta = new_t - cur_t
        if delta < 0 or rng.random() < math.exp(-delta / max(T, 1e-12)):
            pairs = new_pairs_state
            order = new_order
            enable_pairs = new_swap_flag
            cur_t = new_t
            cur_seq = new_seq
            accept += 1
            if cur_t < best_t:
                best_t = cur_t
                best_seq = cur_seq
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
        sequence=best_seq, assignment=final_assignment,
        n_iter=n_iter, accept_count=accept,
        initial_t_end=initial_t, best_t_end=best_t,
        enable_swap_pairs_final=best_swap_pair_flag,
        history_best=history_best, history_current=history_current,
    )


def validate_schedule(sequence: MovingSequence,
                      moves: Sequence[RoutingMove] = (),
                      *,
                      label: str = "") -> dict:
    """Pairwise close-approach check on a committed sequence.

    Walks every atom's segments and checks each one against the other
    atoms' full timelines via `AtomEnsemble._check_candidate_collisions`
    — the same framework helper that drives `append_segment`'s collision
    test. Returns the worst-pair report so callers can pinpoint where
    the schedule goes wrong. `moves` is accepted for API parity with the
    lib's `validate_schedule` signature but not consulted.
    """
    del moves
    ens = sequence.ensemble
    worst_pair = None
    worst_dist = float("inf")
    for atom in ens.atomtrajs:
        for seg in atom.segments:
            rep = ens._check_candidate_collisions({atom.atom_id: seg})
            if not rep.ok and rep.worst_distance < worst_dist:
                worst_pair = rep.worst_pair
                worst_dist = rep.worst_distance
    ok = worst_pair is None
    if label:
        if ok:
            print(f"  validator [{label}]: clean")
        else:
            a, b, t = worst_pair
            print(f"  validator [{label}]: COLLISION  "
                  f"atoms {a}/{b} at t={t*1e6:.1f} us "
                  f"(min distance {worst_dist:.3f} um)")
    return {"ok": ok, "worst_pair": worst_pair, "worst_distance": worst_dist}


# ---------------- Scheduler-class wrapper ----------------


@dataclass
class RIPASearchScheduler(Scheduler):
    """`base.Scheduler` adapter for the simulated-annealing search.

    Wraps `search_labelled` (labeled requests) or `search_uncolored`
    (unlabeled requests). Each candidate is screened by
    `_is_collision_free` (which calls the same `_check_candidate_collisions`
    test that gates `AtomEnsemble.append_segment`); colliding candidates
    are tagged with `+inf` cost and the SA never picks them. On
    completion the best-scoring sequence is hoisted onto `self.sequence`.

    If the algorithm can't find any collision-free schedule within
    `n_iter`, the returned sequence falls back to the best (possibly
    colliding) candidate seen — call `validate_schedule(self.sequence)`
    after `plan()` to find out.
    """

    t_handoff: float = T_HANDOFF_DEFAULT
    n_iter: int = 400
    cooling: float = 0.992
    seed: int = 0
    bijection_swap_prob: float = 0.15
    try_disable_pairs: bool = True
    static_sites: Optional[Sequence[Site]] = None
    T0: Optional[float] = None

    def _plan(self) -> None:
        if self.request.labeled:
            moves = [
                RoutingMove(atom_id=k,
                            src=tuple(self.request.src[k]),
                            dst=tuple(self.request.dst[k]))
                for k in range(len(self.request.src))
            ]
            res = search_labelled(
                moves, self.request.grid,
                t_handoff=self.t_handoff,
                n_iter=self.n_iter, T0=self.T0, cooling=self.cooling,
                try_disable_pairs=self.try_disable_pairs,
                seed=self.seed,
                static_sites=self.static_sites,
                collision_dt=self.collision_dt,
            )
        else:
            req = UncoloredRequest(
                sources=list(self.request.src),
                targets=list(self.request.dst),
            )
            res = search_uncolored(
                req, self.request.grid,
                t_handoff=self.t_handoff,
                n_iter=self.n_iter, T0=self.T0, cooling=self.cooling,
                bijection_swap_prob=self.bijection_swap_prob,
                try_disable_pairs=self.try_disable_pairs,
                seed=self.seed,
                static_sites=self.static_sites,
                collision_dt=self.collision_dt,
            )

        # Hoist the best sequence onto self.sequence so the base class's
        # final-config check sees the right state.
        self.sequence = res.sequence
