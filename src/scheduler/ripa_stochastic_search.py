"""Stochastic-search pebble scheduler — exploits the freedom that the
greedy planner does not.

Motivation
----------
`ripa_pebroute.plan_uncolored` (Stage A) finds the cost-optimal bijection
via Hungarian and then defers to `manhattan.plan`. `manhattan.plan` is a
greedy, order-dependent algorithm: different processing orders produce
different schedules, sometimes by a lot. Greedy also commits irrevocably to
its swap-pair joint planner whenever it can — but in some configurations
the corridor that joint plan grabs would have been better used by the rest
of the schedule.

This module wraps `manhattan.plan` in a stochastic local-search loop
(simulated annealing) over three knobs that the greedy planner picks
deterministically:

  1. **Atom processing order** (the most consequential knob): which atom
     reserves first.
  2. **enable_swap_pairs** flag: sometimes the joint planner is the right
     call, sometimes it isn't.
  3. **Bijection** (uncolored only): the Hungarian-optimal assignment is
     optimal in *Manhattan distance*, but the corridor-time-optimal one
     may differ. Local 2-target swaps explore that.

State decoded into a labelled-request list, then evaluated by
`scheduler.plan(requests, ..., presort=False, enable_swap_pairs=...)`.
The Schedule is the same dataclass as before — fully drop-in.

Why timing/stretch isn't here yet
---------------------------------
The user note about *atoms not always running at a_max* — i.e. choosing
slower velocity profiles to fit two co-directional movers into the same
corridor without conflict — is a real win, but it requires plumbing a
per-atom stretch factor through `_segment_min_duration`, `_commit_segs`
and the joint planner. Left as a follow-up; this file already finds
substantial improvements on the unswapped portion of the schedule.

Port note: this module is the project-side mirror of
`lib/ripa2/visualization/src/schedulers/stochastic_search.py`. Logic is
unchanged; only the imports use absolute lib namespace paths.
"""
import math
import random
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

from lib.ripa2.visualization.src.atoms import Atom, Grid
from lib.ripa2.visualization.src.params import Params
from lib.ripa2.visualization.src.schedulers.manhattan import (
    RoutingRequest,
    Schedule,
    plan as _plan_labelled,
)
from lib.ripa2.visualization.src.sequence import Sequence as _Sequence, StaticTraps
from lib.ripa2.visualization.src.validators import (
    validate_occupancy,
    validate_in_flight,
    validate_kinematic,
    report,
)

from .ripa_pebroute import (
    UncoloredRequest,
    Assignment,
    assign_uncolored,
)


Site = Tuple[int, int]


def _build_seq_for_validation(grid: Grid, params: Params,
                              requests, schedule: Schedule) -> _Sequence:
    """Cheap Sequence construction for the kinematic validator. Atoms are
    placed at their request-source with the schedule's trajectory (or no
    trajectory for atoms the planner left empty)."""
    atoms = []
    for r in requests:
        traj = schedule.trajectories.get(r.atom_id)
        atoms.append(Atom(id=r.atom_id, pos=grid.site_xy(*r.src),
                          trajectory=traj))
    return _Sequence(grid=grid, atoms=atoms, static_traps=StaticTraps(),
                     params=params, t_start=0.0,
                     t_end=max(schedule.t_end, 1e-6) + 1e-6,
                     fps=10, playback_duration=1.0)


def validate_schedule(schedule: Schedule, requests, params: Params,
                      label: str = "", seq=None,
                      kinematic: bool = True) -> dict:
    """Run validators on the schedule.

    If `seq` is supplied, also runs the kinematic (physics-based) validator
    that samples each atom's position over time and flags any pair within
    one site of each other. Returns the dict from `validators.report`.
    """
    if seq is not None:
        return report(seq, requests, schedule, params,
                      label=label, kinematic=kinematic, verbose=True)
    occ = validate_occupancy(list(requests), schedule, params)
    inf = validate_in_flight(schedule, params)
    if label:
        if not occ and not inf:
            print(f"  validator [{label}]: clean (occupancy, corridor)")
        else:
            print(f"  validator [{label}]: "
                  f"{len(occ)} occupancy, {len(inf)} corridor violations")
            for v in occ[:4]:
                print(f"    OCC: {v.describe()}")
            for v in inf[:8]:
                print(f"    INF: {v}")
    return {"occupancy": occ, "in_flight": inf, "kinematic": []}


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

def _evaluate(requests, grid: Grid, params: Params,
              *, enable_swap_pairs: bool,
              t0_global: float = 0.0,
              static_sites=None,
              kinematic_check: bool = True,
              kinematic_n_samples: int = 600) -> Tuple[float, Schedule]:
    """Evaluate a candidate schedule and *reject invalid ones*.

    Returns (effective_t_end, schedule). Effective t_end is +inf if the
    schedule has any:
      - endpoint occupancy violation (atom lands at still-occupied site)
      - corridor-reservation overlap (cheap structural check)
      - kinematic close-approach (atoms within 1 site at any sampled time —
        catches dwell-position collisions, crossing-corner overlaps, and
        anything else the structural checks miss)

    `kinematic_check=True` runs the kinematic validator each iteration with
    `kinematic_n_samples` samples (default 600 = ~10x cheaper than the
    final-report 2500 but still catches every collision the planner missed).
    Set False to fall back to corridor-only checks (faster but unsafe — the
    search will exploit dwell-collisions for fake speedups).
    """
    sched = _plan_labelled(
        list(requests), grid, params,
        t0_global=t0_global,
        static_sites=static_sites,
        verbose=False,
        presort=False,
        enable_swap_pairs=enable_swap_pairs,
    )
    if validate_occupancy(list(requests), sched, params):
        return float("inf"), sched
    if validate_in_flight(sched, params):
        return float("inf"), sched
    if kinematic_check:
        seq_check = _build_seq_for_validation(grid, params,
                                              list(requests), sched)
        kin = validate_kinematic(seq_check, threshold=grid.a,
                                 n_samples=kinematic_n_samples)
        if kin:
            return float("inf"), sched
    return sched.t_end, sched


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

def search_labelled(requests: Sequence[RoutingRequest], grid: Grid,
                    params: Params, *,
                    n_iter: int = 400,
                    T0: Optional[float] = None,
                    cooling: float = 0.992,
                    try_disable_pairs: bool = True,
                    seed: int = 0,
                    static_sites: Optional[Sequence[Site]] = None,
                    verbose: bool = False) -> SearchResult:
    """Stochastic-search labelled scheduler.

    Iterates: perturb processing order (and, with low prob., flip the
    swap-pair toggle); replan via `scheduler.plan(..., presort=False)`;
    accept via simulated annealing on `Schedule.t_end`.
    """
    rng = random.Random(seed)
    requests = list(requests)
    n = len(requests)
    if n == 0:
        sched = _plan_labelled([], grid, params)
        return SearchResult(sched, None, 0, 0, 0.0, 0.0, True)

    order = list(range(n))
    enable_pairs = True
    cur_requests = [requests[k] for k in order]
    cur_t, cur_sched = _evaluate(cur_requests, grid, params,
                                 enable_swap_pairs=enable_pairs,
                                 static_sites=static_sites)
    initial_t = cur_t

    best_t = cur_t
    best_sched = cur_sched
    best_order = order[:]
    best_pairs = enable_pairs

    if T0 is None:
        T0 = max(initial_t * 0.10, 5e-6)
    T = T0
    accept = 0
    history_best = [best_t]
    history_current = [cur_t]

    for it in range(n_iter):
        new_order = _perturb_order(order, rng)
        new_pairs = enable_pairs
        if try_disable_pairs and rng.random() < 0.05:
            new_pairs = not enable_pairs
        new_requests = [requests[k] for k in new_order]
        new_t, new_sched = _evaluate(new_requests, grid, params,
                                     enable_swap_pairs=new_pairs,
                                     static_sites=static_sites)
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

def _build_uncolored_requests(sources, targets, atom_ids, pairs, order):
    labeled = [
        RoutingRequest(atom_id=atom_ids[i], src=sources[i], dst=targets[j])
        for i, j in pairs
    ]
    return [labeled[k] for k in order]


def search_uncolored(req: UncoloredRequest, grid: Grid, params: Params, *,
                     n_iter: int = 400,
                     T0: Optional[float] = None,
                     cooling: float = 0.992,
                     bijection_swap_prob: float = 0.15,
                     try_disable_pairs: bool = True,
                     seed: int = 0,
                     static_sites: Optional[Sequence[Site]] = None,
                     verbose: bool = False) -> SearchResult:
    """Stochastic search for uncolored-pebble scheduling.

    Initial state: Hungarian-optimal bijection from `assign_uncolored`,
    natural processing order, swap-pair planning enabled.

    Search moves:
    - With probability `bijection_swap_prob`, swap two target
      assignments (changes the bijection — gains a few steps of Manhattan
      distance, but may shed bigger corridor congestion).
    - Otherwise perturb the processing order.
    - With probability 0.05, also flip the `enable_swap_pairs` toggle.
    """
    rng = random.Random(seed)

    for s in req.sources:
        if not grid.is_storage_site(*s):
            raise ValueError(f"source {s} not on storage parity site")
    for t in req.targets:
        if not grid.is_storage_site(*t):
            raise ValueError(f"target {t} not on storage parity site")

    initial_asg = assign_uncolored(req.sources, req.targets)
    n = len(req.sources)
    if n == 0:
        sched = _plan_labelled([], grid, params)
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

    cur_requests = _build_uncolored_requests(req.sources, req.targets,
                                             atom_ids, pairs, order)
    cur_t, cur_sched = _evaluate(cur_requests, grid, params,
                                 enable_swap_pairs=enable_pairs,
                                 static_sites=static_sites)
    initial_t = cur_t
    best_t = cur_t
    best_sched = cur_sched
    best_pairs_state = pairs[:]
    best_order = order[:]
    best_swap_pair_flag = enable_pairs

    if T0 is None:
        T0 = max(initial_t * 0.10, 5e-6)
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

        new_requests = _build_uncolored_requests(
            req.sources, req.targets, atom_ids, new_pairs_state, new_order)
        new_t, new_sched = _evaluate(new_requests, grid, params,
                                     enable_swap_pairs=new_swap_flag,
                                     static_sites=static_sites)

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
