"""Pebble-motion scheduler — uncolored / unlabeled assembly variant.

Maps the routing problem to the classical CS pebble-motion problem
(Kornhauser-Miller-Spirakis 1984; see ../pebble_scheduler.md):

    vertices = lattice sites
    pebbles  = atoms
    target   = a SET of occupied sites (no per-atom binding)

Stage A (implemented): Hungarian-assign sources to targets to minimize
total Manhattan distance, then defer to manhattan.plan() for corridor
selection, reservation tables, and a_max-aware timing.

Stages B-D (planned): k-NN greedy matching, cycle-rotation planner
(Yu & Rus 2012), ILP makespan (Yu & LaValle 2015). See md for details.

Output is a regular Schedule from manhattan.py — drop-in compatible with
all downstream consumers (renderers, inspector, Sequence).

Port note: this module is the project-side mirror of
`lib/ripa2/visualization/src/schedulers/pebroute.py`. Logic is unchanged;
only the imports use absolute lib namespace paths so the file can live in
the project's `src/scheduler/` tree without a relative-import chain back
into `lib/ripa2/visualization/src/`.
"""
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from lib.ripa2.visualization.src.atoms import Grid
from lib.ripa2.visualization.src.params import Params
from lib.ripa2.visualization.src.schedulers.manhattan import (
    RoutingRequest,
    Schedule,
    plan as _plan_labelled,
)


Site = Tuple[int, int]


# ---------------- request types ----------------

@dataclass
class UncoloredRequest:
    """A move whose target identity is not bound to a specific source pebble.

    `sources` and `targets` must have the same length. The scheduler will pick
    a bijection minimizing total Manhattan move distance.

    `atom_ids` (optional) lets callers attach stable atom IDs to source
    positions; the chosen bijection then propagates those IDs to the matched
    targets. If omitted, atoms are auto-numbered 0..m-1 in source order.
    """
    sources: List[Site]
    targets: List[Site]
    atom_ids: Optional[List[int]] = None
    deadline: Optional[float] = None


@dataclass
class Assignment:
    """Result of the bipartite-matching step (Stage A)."""
    pairs: List[Tuple[int, int]]              # (src_idx, tgt_idx) under the bijection
    total_cost_sites: float                   # sum of Manhattan distances in lattice steps
    cost_matrix: np.ndarray = field(repr=False)


# ---------------- core helpers ----------------

def _manhattan_sites(a: Site, b: Site) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def assign_uncolored(sources: Sequence[Site],
                     targets: Sequence[Site]) -> Assignment:
    """Min-total-Manhattan-distance bijection sources -> targets via Hungarian.

    Cost matrix C[i, j] = |s_i - t_j|_1 in lattice steps. scipy's
    linear_sum_assignment runs in O(m^3); fine for m <= ~hundreds.
    """
    if len(sources) != len(targets):
        raise ValueError(
            f"uncolored matching requires |sources| == |targets|, "
            f"got {len(sources)} and {len(targets)}"
        )
    m = len(sources)
    if m == 0:
        return Assignment(pairs=[], total_cost_sites=0.0,
                          cost_matrix=np.zeros((0, 0)))
    cost = np.empty((m, m), dtype=float)
    for i, s in enumerate(sources):
        for j, t in enumerate(targets):
            cost[i, j] = _manhattan_sites(s, t)
    row_ind, col_ind = linear_sum_assignment(cost)
    pairs = list(zip(row_ind.tolist(), col_ind.tolist()))
    total = float(cost[row_ind, col_ind].sum())
    return Assignment(pairs=pairs, total_cost_sites=total, cost_matrix=cost)


# ---------------- scheduler entry points ----------------

def plan_uncolored(req: UncoloredRequest,
                   grid: Grid,
                   params: Params,
                   t0_global: float = 0.0,
                   static_sites: Optional[Sequence[Site]] = None,
                   verbose: bool = False) -> Tuple[Schedule, Assignment]:
    """Stage A: Hungarian-matched uncolored pebble scheduling.

    Returns (schedule, assignment). The schedule is a regular
    `scheduler.Schedule` — pass straight to renderers / Sequence.
    The assignment is exposed so callers can audit which atom went where.
    """
    for s in req.sources:
        if not grid.is_storage_site(*s):
            raise ValueError(f"source {s} is not on a storage (even, even) site")
    for t in req.targets:
        if not grid.is_storage_site(*t):
            raise ValueError(f"target {t} is not on a storage (even, even) site")

    assignment = assign_uncolored(req.sources, req.targets)

    if req.atom_ids is None:
        atom_ids = list(range(len(req.sources)))
    else:
        if len(req.atom_ids) != len(req.sources):
            raise ValueError("atom_ids length must match sources")
        atom_ids = list(req.atom_ids)

    requests = [
        RoutingRequest(
            atom_id=atom_ids[i],
            src=req.sources[i],
            dst=req.targets[j],
            deadline=req.deadline,
        )
        for i, j in assignment.pairs
    ]

    if verbose:
        print(f"plan_uncolored: matched {len(requests)} pebbles, "
              f"total Manhattan cost = {assignment.total_cost_sites:.1f} sites")
        for i, j in assignment.pairs:
            print(f"  atom {atom_ids[i]}: {req.sources[i]} -> {req.targets[j]}  "
                  f"(d = {assignment.cost_matrix[i, j]:.0f})")

    schedule = _plan_labelled(
        requests, grid, params,
        t0_global=t0_global,
        static_sites=static_sites,
        verbose=verbose,
    )
    return schedule, assignment


def plan_labelled_pebble(requests: Sequence[RoutingRequest],
                         grid: Grid,
                         params: Params,
                         t0_global: float = 0.0,
                         static_sites: Optional[Sequence[Site]] = None,
                         verbose: bool = False) -> Schedule:
    """Stage A passthrough for the labelled case.

    Provided so downstream code can route every scheduling call through
    pebble_scheduler.* without caring whether the scene is labelled or
    uncolored. Behaves identically to scheduler.plan().
    """
    return _plan_labelled(
        list(requests), grid, params,
        t0_global=t0_global,
        static_sites=static_sites,
        verbose=verbose,
    )


# ---------------- stage stubs (planned) ----------------

def plan_uncolored_greedy(req: UncoloredRequest, grid: Grid, params: Params,
                          **kwargs) -> Tuple[Schedule, Assignment]:
    """Stage B (planned): k-NN greedy matching as a cheaper alternative to
    Hungarian for very large m. Not implemented yet.
    """
    raise NotImplementedError(
        "Stage B (k-NN greedy uncolored matching) not implemented; "
        "use plan_uncolored() for now."
    )


def plan_cycle_rotation(req: UncoloredRequest, grid: Grid, params: Params,
                        **kwargs) -> Tuple[Schedule, Assignment]:
    """Stage C (planned): detect cycles in the source->target permutation and
    rotate them through a buffered corridor cell (Yu & Rus 2012).
    Not implemented yet.
    """
    raise NotImplementedError(
        "Stage C (cycle-rotation planner) not implemented; "
        "use plan_uncolored() for now."
    )


def plan_makespan_ilp(req: UncoloredRequest, grid: Grid, params: Params,
                      **kwargs) -> Tuple[Schedule, Assignment]:
    """Stage D (planned): time-optimal ILP / network-flow scheduling
    (Yu & LaValle 2015). Heavy; intended as a benchmark oracle.
    Not implemented yet.
    """
    raise NotImplementedError(
        "Stage D (ILP makespan) not implemented; reference oracle only."
    )
