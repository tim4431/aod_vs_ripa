"""Pebble-motion scheduler — uncolored / unlabeled assembly variant.

Maps the routing problem to the classical CS pebble-motion problem
(Kornhauser-Miller-Spirakis 1984): vertices = lattice sites; pebbles =
atoms; target = a SET of occupied sites with no per-atom binding.

Stage A (implemented): Hungarian-assign sources to targets to minimise
total Manhattan distance, then defer to the corridor planner in
`_manhattan_planner.plan_labelled` for corridor selection, reservation
tables, and a_max-aware timing.

Stages B-D (planned): k-NN greedy matching, cycle-rotation planner
(Yu & Rus 2012), ILP makespan (Yu & LaValle 2015).

Public API:

* function form — `plan_uncolored`, `plan_labelled_pebble`, `assign_uncolored`
  (used by the lib-side rendering scenes).
* class form — `RIPAPebRouteScheduler` (a `base.Scheduler` subclass that
  plugs into `src.benchmark.benchmark_schedulers` and the existing
  Scheduler framework).

Output is a `MovingSequence`, drop-in compatible with every src-side
validator and visualiser.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from ..atom_config import Grid
from ..moving_sequence import MovingSequence
from ..routing import Site
from ._manhattan_planner import (
    RoutingMove,
    T_HANDOFF_DEFAULT,
    plan_labelled,
)
from .base import Scheduler


# ---------------- request types ----------------


@dataclass
class UncoloredRequest:
    """A move whose target identity is not bound to a specific source pebble.

    `sources` and `targets` must have the same length. The scheduler picks a
    bijection minimising total Manhattan move distance.

    `atom_ids` (optional) lets callers attach stable atom IDs to source
    positions; the chosen bijection then propagates those IDs to the
    matched targets. If omitted, atoms are auto-numbered 0..m-1 in source
    order.
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
    `linear_sum_assignment` runs in O(m^3); fine for m <= ~hundreds.
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


def plan_uncolored(req: UncoloredRequest, grid: Grid,
                   *,
                   t_handoff: float = T_HANDOFF_DEFAULT,
                   static_sites: Optional[Sequence[Site]] = None,
                   collision_dt: float = 1e-6,
                   verbose: bool = False) -> Tuple[MovingSequence, Assignment]:
    """Stage A: Hungarian-matched uncolored pebble scheduling.

    Returns ``(sequence, assignment)``. The bijection-aware planner
    treats any currently-occupied site as an obstacle, so storage layouts
    other than the lib's even/even convention work without extra wiring.
    """
    assignment = assign_uncolored(req.sources, req.targets)

    if req.atom_ids is None:
        atom_ids = list(range(len(req.sources)))
    else:
        if len(req.atom_ids) != len(req.sources):
            raise ValueError("atom_ids length must match sources")
        atom_ids = list(req.atom_ids)

    moves = [
        RoutingMove(
            atom_id=atom_ids[i],
            src=req.sources[i],
            dst=req.targets[j],
            deadline=req.deadline,
        )
        for i, j in assignment.pairs
    ]

    if verbose:
        print(f"plan_uncolored: matched {len(moves)} pebbles, "
              f"total Manhattan cost = {assignment.total_cost_sites:.1f} sites")
        for i, j in assignment.pairs:
            print(f"  atom {atom_ids[i]}: {req.sources[i]} -> {req.targets[j]}  "
                  f"(d = {assignment.cost_matrix[i, j]:.0f})")

    sequence = plan_labelled(
        moves, grid,
        t_handoff=t_handoff,
        static_sites=static_sites,
        collision_dt=collision_dt,
        verbose=verbose,
    )
    return sequence, assignment


def plan_labelled_pebble(moves: Sequence[RoutingMove], grid: Grid,
                         *,
                         t_handoff: float = T_HANDOFF_DEFAULT,
                         static_sites: Optional[Sequence[Site]] = None,
                         collision_dt: float = 1e-6,
                         verbose: bool = False) -> MovingSequence:
    """Pebble-flavoured wrapper around `plan_labelled` — provided so callers
    can route through this module regardless of labelled/uncolored framing.
    """
    return plan_labelled(
        list(moves), grid,
        t_handoff=t_handoff,
        static_sites=static_sites,
        collision_dt=collision_dt,
        verbose=verbose,
    )






# ---------------- Scheduler-class wrapper ----------------


@dataclass
class RIPAPebRouteScheduler(Scheduler):
    """`base.Scheduler` adapter for the Hungarian + Manhattan-corridor planner.

    Handles both labeled and unlabeled `RoutingRequest`s. For the unlabeled
    case it picks the Hungarian-optimal source→target bijection first, then
    delegates to the same corridor planner used by the labeled case.

    Use this class when you want the scheduler to plug into
    `src.benchmark.benchmark_schedulers` or any other framework that expects
    a `Scheduler.plan()` method returning a `MovingSequence`. For raw
    function-form access, call `plan_uncolored` / `plan_labelled_pebble`
    directly — they bypass the base class entirely.
    """

    t_handoff: float = T_HANDOFF_DEFAULT
    static_sites: Optional[Sequence[Site]] = None

    def _plan(self) -> None:
        # Build the per-atom move list. For unlabeled requests we pick the
        # Hungarian-optimal bijection first, then route as if labeled.
        if self.request.labeled:
            moves = [
                RoutingMove(atom_id=k,
                            src=tuple(self.request.src[k]),
                            dst=tuple(self.request.dst[k]))
                for k in range(len(self.request.src))
            ]
        else:
            assignment = assign_uncolored(self.request.src, self.request.dst)
            moves = [
                RoutingMove(atom_id=i,
                            src=tuple(self.request.src[i]),
                            dst=tuple(self.request.dst[j]))
                for (i, j) in assignment.pairs
            ]

        # Plan in place onto the sequence the base class already built from
        # `request.initial`; `self.sequence` ends up holding the result.
        plan_labelled(
            moves, self.request.grid,
            t_handoff=self.t_handoff,
            static_sites=self.static_sites,
            collision_dt=self.collision_dt,
            sequence=self.sequence,
        )
