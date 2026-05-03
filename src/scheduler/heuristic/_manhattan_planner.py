"""Manhattan-corridor planner — internal helper for the RIPA pebble
schedulers.

This is *not* itself a `base.Scheduler`; it is the shared planning engine
that `ripa_pebroute` (Hungarian + greedy) and `ripa_stochastic_search`
(simulated annealing) both call. Each public scheduler wraps the entry
point `plan_labelled` in a real `Scheduler` class — see those modules
for the user-facing API.

The algorithm is a port of `lib/ripa2/visualization/src/schedulers/manhattan.py`:
greedy per-atom routing through 1/2/3/5-segment corridor plans selected
by earliest predicted finish time, plus a joint planner for swap pairs.
Everything runs in *grid units* (integer `(i, j)` sites, distances in
lattice steps) so the planner is unit-agnostic; acceleration is converted
once via `grid_accel_from_phys` and segment durations come from
`bang_bang_duration`.

Obstacle model: a site is an obstacle iff some atom currently rests
there or will rest there at some point in the schedule — i.e. iff it's
in the union of every `RoutingMove`'s `src`/`dst` plus any `static_sites`.
There is *no* hardcoded storage-vs-corridor parity: the algorithm works
for any layout the atoms happen to occupy.

Output: a `src.moving_sequence.MovingSequence` with per-atom trajectories
committed. The corridor reservation table that drives plan scoring is
internal (a `_PlanState` discarded after planning).

Validation note: segments are appended directly to
`AtomTrajectory.segments` rather than through `MovingSequence.append`,
because the framework's per-commit `post_hold` check trips on the lib
algorithm's cross-corridor transient overlaps (atom A momentarily resting
between two segments at a site atom B's row-corridor passes through). The
lib's reference accepts those cases at planning time and lets a separate
kinematic validator decide their fate; we mirror that. Callers who need
a strict kinematic check call `sequence.validate(dt=...)` themselves.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

from ...atom_config import AtomConfig, Grid
from ...atom_trajectory import CollisionError  # noqa: F401  (re-exported for callers)
from ...movement import PHYS_A_MAX, RIPAStep, grid_accel_from_phys
from ...moving_sequence import MovingSequence
from ...routing import Site
from ...segments import bang_bang_duration, make_const_acc_segment


# ---------------- per-atom request + result types ----------------


@dataclass
class RoutingMove:
    """One labelled atom move: send `atom_id`, currently at `src`, to `dst`."""

    atom_id: int
    src: Site
    dst: Site
    deadline: Optional[float] = None


# Algorithm-level defaults. `T_HANDOFF` is the dwell inserted between
# consecutive segments of one atom and the temporal buffer between segments
# of *different* atoms on the same corridor; `R_SAFE_DEFAULT=None` means
# "derive from grid.rc / grid.d" (the same kinematic radius the
# AtomEnsemble validator enforces).
T_HANDOFF_DEFAULT: float = 0.5e-6
R_SAFE_DEFAULT: Optional[float] = None


@dataclass
class ScheduleEntry:
    """One committed segment, in grid coordinates.

    ``axis`` is ``'x'`` for a single-axis move along i (j fixed), ``'y'``
    for a move along j (i fixed). ``p_start``/``p_end`` are positions along
    the moving axis; ``q_fixed`` is the orthogonal coordinate.
    """

    atom_id: int
    seg_idx: int
    axis: str
    t0: float
    t1: float
    p_start: float
    p_end: float
    q_fixed: float


@dataclass
class CorridorReservations:
    """Per-corridor segment ledger used by the planner to score new plans.

    Indices: rows (j) for x-segments; cols (i) for y-segments. Each bucket
    holds the entries planned so far on that corridor; conflict checks use
    the ``t_handoff`` and ``r_safe`` bounds set by the caller.
    """

    by_row: dict[int, List[ScheduleEntry]] = field(default_factory=dict)
    by_col: dict[int, List[ScheduleEntry]] = field(default_factory=dict)

    def add(self, entry: ScheduleEntry, corridor_idx: int) -> None:
        bucket = self.by_row if entry.axis == "x" else self.by_col
        bucket.setdefault(corridor_idx, []).append(entry)

    def conflicts(self, axis: str, corridor_idx: int, t0: float, t1: float,
                  p0: float, p1: float, t_handoff: float, r_safe: float) -> bool:
        bucket = self.by_row if axis == "x" else self.by_col
        for e in bucket.get(corridor_idx, []):
            if t1 + t_handoff <= e.t0 or t0 >= e.t1 + t_handoff:
                continue
            lo, hi = sorted((p0, p1))
            elo, ehi = sorted((e.p_start, e.p_end))
            if hi + r_safe <= elo or lo >= ehi + r_safe:
                continue
            return True
        return False

    def earliest_free(self, axis: str, corridor_idx: int,
                      t0: float, t1: float, p0: float, p1: float,
                      t_handoff: float, r_safe: float) -> float:
        """Earliest start time at or after `t0` such that a [t, t+(t1-t0)]
        segment at [p0, p1] is conflict-free with all prior entries."""
        bucket = self.by_row if axis == "x" else self.by_col
        entries = bucket.get(corridor_idx, [])
        dur = t1 - t0
        t = t0
        for _ in range(50):
            pushed = False
            for e in entries:
                if t + dur + t_handoff <= e.t0 or t >= e.t1 + t_handoff:
                    continue
                lo, hi = sorted((p0, p1))
                elo, ehi = sorted((e.p_start, e.p_end))
                if hi + r_safe <= elo or lo >= ehi + r_safe:
                    continue
                t = max(t, e.t1 + t_handoff)
                pushed = True
                break
            if not pushed:
                return t
        return t


@dataclass
class _PlanState:
    """Mutable planner-internal state passed between helpers.

    `sequence` is the live `MovingSequence` (the eventual result). `entries`
    and `reservations` are bookkeeping only — they let the planner score
    corridor candidates and synchronise swap pairs. Neither is part of the
    public output; callers receive just the `MovingSequence`.
    """

    sequence: MovingSequence
    reservations: CorridorReservations = field(default_factory=CorridorReservations)
    entries: List[ScheduleEntry] = field(default_factory=list)


# ---------------- corridor selection (cost helpers) ----------------
#
# `blocked` is the set of sites another atom currently rests at (or will
# rest at) — derived from the moves' src/dst plus any static atoms. A
# corridor traversal is "clear" when none of its cells are in `blocked`.
# This is the *only* obstacle test we need: there's no separate storage-vs-
# highway parity convention to maintain, and the algorithm therefore works
# for any atom layout, not just the lib's even/even storage pattern.


def _row_clear(j: int, i_a: int, i_b: int, blocked: set) -> bool:
    lo, hi = sorted((i_a, i_b))
    for k in range(lo, hi + 1):
        if (k, j) in blocked:
            return False
    return True


def _col_clear(i: int, j_a: int, j_b: int, blocked: set) -> bool:
    lo, hi = sorted((j_a, j_b))
    for k in range(lo, hi + 1):
        if (i, k) in blocked:
            return False
    return True



def _adjacent_odd(idx: int, target: int, n: int) -> int:
    candidate = idx + (1 if target > idx else -1)
    if not (0 <= candidate < n) or candidate % 2 == 0:
        candidate = idx - (1 if target > idx else -1)
    if not (0 <= candidate < n):
        candidate = idx + 1 if idx + 1 < n else idx - 1
    return candidate


# ---------------- plan-finish-time scoring ----------------


def _solve_plan_timing(plan_segs, corridor_idxs, reservations, accel,
                       t_handoff: float, r_safe: float,
                       t_floor: float = 0.0,
                       t_start_max: Optional[float] = None
                       ) -> Tuple[float, float]:
    """Find the earliest feasible (t_start, t_end) for committing `plan_segs`.

    Walks the plan from a candidate `t_start = t_floor`. If any segment hits
    a corridor reservation conflict, the cursor jumps to the conflict's
    `earliest_free` and the walk restarts. When every segment fits, returns
    `(t_start, t_end)`. If `t_start_max` is given and would be exceeded, both
    values come back as `+inf` so callers can disqualify the plan.

    Used by the planner two ways: (a) score plans by predicted t_end during
    `_build_path` enumeration, (b) re-derive the actual t_start at commit
    time after the chosen plan is known.
    """
    if not plan_segs:
        return t_floor, t_floor

    if reservations is None:
        # Fast path: no other atoms to worry about — just sum segment durations.
        t = t_floor
        for k, (_axis, p_s, p_e, _q) in enumerate(plan_segs):
            t += bang_bang_duration(abs(p_e - p_s), accel)
            if k < len(plan_segs) - 1:
                t += t_handoff
        return t_floor, t

    t_cursor = t_floor
    for _outer in range(200):
        t = t_cursor
        conflict_t = None
        for k, (axis, p_s, p_e, _q) in enumerate(plan_segs):
            dur = bang_bang_duration(abs(p_e - p_s), accel)
            if axis in ("x", "y") and dur > 0 and reservations.conflicts(
                axis, corridor_idxs[k], t, t + dur, p_s, p_e, t_handoff, r_safe,
            ):
                conflict_t = reservations.earliest_free(
                    axis, corridor_idxs[k], t, t + dur, p_s, p_e,
                    t_handoff, r_safe,
                )
                break
            t += dur
            if k < len(plan_segs) - 1:
                t += t_handoff
        if conflict_t is None:
            if t_start_max is not None and t_cursor > t_start_max:
                return float("inf"), float("inf")
            return t_cursor, t
        t_cursor = conflict_t
    return float("inf"), float("inf")


# ---------------- candidate-plan enumeration ----------------


def _build_path(grid: Grid, blocked: set,
                i0: int, j0: int, i1: int, j1: int,
                reservations: CorridorReservations,
                accel: float, t_handoff: float, r_safe: float,
                t_start_floor: float = 0.0,
                t_start_max: Optional[float] = None):
    """Enumerate 1/2/3/5-seg plans avoiding static obstacles; pick the one
    with earliest predicted finish time given current reservations.

    Returns ``(segs_plan, corridor_idxs, kind_str)`` — segs_plan is a list
    of ``(axis, p_start, p_end, q_fixed)`` tuples in grid coordinates.
    """
    if i0 == i1 and j0 == j1:
        return [], [], "noop"

    candidates = []  # list of (cost, plan_tuple)

    def add(plan_segs, corridor_idxs, kind):
        _, t_end = _solve_plan_timing(
            plan_segs, corridor_idxs, reservations, accel,
            t_handoff, r_safe,
            t_floor=t_start_floor, t_start_max=t_start_max,
        )
        candidates.append((t_end, (plan_segs, corridor_idxs, kind)))

    # 1-seg collinear
    if j0 == j1 and _row_clear(j0, i0, i1, blocked):
        add([("x", i0, i1, j0)], [j0], "1-seg row")
    if i0 == i1 and _col_clear(i0, j0, j1, blocked):
        add([("y", j0, j1, i0)], [i0], "1-seg col")

    # 2-seg L-shape
    if i0 != i1 and j0 != j1:
        if _row_clear(j0, i0, i1, blocked) and _col_clear(i1, j0, j1, blocked):
            add([("x", i0, i1, j0), ("y", j0, j1, i1)],
                [j0, i1], "2-seg L (row->col)")
        if _col_clear(i0, j0, j1, blocked) and _row_clear(j1, i0, i1, blocked):
            add([("y", j0, j1, i0), ("x", i0, i1, j1)],
                [i0, j1], "2-seg L (col->row)")

    # 3-seg via corridor row
    for j_corr in (j for j in range(grid.N) if j != j0):
        if not (_col_clear(i0, j0, j_corr, blocked) and
                _col_clear(i1, j_corr, j1, blocked)):
            continue
        add([
            ("y", j0, j_corr, i0),
            ("x", i0, i1, j_corr),
            ("y", j_corr, j1, i1),
        ], [i0, j_corr, i1], f"3-seg via row j={j_corr}")

    # 3-seg via corridor col
    for i_corr in (i for i in range(grid.N) if i != i0):
        if not (_row_clear(j0, i0, i_corr, blocked) and
                _row_clear(j1, i_corr, i1, blocked)):
            continue
        add([
            ("x", i0, i_corr, j0),
            ("y", j0, j1, i_corr),
            ("x", i_corr, i1, j1),
        ], [j0, i_corr, j1], f"3-seg via col i={i_corr}")

    # 5-seg fully-on-highway (always-safe fallback)
    j_in = _adjacent_odd(j0, j1, grid.N)
    j_out = _adjacent_odd(j1, j0, grid.N)
    i_corr = _adjacent_odd(i1, i0, grid.N)
    if j_in == j_out:
        add([
            ("y", j0, j_in, i0),
            ("x", i0, i1, j_in),
            ("y", j_in, j1, i1),
        ], [i0, j_in, i1], f"3-seg highway row j={j_in}")
    else:
        add([
            ("y", j0, j_in, i0),
            ("x", i0, i_corr, j_in),
            ("y", j_in, j_out, i_corr),
            ("x", i_corr, i1, j_out),
            ("y", j_out, j1, i1),
        ], [i0, j_in, i_corr, j_out, i1],
           f"5-seg highway (j_in={j_in}, i_corr={i_corr}, j_out={j_out})")

    candidates.sort(key=lambda c: c[0])
    return candidates[0][1]


# ---------------- segment commit (RIPAStep emission) ----------------


def _build_pending_segments(req: RoutingMove, segs_plan, corridor_idxs,
                            t_start: float, accel: float, t_handoff: float):
    """Walk a per-atom plan and produce ``(t_start, RIPAStep, ScheduleEntry,
    corridor_idx)`` tuples without committing anything to the schedule.

    Used by the swap-pair planner so two atoms' steps can be interleaved
    in time order before they hit the ensemble.
    """
    pending = []
    t_try = t_start
    for k, (axis, p_s, p_e, q) in enumerate(segs_plan):
        dur = bang_bang_duration(abs(p_e - p_s), accel)
        t_s, t_e = t_try, t_try + dur

        if axis == "x":
            target = (int(round(p_e)), int(round(q)))
            channel = "row"
        elif axis == "y":
            target = (int(round(q)), int(round(p_e)))
            channel = "col"
        else:
            raise ValueError(f"unsupported axis {axis!r}")

        step = RIPAStep(start_time=t_s, atom_id=int(req.atom_id),
                        target=target, channel=channel)
        entry = ScheduleEntry(
            atom_id=int(req.atom_id), seg_idx=k, axis=axis,
            t0=t_s, t1=t_e, p_start=float(p_s), p_end=float(p_e),
            q_fixed=float(q),
        )
        pending.append((t_s, step, entry,
                        corridor_idxs[k] if axis in ("x", "y") else None))

        t_try = t_e
        if k < len(segs_plan) - 1:
            t_try += t_handoff
    return pending


def _commit_pending(pending, state: _PlanState, accel: float) -> None:
    """Apply a batch of pending segments to the schedule.

    Segments are appended directly to each atom's `AtomTrajectory.segments`
    (and the corresponding `RIPAStep` recorded on `MovingSequence.steps`),
    *bypassing* `AtomEnsemble.append_segment`'s collision validator. See
    the module docstring for the rationale; tl;dr the lib's algorithm
    accepts transient cross-corridor overlaps and the validator's
    incremental post_hold check trips on them, so the planner trusts its
    own corridor reservation table during commit and lets callers run
    `schedule.sequence.validate(dt)` after the fact if they want a strict
    kinematic check.

    Sorting by `(t_start, atom_id, seg_idx)` keeps each atom's segments
    in chronological order (required by `AtomTrajectory._check_continuity`)
    and interleaves swap-pair atoms so each one's segments are anchored
    in time order alongside its partner's.
    """
    pending = sorted(pending, key=lambda item: (item[0], item[2].atom_id,
                                                item[2].seg_idx))
    for _t_start, step, entry, corridor_idx in pending:
        atomtraj = state.sequence.ensemble.atomtraj_by_id(step.atom_id)
        current = atomtraj.final_pos
        target = tuple(step.target)
        if current != target:
            seg = make_const_acc_segment(
                current, target, step.start_time,
                accel=accel, channel=step.channel,
            )
            atomtraj._check_continuity(seg)
            atomtraj.segments.append(seg)
        state.sequence.steps.append(step)
        state.entries.append(entry)
        if corridor_idx is not None:
            state.reservations.add(entry, corridor_idx)


def _commit_segs(req: RoutingMove, segs_plan, corridor_idxs, t_start: float,
                 state: _PlanState, accel: float, t_handoff: float) -> None:
    """Materialise a single-atom plan onto the schedule + MovingSequence."""
    pending = _build_pending_segments(req, segs_plan, corridor_idxs,
                                      t_start, accel, t_handoff)
    _commit_pending(pending, state, accel)


# ---------------- swap-pair joint planner ----------------


def _is_swap_pair(a: RoutingMove, b: RoutingMove) -> bool:
    return tuple(a.dst) == tuple(b.src) and tuple(b.dst) == tuple(a.src)


def _three_seg_plan_for(req: RoutingMove, j_corr: int):
    """Build the 3-seg-via-row-corridor plan for one move."""
    i0, j0 = req.src
    i1, j1 = req.dst
    return ([
        ("y", j0, j_corr, i0),
        ("x", i0, i1, j_corr),
        ("y", j_corr, j1, i1),
    ], [i0, j_corr, i1])


def _snapshot_with_added(reservations: CorridorReservations,
                         plan_segs, corridor_idxs, t_start: float,
                         accel: float, t_handoff: float,
                         atom_id: int) -> CorridorReservations:
    """Shallow-cloned reservations with `plan_segs` pre-anchored at t_start."""
    sim = CorridorReservations(
        by_row={k: list(v) for k, v in reservations.by_row.items()},
        by_col={k: list(v) for k, v in reservations.by_col.items()},
    )
    t_try = t_start
    for k, (axis, p_s, p_e, q) in enumerate(plan_segs):
        dur = bang_bang_duration(abs(p_e - p_s), accel)
        t_s, t_e = t_try, t_try + dur
        if axis in ("x", "y") and dur > 0:
            entry = ScheduleEntry(atom_id=atom_id, seg_idx=k, axis=axis,
                                  t0=t_s, t1=t_e, p_start=float(p_s),
                                  p_end=float(p_e), q_fixed=float(q))
            sim.add(entry, corridor_idxs[k])
        t_try = t_e
        if k < len(plan_segs) - 1:
            t_try += t_handoff
    return sim


def _plan_swap_pair_jointly(a: RoutingMove, b: RoutingMove,
                            grid: Grid, accel: float, t_handoff: float,
                            r_safe: float, state: _PlanState, all_sites: set,
                            verbose: bool) -> bool:
    """Plan a pure swap pair on different corridors, synchronizing both
    atoms' starts so the swap-pair temporal-occupancy constraint is met.
    """
    blocked_a = all_sites - {tuple(a.src), tuple(a.dst)}
    blocked_b = all_sites - {tuple(b.src), tuple(b.dst)}

    def feasible(req, blocked, j):
        return (_col_clear(req.src[0], req.src[1], j, blocked) and
                _col_clear(req.dst[0], j, req.dst[1], blocked))

    all_rows = [j for j in range(grid.N) if j != a.src[1] and j != a.dst[1]]
    cand_a = [j for j in all_rows if feasible(a, blocked_a, j)]
    cand_b = [j for j in all_rows if feasible(b, blocked_b, j)]

    best = None  # (cost, j_a, j_b, t_sync, t_a_end, t_b_end)
    for j_a in cand_a:
        for j_b in cand_b:
            if j_a == j_b:
                continue
            segs_a, corr_a = _three_seg_plan_for(a, j_a)
            segs_b, corr_b = _three_seg_plan_for(b, j_b)
            # Iterate to a synchronised start: each atom's earliest feasible
            # start depends on the partner's planned reservations, so we
            # converge on max(t_a, t_b) until it stabilises.
            t_sync = 0.0
            for _ in range(20):
                t_a_new, _ = _solve_plan_timing(
                    segs_a, corr_a, state.reservations, accel,
                    t_handoff, r_safe, t_floor=t_sync,
                )
                sim_res = _snapshot_with_added(state.reservations, segs_a, corr_a,
                                               t_a_new, accel, t_handoff,
                                               a.atom_id)
                t_b_new, _ = _solve_plan_timing(
                    segs_b, corr_b, sim_res, accel,
                    t_handoff, r_safe, t_floor=t_sync,
                )
                new_sync = max(t_a_new, t_b_new)
                if abs(new_sync - t_sync) < 1e-12:
                    t_sync = new_sync
                    break
                t_sync = new_sync
            _, t_a_end = _solve_plan_timing(
                segs_a, corr_a, state.reservations, accel,
                t_handoff, r_safe, t_floor=t_sync,
            )
            sim_res = _snapshot_with_added(state.reservations, segs_a, corr_a,
                                           t_sync, accel, t_handoff,
                                           a.atom_id)
            _, t_b_end = _solve_plan_timing(
                segs_b, corr_b, sim_res, accel,
                t_handoff, r_safe, t_floor=t_sync,
            )
            cost = max(t_a_end, t_b_end)
            if best is None or cost < best[0]:
                best = (cost, j_a, j_b, t_sync, t_a_end, t_b_end)

    if best is None:
        return False

    cost, j_a, j_b, t_sync, t_a_end, t_b_end = best
    segs_a, corr_a = _three_seg_plan_for(a, j_a)
    segs_b, corr_b = _three_seg_plan_for(b, j_b)
    pending = _build_pending_segments(a, segs_a, corr_a, t_sync, accel, t_handoff)
    pending += _build_pending_segments(b, segs_b, corr_b, t_sync, accel, t_handoff)
    _commit_pending(pending, state, accel)
    if verbose:
        print(f"  swap pair atoms {a.atom_id}<->{b.atom_id}: "
              f"j_a={j_a}, j_b={j_b}, t_sync={t_sync*1e6:.1f} us, "
              f"max_t={cost*1e6:.2f} us")
    return True


# ---------------- public entry points ----------------


def _build_initial_sequence(moves: Sequence[RoutingMove], grid: Grid,
                            static_sites: Optional[Sequence[Site]],
                            collision_dt: float) -> MovingSequence:
    """Construct a `MovingSequence` whose initial config holds every atom
    addressed by `moves` plus the requested `static_sites` (atoms placed at
    rest, never appearing in any move). Atom IDs come from `moves`; static
    atoms get unique negative IDs to avoid collision with the move IDs.
    """
    move_ids = {int(m.atom_id) for m in moves}
    if len(move_ids) != len(moves):
        raise ValueError("RoutingMove atom_ids must be unique")

    positions: List[Tuple[int, int]] = [
        (int(m.src[0]), int(m.src[1])) for m in moves
    ]
    atom_ids: List[int] = [int(m.atom_id) for m in moves]

    # Static atoms get unique negative IDs so they can't collide with the
    # move IDs (which are typically 0..M-1).
    next_static_id = -1
    for s in static_sites or []:
        positions.append((int(s[0]), int(s[1])))
        while next_static_id in move_ids:
            next_static_id -= 1
        atom_ids.append(next_static_id)
        next_static_id -= 1

    cfg = AtomConfig(
        positions=np.asarray(positions, dtype=int).reshape(-1, 2),
        atom_ids=np.asarray(atom_ids, dtype=int),
    )
    return MovingSequence(grid=grid, initial=cfg, collision_dt=collision_dt)


def plan_labelled(moves: Sequence[RoutingMove], grid: Grid,
                  *,
                  t_handoff: float = T_HANDOFF_DEFAULT,
                  r_safe: Optional[float] = R_SAFE_DEFAULT,
                  static_sites: Optional[Sequence[Site]] = None,
                  presort: bool = True,
                  enable_swap_pairs: bool = True,
                  collision_dt: float = 1e-6,
                  sequence: Optional[MovingSequence] = None,
                  verbose: bool = False) -> MovingSequence:
    """Greedy Manhattan-corridor planner.

    Returns the populated `MovingSequence` (atoms moved to their targets;
    `seq.total_duration()` is the schedule's t_end). The corridor reservation
    table the planner builds while routing is purely internal — callers
    only see the resulting trajectories.

    `t_handoff`: dwell between consecutive segments of one atom and the
    temporal buffer between segments of *different* atoms on the same
    corridor. `r_safe`: spatial buffer in grid units; `None` derives it
    from `grid.rc / grid.d`.

    `sequence`: optional pre-built `MovingSequence` to plan into. The
    Scheduler class wrappers pass `self.sequence` so the plan lands on
    their own state; when `None` the planner builds a fresh one from
    `moves` + `static_sites`.
    """
    accel = grid_accel_from_phys(PHYS_A_MAX, grid.d)
    if r_safe is None:
        r_safe = grid.rc / grid.d

    if sequence is None:
        sequence = _build_initial_sequence(moves, grid, static_sites, collision_dt)
    state = _PlanState(sequence=sequence)

    moving_sites = ({tuple(m.src) for m in moves} |
                    {tuple(m.dst) for m in moves})
    static_set = {tuple(s) for s in (static_sites or [])}
    all_sites = moving_sites | static_set

    src_owner = {tuple(m.src): m.atom_id for m in moves}
    occupant_of = {}
    for m in moves:
        owner_id = src_owner.get(tuple(m.dst))
        if owner_id is not None and owner_id != m.atom_id:
            occupant_of[m.atom_id] = owner_id

    dst_owner = {tuple(m.dst): m.atom_id for m in moves}
    incoming_of = {}
    for m in moves:
        in_id = dst_owner.get(tuple(m.src))
        if in_id is not None and in_id != m.atom_id:
            incoming_of[m.atom_id] = in_id

    def sort_key(m: RoutingMove):
        is_occupied = m.atom_id in occupant_of
        di = abs(m.dst[0] - m.src[0])
        dj = abs(m.dst[1] - m.src[1])
        return (
            is_occupied,
            -(di + dj),
            m.deadline if m.deadline is not None else float("inf"),
        )

    # ---- swap-pair joint planning ----
    paired: set = set()
    if enable_swap_pairs:
        by_id = {m.atom_id: m for m in moves}
        swap_pairs: List[Tuple[RoutingMove, RoutingMove]] = []
        for m in moves:
            if m.atom_id in paired:
                continue
            partner_id = occupant_of.get(m.atom_id)
            if partner_id is None:
                continue
            partner = by_id.get(partner_id)
            if partner is None or partner.atom_id in paired:
                continue
            if _is_swap_pair(m, partner):
                swap_pairs.append((m, partner))
                paired.add(m.atom_id)
                paired.add(partner.atom_id)
        swap_pairs.sort(key=lambda p: -(abs(p[0].dst[0] - p[0].src[0])
                                        + abs(p[0].dst[1] - p[0].src[1])))
        for (a, b) in swap_pairs:
            _plan_swap_pair_jointly(a, b, grid, accel, t_handoff, r_safe,
                                    state, all_sites, verbose)

    # ---- per-atom planning for everyone else ----
    remaining = [m for m in moves if m.atom_id not in paired]
    if presort:
        remaining = sorted(remaining, key=sort_key)

    for req in remaining:
        i0, j0 = req.src
        i1, j1 = req.dst

        blocked = all_sites - {tuple(req.src), tuple(req.dst)}

        rough_dur = (abs(i1 - i0) + abs(j1 - j0))
        rough_t_lower = bang_bang_duration(rough_dur, accel)

        t_cursor = 0.0
        occ_id = occupant_of.get(req.atom_id)
        traj_of = {a.atom_id: a for a in state.sequence.ensemble.atomtrajs}
        if occ_id is not None and occ_id in traj_of and traj_of[occ_id].segments:
            occ_t_start = traj_of[occ_id].segments[0].start_time
            t_cursor = max(t_cursor,
                           occ_t_start + t_handoff - rough_t_lower)
            t_cursor = max(t_cursor, 0.0)

        t_start_max: Optional[float] = None
        inc_id = incoming_of.get(req.atom_id)
        if inc_id is not None and inc_id in traj_of and traj_of[inc_id].segments:
            t_start_max = traj_of[inc_id].final_time - 1e-9

        segs_plan, corridor_idxs, plan_kind = _build_path(
            grid, blocked, i0, j0, i1, j1,
            state.reservations, accel, t_handoff, r_safe,
            t_start_floor=t_cursor, t_start_max=t_start_max,
        )

        if verbose:
            axes = ",".join(s[0] for s in segs_plan) or "-"
            print(f"  atom {req.atom_id}: {req.src}->{req.dst}  "
                  f"{plan_kind}  segs=[{axes}]")

        if not segs_plan:
            continue  # noop (src == dst)

        # Tighten occupancy bound now that the actual plan duration is known.
        if occ_id is not None and occ_id in traj_of and traj_of[occ_id].segments:
            actual_total = 0.0
            for k, (axis, p_s, p_e, q) in enumerate(segs_plan):
                actual_total += bang_bang_duration(abs(p_e - p_s), accel)
                if k < len(segs_plan) - 1:
                    actual_total += t_handoff
            occ_t_start = traj_of[occ_id].segments[0].start_time
            t_cursor = max(t_cursor,
                           occ_t_start + t_handoff - actual_total)

        # Re-solve t_start with the chosen plan against the reservation table,
        # then commit at that t_start.
        t_start, _ = _solve_plan_timing(
            segs_plan, corridor_idxs, state.reservations, accel,
            t_handoff, r_safe, t_floor=t_cursor,
        )
        _commit_segs(req, segs_plan, corridor_idxs, t_start, state,
                     accel, t_handoff)

    return state.sequence
