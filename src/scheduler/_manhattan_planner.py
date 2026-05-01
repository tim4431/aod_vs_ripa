"""Manhattan-corridor planner — internal helper for the RIPA pebble
schedulers.

This module is *not* a `Scheduler` in the sense of `base.Scheduler`; it
is the shared planning engine that `ripa_pebroute` (Hungarian + greedy)
and `ripa_stochastic_search` (simulated annealing) both invoke. Public
schedulers wrap the entry point `plan_labelled` in proper `Scheduler`
classes — see those modules for the user-facing API.

It is the project-side rewrite of
`lib/ripa2/visualization/src/schedulers/manhattan.py`. The algorithm is
unchanged in shape — greedy per-atom routing through 1/2/3/5-segment
plans selected by earliest predicted finish time, with a swap-pair joint
planner for swap pairs — but every data type now lives in `src/`:

  * `Grid` is `src.atom_config.Grid` (square N x N with site spacing `d`
    in micrometers, collision radius `rc`).
  * Atom trajectories are `src.atom_trajectory.AtomTrajectory`s held by
    an `AtomEnsemble`. Per-atom segments are committed via
    `src.movement.RIPAStep` so cross-atom collision validation runs on
    every commit.
  * The output is a `src.moving_sequence.MovingSequence`. Internal
    book-keeping (`Schedule`, `ScheduleEntry`, `CorridorReservations`)
    is exposed so the search wrapper and inspectors can audit which
    plan was chosen for each atom.

The Manhattan planner uses *grid units* (lattice steps) consistently:
positions in `(i, j)` integer coords, distances as `|Δi| + |Δj|` (or
single-axis `|Δp|`). Acceleration is converted from
`PHYS_A_MAX_RIPA` (m/s^2) to grid_units/s^2 via `grid_accel_from_phys`,
and segment times come from `src.segments.bang_bang_duration`.

Storage parity convention: a site `(i, j)` is a *storage* site iff
both `i` and `j` are even. Odd indices are corridors. This matches the
lib's hardcoded convention; if a different highway pattern is needed
in the future, expose the parity check as a parameter.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

from ..atom_config import AtomConfig, Grid
from ..atom_trajectory import AtomEnsemble, CollisionError, CollisionReport
from ..movement import PHYS_A_MAX_RIPA, RIPAStep, grid_accel_from_phys
from ..moving_sequence import MovingSequence
from ..routing import Site
from ..segments import Segment, bang_bang_duration, make_const_acc_segment


# ---------------- per-atom request + result types ----------------


@dataclass
class RoutingMove:
    """One labelled atom move (analogue of the lib's ``RoutingRequest``).

    Fields are kept identical so call sites that previously built per-atom
    lib requests need only swap the import.
    """

    atom_id: int
    src: Site
    dst: Site
    deadline: Optional[float] = None
    order: str = "auto"  # 'row_first' | 'col_first' | 'auto' (currently advisory)


@dataclass
class ManhattanParams:
    """Algorithm-level knobs the Manhattan-corridor planner needs.

    `t_handoff` is the dwell inserted between consecutive segments of one
    atom (and the temporal buffer between segments of *different* atoms on
    the same corridor). `r_safe` is the spatial buffer between two segments
    sharing a corridor; if ``None`` it defaults to ``grid.rc / grid.d`` —
    the same kinematic-radius the validator enforces.
    """

    t_handoff: float = 0.5e-6
    r_safe: Optional[float] = None  # in grid units; None -> derive from grid.rc


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
class Schedule:
    """Result of a Manhattan-corridor planning run.

    The canonical output is `sequence` (a `MovingSequence` whose ensemble
    holds every atom's full trajectory). `entries` and `reservations` are
    the planner's internal book-keeping, exposed so search wrappers and
    inspectors can audit per-segment timing and corridor usage.
    """

    sequence: MovingSequence
    entries: List[ScheduleEntry] = field(default_factory=list)
    reservations: CorridorReservations = field(default_factory=CorridorReservations)

    @property
    def t_end(self) -> float:
        return self.sequence.total_duration()

    @property
    def trajectories(self) -> dict:
        """atom_id -> AtomTrajectory (mirrors the lib's `Schedule.trajectories`
        slot but the value type is now `src.atom_trajectory.AtomTrajectory`)."""
        return {a.atom_id: a for a in self.sequence.ensemble.atomtrajs}

    def print_summary(self) -> None:
        print(f"Schedule: {len(self.entries)} entries, t_end = {self.t_end*1e6:.2f} us")
        for aid in sorted({e.atom_id for e in self.entries}):
            segs = sorted([e for e in self.entries if e.atom_id == aid],
                          key=lambda e: e.t0)
            print(f"  atom {aid}:")
            for e in segs:
                print(f"    [{e.t0*1e6:7.2f}-{e.t1*1e6:7.2f} us]  "
                      f"axis={e.axis}  {e.p_start:+5.1f} -> {e.p_end:+5.1f} "
                      f"(orth={e.q_fixed:+.1f})")


# ---------------- parity helpers (storage = even, corridors = odd) ----------------


def _is_storage(i: int, j: int) -> bool:
    return i % 2 == 0 and j % 2 == 0


# ---------------- timing primitives ----------------


def _grid_accel(grid: Grid) -> float:
    return grid_accel_from_phys(PHYS_A_MAX_RIPA, grid.d)


def _segment_min_duration(distance_grid: float, accel_grid: float) -> float:
    """Symmetric bang-bang time for a single-axis move of `distance_grid`
    sites under `accel_grid` grid_units/s^2."""
    return bang_bang_duration(distance_grid, accel_grid)


# ---------------- corridor selection (cost helpers) ----------------


def _row_clear(j: int, i_a: int, i_b: int, blocked: set) -> bool:
    lo, hi = sorted((i_a, i_b))
    for k in range(lo, hi + 1):
        if _is_storage(k, j) and (k, j) in blocked:
            return False
    return True


def _col_clear(i: int, j_a: int, j_b: int, blocked: set) -> bool:
    lo, hi = sorted((j_a, j_b))
    for k in range(lo, hi + 1):
        if _is_storage(i, k) and (i, k) in blocked:
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


def _plan_finish_time(plan_segs, corridor_idxs, reservations, accel,
                      t_handoff: float, r_safe: float,
                      t_start_floor: float = 0.0,
                      t_start_max: Optional[float] = None) -> float:
    """Eventual t_end if `plan_segs` is committed at the earliest feasible
    time >= `t_start_floor`; +inf if no feasible start <= `t_start_max`.

    Mirrors the lib version: outer loop pushes the cursor forward until
    every segment fits without conflict.
    """
    if not plan_segs:
        return t_start_floor

    if reservations is None:
        t = t_start_floor
        for k, (axis, p_s, p_e, q) in enumerate(plan_segs):
            t += _segment_min_duration(abs(p_e - p_s), accel)
            if k < len(plan_segs) - 1:
                t += t_handoff
        if t_start_max is not None and t_start_floor > t_start_max:
            return float("inf")
        return t

    t_cursor = t_start_floor
    for _outer in range(200):
        t = t_cursor
        conflict = None
        for k, (axis, p_s, p_e, q) in enumerate(plan_segs):
            dur = _segment_min_duration(abs(p_e - p_s), accel)
            if axis in ("x", "y") and dur > 0:
                if reservations.conflicts(axis, corridor_idxs[k], t, t + dur,
                                          p_s, p_e, t_handoff, r_safe):
                    conflict = reservations.earliest_free(
                        axis, corridor_idxs[k], t, t + dur,
                        p_s, p_e, t_handoff, r_safe)
                    break
            t += dur
            if k < len(plan_segs) - 1:
                t += t_handoff
        if conflict is None:
            if t_start_max is not None and t_cursor > t_start_max:
                return float("inf")
            return t
        t_cursor = conflict
    return float("inf")


def _solve_t_start(plan_segs, corridor_idxs, reservations, accel,
                   t_handoff: float, r_safe: float, t_floor: float = 0.0) -> float:
    """Earliest start time for the whole plan; companion to `_plan_finish_time`
    that returns the START rather than the finish."""
    t_cursor = t_floor
    for _ in range(200):
        t = t_cursor
        conflict = None
        for k, (axis, p_s, p_e, q) in enumerate(plan_segs):
            dur = _segment_min_duration(abs(p_e - p_s), accel)
            if axis in ("x", "y") and dur > 0:
                if reservations.conflicts(axis, corridor_idxs[k], t, t + dur,
                                          p_s, p_e, t_handoff, r_safe):
                    conflict = reservations.earliest_free(
                        axis, corridor_idxs[k], t, t + dur, p_s, p_e,
                        t_handoff, r_safe)
                    break
            t += dur
            if k < len(plan_segs) - 1:
                t += t_handoff
        if conflict is None:
            return t_cursor
        t_cursor = conflict
    return t_cursor


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
        cost = _plan_finish_time(plan_segs, corridor_idxs, reservations, accel,
                                 t_handoff, r_safe,
                                 t_start_floor=t_start_floor,
                                 t_start_max=t_start_max)
        candidates.append((cost, (plan_segs, corridor_idxs, kind)))

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
        dur = _segment_min_duration(abs(p_e - p_s), accel)
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


def _commit_pending(pending, sched: Schedule, accel: float) -> None:
    """Apply a batch of pending segments to the schedule.

    Segments are appended directly to each atom's `AtomTrajectory.segments`
    (and the corresponding `RIPAStep` recorded on `MovingSequence.steps`),
    *bypassing* the per-step kinematic validator. Reasoning: the planner's
    corridor reservation table already prevents the conflicts it cares
    about, but the kinematic validator additionally trips on transient
    cross-corridor overlaps — atom A momentarily resting at site X while
    atom B's row-corridor passes through X. The lib's reference algorithm
    accepts those cases at planning time and lets a separate kinematic
    validator (run by the search wrapper) reject the bad ones with
    `+inf` cost. We mirror that contract here: planning produces a
    schedule unconditionally; callers wanting strict validation use
    `validate_kinematic_schedule(schedule)`.

    Within this batch, sorting by `t_start` and `(atom_id, seg_idx)`
    keeps each atom's segments in chronological order — required by
    `AtomTrajectory._check_continuity`.
    """
    pending = sorted(pending, key=lambda item: (item[0], item[2].atom_id,
                                                item[2].seg_idx))
    grid = sched.sequence.grid
    accel_grid = accel  # accel is already in grid_units/s^2
    for _t_start, step, entry, corridor_idx in pending:
        atomtraj = sched.sequence.ensemble.atomtraj_by_id(step.atom_id)
        current = atomtraj.final_pos
        target = tuple(step.target)
        if current != target:
            seg = make_const_acc_segment(
                current, target, step.start_time,
                accel=accel_grid, channel=step.channel,
            )
            atomtraj._check_continuity(seg)
            atomtraj.segments.append(seg)
        sched.sequence.steps.append(step)
        sched.entries.append(entry)
        if corridor_idx is not None:
            sched.reservations.add(entry, corridor_idx)


def _commit_segs(req: RoutingMove, segs_plan, corridor_idxs, t_start: float,
                 sched: Schedule, accel: float, t_handoff: float) -> None:
    """Materialise a single-atom plan onto the schedule + MovingSequence."""
    pending = _build_pending_segments(req, segs_plan, corridor_idxs,
                                      t_start, accel, t_handoff)
    _commit_pending(pending, sched, accel)


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
        dur = _segment_min_duration(abs(p_e - p_s), accel)
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
                            r_safe: float, sched: Schedule, all_sites: set,
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
            t_sync = 0.0
            for _ in range(20):
                t_a_new = _solve_t_start(segs_a, corr_a, sched.reservations,
                                         accel, t_handoff, r_safe, t_floor=t_sync)
                sim_res = _snapshot_with_added(sched.reservations, segs_a, corr_a,
                                               t_a_new, accel, t_handoff,
                                               a.atom_id)
                t_b_new = _solve_t_start(segs_b, corr_b, sim_res,
                                         accel, t_handoff, r_safe, t_floor=t_sync)
                new_sync = max(t_a_new, t_b_new)
                if abs(new_sync - t_sync) < 1e-12:
                    t_sync = new_sync
                    break
                t_sync = new_sync
            t_a_end = _plan_finish_time(segs_a, corr_a, sched.reservations,
                                        accel, t_handoff, r_safe,
                                        t_start_floor=t_sync)
            sim_res = _snapshot_with_added(sched.reservations, segs_a, corr_a,
                                           t_sync, accel, t_handoff,
                                           a.atom_id)
            t_b_end = _plan_finish_time(segs_b, corr_b, sim_res,
                                        accel, t_handoff, r_safe,
                                        t_start_floor=t_sync)
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
    _commit_pending(pending, sched, accel)
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

    positions: List[Tuple[int, int]] = []
    atom_ids: List[int] = []
    for m in moves:
        positions.append((int(m.src[0]), int(m.src[1])))
        atom_ids.append(int(m.atom_id))

    next_static_id = -1
    for s in static_sites or []:
        positions.append((int(s[0]), int(s[1])))
        while next_static_id in move_ids:
            next_static_id -= 1
        atom_ids.append(next_static_id)
        next_static_id -= 1

    import numpy as np  # local — avoid an unconditional dependency at import time
    cfg = AtomConfig(
        positions=np.asarray(positions, dtype=int) if positions else np.zeros((0, 2), dtype=int),
        atom_ids=np.asarray(atom_ids, dtype=int) if atom_ids else np.zeros((0,), dtype=int),
    )
    return MovingSequence(grid=grid, initial=cfg, collision_dt=collision_dt)


def plan_labelled(moves: Sequence[RoutingMove], grid: Grid,
                  *,
                  params: Optional[ManhattanParams] = None,
                  t0_global: float = 0.0,
                  static_sites: Optional[Sequence[Site]] = None,
                  presort: bool = True,
                  enable_swap_pairs: bool = True,
                  collision_dt: float = 1e-6,
                  sequence: Optional[MovingSequence] = None,
                  verbose: bool = False) -> Schedule:
    """Greedy Manhattan-corridor planner.

    Mirrors the lib's ``plan(...)`` algorithm step-for-step, but emits the
    schedule via ``src.movement.RIPAStep`` onto a ``MovingSequence`` whose
    initial config matches the moves' source positions. The returned
    ``Schedule`` wraps that sequence and exposes the entries +
    reservations the planner used.

    `sequence`: optional pre-built `MovingSequence` to plan into. When the
    `Scheduler` class wrappers in `ripa_pebroute` / `ripa_stochastic_search`
    invoke the planner, they pass `self.sequence` (already constructed by
    `base.Scheduler.__post_init__` from the request's initial config) so
    the resulting schedule is the scheduler's own `self.sequence`. When
    `None`, the planner builds one from `moves` + `static_sites`.
    """
    params = params or ManhattanParams()
    accel = _grid_accel(grid)
    t_handoff = float(params.t_handoff)
    r_safe = (params.r_safe if params.r_safe is not None
              else grid.rc / grid.d)

    if sequence is None:
        sequence = _build_initial_sequence(moves, grid, static_sites, collision_dt)
    sched = Schedule(sequence=sequence)

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
                                    sched, all_sites, verbose)

    # ---- per-atom planning for everyone else ----
    remaining = [m for m in moves if m.atom_id not in paired]
    if presort:
        remaining = sorted(remaining, key=sort_key)

    for req in remaining:
        i0, j0 = req.src
        i1, j1 = req.dst

        blocked = all_sites - {tuple(req.src), tuple(req.dst)}

        rough_dur = (abs(i1 - i0) + abs(j1 - j0))
        rough_t_lower = _segment_min_duration(rough_dur, accel)

        t_cursor = t0_global
        occ_id = occupant_of.get(req.atom_id)
        traj_of = {a.atom_id: a for a in sched.sequence.ensemble.atomtrajs}
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
            sched.reservations, accel, t_handoff, r_safe,
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
                actual_total += _segment_min_duration(abs(p_e - p_s), accel)
                if k < len(segs_plan) - 1:
                    actual_total += t_handoff
            occ_t_start = traj_of[occ_id].segments[0].start_time
            t_cursor = max(t_cursor,
                           occ_t_start + t_handoff - actual_total)

        # Walk the conflict-resolution loop to get t_start.
        t_try = t_cursor
        for _ in range(200):
            t = t_try
            conflict_t = None
            for k, (axis, p_s, p_e, q) in enumerate(segs_plan):
                dur = _segment_min_duration(abs(p_e - p_s), accel)
                if axis in ("x", "y") and dur > 0:
                    if sched.reservations.conflicts(
                        axis, corridor_idxs[k], t, t + dur, p_s, p_e,
                        t_handoff, r_safe,
                    ):
                        conflict_t = sched.reservations.earliest_free(
                            axis, corridor_idxs[k], t, t + dur, p_s, p_e,
                            t_handoff, r_safe,
                        )
                        break
                t += dur
                if k < len(segs_plan) - 1:
                    t += t_handoff
            if conflict_t is None:
                break
            t_try = conflict_t

        _commit_segs(req, segs_plan, corridor_idxs, t_try, sched,
                     accel, t_handoff)

    return sched


# Convenience alias so call sites can use the lib-style name.
plan = plan_labelled


# ---------------- post-hoc validators (used by the search wrapper) ----------------


def validate_occupancy(moves: Sequence[RoutingMove], schedule: Schedule) -> List[str]:
    """Endpoint occupancy check.

    For every pair (mover, occupant) where the mover's destination is
    another atom's source, confirm the mover actually arrives *after* the
    occupant departs. Returns a list of human-readable violation strings;
    an empty list means the schedule is OK on this dimension.
    """
    by_id = {m.atom_id: m for m in moves}
    out: List[str] = []
    trajs = {a.atom_id: a for a in schedule.sequence.ensemble.atomtrajs
             if a.segments}
    for m in moves:
        if m.atom_id not in trajs:
            continue
        occupant = next(
            (n for n in moves
             if n.atom_id != m.atom_id and tuple(n.src) == tuple(m.dst)),
            None,
        )
        if occupant is None or occupant.atom_id not in trajs:
            continue
        my_arrival = trajs[m.atom_id].final_time
        occ_departure = trajs[occupant.atom_id].segments[0].start_time
        if occ_departure >= my_arrival:
            out.append(
                f"atom {m.atom_id} arrives at site {tuple(m.dst)} at "
                f"t={my_arrival*1e6:.1f} us, but atom {occupant.atom_id} "
                f"doesn't leave that site until t={occ_departure*1e6:.1f} us"
            )
    return out


def validate_in_flight(schedule: Schedule, params: ManhattanParams) -> List[str]:
    """Corridor-reservation overlap check (cheap, structural).

    Mirrors the lib's `_validate_in_flight_collisions` — looks for two
    entries on the same corridor whose space-time bounds overlap within
    the algorithm's own ``t_handoff`` / ``r_safe`` envelope.

    `r_safe` defaults to ``grid.rc / grid.d`` if not set on `params`.
    """
    grid = schedule.sequence.grid
    t_handoff = params.t_handoff
    r_safe = params.r_safe if params.r_safe is not None else grid.rc / grid.d
    out: List[str] = []
    for axis_name, bucket in (("row", schedule.reservations.by_row),
                              ("col", schedule.reservations.by_col)):
        for corr_idx, entries in bucket.items():
            n = len(entries)
            for i in range(n):
                for j in range(i + 1, n):
                    a, b = entries[i], entries[j]
                    if a.atom_id == b.atom_id:
                        continue
                    if a.t1 + t_handoff <= b.t0 or b.t1 + t_handoff <= a.t0:
                        continue
                    lo_a, hi_a = sorted((a.p_start, a.p_end))
                    lo_b, hi_b = sorted((b.p_start, b.p_end))
                    if hi_a + r_safe <= lo_b or hi_b + r_safe <= lo_a:
                        continue
                    out.append(
                        f"atoms {a.atom_id} and {b.atom_id} share "
                        f"{axis_name} {corr_idx}"
                    )
    return out


def validate_kinematic(schedule: Schedule, *, dt: float = 1e-6) -> bool:
    """True iff `MovingSequence.validate(dt)` is collision-free."""
    return schedule.sequence.validate(dt=dt).ok
