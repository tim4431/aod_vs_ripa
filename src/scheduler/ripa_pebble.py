"""Asynchronous highway-aware RIPA scheduler inspired by pebble/MAPF ideas.

This scheduler is intentionally a pragmatic first RIPA-pebble implementation,
not a full optimal CCBS solver. It keeps the pieces that matter most for the
current simulator:

* macro routes made from long single-axis RIPA legs,
* unlabeled source->target assignment by estimated route cost,
* asynchronous earliest-safe start times, with the trajectory validator as the
  source of truth,
* route retries across alternate highways when a resting atom blocks a leg.

The algorithm is closest to prioritized SIPP: each candidate leg is tried at the
earliest time the addressed atom is free; if validation reports a timed
collision, the start time is advanced and retried. If repeated retries still
fail, the route is treated as statically blocked and another lane is tried.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Literal

from ..atom_trajectory import CollisionError
from ..movement import PHYS_A_MAX_RIPA, RIPAStep, grid_accel_from_phys
from ..routing import Site
from ..segments import bang_bang_duration
from .base import AsyncScheduler

Channel = Literal["row", "col"]
LaneKind = Literal["hrow", "hcol", "direct_row", "direct_col"]
Lane = tuple[LaneKind, int]


@dataclass(frozen=True)
class RIPAPebbleLeg:
    """One macro edge in the RIPA route graph."""

    target: Site
    channel: Channel
    lane: Lane


@dataclass(frozen=True)
class _RouteCandidate:
    legs: tuple[RIPAPebbleLeg, ...]
    lane: Lane
    duration: float
    blockers: int
    score: float


@dataclass
class _AtomRoute:
    target: Site
    legs: list[RIPAPebbleLeg]
    lane: Lane


@dataclass
class RIPAPebbleScheduler(AsyncScheduler):
    """Asynchronous RIPA scheduler using highway macro routes.

    The scheduler supports labeled and unlabeled `RoutingRequest`s. For
    unlabeled requests it chooses a source-target assignment using estimated
    physical route duration plus penalties for handoffs, blockers, and lane use.
    """

    highway_period: int = 2
    storage_offset: int = 0
    allow_direct_storage_moves: bool = False
    max_exact_unlabeled_atoms: int = 12
    max_rounds: int | None = None
    max_start_attempts: int = 12
    route_keep_extra: int | None = None
    wait_increment: float | None = None
    wait_padding: float | None = None
    lane_load_weight: float = 0.25
    blocker_weight: float | None = None
    handoff_weight: float | None = None
    auxiliary_lane_penalty: float = 0.5
    retry_order_reversals: int = 2
    unlabeled_assignment: str = "min_sum"
    """Cost-matrix objective: 'min_sum' (Hungarian) or 'min_max' (bottleneck)."""
    _routes: dict[int, _AtomRoute] = field(default_factory=dict, init=False)
    _stall_count: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.highway_period < 2:
            raise ValueError("highway_period must be >= 2")
        if self.max_start_attempts < 1:
            raise ValueError("max_start_attempts must be >= 1")
        if self.unlabeled_assignment not in ("min_sum", "min_max"):
            raise ValueError("unlabeled_assignment must be 'min_sum' or 'min_max'")

    def target_assignment(self) -> dict[int, Site]:
        """Assign unlabeled atoms by route cost instead of Euclidean distance."""
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
        linear_assignment = self._linear_sum_assignment(costs, targets)
        if linear_assignment is not None:
            return linear_assignment
        return self._greedy_min_cost_assignment(costs, targets)

    def _plan(self) -> None:
        assignment = self.target_assignment()
        max_rounds = self.max_rounds
        if max_rounds is None:
            max_rounds = max(1, 12 * self.request.grid.N * max(1, len(assignment)))

        for _ in range(max_rounds):
            unfinished = self._unfinished_atoms(assignment)
            if not unfinished:
                return

            progress = False
            for atom_id in self._atom_order(unfinished, assignment):
                if self._move_one_leg(atom_id, tuple(assignment[atom_id])):
                    progress = True

            if progress:
                self._stall_count = 0
                continue

            self._stall_count += 1
            self._routes.clear()
            if self._stall_count <= self.retry_order_reversals:
                continue
            raise RuntimeError(
                "RIPAPebbleScheduler is stuck; no feasible async leg from "
                f"{self._debug_positions(unfinished, assignment)}"
            )

        raise RuntimeError(f"RIPAPebbleScheduler exceeded {max_rounds} rounds")

    def _unfinished_atoms(self, assignment: dict[int, Site]) -> list[int]:
        return [
            atom_id
            for atom_id, target in assignment.items()
            if self.ensemble.atomtraj_by_id(atom_id).final_pos != tuple(target)
        ]

    def _move_one_leg(self, atom_id: int, target: Site) -> bool:
        current = self.ensemble.atomtraj_by_id(atom_id).final_pos
        if current == target:
            self._routes.pop(atom_id, None)
            return False

        existing = self._routes.get(atom_id)
        if existing is not None and existing.target == target:
            if self._try_route_next_leg(atom_id, existing):
                return True
            self._routes.pop(atom_id, None)

        lane_load = self._lane_load()
        for candidate in self._route_candidates(atom_id, current, target, lane_load):
            route = _AtomRoute(
                target=target,
                legs=list(candidate.legs),
                lane=candidate.lane,
            )
            if self._try_route_next_leg(atom_id, route):
                if route.legs:
                    self._routes[atom_id] = route
                else:
                    self._routes.pop(atom_id, None)
                return True
        return False

    def _try_route_next_leg(self, atom_id: int, route: _AtomRoute) -> bool:
        current = self.ensemble.atomtraj_by_id(atom_id).final_pos
        while route.legs and route.legs[0].target == current:
            route.legs.pop(0)
        if not route.legs:
            return False

        leg = route.legs[0]
        self._validate_leg_axis(current, leg.target, leg.channel)
        start_time = self._earliest_atom_start(atom_id)
        step = self._append_leg_with_retries(atom_id, leg, start_time)
        if step is None:
            return False

        route.legs.pop(0)
        return True

    def _append_leg_with_retries(
        self,
        atom_id: int,
        leg: RIPAPebbleLeg,
        start_time: float,
    ) -> RIPAStep | None:
        wait = self._wait_increment()
        padding = self._wait_padding()
        t = start_time
        last_error: Exception | None = None

        for _ in range(self.max_start_attempts):
            step = RIPAStep(
                start_time=t,
                atom_id=int(atom_id),
                target=leg.target,
                channel=leg.channel,
            )
            try:
                self.append_step(step)
            except CollisionError as exc:
                last_error = exc
                report_time = (
                    exc.report.worst_pair[2]
                    if exc.report.worst_pair is not None
                    else t
                )
                t = max(t + wait, report_time + padding)
                continue
            except ValueError as exc:
                last_error = exc
                break
            self.last_error = None
            return step

        self.last_error = last_error
        return None

    def _route_candidates(
        self,
        atom_id: int,
        current: Site,
        target: Site,
        lane_load: dict[Lane, int],
    ) -> list[_RouteCandidate]:
        candidates: list[_RouteCandidate] = []
        horizontal_lanes = self._transport_indices("row", atom_id)
        vertical_lanes = self._transport_indices("col", atom_id)

        if current[0] != target[0]:
            for lane_index in horizontal_lanes:
                lane: Lane = ("hrow", lane_index)
                waypoints = [
                    (current[0], lane_index),
                    (target[0], lane_index),
                    target,
                ]
                self._append_candidate(
                    candidates,
                    atom_id,
                    current,
                    self._legs_from_waypoints(current, waypoints, lane),
                    lane,
                    lane_load,
                )

        if current[1] != target[1]:
            for lane_index in vertical_lanes:
                lane = ("hcol", lane_index)
                waypoints = [
                    (lane_index, current[1]),
                    (lane_index, target[1]),
                    target,
                ]
                self._append_candidate(
                    candidates,
                    atom_id,
                    current,
                    self._legs_from_waypoints(current, waypoints, lane),
                    lane,
                    lane_load,
                )

        allow_direct = self.allow_direct_storage_moves or (
            not horizontal_lanes and not vertical_lanes
        )
        if allow_direct and current[1] == target[1]:
            lane = ("direct_row", current[1])
            self._append_candidate(
                candidates,
                atom_id,
                current,
                self._legs_from_waypoints(current, [target], lane),
                lane,
                lane_load,
            )
        if allow_direct and current[0] == target[0]:
            lane = ("direct_col", current[0])
            self._append_candidate(
                candidates,
                atom_id,
                current,
                self._legs_from_waypoints(current, [target], lane),
                lane,
                lane_load,
            )

        return self._rank_candidates(candidates)

    def _append_candidate(
        self,
        candidates: list[_RouteCandidate],
        atom_id: int,
        current: Site,
        legs: tuple[RIPAPebbleLeg, ...],
        lane: Lane,
        lane_load: dict[Lane, int],
    ) -> None:
        if not legs:
            return

        waypoints = [leg.target for leg in legs]
        duration = self._route_duration(current, waypoints)
        blockers = self._blocker_count(atom_id, current, waypoints)
        lane_kind, lane_index = lane
        aux_penalty = (
            self.auxiliary_lane_penalty
            if lane_kind in ("hrow", "hcol")
            and not self._is_highway_index(lane_index)
            else 0.0
        )
        blocker_weight = (
            self.blocker_weight
            if self.blocker_weight is not None
            else 25.0 * self._unit_leg_duration()
        )
        handoff_weight = (
            self.handoff_weight
            if self.handoff_weight is not None
            else 0.35 * self._unit_leg_duration()
        )
        score = (
            duration
            + handoff_weight * max(0, len(legs) - 1)
            + blocker_weight * blockers
            + self.lane_load_weight * self._unit_leg_duration() * lane_load.get(lane, 0)
            + aux_penalty * self._unit_leg_duration()
        )
        candidates.append(
            _RouteCandidate(
                legs=legs,
                lane=lane,
                duration=duration,
                blockers=blockers,
                score=score,
            )
        )

    def _rank_candidates(
        self,
        candidates: list[_RouteCandidate],
    ) -> list[_RouteCandidate]:
        if not candidates:
            return []
        candidates = sorted(
            candidates,
            key=lambda candidate: (
                candidate.score,
                candidate.blockers,
                len(candidate.legs),
                candidate.lane,
            ),
        )
        keep = self.route_keep_extra
        if keep is None:
            keep = max(4, 2 * self.highway_period)
        best = candidates[0].score
        return [
            candidate
            for candidate in candidates
            if candidate.score <= best + keep * self._unit_leg_duration()
        ]

    def _legs_from_waypoints(
        self,
        current: Site,
        waypoints: list[Site],
        lane: Lane,
    ) -> tuple[RIPAPebbleLeg, ...]:
        legs: list[RIPAPebbleLeg] = []
        pos = tuple(current)
        for raw_next in waypoints:
            nxt = tuple(raw_next)
            if nxt == pos:
                continue
            legs.append(
                RIPAPebbleLeg(
                    target=nxt,
                    channel=self._channel_for_leg(pos, nxt),
                    lane=lane,
                )
            )
            pos = nxt
        return tuple(legs)

    def _transport_indices(self, axis: Channel, atom_id: int) -> list[int]:
        occ = self._final_occupancy()
        indices = set(self._highway_indices())
        for idx in range(self.request.grid.N):
            if self._is_highway_index(idx):
                continue
            if self._lane_is_clear(axis, idx, occ, atom_id):
                indices.add(idx)
        return sorted(indices)

    def _highway_indices(self) -> list[int]:
        return [
            idx
            for idx in range(self.request.grid.N)
            if self._is_highway_index(idx)
        ]

    def _is_highway_index(self, idx: int) -> bool:
        return (idx - self.storage_offset) % self.highway_period != 0

    def _lane_is_clear(
        self,
        axis: Channel,
        idx: int,
        occ: dict[Site, int],
        atom_id: int,
    ) -> bool:
        if axis == "row":
            return all(site[1] != idx or owner == atom_id for site, owner in occ.items())
        return all(site[0] != idx or owner == atom_id for site, owner in occ.items())

    def _final_occupancy(self) -> dict[Site, int]:
        return self.sequence.final_config().occupancy()

    def _lane_load(self) -> dict[Lane, int]:
        counts: dict[Lane, int] = {}
        for route in self._routes.values():
            if route.legs:
                counts[route.lane] = counts.get(route.lane, 0) + 1
        return counts

    def _blocker_count(
        self,
        atom_id: int,
        current: Site,
        waypoints: list[Site],
    ) -> int:
        occ = self._final_occupancy()
        blockers = 0
        pos = tuple(current)
        for nxt in waypoints:
            for site in self._sites_on_axis_segment(pos, tuple(nxt)):
                if site == pos:
                    continue
                owner = occ.get(site)
                if owner is not None and owner != atom_id:
                    blockers += 1
            pos = tuple(nxt)
        return blockers

    def _atom_order(
        self,
        unfinished: list[int],
        assignment: dict[int, Site],
    ) -> list[int]:
        target_to_atom = {tuple(target): atom_id for atom_id, target in assignment.items()}

        def key(atom_id: int) -> tuple[int, float, int]:
            current = self.ensemble.atomtraj_by_id(atom_id).final_pos
            target = tuple(assignment[atom_id])
            blocks_other_target = (
                current in target_to_atom and target_to_atom[current] != atom_id
            )
            return (
                0 if blocks_other_target else 1,
                -self._manhattan(current, target),
                atom_id,
            )

        ordered = sorted(unfinished, key=key)
        if self._stall_count % 2 == 1:
            ordered.reverse()
        return ordered

    def _earliest_atom_start(self, atom_id: int) -> float:
        final_time = self.ensemble.atomtraj_by_id(atom_id).final_time
        if final_time <= 0:
            return 0.0
        return final_time

    def _wait_increment(self) -> float:
        if self.wait_increment is not None:
            return self.wait_increment
        return max(self._unit_leg_duration(), self.collision_dt)

    def _wait_padding(self) -> float:
        if self.wait_padding is not None:
            return self.wait_padding
        return max(0.25 * self._unit_leg_duration(), 10.0 * self.collision_dt)

    def _unit_leg_duration(self) -> float:
        a = grid_accel_from_phys(PHYS_A_MAX_RIPA, self.request.grid.d)
        return bang_bang_duration(1.0, a)

    def _route_duration(self, current: Site, waypoints: list[Site]) -> float:
        total = 0.0
        pos = tuple(current)
        a = grid_accel_from_phys(PHYS_A_MAX_RIPA, self.request.grid.d)
        for nxt in waypoints:
            total += bang_bang_duration(self._manhattan(pos, tuple(nxt)), a)
            pos = tuple(nxt)
        return total

    def _estimated_pair_cost(self, atom_id: int, src: Site, dst: Site) -> float:
        if src == dst:
            return 0.0
        candidates = self._route_candidates(atom_id, tuple(src), tuple(dst), {})
        if not candidates:
            return float(self._manhattan(tuple(src), tuple(dst))) * self._unit_leg_duration()
        return min(candidate.score for candidate in candidates)

    def _debug_positions(
        self,
        unfinished: list[int],
        assignment: dict[int, Site],
    ) -> dict[int, tuple[Site, Site]]:
        return {
            atom_id: (
                self.ensemble.atomtraj_by_id(atom_id).final_pos,
                tuple(assignment[atom_id]),
            )
            for atom_id in unfinished
        }

    @staticmethod
    def _channel_for_leg(start: Site, end: Site) -> Channel:
        if start[1] == end[1] and start[0] != end[0]:
            return "row"
        if start[0] == end[0] and start[1] != end[1]:
            return "col"
        raise ValueError(f"RIPA legs must be single-axis, got {start}->{end}")

    @staticmethod
    def _validate_leg_axis(start: Site, end: Site, channel: Channel) -> None:
        expected = RIPAPebbleScheduler._channel_for_leg(start, end)
        if expected != channel:
            raise ValueError(f"leg {start}->{end} needs {expected}, not {channel}")

    @staticmethod
    def _sites_on_axis_segment(start: Site, end: Site) -> list[Site]:
        if start == end:
            return [start]
        if start[0] == end[0]:
            i = start[0]
            lo, hi = sorted((start[1], end[1]))
            return [(i, j) for j in range(lo, hi + 1)]
        if start[1] == end[1]:
            j = start[1]
            lo, hi = sorted((start[0], end[0]))
            return [(i, j) for i in range(lo, hi + 1)]
        raise ValueError(f"segment is not axis aligned: {start}->{end}")

    @staticmethod
    def _manhattan(a: Site, b: Site) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

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
        """Use SciPy's Hungarian/Jonker-Volgenant solver when available."""
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
        breaking ties by min total cost.
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

        lo, hi = 0, len(thresholds) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            masked = np.where(C <= thresholds[mid] + 1e-15, C, BIG)
            row_ind, col_ind = linear_sum_assignment(masked)
            if masked[row_ind, col_ind].max() < BIG:
                hi = mid
            else:
                lo = mid + 1

        masked = np.where(C <= thresholds[lo] + 1e-15, C, BIG)
        row_ind, col_ind = linear_sum_assignment(masked)
        return {
            int(atom_id): targets[int(target_idx)]
            for atom_id, target_idx in zip(row_ind, col_ind)
        }


RIPAPebbleAsyncScheduler = RIPAPebbleScheduler
