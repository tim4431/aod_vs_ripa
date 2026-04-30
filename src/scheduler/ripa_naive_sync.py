"""Naive synchronous RIPA scheduler with highway routing.

The model follows the M-subgrid convention from `doc/aod_vs_ripa.md`:
atoms usually live on storage rows/columns separated by `highway_period`
sites, while the remaining rows/columns are used as obstacle-light
transport highways. For M=2 with `storage_offset=0`, even rows/columns
are storage sites and odd rows/columns are highways.

This scheduler is intentionally heuristic. Each clock cycle, it gives
each unfinished atom at most one RIPA leg, then greedily keeps the largest
feasible batch it can find under the existing collision validator. Candidate
routes are scored by path length, current lane load, and occupied blockers,
so the planner tends to fill many distinct highways while avoiding blocked
lanes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Literal

from ..movement import RIPAStep
from ..routing import Site
from .base import SyncScheduler

Channel = Literal["row", "col"]
Lane = tuple[Literal["hrow", "hcol", "direct_row", "direct_col"], int]


@dataclass(frozen=True)
class RIPARouteLeg:
    target: Site
    channel: Channel
    lane: Lane


@dataclass(frozen=True)
class _RouteCandidate:
    legs: tuple[RIPARouteLeg, ...]
    lane: Lane
    base_score: float
    score: float


@dataclass
class _AtomRoute:
    target: Site
    legs: list[RIPARouteLeg]
    lane: Lane


@dataclass
class RIPANaiveSyncScheduler(SyncScheduler):
    """Synchronous, highway-biased RIPA scheduler.

    The scheduler works for labeled and unlabeled `RoutingRequest`s. For
    unlabeled requests it uses the base class's greedy nearest-target
    assignment, then plans as if the request were labeled.
    """

    highway_period: int = 2
    storage_offset: int = 0
    max_cycles: int | None = None
    lane_load_weight: float | None = None
    blocker_weight: float | None = None
    lane_spread_extra: int | None = None
    auxiliary_lane_penalty: float = 0.5
    allow_direct_storage_moves: bool = False
    max_exact_unlabeled_atoms: int = 12
    _routes: dict[int, _AtomRoute] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.highway_period < 2:
            raise ValueError("highway_period must be >= 2")

    def target_assignment(self) -> dict[int, Site]:
        """Assign unlabeled atoms to targets by estimated highway route cost."""
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
        if n <= self.max_exact_unlabeled_atoms:
            return self._exact_min_cost_assignment(costs, targets)
        return self._greedy_min_cost_assignment(costs, targets)

    def _plan(self) -> None:
        assignment = self.target_assignment()
        max_cycles = self.max_cycles
        if max_cycles is None:
            max_cycles = max(1, 12 * self.request.grid.N * max(1, len(assignment)))

        for _ in range(max_cycles):
            unfinished = [
                atom_id
                for atom_id, target in assignment.items()
                if self.ensemble.atomtraj_by_id(atom_id).final_pos != tuple(target)
            ]
            if not unfinished:
                return

            start_time = self.sequence.next_start_time()
            selected = self._select_cycle_steps(unfinished, assignment, start_time)
            if not selected:
                raise RuntimeError(
                    "RIPANaiveSyncScheduler is stuck: no feasible highway leg "
                    f"from {self._debug_positions(unfinished, assignment)}"
                )

            trial = self.evaluate_cycle([step for _, step in selected])
            self.commit_trial(trial)
            for atom_id, _ in selected:
                route = self._routes.get(atom_id)
                if route is not None and route.legs:
                    route.legs.pop(0)
                    if not route.legs:
                        self._routes.pop(atom_id, None)

        raise RuntimeError(f"RIPANaiveSyncScheduler exceeded {max_cycles} cycles")

    def _select_cycle_steps(
        self,
        unfinished: list[int],
        assignment: dict[int, Site],
        start_time: float,
    ) -> list[tuple[int, RIPAStep]]:
        lane_load = self._lane_load()
        selected: list[tuple[int, RIPAStep]] = []
        selected_steps: list[RIPAStep] = []

        for atom_id in self._atom_order(unfinished, assignment):
            step = self._best_next_step(
                atom_id,
                tuple(assignment[atom_id]),
                start_time,
                selected_steps,
                lane_load,
            )
            if step is None:
                continue

            trial = self.evaluate_cycle(selected_steps + [step])
            if not trial.ok:
                continue

            selected.append((atom_id, step))
            selected_steps.append(step)
            route = self._routes.get(atom_id)
            if route is not None:
                lane_load[route.lane] = lane_load.get(route.lane, 0) + 1

        return selected

    def _best_next_step(
        self,
        atom_id: int,
        target: Site,
        start_time: float,
        selected_steps: list[RIPAStep],
        lane_load: dict[Lane, int],
    ) -> RIPAStep | None:
        current = self.ensemble.atomtraj_by_id(atom_id).final_pos
        existing = self._routes.get(atom_id)
        if existing is not None and existing.target == target:
            step = self._step_from_route(atom_id, current, existing, start_time)
            if step is not None and self.evaluate_cycle(selected_steps + [step]).ok:
                return step

        self._routes.pop(atom_id, None)
        for candidate in self._route_candidates(atom_id, current, target, lane_load):
            route = _AtomRoute(
                target=target,
                legs=list(candidate.legs),
                lane=candidate.lane,
            )
            step = self._step_from_route(atom_id, current, route, start_time)
            if step is None:
                continue
            if self.evaluate_cycle(selected_steps + [step]).ok:
                self._routes[atom_id] = route
                return step

        return None

    def _step_from_route(
        self,
        atom_id: int,
        current: Site,
        route: _AtomRoute,
        start_time: float,
    ) -> RIPAStep | None:
        while route.legs and route.legs[0].target == current:
            route.legs.pop(0)
        if not route.legs:
            return None

        leg = route.legs[0]
        self._validate_leg_axis(current, leg.target, leg.channel)
        return RIPAStep(
            start_time=start_time,
            atom_id=int(atom_id),
            target=leg.target,
            channel=leg.channel,
        )

    def _route_candidates(
        self,
        atom_id: int,
        current: Site,
        target: Site,
        lane_load: dict[Lane, int],
    ) -> list[_RouteCandidate]:
        candidates: list[_RouteCandidate] = []
        horizontal_lanes = self._transport_indices("row")
        vertical_lanes = self._transport_indices("col")
        if current[0] != target[0]:
            for transport_row in horizontal_lanes:
                lane: Lane = ("hrow", transport_row)
                waypoints = [
                    (current[0], transport_row),
                    (target[0], transport_row),
                    target,
                ]
                legs = self._legs_from_waypoints(current, waypoints, lane)
                self._append_candidate(
                    candidates, atom_id, current, legs, lane, lane_load
                )

        if current[1] != target[1]:
            for transport_col in vertical_lanes:
                lane = ("hcol", transport_col)
                waypoints = [
                    (transport_col, current[1]),
                    (transport_col, target[1]),
                    target,
                ]
                legs = self._legs_from_waypoints(current, waypoints, lane)
                self._append_candidate(
                    candidates, atom_id, current, legs, lane, lane_load
                )

        allow_direct = self.allow_direct_storage_moves or (
            not horizontal_lanes and not vertical_lanes
        )
        if allow_direct and current[1] == target[1]:
            lane = ("direct_row", current[1])
            legs = self._legs_from_waypoints(current, [target], lane)
            self._append_candidate(candidates, atom_id, current, legs, lane, lane_load)
        if allow_direct and current[0] == target[0]:
            lane = ("direct_col", current[0])
            legs = self._legs_from_waypoints(current, [target], lane)
            self._append_candidate(candidates, atom_id, current, legs, lane, lane_load)

        return self._rank_candidates(candidates, lane_load)

    def _append_candidate(
        self,
        candidates: list[_RouteCandidate],
        atom_id: int,
        current: Site,
        legs: tuple[RIPARouteLeg, ...],
        lane: Lane,
        lane_load: dict[Lane, int],
    ) -> None:
        if not legs:
            return
        path_length = self._path_length(current, [leg.target for leg in legs])
        blockers = self._blocker_count(atom_id, current, [leg.target for leg in legs])
        lane_kind, lane_index = lane
        auxiliary_penalty = (
            self.auxiliary_lane_penalty
            if lane_kind in ("hrow", "hcol") and not self._is_highway_index(lane_index)
            else 0.0
        )
        blocker_weight = (
            self.blocker_weight
            if self.blocker_weight is not None
            else 20.0 * self.request.grid.N
        )
        score = (
            path_length
            + auxiliary_penalty
            + blocker_weight * blockers
        )
        candidates.append(
            _RouteCandidate(legs=legs, lane=lane, base_score=score, score=score)
        )

    def _rank_candidates(
        self,
        candidates: list[_RouteCandidate],
        lane_load: dict[Lane, int],
    ) -> list[_RouteCandidate]:
        if not candidates:
            return []

        min_base = min(candidate.base_score for candidate in candidates)
        extra = (
            self.lane_spread_extra
            if self.lane_spread_extra is not None
            else 2 * self.highway_period
        )
        load_weight = (
            self.lane_load_weight
            if self.lane_load_weight is not None
            else 2.5 * self.highway_period
        )
        ranked = []
        for candidate in candidates:
            outside_local_band = max(0.0, candidate.base_score - min_base - extra)
            score = (
                candidate.base_score
                + load_weight * lane_load.get(candidate.lane, 0)
                + 1000.0 * outside_local_band
            )
            ranked.append(
                _RouteCandidate(
                    legs=candidate.legs,
                    lane=candidate.lane,
                    base_score=candidate.base_score,
                    score=score,
                )
            )
        return sorted(ranked, key=lambda candidate: (candidate.score, candidate.lane))

    def _legs_from_waypoints(
        self,
        current: Site,
        waypoints: list[Site],
        lane: Lane,
    ) -> tuple[RIPARouteLeg, ...]:
        legs: list[RIPARouteLeg] = []
        pos = tuple(current)
        for raw_next in waypoints:
            nxt = tuple(raw_next)
            if nxt == pos:
                continue
            channel = self._channel_for_leg(pos, nxt)
            legs.append(RIPARouteLeg(target=nxt, channel=channel, lane=lane))
            pos = nxt
        return tuple(legs)

    @staticmethod
    def _channel_for_leg(start: Site, end: Site) -> Channel:
        if start[1] == end[1] and start[0] != end[0]:
            return "row"
        if start[0] == end[0] and start[1] != end[1]:
            return "col"
        raise ValueError(f"RIPA legs must be single-axis, got {start}->{end}")

    @staticmethod
    def _validate_leg_axis(start: Site, end: Site, channel: Channel) -> None:
        expected = RIPANaiveSyncScheduler._channel_for_leg(start, end)
        if expected != channel:
            raise ValueError(f"leg {start}->{end} needs {expected}, not {channel}")

    def _highway_indices(self) -> list[int]:
        return [
            idx
            for idx in range(self.request.grid.N)
            if self._is_highway_index(idx)
        ]

    def _transport_indices(self, axis: Channel) -> list[int]:
        """Dedicated highways plus currently clear auxiliary storage lanes."""
        occ = self.sequence.occupancy_now()
        indices = set(self._highway_indices())
        for idx in range(self.request.grid.N):
            if self._is_highway_index(idx):
                continue
            if self._lane_is_clear(axis, idx, occ):
                indices.add(idx)
        return sorted(indices)

    def _lane_is_clear(
        self,
        axis: Channel,
        idx: int,
        occ: dict[Site, int],
    ) -> bool:
        if axis == "row":
            return all(site[1] != idx for site in occ)
        return all(site[0] != idx for site in occ)

    def _is_highway_index(self, idx: int) -> bool:
        return (idx - self.storage_offset) % self.highway_period != 0

    def _lane_load(self) -> dict[Lane, int]:
        counts: dict[Lane, int] = {}
        for route in self._routes.values():
            if route.legs:
                counts[route.lane] = counts.get(route.lane, 0) + 1
        return counts

    def _atom_order(
        self,
        unfinished: list[int],
        assignment: dict[int, Site],
    ) -> list[int]:
        return sorted(
            unfinished,
            key=lambda atom_id: (
                self._manhattan(
                    self.ensemble.atomtraj_by_id(atom_id).final_pos,
                    tuple(assignment[atom_id]),
                ),
                atom_id,
            ),
        )

    def _blocker_count(
        self,
        atom_id: int,
        current: Site,
        waypoints: list[Site],
    ) -> int:
        occ = self.sequence.occupancy_now()
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
    def _path_length(current: Site, waypoints: list[Site]) -> int:
        total = 0
        pos = tuple(current)
        for nxt in waypoints:
            total += RIPANaiveSyncScheduler._manhattan(pos, tuple(nxt))
            pos = tuple(nxt)
        return total

    @staticmethod
    def _manhattan(a: Site, b: Site) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

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

    def _estimated_pair_cost(self, atom_id: int, src: Site, dst: Site) -> float:
        if src == dst:
            return 0.0
        candidates = self._route_candidates(atom_id, tuple(src), tuple(dst), {})
        if not candidates:
            return float(self._manhattan(tuple(src), tuple(dst)))
        return min(candidate.score for candidate in candidates)

    @staticmethod
    def _exact_min_cost_assignment(
        costs: list[list[float]], targets: list[Site]
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
        costs: list[list[float]], targets: list[Site]
    ) -> dict[int, Site]:
        remaining = set(range(len(targets)))
        assignment: dict[int, Site] = {}
        for atom_id, row in enumerate(costs):
            target_idx = min(remaining, key=lambda idx: (row[idx], targets[idx]))
            remaining.remove(target_idx)
            assignment[atom_id] = targets[target_idx]
        return assignment
