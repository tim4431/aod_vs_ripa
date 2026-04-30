"""Geometry-driven asynchronous RIPA pebble scheduler.

`RIPAPebbleAdvScheduler` is the no-dedicated-highway counterpart to
`RIPAPebbleScheduler`. It does not assume a storage/highway period. Instead it
builds routes from the current geometry: an atom may move in one RIPA leg to
any site visible along its row or column without crossing an occupied site.

The route graph therefore discovers long clear corridors when the atom
configuration happens to provide them, but it does not manufacture highways by
convention. For dense target-set assembly, the scheduler uses a staged
pebble-style plan: clear the target region into buffer sites, reopen boundary
gates when needed, then refill the target from the inside outward.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable, Literal

from ..atom_trajectory import CollisionError
from ..movement import PHYS_A_MAX_RIPA, RIPAStep, grid_accel_from_phys
from ..routing import Site
from ..segments import bang_bang_duration
from .base import AsyncScheduler

Channel = Literal["row", "col"]


@dataclass(frozen=True)
class RIPAPebbleAdvLeg:
    """One geometry-discovered axis-aligned RIPA leg."""

    target: Site
    channel: Channel


@dataclass
class RIPAPebbleAdvScheduler(AsyncScheduler):
    """Asynchronous RIPA scheduler with no assumed highway lattice.

    The planner uses current occupancy as a dynamic obstacle map. Long clear
    row/column moves are preferred because each candidate edge is costed by the
    physical bang-bang duration rather than by unit grid hops.
    """

    max_exact_unlabeled_atoms: int = 12
    max_start_attempts: int = 32
    wait_increment: float | None = None
    wait_padding: float | None = None
    handoff_weight: float = 0.35
    staged_target_fill: bool = True
    parallel_staging: bool = True

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.max_start_attempts < 1:
            raise ValueError("max_start_attempts must be >= 1")
        if self.handoff_weight < 0:
            raise ValueError("handoff_weight must be non-negative")

    def target_assignment(self) -> dict[int, Site]:
        """Assign unlabeled atoms by geometry-route cost."""
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
        linear_assignment = self._linear_sum_assignment(costs, targets)
        if linear_assignment is not None:
            return linear_assignment
        return self._greedy_min_cost_assignment(costs, targets)

    def _plan(self) -> None:
        if (
            self.staged_target_fill
            and not self.request.labeled
            and len(self.request.dst) == len(self.request.src)
            and len(self.request.dst) <= (self.request.grid.N * self.request.grid.N) // 2
        ):
            self._plan_unlabeled_target_set_with_fallback()
            return

        self._plan_assigned_routes(self.target_assignment())

    # ---- staged target-set assembly ---------------------------------------

    def _plan_unlabeled_target_set_with_fallback(self) -> None:
        if not self.parallel_staging:
            self._plan_unlabeled_target_set()
            return

        try:
            self._plan_unlabeled_target_set()
            return
        except RuntimeError:
            self._reset_sequence()

        self.parallel_staging = False
        try:
            self._plan_unlabeled_target_set()
        finally:
            self.parallel_staging = True

    def _reset_sequence(self) -> None:
        super().__post_init__()
        self.last_error = None

    def _plan_unlabeled_target_set(self) -> None:
        target_sites = {tuple(site) for site in self.request.dst}
        depths = self._target_depths(target_sites)

        self._clear_target_region(target_sites, depths)
        self._fill_target_region(target_sites, depths)

    def _clear_target_region(
        self,
        target_sites: set[Site],
        depths: dict[Site, int],
    ) -> None:
        while True:
            occ = self._final_occupancy()
            occupied_targets = [
                (site, atom_id)
                for site, atom_id in occ.items()
                if site in target_sites
            ]
            if not occupied_targets:
                return

            progress = False
            occupied_targets.sort(key=lambda item: (depths[item[0]], item[0]))
            for site, atom_id in occupied_targets:
                path = self._buffer_path(atom_id, site, target_sites, occ)
                if path is None:
                    continue
                self._append_path(
                    atom_id,
                    path,
                    sequential=not self.parallel_staging,
                )
                progress = True
                break

            if not progress:
                if self._relieve_target_gate(target_sites, occ):
                    continue
                raise RuntimeError(
                    "RIPAPebbleAdvScheduler could not clear the target region"
                )

    def _fill_target_region(
        self,
        target_sites: set[Site],
        depths: dict[Site, int],
    ) -> None:
        while True:
            occ = self._final_occupancy()
            occupied = set(occ)
            empty_targets = sorted(
                target_sites - occupied,
                key=lambda site: (-depths[site], site),
            )
            if not empty_targets:
                return

            best: tuple[float, int, Site, list[Site]] | None = None
            for target in empty_targets:
                for site, atom_id in occ.items():
                    if site in target_sites:
                        continue
                    path = self._shortest_path(atom_id, site, target, occ)
                    if path is None:
                        continue
                    score = self._path_cost(path)
                    key = (score, atom_id, target, path)
                    if best is None or key[:3] < best[:3]:
                        best = key

                if best is not None and best[2] == target:
                    break

            if best is None:
                raise RuntimeError(
                    "RIPAPebbleAdvScheduler could not fill the target region"
                )

            _, atom_id, _, path = best
            self._append_path(
                atom_id,
                path,
                sequential=not self.parallel_staging,
            )

    # ---- assigned route fallback ------------------------------------------

    def _plan_assigned_routes(self, assignment: dict[int, Site]) -> None:
        for atom_id in self._assigned_atom_order(assignment):
            target = tuple(assignment[atom_id])
            current = self.ensemble.atomtraj_by_id(atom_id).final_pos
            if current == target:
                continue
            occ = self._final_occupancy()
            path = self._shortest_path(atom_id, current, target, occ)
            if path is None:
                raise RuntimeError(
                    "RIPAPebbleAdvScheduler found no geometry route for "
                    f"atom {atom_id}: {current}->{target}"
                )
            self._append_path(atom_id, path, sequential=False)

    def _assigned_atom_order(self, assignment: dict[int, Site]) -> list[int]:
        return sorted(
            assignment,
            key=lambda atom_id: (
                self._manhattan(
                    self.ensemble.atomtraj_by_id(atom_id).final_pos,
                    tuple(assignment[atom_id]),
                ),
                atom_id,
            ),
        )

    # ---- route search ------------------------------------------------------

    def _shortest_path(
        self,
        atom_id: int,
        start: Site,
        goal: Site,
        occ: dict[Site, int],
    ) -> list[Site] | None:
        return self._shortest_path_to_any(atom_id, start, {goal}, occ)

    def _shortest_path_to_any(
        self,
        atom_id: int,
        start: Site,
        goals: Iterable[Site],
        occ: dict[Site, int],
    ) -> list[Site] | None:
        start = tuple(start)
        goal_set = {tuple(goal) for goal in goals}
        goal_set = {
            goal
            for goal in goal_set
            if occ.get(goal) in (None, atom_id) or goal == start
        }
        if not goal_set:
            return None
        if start in goal_set:
            return [start]

        dist: dict[Site, float] = {start: 0.0}
        prev: dict[Site, Site] = {}
        heap: list[tuple[float, Site]] = [(0.0, start)]
        visited: set[Site] = set()

        while heap:
            cost, site = heapq.heappop(heap)
            if site in visited:
                continue
            visited.add(site)
            if site in goal_set:
                return self._reconstruct_path(prev, site)

            for neighbor in self._visible_neighbors(atom_id, site, occ):
                new_cost = cost + self._leg_cost(site, neighbor)
                if new_cost + 1e-15 < dist.get(neighbor, math.inf):
                    dist[neighbor] = new_cost
                    prev[neighbor] = site
                    heapq.heappush(heap, (new_cost, neighbor))

        return None

    def _visible_neighbors(
        self,
        atom_id: int,
        site: Site,
        occ: dict[Site, int],
    ) -> Iterable[Site]:
        N = self.request.grid.N
        i, j = site
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ni, nj = i + di, j + dj
            while 0 <= ni < N and 0 <= nj < N:
                nxt = (ni, nj)
                owner = occ.get(nxt)
                if owner is not None and owner != atom_id:
                    break
                yield nxt
                ni += di
                nj += dj

    @staticmethod
    def _reconstruct_path(prev: dict[Site, Site], goal: Site) -> list[Site]:
        path = [goal]
        while path[-1] in prev:
            path.append(prev[path[-1]])
        path.reverse()
        return path

    def _path_cost(self, path: list[Site]) -> float:
        return sum(
            self._leg_cost(start, end)
            for start, end in zip(path, path[1:])
        )

    def _leg_cost(self, start: Site, end: Site) -> float:
        distance = self._manhattan(start, end)
        return self._move_duration(distance) + self.handoff_weight * self._unit_leg_duration()

    # ---- movement commit ---------------------------------------------------

    def _append_path(
        self,
        atom_id: int,
        path: list[Site],
        *,
        sequential: bool,
    ) -> None:
        current = self.ensemble.atomtraj_by_id(atom_id).final_pos
        for target in path[1:]:
            target = tuple(target)
            if current == target:
                continue
            channel = self._channel_for_leg(current, target)
            start_time = (
                self.sequence.next_start_time()
                if sequential
                else self._earliest_atom_start(atom_id)
            )
            step = self._append_leg_with_retries(
                atom_id,
                RIPAPebbleAdvLeg(target=target, channel=channel),
                start_time,
            )
            if step is None:
                raise RuntimeError(
                    "RIPAPebbleAdvScheduler could not append geometry leg "
                    f"for atom {atom_id}: {current}->{target}"
                )
            current = target

    def _append_leg_with_retries(
        self,
        atom_id: int,
        leg: RIPAPebbleAdvLeg,
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
                t = self._retry_time_after_collision(exc, t, wait, padding)
                continue
            except ValueError as exc:
                last_error = exc
                break

            self.last_error = None
            return step

        self.last_error = last_error
        return None

    def _retry_time_after_collision(
        self,
        exc: CollisionError,
        start_time: float,
        wait: float,
        padding: float,
    ) -> float:
        if exc.report.worst_pair is None:
            return start_time + wait

        report_time = exc.report.worst_pair[2]
        return max(start_time + wait, float(report_time) + padding)

    # ---- geometry helpers --------------------------------------------------

    def _target_depths(self, target_sites: set[Site]) -> dict[Site, int]:
        remaining = set(target_sites)
        depths: dict[Site, int] = {}
        depth = 0
        while remaining:
            boundary = {
                site
                for site in remaining
                if any(neighbor not in remaining for neighbor in self._grid_neighbors(site))
            }
            if not boundary:
                boundary = set(remaining)
            for site in boundary:
                depths[site] = depth
            remaining -= boundary
            depth += 1
        return depths

    def _grid_neighbors(self, site: Site) -> Iterable[Site]:
        N = self.request.grid.N
        i, j = site
        for ni, nj in ((i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)):
            if 0 <= ni < N and 0 <= nj < N:
                yield (ni, nj)

    def _free_sites(
        self,
        *,
        excluding: set[Site],
        occ: dict[Site, int],
    ) -> list[Site]:
        N = self.request.grid.N
        return [
            (i, j)
            for i in range(N)
            for j in range(N)
            if (i, j) not in excluding and (i, j) not in occ
        ]

    def _buffer_path(
        self,
        atom_id: int,
        start: Site,
        target_sites: set[Site],
        occ: dict[Site, int],
    ) -> list[Site] | None:
        goals = self._free_sites(excluding=target_sites, occ=occ)
        if not goals:
            return None

        by_distance: dict[int, list[Site]] = {}
        for goal in goals:
            distance = self._distance_to_set(goal, target_sites)
            by_distance.setdefault(distance, []).append(goal)

        for distance in sorted(by_distance, reverse=True):
            path = self._shortest_path_to_any(
                atom_id,
                start,
                by_distance[distance],
                occ,
            )
            if path is not None:
                return path
        return None

    def _relieve_target_gate(
        self,
        target_sites: set[Site],
        occ: dict[Site, int],
    ) -> bool:
        relief_moves: list[tuple[int, float, Site, int, list[Site]]] = []
        for site, atom_id in occ.items():
            if site in target_sites:
                continue
            path = self._buffer_path(atom_id, site, target_sites, occ)
            if path is None or path[-1] == site:
                continue
            start_distance = self._distance_to_set(site, target_sites)
            end_distance = self._distance_to_set(path[-1], target_sites)
            if end_distance <= start_distance:
                continue
            relief_moves.append(
                (start_distance, self._path_cost(path), site, atom_id, path)
            )

        for _, _, _, atom_id, path in sorted(relief_moves):
            self._append_path(
                atom_id,
                path,
                sequential=not self.parallel_staging,
            )
            return True
        return False

    @staticmethod
    def _distance_to_set(site: Site, sites: set[Site]) -> int:
        return min(abs(site[0] - other[0]) + abs(site[1] - other[1]) for other in sites)

    def _final_occupancy(self) -> dict[Site, int]:
        return self.sequence.final_config().occupancy()

    def _earliest_atom_start(self, atom_id: int) -> float:
        final_time = self.ensemble.atomtraj_by_id(atom_id).final_time
        if final_time <= 0:
            return 0.0
        return final_time + self.inter_step_gap

    def _wait_increment(self) -> float:
        if self.wait_increment is not None:
            return self.wait_increment
        return max(self._unit_leg_duration(), self.collision_dt)

    def _wait_padding(self) -> float:
        if self.wait_padding is not None:
            return self.wait_padding
        return self.collision_dt

    def _unit_leg_duration(self) -> float:
        return self._move_duration(1.0)

    def _move_duration(self, distance: float) -> float:
        a = grid_accel_from_phys(PHYS_A_MAX_RIPA, self.request.grid.d)
        return bang_bang_duration(float(distance), a)

    @staticmethod
    def _channel_for_leg(start: Site, end: Site) -> Channel:
        if start[1] == end[1] and start[0] != end[0]:
            return "row"
        if start[0] == end[0] and start[1] != end[1]:
            return "col"
        raise ValueError(f"RIPA legs must be single-axis, got {start}->{end}")

    @staticmethod
    def _manhattan(a: Site, b: Site) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    # ---- assignment helpers ------------------------------------------------

    def _estimated_pair_cost(self, atom_id: int, src: Site, dst: Site) -> float:
        if src == dst:
            return 0.0
        occ = self._final_occupancy()
        path = self._shortest_path(atom_id, tuple(src), tuple(dst), occ)
        if path is None:
            return (
                1000.0 * self._unit_leg_duration()
                + self._manhattan(tuple(src), tuple(dst)) * self._unit_leg_duration()
            )
        return self._path_cost(path)

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
        try:
            from scipy.optimize import linear_sum_assignment
        except Exception:
            return None

        row_ind, col_ind = linear_sum_assignment(costs)
        return {
            int(atom_id): targets[int(target_idx)]
            for atom_id, target_idx in zip(row_ind, col_ind)
        }


RIPAPebbleGeometryScheduler = RIPAPebbleAdvScheduler
