"""Geometry-driven asynchronous RIPA pebble scheduler.

`RIPAPebbleAdvScheduler` is the no-dedicated-highway counterpart to
`RIPAPebbleScheduler`. It does not assume a storage/highway period. Instead it
builds routes from the current geometry: an atom may move in one RIPA leg to
any site visible along its row or column without crossing an occupied site.

The route graph therefore discovers long clear corridors when the atom
configuration happens to provide them, but it does not manufacture highways by
convention. The scheduler is labeled-only: an atom at `src[k]` is routed to
`dst[k]`. For unlabeled set→set requests, wrap it in
`UnlabeledRIPAPebbleAdvScheduler`, which picks an atom→target assignment
(min_sum or min_max) and delegates the resulting labeled problem here.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Iterable, Literal

from ..atom_trajectory import CollisionError
from ..movement import PHYS_A_MAX_RIPA, RIPAStep, grid_accel_from_phys
from ..routing import RoutingRequest, Site
from ..segments import bang_bang_duration
from .base import AsyncScheduler, LabeledScheduler, UnlabeledScheduler

Channel = Literal["row", "col"]


@dataclass(frozen=True)
class RIPAPebbleAdvLeg:
    """One geometry-discovered axis-aligned RIPA leg."""

    target: Site
    channel: Channel


@dataclass
class RIPAPebbleAdvScheduler(LabeledScheduler, AsyncScheduler):
    """Labeled asynchronous RIPA scheduler with no assumed highway lattice.

    The planner uses current occupancy as a dynamic obstacle map. Long clear
    row/column moves are preferred because each candidate edge is costed by the
    physical bang-bang duration rather than by unit grid hops.
    """

    max_start_attempts: int = 32
    wait_increment: float | None = None
    wait_padding: float | None = None
    handoff_weight: float = 0.35
    staged_target_fill: bool = True
    async_staging: bool = True
    """Use per-atom async starts for staged moves; false serializes each staged leg."""

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.max_start_attempts < 1:
            raise ValueError("max_start_attempts must be >= 1")
        if self.handoff_weight < 0:
            raise ValueError("handoff_weight must be non-negative")

    def _plan(self) -> None:
        use_staged = (
            self.staged_target_fill
            and len(self.request.dst) == len(self.request.src)
            and len(self.request.dst)
            <= (self.request.grid.N * self.request.grid.N) // 2
        )
        if use_staged:
            self._plan_labeled_target_set_with_fallback()
            return

        self._plan_assigned_routes(self.target_assignment())

    # ---- staged target-set assembly ---------------------------------------

    def _plan_labeled_target_set_with_fallback(self) -> None:
        self._run_staged_with_fallback(self._plan_labeled_target_set)

    def _run_staged_with_fallback(self, plan: Callable[[], None]) -> None:
        if not self.async_staging:
            plan()
            return

        original_async = self.async_staging
        try:
            plan()
            return
        except RuntimeError:
            self._reset_sequence()

        self.async_staging = False
        try:
            plan()
        finally:
            self.async_staging = original_async

    def _reset_sequence(self) -> None:
        super().__post_init__()
        self.last_error = None

    # ---- labeled staged routing -------------------------------------------

    def _plan_labeled_target_set(self) -> None:
        assignment = {
            atom_id: tuple(site) for atom_id, site in enumerate(self.request.dst)
        }
        target_sites = set(assignment.values())
        depths = self._target_depths(target_sites)

        self._clear_wrong_labeled_targets(assignment, target_sites, depths)
        self._fill_labeled_targets(assignment, target_sites, depths)

    def _clear_wrong_labeled_targets(
        self,
        assignment: dict[int, Site],
        target_sites: set[Site],
        depths: dict[Site, int],
    ) -> None:
        while True:
            occ = self._final_occupancy()
            wrong_targets = [
                (site, atom_id)
                for site, atom_id in occ.items()
                if site in target_sites and tuple(assignment[atom_id]) != site
            ]
            if not wrong_targets:
                return

            progress = False
            wrong_targets.sort(key=lambda item: (depths[item[0]], item[0]))
            for site, atom_id in wrong_targets:
                own_target = tuple(assignment[atom_id])
                path = None
                if occ.get(own_target) is None:
                    path = self._shortest_path(atom_id, site, own_target, occ)
                if path is None:
                    path = self._buffer_path(atom_id, site, target_sites, occ)
                if path is None:
                    continue

                self._append_path(
                    atom_id,
                    path,
                    sequential=not self.async_staging,
                )
                progress = True
                break

            if not progress:
                if self._relieve_target_gate(target_sites, occ):
                    continue
                raise RuntimeError(
                    "RIPAPebbleAdvScheduler could not clear wrong labeled targets"
                )

    def _fill_labeled_targets(
        self,
        assignment: dict[int, Site],
        target_sites: set[Site],
        depths: dict[Site, int],
    ) -> None:
        target_to_atom = {target: atom_id for atom_id, target in assignment.items()}

        while True:
            occ = self._final_occupancy()
            site_by_atom = self.sequence.final_config().site_of_atom()
            unfinished_targets = [
                tuple(target)
                for atom_id, target in assignment.items()
                if site_by_atom.get(atom_id) != tuple(target)
                and occ.get(tuple(target)) is None
            ]
            if not unfinished_targets:
                if all(
                    site_by_atom.get(atom_id) == target
                    for atom_id, target in assignment.items()
                ):
                    return
                self._clear_wrong_labeled_targets(assignment, target_sites, depths)
                continue

            progress = False
            for target in sorted(
                unfinished_targets,
                key=lambda site: (-depths.get(site, 0), site),
            ):
                atom_id = target_to_atom[target]
                current = site_by_atom[atom_id]
                path = self._shortest_path(atom_id, current, target, occ)
                if path is None:
                    continue

                self._append_path(
                    atom_id,
                    path,
                    sequential=not self.async_staging,
                )
                progress = True
                break

            if progress:
                continue
            if self._relieve_target_gate(target_sites, occ):
                continue
            raise RuntimeError("RIPAPebbleAdvScheduler could not fill labeled targets")

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
        *,
        allowed_sites: set[Site] | None = None,
    ) -> list[Site] | None:
        return self._shortest_path_to_any(
            atom_id,
            start,
            {goal},
            occ,
            allowed_sites=allowed_sites,
        )

    def _shortest_path_to_any(
        self,
        atom_id: int,
        start: Site,
        goals: Iterable[Site],
        occ: dict[Site, int],
        *,
        allowed_sites: set[Site] | None = None,
    ) -> list[Site] | None:
        start = tuple(start)
        allowed = (
            {tuple(site) for site in allowed_sites} | {start}
            if allowed_sites is not None
            else None
        )
        goal_set = {tuple(goal) for goal in goals}
        if allowed is not None:
            goal_set = {goal for goal in goal_set if goal in allowed}
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

            for neighbor in self._visible_neighbors(
                atom_id,
                site,
                occ,
                allowed_sites=allowed,
            ):
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
        *,
        allowed_sites: set[Site] | None = None,
    ) -> Iterable[Site]:
        N = self.request.grid.N
        i, j = site
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ni, nj = i + di, j + dj
            while 0 <= ni < N and 0 <= nj < N:
                nxt = (ni, nj)
                if allowed_sites is not None and nxt not in allowed_sites:
                    break
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
        return sum(self._leg_cost(start, end) for start, end in zip(path, path[1:]))

    def _leg_cost(self, start: Site, end: Site) -> float:
        distance = self._manhattan(start, end)
        return (
            self._move_duration(distance)
            + self.handoff_weight * self._unit_leg_duration()
        )

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
                if any(
                    neighbor not in remaining for neighbor in self._grid_neighbors(site)
                )
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
                sequential=not self.async_staging,
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
        return final_time

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


RIPAPebbleGeometryScheduler = RIPAPebbleAdvScheduler


@dataclass
class UnlabeledRIPAPebbleAdvScheduler(UnlabeledScheduler, AsyncScheduler):
    """Unlabeled wrapper around `RIPAPebbleAdvScheduler`.

    Two routing modes, mirroring the prior behavior of the combined scheduler:

    * **Staged compaction** (default when `staged_target_fill=True`,
      `unlabeled_assignment="min_sum"`, and the target fits in half the grid).
      Greedy moves are picked from the current geometry without committing to a
      fixed atom→target pairing, so atoms already in target slots can slide
      inward to make room for incoming atoms.
    * **Assignment-based** (when staged compaction is disabled or
      `unlabeled_assignment="min_max"`). Picks an atom→target assignment from
      the unlabeled request and delegates the resulting labeled problem to
      `RIPAPebbleAdvScheduler`.
    """

    unlabeled_assignment: Literal["min_sum", "min_max"] = "min_sum"
    """Cost-matrix objective for the assignment-based path. `min_max`
    (bottleneck) bounds the longest single-atom path so atoms already in
    target slots also get moved when that helps the makespan; ties are broken
    by min total cost. Selecting `min_max` also forces the assignment-based
    path even when `staged_target_fill=True`."""
    max_exact_unlabeled_atoms: int = 12

    # Pass-through configuration for the inner labeled scheduler.
    max_start_attempts: int = 32
    wait_increment: float | None = None
    wait_padding: float | None = None
    handoff_weight: float = 0.35
    staged_target_fill: bool = True
    async_staging: bool = True

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.unlabeled_assignment not in ("min_sum", "min_max"):
            raise ValueError("unlabeled_assignment must be 'min_sum' or 'min_max'")

    def _plan(self) -> None:
        if not self.request.src:
            return

        use_staged = (
            self.staged_target_fill
            and self.unlabeled_assignment != "min_max"
            and len(self.request.dst) == len(self.request.src)
            and len(self.request.dst)
            <= (self.request.grid.N * self.request.grid.N) // 2
        )
        if use_staged:
            self._plan_staged_compaction_with_fallback()
        else:
            self._plan_via_assignment()

    # ---- assignment-based path --------------------------------------------

    def _plan_via_assignment(self) -> None:
        assignment = self.target_assignment()
        labeled_dst = [
            tuple(assignment[atom_id]) for atom_id in range(len(self.request.src))
        ]
        labeled_request = RoutingRequest(
            grid=self.request.grid,
            src=[tuple(s) for s in self.request.src],
            dst=labeled_dst,
            labeled=True,
        )
        # Force per-atom routing on the assignment we picked: the old combined
        # scheduler short-circuited `use_staged=False` for this path, so the
        # labeled clear-and-fill machinery never ran on top of it. Mirror that
        # by disabling staged_target_fill on the inner labeled scheduler.
        labeled = self._make_labeled_scheduler(
            labeled_request,
            staged_target_fill=False,
        )
        labeled.plan()
        self.sequence = labeled.sequence

    def target_assignment(self) -> dict[int, Site]:
        sources = [tuple(site) for site in self.request.src]
        targets = [tuple(site) for site in self.request.dst]
        n = len(sources)
        if n == 0:
            return {}

        helper = self._cost_helper()
        costs = [
            [self._estimated_pair_cost(helper, atom_id, src, dst) for dst in targets]
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

    # ---- staged unlabeled compaction --------------------------------------

    def _plan_staged_compaction_with_fallback(self) -> None:
        original_async = self.async_staging
        if not original_async:
            self._plan_staged_compaction()
            return

        try:
            self._plan_staged_compaction()
            return
        except RuntimeError:
            pass

        self.async_staging = False
        try:
            self._plan_staged_compaction()
        finally:
            self.async_staging = original_async

    def _plan_staged_compaction(self) -> None:
        helper = self._make_labeled_scheduler(
            self._identity_labeled_request(),
            validate_final=False,
        )
        target_sites = {tuple(site) for site in self.request.dst}
        depths = helper._target_depths(target_sites)
        self._compact_target_set(helper, target_sites, depths)
        self.sequence = helper.sequence

    def _compact_target_set(
        self,
        helper: RIPAPebbleAdvScheduler,
        target_sites: set[Site],
        depths: dict[Site, int],
    ) -> None:
        max_rounds = max(
            1,
            8 * self.request.grid.N * self.request.grid.N * max(1, len(target_sites)),
        )
        for _ in range(max_rounds):
            occ = helper._final_occupancy()
            if target_sites <= set(occ):
                return

            move = self._best_compaction_move(helper, target_sites, depths, occ)
            if move is not None:
                atom_id, path = move
                helper._append_path(
                    atom_id,
                    path,
                    sequential=not self.async_staging,
                )
                continue

            if helper._relieve_target_gate(target_sites, occ):
                continue

            raise RuntimeError(
                "UnlabeledRIPAPebbleAdvScheduler could not compact the target set"
            )

        raise RuntimeError(
            "UnlabeledRIPAPebbleAdvScheduler exceeded compaction rounds"
        )

    def _best_compaction_move(
        self,
        helper: RIPAPebbleAdvScheduler,
        target_sites: set[Site],
        depths: dict[Site, int],
        occ: dict[Site, int],
    ) -> tuple[int, list[Site]] | None:
        occupied = set(occ)
        empty_targets = sorted(
            target_sites - occupied,
            key=lambda site: (-depths[site], site),
        )

        for target in empty_targets:
            move = self._best_outside_to_target_move(
                helper, target, target_sites, occ
            )
            if move is not None:
                return move

            move = self._best_inward_target_slide(
                helper, target, target_sites, depths, occ
            )
            if move is not None:
                return move

        return None

    @staticmethod
    def _best_outside_to_target_move(
        helper: RIPAPebbleAdvScheduler,
        target: Site,
        target_sites: set[Site],
        occ: dict[Site, int],
    ) -> tuple[int, list[Site]] | None:
        best: tuple[float, int, list[Site]] | None = None
        for site, atom_id in occ.items():
            if site in target_sites:
                continue
            path = helper._shortest_path(atom_id, site, target, occ)
            if path is None:
                continue
            key = (helper._path_cost(path), atom_id, path)
            if best is None or key[:2] < best[:2]:
                best = key

        if best is None:
            return None
        _, atom_id, path = best
        return atom_id, path

    @staticmethod
    def _best_inward_target_slide(
        helper: RIPAPebbleAdvScheduler,
        target: Site,
        target_sites: set[Site],
        depths: dict[Site, int],
        occ: dict[Site, int],
    ) -> tuple[int, list[Site]] | None:
        target_depth = depths[target]
        best: tuple[int, float, int, list[Site]] | None = None
        for site, atom_id in occ.items():
            if site not in target_sites:
                continue
            source_depth = depths[site]
            if source_depth >= target_depth:
                continue
            path = helper._shortest_path(
                atom_id,
                site,
                target,
                occ,
                allowed_sites=target_sites,
            )
            if path is None:
                continue
            key = (source_depth, helper._path_cost(path), atom_id, path)
            if best is None or key[:3] < best[:3]:
                best = key

        if best is None:
            return None
        _, _, atom_id, path = best
        return atom_id, path

    # ---- helpers ----------------------------------------------------------

    def _make_labeled_scheduler(
        self,
        labeled_request: RoutingRequest,
        *,
        validate_final: bool = False,
        staged_target_fill: bool | None = None,
    ) -> RIPAPebbleAdvScheduler:
        return RIPAPebbleAdvScheduler(
            request=labeled_request,
            collision_dt=self.collision_dt,
            validate_final=validate_final,
            max_start_attempts=self.max_start_attempts,
            wait_increment=self.wait_increment,
            wait_padding=self.wait_padding,
            handoff_weight=self.handoff_weight,
            staged_target_fill=(
                self.staged_target_fill
                if staged_target_fill is None
                else staged_target_fill
            ),
            async_staging=self.async_staging,
        )

    def _identity_labeled_request(self) -> RoutingRequest:
        return RoutingRequest(
            grid=self.request.grid,
            src=[tuple(s) for s in self.request.src],
            dst=[tuple(s) for s in self.request.src],
            labeled=True,
        )

    def _cost_helper(self) -> RIPAPebbleAdvScheduler:
        """Labeled scheduler with an identity placeholder assignment whose
        path-search primitives we use to score candidate atom→target pairings.
        """
        return self._make_labeled_scheduler(self._identity_labeled_request())

    @staticmethod
    def _estimated_pair_cost(
        helper: RIPAPebbleAdvScheduler,
        atom_id: int,
        src: Site,
        dst: Site,
    ) -> float:
        if src == dst:
            return 0.0
        occ = helper._final_occupancy()
        path = helper._shortest_path(atom_id, tuple(src), tuple(dst), occ)
        unit = helper._unit_leg_duration()
        if path is None:
            return (
                1000.0 * unit
                + helper._manhattan(tuple(src), tuple(dst)) * unit
            )
        return helper._path_cost(path)

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
        return {
            atom_id: targets[target_idx] for atom_id, target_idx in enumerate(order)
        }

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

    @staticmethod
    def _bottleneck_assignment(
        costs: list[list[float]],
        targets: list[Site],
    ) -> dict[int, Site] | None:
        """Min-max bipartite assignment: minimize the worst single-atom cost,
        breaking ties by min total cost. Returns None if scipy is unavailable.
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

        # Binary search for the smallest threshold admitting a perfect matching.
        lo, hi = 0, len(thresholds) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            masked = np.where(C <= thresholds[mid] + 1e-15, C, BIG)
            row_ind, col_ind = linear_sum_assignment(masked)
            if masked[row_ind, col_ind].max() < BIG:
                hi = mid
            else:
                lo = mid + 1

        # Final assignment: min-sum within the threshold-feasible subgraph.
        masked = np.where(C <= thresholds[lo] + 1e-15, C, BIG)
        row_ind, col_ind = linear_sum_assignment(masked)
        return {
            int(atom_id): targets[int(target_idx)]
            for atom_id, target_idx in zip(row_ind, col_ind)
        }
