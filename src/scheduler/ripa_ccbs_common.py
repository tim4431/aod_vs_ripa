"""Shared RIPA adapter pieces for CCBS-style schedulers.

This module intentionally contains no CBS search. It owns only the RIPA-facing
data model, route graph construction, target assignment, and conversion from a
timed node path into validated RIPA bang-bang trajectories.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Literal

from ..atom_trajectory import AtomEnsemble, CollisionError, CollisionReport
from ..movement import PHYS_A_MAX_RIPA, Step, grid_accel_from_phys
from ..routing import Site
from ..segments import Segment, bang_bang_duration, make_const_acc_segment
from .base import AsyncScheduler

NodeId = int
Point = tuple[float, float]
Channel = Literal["row", "col"]
GraphMode = Literal["all_axis", "adjacent"]

EPS = 1e-9
INF = math.inf


@dataclass(frozen=True)
class CCBSAgent:
    """One labeled finite-radius agent for a CCBS backend."""

    start: NodeId
    goal: NodeId
    radius: float = math.sqrt(2.0) / 4.0


@dataclass
class CCBSGraph:
    """Finite geometric graph passed to a CCBS backend."""

    coords: dict[NodeId, Point] = field(default_factory=dict)
    edges: dict[NodeId, dict[NodeId, float]] = field(default_factory=dict)

    def add_node(self, node: NodeId, coord: Point) -> None:
        self.coords[int(node)] = (float(coord[0]), float(coord[1]))
        self.edges.setdefault(int(node), {})

    def add_edge(
        self,
        u: NodeId,
        v: NodeId,
        duration: float | None = None,
        *,
        bidirectional: bool = True,
    ) -> None:
        u = int(u)
        v = int(v)
        if u not in self.coords or v not in self.coords:
            raise KeyError(f"both edge endpoints must be nodes, got {u}->{v}")
        if duration is None:
            duration = math.hypot(
                self.coords[v][0] - self.coords[u][0],
                self.coords[v][1] - self.coords[u][1],
            )
        if duration <= 0:
            raise ValueError("edge duration must be positive")
        self.edges.setdefault(u, {})[v] = float(duration)
        if bidirectional:
            self.edges.setdefault(v, {})[u] = float(duration)


@dataclass(frozen=True)
class TimedState:
    node: NodeId
    time: float


@dataclass(frozen=True)
class TimedMove:
    """A timed graph action returned by a CCBS backend."""

    u: NodeId
    v: NodeId
    t1: float
    t2: float

    @property
    def is_wait(self) -> bool:
        return self.u == self.v

    @property
    def duration(self) -> float:
        return self.t2 - self.t1


@dataclass
class TimedPath:
    agent: int
    states: tuple[TimedState, ...]
    expanded: int = 0

    @property
    def cost(self) -> float:
        return self.states[-1].time if self.states else INF

    @property
    def makespan(self) -> float:
        return self.cost

    def moves(self, *, include_final_hold: bool = True) -> list[TimedMove]:
        result: list[TimedMove] = []
        for a, b in zip(self.states, self.states[1:]):
            if b.time <= a.time + EPS:
                continue
            result.append(TimedMove(a.node, b.node, a.time, b.time))
        if include_final_hold and self.states:
            last = self.states[-1]
            result.append(TimedMove(last.node, last.node, last.time, INF))
        return result


@dataclass
class CCBSSolution:
    found: bool
    paths: dict[int, TimedPath] = field(default_factory=dict)
    flowtime: float = INF
    makespan: float = INF
    high_level_expanded: int = 0
    high_level_generated: int = 0
    low_level_expanded: int = 0
    elapsed: float = 0.0


@dataclass(frozen=True)
class _RIPACCBSSegmentSpec:
    atom_id: int
    start: Site
    end: Site
    start_time: float
    channel: Channel


@dataclass(frozen=True)
class _RIPACCBSPlanStep(Step):
    """Whole CCBS plan as one mutation."""

    segments: tuple[_RIPACCBSSegmentSpec, ...]
    start_time: float = 0.0

    def apply(self, ensemble: AtomEnsemble) -> None:
        accel = grid_accel_from_phys(PHYS_A_MAX_RIPA, ensemble.grid.d)
        by_atom: dict[int, list[Segment]] = {}

        for spec in self.segments:
            self._check_axis(spec.start, spec.end, spec.channel)
            segment = make_const_acc_segment(
                spec.start,
                spec.end,
                spec.start_time,
                accel=accel,
                channel=spec.channel,
            )
            by_atom.setdefault(int(spec.atom_id), []).append(segment)

        for atom_id, segments in by_atom.items():
            segments.sort(key=lambda seg: seg.start_time)
            atom = ensemble.atomtraj_by_id(atom_id)
            pos = atom.final_pos
            t = atom.final_time
            for segment in segments:
                if segment.start_pos != pos:
                    raise ValueError(
                        f"atom {atom_id}: segment starts at {segment.start_pos} "
                        f"but atom is at {pos}"
                    )
                if segment.start_time + 1e-12 < t:
                    raise ValueError(
                        f"atom {atom_id}: segment starts at t={segment.start_time} "
                        f"before previous segment ends at t={t}"
                    )
                pos = segment.end_pos
                t = segment.end_time

        old_lengths = {
            atom.atom_id: len(atom.segments)
            for atom in ensemble.atomtrajs
        }
        try:
            for atom_id, segments in by_atom.items():
                ensemble.atomtraj_by_id(atom_id).segments.extend(segments)
            report = self._check_full_timeline(ensemble)
            if not report.ok:
                raise CollisionError(report)
        except Exception:
            for atom in ensemble.atomtrajs:
                del atom.segments[old_lengths[atom.atom_id] :]
            raise

    def end_time(self, ensemble: AtomEnsemble) -> float:
        accel = grid_accel_from_phys(PHYS_A_MAX_RIPA, ensemble.grid.d)
        latest = self.start_time
        for spec in self.segments:
            distance = math.hypot(
                spec.end[0] - spec.start[0],
                spec.end[1] - spec.start[1],
            )
            latest = max(latest, spec.start_time + bang_bang_duration(distance, accel))
        return latest

    @staticmethod
    def _check_axis(start: Site, end: Site, channel: Channel) -> None:
        if channel == "row" and start[1] != end[1]:
            raise ValueError(f"row-channel move must keep j fixed: {start}->{end}")
        if channel == "col" and start[0] != end[0]:
            raise ValueError(f"col-channel move must keep i fixed: {start}->{end}")
        if start != end and start[0] != end[0] and start[1] != end[1]:
            raise ValueError(f"RIPA CCBS move must be single-axis: {start}->{end}")

    @staticmethod
    def _check_full_timeline(ensemble: AtomEnsemble) -> CollisionReport:
        rc_grid = ensemble.grid.rc / ensemble.grid.d
        horizon = ensemble.total_duration()
        worst_grid = math.inf
        worst_pair: tuple[int, int, float] | None = None

        atoms = ensemble.atomtrajs
        for idx, atom_a in enumerate(atoms):
            for atom_b in atoms[idx + 1 :]:
                for seg_a in atom_a.timeline_segments(0.0, horizon):
                    if seg_a.duration <= 0:
                        continue
                    for seg_b in atom_b.timeline_segments(
                        seg_a.start_time,
                        seg_a.end_time,
                    ):
                        t0 = max(seg_a.start_time, seg_b.start_time)
                        t1 = min(seg_a.end_time, seg_b.end_time)
                        if t1 < t0 - 1e-12:
                            continue
                        d_grid, t = ensemble._distance_between_segments(
                            seg_a,
                            seg_b,
                            t0,
                            t1,
                            rc_grid,
                        )
                        if d_grid < worst_grid:
                            worst_grid = d_grid
                            worst_pair = (
                                int(atom_a.atom_id),
                                int(atom_b.atom_id),
                                float(t),
                            )

        worst_um = worst_grid * ensemble.grid.d
        return CollisionReport(
            ok=worst_um + 1e-12 >= ensemble.grid.rc,
            worst_pair=worst_pair,
            worst_distance=worst_um,
        )


@dataclass
class RIPACCBSBaseScheduler(AsyncScheduler):
    """Shared RIPA graph/assignment/emission layer for CCBS backends."""

    graph_mode: GraphMode = "all_axis"
    time_limit: float = 30.0
    max_high_level_nodes: int = 10000
    high_level_order: str = "cost"
    ccbs_precision: float | None = None
    max_exact_unlabeled_atoms: int = 12
    freeze_initial_target_atoms: bool = False
    ccbs_solution: CCBSSolution | None = field(default=None, init=False)
    _node_to_site: dict[int, Site] = field(default_factory=dict, init=False)
    _site_to_node: dict[Site, int] = field(default_factory=dict, init=False)
    _ccbs_agent_to_atom_id: list[int] = field(default_factory=list, init=False)
    _static_blockers: set[Site] = field(default_factory=set, init=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.graph_mode not in ("all_axis", "adjacent"):
            raise ValueError("graph_mode must be 'all_axis' or 'adjacent'")
        if self.max_high_level_nodes < 1:
            raise ValueError("max_high_level_nodes must be positive")
        if self.time_limit <= 0:
            raise ValueError("time_limit must be positive")

    def target_assignment(self) -> dict[int, Site]:
        if self.request.labeled:
            return super().target_assignment()

        sources = [tuple(site) for site in self.request.src]
        targets = [tuple(site) for site in self.request.dst]
        n = len(sources)
        if n == 0:
            return {}

        target_set = set(targets)
        fixed = {
            atom_id: src
            for atom_id, src in enumerate(sources)
            if src in target_set
        }
        remaining_atoms = [atom_id for atom_id in range(n) if atom_id not in fixed]
        fixed_targets = set(fixed.values())
        remaining_targets = [target for target in targets if target not in fixed_targets]
        if not remaining_atoms:
            return fixed

        costs = [
            [self._estimated_pair_cost(sources[atom_id], dst) for dst in remaining_targets]
            for atom_id in remaining_atoms
        ]
        if len(remaining_atoms) <= self.max_exact_unlabeled_atoms:
            partial = self._exact_min_cost_assignment(costs, remaining_targets)
        else:
            partial = self._linear_sum_assignment(costs, remaining_targets)
            if partial is None:
                partial = self._greedy_min_cost_assignment(costs, remaining_targets)

        assignment = dict(fixed)
        assignment.update(
            {
                atom_id: partial[local_id]
                for local_id, atom_id in enumerate(remaining_atoms)
            }
        )
        return assignment

    def _prepare_graph_and_agents(
        self,
    ) -> tuple[dict[int, Site], CCBSGraph, list[CCBSAgent]]:
        assignment = self.target_assignment()
        if self.freeze_initial_target_atoms:
            self._static_blockers = {
                tuple(self.request.src[atom_id])
                for atom_id, target in assignment.items()
                if tuple(self.request.src[atom_id]) == tuple(target)
            }
        else:
            self._static_blockers = set()
        graph = self._build_graph()
        agents = self._ccbs_agents(assignment)
        return assignment, graph, agents

    # ---- graph / agent construction ---------------------------------------

    def _build_graph(self) -> CCBSGraph:
        graph = CCBSGraph()
        self._node_to_site.clear()
        self._site_to_node.clear()

        N = self.request.grid.N
        for i in range(N):
            for j in range(N):
                site = (i, j)
                if site in self._static_blockers:
                    continue
                node = self._site_node(site)
                graph.add_node(node, site)
                self._node_to_site[node] = site
                self._site_to_node[site] = node

        if self.graph_mode == "adjacent":
            self._add_adjacent_edges(graph)
        else:
            self._add_all_axis_edges(graph)
        return graph

    def _add_adjacent_edges(self, graph: CCBSGraph) -> None:
        N = self.request.grid.N
        for i in range(N):
            for j in range(N):
                for ni, nj in ((i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)):
                    if 0 <= ni < N and 0 <= nj < N:
                        self._add_ripa_edge(graph, (i, j), (ni, nj))

    def _add_all_axis_edges(self, graph: CCBSGraph) -> None:
        N = self.request.grid.N
        for i in range(N):
            for j1 in range(N):
                for j2 in range(N):
                    if j1 != j2:
                        self._add_ripa_edge(graph, (i, j1), (i, j2))
        for j in range(N):
            for i1 in range(N):
                for i2 in range(N):
                    if i1 != i2:
                        self._add_ripa_edge(graph, (i1, j), (i2, j))

    def _add_ripa_edge(self, graph: CCBSGraph, start: Site, end: Site) -> None:
        if start not in self._site_to_node or end not in self._site_to_node:
            return
        if not self._axis_clear(start, end):
            return
        distance = math.hypot(end[0] - start[0], end[1] - start[1])
        graph.add_edge(
            self._site_node(start),
            self._site_node(end),
            self._move_duration(distance),
            bidirectional=False,
        )

    def _ccbs_agents(self, assignment: dict[int, Site]) -> list[CCBSAgent]:
        radius = 0.5 * self.request.grid.rc / self.request.grid.d
        agents: list[CCBSAgent] = []
        self._ccbs_agent_to_atom_id.clear()
        for atom_id in sorted(assignment):
            start = tuple(self.request.src[atom_id])
            goal = tuple(assignment[atom_id])
            if self.freeze_initial_target_atoms and start == goal:
                continue
            if start not in self._site_to_node or goal not in self._site_to_node:
                raise RuntimeError(
                    f"RIPA CCBS graph does not contain route endpoint {start}->{goal}"
                )
            self._ccbs_agent_to_atom_id.append(int(atom_id))
            agents.append(
                CCBSAgent(
                    start=self._site_node(start),
                    goal=self._site_node(goal),
                    radius=radius,
                )
            )
        return agents

    def _site_node(self, site: Site) -> int:
        return int(site[0]) * self.request.grid.N + int(site[1])

    def _axis_clear(self, start: Site, end: Site) -> bool:
        if start == end:
            return True
        if start[0] == end[0]:
            i = start[0]
            lo, hi = sorted((start[1], end[1]))
            return all((i, j) not in self._static_blockers for j in range(lo + 1, hi))
        if start[1] == end[1]:
            j = start[1]
            lo, hi = sorted((start[0], end[0]))
            return all((i, j) not in self._static_blockers for i in range(lo + 1, hi))
        return False

    # ---- solution emission -------------------------------------------------

    def _emit_solution(self, solution) -> None:
        segment_specs: list[_RIPACCBSSegmentSpec] = []
        for ccbs_agent_id, path in solution.paths.items():
            atom_id = self._ccbs_agent_to_atom_id[int(ccbs_agent_id)]
            for move in path.moves(include_final_hold=False):
                if move.u == move.v:
                    continue
                segment_specs.append(
                    _RIPACCBSSegmentSpec(
                        atom_id=int(atom_id),
                        start=self._node_to_site[int(move.u)],
                        end=self._node_to_site[int(move.v)],
                        start_time=float(move.t1),
                        channel=self._channel_for_move(move),
                    )
                )
        if segment_specs:
            segment_specs.sort(key=lambda spec: (spec.start_time, spec.atom_id, spec.end))
            self.append_step(_RIPACCBSPlanStep(tuple(segment_specs)))

    def _channel_for_move(self, move) -> Channel:
        start = self._node_to_site[int(move.u)]
        end = self._node_to_site[int(move.v)]
        if start[1] == end[1] and start[0] != end[0]:
            return "row"
        if start[0] == end[0] and start[1] != end[1]:
            return "col"
        raise ValueError(f"CCBS returned a non-RIPA edge: {start}->{end}")

    # ---- cost / assignment helpers ----------------------------------------

    def _estimated_pair_cost(self, src: Site, dst: Site) -> float:
        src = tuple(src)
        dst = tuple(dst)
        if src == dst:
            return 0.0
        if src[0] == dst[0] or src[1] == dst[1]:
            return self._move_duration(self._manhattan(src, dst))
        return (
            self._move_duration(abs(src[0] - dst[0]))
            + self._move_duration(abs(src[1] - dst[1]))
            + 0.25 * self._move_duration(1.0)
        )

    def _move_duration(self, distance: float) -> float:
        accel = grid_accel_from_phys(PHYS_A_MAX_RIPA, self.request.grid.d)
        return bang_bang_duration(float(distance), accel)

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

            best_cost = math.inf
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


__all__ = [
    "CCBSAgent",
    "CCBSGraph",
    "CCBSSolution",
    "Channel",
    "GraphMode",
    "RIPACCBSBaseScheduler",
    "TimedMove",
    "TimedPath",
    "TimedState",
]
