"""Raw Python translation of the relevant Continuous-CBS machinery.

This file is intentionally self-contained. It is not yet a production RIPA
`Scheduler`; it is a compact reference implementation of the algorithmic pieces
from `lib/Continuous-CBS` that are most relevant to RIPA work:

* high-level CBS over a constraint tree,
* low-level SIPP-like single-agent search over safe intervals,
* continuous-time collision detection between finite-radius disk agents,
* conflict resolution by adding unsafe start/occupancy intervals.

The motion model here follows the original CCBS implementation more closely
than the RIPA simulator: graph actions are straight-line constant-velocity
moves with fixed durations. A later RIPA-native version should swap the edge
duration and collision/unsafe-interval calculations for bang-bang RIPA
segments, then keep this same high-level shape.
"""

from __future__ import annotations

import heapq
import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Mapping

NodeId = int
Point = tuple[float, float]
Interval = tuple[float, float]

EPS = 1e-9
INF = math.inf


# ---------------------------------------------------------------------------
# Public data model


@dataclass(frozen=True)
class CCBSAgent:
    """One labeled agent for raw CCBS.

    `radius` is in the same coordinate units as the graph node coordinates.
    Agents are assumed to be circular disks.
    """

    start: NodeId
    goal: NodeId
    radius: float = math.sqrt(2.0) / 4.0


@dataclass
class CCBSGraph:
    """Finite geometric graph used by raw CCBS."""

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
            duration = euclidean(self.coords[u], self.coords[v])
        if duration <= 0:
            raise ValueError("edge duration must be positive")
        self.edges.setdefault(u, {})[v] = float(duration)
        if bidirectional:
            self.edges.setdefault(v, {})[u] = float(duration)

    def neighbors(self, node: NodeId) -> Iterable[tuple[NodeId, float]]:
        return self.edges.get(int(node), {}).items()

    def duration(self, u: NodeId, v: NodeId) -> float:
        return self.edges[int(u)][int(v)]

    def coord(self, node: NodeId) -> Point:
        return self.coords[int(node)]

    @classmethod
    def grid(
        cls,
        width: int,
        height: int | None = None,
        *,
        blocked: Iterable[tuple[int, int]] = (),
        connectivity: int = 4,
        edge_duration_scale: float = 1.0,
    ) -> "CCBSGraph":
        """Build a rectangular grid graph.

        Node ids are `i * width + j`, matching the C++ grid id convention.
        Coordinates are `(i, j)`. `connectivity` may be 4 or 8.
        """

        if height is None:
            height = width
        if connectivity not in (4, 8):
            raise ValueError("connectivity must be 4 or 8")
        blocked_set = {tuple(site) for site in blocked}
        graph = cls()

        for i in range(height):
            for j in range(width):
                if (i, j) in blocked_set:
                    continue
                graph.add_node(i * width + j, (float(i), float(j)))

        steps = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        if connectivity == 8:
            steps.extend([(1, 1), (1, -1), (-1, 1), (-1, -1)])

        for i in range(height):
            for j in range(width):
                if (i, j) in blocked_set:
                    continue
                u = i * width + j
                for di, dj in steps:
                    ni, nj = i + di, j + dj
                    if not (0 <= ni < height and 0 <= nj < width):
                        continue
                    if (ni, nj) in blocked_set:
                        continue
                    v = ni * width + nj
                    duration = edge_duration_scale * math.hypot(di, dj)
                    graph.add_edge(u, v, duration, bidirectional=False)

        return graph


@dataclass(frozen=True)
class TimedState:
    node: NodeId
    time: float


@dataclass(frozen=True)
class TimedMove:
    """A timed action in a path.

    `u == v` is a wait/hold action. A final hold uses `t2 == math.inf`.
    """

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


@dataclass(frozen=True)
class CCBSConstraint:
    """Negative CCBS constraint.

    If `u != v`, the agent may not start action `u -> v` during `[t1, t2)`.
    If `u == v`, the agent may not occupy/wait at vertex `u` during `[t1, t2)`.
    """

    agent: int
    t1: float
    t2: float
    u: NodeId
    v: NodeId

    @property
    def is_vertex(self) -> bool:
        return self.u == self.v


@dataclass(frozen=True)
class CCBSConflict:
    agent1: int
    agent2: int
    move1: TimedMove
    move2: TimedMove
    time: float


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
    constraints: tuple[CCBSConstraint, ...] = ()
    elapsed: float = 0.0


# ---------------------------------------------------------------------------
# SIPP low-level planner


@dataclass
class _SIPPRecord:
    node: NodeId
    interval_id: int
    time: float
    parent: "_SIPPRecord | None" = None
    depart_time: float | None = None


class RawSIPP:
    """Single-agent safe-interval planner for CCBS constraints."""

    def __init__(self, graph: CCBSGraph):
        self.graph = graph
        self._heuristic_cache: dict[NodeId, dict[NodeId, float]] = {}

    def find_path(
        self,
        *,
        agent_id: int,
        start: NodeId,
        goal: NodeId,
        constraints: Iterable[CCBSConstraint] = (),
    ) -> TimedPath | None:
        relevant = [c for c in constraints if c.agent == agent_id]
        vertex_blocks, edge_blocks = self._split_constraints(relevant)
        safe_cache: dict[NodeId, list[Interval]] = {}

        def safe_intervals(node: NodeId) -> list[Interval]:
            if node not in safe_cache:
                safe_cache[node] = complement_intervals(vertex_blocks.get(node, ()))
            return safe_cache[node]

        start_intervals = safe_intervals(start)
        start_interval_id = interval_containing(start_intervals, 0.0)
        if start_interval_id is None:
            return None

        heuristic = self._static_distances_to(goal)
        start_record = _SIPPRecord(start, start_interval_id, 0.0)
        open_heap: list[tuple[float, float, int, _SIPPRecord]] = []
        counter = 0
        heapq.heappush(
            open_heap,
            (heuristic.get(start, INF), -0.0, counter, start_record),
        )

        best_time: dict[tuple[NodeId, int], float] = {(start, start_interval_id): 0.0}
        closed: set[tuple[NodeId, int]] = set()
        expanded = 0

        while open_heap:
            _, _, _, current = heapq.heappop(open_heap)
            key = (current.node, current.interval_id)
            if key in closed:
                continue
            if current.time > best_time.get(key, INF) + EPS:
                continue
            closed.add(key)
            expanded += 1

            cur_interval = safe_intervals(current.node)[current.interval_id]
            if current.node == goal and math.isinf(cur_interval[1]):
                path = self._reconstruct_path(current)
                path.expanded = expanded
                path.agent = agent_id
                return path

            for nxt, duration in self.graph.neighbors(current.node):
                for nxt_interval_id, nxt_interval in enumerate(safe_intervals(nxt)):
                    departure = earliest_departure(
                        current_time=current.time,
                        source_interval=cur_interval,
                        target_interval=nxt_interval,
                        duration=duration,
                        edge_blocks=edge_blocks.get((current.node, nxt), ()),
                    )
                    if departure is None:
                        continue
                    arrive = departure + duration
                    succ_key = (nxt, nxt_interval_id)
                    if arrive + EPS >= best_time.get(succ_key, INF):
                        continue
                    best_time[succ_key] = arrive
                    counter += 1
                    succ = _SIPPRecord(
                        node=nxt,
                        interval_id=nxt_interval_id,
                        time=arrive,
                        parent=current,
                        depart_time=departure,
                    )
                    h = heuristic.get(nxt, INF)
                    heapq.heappush(open_heap, (arrive + h, -arrive, counter, succ))

        return None

    @staticmethod
    def _split_constraints(
        constraints: Iterable[CCBSConstraint],
    ) -> tuple[dict[NodeId, list[Interval]], dict[tuple[NodeId, NodeId], list[Interval]]]:
        vertex_blocks: dict[NodeId, list[Interval]] = defaultdict(list)
        edge_blocks: dict[tuple[NodeId, NodeId], list[Interval]] = defaultdict(list)
        for con in constraints:
            if con.t2 <= con.t1 + EPS:
                continue
            if con.is_vertex:
                vertex_blocks[con.u].append((con.t1, con.t2))
            else:
                edge_blocks[(con.u, con.v)].append((con.t1, con.t2))
        return (
            {node: merge_intervals(blocks) for node, blocks in vertex_blocks.items()},
            {edge: merge_intervals(blocks) for edge, blocks in edge_blocks.items()},
        )

    def _static_distances_to(self, goal: NodeId) -> dict[NodeId, float]:
        if goal in self._heuristic_cache:
            return self._heuristic_cache[goal]

        reverse: dict[NodeId, list[tuple[NodeId, float]]] = defaultdict(list)
        for u, nbrs in self.graph.edges.items():
            for v, cost in nbrs.items():
                reverse[v].append((u, cost))

        dist = {goal: 0.0}
        heap = [(0.0, goal)]
        while heap:
            cost, node = heapq.heappop(heap)
            if cost > dist.get(node, INF) + EPS:
                continue
            for pred, edge_cost in reverse.get(node, ()):
                new_cost = cost + edge_cost
                if new_cost + EPS < dist.get(pred, INF):
                    dist[pred] = new_cost
                    heapq.heappush(heap, (new_cost, pred))

        self._heuristic_cache[goal] = dist
        return dist

    @staticmethod
    def _reconstruct_path(record: _SIPPRecord) -> TimedPath:
        chain: list[_SIPPRecord] = []
        cur: _SIPPRecord | None = record
        while cur is not None:
            chain.append(cur)
            cur = cur.parent
        chain.reverse()

        states: list[TimedState] = [TimedState(chain[0].node, chain[0].time)]
        for prev, cur in zip(chain, chain[1:]):
            depart = cur.depart_time
            if depart is None:
                depart = prev.time
            if depart > prev.time + EPS:
                states.append(TimedState(prev.node, depart))
            states.append(TimedState(cur.node, cur.time))

        return TimedPath(agent=-1, states=tuple(states))


# ---------------------------------------------------------------------------
# CBS high-level planner


@dataclass
class _CTNode:
    id: int
    constraints: tuple[CCBSConstraint, ...]
    paths: dict[int, TimedPath]
    cost: float
    conflicts: tuple[CCBSConflict, ...] = ()


class ContinuousCBSRaw:
    """Raw continuous-time CBS planner.

    This is deliberately small and direct. It recomputes conflicts at each
    high-level node instead of carrying all of the C++ implementation's
    cardinal-conflict and disjoint-splitting machinery.
    """

    def __init__(
        self,
        graph: CCBSGraph,
        agents: Iterable[CCBSAgent],
        *,
        precision: float = 1e-6,
        time_limit: float = 30.0,
        max_high_level_nodes: int = 10000,
    ):
        self.graph = graph
        self.agents = list(agents)
        self.precision = float(precision)
        self.time_limit = float(time_limit)
        self.max_high_level_nodes = int(max_high_level_nodes)
        self.low_level = RawSIPP(graph)

    def find_solution(self) -> CCBSSolution:
        started = time.monotonic()
        root_paths: dict[int, TimedPath] = {}
        low_expanded = 0

        for agent_id, agent in enumerate(self.agents):
            path = self.low_level.find_path(
                agent_id=agent_id,
                start=agent.start,
                goal=agent.goal,
                constraints=(),
            )
            if path is None:
                return CCBSSolution(found=False, elapsed=time.monotonic() - started)
            root_paths[agent_id] = path
            low_expanded += path.expanded

        root = _CTNode(
            id=0,
            constraints=(),
            paths=root_paths,
            cost=sum(path.cost for path in root_paths.values()),
        )
        root.conflicts = tuple(self.get_all_conflicts(root.paths))

        open_heap: list[tuple[float, int, int, _CTNode]] = []
        heapq.heappush(open_heap, (root.cost, len(root.conflicts), root.id, root))
        generated = 1
        expanded = 0
        seen: set[frozenset[CCBSConstraint]] = {frozenset()}

        while open_heap:
            if time.monotonic() - started > self.time_limit:
                break
            if expanded >= self.max_high_level_nodes:
                break

            _, _, _, node = heapq.heappop(open_heap)
            expanded += 1
            conflicts = self.get_all_conflicts(node.paths)
            if not conflicts:
                return self._solution(
                    found=True,
                    node=node,
                    elapsed=time.monotonic() - started,
                    expanded=expanded,
                    generated=generated,
                    low_expanded=low_expanded,
                )

            conflict = min(conflicts, key=lambda item: item.time)
            child_specs = (
                (conflict.agent1, conflict.agent2, conflict.move1, conflict.move2),
                (conflict.agent2, conflict.agent1, conflict.move2, conflict.move1),
            )

            for replanned_agent, other_agent, own_move, other_move in child_specs:
                constraint = self.constraint_from_conflict(
                    replanned_agent,
                    other_agent,
                    own_move,
                    other_move,
                )
                if constraint is None:
                    continue
                new_constraints = node.constraints + (constraint,)
                key = frozenset(new_constraints)
                if key in seen:
                    continue
                seen.add(key)

                agent = self.agents[replanned_agent]
                path = self.low_level.find_path(
                    agent_id=replanned_agent,
                    start=agent.start,
                    goal=agent.goal,
                    constraints=new_constraints,
                )
                if path is None:
                    continue
                low_expanded += path.expanded

                child_paths = dict(node.paths)
                child_paths[replanned_agent] = path
                child_conflicts = tuple(self.get_all_conflicts(child_paths))
                child = _CTNode(
                    id=generated,
                    constraints=new_constraints,
                    paths=child_paths,
                    cost=sum(p.cost for p in child_paths.values()),
                    conflicts=child_conflicts,
                )
                generated += 1
                heapq.heappush(
                    open_heap,
                    (child.cost, len(child.conflicts), child.id, child),
                )

        return CCBSSolution(
            found=False,
            high_level_expanded=expanded,
            high_level_generated=generated,
            low_level_expanded=low_expanded,
            elapsed=time.monotonic() - started,
        )

    def get_all_conflicts(
        self,
        paths: Mapping[int, TimedPath],
    ) -> list[CCBSConflict]:
        conflicts: list[CCBSConflict] = []
        agent_ids = sorted(paths)
        for index, a_id in enumerate(agent_ids):
            for b_id in agent_ids[index + 1 :]:
                conflict = self.check_paths(a_id, paths[a_id], b_id, paths[b_id])
                if conflict is not None:
                    conflicts.append(conflict)
        return conflicts

    def check_paths(
        self,
        agent1: int,
        path1: TimedPath,
        agent2: int,
        path2: TimedPath,
    ) -> CCBSConflict | None:
        best: CCBSConflict | None = None
        radius_sum = self.agents[agent1].radius + self.agents[agent2].radius

        for move1 in path1.moves(include_final_hold=True):
            for move2 in path2.moves(include_final_hold=True):
                interval = collision_interval(
                    self.graph,
                    move1,
                    move2,
                    radius_sum,
                )
                if interval is None:
                    continue
                enter, _, closest = interval
                conflict_time = max(enter, closest)
                conflict = CCBSConflict(
                    agent1=agent1,
                    agent2=agent2,
                    move1=move1,
                    move2=move2,
                    time=conflict_time,
                )
                if best is None or conflict.time < best.time:
                    best = conflict
        return best

    def constraint_from_conflict(
        self,
        agent: int,
        other_agent: int,
        own_move: TimedMove,
        other_move: TimedMove,
    ) -> CCBSConstraint | None:
        radius_sum = self.agents[agent].radius + self.agents[other_agent].radius

        if own_move.is_wait:
            interval = collision_interval(self.graph, own_move, other_move, radius_sum)
            if interval is None:
                return None
            t1, t2, _ = interval
            if t2 <= t1 + EPS:
                t2 = t1 + self.precision
            return CCBSConstraint(agent, t1, t2, own_move.u, own_move.u)

        t1, t2 = self._move_unsafe_start_interval(own_move, other_move, radius_sum)
        if t2 <= t1 + EPS:
            t2 = t1 + self.precision
        return CCBSConstraint(agent, t1, t2, own_move.u, own_move.v)

    def _move_unsafe_start_interval(
        self,
        own_move: TimedMove,
        other_move: TimedMove,
        radius_sum: float,
    ) -> Interval:
        """Approximate CCBS unsafe interval for starting `own_move`.

        This mirrors the C++ implementation's spirit: hold the other timed
        action fixed, slide this action later until it no longer collides, and
        forbid starts from the current start time until that safe start.
        """

        if math.isinf(other_move.t2):
            return own_move.t1, INF

        duration = own_move.duration
        if not math.isfinite(duration) or duration <= 0:
            return own_move.t1, own_move.t1 + self.precision

        def shifted(start_time: float) -> TimedMove:
            return TimedMove(
                own_move.u,
                own_move.v,
                start_time,
                start_time + duration,
            )

        low = own_move.t1
        high = max(low + self.precision, other_move.t2 + duration + self.precision)
        while collision_interval(self.graph, shifted(high), other_move, radius_sum):
            high += max(duration, self.precision)
            if high - own_move.t1 > 1e6:
                return own_move.t1, INF

        for _ in range(80):
            if high - low <= self.precision:
                break
            mid = 0.5 * (low + high)
            if collision_interval(self.graph, shifted(mid), other_move, radius_sum):
                low = mid
            else:
                high = mid
        return own_move.t1, high

    @staticmethod
    def _solution(
        *,
        found: bool,
        node: _CTNode,
        elapsed: float,
        expanded: int,
        generated: int,
        low_expanded: int,
    ) -> CCBSSolution:
        makespan = max((path.cost for path in node.paths.values()), default=0.0)
        return CCBSSolution(
            found=found,
            paths=node.paths,
            flowtime=sum(path.cost for path in node.paths.values()),
            makespan=makespan,
            high_level_expanded=expanded,
            high_level_generated=generated,
            low_level_expanded=low_expanded,
            constraints=node.constraints,
            elapsed=elapsed,
        )


# ---------------------------------------------------------------------------
# Continuous collision helpers


def collision_interval(
    graph: CCBSGraph,
    move_a: TimedMove,
    move_b: TimedMove,
    radius_sum: float,
) -> tuple[float, float, float] | None:
    """Return `(enter, exit, closest_time)` if two timed moves collide."""

    t0 = max(move_a.t1, move_b.t1)
    t1 = min(move_a.t2, move_b.t2)
    if t1 < t0 - EPS:
        return None

    pa = position_at(graph, move_a, t0)
    pb = position_at(graph, move_b, t0)
    va = velocity(graph, move_a)
    vb = velocity(graph, move_b)

    px = pa[0] - pb[0]
    py = pa[1] - pb[1]
    vx = va[0] - vb[0]
    vy = va[1] - vb[1]
    r2 = radius_sum * radius_sum

    a = vx * vx + vy * vy
    b = 2.0 * (px * vx + py * vy)
    c = px * px + py * py - r2

    if a <= EPS:
        if c <= EPS:
            return t0, t1, t0
        return None

    closest_dt = -b / (2.0 * a)
    if math.isinf(t1):
        closest_dt = max(0.0, closest_dt)
    else:
        closest_dt = min(max(0.0, closest_dt), max(0.0, t1 - t0))
    closest_t = t0 + closest_dt

    disc = b * b - 4.0 * a * c
    if disc < -EPS:
        return None
    disc = max(0.0, disc)
    root = math.sqrt(disc)
    enter_dt = (-b - root) / (2.0 * a)
    exit_dt = (-b + root) / (2.0 * a)

    overlap_start = 0.0
    overlap_end = INF if math.isinf(t1) else t1 - t0
    enter = max(enter_dt, overlap_start)
    exit_ = min(exit_dt, overlap_end)
    if exit_ < enter - EPS:
        return None
    return t0 + enter, t0 + exit_, closest_t


def position_at(graph: CCBSGraph, move: TimedMove, t: float) -> Point:
    p0 = graph.coord(move.u)
    if move.is_wait or move.t2 <= move.t1 + EPS or math.isinf(move.t2):
        return p0
    p1 = graph.coord(move.v)
    alpha = min(1.0, max(0.0, (t - move.t1) / (move.t2 - move.t1)))
    return (
        p0[0] + alpha * (p1[0] - p0[0]),
        p0[1] + alpha * (p1[1] - p0[1]),
    )


def velocity(graph: CCBSGraph, move: TimedMove) -> Point:
    if move.is_wait or move.t2 <= move.t1 + EPS or math.isinf(move.t2):
        return (0.0, 0.0)
    p0 = graph.coord(move.u)
    p1 = graph.coord(move.v)
    dt = move.t2 - move.t1
    return ((p1[0] - p0[0]) / dt, (p1[1] - p0[1]) / dt)


# ---------------------------------------------------------------------------
# Interval and timing helpers


def earliest_departure(
    *,
    current_time: float,
    source_interval: Interval,
    target_interval: Interval,
    duration: float,
    edge_blocks: Iterable[Interval],
) -> float | None:
    """Earliest feasible action start under SIPP safe intervals."""

    source_start, source_end = source_interval
    target_start, target_end = target_interval
    depart = max(current_time, source_start, target_start - duration)
    blocks = list(edge_blocks)

    while True:
        if depart > source_end + EPS:
            return None
        arrival = depart + duration
        if arrival < target_start - EPS:
            depart = target_start - duration
            continue
        if arrival > target_end + EPS:
            return None

        shifted = False
        for block_start, block_end in blocks:
            if block_start - EPS <= depart < block_end - EPS:
                depart = block_end
                shifted = True
                break
        if shifted:
            continue
        return depart


def merge_intervals(intervals: Iterable[Interval]) -> list[Interval]:
    cleaned = [
        (max(0.0, float(a)), float(b))
        for a, b in intervals
        if b > max(0.0, a) + EPS
    ]
    if not cleaned:
        return []
    cleaned.sort(key=lambda item: item[0])
    merged = [cleaned[0]]
    for start, end in cleaned[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end + EPS:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def complement_intervals(blocked: Iterable[Interval]) -> list[Interval]:
    blocks = merge_intervals(blocked)
    safe: list[Interval] = []
    cursor = 0.0
    for start, end in blocks:
        if cursor < start - EPS:
            safe.append((cursor, start))
        cursor = max(cursor, end)
        if math.isinf(cursor):
            break
    if not math.isinf(cursor):
        safe.append((cursor, INF))
    return safe


def interval_containing(intervals: Iterable[Interval], t: float) -> int | None:
    for idx, (start, end) in enumerate(intervals):
        if start - EPS <= t <= end + EPS:
            return idx
    return None


def euclidean(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def path_as_node_time_list(path: TimedPath) -> list[tuple[NodeId, float]]:
    """Small display helper for experiments and notebooks."""

    return [(state.node, state.time) for state in path.states]


__all__ = [
    "CCBSAgent",
    "CCBSConflict",
    "CCBSConstraint",
    "CCBSGraph",
    "CCBSSolution",
    "ContinuousCBSRaw",
    "RawSIPP",
    "TimedMove",
    "TimedPath",
    "TimedState",
    "collision_interval",
    "complement_intervals",
    "earliest_departure",
    "merge_intervals",
    "path_as_node_time_list",
]
