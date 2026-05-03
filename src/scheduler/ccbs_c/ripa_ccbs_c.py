"""C++-backed RIPA CCBS scheduler.

The Python side only translates a `RoutingRequest` into the compact graph
format expected by `ccbs_c/ccbs_solver.cpp`, runs that executable, and turns the
returned timed paths into validated RIPA trajectory segments.
"""

from __future__ import annotations

import math
import os
import subprocess
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Literal

from ...atom_trajectory import AtomEnsemble, CollisionError, CollisionReport
from ...movement import PHYS_A_MAX, Step, grid_accel_from_phys
from ...routing import Site
from ...segments import Segment, bang_bang_duration, make_const_acc_segment
from ..base import AsyncScheduler

GraphMode = Literal["all_axis", "adjacent"]
Channel = Literal["row", "col"]
PathState = tuple[int, float]

EPS = 1e-9


@dataclass
class CCBSSolution:
    """Solver summary plus timed node paths keyed by C++ agent id."""

    found: bool
    paths: dict[int, tuple[PathState, ...]] = field(default_factory=dict)
    flowtime: float = math.inf
    makespan: float = math.inf
    high_level_expanded: int = 0
    high_level_generated: int = 0
    low_level_expanded: int = 0
    elapsed: float = 0.0


@dataclass(frozen=True)
class _Agent:
    start: int
    goal: int
    radius: float


@dataclass(frozen=True)
class _SegmentSpec:
    atom_id: int
    start: Site
    end: Site
    start_time: float
    channel: Channel


@dataclass(frozen=True)
class _CCBSPlanStep(Step):
    """One CCBS solution, emitted as one validated mutation."""

    segments: tuple[_SegmentSpec, ...]
    start_time: float = 0.0

    def apply(self, ensemble: AtomEnsemble) -> None:
        accel = grid_accel_from_phys(PHYS_A_MAX, ensemble.grid.d)
        by_atom: dict[int, list[Segment]] = {}

        for spec in self.segments:
            _check_ripa_axis(spec.start, spec.end, spec.channel)
            segment = make_const_acc_segment(
                spec.start,
                spec.end,
                spec.start_time,
                accel=accel,
                channel=spec.channel,
            )
            by_atom.setdefault(spec.atom_id, []).append(segment)

        _check_atom_continuity(ensemble, by_atom)
        old_lengths = {atom.atom_id: len(atom.segments) for atom in ensemble.atomtrajs}
        try:
            for atom_id, segments in by_atom.items():
                ensemble.atomtraj_by_id(atom_id).segments.extend(segments)
            report = _check_full_timeline(ensemble)
            if not report.ok:
                raise CollisionError(report)
        except Exception:
            for atom in ensemble.atomtrajs:
                del atom.segments[old_lengths[atom.atom_id] :]
            raise

    def end_time(self, ensemble: AtomEnsemble) -> float:
        accel = grid_accel_from_phys(PHYS_A_MAX, ensemble.grid.d)
        latest = self.start_time
        for spec in self.segments:
            distance = math.hypot(
                spec.end[0] - spec.start[0],
                spec.end[1] - spec.start[1],
            )
            latest = max(latest, spec.start_time + bang_bang_duration(distance, accel))
        return latest


@dataclass
class RIPACCBSCScheduler(AsyncScheduler):
    """Plan a RIPA request with the standalone C++ CCBS solver."""

    graph_mode: GraphMode = "all_axis"
    time_limit: float = 30.0
    max_high_level_nodes: int = 10000
    high_level_order: str = "cost"
    ccbs_precision: float | None = None
    max_exact_unlabeled_atoms: int = 12
    freeze_initial_target_atoms: bool = False
    solver_path: str | os.PathLike[str] | None = None
    rebuild_solver: bool = False
    cxx: str = "g++"
    solver_timeout_padding: float = 5.0
    ccbs_solution: CCBSSolution | None = field(default=None, init=False)
    _node_to_site: dict[int, Site] = field(default_factory=dict, init=False)
    _site_to_node: dict[Site, int] = field(default_factory=dict, init=False)
    _agent_to_atom_id: list[int] = field(default_factory=list, init=False)
    _static_blockers: set[Site] = field(default_factory=set, init=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.graph_mode not in ("all_axis", "adjacent"):
            raise ValueError("graph_mode must be 'all_axis' or 'adjacent'")
        if self.time_limit <= 0:
            raise ValueError("time_limit must be positive")
        if self.max_high_level_nodes < 1:
            raise ValueError("max_high_level_nodes must be positive")

    def _plan(self) -> None:
        assignment = self.target_assignment()
        self._static_blockers = self._stationary_target_atoms(assignment)

        coords, edges = self._build_graph()
        agents = self._make_agents(assignment)
        if not agents:
            self.ccbs_solution = CCBSSolution(found=True, flowtime=0.0, makespan=0.0)
            return

        precision = self.collision_dt if self.ccbs_precision is None else self.ccbs_precision
        payload = self._solver_payload(coords, edges, agents, precision)
        solution = self._run_solver(payload)
        self.ccbs_solution = solution

        if not solution.found:
            raise RuntimeError(
                "RIPACCBSCScheduler did not find a CCBS solution "
                f"(expanded={solution.high_level_expanded}, "
                f"generated={solution.high_level_generated}, "
                f"elapsed={solution.elapsed:.3f}s)"
            )
        self._emit_solution(solution)

    # ---- assignment -----------------------------------------------------

    def target_assignment(self) -> dict[int, Site]:
        if self.request.labeled:
            return super().target_assignment()

        sources = [tuple(site) for site in self.request.src]
        targets = [tuple(site) for site in self.request.dst]
        if not sources:
            return {}

        target_set = set(targets)
        fixed = {
            atom_id: src
            for atom_id, src in enumerate(sources)
            if src in target_set
        }
        remaining_atoms = [atom_id for atom_id in range(len(sources)) if atom_id not in fixed]
        remaining_targets = [target for target in targets if target not in set(fixed.values())]
        if not remaining_atoms:
            return fixed

        costs = [
            [self._estimated_pair_cost(sources[atom_id], target) for target in remaining_targets]
            for atom_id in remaining_atoms
        ]
        if len(remaining_atoms) <= self.max_exact_unlabeled_atoms:
            partial = self._exact_assignment(costs, remaining_targets)
        else:
            partial = self._greedy_assignment(costs, remaining_targets)

        assignment = dict(fixed)
        assignment.update(
            {
                atom_id: partial[local_id]
                for local_id, atom_id in enumerate(remaining_atoms)
            }
        )
        return assignment

    def _estimated_pair_cost(self, src: Site, dst: Site) -> float:
        if src == dst:
            return 0.0
        if src[0] == dst[0] or src[1] == dst[1]:
            return self._move_duration(abs(src[0] - dst[0]) + abs(src[1] - dst[1]))
        return (
            self._move_duration(abs(src[0] - dst[0]))
            + self._move_duration(abs(src[1] - dst[1]))
            + 0.25 * self._move_duration(1.0)
        )

    @staticmethod
    def _exact_assignment(
        costs: list[list[float]],
        targets: list[Site],
    ) -> dict[int, Site]:
        n = len(costs)

        @lru_cache(maxsize=None)
        def solve(row: int, used_mask: int) -> tuple[float, tuple[int, ...]]:
            if row == n:
                return 0.0, ()

            best_cost = math.inf
            best_cols: tuple[int, ...] = ()
            for col in range(n):
                bit = 1 << col
                if used_mask & bit:
                    continue
                rest_cost, rest_cols = solve(row + 1, used_mask | bit)
                total = costs[row][col] + rest_cost
                if total < best_cost:
                    best_cost = total
                    best_cols = (col,) + rest_cols
            return best_cost, best_cols

        _, cols = solve(0, 0)
        return {row: targets[col] for row, col in enumerate(cols)}

    @staticmethod
    def _greedy_assignment(
        costs: list[list[float]],
        targets: list[Site],
    ) -> dict[int, Site]:
        remaining = set(range(len(targets)))
        assignment: dict[int, Site] = {}
        for row, cost_row in enumerate(costs):
            col = min(remaining, key=lambda idx: (cost_row[idx], targets[idx]))
            remaining.remove(col)
            assignment[row] = targets[col]
        return assignment

    # ---- graph construction --------------------------------------------

    def _stationary_target_atoms(self, assignment: dict[int, Site]) -> set[Site]:
        if not self.freeze_initial_target_atoms:
            return set()
        return {
            tuple(self.request.src[atom_id])
            for atom_id, target in assignment.items()
            if tuple(self.request.src[atom_id]) == tuple(target)
        }

    def _build_graph(self) -> tuple[dict[int, Site], dict[int, dict[int, float]]]:
        self._node_to_site.clear()
        self._site_to_node.clear()

        coords: dict[int, Site] = {}
        edges: dict[int, dict[int, float]] = {}
        for i in range(self.request.grid.N):
            for j in range(self.request.grid.N):
                site = (i, j)
                if site in self._static_blockers:
                    continue
                node = self._site_node(site)
                coords[node] = site
                edges[node] = {}
                self._node_to_site[node] = site
                self._site_to_node[site] = node

        if self.graph_mode == "adjacent":
            self._add_adjacent_edges(edges)
        else:
            self._add_axis_edges(edges)
        return coords, edges

    def _add_adjacent_edges(self, edges: dict[int, dict[int, float]]) -> None:
        N = self.request.grid.N
        for i in range(N):
            for j in range(N):
                for end in ((i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)):
                    if 0 <= end[0] < N and 0 <= end[1] < N:
                        self._add_ripa_edge(edges, (i, j), end)

    def _add_axis_edges(self, edges: dict[int, dict[int, float]]) -> None:
        N = self.request.grid.N
        for i in range(N):
            for j1 in range(N):
                for j2 in range(N):
                    if j1 != j2:
                        self._add_ripa_edge(edges, (i, j1), (i, j2))
        for j in range(N):
            for i1 in range(N):
                for i2 in range(N):
                    if i1 != i2:
                        self._add_ripa_edge(edges, (i1, j), (i2, j))

    def _add_ripa_edge(
        self,
        edges: dict[int, dict[int, float]],
        start: Site,
        end: Site,
    ) -> None:
        if start not in self._site_to_node or end not in self._site_to_node:
            return
        if not self._axis_clear(start, end):
            return

        u = self._site_node(start)
        v = self._site_node(end)
        distance = math.hypot(end[0] - start[0], end[1] - start[1])
        edges[u][v] = self._move_duration(distance)

    def _make_agents(self, assignment: dict[int, Site]) -> list[_Agent]:
        radius = 0.5 * self.request.grid.rc / self.request.grid.d
        agents: list[_Agent] = []
        self._agent_to_atom_id.clear()

        for atom_id in sorted(assignment):
            start = tuple(self.request.src[atom_id])
            goal = tuple(assignment[atom_id])
            if self.freeze_initial_target_atoms and start == goal:
                continue
            if start not in self._site_to_node or goal not in self._site_to_node:
                raise RuntimeError(
                    f"RIPA CCBS graph does not contain route endpoint {start}->{goal}"
                )
            self._agent_to_atom_id.append(atom_id)
            agents.append(
                _Agent(
                    start=self._site_node(start),
                    goal=self._site_node(goal),
                    radius=radius,
                )
            )
        return agents

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

    def _site_node(self, site: Site) -> int:
        return int(site[0]) * self.request.grid.N + int(site[1])

    def _move_duration(self, distance: float) -> float:
        accel = grid_accel_from_phys(PHYS_A_MAX, self.request.grid.d)
        return bang_bang_duration(float(distance), accel)

    # ---- C++ solver IO --------------------------------------------------

    def _run_solver(self, payload: str) -> CCBSSolution:
        solver = self._ensure_solver()
        timeout = max(self.time_limit + self.solver_timeout_padding, self.time_limit * 1.2)
        try:
            completed = subprocess.run(
                [str(solver)],
                input=payload,
                text=True,
                capture_output=True,
                check=False,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"C++ CCBS solver timed out after {timeout:.3f}s"
            ) from exc

        if completed.returncode != 0:
            detail = (completed.stdout + completed.stderr).strip()
            raise RuntimeError(f"C++ CCBS solver failed: {detail}")
        return self._parse_solution(completed.stdout)

    def _ensure_solver(self) -> Path:
        if self.solver_path is not None:
            solver = Path(self.solver_path)
            if not solver.exists():
                raise FileNotFoundError(f"C++ CCBS solver not found: {solver}")
            return solver

        here = Path(__file__).resolve().parent
        source = here / "ccbs_solver.cpp"
        binary = here / "ccbs_solver"
        if (
            not self.rebuild_solver
            and binary.exists()
            and binary.stat().st_mtime >= source.stat().st_mtime
        ):
            return binary

        cmd = [
            self.cxx,
            "-std=c++17",
            "-O3",
            "-DNDEBUG",
            "-Wall",
            "-Wextra",
            str(source),
            "-o",
            str(binary),
        ]
        completed = subprocess.run(cmd, text=True, capture_output=True, check=False)
        if completed.returncode != 0:
            raise RuntimeError(
                "failed to build C++ CCBS solver:\n"
                + (completed.stdout + completed.stderr).strip()
            )
        return binary

    def _solver_payload(
        self,
        coords: dict[int, Site],
        edges: dict[int, dict[int, float]],
        agents: list[_Agent],
        precision: float,
    ) -> str:
        edge_rows = [
            (u, v, duration)
            for u in sorted(edges)
            for v, duration in sorted(edges[u].items())
        ]
        high_level_order = 1 if self.high_level_order == "conflicts" else 0
        motion_profile = 1  # bang-bang RIPA timing in the C++ collision model.

        lines = [
            "CCBS_C_V1",
            (
                f"{len(coords)} {len(edge_rows)} {len(agents)} "
                f"{float(precision):.17g} {float(self.time_limit):.17g} "
                f"{int(self.max_high_level_nodes)} {high_level_order} {motion_profile}"
            ),
        ]
        for node in sorted(coords):
            i, j = coords[node]
            lines.append(f"node {node} {float(i):.17g} {float(j):.17g}")
        for u, v, duration in edge_rows:
            lines.append(f"edge {u} {v} {float(duration):.17g}")
        for agent in agents:
            lines.append(
                f"agent {agent.start} {agent.goal} {float(agent.radius):.17g}"
            )
        return "\n".join(lines) + "\n"

    @staticmethod
    def _parse_solution(stdout: str) -> CCBSSolution:
        lines = [line.strip() for line in stdout.splitlines() if line.strip()]
        if not lines:
            raise RuntimeError("C++ CCBS solver produced no output")
        if lines[0].startswith("ERROR"):
            raise RuntimeError(lines[0])

        header = lines[0].split()
        if len(header) != 8 or header[0] != "FOUND":
            raise RuntimeError(f"unexpected C++ CCBS header: {lines[0]}")

        solution = CCBSSolution(
            found=bool(int(header[1])),
            flowtime=float(header[2]),
            makespan=float(header[3]),
            high_level_expanded=int(header[4]),
            high_level_generated=int(header[5]),
            low_level_expanded=int(header[6]),
            elapsed=float(header[7]),
        )
        if not solution.found:
            return solution

        if len(lines) < 2:
            raise RuntimeError("C++ CCBS solver omitted PATHS section")
        path_header = lines[1].split()
        if len(path_header) != 2 or path_header[0] != "PATHS":
            raise RuntimeError(f"unexpected C++ CCBS paths header: {lines[1]}")

        paths: dict[int, tuple[PathState, ...]] = {}
        idx = 2
        for _ in range(int(path_header[1])):
            if idx >= len(lines):
                raise RuntimeError("C++ CCBS solver ended inside PATH section")
            path_parts = lines[idx].split()
            idx += 1
            if len(path_parts) != 3 or path_parts[0] != "PATH":
                raise RuntimeError(f"unexpected C++ CCBS path header: {lines[idx - 1]}")

            agent = int(path_parts[1])
            state_count = int(path_parts[2])
            states: list[PathState] = []
            for _ in range(state_count):
                if idx >= len(lines):
                    raise RuntimeError("C++ CCBS solver ended inside STATE section")
                state_parts = lines[idx].split()
                idx += 1
                if len(state_parts) != 3 or state_parts[0] != "STATE":
                    raise RuntimeError(
                        f"unexpected C++ CCBS state line: {lines[idx - 1]}"
                    )
                states.append((int(state_parts[1]), float(state_parts[2])))
            paths[agent] = tuple(states)

        solution.paths = paths
        return solution

    # ---- solution emission ---------------------------------------------

    def _emit_solution(self, solution: CCBSSolution) -> None:
        segment_specs: list[_SegmentSpec] = []
        for agent_id, states in solution.paths.items():
            atom_id = self._agent_to_atom_id[int(agent_id)]
            for (u, t1), (v, t2) in zip(states, states[1:]):
                if u == v or t2 <= t1 + EPS:
                    continue
                start = self._node_to_site[int(u)]
                end = self._node_to_site[int(v)]
                segment_specs.append(
                    _SegmentSpec(
                        atom_id=atom_id,
                        start=start,
                        end=end,
                        start_time=float(t1),
                        channel=self._channel_for_move(start, end),
                    )
                )

        if segment_specs:
            segment_specs.sort(key=lambda spec: (spec.start_time, spec.atom_id, spec.end))
            self.append_step(_CCBSPlanStep(tuple(segment_specs)))

    @staticmethod
    def _channel_for_move(start: Site, end: Site) -> Channel:
        if start[1] == end[1] and start[0] != end[0]:
            return "row"
        if start[0] == end[0] and start[1] != end[1]:
            return "col"
        raise ValueError(f"CCBS returned a non-RIPA edge: {start}->{end}")


def _check_ripa_axis(start: Site, end: Site, channel: Channel) -> None:
    if channel == "row" and start[1] != end[1]:
        raise ValueError(f"row-channel move must keep j fixed: {start}->{end}")
    if channel == "col" and start[0] != end[0]:
        raise ValueError(f"col-channel move must keep i fixed: {start}->{end}")
    if start != end and start[0] != end[0] and start[1] != end[1]:
        raise ValueError(f"RIPA CCBS move must be single-axis: {start}->{end}")


def _check_atom_continuity(
    ensemble: AtomEnsemble,
    by_atom: dict[int, list[Segment]],
) -> None:
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
                for seg_b in atom_b.timeline_segments(seg_a.start_time, seg_a.end_time):
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

    return CollisionReport(
        ok=worst_grid * ensemble.grid.d + 1e-12 >= ensemble.grid.rc,
        worst_pair=worst_pair,
        worst_distance=worst_grid * ensemble.grid.d,
    )


__all__ = [
    "CCBSSolution",
    "GraphMode",
    "RIPACCBSCScheduler",
]
