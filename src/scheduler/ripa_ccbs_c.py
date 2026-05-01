"""C++-backed RIPA CCBS scheduler.

This module keeps the Python-facing scheduler API from `ripa_ccbs.py`, but
delegates the high-level CBS and low-level SIPP search to the standalone C++
solver in `ccbs_c/ccbs_solver.cpp`.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .continuous_cbs_raw import CCBSSolution, TimedPath, TimedState
from .ripa_ccbs import GraphMode, RIPACCBSScheduler


@dataclass
class RIPACCBSCScheduler(RIPACCBSScheduler):
    """RIPA CCBS scheduler using a raw C++ search backend.

    The C++ backend receives a finite RIPA graph, labeled start/goal pairs, and
    solver options through stdin. It returns timed node paths, which this class
    emits through the same RIPA segment batch used by `RIPACCBSScheduler`.
    """

    solver_path: str | os.PathLike[str] | None = None
    rebuild_solver: bool = False
    cxx: str = "g++"
    solver_timeout_padding: float = 5.0

    def _plan(self) -> None:
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
        if not agents:
            self.ccbs_solution = CCBSSolution(found=True, flowtime=0.0, makespan=0.0)
            return

        precision = self.collision_dt if self.ccbs_precision is None else self.ccbs_precision
        solver = self._ensure_solver()
        payload = self._solver_payload(graph, agents, precision)
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

        solution = self._parse_solution(completed.stdout)
        self.ccbs_solution = solution
        if not solution.found:
            raise RuntimeError(
                "RIPACCBSCScheduler did not find a CCBS solution "
                f"(expanded={solution.high_level_expanded}, "
                f"generated={solution.high_level_generated}, "
                f"elapsed={solution.elapsed:.3f}s)"
            )

        self._emit_solution(solution)

    def _ensure_solver(self) -> Path:
        if self.solver_path is not None:
            solver = Path(self.solver_path)
            if not solver.exists():
                raise FileNotFoundError(f"C++ CCBS solver not found: {solver}")
            return solver

        here = Path(__file__).resolve().parent
        source = here / "ccbs_c" / "ccbs_solver.cpp"
        binary = here / "ccbs_c" / "ccbs_solver"

        needs_build = (
            self.rebuild_solver
            or not binary.exists()
            or binary.stat().st_mtime < source.stat().st_mtime
        )
        if not needs_build:
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
        completed = subprocess.run(
            cmd,
            text=True,
            capture_output=True,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "failed to build C++ CCBS solver:\n"
                + (completed.stdout + completed.stderr).strip()
            )
        return binary

    def _solver_payload(self, graph, agents, precision: float) -> str:
        nodes = sorted(graph.coords)
        edges = [
            (u, v, duration)
            for u in sorted(graph.edges)
            for v, duration in sorted(graph.edges[u].items())
        ]
        high_level_order = 1 if self.high_level_order == "conflicts" else 0
        motion_profile = 1  # bang-bang RIPA timing in the C++ collision model.

        lines = [
            "CCBS_C_V1",
            (
                f"{len(nodes)} {len(edges)} {len(agents)} "
                f"{float(precision):.17g} {float(self.time_limit):.17g} "
                f"{int(self.max_high_level_nodes)} {high_level_order} {motion_profile}"
            ),
        ]
        for node in nodes:
            i, j = graph.coords[node]
            lines.append(f"node {int(node)} {float(i):.17g} {float(j):.17g}")
        for u, v, duration in edges:
            lines.append(f"edge {int(u)} {int(v)} {float(duration):.17g}")
        for agent in agents:
            lines.append(
                f"agent {int(agent.start)} {int(agent.goal)} {float(agent.radius):.17g}"
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

        found = bool(int(header[1]))
        solution = CCBSSolution(
            found=found,
            flowtime=float(header[2]),
            makespan=float(header[3]),
            high_level_expanded=int(header[4]),
            high_level_generated=int(header[5]),
            low_level_expanded=int(header[6]),
            elapsed=float(header[7]),
        )
        if not found:
            return solution

        if len(lines) < 2:
            raise RuntimeError("C++ CCBS solver omitted PATHS section")
        path_header = lines[1].split()
        if len(path_header) != 2 or path_header[0] != "PATHS":
            raise RuntimeError(f"unexpected C++ CCBS paths header: {lines[1]}")

        expected_paths = int(path_header[1])
        idx = 2
        paths: dict[int, TimedPath] = {}
        for _ in range(expected_paths):
            if idx >= len(lines):
                raise RuntimeError("C++ CCBS solver ended inside PATH section")
            parts = lines[idx].split()
            idx += 1
            if len(parts) != 3 or parts[0] != "PATH":
                raise RuntimeError(f"unexpected C++ CCBS path header: {lines[idx - 1]}")

            agent = int(parts[1])
            state_count = int(parts[2])
            states: list[TimedState] = []
            for _ in range(state_count):
                if idx >= len(lines):
                    raise RuntimeError("C++ CCBS solver ended inside STATE section")
                state_parts = lines[idx].split()
                idx += 1
                if len(state_parts) != 3 or state_parts[0] != "STATE":
                    raise RuntimeError(
                        f"unexpected C++ CCBS state line: {lines[idx - 1]}"
                    )
                states.append(
                    TimedState(
                        node=int(state_parts[1]),
                        time=float(state_parts[2]),
                    )
                )
            paths[agent] = TimedPath(agent=agent, states=tuple(states))

        solution.paths = paths
        return solution


RIPACCBSCppScheduler = RIPACCBSCScheduler
RIPACCBSCBackendScheduler = RIPACCBSCScheduler

__all__ = [
    "GraphMode",
    "RIPACCBSCBackendScheduler",
    "RIPACCBSCppScheduler",
    "RIPACCBSCScheduler",
]
