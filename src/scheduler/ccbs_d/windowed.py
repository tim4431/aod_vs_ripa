"""Windowed/decomposed C++ CCBS scheduler.

This module keeps the exact C++ CCBS backend as the local oracle, but changes
the decomposition around it.  Dense line inversions are planned as small
multi-swap windows so CCBS can discover useful asynchronous overlap, then the
scheduler falls back to pairwise CCBS when a grouped window exceeds its budget.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field, replace
from typing import Iterable

from ...moving_sequence import MovingSequence
from ...movement import Step
from ...routing import RoutingRequest, Site
from ..ripa_ccbs_c import RIPACCBSCScheduler
from ..ripa_ccbs_common import CCBSSolution

Axis = int
Swap = tuple[int, int]
SwapWindow = tuple[Swap, ...]
StageCacheKey = tuple[Axis, bool, tuple[int, ...], tuple[tuple[int, int], ...]]


@dataclass(frozen=True)
class CCBSWindowStats:
    """Planning statistics for one accepted decomposed CCBS window."""

    depth: int
    line: int
    swaps: SwapWindow
    grouped: bool
    parallel: bool
    duration: float
    flowtime: float
    high_level_expanded: int
    high_level_generated: int
    low_level_expanded: int
    elapsed: float
    reused: bool = False


@dataclass(frozen=True)
class _LinePlan:
    line: int
    windows: tuple[SwapWindow, ...]


@dataclass(frozen=True)
class _AxisSwapPlan:
    axis: Axis
    lines: tuple[_LinePlan, ...]


@dataclass(frozen=True)
class _StagePlan:
    depth: int
    line: int
    swaps: SwapWindow
    grouped: bool
    axis: Axis
    steps: tuple[Step, ...]
    duration: float
    solution: CCBSSolution
    start_sites: tuple[tuple[int, Site], ...]
    reused: bool = False


@dataclass(frozen=True)
class _WindowPlan:
    stages: tuple[_StagePlan, ...]


@dataclass
class RIPACCBSDWindowedScheduler(RIPACCBSCScheduler):
    """C++ CCBS scheduler with line-inversion window decomposition.

    The supported fast path is a labeled permutation made of reciprocal swaps
    along one grid axis, such as a row/column inversion.  For each occupied
    line, the scheduler groups nested swaps into small windows:

    ```text
    outer swap + inner swap, then the remaining middle swap
    ```

    Each window is handed to the existing C++ backend.  If the grouped window
    times out or exceeds the node budget, the scheduler falls back to pairwise
    CCBS for that window.  Unsupported requests fall back to the ordinary
    one-shot `RIPACCBSCScheduler` unless `fallback_to_backend=False`.
    """

    max_window_swaps: int = 2
    parallel_line_stride: int = 2
    enable_grouped_windows: bool = True
    fallback_to_pairwise_windows: bool = True
    fallback_to_backend: bool = True
    grouped_time_limit: float | None = 15.0
    grouped_max_high_level_nodes: int | None = 100_000
    pair_time_limit: float | None = None
    pair_max_high_level_nodes: int | None = None
    reuse_symmetric_windows: bool = True
    window_stats: list[CCBSWindowStats] = field(default_factory=list, init=False)
    _stage_cache: dict[StageCacheKey, _StagePlan] = field(
        default_factory=dict,
        init=False,
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.max_window_swaps < 1:
            raise ValueError("max_window_swaps must be positive")
        if self.parallel_line_stride < 1:
            raise ValueError("parallel_line_stride must be positive")

    def _plan(self) -> None:
        self.window_stats.clear()
        self._stage_cache.clear()
        assignment = self.target_assignment()
        plan = self._detect_axis_swap_plan(assignment)

        if plan is None:
            if not self.fallback_to_backend:
                raise RuntimeError(
                    "RIPACCBSDWindowedScheduler only decomposes labeled "
                    "single-axis reciprocal swap permutations"
                )
            super()._plan()
            return

        if not plan.lines:
            self.ccbs_solution = CCBSSolution(found=True, flowtime=0.0, makespan=0.0)
            return

        self._plan_axis_swap_windows(plan)
        self.ccbs_solution = self._combined_solution()

    # ---- decomposition -------------------------------------------------

    def _detect_axis_swap_plan(
        self,
        assignment: dict[int, Site],
    ) -> _AxisSwapPlan | None:
        if not self.request.labeled:
            return None

        start_to_atom = {
            tuple(site): atom_id for atom_id, site in enumerate(self.request.src)
        }
        axis: Axis | None = None
        seen_pairs: set[frozenset[int]] = set()
        pairs_by_line: dict[int, list[Swap]] = defaultdict(list)

        for atom_id, start_site in enumerate(self.request.src):
            start = tuple(start_site)
            goal = tuple(assignment[atom_id])
            if start == goal:
                continue

            move_axis = self._single_axis(start, goal)
            if move_axis is None:
                return None
            if axis is None:
                axis = move_axis
            elif axis != move_axis:
                return None

            partner = start_to_atom.get(goal)
            if partner is None or tuple(assignment[partner]) != start:
                return None

            pair_key = frozenset((atom_id, partner))
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)

            if self._fixed_coord(start, axis) != self._fixed_coord(goal, axis):
                return None
            line = self._fixed_coord(start, axis)
            pairs_by_line[line].append(tuple(sorted((int(atom_id), int(partner)))))

        if axis is None:
            return _AxisSwapPlan(axis=0, lines=())

        lines = tuple(
            _LinePlan(line=line, windows=self._line_windows(pairs, axis))
            for line, pairs in sorted(pairs_by_line.items())
        )
        return _AxisSwapPlan(axis=axis, lines=lines)

    def _line_windows(self, pairs: Iterable[Swap], axis: Axis) -> tuple[SwapWindow, ...]:
        ordered = sorted(
            pairs,
            key=lambda pair: (
                -self._swap_distance(pair, axis),
                self._moving_coord(self.request.src[pair[0]], axis),
                self._moving_coord(self.request.src[pair[1]], axis),
                pair,
            ),
        )

        windows: list[SwapWindow] = []
        remaining = list(ordered)
        while remaining:
            window = [remaining.pop(0)]
            while len(window) < self.max_window_swaps and remaining:
                window.append(remaining.pop())
            windows.append(tuple(window))
        return tuple(windows)

    def _plan_axis_swap_windows(self, plan: _AxisSwapPlan) -> None:
        max_depth = max((len(line.windows) for line in plan.lines), default=0)
        for depth in range(max_depth):
            for offset in range(self.parallel_line_stride):
                selected = [
                    line
                    for line_index, line in enumerate(plan.lines)
                    if line_index % self.parallel_line_stride == offset
                    and depth < len(line.windows)
                ]
                if selected:
                    self._plan_line_window_layer(depth, selected)

    def _plan_line_window_layer(self, depth: int, lines: list[_LinePlan]) -> None:
        current = self._current_sites_by_atom()
        window_plans = [
            self._solve_window(current, depth, line.line, line.windows[depth])
            for line in lines
        ]

        max_substages = max((len(window.stages) for window in window_plans), default=0)
        for substage in range(max_substages):
            stages = tuple(
                window.stages[substage]
                for window in window_plans
                if substage < len(window.stages)
            )
            if not stages:
                continue
            if self._try_append_parallel(stages):
                self._record_stages(stages, parallel=len(stages) > 1)
                continue
            for stage in stages:
                self._append_stage(stage)
                self._record_stages((stage,), parallel=False)

    def _solve_window(
        self,
        current: list[Site],
        depth: int,
        line: int,
        swaps: SwapWindow,
    ) -> _WindowPlan:
        if self.enable_grouped_windows and len(swaps) > 1:
            try:
                return _WindowPlan(
                    stages=(self._solve_stage(current, depth, line, swaps, grouped=True),)
                )
            except Exception as exc:
                self.last_error = exc
                if not self.fallback_to_pairwise_windows:
                    raise

        stages = tuple(
            self._solve_stage(current, depth, line, (swap,), grouped=False)
            for swap in swaps
        )
        return _WindowPlan(stages=stages)

    def _solve_stage(
        self,
        current: list[Site],
        depth: int,
        line: int,
        swaps: SwapWindow,
        *,
        grouped: bool,
    ) -> _StagePlan:
        axis = self._stage_axis(current, swaps)
        cache_key = self._stage_cache_key(current, swaps, grouped, axis)
        cached = self._stage_cache.get(cache_key)
        if self.reuse_symmetric_windows and cached is not None:
            return self._translate_stage(
                cached,
                current=current,
                depth=depth,
                line=line,
                swaps=swaps,
                axis=axis,
            )

        dst = list(current)
        for atom_a, atom_b in swaps:
            dst[atom_a], dst[atom_b] = dst[atom_b], dst[atom_a]

        request = RoutingRequest(
            grid=self.request.grid,
            src=current,
            dst=dst,
            labeled=True,
        )
        scheduler = self._stage_scheduler(request, grouped=grouped)
        sequence = scheduler.plan()
        solution = scheduler.ccbs_solution
        if solution is None or not solution.found:
            raise RuntimeError("decomposed CCBS stage did not produce a solution")

        moving_atoms = sorted({atom_id for swap in swaps for atom_id in swap})
        stage = _StagePlan(
            depth=int(depth),
            line=int(line),
            swaps=tuple(swaps),
            grouped=bool(grouped),
            axis=axis,
            steps=tuple(sequence.steps),
            duration=float(sequence.total_duration()),
            solution=solution,
            start_sites=tuple((atom_id, current[atom_id]) for atom_id in moving_atoms),
        )
        if self.reuse_symmetric_windows:
            self._stage_cache[cache_key] = stage
        return stage

    def _stage_scheduler(
        self,
        request: RoutingRequest,
        *,
        grouped: bool,
    ) -> RIPACCBSCScheduler:
        time_limit = self.time_limit
        max_nodes = self.max_high_level_nodes
        if grouped:
            if self.grouped_time_limit is not None:
                time_limit = self.grouped_time_limit
            if self.grouped_max_high_level_nodes is not None:
                max_nodes = self.grouped_max_high_level_nodes
        else:
            if self.pair_time_limit is not None:
                time_limit = self.pair_time_limit
            if self.pair_max_high_level_nodes is not None:
                max_nodes = self.pair_max_high_level_nodes

        return RIPACCBSCScheduler(
            request,
            collision_dt=self.collision_dt,
            validate_final=True,
            graph_mode=self.graph_mode,
            time_limit=float(time_limit),
            max_high_level_nodes=int(max_nodes),
            high_level_order=self.high_level_order,
            ccbs_precision=self.ccbs_precision,
            max_exact_unlabeled_atoms=self.max_exact_unlabeled_atoms,
            freeze_initial_target_atoms=True,
            solver_path=self.solver_path,
            rebuild_solver=self.rebuild_solver,
            cxx=self.cxx,
            solver_timeout_padding=self.solver_timeout_padding,
        )

    # ---- emission ------------------------------------------------------

    def _try_append_parallel(self, stages: tuple[_StagePlan, ...]) -> bool:
        if not stages:
            return True

        offset = self.sequence.next_start_time()

        def build(sequence: MovingSequence) -> None:
            for stage in stages:
                for step in stage.steps:
                    sequence.append(self._shifted_step(step, offset))

        trial = self.evaluate_candidate(build, clock_cycles=1)
        if not trial.ok:
            self.last_error = trial.error
            return False
        self.commit_trial(trial)
        return True

    def _append_stage(self, stage: _StagePlan) -> None:
        offset = self.sequence.next_start_time()
        for step in stage.steps:
            self.append_step(self._shifted_step(step, offset))

    @staticmethod
    def _shifted_step(step: Step, offset: float) -> Step:
        if hasattr(step, "segments"):
            return replace(
                step,
                start_time=float(getattr(step, "start_time", 0.0)) + offset,
                segments=tuple(
                    replace(spec, start_time=float(spec.start_time) + offset)
                    for spec in getattr(step, "segments")
                ),
            )
        if hasattr(step, "start_time"):
            return replace(step, start_time=float(getattr(step, "start_time")) + offset)
        raise TypeError(f"cannot shift step without start_time: {type(step)!r}")

    # ---- helpers -------------------------------------------------------

    def _current_sites_by_atom(self) -> list[Site]:
        site_by_atom = self.sequence.final_config().site_of_atom()
        return [site_by_atom[atom_id] for atom_id in range(len(self.request.src))]

    def _record_stages(self, stages: Iterable[_StagePlan], *, parallel: bool) -> None:
        for stage in stages:
            solution = stage.solution
            self.window_stats.append(
                CCBSWindowStats(
                    depth=stage.depth,
                    line=stage.line,
                    swaps=stage.swaps,
                    grouped=stage.grouped,
                    parallel=parallel,
                    duration=stage.duration,
                    flowtime=solution.flowtime,
                    high_level_expanded=solution.high_level_expanded,
                    high_level_generated=solution.high_level_generated,
                    low_level_expanded=solution.low_level_expanded,
                    elapsed=solution.elapsed,
                    reused=stage.reused,
                )
            )

    def _combined_solution(self) -> CCBSSolution:
        return CCBSSolution(
            found=True,
            flowtime=sum(stat.flowtime for stat in self.window_stats),
            makespan=self.sequence.total_duration(),
            high_level_expanded=sum(
                stat.high_level_expanded for stat in self.window_stats
            ),
            high_level_generated=sum(
                stat.high_level_generated for stat in self.window_stats
            ),
            low_level_expanded=sum(stat.low_level_expanded for stat in self.window_stats),
            elapsed=sum(stat.elapsed for stat in self.window_stats),
        )

    def _swap_distance(self, pair: Swap, axis: Axis) -> int:
        a, b = pair
        return abs(
            self._moving_coord(self.request.src[a], axis)
            - self._moving_coord(self.request.src[b], axis)
        )

    def _stage_axis(self, current: list[Site], swaps: SwapWindow) -> Axis:
        axes = {self._single_axis(current[a], current[b]) for a, b in swaps}
        if len(axes) != 1:
            raise ValueError(f"window swaps must share one axis: {swaps}")
        axis = axes.pop()
        if axis is None:
            raise ValueError(f"window swap endpoints must be single-axis: {swaps}")
        return axis

    def _stage_cache_key(
        self,
        current: list[Site],
        swaps: SwapWindow,
        grouped: bool,
        axis: Axis,
    ) -> StageCacheKey:
        line_atoms = sorted(
            self._moving_coord(site, axis)
            for site in current
            if self._fixed_coord(site, axis) == self._fixed_coord(current[swaps[0][0]], axis)
        )
        swap_spans = tuple(
            tuple(
                sorted(
                    (
                        self._moving_coord(current[atom_a], axis),
                        self._moving_coord(current[atom_b], axis),
                    )
                )
            )
            for atom_a, atom_b in swaps
        )
        return (axis, bool(grouped), tuple(line_atoms), swap_spans)

    def _translate_stage(
        self,
        template: _StagePlan,
        *,
        current: list[Site],
        depth: int,
        line: int,
        swaps: SwapWindow,
        axis: Axis,
    ) -> _StagePlan:
        delta = int(line) - int(template.line)
        atom_map = self._stage_atom_map(
            template.start_sites,
            current=current,
            template_swaps=template.swaps,
            target_swaps=swaps,
            axis=axis,
        )
        steps = tuple(
            self._translate_window_step(step, axis=axis, delta=delta, atom_map=atom_map)
            for step in template.steps
        )
        solution = CCBSSolution(
            found=True,
            flowtime=template.solution.flowtime,
            makespan=template.solution.makespan,
            high_level_expanded=0,
            high_level_generated=0,
            low_level_expanded=0,
            elapsed=0.0,
        )
        moving_atoms = sorted({atom_id for swap in swaps for atom_id in swap})
        return _StagePlan(
            depth=int(depth),
            line=int(line),
            swaps=tuple(swaps),
            grouped=template.grouped,
            axis=axis,
            steps=steps,
            duration=template.duration,
            solution=solution,
            start_sites=tuple((atom_id, current[atom_id]) for atom_id in moving_atoms),
            reused=True,
        )

    def _stage_atom_map(
        self,
        template_start_sites: tuple[tuple[int, Site], ...],
        *,
        current: list[Site],
        template_swaps: SwapWindow,
        target_swaps: SwapWindow,
        axis: Axis,
    ) -> dict[int, int]:
        template_sites = dict(template_start_sites)
        atom_map: dict[int, int] = {}
        for template_pair, target_pair in zip(template_swaps, target_swaps):
            template_atoms = sorted(
                template_pair,
                key=lambda atom_id: self._moving_coord(template_sites[atom_id], axis),
            )
            target_atoms = sorted(
                target_pair,
                key=lambda atom_id: self._moving_coord(current[atom_id], axis),
            )
            atom_map.update(dict(zip(template_atoms, target_atoms)))
        return atom_map

    def _translate_window_step(
        self,
        step: Step,
        *,
        axis: Axis,
        delta: int,
        atom_map: dict[int, int],
    ) -> Step:
        if hasattr(step, "segments"):
            return replace(
                step,
                segments=tuple(
                    replace(
                        spec,
                        atom_id=atom_map[int(spec.atom_id)],
                        start=self._translate_site(spec.start, axis, delta),
                        end=self._translate_site(spec.end, axis, delta),
                    )
                    for spec in getattr(step, "segments")
                ),
            )
        return step

    @staticmethod
    def _translate_site(site: Site, axis: Axis, delta: int) -> Site:
        if axis == 0:
            return (int(site[0]), int(site[1]) + int(delta))
        return (int(site[0]) + int(delta), int(site[1]))

    @staticmethod
    def _single_axis(start: Site, goal: Site) -> Axis | None:
        if start[1] == goal[1] and start[0] != goal[0]:
            return 0
        if start[0] == goal[0] and start[1] != goal[1]:
            return 1
        return None

    @staticmethod
    def _moving_coord(site: Site, axis: Axis) -> int:
        return int(site[axis])

    @staticmethod
    def _fixed_coord(site: Site, axis: Axis) -> int:
        return int(site[1 - axis])


RIPACCBSDScheduler = RIPACCBSDWindowedScheduler

__all__ = [
    "CCBSWindowStats",
    "RIPACCBSDWindowedScheduler",
    "RIPACCBSDScheduler",
]
