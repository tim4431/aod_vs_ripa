"""Python translation of the sqrt-time AOD reconfiguration planner.

References
----------
Paper:
    Koki Aoyama, Takafumi Tomita, and Fumihiko Ino,
    "Square-root Time Atom Reconfiguration Plan for Lattice-shaped Mobile
    Tweezers", arXiv:2604.05317v1, 2026.
    https://arxiv.org/pdf/2604.05317v1

Reference implementation:
    https://github.com/kotamanegi/sqrt-time-atom-reconfigure
    downloaded locally at `lib/sqrt-time-atom-reconfigure/`.

Translated C++ files:
    - `src/solution/proposed/elementary_operations/elementary_operations.cpp`
    - `src/solution/proposed/alignment/alignment.cpp`
    - `src/solution/proposed/alignment/inverse_alignment.cpp`
    - `src/solution/proposed/problem_solving/gale_ryser.cpp`
    - `src/solution/proposed/problem_solving/arbitrary_reconfiguration.cpp`
    - `src/solution/proposed/problem_solving/util.cpp`

Algorithm notes
---------------
The paper's model plans over binary occupancy grids, not labeled atoms.
Each operation captures atoms in a row-set x column-set AOD lattice and
shifts every captured atom by one grid site left/right/up/down. The
planner decomposes arbitrary unlabeled reconfiguration into row-wise and
column-wise shuttling tasks. It first tries a two-step decomposition, and
falls back to the guaranteed three-step path using an equalized
intermediate grid plus Gale-Ryser construction.

This module preserves that planning model, then converts each translated
C++ `Move` into this repository's `AODStep`. The actual atom identities
remain whichever atoms occupy the selected sites at execution time; only
the final occupied target set is constrained.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Iterable

from ...movement import AODStep
from ...routing import Site
from ..base import SyncScheduler

GridMatrix = list[list[int]]


@dataclass(frozen=True)
class AODMoveSpec:
    old_rows: tuple[int, ...]
    old_cols: tuple[int, ...]
    new_rows: tuple[int, ...]
    new_cols: tuple[int, ...]
    name: str = ""

    @property
    def is_empty(self) -> bool:
        return len(self.old_rows) == 0 or len(self.old_cols) == 0

    def to_step(self, start_time: float) -> AODStep:
        return AODStep(
            start_time=start_time,
            selected_axis_1=self.old_rows,
            selected_axis_2=self.old_cols,
            new_axis_1=self.new_rows,
            new_axis_2=self.new_cols,
        )


@dataclass(frozen=True)
class SqrtTimePlan:
    moves: tuple[AODMoveSpec, ...]
    final_grid: tuple[tuple[int, ...], ...]
    used_three_step: bool = False


@dataclass
class _BinaryPlanState:
    grid: GridMatrix
    moves: list[AODMoveSpec] = field(default_factory=list)
    used_three_step: bool = False


@dataclass
class SqrtTimeAODScheduler(SyncScheduler):
    """Unlabeled set-routing scheduler based on the sqrt-time AOD planner."""

    use_peephole: bool = True
    prefer_two_step: bool = True
    keep_empty_cycles: bool = False
    binary_plan: SqrtTimePlan | None = field(default=None, init=False)

    def _plan(self) -> None:
        if self.request.labeled:
            raise ValueError(
                "SqrtTimeAODScheduler handles unlabeled set routing; "
                "use a direct atom scheduler for labeled requests"
            )

        source_grid = occupancy_grid_from_sites(self.request.grid.N, self.request.src)
        target_grid = occupancy_grid_from_sites(self.request.grid.N, self.request.dst)
        self.binary_plan = plan_arbitrary_reconfiguration(
            source_grid,
            target_grid,
            use_peephole=self.use_peephole,
            prefer_two_step=self.prefer_two_step,
        )

        for move in self.binary_plan.moves:
            if move.is_empty and not self.keep_empty_cycles:
                continue
            self.append_step(move.to_step(self.sequence.next_start_time()))


AODSqrtTimeScheduler = SqrtTimeAODScheduler


def occupancy_grid_from_sites(N: int, sites: Iterable[Site]) -> GridMatrix:
    grid = [[0 for _ in range(N)] for _ in range(N)]
    for i, j in sites:
        grid[int(i)][int(j)] = 1
    return grid


def plan_arbitrary_reconfiguration(
    source_grid: GridMatrix,
    target_grid: GridMatrix,
    *,
    use_peephole: bool = True,
    prefer_two_step: bool = True,
) -> SqrtTimePlan:
    """Plan an arbitrary unlabeled source->target binary reconfiguration."""
    _assert_compatible_grids(source_grid, target_grid)
    _assert_same_atom_count(source_grid, target_grid)

    if prefer_two_step:
        try:
            state = _two_step_arbitrary_reconfiguration(
                source_grid, target_grid, use_peephole=use_peephole
            )
            return _freeze_plan(state)
        except RuntimeError:
            pass

    state = _three_step_arbitrary_reconfiguration(
        source_grid, target_grid, use_peephole=use_peephole
    )
    state.used_three_step = True
    return _freeze_plan(state)


def can_construct_gale_ryser(row_sums: Iterable[int], col_sums: Iterable[int]) -> bool:
    """C++ `GaleRyser::CanConstruct` translated to Python."""
    rows = [int(v) for v in row_sums]
    cols = [int(v) for v in col_sums]
    n = len(rows)
    if n == 0 or len(cols) != n:
        return False
    if sum(rows) != sum(cols):
        return False
    if any(v < 0 or v > n for v in rows + cols):
        return False

    sorted_cols = sorted(cols, reverse=True)
    for k in range(1, n + 1):
        left_side = sum(sorted_cols[:k])
        right_side = sum(min(row_sum, k) for row_sum in rows)
        if left_side > right_side:
            return False
    return True


def construct_gale_ryser_matrix(
    row_sums: Iterable[int],
    col_sums: Iterable[int],
) -> GridMatrix:
    """Greedy matrix constructor used by the reference implementation."""
    rows = [int(v) for v in row_sums]
    cols = [int(v) for v in col_sums]
    n = len(rows)
    if n == 0 or len(cols) != n:
        return []
    if not can_construct_gale_ryser(rows, cols):
        return []

    result = [[0 for _ in range(n)] for _ in range(n)]
    heap: list[tuple[int, int]] = []
    for col_index, remaining_sum in enumerate(cols):
        if remaining_sum > 0:
            heapq.heappush(heap, (-remaining_sum, -col_index))

    for r, atoms_to_place in enumerate(rows):
        temp_columns: list[tuple[int, int]] = []
        while atoms_to_place > 0 and heap:
            neg_remaining, neg_col_index = heapq.heappop(heap)
            remaining_sum = -neg_remaining
            col_index = -neg_col_index

            result[r][col_index] = 1
            atoms_to_place -= 1
            remaining_sum -= 1
            if remaining_sum > 0:
                temp_columns.append((-remaining_sum, -col_index))

        for item in temp_columns:
            heapq.heappush(heap, item)

    return result


def _two_step_arbitrary_reconfiguration(
    source_grid: GridMatrix,
    target_grid: GridMatrix,
    *,
    use_peephole: bool,
) -> _BinaryPlanState:
    state = _BinaryPlanState(grid=_copy_grid(source_grid))

    source_row_sums = _row_sums(source_grid)
    source_col_sums = _col_sums(source_grid)
    target_row_sums = _row_sums(target_grid)
    target_col_sums = _col_sums(target_grid)

    if can_construct_gale_ryser(source_row_sums, target_col_sums):
        intermediate = construct_gale_ryser_matrix(
            source_row_sums, target_col_sums
        )
        if intermediate:
            _row_wise_reconfiguration(state, intermediate, use_peephole)
            _column_wise_reconfiguration(state, target_grid, use_peephole)
            return state

    if can_construct_gale_ryser(target_row_sums, source_col_sums):
        intermediate = construct_gale_ryser_matrix(
            target_row_sums, source_col_sums
        )
        if intermediate:
            _column_wise_reconfiguration(state, intermediate, use_peephole)
            _row_wise_reconfiguration(state, target_grid, use_peephole)
            return state

    raise RuntimeError("two-step arbitrary reconfiguration failed")


def _three_step_arbitrary_reconfiguration(
    source_grid: GridMatrix,
    target_grid: GridMatrix,
    *,
    use_peephole: bool,
) -> _BinaryPlanState:
    state = _BinaryPlanState(grid=_copy_grid(source_grid))
    equalized_grid = _create_equalized_configuration(source_grid)
    if not _verify_equalized_property(equalized_grid):
        raise ValueError("equalized property verification failed")

    intermediate_grid = construct_gale_ryser_matrix(
        _row_sums(equalized_grid), _col_sums(target_grid)
    )
    if not intermediate_grid:
        raise ValueError("Gale-Ryser construction failed")

    _column_wise_reconfiguration(state, equalized_grid, use_peephole)
    _row_wise_reconfiguration(state, intermediate_grid, use_peephole)
    _column_wise_reconfiguration(state, target_grid, use_peephole)
    return state


def _row_wise_reconfiguration(
    state: _BinaryPlanState,
    target_grid: GridMatrix,
    use_peephole: bool,
) -> None:
    if _row_sums(state.grid) != _row_sums(target_grid):
        raise RuntimeError("row sums do not match")
    _left_alignment(state)
    _inverse_left_alignment(state, target_grid, use_peephole)


def _column_wise_reconfiguration(
    state: _BinaryPlanState,
    target_grid: GridMatrix,
    use_peephole: bool,
) -> None:
    if _col_sums(state.grid) != _col_sums(target_grid):
        raise RuntimeError("column sums do not match")
    _up_alignment(state, use_peephole)
    _inverse_up_alignment(state, target_grid, use_peephole)


def _left_alignment(state: _BinaryPlanState) -> None:
    grid = state.grid
    h, w = _shape(grid)
    for c in range(w - 2, -1, -1):
        cols = list(range(c + 1, w))
        rows = [r for r in range(h) if grid[r][c] == 0]
        if rows and cols:
            state.moves.append(_left(rows, cols))

    new_grid = [[0 for _ in range(w)] for _ in range(h)]
    for r in range(h):
        counter = sum(grid[r])
        for c in range(counter):
            new_grid[r][c] = 1
    state.grid = new_grid


def _up_alignment(state: _BinaryPlanState, use_peephole: bool) -> None:
    grid = state.grid
    h, w = _shape(grid)
    for r in range(h - 2, -1, -1):
        rows = list(range(r + 1, h))
        cols = [c for c in range(w) if grid[r][c] == 0]
        if rows and cols:
            state.moves.append(_up(rows, cols))

    new_grid = [[0 for _ in range(w)] for _ in range(h)]
    for c in range(w):
        counter = sum(grid[r][c] for r in range(h))
        for r in range(counter):
            new_grid[r][c] = 1

    if use_peephole is False:
        for r in range(h - 1):
            state.moves.append(_down([r], []))
    state.grid = new_grid


def _inverse_left_alignment(
    state: _BinaryPlanState,
    target_grid: GridMatrix,
    use_peephole: bool,
) -> None:
    target = _copy_grid(target_grid)
    h, w = _shape(target)
    row_remaining_counter = _row_sums(target)

    for c in range(w - 1):
        cols = list(range(c, w - 1))
        rows = []
        for r in range(h):
            if target[r][c] == 0:
                if row_remaining_counter[r] == 0:
                    continue
                rows.append(r)
            else:
                row_remaining_counter[r] -= 1
        if use_peephole is False or (rows and cols):
            state.moves.append(_right(rows, cols))

    state.grid = target


def _inverse_up_alignment(
    state: _BinaryPlanState,
    target_grid: GridMatrix,
    use_peephole: bool,
) -> None:
    target = _copy_grid(target_grid)
    h, w = _shape(target)
    col_remaining_counter = _col_sums(target)

    for r in range(h - 1):
        rows = list(range(r, h - 1))
        cols = []
        for c in range(w):
            if target[r][c] == 0:
                if col_remaining_counter[c] == 0:
                    continue
                cols.append(c)
            else:
                col_remaining_counter[c] -= 1
        if use_peephole is False or (rows and cols):
            state.moves.append(_down(rows, cols))

    state.grid = target


def _create_equalized_configuration(source_grid: GridMatrix) -> GridMatrix:
    h, w = _shape(source_grid)
    equalized_grid = [[0 for _ in range(w)] for _ in range(h)]
    target_row = 0
    for c in range(w):
        column_atoms = sum(source_grid[r][c] for r in range(h))
        for _ in range(column_atoms):
            equalized_grid[target_row][c] = 1
            target_row = (target_row + 1) % h
    return equalized_grid


def _verify_equalized_property(equalized_grid: GridMatrix) -> bool:
    row_sums = _row_sums(equalized_grid)
    return max(row_sums, default=0) - min(row_sums, default=0) <= 1


def _left(rows: Iterable[int], cols: Iterable[int]) -> AODMoveSpec:
    old_rows = _as_tuple(rows)
    old_cols = _as_tuple(cols)
    return AODMoveSpec(
        old_rows=old_rows,
        old_cols=old_cols,
        new_rows=old_rows,
        new_cols=tuple(c - 1 for c in old_cols),
        name="left",
    )


def _right(rows: Iterable[int], cols: Iterable[int]) -> AODMoveSpec:
    old_rows = _as_tuple(rows)
    old_cols = _as_tuple(cols)
    return AODMoveSpec(
        old_rows=old_rows,
        old_cols=old_cols,
        new_rows=old_rows,
        new_cols=tuple(c + 1 for c in old_cols),
        name="right",
    )


def _up(rows: Iterable[int], cols: Iterable[int]) -> AODMoveSpec:
    old_rows = _as_tuple(rows)
    old_cols = _as_tuple(cols)
    return AODMoveSpec(
        old_rows=old_rows,
        old_cols=old_cols,
        new_rows=tuple(r - 1 for r in old_rows),
        new_cols=old_cols,
        name="up",
    )


def _down(rows: Iterable[int], cols: Iterable[int]) -> AODMoveSpec:
    old_rows = _as_tuple(rows)
    old_cols = _as_tuple(cols)
    return AODMoveSpec(
        old_rows=old_rows,
        old_cols=old_cols,
        new_rows=tuple(r + 1 for r in old_rows),
        new_cols=old_cols,
        name="down",
    )


def _row_sums(grid: GridMatrix) -> list[int]:
    return [sum(row) for row in grid]


def _col_sums(grid: GridMatrix) -> list[int]:
    h, w = _shape(grid)
    return [sum(grid[r][c] for r in range(h)) for c in range(w)]


def _copy_grid(grid: GridMatrix) -> GridMatrix:
    return [list(row) for row in grid]


def _shape(grid: GridMatrix) -> tuple[int, int]:
    if not grid:
        return 0, 0
    width = len(grid[0])
    if any(len(row) != width for row in grid):
        raise ValueError("grid rows must have equal width")
    return len(grid), width


def _assert_compatible_grids(a: GridMatrix, b: GridMatrix) -> None:
    if _shape(a) != _shape(b):
        raise ValueError("source and target grids must have the same shape")
    h, w = _shape(a)
    if h != w:
        raise ValueError("sqrt-time planner currently expects a square grid")


def _assert_same_atom_count(a: GridMatrix, b: GridMatrix) -> None:
    if sum(_row_sums(a)) != sum(_row_sums(b)):
        raise ValueError("source and target grids must contain the same atom count")


def _as_tuple(values: Iterable[int]) -> tuple[int, ...]:
    return tuple(int(v) for v in values)


def _freeze_plan(state: _BinaryPlanState) -> SqrtTimePlan:
    return SqrtTimePlan(
        moves=tuple(state.moves),
        final_grid=tuple(tuple(int(v) for v in row) for row in state.grid),
        used_three_step=state.used_three_step,
    )
