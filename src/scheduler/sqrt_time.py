"""SqrtTimeAODScheduler — port of kotamanegi/sqrt-time-atom-reconfigure.

The algorithm plans a small number of AOD lattice shifts: at each step
choose row set R and column set C, then move every atom at the cartesian
product R x C by one site in some direction. Total step count is
O(sqrt(N)) on an N x N grid.

The algorithm itself works on a binary occupancy matrix (1 = atom);
this file translates between that representation and our `AtomConfig`,
and converts each `LatticeMove` into an `AODStep` for the `Sequence`.
The step count and final grid match the reference C++ implementation;
total wall-clock time depends on `a_max`.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field
from typing import Iterable, Sequence as TypingSequence

from ..atoms import AtomConfig
from ..atom_trajectory import AtomEnsemble
from ..routing import RoutingRequest, Site
from ..sequence import Sequence
from .aod_primitives import LatticeMove
from .base import AODScheduler


GridMatrix = list[list[int]]


# --- public scheduler -------------------------------------------------------

@dataclass
class _PlanState:
    grid: GridMatrix
    moves: list[LatticeMove] = field(default_factory=list)
    is_three_step_strategy_enabled: bool = False


@dataclass
class SqrtTimeAODScheduler(AODScheduler):
    """Sqrt-time AOD lattice scheduler.

    `schedule(request)` solves arbitrary unlabeled set->set routing.
    `schedule_grid(initial)` solves the paper's square-grid formation
    variant, gathering atoms into the largest defect-free square.
    """

    a_max: float = 1.0
    inter_step_gap: float = 0.0
    use_peephole: bool = True
    prefer_two_step: bool = True
    keep_empty_steps: bool = False
    last_lattice_moves: list[LatticeMove] = field(default_factory=list, init=False)

    def schedule(self, request: RoutingRequest) -> Sequence:
        if request.labeled:
            raise NotImplementedError(
                "The sqrt-time algorithm handles unlabeled atoms; labeled "
                "pairings need an additional identity-preserving assignment."
            )

        source_grid = _config_to_grid(request.initial)
        target_grid = _target_grid(request.initial.grid.N, request.target_sites())
        if _count_atoms(source_grid) != _count_atoms(target_grid):
            raise ValueError("source and target must have the same atom count")

        if source_grid == target_grid:
            self.last_lattice_moves = []
            return Sequence(initial=request.initial.copy(), inter_step_gap=self.inter_step_gap)

        state = (
            _hybrid_arbitrary_reconfiguration(source_grid, target_grid, self.use_peephole)
            if self.prefer_two_step
            else _three_step_arbitrary_reconfiguration(source_grid, target_grid, self.use_peephole)
        )
        self.last_lattice_moves = list(state.moves)
        return self._moves_to_sequence(request.initial, state.moves, expected_grid=target_grid)

    def schedule_to_targets(self, initial: AtomConfig, targets: Iterable[Site]) -> Sequence:
        return self.schedule(RoutingRequest(initial=initial, targets=set(targets)))

    def schedule_grid(self, initial: AtomConfig) -> Sequence:
        """Gather atoms into the largest top-left square grid possible."""
        source_grid = _config_to_grid(initial)
        state = _grid_reconfiguration(source_grid, self.use_peephole)
        self.last_lattice_moves = list(state.moves)
        return self._moves_to_sequence(initial, state.moves, expected_grid=state.grid)

    def lattice_moves(self, request: RoutingRequest) -> list[LatticeMove]:
        """Return only the row/column lattice operations for a request."""
        self.schedule(request)
        return list(self.last_lattice_moves)

    def _moves_to_sequence(
        self,
        initial: AtomConfig,
        moves: TypingSequence[LatticeMove],
        *,
        expected_grid: GridMatrix | None = None,
    ) -> Sequence:
        seq = Sequence(initial=initial.copy(), inter_step_gap=self.inter_step_gap)
        for move in moves:
            for ni in move.new_rows:
                if not _site_in_bounds(ni, 0, initial.grid.N):
                    raise ValueError(f"lattice move leaves grid: row {ni}")
            for nj in move.new_cols:
                if not _site_in_bounds(0, nj, initial.grid.N):
                    raise ValueError(f"lattice move leaves grid: col {nj}")
            # Skip lattice ops whose selected (row, col) intersection is
            # empty for our actual atom config — the algorithm emits some
            # of these because it works on a generic occupancy template.
            if not self.keep_empty_steps and not _move_affects_any_atom(move, seq.ensemble):
                continue
            seq.append(move.to_aod_step(start_time=seq.next_start_time(), a_max=self.a_max))

        if expected_grid is not None:
            final_grid = _config_to_grid(seq.final_config())
            if final_grid != expected_grid:
                raise RuntimeError("generated lattice moves did not produce the planned target grid")
        return seq


# Short aliases.
SqrtTimeAtomScheduler = SqrtTimeAODScheduler
SqrtTimeScheduler = SqrtTimeAODScheduler


# --- LatticeMove constructors (one-site shifts in each direction) ------------

def _left(rows: Iterable[int], cols: Iterable[int]) -> LatticeMove:
    r, c = tuple(rows), tuple(cols)
    return LatticeMove(r, c, r, tuple(j - 1 for j in c), "left")


def _right(rows: Iterable[int], cols: Iterable[int]) -> LatticeMove:
    r, c = tuple(rows), tuple(cols)
    return LatticeMove(r, c, r, tuple(j + 1 for j in c), "right")


def _up(rows: Iterable[int], cols: Iterable[int]) -> LatticeMove:
    r, c = tuple(rows), tuple(cols)
    return LatticeMove(r, c, tuple(i - 1 for i in r), c, "up")


def _down(rows: Iterable[int], cols: Iterable[int]) -> LatticeMove:
    r, c = tuple(rows), tuple(cols)
    return LatticeMove(r, c, tuple(i + 1 for i in r), c, "down")


def _append_if_useful(state: _PlanState, move: LatticeMove, *, force: bool = False) -> None:
    if force or (move.old_rows and move.old_cols):
        state.moves.append(move)


# --- alignments (forward + inverse) -----------------------------------------

def _left_alignment(state: _PlanState) -> None:
    h, w = _shape(state.grid)
    for c in range(w - 2, -1, -1):
        cols = range(c + 1, w)
        rows = [r for r in range(h) if state.grid[r][c] == 0]
        _append_if_useful(state, _left(rows, cols))
    state.grid = [[1 if c < sum(row) else 0 for c in range(w)] for row in state.grid]


def _right_alignment(state: _PlanState) -> None:
    h, w = _shape(state.grid)
    for c in range(1, w):
        cols = range(c)
        rows = [r for r in range(h) if state.grid[r][c] == 0]
        _append_if_useful(state, _right(rows, cols))
    new_grid = _zero_grid(h, w)
    for r, row in enumerate(state.grid):
        row_atoms = sum(row)
        for c in range(w - row_atoms, w):
            new_grid[r][c] = 1
    state.grid = new_grid


def _up_alignment(state: _PlanState, use_peephole: bool = True) -> None:
    h, w = _shape(state.grid)
    for r in range(h - 2, -1, -1):
        rows = range(r + 1, h)
        cols = [c for c in range(w) if state.grid[r][c] == 0]
        _append_if_useful(state, _up(rows, cols))
    new_grid = _zero_grid(h, w)
    for c in range(w):
        col_atoms = sum(state.grid[r][c] for r in range(h))
        for r in range(col_atoms):
            new_grid[r][c] = 1
    if not use_peephole:
        for r in range(h - 1):
            _append_if_useful(state, _down([r], []), force=True)
    state.grid = new_grid


def _down_alignment(state: _PlanState) -> None:
    h, w = _shape(state.grid)
    for r in range(1, h):
        rows = range(r)
        cols = [c for c in range(w) if state.grid[r][c] == 0]
        _append_if_useful(state, _down(rows, cols))
    new_grid = _zero_grid(h, w)
    for c in range(w):
        col_atoms = sum(state.grid[r][c] for r in range(h))
        for r in range(h - col_atoms, h):
            new_grid[r][c] = 1
    state.grid = new_grid


def _inverse_left_alignment(state: _PlanState, target_grid: GridMatrix,
                            use_peephole: bool = True) -> None:
    h, w = _shape(state.grid)
    remaining = _row_sums(target_grid)
    for c in range(w - 1):
        cols = range(c, w - 1)
        rows = []
        for r in range(h):
            if target_grid[r][c] == 0:
                if remaining[r] == 0:
                    continue
                rows.append(r)
            else:
                remaining[r] -= 1
        _append_if_useful(state, _right(rows, cols), force=not use_peephole)
    state.grid = _copy_grid(target_grid)


def _inverse_right_alignment(state: _PlanState, target_grid: GridMatrix) -> None:
    h, w = _shape(state.grid)
    remaining = _row_sums(target_grid)
    for c in range(w - 1, 0, -1):
        cols = range(1, c + 1)
        rows = []
        for r in range(h):
            if target_grid[r][c] == 0:
                if remaining[r] == 0:
                    continue
                rows.append(r)
            else:
                remaining[r] -= 1
        _append_if_useful(state, _left(rows, cols))
    state.grid = _copy_grid(target_grid)


def _inverse_up_alignment(state: _PlanState, target_grid: GridMatrix) -> None:
    h, w = _shape(state.grid)
    remaining = _col_sums(target_grid)
    for r in range(h - 1):
        rows = range(r, h - 1)
        cols = []
        for c in range(w):
            if target_grid[r][c] == 0:
                if remaining[c] == 0:
                    continue
                cols.append(c)
            else:
                remaining[c] -= 1
        _append_if_useful(state, _down(rows, cols))
    state.grid = _copy_grid(target_grid)


def _inverse_down_alignment(state: _PlanState, target_grid: GridMatrix) -> None:
    h, w = _shape(state.grid)
    remaining = _col_sums(target_grid)
    for r in range(h - 1, 0, -1):
        rows = range(1, r + 1)
        cols = []
        for c in range(w):
            if target_grid[r][c] == 0:
                if remaining[c] == 0:
                    continue
                cols.append(c)
            else:
                remaining[c] -= 1
        _append_if_useful(state, _up(rows, cols))
    state.grid = _copy_grid(target_grid)


# --- reconfigurations -------------------------------------------------------

def _row_wise_reconfiguration(state: _PlanState, target_grid: GridMatrix,
                              use_peephole: bool = True) -> None:
    if _row_sums(state.grid) != _row_sums(target_grid):
        raise RuntimeError("row sums do not match")
    _left_alignment(state)
    _inverse_left_alignment(state, target_grid, use_peephole)


def _column_wise_reconfiguration(state: _PlanState, target_grid: GridMatrix,
                                 use_peephole: bool = True) -> None:
    if _col_sums(state.grid) != _col_sums(target_grid):
        raise RuntimeError("column sums do not match")
    _up_alignment(state, use_peephole)
    _inverse_up_alignment(state, target_grid)


def _partial_row_wise_reconfiguration(state: _PlanState, target_grid: GridMatrix,
                                      columns_to_configure: int,
                                      use_peephole: bool = True) -> None:
    h, w = _shape(state.grid)
    columns_to_configure = min(columns_to_configure, w)
    for r in range(h):
        if sum(state.grid[r]) < sum(target_grid[r][:columns_to_configure]):
            raise RuntimeError("not enough atoms in source row for partial reconfiguration")
    _left_alignment(state)
    partial_target = _zero_grid(h, w)
    for r in range(h):
        for c in range(columns_to_configure):
            partial_target[r][c] = target_grid[r][c]
        atoms_used = sum(partial_target[r][:columns_to_configure])
        remaining = sum(state.grid[r]) - atoms_used
        for c in range(w - 1, columns_to_configure - 1, -1):
            if remaining <= 0:
                break
            partial_target[r][c] = 1
            remaining -= 1
    _inverse_left_alignment(state, partial_target, use_peephole)


def _partial_column_wise_reconfiguration(state: _PlanState, target_grid: GridMatrix,
                                         rows_to_configure: int) -> None:
    h, w = _shape(state.grid)
    rows_to_configure = min(rows_to_configure, h)
    for c in range(w):
        if sum(state.grid[r][c] for r in range(h)) < sum(target_grid[r][c] for r in range(rows_to_configure)):
            raise RuntimeError("not enough atoms in source column for partial reconfiguration")
    _up_alignment(state)
    partial_target = _zero_grid(h, w)
    for c in range(w):
        for r in range(rows_to_configure):
            partial_target[r][c] = target_grid[r][c]
        atoms_used = sum(partial_target[r][c] for r in range(rows_to_configure))
        remaining = sum(state.grid[r][c] for r in range(h)) - atoms_used
        for r in range(h - 1, rows_to_configure - 1, -1):
            if remaining <= 0:
                break
            partial_target[r][c] = 1
            remaining -= 1
    _inverse_up_alignment(state, partial_target)


def _three_step_arbitrary_reconfiguration(source_grid: GridMatrix, target_grid: GridMatrix,
                                          use_peephole: bool = True) -> _PlanState:
    _require_same_shape(source_grid, target_grid)
    if _count_atoms(source_grid) != _count_atoms(target_grid):
        raise ValueError("source and target must have the same atom count")
    state = _PlanState(grid=_copy_grid(source_grid))
    equalized_grid = _create_equalized_configuration(source_grid)
    if not _verify_equalized_property(equalized_grid):
        raise ValueError("equalized property verification failed")
    intermediate_grid = _construct_matrix(_row_sums(equalized_grid), _col_sums(target_grid))
    if not intermediate_grid:
        raise ValueError("Gale-Ryser construction failed")
    _column_wise_reconfiguration(state, equalized_grid, use_peephole)
    _row_wise_reconfiguration(state, intermediate_grid, use_peephole)
    _column_wise_reconfiguration(state, target_grid, use_peephole)
    state.is_three_step_strategy_enabled = True
    return state


def _two_step_arbitrary_reconfiguration(source_grid: GridMatrix, target_grid: GridMatrix,
                                        use_peephole: bool = True) -> _PlanState:
    _require_same_shape(source_grid, target_grid)
    if _count_atoms(source_grid) != _count_atoms(target_grid):
        raise ValueError("source and target must have the same atom count")
    state = _PlanState(grid=_copy_grid(source_grid))
    s_row, s_col = _row_sums(source_grid), _col_sums(source_grid)
    t_row, t_col = _row_sums(target_grid), _col_sums(target_grid)

    if _can_construct(s_row, t_col):
        inter = _construct_matrix(s_row, t_col)
        if inter:
            _row_wise_reconfiguration(state, inter, use_peephole)
            _column_wise_reconfiguration(state, target_grid, use_peephole)
            return state
    if _can_construct(t_row, s_col):
        inter = _construct_matrix(t_row, s_col)
        if inter:
            _column_wise_reconfiguration(state, inter, use_peephole)
            _row_wise_reconfiguration(state, target_grid, use_peephole)
            return state
    raise RuntimeError("two-step arbitrary reconfiguration failed")


def _hybrid_arbitrary_reconfiguration(source_grid: GridMatrix, target_grid: GridMatrix,
                                      use_peephole: bool = True) -> _PlanState:
    try:
        return _two_step_arbitrary_reconfiguration(source_grid, target_grid, use_peephole)
    except (RuntimeError, ValueError):
        return _three_step_arbitrary_reconfiguration(source_grid, target_grid, use_peephole)


def _grid_reconfiguration(source_grid: GridMatrix, use_peephole: bool = True) -> _PlanState:
    h, w = _shape(source_grid)
    total_atoms = _count_atoms(source_grid)
    side = int(math.sqrt(total_atoms))
    if side == 0:
        return _PlanState(grid=_copy_grid(source_grid))
    if side > h or side > w:
        raise ValueError("not enough grid area to form the square target")

    available_atoms = sum(min(sum(source_grid[r]), side) for r in range(h))
    if available_atoms < side * side:
        target_grid = _zero_grid(h, w)
        placed = 0
        for r in range(side):
            for c in range(side):
                target_grid[r][c] = 1
                placed += 1
        for r in range(h):
            for c in range(w):
                if placed >= total_atoms:
                    break
                if target_grid[r][c] == 0:
                    target_grid[r][c] = 1
                    placed += 1
            if placed >= total_atoms:
                break
        return _three_step_arbitrary_reconfiguration(source_grid, target_grid, use_peephole)

    intermediate_grid = _zero_grid(h, w)
    target_col = 0
    for r in range(h):
        row_atoms = sum(source_grid[r])
        atoms_to_use = min(row_atoms, side)
        for _ in range(atoms_to_use):
            intermediate_grid[r][target_col] = 1
            target_col = (target_col + 1) % side
        remaining = row_atoms - atoms_to_use
        for c in range(side, w):
            if remaining <= 0:
                break
            intermediate_grid[r][c] = 1
            remaining -= 1

    state = _PlanState(grid=_copy_grid(source_grid))
    _partial_row_wise_reconfiguration(state, intermediate_grid, side, use_peephole)
    _up_alignment(state, use_peephole)
    return state


# --- Gale-Ryser construction utilities --------------------------------------

def _can_construct(row_sums: list[int], col_sums: list[int]) -> bool:
    n = len(row_sums)
    if n == 0 or len(col_sums) != n:
        return False
    if any(v < 0 or v > n for v in row_sums + col_sums):
        return False
    if sum(row_sums) != sum(col_sums):
        return False
    sorted_col = sorted(col_sums, reverse=True)
    for k in range(1, n + 1):
        if sum(sorted_col[:k]) > sum(min(rs, k) for rs in row_sums):
            return False
    return True


def _construct_matrix(row_sums: list[int], col_sums: list[int]) -> GridMatrix:
    n = len(row_sums)
    if n == 0 or len(col_sums) != n or not _can_construct(row_sums, col_sums):
        return []
    result = _zero_grid(n, n)
    heap = [(-rem, -c) for c, rem in enumerate(col_sums) if rem > 0]
    heapq.heapify(heap)
    for r, atoms_to_place in enumerate(row_sums):
        updated: list[tuple[int, int]] = []
        while atoms_to_place > 0 and heap:
            neg_rem, neg_c = heapq.heappop(heap)
            rem, c = -neg_rem, -neg_c
            result[r][c] = 1
            atoms_to_place -= 1
            rem -= 1
            if rem > 0:
                updated.append((-rem, -c))
        if atoms_to_place != 0:
            return []
        for item in updated:
            heapq.heappush(heap, item)
    if _row_sums(result) != row_sums or _col_sums(result) != col_sums:
        return []
    return result


def _create_equalized_configuration(source_grid: GridMatrix) -> GridMatrix:
    h, w = _shape(source_grid)
    out = _zero_grid(h, w)
    target_row = 0
    for c in range(w):
        atoms_in_col = sum(source_grid[r][c] for r in range(h))
        for _ in range(atoms_in_col):
            out[target_row][c] = 1
            target_row = (target_row + 1) % h
    return out


def _verify_equalized_property(equalized_grid: GridMatrix) -> bool:
    rs = _row_sums(equalized_grid)
    return all(abs(a - b) <= 1 for a in rs for b in rs)


# --- AtomConfig <-> binary grid bridge --------------------------------------

def _config_to_grid(cfg: AtomConfig) -> GridMatrix:
    grid = _zero_grid(cfg.grid.N, cfg.grid.N)
    for pos in cfg.positions:
        i, j = _integer_site(pos, cfg.grid.N)
        if grid[i][j]:
            raise ValueError(f"duplicate atom at {(i, j)}")
        grid[i][j] = 1
    return grid


def _target_grid(n: int, targets: set[Site]) -> GridMatrix:
    grid = _zero_grid(n, n)
    for site in targets:
        i, j = _integer_site(site, n)
        if grid[i][j]:
            raise ValueError(f"duplicate target at {(i, j)}")
        grid[i][j] = 1
    return grid


def _integer_site(pos: Iterable[float], n: int, *, tol: float = 1e-9) -> tuple[int, int]:
    i_f, j_f = pos
    i, j = int(round(float(i_f))), int(round(float(j_f)))
    if abs(float(i_f) - i) > tol or abs(float(j_f) - j) > tol:
        raise ValueError(f"position {tuple(pos)} is not on an integer lattice site")
    if not (0 <= i < n and 0 <= j < n):
        raise ValueError(f"site {(i, j)} is outside a {n}x{n} grid")
    return i, j


def _site_in_bounds(i: float, j: float, n: int, *, tol: float = 1e-9) -> bool:
    return -tol <= i <= n - 1 + tol and -tol <= j <= n - 1 + tol


def _move_affects_any_atom(move: LatticeMove, ensemble: AtomEnsemble) -> bool:
    """Does any atom currently sit at a (row, col) selected by `move`?"""
    sel_r = set(move.old_rows)
    sel_c = set(move.old_cols)
    t = ensemble.total_duration()
    for atom in ensemble.atoms:
        i, j = atom.resting_position_at(t)
        if i in sel_r and j in sel_c:
            return True
    return False


# --- generic grid utilities -------------------------------------------------

def _shape(grid: GridMatrix) -> tuple[int, int]:
    if not grid or not grid[0]:
        raise ValueError("grid must be non-empty")
    width = len(grid[0])
    if any(len(row) != width for row in grid):
        raise ValueError("grid rows must have equal length")
    return len(grid), width


def _require_same_shape(a: GridMatrix, b: GridMatrix) -> None:
    if _shape(a) != _shape(b):
        raise ValueError("source and target grids must have the same shape")


def _zero_grid(h: int, w: int) -> GridMatrix:
    return [[0 for _ in range(w)] for _ in range(h)]


def _copy_grid(grid: GridMatrix) -> GridMatrix:
    return [list(row) for row in grid]


def _row_sums(grid: GridMatrix) -> list[int]:
    return [sum(row) for row in grid]


def _col_sums(grid: GridMatrix) -> list[int]:
    h, w = _shape(grid)
    return [sum(grid[r][c] for r in range(h)) for c in range(w)]


def _count_atoms(grid: GridMatrix) -> int:
    return sum(sum(row) for row in grid)
