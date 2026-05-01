"""Core abstract realization of the Tetris atom-rearrangement algorithm.

Reference
---------
Shuai Wang et al.,
"Accelerating the Assembly of Defect-Free Atomic Arrays with Maximum
Parallelisms", Phys. Rev. Applied 19, 054032 (2023).

The paper describes a target geometry as ordered sets R_i, one set per target
column, where each set stores the target rows that must be filled in that
column. The algorithm has two physical phases:

1. Tetrimino construction: process loaded reservoir rows in order. For the
   current row, assign its atoms to the target columns whose smallest unfilled
   target row is most urgent, then move those atoms horizontally in one
   parallel row move.
2. Tetrimino elimination: once every target site has been assigned an atom,
   compress each target column vertically in one parallel column move.

This module intentionally emits only integer-grid move specifications. It does
not know about AOD timing, collision radii, waveform generation, or atom
identity. The scheduler layer is responsible for translating these abstract
operations into hardware-specific steps.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

Site = tuple[int, int]
TetrisMoveKind = Literal["row", "column"]


class TetrisPlanningError(ValueError):
    """Raised when the Tetris construction cannot assign the loaded atoms."""


class TetrisPostselectionError(TetrisPlanningError):
    """Raised when the postselection check finds unfilled target sites."""


@dataclass(frozen=True)
class TetrisMove:
    """One abstract parallel row or column move.

    The field names mirror `src.movement.AODStep`, but this dataclass is kept
    independent so the raw planner can be reused without importing `src`.
    """

    kind: TetrisMoveKind
    selected_rows: tuple[int, ...]
    selected_cols: tuple[int, ...]
    new_rows: tuple[int, ...]
    new_cols: tuple[int, ...]
    source_sites: tuple[Site, ...] = ()
    target_sites: tuple[Site, ...] = ()

    def __post_init__(self) -> None:
        if self.kind not in ("row", "column"):
            raise ValueError(f"unknown Tetris move kind {self.kind!r}")
        if len(self.selected_rows) != len(self.new_rows):
            raise ValueError("selected_rows and new_rows must have the same length")
        if len(self.selected_cols) != len(self.new_cols):
            raise ValueError("selected_cols and new_cols must have the same length")
        if self.kind == "row" and len(self.selected_rows) != 1:
            raise ValueError("row Tetris moves must address exactly one row")
        if self.kind == "column" and len(self.selected_cols) != 1:
            raise ValueError("column Tetris moves must address exactly one column")

    @property
    def is_empty(self) -> bool:
        return len(self.selected_rows) == 0 or len(self.selected_cols) == 0

    @property
    def is_noop(self) -> bool:
        return (
            self.selected_rows == self.new_rows
            and self.selected_cols == self.new_cols
        )

    @property
    def max_grid_displacement(self) -> int:
        """Largest single-atom grid displacement in this parallel move."""
        if self.is_empty:
            return 0
        if self.kind == "row":
            return max(
                abs(new_col - old_col)
                for old_col, new_col in zip(self.selected_cols, self.new_cols)
            )
        return max(
            abs(new_row - old_row)
            for old_row, new_row in zip(self.selected_rows, self.new_rows)
        )


@dataclass(frozen=True)
class TetrisRowAssignment:
    """Bookkeeping for one Tetrimino-construction row."""

    source_row: int
    source_cols: tuple[int, ...]
    target_cols: tuple[int, ...]
    consumed_target_rows: tuple[int, ...]


@dataclass(frozen=True)
class TetrisColumnTarget:
    """Target rows for one target column, equivalent to one paper set R_i."""

    col: int
    rows: tuple[int, ...]


@dataclass(frozen=True)
class TetrisPlan:
    """Complete raw Tetris plan."""

    moves: tuple[TetrisMove, ...]
    row_moves: tuple[TetrisMove, ...]
    column_moves: tuple[TetrisMove, ...]
    row_assignments: tuple[TetrisRowAssignment, ...]
    target_columns: tuple[TetrisColumnTarget, ...]
    source_sites: tuple[Site, ...]
    target_sites: tuple[Site, ...]
    assigned_source_sites: tuple[Site, ...]
    intermediate_sites: tuple[Site, ...]

    @property
    def parallel_displacements(self) -> int:
        """Paper metric: sum of the largest displacement in each move."""
        return sum(move.max_grid_displacement for move in self.moves)


def plan_tetris(
    source_sites: Iterable[Site],
    target_sites: Iterable[Site],
    *,
    allow_extra_atoms: bool = False,
    include_noop_moves: bool = True,
) -> TetrisPlan:
    """Plan source-to-target set assembly with the Tetris algorithm.

    Coordinates are ordinary `(row, col)` integer sites. The returned row moves
    keep `row` fixed and change columns; column-compression moves keep `col`
    fixed and change rows.

    `allow_extra_atoms=False` matches this repository's `RoutingRequest`
    contract, where every source atom must end on a target site. Setting it to
    true exposes the paper-style postselection model where surplus loaded atoms
    may be ignored after enough atoms have been assigned.
    """

    sources = _normalize_sites(source_sites, label="source")
    targets = _normalize_sites(target_sites, label="target")

    if len(sources) < len(targets):
        raise TetrisPostselectionError(
            "Tetris postselection failed: fewer source atoms than target sites"
        )
    if not allow_extra_atoms and len(sources) != len(targets):
        raise TetrisPlanningError(
            "Tetris planning requires the same number of source and target "
            "sites unless allow_extra_atoms=True"
        )

    target_rows_by_col = _target_rows_by_col(targets)
    remaining_rows_by_col = {
        col: list(rows) for col, rows in target_rows_by_col.items()
    }
    source_cols_by_row = _source_cols_by_row(sources)

    assigned_rows_by_col: dict[int, list[int]] = {
        col: [] for col in target_rows_by_col
    }
    assigned_sources: list[Site] = []
    intermediate_sites: list[Site] = []
    row_assignments: list[TetrisRowAssignment] = []
    row_moves: list[TetrisMove] = []
    all_moves: list[TetrisMove] = []

    for source_row in sorted(source_cols_by_row):
        source_cols = tuple(source_cols_by_row[source_row])
        if not source_cols:
            continue

        priority = _target_column_priority(remaining_rows_by_col)
        if not priority:
            if allow_extra_atoms:
                continue
            raise TetrisPlanningError(
                f"source row {source_row} still has atoms after all target "
                "sites have been assigned"
            )

        assign_count = min(len(source_cols), len(priority))
        if assign_count < len(source_cols) and not allow_extra_atoms:
            raise TetrisPlanningError(
                f"source row {source_row} has {len(source_cols)} atoms, but "
                f"only {len(priority)} target columns can accept one atom in "
                "this Tetris row move"
            )
        if assign_count == 0:
            continue

        selected_source_cols = source_cols[:assign_count]
        selected_target_cols_by_urgency = [
            col for _, col in priority[:assign_count]
        ]

        consumed_target_row_by_col: dict[int, int] = {}
        for target_col in selected_target_cols_by_urgency:
            target_row = remaining_rows_by_col[target_col].pop(0)
            consumed_target_row_by_col[target_col] = target_row
            assigned_rows_by_col[target_col].append(source_row)

        selected_target_cols = tuple(sorted(selected_target_cols_by_urgency))
        consumed_target_rows = tuple(
            consumed_target_row_by_col[col] for col in selected_target_cols
        )

        assignment = TetrisRowAssignment(
            source_row=source_row,
            source_cols=selected_source_cols,
            target_cols=selected_target_cols,
            consumed_target_rows=consumed_target_rows,
        )
        row_assignments.append(assignment)

        move = TetrisMove(
            kind="row",
            selected_rows=(source_row,),
            selected_cols=selected_source_cols,
            new_rows=(source_row,),
            new_cols=selected_target_cols,
            source_sites=tuple((source_row, col) for col in selected_source_cols),
            target_sites=tuple((source_row, col) for col in selected_target_cols),
        )
        if include_noop_moves or not move.is_noop:
            row_moves.append(move)
            all_moves.append(move)

        for source_col, target_col in zip(selected_source_cols, selected_target_cols):
            assigned_sources.append((source_row, source_col))
            intermediate_sites.append((source_row, target_col))

    missing_targets = _remaining_target_sites(remaining_rows_by_col)
    if missing_targets:
        raise TetrisPostselectionError(
            "Tetris postselection failed: unfilled target sites "
            f"{missing_targets}"
        )

    if not allow_extra_atoms:
        assigned_source_set = set(assigned_sources)
        source_set = set(sources)
        if assigned_source_set != source_set:
            unused = tuple(sorted(source_set - assigned_source_set))
            raise TetrisPlanningError(
                "Tetris construction left source atoms unassigned: "
                f"{unused}"
            )

    column_moves: list[TetrisMove] = []
    for target_col, target_rows in target_rows_by_col.items():
        source_rows = tuple(sorted(assigned_rows_by_col[target_col]))
        if len(source_rows) != len(target_rows):
            raise TetrisPostselectionError(
                f"column {target_col} has {len(source_rows)} assigned atoms "
                f"for {len(target_rows)} target rows"
            )

        move = TetrisMove(
            kind="column",
            selected_rows=source_rows,
            selected_cols=(target_col,),
            new_rows=target_rows,
            new_cols=(target_col,),
            source_sites=tuple((row, target_col) for row in source_rows),
            target_sites=tuple((row, target_col) for row in target_rows),
        )
        if include_noop_moves or not move.is_noop:
            column_moves.append(move)
            all_moves.append(move)

    return TetrisPlan(
        moves=tuple(all_moves),
        row_moves=tuple(row_moves),
        column_moves=tuple(column_moves),
        row_assignments=tuple(row_assignments),
        target_columns=tuple(
            TetrisColumnTarget(col=col, rows=rows)
            for col, rows in target_rows_by_col.items()
        ),
        source_sites=sources,
        target_sites=targets,
        assigned_source_sites=tuple(assigned_sources),
        intermediate_sites=tuple(intermediate_sites),
    )


def _normalize_sites(sites: Iterable[Site], *, label: str) -> tuple[Site, ...]:
    normalized = tuple(sorted((int(row), int(col)) for row, col in sites))
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{label} sites contain duplicates")
    return normalized


def _target_rows_by_col(targets: tuple[Site, ...]) -> dict[int, tuple[int, ...]]:
    by_col: dict[int, list[int]] = {}
    for row, col in targets:
        by_col.setdefault(col, []).append(row)
    return {
        col: tuple(sorted(rows))
        for col, rows in sorted(by_col.items(), key=lambda item: item[0])
    }


def _source_cols_by_row(sources: tuple[Site, ...]) -> dict[int, tuple[int, ...]]:
    by_row: dict[int, list[int]] = {}
    for row, col in sources:
        by_row.setdefault(row, []).append(col)
    return {
        row: tuple(sorted(cols))
        for row, cols in sorted(by_row.items(), key=lambda item: item[0])
    }


def _target_column_priority(
    remaining_rows_by_col: dict[int, list[int]],
) -> list[tuple[int, int]]:
    """Return `(smallest_unfilled_target_row, col)` sorted by urgency."""
    return sorted(
        (rows[0], col)
        for col, rows in remaining_rows_by_col.items()
        if rows
    )


def _remaining_target_sites(
    remaining_rows_by_col: dict[int, list[int]],
) -> tuple[Site, ...]:
    return tuple(
        sorted(
            (row, col)
            for col, rows in remaining_rows_by_col.items()
            for row in rows
        )
    )
