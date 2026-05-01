# Tetris AOD Scheduler

This folder contains the raw Tetris rearrangement algorithm from
`PhysRevApplied.19.054032.pdf` and the thin AOD scheduler wrapper that converts
the raw row/column moves into this repository's `AODStep` objects.

## Original Algorithm

The Tetris algorithm was proposed in Shuai Wang et al., "Accelerating the
Assembly of Defect-Free Atomic Arrays with Maximum Parallelisms", Phys. Rev.
Applied 19, 054032 (2023). The paper targets fast assembly of defect-free
two-dimensional neutral-atom arrays from stochastic loading.

The hardware idea is maximum parallelism:

- image and occupation data are decoded row by row;
- the rearrangement strategy for a row can be computed before the full image is
  read out;
- all atoms in one source row can be moved horizontally at the same time;
- after row construction succeeds, all atoms in one target column can be
  compressed vertically at the same time.

The target geometry is represented as one ordered set per target column:

```text
R_i = {target rows that must be filled in target column i}
```

For a compact `L x L` square, there are `L` target columns and each `R_i`
contains `L` rows. More general pixelated geometries use different ordered
sets.

## Core Idea

Tetris separates assembly into two phases.

The first phase is **Tetrimino construction**. The algorithm scans source rows
from top to bottom. In each source row, it assigns the loaded atoms to target
columns whose current earliest unfilled target row is most urgent. Those atoms
are then moved horizontally into the chosen columns in one parallel row move.
This builds connected vertical blocks of atoms inside the target columns.

The second phase is **Tetrimino elimination**. If every target row in every
target column has received an atom, each target column is compressed vertically
into the final target rows. Each column compression is one parallel column
move.

The algorithm is intentionally postselective. A loading pattern can fail if the
row-by-row construction cannot assign enough atoms to the target columns. In
this repository, `AODTetrisScheduler` uses the stricter `RoutingRequest` model:
the number of source and target atoms are equal, and every source atom must end
in the target set. That means a source row with more atoms than active target
columns is treated as a planning failure.

## Pseudo-Code

```text
input:
    source sites S
    target sites D

build target column sets:
    for each target column c:
        R[c] = sorted target rows in column c
        remaining[c] = copy of R[c]

group source atoms by source row:
    C[k] = sorted source columns occupied in source row k

row_moves = []
assigned_rows_by_column = empty lists

for each source row k in increasing order:
    source_cols = C[k]

    active_columns = columns c where remaining[c] is not empty
    if source_cols is larger than active_columns:
        fail or postselect this loading attempt

    priority = sort active columns by smallest remaining target row:
        priority = sorted((remaining[c][0], c) for c in active_columns)

    chosen_columns = first len(source_cols) columns from priority
    sort chosen_columns in increasing column order

    emit one horizontal row move:
        selected_rows = [k]
        selected_cols = source_cols
        new_rows = [k]
        new_cols = chosen_columns

    for each chosen column c:
        consume the smallest row from remaining[c]
        record that source row k contributes one atom to column c

if any remaining[c] is not empty:
    fail or postselect this loading attempt

column_moves = []
for each target column c:
    source_rows = sorted rows that contributed atoms to c
    target_rows = R[c]

    emit one vertical column compression move:
        selected_rows = source_rows
        selected_cols = [c]
        new_rows = target_rows
        new_cols = [c]

return row_moves followed by column_moves
```

## Files

- `core.py`: raw Tetris planner. It emits abstract row/column move specs and
  does not import trajectory or scheduler classes.
- `aod_tetris.py`: scheduler wrapper. It translates raw Tetris moves into
  `AODStep`s and lets the normal sequence validator check trajectories.
- `PhysRevApplied.19.054032.pdf`: local copy of the paper.
