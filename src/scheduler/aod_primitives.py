"""Shared AOD building blocks.

`LatticeMove` is the natural "what an AOD does in one shot": it captures
a set of rows and a set of columns (the addressed cartesian product),
plus their new positions after the shift. Multiple AOD schedulers will
emit lists of these and convert each one into an `AODStep` for the
sequence builder.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..movement import AODStep


Direction = str   # "left" | "right" | "up" | "down"


@dataclass(frozen=True)
class LatticeMove:
    """One paper-style 2D AOD tweezer operation.

    `old_rows`/`old_cols` identify the captured lattice sites (the AOD
    addresses their cartesian product). `new_rows`/`new_cols` have
    matching lengths and describe where each captured row/column ends
    up. Atoms move only at the (old_rows x old_cols) intersections;
    captured intersections that happen to be empty are harmless.
    """
    old_rows: tuple[int, ...]
    old_cols: tuple[int, ...]
    new_rows: tuple[int, ...]
    new_cols: tuple[int, ...]
    direction: Direction

    @property
    def rows(self) -> tuple[int, ...]:
        return self.old_rows

    @property
    def cols(self) -> tuple[int, ...]:
        return self.old_cols

    def as_dict(self) -> dict[str, list[int] | str]:
        """Same shape as the reference C++ JSON output (debug-friendly)."""
        return {
            "old_row": list(self.old_rows),
            "old_col": list(self.old_cols),
            "new_row": list(self.new_rows),
            "new_col": list(self.new_cols),
            "direction": self.direction,
        }

    def to_aod_step(self, *, start_time: float, a_max: float = 1.0) -> AODStep:
        """Wrap this lattice shift in an `AODStep` for a `Sequence`."""
        return AODStep(
            start_time=start_time,
            selected_rows=tuple(self.old_rows),
            selected_cols=tuple(self.old_cols),
            new_rows=tuple(self.new_rows),
            new_cols=tuple(self.new_cols),
            a_max=a_max,
        )
