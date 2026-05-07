"""AOD scheduler using the Tetris rearrangement algorithm.

This is a thin hardware wrapper around `src.scheduler.tetris.plan_tetris`, the raw
abstract implementation of the algorithm from Phys. Rev. Applied 19, 054032
(2023). The raw plan performs Tetrimino construction with horizontal row moves
followed by Tetrimino elimination with vertical column-compression moves. This
scheduler translates each abstract move into one `AODStep`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .core import TetrisPlan, plan_tetris

from ...movement import AODStep
from ..base import SyncScheduler


@dataclass
class AODTetrisScheduler(SyncScheduler):
    """Unlabeled AOD set-routing scheduler based on the Tetris algorithm."""

    keep_noop_moves: bool = False
    tetris_plan: TetrisPlan | None = field(default=None, init=False)

    def _plan(self) -> None:
        if self.request.labeled:
            raise ValueError(
                "AODTetrisScheduler handles unlabeled set routing; use a "
                "direct atom scheduler for labeled requests"
            )

        self.tetris_plan = plan_tetris(
            self.request.src,
            self.request.dst,
            allow_extra_atoms=False,
            include_noop_moves=True,
        )

        for move in self.tetris_plan.moves:
            if move.is_empty:
                continue
            if move.is_noop and not self.keep_noop_moves:
                continue
            self.append_step(
                AODStep(
                    start_time=self.sequence.next_start_time(),
                    selected_axis_1=move.selected_rows,
                    selected_axis_2=move.selected_cols,
                    new_axis_1=move.new_rows,
                    new_axis_2=move.new_cols,
                )
            )


AODTetris = AODTetrisScheduler
