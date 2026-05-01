"""Tetris AOD scheduler and raw rearrangement planner.

The planner implements the abstract row-construction and column-compression
algorithm from Phys. Rev. Applied 19, 054032 (2023). It is independent of the
project's trajectory/timing classes; schedulers translate its moves to hardware
steps.
"""

from .aod_tetris import AODTetris, AODTetrisScheduler
from .core import (
    Site,
    TetrisColumnTarget,
    TetrisMove,
    TetrisPlan,
    TetrisPlanningError,
    TetrisPostselectionError,
    TetrisRowAssignment,
    plan_tetris,
)

__all__ = [
    "AODTetris",
    "AODTetrisScheduler",
    "Site",
    "TetrisColumnTarget",
    "TetrisMove",
    "TetrisPlan",
    "TetrisPlanningError",
    "TetrisPostselectionError",
    "TetrisRowAssignment",
    "plan_tetris",
]
