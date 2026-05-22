"""Single-atom RIPA col -> row handoff with trap-intensity ramping.

Manually scheduled RIPA moves -- no RIPA scheduler. One atom descends in
a col trap, then hands off to a row trap and slides right, mirroring
atom "1" in the asynchronous-scheduling slide.

All visual style + rendering lives in `_panel_render`; this file only
owns the sequence definition.

Usage:
    python example/presentation_demo/row_col_transport.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402

from _panel_render import (  # noqa: E402
    COLLISION_RADIUS_UM,
    GRID_SPACING_UM,
    HANDOFF_HOLD,
    RAMP_PROFILE,
    RAMP_TIME,
    render_gif,
)

from src.atom_config import AtomConfig, Grid  # noqa: E402
from src.movement import RIPAStep  # noqa: E402
from src.moving_sequence import MovingSequence  # noqa: E402
from src.visualization import _format_time_us  # noqa: E402

N = 4
PREFIX = "row_col_transport"
OUT_DIR = HERE / "render"


def build_sequence() -> MovingSequence:
    """One atom: col-down (0, 3) -> (0, 1), stationary handoff at (0, 1),
    then row-right (0, 1) -> (2, 1). The gap between col-end and row-start
    is `HANDOFF_HOLD` so the two trap ramps cross-fade with the atom at rest."""
    grid = Grid(N=N, d=GRID_SPACING_UM, rc=COLLISION_RADIUS_UM)
    initial = AtomConfig(positions=np.asarray([(0.0, 3.0)], dtype=float))
    sequence = MovingSequence(grid=grid, initial=initial)
    sequence.append(RIPAStep(0.0, atom_id=0, target=(0.0, 1.0), channel="col"))
    t_row_start = sequence.next_start_time() + HANDOFF_HOLD
    sequence.append(
        RIPAStep(t_row_start, atom_id=0, target=(2.0, 1.0), channel="row")
    )
    return sequence


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Single-atom RIPA col -> row handoff demo with ramped trap intensity."
    )
    parser.add_argument(
        "--no-render", action="store_true",
        help="build the sequence only; skip GIF rendering",
    )
    args = parser.parse_args()

    sequence = build_sequence()
    print(
        f"N={N}, total_duration={_format_time_us(sequence.total_duration())}, "
        f"ramp={_format_time_us(RAMP_TIME)} ({RAMP_PROFILE.name})"
    )
    if args.no_render:
        return

    out_path = OUT_DIR / f"{PREFIX}.gif"
    render_gif(sequence, out_path)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
