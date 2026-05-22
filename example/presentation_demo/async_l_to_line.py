"""Three-atom asynchronous RIPA scheduling: L-shape -> vertical line.

Manually scheduled RIPA moves -- no RIPA scheduler. Mirrors panel "c"
("asynchronous scheduling") of the slide:

    atom 0:  col-down  ->  (handoff)  ->  row-right
    atom 1:  row-right ->  (handoff)  ->  col-up
    atom 2:  col-up

Each atom continues on its own timeline -- the col -> row handoff for
atom 0 fires after atom 0's first phase ends, independent of where
atoms 1 and 2 are in their own schedules. Atom 2 has a single phase and
parks at the line top while the other two are still mid-handoff.

Visual style and rendering live in `_panel_render`.

Usage:
    python example/presentation_demo/async_l_to_line.py
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
PREFIX = "async_l_to_line"
OUT_DIR = HERE / "render"


def build_sequence() -> MovingSequence:
    """Three atoms moving from an L-ish initial layout to a near-vertical line."""
    grid = Grid(N=N, d=GRID_SPACING_UM, rc=COLLISION_RADIUS_UM)
    sources = [
        (0.0, 3.0),  # atom 0: top of the L's vertical arm
        (0.0, 1.0),  # atom 1: corner of the L
        (2.0, 1.0),  # atom 2: tip of the L's horizontal arm
    ]
    initial = AtomConfig(positions=np.asarray(sources, dtype=float))
    sequence = MovingSequence(grid=grid, initial=initial)

    # Phase 1: all three atoms start at t = 0 on different channels.
    # `append_sync_batch` validates the three segments together, so the
    # order of steps inside the batch doesn't matter even though atom 1's
    # row-right ends at atom 2's initial site (2, 1).
    sequence.append_sync_batch([
        RIPAStep(0.0, atom_id=0, target=(0.0, 1.0), channel="col"),
        RIPAStep(0.0, atom_id=1, target=(2.0, 1.0), channel="row"),
        RIPAStep(0.0, atom_id=2, target=(2.0, 3.0), channel="col"),
    ])

    # Phase 2 per atom on its own timeline. Atom 2 finished its single phase
    # already (no phase 2). Atoms 0 and 1 each get a stationary HANDOFF_HOLD
    # so their col<->row ramps cross-fade with the atom at rest at the corner.
    atom0_p1_end = sequence.ensemble.atomtraj_by_id(0).final_time
    atom1_p1_end = sequence.ensemble.atomtraj_by_id(1).final_time

    sequence.append_sync_batch([
        RIPAStep(
        atom0_p1_end + HANDOFF_HOLD, atom_id=0,
        target=(2.0, 1.0), channel="row",
    ), RIPAStep(
        atom1_p1_end + HANDOFF_HOLD, atom_id=1,
        target=(2.0, 2.0), channel="col",
    )])
    return sequence


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Three-atom asynchronous RIPA L -> line demo."
    )
    parser.add_argument(
        "--no-render", action="store_true",
        help="build the sequence only; skip GIF rendering",
    )
    args = parser.parse_args()

    sequence = build_sequence()
    print(
        f"N={N}, atoms={len(sequence.ensemble.atomtrajs)}, "
        f"total_duration={_format_time_us(sequence.total_duration())}, "
        f"ramp={_format_time_us(RAMP_TIME)} ({RAMP_PROFILE.name})"
    )
    if args.no_render:
        return

    out_path = OUT_DIR / f"{PREFIX}.gif"
    render_gif(sequence, out_path)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
