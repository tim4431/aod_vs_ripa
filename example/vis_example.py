"""Three-atom hand-routed demo for `visualization_stack_time`.

Implements the exact sequence the user specified on a 4x4 grid:

    t=0     atom 0 = (0,3)  atom 1 = (0,1)  atom 2 = (2,1)
    t=t0    atom 0 = (0,1)  atom 1 = (2,1)  atom 2 = (2,3)
    t=2*t0  atom 0 = (2,1)  atom 1 = (2,2)  atom 2 = (2,3)

Phase 1 (0..t0): atom 0 col-down, atom 1 row-right, atom 2 col-up — all
moving in parallel; with rc=2um the closest pair is sqrt(2)·d ≈ 7.07um
apart so no collision. Phase 2 (t0..2t0): atom 0 row-right, atom 1
col-up, atom 2 stays.

The three time slices are exactly the requested t_values. The bottom
plate of the figure shows the 2D grid and the projected xy trajectory
of every atom; layer alpha fades earlier layers so time depth reads.

Usage:
    python trial/vis_example.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from src.atom_config import AtomConfig, Grid
from src.atom_trajectory import AtomEnsemble
from src.movement import PHYS_A_MAX, grid_accel_from_phys
from src.segments import bang_bang_duration, make_const_acc_segment

from src.visualization_stack_time import save_stack_time

N = 4
GRID_SPACING_UM = 5.0
COLLISION_RADIUS_UM = 2.0
OUTPUT_PATH = ROOT / "render" / "vis_example.png"


def main() -> None:
    grid = Grid(N=N, d=GRID_SPACING_UM, rc=COLLISION_RADIUS_UM)
    accel = grid_accel_from_phys(PHYS_A_MAX, GRID_SPACING_UM)

    config = AtomConfig(positions=np.asarray(
        [(0.0, 3.0), (0.0, 1.0), (2.0, 1.0)]
    ))
    ensemble = AtomEnsemble.from_config(grid, config)

    t0 = bang_bang_duration(2.0, accel)

    # Phase 1: 0..t0 — three parallel moves.
    ensemble.append_segments_batch([
        (0, make_const_acc_segment(
            (0.0, 3.0), (0.0, 1.0), 0.0, accel=accel, channel="col",
        )),
        (1, make_const_acc_segment(
            (0.0, 1.0), (2.0, 1.0), 0.0, accel=accel, channel="row",
        )),
        (2, make_const_acc_segment(
            (2.0, 1.0), (2.0, 3.0), 0.0, accel=accel, channel="col",
        )),
    ])

    # Phase 2: t0..2t0 — atom 0 across, atom 1 up, atom 2 stays.
    ensemble.append_segments_batch([
        (0, make_const_acc_segment(
            (0.0, 1.0), (2.0, 1.0), t0, accel=accel, channel="row",
        )),
        (1, make_const_acc_segment(
            (2.0, 1.0), (2.0, 2.0), t0, accel=accel, channel="col",
        )),
    ])

    total = ensemble.total_duration()
    print(
        f"N={N}, atoms=3, t0={t0 * 1e6:.2f} us, "
        f"total={total * 1e6:.2f} us, "
        f"segments={[len(a.segments) for a in ensemble.atomtrajs]}"
    )

    t_values = [0.0, t0, 2.0 * t0]

    save_stack_time(
        ensemble,
        t_values,
        OUTPUT_PATH,
        atom_colors={0: "#1f77b4", 1: "#2ca02c", 2: "#9467bd"},
        show_atom_ids=True,
        show_traps=True,
        layer_label_style="us",
        title="three atoms, asynchronous moves",
    )
    print(f"wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
