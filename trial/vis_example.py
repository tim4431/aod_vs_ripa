"""Three-atom hand-routed demo for `visualization_stack_time` on a 6x6 grid.

Sequence (atom positions in (i, j) grid units):

    t=0     atom 0 = (0,3)  atom 1 = (0,1)  atom 2 = (2,1)
    t=t0    atom 0 = (0,1)  atom 1 = (2,1)  atom 2 = (2,3)
    t=2*t0  atom 0 = (2,1)  atom 1 = (2,2)  atom 2 = (2,3)

Phase 1 (0..t0): atom 0 col-down, atom 1 row-right, atom 2 col-up — all
moving in parallel; with rc=2um the closest pair is sqrt(2)·d ≈ 7.07um
apart so no collision. Phase 2 (t0..2t0): atom 0 row-right, atom 1
col-up, atom 2 stays.

Rendering style: each layer is a near-transparent rectangular plane
with faint extending row (red) / col (blue) lattice lines and dashed
gray circles at every site. On each layer, an arrow on each atom
points in its intended motion direction at that snapshot. Time
arrow and time labels are off — the per-layer arrows carry the
"what's happening here" load.

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

from src.visualization_stack_time import StackTimeStyle, save_stack_time

N = 6
GRID_SPACING_UM = 5.0
COLLISION_RADIUS_UM = 2.0
OUTPUT_PATH = ROOT / "trial" / "vis_example.png"


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

    style = StackTimeStyle(
        figsize=(7.6, 7.8),
        dpi=240,
        layer_label_style="none",
        # dashed open circles at every lattice site
        grid_circle_color="#7a7a7a",
        grid_circle_radius_frac=0.36,
        grid_circle_lw=1.0,
        grid_circle_alpha=0.85,
        grid_circle_dashes=(2.5, 2.0),
        # faint extending lattice lines (row=red, col=blue)
        lattice_line_colors={"row": "#ff2828", "col": "#29b0ff"},
        lattice_line_lw=1.0,
        lattice_line_alpha=0.22,
        lattice_line_extension_frac=0.7,
        # near-transparent layer plane
        layer_plane_color="#c8d0dc",
        layer_plane_alpha=0.10,
        layer_plane_extension_frac=0.55,
        # per-atom intended-motion arrows
        motion_arrow_lw=1.6,
        motion_arrow_alpha=0.95,
        motion_arrow_length_frac=0.85,
        motion_arrow_head_frac=0.32,
        motion_arrow_darken=0.55,
        # atoms
        atom_size=180.0,
        # trap channel colors aligned with lattice lines
        trap_channel_colors={
            "row": "#ff2828", "col": "#29b0ff",
            "aod": "#9a9a9a", None: "#9a9a9a",
        },
    )

    save_stack_time(
        ensemble,
        t_values,
        OUTPUT_PATH,
        style=style,
        atom_colors={0: "#1f3a93", 1: "#1f3a93", 2: "#1f3a93"},
        show_atom_ids=False,
        show_traps=True,
        show_grid_dots=False,
        show_grid_circles=True,
        show_grid_frame=False,
        show_lattice_lines=True,
        show_layer_plane=True,
        show_motion_arrows=True,
        show_trajectory=False,
        show_time_arrow=False,
        show_bottom_grid=False,
        show_bottom_trajectory=False,
    )
    print(f"wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
