"""Demo: one transport loss followed by a two-step RIPA refill.

Sequence on a 6x6 grid:

    Phase 1 (0..t0): atoms 0,1,2,3 move; atom 2 is lost at 0.5t0
                     0: (0,3)->(1,3)  row-right
                     1: (2,4)->(3,4)  row-right
                     2: (3,2)->(2,2)  row-left, LOST halfway
                     3: (1,1)->(1,2)  col-up
                     atom 4 waits fresh/gray at (4,5)

    Phase 2 (t0..2t0): atom 4 (4,5)->(4,3)  col-down
                       atom 0: stationary Rz
                       atom 3: stationary Rx

    Phase 3 (2t0..3t0): atom 4 (4,3)->(2,3)  row-left

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
from src.segments import make_hold, make_smooth_segment, min_jerk_duration

from src.visualization_stack_time import StackTimeStyle, save_stack_time

N = 6
GRID_SPACING_UM = 5.0
COLLISION_RADIUS_UM = 2.0
OUTPUT_PATH = ROOT / "trial" / "vis_example.png"


def main() -> None:
    grid = Grid(N=N, d=GRID_SPACING_UM, rc=COLLISION_RADIUS_UM)
    accel = grid_accel_from_phys(PHYS_A_MAX, GRID_SPACING_UM)

    # Atom 4 is the incoming fresh atom. It starts gray and is only
    # addressed after the loss is detected at t=t0.
    config = AtomConfig(positions=np.asarray(
        [
            (1, 1),
            (0, 3),
            (2, 4),
            (3, 2),
            (5, 3),
            (5,4),
            (5,5)
        ]
    ))
    ensemble = AtomEnsemble.from_config(grid, config)

    # Use one visual cycle per storyboard step. The refill legs are the
    # longest moves in this example, spanning two lattice sites.
    t0 = min_jerk_duration(2.0, accel)

    # Phase 1: 0..t0 - four parallel RIPA moves. Atom 2 is lost mid-flight.
    ensemble.append_segments_batch([
        (0, make_smooth_segment(
            (1, 1), (1, 2), 0.0, duration=t0, channel="col",
        )),
        (1, make_smooth_segment(
            (0, 3), (1, 3), 0.0, duration=t0, channel="row",
        )),
        (2, make_smooth_segment(
            (2, 4), (2, 3), 0.0, duration=t0, channel="col",
        )),
        (3, make_smooth_segment(
            (3, 2), (2, 2), 0.0, duration=t0, channel="row",
        )),
    ])
    lost_at = 0.5 * t0
    ensemble.mark_lost(2, lost_at, cause="transport_loss")

    # Phase 2: t0..2t0 - loss is detected, atom 4 starts refill, and
    # stationary single-qubit rotations run on atom 0 and atom 3.
    ensemble.append_segments_batch([
        (0, make_hold((1, 2), t0, t0, channel="rz")),
        (3, make_hold((2, 2), t0, t0, channel="rx")),
        (4, make_smooth_segment(
            (5, 3), (2, 3), t0, duration=t0, channel="row",
        )),
    ])



    t_values = [k * t0 for k in range(4)]

    style = StackTimeStyle(
        figsize=(7.6, 8.6),
        dpi=240,
        projection_type="ortho",
        # all layers similarly opaque so the top doesn't wash out
        layer_alpha_floor=0.85,
        # row/col palette — used by motion arrows AND the bottom-plate
        # lattice; lattice_lines on individual layers is OFF below
        lattice_line_colors={"row": "#ff2828", "col": "#29b0ff"},
        lattice_line_extension_frac=0.7,
        # noticeably translucent layer plane — each layer reads as a plate
        layer_plane_color="#c8d0dc",
        layer_plane_alpha=0.50,
        layer_plane_extension_frac=0.55,
        # small solid in-plane arrows: current move direction at each layer,
        # color-keyed by RIPA row/col channel
        motion_arrow_lw=1.6,
        motion_arrow_alpha=1.0,
        motion_arrow_length_frac=1.0,
        motion_arrow_max_length_frac=0.58,
        motion_arrow_head_length_frac=0.20,
        motion_arrow_head_width_frac=0.24,
        motion_arrow_start_offset_frac=0.20,
        motion_arrow_z_offset=0.08,
        motion_arrow_channel_colors={
            "row": "#ff2828", "col": "#29b0ff",
            "rz": "#f97316", "rx": "#facc15",
            "aod": "#444444", None: "#444444",
        },
        motion_target_circle_color="#7d8795",
        motion_target_circle_alpha=0.78,
        motion_target_circle_lw=1.2,
        motion_target_circle_radius_frac=0.33,
        motion_target_circle_dashes=(3.0, 2.4),
        # atoms: dark blue baseline, fresh (un-addressed) color = gray
        atom_radius_frac=0.28,
        atom_disc_segments=72,
        atom_plane_z_offset=0.03,
        fresh_atom_color="#b9c0ca",
        # loss marker
        loss_marker_color="#000000",
        loss_marker_size=110.0,
        loss_marker_lw=2.4,
        loss_fade_frac=0.35,
        # trap/gate channels: RIPA row/col moves plus stationary rotations
        trap_channel_colors={
            "row": "#ff2828", "col": "#29b0ff",
            "rz": "#f97316",
            "rx": ("#f97316", "#facc15"),
            "aod": "#888888", None: "#888888",
        },
        trap_ramp_frac=0.30,
        trap_ramp_alpha_floor=0.12,
        trap_samples_per_segment=56,
        trajectory_lw=1.0,
        trajectory_alpha=0.42,
        trajectory_darken=0.75,
        bottom_plane_alpha=0.22,
        bottom_trap_lw=5.0,
        bottom_trap_alpha=0.34,
        bottom_trap_width_frac=0.26,
        bottom_trap_arrow_length_frac=0.42,
        bottom_trap_arrow_lw=1.4,
        bottom_trap_arrow_alpha=0.95,
        trap_event_guide_color="#7d8795",
        trap_event_guide_lw=1.0,
        trap_event_guide_alpha=0.32,
        trap_event_guide_dashes=(3.0, 3.0),
        # bottom plate uses the lattice_line_colors above (row/col)
        bottom_grid_lw=1.0,
        bottom_grid_alpha=0.55,
        bottom_traj_lw=1.1,
        bottom_traj_alpha=0.18,
    )

    save_stack_time(
        ensemble,
        t_values,
        OUTPUT_PATH,
        style=style,
        atom_colors={
            0: "#1f3a93",
            1: "#1f3a93",
            2: "#1f3a93",
            3: "#1f3a93",
            4: "#1f3a93",
        },
        show_traps=True,
        show_layer_plane=True,
        show_motion_arrows=True,
        show_motion_targets=True,
        show_trajectory=True,
        show_bottom_plane=True,
        show_bottom_grid=True,
        show_bottom_traps=True,
        show_bottom_trajectory=True,
        show_trap_event_guides=True,
        transparent=False,
    )
    print(f"wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
