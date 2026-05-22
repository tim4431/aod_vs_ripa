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

from src.visualization_stack_time import (
    ProjectPlaneStyle,
    StackTimeStyle,
    save_ground_plane_2d,
    save_stack_time,
)

N = 6
GRID_SPACING_UM = 5.0
COLLISION_RADIUS_UM = 2.0
OUTPUT_PATH = ROOT / "trial" / "vis_example.png"
PER_TIMESTEP_DIR = ROOT / "trial" / "vis_example_per_timestep"
PER_TIMESTEP_3D_DIR = ROOT / "trial" / "vis_example_per_timestep_3d"


def save_per_timestep_frames(
    ensemble,
    t_values,
    output_dir,
    *,
    style,
    atom_colors,
    filename_template="frame_{idx:02d}.png",
):
    """Render one 2D top-down PNG per timestep.

    Calls `save_ground_plane_2d` so each frame shows the current atoms,
    the active trap motion bars (thick row/col strips with arrowheads),
    target site outlines, and the row/col lattice — no 3D camera, no
    projection guides, no loss cross.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for idx, t in enumerate(t_values):
        out = output_dir / filename_template.format(idx=idx, t=float(t))
        save_ground_plane_2d(
            ensemble,
            float(t),
            out,
            style=style,
            atom_colors=atom_colors,
            show_grid=True,
            show_traps=True,
            show_motion_targets=True,
            show_plane=False,
            transparent=False,
        )
        paths.append(out)
        print(f"wrote {out}")
    return paths


def save_per_timestep_frames_3d(
    ensemble,
    t_values,
    output_dir,
    *,
    style,
    atom_colors,
    filename_template="frame_{idx:02d}.png",
):
    """Render one 3D-looking PNG per timestep.

    Calls `save_stack_time` with a single-element `t_values=[t]` and a
    `ProjectPlaneStyle` with every projection enabled, so each frame
    shows the current-time layer alongside its full projected trap
    trajectory on the plate.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    project_plane = ProjectPlaneStyle(
        show_plane=True,
        show_grid=True,
        show_traps=True,
        show_trajectory=True,
    )
    paths = []
    for idx, t in enumerate(t_values):
        out = output_dir / filename_template.format(idx=idx, t=float(t))
        save_stack_time(
            ensemble,
            [float(t)],
            out,
            style=style,
            project_plane=project_plane,
            atom_colors=atom_colors,
            show_traps=True,
            show_layers=False,
            show_layer_plane=False,
            show_motion_arrows=False,
            show_motion_targets=False,
            show_trajectory=False,
            show_trap_event_guides=False,
            transparent=False,
        )
        paths.append(out)
        print(f"wrote {out}")
    return paths


def main() -> None:
    grid = Grid(N=N, d=GRID_SPACING_UM, rc=COLLISION_RADIUS_UM)
    accel = grid_accel_from_phys(PHYS_A_MAX, GRID_SPACING_UM)

    # Atom 4 is the incoming fresh atom. It starts gray and is only
    # addressed after the loss is detected at t=t0.
    config = AtomConfig(positions=np.asarray(
        [
            (1, 2),
            (1, 3),
            (2, 3),
            (2, 2),
            (5, 3),
            (5, 4),
            (5, 5)
        ]
    ))
    ensemble = AtomEnsemble.from_config(grid, config)

    # Use one visual cycle per storyboard step. The refill legs are the
    # longest moves in this example, spanning two lattice sites.
    t0 = min_jerk_duration(2.0, accel)

    # Phase 1: 0..t0 - four parallel RIPA moves. Atom 2 is lost mid-flight.
    ensemble.append_segments_batch([
        (0, make_smooth_segment(
            (1, 2), (0, 2), 0.0, duration=t0, channel="row",
        )),
        (1, make_smooth_segment(
            (1, 3), (1, 4), 0.0, duration=t0, channel="col",
        )),
        (2, make_smooth_segment(
            (2, 3), (3, 3), 0.0, duration=t0, channel="row",
        )),
        (3, make_smooth_segment(
            (2, 2), (2, 1), 0.0, duration=t0, channel="col",
        )),
    ])
    lost_at = 0.5 * t0
    ensemble.mark_lost(2, lost_at, cause="transport_loss")

    # Phase 2: t0..2t0 - loss is detected, atom 4 starts refill, and
    # stationary single-qubit rotations run on atom 0 and atom 3.
    ensemble.append_segments_batch([
        (0, make_hold((0, 2), t0, t0, channel="rz")),
        (3, make_hold((2, 1), t0, t0, channel="rx")),
        (4, make_smooth_segment(
            (5, 3), (3, 3), t0, duration=t0, channel="row",
        )),
    ])



    t_values = [k * t0 for k in range(3)]

    style = StackTimeStyle(
        figsize=(7.6, 8.6),
        dpi=600,
        projection_type="ortho",
        # all layers similarly opaque so the top doesn't wash out
        layer_alpha_floor=0.9,
        # row/col palette — used by motion arrows AND the bottom-plate
        # lattice; lattice_lines on individual layers is OFF below
        lattice_line_colors={"row": "#ff2828", "col": "#29b0ff"},
        lattice_line_extension_frac=0.7,
        # noticeably translucent layer plane — each layer reads as a plate
        layer_plane_color="#dce2e9",
        layer_plane_alpha=0.5,
        layer_plane_extension_frac=0.55,
        # small solid in-plane arrows: current move direction at each layer,
        # color-keyed by RIPA row/col channel
        motion_arrow_lw=1.8,
        motion_arrow_alpha=1.0,
        motion_arrow_length_frac=1.0,
        motion_arrow_max_length_frac=0.58,
        motion_arrow_head_length_frac=0.20,
        motion_arrow_head_width_frac=0.24,
        motion_arrow_start_offset_frac=0.30,
        motion_arrow_z_offset=0.04,
        motion_arrow_channel_colors={
            "row": "#ff2828", "col": "#29b0ff",
            "rz": "#f97316", "rx": "#facc15",
            "aod": "#444444", None: "#444444",
        },
        motion_target_circle_color="#7d8795",
        motion_target_circle_alpha=0.78,
        motion_target_circle_lw=1.2,
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
        trap_two_tone_theta=3*np.pi/4-0.33,
        # Rx uses orange+yellow; force an abrupt boundary between them
        # (no white smoothstep band).
        trap_two_tone_transition_widths={"rx": 0.0},
        trap_ramp_frac=0.30,
        trap_ramp_alpha_floor=0.12,
        trap_samples_per_segment=256,
        trajectory_lw=1.0,
        trajectory_alpha=0.42,
        trajectory_darken=0.75,
        trap_event_guide_color="#7d8795",
        trap_event_guide_lw=1.0,
        trap_event_guide_alpha=0.32,
        trap_event_guide_dashes=(3.0, 3.0),
    )

    # Projection plate uses the StackTimeStyle.lattice_line_colors above
    # (row/col) for its grid. All projections are off for the main stack
    # render — toggle on per call via a different ProjectPlaneStyle.
    project_plane = ProjectPlaneStyle(
        show_plane=True,
        show_grid=True,
        show_traps=True,
        show_trajectory=True,
        plane_alpha=0.22,
        trap_lw=5.0,
        trap_alpha=0.34,
        trap_width_frac=0.26,
        trap_arrow_length_frac=0.42,
        trap_arrow_lw=1.4,
        trap_arrow_alpha=0.95,
        grid_lw=1.0,
        grid_alpha=0.55,
        traj_lw=1.1,
        traj_alpha=0.18,
    )

    # Per-layer dashed-gray connections — one closed quad through the
    # four working atoms on each layer. Atoms (non-fresh) sit on the
    # vertices.
    layer_connections = {
        0: [[(1, 2), (1, 3), (2, 3), (2, 2), (1, 2)]],
        1: [[(0, 2), (1, 4), (3, 3), (2, 1), (0, 2)]],
        2: [[(0, 2), (1, 4), (3, 3), (2, 1), (0, 2)]],
    }

    save_stack_time(
        ensemble,
        t_values,
        OUTPUT_PATH,
        style=style,
        project_plane=project_plane,
        layer_connections=layer_connections,
        atom_colors={
            0: "#1f3a93",
            1: "#1f3a93",
            2: "#1f3a93",
            3: "#1f3a93",
            4: "#1f3a93",
        },
        show_traps=True,
        show_idle_traps=True,
        show_layer_plane=True,
        show_motion_arrows=True,
        show_motion_targets=True,
        show_trajectory=False,
        show_trap_event_guides=False,
        transparent=False,
        show_project_plane=False,
    )
    print(f"wrote {OUTPUT_PATH}")

    # per_timestep_atom_colors = {
    #     0: "#1f3a93",
    #     1: "#1f3a93",
    #     2: "#1f3a93",
    #     3: "#1f3a93",
    #     4: "#1f3a93",
    # }

    # save_per_timestep_frames(
    #     ensemble,
    #     t_values,
    #     PER_TIMESTEP_DIR,
    #     style=style,
    #     atom_colors=per_timestep_atom_colors,
    # )

    # save_per_timestep_frames_3d(
    #     ensemble,
    #     t_values,
    #     PER_TIMESTEP_3D_DIR,
    #     style=style,
    #     atom_colors=per_timestep_atom_colors,
    # )


if __name__ == "__main__":
    main()
