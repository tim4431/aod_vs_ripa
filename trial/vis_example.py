"""Demo: three atoms transport, one is lost mid-flight, a fresh atom
starts at the upper-right corner and moves by RIPA row/col legs to replace it.

Sequence on a 6x6 grid:

    Phase 1 (0..t0):   atom 0 (0,3)->(0,1)  col-down
                       atom 1 (0,1)->(2,1)  row-right
                       atom 2 (2,1)->(2,3)  col-up
                       atom 3 parked at (5,5)

    Phase 2 (t0..2t0): atom 0 (0,1)->(2,1)  row-right
                       atom 1 (2,1)->(2,2)  col-up — LOST mid-flight
                                              at t = t0 + 0.5*duration
                       atom 2 stays
                       atom 3 still parked

    Phase 3 (2t0..3t0): planning / detection delay

    Phase 4 (3t0..4t0): atom 3 (5,5)->(5,2)  col-down

    Phase 5 (4t0..5t0): atom 3 (5,2)->(2,2)  row-left

    Phase 6 (5t0..6t0): stationary single-qubit gates
                       atom 0 at (2,1): Rz, one orange detuning tone
                       atom 2 at (2,3): Rx, two Raman tones (orange + yellow)

The visualization uses a *time-based* fresh-color rule: atom 3 renders
in the fresh color only while no segment has yet started for it (the
parked-reservoir state). The instant the trap engages — i.e., the
loading segment begins — atom 3 (and its trajectory polyline) switch
to the normal color.

A dashed connector links the lost atom's last in-array position
(atom 1 at the loss time) to the new atom's final replacement position,
making the replacement relationship visible.

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

    # Atom 3 is the incoming replacement atom, initially parked at the
    # upper-right corner. Its later refill path is split into row/col RIPA
    # legs rather than a diagonal move.
    config = AtomConfig(positions=np.asarray(
        [(0.0, 3.0), (0.0, 1.0), (2.0, 1.0), (5.0, 5.0)]
    ))
    ensemble = AtomEnsemble.from_config(grid, config)

    # Use one visual cycle for every RIPA leg. A 3-site leg is the longest
    # one in this storyboard, so shorter moves simply run below max accel.
    t0 = min_jerk_duration(3.0, accel)

    # Phase 1: 0..t0 — three parallel moves; atom 3 stays put.
    ensemble.append_segments_batch([
        (0, make_smooth_segment(
            (0.0, 3.0), (0.0, 1.0), 0.0, duration=t0, channel="col",
        )),
        (1, make_smooth_segment(
            (0.0, 1.0), (2.0, 1.0), 0.0, duration=t0, channel="row",
        )),
        (2, make_smooth_segment(
            (2.0, 1.0), (2.0, 3.0), 0.0, duration=t0, channel="col",
        )),
    ])

    # Phase 2: t0..2t0 — atom 0 across, atom 1 up (will be lost mid-move).
    ensemble.append_segments_batch([
        (0, make_smooth_segment(
            (0.0, 1.0), (2.0, 1.0), t0, duration=t0, channel="row",
        )),
        (1, make_smooth_segment(
            (2.0, 1.0), (2.0, 2.0), t0, duration=t0, channel="col",
        )),
    ])

    # Atom 1 is lost halfway through phase 2.
    lost_at = t0 + 0.5 * t0
    ensemble.mark_lost(1, lost_at, cause="transport_dropout")

    # Phase 3 (2t0..3t0): detection / planning cycle — atom 3 still
    # parked, no motion yet. The system uses this cycle to register the
    # loss before launching the replenishment.
    # Phase 4/5: RIPA replenishment into the (2, 2) slot. The path is split
    # into row and column legs so every segment starts/ends on a layer.
    load_start = 3.0 * t0
    ensemble.append_segment(
        3,
        make_smooth_segment(
            (5.0, 5.0), (5.0, 2.0), load_start,
            duration=t0, channel="col",
        ),
    )
    ensemble.append_segment(
        3,
        make_smooth_segment(
            (5.0, 2.0), (2.0, 2.0), load_start + t0,
            duration=t0, channel="row",
        ),
    )

    # Phase 6: internal single-qubit rotations while atoms are stationary.
    # They reuse trap rendering by adding hold segments with gate channels:
    # Rz is one orange tone, Rx is a two-color Raman pair.
    gate_start = load_start + 2.0 * t0
    gate_duration = t0
    ensemble.append_segments_batch([
        (0, make_hold((2.0, 1.0), gate_start, gate_duration, channel="rz")),
        (2, make_hold((2.0, 3.0), gate_start, gate_duration, channel="rx")),
    ])

    total = ensemble.total_duration()
    print(
        f"N={N}, atoms=4, t0={t0 * 1e6:.2f} us, "
        f"total={total * 1e6:.2f} us, "
        f"lost_at={lost_at * 1e6:.2f} us, "
        f"load_start={load_start * 1e6:.2f} us, "
        f"gate_start={gate_start * 1e6:.2f} us"
    )

    # Layer at 3t0 makes the one-cycle delay between loss and load visible;
    # 4t0 exposes the column-to-row corner of the refill path. The final
    # slab shows the stationary Rz/Rx single-qubit gate pulses.
    t_values = [k * t0 for k in range(7)]

    style = StackTimeStyle(
        figsize=(7.6, 8.6),
        dpi=240,
        projection_type="ortho",
        layer_label_style="none",
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
        # atoms: dark blue baseline, fresh (un-addressed) color = yellow
        atom_size=180.0,
        atom_disc_mode="plane",
        atom_radius_frac=0.28,
        atom_disc_segments=72,
        atom_plane_z_offset=0.03,
        fresh_atom_color="#f1c40f",
        # loss marker
        loss_marker_color="#000000",
        loss_marker_size=110.0,
        loss_marker_lw=2.4,
        loss_fade_frac=0.35,
        # replacement connector — dashed, picks up the lost atom's color
        replacement_connector_lw=1.5,
        replacement_connector_alpha=0.9,
        replacement_connector_dashes=(5.0, 4.0),
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
        bottom_trap_alpha=0.20,
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
        atom_colors={0: "#1f3a93", 1: "#1f3a93", 2: "#1f3a93", 3: "#1f3a93"},
        show_atom_ids=False,
        show_traps=True,
        show_grid_dots=False,
        show_grid_circles=False,
        show_grid_frame=False,
        show_lattice_lines=False,
        show_layer_plane=True,
        show_motion_arrows=True,
        show_trajectory=True,
        show_time_arrow=False,
        show_bottom_plane=True,
        show_bottom_grid=True,
        show_bottom_traps=True,
        show_bottom_trajectory=True,
        show_trap_event_guides=True,
        replacements=[(1, 3)],
        transparent=False,
    )
    print(f"wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
