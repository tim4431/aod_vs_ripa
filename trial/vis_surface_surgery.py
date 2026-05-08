"""Demo: two 3x3 rotated surface code patches sit side by side. The
left patch undergoes a 90-degree Hadamard rotation driven by synchronous
RIPA (one-slot lockstep ring shifts at a_max), and afterwards three
ancilla atoms move up from the reservoir below into the gap column to
perform a lattice-surgery merge between the two patches.

Atom replenishment is neglected — no atom loss, no fresh-loaded atom.

Sequence on an 8x8 grid:

    Phase 1 (0..t0):   First lockstep RIPA shift of patch A's outer ring
                       — every ring atom hops one perimeter slot clockwise.
                       Patch A center, patch B, and ancillas stay put.

    Phase 2 (t0..2t0): Second lockstep shift completes the 90-degree
                       rotation (8-atom ring rotated by 2 of its 8 slots).

    Phase 3 (2t0..3t0): Lattice-surgery merge — three ancilla atoms move
                        up the gap column j=4 in one synchronous batch
                        (row channel, j fixed), settling into rows 1..3
                        between patch A and patch B.

The lockstep ring decomposition mirrors `LockstepRIPAHadamardPatchRotationScheduler`
in `example/hardmard_patch_rotation.py`; the visualization style mirrors
`trial/vis_example.py`.

Usage:
    python trial/vis_surface_surgery.py
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
from src.segments import make_smooth_segment, min_jerk_duration

import src.visualization_stack_time as vst
from src.visualization_stack_time import StackTimeStyle, save_stack_time

# Every atom is part of the array from t=0 — there is no off-grid
# reservoir loading in this demo. Suppress the "fresh-loaded" recolor
# rule so patch B and patch A's stationary center qubit render in their
# assigned colors instead of the fresh-atom yellow.
vst._is_fresh_at = lambda atom, t, eps=1e-9: False

N = 8
GRID_SPACING_UM = 5.0
COLLISION_RADIUS_UM = 2.0
OUTPUT_PATH = ROOT / "trial" / "vis_surface_surgery.png"

PATCH_A_ROWS = (1, 2, 3)
PATCH_A_COLS = (1, 2, 3)
PATCH_B_ROWS = (1, 2, 3)
PATCH_B_COLS = (5, 6, 7)
GAP_COL = 4
ANCILLA_ROWS = (5, 6, 7)


def _outer_ring(rows: tuple[int, int, int], cols: tuple[int, int, int]):
    """Outer perimeter of a 3x3 patch in clockwise order from top-left.

    Mirrors `_ring_positions(coords, layer=0)` for vals=(lo, mid, hi) in
    `ManualRIPAHadamardPatchRotationScheduler` so the lockstep rotation
    follows the same site ordering as the scheduler-driven demo.
    """
    lo_r, mid_r, hi_r = rows
    lo_c, mid_c, hi_c = cols
    return [
        (lo_r, lo_c), (lo_r, mid_c), (lo_r, hi_c),
        (mid_r, hi_c), (hi_r, hi_c),
        (hi_r, mid_c), (hi_r, lo_c),
        (mid_r, lo_c),
    ]


def _channel(current, target):
    if current[1] == target[1]:
        return "row"
    if current[0] == target[0]:
        return "col"
    raise ValueError(f"non-axis-aligned hop: {current} -> {target}")


def main() -> None:
    grid = Grid(N=N, d=GRID_SPACING_UM, rc=COLLISION_RADIUS_UM)
    accel = grid_accel_from_phys(PHYS_A_MAX, GRID_SPACING_UM)

    patch_a_sites = [(i, j) for i in PATCH_A_ROWS for j in PATCH_A_COLS]
    patch_b_sites = [(i, j) for i in PATCH_B_ROWS for j in PATCH_B_COLS]
    ancilla_sites = [(i, GAP_COL) for i in ANCILLA_ROWS]

    all_sites = patch_a_sites + patch_b_sites + ancilla_sites
    config = AtomConfig(positions=np.asarray(all_sites, dtype=float))
    ensemble = AtomEnsemble.from_config(grid, config)

    site_to_atom_id = {site: idx for idx, site in enumerate(all_sites)}

    # One layer per phase; t0 is set by the longest single move (the
    # 4-slot ancilla merge). Single-slot ring hops simply run below
    # max accel inside the same window, keeping every layer one t0 wide.
    t0 = min_jerk_duration(4.0, accel)

    # Phase 1 / Phase 2: synchronous RIPA Hadamard rotation of patch A's
    # 8-atom outer ring. Two lockstep shifts of one perimeter slot each
    # produce a 90-degree rotation. Patch A's center atom (2, 2) stays.
    ring = _outer_ring(PATCH_A_ROWS, PATCH_A_COLS)
    ring_len = len(ring)
    for shift in range(2):
        start_time = shift * t0
        moves = []
        for source_index, source_site in enumerate(ring):
            current = ring[(source_index + shift) % ring_len]
            target = ring[(source_index + shift + 1) % ring_len]
            atom_id = site_to_atom_id[source_site]
            moves.append((atom_id, make_smooth_segment(
                current, target, start_time,
                duration=t0, channel=_channel(current, target),
            )))
        ensemble.append_segments_batch(moves)

    # Phase 3: lattice-surgery merge. Three ancilla atoms parked four
    # slots below the gap column move up in lockstep into rows 1..3 of
    # column GAP_COL. They share j (row channel), share start time, share
    # duration — so the column stays rigid and never collides with itself.
    merge_start = 2.0 * t0
    merge_moves = []
    for src_row, dst_row in zip(ANCILLA_ROWS, PATCH_A_ROWS):
        src = (src_row, GAP_COL)
        dst = (dst_row, GAP_COL)
        merge_moves.append((site_to_atom_id[src], make_smooth_segment(
            src, dst, merge_start, duration=t0, channel="row",
        )))
    ensemble.append_segments_batch(merge_moves)

    total = ensemble.total_duration()
    print(
        f"N={N}, atoms={len(all_sites)}, t0={t0 * 1e6:.2f} us, "
        f"total={total * 1e6:.2f} us"
    )

    # Layers at 0, t0, 2t0, 3t0: initial state, after shift 1, after
    # rotation complete, after merge complete.
    t_values = [k * t0 for k in range(4)]

    atom_colors = {}
    for site in patch_a_sites:
        atom_colors[site_to_atom_id[site]] = "#1f3a93"  # patch A: deep blue
    for site in patch_b_sites:
        atom_colors[site_to_atom_id[site]] = "#7d3c98"  # patch B: plum
    for site in ancilla_sites:
        atom_colors[site_to_atom_id[site]] = "#27ae60"  # ancillas: green

    style = StackTimeStyle(
        figsize=(8.4, 8.6),
        dpi=240,
        projection_type="ortho",
        layer_label_style="none",
        layer_alpha_floor=0.85,
        lattice_line_colors={"row": "#ff2828", "col": "#29b0ff"},
        lattice_line_extension_frac=0.7,
        layer_plane_color="#c8d0dc",
        layer_plane_alpha=0.30,
        layer_plane_extension_frac=0.55,
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
            "aod": "#444444", None: "#444444",
        },
        atom_size=180.0,
        atom_disc_mode="plane",
        atom_radius_frac=0.28,
        atom_disc_segments=72,
        atom_plane_z_offset=0.03,
        fresh_atom_color="#1f3a93",
        trap_channel_colors={
            "row": "#ff2828", "col": "#29b0ff",
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
        bottom_grid_lw=1.0,
        bottom_grid_alpha=0.55,
        bottom_traj_lw=2.0,
        bottom_traj_alpha=0.85,
    )

    save_stack_time(
        ensemble,
        t_values,
        OUTPUT_PATH,
        style=style,
        atom_colors=atom_colors,
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
        transparent=False,
    )
    print(f"wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
