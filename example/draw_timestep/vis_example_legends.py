"""Demo: 2D legend overlay for the time-stacked 3D visualization.

Renders the `vis_example.py` storyboard with a matplotlib `ax.legend()`
overlay anchored to the *left* of the axes (outside the scene, so it
doesn't block any of the 3D data). A custom `HandlerTwoTone` paints
the Rx swatch as half orange / half yellow with no transition band,
matching the trap-tube styling.

Legend entries:
    blue atom       -> computation qubit
    gray atom       -> idle qubit
    red             -> row channel
    blue            -> col channel
    orange          -> Rz rotation
    orange + yellow -> Rx rotation (abrupt boundary, no white band)
    gray            -> idle trap
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.legend_handler import HandlerBase
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

from src.atom_config import AtomConfig, Grid
from src.atom_trajectory import AtomEnsemble
from src.movement import PHYS_A_MAX, grid_accel_from_phys
from src.segments import make_hold, make_smooth_segment, min_jerk_duration

from src.visualization_stack_time import (
    ProjectPlaneStyle,
    StackTimeStyle,
    draw_stack_time,
)


N = 6
GRID_SPACING_UM = 5.0
COLLISION_RADIUS_UM = 2.0
OUT_2D = ROOT / "example" / "draw_timestep" / "vis_example_legends_2d.png"

# Single source of truth for swatch colors. The StackTimeStyle below
# keys off these so the legend always matches the rendered scene.
PALETTE = {
    "computation_atom": "#1f3a93",
    "idle_atom": "#b9c0ca",
    "row": "#ff2828",
    "col": "#29b0ff",
    "rz": "#f97316",
    "rx": ("#f97316", "#facc15"),
    "idle": "#9a9a9a",
}


class HandlerTwoTone(HandlerBase):
    """Legend handler: split swatch into two halves of `color_a` / `color_b`."""

    def __init__(self, color_a, color_b, **kwargs):
        super().__init__(**kwargs)
        self.color_a = color_a
        self.color_b = color_b

    def create_artists(
        self, legend, orig_handle, xdescent, ydescent,
        width, height, fontsize, trans,
    ):
        x0, y0 = -xdescent, -ydescent
        a = Rectangle(
            (x0, y0), width / 2, height,
            facecolor=self.color_a, edgecolor="none", transform=trans,
        )
        b = Rectangle(
            (x0 + width / 2, y0), width / 2, height,
            facecolor=self.color_b, edgecolor="none", transform=trans,
        )
        return [a, b]


def build_ensemble():
    """Reproduce the vis_example.py storyboard (loss + RIPA refill + gates)."""
    grid = Grid(N=N, d=GRID_SPACING_UM, rc=COLLISION_RADIUS_UM)
    accel = grid_accel_from_phys(PHYS_A_MAX, GRID_SPACING_UM)
    config = AtomConfig(positions=np.asarray([
        (1, 2), (1, 3), (2, 3), (2, 2),
        (5, 3), (5, 4), (5, 5),
    ]))
    ensemble = AtomEnsemble.from_config(grid, config)
    t0 = min_jerk_duration(2.0, accel)

    ensemble.append_segments_batch([
        (0, make_smooth_segment((1, 2), (0, 2), 0.0, duration=t0, channel="row")),
        (1, make_smooth_segment((1, 3), (1, 4), 0.0, duration=t0, channel="col")),
        (2, make_smooth_segment((2, 3), (3, 3), 0.0, duration=t0, channel="row")),
        (3, make_smooth_segment((2, 2), (2, 1), 0.0, duration=t0, channel="col")),
    ])
    ensemble.mark_lost(2, 0.5 * t0, cause="transport_loss")
    ensemble.append_segments_batch([
        (0, make_hold((0, 2), t0, t0, channel="rz")),
        (3, make_hold((2, 1), t0, t0, channel="rx")),
        (4, make_smooth_segment((5, 3), (3, 3), t0, duration=t0, channel="row")),
    ])
    return ensemble, t0


def build_style():
    return StackTimeStyle(
        figsize=(7.6, 8.6),
        dpi=240,
        projection_type="ortho",
        layer_alpha_floor=0.9,
        lattice_line_colors={"row": PALETTE["row"], "col": PALETTE["col"]},
        lattice_line_extension_frac=0.7,
        layer_plane_color="#dce2e9",
        layer_plane_alpha=0.5,
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
            "row": PALETTE["row"], "col": PALETTE["col"],
            "rz": PALETTE["rz"], "rx": PALETTE["rx"][1],
            "aod": "#444444", None: "#444444",
        },
        motion_target_circle_color="#7d8795",
        motion_target_circle_alpha=0.78,
        motion_target_circle_lw=1.2,
        motion_target_circle_dashes=(3.0, 2.4),
        atom_radius_frac=0.28,
        atom_disc_segments=72,
        atom_plane_z_offset=0.03,
        fresh_atom_color=PALETTE["idle_atom"],
        loss_marker_color="#000000",
        loss_marker_size=110.0,
        loss_marker_lw=2.4,
        loss_fade_frac=0.35,
        trap_channel_colors={
            "row": PALETTE["row"], "col": PALETTE["col"],
            "rz": PALETTE["rz"],
            "rx": PALETTE["rx"],
            "aod": "#888888", None: "#888888",
        },
        trap_two_tone_theta=3 * np.pi / 4 - 0.33,
        trap_two_tone_transition_widths={"rx": 0.0},
        trap_ramp_frac=0.30,
        trap_ramp_alpha_floor=0.12,
        trap_samples_per_segment=128,
        trajectory_lw=1.0,
        trajectory_alpha=0.42,
        trajectory_darken=0.75,
        idle_trap_color=PALETTE["idle"],
        idle_trap_alpha=0.55,
    )


def _legend_entries():
    """The seven semantic legend rows."""
    return [
        ("computation qubit", "atom",     PALETTE["computation_atom"]),
        ("idle qubit",        "atom",     PALETTE["idle_atom"]),
        ("row channel",       "rect",     PALETTE["row"]),
        ("col channel",       "rect",     PALETTE["col"]),
        ("Rz rotation",       "rect",     PALETTE["rz"]),
        ("Rx rotation",       "two_tone", PALETTE["rx"]),
        ("idle trap",         "rect",     PALETTE["idle"]),
    ]


def _build_legend_handles():
    """Matplotlib `Line2D` / `Patch` handles for `ax.legend()`."""
    # Sentinel patch routed through HandlerTwoTone for the Rx row.
    rx_sentinel = Patch(facecolor="none", edgecolor="none")
    handles = []
    labels = []
    for label, kind, color in _legend_entries():
        if kind == "atom":
            handles.append(Line2D(
                [0], [0], marker="o", color="none",
                markerfacecolor=color, markeredgecolor="white",
                markeredgewidth=1.4, markersize=12,
            ))
        elif kind == "rect":
            handles.append(Patch(facecolor=color, edgecolor="none"))
        elif kind == "two_tone":
            handles.append(rx_sentinel)
        labels.append(label)
    handler_map = {
        rx_sentinel: HandlerTwoTone(PALETTE["rx"][0], PALETTE["rx"][1]),
    }
    return handles, labels, handler_map


def render_legend(ensemble, t_values, style, project_plane, atom_colors, out):
    fig = plt.figure(figsize=style.figsize, dpi=style.dpi)
    ax = fig.add_subplot(111, projection="3d")
    draw_stack_time(
        ax, ensemble, t_values,
        style=style, project_plane=project_plane,
        atom_colors=atom_colors,
        show_traps=True, show_idle_traps=True,
        show_layer_plane=True, show_motion_arrows=True,
        show_motion_targets=True, show_trajectory=False,
        show_trap_event_guides=False, show_project_plane=False,
    )

    handles, labels, handler_map = _build_legend_handles()
    # Anchor the legend's upper-RIGHT corner just outside the axes' left
    # edge, so the legend sits to the left of the scene and never
    # overlaps any of the 3D content. `bbox_inches="tight"` on savefig
    # extends the cropped image to include this region.
    ax.legend(
        handles, labels,
        loc="upper right",
        bbox_to_anchor=(-0.02, 1.0),
        handler_map=handler_map,
        frameon=True, framealpha=0.95,
        fontsize=10, labelspacing=0.9,
        handlelength=2.4, handleheight=1.4,
    )

    fig.savefig(
        out, dpi=style.dpi, bbox_inches="tight",
        facecolor="white", transparent=False,
    )
    plt.close(fig)
    print(f"wrote {out}")


def main():
    ensemble, t0 = build_ensemble()
    t_values = [k * t0 for k in range(3)]
    style = build_style()
    project_plane = ProjectPlaneStyle()
    atom_colors = {i: PALETTE["computation_atom"] for i in range(5)}
    render_legend(ensemble, t_values, style, project_plane, atom_colors, OUT_2D)


if __name__ == "__main__":
    main()
