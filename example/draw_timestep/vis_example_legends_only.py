"""Legend-only render: column of realistic trap/atom swatches.

No atoms, no time stack, no scene — just the legend rows, drawn with
the same `_draw_trap_tube` / `_draw_planar_discs` primitives the main
visualization uses, viewed at the same camera angle. Each swatch is
therefore a miniature of how that element looks in a real render.

Rows (top to bottom), matching the same palette as `vis_example.py`:

    blue atom       computation qubit
    gray atom       idle qubit
    black x         atom loss
    red tube        row channel
    blue tube       col channel
    orange tube     Rz rotation
    orange/yellow   Rx rotation (abrupt boundary)
    gray tube       idle trap

Text labels are intentionally omitted — add them in post-processing.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb, to_rgba

from src.visualization_stack_time import (
    StackTimeStyle,
    _draw_planar_discs,
    _draw_trap_tube,
)


OUTPUT_PATH = Path(__file__).resolve().parent / "vis_example_legends_only.png"

# Match the colors used in vis_example.py / vis_example_legends.py.
PALETTE = {
    "computation_atom": "#1f3a93",
    "idle_atom": "#b9c0ca",
    "row": "#ff2828",
    "col": "#29b0ff",
    "rz": "#f97316",
    "rx": ("#f97316", "#facc15"),
    "idle": "#9a9a9a",
}

# Grid-spacing baseline (um) — sets the physical scale of the swatches
# to roughly one lattice cell, matching the main scene.
D = 8.0


def build_style() -> StackTimeStyle:
    """Reuse the same style as the main demo so swatch geometry / colors match."""
    return StackTimeStyle(
        figsize=(3.6, 10.5),
        dpi=1200,
        projection_type="ortho",
        atom_radius_frac=0.28,
        atom_disc_segments=72,
        atom_edge_lw=0.6,
        trap_radius_frac=0.30,
        trap_max_alpha=0.62,
        trap_samples_per_segment=160,
        tube_facets=160,
        trap_two_tone_theta=3 * np.pi / 4 - 0.33,
        trap_two_tone_transition_widths={"rx": 0.0},
        loss_marker_color="#000000",
        loss_marker_size=40.0,
        loss_marker_lw=1.5,
        idle_trap_color=PALETTE["idle"],
        idle_trap_alpha=0.55,
    )


def render_legend(out: Path) -> None:
    s = build_style()

    trap_radius = s.trap_radius_frac * D
    atom_radius = np.sqrt(0.7) * s.atom_radius_frac * D

    entries = [
        ("atom",     PALETTE["computation_atom"]),
        ("atom",     PALETTE["idle_atom"]),
        ("loss",     s.loss_marker_color),
        ("trap",     PALETTE["row"]),
        ("trap",     PALETTE["col"]),
        ("trap",     PALETTE["rz"]),
        ("two_tone", PALETTE["rx"]),
        ("idle",     PALETTE["idle"]),
    ]

    # Column layout: each swatch occupies a vertical slot of `row_pitch`,
    # with the trap-tube body taking `swatch_height` and a generous gap
    # above/below so rows read as distinct entries.
    swatch_height = 1.6 * D
    row_pitch = 2.8 * D
    n = len(entries)
    z_top = (n - 1) * row_pitch

    fig = plt.figure(figsize=s.figsize, dpi=s.dpi)
    ax = fig.add_subplot(111, projection="3d")
    # Use draw order, not matplotlib's auto z-sort, so swatch ordering
    # stays predictable along the column.
    ax.computed_zorder = False

    for i, (kind, color) in enumerate(entries):
        z0 = z_top - i * row_pitch
        z_center = z0 + 0.5 * swatch_height
        xy_single = np.asarray([[0.0, 0.0]], dtype=float)

        if kind == "atom":
            _draw_planar_discs(
                ax,
                xy_single,
                z_center,
                atom_radius,
                [to_rgba(color)],
                [to_rgba("white")],
                s.atom_edge_lw,
                s.atom_disc_segments,
                0.0,
            )
            continue

        if kind == "loss":
            # Same scatter call the main scene uses for the loss cross.
            ax.scatter(
                [0.0], [0.0], [z_center],
                s=s.loss_marker_size, c=color, marker="x",
                linewidths=s.loss_marker_lw, depthshade=False,
            )
            continue

        # All other kinds are short vertical trap tubes at a single xy.
        samples = max(8, int(s.trap_samples_per_segment) // 2)
        zs = np.linspace(z0, z0 + swatch_height, samples)
        xy_arr = np.tile(xy_single, (samples, 1))

        if kind == "trap":
            rgb = [to_rgb(color)]
            wall_alpha = np.full_like(zs, s.trap_max_alpha)
            two_tone_half_width = 0.0
        elif kind == "two_tone":
            color_a, color_b = color
            rgb = [to_rgb(color_a), to_rgb(color_b)]
            wall_alpha = np.full_like(zs, s.trap_max_alpha)
            # Abrupt boundary — Rx with no white band, matching the
            # `trap_two_tone_transition_widths={"rx": 0.0}` override.
            two_tone_half_width = 0.0
        elif kind == "idle":
            rgb = [to_rgb(color)]
            wall_alpha = np.full_like(zs, s.trap_max_alpha * s.idle_trap_alpha)
            two_tone_half_width = 0.0
        else:
            raise ValueError(f"unknown legend kind {kind!r}")

        _draw_trap_tube(
            ax,
            xy_arr,
            zs,
            wall_alpha,
            trap_radius,
            rgb,
            s.tube_facets,
            sort_zpos=float(np.mean(zs)),
            theta0=s.trap_two_tone_theta,
            two_tone_half_width=two_tone_half_width,
        )

    # View setup: same camera as the main scene so each swatch reads
    # exactly like its counterpart in the real render.
    pad_xy = trap_radius * 2.5
    ax.set_xlim(-pad_xy, pad_xy)
    ax.set_ylim(-pad_xy, pad_xy)
    ax.set_zlim(-row_pitch * 0.3, z_top + swatch_height + row_pitch * 0.3)
    ax.view_init(elev=s.elev, azim=s.azim, vertical_axis=s.vertical_axis)
    if s.projection_type == "ortho":
        ax.set_proj_type("ortho")
    else:
        ax.set_proj_type("persp", focal_length=s.focal_length)

    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False
        axis.pane.set_edgecolor("none")
        axis.line.set_color((1.0, 1.0, 1.0, 0.0))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.grid(False)
    # Tall, narrow box so the column reads as a column.
    ax.set_box_aspect((1.0, 1.0,  n))

    fig.savefig(
        out, dpi=s.dpi, bbox_inches="tight",
        transparent=True,
    )
    plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    render_legend(OUTPUT_PATH)


if __name__ == "__main__":
    main()
