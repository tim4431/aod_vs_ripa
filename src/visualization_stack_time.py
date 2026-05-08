"""Time-stacked 3D visualization for atom rearrangement motion.

Stacks time slices of an `AtomEnsemble` / `MovingSequence` along z, with
t=0 at the top and t=t_max at the bottom. Atoms are 2D cross-section
discs on every layer; trap motion can render as 3D tubes; a continuous
trajectory polyline threads through the stack between layers; per-layer
alpha gives a "depth of time" cue.

All visual styling — view angle, colors, sizes, alphas, the choice of
grid decorations (border / dots / circles / lattice bonds) — lives on
the `StackTimeStyle` dataclass. Pass a custom `style=` to change the
look without touching the module. The data-flow kwargs of
`draw_stack_time` (motion, t_values, atom_colors, show_*) control
*what* is drawn; the Style controls *how*.

Top-level entry points:

    `draw_stack_time(ax, motion, t_values, *, style=..., ...)`
    `draw_stack_time_frame(motion, t_values, *, style=..., ...)`
    `save_stack_time(motion, t_values, output_path, *, style=..., ...)`
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb, to_rgba
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers '3d' projection)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.atom_trajectory import AtomEnsemble  # noqa: E402
from src.visualization import ATOM_SIZE, _atom_colors_rgba  # noqa: E402

LayerLabelStyle = Literal["us", "ns", "none"]


@dataclass
class StackTimeStyle:
    """Visual styling for `draw_stack_time` and friends.

    Every knob that controls *appearance* (view angle, colors, sizes,
    alphas, fonts, layer fade) lives here. The drawing functions hold
    no hardcoded style — pass an instance to override defaults.
    """

    # Figure
    figsize: tuple[float, float] = (6.6, 6.4)
    dpi: int = 220

    # Camera
    elev: float = 32.0
    azim: float = -62.0
    vertical_axis: str = "z"
    focal_length: float = 0.45

    # Layer stack geometry / depth fade
    layer_spacing: float = 6.0
    layer_alpha_floor: float = 0.32

    # Atom discs
    atom_size: float = ATOM_SIZE
    atom_edge_color: Any = "#ffffff"
    atom_edge_lw: float = 0.6
    atom_id_fontsize: float = 6.5

    # Layer rectangular border (split into near/far halves)
    border_color: Any = "black"
    border_lw: float = 0.9
    border_alpha_floor: float = 0.18
    border_alpha_falloff: float = 0.72
    border_split_frac: float = 0.82

    # Grid dots (small filled scatter at each lattice site)
    grid_dot_color: Any = "#cccccc"
    grid_dot_alpha: float = 0.18
    grid_dot_size: float = 4.0

    # Grid circles (open dashed circle at each lattice site)
    grid_circle_color: Any = "#9a9a9a"
    grid_circle_radius_frac: float = 0.30
    grid_circle_lw: float = 0.7
    grid_circle_alpha: float = 0.7
    grid_circle_dashes: tuple[float, float] | None = (3.0, 2.5)
    grid_circle_segments: int = 36

    # Lattice lines: one continuous line per row of sites (red, varying i
    # at fixed j) and per column of sites (blue, varying j at fixed i).
    # Each line extends past the outermost lattice site by a fraction of d
    # so the layer's edge looks like an open mesh, not a closed rectangle.
    lattice_line_colors: Mapping[str, Any] = field(
        default_factory=lambda: {"row": "#ff2828", "col": "#29b0ff"}
    )
    lattice_line_lw: float = 1.0
    lattice_line_alpha: float = 0.25
    lattice_line_extension_frac: float = 0.7

    # Translucent rectangular plane drawn under each layer.
    layer_plane_color: Any = "#c8d0dc"
    layer_plane_alpha: float = 0.10
    layer_plane_extension_frac: float = 0.55

    # Per-layer motion arrows: an arrow on each atom showing the direction
    # of the segment that's active or about to start at the layer time.
    motion_arrow_lw: float = 1.6
    motion_arrow_alpha: float = 0.95
    motion_arrow_length_frac: float = 0.70   # arrow length in units of d
    motion_arrow_head_frac: float = 0.32     # head length / total length
    motion_arrow_darken: float = 0.55        # applied to atom color

    # Trap tubes
    trap_radius_frac: float = 0.30
    trap_max_alpha: float = 0.62
    trap_channel_colors: Mapping[Any, Any] = field(
        default_factory=lambda: {
            "row": "#ff2828",
            "col": "#29b0ff",
            "aod": "#9a9a9a",
            None: "#9a9a9a",
        }
    )
    trap_ramp_frac: float = 0.18
    trap_samples_per_segment: int = 36
    tube_facets: int = 32

    # Trajectory polylines through the stack
    trajectory_lw: float = 1.4
    trajectory_alpha: float = 0.7
    trajectory_darken: float = 0.65
    trajectory_samples_per_segment: int = 24

    # Bottom plate grid lines
    bottom_grid_color: Any = "#9a9a9a"
    bottom_grid_lw: float = 0.6
    bottom_grid_alpha: float = 0.55

    # Bottom plate trajectory projections
    bottom_traj_lw: float = 2.0
    bottom_traj_alpha: float = 0.85
    bottom_traj_samples_per_segment: int = 24

    # Per-layer time label
    layer_label_style: LayerLabelStyle = "us"
    label_color: Any = (0.15, 0.15, 0.15)
    label_fontsize_base: float = 6.5
    label_fontsize_growth: float = 1.2

    # Time arrow
    time_arrow_color: Any = "#444444"
    time_arrow_lw: float = 1.4
    time_arrow_label_color: Any = "#333333"
    time_arrow_fontsize: float = 8.5

    # Title
    title_pad: float = 18
    title_fontsize: float = 11


DEFAULT_STYLE = StackTimeStyle()


def _smoothstep(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _darken(color: Any, k: float) -> tuple[float, float, float, float]:
    r, g, b, a = to_rgba(color)
    return (k * r, k * g, k * b, a)


def _intended_direction(
    atom: Any, t: float, eps: float = 1e-9,
) -> tuple[float, float] | None:
    """Return (di, dj) for the move active at or beginning at time `t`.

    Picks the segment whose start_time matches `t` (atom about to move)
    or which strictly contains `t` (atom mid-motion). Returns `None` if
    no such segment exists (atom is stationary at this snapshot).
    """
    for seg in atom.segments:
        if seg.duration <= 0:
            continue
        if abs(seg.start_time - t) <= eps or (
            seg.start_time < t < seg.end_time - eps
        ):
            return (
                seg.end_pos[0] - seg.start_pos[0],
                seg.end_pos[1] - seg.start_pos[1],
            )
    return None


def draw_stack_time(
    ax: Any,
    motion: Any,
    t_values: Sequence[float],
    *,
    atom_colors: Mapping[int, Any] | Iterable[Any] | None = None,
    show_atom_ids: bool = False,
    show_traps: bool = True,
    show_grid_dots: bool = True,
    show_grid_circles: bool = False,
    show_grid_frame: bool = True,
    show_lattice_lines: bool = False,
    show_layer_plane: bool = False,
    show_motion_arrows: bool = False,
    show_trajectory: bool = True,
    show_time_arrow: bool = True,
    show_bottom_grid: bool = True,
    show_bottom_trajectory: bool = True,
    style: StackTimeStyle | None = None,
    title: str | None = None,
) -> None:
    """Draw a time-stacked 3D figure of `motion` on a 3D `ax`.

    `style` selects all visual parameters; the `show_*` flags toggle
    which decorations are rendered. Atom layers always render.
    """
    s = style if style is not None else DEFAULT_STYLE
    ensemble = motion if isinstance(motion, AtomEnsemble) else motion.ensemble

    t_arr = sorted(float(t) for t in t_values)
    n_layers = len(t_arr)
    if n_layers == 0:
        raise ValueError("t_values must be non-empty")

    colors = _atom_colors_rgba(
        [int(a.atom_id) for a in ensemble.atomtrajs], atom_colors
    )

    grid = ensemble.grid
    d = grid.d
    site_lo = (0.0 - grid.center) * d
    site_hi = ((grid.N - 1.0) - grid.center) * d
    bound_lo = site_lo - d / 2.0
    bound_hi = site_hi + d / 2.0
    span = bound_hi - bound_lo
    y_split = bound_lo + s.border_split_frac * span
    trap_radius = s.trap_radius_frac * d
    line_lo = site_lo - s.lattice_line_extension_frac * d
    line_hi = site_hi + s.lattice_line_extension_frac * d
    plane_lo = site_lo - s.layer_plane_extension_frac * d
    plane_hi = site_hi + s.layer_plane_extension_frac * d

    # Top layer (smallest t) at largest z; layers descend toward z=0.
    layer_zs = [(n_layers - 1 - i) * s.layer_spacing for i in range(n_layers)]
    z_top = float(layer_zs[0])
    z_bot = float(layer_zs[-1])
    z_max = max(z_top, 1.0)
    t_min, t_max = t_arr[0], t_arr[-1]

    def t_to_z(ts: np.ndarray) -> np.ndarray:
        if t_max <= t_min:
            return np.full_like(np.asarray(ts, dtype=float), z_top)
        return z_top + (z_bot - z_top) * (
            (np.asarray(ts, dtype=float) - t_min) / (t_max - t_min)
        )

    if n_layers == 1:
        layer_alphas = [1.0]
    else:
        layer_alphas = [
            s.layer_alpha_floor
            + (1.0 - s.layer_alpha_floor) * (i / (n_layers - 1))
            for i in range(n_layers)
        ]

    def alpha_at_z(z: float | np.ndarray) -> np.ndarray | float:
        if z_top <= z_bot:
            return 1.0
        frac = (z_top - np.asarray(z, dtype=float)) / (z_top - z_bot)
        return s.layer_alpha_floor + (1.0 - s.layer_alpha_floor) * frac

    # --- bottom plate: 2D grid lines + xy trajectory projections ---------
    if show_bottom_grid:
        for k in range(grid.N):
            line_xy = (k - grid.center) * d
            ax.plot(
                [line_xy, line_xy], [bound_lo, bound_hi], [0, 0],
                color=s.bottom_grid_color, lw=s.bottom_grid_lw,
                alpha=s.bottom_grid_alpha,
            )
            ax.plot(
                [bound_lo, bound_hi], [line_xy, line_xy], [0, 0],
                color=s.bottom_grid_color, lw=s.bottom_grid_lw,
                alpha=s.bottom_grid_alpha,
            )

    if show_bottom_trajectory and t_max > t_min:
        for idx, atom in enumerate(ensemble.atomtrajs):
            sample_times = {t_min, t_max}
            for seg in atom.segments:
                if seg.duration <= 0:
                    continue
                if seg.end_time < t_min - 1e-15 or seg.start_time > t_max + 1e-15:
                    continue
                seg_lo = max(seg.start_time, t_min)
                seg_hi = min(seg.end_time, t_max)
                sample_times.update(
                    np.linspace(seg_lo, seg_hi, s.bottom_traj_samples_per_segment)
                    .tolist()
                )
            ts = np.asarray(sorted(sample_times))
            ij = np.asarray(
                [atom.position_at(float(tk)) for tk in ts], dtype=float,
            )
            xy = grid.ij_to_xy(ij)
            ax.plot(
                xy[:, 0], xy[:, 1], np.zeros_like(ts),
                color=colors[idx], lw=s.bottom_traj_lw, alpha=s.bottom_traj_alpha,
                solid_capstyle="round",
            )
            ax.scatter(
                [xy[0, 0]], [xy[0, 1]], [0],
                s=s.atom_size * 0.7, marker="o",
                facecolors="white", edgecolors=colors[idx],
                linewidths=1.0, depthshade=False,
            )
            ax.scatter(
                [xy[-1, 0]], [xy[-1, 1]], [0],
                s=s.atom_size * 0.7, marker="o",
                c=[colors[idx]], edgecolors="white",
                linewidths=0.6, depthshade=False,
            )

    # --- per-layer dressing ---------------------------------------------
    if show_grid_circles:
        circle_angles = np.linspace(
            0.0, 2.0 * np.pi, s.grid_circle_segments + 1
        )
        circle_cos = np.cos(circle_angles)
        circle_sin = np.sin(circle_angles)
        circle_radius = s.grid_circle_radius_frac * d

    if show_lattice_lines:
        row_color = to_rgba(s.lattice_line_colors.get("row", "#ff2828"))
        col_color = to_rgba(s.lattice_line_colors.get("col", "#29b0ff"))

    for i, (t, z) in enumerate(zip(t_arr, layer_zs)):
        layer_alpha = layer_alphas[i]

        if i == n_layers - 1:
            border_alpha_bot = layer_alpha
        else:
            border_alpha_bot = (
                layer_alpha
                * max(s.border_alpha_floor
                      * (s.border_alpha_falloff ** (n_layers - 1 - i)), 0.0)
            )
        border_alpha_top = layer_alpha

        if show_layer_plane:
            xx = np.array([[plane_lo, plane_hi], [plane_lo, plane_hi]])
            yy = np.array([[plane_lo, plane_lo], [plane_hi, plane_hi]])
            zz = np.full_like(xx, z, dtype=float)
            ax.plot_surface(
                xx, yy, zz,
                color=s.layer_plane_color,
                alpha=s.layer_plane_alpha * layer_alpha,
                shade=False, linewidth=0, edgecolor="none",
                antialiased=False,
            )

        if show_grid_dots:
            ii, jj = np.meshgrid(np.arange(grid.N), np.arange(grid.N), indexing="ij")
            dot_xy = grid.ij_to_xy(
                np.column_stack([ii.ravel(), jj.ravel()]).astype(float)
            )
            ax.scatter(
                dot_xy[:, 0], dot_xy[:, 1], np.full(len(dot_xy), z),
                s=s.grid_dot_size, c=s.grid_dot_color,
                alpha=s.grid_dot_alpha * layer_alpha,
                linewidths=0, depthshade=False,
            )

        if show_grid_circles:
            for ii in range(grid.N):
                for jj in range(grid.N):
                    cx, cy = grid.ij_to_xy(np.array([ii, jj], dtype=float))
                    line, = ax.plot(
                        cx + circle_radius * circle_cos,
                        cy + circle_radius * circle_sin,
                        np.full_like(circle_cos, z),
                        color=s.grid_circle_color,
                        lw=s.grid_circle_lw,
                        alpha=s.grid_circle_alpha * layer_alpha,
                        solid_capstyle="round",
                    )
                    if s.grid_circle_dashes is not None:
                        line.set_dashes(list(s.grid_circle_dashes))

        if show_lattice_lines:
            # one row line per j (varying i, fixed j) — colored row_color
            for jj in range(grid.N):
                y_row = (jj - grid.center) * d
                ax.plot(
                    [line_lo, line_hi], [y_row, y_row], [z, z],
                    color=row_color, lw=s.lattice_line_lw,
                    alpha=s.lattice_line_alpha * layer_alpha,
                    solid_capstyle="round",
                )
            # one col line per i (varying j, fixed i) — colored col_color
            for ii in range(grid.N):
                x_col = (ii - grid.center) * d
                ax.plot(
                    [x_col, x_col], [line_lo, line_hi], [z, z],
                    color=col_color, lw=s.lattice_line_lw,
                    alpha=s.lattice_line_alpha * layer_alpha,
                    solid_capstyle="round",
                )

        if show_grid_frame:
            ax.plot(
                [bound_lo, bound_hi], [bound_lo, bound_lo], [z, z],
                color=s.border_color, lw=s.border_lw, alpha=border_alpha_bot,
            )
            ax.plot(
                [bound_lo, bound_hi], [bound_hi, bound_hi], [z, z],
                color=s.border_color, lw=s.border_lw, alpha=border_alpha_top,
            )
            for xv in (bound_lo, bound_hi):
                ax.plot(
                    [xv, xv], [bound_lo, y_split], [z, z],
                    color=s.border_color, lw=s.border_lw,
                    alpha=border_alpha_bot,
                )
                ax.plot(
                    [xv, xv], [y_split, bound_hi], [z, z],
                    color=s.border_color, lw=s.border_lw,
                    alpha=border_alpha_top,
                )

        if s.layer_label_style != "none":
            denom = max(1, n_layers - 1)
            label_text = (
                f"{t * 1e9:.0f} ns" if s.layer_label_style == "ns"
                else f"{t * 1e6:.2f} us"
            )
            r, g, b, _ = to_rgba(s.label_color)
            ax.text(
                bound_lo + 0.02 * span, bound_hi - 0.02 * span, z, label_text,
                fontsize=s.label_fontsize_base + s.label_fontsize_growth * (i / denom),
                ha="left", va="top",
                color=(r, g, b, layer_alpha),
            )

    # --- trap tubes -----------------------------------------------------
    if show_traps:
        for atom in ensemble.atomtrajs:
            for seg in atom.segments:
                if seg.duration <= 0:
                    continue
                seg_lo = max(seg.start_time, t_min)
                seg_hi = min(seg.end_time, t_max)
                if seg_hi <= seg_lo:
                    continue
                ts = np.linspace(seg_lo, seg_hi, s.trap_samples_per_segment)
                ij = np.asarray(
                    [seg.position_at(float(tk)) for tk in ts], dtype=float,
                )
                xy = grid.ij_to_xy(ij)
                zs = t_to_z(ts)
                ramp = max(seg.duration * s.trap_ramp_frac, 1e-15)
                local = ts - seg.start_time
                env_in = _smoothstep(local / ramp)
                env_out = _smoothstep((seg.duration - local) / ramp)
                envelope = np.minimum(env_in, env_out)
                z_alpha = np.asarray(alpha_at_z(zs), dtype=float)
                wall_alpha = envelope * s.trap_max_alpha * z_alpha

                channel_color = s.trap_channel_colors.get(
                    seg.channel, s.trap_channel_colors.get(None, "#9a9a9a"),
                )
                _draw_trap_tube(
                    ax, xy, zs, wall_alpha, trap_radius,
                    to_rgb(channel_color), s.tube_facets,
                )

    # --- continuous trajectory ribbons ----------------------------------
    if show_trajectory and n_layers >= 2 and t_max > t_min:
        for idx, atom in enumerate(ensemble.atomtrajs):
            sample_times = {t_min, t_max}
            for seg in atom.segments:
                if seg.duration <= 0:
                    continue
                if seg.end_time < t_min - 1e-15 or seg.start_time > t_max + 1e-15:
                    continue
                seg_lo = max(seg.start_time, t_min)
                seg_hi = min(seg.end_time, t_max)
                sample_times.update(
                    np.linspace(seg_lo, seg_hi, s.trajectory_samples_per_segment)
                    .tolist()
                )
            ts = np.asarray(sorted(sample_times))
            ij = np.asarray(
                [atom.position_at(float(tk)) for tk in ts], dtype=float,
            )
            xy = grid.ij_to_xy(ij)
            zs = t_to_z(ts)
            ax.plot(
                xy[:, 0], xy[:, 1], zs,
                color=_darken(colors[idx], s.trajectory_darken),
                lw=s.trajectory_lw, alpha=s.trajectory_alpha,
            )

    # --- atom 2D cross-section discs per slice --------------------------
    for i, (t, z) in enumerate(zip(t_arr, layer_zs)):
        layer_alpha = layer_alphas[i]
        atom_xy = grid.ij_to_xy(ensemble.positions_at(t))
        rgba = np.array([to_rgba(c) for c in colors], dtype=float)
        rgba[:, 3] *= layer_alpha
        edge_r, edge_g, edge_b, edge_a = to_rgba(s.atom_edge_color)
        ax.scatter(
            atom_xy[:, 0], atom_xy[:, 1], np.full(len(atom_xy), z),
            s=s.atom_size, c=rgba,
            edgecolors=(edge_r, edge_g, edge_b, edge_a * layer_alpha),
            linewidths=s.atom_edge_lw, depthshade=False,
        )
        if show_atom_ids:
            label_offset = 0.32 * d
            for atom, (x, y), color in zip(ensemble.atomtrajs, atom_xy, colors):
                r, g, b, _ = to_rgba(color)
                ax.text(
                    x, y + label_offset, z,
                    str(int(atom.atom_id)),
                    ha="center", va="bottom", fontsize=s.atom_id_fontsize,
                    color=(0.55 * r, 0.55 * g, 0.55 * b, layer_alpha),
                )

        if show_motion_arrows:
            arrow_len = s.motion_arrow_length_frac * d
            for atom, (x, y), color in zip(ensemble.atomtrajs, atom_xy, colors):
                direction = _intended_direction(atom, t)
                if direction is None:
                    continue
                # convert ij-displacement into xy-displacement (i->x, j->y)
                ux, uy = grid.ij_to_xy(np.asarray(direction)) - grid.ij_to_xy(
                    np.zeros(2)
                )
                norm = float(np.hypot(ux, uy))
                if norm <= 0.0:
                    continue
                ux /= norm
                uy /= norm
                ax.quiver(
                    x, y, z, ux, uy, 0.0,
                    length=arrow_len,
                    arrow_length_ratio=s.motion_arrow_head_frac,
                    color=_darken(color, s.motion_arrow_darken),
                    lw=s.motion_arrow_lw,
                    alpha=s.motion_arrow_alpha * layer_alpha,
                    normalize=True,
                )

    # --- "time" arrow on the side ---------------------------------------
    if show_time_arrow and n_layers >= 2:
        arrow_x = bound_lo - 0.18 * span
        arrow_y = bound_lo - 0.05 * span
        z_pad = 0.4 * s.layer_spacing
        ax.plot(
            [arrow_x, arrow_x], [arrow_y, arrow_y],
            [z_top + z_pad, z_bot - z_pad],
            color=s.time_arrow_color, lw=s.time_arrow_lw, solid_capstyle="round",
        )
        head = 0.6 * s.layer_spacing
        ax.plot(
            [arrow_x, arrow_x - 0.04 * span], [arrow_y, arrow_y],
            [z_bot - z_pad, z_bot - z_pad + head],
            color=s.time_arrow_color, lw=s.time_arrow_lw, solid_capstyle="round",
        )
        ax.plot(
            [arrow_x, arrow_x + 0.04 * span], [arrow_y, arrow_y],
            [z_bot - z_pad, z_bot - z_pad + head],
            color=s.time_arrow_color, lw=s.time_arrow_lw, solid_capstyle="round",
        )
        ax.text(
            arrow_x, arrow_y, 0.5 * (z_top + z_bot), "time",
            ha="right", va="center",
            fontsize=s.time_arrow_fontsize, color=s.time_arrow_label_color,
        )

    # --- view + axes ----------------------------------------------------
    asp_z = max(1.1, n_layers / 6.0)
    ax.set_xlim(bound_lo, bound_hi)
    ax.set_ylim(bound_lo, bound_hi)
    ax.set_zlim(0, z_max)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False
        axis.pane.set_edgecolor("none")
        axis.line.set_color((1.0, 1.0, 1.0, 0.0))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.grid(False)
    ax.set_box_aspect((1.0, 1.0, asp_z))
    ax.view_init(elev=s.elev, azim=s.azim, vertical_axis=s.vertical_axis)
    ax.set_proj_type("persp", focal_length=s.focal_length)

    if title:
        ax.set_title(title, pad=s.title_pad, fontsize=s.title_fontsize)


def _draw_trap_tube(ax, centers_xy, zs_along, wall_alpha, radius, rgb, facets):
    """Render one trap as a 3D tube whose wall alpha varies along z."""
    n_along = len(centers_xy)
    if n_along < 2:
        return
    angles = np.linspace(0.0, 2.0 * np.pi, facets)
    cos_a = np.cos(angles)[None, :]
    sin_a = np.sin(angles)[None, :]
    xs = centers_xy[:, 0:1] + radius * cos_a
    ys = centers_xy[:, 1:2] + radius * sin_a
    zs_grid = np.broadcast_to(
        np.asarray(zs_along, dtype=float)[:, None], (n_along, facets),
    )

    face_alpha = 0.5 * (wall_alpha[:-1] + wall_alpha[1:])
    facecolors = np.zeros((n_along - 1, facets - 1, 4))
    facecolors[..., 0] = rgb[0]
    facecolors[..., 1] = rgb[1]
    facecolors[..., 2] = rgb[2]
    facecolors[..., 3] = face_alpha[:, None]
    ax.plot_surface(
        xs, ys, zs_grid, facecolors=facecolors, shade=False,
        linewidth=0, antialiased=True, edgecolor="none",
    )


def draw_stack_time_frame(
    motion: Any,
    t_values: Sequence[float],
    *,
    style: StackTimeStyle | None = None,
    title: str | None = None,
    **draw_kwargs: Any,
) -> tuple[Any, Any]:
    """Build a fresh figure with a 3D Axes and draw the stack on it."""
    s = style if style is not None else DEFAULT_STYLE
    fig = plt.figure(figsize=s.figsize, dpi=s.dpi)
    ax = fig.add_subplot(111, projection="3d")
    draw_stack_time(ax, motion, t_values, style=s, title=title, **draw_kwargs)
    return fig, ax


def save_stack_time(
    motion: Any,
    t_values: Sequence[float],
    output_path: str | Path,
    *,
    style: StackTimeStyle | None = None,
    title: str | None = None,
    transparent: bool = True,
    **draw_kwargs: Any,
) -> Path:
    """Render one time-stacked figure to a PNG file via `draw_stack_time`."""
    s = style if style is not None else DEFAULT_STYLE
    fig, _ = draw_stack_time_frame(
        motion, t_values, style=s, title=title, **draw_kwargs,
    )
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=s.dpi, transparent=transparent, bbox_inches="tight")
    plt.close(fig)
    return out
