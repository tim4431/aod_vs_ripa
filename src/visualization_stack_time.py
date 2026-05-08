"""Time-stacked 3D visualization for atom rearrangement motion.

Stacks time slices of an `AtomEnsemble` / `MovingSequence` along z, with
t=0 at the top and t=t_max at the bottom. Each atom is drawn as a 2D
cross-section circle (filled disc + thin white edge, matching
`src/visualization.draw_atoms`) on every requested layer; a continuous
trajectory polyline threads through the stack between layers. Each
motion segment renders as a 3D trap tube around the atom's path; the
tube is colored by the segment's RIPA channel (row=#ff2828,
col=#29b0ff) and its wall alpha follows a smoothstep on/off envelope.

Per-layer alpha is used as a "depth of time" cue: the bottom (latest)
layer is fully opaque and earlier layers fade toward the top, giving
the feel of recent state in front and past state receding behind.

Top-level entry points mirror `src.visualization`:

    `draw_stack_time(ax, motion, t_values, ...)` — populate a 3D Axes
    `draw_stack_time_frame(motion, t_values, ...)` — fresh figure + 3D ax
    `save_stack_time(motion, t_values, output_path, ...)` — write a PNG
"""

from __future__ import annotations

import sys
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

DEFAULT_FIGSIZE = (6.6, 6.4)
DEFAULT_DPI = 220
LAYER_SPACING = 6.0
BORDER_LW = 0.9
BORDER_ALPHA_FLOOR = 0.18
BORDER_ALPHA_FALLOFF = 0.72
BORDER_SPLIT_FRAC = 0.82
GRID_DOT_COLOR = "#cccccc"
GRID_DOT_ALPHA = 0.18
GRID_DOT_SIZE = 4.0
ATOM_EDGE_COLOR = "#ffffff"
ATOM_EDGE_LW = 0.6
TRAP_RADIUS_FRAC = 0.30
TRAP_RAMP_FRAC = 0.18
TRAP_MAX_ALPHA = 0.62
TRAP_CHANNEL_COLORS: dict[str | None, str] = {
    "row": "#ff2828",
    "col": "#29b0ff",
    "aod": "#9a9a9a",
    None: "#9a9a9a",
}
LAYER_ALPHA_FLOOR = 0.32           # alpha of the top (earliest) layer
TRAJECTORY_LW = 1.4
TRAJECTORY_ALPHA = 0.7
TRAJECTORY_SAMPLES_PER_SEGMENT = 24
TRAP_SAMPLES_PER_SEGMENT = 36
TUBE_FACETS = 32

VIEW_ELEV = 32.0
VIEW_AZIM = -62.0
VIEW_VERTICAL_AXIS = "z"
BOTTOM_GRID_COLOR = "#9a9a9a"
BOTTOM_GRID_LW = 0.6
BOTTOM_GRID_ALPHA = 0.55
BOTTOM_TRAJ_LW = 2.0
BOTTOM_TRAJ_ALPHA = 0.85
BOTTOM_TRAJ_SAMPLES_PER_SEGMENT = 24


def _smoothstep(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _darken(color: Any, k: float = 0.55) -> tuple[float, float, float, float]:
    r, g, b, a = to_rgba(color)
    return (k * r, k * g, k * b, a)


def draw_stack_time(
    ax: Any,
    motion: Any,
    t_values: Sequence[float],
    *,
    atom_colors: Mapping[int, Any] | Iterable[Any] | None = None,
    show_atom_ids: bool = False,
    show_traps: bool = True,
    show_grid_dots: bool = True,
    show_grid_frame: bool = True,
    show_trajectory: bool = True,
    show_time_arrow: bool = True,
    show_bottom_grid: bool = True,
    show_bottom_trajectory: bool = True,
    layer_label_style: LayerLabelStyle = "us",
    layer_spacing: float = LAYER_SPACING,
    layer_alpha_floor: float = LAYER_ALPHA_FLOOR,
    atom_size: float = ATOM_SIZE,
    trap_radius_frac: float = TRAP_RADIUS_FRAC,
    trap_max_alpha: float = TRAP_MAX_ALPHA,
    trap_channel_colors: Mapping[str | None, Any] = TRAP_CHANNEL_COLORS,
    elev: float = VIEW_ELEV,
    azim: float = VIEW_AZIM,
    vertical_axis: str = VIEW_VERTICAL_AXIS,
    lw: float = 1.0,
    title: str | None = None,
) -> None:
    """Draw a time-stacked 3D figure of `motion` on a 3D `ax`.

    Atoms are 2D scatter discs (one per layer); the trap tubes use
    `trap_channel_colors` to color row vs col motion separately. Each
    layer's drawn elements are scaled by a depth-based alpha — bottom
    (latest) layer fully opaque, top fades toward `layer_alpha_floor`.
    """
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
    bound_lo = (0.0 - grid.center) * d - d / 2.0
    bound_hi = ((grid.N - 1.0) - grid.center) * d + d / 2.0
    span = bound_hi - bound_lo
    y_split = bound_lo + BORDER_SPLIT_FRAC * span
    trap_radius = trap_radius_frac * d

    # Top layer (smallest t) at largest z; layers descend toward z=0.
    layer_zs = [(n_layers - 1 - i) * layer_spacing for i in range(n_layers)]
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

    # Layer alpha schedule: bottom (latest) = 1, top (earliest) = floor.
    # Linear in stack position so the gradient reads as smooth depth.
    if n_layers == 1:
        layer_alphas = [1.0]
    else:
        layer_alphas = [
            layer_alpha_floor
            + (1.0 - layer_alpha_floor) * (i / (n_layers - 1))
            for i in range(n_layers)
        ]

    def alpha_at_z(z: float | np.ndarray) -> np.ndarray | float:
        # Alpha as a continuous function of z, matching the per-layer schedule.
        # z_top -> floor, z_bot -> 1.0.
        if z_top <= z_bot:
            return 1.0
        frac = (z_top - np.asarray(z, dtype=float)) / (z_top - z_bot)
        return layer_alpha_floor + (1.0 - layer_alpha_floor) * frac

    # --- bottom plate: 2D grid lines + xy trajectory projections at z=0 --
    # Drawn first so layers above paint on top of it.
    if show_bottom_grid:
        for k in range(grid.N):
            line_xy = (k - grid.center) * d
            ax.plot(
                [line_xy, line_xy], [bound_lo, bound_hi], [0, 0],
                color=BOTTOM_GRID_COLOR, lw=BOTTOM_GRID_LW,
                alpha=BOTTOM_GRID_ALPHA,
            )
            ax.plot(
                [bound_lo, bound_hi], [line_xy, line_xy], [0, 0],
                color=BOTTOM_GRID_COLOR, lw=BOTTOM_GRID_LW,
                alpha=BOTTOM_GRID_ALPHA,
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
                    np.linspace(seg_lo, seg_hi, BOTTOM_TRAJ_SAMPLES_PER_SEGMENT)
                    .tolist()
                )
            ts = np.asarray(sorted(sample_times))
            ij = np.asarray(
                [atom.position_at(float(tk)) for tk in ts], dtype=float,
            )
            xy = grid.ij_to_xy(ij)
            ax.plot(
                xy[:, 0], xy[:, 1], np.zeros_like(ts),
                color=colors[idx], lw=BOTTOM_TRAJ_LW, alpha=BOTTOM_TRAJ_ALPHA,
                solid_capstyle="round",
            )
            # mark start (open circle) and end (filled) on the floor
            ax.scatter(
                [xy[0, 0]], [xy[0, 1]], [0],
                s=atom_size * 0.7, marker="o",
                facecolors="white", edgecolors=colors[idx],
                linewidths=1.0, depthshade=False,
            )
            ax.scatter(
                [xy[-1, 0]], [xy[-1, 1]], [0],
                s=atom_size * 0.7, marker="o",
                c=[colors[idx]], edgecolors="white",
                linewidths=0.6, depthshade=False,
            )

    # --- per-layer dressing: frame + grid dots + time label --------------
    border_lw = BORDER_LW * lw
    for i, (t, z) in enumerate(zip(t_arr, layer_zs)):
        layer_alpha = layer_alphas[i]

        if i == n_layers - 1:
            border_alpha_bot = layer_alpha
        else:
            border_alpha_bot = (
                layer_alpha
                * max(BORDER_ALPHA_FLOOR
                      * (BORDER_ALPHA_FALLOFF ** (n_layers - 1 - i)), 0.0)
            )
        border_alpha_top = layer_alpha

        if show_grid_dots:
            ii, jj = np.meshgrid(np.arange(grid.N), np.arange(grid.N), indexing="ij")
            dot_xy = grid.ij_to_xy(
                np.column_stack([ii.ravel(), jj.ravel()]).astype(float)
            )
            ax.scatter(
                dot_xy[:, 0], dot_xy[:, 1], np.full(len(dot_xy), z),
                s=GRID_DOT_SIZE, c=GRID_DOT_COLOR,
                alpha=GRID_DOT_ALPHA * layer_alpha,
                linewidths=0, depthshade=False,
            )

        if show_grid_frame:
            ax.plot(
                [bound_lo, bound_hi], [bound_lo, bound_lo], [z, z],
                color="black", lw=border_lw, alpha=border_alpha_bot,
            )
            ax.plot(
                [bound_lo, bound_hi], [bound_hi, bound_hi], [z, z],
                color="black", lw=border_lw, alpha=border_alpha_top,
            )
            for xv in (bound_lo, bound_hi):
                ax.plot(
                    [xv, xv], [bound_lo, y_split], [z, z],
                    color="black", lw=border_lw, alpha=border_alpha_bot,
                )
                ax.plot(
                    [xv, xv], [y_split, bound_hi], [z, z],
                    color="black", lw=border_lw, alpha=border_alpha_top,
                )

        if layer_label_style != "none":
            denom = max(1, n_layers - 1)
            label_text = (
                f"{t * 1e9:.0f} ns" if layer_label_style == "ns"
                else f"{t * 1e6:.2f} us"
            )
            ax.text(
                bound_lo + 0.02 * span, bound_hi - 0.02 * span, z, label_text,
                fontsize=6.5 + 1.2 * (i / denom),
                ha="left", va="top",
                color=(0.15, 0.15, 0.15, layer_alpha),
            )

    # --- trap tubes: per-channel color, layer-alpha modulated -----------
    if show_traps:
        for atom in ensemble.atomtrajs:
            for seg in atom.segments:
                if seg.duration <= 0:
                    continue
                seg_lo = max(seg.start_time, t_min)
                seg_hi = min(seg.end_time, t_max)
                if seg_hi <= seg_lo:
                    continue
                ts = np.linspace(seg_lo, seg_hi, TRAP_SAMPLES_PER_SEGMENT)
                ij = np.asarray(
                    [seg.position_at(float(tk)) for tk in ts], dtype=float,
                )
                xy = grid.ij_to_xy(ij)
                zs = t_to_z(ts)
                ramp = max(seg.duration * TRAP_RAMP_FRAC, 1e-15)
                local = ts - seg.start_time
                env_in = _smoothstep(local / ramp)
                env_out = _smoothstep((seg.duration - local) / ramp)
                envelope = np.minimum(env_in, env_out)
                z_alpha = np.asarray(alpha_at_z(zs), dtype=float)
                wall_alpha = envelope * trap_max_alpha * z_alpha

                channel_color = trap_channel_colors.get(
                    seg.channel, trap_channel_colors.get(None, "#9a9a9a"),
                )
                _draw_trap_tube(
                    ax, xy, zs, wall_alpha, trap_radius, to_rgb(channel_color),
                )

    # --- continuous trajectory ribbons through the stack -----------------
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
                    np.linspace(seg_lo, seg_hi, TRAJECTORY_SAMPLES_PER_SEGMENT)
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
                color=_darken(colors[idx], 0.65),
                lw=TRAJECTORY_LW, alpha=TRAJECTORY_ALPHA,
            )

    # --- atom 2D cross-section discs per slice ---------------------------
    for i, (t, z) in enumerate(zip(t_arr, layer_zs)):
        layer_alpha = layer_alphas[i]
        atom_xy = grid.ij_to_xy(ensemble.positions_at(t))
        # Per-atom RGBA so each atom's alpha is multiplied with the layer's.
        rgba = np.array([to_rgba(c) for c in colors], dtype=float)
        rgba[:, 3] *= layer_alpha
        ax.scatter(
            atom_xy[:, 0], atom_xy[:, 1], np.full(len(atom_xy), z),
            s=atom_size, c=rgba,
            edgecolors=(1.0, 1.0, 1.0, layer_alpha),
            linewidths=ATOM_EDGE_LW, depthshade=False,
        )
        if show_atom_ids:
            label_offset = 0.32 * d
            for atom, (x, y), color in zip(ensemble.atomtrajs, atom_xy, colors):
                r, g, b, _ = to_rgba(color)
                ax.text(
                    x, y + label_offset, z,
                    str(int(atom.atom_id)),
                    ha="center", va="bottom", fontsize=6.5,
                    color=(0.55 * r, 0.55 * g, 0.55 * b, layer_alpha),
                )

    # --- "time" arrow on the side, downward ------------------------------
    if show_time_arrow and n_layers >= 2:
        arrow_x = bound_lo - 0.18 * span
        arrow_y = bound_lo - 0.05 * span
        z_pad = 0.4 * layer_spacing
        ax.plot(
            [arrow_x, arrow_x], [arrow_y, arrow_y],
            [z_top + z_pad, z_bot - z_pad],
            color="#444444", lw=1.4, solid_capstyle="round",
        )
        head = 0.6 * layer_spacing
        ax.plot(
            [arrow_x, arrow_x - 0.04 * span], [arrow_y, arrow_y],
            [z_bot - z_pad, z_bot - z_pad + head],
            color="#444444", lw=1.4, solid_capstyle="round",
        )
        ax.plot(
            [arrow_x, arrow_x + 0.04 * span], [arrow_y, arrow_y],
            [z_bot - z_pad, z_bot - z_pad + head],
            color="#444444", lw=1.4, solid_capstyle="round",
        )
        ax.text(
            arrow_x, arrow_y, 0.5 * (z_top + z_bot), "time",
            ha="right", va="center", fontsize=8.5, color="#333333",
        )

    # --- view + axes -----------------------------------------------------
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
    ax.view_init(elev=elev, azim=azim, vertical_axis=vertical_axis)
    ax.set_proj_type("persp", focal_length=0.45)

    if title:
        ax.set_title(title, pad=18, fontsize=11)


def _draw_trap_tube(ax, centers_xy, zs_along, wall_alpha, radius, rgb):
    """Render one trap as a 3D tube whose wall alpha varies along z.

    Cross-section is a horizontal circle (xy plane). `wall_alpha` is the
    final per-vertex tube alpha (already including the on/off envelope
    AND the layer-alpha depth modulation).
    """
    n_along = len(centers_xy)
    if n_along < 2:
        return
    angles = np.linspace(0.0, 2.0 * np.pi, TUBE_FACETS)
    cos_a = np.cos(angles)[None, :]
    sin_a = np.sin(angles)[None, :]
    xs = centers_xy[:, 0:1] + radius * cos_a
    ys = centers_xy[:, 1:2] + radius * sin_a
    zs_grid = np.broadcast_to(
        np.asarray(zs_along, dtype=float)[:, None], (n_along, TUBE_FACETS),
    )

    face_alpha = 0.5 * (wall_alpha[:-1] + wall_alpha[1:])
    facecolors = np.zeros((n_along - 1, TUBE_FACETS - 1, 4))
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
    figsize: tuple[float, float] = DEFAULT_FIGSIZE,
    dpi: int = DEFAULT_DPI,
    title: str | None = None,
    **draw_kwargs: Any,
) -> tuple[Any, Any]:
    """Build a fresh figure with a 3D Axes and draw the stack on it."""
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111, projection="3d")
    draw_stack_time(ax, motion, t_values, title=title, **draw_kwargs)
    return fig, ax


def save_stack_time(
    motion: Any,
    t_values: Sequence[float],
    output_path: str | Path,
    *,
    figsize: tuple[float, float] = DEFAULT_FIGSIZE,
    dpi: int = DEFAULT_DPI,
    title: str | None = None,
    transparent: bool = True,
    **draw_kwargs: Any,
) -> Path:
    """Render one time-stacked figure to a PNG file via `draw_stack_time`."""
    fig, _ = draw_stack_time_frame(
        motion, t_values, figsize=figsize, dpi=dpi, title=title, **draw_kwargs,
    )
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, transparent=transparent, bbox_inches="tight")
    plt.close(fig)
    return out
