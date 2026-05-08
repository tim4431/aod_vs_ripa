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
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

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

    # Layer stack geometry / depth fade. `layer_alpha_floor` is the
    # alpha of the top (earliest) layer; the bottom layer is at 1.0.
    # Keep this fairly high so each layer reads as a distinct plate
    # without the top one washing out — the bottom-plate trajectory and
    # trap tubes provide the "depth" cue, not faded plates.
    layer_spacing: float = 6.0
    layer_alpha_floor: float = 0.85

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

    # Grid circles (filled disc at each lattice site, no edge)
    grid_circle_color: Any = "#9a9a9a"
    grid_circle_radius_frac: float = 0.30
    grid_circle_alpha: float = 0.18
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
    layer_plane_alpha: float = 0.22
    layer_plane_extension_frac: float = 0.55

    # Per-layer planned-trajectory dashes: a *dashed line* drawn on
    # each atom from its current xy to its segment's end_pos at the
    # layer time. No arrow head — the dashes carry the "planned, not
    # yet executed" cue. Color is keyed by the segment's `channel`.
    motion_arrow_lw: float = 1.8
    motion_arrow_alpha: float = 1.0
    motion_arrow_length_frac: float = 1.0   # line length / actual move xy length
    motion_arrow_dashes: tuple[float, float] = (4.0, 3.0)
    motion_arrow_channel_colors: Mapping[Any, Any] = field(
        default_factory=lambda: {
            "row": "#ff2828",
            "col": "#29b0ff",
            "aod": "#444444",
            None: "#444444",
        }
    )
    motion_arrow_default_color: Any = "#444444"

    # Fresh atom color: used for atom discs (and trajectory polylines)
    # while the atom's grid coordinate is outside [0, N-1] in either
    # axis — visual cue that the atom is a fresh-loaded replenishment
    # not yet incorporated into the lattice.
    fresh_atom_color: Any = "#f1c40f"

    # Loss marker: black cross drawn on the trajectory at the loss
    # position, plus an extra fade-out window on the trap tube
    # approaching `lost_at` (mirrors `trap_ramp_frac` semantics).
    loss_marker_color: Any = "#000000"
    loss_marker_size: float = 90.0
    loss_marker_lw: float = 2.0
    loss_fade_frac: float = 0.20

    # Replacement connector: dashed line drawn between a lost atom's
    # last in-array position and the replacement atom's first in-array
    # position. Color defaults to the lost atom's normal color, but
    # can be overridden via `replacement_connector_color`.
    replacement_connector_color: Any | None = None
    replacement_connector_lw: float = 1.4
    replacement_connector_alpha: float = 0.85
    replacement_connector_dashes: tuple[float, float] = (5.0, 4.0)

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
    # Smoothstep fade ramp at each end of a normal trap tube, expressed
    # as a fraction of the segment duration. The segment is sampled in
    # time but rendered along z, so this is effectively a *depth* fade
    # near the segment's start_time (high z) and end_time (low z). Keep
    # the value small so the trap stays visually aligned with the atom
    # at both ends; large values would lag the atom by the ramp width.
    trap_ramp_frac: float = 0.10
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


def _draw_loss_cross(
    ax: Any, atom: Any, grid: Any, t_to_z: Any, s: "StackTimeStyle",
) -> None:
    """Black `x` marker at the (xy, z) where this atom was lost."""
    ij_loss = np.asarray(atom.position_at(atom.lost_at), dtype=float)
    xy_loss = grid.ij_to_xy(ij_loss)
    z_loss = float(t_to_z(np.array([atom.lost_at], dtype=float))[0])
    ax.scatter(
        [float(xy_loss[0])], [float(xy_loss[1])], [z_loss],
        s=s.loss_marker_size, c=s.loss_marker_color, marker="x",
        linewidths=s.loss_marker_lw, depthshade=False,
    )


def _is_outside_grid(ij: Any, n: int, eps: float = 1e-6) -> bool:
    """True if the (i, j) grid coordinate falls outside `[0, n-1]` in either axis."""
    i, j = float(ij[0]), float(ij[1])
    return i < -eps or i > n - 1 + eps or j < -eps or j > n - 1 + eps


def _is_fresh_at(atom: Any, t: float, eps: float = 1e-9) -> bool:
    """True while the trap has not yet addressed `atom` at time `t`.

    An atom is "fresh" (just-loaded reservoir state) until its first
    segment starts; once any segment has begun, the atom counts as
    addressed-by-the-trap and renders in its normal color regardless
    of where it currently sits geometrically. This decouples the
    visual "freshness" cue from grid bounds — useful for replenishment
    atoms whose first move begins outside the lattice and ends inside.
    """
    for seg in atom.segments:
        if seg.duration > 0 and seg.start_time <= t + eps:
            return False
    return True


def _last_grid_site_and_time(atom: Any) -> tuple[tuple[float, float], float] | None:
    """The integer grid site the atom was last at, and the time it was there.

    For an atom lost mid-segment, this is `(seg.start_pos, seg.start_time)`
    of the segment in which the loss occurred — i.e., the last lattice
    site the atom physically occupied before it went mid-flight. For a
    loss during a rest period, returns the prior segment's end_pos at
    its end_time. Returns `None` if the atom was never lost.
    """
    if atom.lost_at is None:
        return None
    lost_at = float(atom.lost_at)
    for seg in atom.segments:
        if (
            seg.duration > 0
            and seg.start_time - 1e-12 <= lost_at <= seg.end_time + 1e-12
        ):
            return (
                (float(seg.start_pos[0]), float(seg.start_pos[1])),
                float(seg.start_time),
            )
    last_pos = (float(atom.initial_pos[0]), float(atom.initial_pos[1]))
    last_time = 0.0
    for seg in atom.segments:
        if seg.duration > 0 and seg.end_time <= lost_at + 1e-12:
            last_pos = (float(seg.end_pos[0]), float(seg.end_pos[1]))
            last_time = float(seg.end_time)
    return last_pos, last_time


def _loading_destination_and_time(
    atom: Any, n: int,
) -> tuple[tuple[float, float], float] | None:
    """The grid site where this atom is first loaded and the arrival time.

    Walks the atom's segments and returns the first one whose `end_pos`
    sits inside `[0, n-1]` — that's the segment that brings the atom
    into the array. `None` if no such segment exists.
    """
    for seg in atom.segments:
        if seg.duration <= 0:
            continue
        if not _is_outside_grid(seg.end_pos, n):
            return (
                (float(seg.end_pos[0]), float(seg.end_pos[1])),
                float(seg.end_time),
            )
    return None


def _array_entry_time(atom: Any, n: int, eps: float = 1e-9) -> float | None:
    """Earliest time the atom's position is inside grid `[0, n-1]`.

    Returns `None` if the atom never enters the grid. Used to anchor
    the replacement connector at the moment the new atom takes over
    the lost atom's slot.
    """
    if not _is_outside_grid(atom.initial_pos, n):
        return 0.0
    for seg in atom.segments:
        if seg.duration <= 0:
            continue
        if not _is_outside_grid(seg.start_pos, n):
            return seg.start_time
        if not _is_outside_grid(seg.end_pos, n):
            # crosses inside during this segment — bisect on path fraction
            lo, hi = 0.0, 1.0
            for _ in range(40):
                mid = 0.5 * (lo + hi)
                t_mid = seg.start_time + mid * seg.duration
                if _is_outside_grid(seg.position_at(t_mid), n):
                    lo = mid
                else:
                    hi = mid
            return seg.start_time + hi * seg.duration
    return None


def _intended_segment(atom: Any, t: float, eps: float = 1e-9) -> Any | None:
    """Return the move active at or beginning at time `t`.

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
            return seg
    return None


def _draw_planned_dash(
    ax: Any,
    x0: float, y0: float, z: float,
    ux: float, uy: float,
    length: float,
    dashes: tuple[float, float],
    color: Any,
    lw: float,
    alpha: float,
) -> None:
    """Dashed line from (x0, y0, z) extending in (ux, uy) direction for `length` units."""
    end_x = x0 + ux * length
    end_y = y0 + uy * length
    line, = ax.plot(
        [x0, end_x], [y0, end_y], [z, z],
        color=color, lw=lw, alpha=alpha, solid_capstyle="butt",
    )
    line.set_dashes(list(dashes))


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
    replacements: Sequence[tuple[int, int]] | None = None,
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

    # Auto-expand xy bounds to include any segment endpoints / initial
    # positions that fall outside the lattice (replenishment trajectories
    # loaded from off-grid). Lattice-only decorations still use bound_lo /
    # bound_hi; only the final ax.set_xlim / set_ylim use the expanded box.
    xy_lo = bound_lo
    xy_hi = bound_hi
    margin = 0.5 * d
    for atom in ensemble.atomtrajs:
        ij_pts = [np.asarray(atom.initial_pos, dtype=float)]
        for seg in atom.segments:
            ij_pts.append(np.asarray(seg.start_pos, dtype=float))
            ij_pts.append(np.asarray(seg.end_pos, dtype=float))
        for ij in ij_pts:
            x, y = grid.ij_to_xy(ij)
            xy_lo = min(xy_lo, float(x) - margin, float(y) - margin)
            xy_hi = max(xy_hi, float(x) + margin, float(y) + margin)

    # When the bottom plate is enabled, lift the whole stack by one
    # `layer_spacing` so the bottom layer doesn't share z=0 with the
    # plate (would otherwise z-fight). Top layer (smallest t) ends up
    # at the largest z; layers descend with increasing t.
    z_floor = s.layer_spacing if (show_bottom_grid or show_bottom_trajectory) else 0.0
    layer_zs = [
        z_floor + (n_layers - 1 - i) * s.layer_spacing for i in range(n_layers)
    ]
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

    # --- bottom plate: row/col-colored extending lattice lines + xy
    #     trajectory projections at z=0. The "grid" on the bottom uses
    #     the same row/col color rule as the layer lattice lines, but
    #     with the bottom-plate styling (alpha, lw).
    if show_bottom_grid:
        bot_row_color = to_rgba(s.lattice_line_colors.get("row", "#ff2828"))
        bot_col_color = to_rgba(s.lattice_line_colors.get("col", "#29b0ff"))
        for jj in range(grid.N):
            y_row = (jj - grid.center) * d
            ax.plot(
                [line_lo, line_hi], [y_row, y_row], [0, 0],
                color=bot_row_color, lw=s.bottom_grid_lw,
                alpha=s.bottom_grid_alpha,
                solid_capstyle="round",
            )
        for ii in range(grid.N):
            x_col = (ii - grid.center) * d
            ax.plot(
                [x_col, x_col], [line_lo, line_hi], [0, 0],
                color=bot_col_color, lw=s.bottom_grid_lw,
                alpha=s.bottom_grid_alpha,
                solid_capstyle="round",
            )

    if show_bottom_trajectory and t_max > t_min:
        fresh_rgba = to_rgba(s.fresh_atom_color)

        def _bottom_seg_color(atom_, t_mid: float, normal_rgba_):
            # color as fresh while no segment has yet started for this atom
            return fresh_rgba if _is_fresh_at(atom_, t_mid) else normal_rgba_

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
            # truncate the trajectory at the loss event
            traj_end = t_max if atom.lost_at is None else min(t_max, atom.lost_at)
            ts = np.asarray(sorted(tk for tk in sample_times if tk <= traj_end + 1e-15))
            if atom.lost_at is not None:
                ts = np.append(ts[ts < atom.lost_at - 1e-15], atom.lost_at)
            if len(ts) < 2:
                continue
            ij = np.asarray(
                [atom.position_at(float(tk)) for tk in ts], dtype=float,
            )
            xy = grid.ij_to_xy(ij)

            # per-segment color: fresh while either endpoint is off-grid,
            # else atom's normal color
            normal_rgba = to_rgba(colors[idx])
            points = np.stack([xy[:, 0], xy[:, 1], np.zeros(len(ts))], axis=1)
            segs = np.stack([points[:-1], points[1:]], axis=1)
            seg_colors = []
            for k in range(len(segs)):
                t_mid = 0.5 * (float(ts[k]) + float(ts[k + 1]))
                seg_colors.append(_bottom_seg_color(atom, t_mid, normal_rgba))
            lc = Line3DCollection(
                segs, colors=seg_colors,
                linewidths=s.bottom_traj_lw,
                alpha=s.bottom_traj_alpha,
            )
            ax.add_collection3d(lc)

            start_color = (
                fresh_rgba if _is_fresh_at(atom, float(ts[0])) else normal_rgba
            )
            ax.scatter(
                [xy[0, 0]], [xy[0, 1]], [0],
                s=s.atom_size * 0.7, marker="o",
                facecolors="white", edgecolors=[start_color],
                linewidths=1.0, depthshade=False,
            )
            if atom.lost_at is None:
                # only mark a finished trajectory with the closed end disc;
                # for a lost atom the projection just stops at the loss
                # position with no extra marker — the in-stack cross is
                # the single source of truth.
                end_color = (
                    fresh_rgba if _is_fresh_at(atom, float(ts[-1])) else normal_rgba
                )
                ax.scatter(
                    [xy[-1, 0]], [xy[-1, 1]], [0],
                    s=s.atom_size * 0.7, marker="o",
                    c=[end_color], edgecolors="white",
                    linewidths=0.6, depthshade=False,
                )

    # --- per-layer dressing ---------------------------------------------
    if show_grid_circles:
        # filled disc at every site — no edge; one batched Poly3DCollection
        # per layer keeps it cheap.
        circle_angles = np.linspace(
            0.0, 2.0 * np.pi, s.grid_circle_segments + 1,
        )[:-1]
        circle_cos = np.cos(circle_angles)
        circle_sin = np.sin(circle_angles)
        circle_radius = s.grid_circle_radius_frac * d
        ii_grid, jj_grid = np.meshgrid(
            np.arange(grid.N), np.arange(grid.N), indexing="ij",
        )
        site_xy = grid.ij_to_xy(
            np.column_stack([ii_grid.ravel(), jj_grid.ravel()]).astype(float)
        )

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
            polys = []
            for cx, cy in site_xy:
                polys.append([
                    (cx + circle_radius * circle_cos[k],
                     cy + circle_radius * circle_sin[k],
                     z)
                    for k in range(len(circle_angles))
                ])
            cr, cg, cb, _ca = to_rgba(s.grid_circle_color)
            face = (cr, cg, cb, s.grid_circle_alpha * layer_alpha)
            coll = Poly3DCollection(
                polys,
                facecolors=[face] * len(polys),
                edgecolors="none",
                linewidths=0,
            )
            ax.add_collection3d(coll)

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

    # --- interleaved stack pass: from bottom z up, draw each layer's
    #     plane + the trajectory polylines + trap tubes that exit upward
    #     from this plane to the next-up plane. The order trajectory →
    #     trap → next plane ensures: (a) the trap covers the trajectory,
    #     (b) traps/trajectories sit above the next-time plane (lower z)
    #     but below the current-time plane (higher z, drawn next).
    fresh_dark = _darken(s.fresh_atom_color, s.trajectory_darken)

    def _draw_layer_plane(z_layer: float, layer_alpha_: float) -> None:
        if not show_layer_plane:
            return
        xx = np.array([[plane_lo, plane_hi], [plane_lo, plane_hi]])
        yy = np.array([[plane_lo, plane_lo], [plane_hi, plane_hi]])
        zz = np.full_like(xx, z_layer, dtype=float)
        surf = ax.plot_surface(
            xx, yy, zz,
            color=s.layer_plane_color,
            alpha=s.layer_plane_alpha * layer_alpha_,
            shade=False, linewidth=0, edgecolor="none",
            antialiased=False,
        )
        # nail the depth-sort to the plane's z so the surface always
        # sorts above traps with avg-z below it and below traps above it
        try:
            surf.set_sort_zpos(float(z_layer))
        except AttributeError:
            pass

    def _draw_trap_for_seg(atom: Any, seg: Any, t_lo: float, t_hi: float) -> None:
        if not show_traps or seg.duration <= 0:
            return
        seg_lo = max(seg.start_time, t_lo)
        seg_hi = min(seg.end_time, t_hi)
        truncated_by_loss = False
        if (
            atom.lost_at is not None
            and atom.lost_at < seg_hi - 1e-15
            and atom.lost_at >= seg_lo - 1e-15
        ):
            seg_hi = atom.lost_at
            truncated_by_loss = True
        if (
            atom.lost_at is not None
            and atom.lost_at <= seg_lo + 1e-15
        ):
            return
        if seg_hi <= seg_lo:
            return
        ts = np.linspace(seg_lo, seg_hi, s.trap_samples_per_segment)
        ij = np.asarray(
            [seg.position_at(float(tk)) for tk in ts], dtype=float,
        )
        xy = grid.ij_to_xy(ij)
        zs = t_to_z(ts)
        effective_end = seg_hi
        if s.trap_ramp_frac > 0.0:
            ramp_in = max(seg.duration * s.trap_ramp_frac, 1e-15)
            env_in = _smoothstep((ts - seg.start_time) / ramp_in)
        else:
            env_in = np.ones_like(ts)
        if truncated_by_loss:
            ramp_out_frac = s.loss_fade_frac
        else:
            ramp_out_frac = s.trap_ramp_frac
        if ramp_out_frac > 0.0:
            ramp_out = max(seg.duration * ramp_out_frac, 1e-15)
            env_out = _smoothstep((effective_end - ts) / ramp_out)
        else:
            env_out = np.ones_like(ts)
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

    def _draw_trajectory_in_range(t_lo: float, t_hi: float) -> None:
        if not show_trajectory or n_layers < 2 or t_max <= t_min:
            return
        for idx_, atom in enumerate(ensemble.atomtrajs):
            sample_times = {t_lo, t_hi}
            for seg in atom.segments:
                if seg.duration <= 0:
                    continue
                if seg.end_time < t_lo - 1e-15 or seg.start_time > t_hi + 1e-15:
                    continue
                seg_lo = max(seg.start_time, t_lo)
                seg_hi = min(seg.end_time, t_hi)
                sample_times.update(
                    np.linspace(seg_lo, seg_hi, s.trajectory_samples_per_segment)
                    .tolist()
                )
            range_end = t_hi if atom.lost_at is None else min(t_hi, atom.lost_at)
            ts = np.asarray(
                sorted(tk for tk in sample_times if t_lo - 1e-15 <= tk <= range_end + 1e-15)
            )
            if (
                atom.lost_at is not None
                and t_lo - 1e-15 <= atom.lost_at <= t_hi + 1e-15
            ):
                ts = np.unique(np.append(
                    ts[ts < atom.lost_at - 1e-15], atom.lost_at,
                ))
            if len(ts) < 2:
                continue
            ij = np.asarray(
                [atom.position_at(float(tk)) for tk in ts], dtype=float,
            )
            xy = grid.ij_to_xy(ij)
            zs = t_to_z(ts)
            normal_dark = _darken(colors[idx_], s.trajectory_darken)
            points = np.stack([xy[:, 0], xy[:, 1], zs], axis=1)
            segs = np.stack([points[:-1], points[1:]], axis=1)
            seg_colors = []
            for k in range(len(segs)):
                t_mid = 0.5 * (float(ts[k]) + float(ts[k + 1]))
                seg_colors.append(
                    fresh_dark if _is_fresh_at(atom, t_mid) else normal_dark
                )
            lc = Line3DCollection(
                segs, colors=seg_colors,
                linewidths=s.trajectory_lw,
                alpha=s.trajectory_alpha,
            )
            ax.add_collection3d(lc)

    # Walk from the bottom-most layer up to the top.
    # i = n_layers - 1 is at lowest z; i = 0 is at highest z.
    for i in range(n_layers - 1, -1, -1):
        # 1. draw this layer's plane first — it sits below any trap that
        #    rises above it.
        _draw_layer_plane(layer_zs[i], layer_alphas[i])
        # 2. trajectory polylines in the range above this layer
        if i > 0:
            t_above_lo = t_arr[i - 1]
            t_above_hi = t_arr[i]
            _draw_trajectory_in_range(t_above_lo, t_above_hi)
            # 3. then trap tubes (drawn after trajectory so trap covers it)
            for atom in ensemble.atomtrajs:
                for seg in atom.segments:
                    if seg.end_time <= t_above_lo - 1e-15:
                        continue
                    if seg.start_time >= t_above_hi + 1e-15:
                        continue
                    _draw_trap_for_seg(atom, seg, t_above_lo, t_above_hi)

    # Loss markers, drawn after all trap/trajectory passes so the cross
    # sits on top of the (already truncated) trajectory line.
    for atom in ensemble.atomtrajs:
        if atom.lost_at is not None and t_min <= atom.lost_at <= t_max:
            _draw_loss_cross(ax, atom, grid, t_to_z, s)

    # --- replacement connectors: a straight dashed line from the lost
    #     atom's *last grid site* to the new atom's *loaded grid site*,
    #     with z matching the time the atom was (or arrived) at each
    #     site. Endpoints are integer lattice sites by construction.
    if replacements:
        for lost_id, new_id in replacements:
            try:
                lost = ensemble.atomtraj_by_id(int(lost_id))
                new = ensemble.atomtraj_by_id(int(new_id))
            except KeyError:
                continue
            last = _last_grid_site_and_time(lost)
            loaded = _loading_destination_and_time(new, grid.N)
            if last is None or loaded is None:
                continue
            site_a, t_a = last
            site_b, t_b = loaded
            if not (t_min - 1e-12 <= t_a <= t_max + 1e-12):
                continue
            if not (t_min - 1e-12 <= t_b <= t_max + 1e-12):
                continue
            xy_a = grid.ij_to_xy(np.asarray(site_a, dtype=float))
            xy_b = grid.ij_to_xy(np.asarray(site_b, dtype=float))
            z_a = float(t_to_z(np.array([t_a], dtype=float))[0])
            z_b = float(t_to_z(np.array([t_b], dtype=float))[0])
            color = (
                s.replacement_connector_color
                if s.replacement_connector_color is not None
                else _darken(colors[ensemble.index_of(int(lost_id))],
                             s.trajectory_darken)
            )
            line, = ax.plot(
                [float(xy_a[0]), float(xy_b[0])],
                [float(xy_a[1]), float(xy_b[1])],
                [z_a, z_b],
                color=color, lw=s.replacement_connector_lw,
                alpha=s.replacement_connector_alpha,
                solid_capstyle="butt",
            )
            line.set_dashes(list(s.replacement_connector_dashes))
            # also project the connector onto the bottom plate, if drawn
            if show_bottom_trajectory or show_bottom_grid:
                line2, = ax.plot(
                    [float(xy_a[0]), float(xy_b[0])],
                    [float(xy_a[1]), float(xy_b[1])],
                    [0.0, 0.0],
                    color=color, lw=s.replacement_connector_lw,
                    alpha=s.replacement_connector_alpha,
                    solid_capstyle="butt",
                )
                line2.set_dashes(list(s.replacement_connector_dashes))

    # --- atom 2D cross-section discs per slice --------------------------
    fresh_rgba_full = to_rgba(s.fresh_atom_color)
    for i, (t, z) in enumerate(zip(t_arr, layer_zs)):
        layer_alpha = layer_alphas[i]
        ij_now = ensemble.positions_at(t)
        atom_xy = grid.ij_to_xy(ij_now)

        edge_r, edge_g, edge_b, edge_a = to_rgba(s.atom_edge_color)

        visible_idx = []
        rgba_visible = []
        for k, atom in enumerate(ensemble.atomtrajs):
            if atom.is_lost_at(t):
                continue
            if _is_fresh_at(atom, t):
                base = fresh_rgba_full
            else:
                base = to_rgba(colors[k])
            rgba_visible.append((base[0], base[1], base[2], base[3] * layer_alpha))
            visible_idx.append(k)

        if visible_idx:
            visible_idx = np.asarray(visible_idx)
            ax.scatter(
                atom_xy[visible_idx, 0],
                atom_xy[visible_idx, 1],
                np.full(len(visible_idx), z),
                s=s.atom_size,
                c=np.asarray(rgba_visible),
                edgecolors=(edge_r, edge_g, edge_b, edge_a * layer_alpha),
                linewidths=s.atom_edge_lw, depthshade=False,
            )
        if show_atom_ids:
            label_offset = 0.32 * d
            for atom, (x, y), color in zip(ensemble.atomtrajs, atom_xy, colors):
                if atom.is_lost_at(t):
                    continue
                r, g, b, _ = to_rgba(color)
                ax.text(
                    x, y + label_offset, z,
                    str(int(atom.atom_id)),
                    ha="center", va="bottom", fontsize=s.atom_id_fontsize,
                    color=(0.55 * r, 0.55 * g, 0.55 * b, layer_alpha),
                )

        if show_motion_arrows:
            for atom, (x, y), _atom_color in zip(
                ensemble.atomtrajs, atom_xy, colors
            ):
                if atom.is_lost_at(t):
                    continue
                seg = _intended_segment(atom, t)
                if seg is None:
                    continue
                # arrow goes from the atom's current xy to the segment's
                # end_pos; length scales by `motion_arrow_length_frac`
                # (default 1.0 = exactly the remaining xy distance).
                end_xy = grid.ij_to_xy(np.asarray(seg.end_pos, dtype=float))
                ux_full = float(end_xy[0]) - float(x)
                uy_full = float(end_xy[1]) - float(y)
                norm = float(np.hypot(ux_full, uy_full))
                if norm <= 0.0:
                    continue
                ux = ux_full / norm
                uy = uy_full / norm
                arrow_len = norm * s.motion_arrow_length_frac
                arrow_color = s.motion_arrow_channel_colors.get(
                    seg.channel, s.motion_arrow_default_color,
                )
                _draw_planned_dash(
                    ax, x, y, z, ux, uy, arrow_len,
                    s.motion_arrow_dashes,
                    arrow_color,
                    s.motion_arrow_lw,
                    s.motion_arrow_alpha * layer_alpha,
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
    ax.set_xlim(xy_lo, xy_hi)
    ax.set_ylim(xy_lo, xy_hi)
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
