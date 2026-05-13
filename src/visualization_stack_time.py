"""Time-stacked 3D visualization for atom rearrangement motion.

Stacks time slices of an `AtomEnsemble` / `MovingSequence` along z, with
t=0 at the bottom and t=t_max at the top. Atoms are 2D cross-section
discs on every layer; trap motion can render as 3D tubes; a continuous
trajectory polyline threads through the stack between layers; per-layer
alpha gives a "depth of time" cue.

All visual styling — view angle, colors, sizes, alphas, layer planes,
arrows, traps — lives on the `StackTimeStyle` dataclass. The optional
projection plate (a "ground" plane above the latest-time layer that
shows trajectory / trap / grid summaries for the whole run) is styled
separately via `ProjectPlaneStyle` and passed to the drawing functions
as `project_plane=...`. Pass custom styles to change the look without
touching the module.

Top-level entry points:

    `draw_stack_time(ax, motion, t_values, *, style=..., project_plane=..., ...)`
    `draw_stack_time_frame(motion, t_values, *, style=..., project_plane=..., ...)`
    `save_stack_time(motion, t_values, output_path, *, style=..., project_plane=..., ...)`
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import is_color_like, to_rgb, to_rgba
from matplotlib.patches import Circle as MplCircle, Polygon as MplPolygon
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers '3d' projection)
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.atom_trajectory import AtomEnsemble  # noqa: E402
from src.visualization import _atom_colors_rgba  # noqa: E402

ProjectionType = Literal["persp", "ortho"]


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
    projection_type: ProjectionType = "persp"
    focal_length: float = 0.45

    # Layer stack geometry / depth fade. `layer_alpha_floor` is the
    # alpha of the top (earliest) layer; the bottom layer is at 1.0.
    # Keep this fairly high so each layer reads as a distinct plate
    # without the top one washing out — the bottom-plate trajectory and
    # trap tubes provide the "depth" cue, not faded plates.
    layer_spacing: float = 6.0
    layer_alpha_floor: float = 0.85

    # Atom discs
    atom_edge_color: Any = "#ffffff"
    atom_edge_lw: float = 1.4
    atom_radius_frac: float = 0.30
    atom_disc_segments: int = 48
    atom_plane_z_offset: float = 0.02

    # Lattice lines: one continuous line per row of sites (red, varying i
    # at fixed j) and per column of sites (blue, varying j at fixed i).
    # Used for the bottom projected grid.
    lattice_line_colors: Mapping[str, Any] = field(
        default_factory=lambda: {"row": "#ff2828", "col": "#29b0ff"}
    )
    lattice_line_extension_frac: float = 0.7

    # Translucent rectangular plane drawn under each layer.
    layer_plane_color: Any = "#c8d0dc"
    layer_plane_alpha: float = 0.22
    layer_plane_extension_frac: float = 0.55

    # Per-layer planned-motion arrows: a small solid in-plane arrow near
    # each moving atom, color-keyed by the segment's `channel`.
    motion_arrow_lw: float = 1.8
    motion_arrow_alpha: float = 1.0
    motion_arrow_length_frac: float = 1.0   # arrow length / remaining xy length
    motion_arrow_max_length_frac: float = 0.65  # arrow length / grid spacing
    motion_arrow_dashes: tuple[float, float] = (4.0, 3.0)
    motion_arrow_head_length_frac: float = 0.20
    motion_arrow_head_width_frac: float = 0.26
    motion_arrow_start_offset_frac: float = 0.18
    motion_arrow_z_offset: float = 0.08
    motion_arrow_channel_colors: Mapping[Any, Any] = field(
        default_factory=lambda: {
            "row": "#ff2828",
            "col": "#29b0ff",
            "aod": "#444444",
            None: "#444444",
        }
    )
    motion_arrow_default_color: Any = "#444444"
    motion_target_circle_color: Any = "#7d8795"
    motion_target_circle_alpha: float = 0.75
    motion_target_circle_lw: float = 1.0
    motion_target_circle_segments: int = 80
    motion_target_circle_dashes: tuple[float, float] = (3.0, 2.5)
    motion_target_circle_z_offset: float = 0.04

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
    loss_target_cross_color: Any = "#4b5563"
    loss_target_cross_lw: float = 2.2
    loss_target_cross_alpha: float = 0.90
    loss_target_cross_size_frac: float = 0.4
    loss_target_cross_z_offset: float = 0.06

    # Trap tubes. `trap_channel_colors` values may be one color or a
    # two-color pair; pairs render as angular half-cylinders with a white
    # transition band at `trap_two_tone_theta + pi`.
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
    # Start angle for two-tone trap tubes. Color 0 covers
    # [theta, theta + pi], color 1 covers [theta + pi, theta + 2pi].
    trap_two_tone_theta: float = 0
    # Half-width (in radians) of the white smoothstep transition band
    # between the two tones, centered on the boundary. 0 means an abrupt
    # boundary. Per-channel overrides via `trap_two_tone_transition_widths`.
    trap_two_tone_transition_width: float = np.pi / 9.0
    trap_two_tone_transition_widths: Mapping[Any, float] = field(
        default_factory=dict
    )
    # Smoothstep fade ramp at each end of a normal trap tube, expressed
    # as a fraction of the segment duration. The segment is sampled in
    # time but rendered along z, so this is effectively a *depth* fade
    # near the segment's start_time (high z) and end_time (low z). Keep
    # the value small so the trap stays visually aligned with the atom
    # at both ends; large values would lag the atom by the ramp width.
    trap_ramp_frac: float = 0.10
    trap_ramp_alpha_floor: float = 0.0
    trap_samples_per_segment: int = 96
    tube_facets: int = 256

    # Idle trap tube: drawn for time intervals where an atom has no
    # active segment (no row/col move, no aod, no stationary gate).
    # The atom is stationary during idle, so the tube is a vertical
    # gray cylinder threading through the slab between layers.
    idle_trap_color: Any = "#9a9a9a"
    idle_trap_alpha: float = 0.4
    idle_trap_radius_frac: float | None = None  # None inherits trap_radius_frac
    idle_trap_samples_per_segment: int = 16

    # Trajectory polylines through the stack
    trajectory_lw: float = 1.4
    trajectory_alpha: float = 0.7
    trajectory_darken: float = 0.65
    trajectory_samples_per_segment: int = 24
    trajectory_z_offset: float = 0.06

    # 2D ground-plane gate blobs (single-qubit rotations rendered as a
    # Gaussian glow at the atom's site). Single-color channels render
    # solid; two-color channels split angularly using `trap_two_tone_theta`.
    gate_blob_radius_frac: float = 0.55
    gate_blob_sigma_frac: float = 0.42
    gate_blob_alpha_max: float = 0.85
    gate_blob_grid_size: int = 96

    # Vertical guide lines at trap start/end/handoff positions
    trap_event_guide_color: Any = "#7d8795"
    trap_event_guide_lw: float = 1.0
    trap_event_guide_alpha: float = 0.32
    trap_event_guide_dashes: tuple[float, float] = (3.0, 3.0)

    # Per-layer connection polylines joining caller-supplied vertex
    # lists at each layer's z (see `layer_connections=` on
    # `draw_stack_time`). Drawn under the atom discs.
    # `layer_connection_dashes=None` -> solid line.
    layer_connection_color: Any = "#9a9a9a"
    layer_connection_lw: float = 1.0
    layer_connection_alpha: float = 0.85
    layer_connection_dashes: tuple[float, float] | None = None
    layer_connection_z_offset: float = -0.02

    # Title
    title_pad: float = 18
    title_fontsize: float = 11


DEFAULT_STYLE = StackTimeStyle()


@dataclass
class ProjectPlaneStyle:
    """Visual styling for the projection plate.

    The plate is a translucent rectangular surface drawn above the
    latest-time layer of a stack-time render. Trajectories, traps, and
    the row/col grid can be projected onto it as a summary view of the
    whole run. The four `show_*` flags pick which projections render;
    the rest are appearance knobs (color, alpha, line width, etc.).
    """

    # Render toggles
    show_plane: bool = False
    show_grid: bool = True
    show_traps: bool = False
    show_trajectory: bool = True

    # Translucent plate surface. `plane_color=None` inherits
    # `StackTimeStyle.layer_plane_color`.
    plane_color: Any | None = None
    plane_alpha: float = 0.20

    # Row/col grid lines on the plate.
    grid_color: Any = "#9a9a9a"
    grid_lw: float = 0.6
    grid_alpha: float = 0.55

    # Atom xy trajectory polylines projected onto the plate.
    traj_lw: float = 2.0
    traj_alpha: float = 0.85
    traj_samples_per_segment: int = 24

    # Trap motion bars projected onto the plate.
    trap_lw: float = 5.0
    trap_alpha: float = 0.18
    trap_samples_per_segment: int = 36
    trap_width_frac: float = 0.14
    trap_arrow_length_frac: float = 0.38
    trap_arrow_lw: float = 1.1
    trap_arrow_alpha: float = 0.88
    trap_arrow_head_length_frac: float = 0.13
    trap_arrow_head_width_frac: float = 0.16
    trap_arrow_z_offset: float = 0.02


DEFAULT_PROJECT_PLANE = ProjectPlaneStyle()


def _smoothstep(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _darken(color: Any, k: float) -> tuple[float, float, float, float]:
    r, g, b, a = to_rgba(color)
    return (k * r, k * g, k * b, a)


def _colorway(color_spec: Any) -> list[Any]:
    """Return one or more colors from a channel color specification.

    Most channels use a single Matplotlib color. A channel may also provide
    a two-color sequence, which the trap tube renders as angular half-cylinders
    with a white transition band; this is useful for multi-tone optical gates.
    """
    if is_color_like(color_spec):
        return [color_spec]
    if isinstance(color_spec, Sequence) and not isinstance(color_spec, (str, bytes)):
        colors = list(color_spec)
        if colors and all(is_color_like(c) for c in colors):
            return colors
    return [color_spec]


def _blend_colorway(color_spec: Any) -> tuple[float, float, float, float]:
    rgba = np.asarray([to_rgba(c) for c in _colorway(color_spec)], dtype=float)
    rgb = np.mean(rgba[:, :3], axis=0)
    alpha = np.max(rgba[:, 3])
    return (float(rgb[0]), float(rgb[1]), float(rgb[2]), float(alpha))


def _angular_two_tone_rgbs(
    rgbs: np.ndarray,
    theta: np.ndarray,
    theta0: float = 0.0,
    half_width: float = np.pi / 9.0,
) -> np.ndarray:
    """Color [theta0, theta0+pi] with color 0 and the opposite half with color 1.

    `half_width` is the half-angular extent (radians) of the transition
    band centered at theta0+pi, ramping color0 -> white -> color1 with a
    smoothstep profile. Pass 0 for an abrupt color1<->color0 boundary
    with no white blend.
    """
    stops = np.asarray(rgbs, dtype=float).reshape(-1, 3)
    phase = np.mod(np.asarray(theta, dtype=float) - float(theta0), 2.0 * np.pi)
    if len(stops) != 2:
        idx = np.minimum(
            (phase / (2.0 * np.pi) * len(stops)).astype(int),
            len(stops) - 1,
        )
        return stops[idx]

    rgb = np.empty(phase.shape + (3,), dtype=float)
    if half_width <= 0.0:
        rgb[phase < np.pi] = stops[0]
        rgb[phase >= np.pi] = stops[1]
        return rgb

    white = np.ones(3, dtype=float)
    orange_side = phase <= np.pi - half_width
    yellow_side = phase >= np.pi + half_width
    lower_ramp = (phase > np.pi - half_width) & (phase <= np.pi)
    upper_ramp = (phase > np.pi) & (phase < np.pi + half_width)

    rgb[orange_side] = stops[0]
    rgb[yellow_side] = stops[1]
    if np.any(lower_ramp):
        u = _smoothstep((phase[lower_ramp] - (np.pi - half_width)) / half_width)
        rgb[lower_ramp] = (1.0 - u)[:, None] * stops[0] + u[:, None] * white
    if np.any(upper_ramp):
        u = _smoothstep((phase[upper_ramp] - np.pi) / half_width)
        rgb[upper_ramp] = (1.0 - u)[:, None] * white + u[:, None] * stops[1]
    return rgb


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


def _draw_solid_planar_arrow(
    ax: Any,
    x0: float, y0: float, z: float,
    ux: float, uy: float,
    length: float,
    color: Any,
    lw: float,
    alpha: float,
    head_length: float,
    head_width: float,
    start_offset: float,
    z_offset: float = 0.0,
    sort_zpos: float | None = None,
) -> None:
    """Draw a compact filled arrow parallel to the xy layer plane."""
    if length <= 0.0:
        return
    z_draw = float(z) + float(z_offset)
    r, g, b, a = to_rgba(color)
    rgba = (r, g, b, a * alpha)
    start_x = float(x0) + ux * start_offset
    start_y = float(y0) + uy * start_offset
    tip_x = start_x + ux * length
    tip_y = start_y + uy * length
    head_length = min(max(0.0, head_length), 0.72 * length)
    head_width = max(0.0, head_width)
    base_x = tip_x - ux * head_length
    base_y = tip_y - uy * head_length
    shaft_len = length - head_length
    if shaft_len > 0.0:
        line, = ax.plot(
            [start_x, base_x], [start_y, base_y], [z_draw, z_draw],
            color=rgba, lw=lw, solid_capstyle="round",
        )
        try:
            line.set_sort_zpos(
                z_draw if sort_zpos is None else float(sort_zpos)
            )
        except AttributeError:
            pass
    perp_x = -uy
    perp_y = ux
    tri = [
        (tip_x, tip_y, z_draw),
        (base_x + 0.5 * head_width * perp_x,
         base_y + 0.5 * head_width * perp_y,
         z_draw),
        (base_x - 0.5 * head_width * perp_x,
         base_y - 0.5 * head_width * perp_y,
         z_draw),
    ]
    coll = Poly3DCollection(
        [tri], facecolors=[rgba], edgecolors="none", linewidths=0,
    )
    ax.add_collection3d(coll)
    try:
        coll.set_sort_zpos(z_draw if sort_zpos is None else float(sort_zpos))
    except AttributeError:
        pass


def _draw_planar_circle_outline(
    ax: Any,
    center_xy: Sequence[float],
    z: float,
    radius: float,
    color: Any,
    lw: float,
    alpha: float,
    segments: int,
    dashes: Sequence[float],
    z_offset: float = 0.0,
    sort_zpos: float | None = None,
) -> None:
    """Draw a dashed circular outline parallel to the xy layer plane."""
    if radius <= 0.0:
        return
    n_segments = max(16, int(segments))
    theta = np.linspace(0.0, 2.0 * np.pi, n_segments + 1)
    cx, cy = np.asarray(center_xy, dtype=float)
    z_draw = float(z) + float(z_offset)
    r, g, b, a = to_rgba(color)
    line, = ax.plot(
        cx + radius * np.cos(theta),
        cy + radius * np.sin(theta),
        np.full_like(theta, z_draw, dtype=float),
        color=(r, g, b, a * alpha),
        lw=lw,
        solid_capstyle="butt",
    )
    line.set_dashes(list(dashes))
    try:
        line.set_sort_zpos(z_draw if sort_zpos is None else float(sort_zpos))
    except AttributeError:
        pass


def _draw_planar_cross(
    ax: Any,
    center_xy: Sequence[float],
    z: float,
    size: float,
    color: Any,
    lw: float,
    alpha: float,
    z_offset: float = 0.0,
    sort_zpos: float | None = None,
) -> None:
    """Draw an in-plane X centered on an xy point."""
    if size <= 0.0:
        return
    cx, cy = np.asarray(center_xy, dtype=float)
    z_draw = float(z) + float(z_offset)
    r, g, b, a = to_rgba(color)
    rgba = (r, g, b, a * alpha)
    half = 0.5 * float(size)
    for sx, sy in ((1.0, 1.0), (1.0, -1.0)):
        line, = ax.plot(
            [cx - sx * half, cx + sx * half],
            [cy - sy * half, cy + sy * half],
            [z_draw, z_draw],
            color=rgba,
            lw=lw,
            solid_capstyle="round",
        )
        try:
            line.set_sort_zpos(
                z_draw if sort_zpos is None else float(sort_zpos)
            )
        except AttributeError:
            pass


def _draw_planar_discs(
    ax: Any,
    centers_xy: np.ndarray,
    z: float,
    radius: float,
    facecolors: Sequence[Any],
    edgecolors: Any,
    edge_lw: float,
    segments: int,
    z_offset: float = 0.0,
    sort_zpos: float | None = None,
) -> None:
    """Draw filled circular polygons parallel to the xy layer plane."""
    if len(centers_xy) == 0:
        return
    n_segments = max(12, int(segments))
    angles = np.linspace(0.0, 2.0 * np.pi, n_segments, endpoint=False)
    cos_a = np.cos(angles)
    sin_a = np.sin(angles)
    z_draw = float(z) + float(z_offset)
    polys = [
        [
            (
                float(cx) + radius * float(cos_a[k]),
                float(cy) + radius * float(sin_a[k]),
                z_draw,
            )
            for k in range(n_segments)
        ]
        for cx, cy in np.asarray(centers_xy, dtype=float)
    ]
    coll = Poly3DCollection(
        polys,
        facecolors=list(facecolors),
        edgecolors=edgecolors,
        linewidths=edge_lw,
    )
    ax.add_collection3d(coll)
    try:
        coll.set_sort_zpos(z_draw if sort_zpos is None else float(sort_zpos))
    except AttributeError:
        pass


def _draw_planar_segment_strip(
    ax: Any,
    xy0: Sequence[float],
    xy1: Sequence[float],
    z: float,
    width: float,
    rgba: Any,
    *,
    sort_zpos: float | None = None,
) -> bool:
    """Draw one square-ended xy-plane strip between two points."""
    p0 = np.asarray(xy0, dtype=float)
    p1 = np.asarray(xy1, dtype=float)
    delta = p1 - p0
    length = float(np.hypot(delta[0], delta[1]))
    if length <= 1e-12:
        return False

    perp = np.array([-delta[1], delta[0]], dtype=float) / length
    half_width = 0.5 * float(width)
    offset = half_width * perp
    z_draw = float(z)
    poly = [
        (float(p0[0] + offset[0]), float(p0[1] + offset[1]), z_draw),
        (float(p1[0] + offset[0]), float(p1[1] + offset[1]), z_draw),
        (float(p1[0] - offset[0]), float(p1[1] - offset[1]), z_draw),
        (float(p0[0] - offset[0]), float(p0[1] - offset[1]), z_draw),
    ]
    coll = Poly3DCollection(
        [poly],
        facecolors=[rgba],
        edgecolors="none",
        linewidths=0,
    )
    ax.add_collection3d(coll)
    try:
        coll.set_sort_zpos(z_draw if sort_zpos is None else float(sort_zpos))
    except AttributeError:
        pass
    return True


def draw_stack_time(
    ax: Any,
    motion: Any,
    t_values: Sequence[float],
    *,
    atom_colors: Mapping[int, Any] | Iterable[Any] | None = None,
    show_traps: bool = True,
    show_idle_traps: bool = False,
    show_layer_plane: bool = False,
    show_motion_arrows: bool = False,
    show_motion_targets: bool = False,
    show_trajectory: bool = True,
    show_trap_event_guides: bool = False,
    show_layers: bool = True,
    show_project_plane: bool = True,
    style: StackTimeStyle | None = None,
    project_plane: ProjectPlaneStyle | None = None,
    layer_connections: Mapping[int, Sequence[Sequence[tuple[int, int]]]] | None = None,
    title: str | None = None,
) -> None:
    """Draw a time-stacked 3D figure of `motion` on a 3D `ax`.

    `style` selects stack visual parameters; `project_plane` selects the
    projection-plate styling (a "ground" plate drawn above the latest-
    time layer that summarizes trajectories, traps, and the row/col
    grid for the whole run). The remaining `show_*` flags cover major
    layers of the stack itself. Set `show_layers=False` to suppress the
    time-slice layers (atoms, layer plane, motion arrows / targets,
    trajectory polylines) so only the projection plate renders;
    set `show_project_plane=False` to suppress the projection plate
    entirely (overrides the per-element flags on `project_plane`).

    `layer_connections` is an optional `{layer_idx: [polyline, ...]}`
    map. Each polyline is a sequence of (i, j) grid sites; consecutive
    sites are joined by a dashed gray segment drawn on that layer's
    plane (styled via `StackTimeStyle.layer_connection_*`). Use it to
    show static structure on a layer (patch outlines, ancilla chains,
    etc.) — the caller is responsible for picking vertex lists that
    line up with where the atoms actually sit on that layer.
    """
    s = style if style is not None else DEFAULT_STYLE
    pp = project_plane if project_plane is not None else DEFAULT_PROJECT_PLANE
    ensemble = motion if isinstance(motion, AtomEnsemble) else motion.ensemble

    # Matplotlib's automatic 3D z-sort uses min(projected_zs) for
    # Line3DCollection, which can put trajectory lines in front of the
    # layer atoms when their projected camera-z falls below the atom's
    # set_sort_zpos. Disable computed_zorder so the painter's algorithm
    # uses the draw order we control: lower layers and their slab content
    # are emitted before higher layers, so atoms and planes always end up
    # in front of the slab below them.
    ax.computed_zorder = False

    t_arr = sorted(float(t) for t in t_values)
    n_layers = len(t_arr)
    if n_layers == 0:
        raise ValueError("t_values must be non-empty")

    colors = _atom_colors_rgba(
        [int(a.atom_id) for a in ensemble.atomtrajs], atom_colors
    )

    grid = ensemble.grid
    d = grid.d
    atom_radius = np.sqrt(0.7) * s.atom_radius_frac * d
    site_lo = (0.0 - grid.center) * d
    site_hi = ((grid.N - 1.0) - grid.center) * d
    bound_lo = site_lo - d / 2.0
    bound_hi = site_hi + d / 2.0
    trap_radius = s.trap_radius_frac * d
    line_lo = site_lo - s.lattice_line_extension_frac * d
    line_hi = site_hi + s.lattice_line_extension_frac * d
    plane_lo = site_lo - s.layer_plane_extension_frac * d
    plane_hi = site_hi + s.layer_plane_extension_frac * d
    proj_plane_lo = min(plane_lo, line_lo)
    proj_plane_hi = max(plane_hi, line_hi)

    # Auto-expand xy bounds to include any segment endpoints / initial
    # positions that fall outside the lattice.
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

    # Bottom of the stack is the earliest time (lowest z), top is the
    # latest time (highest z). Layers ascend with increasing t.
    project_any = show_project_plane and (
        pp.show_plane or pp.show_grid or pp.show_traps or pp.show_trajectory
    )
    layer_zs = [i * s.layer_spacing for i in range(n_layers)]
    z_bot = float(layer_zs[0])
    z_top = float(layer_zs[-1])
    z_max = max(z_top, 1.0)
    t_min, t_max = t_arr[0], t_arr[-1]

    def _is_freshly_loaded(atom: Any, t: float, eps: float = 1e-9) -> bool:
        """Fresh-loaded recolor rule, anchored to the first stack time.

        An atom whose first non-zero segment starts strictly after the
        first stack time `t_min` counts as a reservoir delivery: it stays
        in the fresh color until that first segment ends, so the layer
        showing the moment of first addressing still renders gray.
        Atoms whose first segment starts at `t_min` are part of the
        initial configuration and are never fresh. Defers to the
        module-level `_is_fresh_at` so monkey-patches still take effect.
        """
        if _is_fresh_at(atom, t, eps):
            return True
        first_seg = next(
            (seg for seg in atom.segments if seg.duration > 0), None,
        )
        if first_seg is None:
            return False
        return (
            first_seg.start_time > t_min + eps
            and t < first_seg.end_time - eps
        )

    def t_to_z(ts: np.ndarray) -> np.ndarray:
        if t_max <= t_min:
            return np.full_like(np.asarray(ts, dtype=float), z_bot)
        return z_bot + (z_top - z_bot) * (
            (np.asarray(ts, dtype=float) - t_min) / (t_max - t_min)
        )

    # Latest time (top layer) is fully opaque; earlier times fade toward
    # `layer_alpha_floor`. layer_zs is ascending in t, so the same formula
    # over i works (i=n-1 latest → alpha=1).
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
        frac = (np.asarray(z, dtype=float) - z_bot) / (z_top - z_bot)
        return s.layer_alpha_floor + (1.0 - s.layer_alpha_floor) * frac

    def channel_at_time(atom: Any, t: float, eps: float = 1e-9) -> Any:
        for seg in atom.segments:
            if seg.duration <= 0:
                continue
            if seg.start_time - eps <= t <= seg.end_time + eps:
                return seg.channel
        return None

    def trajectory_channel_rgba(t_mid: float, atom: Any) -> tuple[float, ...]:
        channel = channel_at_time(atom, t_mid)
        color_spec = s.trap_channel_colors.get(
            channel, s.trap_channel_colors.get(None, "#9a9a9a"),
        )
        return _darken(_blend_colorway(color_spec), s.trajectory_darken)

    # The projection plate sits one `layer_spacing` above the latest-time
    # layer when the time stack is rendered. In single-frame mode
    # (`show_layers=False`) the plate is the only visual, so anchor it at
    # z=0 to match the framing of a pure ground-plane render.
    proj_plane_base = (z_top + s.layer_spacing) if show_layers else 0.0
    proj_atom_z_offset = max(float(s.atom_plane_z_offset), 1e-6)
    proj_grid_z = proj_plane_base + 0.20 * proj_atom_z_offset
    proj_trap_z = proj_plane_base + 0.45 * proj_atom_z_offset
    proj_traj_z = proj_plane_base + 0.70 * proj_atom_z_offset
    proj_marker_z = proj_plane_base + 1.40 * proj_atom_z_offset
    proj_trap_width = max(
        0.006 * d,
        float(pp.trap_width_frac) * d,
        0.018 * float(pp.trap_lw) * d,
    )
    proj_traj_width = max(0.006 * d, 0.018 * float(pp.traj_lw) * d)
    proj_marker_draws: list[tuple[np.ndarray, float, list[Any], list[Any], float]] = []
    loss_target_cross_size = (
        2.0 * atom_radius * max(0.0, float(s.loss_target_cross_size_frac))
    )
    loss_targets_by_layer: dict[int, list[np.ndarray]] = {}
    proj_loss_points: list[np.ndarray] = []

    def _lost_positions(atom: Any) -> tuple[np.ndarray, np.ndarray] | None:
        if atom.lost_at is None:
            return None
        lost_at = float(atom.lost_at)
        loss_pos = np.asarray(atom.position_at(lost_at), dtype=float)
        for seg in atom.segments:
            if (
                seg.duration > 0
                and seg.start_time - 1e-12 <= lost_at <= seg.end_time + 1e-12
            ):
                return loss_pos, np.asarray(seg.end_pos, dtype=float)
        return loss_pos, loss_pos

    for atom in ensemble.atomtrajs:
        lost_positions = _lost_positions(atom)
        if lost_positions is None:
            continue
        loss_ij, target_ij = lost_positions
        target_xy = grid.ij_to_xy(target_ij)
        proj_loss_points.append(grid.ij_to_xy(loss_ij))
        for layer_idx, layer_t in enumerate(t_arr):
            if layer_t > float(atom.lost_at) + 1e-12:
                loss_targets_by_layer.setdefault(layer_idx, []).append(target_xy)
                break

    # --- projection plate: row/col-colored extending lattice lines + xy
    #     trajectory projections at `proj_plane_base`. The "grid" uses
    #     the same row/col color rule as the layer lattice lines, but
    #     with the plate styling (alpha, lw).
    if show_project_plane and pp.show_plane:
        xx = np.array(
            [[proj_plane_lo, proj_plane_hi],
             [proj_plane_lo, proj_plane_hi]]
        )
        yy = np.array(
            [[proj_plane_lo, proj_plane_lo],
             [proj_plane_hi, proj_plane_hi]]
        )
        zz = np.full_like(xx, proj_plane_base, dtype=float)
        surf = ax.plot_surface(
            xx, yy, zz,
            color=(
                s.layer_plane_color
                if pp.plane_color is None
                else pp.plane_color
            ),
            alpha=pp.plane_alpha,
            shade=False,
            linewidth=0,
            edgecolor="none",
            antialiased=False,
        )
        try:
            surf.set_sort_zpos(proj_plane_base - 0.02 * s.layer_spacing)
        except AttributeError:
            pass

    if show_project_plane and pp.show_grid:
        bot_row_color = to_rgba(s.lattice_line_colors.get("row", "#ff2828"))
        bot_col_color = to_rgba(s.lattice_line_colors.get("col", "#29b0ff"))
        for jj in range(grid.N):
            y_row = (jj - grid.center) * d
            ax.plot(
                [line_lo, line_hi], [y_row, y_row], [proj_grid_z, proj_grid_z],
                color=bot_row_color, lw=pp.grid_lw,
                alpha=pp.grid_alpha,
                solid_capstyle="round",
            )
        for ii in range(grid.N):
            x_col = (ii - grid.center) * d
            ax.plot(
                [x_col, x_col], [line_lo, line_hi], [proj_grid_z, proj_grid_z],
                color=bot_col_color, lw=pp.grid_lw,
                alpha=pp.grid_alpha,
                solid_capstyle="round",
            )

    if show_project_plane and pp.show_traps and show_traps:
        single_time = t_max <= t_min
        for atom in ensemble.atomtrajs:
            for seg in atom.segments:
                if seg.duration <= 0:
                    continue
                if seg.end_time < t_min - 1e-15 or seg.start_time > t_max + 1e-15:
                    continue
                if single_time:
                    # Single-time snapshot: draw the full start->end bar of
                    # any segment that is active at t (mirrors the 2D
                    # ground-plane trap rendering).
                    if atom.lost_at is not None and atom.lost_at <= t_min + 1e-15:
                        continue
                    xy = grid.ij_to_xy(np.asarray([
                        seg.start_pos,
                        seg.end_pos,
                    ], dtype=float))
                else:
                    seg_lo = max(seg.start_time, t_min)
                    seg_hi = min(seg.end_time, t_max)
                    if atom.lost_at is not None:
                        if atom.lost_at <= seg_lo + 1e-15:
                            continue
                        if atom.lost_at < seg_hi - 1e-15:
                            seg_hi = atom.lost_at
                    if seg_hi <= seg_lo:
                        continue
                    xy = grid.ij_to_xy(np.asarray([
                        seg.position_at(float(seg_lo)),
                        seg.position_at(float(seg_hi)),
                    ], dtype=float))
                if np.hypot(*(xy[1] - xy[0])) <= 1e-12:
                    continue
                r, g, b, _ = _blend_colorway(s.trap_channel_colors.get(
                    seg.channel, s.trap_channel_colors.get(None, "#9a9a9a"),
                ))
                _draw_planar_segment_strip(
                    ax,
                    xy[0],
                    xy[1],
                    proj_trap_z,
                    proj_trap_width,
                    (r, g, b, pp.trap_alpha),
                    sort_zpos=proj_trap_z,
                )
                arrow_len = max(0.0, float(pp.trap_arrow_length_frac)) * d
                if arrow_len > 0.0:
                    direction = xy[1] - xy[0]
                    direction_norm = float(np.hypot(direction[0], direction[1]))
                    if direction_norm > 1e-12:
                        ux = float(direction[0]) / direction_norm
                        uy = float(direction[1]) / direction_norm
                        center = 0.5 * (xy[0] + xy[1])
                        arrow_start = center - 0.5 * arrow_len * np.array([ux, uy])
                        arrow_color = _darken((r, g, b, 1.0), 0.62)
                        _draw_solid_planar_arrow(
                            ax,
                            float(arrow_start[0]),
                            float(arrow_start[1]),
                            proj_trap_z,
                            ux,
                            uy,
                            arrow_len,
                            arrow_color,
                            pp.trap_arrow_lw,
                            pp.trap_arrow_alpha,
                            min(
                                max(0.0, pp.trap_arrow_head_length_frac) * d,
                                0.45 * arrow_len,
                            ),
                            max(0.0, pp.trap_arrow_head_width_frac) * d,
                            0.0,
                            pp.trap_arrow_z_offset,
                            proj_trap_z + pp.trap_arrow_z_offset,
                        )

    if show_project_plane and pp.show_trajectory and t_max > t_min:
        fresh_rgba = to_rgba(s.fresh_atom_color)

        for idx, atom in enumerate(ensemble.atomtrajs):
            traj_end = t_max if atom.lost_at is None else min(t_max, atom.lost_at)
            drew_trajectory = False
            for seg in atom.segments:
                if seg.duration <= 0:
                    continue
                if seg.end_time < t_min - 1e-15 or seg.start_time > t_max + 1e-15:
                    continue
                seg_lo = max(seg.start_time, t_min)
                seg_hi = min(seg.end_time, traj_end)
                if seg_hi <= seg_lo:
                    continue
                xy = grid.ij_to_xy(np.asarray([
                    seg.position_at(float(seg_lo)),
                    seg.position_at(float(seg_hi)),
                ], dtype=float))
                if np.hypot(*(xy[1] - xy[0])) <= 1e-12:
                    continue
                t_mid = 0.5 * (float(seg_lo) + float(seg_hi))
                r, g, b, a = trajectory_channel_rgba(t_mid, atom)
                _draw_planar_segment_strip(
                    ax,
                    xy[0],
                    xy[1],
                    proj_traj_z,
                    proj_traj_width,
                    (r, g, b, a * pp.traj_alpha),
                    sort_zpos=proj_traj_z,
                )
                drew_trajectory = True

            if not drew_trajectory:
                continue
            ij = np.asarray(
                [atom.position_at(t_min), atom.position_at(traj_end)],
                dtype=float,
            )
            xy = grid.ij_to_xy(ij)

            normal_rgba = to_rgba(colors[idx])
            start_color = (
                fresh_rgba if _is_freshly_loaded(atom, t_min) else normal_rgba
            )
            proj_marker_draws.append((
                xy[0:1],
                proj_marker_z,
                atom_radius,
                [to_rgba("white")],
                [start_color],
                1.0,
            ))
            if atom.lost_at is None:
                # only mark a finished trajectory with the closed end disc;
                # for a lost atom the projection just stops at the loss
                # position with no extra marker — the in-stack cross is
                # the single source of truth.
                end_color = (
                    fresh_rgba if _is_freshly_loaded(atom, traj_end) else normal_rgba
                )
                proj_marker_draws.append((
                    xy[-1:],
                    proj_marker_z,
                    atom_radius,
                    [end_color],
                    [to_rgba("white")],
                    0.6,
                ))

    for centers_xy, z_marker, radius, facecolors, edgecolors, edge_lw in proj_marker_draws:
        _draw_planar_discs(
            ax,
            centers_xy,
            z_marker,
            radius,
            facecolors,
            edgecolors,
            edge_lw,
            s.atom_disc_segments,
            0.0,
            10.0 * s.layer_spacing,
        )

    if show_project_plane and pp.show_plane:
        for loss_xy in proj_loss_points:
            ax.scatter(
                [float(loss_xy[0])], [float(loss_xy[1])], [proj_marker_z],
                s=s.loss_marker_size, c=s.loss_marker_color, marker="x",
                linewidths=s.loss_marker_lw, depthshade=False,
            )

    if show_trap_event_guides and show_traps:
        event_points: list[tuple[float, float, float]] = []
        seen_events: set[tuple[int, int, int]] = set()

        def _add_trap_event(t_event: float, ij_event: Any) -> None:
            if not (t_min - 1e-12 <= t_event <= t_max + 1e-12):
                return
            xy_event = grid.ij_to_xy(np.asarray(ij_event, dtype=float))
            z_event = float(t_to_z(np.array([t_event], dtype=float))[0])
            key = (
                int(round(float(xy_event[0]) / 1e-9)),
                int(round(float(xy_event[1]) / 1e-9)),
                int(round(float(z_event) / 1e-9)),
            )
            if key in seen_events:
                return
            seen_events.add(key)
            event_points.append((float(xy_event[0]), float(xy_event[1]), z_event))

        for atom in ensemble.atomtrajs:
            for seg in atom.segments:
                if seg.duration <= 0:
                    continue
                if seg.end_time < t_min - 1e-15 or seg.start_time > t_max + 1e-15:
                    continue
                if atom.lost_at is not None and atom.lost_at <= seg.start_time + 1e-15:
                    continue
                _add_trap_event(float(seg.start_time), seg.start_pos)
                if (
                    atom.lost_at is not None
                    and seg.start_time - 1e-15 <= atom.lost_at <= seg.end_time + 1e-15
                ):
                    _add_trap_event(
                        float(atom.lost_at),
                        seg.position_at(float(atom.lost_at)),
                    )
                else:
                    _add_trap_event(float(seg.end_time), seg.end_pos)

        for x_event, y_event, z_event in event_points:
            if z_event <= 0.0:
                continue
            line, = ax.plot(
                [x_event, x_event], [y_event, y_event], [0.0, z_event],
                color=s.trap_event_guide_color,
                lw=s.trap_event_guide_lw,
                alpha=s.trap_event_guide_alpha,
                solid_capstyle="butt",
            )
            line.set_dashes(list(s.trap_event_guide_dashes))

    # --- interleaved stack pass: from bottom z up, draw each layer's
    #     plane + the trajectory polylines + trap tubes that exit upward
    #     from this plane to the next-up plane. The order trajectory →
    #     trap → next plane ensures: (a) the trap covers the trajectory,
    #     (b) traps/trajectories sit above the next-time plane (lower z)
    #     but below the current-time plane (higher z, drawn next).
    fresh_rgba_full = to_rgba(s.fresh_atom_color)

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

    def _draw_layer_connections(z_layer: float, layer_alpha_: float, i: int) -> None:
        if not layer_connections:
            return
        polylines = layer_connections.get(i)
        if not polylines:
            return
        z_draw = float(z_layer) + float(s.layer_connection_z_offset)
        color = s.layer_connection_color
        for polyline in polylines:
            ij = np.asarray(list(polyline), dtype=float)
            if len(ij) < 2:
                continue
            xy = grid.ij_to_xy(ij)
            zs_line = np.full(len(xy), z_draw, dtype=float)
            # Low zorder + sort_zpos pinned below the layer's z keeps the
            # line under the atom discs on this layer.
            line, = ax.plot(
                xy[:, 0], xy[:, 1], zs_line,
                color=color,
                lw=s.layer_connection_lw,
                alpha=s.layer_connection_alpha * layer_alpha_,
                solid_capstyle="round",
                zorder=0.5,
            )
            if s.layer_connection_dashes is not None:
                line.set_dashes(list(s.layer_connection_dashes))
            try:
                line.set_sort_zpos(z_draw)
            except AttributeError:
                pass

    def _draw_layer(i: int) -> None:
        t = t_arr[i]
        z = layer_zs[i]
        layer_alpha = layer_alphas[i]

        _draw_layer_plane(z, layer_alpha)
        _draw_layer_connections(z, layer_alpha, i)

        ij_now = ensemble.positions_at(t)
        atom_xy = grid.ij_to_xy(ij_now)

        if show_motion_targets:
            for atom, (x, y) in zip(ensemble.atomtrajs, atom_xy):
                if atom.is_lost_at(t):
                    continue
                seg = _intended_segment(atom, t)
                if seg is None:
                    continue
                target_xy = grid.ij_to_xy(np.asarray(seg.end_pos, dtype=float))
                if np.hypot(float(target_xy[0]) - float(x),
                            float(target_xy[1]) - float(y)) <= 1e-12:
                    continue
                _draw_planar_circle_outline(
                    ax,
                    target_xy,
                    z,
                    atom_radius,
                    s.motion_target_circle_color,
                    s.motion_target_circle_lw,
                    s.motion_target_circle_alpha * layer_alpha,
                    s.motion_target_circle_segments,
                    s.motion_target_circle_dashes,
                    s.motion_target_circle_z_offset,
                    float(z) + s.motion_target_circle_z_offset,
                )

        if show_motion_targets:
            for target_xy in loss_targets_by_layer.get(i, []):
                _draw_planar_circle_outline(
                    ax,
                    target_xy,
                    z,
                    atom_radius,
                    s.motion_target_circle_color,
                    s.motion_target_circle_lw,
                    s.motion_target_circle_alpha * layer_alpha,
                    s.motion_target_circle_segments,
                    s.motion_target_circle_dashes,
                    s.motion_target_circle_z_offset,
                    float(z) + s.motion_target_circle_z_offset,
                )
                _draw_planar_cross(
                    ax,
                    target_xy,
                    z,
                    loss_target_cross_size,
                    s.loss_target_cross_color,
                    s.loss_target_cross_lw,
                    s.loss_target_cross_alpha * layer_alpha,
                    s.loss_target_cross_z_offset,
                    float(z) + s.loss_target_cross_z_offset,
                )

        edge_r, edge_g, edge_b, edge_a = to_rgba(s.atom_edge_color)

        visible_idx = []
        rgba_visible = []
        for k, atom in enumerate(ensemble.atomtrajs):
            if atom.is_lost_at(t):
                continue
            if _is_freshly_loaded(atom, t):
                base = fresh_rgba_full
            else:
                base = to_rgba(colors[k])
            rgba_visible.append((base[0], base[1], base[2], base[3] * layer_alpha))
            visible_idx.append(k)

        if visible_idx:
            visible_idx = np.asarray(visible_idx)
            edge_rgba = (edge_r, edge_g, edge_b, edge_a * layer_alpha)
            _draw_planar_discs(
                ax,
                atom_xy[visible_idx],
                z,
                atom_radius,
                rgba_visible,
                [edge_rgba] * len(visible_idx),
                s.atom_edge_lw,
                s.atom_disc_segments,
                s.atom_plane_z_offset,
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
                end_xy = grid.ij_to_xy(np.asarray(seg.end_pos, dtype=float))
                ux_full = float(end_xy[0]) - float(x)
                uy_full = float(end_xy[1]) - float(y)
                norm = float(np.hypot(ux_full, uy_full))
                if norm <= 0.0:
                    continue
                ux = ux_full / norm
                uy = uy_full / norm
                arrow_len = min(
                    norm * max(0.0, s.motion_arrow_length_frac),
                    d * max(0.0, s.motion_arrow_max_length_frac),
                )
                if arrow_len <= 0.0:
                    continue
                head_length = min(
                    max(0.0, s.motion_arrow_head_length_frac) * d,
                    0.45 * arrow_len,
                )
                head_width = max(0.0, s.motion_arrow_head_width_frac) * d
                start_offset = max(0.0, s.motion_arrow_start_offset_frac) * d
                arrow_color = s.motion_arrow_channel_colors.get(
                    seg.channel, s.motion_arrow_default_color,
                )
                _draw_solid_planar_arrow(
                    ax, x, y, z, ux, uy, arrow_len,
                    arrow_color,
                    s.motion_arrow_lw,
                    s.motion_arrow_alpha * layer_alpha,
                    head_length,
                    head_width,
                    start_offset,
                    s.motion_arrow_z_offset,
                )

    def _draw_trap_for_seg(atom: Any, seg: Any, t_lo: float, t_hi: float) -> None:
        if not show_traps or seg.duration <= 0:
            return
        seg_lo = max(seg.start_time, t_lo)
        seg_hi = min(seg.end_time, t_hi)
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
        ramp_out_frac = s.trap_ramp_frac
        if ramp_out_frac > 0.0:
            ramp_out = max(seg.duration * ramp_out_frac, 1e-15)
            env_out = _smoothstep((effective_end - ts) / ramp_out)
        else:
            env_out = np.ones_like(ts)
        envelope = np.minimum(env_in, env_out)
        ramp_floor = float(np.clip(s.trap_ramp_alpha_floor, 0.0, 1.0))
        if ramp_floor > 0.0 and (s.trap_ramp_frac > 0.0 or ramp_out_frac > 0.0):
            envelope = ramp_floor + (1.0 - ramp_floor) * envelope
        z_alpha = np.asarray(alpha_at_z(zs), dtype=float)
        wall_alpha = envelope * s.trap_max_alpha * z_alpha
        channel_color = s.trap_channel_colors.get(
            seg.channel, s.trap_channel_colors.get(None, "#9a9a9a"),
        )
        _draw_trap_tube(
            ax, xy, zs, wall_alpha, trap_radius,
            [to_rgb(c) for c in _colorway(channel_color)], s.tube_facets,
            sort_zpos=float(np.mean(zs)),
            theta0=s.trap_two_tone_theta,
            two_tone_half_width=s.trap_two_tone_transition_widths.get(
                seg.channel, s.trap_two_tone_transition_width,
            ),
        )

    idle_trap_radius = (
        trap_radius
        if s.idle_trap_radius_frac is None
        else float(s.idle_trap_radius_frac) * d
    )
    idle_trap_rgb = [to_rgb(s.idle_trap_color)]

    def _draw_idle_trap_in_range(atom: Any, t_lo: float, t_hi: float) -> None:
        """Vertical gray trap tube for `atom` over the idle gaps inside
        [t_lo, t_hi].

        An "idle" interval is any sub-range of [t_lo, t_hi] not covered by
        a positive-duration segment of `atom` (and not after lost_at).
        The atom is stationary while idle, so the tube is a cylinder.
        """
        if not (show_idle_traps and show_traps):
            return
        end_bound = t_hi if atom.lost_at is None else min(t_hi, float(atom.lost_at))
        if end_bound <= t_lo + 1e-15:
            return
        covered: list[tuple[float, float]] = []
        for seg in atom.segments:
            if seg.duration <= 0:
                continue
            a = max(float(seg.start_time), t_lo)
            b = min(float(seg.end_time), end_bound)
            if b > a:
                covered.append((a, b))
        covered.sort()
        merged: list[list[float]] = []
        for a, b in covered:
            if merged and a <= merged[-1][1] + 1e-15:
                merged[-1][1] = max(merged[-1][1], b)
            else:
                merged.append([a, b])
        gaps: list[tuple[float, float]] = []
        cursor = t_lo
        for a, b in merged:
            if a > cursor + 1e-15:
                gaps.append((cursor, a))
            cursor = max(cursor, b)
        if end_bound > cursor + 1e-15:
            gaps.append((cursor, end_bound))
        samples_n = max(4, int(s.idle_trap_samples_per_segment))
        for a, b in gaps:
            if b - a <= 1e-15:
                continue
            ij = np.asarray(atom.position_at(float(a)), dtype=float)
            xy0 = grid.ij_to_xy(ij)
            ts = np.linspace(a, b, samples_n)
            zs = t_to_z(ts)
            xy_arr = np.tile(xy0.reshape(1, 2), (samples_n, 1))
            z_alpha = np.asarray(alpha_at_z(zs), dtype=float)
            wall_alpha = s.idle_trap_alpha * s.trap_max_alpha * z_alpha
            _draw_trap_tube(
                ax, xy_arr, zs, wall_alpha, idle_trap_radius,
                idle_trap_rgb, s.tube_facets,
                sort_zpos=float(np.mean(zs)),
                theta0=s.trap_two_tone_theta,
            )

    def _draw_trajectory_in_range(t_lo: float, t_hi: float) -> None:
        if not show_trajectory or n_layers < 2 or t_max <= t_min:
            return
        for atom in ensemble.atomtrajs:
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
            zs = t_to_z(ts) + s.trajectory_z_offset
            points = np.stack([xy[:, 0], xy[:, 1], zs], axis=1)
            segs = np.stack([points[:-1], points[1:]], axis=1)
            seg_colors = []
            for k in range(len(segs)):
                t_mid = 0.5 * (float(ts[k]) + float(ts[k + 1]))
                seg_colors.append(trajectory_channel_rgba(t_mid, atom))
            lc = Line3DCollection(
                segs, colors=seg_colors,
                linewidths=s.trajectory_lw,
                alpha=s.trajectory_alpha,
            )
            ax.add_collection3d(lc)

    def _draw_loss_markers_in_range(t_lo: float, t_hi: float) -> None:
        # When the projection plate is shown, the loss cross is projected
        # onto it once; suppress the in-stack copy so we don't render two
        # crosses.
        if show_project_plane and pp.show_plane:
            return
        for atom in ensemble.atomtrajs:
            if atom.lost_at is None:
                continue
            if t_lo - 1e-15 <= atom.lost_at <= t_hi + 1e-15:
                _draw_loss_cross(ax, atom, grid, t_to_z, s)

    # Walk from the bottom-most layer up to the top.
    # i = 0 is at lowest z (earliest); i = n_layers - 1 is at highest z (latest).
    if show_layers:
        for i in range(n_layers):
            # Draw the earlier-time layer first; then draw the slab above it.
            # On the next iteration the later-time layer is drawn, masking
            # the trajectory/traps in between.
            _draw_layer(i)
            if i + 1 < n_layers:
                t_above_lo = t_arr[i]
                t_above_hi = t_arr[i + 1]
                # Stack order inside each time slab:
                # earlier layer < trajectory < traps/loss < later layer.
                _draw_trajectory_in_range(t_above_lo, t_above_hi)
                for atom in ensemble.atomtrajs:
                    _draw_idle_trap_in_range(atom, t_above_lo, t_above_hi)
                    for seg in atom.segments:
                        if seg.end_time <= t_above_lo - 1e-15:
                            continue
                        if seg.start_time >= t_above_hi + 1e-15:
                            continue
                        _draw_trap_for_seg(atom, seg, t_above_lo, t_above_hi)
                _draw_loss_markers_in_range(t_above_lo, t_above_hi)

    # --- view + axes ----------------------------------------------------
    if show_layers:
        asp_z = max(1.1, n_layers / 6.0)
        z_view_max = (
            proj_marker_z + 0.5 * s.layer_spacing if project_any else z_max
        )
    else:
        # Projection-plate-only view: shrink the z box so the plate fills
        # the figure instead of floating in a tall stack box.
        z_view_max = max(proj_marker_z * 4.0, 1e-3)
        asp_z = 0.18
    ax.set_xlim(xy_lo, xy_hi)
    ax.set_ylim(xy_lo, xy_hi)
    ax.set_zlim(0, z_view_max)
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
    if s.projection_type == "ortho":
        ax.set_proj_type("ortho")
    elif s.projection_type == "persp":
        ax.set_proj_type("persp", focal_length=s.focal_length)
    else:
        raise ValueError(f"unknown projection_type={s.projection_type!r}")

    if title:
        ax.set_title(title, pad=s.title_pad, fontsize=s.title_fontsize)


def _draw_trap_tube(
    ax, centers_xy, zs_along, wall_alpha, radius, rgb, facets, *,
    sort_zpos=None, theta0: float = 0.0,
    two_tone_half_width: float = np.pi / 9.0,
):
    """Render one trap as a 3D tube whose wall alpha varies along z.

    A two-color channel renders as an angular split: theta in
    [theta0, theta0 + pi] uses color 0, the opposite half uses color 1,
    and a smoothstep band of half-width `two_tone_half_width` centered
    on theta0 + pi ramps through white. Pass `two_tone_half_width=0`
    for an abrupt boundary with no transition.
    """
    n_along = len(centers_xy)
    if n_along < 2:
        return
    rgbs = np.asarray(rgb, dtype=float).reshape(-1, 3)
    angles = float(theta0) + np.linspace(0.0, 2.0 * np.pi, facets)
    cos_a = np.cos(angles)[None, :]
    sin_a = np.sin(angles)[None, :]
    xs = centers_xy[:, 0:1] + radius * cos_a
    ys = centers_xy[:, 1:2] + radius * sin_a
    zs_grid = np.broadcast_to(
        np.asarray(zs_along, dtype=float)[:, None], (n_along, facets),
    )

    face_alpha = 0.5 * (wall_alpha[:-1] + wall_alpha[1:])
    facecolors = np.zeros((n_along - 1, facets - 1, 4))
    if len(rgbs) == 1:
        facecolors[..., 0] = rgbs[0, 0]
        facecolors[..., 1] = rgbs[0, 1]
        facecolors[..., 2] = rgbs[0, 2]
    else:
        face_theta = 0.5 * (angles[:-1] + angles[1:])
        facecolors[..., :3] = _angular_two_tone_rgbs(
            rgbs, face_theta, theta0, half_width=two_tone_half_width,
        )[None, :, :]
    facecolors[..., 3] = face_alpha[:, None]
    surf = ax.plot_surface(
        xs, ys, zs_grid, facecolors=facecolors, shade=False,
        linewidth=0, antialiased=True, edgecolor="none",
    )
    try:
        surf.set_sort_zpos(
            float(np.mean(zs_along)) if sort_zpos is None else float(sort_zpos)
        )
    except AttributeError:
        pass


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


# ---------------------------------------------------------------------------
# 2D top-down ground-plane rendering. Mirrors the bottom-plate look from
# `draw_stack_time` (row/col grid, thick trap bars with arrowheads, target
# outlines, atom discs) but on a regular 2D Matplotlib Axes — no 3D camera,
# no projection guides, no loss cross.
# ---------------------------------------------------------------------------


def _strip_polygon_2d(
    xy0: Sequence[float],
    xy1: Sequence[float],
    width: float,
) -> list[tuple[float, float]] | None:
    p0 = np.asarray(xy0, dtype=float)
    p1 = np.asarray(xy1, dtype=float)
    delta = p1 - p0
    length = float(np.hypot(delta[0], delta[1]))
    if length <= 1e-12:
        return None
    perp = np.array([-delta[1], delta[0]], dtype=float) / length
    half = 0.5 * float(width)
    offset = half * perp
    return [
        (float(p0[0] + offset[0]), float(p0[1] + offset[1])),
        (float(p1[0] + offset[0]), float(p1[1] + offset[1])),
        (float(p1[0] - offset[0]), float(p1[1] - offset[1])),
        (float(p0[0] - offset[0]), float(p0[1] - offset[1])),
    ]


def _draw_planar_segment_strip_2d(
    ax: Any,
    xy0: Sequence[float],
    xy1: Sequence[float],
    width: float,
    rgba: Any,
    *,
    zorder: float = 1.0,
) -> None:
    poly = _strip_polygon_2d(xy0, xy1, width)
    if poly is None:
        return
    ax.add_patch(MplPolygon(
        poly, closed=True, facecolor=rgba, edgecolor="none", linewidth=0,
        zorder=zorder,
    ))


def _draw_solid_planar_arrow_2d(
    ax: Any,
    x0: float, y0: float,
    ux: float, uy: float,
    length: float,
    color: Any,
    lw: float,
    alpha: float,
    head_length: float,
    head_width: float,
    start_offset: float,
    *,
    zorder: float = 1.0,
) -> None:
    if length <= 0.0:
        return
    r, g, b, a = to_rgba(color)
    rgba = (r, g, b, a * alpha)
    start_x = float(x0) + ux * start_offset
    start_y = float(y0) + uy * start_offset
    tip_x = start_x + ux * length
    tip_y = start_y + uy * length
    head_length = min(max(0.0, head_length), 0.72 * length)
    head_width = max(0.0, head_width)
    base_x = tip_x - ux * head_length
    base_y = tip_y - uy * head_length
    shaft_len = length - head_length
    if shaft_len > 0.0:
        ax.plot(
            [start_x, base_x], [start_y, base_y],
            color=rgba, lw=lw, solid_capstyle="round",
            zorder=zorder,
        )
    perp_x = -uy
    perp_y = ux
    tri = [
        (tip_x, tip_y),
        (base_x + 0.5 * head_width * perp_x,
         base_y + 0.5 * head_width * perp_y),
        (base_x - 0.5 * head_width * perp_x,
         base_y - 0.5 * head_width * perp_y),
    ]
    ax.add_patch(MplPolygon(
        tri, closed=True, facecolor=rgba, edgecolor="none", linewidth=0,
        zorder=zorder,
    ))


def _draw_planar_circle_outline_2d(
    ax: Any,
    center_xy: Sequence[float],
    radius: float,
    color: Any,
    lw: float,
    alpha: float,
    dashes: Sequence[float],
    *,
    zorder: float = 1.0,
) -> None:
    if radius <= 0.0:
        return
    r, g, b, a = to_rgba(color)
    circle = MplCircle(
        (float(center_xy[0]), float(center_xy[1])),
        float(radius),
        facecolor="none",
        edgecolor=(r, g, b, a * alpha),
        lw=lw,
        zorder=zorder,
    )
    ax.add_patch(circle)
    circle.set_linestyle((0, tuple(float(x) for x in dashes)))


def _draw_planar_disc_2d(
    ax: Any,
    center_xy: Sequence[float],
    radius: float,
    facecolor: Any,
    edgecolor: Any,
    edge_lw: float,
    *,
    zorder: float = 1.0,
) -> None:
    ax.add_patch(MplCircle(
        (float(center_xy[0]), float(center_xy[1])),
        float(radius),
        facecolor=facecolor,
        edgecolor=edgecolor,
        lw=edge_lw,
        zorder=zorder,
    ))


def _draw_planar_cross_2d(
    ax: Any,
    center_xy: Sequence[float],
    size: float,
    color: Any,
    lw: float,
    alpha: float,
    *,
    zorder: float = 1.0,
) -> None:
    if size <= 0.0:
        return
    cx, cy = float(center_xy[0]), float(center_xy[1])
    r, g, b, a = to_rgba(color)
    rgba = (r, g, b, a * alpha)
    half = 0.5 * float(size)
    for sx, sy in ((1.0, 1.0), (1.0, -1.0)):
        ax.plot(
            [cx - sx * half, cx + sx * half],
            [cy - sy * half, cy + sy * half],
            color=rgba, lw=lw, solid_capstyle="round",
            zorder=zorder,
        )


def _draw_gaussian_blob_2d(
    ax: Any,
    center_xy: Sequence[float],
    radius: float,
    color: Any,
    *,
    sigma_frac: float = 0.42,
    alpha_max: float = 0.85,
    grid_size: int = 96,
    zorder: float = 1.0,
) -> None:
    """Soft single-color Gaussian glow centered on `center_xy`."""
    if radius <= 0.0:
        return
    cx, cy = float(center_xy[0]), float(center_xy[1])
    r_box = float(radius)
    sigma = max(float(sigma_frac) * r_box, 1e-9)
    n = max(8, int(grid_size))
    xs = np.linspace(-r_box, r_box, n)
    ys = np.linspace(-r_box, r_box, n)
    XX, YY = np.meshgrid(xs, ys)
    dist2 = XX * XX + YY * YY
    alpha = np.exp(-dist2 / (2.0 * sigma * sigma)) * float(alpha_max)
    alpha[dist2 > r_box * r_box] = 0.0
    rgb = np.array(to_rgba(color))[:3]
    img = np.empty((n, n, 4), dtype=float)
    img[..., 0] = rgb[0]
    img[..., 1] = rgb[1]
    img[..., 2] = rgb[2]
    img[..., 3] = alpha
    ax.imshow(
        img,
        extent=(cx - r_box, cx + r_box, cy - r_box, cy + r_box),
        origin="lower",
        interpolation="bilinear",
        aspect="auto",
        zorder=zorder,
    )


def _draw_gaussian_blob_two_tone_2d(
    ax: Any,
    center_xy: Sequence[float],
    radius: float,
    color_a: Any,
    color_b: Any,
    *,
    theta0: float = 0.0,
    half_width: float = np.pi / 9.0,
    sigma_frac: float = 0.42,
    alpha_max: float = 0.85,
    grid_size: int = 96,
    zorder: float = 1.0,
) -> None:
    """Two-tone Gaussian glow: angular split with smooth transition band.

    Uses the same `_angular_two_tone_rgbs` color mixing as the 3D trap
    tubes, so two-color gate channels (e.g. Rx = orange/yellow) read
    identically across views.
    """
    if radius <= 0.0:
        return
    cx, cy = float(center_xy[0]), float(center_xy[1])
    r_box = float(radius)
    sigma = max(float(sigma_frac) * r_box, 1e-9)
    n = max(8, int(grid_size))
    xs = np.linspace(-r_box, r_box, n)
    ys = np.linspace(-r_box, r_box, n)
    XX, YY = np.meshgrid(xs, ys)
    dist2 = XX * XX + YY * YY
    alpha = np.exp(-dist2 / (2.0 * sigma * sigma)) * float(alpha_max)
    alpha[dist2 > r_box * r_box] = 0.0
    theta = np.arctan2(YY, XX)
    rgbs = np.asarray([to_rgba(color_a)[:3], to_rgba(color_b)[:3]], dtype=float)
    rgb = _angular_two_tone_rgbs(rgbs, theta, theta0=theta0, half_width=half_width)
    img = np.empty((n, n, 4), dtype=float)
    img[..., :3] = rgb
    img[..., 3] = alpha
    ax.imshow(
        img,
        extent=(cx - r_box, cx + r_box, cy - r_box, cy + r_box),
        origin="lower",
        interpolation="bilinear",
        aspect="auto",
        zorder=zorder,
    )


def draw_ground_plane_2d(
    ax: Any,
    motion: Any,
    t: float,
    *,
    atom_colors: Mapping[int, Any] | Iterable[Any] | None = None,
    show_grid: bool = True,
    show_traps: bool = True,
    show_motion_targets: bool = True,
    show_plane: bool = False,
    show_loss_cross: bool = True,
    style: StackTimeStyle | None = None,
    project_plane: ProjectPlaneStyle | None = None,
    title: str | None = None,
) -> None:
    """Draw a single timestep as a 2D top-down ground-plane figure.

    Reuses the StackTimeStyle knobs that drive the 3D stack styling
    (motion_target_circle_*, atom_*, gate_blob_*, etc.) plus the
    ProjectPlaneStyle knobs that drive the projection plate (grid, trap
    bars with arrowheads, plate alpha) so the look matches the 3D
    projection-plate rendering.
    """
    s = style if style is not None else DEFAULT_STYLE
    pp = project_plane if project_plane is not None else DEFAULT_PROJECT_PLANE
    ensemble = motion if isinstance(motion, AtomEnsemble) else motion.ensemble
    t = float(t)

    colors = _atom_colors_rgba(
        [int(a.atom_id) for a in ensemble.atomtrajs], atom_colors,
    )

    grid = ensemble.grid
    d = grid.d
    atom_radius = np.sqrt(0.7) * s.atom_radius_frac * d
    site_lo = (0.0 - grid.center) * d
    site_hi = ((grid.N - 1.0) - grid.center) * d
    line_lo = site_lo - s.lattice_line_extension_frac * d
    line_hi = site_hi + s.lattice_line_extension_frac * d
    plane_lo = site_lo - s.layer_plane_extension_frac * d
    plane_hi = site_hi + s.layer_plane_extension_frac * d
    bound_lo = min(plane_lo, line_lo)
    bound_hi = max(plane_hi, line_hi)
    proj_trap_width_2d = max(
        0.006 * d,
        float(pp.trap_width_frac) * d,
        0.018 * float(pp.trap_lw) * d,
    )

    # Auto-expand bounds for off-lattice endpoints (fresh-load reservoirs).
    margin = 0.5 * d
    for atom in ensemble.atomtrajs:
        ij_pts = [np.asarray(atom.initial_pos, dtype=float)]
        for seg in atom.segments:
            ij_pts.append(np.asarray(seg.start_pos, dtype=float))
            ij_pts.append(np.asarray(seg.end_pos, dtype=float))
        for ij in ij_pts:
            x, y = grid.ij_to_xy(ij)
            bound_lo = min(bound_lo, float(x) - margin, float(y) - margin)
            bound_hi = max(bound_hi, float(x) + margin, float(y) + margin)

    plane_color = (
        s.layer_plane_color
        if pp.plane_color is None
        else pp.plane_color
    )

    # Layer order (lowest -> highest matplotlib zorder):
    #   plane bg  ->  grid lines  ->  trap projection  ->  target outline
    #             ->  atom disc
    Z_PLANE = 0.0
    Z_GRID = 1.0
    Z_TRAP = 2.0
    Z_TARGET = 3.0
    Z_ATOM = 4.0

    if show_plane:
        ax.add_patch(MplPolygon(
            [
                (bound_lo, bound_lo),
                (bound_hi, bound_lo),
                (bound_hi, bound_hi),
                (bound_lo, bound_hi),
            ],
            closed=True,
            facecolor=plane_color,
            alpha=pp.plane_alpha,
            edgecolor="none",
            linewidth=0,
            zorder=Z_PLANE,
        ))

    if show_grid:
        bot_row_color = to_rgba(s.lattice_line_colors.get("row", "#ff2828"))
        bot_col_color = to_rgba(s.lattice_line_colors.get("col", "#29b0ff"))
        for jj in range(grid.N):
            y_row = (jj - grid.center) * d
            ax.plot(
                [line_lo, line_hi], [y_row, y_row],
                color=bot_row_color, lw=pp.grid_lw,
                alpha=pp.grid_alpha, solid_capstyle="round",
                zorder=Z_GRID,
            )
        for ii in range(grid.N):
            x_col = (ii - grid.center) * d
            ax.plot(
                [x_col, x_col], [line_lo, line_hi],
                color=bot_col_color, lw=pp.grid_lw,
                alpha=pp.grid_alpha, solid_capstyle="round",
                zorder=Z_GRID,
            )

    # Currently-active trap segments. Non-zero displacement renders as a
    # thick colored bar with arrowhead (mirrors `project_plane.show_traps`
    # in `draw_stack_time`); zero-displacement segments are stationary
    # single-qubit gates and render as a Gaussian glow at the atom's site.
    if show_traps:
        gate_blob_radius = float(s.gate_blob_radius_frac) * d
        for atom in ensemble.atomtrajs:
            if atom.is_lost_at(t):
                continue
            seg = _intended_segment(atom, t)
            if seg is None:
                continue
            xy0 = grid.ij_to_xy(np.asarray(seg.start_pos, dtype=float))
            xy1 = grid.ij_to_xy(np.asarray(seg.end_pos, dtype=float))
            color_spec = s.trap_channel_colors.get(
                seg.channel, s.trap_channel_colors.get(None, "#9a9a9a"),
            )
            if np.hypot(*(xy1 - xy0)) <= 1e-12:
                center_xy = grid.ij_to_xy(np.asarray(
                    atom.position_at(t), dtype=float,
                ))
                colors_list = _colorway(color_spec)
                if len(colors_list) >= 2:
                    _draw_gaussian_blob_two_tone_2d(
                        ax, center_xy, gate_blob_radius,
                        colors_list[0], colors_list[1],
                        theta0=s.trap_two_tone_theta,
                        half_width=s.trap_two_tone_transition_widths.get(
                            seg.channel, s.trap_two_tone_transition_width,
                        ),
                        sigma_frac=s.gate_blob_sigma_frac,
                        alpha_max=s.gate_blob_alpha_max,
                        grid_size=s.gate_blob_grid_size,
                        zorder=Z_TRAP,
                    )
                else:
                    _draw_gaussian_blob_2d(
                        ax, center_xy, gate_blob_radius,
                        colors_list[0],
                        sigma_frac=s.gate_blob_sigma_frac,
                        alpha_max=s.gate_blob_alpha_max,
                        grid_size=s.gate_blob_grid_size,
                        zorder=Z_TRAP,
                    )
                continue
            r, g, b, _ = _blend_colorway(color_spec)
            _draw_planar_segment_strip_2d(
                ax, xy0, xy1, proj_trap_width_2d,
                (r, g, b, pp.trap_alpha),
                zorder=Z_TRAP,
            )
            arrow_len = max(0.0, float(pp.trap_arrow_length_frac)) * d
            if arrow_len > 0.0:
                direction = xy1 - xy0
                norm = float(np.hypot(direction[0], direction[1]))
                if norm > 1e-12:
                    ux = float(direction[0]) / norm
                    uy = float(direction[1]) / norm
                    center = 0.5 * (xy0 + xy1)
                    arrow_start = center - 0.5 * arrow_len * np.array([ux, uy])
                    arrow_color = _darken((r, g, b, 1.0), 0.62)
                    _draw_solid_planar_arrow_2d(
                        ax,
                        float(arrow_start[0]), float(arrow_start[1]),
                        ux, uy, arrow_len, arrow_color,
                        pp.trap_arrow_lw,
                        pp.trap_arrow_alpha,
                        min(
                            max(0.0, pp.trap_arrow_head_length_frac) * d,
                            0.45 * arrow_len,
                        ),
                        max(0.0, pp.trap_arrow_head_width_frac) * d,
                        0.0,
                        zorder=Z_TRAP,
                    )

    if show_motion_targets:
        for atom in ensemble.atomtrajs:
            if atom.is_lost_at(t):
                continue
            seg = _intended_segment(atom, t)
            if seg is None:
                continue
            target_xy = grid.ij_to_xy(np.asarray(seg.end_pos, dtype=float))
            current_xy = grid.ij_to_xy(np.asarray(
                atom.position_at(t), dtype=float,
            ))
            if np.hypot(
                float(target_xy[0]) - float(current_xy[0]),
                float(target_xy[1]) - float(current_xy[1]),
            ) <= 1e-12:
                continue
            _draw_planar_circle_outline_2d(
                ax, target_xy, atom_radius,
                s.motion_target_circle_color,
                s.motion_target_circle_lw,
                s.motion_target_circle_alpha,
                s.motion_target_circle_dashes,
                zorder=Z_TARGET,
            )

    if show_loss_cross:
        # Mark the *future* loss site as a projection: only render the X
        # while the loss is still upcoming (t < lost_at). Once the atom is
        # lost, the X disappears.
        loss_cross_size = 0.5 * d
        for atom in ensemble.atomtrajs:
            if atom.lost_at is None:
                continue
            if float(atom.lost_at) <= t + 1e-12:
                continue
            loss_xy = grid.ij_to_xy(np.asarray(
                atom.position_at(float(atom.lost_at)), dtype=float,
            ))
            _draw_planar_cross_2d(
                ax, loss_xy, loss_cross_size,
                s.loss_marker_color, s.loss_marker_lw, 1.0,
                zorder=Z_TARGET,
            )

    fresh_rgba_full = to_rgba(s.fresh_atom_color)
    edge_rgba = to_rgba(s.atom_edge_color)
    ij_now = ensemble.positions_at(t)
    atom_xy = grid.ij_to_xy(ij_now)
    for k, atom in enumerate(ensemble.atomtrajs):
        if atom.is_lost_at(t):
            continue
        if _is_fresh_at(atom, t):
            face = fresh_rgba_full
        else:
            face = to_rgba(colors[k])
        _draw_planar_disc_2d(
            ax, atom_xy[k], atom_radius, face, edge_rgba, s.atom_edge_lw,
            zorder=Z_ATOM,
        )

    ax.set_facecolor("white")

    ax.set_xlim(bound_lo, bound_hi)
    ax.set_ylim(bound_lo, bound_hi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    if title:
        ax.set_title(title, pad=s.title_pad, fontsize=s.title_fontsize)


def save_ground_plane_2d(
    motion: Any,
    t: float,
    output_path: str | Path,
    *,
    style: StackTimeStyle | None = None,
    title: str | None = None,
    transparent: bool = True,
    **draw_kwargs: Any,
) -> Path:
    """Render a single 2D top-down ground-plane snapshot to a PNG file."""
    s = style if style is not None else DEFAULT_STYLE
    fig, ax = plt.subplots(figsize=s.figsize, dpi=s.dpi)
    draw_ground_plane_2d(ax, motion, t, style=s, title=title, **draw_kwargs)
    if not transparent:
        fig.patch.set_facecolor("white")
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        out,
        dpi=s.dpi,
        transparent=transparent,
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
    )
    plt.close(fig)
    return out
