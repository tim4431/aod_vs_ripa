"""Visualization for atom rearrangement motion.

A flat collection of `draw_*(ax, ...)` primitives plus three composers:
`draw_atom_panel(ax, motion, t, ...)` wires the atom-plane primitives,
`draw_frame(motion, t, view=...)` builds a fresh figure for one of
`demo` / `benchmark` / `detail`, `save_frame(...)` writes one PNG, and
`render_animation(...)` calls `draw_frame` per timestep and stitches the
PNGs into a GIF.

Tone frequencies are normalized to [0, 1) — i.e. `nu / FSR1`. With
FSR2 = FSR1 / N, the RIPA spectrometer mapping reduces to:

    nu_row(i, j) = (j + i / N) / N   (mod 1)
    nu_col(i, j) = (i + j / N) / N   (mod 1)
"""

from __future__ import annotations

import math
import os
import shutil
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Literal, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import to_rgba  # noqa: E402
from matplotlib.patches import Circle, Rectangle  # noqa: E402

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
from tqdm.auto import tqdm  # noqa: E402

from .atom_config import Grid  # noqa: E402
from .atom_trajectory import AtomEnsemble, AtomTrajectory  # noqa: E402
from .drawing import Z_COLOR_CODE, draw_color_code_pattern  # noqa: E402, F401
from .segments import BANG_BANG, Segment  # noqa: E402

ToneChannel = Literal["row", "col"]
ToneHardware = Literal["AOD", "EOM", "AOD/EOM", "tone"]
PlannedTrajectoryMode = Literal["none", "full", "next"]
AddressedStyle = Literal["edge", "blob", "both", "none"]
RenderView = Literal["demo", "benchmark", "detail"]
RenderQuality = Literal["speed", "quality"]
RenderFormat = Literal["gif", "webp", "apng"]

# Callback used to paint an extra layer on a single atom-panel axes. Called
# with (ax, ensemble) once per panel on the start hold or the trailing
# (finished) hold; the panel passes its OWN ensemble so per-panel overlays
# (e.g. a Reg(3) code overlay drawn only when that panel reaches its end)
# work naturally in the benchmark view. Must be picklable -- use a module-
# level function or `functools.partial` of one for the worker pool.
PanelOverlay = Callable[[Any, AtomEnsemble], None]

ATOM_SIZE = 40.0
GRIDPOINT_SIZE = 12.0
BLOB_SIGMA = 1.5  # gaussian halo std-dev (um, in xy data coords)
TIME_FACTOR_US = 1e6
FREQ_LABEL = "nu / FSR1 (mod 1)"

# Render z-order layering, low -> high. Anything painted with a higher
# zorder draws on top of lower zorder layers. `Z_COLOR_CODE` lives in
# `drawing.py` (the pattern itself is defined there) and is re-exported
# above for callers that want to layer against it.
Z_GRID_DOTS = 1.0       # background lattice dots
Z_GRID_FRAME = 1.2      # dashed grid bbox
Z_PLANNED_PATH = 2.0    # dashed future trajectories
Z_ROUTING_ARROW = 2.2   # src->dst arrows
Z_MOTION_TRAIL = 2.5    # motion blur trail line
Z_MOTION_DOT = 2.7      # motion blur fade dots
Z_TRAP_BLOB = 4.0       # soft red gaussian halo (under atoms)
Z_ATOM = 5.0            # atom marker
Z_TRAP_OUTLINE = 6.0    # ring around addressed atom / square for empty AOD trap
Z_GATE_OVERLAY = 6.5    # dashed Rydberg-blockade ring + beam blob for active gate
Z_ATOM_ID = 7.0         # atom-id text label

GATE_BLOCKADE_RADIUS_UM = 1.4
GATE_RING_RGBA = (0.85, 0.18, 0.18, 0.90)
GATE_BEAM_RGBA = (0.96, 0.55, 0.10, 0.70)
GATE_RAMP_TIME = 10.0e-6

DEMO_FIGSIZE = (5.4, 5.2)
DETAIL_FIGSIZE = (12.0, 6.2)
BENCH_HEIGHT = 4.6
BENCH_WIDTH_PER_PANEL = 4.0


@dataclass(frozen=True)
class _Style:
    dpi: int
    fps: int
    addressed_style: AddressedStyle
    show_motion_blur: bool
    trail_samples: int
    tone_samples_per_segment: int
    planned_samples_per_segment: int


_QUALITY: dict[RenderQuality, _Style] = {
    "speed": _Style(
        dpi=60,
        fps=6,
        addressed_style="edge",
        show_motion_blur=False,
        trail_samples=0,
        tone_samples_per_segment=32,
        planned_samples_per_segment=18,
    ),
    "quality": _Style(
        dpi=100,
        fps=20,
        addressed_style="blob",
        show_motion_blur=True,
        trail_samples=12,
        tone_samples_per_segment=96,
        planned_samples_per_segment=48,
    ),
}


# --- draw primitives -------------------------------------------------------


def draw_grid_dots(ax: Any, grid: Grid) -> None:
    """Draw the gray underlying lattice dot at every grid site."""
    ii, jj = np.meshgrid(np.arange(grid.N), np.arange(grid.N), indexing="ij")
    xy = grid.ij_to_xy(np.column_stack([ii.ravel(), jj.ravel()]).astype(float))
    ax.scatter(
        xy[:, 0], xy[:, 1], s=GRIDPOINT_SIZE * 0.7, c="#9a9a9a",
        alpha=0.22, linewidths=0, zorder=Z_GRID_DOTS,
    )


def draw_grid_frame(ax: Any, grid: Grid) -> None:
    """Draw the dashed grey rectangle that frames the grid extent and set axis limits."""
    lo = (0.0 - grid.center) * grid.d
    hi = ((grid.N - 1.0) - grid.center) * grid.d
    b0 = lo - grid.d / 2.0
    b1 = hi + grid.d / 2.0
    width = b1 - b0
    ax.add_patch(
        Rectangle(
            (b0, b0), width, width, fill=False, linestyle="--",
            linewidth=1.0, edgecolor="#bbbbbb", zorder=Z_GRID_FRAME,
        )
    )
    ax.set_xlim(lo - grid.d, hi + grid.d)
    ax.set_ylim(lo - grid.d, hi + grid.d)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (um)")
    ax.set_ylabel("y (um)")


def draw_traps(
    ax: Any,
    sequence: Any,
    t: float,
    *,
    addressed_style: AddressedStyle = "edge",
    trap_scale: float = 1.0,
) -> None:
    """Mark active traps for AOD and RIPA hardware.

    Two styles share one red color. Traps that hold an atom (any active
    AOD or RIPA segment) are drawn as a circle ring around the atom.
    AOD lattice intersections that hold no atom are drawn as an empty
    red square. `addressed_style`: "edge" uses ring/square outlines,
    "blob" uses a soft Gaussian halo, "both" draws each, "none" skips.
    """
    if addressed_style == "none":
        return
    ensemble = getattr(sequence, "ensemble", None)
    steps = getattr(sequence, "steps", None)
    if ensemble is None:
        return

    # Atoms in any active segment (AOD or RIPA) -> ring around the atom.
    addressed_ij = [
        atom.position_at(t)
        for atom in ensemble.atomtrajs
        if _active_segment(atom, t) is not None
    ]
    addressed_xy = (
        ensemble.grid.ij_to_xy(np.asarray(addressed_ij, dtype=float))
        if addressed_ij else np.empty((0, 2), dtype=float)
    )

    # AOD lattice intersections with no atom -> empty square.
    empty_aod_xy = (
        _active_aod_traps_xy(steps, ensemble, t)
        if steps else np.empty((0, 2), dtype=float)
    )
    if len(empty_aod_xy):
        atom_xy = ensemble.grid.ij_to_xy(ensemble.positions_at(t))
        if len(atom_xy):
            tol = ensemble.grid.d * 1e-3
            dists = np.linalg.norm(
                empty_aod_xy[:, None, :] - atom_xy[None, :, :], axis=2,
            )
            empty_aod_xy = empty_aod_xy[~(dists < tol).any(axis=1)]

    if addressed_style in ("blob", "both"):
        for x, y in addressed_xy:
            _draw_gaussian_blob(ax, float(x), float(y), trap_scale=trap_scale)
        for x, y in empty_aod_xy:
            _draw_gaussian_blob(ax, float(x), float(y), trap_scale=trap_scale)
    if addressed_style in ("edge", "both"):
        if len(addressed_xy):
            ax.scatter(
                addressed_xy[:, 0], addressed_xy[:, 1],
                s=ATOM_SIZE * 2.0 * trap_scale**2, marker="o",
                facecolors="none", edgecolors="#d62728",
                linewidths=1.4, alpha=0.85, zorder=Z_TRAP_OUTLINE,
            )
        if len(empty_aod_xy):
            ax.scatter(
                empty_aod_xy[:, 0], empty_aod_xy[:, 1],
                s=GRIDPOINT_SIZE * 5.0 * trap_scale**2, marker="s",
                facecolors="none", edgecolors="#d62728",
                linewidths=1.1, alpha=0.70, zorder=Z_TRAP_OUTLINE,
            )


def draw_gate_overlay(
    ax: Any,
    sequence: Any,
    t: float,
    *,
    gate_type: str = "CZ",
    blockade_radius_um: float = GATE_BLOCKADE_RADIUS_UM,
    ring_rgba: tuple[float, float, float, float] = GATE_RING_RGBA,
    beam_rgba: tuple[float, float, float, float] = GATE_BEAM_RGBA,
    ramp_time: float = GATE_RAMP_TIME,
) -> None:
    """For each currently-active gate pulse whose `gate_type` matches the
    `gate_type` argument, draw dashed Rydberg-blockade rings around its
    participating atoms plus a Gaussian "beam" blob at their midpoint.

    Steps are recognized by duck typing: any item in `sequence.steps`
    carrying `atom_ids` (a non-empty tuple), `start_time`, `duration`, and
    matching `gate_type` is treated as a gate pulse -- matches
    `src.movement.GateStep`. `AODStep`/`RIPAStep` have no `gate_type`, so
    they're skipped. This blockade-ring + beam style is the canonical CZ
    look; other gate types should be rendered by a different primitive
    (call this function again with a different `gate_type` + colors, or
    layer your own overlay).

    Intensity follows a linear ramp-in -> plateau -> ramp-out envelope
    sized by `ramp_time`, all contained within `[start_time, start_time +
    duration]` so the overlay only appears once the prior move has ended.
    """
    ensemble = getattr(sequence, "ensemble", None)
    steps = getattr(sequence, "steps", None)
    if ensemble is None or not steps:
        return
    grid = ensemble.grid

    for step in steps:
        if getattr(step, "gate_type", None) != gate_type:
            continue
        atom_ids = getattr(step, "atom_ids", None)
        start_time = getattr(step, "start_time", None)
        duration = getattr(step, "duration", None)
        if not atom_ids or start_time is None or duration is None:
            continue
        intensity = _gate_envelope(
            float(start_time), float(duration), float(t), ramp_time,
        )
        if intensity <= 0.0:
            continue
        ij = np.asarray(
            [ensemble.atomtraj_by_id(aid).position_at(t) for aid in atom_ids],
            dtype=float,
        )
        xy = grid.ij_to_xy(ij)
        rrgba = (ring_rgba[0], ring_rgba[1], ring_rgba[2], ring_rgba[3] * intensity)
        for x, y in xy:
            ax.add_patch(
                Circle(
                    (float(x), float(y)),
                    blockade_radius_um,
                    fill=False, linestyle="--", linewidth=1.6,
                    edgecolor=rrgba, zorder=Z_GATE_OVERLAY,
                )
            )
        mid_x, mid_y = xy.mean(axis=0)
        brgba = (beam_rgba[0], beam_rgba[1], beam_rgba[2], beam_rgba[3] * intensity)
        _draw_gaussian_blob(
            ax, float(mid_x), float(mid_y), trap_scale=1.6, rgba=brgba,
        )


def _gate_envelope(start: float, duration: float, t: float, ramp_time: float) -> float:
    """0 -> linear ramp-in -> plateau -> linear ramp-out -> 0, contained in
    `[start, start + duration]` so the ramp-in only fires after the prior
    move has actually ended."""
    end = start + duration
    if t < start or t >= end:
        return 0.0
    elapsed = t - start
    remaining = end - t
    if elapsed < ramp_time:
        return elapsed / ramp_time
    if remaining < ramp_time:
        return remaining / ramp_time
    return 1.0


def draw_routing_request(
    ax: Any,
    ensemble: AtomEnsemble,
    *,
    colors: list[Any] | None = None,
) -> None:
    """Draw a per-atom arrow from its initial site to its final site.

    For unlabeled requests, the realized atom-by-atom pairing is what the
    scheduler chose, not the abstract source/target sets — this primitive
    walks the actual ensemble state, so the arrows always reflect the
    pairwise routing that the run produced.
    """
    colors = colors or _default_colors(ensemble)
    grid = ensemble.grid
    initial_xy = grid.ij_to_xy(ensemble.positions_at(0.0))
    final_xy = grid.ij_to_xy(ensemble.positions_at(ensemble.total_duration()))
    for (x0, y0), (x1, y1), color in zip(initial_xy, final_xy, colors):
        if math.isclose(x0, x1) and math.isclose(y0, y1):
            continue
        ax.annotate(
            "",
            xy=(x1, y1),
            xytext=(x0, y0),
            arrowprops=dict(
                arrowstyle="->",
                color=color,
                lw=1.6,
                alpha=1.0,
                shrinkA=4,
                shrinkB=4,
            ),
            zorder=Z_ROUTING_ARROW,
        )


def draw_planned_paths(
    ax: Any,
    ensemble: AtomEnsemble,
    t: float,
    *,
    mode: PlannedTrajectoryMode = "full",
    colors: list[Any] | None = None,
    samples: int = 18,
) -> None:
    """Draw a dashed line for each atom's future path: full plan or just the next segment."""
    if mode == "none" or samples <= 1:
        return
    colors = colors or _default_colors(ensemble)
    grid = ensemble.grid
    for idx, atom in enumerate(ensemble.atomtrajs):
        future = list(_future_segments(atom, t))
        if mode == "next":
            future = future[:1]
        for seg in future:
            t0 = max(float(t), seg.start_time)
            xy = _sample_segment_xy(grid, seg, t0, seg.end_time, samples)
            if len(xy) > 1:
                ax.plot(
                    xy[:, 0], xy[:, 1], linestyle="--", linewidth=1.1,
                    color=colors[idx], alpha=0.34, zorder=Z_PLANNED_PATH,
                )


def draw_motion_blur(
    ax: Any,
    ensemble: AtomEnsemble,
    t: float,
    *,
    samples: int = 12,
    colors: list[Any] | None = None,
    atom_scale: float = 1.0,
) -> None:
    """Draw a fading speed-scaled trail behind every currently-moving atom."""
    if samples <= 1:
        return
    colors = colors or _default_colors(ensemble)
    grid = ensemble.grid
    for idx, atom in enumerate(ensemble.atomtrajs):
        seg = _active_segment(atom, t)
        if seg is None:
            continue
        xy = _motion_blur_xy(grid, seg, float(t), samples)
        if len(xy) <= 1:
            continue
        color = colors[idx]
        ax.plot(
            xy[:, 0], xy[:, 1], color=color, linewidth=4.0, alpha=0.20,
            zorder=Z_MOTION_TRAIL,
        )
        fade = np.linspace(0.08, 0.32, len(xy))
        sizes = np.linspace(
            ATOM_SIZE * 0.15 * atom_scale**2,
            ATOM_SIZE * 0.55 * atom_scale**2,
            len(xy),
        )
        for (x, y), alpha, size in zip(xy, fade, sizes):
            ax.scatter(
                [x], [y], s=size, c=[color], alpha=float(alpha),
                linewidths=0, zorder=Z_MOTION_DOT,
            )


def draw_atoms(
    ax: Any,
    ensemble: AtomEnsemble,
    t: float,
    *,
    colors: list[Any] | None = None,
    show_atom_ids: bool = False,
    atom_scale: float = 1.0,
    edge_color: Any = "#ffffff",
    edge_linewidth: float = 0.5,
) -> None:
    """Draw atoms at their positions at time `t`. Trap markers live in `draw_traps`."""
    colors = colors or _default_colors(ensemble)
    grid = ensemble.grid
    positions_xy = grid.ij_to_xy(ensemble.positions_at(t))
    ax.scatter(
        positions_xy[:, 0], positions_xy[:, 1], s=ATOM_SIZE * atom_scale**2,
        c=colors, edgecolors=edge_color, linewidths=edge_linewidth,
        zorder=Z_ATOM,
    )

    if show_atom_ids:
        offset = 0.22 * grid.d
        for atom, (x, y), color in zip(ensemble.atomtrajs, positions_xy, colors):
            ax.text(
                x, y + offset, str(int(atom.atom_id)),
                ha="center", va="bottom", fontsize=7,
                color=_darken_color(color), zorder=Z_ATOM_ID,
            )


def draw_current_tones(
    ax: Any,
    ensemble: AtomEnsemble,
    t: float,
    channel: ToneChannel,
    *,
    colors: list[Any] | None = None,
) -> None:
    """Draw the instantaneous spectrum of currently-active tones for one channel as vlines."""
    ax.clear()
    colors = colors or _default_colors(ensemble)
    N = ensemble.grid.N
    for idx, atom in enumerate(ensemble.atomtrajs):
        seg = _active_segment(atom, t)
        if seg is None or channel not in _channels_for_segment(seg):
            continue
        i, j = atom.position_at(t)
        freq = _tone_freq(channel, i, j, N, segment=seg)
        color = colors[idx]
        ax.vlines(freq, 0.0, 1.0, color=color, linewidth=2.5, alpha=0.9)
        ax.scatter([freq], [1.0], c=[color], s=22.0, zorder=3)
        ax.text(
            freq, 1.04, str(int(atom.atom_id)),
            ha="center", va="bottom", fontsize=6,
            color=_darken_color(color), clip_on=True,
        )
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.2)
    ax.set_xlabel(FREQ_LABEL)
    ax.set_ylabel("amplitude")
    hardware = _tone_hardware_name(ensemble)
    axis = "x" if channel == "row" else "y"
    panel_title = (
        f"{channel} tone (controls {axis})"
        if hardware == "tone"
        else f"{channel} {hardware} (controls {axis})"
    )
    ax.set_title(panel_title)
    ax.grid(True, color="#e5e5e5", linewidth=0.6)


def draw_tone_history(
    ax: Any,
    ensemble: AtomEnsemble,
    t: float,
    channel: ToneChannel,
    *,
    mode: PlannedTrajectoryMode = "full",
    colors: list[Any] | None = None,
    samples: int = 32,
) -> None:
    """Draw the tone-vs-time history for one channel, with optional planned future."""
    ax.clear()
    colors = colors or _default_colors(ensemble)
    trajs = _collect_tone_trajectories(ensemble, channel, max(2, int(samples)))
    t_now = float(t)
    next_by_atom = _next_tone_trajectory_by_atom(trajs, t_now) if mode == "next" else {}

    for traj in trajs:
        ts = traj["times"]
        nus = traj["freqs"]
        idx = traj["idx"]
        if mode == "full":
            _plot_wrapped_freq_line(
                ax, ts * TIME_FACTOR_US, nus, 1.0,
                color=colors[idx], linewidth=1.4, alpha=0.35,
            )
        else:
            mask = ts <= t_now + 1e-15
            if np.count_nonzero(mask) >= 2:
                _plot_wrapped_freq_line(
                    ax, ts[mask] * TIME_FACTOR_US, nus[mask], 1.0,
                    color=colors[idx], linewidth=1.4, alpha=0.45,
                )
            if mode == "next" and next_by_atom.get(idx) is traj:
                mask = ts >= t_now - 1e-15
                if np.count_nonzero(mask) >= 2:
                    _plot_wrapped_freq_line(
                        ax, ts[mask] * TIME_FACTOR_US, nus[mask], 1.0,
                        color=colors[idx], linewidth=1.4, alpha=0.35,
                    )

    total = max(ensemble.total_duration(), t_now)
    x_hi = max(total * TIME_FACTOR_US, 1.0)
    ax.axvline(t_now * TIME_FACTOR_US, color="#b8b8b8", linewidth=1.0)
    ax.set_xlim(0.0, x_hi)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("t (us)")
    ax.set_ylabel(FREQ_LABEL)
    ax.grid(True, color="#e5e5e5", linewidth=0.6)


def draw_atom_panel(
    ax: Any,
    motion: Any,
    t: float,
    *,
    quality: RenderQuality = "speed",
    atom_colors: Mapping[int, Any] | Iterable[Any] | None = None,
    planned_trajectory: PlannedTrajectoryMode = "none",
    show_atom_ids: bool = False,
    show_routing: bool = False,
    show_color_code: bool = False,
    color_code_swap: bool = False,
    start_overlay: PanelOverlay | None = None,
    end_overlay: PanelOverlay | None = None,
    is_start_hold: bool = False,
    title: str | None = None,
    atom_scale: float = 1.0,
    trap_scale: float = 1.0,
) -> None:
    """Compose one atom-plane panel by calling the visual primitives in z-order."""
    style = _QUALITY[quality]
    if isinstance(motion, AtomEnsemble):
        ensemble, sequence = motion, None
    else:
        ensemble, sequence = motion.ensemble, motion
    colors = _atom_colors_rgba(
        [int(a.atom_id) for a in ensemble.atomtrajs], atom_colors
    )

    finished = t >= ensemble.total_duration() - 1e-12
    # When `color_code_swap` is set, the overlay only fires on a panel
    # once its own ensemble has finished moving (lets each scheduler
    # transition into the post-Hadamard state independently). Without
    # the swap flag (start hold) the un-swapped overlay is drawn on
    # every panel regardless.
    draw_overlay = show_color_code and (finished or not color_code_swap)
    addressed_style: AddressedStyle = (
        "none" if finished or draw_overlay else style.addressed_style
    )

    if draw_overlay:
        draw_color_code_pattern(ax, ensemble, swap_colors=color_code_swap)
    if is_start_hold and start_overlay is not None:
        start_overlay(ax, ensemble)
    if finished and end_overlay is not None:
        end_overlay(ax, ensemble)
    draw_grid_dots(ax, ensemble.grid)
    if show_routing:
        draw_routing_request(ax, ensemble, colors=colors)
    draw_planned_paths(
        ax, ensemble, t, mode=planned_trajectory, colors=colors,
        samples=style.planned_samples_per_segment,
    )
    if style.show_motion_blur and not finished:
        draw_motion_blur(
            ax, ensemble, t, samples=style.trail_samples, colors=colors,
            atom_scale=atom_scale,
        )
    draw_atoms(
        ax, ensemble, t, colors=colors, show_atom_ids=show_atom_ids,
        atom_scale=atom_scale,
    )
    if sequence is not None and not finished:
        draw_traps(
            ax, sequence, t, addressed_style=addressed_style,
            trap_scale=trap_scale,
        )
        draw_gate_overlay(ax, sequence, t)
    draw_grid_frame(ax, ensemble.grid)
    ax.set_title(title or f"atom motion  -  t = {_format_time_us(t)}")


def _draw_speedup_indicator(ax: Any, speed_str: str, frame_index: int) -> None:
    """Draw a "Nx ›››" badge in the panel's upper-left with a chasing-chevron
    pulse so it visually reads as a fast-forward playback overlay."""
    color = "#d62728"
    ax.text(
        0.02, 0.97, f"{speed_str}x",
        transform=ax.transAxes, ha="left", va="top",
        fontsize=14, fontweight="bold", color=color, zorder=10,
    )
    phase = int(frame_index) % 3
    for i in range(3):
        alpha = 1.0 if i == phase else 0.25
        ax.text(
            0.10 + i * 0.035, 0.97, "›",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=16, fontweight="bold", color=color, alpha=alpha,
            zorder=10,
        )


# --- frame composer + IO ---------------------------------------------------


def draw_frame(
    motion: Any,
    t: float,
    *,
    view: RenderView = "demo",
    quality: RenderQuality = "speed",
    labels: Iterable[str] | None = None,
    atom_colors: Mapping[int, Any] | Iterable[Any] | None = None,
    planned_trajectory: PlannedTrajectoryMode = "none",
    show_atom_ids: bool = False,
    show_routing: bool = False,
    show_color_code: bool = False,
    color_code_swap: bool = False,
    start_overlay: PanelOverlay | None = None,
    end_overlay: PanelOverlay | None = None,
    is_start_hold: bool = False,
    title: str | None = None,
    atom_scale: float = 1.0,
    trap_scale: float = 1.0,
    panel_speedup: Mapping[str, float] | None = None,
    frame_index: int = 0,
    is_motion_frame: bool = True,
) -> tuple[Any, list[Any]]:
    """Build a fresh figure for `view` and draw the primitives on it; return (fig, axes)."""
    style = _QUALITY[quality]

    if view == "demo":
        fig, ax = plt.subplots(figsize=DEMO_FIGSIZE, constrained_layout=True)
        draw_atom_panel(
            ax, motion, t, quality=quality, atom_colors=atom_colors,
            planned_trajectory=planned_trajectory,
            show_atom_ids=show_atom_ids, show_routing=show_routing,
            show_color_code=show_color_code, color_code_swap=color_code_swap, start_overlay=start_overlay, end_overlay=end_overlay, is_start_hold=is_start_hold,
            title=title, atom_scale=atom_scale, trap_scale=trap_scale,
        )
        return fig, [ax]

    if view == "benchmark":
        items = _normalize_motion_items(motion, labels)
        if not items:
            raise ValueError("benchmark view needs at least one motion object")
        n = len(items)
        figsize = (max(BENCH_WIDTH_PER_PANEL, BENCH_WIDTH_PER_PANEL * n), BENCH_HEIGHT)
        fig, axes_array = plt.subplots(
            1, n, figsize=figsize, squeeze=False, constrained_layout=True
        )
        axes_list = list(axes_array.ravel())
        for (label, panel_motion), ax in zip(items, axes_list):
            panel_ensemble = (
                panel_motion if isinstance(panel_motion, AtomEnsemble)
                else panel_motion.ensemble
            )
            factor = (
                float(panel_speedup.get(label, 1.0))
                if panel_speedup is not None else 1.0
            )
            panel_total = panel_ensemble.total_duration()
            panel_t = min(float(t) * factor, panel_total)
            speed_tag = ""
            speed_str = ""
            if factor != 1.0:
                speed_str = f"{int(factor)}" if factor.is_integer() else f"{factor:g}"
                speed_tag = rf"  ($\mathbf{{{speed_str}x}}$ speed)"
            draw_atom_panel(
                ax, panel_motion, panel_t, quality=quality, atom_colors=atom_colors,
                planned_trajectory=planned_trajectory,
                show_atom_ids=show_atom_ids, show_routing=show_routing,
                show_color_code=show_color_code, color_code_swap=color_code_swap, start_overlay=start_overlay, end_overlay=end_overlay, is_start_hold=is_start_hold,
                title=f"{label}{speed_tag}",
                atom_scale=atom_scale, trap_scale=trap_scale,
            )
            # Right-anchored so digits of varying width don't shift its position.
            ax.text(
                0.98, 0.97, f"t = {_format_time_us(panel_t)}",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=10, zorder=10,
            )
            if (
                is_motion_frame and factor != 1.0
                and panel_t < panel_total - 1e-12
            ):
                _draw_speedup_indicator(ax, speed_str, frame_index)
        if title:
            fig.suptitle(title)
        return fig, axes_list

    if view == "detail":
        fig = plt.figure(figsize=DETAIL_FIGSIZE, constrained_layout=True)
        gs = fig.add_gridspec(
            2, 3, width_ratios=(2.5, 1.0, 1.0), height_ratios=(1.0, 1.0)
        )
        atom_ax = fig.add_subplot(gs[:, 0])
        row_now = fig.add_subplot(gs[0, 1])
        col_now = fig.add_subplot(gs[0, 2])
        row_hist = fig.add_subplot(gs[1, 1])
        col_hist = fig.add_subplot(gs[1, 2])
        ensemble = motion if isinstance(motion, AtomEnsemble) else motion.ensemble
        colors = _atom_colors_rgba(
            [int(a.atom_id) for a in ensemble.atomtrajs], atom_colors
        )
        draw_atom_panel(
            atom_ax, motion, t, quality=quality, atom_colors=atom_colors,
            planned_trajectory=planned_trajectory,
            show_atom_ids=show_atom_ids, show_routing=show_routing,
            show_color_code=show_color_code, color_code_swap=color_code_swap, start_overlay=start_overlay, end_overlay=end_overlay, is_start_hold=is_start_hold,
            title=None, atom_scale=atom_scale, trap_scale=trap_scale,
        )
        draw_current_tones(row_now, ensemble, t, "row", colors=colors)
        draw_current_tones(col_now, ensemble, t, "col", colors=colors)
        draw_tone_history(
            row_hist, ensemble, t, "row", mode=planned_trajectory,
            colors=colors, samples=style.tone_samples_per_segment,
        )
        draw_tone_history(
            col_hist, ensemble, t, "col", mode=planned_trajectory,
            colors=colors, samples=style.tone_samples_per_segment,
        )
        fig.suptitle(title or f"Atom rearrangement  -  t = {_format_time_us(t)}")
        return fig, [atom_ax, row_now, col_now, row_hist, col_hist]

    raise ValueError("view must be 'demo', 'benchmark', or 'detail'")


def save_frame(
    motion: Any,
    t: float,
    output_path: str | Path,
    **draw_kwargs: Any,
) -> Path:
    """Render one frame to a PNG file via `draw_frame`."""
    quality: RenderQuality = draw_kwargs.get("quality", "speed")
    dpi = _QUALITY[quality].dpi
    fig, _ = draw_frame(motion, t, **draw_kwargs)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, facecolor="white")
    plt.close(fig)
    return path


def render_animation(
    motion: Any,
    output_path: str | Path,
    *,
    view: RenderView = "demo",
    quality: RenderQuality = "speed",
    labels: Iterable[str] | None = None,
    fps: int | None = None,
    time_dilation: float = 1e4,
    hold_seconds: float = 1.0,
    atom_colors: Mapping[int, Any] | Iterable[Any] | None = None,
    planned_trajectory: PlannedTrajectoryMode = "none",
    show_atom_ids: bool = False,
    show_routing_on_start: bool = True,
    show_color_code: bool = False,
    show_color_code_on_start: bool = True,
    start_overlay: PanelOverlay | None = None,
    end_overlay: PanelOverlay | None = None,
    title: str | None = None,
    show_progress: bool = True,
    fmt: RenderFormat | None = None,
    atom_scale: float = 1.0,
    trap_scale: float = 1.0,
    panel_speedup: Mapping[str, float] | None = None,
) -> Path:
    """Render a PNG sequence by calling `draw_frame` per timestep, then stitch into an animation.

    Playback speed is controlled by exactly two parameters:
      - `time_dilation`: animation seconds per 1 second of execution time.
        e.g. `time_dilation=1e4` plays a 100 us run as a 1 s GIF.
      - `fps`: playback frame rate. Defaults to the quality preset
        (6 for "speed", 20 for "quality") when None.
    The physics-time step per frame is `1 / (time_dilation * fps)`.

    Output format is `fmt` if given, otherwise inferred from
    `output_path`'s extension (.gif, .webp, .apng/.png). WebP is lossless
    and typically the smallest; GIF output is post-processed with
    `gifsicle -O3` when that binary is on PATH. Start/end hold frames are
    encoded as a single frame with an extended per-frame duration instead
    of repeating identical frames.

    Frame rendering is parallelized across `os.cpu_count()` worker
    processes; stitching itself remains serial.
    """
    style = _QUALITY[quality]
    if fps is None:
        fps = style.fps
    if fps <= 0:
        raise ValueError("fps must be positive")
    if time_dilation <= 0:
        raise ValueError("time_dilation must be positive")
    if view not in ("demo", "benchmark", "detail"):
        raise ValueError("view must be 'demo', 'benchmark', or 'detail'")

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fmt = fmt or _infer_render_format(out)
    dpi = style.dpi

    if view == "benchmark":
        items_norm = _normalize_motion_items(motion, labels)
        if not items_norm:
            raise ValueError("benchmark view needs at least one motion object")
        total = max(
            (
                (m if isinstance(m, AtomEnsemble) else m.ensemble).total_duration()
                / (float(panel_speedup.get(label, 1.0)) if panel_speedup else 1.0)
                for label, m in items_norm
            ),
            default=0.0,
        )
    else:
        ensemble = motion if isinstance(motion, AtomEnsemble) else motion.ensemble
        total = ensemble.total_duration()

    frame_dt = 1.0 / (time_dilation * fps)
    moving_times = (
        np.array([0.0], dtype=float)
        if total <= 0
        else np.linspace(
            0.0, total, max(2, int(math.ceil(total / frame_dt)) + 1), dtype=float
        )
    )

    frame_dur_ms = max(1, int(round(1000.0 / fps)))
    hold_ms = max(0, int(round(hold_seconds * 1000.0)))
    has_routing_frame = bool(show_routing_on_start and hold_ms > 0)
    has_start_overlay = bool(
        show_color_code and show_color_code_on_start and hold_ms > 0
    )
    needs_real_start_t = (
        has_routing_frame or has_start_overlay or start_overlay is not None
    )

    # Plan tuple: (t, show_routing, show_color_code, color_code_swap, is_start_hold).
    # Start hold draws the un-swapped overlay on every panel (the
    # initial code state). During motion `color_code_swap=True` lets
    # each panel pick up the swapped overlay independently the moment
    # its own ensemble finishes; the last moving frame thus already
    # has every panel finished and overlaid, so the trailing hold is
    # just an extended duration on that frame.
    plan: list[tuple[float, bool, bool, bool, bool]] = []
    durations: list[int] = []
    if hold_ms > 0:
        # If nothing on the start hold needs a "real" render, dip below 0 so
        # no segments are active and `draw_traps` finds nothing to draw.
        start_t = float(moving_times[0]) if needs_real_start_t else -1e-9
        plan.append((start_t, has_routing_frame, has_start_overlay, False, True))
        durations.append(hold_ms)
    for t in moving_times:
        plan.append((float(t), False, show_color_code, True, False))
        durations.append(frame_dur_ms)
    if hold_ms > 0:
        durations[-1] += hold_ms

    draw_kwargs = dict(
        view=view, quality=quality,
        labels=list(labels) if labels is not None else None,
        atom_colors=atom_colors,
        planned_trajectory=planned_trajectory,
        show_atom_ids=show_atom_ids, title=title,
        atom_scale=atom_scale, trap_scale=trap_scale,
        panel_speedup=dict(panel_speedup) if panel_speedup is not None else None,
        start_overlay=start_overlay,
        end_overlay=end_overlay,
    )

    items: list[tuple[int, float, bool, bool, bool, bool]] = [
        (k, t, sr, scc, swap, ish)
        for k, (t, sr, scc, swap, ish) in enumerate(plan)
    ]
    n_workers = min(os.cpu_count() or 1, len(items))

    with tempfile.TemporaryDirectory(prefix="aod_vs_ripa_frames_") as tmp:
        tmp_path = Path(tmp)
        chunksize = max(1, len(items) // (n_workers * 4))
        with ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=_frame_worker_init,
            initargs=(motion, draw_kwargs, dpi, str(tmp_path)),
        ) as ex:
            results = ex.map(_frame_worker_render, items, chunksize=chunksize)
            frame_paths = [
                Path(p) for p in _progress_iter(
                    results, total=len(items), desc="render frames",
                    enabled=show_progress,
                )
            ]
        _save_animation(
            frame_paths, out, fmt=fmt,
            durations=durations, show_progress=show_progress,
        )
    return out



# --- color helpers ---------------------------------------------------------


def _default_colors(ensemble: AtomEnsemble) -> list[Any]:
    """Compute the default tab20-cycled RGBA color list for an ensemble's atoms."""
    return _atom_colors_rgba([int(a.atom_id) for a in ensemble.atomtrajs], None)


def _atom_colors_rgba(
    atom_ids: list[int],
    atom_colors: Mapping[int, Any] | Iterable[Any] | None,
) -> list[Any]:
    """Pick one RGBA color per atom, defaulting to a tab20 cycle."""
    n = len(atom_ids)
    cmap = plt.get_cmap("tab20")
    defaults = [to_rgba(cmap((k % 20) / 19.0)) for k in range(n)]

    if atom_colors is None:
        return defaults
    if isinstance(atom_colors, Mapping):
        return [
            to_rgba(atom_colors.get(atom_id, defaults[k]))
            for k, atom_id in enumerate(atom_ids)
        ]

    colors = [to_rgba(c) for c in atom_colors]
    if len(colors) != n:
        raise ValueError(f"atom_colors length {len(colors)} != atom count {n}")
    return colors


def _darken_color(color: Any) -> Any:
    """Return a 55%-luminosity version of the given color, used for label text."""
    r, g, b, a = to_rgba(color)
    return (0.55 * r, 0.55 * g, 0.55 * b, a)


# --- geometry / segment helpers --------------------------------------------


def _active_segment(
    atom: AtomTrajectory, t: float, *, tol: float = 1e-12,
) -> Segment | None:
    """Return the atom's currently-moving segment at time `t`, or None at rest."""
    for seg in atom.segments:
        if seg.duration > 0 and seg.start_time - tol <= t <= seg.end_time + tol:
            return seg
    return None


def _future_segments(
    atom: AtomTrajectory, t: float, *, tol: float = 1e-12,
) -> Iterable[Segment]:
    """Yield the atom's segments that have not finished yet at time `t`."""
    for seg in atom.segments:
        if seg.duration > 0 and seg.end_time > t + tol:
            yield seg


def _active_aod_traps_xy(
    steps: list[Any],
    ensemble: AtomEnsemble,
    t: float,
    *,
    tol: float = 1e-12,
) -> np.ndarray:
    """Positions of every active AOD trap, including empty intersections."""
    traps: list[tuple[float, float]] = []
    for step in steps:
        selected_axis_1 = getattr(step, "selected_axis_1", None)
        selected_axis_2 = getattr(step, "selected_axis_2", None)
        new_axis_1 = getattr(step, "new_axis_1", None)
        new_axis_2 = getattr(step, "new_axis_2", None)
        start_time = getattr(step, "start_time", None)
        if (
            start_time is None
            or selected_axis_1 is None
            or selected_axis_2 is None
            or new_axis_1 is None
            or new_axis_2 is None
        ):
            continue
        try:
            end_time = step.end_time(ensemble)
        except Exception:
            continue
        duration = float(end_time - start_time)
        if duration <= 0 or not (start_time - tol <= t < end_time - tol):
            continue
        s = max(0.0, min(1.0, float(t - start_time) / duration))
        u = float(BANG_BANG.value(s))
        coords_1 = [
            float(old) + u * (float(new) - float(old))
            for old, new in zip(selected_axis_1, new_axis_1)
        ]
        coords_2 = [
            float(old) + u * (float(new) - float(old))
            for old, new in zip(selected_axis_2, new_axis_2)
        ]
        axis_1 = getattr(step, "axis_1", (1.0, 0.0))
        axis_2 = getattr(step, "axis_2", (0.0, 1.0))
        origin = getattr(step, "origin", (0.0, 0.0))
        traps.extend(
            (
                origin[0] + coord_1 * axis_1[0] + coord_2 * axis_2[0],
                origin[1] + coord_1 * axis_1[1] + coord_2 * axis_2[1],
            )
            for coord_1 in coords_1
            for coord_2 in coords_2
        )

    if not traps:
        return np.empty((0, 2), dtype=float)
    return ensemble.grid.ij_to_xy(np.asarray(traps, dtype=float))


def _sample_segment_xy(
    grid: Grid, seg: Segment, t0: float, t1: float, samples: int,
) -> np.ndarray:
    """Return `samples` xy positions along `seg` between `t0` and `t1`."""
    if t1 <= t0:
        return np.empty((0, 2), dtype=float)
    ts = np.linspace(t0, t1, max(2, int(samples)))
    ij = np.asarray([seg.position_at(float(t)) for t in ts], dtype=float)
    return grid.ij_to_xy(ij)


def _motion_blur_xy(grid: Grid, seg: Segment, t: float, samples: int) -> np.ndarray:
    """Speed-scaled tail ending at the atom's current position."""
    t_now = min(max(float(t), seg.start_time), seg.end_time)
    if t_now <= seg.start_time or seg.duration <= 0:
        return np.empty((0, 2), dtype=float)

    total_distance = math.hypot(
        seg.end_pos[0] - seg.start_pos[0],
        seg.end_pos[1] - seg.start_pos[1],
    )
    if total_distance <= 0:
        return np.empty((0, 2), dtype=float)

    speed = _segment_speed_grid(seg, t_now)
    if speed <= 1e-12:
        return np.empty((0, 2), dtype=float)

    shutter = 0.18 * seg.duration
    current_fraction = seg.path_fraction_at(t_now)
    current_distance = current_fraction * total_distance
    blur_distance = min(speed * shutter, current_distance, 0.45 * total_distance)
    if blur_distance <= 1e-12:
        return np.empty((0, 2), dtype=float)

    start_fraction = max(0.0, (current_distance - blur_distance) / total_distance)
    t0 = _time_for_path_fraction(seg, start_fraction, seg.start_time, t_now)
    return _sample_segment_xy(grid, seg, t0, t_now, samples)


def _segment_speed_grid(seg: Segment, t: float) -> float:
    """Instantaneous speed in grid units per second from the segment's analytic velocity."""
    vx, vy = seg.velocity_at(t)
    return math.hypot(vx, vy)


def _time_for_path_fraction(
    seg: Segment, fraction: float, lo: float, hi: float,
) -> float:
    """Bisect within [lo, hi] to find the time at which the segment reaches `fraction`."""
    fraction = min(max(float(fraction), 0.0), 1.0)
    if fraction <= 0.0:
        return seg.start_time
    if fraction >= seg.path_fraction_at(hi):
        return hi
    left = max(seg.start_time, lo)
    right = min(seg.end_time, hi)
    for _ in range(48):
        mid = (left + right) / 2.0
        if seg.path_fraction_at(mid) < fraction:
            left = mid
        else:
            right = mid
    return (left + right) / 2.0


def _channels_for_segment(seg: Segment) -> tuple[ToneChannel, ...]:
    """Return which tone channels (row, col, or both) the segment drives."""
    if seg.channel == "row":
        return ("row",)
    if seg.channel == "col":
        return ("col",)
    if seg.channel == "aod":
        return ("row", "col")
    di = seg.end_pos[0] - seg.start_pos[0]
    dj = seg.end_pos[1] - seg.start_pos[1]
    if dj == 0 and di != 0:
        return ("row",)
    if di == 0 and dj != 0:
        return ("col",)
    return ("row", "col")


def _tone_hardware_name(ensemble: AtomEnsemble) -> ToneHardware:
    """Classify the ensemble's hardware as 'AOD', 'EOM', 'AOD/EOM', or 'tone'."""
    hardware: set[ToneHardware] = set()
    for atom in ensemble.atomtrajs:
        for seg in atom.segments:
            if seg.duration <= 0:
                continue
            if seg.channel == "aod":
                hardware.add("AOD")
            elif seg.channel in ("row", "col"):
                hardware.add("EOM")
    if hardware == {"AOD"}:
        return "AOD"
    if hardware == {"EOM"}:
        return "EOM"
    if hardware == {"AOD", "EOM"}:
        return "AOD/EOM"
    return "tone"


# --- tone frequencies (normalized to [0, 1)) -------------------------------


def _ripa_tone_freq(channel: ToneChannel, i: float, j: float, N: int) -> float:
    """Normalized RIPA EOM tone in [0, 1).

    With FSR2 = FSR1 / N, the row-channel frequency `(j + i/N) * FSR2 mod
    FSR1` divided by FSR1 reduces to `(j + i/N) / N mod 1`. Col is the
    same with i and j swapped.
    """
    if channel == "row":
        return float(((j + i / N) / N) % 1.0)
    return float(((i + j / N) / N) % 1.0)


def _aod_tone_freq(channel: ToneChannel, i: float, j: float, N: int) -> float:
    """Normalized AOD tone in [0, 1) — one separable RF tone per row/column."""
    coord = i if channel == "row" else j
    return float((coord / N) % 1.0)


def _tone_freq(
    channel: ToneChannel, i: float, j: float, N: int, *, segment: Segment,
) -> float:
    """Normalized tone in [0, 1) for the segment's hardware (AOD or RIPA EOM)."""
    if segment.channel == "aod":
        return _aod_tone_freq(channel, i, j, N)
    return _ripa_tone_freq(channel, i, j, N)


def _collect_tone_trajectories(
    ensemble: AtomEnsemble, channel: ToneChannel, samples_per_segment: int,
) -> list[dict[str, Any]]:
    """Sample each segment to build tone-vs-time trajectories for one channel."""
    trajs: list[dict[str, Any]] = []
    N = ensemble.grid.N
    for idx, atom in enumerate(ensemble.atomtrajs):
        for seg in atom.segments:
            if seg.duration <= 0:
                continue
            if channel not in _channels_for_segment(seg):
                continue
            ts = np.linspace(seg.start_time, seg.end_time, samples_per_segment)
            ij = np.asarray([seg.position_at(float(t)) for t in ts], dtype=float)
            trajs.append(
                {
                    "idx": idx,
                    "atom_id": int(atom.atom_id),
                    "times": ts,
                    "start_time": float(seg.start_time),
                    "end_time": float(seg.end_time),
                    "freqs": np.asarray(
                        [_tone_freq(channel, i, j, N, segment=seg) for i, j in ij],
                        dtype=float,
                    ),
                }
            )
    return trajs


def _next_tone_trajectory_by_atom(
    trajectories: list[dict[str, Any]], t: float,
) -> dict[int, dict[str, Any]]:
    """Return each atom's earliest tone trajectory that ends after `t`."""
    selected: dict[int, dict[str, Any]] = {}
    for traj in trajectories:
        idx = int(traj["idx"])
        if idx in selected:
            continue
        if float(traj["end_time"]) > t + 1e-15:
            selected[idx] = traj
    return selected


# --- assorted small helpers ------------------------------------------------


def _draw_gaussian_blob(
    ax: Any,
    x: float,
    y: float,
    *,
    trap_scale: float = 1.0,
    rgba: tuple[float, float, float, float] | None = None,
) -> None:
    """Draw a soft Gaussian halo at (x, y) using BLOB_SIGMA, under the atom
    layer. `rgba` overrides the default red trap-outline tint; its alpha
    is multiplied by the Gaussian so callers pass the peak alpha they want
    at the center."""
    sigma = BLOB_SIGMA * trap_scale
    r = 3.0 * sigma
    vals = np.linspace(-r, r, 21)
    xx, yy = np.meshgrid(vals, vals)
    zz = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
    if rgba is None:
        rgba = (0.839, 0.153, 0.157, 0.55)  # #d62728 — match trap-outline red
    rgba_arr = np.empty((*zz.shape, 4), dtype=float)
    rgba_arr[..., 0] = rgba[0]
    rgba_arr[..., 1] = rgba[1]
    rgba_arr[..., 2] = rgba[2]
    rgba_arr[..., 3] = rgba[3] * zz
    ax.imshow(
        rgba_arr,
        extent=(x - r, x + r, y - r, y + r),
        origin="lower",
        interpolation="bilinear",
        zorder=Z_TRAP_BLOB,
    )


def _plot_wrapped_freq_line(
    ax: Any, xs: np.ndarray, ys: np.ndarray, fsr: float, **kw: Any,
) -> None:
    """Plot modulo-wrapped frequencies without drawing the wrap jumps as lines."""
    if len(xs) < 2:
        return
    jump = abs(float(fsr)) / 2.0
    breaks = np.flatnonzero(np.abs(np.diff(ys)) > jump) + 1
    starts = np.r_[0, breaks]
    ends = np.r_[breaks, len(xs)]
    for s, e in zip(starts, ends):
        if e - s >= 2:
            ax.plot(xs[s:e], ys[s:e], **kw)


def _normalize_motion_items(
    motions: Mapping[str, Any] | Iterable[Any],
    labels: Iterable[str] | None,
) -> list[tuple[str, Any]]:
    """Pair each benchmark motion with its label, defaulting to 'scheduler N' names."""
    if isinstance(motions, Mapping):
        items = [(str(label), motion) for label, motion in motions.items()]
        if labels is None:
            return items
        label_list = [str(label) for label in labels]
        if len(label_list) != len(items):
            raise ValueError("labels length must match benchmark motion count")
        return [(label, motion) for label, (_, motion) in zip(label_list, items)]

    motion_list = list(motions)
    if labels is None:
        label_list = [f"scheduler {index + 1}" for index in range(len(motion_list))]
    else:
        label_list = [str(label) for label in labels]
        if len(label_list) != len(motion_list):
            raise ValueError("labels length must match benchmark motion count")
    return list(zip(label_list, motion_list))


def _format_time_us(t: float) -> str:
    """Format a time in seconds as a microsecond string with adaptive precision."""
    value = t * TIME_FACTOR_US
    if abs(value) >= 100:
        return f"{value:.1f} us"
    if abs(value) >= 10:
        return f"{value:.2f} us"
    return f"{value:.3f} us"


_FRAME_WORKER: dict[str, Any] = {}


def _frame_worker_init(
    motion: Any,
    draw_kwargs: dict[str, Any],
    dpi: int,
    tmp_path_str: str,
) -> None:
    _FRAME_WORKER["motion"] = motion
    _FRAME_WORKER["draw_kwargs"] = draw_kwargs
    _FRAME_WORKER["dpi"] = dpi
    _FRAME_WORKER["tmp_path"] = Path(tmp_path_str)


def _frame_worker_render(item: tuple[int, float, bool, bool, bool, bool]) -> str:
    k, t, show_routing, show_color_code, color_code_swap, is_start_hold = item
    # Plan invariant in `render_animation`: motion frames carry
    # color_code_swap=True; the start-hold frame carries False AND
    # is_start_hold=True. End-of-run idleness is implicit via panel_t
    # saturating at panel_total.
    fig, _ = draw_frame(
        _FRAME_WORKER["motion"], float(t),
        show_routing=show_routing, show_color_code=show_color_code,
        color_code_swap=color_code_swap, is_start_hold=is_start_hold,
        frame_index=int(k),
        is_motion_frame=bool(color_code_swap),
        **_FRAME_WORKER["draw_kwargs"],
    )
    path = _FRAME_WORKER["tmp_path"] / f"frame_{k:05d}.png"
    fig.savefig(path, dpi=_FRAME_WORKER["dpi"], facecolor="white")
    plt.close(fig)
    return str(path)


def _progress_iter(
    items: Iterable[Any], *, total: int, desc: str, enabled: bool,
) -> Iterable[Any]:
    """Wrap an iterable with a tqdm progress bar when `enabled`, else pass through."""
    if not enabled:
        yield from items
        return
    yield from tqdm(items, total=total, desc=desc, unit="frame")


def _infer_render_format(path: Path) -> RenderFormat:
    """Map a file extension to a `RenderFormat`."""
    ext = path.suffix.lower()
    if ext == ".gif":
        return "gif"
    if ext == ".webp":
        return "webp"
    if ext in (".apng", ".png"):
        return "apng"
    raise ValueError(
        f"cannot infer animation format from extension {ext!r}; "
        "pass fmt='gif' | 'webp' | 'apng'"
    )


def _save_animation(
    frame_paths: list[Path],
    output_path: Path,
    *,
    fmt: RenderFormat,
    durations: list[int],
    show_progress: bool,
) -> None:
    """Stitch PNG frames into an animation in `fmt`.

    GIF goes through a shared 128-color palette (from the middle frame) so
    `optimize=True` can encode inter-frame diffs, then `gifsicle -O3` if
    available. WebP/APNG keep RGBA and rely on the codec.
    """
    if not frame_paths:
        raise ValueError("no frames to save")
    if len(durations) != len(frame_paths):
        raise ValueError("durations length must match frame_paths length")

    palette: Image.Image | None = None
    if fmt == "gif":
        with Image.open(frame_paths[len(frame_paths) // 2]) as ref:
            palette = ref.convert("RGB").quantize(
                colors=128,
                method=Image.Quantize.MEDIANCUT,
                dither=Image.Dither.NONE,
            )
        convert = lambda raw: raw.convert("RGB").quantize(
            palette=palette, dither=Image.Dither.NONE,
        )
        save_kwargs: dict[str, Any] = dict(loop=0, optimize=True)
    elif fmt == "webp":
        convert = lambda raw: raw.convert("RGBA")
        save_kwargs = dict(
            format="WebP", loop=0, lossless=True, method=6, minimize_size=True,
        )
    elif fmt == "apng":
        convert = lambda raw: raw.convert("RGBA")
        save_kwargs = dict(format="PNG", loop=0)
    else:
        raise ValueError(f"unsupported fmt {fmt!r}")

    images: list[Image.Image] = []
    for path in _progress_iter(
        frame_paths, total=len(frame_paths),
        desc=f"load {fmt} frames", enabled=show_progress,
    ):
        with Image.open(path) as raw:
            images.append(convert(raw))
    try:
        images[0].save(
            output_path,
            save_all=True,
            append_images=images[1:],
            duration=durations,
            **save_kwargs,
        )
    finally:
        for image in images:
            image.close()
        if palette is not None:
            palette.close()

    if fmt == "gif":
        _try_gifsicle_optimize(output_path)


def _try_gifsicle_optimize(path: Path) -> None:
    """Run `gifsicle -O3` on `path` to shrink it in place when available.

    Silently skipped if gifsicle is not on PATH; subprocess failure leaves
    the original file untouched.
    """
    if shutil.which("gifsicle") is None:
        return
    tmp = path.with_suffix(path.suffix + ".opt.tmp")
    try:
        subprocess.run(
            ["gifsicle", "-O3", str(path), "-o", str(tmp)],
            check=True,
            capture_output=True,
        )
        tmp.replace(path)
    except subprocess.CalledProcessError:
        if tmp.exists():
            tmp.unlink()
