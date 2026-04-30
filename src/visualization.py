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
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import to_rgba  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
from tqdm.auto import tqdm  # noqa: E402

from .atom_config import Grid  # noqa: E402
from .atom_trajectory import AtomEnsemble, AtomTrajectory  # noqa: E402
from .segments import Segment  # noqa: E402

Site = tuple[int, int]
ToneChannel = Literal["row", "col"]
ToneHardware = Literal["AOD", "EOM", "AOD/EOM", "tone"]
PlannedTrajectoryMode = Literal["none", "full", "next"]
AddressedStyle = Literal["edge", "blob", "both", "none"]
RenderView = Literal["demo", "benchmark", "detail"]
RenderQuality = Literal["speed", "quality"]

ATOM_SIZE = 70.0
TRAP_SIZE = 12.0
TIME_FACTOR_US = 1e6
FREQ_LABEL = "nu / FSR1 (mod 1)"

DEMO_FIGSIZE = (5.4, 5.2)
DETAIL_FIGSIZE = (12.0, 6.2)
BENCH_HEIGHT = 4.6
BENCH_WIDTH_PER_PANEL = 4.0


@dataclass(frozen=True)
class _Style:
    dpi: int
    addressed_style: AddressedStyle
    show_motion_blur: bool
    trail_samples: int
    tone_samples_per_segment: int
    planned_samples_per_segment: int


_QUALITY: dict[RenderQuality, _Style] = {
    "speed": _Style(
        dpi=90,
        addressed_style="edge",
        show_motion_blur=False,
        trail_samples=0,
        tone_samples_per_segment=32,
        planned_samples_per_segment=18,
    ),
    "quality": _Style(
        dpi=150,
        addressed_style="both",
        show_motion_blur=True,
        trail_samples=12,
        tone_samples_per_segment=96,
        planned_samples_per_segment=48,
    ),
}


# --- draw primitives -------------------------------------------------------


def draw_grid_dots(ax: Any, grid: Grid) -> None:
    """Draw the gray underlying lattice dot at every grid site."""
    xy = grid.ij_to_xy(_all_grid_sites(grid.N))
    ax.scatter(
        xy[:, 0], xy[:, 1], s=TRAP_SIZE, c="#9a9a9a",
        alpha=0.45, linewidths=0, zorder=1,
    )


def draw_static_traps(ax: Any, grid: Grid, sites: Iterable[Site]) -> None:
    """Draw a blue circle marker at each listed static-trap site."""
    arr = np.asarray(list(sites), dtype=float).reshape(-1, 2)
    if not len(arr):
        return
    xy = grid.ij_to_xy(arr)
    ax.scatter(
        xy[:, 0], xy[:, 1], s=TRAP_SIZE * 2.3,
        facecolors="none", edgecolors="#4b8bbe",
        linewidths=0.8, alpha=0.55, zorder=2,
    )


def draw_grid_frame(ax: Any, grid: Grid) -> None:
    """Draw the dashed grey rectangle that frames the grid extent and set axis limits."""
    lo, hi = _site_extent_um(grid)
    b0 = lo - grid.d / 2.0
    b1 = hi + grid.d / 2.0
    width = b1 - b0
    ax.add_patch(
        Rectangle(
            (b0, b0), width, width, fill=False, linestyle="--",
            linewidth=1.0, edgecolor="#bbbbbb", zorder=1.5,
        )
    )
    ax.set_xlim(lo - grid.d, hi + grid.d)
    ax.set_ylim(lo - grid.d, hi + grid.d)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (um)")
    ax.set_ylabel("y (um)")


def draw_aod_traps(ax: Any, sequence: Any, t: float) -> None:
    """Draw red square markers at every active AOD trap (no-op when not a Sequence)."""
    ensemble = getattr(sequence, "ensemble", None)
    steps = getattr(sequence, "steps", None)
    if ensemble is None or not steps:
        return
    xy = _active_aod_traps_xy(steps, ensemble, t)
    if not len(xy):
        return
    ax.scatter(
        xy[:, 0], xy[:, 1], s=TRAP_SIZE * 5.0, marker="s",
        facecolors="none", edgecolors="#d62728",
        linewidths=1.1, alpha=0.70, zorder=3.6,
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
                    color=colors[idx], alpha=0.34, zorder=2.5,
                )


def draw_motion_blur(
    ax: Any,
    ensemble: AtomEnsemble,
    t: float,
    *,
    samples: int = 12,
    colors: list[Any] | None = None,
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
        ax.plot(xy[:, 0], xy[:, 1], color=color, linewidth=4.0, alpha=0.20, zorder=3)
        fade = np.linspace(0.08, 0.32, len(xy))
        sizes = np.linspace(ATOM_SIZE * 0.15, ATOM_SIZE * 0.55, len(xy))
        for (x, y), alpha, size in zip(xy, fade, sizes):
            ax.scatter(
                [x], [y], s=size, c=[color], alpha=float(alpha),
                linewidths=0, zorder=3.2,
            )


def draw_atoms(
    ax: Any,
    ensemble: AtomEnsemble,
    t: float,
    *,
    colors: list[Any] | None = None,
    addressed_style: AddressedStyle = "edge",
    show_atom_ids: bool = False,
) -> None:
    """Draw atoms at their positions at time `t`, highlighting those currently moving."""
    colors = colors or _default_colors(ensemble)
    grid = ensemble.grid
    positions_xy = grid.ij_to_xy(ensemble.positions_at(t))
    n = len(positions_xy)
    active = np.zeros(n, dtype=bool)
    for idx, atom in enumerate(ensemble.atomtrajs):
        if _active_segment(atom, t) is not None:
            active[idx] = True

    if addressed_style in ("blob", "both"):
        sigma = max(grid.d * 0.22, 1e-9)
        for idx in np.flatnonzero(active):
            x, y = positions_xy[idx]
            _draw_gaussian_blob(ax, float(x), float(y), sigma)

    edgecolors = [
        "#d62728" if a and addressed_style in ("edge", "both") else "#ffffff"
        for a in active
    ]
    linewidths = [
        2.3 if a and addressed_style in ("edge", "both") else 0.8 for a in active
    ]
    ax.scatter(
        positions_xy[:, 0], positions_xy[:, 1], s=ATOM_SIZE,
        c=colors, edgecolors=edgecolors, linewidths=linewidths, zorder=5,
    )

    if show_atom_ids:
        offset = 0.22 * grid.d
        for atom, (x, y), color in zip(ensemble.atomtrajs, positions_xy, colors):
            ax.text(
                x, y + offset, str(int(atom.atom_id)),
                ha="center", va="bottom", fontsize=7,
                color=_darken_color(color), zorder=6,
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
    ax.set_title(_tone_panel_title(channel, _tone_hardware_name(ensemble)))
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
    static_traps: Iterable[Site] | None = None,
    planned_trajectory: PlannedTrajectoryMode = "full",
    show_atom_ids: bool = False,
    title: str | None = None,
) -> None:
    """Compose one atom-plane panel by calling the seven visual primitives in z-order."""
    style = _QUALITY[quality]
    if isinstance(motion, AtomEnsemble):
        ensemble, sequence = motion, None
    else:
        ensemble, sequence = motion.ensemble, motion
    colors = _atom_colors_rgba(
        [int(a.atom_id) for a in ensemble.atomtrajs], atom_colors
    )

    draw_grid_dots(ax, ensemble.grid)
    if static_traps is not None:
        draw_static_traps(ax, ensemble.grid, static_traps)
    draw_planned_paths(
        ax, ensemble, t, mode=planned_trajectory, colors=colors,
        samples=style.planned_samples_per_segment,
    )
    if style.show_motion_blur:
        draw_motion_blur(
            ax, ensemble, t, samples=style.trail_samples, colors=colors
        )
    draw_atoms(
        ax, ensemble, t, colors=colors,
        addressed_style=style.addressed_style, show_atom_ids=show_atom_ids,
    )
    if sequence is not None:
        draw_aod_traps(ax, sequence, t)
    draw_grid_frame(ax, ensemble.grid)
    ax.set_title(title or f"atom motion  -  t = {_format_time_us(t)}")


# --- frame composer + IO ---------------------------------------------------


def draw_frame(
    motion: Any,
    t: float,
    *,
    view: RenderView = "demo",
    quality: RenderQuality = "speed",
    labels: Iterable[str] | None = None,
    atom_colors: Mapping[int, Any] | Iterable[Any] | None = None,
    static_traps: Iterable[Site] | None = None,
    planned_trajectory: PlannedTrajectoryMode = "full",
    show_atom_ids: bool = False,
    title: str | None = None,
) -> tuple[Any, list[Any]]:
    """Build a fresh figure for `view` and draw the primitives on it; return (fig, axes)."""
    style = _QUALITY[quality]

    if view == "demo":
        fig, ax = plt.subplots(figsize=DEMO_FIGSIZE, constrained_layout=True)
        draw_atom_panel(
            ax, motion, t, quality=quality, atom_colors=atom_colors,
            static_traps=static_traps, planned_trajectory=planned_trajectory,
            show_atom_ids=show_atom_ids, title=title,
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
            draw_atom_panel(
                ax, panel_motion, t, quality=quality, atom_colors=atom_colors,
                static_traps=static_traps, planned_trajectory=planned_trajectory,
                show_atom_ids=show_atom_ids,
                title=f"{label}  -  t = {_format_time_us(t)}",
            )
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
            static_traps=static_traps, planned_trajectory=planned_trajectory,
            show_atom_ids=show_atom_ids, title=None,
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
    fps: int = 20,
    frame_dt: float = 10e-6,
    hold_seconds: float = 1.0,
    atom_colors: Mapping[int, Any] | Iterable[Any] | None = None,
    static_traps: Iterable[Site] | None = None,
    planned_trajectory: PlannedTrajectoryMode = "full",
    show_atom_ids: bool = False,
    title: str | None = None,
    show_progress: bool = True,
) -> Path:
    """Render a PNG sequence by calling `draw_frame` per timestep, then stitch into a GIF.

    The frame count is `ceil(total_duration / frame_dt) + 1`. `fps` controls
    GIF playback only; the physics timeline is visualized, not played in
    real time.
    """
    if fps <= 0:
        raise ValueError("fps must be positive")
    if frame_dt <= 0:
        raise ValueError("frame_dt must be positive")
    if view not in ("demo", "benchmark", "detail"):
        raise ValueError("view must be 'demo', 'benchmark', or 'detail'")

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    dpi = _QUALITY[quality].dpi

    if view == "benchmark":
        items = _normalize_motion_items(motion, labels)
        if not items:
            raise ValueError("benchmark view needs at least one motion object")
        total = max(
            (
                (m if isinstance(m, AtomEnsemble) else m.ensemble).total_duration()
                for _, m in items
            ),
            default=0.0,
        )
    else:
        ensemble = motion if isinstance(motion, AtomEnsemble) else motion.ensemble
        total = ensemble.total_duration()

    moving_times = (
        np.array([0.0], dtype=float)
        if total <= 0
        else np.linspace(
            0.0, total, max(2, int(math.ceil(total / frame_dt)) + 1), dtype=float
        )
    )
    hold_count = max(0, int(round(hold_seconds * fps)))
    schedule = np.concatenate(
        [
            np.full(hold_count, moving_times[0], dtype=float),
            moving_times,
            np.full(hold_count, moving_times[-1], dtype=float),
        ]
    )

    draw_kwargs = dict(
        view=view, quality=quality,
        labels=list(labels) if labels is not None else None,
        atom_colors=atom_colors, static_traps=static_traps,
        planned_trajectory=planned_trajectory,
        show_atom_ids=show_atom_ids, title=title,
    )

    with tempfile.TemporaryDirectory(prefix="aod_vs_ripa_frames_") as tmp:
        tmp_path = Path(tmp)
        frame_paths: list[Path] = []
        for k, t in enumerate(
            _progress_iter(
                schedule, total=len(schedule), desc="render frames",
                enabled=show_progress,
            )
        ):
            fig, _ = draw_frame(motion, float(t), **draw_kwargs)
            path = tmp_path / f"frame_{k:05d}.png"
            fig.savefig(path, dpi=dpi, facecolor="white")
            plt.close(fig)
            frame_paths.append(path)
        _save_gif_from_pngs(frame_paths, out, fps=fps, show_progress=show_progress)
    return out


def render_check_outputs(
    motion: Any,
    output_dir: str | Path,
    prefix: str,
    *,
    render_gif: bool = True,
    gif_fps: int = 8,
    gif_frame_dt: float = 10e-6,
    gif_hold_seconds: float = 1.0,
    title_prefix: str | None = None,
    **visual_kwargs: Any,
) -> dict[str, Path]:
    """Render the standard t=0, t=final, and optional GIF check outputs."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    visual_kwargs = {"view": "demo", **visual_kwargs}
    if visual_kwargs.get("labels") is not None:
        visual_kwargs["labels"] = list(visual_kwargs["labels"])
    view = visual_kwargs["view"]

    if view == "benchmark":
        items = _normalize_motion_items(motion, visual_kwargs.get("labels"))
        final_t = max(
            (
                (m if isinstance(m, AtomEnsemble) else m.ensemble).total_duration()
                for _, m in items
            ),
            default=0.0,
        )
    else:
        ensemble = motion if isinstance(motion, AtomEnsemble) else motion.ensemble
        final_t = ensemble.total_duration()
    label = title_prefix or prefix

    paths = {
        "t0": out_dir / f"{prefix}_t0.png",
        "tfinal": out_dir / f"{prefix}_tfinal.png",
    }
    animation_only = {"show_progress"}
    frame_kwargs = {k: v for k, v in visual_kwargs.items() if k not in animation_only}

    save_frame(
        motion, 0.0, paths["t0"],
        **{**frame_kwargs, "title": frame_kwargs.get("title", f"{label} - t=0")},
    )
    save_frame(
        motion, final_t, paths["tfinal"],
        **{**frame_kwargs, "title": frame_kwargs.get("title", f"{label} - final")},
    )

    if render_gif:
        paths["gif"] = out_dir / f"{prefix}.gif"
        anim_kwargs = {k: v for k, v in visual_kwargs.items() if k != "title"}
        render_animation(
            motion, paths["gif"], fps=gif_fps,
            frame_dt=gif_frame_dt, hold_seconds=gif_hold_seconds,
            **anim_kwargs,
        )
    return paths


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


def _all_grid_sites(N: int) -> np.ndarray:
    """Return every (i, j) site of an N x N grid as a flat (N*N, 2) float array."""
    ii, jj = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    return np.column_stack([ii.ravel(), jj.ravel()]).astype(float)


def _site_extent_um(grid: Grid) -> tuple[float, float]:
    """Return (lo, hi) micrometer coordinates of the grid's extreme sites."""
    lo = (0.0 - grid.center) * grid.d
    hi = ((grid.N - 1.0) - grid.center) * grid.d
    return float(lo), float(hi)


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
        selected_rows = getattr(step, "selected_rows", None)
        selected_cols = getattr(step, "selected_cols", None)
        new_rows = getattr(step, "new_rows", None)
        new_cols = getattr(step, "new_cols", None)
        start_time = getattr(step, "start_time", None)
        if (
            selected_rows is None
            or selected_cols is None
            or new_rows is None
            or new_cols is None
            or start_time is None
        ):
            continue
        try:
            end_time = step.end_time(ensemble)
        except Exception:
            continue
        duration = float(end_time - start_time)
        if duration <= 0 or not (start_time - tol <= t < end_time - tol):
            continue
        u = _bang_bang_fraction(float(t - start_time), duration)
        rows = [
            float(old) + u * (float(new) - float(old))
            for old, new in zip(selected_rows, new_rows)
        ]
        cols = [
            float(old) + u * (float(new) - float(old))
            for old, new in zip(selected_cols, new_cols)
        ]
        traps.extend((i, j) for i in rows for j in cols)

    if not traps:
        return np.empty((0, 2), dtype=float)
    return ensemble.grid.ij_to_xy(np.asarray(traps, dtype=float))


def _bang_bang_fraction(t_local: float, duration: float) -> float:
    """Path fraction in [0, 1] for a symmetric bang-bang move at local time `t_local`."""
    if duration <= 0:
        return 1.0
    t = min(max(float(t_local), 0.0), float(duration))
    half = duration / 2.0
    if t <= half:
        return 2.0 * (t / duration) ** 2
    return 1.0 - 2.0 * ((duration - t) / duration) ** 2


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
    current_fraction = _segment_path_fraction(seg, t_now)
    current_distance = current_fraction * total_distance
    blur_distance = min(speed * shutter, current_distance, 0.45 * total_distance)
    if blur_distance <= 1e-12:
        return np.empty((0, 2), dtype=float)

    start_fraction = max(0.0, (current_distance - blur_distance) / total_distance)
    t0 = _time_for_path_fraction(seg, start_fraction, seg.start_time, t_now)
    return _sample_segment_xy(grid, seg, t0, t_now, samples)


def _segment_speed_grid(seg: Segment, t: float) -> float:
    """Numerically estimate the segment's instantaneous speed in grid units per second."""
    eps = min(max(seg.duration * 1e-3, 1e-12), 1e-6)
    t0 = max(seg.start_time, t - eps)
    t1 = min(seg.end_time, t + eps)
    if t1 <= t0:
        return 0.0
    p0 = seg.position_at(t0)
    p1 = seg.position_at(t1)
    return math.hypot(p1[0] - p0[0], p1[1] - p0[1]) / (t1 - t0)


def _segment_path_fraction(seg: Segment, t: float) -> float:
    """Return how far along the segment's path the atom is at absolute time `t`, in [0, 1]."""
    if seg.profile is not None:
        return min(max(float(seg.profile(t - seg.start_time)), 0.0), 1.0)

    di = seg.end_pos[0] - seg.start_pos[0]
    dj = seg.end_pos[1] - seg.start_pos[1]
    length2 = di * di + dj * dj
    if length2 <= 0:
        return 1.0
    pos = seg.position_at(t)
    u = ((pos[0] - seg.start_pos[0]) * di + (pos[1] - seg.start_pos[1]) * dj) / length2
    return min(max(float(u), 0.0), 1.0)


def _time_for_path_fraction(
    seg: Segment, fraction: float, lo: float, hi: float,
) -> float:
    """Bisect within [lo, hi] to find the time at which the segment reaches `fraction`."""
    fraction = min(max(float(fraction), 0.0), 1.0)
    if fraction <= 0.0:
        return seg.start_time
    if fraction >= _segment_path_fraction(seg, hi):
        return hi
    left = max(seg.start_time, lo)
    right = min(seg.end_time, hi)
    for _ in range(48):
        mid = (left + right) / 2.0
        if _segment_path_fraction(seg, mid) < fraction:
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


def _tone_panel_title(channel: ToneChannel, hardware: ToneHardware) -> str:
    """Compose the per-panel title that names the channel and the controlled axis."""
    axis = "x" if channel == "row" else "y"
    if hardware == "tone":
        return f"{channel} tone (controls {axis})"
    return f"{channel} {hardware} (controls {axis})"


# --- assorted small helpers ------------------------------------------------


def _draw_gaussian_blob(ax: Any, x: float, y: float, sigma: float) -> None:
    """Draw a soft red Gaussian halo at (x, y) to highlight an addressed atom."""
    r = 3.0 * sigma
    vals = np.linspace(-r, r, 31)
    xx, yy = np.meshgrid(vals, vals)
    zz = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
    ax.imshow(
        zz,
        extent=(x - r, x + r, y - r, y + r),
        origin="lower",
        cmap="Reds",
        alpha=0.33 * zz,
        interpolation="bilinear",
        zorder=4,
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


def _progress_iter(
    items: Iterable[Any], *, total: int, desc: str, enabled: bool,
) -> Iterable[Any]:
    """Wrap an iterable with a tqdm progress bar when `enabled`, else pass through."""
    if not enabled:
        yield from items
        return
    yield from tqdm(items, total=total, desc=desc, unit="frame")


def _save_gif_from_pngs(
    frame_paths: list[Path],
    output_path: Path,
    *,
    fps: int,
    show_progress: bool,
) -> None:
    """Stitch a sequence of PNG frames into an animated GIF via Pillow."""
    if not frame_paths:
        raise ValueError("no frames to save")
    duration_ms = int(round(1000 / fps))
    images = [
        Image.open(p)
        for p in _progress_iter(
            frame_paths, total=len(frame_paths),
            desc="load gif frames", enabled=show_progress,
        )
    ]
    try:
        images[0].save(
            output_path,
            save_all=True,
            append_images=images[1:],
            duration=duration_ms,
            loop=0,
        )
    finally:
        for image in images:
            image.close()
