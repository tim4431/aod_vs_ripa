"""Visualization helpers for atom rearrangement motion.

Most public functions accept either a `Sequence` (which also exposes hardware
steps) or an `AtomEnsemble` (just the atom trajectories). The lighter demo and
benchmark views render only atom motion; the detail view adds the EOM/AOD tone
panels used for hardware debugging.
"""

from __future__ import annotations

import multiprocessing as mp
import math
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping

import numpy as np

from .atom_config import Grid
from .atom_trajectory import AtomEnsemble, AtomTrajectory
from .ripa_freq import RIPASpec, nu_col, nu_row
from .segments import Segment

Site = tuple[int, int]
ToneChannel = Literal["row", "col"]
ToneHardware = Literal["AOD", "EOM", "AOD/EOM", "tone"]
PlannedTrajectoryMode = Literal["none", "full", "next"]
AddressedStyle = Literal["edge", "blob", "both", "none"]
RenderView = Literal["demo", "benchmark", "detail"]
RenderQuality = Literal["speed", "quality"]
OptimizeMode = Literal["speed", "performance", "quality"]

DEFAULT_FSR1_GHZ = 3.0
_TIME_FACTORS = {"s": 1.0, "ms": 1e3, "us": 1e6, "ns": 1e9}

_RENDER_BASE: dict[str, Any] | None = None
_RENDER_OPTIONS: dict[str, Any] | None = None


@dataclass(frozen=True)
class VisualizationAxes:
    """Axes returned by `plot_frame`.

    `frequency` panels are ordered like the reference image:
    current row/col tones on top, tone trajectories below.
    """

    atom: Any
    row_tones: Any
    col_tones: Any
    row_trajectories: Any
    col_trajectories: Any


@dataclass(frozen=True)
class RenderStyle:
    """Resolved visual tradeoffs for frame rendering."""

    dpi: int
    addressed_style: AddressedStyle
    show_motion_blur: bool
    trail_samples: int
    tone_samples_per_segment: int
    planned_samples_per_segment: int
    atom_size: float
    trap_size: float


def plot_atom_motion(
    ax: Any,
    motion: Any,
    t: float,
    *,
    quality: RenderQuality = "speed",
    atom_colors: Mapping[int, Any] | Iterable[Any] | None = None,
    static_traps: bool | Iterable[Site] = True,
    addressed_style: AddressedStyle | None = None,
    show_motion_blur: bool | None = None,
    show_planned: bool = True,
    planned_trajectory: PlannedTrajectoryMode | None = None,
    show_atom_ids: bool = True,
    trail_samples: int | None = None,
    trail_duration: float | None = None,
    planned_samples_per_segment: int | None = None,
    atom_size: float | None = None,
    trap_size: float | None = None,
    title: str | None = None,
) -> Any:
    """Draw only the atom plane for a `Sequence` or `AtomEnsemble`.

    This is the reusable renderer behind the demo and benchmark views. In
    `quality="speed"` it uses an edge highlight for addressed atoms and skips
    motion blur; `quality="quality"` uses a higher-detail addressed highlight
    and motion trail.
    """

    style = _resolve_render_style(
        quality,
        dpi=None,
        addressed_style=addressed_style,
        show_motion_blur=show_motion_blur,
        trail_samples=trail_samples,
        tone_samples_per_segment=None,
        planned_samples_per_segment=planned_samples_per_segment,
        atom_size=atom_size,
        trap_size=trap_size,
    )
    ensemble = _as_ensemble(motion)
    planned_mode = _resolve_planned_trajectory_mode(
        show_planned, planned_trajectory
    )
    base = _build_base_payload(
        ensemble,
        spec=None,
        atom_colors=atom_colors,
        static_traps=static_traps,
        time_unit="us",
        frequency_label="nu / FSR1 (mod 1)",
        tone_samples_per_segment=style.tone_samples_per_segment,
        include_tones=False,
    )
    frame = _build_frame_payload(
        ensemble,
        float(t),
        spec=None,
        motion=motion,
        include_tones=False,
        show_motion_blur=style.show_motion_blur,
        planned_trajectory=planned_mode,
        trail_samples=style.trail_samples,
        trail_duration=trail_duration,
        planned_samples_per_segment=style.planned_samples_per_segment,
    )
    _plot_atom_plane_payload(
        ax,
        base,
        frame,
        addressed_style=style.addressed_style,
        show_atom_ids=show_atom_ids,
        atom_size=style.atom_size,
        trap_size=style.trap_size,
        title=title,
    )
    return ax


def plot_demo_frame(
    motion: Any,
    t: float,
    *,
    fig: Any | None = None,
    ax: Any | None = None,
    quality: RenderQuality = "speed",
    atom_colors: Mapping[int, Any] | Iterable[Any] | None = None,
    static_traps: bool | Iterable[Site] = True,
    addressed_style: AddressedStyle | None = None,
    show_motion_blur: bool | None = None,
    show_planned: bool = True,
    planned_trajectory: PlannedTrajectoryMode | None = None,
    show_atom_ids: bool = True,
    trail_samples: int | None = None,
    trail_duration: float | None = None,
    planned_samples_per_segment: int | None = None,
    atom_size: float | None = None,
    trap_size: float | None = None,
    figsize: tuple[float, float] = (5.4, 5.2),
    title: str | None = None,
) -> tuple[Any, Any]:
    """Draw the demo view: one atom-motion plot."""

    plt = _load_pyplot()
    if ax is None:
        fig = plt.figure(figsize=figsize, constrained_layout=True) if fig is None else fig
        ax = fig.add_subplot(1, 1, 1)
    else:
        fig = ax.figure if fig is None else fig

    plot_atom_motion(
        ax,
        motion,
        t,
        quality=quality,
        atom_colors=atom_colors,
        static_traps=static_traps,
        addressed_style=addressed_style,
        show_motion_blur=show_motion_blur,
        show_planned=show_planned,
        planned_trajectory=planned_trajectory,
        show_atom_ids=show_atom_ids,
        trail_samples=trail_samples,
        trail_duration=trail_duration,
        planned_samples_per_segment=planned_samples_per_segment,
        atom_size=atom_size,
        trap_size=trap_size,
        title=title,
    )
    return fig, ax


def plot_benchmark_frame(
    motions: Mapping[str, Any] | Iterable[Any],
    t: float,
    *,
    labels: Iterable[str] | None = None,
    fig: Any | None = None,
    axes: Iterable[Any] | Any | None = None,
    quality: RenderQuality = "speed",
    atom_colors: Mapping[int, Any] | Iterable[Any] | None = None,
    static_traps: bool | Iterable[Site] | Mapping[str, Iterable[Site]] = True,
    addressed_style: AddressedStyle | None = None,
    show_motion_blur: bool | None = None,
    show_planned: bool = True,
    planned_trajectory: PlannedTrajectoryMode | None = None,
    show_atom_ids: bool = True,
    trail_samples: int | None = None,
    trail_duration: float | None = None,
    planned_samples_per_segment: int | None = None,
    atom_size: float | None = None,
    trap_size: float | None = None,
    figsize: tuple[float, float] | None = None,
    title: str | None = None,
) -> tuple[Any, list[Any]]:
    """Draw the benchmark view: one row of atom-motion plots."""

    items = _normalize_motion_items(motions, labels)
    if not items:
        raise ValueError("benchmark view needs at least one motion object")

    plt = _load_pyplot()
    n = len(items)
    if axes is None:
        size = figsize or (max(4.0, 4.0 * n), 4.6)
        fig, axes_obj = plt.subplots(
            1,
            n,
            figsize=size,
            squeeze=False,
            constrained_layout=True,
        )
        axes_list = list(axes_obj.ravel())
    else:
        axes_list = list(np.asarray(axes, dtype=object).ravel())
        if len(axes_list) != n:
            raise ValueError(f"expected {n} benchmark axes, got {len(axes_list)}")
        fig = axes_list[0].figure if fig is None else fig

    for index, ((label, motion), ax) in enumerate(zip(items, axes_list)):
        panel_title = f"{label}  -  t = {_format_time(float(t), 'us')}"
        plot_atom_motion(
            ax,
            motion,
            t,
            quality=quality,
            atom_colors=atom_colors,
            static_traps=_panel_value(static_traps, index, label),
            addressed_style=addressed_style,
            show_motion_blur=show_motion_blur,
            show_planned=show_planned,
            planned_trajectory=planned_trajectory,
            show_atom_ids=show_atom_ids,
            trail_samples=trail_samples,
            trail_duration=trail_duration,
            planned_samples_per_segment=planned_samples_per_segment,
            atom_size=atom_size,
            trap_size=trap_size,
            title=panel_title,
        )

    if title:
        fig.suptitle(title)
    return fig, axes_list


def plot_atom_plane(
    ax: Any,
    timeline: Any,
    t: float,
    *,
    atom_colors: Mapping[int, Any] | Iterable[Any] | None = None,
    static_traps: bool | Iterable[Site] = True,
    addressed_style: AddressedStyle = "edge",
    show_motion_blur: bool = True,
    show_planned: bool = True,
    planned_trajectory: PlannedTrajectoryMode | None = None,
    show_atom_ids: bool = True,
    trail_samples: int = 8,
    trail_duration: float | None = None,
    planned_samples_per_segment: int = 32,
    atom_size: float = 70.0,
    trap_size: float = 12.0,
) -> Any:
    """Draw the atom plane at time `t` on `ax`.

    Coordinates are physical micrometers. The underlying atom labels remain
    the codebase's `(i, j)` grid units, with `(0, 0)` at the lower-left site.
    `planned_trajectory` may be "none", "full", or "next"; when omitted,
    the old `show_planned` boolean selects "full" or "none".
    """

    ensemble = _as_ensemble(timeline)
    planned_mode = _resolve_planned_trajectory_mode(
        show_planned, planned_trajectory
    )
    base = _build_base_payload(
        ensemble,
        spec=None,
        atom_colors=atom_colors,
        static_traps=static_traps,
        time_unit="us",
        frequency_label="nu / FSR1 (mod 1)",
        tone_samples_per_segment=48,
        include_tones=False,
    )
    frame = _build_frame_payload(
        ensemble,
        float(t),
        spec=None,
        motion=timeline,
        include_tones=False,
        show_motion_blur=show_motion_blur,
        planned_trajectory=planned_mode,
        trail_samples=trail_samples,
        trail_duration=trail_duration,
        planned_samples_per_segment=planned_samples_per_segment,
    )
    _plot_atom_plane_payload(
        ax,
        base,
        frame,
        addressed_style=addressed_style,
        show_atom_ids=show_atom_ids,
        atom_size=atom_size,
        trap_size=trap_size,
        title=None,
    )
    return ax


def plot_frequency_tones(
    axes: Any,
    timeline: Any,
    t: float,
    *,
    spec: RIPASpec | None = None,
    atom_colors: Mapping[int, Any] | Iterable[Any] | None = None,
    show_future: bool = True,
    time_unit: Literal["s", "ms", "us", "ns"] = "us",
    frequency_label: str = "nu / FSR1 (mod 1)",
    tone_samples_per_segment: int = 64,
) -> Any:
    """Draw row/col tone panels on a 2x2 axes object.

    `axes` may be a numpy array from `plt.subplots(2, 2)` or a nested
    sequence in the order `[[row_tones, col_tones], [row_history, col_history]]`.
    """

    ensemble = _as_ensemble(timeline)
    base = _build_base_payload(
        ensemble,
        spec=spec,
        atom_colors=atom_colors,
        static_traps=False,
        time_unit=time_unit,
        frequency_label=frequency_label,
        tone_samples_per_segment=tone_samples_per_segment,
        include_tones=True,
    )
    frame = _build_frame_payload(
        ensemble,
        float(t),
        spec=base["spec"],
        motion=timeline,
        include_tones=True,
        show_motion_blur=False,
        planned_trajectory="none",
        trail_samples=0,
        trail_duration=None,
        planned_samples_per_segment=0,
    )
    row_tones, col_tones, row_hist, col_hist = _normalize_frequency_axes(axes)
    _plot_current_tones_payload(row_tones, base, frame, "row")
    _plot_current_tones_payload(col_tones, base, frame, "col")
    planned_mode = "full" if show_future else "none"
    _plot_tone_history_payload(row_hist, base, frame, "row", planned_mode)
    _plot_tone_history_payload(col_hist, base, frame, "col", planned_mode)
    return axes


def plot_frame(
    timeline: Any,
    t: float,
    *,
    view: RenderView = "detail",
    quality: RenderQuality = "speed",
    labels: Iterable[str] | None = None,
    fig: Any | None = None,
    axes: VisualizationAxes | tuple[Any, Any, Any, Any, Any] | Iterable[Any] | None = None,
    spec: RIPASpec | None = None,
    atom_colors: Mapping[int, Any] | Iterable[Any] | None = None,
    static_traps: bool | Iterable[Site] = True,
    addressed_style: AddressedStyle | None = None,
    show_motion_blur: bool | None = None,
    show_planned: bool = True,
    planned_trajectory: PlannedTrajectoryMode | None = None,
    show_atom_ids: bool = True,
    time_unit: Literal["s", "ms", "us", "ns"] = "us",
    frequency_label: str = "nu / FSR1 (mod 1)",
    tone_samples_per_segment: int = 64,
    trail_samples: int | None = None,
    trail_duration: float | None = None,
    planned_samples_per_segment: int | None = None,
    atom_size: float | None = None,
    trap_size: float | None = None,
    figsize: tuple[float, float] = (12.0, 6.2),
    title: str | None = None,
) -> tuple[Any, Any]:
    """Draw a demo, benchmark, or detail animation frame.

    `view="demo"` draws one atom-motion panel, `view="benchmark"` draws a
    row of atom-motion panels for multiple schedulers, and `view="detail"`
    draws the atom plane plus EOM/AOD tone panels.
    """

    if view == "demo":
        return plot_demo_frame(
            timeline,
            t,
            fig=fig,
            ax=axes,  # type: ignore[arg-type]
            quality=quality,
            atom_colors=atom_colors,
            static_traps=static_traps,
            addressed_style=addressed_style,
            show_motion_blur=show_motion_blur,
            show_planned=show_planned,
            planned_trajectory=planned_trajectory,
            show_atom_ids=show_atom_ids,
            trail_samples=trail_samples,
            trail_duration=trail_duration,
            planned_samples_per_segment=planned_samples_per_segment,
            atom_size=atom_size,
            trap_size=trap_size,
            figsize=figsize,
            title=title,
        )
    if view == "benchmark":
        return plot_benchmark_frame(
            timeline,
            t,
            labels=labels,
            fig=fig,
            axes=axes,
            quality=quality,
            atom_colors=atom_colors,
            static_traps=static_traps,
            addressed_style=addressed_style,
            show_motion_blur=show_motion_blur,
            show_planned=show_planned,
            planned_trajectory=planned_trajectory,
            show_atom_ids=show_atom_ids,
            trail_samples=trail_samples,
            trail_duration=trail_duration,
            planned_samples_per_segment=planned_samples_per_segment,
            atom_size=atom_size,
            trap_size=trap_size,
            figsize=None if figsize == (12.0, 6.2) else figsize,
            title=title,
        )
    if view != "detail":
        raise ValueError("view must be 'demo', 'benchmark', or 'detail'")

    plt = _load_pyplot()
    ensemble = _as_ensemble(timeline)
    style = _resolve_render_style(
        quality,
        dpi=None,
        addressed_style=addressed_style,
        show_motion_blur=show_motion_blur,
        trail_samples=trail_samples,
        tone_samples_per_segment=tone_samples_per_segment,
        planned_samples_per_segment=planned_samples_per_segment,
        atom_size=atom_size,
        trap_size=trap_size,
    )
    planned_mode = _resolve_planned_trajectory_mode(
        show_planned, planned_trajectory
    )

    if axes is None:
        fig = (
            plt.figure(figsize=figsize, constrained_layout=True) if fig is None else fig
        )
        gs = fig.add_gridspec(
            2,
            3,
            width_ratios=(2.5, 1.0, 1.0),
            height_ratios=(1.0, 1.0),
        )
        axes_obj = VisualizationAxes(
            atom=fig.add_subplot(gs[:, 0]),
            row_tones=fig.add_subplot(gs[0, 1]),
            col_tones=fig.add_subplot(gs[0, 2]),
            row_trajectories=fig.add_subplot(gs[1, 1]),
            col_trajectories=fig.add_subplot(gs[1, 2]),
        )
    else:
        axes_obj = _coerce_visualization_axes(axes)
        fig = axes_obj.atom.figure if fig is None else fig

    base = _build_base_payload(
        ensemble,
        spec=spec,
        atom_colors=atom_colors,
        static_traps=static_traps,
        time_unit=time_unit,
        frequency_label=frequency_label,
        tone_samples_per_segment=style.tone_samples_per_segment,
        include_tones=True,
    )
    frame = _build_frame_payload(
        ensemble,
        float(t),
        spec=base["spec"],
        motion=timeline,
        include_tones=True,
        show_motion_blur=style.show_motion_blur,
        planned_trajectory=planned_mode,
        trail_samples=style.trail_samples,
        trail_duration=trail_duration,
        planned_samples_per_segment=style.planned_samples_per_segment,
    )
    _plot_payload_frame(
        fig,
        axes_obj,
        base,
        frame,
        addressed_style=style.addressed_style,
        show_atom_ids=show_atom_ids,
        planned_trajectory=planned_mode,
        atom_size=style.atom_size,
        trap_size=style.trap_size,
        title=title,
    )
    return fig, axes_obj


def save_frame(
    timeline: Any,
    t: float,
    output_path: str | Path,
    **plot_kwargs: Any,
) -> Path:
    """Render one frame to a PNG file."""

    plt = _load_pyplot(force_agg=True)
    optimize = plot_kwargs.pop("optimize", None)
    if "quality" not in plot_kwargs and optimize is not None:
        plot_kwargs["quality"] = _normalize_optimize(optimize)
    resolved_quality = _resolve_quality_option(plot_kwargs.get("quality"), None)
    dpi = int(
        plot_kwargs.pop(
            "dpi",
            _resolve_render_style(
                resolved_quality,
                dpi=None,
                addressed_style=None,
                show_motion_blur=None,
                trail_samples=None,
                tone_samples_per_segment=None,
                planned_samples_per_segment=None,
                atom_size=None,
                trap_size=None,
            ).dpi,
        )
    )
    fig, _ = plot_frame(timeline, t, **plot_kwargs)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, facecolor="white")
    plt.close(fig)
    return path


def render_animation(
    timeline: Any,
    output_path: str | Path,
    *,
    view: RenderView = "demo",
    quality: RenderQuality | None = None,
    labels: Iterable[str] | None = None,
    fps: int = 20,
    n_frames: int | None = None,
    frame_dt: float = 10e-6,
    frame_times: Iterable[float] | None = None,
    hold_seconds: float = 1.0,
    optimize: OptimizeMode | None = None,
    use_multiprocessing: bool = True,
    workers: int | None = None,
    frames_dir: str | Path | None = None,
    keep_frames: bool = False,
    spec: RIPASpec | None = None,
    atom_colors: Mapping[int, Any] | Iterable[Any] | None = None,
    static_traps: bool | Iterable[Site] = True,
    addressed_style: AddressedStyle | None = None,
    show_motion_blur: bool | None = None,
    show_planned: bool = True,
    planned_trajectory: PlannedTrajectoryMode | None = None,
    show_atom_ids: bool = True,
    time_unit: Literal["s", "ms", "us", "ns"] = "us",
    frequency_label: str = "nu / FSR1 (mod 1)",
    figsize: tuple[float, float] | None = None,
    dpi: int | None = None,
    show_progress: bool = True,
    title: str | None = None,
) -> Path:
    """Render a PNG frame sequence and combine it into a GIF.

    By default, frame count is proportional to physical rearrangement time:
    one moving frame every `frame_dt` seconds. Pass `n_frames` for a fixed
    sample count, or `frame_times` for an explicit schedule. GIF playback
    speed is controlled by `fps`; the physics timeline is visualized, not
    played in real time.

    Frame PNGs are rendered in parallel when `use_multiprocessing` is true.
    Progress bars use `tqdm` when installed, with a simple stderr fallback.
    `planned_trajectory` controls future paths: "none", "full", or "next".
    """

    if fps <= 0:
        raise ValueError("fps must be positive")
    if n_frames is not None and n_frames <= 0:
        raise ValueError("n_frames must be positive")
    if frame_dt <= 0:
        raise ValueError("frame_dt must be positive")

    if view not in ("demo", "benchmark", "detail"):
        raise ValueError("view must be 'demo', 'benchmark', or 'detail'")

    mode = _resolve_quality_option(quality, optimize)
    style = _resolve_render_style(
        mode,
        dpi=dpi,
        addressed_style=addressed_style,
        show_motion_blur=show_motion_blur,
        trail_samples=None,
        tone_samples_per_segment=None,
        planned_samples_per_segment=None,
        atom_size=None,
        trap_size=None,
    )
    planned_mode = _resolve_planned_trajectory_mode(
        show_planned, planned_trajectory
    )

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    if view == "benchmark":
        motion_items = _normalize_motion_items(timeline, labels)
        if not motion_items:
            raise ValueError("benchmark view needs at least one motion object")
        total = max((_as_ensemble(motion).total_duration() for _, motion in motion_items), default=0.0)
    else:
        motion_items = []
        ensemble = _as_ensemble(timeline)
        total = ensemble.total_duration()

    if frame_times is None:
        if total <= 0:
            moving_times = np.array([0.0], dtype=float)
        else:
            frame_count = (
                int(n_frames)
                if n_frames is not None
                else int(math.ceil(total / frame_dt)) + 1
            )
            moving_times = np.linspace(0.0, total, max(2, frame_count), dtype=float)
    else:
        moving_times = np.asarray(list(frame_times), dtype=float)
        if moving_times.size == 0:
            raise ValueError("frame_times must contain at least one time")

    hold_count = max(0, int(round(hold_seconds * fps)))
    frame_schedule = np.concatenate(
        [
            np.full(hold_count, moving_times[0], dtype=float),
            moving_times,
            np.full(hold_count, moving_times[-1], dtype=float),
        ]
    )

    if view == "benchmark":
        panel_bases = []
        for index, (label, motion) in enumerate(motion_items):
            panel_ensemble = _as_ensemble(motion)
            panel_bases.append(
                {
                    "label": label,
                    "motion": motion,
                    "ensemble": panel_ensemble,
                    "base": _build_base_payload(
                        panel_ensemble,
                        spec=None,
                        atom_colors=atom_colors,
                        static_traps=_panel_value(static_traps, index, label),
                        time_unit=time_unit,
                        frequency_label=frequency_label,
                        tone_samples_per_segment=style.tone_samples_per_segment,
                        include_tones=False,
                    ),
                }
            )
        base = {
            "view": "benchmark",
            "panels": [
                {"label": panel["label"], "base": panel["base"]}
                for panel in panel_bases
            ],
        }
        frame_payloads = [
            {
                "t": float(t),
                "panels": [
                    {
                        "label": panel["label"],
                        "frame": _build_frame_payload(
                            panel["ensemble"],
                            float(t),
                            spec=None,
                            motion=panel["motion"],
                            include_tones=False,
                            show_motion_blur=style.show_motion_blur,
                            planned_trajectory=planned_mode,
                            trail_samples=style.trail_samples,
                            trail_duration=None,
                            planned_samples_per_segment=style.planned_samples_per_segment,
                        ),
                    }
                    for panel in panel_bases
                ],
            }
            for t in _progress_iter(
                frame_schedule,
                total=len(frame_schedule),
                desc="prepare frames",
                enabled=show_progress,
            )
        ]
    else:
        include_tones = view == "detail"
        base = _build_base_payload(
            ensemble,
            spec=spec,
            atom_colors=atom_colors,
            static_traps=static_traps,
            time_unit=time_unit,
            frequency_label=frequency_label,
            tone_samples_per_segment=style.tone_samples_per_segment,
            include_tones=include_tones,
        )
        frame_payloads = [
            _build_frame_payload(
                ensemble,
                float(t),
                spec=base["spec"],
                motion=timeline,
                include_tones=include_tones,
                show_motion_blur=style.show_motion_blur,
                planned_trajectory=planned_mode,
                trail_samples=style.trail_samples,
                trail_duration=None,
                planned_samples_per_segment=style.planned_samples_per_segment,
            )
            for t in _progress_iter(
                frame_schedule,
                total=len(frame_schedule),
                desc="prepare frames",
                enabled=show_progress,
            )
        ]

    if figsize is None:
        if view == "detail":
            resolved_figsize = (12.0, 6.2)
        elif view == "benchmark":
            resolved_figsize = (max(4.0, 4.0 * len(motion_items)), 4.6)
        else:
            resolved_figsize = (5.4, 5.2)
    else:
        resolved_figsize = figsize

    options = {
        "view": view,
        "figsize": resolved_figsize,
        "dpi": style.dpi,
        "addressed_style": style.addressed_style,
        "show_atom_ids": show_atom_ids,
        "planned_trajectory": planned_mode,
        "atom_size": style.atom_size,
        "trap_size": style.trap_size,
        "title": title,
    }

    if frames_dir is None and not keep_frames:
        with tempfile.TemporaryDirectory(prefix="aod_vs_ripa_frames_") as tmp:
            frame_paths = _render_png_frames(
                Path(tmp),
                base,
                frame_payloads,
                options,
                use_multiprocessing=use_multiprocessing,
                workers=workers,
                show_progress=show_progress,
            )
            _save_gif_from_pngs(
                frame_paths, out, fps=fps, show_progress=show_progress
            )
    else:
        frame_root = Path(frames_dir) if frames_dir is not None else out.with_suffix("")
        frame_root.mkdir(parents=True, exist_ok=True)
        frame_paths = _render_png_frames(
            frame_root,
            base,
            frame_payloads,
            options,
            use_multiprocessing=use_multiprocessing,
            workers=workers,
            show_progress=show_progress,
        )
        _save_gif_from_pngs(frame_paths, out, fps=fps, show_progress=show_progress)

    return out


def render_check_outputs(
    timeline: Any,
    output_dir: str | Path,
    prefix: str,
    *,
    render_gif: bool = True,
    gif_fps: int = 8,
    gif_frames: int | None = None,
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
    view = visual_kwargs.get("view", "demo")
    if view == "benchmark":
        final_t = max(
            (
                _as_ensemble(motion).total_duration()
                for _, motion in _normalize_motion_items(
                    timeline, visual_kwargs.get("labels")
                )
            ),
            default=0.0,
        )
    else:
        ensemble = _as_ensemble(timeline)
        final_t = ensemble.total_duration()
    label = title_prefix or prefix

    paths = {
        "t0": out_dir / f"{prefix}_t0.png",
        "tfinal": out_dir / f"{prefix}_tfinal.png",
    }
    animation_only = {
        "frames_dir",
        "keep_frames",
        "show_progress",
        "use_multiprocessing",
        "workers",
    }
    frame_kwargs = {
        key: value for key, value in visual_kwargs.items() if key not in animation_only
    }

    save_frame(
        timeline,
        0.0,
        paths["t0"],
        **{
            **frame_kwargs,
            "title": frame_kwargs.get("title", f"{label} - t=0"),
        },
    )
    save_frame(
        timeline,
        final_t,
        paths["tfinal"],
        **{
            **frame_kwargs,
            "title": frame_kwargs.get("title", f"{label} - final"),
        },
    )

    if render_gif:
        paths["gif"] = out_dir / f"{prefix}.gif"
        animation_kwargs = {
            key: value for key, value in visual_kwargs.items() if key != "title"
        }
        render_animation(
            timeline,
            paths["gif"],
            fps=gif_fps,
            n_frames=gif_frames,
            frame_dt=gif_frame_dt,
            hold_seconds=gif_hold_seconds,
            **animation_kwargs,
        )

    return paths


def _as_ensemble(timeline: Any) -> AtomEnsemble:
    """Return trajectories from either an AtomEnsemble or a Sequence-like object."""
    if isinstance(timeline, AtomEnsemble):
        return timeline
    ensemble = getattr(timeline, "ensemble", None)
    if isinstance(ensemble, AtomEnsemble):
        return ensemble
    raise TypeError(
        "expected an AtomEnsemble or a Sequence-like object with .ensemble"
    )


def _load_pyplot(*, force_agg: bool = False) -> Any:
    if force_agg:
        import matplotlib

        matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt


def _build_base_payload(
    ensemble: AtomEnsemble,
    *,
    spec: RIPASpec | None,
    atom_colors: Mapping[int, Any] | Iterable[Any] | None,
    static_traps: bool | Iterable[Site],
    time_unit: Literal["s", "ms", "us", "ns"],
    frequency_label: str,
    tone_samples_per_segment: int,
    include_tones: bool,
) -> dict[str, Any]:
    if time_unit not in _TIME_FACTORS:
        raise ValueError(f"unknown time_unit {time_unit!r}")

    grid = ensemble.grid
    resolved_spec = spec or (RIPASpec(N=grid.N, FSR1=DEFAULT_FSR1_GHZ) if include_tones else None)
    if resolved_spec is not None and resolved_spec.N != grid.N:
        raise ValueError(f"RIPASpec.N={resolved_spec.N} does not match grid.N={grid.N}")

    atom_ids = [int(a.atom_id) for a in ensemble.atomtrajs]
    colors = _resolve_atom_colors(atom_ids, atom_colors)
    grid_sites = _all_grid_sites(grid.N)
    grid_xy = grid.ij_to_xy(grid_sites)
    static_xy = _resolve_static_traps(grid, static_traps)
    if include_tones:
        assert resolved_spec is not None
        row_trajs, col_trajs = _collect_tone_trajectories(
            ensemble, resolved_spec, max(2, int(tone_samples_per_segment))
        )
        tone_hardware = _tone_hardware_name(ensemble)
    else:
        row_trajs, col_trajs = [], []
        tone_hardware = "tone"
    lo, hi = _site_extent_um(grid)

    return {
        "grid_N": grid.N,
        "grid_d": grid.d,
        "grid_rc": grid.rc,
        "grid_xy": grid_xy,
        "static_xy": static_xy,
        "xlim": (lo - grid.d, hi + grid.d),
        "ylim": (lo - grid.d, hi + grid.d),
        "boundary": (lo - grid.d / 2.0, hi + grid.d / 2.0),
        "atom_ids": atom_ids,
        "colors": colors,
        "spec": resolved_spec,
        "row_trajectories": row_trajs,
        "col_trajectories": col_trajs,
        "tone_hardware": tone_hardware,
        "total_duration": ensemble.total_duration(),
        "time_unit": time_unit,
        "time_factor": _TIME_FACTORS[time_unit],
        "frequency_label": frequency_label,
    }


def _build_frame_payload(
    ensemble: AtomEnsemble,
    t: float,
    spec: RIPASpec | None,
    *,
    motion: Any | None = None,
    include_tones: bool,
    show_motion_blur: bool,
    planned_trajectory: PlannedTrajectoryMode,
    trail_samples: int,
    trail_duration: float | None,
    planned_samples_per_segment: int,
) -> dict[str, Any]:
    if include_tones and spec is None:
        raise ValueError("tone panels need a RIPASpec")

    positions_ij = ensemble.positions_at(t)
    positions_xy = ensemble.grid.ij_to_xy(positions_ij)
    active_indices: list[int] = []
    row_active: list[dict[str, Any]] = []
    col_active: list[dict[str, Any]] = []
    trails: list[dict[str, Any]] = []
    planned: list[dict[str, Any]] = []
    aod_traps_xy = _active_aod_traps_xy(motion, ensemble, t)

    for idx, atom in enumerate(ensemble.atomtrajs):
        seg = _active_segment(atom, t)
        if seg is not None:
            active_indices.append(idx)
            channels = _channels_for_segment(seg)
            pos = atom.position_at(t)
            if include_tones and "row" in channels:
                assert spec is not None
                row_active.append(
                    {
                        "idx": idx,
                        "atom_id": int(atom.atom_id),
                        "freq": _tone_frequency("row", pos, spec, segment=seg),
                    }
                )
            if include_tones and "col" in channels:
                assert spec is not None
                col_active.append(
                    {
                        "idx": idx,
                        "atom_id": int(atom.atom_id),
                        "freq": _tone_frequency("col", pos, spec, segment=seg),
                    }
                )
            if show_motion_blur and trail_samples > 1:
                xy = _motion_blur_xy(
                    ensemble.grid,
                    seg,
                    float(t),
                    trail_samples,
                    trail_duration,
                )
                if len(xy) > 1:
                    trails.append({"idx": idx, "xy": xy})

        if planned_trajectory != "none" and planned_samples_per_segment > 1:
            future_segments = list(_future_segments(atom, t))
            if planned_trajectory == "next":
                future_segments = future_segments[:1]

            for future_seg in future_segments:
                t0 = max(float(t), future_seg.start_time)
                xy = _sample_segment_xy(
                    ensemble.grid,
                    future_seg,
                    t0,
                    future_seg.end_time,
                    planned_samples_per_segment,
                )
                if len(xy) > 1:
                    planned.append(
                        {
                            "idx": idx,
                            "xy": xy,
                            "channels": _channels_for_segment(future_seg),
                        }
                    )

    return {
        "t": float(t),
        "positions_ij": positions_ij,
        "positions_xy": positions_xy,
        "active_indices": np.asarray(active_indices, dtype=int),
        "row_active": row_active,
        "col_active": col_active,
        "trails": trails,
        "planned": planned,
        "aod_traps_xy": aod_traps_xy,
    }


def _plot_payload_frame(
    fig: Any,
    axes: VisualizationAxes,
    base: dict[str, Any],
    frame: dict[str, Any],
    *,
    addressed_style: AddressedStyle,
    show_atom_ids: bool,
    planned_trajectory: PlannedTrajectoryMode,
    atom_size: float,
    trap_size: float,
    title: str | None,
) -> None:
    _plot_atom_plane_payload(
        axes.atom,
        base,
        frame,
        addressed_style=addressed_style,
        show_atom_ids=show_atom_ids,
        atom_size=atom_size,
        trap_size=trap_size,
        title=None,
    )
    _plot_current_tones_payload(axes.row_tones, base, frame, "row")
    _plot_current_tones_payload(axes.col_tones, base, frame, "col")
    _plot_tone_history_payload(
        axes.row_trajectories, base, frame, "row", planned_trajectory
    )
    _plot_tone_history_payload(
        axes.col_trajectories, base, frame, "col", planned_trajectory
    )

    if title is None:
        t_label = _format_time(frame["t"], base["time_unit"])
        title = f"Atom rearrangement  -  t = {t_label}"
    fig.suptitle(title)


def _plot_atom_plane_payload(
    ax: Any,
    base: dict[str, Any],
    frame: dict[str, Any],
    *,
    addressed_style: AddressedStyle,
    show_atom_ids: bool,
    atom_size: float,
    trap_size: float,
    title: str | None,
) -> None:
    ax.clear()
    ax.set_facecolor("white")

    grid_xy = base["grid_xy"]
    ax.scatter(
        grid_xy[:, 0],
        grid_xy[:, 1],
        s=trap_size,
        c="#9a9a9a",
        alpha=0.45,
        linewidths=0,
        zorder=1,
    )

    static_xy = base["static_xy"]
    if len(static_xy):
        ax.scatter(
            static_xy[:, 0],
            static_xy[:, 1],
            s=trap_size * 2.3,
            facecolors="none",
            edgecolors="#4b8bbe",
            linewidths=0.8,
            alpha=0.55,
            zorder=2,
        )

    aod_traps_xy = frame.get("aod_traps_xy")
    if aod_traps_xy is not None and len(aod_traps_xy):
        ax.scatter(
            aod_traps_xy[:, 0],
            aod_traps_xy[:, 1],
            s=trap_size * 5.0,
            marker="s",
            facecolors="none",
            edgecolors="#d62728",
            linewidths=1.1,
            alpha=0.70,
            zorder=3.6,
        )

    for path in frame["planned"]:
        idx = int(path["idx"])
        xy = path["xy"]
        ax.plot(
            xy[:, 0],
            xy[:, 1],
            linestyle="--",
            linewidth=1.1,
            color=base["colors"][idx],
            alpha=0.34,
            zorder=2.5,
        )

    for trail in frame["trails"]:
        idx = int(trail["idx"])
        xy = trail["xy"]
        color = base["colors"][idx]
        ax.plot(xy[:, 0], xy[:, 1], color=color, linewidth=4.0, alpha=0.20, zorder=3)
        fade = np.linspace(0.08, 0.32, len(xy))
        sizes = np.linspace(atom_size * 0.15, atom_size * 0.55, len(xy))
        for (x, y), alpha, size in zip(xy, fade, sizes):
            ax.scatter(
                [x],
                [y],
                s=size,
                c=[color],
                alpha=float(alpha),
                linewidths=0,
                zorder=3.2,
            )

    positions_xy = frame["positions_xy"]
    active = np.zeros(len(positions_xy), dtype=bool)
    if len(frame["active_indices"]):
        active[frame["active_indices"]] = True

    if addressed_style in ("blob", "both"):
        sigma = max(base["grid_d"] * 0.22, 1e-9)
        for idx in np.flatnonzero(active):
            x, y = positions_xy[idx]
            _draw_gaussian_blob(ax, float(x), float(y), sigma)

    edgecolors = [
        "#d62728" if is_active and addressed_style in ("edge", "both") else "#ffffff"
        for is_active in active
    ]
    linewidths = [
        2.3 if is_active and addressed_style in ("edge", "both") else 0.8
        for is_active in active
    ]
    ax.scatter(
        positions_xy[:, 0],
        positions_xy[:, 1],
        s=atom_size,
        c=base["colors"],
        edgecolors=edgecolors,
        linewidths=linewidths,
        zorder=5,
    )

    if show_atom_ids:
        label_offset = 0.22 * base["grid_d"]
        for atom_id, (x, y), color in zip(
            base["atom_ids"], positions_xy, base["colors"]
        ):
            ax.text(
                x,
                y + label_offset,
                str(atom_id),
                ha="center",
                va="bottom",
                fontsize=7,
                color=_darken_color(color),
                zorder=6,
            )

    b0, b1 = base["boundary"]
    width = b1 - b0
    rect = _rectangle_patch((b0, b0), width, width)
    ax.add_patch(rect)

    ax.set_xlim(*base["xlim"])
    ax.set_ylim(*base["ylim"])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (um)")
    ax.set_ylabel("y (um)")
    ax.set_title(title or f"atom motion  -  t = {_format_time(frame['t'], 'us')}")


def _tone_panel_title(channel: ToneChannel, hardware: ToneHardware) -> str:
    axis = "x" if channel == "row" else "y"
    if hardware == "tone":
        return f"{channel} tone (controls {axis})"
    return f"{channel} {hardware} (controls {axis})"


def _plot_current_tones_payload(
    ax: Any,
    base: dict[str, Any],
    frame: dict[str, Any],
    channel: ToneChannel,
) -> None:
    ax.clear()
    spec: RIPASpec = base["spec"]
    entries = frame["row_active"] if channel == "row" else frame["col_active"]
    colors = base["colors"]
    for entry in entries:
        idx = int(entry["idx"])
        freq = _frequency_for_display(float(entry["freq"]), spec)
        color = colors[idx]
        ax.vlines(freq, 0.0, 1.0, color=color, linewidth=2.5, alpha=0.9)
        ax.scatter([freq], [1.0], c=[color], s=22.0, zorder=3)
        ax.text(
            freq,
            1.04,
            str(entry["atom_id"]),
            ha="center",
            va="bottom",
            fontsize=6,
            color=_darken_color(color),
            clip_on=True,
        )

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.2)
    ax.set_xlabel(base["frequency_label"])
    ax.set_ylabel("amplitude")
    ax.set_title(_tone_panel_title(channel, base["tone_hardware"]))
    ax.grid(True, color="#e5e5e5", linewidth=0.6)


def _plot_tone_history_payload(
    ax: Any,
    base: dict[str, Any],
    frame: dict[str, Any],
    channel: ToneChannel,
    planned_trajectory: PlannedTrajectoryMode,
) -> None:
    ax.clear()
    spec: RIPASpec = base["spec"]
    time_factor = float(base["time_factor"])
    trajs = base["row_trajectories"] if channel == "row" else base["col_trajectories"]
    colors = base["colors"]
    t_now = float(frame["t"])
    next_by_atom = (
        _next_tone_trajectory_by_atom(trajs, t_now)
        if planned_trajectory == "next"
        else {}
    )

    for traj in trajs:
        ts = np.asarray(traj["times"], dtype=float)
        nus = _frequencies_for_display(np.asarray(traj["freqs"], dtype=float), spec)
        idx = int(traj["idx"])
        if planned_trajectory == "full":
            _plot_wrapped_frequency_line(
                ax,
                ts * time_factor,
                nus,
                1.0,
                color=colors[idx],
                linewidth=1.4,
                alpha=0.35,
            )
        else:
            mask = ts <= t_now + 1e-15
            if np.count_nonzero(mask) >= 2:
                _plot_wrapped_frequency_line(
                    ax,
                    ts[mask] * time_factor,
                    nus[mask],
                    1.0,
                    color=colors[idx],
                    linewidth=1.4,
                    alpha=0.45,
                )
            if planned_trajectory == "next" and next_by_atom.get(idx) is traj:
                mask = ts >= t_now - 1e-15
                if np.count_nonzero(mask) >= 2:
                    _plot_wrapped_frequency_line(
                        ax,
                        ts[mask] * time_factor,
                        nus[mask],
                        1.0,
                        color=colors[idx],
                        linewidth=1.4,
                        alpha=0.35,
                    )

    total = max(float(base["total_duration"]), t_now)
    x_hi = max(total * time_factor, 1.0)
    ax.axvline(t_now * time_factor, color="#b8b8b8", linewidth=1.0)
    ax.set_xlim(0.0, x_hi)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel(f"t ({base['time_unit']})")
    ax.set_ylabel(base["frequency_label"])
    ax.grid(True, color="#e5e5e5", linewidth=0.6)


def _plot_wrapped_frequency_line(
    ax: Any,
    xs: np.ndarray,
    ys: np.ndarray,
    fsr: float,
    **plot_kwargs: Any,
) -> None:
    """Plot modulo-wrapped frequencies without drawing wrap jumps."""
    if len(xs) < 2:
        return
    jump = abs(float(fsr)) / 2.0
    breaks = np.flatnonzero(np.abs(np.diff(ys)) > jump) + 1
    starts = np.r_[0, breaks]
    ends = np.r_[breaks, len(xs)]
    for start, end in zip(starts, ends):
        if end - start >= 2:
            ax.plot(xs[start:end], ys[start:end], **plot_kwargs)


def _next_tone_trajectory_by_atom(
    trajectories: list[dict[str, Any]],
    t: float,
) -> dict[int, dict[str, Any]]:
    selected: dict[int, dict[str, Any]] = {}
    for traj in trajectories:
        idx = int(traj["idx"])
        if idx in selected:
            continue
        if float(traj["end_time"]) > t + 1e-15:
            selected[idx] = traj
    return selected


def _frequency_for_display(
    nu: float,
    spec: RIPASpec,
) -> float:
    return float((nu % spec.FSR1) / spec.FSR1)


def _frequencies_for_display(
    nus: np.ndarray,
    spec: RIPASpec,
) -> np.ndarray:
    return np.mod(nus, spec.FSR1) / spec.FSR1


def _draw_gaussian_blob(ax: Any, x: float, y: float, sigma: float) -> None:
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


def _rectangle_patch(xy: tuple[float, float], width: float, height: float) -> Any:
    from matplotlib.patches import Rectangle

    return Rectangle(
        xy,
        width,
        height,
        fill=False,
        linestyle="--",
        linewidth=1.0,
        edgecolor="#bbbbbb",
        zorder=1.5,
    )


def _all_grid_sites(N: int) -> np.ndarray:
    ii, jj = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    return np.column_stack([ii.ravel(), jj.ravel()]).astype(float)


def _site_extent_um(grid: Grid) -> tuple[float, float]:
    lo = (0.0 - grid.center) * grid.d
    hi = ((grid.N - 1.0) - grid.center) * grid.d
    return float(lo), float(hi)


def _resolve_static_traps(
    grid: Grid, static_traps: bool | Iterable[Site]
) -> np.ndarray:
    if static_traps is False or static_traps is None:
        return np.empty((0, 2), dtype=float)
    if static_traps is True:
        sites = _all_grid_sites(grid.N)
    else:
        sites = np.asarray(list(static_traps), dtype=float).reshape(-1, 2)
    return grid.ij_to_xy(sites)


def _resolve_atom_colors(
    atom_ids: list[int],
    atom_colors: Mapping[int, Any] | Iterable[Any] | None,
) -> list[Any]:
    plt = _load_pyplot()
    from matplotlib.colors import to_rgba

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
    from matplotlib.colors import to_rgba

    r, g, b, a = to_rgba(color)
    return (0.55 * r, 0.55 * g, 0.55 * b, a)


def _active_segment(
    atom: AtomTrajectory, t: float, *, tol: float = 1e-12
) -> Segment | None:
    for seg in atom.segments:
        if seg.duration > 0 and seg.start_time - tol <= t <= seg.end_time + tol:
            return seg
    return None


def _future_segments(
    atom: AtomTrajectory, t: float, *, tol: float = 1e-12
) -> Iterable[Segment]:
    for seg in atom.segments:
        if seg.duration > 0 and seg.end_time > t + tol:
            yield seg


def _active_aod_traps_xy(
    timeline: Any | None,
    ensemble: AtomEnsemble,
    t: float,
    *,
    tol: float = 1e-12,
) -> np.ndarray:
    """Positions of every active AOD trap, including empty intersections."""
    steps = getattr(timeline, "steps", None)
    if not steps:
        return np.empty((0, 2), dtype=float)

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
    if duration <= 0:
        return 1.0
    t = min(max(float(t_local), 0.0), float(duration))
    half = duration / 2.0
    if t <= half:
        return 2.0 * (t / duration) ** 2
    return 1.0 - 2.0 * ((duration - t) / duration) ** 2


def _sample_segment_xy(
    grid: Grid,
    seg: Segment,
    t0: float,
    t1: float,
    samples: int,
) -> np.ndarray:
    if t1 <= t0:
        return np.empty((0, 2), dtype=float)
    ts = np.linspace(t0, t1, max(2, int(samples)))
    ij = np.asarray([seg.position_at(float(t)) for t in ts], dtype=float)
    return grid.ij_to_xy(ij)


def _motion_blur_xy(
    grid: Grid,
    seg: Segment,
    t: float,
    samples: int,
    trail_duration: float | None,
) -> np.ndarray:
    """Speed-scaled tail ending at the atom's current position."""
    t_now = min(max(float(t), seg.start_time), seg.end_time)
    if t_now <= seg.start_time or seg.duration <= 0:
        return np.empty((0, 2), dtype=float)

    if trail_duration is not None:
        t0 = max(seg.start_time, t_now - float(trail_duration))
        return _sample_segment_xy(grid, seg, t0, t_now, samples)

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
    eps = min(max(seg.duration * 1e-3, 1e-12), 1e-6)
    t0 = max(seg.start_time, t - eps)
    t1 = min(seg.end_time, t + eps)
    if t1 <= t0:
        return 0.0
    p0 = seg.position_at(t0)
    p1 = seg.position_at(t1)
    return math.hypot(p1[0] - p0[0], p1[1] - p0[1]) / (t1 - t0)


def _segment_path_fraction(seg: Segment, t: float) -> float:
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
    seg: Segment,
    fraction: float,
    lo: float,
    hi: float,
) -> float:
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


def _tone_frequency(
    channel: ToneChannel,
    position_ij: tuple[float, float],
    spec: RIPASpec,
    *,
    segment: Segment,
) -> float:
    """RF frequency for the hardware channel that generated `segment`.

    RIPA row/col moves use the spectrometer mapping from `ripa_freq.py`.
    AOD lattice moves use one separable RF tone per selected row/column, so
    their displayed frequency is based only on that row or column coordinate.
    """
    if segment.channel == "aod":
        return _aod_rf_frequency(channel, position_ij, spec)
    return _ripa_eom_frequency(channel, position_ij, spec)


def _ripa_eom_frequency(
    channel: ToneChannel,
    position_ij: tuple[float, float],
    spec: RIPASpec,
) -> float:
    i, j = position_ij
    return nu_row(i, j, spec) if channel == "row" else nu_col(i, j, spec)


def _aod_rf_frequency(
    channel: ToneChannel,
    position_ij: tuple[float, float],
    spec: RIPASpec,
) -> float:
    """AOD tone for a row/column deflection, normalized to the same FSR."""
    i, j = position_ij
    coordinate = i if channel == "row" else j
    return (float(coordinate) * spec.fsr2()) % spec.FSR1


def _collect_tone_trajectories(
    ensemble: AtomEnsemble,
    spec: RIPASpec,
    samples_per_segment: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    row_trajs: list[dict[str, Any]] = []
    col_trajs: list[dict[str, Any]] = []

    for idx, atom in enumerate(ensemble.atomtrajs):
        for seg in atom.segments:
            if seg.duration <= 0:
                continue
            ts = np.linspace(seg.start_time, seg.end_time, samples_per_segment)
            ij = np.asarray([seg.position_at(float(t)) for t in ts], dtype=float)
            channels = _channels_for_segment(seg)
            if "row" in channels:
                row_trajs.append(
                    {
                        "idx": idx,
                        "atom_id": int(atom.atom_id),
                        "times": ts,
                        "start_time": float(seg.start_time),
                        "end_time": float(seg.end_time),
                        "freqs": np.asarray(
                            [
                                _tone_frequency("row", (i, j), spec, segment=seg)
                                for i, j in ij
                            ],
                            dtype=float,
                        ),
                    }
                )
            if "col" in channels:
                col_trajs.append(
                    {
                        "idx": idx,
                        "atom_id": int(atom.atom_id),
                        "times": ts,
                        "start_time": float(seg.start_time),
                        "end_time": float(seg.end_time),
                        "freqs": np.asarray(
                            [
                                _tone_frequency("col", (i, j), spec, segment=seg)
                                for i, j in ij
                            ],
                            dtype=float,
                        ),
                    }
                )
    return row_trajs, col_trajs


def _normalize_frequency_axes(axes: Any) -> tuple[Any, Any, Any, Any]:
    arr = np.asarray(axes, dtype=object)
    if arr.shape != (2, 2):
        raise ValueError("frequency axes must be shaped like (2, 2)")
    return arr[0, 0], arr[0, 1], arr[1, 0], arr[1, 1]


def _coerce_visualization_axes(
    axes: VisualizationAxes | tuple[Any, Any, Any, Any, Any],
) -> VisualizationAxes:
    if isinstance(axes, VisualizationAxes):
        return axes
    if len(axes) != 5:
        raise ValueError("axes must be a VisualizationAxes or five axes")
    return VisualizationAxes(*axes)


def _normalize_motion_items(
    motions: Mapping[str, Any] | Iterable[Any],
    labels: Iterable[str] | None,
) -> list[tuple[str, Any]]:
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


def _panel_value(value: Any, index: int, label: str) -> Any:
    if isinstance(value, Mapping):
        return value.get(label, True)
    return value


def _normalize_render_quality(quality: RenderQuality) -> RenderQuality:
    if quality in ("speed", "quality"):
        return quality
    raise ValueError("quality must be 'speed' or 'quality'")


def _resolve_quality_option(
    quality: RenderQuality | None,
    optimize: OptimizeMode | None,
) -> RenderQuality:
    if quality is not None:
        return _normalize_render_quality(quality)
    if optimize is not None:
        return _normalize_optimize(optimize)
    return "speed"


def _resolve_render_style(
    quality: RenderQuality,
    *,
    dpi: int | None,
    addressed_style: AddressedStyle | None,
    show_motion_blur: bool | None,
    trail_samples: int | None,
    tone_samples_per_segment: int | None,
    planned_samples_per_segment: int | None,
    atom_size: float | None,
    trap_size: float | None,
) -> RenderStyle:
    mode = _normalize_render_quality(quality)
    defaults = (
        RenderStyle(
            dpi=90,
            addressed_style="edge",
            show_motion_blur=False,
            trail_samples=0,
            tone_samples_per_segment=32,
            planned_samples_per_segment=18,
            atom_size=62.0,
            trap_size=10.0,
        )
        if mode == "speed"
        else RenderStyle(
            dpi=150,
            addressed_style="both",
            show_motion_blur=True,
            trail_samples=12,
            tone_samples_per_segment=96,
            planned_samples_per_segment=48,
            atom_size=75.0,
            trap_size=14.0,
        )
    )

    blur = defaults.show_motion_blur if show_motion_blur is None else show_motion_blur
    trails = defaults.trail_samples if trail_samples is None else int(trail_samples)
    if blur and trail_samples is None and trails == 0:
        trails = 5
    if not blur:
        trails = 0

    return RenderStyle(
        dpi=defaults.dpi if dpi is None else int(dpi),
        addressed_style=addressed_style or defaults.addressed_style,
        show_motion_blur=bool(blur),
        trail_samples=max(0, trails),
        tone_samples_per_segment=(
            defaults.tone_samples_per_segment
            if tone_samples_per_segment is None
            else int(tone_samples_per_segment)
        ),
        planned_samples_per_segment=(
            defaults.planned_samples_per_segment
            if planned_samples_per_segment is None
            else int(planned_samples_per_segment)
        ),
        atom_size=defaults.atom_size if atom_size is None else float(atom_size),
        trap_size=defaults.trap_size if trap_size is None else float(trap_size),
    )


def _format_time(t: float, unit: Literal["s", "ms", "us", "ns"]) -> str:
    value = t * _TIME_FACTORS[unit]
    if abs(value) >= 100:
        return f"{value:.1f} {unit}"
    if abs(value) >= 10:
        return f"{value:.2f} {unit}"
    return f"{value:.3f} {unit}"


def _normalize_optimize(optimize: OptimizeMode) -> RenderQuality:
    if optimize in ("speed", "performance"):
        return "speed"
    if optimize == "quality":
        return "quality"
    raise ValueError("optimize must be 'speed', 'performance', or 'quality'")


def _resolve_planned_trajectory_mode(
    show_planned: bool,
    planned_trajectory: PlannedTrajectoryMode | None,
) -> PlannedTrajectoryMode:
    if planned_trajectory is None:
        return "full" if show_planned else "none"
    if planned_trajectory in ("none", "full", "next"):
        return planned_trajectory
    raise ValueError("planned_trajectory must be 'none', 'full', or 'next'")


def _progress_iter(
    items: Iterable[Any],
    *,
    total: int,
    desc: str,
    enabled: bool,
) -> Iterable[Any]:
    if not enabled:
        yield from items
        return

    try:
        from tqdm.auto import tqdm
    except ImportError:
        for index, item in enumerate(items, start=1):
            if index == 1 or index == total or index % max(1, total // 10) == 0:
                print(f"{desc}: {index}/{total}", file=sys.stderr, flush=True)
            yield item
        return

    yield from tqdm(items, total=total, desc=desc, unit="frame")


def _render_worker_count(workers: int | None, task_count: int) -> int:
    if task_count <= 0:
        return 0
    if workers is not None:
        if workers <= 0:
            raise ValueError("workers must be positive")
        return min(int(workers), task_count)
    return min(os.cpu_count() or 1, task_count)


def _render_pool_context() -> Any | None:
    methods = mp.get_all_start_methods()
    if sys.platform.startswith("linux") and "fork" in methods:
        return mp.get_context("fork")
    if "spawn" in methods and _can_spawn_render_workers():
        return mp.get_context("spawn")
    return None


def _render_png_frames(
    frame_dir: Path,
    base: dict[str, Any],
    frames: list[dict[str, Any]],
    options: dict[str, Any],
    *,
    use_multiprocessing: bool,
    workers: int | None,
    show_progress: bool,
) -> list[Path]:
    frame_dir.mkdir(parents=True, exist_ok=True)
    frame_paths = [frame_dir / f"frame_{k:05d}.png" for k in range(len(frames))]
    tasks = list(zip(frames, frame_paths))

    worker_count = (
        _render_worker_count(workers, len(tasks)) if use_multiprocessing else 0
    )
    pool_context = _render_pool_context() if use_multiprocessing else None
    if pool_context is not None and worker_count > 1:
        try:
            chunk_size = max(1, len(tasks) // (worker_count * 4))
            with pool_context.Pool(
                processes=worker_count,
                initializer=_init_render_worker,
                initargs=(base, options),
            ) as pool:
                rendered = pool.imap_unordered(
                    _render_worker_frame, tasks, chunksize=chunk_size
                )
                list(
                    _progress_iter(
                        rendered,
                        total=len(tasks),
                        desc=f"render frames ({worker_count} workers)",
                        enabled=show_progress,
                    )
                )
            return frame_paths
        except Exception:
            for frame, path in _progress_iter(
                tasks,
                total=len(tasks),
                desc="render missing frames",
                enabled=show_progress,
            ):
                if not path.exists():
                    _render_payload_to_png(base, frame, path, options)
            return frame_paths

    for frame, path in _progress_iter(
        tasks,
        total=len(tasks),
        desc="render frames",
        enabled=show_progress,
    ):
        _render_payload_to_png(base, frame, path, options)
    return frame_paths


def _can_spawn_render_workers() -> bool:
    """Process pools need an importable `__main__` on Windows spawn."""

    import __main__

    main_file = getattr(__main__, "__file__", None)
    if not main_file or str(main_file).startswith("<"):
        return False
    if sys.argv and sys.argv[0] in {"", "-"}:
        return False
    return True


def _init_render_worker(base: dict[str, Any], options: dict[str, Any]) -> None:
    global _RENDER_BASE, _RENDER_OPTIONS
    _RENDER_BASE = base
    _RENDER_OPTIONS = options


def _render_worker_frame(task: tuple[dict[str, Any], Path]) -> str:
    if _RENDER_BASE is None or _RENDER_OPTIONS is None:
        raise RuntimeError("render worker was not initialized")
    frame, path = task
    _render_payload_to_png(_RENDER_BASE, frame, path, _RENDER_OPTIONS)
    return str(path)


def _render_payload_to_png(
    base: dict[str, Any],
    frame: dict[str, Any],
    path: Path,
    options: dict[str, Any],
) -> None:
    plt = _load_pyplot(force_agg=True)
    fig = plt.figure(
        figsize=options["figsize"], dpi=options["dpi"], constrained_layout=True
    )
    view = options.get("view", "detail")
    if view == "demo":
        ax = fig.add_subplot(1, 1, 1)
        _plot_atom_plane_payload(
            ax,
            base,
            frame,
            addressed_style=options["addressed_style"],
            show_atom_ids=bool(options["show_atom_ids"]),
            atom_size=float(options["atom_size"]),
            trap_size=float(options["trap_size"]),
            title=options.get("title"),
        )
    elif view == "benchmark":
        panels = list(base["panels"])
        gs = fig.add_gridspec(1, len(panels))
        for index, (panel, panel_frame) in enumerate(zip(panels, frame["panels"])):
            ax = fig.add_subplot(gs[0, index])
            title = f"{panel['label']}  -  t = {_format_time(frame['t'], 'us')}"
            _plot_atom_plane_payload(
                ax,
                panel["base"],
                panel_frame["frame"],
                addressed_style=options["addressed_style"],
                show_atom_ids=bool(options["show_atom_ids"]),
                atom_size=float(options["atom_size"]),
                trap_size=float(options["trap_size"]),
                title=title,
            )
        if options.get("title"):
            fig.suptitle(options["title"])
    else:
        gs = fig.add_gridspec(
            2,
            3,
            width_ratios=(2.5, 1.0, 1.0),
            height_ratios=(1.0, 1.0),
        )
        axes = VisualizationAxes(
            atom=fig.add_subplot(gs[:, 0]),
            row_tones=fig.add_subplot(gs[0, 1]),
            col_tones=fig.add_subplot(gs[0, 2]),
            row_trajectories=fig.add_subplot(gs[1, 1]),
            col_trajectories=fig.add_subplot(gs[1, 2]),
        )
        _plot_payload_frame(
            fig,
            axes,
            base,
            frame,
            addressed_style=options["addressed_style"],
            show_atom_ids=bool(options["show_atom_ids"]),
            planned_trajectory=options["planned_trajectory"],
            atom_size=float(options["atom_size"]),
            trap_size=float(options["trap_size"]),
            title=options.get("title"),
        )
    fig.savefig(path, dpi=options["dpi"], facecolor="white")
    plt.close(fig)


def _save_gif_from_pngs(
    frame_paths: list[Path],
    output_path: Path,
    *,
    fps: int,
    show_progress: bool,
) -> None:
    if not frame_paths:
        raise ValueError("no frames to save")

    duration_ms = int(round(1000 / fps))
    try:
        from PIL import Image

        images = [
            Image.open(path)
            for path in _progress_iter(
                frame_paths,
                total=len(frame_paths),
                desc="load gif frames",
                enabled=show_progress,
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
        return
    except ImportError:
        pass

    try:
        import imageio.v2 as imageio
    except ImportError as exc:
        raise RuntimeError("saving GIFs requires Pillow or imageio") from exc

    imageio.mimsave(
        output_path,
        [
            imageio.imread(path)
            for path in _progress_iter(
                frame_paths,
                total=len(frame_paths),
                desc="load gif frames",
                enabled=show_progress,
            )
        ],
        duration=1 / fps,
    )
