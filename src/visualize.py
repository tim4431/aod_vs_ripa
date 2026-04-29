"""Visualization: atom plane, EOM tone panels, trajectory traces.

All draw helpers take an explicit ``ax`` so they compose into any layout.
``animate(...)`` assembles the panels shown in ``doc/vis_example.png``
and saves a GIF.

The data model is the AtomEnsemble: each atom carries its own timeline
of segments. We sample positions at frame times via
``ensemble.positions_at(t)`` and decide who is "currently held by the
tweezer" by checking which atoms are mid-segment at that t.

Performance switch: ``fast=True`` uses a thin red edge marker for held
atoms (cheap); ``fast=False`` adds a soft gaussian blob (prettier).
"""

from __future__ import annotations

import multiprocessing as mp
import os
import tempfile
from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np

from .atom_trajectory import AtomEnsemble
from .atoms import Grid
from .ripa_freq import RIPASpec, nu_col, nu_row
from .sequence import Sequence
from .segments import Segment

# --- Atom plane -------------------------------------------------------------


def draw_grid(ax, grid: Grid, *, color: str = "lightgray", size: float = 4.0):
    """Faint dot for every grid site, plus a dashed bounding box."""
    N, d = grid.N, grid.d
    c = (N - 1) / 2.0
    xs = (np.arange(N) - c) * d
    X, Y = np.meshgrid(xs, xs, indexing="ij")
    ax.scatter(X, Y, s=size, color=color, zorder=0)
    half = (N - 1) / 2.0 * d + d / 2.0
    ax.plot(
        [-half, half, half, -half, -half],
        [-half, -half, half, half, -half],
        "--",
        color="gray",
        lw=0.6,
        zorder=0,
    )
    ax.set_aspect("equal")
    ax.set_xlabel("x (um)")
    ax.set_ylabel("y (um)")


def _held_atom_indices(ensemble: AtomEnsemble, t: float) -> set[int]:
    """Atoms currently in motion at global time t (= 'in tweezer')."""
    held: set[int] = set()
    for atom in ensemble.atoms:
        for seg in atom.segments:
            if seg.start_time <= t <= seg.end_time and seg.duration > 0:
                held.add(atom.atom_id)
                break
    return held


def draw_atoms(
    ax,
    ensemble: AtomEnsemble,
    t: float,
    *,
    held: Optional[Iterable[int]] = None,
    static_traps: Iterable[tuple[float, float]] = (),
    colors: Optional[list] = None,
    size: float = 60,
    static_color: str = "tab:blue",
    edgecolor_held: str = "red",
    fast: bool = True,
):
    """Atoms at time t. Held atoms get a red edge (and an optional blob)."""
    xy = ensemble.xy_at(t)
    held_set = set(
        int(k) for k in (held if held is not None else _held_atom_indices(ensemble, t))
    )
    edges = [edgecolor_held if k in held_set else "none" for k in range(len(xy))]
    lws = [1.5 if k in held_set else 0.0 for k in range(len(xy))]
    face = colors if colors is not None else ["k"] * len(xy)
    if not fast and held_set:
        held_xy = xy[sorted(held_set)]
        ax.scatter(
            held_xy[:, 0], held_xy[:, 1], s=size * 6, c="red", alpha=0.18, zorder=2
        )
    ax.scatter(
        xy[:, 0], xy[:, 1], s=size, c=face, edgecolors=edges, linewidths=lws, zorder=3
    )
    if static_traps:
        st = np.array([ensemble.grid.ij_to_xy(i, j) for i, j in static_traps])
        ax.scatter(
            st[:, 0],
            st[:, 1],
            s=size * 0.6,
            facecolor="none",
            edgecolor=static_color,
            lw=1.0,
            zorder=2,
        )


def draw_motion_blur(
    ax, ensemble: AtomEnsemble, t: float, *, window: float = 5e-6, alpha: float = 0.25
):
    """Short trail from t-window to t for atoms in motion (speed cue)."""
    xy_now = ensemble.xy_at(t)
    xy_before = ensemble.xy_at(max(0.0, t - window))
    for (x0, y0), (x1, y1) in zip(xy_before, xy_now):
        if (x0, y0) != (x1, y1):
            ax.plot([x0, x1], [y0, y1], color="0.5", alpha=alpha, lw=2, zorder=2)


def draw_planned_trajectories(
    ax, ensemble: AtomEnsemble, t: float, *, n_samples: int = 30, alpha: float = 0.4
):
    """Faint dashed line for each atom's *future* path beyond t."""
    grid = ensemble.grid
    c = (grid.N - 1) / 2.0
    for atom in ensemble.atoms:
        for seg in atom.segments:
            if seg.end_time < t or seg.duration <= 0:
                continue
            t0 = max(seg.start_time, t)
            ts = np.linspace(t0, seg.end_time, n_samples)
            pts = np.array([seg.position_at(float(tt)) for tt in ts])
            pts = (pts - c) * grid.d
            ax.plot(
                pts[:, 0], pts[:, 1], "--", color="0.4", alpha=alpha, lw=0.8, zorder=1
            )


# --- Frequency panels --------------------------------------------------------


def _active_segments(ensemble: AtomEnsemble, t: float, channel: str) -> list[Segment]:
    """Currently-active segments of a given channel ('row' or 'col')."""
    out: list[Segment] = []
    for atom in ensemble.atoms:
        for seg in atom.segments:
            if (
                seg.channel == channel
                and seg.start_time <= t <= seg.end_time
                and seg.duration > 0
            ):
                out.append(seg)
    return out


def draw_tone_stems(
    ax, freqs: list[float], *, FSR1: float, color: str = "k", label_each: bool = False
):
    """Vertical stems (one per active tone) on the freq axis."""
    for k, f in enumerate(freqs):
        ax.vlines(f, 0, 1, colors=color, lw=1.2)
        if label_each:
            ax.text(f, 1.02, str(k), ha="center", va="bottom", fontsize=7)
    ax.set_xlim(-FSR1 / 2, FSR1 / 2)
    ax.set_ylim(0, 1.1)
    ax.set_xlabel("Δν (mod FSR1)")
    ax.set_ylabel("amplitude")


def draw_tone_trajectories(
    ax,
    ensemble: AtomEnsemble,
    channel: str,
    *,
    spec: RIPASpec,
    t: float,
    n_samples: int = 50,
):
    """Tone frequency vs time for every segment of `channel` (full timeline)."""
    fn = nu_row if channel == "row" else nu_col
    for atom in ensemble.atoms:
        for seg in atom.segments:
            if seg.channel != channel or seg.duration <= 0:
                continue
            ts = np.linspace(seg.start_time, seg.end_time, n_samples)
            ij = np.array([seg.position_at(float(tt)) for tt in ts])
            freqs = [fn(i, j, spec) for (i, j) in ij]
            ax.plot(ts, freqs, lw=0.8)
    ax.axvline(t, color="0.5", lw=0.6)
    ax.set_xlabel("t (s)")
    ax.set_ylabel("Δν (mod FSR1)")
    ax.set_ylim(-spec.FSR1 / 2, spec.FSR1 / 2)


# --- Animation --------------------------------------------------------------


@dataclass
class AnimationOptions:
    fps: int = 30
    fast: bool = True
    pad_seconds: float = 1.0  # freeze before & after
    out_path: str = "sequence.gif"
    n_workers: int = max(1, (os.cpu_count() or 2) - 1)
    spec: Optional[RIPASpec] = None  # RIPA tone panels (omit for AOD)
    show_planned: bool = True
    static_traps: tuple = ()  # external SLM traps to render


def _render_frame(args):
    """Worker: render one frame to PNG. Top-level so it pickles."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    idx, t, ensemble, opts_dict, out_dir = args
    fast = opts_dict["fast"]
    show_planned = opts_dict["show_planned"]
    spec_dict = opts_dict["spec"]
    static_traps = opts_dict["static_traps"]
    spec = RIPASpec(**spec_dict) if spec_dict is not None else None

    fig = plt.figure(figsize=(11, 5))
    if spec is not None:
        gs = fig.add_gridspec(2, 3, width_ratios=[2, 1, 1])
        ax_atoms = fig.add_subplot(gs[:, 0])
        ax_row = fig.add_subplot(gs[0, 1])
        ax_col = fig.add_subplot(gs[0, 2])
        ax_row_t = fig.add_subplot(gs[1, 1])
        ax_col_t = fig.add_subplot(gs[1, 2])
    else:
        ax_atoms = fig.add_subplot(1, 1, 1)
        ax_row = ax_col = ax_row_t = ax_col_t = None

    draw_grid(ax_atoms, ensemble.grid)
    draw_atoms(ax_atoms, ensemble, t, fast=fast, static_traps=static_traps)
    draw_motion_blur(ax_atoms, ensemble, t)
    if show_planned:
        draw_planned_trajectories(ax_atoms, ensemble, t)
    ax_atoms.set_title(f"t = {t*1e6:.1f} us")

    if spec is not None:
        held = _held_atom_indices(ensemble, t)
        held_pos = [ensemble.atoms[k].position_at(t) for k in held]
        # Split tones by which channel is currently driving each atom.
        row_active = {
            atom.atom_id
            for atom in ensemble.atoms
            for seg in atom.segments
            if seg.channel == "row"
            and seg.start_time <= t <= seg.end_time
            and seg.duration > 0
        }
        col_active = {
            atom.atom_id
            for atom in ensemble.atoms
            for seg in atom.segments
            if seg.channel == "col"
            and seg.start_time <= t <= seg.end_time
            and seg.duration > 0
        }
        row_pos = [ensemble.atoms[k].position_at(t) for k in row_active]
        col_pos = [ensemble.atoms[k].position_at(t) for k in col_active]
        draw_tone_stems(
            ax_row, [nu_row(i, j, spec) for i, j in row_pos], FSR1=spec.FSR1
        )
        draw_tone_stems(
            ax_col, [nu_col(i, j, spec) for i, j in col_pos], FSR1=spec.FSR1
        )
        ax_row.set_title("row EOM (controls x)")
        ax_col.set_title("col EOM (controls y)")
        draw_tone_trajectories(ax_row_t, ensemble, "row", spec=spec, t=t)
        draw_tone_trajectories(ax_col_t, ensemble, "col", spec=spec, t=t)
        del held_pos  # held_pos not directly drawn; tones are computed from row/col_pos.

    fig.tight_layout()
    path = os.path.join(out_dir, f"frame_{idx:05d}.png")
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def animate(seq: Sequence, opts: AnimationOptions = AnimationOptions()) -> str:
    """Render a Sequence to GIF. Returns the output path."""
    import imageio.v2 as imageio

    ensemble = seq.ensemble
    total = ensemble.total_duration()
    fps = opts.fps
    pad_frames = int(round(opts.pad_seconds * fps))
    n_frames = int(round(total * fps)) + 2 * pad_frames

    spec_dict = None
    if opts.spec is not None:
        spec_dict = dict(N=opts.spec.N, FSR1=opts.spec.FSR1, FSR2=opts.spec.FSR2)
    opts_dict = dict(
        fast=opts.fast,
        show_planned=opts.show_planned,
        spec=spec_dict,
        static_traps=tuple(opts.static_traps),
    )

    # Build (idx, t) per frame; pad freezes before/after.
    times: list[float] = []
    for f in range(n_frames):
        if f < pad_frames:
            times.append(0.0)
        elif f >= n_frames - pad_frames:
            times.append(total)
        else:
            times.append((f - pad_frames) / fps)

    with tempfile.TemporaryDirectory() as tmp:
        tasks = [(idx, t, ensemble, opts_dict, tmp) for idx, t in enumerate(times)]
        if opts.n_workers > 1:
            with mp.Pool(opts.n_workers) as pool:
                paths = pool.map(_render_frame, tasks)
        else:
            paths = [_render_frame(a) for a in tasks]

        with imageio.get_writer(opts.out_path, mode="I", fps=fps) as w:
            for p in paths:
                w.append_data(imageio.imread(p))
    return opts.out_path
