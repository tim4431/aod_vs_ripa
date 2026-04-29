"""Visualization: atom plane + EOM tone panels + trajectory traces.

All draw helpers take an explicit `ax` so they can be composed into any
figure layout. The `animate(...)` entry point assembles the panels shown
in `doc/vis_example.png` and saves a GIF.

Performance switch: `fast=True` uses simple markers and disables the
gaussian-blob render of the tweezer; `fast=False` draws filled gaussians
(slower, prettier).
"""
from __future__ import annotations

import multiprocessing as mp
import os
import tempfile
from dataclasses import dataclass, field
from typing import Iterable, Optional

import numpy as np

from .atoms import AtomConfig
from .movement import AODStep, RIPAStep
from .ripa_freq import RIPASpec, nu_col, nu_row
from .sequence import Sequence


# --- Atom plane -------------------------------------------------------------

def draw_grid(ax, cfg: AtomConfig, *, color: str = "lightgray", size: float = 4.0):
    """Faint dot for every grid site."""
    N, d = cfg.grid.N, cfg.grid.d
    c = (N - 1) / 2.0
    xs = (np.arange(N) - c) * d
    X, Y = np.meshgrid(xs, xs, indexing="ij")
    ax.scatter(X, Y, s=size, color=color, zorder=0)
    # Brillouin-zone-like dashed bounding box.
    half = (N - 1) / 2.0 * d + d / 2.0
    ax.plot([-half, half, half, -half, -half],
            [-half, -half, half, half, -half],
            "--", color="gray", lw=0.6, zorder=0)
    ax.set_aspect("equal")
    ax.set_xlabel("x (um)")
    ax.set_ylabel("y (um)")


def draw_atoms(ax, cfg: AtomConfig, *,
               held_indices: Iterable[int] = (),
               colors: Optional[list] = None,
               edgecolor_held: str = "red",
               static_color: str = "tab:blue",
               size: float = 60):
    """Draw atoms; atoms in `held_indices` get a red edge to mark tweezer addressing.

    `colors` is per-atom face color (length = n_atoms). `held_indices`
    uses the cheap edge-color marker (the performance-friendly switch).
    """
    xy = cfg.xy()
    held = set(int(i) for i in held_indices)
    edges = [edgecolor_held if k in held else "none" for k in range(cfg.n_atoms)]
    lws = [1.5 if k in held else 0.0 for k in range(cfg.n_atoms)]
    face = colors if colors is not None else ["k"] * cfg.n_atoms
    ax.scatter(xy[:, 0], xy[:, 1], s=size, c=face,
               edgecolors=edges, linewidths=lws, zorder=3)
    if cfg.static_traps:
        st = np.array([cfg.grid.ij_to_xy(i, j) for i, j in cfg.static_traps])
        ax.scatter(st[:, 0], st[:, 1], s=size * 0.6,
                   facecolor="none", edgecolor=static_color, lw=1.0, zorder=2)


def draw_motion_blur(ax, cfg: AtomConfig, prev_xy: np.ndarray, *, alpha: float = 0.25):
    """Short trail from previous frame to current frame as a blur cue."""
    cur = cfg.xy()
    for (x0, y0), (x1, y1) in zip(prev_xy, cur):
        if (x0, y0) != (x1, y1):
            ax.plot([x0, x1], [y0, y1], color="0.5", alpha=alpha, lw=2, zorder=2)


def draw_planned_trajectories(ax, cfg: AtomConfig, step, *, n_samples: int = 30,
                              alpha: float = 0.4):
    """Faint dashed line for the planned future path of each moving atom."""
    if isinstance(step, AODStep):
        step = step.to_ripa_step(cfg)
    if not isinstance(step, RIPAStep):
        return
    ts = np.linspace(0.0, max(step.duration, 1e-12), n_samples)
    for k, tr in zip(step.atom_indices, step.trajectories):
        path = tr.sample(ts)
        c = (cfg.grid.N - 1) / 2.0
        path = (path - c) * cfg.grid.d
        ax.plot(path[:, 0], path[:, 1], "--", color="0.4", alpha=alpha, lw=0.8, zorder=1)


# --- Frequency panels --------------------------------------------------------

def draw_tone_stems(ax, freqs: list[float], *, FSR1: float, color: str = "k",
                    label_each: bool = False):
    """Vertical stems at each tone frequency in [-FSR1/2, +FSR1/2)."""
    for k, f in enumerate(freqs):
        ax.vlines(f, 0, 1, colors=color, lw=1.2)
        if label_each:
            ax.text(f, 1.02, str(k), ha="center", va="bottom", fontsize=7)
    ax.set_xlim(-FSR1 / 2, FSR1 / 2)
    ax.set_ylim(0, 1.1)
    ax.set_xlabel("Δν (mod FSR1)")
    ax.set_ylabel("amplitude")


def draw_tone_trajectories(ax, channel_history: list[tuple[float, list[float]]],
                           *, FSR1: float):
    """Each tone's freq vs time (one trace per active tone-id).

    `channel_history` = list of (t, [freq, ...]) snapshots.
    """
    if not channel_history:
        return
    ts = [t for t, _ in channel_history]
    # Pad each snapshot to the same length so we can stack into an array.
    width = max(len(f) for _, f in channel_history)
    M = np.full((len(ts), width), np.nan)
    for r, (_, f) in enumerate(channel_history):
        M[r, : len(f)] = f
    for col in range(width):
        ax.plot(ts, M[:, col], lw=0.8)
    ax.set_xlabel("t (s)")
    ax.set_ylabel("Δν (mod FSR1)")
    ax.set_ylim(-FSR1 / 2, FSR1 / 2)


# --- Animation --------------------------------------------------------------

@dataclass
class AnimationOptions:
    fps: int = 30
    fast: bool = True            # edge-color tweezer marker vs gaussian blob
    pad_seconds: float = 1.0     # freeze frames before & after sequence
    out_path: str = "sequence.gif"
    n_workers: int = max(1, (os.cpu_count() or 2) - 1)
    spec: Optional[RIPASpec] = None      # for tone panels (RIPA only)
    show_planned: bool = True


def _render_frame(args):
    """Worker: render a single frame to PNG. Top-level so it pickles."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    (idx, t, cfg_state, active_step, prev_xy, opts_dict, out_dir) = args
    fast = opts_dict["fast"]
    show_planned = opts_dict["show_planned"]
    spec_dict = opts_dict["spec"]

    fig = plt.figure(figsize=(11, 5))
    gs = fig.add_gridspec(2, 3, width_ratios=[2, 1, 1])
    ax_atoms = fig.add_subplot(gs[:, 0])
    ax_row = fig.add_subplot(gs[0, 1])
    ax_col = fig.add_subplot(gs[0, 2])
    ax_row_t = fig.add_subplot(gs[1, 1])
    ax_col_t = fig.add_subplot(gs[1, 2])

    draw_grid(ax_atoms, cfg_state)
    held = []
    if active_step is not None:
        held = list(active_step.atom_indices) if isinstance(active_step, RIPAStep) \
            else list(active_step.to_ripa_step(cfg_state).atom_indices)
    draw_atoms(ax_atoms, cfg_state, held_indices=held)
    if prev_xy is not None:
        draw_motion_blur(ax_atoms, cfg_state, prev_xy)
    if show_planned and active_step is not None:
        draw_planned_trajectories(ax_atoms, cfg_state, active_step)
    ax_atoms.set_title(f"t = {t*1e6:.1f} us")

    # Tone panels (only meaningful if a RIPASpec is provided).
    if spec_dict is not None:
        spec = RIPASpec(**spec_dict)
        held_pos = cfg_state.positions[held] if held else np.empty((0, 2))
        row_freqs = [nu_row(i, j, spec) for i, j in held_pos]
        col_freqs = [nu_col(i, j, spec) for i, j in held_pos]
        draw_tone_stems(ax_row, row_freqs, FSR1=spec.FSR1)
        draw_tone_stems(ax_col, col_freqs, FSR1=spec.FSR1)
        ax_row.set_title("row EOM (controls x)")
        ax_col.set_title("col EOM (controls y)")
    ax_row_t.axis("off")
    ax_col_t.axis("off")

    fig.tight_layout()
    path = os.path.join(out_dir, f"frame_{idx:05d}.png")
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def animate(seq: Sequence, opts: AnimationOptions = AnimationOptions()) -> str:
    """Render a Sequence to GIF. Returns the output path."""
    import imageio.v2 as imageio  # lazy import to keep test envs lightweight

    timed = seq.timed()
    total = seq.total_duration()
    fps = opts.fps
    pad_frames = int(round(opts.pad_seconds * fps))
    n_frames = int(round(total * fps)) + 2 * pad_frames

    # Build (t, active_step, cfg_at_t, prev_xy) per frame.
    frames_meta = []
    cfg_at_t = seq.initial.copy()
    prev_xy = cfg_at_t.xy().copy()

    for f in range(n_frames):
        if f < pad_frames:
            t = 0.0
        elif f >= n_frames - pad_frames:
            t = total
        else:
            t = (f - pad_frames) / fps

        # Find which step is active and propagate config to its start.
        cfg_state = seq.initial.copy()
        active = None
        for ts in timed:
            if t >= ts.end_time:
                cfg_state = ts.step.apply(cfg_state)
                continue
            if t >= ts.start_time:
                active = ts.step
                # For frame rendering we show atoms at their *step start*
                # config and let `draw_planned_trajectories` show the path.
                break

        frames_meta.append((f, t, cfg_state, active, prev_xy.copy()))
        prev_xy = cfg_state.xy().copy()

    spec_dict = None
    if opts.spec is not None:
        spec_dict = dict(N=opts.spec.N, FSR1=opts.spec.FSR1, FSR2=opts.spec.FSR2)
    opts_dict = dict(fast=opts.fast, show_planned=opts.show_planned, spec=spec_dict)

    with tempfile.TemporaryDirectory() as tmp:
        tasks = [(idx, t, cfg, step, prev, opts_dict, tmp)
                 for (idx, t, cfg, step, prev) in frames_meta]
        if opts.n_workers > 1:
            with mp.Pool(opts.n_workers) as pool:
                paths = pool.map(_render_frame, tasks)
        else:
            paths = [_render_frame(a) for a in tasks]

        with imageio.get_writer(opts.out_path, mode="I", fps=fps) as w:
            for p in paths:
                w.append_data(imageio.imread(p))
    return opts.out_path
