"""Shared panel rendering for presentation_demo gifs.

Channel-colored Gaussian trap halos with a QUINTIC_MIN_JERK intensity
envelope around each segment's motion window. Layers on top of
`src/visualization.py` primitives (`draw_grid_dots`, `draw_grid_frame`,
`draw_atoms`, `_draw_gaussian_blob`); the bits that don't exist there
-- per-channel coloring and per-segment intensity ramps -- live here.

Demos import the constants, `render_gif`, and (if they need a frame)
`draw_panel` / `_render_frame`; they only own their own `build_sequence`.
"""

from __future__ import annotations

import io
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
from tqdm.auto import tqdm  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.moving_sequence import MovingSequence  # noqa: E402
from src.segments import QUINTIC_MIN_JERK, Segment  # noqa: E402
from src.visualization import (  # noqa: E402
    _draw_gaussian_blob,
    draw_atoms,
    draw_grid_dots,
    draw_grid_frame,
)

GRID_SPACING_UM = 5.0
COLLISION_RADIUS_UM = 1.0

ROW_RGBA = (1.0, 40 / 255, 40 / 255, 0.5)   # #ff2828
COL_RGBA = (41 / 255, 176 / 255, 1.0, 0.5)  # #29b0ff

RAMP_TIME = 40e-6
# Stationary handoff window. Equal to RAMP_TIME so the col ramp-out and
# row ramp-in share one window and their summed intensity stays ~constant.
HANDOFF_HOLD = RAMP_TIME
RAMP_PROFILE = QUINTIC_MIN_JERK

FIGSIZE = (5.0, 5.0)
DPI = 150
FPS = 30
TIME_DILATION = 1.5e4
HOLD_SECONDS = 1.0
TRAP_SCALE = 1.0
ATOM_SCALE = 2.0
ATOM_EDGE_LINEWIDTH = 2.0


def _channel_rgba(channel: str | None) -> tuple[float, float, float, float] | None:
    if channel == "row":
        return ROW_RGBA
    if channel == "col":
        return COL_RGBA
    return None


def _trap_intensity(seg: Segment, t: float) -> float:
    """Smooth on/off envelope around `seg`'s motion window: ramp up over
    RAMP_TIME before motion, plateau during motion, ramp down over
    RAMP_TIME after motion. Profile is `RAMP_PROFILE`."""
    on_start = seg.start_time - RAMP_TIME
    on_end = seg.end_time + RAMP_TIME
    if t <= on_start or t >= on_end:
        return 0.0
    if t < seg.start_time:
        return float(RAMP_PROFILE.value((t - on_start) / RAMP_TIME))
    if t > seg.end_time:
        return float(RAMP_PROFILE.value((on_end - t) / RAMP_TIME))
    return 1.0


def _trap_position(seg: Segment, t: float) -> tuple[float, float]:
    """Where the trap is addressing at `t`: parked at start_pos while
    ramping in, follows the atom during motion, parked at end_pos while
    ramping out."""
    if t < seg.start_time:
        return seg.start_pos
    if t > seg.end_time:
        return seg.end_pos
    return seg.position_at(t)


def draw_panel(ax, sequence: MovingSequence, t: float) -> None:
    ensemble = sequence.ensemble
    grid = ensemble.grid

    draw_grid_dots(ax, grid)

    for atom in ensemble.atomtrajs:
        for seg in atom.segments:
            intensity = _trap_intensity(seg, t)
            if intensity <= 0.0:
                continue
            base_rgba = _channel_rgba(seg.channel)
            if base_rgba is None:
                continue
            x, y = grid.ij_to_xy(
                np.asarray([_trap_position(seg, t)], dtype=float)
            )[0]
            rgba = (
                base_rgba[0], base_rgba[1], base_rgba[2],
                base_rgba[3] * intensity,
            )
            _draw_gaussian_blob(
                ax, float(x), float(y),
                trap_scale=TRAP_SCALE, rgba=rgba,
            )

    draw_atoms(
        ax, ensemble, t,
        colors=["gray"] * len(ensemble.atomtrajs),
        atom_scale=ATOM_SCALE,
        edge_linewidth=ATOM_EDGE_LINEWIDTH,
    )

    draw_grid_frame(ax, grid)
    ax.set_axis_off()


def _render_frame(sequence: MovingSequence, t: float) -> Image.Image:
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI, constrained_layout=True)
    draw_panel(ax, sequence, t)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor="white")
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf)
    img.load()
    return img.convert("RGB")


def _build_palette(frames: list[Image.Image]) -> Image.Image:
    """256-color palette from a col-phase + row-phase composite so neither
    tint gets starved by the quantizer."""
    n = len(frames)
    col_frame = frames[max(0, n // 4)]
    row_frame = frames[min(n - 1, (3 * n) // 4)]
    combined = Image.new(
        "RGB", (col_frame.width, col_frame.height + row_frame.height), "white",
    )
    combined.paste(col_frame, (0, 0))
    combined.paste(row_frame, (0, col_frame.height))
    palette = combined.quantize(
        colors=256, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE,
    )
    combined.close()
    return palette


def render_gif(sequence: MovingSequence, output_path: Path) -> None:
    # Window stretches beyond the motion span so leading fade-ins and
    # trailing fade-outs are both visible.
    t_start = -RAMP_TIME
    t_end = sequence.total_duration() + RAMP_TIME
    frame_dt = 1.0 / (TIME_DILATION * FPS)
    n_frames = max(2, int(math.ceil((t_end - t_start) / frame_dt)) + 1)
    times = np.linspace(t_start, t_end, n_frames)

    frame_ms = max(1, int(round(1000.0 / FPS)))
    hold_ms = max(0, int(round(HOLD_SECONDS * 1000.0)))

    frames: list[Image.Image] = []
    durations: list[int] = []

    if hold_ms > 0:
        frames.append(_render_frame(sequence, t_start - 1e-9))
        durations.append(hold_ms)
    for t in tqdm(times, desc="render frames", unit="frame"):
        frames.append(_render_frame(sequence, float(t)))
        durations.append(frame_ms)
    if hold_ms > 0:
        durations[-1] += hold_ms

    palette = _build_palette(frames)
    # Floyd-Steinberg dithers the gaussian gradients across the 256-color
    # palette so the red and blue halos render smoothly instead of banding.
    quantized = [
        f.quantize(palette=palette, dither=Image.Dither.FLOYDSTEINBERG)
        for f in frames
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    quantized[0].save(
        output_path,
        save_all=True, append_images=quantized[1:],
        duration=durations, loop=0, optimize=True,
    )
    for img in frames:
        img.close()
    for img in quantized:
        img.close()
    palette.close()
