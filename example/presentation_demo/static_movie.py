"""Stationary-trap RIPA-tone movies.

Two demos share one renderer:
  - static scan: 100 frames, one site lit each, advancing
    (0,0) -> (1,0) -> ... -> (9,0) -> (0,1) -> ... -> (9,9). Tone
    sweeps monotonically across [0, FSR1].
  - tetris: T frames loaded from `tetris_movie.npy` (binary
    `(T, 10, 10)` mask); every lit site lights a halo + tick.

`render_gif(grid, masks, output_path)` takes a sequence of binary
(N, N) masks; per frame, every lit site paints a Gaussian halo on the
plane and a matching RIPA row tone tick (color-mapped) in the freq bar
above. Traps and tones are drawn directly via `src/visualization.py`
primitives -- no scheduler, no motion.

Usage:
    python example/presentation_demo/static_movie.py             # both
    python example/presentation_demo/static_movie.py --mode static
    python example/presentation_demo/static_movie.py --mode tetris
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
from tqdm.auto import tqdm  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.atom_config import Grid  # noqa: E402
from src.visualization import (  # noqa: E402
    Z_GRID_DOTS,
    _draw_gaussian_blob,
    _ripa_tone_freq,
    draw_grid_frame,
)

N = 10
GRID_SPACING_UM = 5.0
COLLISION_RADIUS_UM = 1.0
CHANNEL = "row"

FIGSIZE = (5.0, 5.6)
DPI = 150
FRAME_MS = 50        # display duration per frame (ms)
START_END_HOLD_MS = 0  # extra dwell on first + last frame so the loop is readable
TRAP_SCALE = 1.0
TRAP_ALPHA = 0.75

GRID_DOT_SIZE = 28.0
GRID_DOT_COLOR = "#9a9a9a"
GRID_DOT_ALPHA = 0.55

TONE_COLOR_STOPS = ["#ff5e39", "#f30a0a", "#dc0404", "#aa0f0f", "#83030f"]
TONE_CMAP = LinearSegmentedColormap.from_list("ripa_tone", TONE_COLOR_STOPS, N=256)

# Aim for ~this many sample frames in the palette composite so the red
# ramp is well-represented across the whole run; the stride is set per
# render based on frame count, capped so we never under-sample.
PALETTE_MAX_SAMPLES = 24

OUT_DIR = HERE / "render"
STATIC_PREFIX = "static_movie_row_scan"
TETRIS_PREFIX = "tetris_movie"
TETRIS_MOVIE_PATH = HERE / "tetris_movie.npy"
TETRIS_FRAME_DUR_MULT = 4  # hold each tetris frame 5x longer


def _tone_rgba(nu: float, alpha: float = 1.0) -> tuple[float, float, float, float]:
    r, g, b, _ = TONE_CMAP(float(np.clip(nu, 0.0, 1.0)))
    return (r, g, b, alpha)


def _draw_grid_dots(ax, grid: Grid) -> None:
    ii, jj = np.meshgrid(np.arange(grid.N), np.arange(grid.N), indexing="ij")
    xy = grid.ij_to_xy(np.column_stack([ii.ravel(), jj.ravel()]).astype(float))
    ax.scatter(
        xy[:, 0], xy[:, 1], s=GRID_DOT_SIZE, c=GRID_DOT_COLOR,
        alpha=GRID_DOT_ALPHA, linewidths=0, zorder=Z_GRID_DOTS,
    )


def _render_frame(grid: Grid, mask: np.ndarray) -> Image.Image:
    """One frame: every lit (i,j) -> Gaussian halo + matching freq tick."""
    fig = plt.figure(
        figsize=FIGSIZE, dpi=DPI, constrained_layout=True, facecolor="white",
    )
    gs = fig.add_gridspec(2, 1, height_ratios=(0.12, 1.0))
    freq_ax = fig.add_subplot(gs[0, 0], facecolor="white")
    plane_ax = fig.add_subplot(gs[1, 0], facecolor="white")

    sites = np.argwhere(mask == 1)
    nus = [_ripa_tone_freq(CHANNEL, float(i), float(j), grid.N) for i, j in sites]

    for nu in nus:
        freq_ax.vlines(nu, 0.05, 0.95, color=_tone_rgba(nu, alpha=1.0), linewidth=2.5)
    freq_ax.set_xlim(-0.08, 1.08)
    freq_ax.set_ylim(0.0, 1.0)
    freq_ax.set_axis_off()

    _draw_grid_dots(plane_ax, grid)
    for (i, j), nu in zip(sites, nus):
        x, y = grid.ij_to_xy([[float(i), float(j)]])[0]
        _draw_gaussian_blob(
            plane_ax, float(x), float(y),
            trap_scale=TRAP_SCALE, rgba=_tone_rgba(nu, alpha=TRAP_ALPHA),
        )
    draw_grid_frame(plane_ax, grid)
    plane_ax.set_axis_off()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor="white")
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf)
    img.load()
    return img.convert("RGB")


def render_gif(
    grid: Grid,
    masks: Iterable[np.ndarray],
    output_path: Path,
    *,
    frame_dur_mult: int = 1,
) -> None:
    masks = list(masks)
    n_frames = len(masks)
    if n_frames == 0:
        raise ValueError("masks is empty")
    frame_ms = FRAME_MS * int(frame_dur_mult)
    hold_ms = max(0, int(START_END_HOLD_MS))

    frames: list[Image.Image] = []
    durations: list[int] = []
    desc = f"render {output_path.name}"
    for k, mask in enumerate(tqdm(masks, desc=desc, unit="frame")):
        frames.append(_render_frame(grid, mask))
        dur = frame_ms
        if hold_ms > 0 and (k == 0 or k == n_frames - 1):
            dur += hold_ms
        durations.append(dur)

    # Stride-sample frames across the run so the palette covers the
    # whole red ramp; a single mid-frame would starve one tint and
    # Floyd-Steinberg would visibly band at the start/end colors.
    stride = max(1, n_frames // PALETTE_MAX_SAMPLES)
    sample_idx = list(range(0, n_frames, stride))
    if sample_idx[-1] != n_frames - 1:
        sample_idx.append(n_frames - 1)
    samples = [frames[k] for k in sample_idx]
    w, h = samples[0].size
    composite = Image.new("RGB", (w, h * len(samples)), "white")
    for k, f in enumerate(samples):
        composite.paste(f, (0, h * k))
    palette = composite.quantize(
        colors=256, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE,
    )
    composite.close()
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


def _static_scan_masks(grid_n: int) -> Iterable[np.ndarray]:
    for j in range(grid_n):
        for i in range(grid_n):
            mask = np.zeros((grid_n, grid_n), dtype=np.uint8)
            mask[i, j] = 1
            yield mask


def _tetris_masks() -> Iterable[np.ndarray]:
    movie = np.load(TETRIS_MOVIE_PATH)
    if movie.ndim != 3 or movie.shape[1:] != (N, N):
        raise ValueError(f"expected (T, {N}, {N}) movie, got {movie.shape}")
    for k in range(movie.shape[0]):
        yield np.fliplr(movie[k].T)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stationary RIPA-tone trap movies: static scan + tetris."
    )
    parser.add_argument(
        "--mode", choices=("both", "static", "tetris"), default="both",
        help="which GIF to render (default: both)",
    )
    args = parser.parse_args()

    grid = Grid(N=N, d=GRID_SPACING_UM, rc=COLLISION_RADIUS_UM)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.mode in ("both", "static"):
        out = OUT_DIR / f"{STATIC_PREFIX}.gif"
        render_gif(grid, _static_scan_masks(N), out)
        print(f"wrote {out}")
    if args.mode in ("both", "tetris"):
        out = OUT_DIR / f"{TETRIS_PREFIX}.gif"
        render_gif(grid, _tetris_masks(), out, frame_dur_mult=TETRIS_FRAME_DUR_MULT)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
