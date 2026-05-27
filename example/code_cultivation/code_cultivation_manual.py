"""Manual AOD movement script with CZ-beam visualization.

The SCRIPT below is a list of **sync batches**. Each batch is a list of
operations that fire at the same start_time via `MovingSequence.append_sync_batch`
(same pattern as `example/presentation_demo/async_l_to_line.py`); the batch
ends when its longest sub-step ends, and the next batch begins from there.

Op forms:
    ("mv", atom_name, (x, y))   # AOD single-atom move to absolute coords
    ("cz", atom_a, atom_b)      # CZ dwell on the (already adjacent) pair

The renderer draws, on top of the normal atom/trap layer, a dashed Rydberg-
blockade ring around each atom in an active CZ and a Gaussian "beam" glow
midway between them. The CZ dwell is only ~1 us, so the overlay carries a
QUINTIC ramp-in / ramp-out envelope of CZ_VIZ_RAMP so it stays on-screen
long enough to read.

Usage:
    python example/code_cultivation/code_cultivation_manual.py
"""

from __future__ import annotations

import io
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Circle  # noqa: E402
from PIL import Image  # noqa: E402
from tqdm.auto import tqdm  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.atom_config import AtomConfig, Grid  # noqa: E402
from src.movement import AODStep, Step  # noqa: E402
from src.moving_sequence import MovingSequence  # noqa: E402
from src.segments import QUINTIC_MIN_JERK, make_hold  # noqa: E402
from src.visualization import (  # noqa: E402
    _active_aod_traps_xy,
    _draw_gaussian_blob,
    draw_atoms,
    draw_grid_dots,
    draw_grid_frame,
)


# ----------------------------------------------------------------------------- #
# Layout + timing
# ----------------------------------------------------------------------------- #

GRID = Grid(N=11, d=3.0, rc=0.5)

# Atom name -> initial (x, y) grid site. r* are Rot(3) data on a coarse grid;
# a* are ancilla atoms sitting in the plaquette centers.
INITIAL: dict[str, tuple[float, float]] = {
    "r00": (1.0, 1.0),  "r01": (1.0, 5.0),  "r02": (1.0, 9.0),
    "r10": (5.0, 1.0),  "r11": (5.0, 5.0),  "r12": (5.0, 9.0),
    "r20": (9.0, 1.0),  "r21": (9.0, 5.0),  "r22": (9.0, 9.0),
    "a00": (3.0, 3.0),  "a01": (3.0, 7.0),
    "a10": (7.0, 3.0),  "a11": (7.0, 7.0),
}

CZ_DWELL = 120.0e-6            # CZ pulse window; atoms hold still throughout (s)
CZ_VIZ_RAMP = 30.0e-6          # quintic ramp-in + ramp-out at each end of the dwell (s)
BLOCKADE_RADIUS_UM = 1.4       # dashed-ring radius around each CZ atom (um)


# ----------------------------------------------------------------------------- #
# SCRIPT: one batch per row; all ops in a row fire on the same sync_batch.
# ----------------------------------------------------------------------------- #

SCRIPT_RIPA: list[list[tuple]] = [
    # === Fig. A1 tick 2: 4 parallel CZs (single-leg movers) ===
    [("mv", "r01", (1, 8)),("mv", "r11", (5, 2)),
    ("mv", "r21", (9, 2)),("mv", "r12", (8, 9))],
    [("cz", "r01", "r02"), ("cz", "r11", "r10"),
     ("cz", "r12", "r22"), ("cz", "r20", "r21")],
    [("mv", "r01", (1, 5)), ("mv", "r11", (5, 5)),
     ("mv", "r21", (9, 5)), ("mv", "r12", (5, 9))],

    # === Fig. A1 tick 3: r01 short, r10 takes a clearance lane around r11.
    # We share batch 1 with both atoms; r10's longer route then trails on its own.
    [("mv", "r11", (2, 5)), ("mv", "r10", (5, 8))],
    [("cz", "r01", "r11"), ("cz", "r10", "r12")],
    [("mv", "r10", (5, 1)), ("mv", "r11", (5, 5))],

    # === Fig. A1 tick 4: 3 parallel CZs; r00 is single-leg, r01 + r20 are 2-leg.
    [("mv", "r01", (4, 5)),("mv", "r00", (4, 1)), ("mv", "r20", (6, 1))],
    [("mv", "r01", (4, 9)),("mv", "r20", (6, 5))],
    [("cz", "r01", "r12"), ("cz", "r00", "r10"), ("cz", "r20", "r11")],
    [("mv", "r01", (4, 5)),("mv", "r20", (6, 1))],
    [("mv", "r00", (1, 1)),("mv", "r01", (1, 5)),("mv", "r20", (9, 1)),
    # === Fig. A1 tick 5: lone CZ ===
     ("mv", "r10", (8, 1))],
    [("cz", "r10", "r20")],
    [("mv", "r10", (5, 1)),
    # === Fig. A2 tick 20: 4 parallel CZs (Rot(3) ancillas -> Reg(3) sites) ===
     ("mv", "a01", (2, 7)), ("mv", "a11", (6, 7)),
     ("mv", "a00", (2, 3)), ("mv", "a10", (6, 3))],
    [("mv", "a01", (2, 9)), ("mv", "a11", (6, 9)),
     ("mv", "a00", (2, 5)), ("mv", "a10", (6, 5))],
    [("cz", "a01", "r02"), ("cz", "a11", "r12"),
     ("cz", "a00", "r01"), ("cz", "a10", "r11")],
    [("mv", "a01", (2, 7)), ("mv", "a11", (6, 7)),
     ("mv", "a00", (2, 3)), ("mv", "a10", (6, 3))],
    [("mv", "a01", (3, 7)), ("mv", "a11", (7, 7)),
     ("mv", "a00", (3, 3)), ("mv", "a10", (7, 3))],

    # === Fig. A2 tick 21: 4 parallel CZs ===
    [("mv", "a01", (4, 7)), ("mv", "r11", (6, 5)),
     ("mv", "r00", (2, 1)), ("mv", "a10", (8, 3))],
    [("mv", "a01", (4, 9)), ("mv", "r11", (6, 7)),
     ("mv", "r00", (2, 3)), ("mv", "a10", (8, 5))],
    [("cz", "a01", "r12"), ("cz", "a11", "r11"),
     ("cz", "a10", "r21"), ("cz", "a00", "r00")],
    [("mv", "a01", (4, 7)), ("mv", "r11", (6, 5)),
     ("mv", "r00", (2, 1)), ("mv", "a10", (8, 3))],
    [("mv", "a01", (3, 7)), ("mv", "r11", (5, 5)),
     ("mv", "r00", (1, 1)), ("mv", "a10", (7, 3))],
]


SCRIPT_AOD: list[list[tuple]] = [
    # === Fig. A1 tick 2: 4 parallel CZs (single-leg movers) ===
    [("mv", "r01", (1, 8))],
    [("mv", "r11", (5, 2)), ("mv", "r21", (9, 2))],
    [("mv", "r12", (8, 9))],
    [("cz", "r01", "r02"), ("cz", "r11", "r10"),
     ("cz", "r12", "r22"), ("cz", "r20", "r21")],
    [("mv", "r01", (1, 5))],
    [("mv", "r11", (5, 5)), ("mv", "r21", (9, 5))],
    [("mv", "r12", (5, 9))],

    # === Fig. A1 tick 3: r01 short, r10 takes a clearance lane around r11.
    # We share batch 1 with both atoms; r10's longer route then trails on its own.
    [("mv", "r11", (2, 5))],
    [("mv", "r10", (5, 8))],
    [("cz", "r01", "r11"), ("cz", "r10", "r12")],
    [("mv", "r10", (5, 1))],
    [("mv", "r11", (5, 5))],

    # === Fig. A1 tick 4: 3 parallel CZs; r00 is single-leg, r01 + r20 are 2-leg.
    [("mv", "r01", (4, 5)),("mv", "r00", (4, 1))],
    [("mv", "r01", (4, 9))],
    [("mv", "r20", (6, 1))],
    [("mv", "r20", (6, 5))],
    [("cz", "r01", "r12"), ("cz", "r00", "r10"), ("cz", "r20", "r11")],
    [("mv", "r01", (4, 5))],
    [("mv", "r00", (1, 1))],
    [("mv", "r01", (1, 5))],
    [("mv", "r20", (6, 1))],
    [("mv", "r20", (9, 1))],

    # === Fig. A1 tick 5: lone CZ ===
    [("mv", "r10", (8, 1))],
    [("cz", "r10", "r20")],
    [("mv", "r10", (5, 1))],

    # === Fig. A2 tick 20: 4 parallel CZs (Rot(3) ancillas -> Reg(3) sites) ===
    [("mv", "a01", (2, 9)), ("mv", "a11", (6, 9)),
     ("mv", "a00", (2, 5)), ("mv", "a10", (6, 5))],
    [("cz", "a01", "r02"), ("cz", "a11", "r12"),
     ("cz", "a00", "r01"), ("cz", "a10", "r11")],
    [("mv", "a01", (3, 7)), ("mv", "a11", (7, 7)),
     ("mv", "a00", (3, 3)), ("mv", "a10", (7, 3))],

    # === Fig. A2 tick 21: 4 parallel CZs ===
    [("mv", "a01", (4, 8)), ("mv", "r11", (6, 6))],
    [("mv", "r00", (2, 2)), ("mv", "a10", (8, 4))],
    [("cz", "a01", "r12"), ("cz", "a11", "r11"),
     ("cz", "a10", "r21"), ("cz", "a00", "r00")],
    [("mv", "r00", (1, 1)), ("mv", "a10", (7, 3))],
    [("mv", "a01", (3, 7)), ("mv", "r11", (5, 5))],
]


# ----------------------------------------------------------------------------- #
# CZ dwell step (local, so this file does not import from code_cultivation.py).
# ----------------------------------------------------------------------------- #

@dataclass
class CZDwellStep(Step):
    """Hold a pair stationary long enough to represent the CZ pulse."""

    start_time: float
    atom_ids: tuple[int, ...]
    duration: float = CZ_DWELL
    label: str = ""
    channel: str = "aod"

    def apply(self, ensemble) -> None:
        if self.duration <= 0.0:
            return
        segments = []
        for atom_id in self.atom_ids:
            pos = ensemble.atomtraj_by_id(atom_id).resting_position_at(
                self.start_time
            )
            segments.append(
                (atom_id, make_hold(pos, self.start_time, self.duration,
                                    channel=self.channel))
            )
        ensemble.append_segments_batch(segments)

    def end_time(self, ensemble) -> float:
        return self.start_time + max(0.0, float(self.duration))


# ----------------------------------------------------------------------------- #
# Build
# ----------------------------------------------------------------------------- #

def _build_aod_step(
    seq: MovingSequence,
    t0: float,
    name_to_id: dict[str, int],
    mv_ops: list[tuple],
) -> AODStep:
    """Pack a batch's "mv" ops into ONE multi-atom AODStep on the default
    (x, y) basis, then validate it against `AODStep._affected`.

    A real AOD shot moves the Cartesian product `selected_axis_1 x
    selected_axis_2` as a single tone sweep per axis. Two atoms sharing a
    source column must therefore share a target column (same for rows), and
    the product must not pick up any atom we did not intend to move.
    """
    id_to_name = {v: k for k, v in name_to_id.items()}
    sources: list[tuple[float, float]] = []
    targets: list[tuple[float, float]] = []
    ids: list[int] = []
    for _, name, target in mv_ops:
        aid = name_to_id[name]
        cur = seq.ensemble.atomtraj_by_id(aid).resting_position_at(t0)
        sources.append((float(cur[0]), float(cur[1])))
        targets.append((float(target[0]), float(target[1])))
        ids.append(aid)

    col_map: dict[float, float] = {}
    row_map: dict[float, float] = {}
    for (sx, sy), (tx, ty), (_, name, _) in zip(sources, targets, mv_ops):
        if sx in col_map and abs(col_map[sx] - tx) > 1e-8:
            raise ValueError(
                f"AOD batch at t={t0 * 1e6:.3f}us: source column {sx} maps to "
                f"both {col_map[sx]} and {tx} (atom {name!r}). One AOD shot "
                f"sweeps a whole column tone, so atoms sharing a column must "
                f"share a target column -- split this batch."
            )
        col_map[sx] = tx
        if sy in row_map and abs(row_map[sy] - ty) > 1e-8:
            raise ValueError(
                f"AOD batch at t={t0 * 1e6:.3f}us: source row {sy} maps to "
                f"both {row_map[sy]} and {ty} (atom {name!r}). One AOD shot "
                f"sweeps a whole row tone, so atoms sharing a row must share "
                f"a target row -- split this batch."
            )
        row_map[sy] = ty

    selected_axis_1 = tuple(sorted(col_map))
    new_axis_1 = tuple(col_map[c] for c in selected_axis_1)
    selected_axis_2 = tuple(sorted(row_map))
    new_axis_2 = tuple(row_map[r] for r in selected_axis_2)

    step = AODStep(
        start_time=t0,
        selected_axis_1=selected_axis_1,
        selected_axis_2=selected_axis_2,
        new_axis_1=new_axis_1,
        new_axis_2=new_axis_2,
        match_tol=1e-8,
    )

    affected = {aid: (old, new) for aid, old, new in step._affected(seq.ensemble)}
    expected_ids = set(ids)
    extras = set(affected) - expected_ids
    if extras:
        raise ValueError(
            f"AOD batch at t={t0 * 1e6:.3f}us: the Cartesian product "
            f"sel_1={selected_axis_1} x sel_2={selected_axis_2} also "
            f"addresses unintended atoms "
            f"{sorted(id_to_name[a] for a in extras)}. Pick different rows/"
            f"columns or move those atoms in the same shot."
        )
    missing = expected_ids - set(affected)
    if missing:
        raise ValueError(
            f"AOD batch at t={t0 * 1e6:.3f}us: requested atoms "
            f"{sorted(id_to_name[a] for a in missing)} are not at the "
            f"Cartesian product intersections -- check the source positions."
        )
    for aid, (sx, sy), (tx, ty) in zip(ids, sources, targets):
        got_new = affected[aid][1]
        if abs(got_new[0] - tx) > 1e-7 or abs(got_new[1] - ty) > 1e-7:
            raise ValueError(
                f"AOD batch at t={t0 * 1e6:.3f}us: atom {id_to_name[aid]!r} "
                f"is routed to {got_new} instead of {(tx, ty)} by this shot."
            )
    return step


def build_sequence() -> tuple[MovingSequence, dict[str, int]]:
    names = list(INITIAL)
    name_to_id = {name: idx for idx, name in enumerate(names)}
    positions = np.asarray([INITIAL[n] for n in names], dtype=float)
    initial = AtomConfig(positions=positions, atom_ids=np.arange(len(names)))
    seq = MovingSequence(grid=GRID, initial=initial, collision_dt=1e-7)

    for batch in SCRIPT_AOD:
        t0 = seq.next_start_time()
        mvs = [op for op in batch if op[0] == "mv"]
        czs = [op for op in batch if op[0] == "cz"]
        unknown = [op for op in batch if op[0] not in ("mv", "cz")]
        if unknown:
            raise ValueError(f"unknown op kind {unknown[0][0]!r}")

        steps: list[Step] = []
        if mvs:
            steps.append(_build_aod_step(seq, t0, name_to_id, mvs))
        for _, a, b in czs:
            steps.append(
                CZDwellStep(
                    start_time=t0,
                    atom_ids=(name_to_id[a], name_to_id[b]),
                    duration=CZ_DWELL,
                    label=f"CZ {a}-{b}",
                )
            )
        seq.append_sync_batch(steps)
    return seq, name_to_id


# ----------------------------------------------------------------------------- #
# Render
# ----------------------------------------------------------------------------- #

FIGSIZE = (5.5, 5.5)
DPI = 120
FPS = 24
TIME_DILATION = 8.0e3
HOLD_SECONDS = 0.8
BG_COLOR = "#f7f7f7"
CZ_RING_RGBA = (0.85, 0.18, 0.18, 0.90)
CZ_BEAM_RGBA = (0.96, 0.55, 0.10, 0.70)


def _cz_envelope(step: CZDwellStep, t: float) -> float:
    """0 -> quintic ramp-in -> plateau -> quintic ramp-out -> 0, entirely
    inside the physical dwell `[start_time, start_time + duration]` so the
    ramp-in only begins after the prior AOD batch has finished."""
    dwell_end = step.start_time + step.duration
    if t < step.start_time or t >= dwell_end:
        return 0.0
    elapsed = t - step.start_time
    remaining = dwell_end - t
    if elapsed < CZ_VIZ_RAMP:
        return float(QUINTIC_MIN_JERK.value(elapsed / CZ_VIZ_RAMP))
    if remaining < CZ_VIZ_RAMP:
        return float(QUINTIC_MIN_JERK.value(remaining / CZ_VIZ_RAMP))
    return 1.0


def _render_frame(seq: MovingSequence, atom_colors: list, t: float) -> Image.Image:
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI, constrained_layout=True)
    ensemble = seq.ensemble
    grid = ensemble.grid

    draw_grid_dots(ax, grid)

    # Red Gaussian trap halo around any atom that is currently moving, and
    # around every empty AOD trap intersection (Cartesian product addresses
    # that don't hold an atom) -- same "quality" addressed_style as
    # src/visualization.draw_traps.
    moving_xy = []
    for atom in ensemble.atomtrajs:
        for seg in atom.segments:
            if seg.start_time - 1e-12 <= t <= seg.end_time + 1e-12 and seg.length > 0:
                moving_xy.append(atom.position_at(t))
                break
    if moving_xy:
        xy = grid.ij_to_xy(np.asarray(moving_xy, dtype=float))
        for x, y in xy:
            _draw_gaussian_blob(ax, float(x), float(y), trap_scale=0.9)

    empty_aod_xy = _active_aod_traps_xy(seq.steps, ensemble, t)
    if len(empty_aod_xy):
        atom_xy = grid.ij_to_xy(ensemble.positions_at(t))
        if len(atom_xy):
            atom_tol = grid.d * 1e-3
            dists = np.linalg.norm(
                empty_aod_xy[:, None, :] - atom_xy[None, :, :], axis=2,
            )
            empty_aod_xy = empty_aod_xy[~(dists < atom_tol).any(axis=1)]
        for x, y in empty_aod_xy:
            _draw_gaussian_blob(ax, float(x), float(y), trap_scale=0.9)

    # CZ overlay: dashed blockade rings on each pair atom + a Gaussian beam blob
    # at the midpoint. Each CZ event carries its own ramp envelope.
    for step in seq.steps:
        if not isinstance(step, CZDwellStep):
            continue
        intensity = _cz_envelope(step, t)
        if intensity <= 0.0:
            continue
        ij = np.asarray(
            [ensemble.atomtraj_by_id(aid).position_at(t) for aid in step.atom_ids],
            dtype=float,
        )
        xy = grid.ij_to_xy(ij)
        ring_rgba = (
            CZ_RING_RGBA[0], CZ_RING_RGBA[1], CZ_RING_RGBA[2],
            CZ_RING_RGBA[3] * intensity,
        )
        for x, y in xy:
            ax.add_patch(
                Circle(
                    (float(x), float(y)),
                    BLOCKADE_RADIUS_UM,
                    fill=False, linestyle="--", linewidth=1.6,
                    edgecolor=ring_rgba, zorder=6,
                )
            )
        mid_x, mid_y = xy.mean(axis=0)
        beam_rgba = (
            CZ_BEAM_RGBA[0], CZ_BEAM_RGBA[1], CZ_BEAM_RGBA[2],
            CZ_BEAM_RGBA[3] * intensity,
        )
        _draw_gaussian_blob(
            ax, float(mid_x), float(mid_y), trap_scale=1.6, rgba=beam_rgba,
        )

    draw_atoms(
        ax, ensemble, t, colors=atom_colors,
        atom_scale=1.8, edge_linewidth=1.4,
    )
    draw_grid_frame(ax, grid)
    ax.set_title(f"t = {t * 1e6:7.1f} us", fontsize=10, loc="right")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=BG_COLOR)
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf)
    img.load()
    return img.convert("RGB")


def render_gif(seq: MovingSequence, name_to_id: dict[str, int], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    atom_colors = [
        ("#4c78a8" if name.startswith("r") else "#f58518")
        for name, _ in sorted(name_to_id.items(), key=lambda kv: kv[1])
    ]

    total = seq.total_duration()
    frame_dt = 1.0 / (TIME_DILATION * FPS)
    n_frames = max(2, int(math.ceil(total / frame_dt)) + 1)
    times = np.linspace(0.0, total, n_frames)
    frame_ms = max(1, int(round(1000.0 / FPS)))
    hold_ms = max(0, int(round(HOLD_SECONDS * 1000.0)))

    frames: list[Image.Image] = []
    durations: list[int] = []
    if hold_ms > 0:
        frames.append(_render_frame(seq, atom_colors, -1e-9))
        durations.append(hold_ms)
    for t in tqdm(times, desc="render frames", unit="frame"):
        frames.append(_render_frame(seq, atom_colors, float(t)))
        durations.append(frame_ms)
    if hold_ms > 0:
        durations[-1] += hold_ms

    palette = frames[len(frames) // 2].quantize(
        colors=256, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE,
    )
    quantized = [
        f.quantize(palette=palette, dither=Image.Dither.FLOYDSTEINBERG) for f in frames
    ]
    quantized[0].save(
        out_path,
        save_all=True, append_images=quantized[1:],
        duration=durations, loop=0, optimize=True,
    )
    for img in frames:
        img.close()
    for img in quantized:
        img.close()
    palette.close()


def main() -> None:
    seq, name_to_id = build_sequence()
    report = seq.validate(dt=1e-7)
    if not report.ok:
        raise RuntimeError(report)

    n_cz = sum(1 for s in seq.steps if isinstance(s, CZDwellStep))
    n_aod = sum(1 for s in seq.steps if isinstance(s, AODStep))
    print(f"atoms: {len(name_to_id)}")
    print(f"sync batches: {len(SCRIPT_AOD)}")
    print(f"AOD steps: {n_aod}   CZ steps: {n_cz}")
    print(f"duration: {seq.total_duration() * 1e6:.3f} us")

    out_path = ROOT / "render" / "code_cultivation_manual.gif"
    render_gif(seq, name_to_id, out_path)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
