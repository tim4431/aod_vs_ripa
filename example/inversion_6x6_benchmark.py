"""RIPA inversion benchmark for a 6x6 atom array on a 24x24 grid.

A 6x6 storage square (period 2) sits in the center of a 24x24 grid, occupying
coordinates 6, 8, 10, 12, 14, 16 along each axis. The labeled target is a
horizontal (x-direction) inversion: the atom at (i, j) must reach (i, 22 - j).

The inversion is a non-trivial labeled permutation, even though the destination
sites coincide with the source sites as a set (the centered square is symmetric
about its own center). The request therefore uses `labeled=True`.

Usage:
    python example/inversion_6x6_benchmark.py            # quick check -> render/
    python example/inversion_6x6_benchmark.py --demo     # online quality -> demo/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib as mpl

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.atom_config import Grid
from src.benchmark import benchmark_schedulers, format_benchmark_table
from src.routing import RoutingRequest, centered_storage_square
from src.scheduler.ripa_naive_sync import RIPANaiveSyncScheduler
from src.scheduler.ripa_pebble import RIPAPebbleScheduler
from src.scheduler.ripa_pebble_adv import RIPAPebbleAdvScheduler
from src.scheduler.ripa_pebble_b import RIPAPebbleBScheduler
from src.scheduler.ripa_pebble_search import RIPAPebbleSearchScheduler
from src.visualization import render_animation

N = 24
TARGET_SIDE = 6
STORAGE_PERIOD = 2
GRID_SPACING_UM = 5.0
COLLISION_RADIUS_UM = 4.0

PREFIX = "inversion_6x6_benchmark"

# Dict order = panel order. Keys also become panel labels.
SCHEDULERS = {
    "ripa_pebble": lambda req: RIPAPebbleScheduler(req, highway_period=STORAGE_PERIOD),
    "ripa_pebble_adv": RIPAPebbleAdvScheduler,
    # "ripa_pebble_b": RIPAPebbleBScheduler,
    # "ripa_pebble_search": lambda req: RIPAPebbleSearchScheduler(req, n_iter=200, seed=0),
    "ripa_naive_sync": lambda req: RIPANaiveSyncScheduler(req, highway_period=STORAGE_PERIOD),
}


def x_inverted_targets(src):
    """Mirror each site along the x direction (the first site index)."""
    xs = sorted({i for i, _ in src})
    pivot_sum = xs[0] + xs[-1]
    return [(pivot_sum - i, j) for i, j in src]


def x_gradient_colors(src):
    """Color each atom by initial x position so the inversion is visually obvious."""
    xs = sorted({i for i, _ in src})
    norm = mpl.colors.Normalize(vmin=xs[0], vmax=xs[-1])
    cmap = mpl.colormaps["coolwarm"]
    return {idx: cmap(norm(i)) for idx, (i, _) in enumerate(src)}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RIPA inversion benchmark for a 6x6 atom array."
    )
    parser.add_argument(
        "--demo", action="store_true",
        help="render a high-quality GIF into demo/ instead of a quick render/ check",
    )
    parser.add_argument(
        "--no-render", action="store_true",
        help="print the benchmark table only and skip rendering",
    )
    args = parser.parse_args()

    grid = Grid(N=N, d=GRID_SPACING_UM, rc=COLLISION_RADIUS_UM)
    src = centered_storage_square(N, TARGET_SIDE, STORAGE_PERIOD)
    dst = x_inverted_targets(src)
    request = RoutingRequest(grid=grid, src=src, dst=dst, labeled=True)
    print(
        f"N={N}, storage_period={STORAGE_PERIOD}, "
        f"atoms={len(src)} ({TARGET_SIDE}x{TARGET_SIDE}), "
        f"task=horizontal inversion (labeled)"
    )

    results = benchmark_schedulers(request, SCHEDULERS)
    print(format_benchmark_table(results))
    for r in results:
        if not r.ok:
            print(f"{r.name} error: {type(r.error).__name__}: {r.error}")

    if args.no_render:
        return

    by_name = {r.name: r.sequence for r in results if r.ok and r.sequence is not None}
    if not by_name:
        raise SystemExit("no scheduler produced a valid sequence")
    panels = {name: by_name[name] for name in SCHEDULERS if name in by_name}

    if args.demo:
        out_path = ROOT / "demo" / f"{PREFIX}.gif"
        quality = "quality"
    else:
        out_path = ROOT / "render" / f"{PREFIX}.gif"
        quality = "speed"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    render_animation(
        panels,
        out_path,
        view="benchmark",
        quality=quality,
        time_dilation=1e4,
        hold_seconds=1.5,
        atom_colors=x_gradient_colors(src),
        title=f"RIPA inversion - {TARGET_SIDE}x{TARGET_SIDE} array on {N}x{N} grid",
    )
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
