"""RIPA inversion benchmark for a 1x6 atom row on a 24x24 grid.

A 1x6 storage row (period 2) sits in the center of a 24x24 grid, occupying
x-coordinates 6, 8, 10, 12, 14, 16 at a fixed y. The labeled target is a
horizontal (x-direction) inversion: the atom at (i, j) must reach
(pivot_sum - i, j).

Usage:
    python example/inversion_1x6_benchmark.py            # quick check -> render/
    python example/inversion_1x6_benchmark.py --demo     # online quality -> demo/
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
from src.routing import RoutingRequest
from src.scheduler.ripa_naive_sync import RIPANaiveSyncScheduler
from src.scheduler.ripa_pebble import RIPAPebbleScheduler
from src.scheduler.ripa_pebble_adv import RIPAPebbleAdvScheduler
from src.scheduler.ripa_pebroute import RIPAPebRouteScheduler
from src.scheduler.ripa_stochastic_search import RIPASearchScheduler
from src.visualization import render_animation

N = 24
TARGET_LENGTH = 6
STORAGE_PERIOD = 2
GRID_SPACING_UM = 5.0
COLLISION_RADIUS_UM = 4.0

PREFIX = "inversion_1x6_benchmark"

# Dict order = panel order. Keys also become panel labels.
SCHEDULERS = {
    "ripa_pebble": lambda req: RIPAPebbleScheduler(req, highway_period=STORAGE_PERIOD),
    "ripa_pebble_adv": RIPAPebbleAdvScheduler,
    "ripa_pebroute": RIPAPebRouteScheduler,
    "ripa_search": lambda req: RIPASearchScheduler(req, n_iter=200, seed=0),
    # "ripa_pebble_b": RIPAPebbleBScheduler,
    "ripa_naive_sync": lambda req: RIPANaiveSyncScheduler(req, highway_period=STORAGE_PERIOD),
}


def centered_storage_row(N: int, length: int, period: int) -> list[tuple[int, int]]:
    """Centered length-atom row on the storage subgrid, along the x direction."""
    storage = list(range(0, N, period))
    center = (N - 1) / 2
    s = min(
        range(len(storage) - length + 1),
        key=lambda i: abs((storage[i] + storage[i + length - 1]) / 2 - center),
    )
    j_idx = min(range(len(storage)), key=lambda k: abs(storage[k] - center))
    j = storage[j_idx]
    return [(storage[i], j) for i in range(s, s + length)]


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
        description="RIPA inversion benchmark for a 1x6 atom row."
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
    src = centered_storage_row(N, TARGET_LENGTH, STORAGE_PERIOD)
    dst = x_inverted_targets(src)
    request = RoutingRequest(grid=grid, src=src, dst=dst, labeled=True)
    print(
        f"N={N}, storage_period={STORAGE_PERIOD}, "
        f"atoms={len(src)} (1x{TARGET_LENGTH}), "
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
        title=f"RIPA inversion - 1x{TARGET_LENGTH} row on {N}x{N} grid",
    )
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
