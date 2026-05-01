"""RIPA vs AOD synchronized benchmark GIF.

Renders RIPA pebble-adv (left) and AOD sqrt-time (right) on one shared
physics clock. When one scheduler finishes early, its panel freezes in
place while the other keeps moving until both are done -- AtomEnsemble's
`position_at(t)` clamps each atom to its final site past its sequence end.

Usage:
    python example/defect_free_assembly_benchmark.py            # quick check -> render/
    python example/defect_free_assembly_benchmark.py --demo     # online quality -> demo/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.atom_config import Grid
from src.benchmark import benchmark_schedulers, format_benchmark_table
from src.routing import (
    RoutingRequest,
    centered_square_targets,
    stochastically_loaded_sites,
)
from src.scheduler.aod_sqrt_time import SqrtTimeAODScheduler
from src.scheduler.ripa_pebble_adv import RIPAPebbleAdvScheduler
from src.visualization import render_animation

N = 10
TARGET_SIDE = 7
SEED = 260405317
GRID_SPACING_UM = 5.0
COLLISION_RADIUS_UM = 4.0

PREFIX = "defect_free_assembly_benchmark"

# Dict order = panel order (RIPA left, AOD right). Keys also become panel labels.
SCHEDULERS = {
    "RIPA pebble adv": RIPAPebbleAdvScheduler,
    "AOD sqrt-time": SqrtTimeAODScheduler,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="RIPA vs AOD synchronized benchmark GIF.")
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
    src = stochastically_loaded_sites(N, TARGET_SIDE * TARGET_SIDE, SEED)
    dst = centered_square_targets(N, TARGET_SIDE)
    request = RoutingRequest(grid=grid, src=src, dst=dst, labeled=False)
    print(
        f"seed={SEED}, N={N}, loaded={len(src)} atoms "
        f"({len(src) / (N * N):.1%}), target={TARGET_SIDE}x{TARGET_SIDE}"
    )

    results = benchmark_schedulers(request, SCHEDULERS)
    print(format_benchmark_table(results))

    failures = [r for r in results if not r.ok]
    if failures:
        for r in failures:
            print(f"{r.name} error: {type(r.error).__name__}: {r.error}")
        raise SystemExit(1)

    if args.no_render:
        return

    by_name = {r.name: r.sequence for r in results}
    panels = {name: by_name[name] for name in SCHEDULERS}

    if args.demo:
        out_path = ROOT / "demo" / f"{PREFIX}.gif"
        quality, fps = "quality", 20
    else:
        out_path = ROOT / "render" / f"{PREFIX}.gif"
        quality, fps = "speed", 6

    out_path.parent.mkdir(parents=True, exist_ok=True)
    render_animation(
        panels,
        out_path,
        view="benchmark",
        quality=quality,
        fps=fps,
        time_dilation=2e4,
        hold_seconds=1.5,
        title=f"RIPA vs AOD - defect-free assembly ({TARGET_SIDE}x{TARGET_SIDE})",
    )
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
