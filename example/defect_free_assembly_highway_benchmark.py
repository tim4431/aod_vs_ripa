"""RIPA highway benchmark for unlabeled defect-free assembly.

49 atoms randomly placed on the even storage sites of a 20x20 grid are routed
to a centered 7x7 target square. Odd rows/columns are left empty as transport
highways for RIPA.

Usage:
    python example/defect_free_assembly_highway_benchmark.py            # quick check -> render/
    python example/defect_free_assembly_highway_benchmark.py --demo     # online quality -> demo/
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
    centered_storage_square,
    random_sample_sites,
    storage_subgrid,
)
from src.scheduler.ccbs_c import RIPACCBSCScheduler
from src.scheduler.ripa_naive_sync import RIPANaiveSyncScheduler
from src.scheduler.ripa_pebble import RIPAPebbleScheduler
from src.scheduler.ripa_pebble_adv import UnlabeledRIPAPebbleAdvScheduler
from src.visualization import render_animation

N = 20
TARGET_SIDE = 7
SEED = 260405317
STORAGE_PERIOD = 2
GRID_SPACING_UM = 5.0
COLLISION_RADIUS_UM = 4.0

PREFIX = "defect_free_assembly_highway"

# Dict order = panel order. Keys also become panel labels.
# `ripa_pebble_adv` rediscovers the highway from occupancy alone; the others
# need `highway_period` set explicitly. `min_max` switches the unlabeled
# assignment from Hungarian to bottleneck so atoms already in target slots
# also participate in routing instead of self-assigning at zero cost.
SCHEDULERS = {
    # "ripa_ccbs_c": lambda req: RIPACCBSCScheduler(
    #     req, time_limit=10.0, max_high_level_nodes=5_000,
    # ),
    "ripa_pebble": lambda req: RIPAPebbleScheduler(req, highway_period=STORAGE_PERIOD),
    "ripa_pebble_adv": UnlabeledRIPAPebbleAdvScheduler,
    "ripa_pebble_adv_min_max": lambda req: UnlabeledRIPAPebbleAdvScheduler(
        req, unlabeled_assignment="min_max",
    ),
    "ripa_naive_sync": lambda req: RIPANaiveSyncScheduler(req, highway_period=STORAGE_PERIOD),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="RIPA highway benchmark for defect-free assembly.")
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
    storage = storage_subgrid(N, STORAGE_PERIOD)
    src = random_sample_sites(storage, TARGET_SIDE * TARGET_SIDE, SEED)
    dst = centered_storage_square(N, TARGET_SIDE, STORAGE_PERIOD)
    request = RoutingRequest(grid=grid, src=src, dst=dst, labeled=False)
    print(
        f"seed={SEED}, N={N}, storage_sites={len(storage)}, "
        f"loaded={len(src)} atoms ({len(src) / len(storage):.1%}), "
        f"target={TARGET_SIDE}x{TARGET_SIDE}"
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
        title=f"RIPA defect-free assembly ({TARGET_SIDE}x{TARGET_SIDE})",
    )
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
