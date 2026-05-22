"""RIPA defect-free assembly snapshots for paper figures.

Runs the RIPA pebble-adv scheduler on the same defect-free assembly
problem as `defect_free_assembly_benchmark.py` and saves single-frame
PNGs at t = 80 us, 100 us, and 120 us into `render/` at quality preset.

Usage:
    python example/defect_free_paper_demo.py
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
from src.scheduler.heuristic.ripa_pebble_adv import UnlabeledRIPAPebbleAdvScheduler
from src.visualization import save_frame

N = 10
TARGET_SIDE = 7
ATOM_COUNT = TARGET_SIDE * TARGET_SIDE
SEED = 260405317
GRID_SPACING_UM = 5.0
COLLISION_RADIUS_UM = 4.0

PREFIX = "defect_free_paper_demo"

SNAPSHOT_TIMES_US = (80.0, 100.0, 120.0)

SCHEDULERS = {
    "RIPA pebble adv": UnlabeledRIPAPebbleAdvScheduler,
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RIPA defect-free assembly paper snapshots."
    )
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="print the benchmark table only and skip rendering",
    )
    args = parser.parse_args()

    grid = Grid(N=N, d=GRID_SPACING_UM, rc=COLLISION_RADIUS_UM)
    storage = storage_subgrid(N, 1)
    src = random_sample_sites(storage, ATOM_COUNT, SEED)
    dst = centered_storage_square(N, TARGET_SIDE, 1)
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

    sequence = next(r.sequence for r in results if r.name == "RIPA pebble adv")

    out_dir = ROOT / "render"
    out_dir.mkdir(parents=True, exist_ok=True)

    for t_us in SNAPSHOT_TIMES_US:
        out_path = out_dir / f"{PREFIX}_t{int(t_us)}us.png"
        save_frame(
            sequence,
            t_us * 1e-6,
            out_path,
            view="demo",
            quality="quality",
            atom_colors={idx: "gray" for idx in range(N * N)},
            title=f"RIPA pebble adv - defect-free assembly ({TARGET_SIDE}x{TARGET_SIDE}) - t = {t_us:.0f} us",
        )
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
