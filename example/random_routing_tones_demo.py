"""RIPA random pairwise routing demo with EOM tone visualization.

Runs a small (~10 atom) labeled random source-to-target routing problem
through the C++ CCBS scheduler, then renders the resulting motion with
the `detail` view -- a side-by-side atom panel and live row/col EOM
tone spectra plus tone-vs-time history. The point is to make the
underlying physical control scheme visible: each moving atom shows up
as a single normalized RIPA EOM tone whose frequency tracks its grid
position over time.

Usage:
    python example/random_routing_tones_demo.py            # quick check -> render/
    python example/random_routing_tones_demo.py --demo     # quality GIF -> demo/
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.atom_config import Grid
from src.routing import RoutingRequest
from src.scheduler.ccbs_c import RIPACCBSCScheduler
from src.visualization import render_animation

N = 8
ATOM_COUNT = 10
SEED = 20260501
GRID_SPACING_UM = 5.0
COLLISION_RADIUS_UM = 4.0
COLLISION_DT = 5e-6
CCBS_PRECISION = 1e-7
TIME_LIMIT = 10.0
MAX_HIGH_LEVEL_NODES = 50_000

PREFIX = "random_routing_tones_demo"


def make_request(grid: Grid, atom_count: int, seed: int) -> RoutingRequest:
    rng = random.Random(seed)
    sites = [(i, j) for i in range(grid.N) for j in range(grid.N)]
    src = rng.sample(sites, atom_count)
    dst = rng.sample(sites, atom_count)
    return RoutingRequest(grid=grid, src=src, dst=dst, labeled=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RIPA random pairwise routing demo with EOM tone view."
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="render a high-quality GIF into demo/ instead of a quick render/ check",
    )
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="plan only and skip rendering",
    )
    args = parser.parse_args()

    grid = Grid(N=N, d=GRID_SPACING_UM, rc=COLLISION_RADIUS_UM)
    request = make_request(grid, ATOM_COUNT, SEED)
    print(
        f"seed={SEED}, N={N}, atoms={ATOM_COUNT}, "
        f"task=random labeled pairwise routing"
    )

    scheduler = RIPACCBSCScheduler(
        request,
        collision_dt=COLLISION_DT,
        ccbs_precision=CCBS_PRECISION,
        high_level_order="conflicts",
        time_limit=TIME_LIMIT,
        max_high_level_nodes=MAX_HIGH_LEVEL_NODES,
        freeze_initial_target_atoms=True,
    )
    sequence = scheduler.plan()

    report = sequence.validate(dt=COLLISION_DT)
    if not report.ok:
        raise RuntimeError(f"collision validation failed: {report}")

    print(
        f"planned: total={sequence.total_duration() * 1e6:.2f} us, "
        f"steps={len(sequence.steps)}, "
        f"segments={sum(len(a.segments) for a in sequence.ensemble.atomtrajs)}"
    )

    if args.no_render:
        return

    if args.demo:
        out_path = ROOT / "demo" / f"{PREFIX}.gif"
        quality = "quality"
    else:
        out_path = ROOT / "render" / f"{PREFIX}.gif"
        quality = "speed"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    render_animation(
        sequence,
        out_path,
        view="detail",
        quality=quality,
        time_dilation=1e4,
        hold_seconds=1.5,
        planned_trajectory="full",
        show_atom_ids=True,
        title=(
            f"RIPA random pairwise routing with EOM tones "
            f"({ATOM_COUNT} atoms, {N}x{N})"
        ),
    )
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
