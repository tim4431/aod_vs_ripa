"""Demo and benchmark: unlabeled defect-free assembly with RIPA pebble routing.

49 atoms randomly placed on the even storage sites of a 20x20 grid are routed
to a centered 7x7 target square. Odd rows/columns are left empty as transport
highways for RIPA.

Outputs are written to `render/`:
    - `defect_free_assembly_t0.png`
    - `defect_free_assembly_tfinal.png`
    - `defect_free_assembly.gif`
    - `defect_free_assembly_benchmark.png`
    - `defect_free_assembly_benchmark.gif`
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
from src.benchmark import benchmark_schedulers, format_benchmark_table
from src.routing import RoutingRequest
from src.scheduler.ripa_naive_sync import RIPANaiveSyncScheduler
from src.scheduler.ripa_ccbs_c import RIPACCBSCScheduler
from src.scheduler.ripa_pebble import RIPAPebbleScheduler
from src.scheduler.ripa_pebble_adv import RIPAPebbleAdvScheduler
from src.visualization import render_animation,  save_frame

N = 20
TARGET_SIDE = 7
SEED = 260405317
STORAGE_PERIOD = 2
GRID_SPACING_UM = 5.0
COLLISION_RADIUS_UM = 4.0
COLLISION_DT = 5e-6

PREFIX = "defect_free_assembly_highway"
OUT_DIR = ROOT / "render"

# `ripa_pebble_adv` takes no highway_period: it should rediscover the highway
# from occupancy alone. The other RIPA schedulers are told the period explicitly.
# `_min_max` switches the unlabeled assignment from min-sum (Hungarian) to
# bottleneck (min-max), which forces atoms already in target slots to also
# participate in the routing instead of self-assigning at zero cost.
BASE_SCHEDULERS = {
    "ripa_pebble": lambda req: RIPAPebbleScheduler(
        req, collision_dt=COLLISION_DT, highway_period=STORAGE_PERIOD,
    ),
    "ripa_pebble_adv": lambda req: RIPAPebbleAdvScheduler(
        req, collision_dt=COLLISION_DT,
    ),
    "ripa_pebble_adv_min_max": lambda req: RIPAPebbleAdvScheduler(
        req, collision_dt=COLLISION_DT, unlabeled_assignment="min_max",
    ),
    "ripa_naive_sync": lambda req: RIPANaiveSyncScheduler(
        req, collision_dt=COLLISION_DT, highway_period=STORAGE_PERIOD,
    ),
}


def scheduler_factories(
    *,
    include_ccbs_c: bool,
    ccbs_time_limit: float,
    ccbs_max_high_level_nodes: int,
):
    schedulers = {}
    if include_ccbs_c:
        schedulers["ripa_ccbs_c"] = lambda req: RIPACCBSCScheduler(
            req,
            collision_dt=COLLISION_DT,
            time_limit=ccbs_time_limit,
            max_high_level_nodes=ccbs_max_high_level_nodes,
            unlabeled_assignment="min_max",
        )
    schedulers.update(BASE_SCHEDULERS)
    return schedulers


def storage_subgrid(N: int, period: int) -> list[tuple[int, int]]:
    """Sites on the even-storage subgrid (odd rows/cols are highways)."""
    return [(i, j) for i in range(0, N, period) for j in range(0, N, period)]


def random_sample_sites(sites, count, seed):
    rng = random.Random(seed)
    chosen = list(sites)
    rng.shuffle(chosen)
    return sorted(chosen[:count])


def centered_storage_square(N: int, side: int, period: int) -> list[tuple[int, int]]:
    """Centered side x side target square on the storage subgrid."""
    storage = list(range(0, N, period))
    center = (N - 1) / 2
    s = min(
        range(len(storage) - side + 1),
        key=lambda i: abs((storage[i] + storage[i + side - 1]) / 2 - center),
    )
    return [(storage[i], storage[j]) for i in range(s, s + side) for j in range(s, s + side)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-render", action="store_true",
                        help="skip PNG/GIF rendering and only print the benchmark table")
    parser.add_argument("--skip-ccbs-c", action="store_true",
                        help="omit the exact C++ CCBS backend from the benchmark")
    parser.add_argument("--ccbs-time-limit", type=float, default=10.0,
                        help="planning time limit for ripa_ccbs_c")
    parser.add_argument("--ccbs-max-high-level-nodes", type=int, default=5_000,
                        help="high-level node cap for ripa_ccbs_c")
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

    schedulers = scheduler_factories(
        include_ccbs_c=not args.skip_ccbs_c,
        ccbs_time_limit=args.ccbs_time_limit,
        ccbs_max_high_level_nodes=args.ccbs_max_high_level_nodes,
    )
    results = benchmark_schedulers(request, schedulers, validate_dt=COLLISION_DT)
    print(format_benchmark_table(results))
    for r in results:
        if not r.ok:
            print(f"{r.name} error: {type(r.error).__name__}: {r.error}")

    pebble = next(r for r in results if r.name == "ripa_pebble")
    if not pebble.ok or pebble.sequence is None:
        raise SystemExit("ripa_pebble did not produce a valid defect-free assembly")

    if args.no_render:
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Synchronized side-by-side panels in scheduler dict order.
    panels = {name: r.sequence for name in schedulers for r in results
              if r.name == name and r.ok and r.sequence is not None}

    save_frame(
        panels, 0.0, OUT_DIR / f"{PREFIX}_benchmark.png",
        view="benchmark", quality="speed",
        title="Defect-free assembly benchmark",
    )
    render_animation(
        panels, OUT_DIR / f"{PREFIX}_benchmark.gif",
        view="benchmark", quality="speed",
        fps=6, time_dilation=2e4, hold_seconds=1.5,
        title=f"RIPA defect-free assembly ({TARGET_SIDE}x{TARGET_SIDE})",
    )

if __name__ == "__main__":
    main()
