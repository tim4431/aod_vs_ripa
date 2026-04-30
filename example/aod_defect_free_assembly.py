"""Demo: assemble a defect-free 2D atom array with the AOD scheduler.

The source array is randomly loaded at about 50% filling. The target uses
the largest square that can be made defect-free with the available atoms;
any extra atoms are parked outside that square so the unlabeled routing
request preserves atom count.

Outputs are written to `render/`:
    - `aod_defect_free_assembly_t0.png`
    - `aod_defect_free_assembly_tfinal.png`
    - `aod_defect_free_assembly.gif`
"""

from __future__ import annotations

import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.atom_config import Grid
from src.routing import RoutingRequest, Site
from src.scheduler.aod_sqrt_time import SqrtTimeAODScheduler
from src.visualization import render_check_outputs


N = 10
FILL_RATE = 0.50
SEED = 260405317
GRID_SPACING_UM = 5.0
COLLISION_RADIUS_UM = 4.0
COLLISION_DT = 5e-6

OUT_DIR = ROOT / "render"
PREFIX = "aod_defect_free_assembly"


def random_loaded_sites(N: int, fill_rate: float, seed: int) -> list[Site]:
    rng = random.Random(seed)
    sites = [
        (i, j)
        for i in range(N)
        for j in range(N)
        if rng.random() < fill_rate
    ]
    if not sites:
        center = N // 2
        sites.append((center, center))
    return sites


def square_plus_parking_targets(N: int, atom_count: int) -> tuple[list[Site], int]:
    """Largest centered LxL defect-free square plus parking for leftovers."""
    L = int(math.isqrt(atom_count))
    if L == 0:
        return [], 0

    start = (N - L) // 2
    square = [
        (i, j)
        for i in range(start, start + L)
        for j in range(start, start + L)
    ]
    square_set = set(square)
    leftovers = atom_count - len(square)

    parking: list[Site] = []
    for site in _parking_site_order(N, start, L):
        if site not in square_set:
            parking.append(site)
            if len(parking) == leftovers:
                break

    if len(parking) != leftovers:
        raise ValueError("not enough parking sites for leftover atoms")
    return square + parking, L


def _parking_site_order(N: int, square_start: int, L: int) -> list[Site]:
    center = (N - 1) / 2
    square_center = square_start + (L - 1) / 2
    sites = [(i, j) for i in range(N) for j in range(N)]
    return sorted(
        sites,
        key=lambda site: (
            abs(site[0] - square_center) + abs(site[1] - square_center),
            abs(site[0] - center) + abs(site[1] - center),
            site,
        ),
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    grid = Grid(N=N, d=GRID_SPACING_UM, rc=COLLISION_RADIUS_UM)
    src = random_loaded_sites(N, FILL_RATE, SEED)
    dst, square_side = square_plus_parking_targets(N, len(src))
    request = RoutingRequest(grid=grid, src=src, dst=dst, labeled=False)

    scheduler = SqrtTimeAODScheduler(
        request,
        collision_dt=COLLISION_DT,
        validate_final=True,
    )
    sequence = scheduler.plan()
    report = sequence.validate(dt=COLLISION_DT)
    if not report.ok:
        raise RuntimeError(f"collision validation failed: {report}")

    total_us = sequence.total_duration() * 1e6
    print(
        f"seed={SEED}, N={N}, loaded={len(src)} atoms "
        f"({len(src) / (N * N):.1%}), square={square_side}x{square_side}"
    )
    print(
        f"aod_steps={len(sequence.steps)}, "
        f"duration={total_us:.3f} us, "
        f"three_step={scheduler.binary_plan.used_three_step}"
    )

    static_traps = dst[: square_side * square_side]
    render_check_outputs(
        sequence,
        OUT_DIR,
        PREFIX,
        static_traps=static_traps,
        show_planned=True,
        addressed_style="edge",
        title_prefix=f"AOD defect-free assembly - target {square_side}x{square_side}",
        gif_fps=6,
        gif_hold_seconds=1.5,
        optimize="speed",
        use_multiprocessing=False,
    )


if __name__ == "__main__":
    main()
