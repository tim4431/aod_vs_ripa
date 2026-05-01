"""Demo: assemble a centered 7x7 defect-free atom array with the AOD scheduler.

The source array contains exactly 49 atoms randomly placed on an N x N grid.
Those atoms are routed into the centered 7x7 target square.

Outputs are written to `render/`:
    - `aod_defect_free_assembly_t0.png`
    - `aod_defect_free_assembly_tfinal.png`
    - `aod_defect_free_assembly.gif`
"""

from __future__ import annotations


import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.atom_config import Grid
from src.routing import *
from src.scheduler.aod_sqrt_time import SqrtTimeAODScheduler

N = 10
TARGET_SIDE = 7
SEED = 260405317
GRID_SPACING_UM = 5.0
COLLISION_RADIUS_UM = 4.0
COLLISION_DT = 5e-6

OUT_DIR = ROOT / "render"
PREFIX = "aod_defect_free_assembly"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    grid = Grid(N=N, d=GRID_SPACING_UM, rc=COLLISION_RADIUS_UM)
    target_atom_count = TARGET_SIDE * TARGET_SIDE
    src = stochastically_loaded_sites(N, target_atom_count, SEED)
    dst = centered_square_targets(N, TARGET_SIDE)
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
        f"({len(src) / (N * N):.1%}), "
        f"target={TARGET_SIDE}x{TARGET_SIDE}"
    )
    print(
        f"aod_steps={len(sequence.steps)}, "
        f"duration={total_us:.3f} us, "
        f"three_step={scheduler.binary_plan.used_three_step}"
    )


if __name__ == "__main__":
    main()
