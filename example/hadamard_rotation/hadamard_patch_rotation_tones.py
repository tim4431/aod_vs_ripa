"""EOM tone view of the RIPA-async Hadamard patch rotation.

Reuses `ManualRIPAHadamardPatchRotationScheduler` from the side-by-side
benchmark example, but renders only the RIPA-async sequence in the
`detail` view so each atom's row/col EOM tone is visible. The patch
sits on a 5x5 grid with storage period 1 (atom spacing == grid
spacing), which keeps the tone axes compact and readable.

Usage:
    python example/hadamard_rotation/hadamard_patch_rotation_tones.py            # quick check -> render/
    python example/hadamard_rotation/hadamard_patch_rotation_tones.py --demo     # quality GIF -> demo/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from example.hadamard_rotation.hadamard_patch_rotation import (
    ManualRIPAHadamardPatchRotationScheduler,
    hadamard_rotation_targets,
    patch_rotation_colors,
)
from src.atom_config import Grid
from src.routing import RoutingRequest, centered_storage_square
from src.visualization import render_animation

N = 7
PATCH_SIDE = 5
STORAGE_PERIOD = 1
GRID_SPACING_UM = 15.0
COLLISION_RADIUS_UM = 3.0

ATOM_SCALE = 1.4
TRAP_SCALE = 0.4

PREFIX = "hadamard_patch_rotation_tones"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="EOM tone view of the RIPA-async Hadamard patch rotation."
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
    src = centered_storage_square(N, PATCH_SIDE, STORAGE_PERIOD)
    dst = hadamard_rotation_targets(src)
    request = RoutingRequest(grid=grid, src=src, dst=dst, labeled=True)

    sequence = ManualRIPAHadamardPatchRotationScheduler(request).plan()
    report = sequence.validate(dt=2e-7)
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
        time_dilation=2e4,
        hold_seconds=1.5,
        planned_trajectory="full",
        atom_colors=patch_rotation_colors(src),
        show_atom_ids=True,
        show_routing_on_start=True,
        show_color_code=True,
        atom_scale=ATOM_SCALE,
        trap_scale=TRAP_SCALE,
        title=f"RIPA Hadamard patch rotation EOM tones ({PATCH_SIDE}x{PATCH_SIDE})",
    )
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
