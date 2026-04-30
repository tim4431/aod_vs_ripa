"""Demo: pairwise random-to-random routing with RIPAPebbleAdvScheduler.

Unlike the defect-free assembly examples, this request is labeled: atom `k`
starts at `src[k]` and must end at `dst[k]`. The rendered atom IDs therefore
matter and should match the requested pairwise assignment at the final frame.

Outputs are written to `render/`:
    - `ripa_pairwise_random_adv_t0.png`
    - `ripa_pairwise_random_adv_tfinal.png`
    - `ripa_pairwise_random_adv.gif`
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
from src.routing import RoutingRequest, stochastically_loaded_sites
from src.scheduler.ripa_pebble_adv import RIPAPebbleAdvScheduler
from src.sequence import Sequence
from src.visualization import render_check_outputs

N = 10
ATOM_COUNT = 24
SRC_SEED = 260405318
DST_SEED = 260406318
PAIRING_SEED = 260407318
GRID_SPACING_UM = 5.0
COLLISION_RADIUS_UM = 4.0
COLLISION_DT = 5e-6

OUT_DIR = ROOT / "render"
PREFIX = "ripa_pairwise_random_adv"


def make_request() -> RoutingRequest:
    grid = Grid(N=N, d=GRID_SPACING_UM, rc=COLLISION_RADIUS_UM)
    src = stochastically_loaded_sites(N, ATOM_COUNT, SRC_SEED)
    dst = stochastically_loaded_sites(N, ATOM_COUNT, DST_SEED)
    random.Random(PAIRING_SEED).shuffle(dst)
    return RoutingRequest(grid=grid, src=src, dst=dst, labeled=True)


def assert_request_satisfied(sequence: Sequence, request: RoutingRequest) -> None:
    report = sequence.validate(dt=COLLISION_DT)
    if not report.ok:
        raise RuntimeError(f"collision validation failed: {report}")

    site_by_atom = sequence.final_config().site_of_atom()
    for atom_id, target in enumerate(request.dst):
        if site_by_atom.get(atom_id) != tuple(target):
            raise RuntimeError(
                f"atom {atom_id} ended at {site_by_atom.get(atom_id)}, "
                f"expected {tuple(target)}"
            )


def render_pairwise_outputs(sequence: Sequence, request: RoutingRequest) -> None:
    gif_frame_dt = max(sequence.total_duration() / 100.0, 10e-6)
    render_check_outputs(
        sequence,
        OUT_DIR,
        PREFIX,
        static_traps=request.dst,
        view="demo",
        show_atom_ids=True,
        title_prefix=f"RIPA pebble adv pairwise random - {ATOM_COUNT} atoms",
        gif_fps=6,
        gif_frame_dt=gif_frame_dt,
        gif_hold_seconds=1.5,
        quality="speed",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="skip PNG/GIF rendering and only print the benchmark table",
    )
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    request = make_request()
    print(
        f"N={N}, labeled_atoms={len(request.src)}, "
        f"src_seed={SRC_SEED}, dst_seed={DST_SEED}, pairing_seed={PAIRING_SEED}"
    )

    results = benchmark_schedulers(
        request,
        {
            "ripa_pebble_adv_labeled": lambda req: RIPAPebbleAdvScheduler(
                req,
                collision_dt=COLLISION_DT,
            ),
        },
        validate_dt=COLLISION_DT,
    )
    print(format_benchmark_table(results))
    for result in results:
        if not result.ok and result.error is not None:
            print(f"{result.name} error: {type(result.error).__name__}: {result.error}")

    result = results[0]
    if not result.ok or result.sequence is None:
        raise RuntimeError("ripa_pebble_adv_labeled did not produce a valid route")
    assert_request_satisfied(result.sequence, request)

    if not args.no_render:
        render_pairwise_outputs(result.sequence, request)


if __name__ == "__main__":
    main()
