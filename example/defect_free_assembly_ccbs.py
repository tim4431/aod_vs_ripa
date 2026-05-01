"""Demo: CCBS-backed RIPA assembly on the same grid as the AOD example.

This uses a fully populated 7x7 target set inside a 10x10 array, matching
`aod_defect_free_assembly.py` rather than the older sparse storage/highway
RIPA setup.

Outputs are written to `render/`:
    - `defect_free_assembly_ccbs_t0.png`
    - `defect_free_assembly_ccbs_tfinal.png`
    - `defect_free_assembly_ccbs.gif`
    - `defect_free_assembly_ccbs_benchmark.png`
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
from src.benchmark import BenchmarkResult, benchmark_schedulers, format_benchmark_table
from src.routing import *
from src.scheduler.aod_sqrt_time import SqrtTimeAODScheduler
from src.scheduler.ripa_ccbs import RIPACCBSScheduler
from src.sequence import Sequence
from src.visualization import draw_frame, render_check_outputs

N = 10
TARGET_SIDE = 7
SEED = 260405317
GRID_SPACING_UM = 5.0
COLLISION_RADIUS_UM = 4.0
COLLISION_DT = 5e-6

OUT_DIR = ROOT / "render"
PREFIX = "defect_free_assembly_ccbs"


def make_request(
    *,
    N_: int = N,
    target_side: int = TARGET_SIDE,
    seed: int = SEED,
) -> RoutingRequest:
    grid = Grid(N=N_, d=GRID_SPACING_UM, rc=COLLISION_RADIUS_UM)
    target_atom_count = target_side * target_side
    src = stochastically_loaded_sites(N_, target_atom_count, seed)
    dst = centered_square_targets(N_, target_side)
    return RoutingRequest(grid=grid, src=src, dst=dst, labeled=False)


def scheduler_factories(*, ccbs_time_limit: float):
    return {
        "ripa_ccbs": lambda req: RIPACCBSScheduler(
            req,
            collision_dt=COLLISION_DT,
            time_limit=ccbs_time_limit,
            max_high_level_nodes=20_000,
        ),
        "aod_sqrt_time": lambda req: SqrtTimeAODScheduler(
            req,
            collision_dt=COLLISION_DT,
            validate_final=True,
        ),
    }


def assert_request_satisfied(sequence: Sequence, request: RoutingRequest) -> None:
    report = sequence.validate(dt=COLLISION_DT)
    if not report.ok:
        raise RuntimeError(f"collision validation failed: {report}")
    if sequence.final_config().occupied_sites() != request.target_sites():
        raise RuntimeError("final atom set does not match the 7x7 target set")


def plot_benchmark(
    results: list[BenchmarkResult],
    out_path: Path,
    static_traps: list[Site],
) -> None:
    import matplotlib.pyplot as plt

    sequences = {
        result.name: result.sequence
        for result in results
        if result.ok and result.sequence is not None
    }
    if not sequences:
        raise RuntimeError("no successful scheduler result to render")

    fig, _ = draw_frame(
        sequences,
        0.0,
        view="benchmark",
        static_traps=static_traps,
        quality="speed",
        show_atom_ids=False,
        title="Defect-free assembly benchmark - dense geometry",
    )
    fig.savefig(out_path, dpi=120, facecolor="white")
    plt.close(fig)


def render_ccbs_outputs(
    sequence: Sequence,
    request: RoutingRequest,
    *,
    target_side: int,
) -> None:
    render_check_outputs(
        sequence,
        OUT_DIR,
        PREFIX,
        static_traps=request.dst,
        view="demo",
        title_prefix=f"RIPA CCBS defect-free assembly - target {target_side}x{target_side}",
        gif_fps=6,
        gif_time_dilation=3e4,
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
    parser.add_argument("--N", type=int, default=N, help="grid side length")
    parser.add_argument(
        "--target-side",
        type=int,
        default=TARGET_SIDE,
        help="side length of the centered target square",
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--ccbs-time-limit",
        type=float,
        default=30.0,
        help="CCBS planning time limit in seconds",
    )
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    request = make_request(
        N_=args.N,
        target_side=args.target_side,
        seed=args.seed,
    )
    print(
        f"seed={args.seed}, N={args.N}, loaded={len(request.src)} atoms "
        f"({len(request.src) / (args.N * args.N):.1%}), "
        f"target={args.target_side}x{args.target_side}"
    )

    results = benchmark_schedulers(
        request,
        scheduler_factories(ccbs_time_limit=args.ccbs_time_limit),
        validate_dt=COLLISION_DT,
    )
    print(format_benchmark_table(results))
    for result in results:
        if not result.ok and result.error is not None:
            print(f"{result.name} error: {type(result.error).__name__}: {result.error}")

    ccbs = next(result for result in results if result.name == "ripa_ccbs")
    if not ccbs.ok or ccbs.sequence is None:
        raise RuntimeError("ripa_ccbs did not produce a valid assembly")
    assert_request_satisfied(ccbs.sequence, request)

    if not args.no_render:
        plot_benchmark(
            results,
            OUT_DIR / f"{PREFIX}_benchmark.png",
            list(request.dst),
        )
        render_ccbs_outputs(
            ccbs.sequence,
            request,
            target_side=args.target_side,
        )


if __name__ == "__main__":
    main()
