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


def make_request() -> RoutingRequest:
    grid = Grid(N=N, d=GRID_SPACING_UM, rc=COLLISION_RADIUS_UM)
    target_atom_count = TARGET_SIDE * TARGET_SIDE
    src = stochastically_loaded_sites(N, target_atom_count, SEED)
    dst = centered_square_targets(N, TARGET_SIDE)
    return RoutingRequest(grid=grid, src=src, dst=dst, labeled=False)


def scheduler_factories():
    return {
        "ripa_ccbs": lambda req: RIPACCBSScheduler(
            req,
            collision_dt=COLLISION_DT,
            time_limit=30.0,
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


def render_ccbs_outputs(sequence: Sequence, request: RoutingRequest) -> None:
    gif_frame_dt = max(sequence.total_duration() / 100.0, 10e-6)
    render_check_outputs(
        sequence,
        OUT_DIR,
        PREFIX,
        static_traps=request.dst,
        view="demo",
        title_prefix=f"RIPA CCBS defect-free assembly - target {TARGET_SIDE}x{TARGET_SIDE}",
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
        f"seed={SEED}, N={N}, loaded={len(request.src)} atoms "
        f"({len(request.src) / (N * N):.1%}), target={TARGET_SIDE}x{TARGET_SIDE}"
    )

    results = benchmark_schedulers(
        request,
        scheduler_factories(),
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
        render_ccbs_outputs(ccbs.sequence, request)


if __name__ == "__main__":
    main()
