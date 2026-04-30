"""Demo and benchmark: unlabeled defect-free assembly with RIPA pebble routing.

The source contains exactly 49 atoms randomly placed on the even storage sites
of a 20x20 grid. Those atoms are routed to a centered 7x7 target square that
also lies on the even storage sites.

For the RIPA highway scheduler, the source/target sites are drawn from the
storage subgrid. With `STORAGE_PERIOD = 2`, even rows/columns store atoms and
odd rows/columns are left as transport highways.

Outputs are written to `render/`:
    - `defect_free_assembly_t0.png`
    - `defect_free_assembly_tfinal.png`
    - `defect_free_assembly.gif`
    - `defect_free_assembly_benchmark.png`
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
from src.routing import RoutingRequest, Site
from src.scheduler.aod_sqrt_time import SqrtTimeAODScheduler
from src.scheduler.ripa_naive_sync import RIPANaiveSyncScheduler
from src.scheduler.ripa_pebble import RIPAPebbleScheduler
from src.sequence import Sequence
from src.visualization import draw_frame, render_check_outputs

N = 20
TARGET_SIDE = 7
SEED = 260405317
STORAGE_PERIOD = 2
STORAGE_OFFSET = 0
GRID_SPACING_UM = 5.0
COLLISION_RADIUS_UM = 4.0
COLLISION_DT = 5e-6

OUT_DIR = ROOT / "render"
PREFIX = "defect_free_assembly"


def storage_sites(
    N: int,
    *,
    period: int = STORAGE_PERIOD,
    offset: int = STORAGE_OFFSET,
) -> list[Site]:
    return [(i, j) for i in range(offset, N, period) for j in range(offset, N, period)]


def random_loaded_sites(sites: list[Site], atom_count: int, seed: int) -> list[Site]:
    if not 0 <= atom_count <= len(sites):
        raise ValueError("atom_count must be between 0 and the number of sites")

    rng = random.Random(seed)
    loaded = list(sites)
    rng.shuffle(loaded)
    return sorted(loaded[:atom_count])


def centered_storage_square_targets(N: int, sites: list[Site], side: int) -> list[Site]:
    """Centered side x side target square on the provided storage sites."""
    rows = sorted({site[0] for site in sites})
    cols = sorted({site[1] for site in sites})
    if not 0 < side <= min(len(rows), len(cols)):
        raise ValueError("target side must fit within the storage subgrid")

    row_start = _centered_window_start(rows, side, (N - 1) / 2.0)
    col_start = _centered_window_start(cols, side, (N - 1) / 2.0)
    return [
        (rows[ri], cols[cj])
        for ri in range(row_start, row_start + side)
        for cj in range(col_start, col_start + side)
    ]


def _centered_window_start(values: list[int], side: int, center: float) -> int:
    return min(
        range(len(values) - side + 1),
        key=lambda start: abs((values[start] + values[start + side - 1]) / 2.0 - center),
    )


def make_request() -> tuple[RoutingRequest, int, int]:
    grid = Grid(N=N, d=GRID_SPACING_UM, rc=COLLISION_RADIUS_UM)
    allowed_sites = storage_sites(N)
    src = random_loaded_sites(allowed_sites, TARGET_SIDE * TARGET_SIDE, SEED)
    dst = centered_storage_square_targets(N, allowed_sites, TARGET_SIDE)
    return (
        RoutingRequest(grid=grid, src=src, dst=dst, labeled=False),
        TARGET_SIDE,
        len(allowed_sites),
    )


def scheduler_factories():
    return {
        "ripa_pebble": lambda req: RIPAPebbleScheduler(
            req,
            collision_dt=COLLISION_DT,
            highway_period=STORAGE_PERIOD,
            storage_offset=STORAGE_OFFSET,
        ),
        "ripa_naive_sync": lambda req: RIPANaiveSyncScheduler(
            req,
            collision_dt=COLLISION_DT,
            highway_period=STORAGE_PERIOD,
            storage_offset=STORAGE_OFFSET,
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
        raise RuntimeError("final atom set does not match defect-free target set")


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
        title="Defect-free assembly benchmark",
    )
    fig.savefig(out_path, dpi=120, facecolor="white")
    plt.close(fig)


def render_pebble_outputs(
    sequence: Sequence,
    request: RoutingRequest,
    square_side: int,
) -> None:
    static_traps = request.dst
    render_check_outputs(
        sequence,
        OUT_DIR,
        PREFIX,
        static_traps=static_traps,
        view="demo",
        title_prefix=f"RIPA pebble defect-free assembly - target {square_side}x{square_side}",
        gif_fps=6,
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
    request, square_side, storage_count = make_request()
    print(
        f"seed={SEED}, N={N}, storage_sites={storage_count}, "
        f"loaded={len(request.src)} atoms ({len(request.src) / storage_count:.1%}), "
        f"target={square_side}x{square_side}"
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

    pebble = next(result for result in results if result.name == "ripa_pebble")
    if not pebble.ok or pebble.sequence is None:
        raise RuntimeError("ripa_pebble did not produce a valid defect-free assembly")
    assert_request_satisfied(pebble.sequence, request)

    if not args.no_render:
        static_traps = request.dst
        plot_benchmark(results, OUT_DIR / f"{PREFIX}_benchmark.png", static_traps)
        render_pebble_outputs(pebble.sequence, request, square_side)


if __name__ == "__main__":
    main()
