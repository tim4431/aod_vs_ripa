"""Demo and benchmark: unlabeled defect-free assembly with RIPA pebble routing.

This mirrors the task shape from `aod_defect_free_assembly.py`: a stochastic
loaded source is routed to the largest compact defect-free square possible, with
leftover atoms parked nearby so source and target counts match.

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
import math
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
from src.visualization import render_check_outputs


N = 16
FILL_RATE = 0.50
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
    return [
        (i, j)
        for i in range(offset, N, period)
        for j in range(offset, N, period)
    ]


def random_loaded_sites(sites: list[Site], fill_rate: float, seed: int) -> list[Site]:
    rng = random.Random(seed)
    loaded = [site for site in sites if rng.random() < fill_rate]
    if not loaded:
        loaded.append(sites[len(sites) // 2])
    return loaded


def square_plus_parking_targets(sites: list[Site], atom_count: int) -> tuple[list[Site], int]:
    """Largest centered LxL defect-free square plus parking for leftovers."""
    if atom_count <= 0:
        return [], 0

    rows = sorted({site[0] for site in sites})
    cols = sorted({site[1] for site in sites})
    side_limit = min(len(rows), len(cols))
    L = min(int(math.isqrt(atom_count)), side_limit)
    if L == 0:
        return [], 0

    row_start = (len(rows) - L) // 2
    col_start = (len(cols) - L) // 2
    square = [
        (rows[ri], cols[cj])
        for ri in range(row_start, row_start + L)
        for cj in range(col_start, col_start + L)
    ]
    square_set = set(square)
    leftovers = atom_count - len(square)

    parking: list[Site] = []
    for site in _parking_site_order(sites, rows, cols, row_start, col_start, L):
        if site not in square_set:
            parking.append(site)
            if len(parking) == leftovers:
                break

    if len(parking) != leftovers:
        raise ValueError("not enough parking sites for leftover atoms")
    return square + parking, L


def _parking_site_order(
    sites: list[Site],
    rows: list[int],
    cols: list[int],
    row_start: int,
    col_start: int,
    L: int,
) -> list[Site]:
    row_index = {row: idx for idx, row in enumerate(rows)}
    col_index = {col: idx for idx, col in enumerate(cols)}
    row_center = (len(rows) - 1) / 2.0
    col_center = (len(cols) - 1) / 2.0
    square_row_center = row_start + (L - 1) / 2.0
    square_col_center = col_start + (L - 1) / 2.0

    return sorted(
        sites,
        key=lambda site: (
            abs(row_index[site[0]] - square_row_center)
            + abs(col_index[site[1]] - square_col_center),
            abs(row_index[site[0]] - row_center)
            + abs(col_index[site[1]] - col_center),
            site,
        ),
    )


def make_request() -> tuple[RoutingRequest, int, int]:
    grid = Grid(N=N, d=GRID_SPACING_UM, rc=COLLISION_RADIUS_UM)
    allowed_sites = storage_sites(N)
    src = random_loaded_sites(allowed_sites, FILL_RATE, SEED)
    dst, square_side = square_plus_parking_targets(allowed_sites, len(src))
    return RoutingRequest(grid=grid, src=src, dst=dst, labeled=False), square_side, len(allowed_sites)


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


def plot_benchmark(results: list[BenchmarkResult], out_path: Path) -> None:
    import matplotlib.pyplot as plt

    names = [result.name for result in results]
    values = [result.total_time_us if result.ok else 0.0 for result in results]
    colors = ["#2f6f6d" if result.ok else "#b8b8b8" for result in results]

    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    bars = ax.bar(names, values, color=colors)
    ax.set_ylabel("arrangement time (us)")
    ax.set_title("Defect-free assembly benchmark")
    ax.grid(axis="y", color="#e6e6e6", linewidth=0.8)
    ax.set_axisbelow(True)

    for bar, result in zip(bars, results):
        x = bar.get_x() + bar.get_width() / 2.0
        if result.ok:
            label = f"{result.total_time_us:.0f} us"
            y = bar.get_height()
        else:
            label = "failed"
            y = max(values) * 0.04 if values else 1.0
        ax.text(x, y, label, ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def render_pebble_outputs(
    sequence: Sequence,
    request: RoutingRequest,
    square_side: int,
) -> None:
    static_traps = request.dst[: square_side * square_side]
    render_check_outputs(
        sequence,
        OUT_DIR,
        PREFIX,
        static_traps=static_traps,
        show_planned=True,
        addressed_style="edge",
        title_prefix=f"RIPA pebble defect-free assembly - target {square_side}x{square_side}",
        gif_fps=6,
        gif_frames=100,
        gif_hold_seconds=1.5,
        optimize="speed",
        use_multiprocessing=False,
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
        f"target={square_side}x{square_side} plus parking"
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
        plot_benchmark(results, OUT_DIR / f"{PREFIX}_benchmark.png")
        render_pebble_outputs(pebble.sequence, request, square_side)


if __name__ == "__main__":
    main()
