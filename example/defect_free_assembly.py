"""Defect-free assembly benchmark: AOD vs the dense-aware RIPA pebble.

Mirrors `aod_defect_free_assembly.py` for the problem setup (same N,
seed, fill rate, grid spacing, collision radius). The same dense
RoutingRequest is fed to two schedulers via `benchmark_schedulers`:

  - `aod_sqrt_time`   — SqrtTimeAODScheduler (sync AOD lattice ops).
  - `ripa_pebble_adv` — RIPAPebbleAdvScheduler (async per-atom RIPA,
                        no highway concept; empty cells used as buffers).

Outputs in `render/`:
    - `defect_free_assembly_t0.png`
    - `defect_free_assembly_tfinal.png`
    - `defect_free_assembly.gif`        (rendered for the RIPA result)
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
from src.scheduler.ripa_pebble_adv import RIPAPebbleAdvScheduler
from src.sequence import Sequence
from src.visualization import render_check_outputs


N = 10
FILL_RATE = 0.50
SEED = 260405317
GRID_SPACING_UM = 5.0
COLLISION_RADIUS_UM = 4.0
COLLISION_DT = 5e-6

OUT_DIR = ROOT / "render"
PREFIX = "defect_free_assembly"


def random_loaded_sites(N: int, fill_rate: float, seed: int) -> list[Site]:
    """Random ~50% fill across the full N x N grid (matches AOD example)."""
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


def make_request() -> tuple[RoutingRequest, int]:
    grid = Grid(N=N, d=GRID_SPACING_UM, rc=COLLISION_RADIUS_UM)
    src = random_loaded_sites(N, FILL_RATE, SEED)
    dst, square_side = square_plus_parking_targets(N, len(src))
    return RoutingRequest(grid=grid, src=src, dst=dst, labeled=False), square_side


def scheduler_factories():
    return {
        "aod_sqrt_time": lambda req: SqrtTimeAODScheduler(
            req,
            collision_dt=COLLISION_DT,
            validate_final=True,
        ),
        "ripa_pebble_adv": lambda req: RIPAPebbleAdvScheduler(
            req,
            collision_dt=COLLISION_DT,
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


def render_pebble_adv_outputs(
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
        title_prefix=f"RIPA pebble-adv defect-free assembly - target {square_side}x{square_side}",
        gif_fps=6,
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
    request, square_side = make_request()
    print(
        f"seed={SEED}, N={N}, loaded={len(request.src)} atoms "
        f"({len(request.src) / (N * N):.1%}), target={square_side}x{square_side} "
        f"plus parking"
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

    pebble = next(result for result in results if result.name == "ripa_pebble_adv")
    if not pebble.ok or pebble.sequence is None:
        raise RuntimeError("ripa_pebble_adv did not produce a valid defect-free assembly")
    assert_request_satisfied(pebble.sequence, request)

    if not args.no_render:
        plot_benchmark(results, OUT_DIR / f"{PREFIX}_benchmark.png")
        render_pebble_adv_outputs(pebble.sequence, request, square_side)


if __name__ == "__main__":
    main()
