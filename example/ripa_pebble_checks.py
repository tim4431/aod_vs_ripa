"""Executable checks for `RIPAPebbleScheduler`.

The examples cover:
1. labeled line reversal with atom swaps,
2. unlabeled set routing with route-cost assignment,
3. a random labeled request compared with the synchronous highway baseline.

Run with `--render` to save a small set of visual checks under `render/`.
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
from src.scheduler.ripa_pebble import RIPAPebbleScheduler
from src.sequence import Sequence
from src.visualization import render_check_outputs


GRID = Grid(N=10, d=5.0, rc=4.0)
COLLISION_DT = 2e-6
OUT_DIR = ROOT / "render"


def assert_request_satisfied(sequence: Sequence, request: RoutingRequest) -> None:
    report = sequence.validate(dt=COLLISION_DT)
    assert report.ok, report

    final = sequence.final_config()
    if request.labeled:
        by_atom = final.site_of_atom()
        for atom_id, target in enumerate(request.dst):
            assert by_atom[atom_id] == tuple(target), (atom_id, by_atom[atom_id], target)
        return

    assert final.occupied_sites() == request.target_sites()


def maybe_render(prefix: str, sequence: Sequence, request: RoutingRequest, render: bool) -> None:
    if not render:
        return
    render_check_outputs(
        sequence,
        OUT_DIR,
        prefix,
        static_traps=request.dst,
        view="demo",
        show_planned=True,
        addressed_style="edge",
        title_prefix=prefix,
        gif_fps=8,
        gif_hold_seconds=0.8,
        quality="speed",
        use_multiprocessing=False,
    )


def run_line_reversal(render: bool) -> None:
    y = 4
    src = [(0, y), (2, y), (4, y), (6, y)]
    dst = list(reversed(src))
    request = RoutingRequest(GRID, src, dst, labeled=True)
    scheduler = RIPAPebbleScheduler(request, collision_dt=COLLISION_DT)
    sequence = scheduler.plan()
    assert_request_satisfied(sequence, request)
    maybe_render("ripa_pebble_line_reversal", sequence, request, render)
    print(
        "line reversal:",
        f"steps={len(sequence.steps)}",
        f"duration_us={sequence.total_duration() * 1e6:.3f}",
    )


def run_unlabeled_set_routing(render: bool) -> None:
    src = [(0, 0), (2, 0), (4, 0), (6, 0), (8, 0)]
    dst = [(0, 4), (2, 4), (4, 4), (6, 4), (8, 4)]
    request = RoutingRequest(GRID, src, dst, labeled=False)
    scheduler = RIPAPebbleScheduler(request, collision_dt=COLLISION_DT)
    sequence = scheduler.plan()
    assert_request_satisfied(sequence, request)
    maybe_render("ripa_pebble_unlabeled_set", sequence, request, render)
    print(
        "unlabeled set routing:",
        f"steps={len(sequence.steps)}",
        f"duration_us={sequence.total_duration() * 1e6:.3f}",
    )


def run_random_benchmark(render: bool) -> None:
    rng = random.Random(20260430)
    storage_sites = [(i, j) for i in range(0, GRID.N, 2) for j in range(0, GRID.N, 2)]
    src = rng.sample(storage_sites, 12)
    dst = rng.sample(storage_sites, 12)
    request = RoutingRequest(GRID, src, dst, labeled=True)

    results = benchmark_schedulers(
        request,
        {
            "ripa_pebble": lambda req: RIPAPebbleScheduler(
                req,
                collision_dt=COLLISION_DT,
            ),
            "ripa_naive_sync": lambda req: RIPANaiveSyncScheduler(
                req,
                collision_dt=COLLISION_DT,
            ),
        },
        validate_dt=COLLISION_DT,
    )
    print("random labeled benchmark:")
    print(format_benchmark_table(results))

    best = next(result for result in results if result.name == "ripa_pebble")
    assert best.sequence is not None
    assert_request_satisfied(best.sequence, request)
    maybe_render("ripa_pebble_random", best.sequence, request, render)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--render",
        action="store_true",
        help="write PNG/GIF visual checks under render/",
    )
    args = parser.parse_args()

    run_line_reversal(args.render)
    run_unlabeled_set_routing(args.render)
    run_random_benchmark(args.render)


if __name__ == "__main__":
    main()
