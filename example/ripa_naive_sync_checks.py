"""Small correctness checks for `RIPANaiveSyncScheduler`.

These are executable examples rather than a full test suite. They cover:
1. labeled reversal of atoms arranged in a line along x,
2. a random labeled routing request with about eight atoms,
3. the same style of route with a stationary obstacle placed on the
   nearest planned highway, forcing the scheduler to choose another lane.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.atom_config import Grid
from src.routing import RoutingRequest, Site
from src.scheduler.ripa_naive_sync import RIPANaiveSyncScheduler
from src.visualization import render_check_outputs


GRID = Grid(N=8, d=5.0, rc=4.0)
COLLISION_DT = 2e-6
OUT_DIR = ROOT / "render"


def assert_request_satisfied(sequence, request: RoutingRequest) -> None:
    report = sequence.validate(dt=COLLISION_DT)
    assert report.ok, report
    final = sequence.final_config()
    if request.labeled:
        by_atom = final.site_of_atom()
        for atom_id, target in enumerate(request.dst):
            assert by_atom[atom_id] == tuple(target), (atom_id, by_atom[atom_id], target)
    else:
        assert final.occupied_sites() == request.target_sites()


def save_check_frames(prefix: str, sequence, request: RoutingRequest) -> None:
    render_check_outputs(
        sequence,
        OUT_DIR,
        prefix,
        static_traps=request.dst,
        view="demo",
        title_prefix=prefix,
        gif_fps=8,
        gif_hold_seconds=0.8,
        quality="speed",
    )


def run_line_reversal() -> None:
    y = 4
    src = [(0, y), (2, y), (4, y), (6, y)]
    dst = list(reversed(src))
    request = RoutingRequest(GRID, src, dst, labeled=True)
    scheduler = RIPANaiveSyncScheduler(request, collision_dt=COLLISION_DT)
    sequence = scheduler.plan()
    assert_request_satisfied(sequence, request)
    save_check_frames("ripa_naive_sync_line_reversal", sequence, request)
    print(
        "line reversal:",
        f"cycles={scheduler.clock_cycle}",
        f"steps={len(sequence.steps)}",
        f"duration_us={sequence.total_duration() * 1e6:.3f}",
    )


def run_random_routing() -> None:
    rng = random.Random(20260429)
    storage_sites = [(i, j) for i in range(0, GRID.N, 2) for j in range(0, GRID.N, 2)]
    src = rng.sample(storage_sites, 8)
    dst = rng.sample(storage_sites, 8)
    request = RoutingRequest(GRID, src, dst, labeled=True)
    scheduler = RIPANaiveSyncScheduler(request, collision_dt=COLLISION_DT)
    sequence = scheduler.plan()
    assert_request_satisfied(sequence, request)
    save_check_frames("ripa_naive_sync_random", sequence, request)
    print(
        "random labeled routing:",
        f"cycles={scheduler.clock_cycle}",
        f"steps={len(sequence.steps)}",
        f"duration_us={sequence.total_duration() * 1e6:.3f}",
    )


def run_obstacle_routing() -> None:
    src: list[Site] = [(0, 0), (2, 1)]
    dst: list[Site] = [(6, 0), (2, 1)]
    request = RoutingRequest(GRID, src, dst, labeled=True)
    scheduler = RIPANaiveSyncScheduler(request, collision_dt=COLLISION_DT)
    sequence = scheduler.plan()
    assert_request_satisfied(sequence, request)
    save_check_frames("ripa_naive_sync_obstacle", sequence, request)

    obstacle = sequence.ensemble.atomtraj_by_id(1)
    assert obstacle.segments == []

    moving = sequence.ensemble.atomtraj_by_id(0)
    blocked_highway_used = any(
        seg.channel == "row"
        and seg.start_pos[1] == 1
        and seg.end_pos[1] == 1
        and min(seg.start_pos[0], seg.end_pos[0]) <= 2 <= max(seg.start_pos[0], seg.end_pos[0])
        for seg in moving.segments
    )
    assert not blocked_highway_used
    print(
        "obstacle reroute:",
        f"cycles={scheduler.clock_cycle}",
        f"steps={len(sequence.steps)}",
        f"duration_us={sequence.total_duration() * 1e6:.3f}",
    )


def main() -> None:
    run_line_reversal()
    run_random_routing()
    run_obstacle_routing()


if __name__ == "__main__":
    main()
