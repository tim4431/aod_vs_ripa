"""Small labeled routing checks for the C++ RIPA CCBS scheduler.

These are correctness-oriented smoke tests rather than benchmarks. They use
pairwise/labeled `RoutingRequest`s so atom `k` must end at `dst[k]`.

By default the deterministic cases are rendered side-by-side to
`render/ripa_ccbs_c_pairwise_checks.gif`.

Run:
    conda run -n claude python example/ripa_ccbs_c_pairwise_checks.py
"""

from __future__ import annotations

import argparse
import random
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.atom_config import Grid
from src.routing import RoutingRequest, Site, stochastically_loaded_sites
from src.scheduler.ripa_ccbs_c import RIPACCBSCScheduler
from src.sequence import Sequence
from src.visualization import render_animation

GRID_SPACING_UM = 5.0
COLLISION_RADIUS_UM = 4.0
COLLISION_DT = 5e-6
OUT_DIR = ROOT / "render"
PREFIX = "ripa_ccbs_c_pairwise_checks"


@dataclass(frozen=True)
class PairwiseCase:
    name: str
    N: int
    src: list[Site]
    dst: list[Site]
    expect_repair: bool = False
    max_high_level_nodes: int = 5_000
    time_limit: float = 5.0


@dataclass(frozen=True)
class CaseResult:
    name: str
    sequence: Sequence
    duration: float
    high_level_expanded: int
    high_level_generated: int
    segments: int


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


def run_case(case: PairwiseCase, *, verbose: bool) -> CaseResult:
    grid = Grid(N=case.N, d=GRID_SPACING_UM, rc=COLLISION_RADIUS_UM)
    request = RoutingRequest(grid=grid, src=case.src, dst=case.dst, labeled=True)
    scheduler = RIPACCBSCScheduler(
        request,
        collision_dt=COLLISION_DT,
        ccbs_precision=1e-7,
        time_limit=case.time_limit,
        max_high_level_nodes=case.max_high_level_nodes,
    )
    sequence = scheduler.plan()
    assert_request_satisfied(sequence, request)

    solution = scheduler.ccbs_solution
    if solution is None or not solution.found:
        raise RuntimeError(f"{case.name}: CCBS did not report a found solution")
    if case.expect_repair and solution.high_level_expanded <= 1:
        raise RuntimeError(
            f"{case.name}: expected a conflict-repair branch, "
            f"but high_level_expanded={solution.high_level_expanded}"
        )

    segments = sum(len(atom.segments) for atom in sequence.ensemble.atomtrajs)
    if verbose:
        paths = {
            atom_id: [(state.node, round(state.time, 12)) for state in path.states]
            for atom_id, path in sorted(solution.paths.items())
        }
        print(f"{case.name} paths: {paths}")

    return CaseResult(
        name=case.name,
        sequence=sequence,
        duration=sequence.total_duration(),
        high_level_expanded=solution.high_level_expanded,
        high_level_generated=solution.high_level_generated,
        segments=segments,
    )


def deterministic_cases() -> list[PairwiseCase]:
    return [
        PairwiseCase(
            name="single_axis",
            N=5,
            src=[(0, 0)],
            dst=[(4, 0)],
        ),
        PairwiseCase(
            name="independent_parallel",
            N=5,
            src=[(0, 0), (0, 4)],
            dst=[(4, 0), (4, 4)],
        ),
        PairwiseCase(
            name="perpendicular_crossing",
            N=5,
            src=[(0, 2), (2, 0)],
            dst=[(4, 2), (2, 4)],
            expect_repair=True,
        ),
        PairwiseCase(
            name="same_line_opposite",
            N=5,
            src=[(0, 2), (4, 2)],
            dst=[(4, 2), (0, 2)],
            expect_repair=True,
            max_high_level_nodes=20_000,
            time_limit=10.0,
        ),
    ]


def random_case(seed: int, *, N: int, atom_count: int) -> PairwiseCase:
    src = stochastically_loaded_sites(N, atom_count, seed)
    dst = stochastically_loaded_sites(N, atom_count, seed + 1000)
    random.Random(seed + 2000).shuffle(dst)
    return PairwiseCase(
        name=f"random_{seed}",
        N=N,
        src=src,
        dst=dst,
        expect_repair=False,
        max_high_level_nodes=20_000,
        time_limit=10.0,
    )


def print_results(results: list[CaseResult]) -> None:
    print("case                    duration_us  expanded  generated  segments")
    for result in results:
        print(
            f"{result.name:<23} {result.duration * 1e6:>11.3f}  "
            f"{result.high_level_expanded:>8}  "
            f"{result.high_level_generated:>9}  "
            f"{result.segments:>8}"
        )


def render_checks(results: list[CaseResult], *, quality: str) -> Path:
    deterministic_names = {case.name for case in deterministic_cases()}
    panels = {
        result.name: result.sequence
        for result in results
        if result.name in deterministic_names
    }
    out_path = OUT_DIR / f"{PREFIX}.gif"
    render_animation(
        panels,
        out_path,
        view="benchmark",
        quality=quality,
        fps=6 if quality == "speed" else 12,
        time_dilation=2e4,
        hold_seconds=1.0,
        planned_trajectory="full",
        show_atom_ids=True,
        show_routing_on_start=True,
        title="C++ CCBS labeled RIPA routing checks",
    )
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--random-cases", type=int, default=5)
    parser.add_argument("--random-seed-start", type=int, default=101)
    parser.add_argument("--random-atoms", type=int, default=6)
    parser.add_argument("--random-N", type=int, default=6)
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--quality", choices=("speed", "quality"), default="speed")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    cases = deterministic_cases()
    cases.extend(
        random_case(
            args.random_seed_start + k,
            N=args.random_N,
            atom_count=args.random_atoms,
        )
        for k in range(args.random_cases)
    )

    results = [run_case(case, verbose=args.verbose) for case in cases]
    print_results(results)
    if not args.no_render:
        out_path = render_checks(results, quality=args.quality)
        print(f"wrote {out_path}")
    print("all C++ CCBS pairwise checks passed")


if __name__ == "__main__":
    main()
