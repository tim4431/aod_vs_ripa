"""Harder labeled routing checks for the C++ RIPA CCBS scheduler.

These are correctness-oriented checks rather than benchmarks. They use
pairwise/labeled `RoutingRequest`s so atom `k` must end at `dst[k]`.
The default fixtures are dense enough to force non-trivial high-level CCBS
repair while still finishing quickly on a development machine.

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
    min_high_level_expanded: int = 1
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
    if solution.high_level_expanded < case.min_high_level_expanded:
        raise RuntimeError(
            f"{case.name}: expected at least {case.min_high_level_expanded} "
            "high-level expansions, "
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
            name="eight_lane_weave",
            N=8,
            src=[
                (0, 3),
                (1, 0),
                (2, 6),
                (3, 1),
                (4, 7),
                (5, 2),
                (6, 5),
                (7, 4),
            ],
            dst=[
                (7, 4),
                (6, 5),
                (5, 2),
                (4, 7),
                (3, 1),
                (2, 6),
                (1, 0),
                (0, 3),
            ],
            min_high_level_expanded=10,
            max_high_level_nodes=60_000,
            time_limit=10.0,
        ),
        seeded_dense_case(
            name="dense_10_seed202",
            N=8,
            atom_count=10,
            seed=202,
            min_high_level_expanded=100,
            max_high_level_nodes=60_000,
            time_limit=12.0,
        ),
        seeded_dense_case(
            name="dense_12_seed201",
            N=8,
            atom_count=12,
            seed=201,
            min_high_level_expanded=100,
            max_high_level_nodes=60_000,
            time_limit=12.0,
        ),
        seeded_dense_case(
            name="dense_14_seed206",
            N=10,
            atom_count=14,
            seed=206,
            min_high_level_expanded=100,
            max_high_level_nodes=80_000,
            time_limit=12.0,
        ),
    ]


def seeded_dense_case(
    *,
    name: str,
    N: int,
    atom_count: int,
    seed: int,
    min_high_level_expanded: int,
    max_high_level_nodes: int,
    time_limit: float,
) -> PairwiseCase:
    src = stochastically_loaded_sites(N, atom_count, seed)
    dst = stochastically_loaded_sites(N, atom_count, seed + 1000)
    random.Random(seed + 2000).shuffle(dst)
    return PairwiseCase(
        name=name,
        N=N,
        src=src,
        dst=dst,
        min_high_level_expanded=min_high_level_expanded,
        max_high_level_nodes=max_high_level_nodes,
        time_limit=time_limit,
    )


def four_by_four_reversal_stress_case(
    *,
    time_limit: float,
    max_high_level_nodes: int,
) -> PairwiseCase:
    """Literal 16-atom 4x4 reversal.

    This is intentionally opt-in: it is a useful stress probe, but the current
    exact labeled CCBS backend usually exhausts the default search budget.
    """

    src = [(i, j) for i in range(4) for j in range(4)]
    dst = [(3 - i, 3 - j) for i, j in src]
    return PairwiseCase(
        name="reverse_4x4_array_stress",
        N=4,
        src=src,
        dst=dst,
        min_high_level_expanded=1,
        max_high_level_nodes=max_high_level_nodes,
        time_limit=time_limit,
    )


def random_case(seed: int, *, N: int, atom_count: int) -> PairwiseCase:
    src = stochastically_loaded_sites(N, atom_count, seed)
    dst = stochastically_loaded_sites(N, atom_count, seed + 1000)
    random.Random(seed + 2000).shuffle(dst)
    return PairwiseCase(
        name=f"random_{seed}",
        N=N,
        src=src,
        dst=dst,
        min_high_level_expanded=1,
        max_high_level_nodes=60_000,
        time_limit=12.0,
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
    deterministic_names.add("reverse_4x4_array_stress")
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
    parser.add_argument("--random-cases", type=int, default=0)
    parser.add_argument("--random-seed-start", type=int, default=301)
    parser.add_argument("--random-atoms", type=int, default=12)
    parser.add_argument("--random-N", type=int, default=8)
    parser.add_argument(
        "--include-4x4-reversal-stress",
        action="store_true",
        help="also try the literal 16-atom 4x4 reversal stress case",
    )
    parser.add_argument("--stress-time-limit", type=float, default=30.0)
    parser.add_argument("--stress-max-high-level-nodes", type=int, default=100_000)
    parser.add_argument(
        "--strict-stress",
        action="store_true",
        help="fail the script if the opt-in 4x4 stress case does not solve",
    )
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
    stress_error: RuntimeError | None = None
    if args.include_4x4_reversal_stress:
        stress_case = four_by_four_reversal_stress_case(
            time_limit=args.stress_time_limit,
            max_high_level_nodes=args.stress_max_high_level_nodes,
        )
        try:
            results.append(run_case(stress_case, verbose=args.verbose))
        except RuntimeError as exc:
            stress_error = exc
            if args.strict_stress:
                raise

    print_results(results)
    if stress_error is not None:
        print(f"reverse_4x4_array_stress did not solve: {stress_error}")
    if not args.no_render:
        out_path = render_checks(results, quality=args.quality)
        print(f"wrote {out_path}")
    print("all C++ CCBS pairwise checks passed")


if __name__ == "__main__":
    main()
