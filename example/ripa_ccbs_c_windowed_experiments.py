"""Experiments for whether the windowed C++ CCBS wrapper is generally better.

The current windowed scheduler is specialized: it decomposes reciprocal
single-axis line swaps, and otherwise falls back to the plain C++ backend. This
script makes that behavior visible with two deterministic workloads:

1. a 1D atom train inversion embedded in the 2D RIPA grid;
2. a labeled 25-atom random source-to-random-target routing instance.

Usage:
    python example/ripa_ccbs_c_windowed_experiments.py
    python example/ripa_ccbs_c_windowed_experiments.py --render
    python example/ripa_ccbs_c_windowed_experiments.py --only random
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.atom_config import Grid
from src.moving_sequence import MovingSequence
from src.routing import RoutingRequest, Site
from src.scheduler.ccbs_c import RIPACCBSCScheduler, RIPACCBSWindowedScheduler
from src.visualization import render_animation

GRID_SPACING_UM = 5.0
COLLISION_RADIUS_UM = 4.0
COLLISION_DT = 5e-6
CCBS_PRECISION = 1e-7
PREFIX = "ripa_ccbs_c_windowed_experiments"

SchedulerFactory = Callable[[RoutingRequest], object]
ExperimentName = Literal["train", "random"]


@dataclass(frozen=True)
class Experiment:
    name: ExperimentName
    description: str
    request: RoutingRequest
    time_limit: float
    max_high_level_nodes: int


@dataclass(frozen=True)
class ExperimentResult:
    experiment: str
    scheduler: str
    ok: bool
    total_time_s: float = float("inf")
    planning_time_s: float = 0.0
    steps: int = 0
    segments: int = 0
    high_level_expanded: int = 0
    high_level_generated: int = 0
    window_stages: int = 0
    grouped_windows: int = 0
    sequence: MovingSequence | None = None
    error: Exception | None = None

    @property
    def total_time_us(self) -> float:
        return self.total_time_s * 1e6


def make_train_inversion() -> Experiment:
    grid = Grid(N=24, d=GRID_SPACING_UM, rc=COLLISION_RADIUS_UM)
    length = 11
    row = grid.N // 2
    first_col = grid.N // 2 - (length // 2) * 2
    src = [(row, col) for col in range(first_col, first_col + 2 * length, 2)]
    dst = list(reversed(src))
    return Experiment(
        name="train",
        description="1D 11-atom train inversion on one row of a 24x24 grid",
        request=RoutingRequest(grid=grid, src=src, dst=dst, labeled=True),
        time_limit=3.0,
        max_high_level_nodes=100_000,
    )


def make_random_routing() -> Experiment:
    grid = Grid(N=14, d=GRID_SPACING_UM, rc=COLLISION_RADIUS_UM)
    rng = random.Random(1000)
    sites = [(i, j) for i in range(grid.N) for j in range(grid.N)]
    src = rng.sample(sites, 25)
    dst = rng.sample(sites, 25)
    return Experiment(
        name="random",
        description="25 labeled atoms from random sites to random sites on a 14x14 grid",
        request=RoutingRequest(grid=grid, src=src, dst=dst, labeled=True),
        time_limit=3.0,
        max_high_level_nodes=50_000,
    )


def plain_scheduler(experiment: Experiment) -> SchedulerFactory:
    return lambda request: RIPACCBSCScheduler(
        request,
        collision_dt=COLLISION_DT,
        ccbs_precision=CCBS_PRECISION,
        high_level_order="conflicts",
        time_limit=experiment.time_limit,
        max_high_level_nodes=experiment.max_high_level_nodes,
        freeze_initial_target_atoms=True,
    )


def windowed_scheduler(experiment: Experiment) -> SchedulerFactory:
    return lambda request: RIPACCBSWindowedScheduler(
        request,
        collision_dt=COLLISION_DT,
        ccbs_precision=CCBS_PRECISION,
        high_level_order="conflicts",
        time_limit=experiment.time_limit,
        max_high_level_nodes=experiment.max_high_level_nodes,
        grouped_time_limit=experiment.time_limit,
        grouped_max_high_level_nodes=experiment.max_high_level_nodes,
        freeze_initial_target_atoms=True,
    )


def run_scheduler(
    experiment: Experiment,
    scheduler_name: str,
    factory: SchedulerFactory,
) -> ExperimentResult:
    started = time.perf_counter()
    try:
        scheduler = factory(experiment.request)
        sequence = scheduler.plan()
        assert_solution_matches_request(sequence, experiment.request)
        solution = getattr(scheduler, "ccbs_solution", None)
        stats = list(getattr(scheduler, "window_stats", []))
        return ExperimentResult(
            experiment=experiment.name,
            scheduler=scheduler_name,
            ok=True,
            total_time_s=sequence.total_duration(),
            planning_time_s=time.perf_counter() - started,
            steps=len(sequence.steps),
            segments=sum(len(atom.segments) for atom in sequence.ensemble.atomtrajs),
            high_level_expanded=(
                0 if solution is None else int(solution.high_level_expanded)
            ),
            high_level_generated=(
                0 if solution is None else int(solution.high_level_generated)
            ),
            window_stages=len(stats),
            grouped_windows=sum(int(stat.grouped) for stat in stats),
            sequence=sequence,
        )
    except Exception as exc:
        return ExperimentResult(
            experiment=experiment.name,
            scheduler=scheduler_name,
            ok=False,
            planning_time_s=time.perf_counter() - started,
            error=exc,
        )


def assert_solution_matches_request(
    sequence: MovingSequence,
    request: RoutingRequest,
) -> None:
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


def run_experiment(experiment: Experiment) -> list[ExperimentResult]:
    return [
        run_scheduler(experiment, "C++ CCBS plain", plain_scheduler(experiment)),
        run_scheduler(experiment, "C++ CCBS windowed", windowed_scheduler(experiment)),
    ]


def assert_expected_behavior(results: list[ExperimentResult]) -> None:
    by_experiment = {(result.experiment, result.scheduler): result for result in results}

    plain_train = by_experiment.get(("train", "C++ CCBS plain"))
    windowed_train = by_experiment.get(("train", "C++ CCBS windowed"))
    if windowed_train is not None:
        if not windowed_train.ok:
            raise AssertionError("windowed train inversion did not solve")

    plain_random = by_experiment.get(("random", "C++ CCBS plain"))
    windowed_random = by_experiment.get(("random", "C++ CCBS windowed"))
    if plain_random is not None and windowed_random is not None:
        if not plain_random.ok or not windowed_random.ok:
            raise AssertionError("random routing should solve for both schedulers")
        if abs(plain_random.total_time_s - windowed_random.total_time_s) > 1e-12:
            raise AssertionError("random routing changed under windowed fallback")
        if windowed_random.window_stages != 0:
            raise AssertionError("random routing unexpectedly used window decomposition")


def format_results(results: list[ExperimentResult]) -> str:
    rows = [
        (
            "experiment",
            "scheduler",
            "ok",
            "total_us",
            "plan_ms",
            "steps",
            "segments",
            "stages",
            "grouped",
            "expanded",
            "generated",
        )
    ]
    for result in results:
        rows.append(
            (
                result.experiment,
                result.scheduler,
                "yes" if result.ok else "no",
                f"{result.total_time_us:.3f}" if result.ok else "-",
                f"{result.planning_time_s * 1e3:.2f}",
                str(result.steps) if result.ok else "-",
                str(result.segments) if result.ok else "-",
                str(result.window_stages) if result.ok else "-",
                str(result.grouped_windows) if result.ok else "-",
                str(result.high_level_expanded) if result.ok else "-",
                str(result.high_level_generated) if result.ok else "-",
            )
        )

    widths = [max(len(row[col]) for row in rows) for col in range(len(rows[0]))]
    return "\n".join(
        "  ".join(value.ljust(widths[col]) for col, value in enumerate(row))
        for row in rows
    )


def format_takeaways(results: list[ExperimentResult]) -> str:
    by_experiment = {(result.experiment, result.scheduler): result for result in results}
    lines = ["takeaways"]

    plain_train = by_experiment.get(("train", "C++ CCBS plain"))
    windowed_train = by_experiment.get(("train", "C++ CCBS windowed"))
    if plain_train is not None and windowed_train is not None and windowed_train.ok:
        if plain_train.ok:
            delta_us = windowed_train.total_time_us - plain_train.total_time_us
            if delta_us < -1e-6:
                speedup = plain_train.total_time_us / windowed_train.total_time_us
                lines.append(f"train: windowed motion is {speedup:.2f}x shorter")
            elif delta_us > 1e-6:
                slowdown = windowed_train.total_time_us / plain_train.total_time_us
                lines.append(f"train: windowed motion is {slowdown:.2f}x longer")
            else:
                lines.append("train: windowed matches plain motion duration")
        else:
            lines.append(
                "train: windowed solved inside the budget; plain one-shot did not"
            )

    plain_random = by_experiment.get(("random", "C++ CCBS plain"))
    windowed_random = by_experiment.get(("random", "C++ CCBS windowed"))
    if plain_random is not None and windowed_random is not None:
        if plain_random.ok and windowed_random.ok:
            delta_us = abs(plain_random.total_time_us - windowed_random.total_time_us)
            lines.append(
                "random: windowed used no decomposition and matched plain "
                f"(delta={delta_us:.6f} us)"
            )
    return "\n".join(lines)


def render_experiment_gifs(
    results: list[ExperimentResult],
    *,
    quality: str,
    out_dir: Path,
) -> None:
    by_experiment: dict[str, dict[str, MovingSequence]] = {}
    for result in results:
        if result.ok and result.sequence is not None:
            by_experiment.setdefault(result.experiment, {})[result.scheduler] = (
                result.sequence
            )

    out_dir.mkdir(parents=True, exist_ok=True)
    for experiment, panels in sorted(by_experiment.items()):
        if not panels:
            continue
        out_path = out_dir / f"{PREFIX}_{experiment}.gif"
        render_animation(
            panels,
            out_path,
            view="benchmark",
            quality=quality,
            time_dilation=1e4,
            hold_seconds=1.5,
            title=f"C++ CCBS windowed experiment: {experiment}",
        )
        print(f"wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check whether windowed C++ CCBS is a general improvement."
    )
    parser.add_argument(
        "--only",
        choices=("all", "train", "random"),
        default="all",
        help="run both workloads or just one",
    )
    parser.add_argument(
        "--no-assert",
        action="store_true",
        help="print results without enforcing the expected behavior checks",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="write speed GIFs to render/",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="write higher-quality GIFs to demo/; implies --render",
    )
    args = parser.parse_args()

    experiments = {
        "train": make_train_inversion(),
        "random": make_random_routing(),
    }
    selected = ("train", "random") if args.only == "all" else (args.only,)

    results: list[ExperimentResult] = []
    for key in selected:
        experiment = experiments[key]
        print(f"{experiment.name}: {experiment.description}")
        results.extend(run_experiment(experiment))

    print(format_results(results))
    print(format_takeaways(results))

    for result in results:
        if not result.ok:
            print(
                f"{result.experiment}/{result.scheduler} error: "
                f"{type(result.error).__name__}: {result.error}"
            )

    if not args.no_assert:
        assert_expected_behavior(results)

    if args.render or args.demo:
        render_experiment_gifs(
            results,
            quality="quality" if args.demo else "speed",
            out_dir=ROOT / ("demo" if args.demo else "render"),
        )


if __name__ == "__main__":
    main()
