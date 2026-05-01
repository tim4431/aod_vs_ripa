"""Benchmark helpers for scheduler comparisons."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Iterable, Mapping

from .routing import RoutingRequest
from .moving_sequence import MovingSequence

SchedulerFactory = Callable[[RoutingRequest], object]


@dataclass(frozen=True)
class BenchmarkResult:
    name: str
    ok: bool
    total_time_s: float = float("inf")
    planning_time_s: float = 0.0
    steps: int = 0
    segments: int = 0
    clock_cycles: int | None = None
    sequence: MovingSequence | None = None
    error: Exception | None = None

    @property
    def total_time_us(self) -> float:
        return self.total_time_s * 1e6


def benchmark_schedulers(
    request: RoutingRequest,
    schedulers: Mapping[str, SchedulerFactory] | Iterable[type | SchedulerFactory],
    *,
    validate_dt: float | None = None,
) -> list[BenchmarkResult]:
    """Run schedulers on the same request and compare arrangement duration.

    `schedulers` can be either a mapping of display name -> factory, or an
    iterable of scheduler classes/factories that accept `RoutingRequest`.
    The benchmark objective is the final sequence duration, not wall-clock
    planning time. Planning time is returned separately.
    """
    results: list[BenchmarkResult] = []
    for name, factory in _iter_scheduler_factories(schedulers):
        t0 = time.perf_counter()
        try:
            scheduler = factory(request)
            sequence = scheduler.plan()
            planning_time = time.perf_counter() - t0
            if validate_dt is not None:
                report = sequence.validate(dt=validate_dt)
                if not report.ok:
                    raise ValueError(f"validation failed: {report}")
            results.append(
                BenchmarkResult(
                    name=name,
                    ok=True,
                    total_time_s=sequence.total_duration(),
                    planning_time_s=planning_time,
                    steps=len(sequence.steps),
                    segments=sum(len(atom.segments) for atom in sequence.ensemble.atomtrajs),
                    clock_cycles=getattr(scheduler, "clock_cycle", None),
                    sequence=sequence,
                )
            )
        except Exception as exc:
            results.append(
                BenchmarkResult(
                    name=name,
                    ok=False,
                    planning_time_s=time.perf_counter() - t0,
                    error=exc,
                )
            )
    return sorted(results, key=lambda result: (not result.ok, result.total_time_s))


def format_benchmark_table(results: Iterable[BenchmarkResult]) -> str:
    """Return a compact text table for benchmark results."""
    rows = [
        ("scheduler", "ok", "total_us", "plan_ms", "steps", "cycles", "segments"),
    ]
    for result in results:
        rows.append(
            (
                result.name,
                "yes" if result.ok else "no",
                f"{result.total_time_us:.3f}" if result.ok else "-",
                f"{result.planning_time_s * 1e3:.2f}",
                str(result.steps) if result.ok else "-",
                str(result.clock_cycles) if result.clock_cycles is not None else "-",
                str(result.segments) if result.ok else "-",
            )
        )

    widths = [max(len(row[col]) for row in rows) for col in range(len(rows[0]))]
    lines = []
    for row in rows:
        lines.append(
            "  ".join(value.ljust(widths[col]) for col, value in enumerate(row))
        )
    return "\n".join(lines)


def _iter_scheduler_factories(
    schedulers: Mapping[str, SchedulerFactory] | Iterable[type | SchedulerFactory],
) -> Iterable[tuple[str, SchedulerFactory]]:
    if isinstance(schedulers, Mapping):
        yield from schedulers.items()
        return

    for factory in schedulers:
        name = getattr(factory, "__name__", factory.__class__.__name__)
        yield name, factory
