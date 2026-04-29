"""A Sequence is an ordered list of MovementSteps.

Each step gets a start time equal to the cumulative duration of its
predecessors plus a small inter-step gap (default 0). This is the
fundamental object schedulers emit and benchmarks read.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .atoms import AtomConfig
from .movement import AODStep, RIPAStep
from .validator import CollisionReport, validate_step


@dataclass
class TimedStep:
    step: AODStep | RIPAStep
    start_time: float
    duration: float

    @property
    def end_time(self) -> float:
        return self.start_time + self.duration


@dataclass
class Sequence:
    initial: AtomConfig
    steps: list[AODStep | RIPAStep] = field(default_factory=list)
    inter_step_gap: float = 0.0   # seconds between steps (e.g. trap settle time)

    def append(self, step: AODStep | RIPAStep) -> None:
        self.steps.append(step)

    def timed(self) -> list[TimedStep]:
        """Resolve concrete (start, duration) for every step.

        AODStep duration depends on the atom config it acts on, so we
        propagate the config through the sequence as we go.
        """
        out: list[TimedStep] = []
        cfg = self.initial
        t = 0.0
        for step in self.steps:
            if isinstance(step, AODStep):
                lifted = step.to_ripa_step(cfg)
                dur = lifted.duration
            else:
                dur = step.duration
            out.append(TimedStep(step=step, start_time=t, duration=dur))
            t += dur + self.inter_step_gap
            cfg = step.apply(cfg)
        return out

    def total_duration(self) -> float:
        ts = self.timed()
        return ts[-1].end_time if ts else 0.0

    def final_config(self) -> AtomConfig:
        cfg = self.initial
        for step in self.steps:
            cfg = step.apply(cfg)
        return cfg

    def validate(self, dt: float = 1e-6) -> list[CollisionReport]:
        """Validate every step in order, threading the config through."""
        reports: list[CollisionReport] = []
        cfg = self.initial
        for step in self.steps:
            reports.append(validate_step(cfg, step, dt=dt))
            cfg = step.apply(cfg)
        return reports
