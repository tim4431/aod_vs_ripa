"""Scheduler interface — concrete schedulers are TODO.

A scheduler turns a `RoutingRequest` into a `Sequence` of MovementSteps.
We split the interface into two abstract bases so that AOD vs RIPA
schedulers can share the same calling convention while emitting their
respective step types.

Common AOD schedulers we plan to implement (later, per case 1 / case 2):
  - Hungarian assignment + row-then-col stretching baseline.
  - Lookahead / parallel-row schedulers from the literature.

RIPA schedulers (our novel contribution):
  - Per-channel pipelining over independent tones.
  - Sub-grid "highway" routing using the M-spaced corridor convention.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from .routing import RoutingRequest
from .sequence import Sequence


class Scheduler(ABC):
    """Common interface. Subclasses pick a backend (AOD or RIPA)."""

    @abstractmethod
    def schedule(self, request: RoutingRequest) -> Sequence: ...


class AODScheduler(Scheduler):
    """Marker base for AOD schedulers. Implementations TBD."""


class RIPAScheduler(Scheduler):
    """Marker base for RIPA schedulers. Implementations TBD."""
