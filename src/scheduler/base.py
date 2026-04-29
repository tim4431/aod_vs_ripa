"""Scheduler base classes.

Every scheduler maps a `RoutingRequest` to a `Sequence`. We provide:

* `Scheduler` — the bare ABC.
* `AODScheduler` — marker base for AOD backends (synchronous lattice ops).
* `RIPAScheduler` — marker base for RIPA backends (per-atom async addressing).
* `SyncScheduler` — convenience base for "one synchronous batch at a time"
  schedulers. Both AOD and a sync-RIPA *baseline* fit this pattern.

Sync semantics:
    - Each batch shares one `start_time = seq.next_start_time()`.
    - The next batch starts after the slowest move in the current batch
      finishes — that's automatically the case because `next_start_time`
      reads the ensemble's max end_time.
    - The scheduler reasons about `seq.current_config()` /
      `seq.occupancy_now()` between batches; those return the resting
      snapshot at the moment the next batch will begin.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..routing import RoutingRequest
from ..sequence import Sequence


class Scheduler(ABC):
    @abstractmethod
    def schedule(self, request: RoutingRequest) -> Sequence: ...


class AODScheduler(Scheduler):
    """Marker base for AOD schedulers (synchronous lattice ops)."""


class RIPAScheduler(Scheduler):
    """Marker base for RIPA schedulers (native per-atom async)."""


class SyncScheduler(Scheduler):
    """Plan one synchronous batch at a time.

    Subclasses implement `plan_next_batch(seq) -> list[Step]` (returning
    the empty list signals "done"). The driver loop appends each batch
    via `seq.append_sync_batch(...)`, which advances `next_start_time`
    to the max end_time of the batch.

    This base is identical for AOD and a synchronous-RIPA baseline; the
    difference is the step type the subclass produces.
    """

    @abstractmethod
    def plan_next_batch(self, request: RoutingRequest, seq: Sequence) -> list:
        """Return the next batch of Steps, or [] when finished."""

    def schedule(self, request: RoutingRequest) -> Sequence:
        seq = Sequence(initial=request.initial.copy(),
                       inter_step_gap=getattr(self, "inter_step_gap", 0.0))
        while True:
            batch = self.plan_next_batch(request, seq)
            if not batch:
                return seq
            seq.append_sync_batch(batch)
