"""Synchronous RIPA baseline (skeleton).

RIPA is natively asynchronous (each EOM tone is independent), but for
benchmarking it's useful to also run RIPA in a *synchronous* mode that
mimics how AOD operates: pick a batch of moves, run them all in
parallel, wait for the slowest, then plan the next batch.

This is the straw-man baseline against which a full async-RIPA
scheduler should win.

Mechanics:
    next_start_time() returns the latest end_time across the ensemble.
    If we emit several RIPASteps with the same start_time, they run
    concurrently; the next call to next_start_time() then sits at the
    max of their end_times — exactly the synchronous semantics.

Usage pattern (concrete subclass):

    class MySyncRIPAScheduler(SyncScheduler, RIPAScheduler):
        a_max: float = 1.0

        def plan_next_batch(self, request, seq):
            occ = seq.occupancy_now()      # site -> atom_id snapshot
            t = seq.next_start_time()
            # ... pick which atoms to move, on which channel, where ...
            return [
                RIPAStep(start_time=t, atom_id=k, target=tgt,
                         channel=ch, a_max=self.a_max)
                for k, tgt, ch in batch
            ]

The SyncScheduler driver loop calls plan_next_batch repeatedly and
appends each batch via append_sync_batch. Subclasses only worry about
the routing logic, never about timing.
"""

from __future__ import annotations

from .base import RIPAScheduler, SyncScheduler


class SyncRIPAScheduler(SyncScheduler, RIPAScheduler):
    """Marker base for synchronous-RIPA schedulers (concrete impls TBD)."""
