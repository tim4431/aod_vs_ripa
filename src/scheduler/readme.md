## Schedulers
By scheduler i mean it can handle a `RoutingRequest`. The scheduler automatically create necessary info (like `AtomEnsemble` to keep the information along movements), and try adding segments to atom trajectory with naive/crazy algorithms.

**Types of Schedulers**
- The scheduler can be either synchronous or asynchronous.
- It can be labeled and un-labeled (labeled meaning it handles (src, dst) pairs, and un-labeled meaning it performs routing so that the final configuration is equal to (dst), atoms are treated as identical).
- It can be an AOD scheduler, which maps (row->new_row, col->new_col), or a RIPA scheduler, which maps (row,col) -> (new_row, new_col) for each individual atom.

## Description of the schedulers
`AODStep` is synchronous: atoms in `selected_rows x selected_cols` move together and share the longest required duration.
`RIPAStep` moves one atom along one channel, `"row"` or `"col"`; asynchronous behavior comes from giving different `RIPAStep`s different `start_time`s. For sync-style batches, use `Sequence.append_sync_batch(...)`.

`SqrtTimeAODScheduler` emits one AOD lattice shift per scheduler clock cycle. Its binary planner is a Python translation of the proposed sqrt-time algorithm's alignment, inverse-alignment, Gale-Ryser, and two-step/three-step arbitrary reconfiguration routines. Because an AOD cycle moves a lattice of atoms at once, `AtomEnsemble.append_segments_batch(...)` validates all same-cycle atom segments as one simultaneous mutation.

`RIPANaiveSyncScheduler` assumes an M-period storage/highway pattern, e.g. `M=2` means even rows/columns hold atoms and odd rows/columns are highways. Each clock cycle it proposes one next RIPA leg per unfinished atom, scores route options by path length, lane load, and occupied blockers, then commits the largest greedy collision-free same-start batch. Individual RIPA legs keep their own durations; the next clock cycle starts after the slowest leg in the batch. For unlabeled requests it first assigns atoms to targets by estimated highway route cost, using exact dynamic programming for small atom counts and a greedy fallback for larger ones.

`RIPAPebbleScheduler` keeps the same storage/highway convention but drops global clock cycles. It builds short macro routes made of long row/column RIPA legs, assigns unlabeled atoms by estimated route cost, and appends each leg at the earliest safe atom-specific start time. On a timed collision it advances the start time and retries; if a lane remains blocked it tries alternate highways or clear auxiliary storage lanes. This is a prioritized/SIPP-style heuristic, not an optimal CBS solver.

Schedulers that search or backtrack should evaluate candidates with `Scheduler.evaluate_steps(...)`, `SyncScheduler.evaluate_cycle(...)`, or `Scheduler.evaluate_candidate(...)`. These helpers replay the current `Sequence` into a fresh trial sequence, append the candidate there, and score it with `Scheduler.cost(...)` (default: total arrangement duration). The live sequence is changed only when `commit_trial(...)` is called on a successful trial.

Use `benchmark_schedulers(request, schedulers, validate_dt=...)` to compare total arrangement time across scheduler factories. It returns per-scheduler results including physical total duration, planning wall time, step count, segment count, and clock cycles when available.