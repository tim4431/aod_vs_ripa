## Schedulers
By scheduler i mean it can handle a `RoutingRequest`. The scheduler automatically creates necessary info (like `AtomEnsemble` to keep the information along movements), and tries adding segments to atom trajectories with naive/crazy algorithms.

**Types of schedulers** (see [base.py](base.py))
- **Sync vs async**: `SyncScheduler` advances in clocked cycles where same-cycle steps share a start time; `AsyncScheduler` lets each step pick its own start time independently.
- **Labeled vs unlabeled**: `LabeledScheduler` only accepts requests where atom `k` must reach `dst[k]`; `UnlabeledScheduler` only accepts set→set requests where atoms are interchangeable. The two axes are orthogonal — a concrete scheduler can pick any combination via multiple inheritance.
- **AOD vs RIPA hardware model**: AOD steps map `(row→new_row, col→new_col)` so a whole sub-lattice moves together; RIPA steps move one atom along one channel (`"row"` or `"col"`) at a time.

`AODStep` is synchronous: atoms in `selected_rows × selected_cols` move together and share the longest required duration. `RIPAStep` moves one atom along one channel; asynchronous behavior comes from giving different `RIPAStep`s different `start_time`s. For sync-style RIPA batches, use `Sequence.append_sync_batch(...)`.

## List of schedulers

| Scheduler | Sync/Async | Labeled/Unlabeled | Hardware |
|---|---|---|---|
| [`SqrtTimeAODScheduler`](aod_sqrt_time.py) | Sync | Unlabeled | AOD |
| [`RIPANaiveSyncScheduler`](ripa_naive_sync.py) | Sync | Both | RIPA |
| [`RIPAPebbleScheduler`](ripa_pebble.py) | Async | Both | RIPA |
| [`RIPAPebbleAdvScheduler`](ripa_pebble_adv.py) | Async | Labeled | RIPA |
| [`UnlabeledRIPAPebbleAdvScheduler`](ripa_pebble_adv.py) | Async | Unlabeled | RIPA |
| [`RIPACCBSScheduler`](ripa_ccbs.py) | Async | Both | RIPA |
| [`RIPACCBSCScheduler`](ripa_ccbs_c.py) | Async | Both | RIPA |
| [`RIPACCBSDWindowedScheduler`](ccbs_d/windowed.py) | Async | Both | RIPA |

## Description of the schedulers

`SqrtTimeAODScheduler` emits one AOD lattice shift per scheduler clock cycle, using a Python translation of the proposed sqrt-time algorithm's alignment, inverse-alignment, Gale-Ryser, and two-step/three-step arbitrary reconfiguration routines. Because an AOD cycle moves a whole lattice of atoms at once, `AtomEnsemble.append_segments_batch(...)` validates all same-cycle atom segments as one simultaneous mutation.

`RIPANaiveSyncScheduler` assumes an M-period storage/highway pattern (e.g. `M=2`: even rows/columns hold atoms, odd ones are highways), and each clock cycle proposes one next RIPA leg per unfinished atom, scoring routes by path length, lane load, and occupied blockers before committing the largest greedy collision-free same-start batch. For unlabeled requests it first assigns atoms to targets by estimated highway route cost (exact DP for small atom counts, greedy fallback for larger).

`RIPAPebbleScheduler` keeps the same storage/highway convention but drops global clock cycles, building short macro routes of long row/column RIPA legs and appending each leg at the earliest safe atom-specific start time. On a timed collision it advances the start time and retries; if a lane remains blocked it tries alternate highways or clear auxiliary storage lanes — a prioritized/SIPP-style heuristic, not an optimal CBS solver.

`RIPAPebbleAdvScheduler` is the no-dedicated-highway labeled variant of the pebble scheduler: an atom may move in one RIPA leg to any site visible along its row or column without crossing an occupied site, so long clear corridors are discovered from the current geometry rather than assumed by convention. For dense target sets it runs a staged clear-and-fill (`_clear_wrong_labeled_targets` then `_fill_labeled_targets` ordered by target depth); otherwise it falls back to per-atom Dijkstra routing in distance order.

`UnlabeledRIPAPebbleAdvScheduler` is the unlabeled (set→set) sibling of `RIPAPebbleAdvScheduler`. Both share the path-finding and movement-commit primitives via the `PebbleWorkspace` mixin, so the unlabeled scheduler routes on its own workspace rather than delegating to an inner labeled instance. Two routing modes: a staged unlabeled compaction (default, `unlabeled_assignment="min_sum"`) that picks moves greedily from the current occupancy without committing to a fixed pairing, and an assignment-based path (`unlabeled_assignment="min_max"` or when staged compaction is disabled) that solves a Hungarian/bottleneck assignment on path-cost estimates and routes per-atom via the shared `_plan_assigned_routes`.

`RIPACCBSScheduler` builds a finite RIPA route graph and solves a labeled continuous-time MAPF instance via Continuous-time Conflict-Based Search, returning provably collision-free trajectories under the discretization. For unlabeled requests it first picks a source→target assignment by estimated RIPA move duration, then solves the resulting labeled CCBS instance.

`RIPACCBSCScheduler` is a thin wrapper around `RIPACCBSScheduler` that hands the labeled CCBS instance (route graph + start/goal pairs) to a C++ backend instead of the Python solver, trading implementation simplicity for substantially faster high-level search.

`RIPACCBSDWindowedScheduler` is an experimental decomposed wrapper around the C++ backend. For labeled single-axis reciprocal swap permutations, such as row/column inversions, it groups small nested swap windows before calling CCBS so local asynchronous overlap can be discovered; symmetric windows are translated across identical rows/columns, hard windows fall back to pairwise CCBS, and unsupported requests fall back to `RIPACCBSCScheduler`.

## Search and benchmarking helpers

Schedulers that search or backtrack should evaluate candidates with `Scheduler.evaluate_steps(...)`, `SyncScheduler.evaluate_cycle(...)`, or `Scheduler.evaluate_candidate(...)`. These helpers replay the current `Sequence` into a fresh trial sequence, append the candidate there, and score it with `Scheduler.cost(...)` (default: total arrangement duration). The live sequence is changed only when `commit_trial(...)` is called on a successful trial.

Use `benchmark_schedulers(request, schedulers, validate_dt=...)` to compare total arrangement time across scheduler factories. It returns per-scheduler results including physical total duration, planning wall time, step count, segment count, and clock cycles when available.
