# Architecture

## Core dataclasses

| Class                            | Holds                                                       | File |
|----------------------------------|-------------------------------------------------------------|------|
| `Grid(N, d, rc)`                 | grid size, site spacing, collision radius                   | [src/atoms.py](../src/atoms.py) |
| `AtomConfig(grid, positions)`    | resting-state snapshot, **integer (i, j) only**             | [src/atoms.py](../src/atoms.py) |
| `Trajectory(start_time, duration, start_pos, end_pos, fn, channel)` | one timed segment of motion; integer endpoints, float during flight | [src/trajectories.py](../src/trajectories.py) |
| `AtomTrajectory(atom_id, initial_pos, segments)` | one atom's full life (`atom_id` is internal, *not* a distinguishability marker) | [src/atom_trajectory.py](../src/atom_trajectory.py) |
| `AtomEnsemble(grid, atoms)`      | collection of `AtomTrajectory` — the live state             | [src/atom_trajectory.py](../src/atom_trajectory.py) |
| `Step` (ABC)                     | `apply(ensemble)` appends segments                          | [src/movement.py](../src/movement.py) |
| `AODStep`                        | sync lattice op: `selected_rows × selected_cols → new_*`    | [src/movement.py](../src/movement.py) |
| `RIPAStep`                       | single-atom move along one EOM channel ('row' or 'col')     | [src/movement.py](../src/movement.py) |
| `Sequence(initial, steps)`       | ordered list of Steps; builds the ensemble incrementally    | [src/sequence.py](../src/sequence.py) |
| `RoutingRequest`                 | `targets` (Case 1) **xor** `pairing: Site→Site` (Case 2)    | [src/routing.py](../src/routing.py) |

Snapshot ↔ live state: `AtomConfig` is a resting snapshot at one instant; `AtomEnsemble` is the timeline. Convert via `AtomEnsemble.from_config(cfg)` and `ensemble.final_config()`.

## Async vs sync

RIPA is natively asynchronous (independent EOM tones); AOD is synchronous (one lattice op = one batch). Both backends share one timing model:

- `Trajectory.start_time` is **absolute global time**.
- `seq.next_start_time()` returns `max(end_time)` across the ensemble — the earliest moment when no atom is mid-flight.

That single rule produces both behaviors:

- **AOD (sync, native).** One `AODStep` per lattice op, started at `seq.next_start_time()`. All affected atoms share one start_time and one duration (longest move sets the duration).
- **RIPA (async, native).** A scheduler emits `RIPAStep`s with arbitrary start_times, possibly overlapping, on different atoms.
- **RIPA (sync baseline).** Emit several `RIPAStep`s with the **same** `start_time = seq.next_start_time()`. The next `next_start_time()` jumps to the slowest atom's end → automatic "wait for the batch" semantics. No new step type required.

## Sync interface (shared by AOD + sync-RIPA schedulers)

`Sequence` exposes the resting snapshot at the next batch boundary:

- `current_config() -> AtomConfig`
- `occupancy_now() -> dict[Site, int]`   (site → atom_id)
- `append_sync_batch(steps)`             (atomic: all share one start_time)

`SyncScheduler` ([src/scheduler/base.py](../src/scheduler/base.py)) drives the loop: subclasses implement `plan_next_batch(request, seq) -> list[Step]` and return `[]` when done. AOD and sync-RIPA differ only in which `Step` type they emit.

## Validation

`validate_ensemble(ensemble, dt)` ([src/validator.py](../src/validator.py)) walks the timeline, sampling every atom's position via `AtomTrajectory.position_at(t)` only inside `moving_intervals()` plus segment boundaries, then runs an O(M²) pairwise check against `grid.rc`.
