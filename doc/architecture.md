# Architecture

## Core Objects

| Object | Role | File |
|---|---|---|
| `Grid(N, d, rc)` | grid size, physical spacing in um, collision radius | [`src/atom_config.py`](../src/atom_config.py) |
| `AtomConfig(positions, atom_ids)` | resting snapshot; positions are integer `(i, j)` sites | [`src/atom_config.py`](../src/atom_config.py) |
| `Segment` | one timed move with absolute `start_time`, `duration`, endpoints, profile, and channel | [`src/segments.py`](../src/segments.py) |
| `AtomTrajectory` | one atom's full timeline; `atom_id` is only an internal handle | [`src/atom_trajectory.py`](../src/atom_trajectory.py) |
| `AtomEnsemble` | live collection of atom trajectories plus collision-checked mutation | [`src/atom_trajectory.py`](../src/atom_trajectory.py) |
| `Step`, `AODStep`, `RIPAStep` | scheduler commands that append segments to an ensemble | [`src/movement.py`](../src/movement.py) |
| `Sequence` | ordered steps, initial config, ensemble state, timing helpers | [`src/sequence.py`](../src/sequence.py) |
| `RoutingRequest` | routing input: `grid`, `src`, `dst`, `labeled` | [`src/routing.py`](../src/routing.py) |
| `Scheduler`, `SyncScheduler` | request-to-sequence planners with clock-cycle helpers | [`src/scheduler/`](../src/scheduler/) |
| `SqrtTimeAODScheduler` | unlabeled AOD planner translated from the sqrt-time reference repo | [`src/scheduler/sqrt_time.py`](../src/scheduler/sqrt_time.py) |
| `NaiveRIPAScheduler` | simple direct single-atom RIPA baseline | [`src/scheduler/naive.py`](../src/scheduler/naive.py) |
| `RIPASpec`, `nu_row`, `nu_col` | RIPA position-to-frequency mapping | [`src/ripa_freq.py`](../src/ripa_freq.py) |

Grid coordinates use `(i, j)` in code. Physical plotting coordinates are produced by `Grid.ij_to_xy(...)`, centered so the array center is `(0 um, 0 um)`.

## Data Flow

`RoutingRequest.initial` builds an `AtomConfig`. A `Sequence` wraps that config in an `AtomEnsemble`. Appending a `Step` creates one or more `Segment`s, and `AtomEnsemble.append_segment(...)` checks per-atom continuity and cross-atom collision before committing.

Use `Sequence.final_config()` for the final resting snapshot, and `AtomEnsemble.positions_at(t)` / `xy_at(t)` for timeline samples.

## Timing

All segment times are absolute. `Sequence.next_start_time()` returns the current total duration plus `inter_step_gap`, or `0` before any motion.

`AODStep` is synchronous: atoms in `selected_rows x selected_cols` move together and share the longest required duration. `RIPAStep` moves one atom along one channel, `"row"` or `"col"`; asynchronous behavior comes from giving different `RIPAStep`s different `start_time`s. For sync-style batches, use `Sequence.append_sync_batch(...)`.

`SqrtTimeAODScheduler` emits one AOD lattice shift per scheduler clock cycle. Its binary planner is a Python translation of the proposed sqrt-time algorithm's alignment, inverse-alignment, Gale-Ryser, and two-step/three-step arbitrary reconfiguration routines. Because an AOD cycle moves a lattice of atoms at once, `AtomEnsemble.append_segments_batch(...)` validates all same-cycle atom segments as one simultaneous mutation.

Schedulers that search or backtrack should evaluate candidates with `Scheduler.evaluate_steps(...)`, `SyncScheduler.evaluate_cycle(...)`, or `Scheduler.evaluate_candidate(...)`. These helpers replay the current `Sequence` into a fresh trial sequence, append the candidate there, and score it with `Scheduler.cost(...)` (default: total arrangement duration). The live sequence is changed only when `commit_trial(...)` is called on a successful trial.

## Validation

Collision validation is built into `AtomEnsemble.append_segment(...)`. It samples candidate motion against every other atom using `ensemble.collision_dt` and compares physical distances to `grid.rc`. `Sequence.validate(dt)` replays the stored steps at a different sampling interval.

## Visualization

[`src/visualization.py`](../src/visualization.py) renders existing `Sequence` or `AtomEnsemble` timelines.

- `plot_atom_plane(ax, timeline, t, ...)`: atom plane, static traps, addressed atoms, trails, planned paths.
- `plot_frequency_tones(axes, timeline, t, ...)`: current row/col tones and tone trajectories.
- `plot_frame(...)` / `save_frame(...)`: combined atom-plane plus frequency panels.
- `render_gif(...)` / `render_animation(...)`: frame rendering plus GIF export, with optional multiprocessing.
