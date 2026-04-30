# Architecture

## Timing

All segment times are absolute. `Sequence.next_start_time()` returns the current total duration plus `inter_step_gap`, or `0` before any motion.


## Validation

Collision validation is built into `AtomEnsemble.append_segment(...)`. It samples candidate motion against every other atom using `ensemble.collision_dt` and compares physical distances to `grid.rc`. For asynchronous appends, `AtomEnsemble.check_segment(...)` also validates the addressed atom's implicit rest before the new segment and, when there is already a longer planned horizon, its implicit rest after the segment. This prevents out-of-order async planning from creating hidden endpoint conflicts. `Sequence.validate(dt)` replays the stored steps at a different sampling interval.

## Visualization

[`src/visualization.py`](../src/visualization.py) renders existing `Sequence` or `AtomEnsemble` timelines.

- `plot_atom_plane(ax, timeline, t, ...)`: atom plane, static traps, addressed atoms, trails, planned paths.
- `plot_frequency_tones(axes, timeline, t, ...)`: current row/col tones and tone trajectories.
- `plot_frame(...)` / `save_frame(...)`: combined atom-plane plus frequency panels.
- `render_gif(...)` / `render_animation(...)`: frame rendering plus GIF export, with optional multiprocessing.
