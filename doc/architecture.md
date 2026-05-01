# Architecture


## Validation

Collision validation is built into `AtomEnsemble.append_segment(...)`. It samples candidate motion against every other atom using `ensemble.collision_dt` and compares physical distances to `grid.rc`. For asynchronous appends, `AtomEnsemble.check_segment(...)` also validates the addressed atom's implicit rest before the new segment and, when there is already a longer planned horizon, its implicit rest after the segment. This prevents out-of-order async planning from creating hidden endpoint conflicts. `Sequence.validate(dt)` replays the stored steps at a different sampling interval.

## Visualization

[`src/visualization.py`](../src/visualization.py) renders existing `Sequence` or `AtomEnsemble` motion.

- `plot_atom_motion(ax, motion, t, ...)`: reusable atom plane, static traps, addressed atoms, optional trails, planned paths.
- `plot_demo_frame(...)`: one atom-motion plot.
- `plot_benchmark_frame(...)`: one row of atom-motion plots for several schedulers.
- `plot_frequency_tones(axes, timeline, t, ...)`: current row/col tones and tone trajectories.
- `plot_frame(...)` / `save_frame(...)`: dispatches `view="demo"`, `view="benchmark"`, or `view="detail"`.
- `render_animation(...)`: frame rendering plus GIF export, with optional multiprocessing and `quality="speed" | "quality"`.
