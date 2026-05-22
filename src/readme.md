
## Architecture
[Atom Config](./atom_config.py)
- `Grid(N, d, rc)` describe a N*N grid with grid size d, collision radius rc.
- `AtomConfig(positions, atom_ids)` stores configuration. positions are integer `(i, j)` sites, id for indexing.

[Atom Trajectory](./atom_trajectory.py)
- `AtomTrajectory` encodes one atom's full timeline
- `AtomEnsemble` collection of atom trajectories. When adding a new segment, it checks for collisions with existing trajectories.

[Segments](./segments.py)
- `Segment` a single movement of atom.

[Movement](./movement.py)
- `Step` an action that generate new trajectories based on current AtomEnsemble.
-  `AODStep` moves atom that the grid address
-  `RIPAStep` moves a single atom that each RIPA tone address

[Moving Sequence](./moving_sequence.py)
- `MovingSequence` A list of Steps, owning the live `AtomEnsemble`.
- `append(step)` validates and commits one step.
- `append_sync_batch(steps)` commits a same-start batch of steps with order-independent validation: a step that trips a collision against a still-pending batch member is deferred and retried after the others commit, and the whole batch rolls back atomically if a real mutual collision is detected.

[Routing](./routing.py)
- `RoutingRequest(grid, src, dst, labeled)`: route atoms from src to dst, if labeled, pair-wise routing; it not labeled, configuraion matching.

[Schedulers](./scheduler/)
See readme there [schedulers_readme](./scheduler/readme.md)

[Visualization](./visualization.py)
- `render_animation(motion, out, view=...)` stitches per-frame plots into a GIF/WebP/APNG; views are `demo`, `benchmark` (multi-panel side-by-side), and `detail` (atom plane + per-channel tone spectra and history).
- Composable primitives — `draw_grid_dots`, `draw_grid_frame`, `draw_atoms`, `draw_traps`, `draw_motion_blur`, `_draw_gaussian_blob` — are exposed for custom panels (see [`example/presentation_demo/_panel_render.py`](../example/presentation_demo/_panel_render.py) for a custom panel that channel-tints trap halos and adds a `QUINTIC_MIN_JERK` intensity envelope on top of these primitives).

## Data Flow

`RoutingRequest.initial` builds an `AtomConfig`. A `MovingSequence` wraps that config in an `AtomEnsemble`. Appending a `Step` creates one or more `Segment`s, and `AtomEnsemble.append_segment(...)` checks per-atom continuity and cross-atom collision before committing; `append_sync_batch(...)` performs the same checks across a group of same-start steps.

Use `MovingSequence.final_config()` for the final resting snapshot, and `AtomEnsemble.positions_at(t)` / `xy_at(t)` for timeline samples.
