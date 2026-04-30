
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

[Sequence](./sequence.py)
- `Sequence` A list of Steps.

[Routing](./routing.py)
- `RoutingRequest(grid, src, dst, labeled)`: route atoms from src to dst, if labeled, pair-wise routing; it not labeled, configuraion matching.

[Schedulers](./scheduler/)
See readme there [schedulers_readme](./scheduler/readme.md)

## Data Flow

`RoutingRequest.initial` builds an `AtomConfig`. A `Sequence` wraps that config in an `AtomEnsemble`. Appending a `Step` creates one or more `Segment`s, and `AtomEnsemble.append_segment(...)` checks per-atom continuity and cross-atom collision before committing.

Use `Sequence.final_config()` for the final resting snapshot, and `AtomEnsemble.positions_at(t)` / `xy_at(t)` for timeline samples.
