# RIPA-SLM vs AOD: Moving Atoms

## General rules of moving atoms
- We assume we deal with 2D atom array, where atoms sits on a N*N grid (grid size d). We label the atom position as (i,j), i is the horizontal coordinate, j is the vertical coordinate. When atom is moving, ij could be float number. But they have to be integer when they are on the grid, when you store the state.
- Atoms during movement should not collide with each other, the criterion that we set for colliding is when two atom have a distance smaller than a radius (colliding radius rc). Typically rc<=d, meaning that if two atom moves in adjacent row(columns) of the grid, they won't collide. However, if atoms move in the same row(column), they may collide. If they moves towards each other, and their destination cross, they definitely collide; if the moves in the same direction, should consider their actual moving profile x(t). Also, if one atom moves in row, and one atom moves in column, their planned trajectory crosses, should also consider their actual trajectories.
- Atoms can have intricated moving profile along a trajectory x(t). But usually people just do constant accel/deaccel. This has a maximum acceleration number max_acc, so that atom won't be heated too much.
- Your job is to schedule the atom movement based on clever compilation of movements, consider the physical properties of the moving tools (here we consider AOD and RIPA-SLM) that you have. Benchmark the schedulers based on their total arrangement speed.

## Two types of Routing problems

### Case 1 — Identical atoms (set → set routing)
Atoms are interchangeable. The mover only needs to fill a target set of sites from a source set; which source atom ends up at which target site is irrelevant.

- Example: initializing a **defect-free atom array** from a stochastically loaded array.

### Case 2 — Non-identical atoms (pairwise routing)
Atoms carry distinguishable state (e.g. encoded quantum information), so the mover must respect a specific source→target *pairing*. This is a strictly harder routing problem than Case 1.

- Example: rearranging qubits during a quantum computation.



## crossed AOD: row x column grid-like patterns, no crossing.

- Each AOD turns one RF tone `f_RF` into a deflection angle `θ ≈ λ·f_RF/v_s` → spot position `y = f·θ ∝ f_RF` along its axis.
- Crossed AODs (x-AOD × y-AOD): the trap pattern is the **outer product** `{f_x} × {f_y}` — a rectangular grid. You cannot light up a single off-grid site without also lighting up its row/column intersections.
- During atom-moving operations AODs are restricted to **stretches, compressions, and translations of whole rows and columns**. Rows and columns must **never cross**, because two RF tones sweeping past each other on the same AOD heat the atoms.
- Consequence: any operation that requires atoms to swap positions across rows/columns (e.g. mirror-symmetric inversion of the array) cannot be done in one shot — atoms must first be **handed off to static SLM traps**, rearranged serially, then handed back.
- Native way to represents its operation on atom movement: list(row -> new row), and a list(col -> new col). during the changing, the order of these rows/cols should not change (no crossing limit). Based on this description of movement, with an input of atom configuration, you can compute the resulting atom configuration.


## RIPA-SLM: independent per-pixel addressing
- For the original proposal, see [2601.08906v1.pdf](https://arxiv.org/abs/2601.08906)
- A single laser + EOM produces a comb of optical tones `{ν_k}`. Each tone is mapped by the cascaded RIPAs + lens to a unique `(x_k, y_k)` in the focal plane.
- Because every site has its own dedicated frequency tone (frequency-bin encoding), spots are **independent**: amplitude, phase, and trajectory of each tone are programmed separately by the AWG/EOM. No outer-product constraint.
- Continuous frequency sweeps move spots smoothly along either a row or a column. The way to realize this: using a **frequency- or polarization-multiplexed second channel** through the RIPA whose 2D output is **rotated 90°** relative to the primary channel (row channel). The primary channel (row channel) moves atom continuously along its own row, and the second channel (col channel) moves atom continuously along its own column. The EOM RF drive switches each channel on/off independently, and they can be simultaneously on, they can have independent timing for multiple frequency tones (so does multiple tweezer traps).
- To move an atom A → B: pick up atom from A, transport continuously along row/column. **At the intersection of row/column (i.e., at integer grid point (i,j))**, if the atom is stationary, it could be hand-off from one channel to another. After several row(column)-continuous transport, the atom reaches B.
- In order to make ways for atom transport, we want to atoms to stay on a subgrid of the N*N grid. Before and after the routing, atoms only live every M rows/cols. For example if M=2, then atoms only lives in even rows and columns before and after the routing. The odd rows/cols serves like a "highway" to transport atoms
- However, since atoms may not fill a regular grid on this subgrid, (again with M=2 example), the "even" rows can also used for transporting atoms if there isn't any atoms blocking the way. Sometimes it is even possible to move some atoms out of the way to create a new, free, highway.


## Things to implement
- class that store atom configuration
- class to store a movement step. For RIPA this directly translate to atom trajectory, for AOD this first translate to a grid, than to atom trajectory.
- A RoutingRequest that is either **labeled** or **unlabeled**. For the case 2 and case 1 routing problems.
- A scheduler that generate movement steps from the RoutingRequest (dont implement it yet)!
- A validator that validate if the trajectory will cause atom collision.
- Timing of the whole sequence.
