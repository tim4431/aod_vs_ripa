# The pebble problem

This note is for translating the computer-science "pebble motion" literature
into a useful RIPA scheduler design. Local copies of the papers I read are in
`./tmp/pebble_papers/`.

## Original pointer from Adam

Relevant search terms are "pebble moving problem" and, for our interchangeable
atom case, "uncolored" or "unlabeled pebble problem". The same family of
problems appears in automated warehouse routing, but most CS algorithms are not
directly used for optical tweezer rearrangement because AODs add constraints
that those papers do not model.

Adam's reference list:

1. <https://arxiv.org/pdf/1507.03290.pdf>
2. <https://arxiv.org/pdf/1205.5263.pdf>
3. <https://link.springer.com/content/pdf/10.1007/s00373-005-0640-1.pdf>
4. <https://arxiv.org/pdf/1504.05218.pdf>
5. <https://arc.cs.rutgers.edu/files/KatYuLav13ICRA.pdf>

The ARC URL for `KatYuLav13ICRA.pdf` timed out during download; the same paper is
available from Jingjin Yu's Rutgers page as
<https://people.cs.rutgers.edu/~jy512/files/KatYuLav13ICRA.pdf>.

The following are unavailable / not needed for now:

<!--
6. https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6426233&tag=1
7. https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=5354291
-->

## Problem variants

Classic pebble motion on graphs (PMG):

- Input: graph `G=(V,E)`, `p <= |V|` pebbles on distinct vertices, start
  configuration `S`, destination configuration `D`.
- Move model: one pebble moves along one edge into an empty adjacent vertex.
- Usually labeled: pebble identity matters.
- Goal: decide reachability, and sometimes output a move sequence.

Pebble motion with rotations (PMR):

- Extends PMG by allowing synchronous rotations on fully occupied cycles.
- This is closer to multi-robot routing than the strict 15-puzzle model because
  several agents can move in one time step.

Multi-agent path finding (MAPF):

- Agents move on a graph with discrete or continuous time.
- Typical collision rules forbid two agents at the same vertex at the same time
  and forbid opposite traversal of the same edge. Continuous-time MAPF extends
  this to geometric bodies, non-unit edge durations, and unsafe time intervals.
- Common objectives are makespan, sum of costs, total distance, and arrival time.

Unlabeled / uncolored variants:

- Targets are a set, not an atom-by-atom pairing.
- The solver can choose the source->target assignment. This is essential for
  defect-free atom array assembly.
- A common structure is: estimate source-target path costs, solve an assignment
  problem, then schedule collision-free paths.

## Useful literature facts

- Labeled PMG feasibility can be tested in linear time on general graphs, while
  constructing a full step-by-step plan has classic `O(n^3)` upper bounds and
  can require cubic-length outputs in hard cases. Optimal versions are generally
  NP-hard, so exact optimal scheduling is not the right default for large arrays.
- Yu and Rus's PMR work adds synchronous cycle rotations and gives linear
  feasibility plus cubic planning. The conceptual takeaway is that extra
  simultaneous motion primitives change both feasibility and plan length.
- Yu and LaValle map labeled MPP to time-expanded network flow / ILP and optimize
  makespan, max distance, total arrival time, or total distance. This is useful
  as an exact small-instance oracle or benchmark, not as the first large-array
  RIPA scheduler.
- Katsev, Yu, and LaValle's formation planner is especially relevant to
  unlabeled routing: use shortest paths + assignment for distance-optimal plans,
  then scale by partitioning the graph, balancing subproblems with min-cost
  flow, and scheduling local paths. This suggests using coarse lanes/highways
  before per-atom timing.
- Solovey, Yu, Zamir, and Halperin solve unlabeled disc routing under separation
  assumptions by repeatedly choosing a standalone goal, using optimal assignment
  paths, and, when a direct path is blocked, switching through a short detour.
  The RIPA highway layout is a physical way to manufacture such separation.
- Continuous-time CBS (CCBS) combines Conflict-Based Search with Safe Interval
  Path Planning. This is the closest algorithmic match for RIPA timing because
  RIPA legs have non-unit durations and collisions depend on continuous
  trajectories, not just grid occupancy.
- Warehouse "Push, Stop, and Replan" is a practical lesson: start with shortest
  individual paths, execute in parallel, and resolve conflicts by stopping one
  robot, pushing a blocker aside, or replanning around finished/blocked robots.
- "Pushing Squares Around" is more about connected modular metamorphic robots.
  It is not a direct fit for independent atoms, but it reinforces the idea that
  geometry and connectivity constraints can dominate the graph abstraction.

## Pebble problem vs. RIPA scheduler

RIPA is not just PMG on the atom grid.

1. A RIPA leg is a macro-edge: an atom can move many grid sites along one row or
   column in one acceleration/deceleration segment. Cost scales like the physical
   bang-bang duration, roughly `2*sqrt(length/a_max)`, not like the number of
   adjacent grid edges.
2. Extra stops are expensive. A route with fewer long legs and fewer handoffs can
   beat a shortest Manhattan path with many local detours.
3. Collisions are continuous-time geometric events. Two atoms can collide while
   crossing paths even if no grid vertex conflict appears; adjacent rows/columns
   are safe only because `rc <= d`.
4. RIPA has many independently addressable tones, so the scheduler should
   minimize physical makespan with asynchronous parallel motion, not just number
   of discrete pebble moves.
5. Atoms live on a storage subgrid, while non-storage rows/columns act as
   highways. The scheduler can choose lanes, create temporary vacancies, and
   move blockers away to open a long transport path.
6. RIPA row/column channels require axis-aligned legs and stationary handoff
   points at grid intersections. This is different from generic MAPF edges.
7. Unlabeled atom assembly should choose assignments jointly with routes. A
   nearest-target assignment can be poor if it overloads one highway or blocks a
   later long move.
8. AOD constraints are different again: AOD rows/columns move as an outer-product
   lattice and cannot cross. Pebble/MAPF methods are much more applicable to RIPA
   than to crossed-AOD scheduling.

## Direction for a good RIPA scheduler

A good next scheduler should be an asynchronous, highway-aware MAPF planner, with
the existing trajectory validator as the source of truth.

1. Build a macro-route graph.
   - Nodes are storage sites plus legal handoff/intersection sites.
   - Edges are single RIPA row/column legs with physical duration from the current
     acceleration model.
   - Candidate routes should include direct legs, row-highway detours,
     column-highway detours, and a small number of alternate lanes.

2. Solve assignment for unlabeled requests.
   - Estimate each source-target cost using the macro-route graph.
   - Include penalties for lane congestion, stationary blockers, handoffs, and
     expected waiting.
   - Use exact DP/Hungarian matching for modest atom counts, with a greedy or
     min-cost-flow fallback for larger arrays.

3. Plan individual routes first, then schedule in time.
   - Start from shortest / lowest-cost macro routes.
   - Maintain a reservation table for occupied sites, moving line segments,
     highway lanes, and crossing points over time.
   - Use Safe Interval Path Planning ideas to choose the earliest safe start time
     for each leg instead of forcing all atoms into global clock cycles.

4. Resolve conflicts with a bounded high-level search.
   - For small or congested windows, use a CCBS-like branch: if two atoms collide,
     add a time/segment constraint to one of them and replan that atom.
   - For larger cases, use prioritized planning with retries: reorder atoms,
     reassign targets, or add waits.
   - Keep the search window local in time and space so the scheduler remains fast.

5. Add blocker-clearing moves.
   - If a long leg is blocked by resting atoms, compare three choices: alternate
     lane, wait, or move blockers to nearby spare storage/highway sites.
   - This is the atom analogue of push/stop/replan. The cost function should
     charge blocker moves, extra handoffs, and extra acceleration cycles.

6. Optimize after a feasible plan exists.
   - Try shifting each leg earlier while preserving safety.
   - Try lane swaps and assignment swaps that reduce makespan.
   - Objective should be physical total duration first, then secondary penalties
     for number of moved atoms, total distance, handoffs, and heating risk.

## Implementation notes for this repo

- `RIPANaiveSyncScheduler` is a useful baseline, but it waits for the slowest leg
  in each cycle. The next scheduler should allow atom-specific start times.
- Waiting does not need a new movement primitive if atoms simply remain in their
  last resting site until their next segment. The planner does need to reserve
  that resting site during the wait.
- Same-start RIPA batches may need true batch validation. Sequentially appending
  same-time moves can falsely reject a move into a site that another atom vacates
  simultaneously. `AtomEnsemble.append_segments_batch(...)` already has the right
  shape; RIPA steps may need a way to expose candidate segments before commit.
- Use `Scheduler.evaluate_candidate(...)` and `commit_trial(...)` for local
  replanning experiments so failed conflict resolutions do not mutate the live
  sequence.
- The first serious benchmark should compare:
  - current `RIPANaiveSyncScheduler`,
  - an asynchronous prioritized/SIPP scheduler,
  - the same scheduler with reassignment enabled,
  - the same scheduler with blocker-clearing enabled.

