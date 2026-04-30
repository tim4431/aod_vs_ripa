# Continuous-CBS vs. the RIPA scheduling problem

This note compares the local `lib/Continuous-CBS` implementation with the
RIPA-SLM routing problem described in `doc/aod_vs_ripa.md`,
`doc/pebble_problem.md`, and `example/instruct.md`.

Short answer: **Continuous-time CBS is highly relevant, but it is not a direct
solution to our problem.** It is the closest match on the timing/collision
side, because it plans many finite-radius agents in continuous time with
non-unit move durations. It is a weaker match on the atom-compiler side,
because our RIPA problem has unlabeled target-set assembly, axis-aligned
row/column handoffs, bang-bang trap motion, geometry-discovered corridors, and
often a makespan objective.

## What Continuous-CBS solves

Continuous-CBS, or CCBS, is a continuous-time version of multi-agent path
finding (MAPF). Its input is essentially:

- a graph, either a grid or a roadmap;
- one labeled start and one labeled goal per agent;
- a geometric coordinate for each vertex;
- finite-radius agents;
- a finite set of graph actions, where each action is a straight-line move or a
  wait action with a duration;
- a cost objective, implemented here as sum of individual path durations
  (`flowtime`).

The local implementation is the public
[`PathPlanning/Continuous-CBS`](https://github.com/PathPlanning/Continuous-CBS)
codebase. Important files:

- `lib/Continuous-CBS/cbs.cpp`: high-level conflict tree search. It initializes
  an unconstrained path for each agent, detects the first pairwise continuous
  collision, branches by adding a time interval constraint to one agent or the
  other, and replans affected agents.
- `lib/Continuous-CBS/sipp.cpp`: low-level single-agent planner. It is an
  A*/SIPP-style search over `(vertex, safe interval)` states. Negative
  constraints remove unsafe start-time intervals for moves or unsafe occupancy
  intervals for waits. Positive constraints act as ordered landmarks for
  disjoint splitting.
- `lib/Continuous-CBS/map.cpp`: graph construction. Grid maps can use 4, 8, 16,
  or 32-neighbor connectivity; roadmap maps use graph nodes with floating-point
  coordinates.
- `lib/Continuous-CBS/structs.h`: core records for `Agent`, `Path`, `Move`,
  `Constraint`, `Conflict`, and constraint-tree nodes.

The original CCBS paper frames the key change from ordinary CBS like this:
conflicts are no longer vertex-at-timestep or edge-at-timestep events. They are
collisions between two timed actions, and resolving a collision means forbidding
one of those actions from starting during an unsafe time interval. The low-level
planner then finds the shortest path satisfying those constraints by using safe
intervals.

The 2021 improvement paper adds common CBS accelerations to CCBS:

- prioritizing cardinal/semi-cardinal conflicts;
- disjoint splitting with positive constraints;
- high-level admissible heuristics.

Those improvements are present in the local repo through options such as
`use_cardinal`, `use_disjoint_splitting`, and `hlh_type`.

One caution from the newer literature: a 2025 revisit paper argues that general
CCBS is incomplete when arbitrary real-valued wait durations are part of the
model. That does not make the ideas useless, but it does mean we should treat
CCBS as an algorithmic template and practical heuristic/oracle, not as a
guaranteed optimal solver for unconstrained continuous-wait RIPA scheduling.

## Why it is similar to RIPA

The overlap is real and important.

1. **Continuous-time collisions.** RIPA atoms collide by physical distance along
   continuous trajectories, not by occupying the same discrete site in the same
   timestep. CCBS is designed for this exact conceptual gap.

2. **Non-unit move durations.** RIPA legs have physical durations derived from
   travel length and acceleration limits. CCBS already allows actions whose
   durations are not all one global timestep.

3. **Agents have size.** CCBS treats agents as geometric bodies, implemented in
   this repo as same-radius disks. Our atoms have a collision radius `rc`.

4. **Waiting matters.** RIPA schedules are asynchronous: one atom can wait at a
   resting site while another passes. SIPP's safe intervals are a natural way to
   ask "when may this atom occupy this site or start this leg?"

5. **Conflict repair is local.** A CCBS branch only replans one of the two
   conflicting agents under one extra constraint. This is attractive for RIPA
   because most attempted routes are probably fine; the hard part is resolving
   occasional blocked corridors, crossing times, and target-region congestion.

6. **Roadmaps can encode handoff sites.** CCBS can run on a general graph, so a
   RIPA route graph could use grid intersections as nodes and row/column RIPA
   legs as edges.

In other words, CCBS is much closer to RIPA than classic pebble motion on
graphs. Classic pebble motion only knows "move one pebble to an adjacent empty
vertex." RIPA knows long continuous legs, nonuniform durations, finite atom
radius, and asynchronous overlap. CCBS lives in that same world.

## Why it is not the same problem

The mismatches are the part that matters for implementation.

### 1. CCBS is labeled; defect-free assembly is often unlabeled

The local CCBS task file assigns each agent a specific goal. Our
`RoutingRequest` supports both pairwise/labeled routing and set-to-set
unlabeled assembly. For defect-free assembly, source atoms are interchangeable.

To use CCBS for unlabeled RIPA assembly, we would need an outer assignment
stage:

1. estimate source-target route costs on a RIPA route graph;
2. solve assignment, e.g. Hungarian/min-cost flow/bitmask DP;
3. pass the resulting labeled pairs into CCBS;
4. optionally reassign if the CCBS schedule becomes congested.

This is exactly why the current `RIPAPebbleScheduler` and
`RIPAPebbleAdvScheduler` already estimate pair costs before assigning targets.

### 2. CCBS actions are static graph edges; RIPA corridors are geometry-driven

In `Continuous-CBS`, `Map::get_valid_moves()` returns a fixed neighbor list for
each node. RIPA's useful macro-edges depend on the current atom geometry:

- an atom can move along a row/column until another resting atom blocks the
  corridor;
- a long clear leg may appear only after moving a blocker;
- a dense target region may require sliding vacancies outward.

We can still build a finite RIPA route graph, but it is a modeling choice:

- **Conservative graph:** only adjacent grid edges. This is safe but loses the
  RIPA advantage of long one-shot legs.
- **All row/column macro-edges:** every pair of sites sharing a row/column.
  This captures long legs, but many edges are invalid in the current occupancy
  and must be filtered by collision/visibility constraints.
- **Dynamic visible graph:** recompute line-of-sight neighbors from current
  occupancy, like `RIPAPebbleAdvScheduler._visible_neighbors()`. This matches
  RIPA better, but it is no longer the static graph assumed by the C++ CCBS
  implementation.

### 3. CCBS uses straight-line constant-speed moves; RIPA uses bang-bang motion

The C++ code costs a move by Euclidean distance and collision-checks two
constant-velocity line segments. RIPA moves use constant acceleration then
constant deceleration:

```text
duration(L) = 2 * sqrt(L / a_max)
```

The position profile `x(t)` is nonlinear. Two RIPA legs that share the same
geometric line segment may be safe or unsafe depending on acceleration phase,
not just start/end velocity. Our existing `AtomEnsemble` validator already
handles these profiles, including moving-vs-hold checks and row/column path
crossing times. A RIPA CCBS would need to replace CCBS's `check_conflict()` and
unsafe interval generation with RIPA segment math.

### 4. CCBS minimizes sum of costs, while our benchmark cares about arrangement time

The CCBS high-level priority is `root.cost += path.cost`, i.e. total flowtime.
The solution logger also reports makespan, but the search is not makespan-first.

For atom rearrangement, the natural metric in this repo is total arrangement
duration, i.e. makespan. A flowtime-optimal plan may serialize or delay atoms in
ways that are unattractive for hardware throughput. A RIPA adaptation should
rank high-level nodes by makespan or by a hardware-aware objective:

```text
primary:   final sequence duration
secondary: total moved distance / handoffs / heating-risk penalties
```

### 5. RIPA has channel and handoff semantics

In RIPA:

- a row-channel leg changes `i` while keeping `j` fixed;
- a column-channel leg changes `j` while keeping `i` fixed;
- direction changes require a stationary handoff at a grid intersection;
- diagonal moves are not primitive actions;
- hardware may impose practical limits on simultaneous tones, channel power, or
  frequency separation.

CCBS does not know these semantics. It can represent them if we only put legal
row/column edges into the graph, but the cost model and constraint model still
need to preserve handoff penalties and channel constraints.

### 6. Resting atoms are both agents and obstacles

CCBS treats every agent as moving or waiting, and conflicts with a waiting agent
are just another collision. That is conceptually fine. The catch is planner
behavior: a RIPA scheduler often wants to move blockers out of the way to open a
long corridor. CCBS can discover this only if the blocker has a goal that makes
such motion useful, or if the high-level search constraints force it to replan
through a detour. It does not naturally create "temporary buffer goals" for
unlabeled compaction.

This is why current geometry-driven code has explicit target-set staging:
compact inward, slide vacancies outward, and move blockers to buffers when a
gate is blocked. CCBS does not provide that domain strategy by itself.

### 7. Continuous waits are a theoretical and practical trap

The 2025 revisit paper is relevant to us because RIPA schedules are naturally
asynchronous and allow real-valued start times. If we pursue exact CCBS, we must
decide what wait model we actually want:

- arbitrary real-valued waits: closest to physics, but the CCBS completeness
  story is suspect;
- finite candidate waits from safe intervals/conflict endpoints: practical and
  probably enough for scheduling;
- discretized waits at `collision_dt` or at one-leg-duration ticks: finite and
  implementable, but sacrifices exact optimality and can miss narrow solutions.

For this repo, a bounded finite candidate-wait model is probably the right
engineering choice.

## Can we use `lib/Continuous-CBS` directly?

Not directly.

The local C++ implementation is useful to read and possibly useful as a
small-instance benchmark after modification, but it does not plug into the
Python scheduler stack:

- it uses XML inputs and emits XML logs;
- it is labeled-only;
- roadmap edges do not carry arbitrary RIPA duration functions;
- move cost is geometric distance, not bang-bang duration;
- collision detection assumes constant-velocity disk motion;
- the high-level objective is sum of costs;
- it is not written as a reusable Python library.

Even if we generated XML from a `RoutingRequest`, the result would answer a
different problem unless we patched the C++ core.

## What to borrow

The best thing to borrow is not the code wholesale, but the architecture.

### Low-level: RIPA-SIPP

Build a single-atom planner over RIPA macro-edges:

- state: `(site, safe_interval_id)`;
- action: one legal row/column RIPA leg to another site;
- action duration: `bang_bang_duration(length, a_max)`;
- legal successor time: earliest departure that keeps the source wait, moving
  leg, and destination arrival inside safe intervals;
- constraints: "atom cannot start leg A->B during `[t1, t2)`" and "atom cannot
  occupy/wait at site S during `[t1, t2)`";
- heuristic: bang-bang lower bound from Manhattan distance, plus handoff lower
  bound when both coordinates differ.

This is the cleanest way to upgrade the current "retry start time after
collision" behavior. Instead of repeatedly appending and waiting after failure,
the low-level planner would know forbidden time intervals before proposing the
next leg.

### High-level: bounded RIPA-CBS

Use high-level CBS only when prioritized/SIPP scheduling gets stuck or when a
small congested window matters:

1. Start from assigned RIPA routes.
2. Schedule each atom with RIPA-SIPP against accumulated constraints.
3. Validate the full sequence with `AtomEnsemble`.
4. If two atoms collide, compute an unsafe start interval for one conflicting
   leg against the other conflicting leg or wait.
5. Branch: add that constraint to atom A in one child and to atom B in the
   other.
6. Replan only the constrained atom.
7. Bound the search by time window, number of high-level nodes, or conflicted
   subset size.

This would be CCBS-shaped, but with RIPA-native action generation and our
existing validator as the source of truth.

### Assignment wrapper for unlabeled assembly

For `labeled=False`, keep assignment outside the CBS core:

- first assignment by estimated RIPA-SIPP cost;
- high-level CBS solves the resulting labeled instance;
- if CBS becomes expensive, retry with alternate assignments for conflicted
  atoms or target-region boundary atoms.

This preserves the key freedom of defect-free assembly without forcing the
CBS tree to reason about target identity swaps at every node.

## Suggested path for `RIPAPebbleAdv1Scheduler`

If the goal is a geometry-driven scheduler "from the original pebble problem"
but with RIPA timing, I would not port `Continuous-CBS` line by line. I would
build a staged hybrid:

1. **Keep geometry-discovered visible neighbors.** The current
   `RIPAPebbleAdvScheduler._visible_neighbors()` is the right RIPA primitive:
   long clear row/column moves appear naturally from occupancy.

2. **Replace append-and-retry with safe intervals.** Maintain reservations for
   resting sites and moving RIPA legs. When planning one atom, compute safe
   intervals and choose the earliest feasible start time directly.

3. **Add a small CBS repair loop.** When the validator finds a collision that
   prioritized planning did not avoid, branch on the two involved atoms and
   replan one atom with an extra forbidden leg-start or forbidden-wait
   interval.

4. **Keep target-set staging for dense unlabeled assembly.** CCBS does not
   solve the "which vacancy should move outward?" problem. The target-depth and
   gate-relief logic in `RIPAPebbleAdvScheduler` is domain knowledge worth
   keeping.

5. **Use makespan-aware ordering.** Branch/node priority should be based on
   total sequence duration, not only sum of individual path costs.

6. **Use the existing validator as final authority.** Any analytical unsafe
   interval math will have edge cases. `AtomEnsemble` already checks nonlinear
   RIPA trajectories, holds, and crossing times; keep it in the loop.

## Bottom line

Continuous-CBS is solving a sibling of our problem:

```text
CCBS: labeled continuous-time MAPF on a finite graph
RIPA: labeled or unlabeled continuous-time atom routing with RIPA-native moves
```

The shared core is **continuous-time, finite-radius, asynchronous multi-agent
path planning**. The non-shared core is **RIPA hardware semantics and
unlabeled geometry-driven assembly**.

So the answer is: **yes, similar enough to guide the next scheduler; no, not
similar enough to use directly.** Treat `lib/Continuous-CBS` as the reference
architecture for a RIPA-SIPP plus bounded-CBS repair layer, while preserving our
own assignment, bang-bang motion, row/column handoff, blocker-clearing, and
target-compaction logic.

## References

- PathPlanning Continuous-CBS repository:
  <https://github.com/PathPlanning/Continuous-CBS>
- Anton Andreychuk, Konstantin Yakovlev, Dor Atzmon, and Roni Stern,
  "Multi-Agent Pathfinding with Continuous Time", IJCAI 2019:
  <https://arxiv.org/abs/1901.05506>
- Anton Andreychuk, Konstantin Yakovlev, Eli Boyarski, and Roni Tzvi Stern,
  "Improving Continuous-time Conflict Based Search", AAAI 2021:
  <https://arxiv.org/abs/2101.09723>
- Andy Li, Zhe Chen, Danial Harabor, and Mor Vered,
  "CBS with Continuous-Time Revisit", arXiv 2025:
  <https://arxiv.org/abs/2501.07744>
