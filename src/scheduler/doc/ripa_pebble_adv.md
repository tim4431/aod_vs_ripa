# RIPAPebbleAdvScheduler

`RIPAPebbleAdvScheduler` is a geometry-driven hybrid RIPA scheduler. It does
not assume a sparse storage/highway pattern. Instead, it looks at the current
atom configuration and builds row/column routes through whatever clear
corridors already exist.

## Modes

Unlabeled set assembly is used for defect-free filling. Atoms are
interchangeable, so only the final occupied set matters.

Labeled pairwise routing is used for random-to-random tests. Atom identity
matters, so atom `k` must reach `dst[k]`.

## Route Search

`_visible_neighbors()` scans along each row/column direction until another atom
blocks the line. Every visible empty site is a possible one-leg RIPA move.

The path cost uses physical bang-bang duration plus a small handoff weight. This
favors fewer long clear moves over many short stop-start moves, without
declaring any permanent highway lanes.

## Unlabeled Compaction

The unlabeled target-set planner now behaves like inward compaction, not
evacuation.

It keeps atoms that are already in the target region whenever possible. On each
iteration it looks at empty target sites from deepest to shallowest:

1. If an outside atom can reach that empty target site, move it in.
2. Otherwise, slide a shallower in-target atom inward to that empty site.
3. This moves the vacancy outward, so later outside atoms can fill it.

This means the target occupancy count never decreases, and in-target moves only
increase how deeply atoms are packed. Visually, the square should now look more
like atoms falling and piling toward the center rather than the previous
"clear everything out, then refill" behavior.

## Async Staging

`async_staging=True` means staged moves start at the addressed atom's earliest
available time. Different atoms may overlap in time if the collision validator
accepts the trajectories.

`async_staging=False` serializes staged legs at the global sequence tail.

If async staging fails, the scheduler resets the tentative sequence and retries
the same staged plan in serialized mode.

## Labeled Staging

Labeled mode still uses a more conservative staged plan: atoms already on their
own targets are preserved, atoms on the wrong target sites are moved away or to
their own targets, and then exact destinations are filled.

## Limitations

This is still a heuristic planner, not an optimal multi-agent solver. Dense
labeled permutations or heavily boxed-in unlabeled vacancies can still require a
stronger search method.
