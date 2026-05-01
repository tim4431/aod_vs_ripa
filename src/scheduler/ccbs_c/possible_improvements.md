# CCBS Improvement Notes

## Main Finding

CCBS can represent the useful asynchronous behavior where a short inner swap
uses a lane while a longer outer swap is still taking an outside detour. The
current x-inversion demo mostly hides that behavior because it decomposes the
task into isolated two-atom swaps before CCBS sees the local interaction.

For one 6-site column, solving these two swaps together:

```text
0 <-> 5
2 <-> 3
```

produced a valid grouped schedule around `471 us`. The short `2 <-> 3` swap
finishes while the long `0 <-> 5` swap is still routing outside the array.

The tradeoff is planning cost. A simple pairwise solve often expands only a few
CBS nodes, while the grouped one-column solve expanded roughly `45k` nodes in
the current C++ backend. Full-array one-shot CCBS is therefore still too
expensive for the 6x6 inversion benchmark.

## Why the Existing Demo Misses Throughput

1. It solves most swaps as independent two-agent CCBS problems.
2. It batches only obviously independent columns afterward.
3. The C++ backend optimizes flowtime, not directly layer makespan or hardware
   throughput.
4. A useful outside-array route can be longer for one atom but better for total
   completion time if it lets another swap overlap safely.

This is a decomposition issue, not a fundamental CCBS limitation.

## Concrete Improvement Path

1. Detect local line-inversion windows instead of individual swaps.
2. Try small grouped windows such as `(0 <-> 5) + (2 <-> 3)`.
3. Keep grouped windows only when they solve within the configured node/time
   budget and validate in the full RIPA timeline.
4. Fall back to pairwise CCBS for windows that are too hard.
5. Launch separated rows/columns in parallel when validation says the combined
   timeline is safe.
6. Cache or translate symmetric window solutions where possible, but keep final
   RIPA trajectory validation as the source of truth.

The first implementation of this idea lives in
`src/scheduler/ccbs_c/ripa_ccbs_c_windowed.py`: a windowed/decomposed C++ CCBS
scheduler that targets dense line inversions while falling back to the existing
backend for unsupported requests. It now translates solved symmetric windows
across identical rows/columns, so the 6x6 inversion needs one grouped solve plus
one pairwise middle solve instead of six of each.
