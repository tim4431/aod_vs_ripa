# Decomposed CCBS Experiments

`RIPACCBSDWindowedScheduler` is an experimental layer around the C++ CCBS
backend. It targets dense labeled line inversions by grouping small nested swap
windows before calling CCBS.

The useful case is a line like:

```text
0 1 2 3 4 5
```

where the target asks for:

```text
0 <-> 5
1 <-> 4
2 <-> 3
```

The scheduler tries `0 <-> 5` and `2 <-> 3` as one local CCBS problem, so the
short center swap can finish while the outer swap is still taking an outside
route. If that grouped solve exceeds the configured budget, it falls back to
pairwise CCBS for that window.

Unsupported requests fall back to `RIPACCBSCScheduler` by default.

Symmetric windows are translated across identical rows/columns by default. On
the current 6x6 x-inversion benchmark, the windowed path validates at about
`1701.750 us` versus `2221.719 us` for the conservative pairwise C++ demo. With
cache reuse, only one grouped column window and one pairwise middle window need
fresh C++ solves (`~44.9k` high-level expansions total); the other ten windows
are translated and revalidated in the full RIPA timeline.
