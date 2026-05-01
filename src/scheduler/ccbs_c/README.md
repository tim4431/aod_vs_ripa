# C++ CCBS Backend

This program is adapted from [Continuous-CBS](https://github.com/PathPlanning/Continuous-CBS), refractored by Codex.

## Usage
`ccbs_solver.cpp` is a standalone C++17 implementation of the raw CCBS/SIPP
search used by `src.scheduler.ripa_ccbs_c.RIPACCBSCScheduler`.

The Python wrapper builds the executable on demand with:

```bash
g++ -std=c++17 -O3 -DNDEBUG -Wall -Wextra ccbs_solver.cpp -o ccbs_solver
```

The solver uses a compact stdin/stdout protocol instead of XML or pybind11 so
the repo does not need extra Python build dependencies. The wrapper is
responsible for translating a `RoutingRequest` into a finite RIPA graph and for
validating the returned bang-bang RIPA segments with the existing trajectory
validator.
