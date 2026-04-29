# CLAUDE.md

Guidance for Claude Code when working in this repo.

## Environment

**Always use the local conda env `vipa`** for running, testing, and installing Python in this repo. Activate before any Python invocation:

```bash
conda activate vipa
```

Do not use the base env, system Python, or create new envs without being asked.

## Project layout

- `src/` — implementation (AOD schedulers, RIPA schedulers, benchmarking).
- `lib/` — shared utilities / primitives.
- `doc/` — design notes and visualizations.
- `README.md` — project scope and routing-problem definitions.

## Project context

See `README.md`. Two routing problems to keep distinct:
- **Set → set** (identical atoms; defect-free array init).
- **Pairwise** (distinguishable atoms; qubit rearrangement).
