# aod_vs_ripa

A direct comparison of two tools for moving neutral atoms in reconfigurable atom arrays:

- **AOD** (Acousto-Optic Deflector) — the traditional tool.
- **RIPA** — proposed in [arXiv:2601.08906](https://arxiv.org/abs/2601.08906).

## Goals

1. Implement common schedulers for AOD (baselines from the literature).
2. Develop our own schedulers for RIPA.
3. Benchmark AOD vs RIPA across the routing problems below.

## Routing problems

The atom-movement task has two distinct flavors, and the routing problem is different for each:

### Case 1 — Identical atoms (set → set routing)
Atoms are interchangeable. The mover only needs to fill a target set of sites from a source set; which source atom ends up at which target site is irrelevant.

- Canonical use case: initializing a **defect-free atom array** from a stochastically loaded array.

### Case 2 — Non-identical atoms (pairwise routing)
Atoms carry distinguishable state (e.g. encoded quantum information), so the mover must respect a specific source→target *pairing*. This is a strictly harder routing problem than Case 1.

- Canonical use case: rearranging qubits during a quantum computation.

## Status

Scaffolding only. Schedulers, simulators, and benchmarks to come.
