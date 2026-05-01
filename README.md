# aod_vs_ripa

A direct comparison of two tools for moving neutral atoms in reconfigurable atom arrays:

- **AOD** (Acousto-Optic Deflector) - the traditional tool that moves atoms in a grid geometry configuration.
- **RIPA** (Re-imaging Phased Array) - a fast SLM proposed in [arXiv:2601.08906](https://arxiv.org/abs/2601.08906) that can move atoms in arbitrary patterns.



## Identical

Routing identical atoms, for example, in the case of assembling a defect-free array, is a good example to compare the two tools.

The RIPA scheduler is the heuristic scheduler [RIPAPebbleAdvScheduler](./src/scheduler/ripa_pebble_adv.py).

The AOD scheduler is a [Sqrt-time scheduler](https://arxiv.org/pdf/2604.05317v1) that fully utilizes the parallelism of the AOD.

![defect free assembly](demo/defect_free_assembly_benchmark.gif)


## Pair-wise rearrangment

Routing atoms encoded with quantum information, now the atoms are not treated as identical, and the routing is a pair-wise rearrangement problem.

Here we represent a


## How does a RIPA work