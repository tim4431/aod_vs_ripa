# aod_vs_ripa

A direct comparison of two tools for moving neutral atoms in reconfigurable atom arrays:

- **AOD** (Acousto-Optic Deflector) - the traditional tool that moves atoms in a grid geometry configuration.
- **RIPA** (Re-imaging Phased Array) - a fast SLM proposed in [arXiv:2601.08906](https://arxiv.org/abs/2601.08906) that can move atoms in arbitrary patterns.

**What is so different about a RIPA ?**
- It can move atoms independently in an arbitrary pattern (continuously along the cartesian grid).
- It can be scheduled asynchronously.

For detailed comparison, see [aod_vs_ripa.md](doc/aod_vs_ripa.md)


### Identical atom routing

**Defect-free array assembly**

Routing identical atoms, for example, in the case of assembling a defect-free array, is a good example to compare the two tools.

The RIPA scheduler used here is a simple heuristic scheduler [`RIPAPebbleAdvScheduler`](./src/scheduler/ripa_pebble_adv.py). We choose two AOD schedulers to compare with, one is the [Tetris](https://journals.aps.org/prapplied/abstract/10.1103/PhysRevApplied.19.054032) suitable for stochastic reservoir loading with postselection; and the other one is a [Sqrt-time scheduler](https://arxiv.org/pdf/2604.05317v1) that fully utilizes the parallelism of the AOD to solve a binary-grid reconfiguration problem.

![defect free assembly](demo/defect_free_assembly_benchmark.gif)

The RIPA completes the assembly within `497 us`, while both aod compiles to `~2ms` operation times.


## Pair-wise rearrangment

**Inversion of an atom array**

Routing atoms encoded with quantum information, now the atoms are not treated as identical, and the routing is a pair-wise rearrangement problem.

Here we represent an inversion operation of a `1xN (N=6)` atom array. We can see the RIPA arranger can perform the inversion in a 1-step fashion, while the AOD arranger needs to perform this in multiple (N) steps. Moreover, RIPA does not require a hand-off between AOD and SLM traps which could potentially introduce atom loss.

![inversion 1x6 benchmark](demo/inversion_1x6_ripa_aod.gif)

With a 6x6 atom array, where AOD gains more parallelism, now the total time cost to rearrange is limited by the maximum throughput. Assume we have the same throughput (1 channel per adjacent row/column within atom array), the RIPA scheduler still takes less time due to its flexible routing capability.

![inversion 6x6 benchmark](demo/inversion_6x6_ripa_aod.gif)


## How does a RIPA transportation work

- Continuous frequency sweeps move spots smoothly along either a row or a column.
- Using frequency- or polarization- multiplexing to replicate the primary channel (row channel, continuously moving along x) and rotate it by 90 deg, so that the second channel (col channel) continuously moves along y.
- The EOM RF drive switches each channel on/off independently, and they can be simultaneously on, they can have independent timing for multiple frequency tones (so does multiple tweezer traps).
- To move an atom A → B: pick up atom from A, transport continuously along row/column. **At the intersection of row/column (i.e., at integer grid point (i,j))**, hand-off from one channel to the other. After several row(column)-continuous transport, the atom reaches B.

![random routing tones demo](demo/random_routing_tones_demo.gif)


## How to design a good RIPA scheduler
RIPA scheduling is a very hard problem, mostly in its vast parameter space and its asynchronous nature. This repo implements some heuristic algorithms, as well as a searching algorithm, see [scheduler/readme.md](src/scheduler/readme.md).

The RIPA scheduler, in its geometric nature, is similar to the pebble motion problem, see [Pebble Problem](doc/pebble_problem.md). We also adapted some path-planning algorithms like [Continuous-CBS](https://github.com/PathPlanning/Continuous-CBS).

Different schedulers will yield different total arrangement times. See [inversion 6x6 benchmark](demo/inversion_6x6_benchmark.gif)

## Understanding this repository
- `src/`: source code, see [src/readme.md](src/readme.md).
- `src/scheduler/`: contains the scheduling algorithms, see [scheduler/readme.md](src/scheduler/readme.md).
- `example/`: example scripts to demonstrate the scheduling and to do benchmarking.