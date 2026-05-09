# Code Cultivation Movement Notes

Paper: arXiv:2509.05212, "Fold-transversal surface code cultivation"

Local paper artifacts:
- `tmp/2509.05212.pdf`
- `tmp/2509.05212.txt`
- `tmp/2509.05212-src/`
- Rendered figure pages: `tmp/2509.05212-pages/`
- Cropped source-figure panels:
  - `tmp/2509.05212-src/render/a1_panels/`
  - `tmp/2509.05212-src/render/a2_panels/`

## Relevant Paper Points

- The protocol starts by preparing a magic state in `Rot(3)` and measuring its stabilizers.
- The cultivation stage transforms `Rot(3)` into `Reg(3)`. The paper states that this is "simply the first two steps of the stabilizer measurement circuit for Rot(3)".
- Figure A1 gives the unitary encoder for `Rot(3)`.
- Figure A2 gives the growth from `Rot(3)` to `Reg(3)`.
- For neutral atoms, the paper later uses a CZ-native physical-noise model and notes that CX gates are decomposed into CZ gates plus local single-qubit rotations. Therefore the movement problem can ignore single-qubit gates and compile every two-qubit edge into a CZ interaction.

## AOD Movement Model Implemented

`example/code_cultivation/code_cultivation.py` models only one axis-aligned crossed-AOD set:

1. Read each two-qubit edge in Figs. A1 and A2 as a CZ interaction edge.
2. Ignore reset, H, phase, measurement, and other single-qubit operations.
3. Serialize the edges within each source tick; do not move multiple CZ pairs together.
4. For every two-qubit edge, move only the first atom in the listed pair to a
   cardinal CZ site one grid unit from the second atom.
5. Route that one atom with axis-aligned AOD legs. The router tries direct
   Manhattan paths first, then orthogonal clearance lanes to avoid collisions.
6. Hold both atoms for `CZ_DWELL = 1 us`.
7. Return the moved atom to its code site by reversing the selected route.

The code uses the existing `MovingSequence`, `AODStep`, and collision validator.
Every emitted transport leg changes only one coordinate of one selected atom,
using the default `(x, y)` AOD axes.

The compiler asserts that every AOD movement step affects exactly one atom, and every CZ dwell step holds exactly two atoms. This prevents the same-tick multi-pair motion that appeared in the earlier GIF.

The hardware-neutral atom labels and interaction order are written in `example/code_cultivation/code_cultivation_sequence.md`.
The same interaction data is available to the AOD implementation as
`example/code_cultivation/code_cultivation_sequence.py`;
`example/code_cultivation/code_cultivation.py` imports it and derives the atom
arrangement from the labels.

## Transcribed Two-Qubit Layers

`Rot(3)` initialization from Fig. A1:

- Tick 2: `r01-r02`, `r11-r10`, `r12-r22`, `r20-r21`
- Tick 3: `r01-r11`, `r10-r12`
- Tick 4: `r01-r12`, `r00-r10`, `r20-r11`
- Tick 5: `r10-r20`

`Rot(3) -> Reg(3)` from Fig. A2:

- Tick 20: `a01-r02`, `a11-r12`, `a00-r01`, `a10-r11`
- Tick 21: `a01-r12`, `a11-r11`, `a10-r21`, `a00-r00`

Ticks 0, 1, 6, 18, and 19 are single-qubit/reset/layout ticks for this transport model and emit no motion.

## Assumptions

- The appendix figures are circuit-layout diagrams, not calibrated atom-shuttling diagrams. The implementation preserves the two-qubit layer structure and supplies a local neutral-atom shuttling envelope.
- The four `a..` sites are modeled as preloaded Rot(3) stabilizer ancillae at their eventual Reg(3) data positions. On a reconfigurable machine, these could instead be introduced immediately before Fig. A2.
- CNOT direction is irrelevant for movement after conversion to CZ, so the code stores undirected gate edges.
- RIPA is not mixed into this pass. A later RIPA comparison should compile the same CZ edge list into its own row/column movement model.
- The AOD pass uses one crossed-AOD set whose axes are aligned with the code coordinate axes.
- This is a faithful AOD movement scaffold for the requested stages, not an optimized neutral-atom scheduler.
