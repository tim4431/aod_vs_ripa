# Code Cultivation Interaction Sequence

This note records only the two-qubit interaction sequence for the requested
part of arXiv:2509.05212:

- initializing `Rot(3)` from Fig. A1
- converting `Rot(3)` to `Reg(3)` from Fig. A2

It does not specify how the interactions are physically realized. In
particular, it does not choose hardware, paths, close-gate coordinates, or
parallelization.

## Figures

Fig. A1, unitary encoder for `Rot(3)`:

![Fig. A1 unitary encoder for Rot(3)](../../tmp/2509.05212-pages/fig-a1-panels.png)

Fig. A2, growth from `Rot(3)` to `Reg(3)`:

![Fig. A2 growth from Rot(3) to Reg(3)](../../tmp/2509.05212-pages/fig-a2-panels.png)

Higher-resolution extracted source renders:

![Fig. A1 high-resolution source render](../../tmp/2509.05212-src/render/appa_unitstabcircuit_144-1.png)

![Fig. A2 high-resolution source render](../../tmp/2509.05212-src/render/appa_rot3reg3_144-1.png)

Tick 20 and Tick 21 panels used for the `Rot(3) -> Reg(3)` interaction
read-off:

![Fig. A2 Tick 20 interaction panel](../../tmp/2509.05212-src/render/a2_panels/a2_tick20.png)

![Fig. A2 Tick 21 interaction panel](../../tmp/2509.05212-src/render/a2_panels/a2_tick21.png)

## Atom Labels

Rot(3) data atoms:

| label | logical position |
|---|---:|
| `r00` | `(0, 0)` |
| `r01` | `(0, 1)` |
| `r02` | `(0, 2)` |
| `r10` | `(1, 0)` |
| `r11` | `(1, 1)` |
| `r12` | `(1, 2)` |
| `r20` | `(2, 0)` |
| `r21` | `(2, 1)` |
| `r22` | `(2, 2)` |

Rot(3) stabilizer ancillae that become Reg(3) data atoms:

| label | logical position |
|---|---:|
| `a00` | `(0.5, 0.5)` |
| `a10` | `(1.5, 0.5)` |
| `a01` | `(0.5, 1.5)` |
| `a11` | `(1.5, 1.5)` |

## Initialize Rot(3)

Source: Fig. A1. Single-qubit/reset-only ticks are omitted.

| step | source tick | interacting atoms |
|---:|---:|---|
| 1 | 2 | `r01` with `r02` |
| 2 | 2 | `r11` with `r10` |
| 3 | 2 | `r12` with `r22` |
| 4 | 2 | `r20` with `r21` |
| 5 | 3 | `r01` with `r11` |
| 6 | 3 | `r10` with `r12` |
| 7 | 4 | `r01` with `r12` |
| 8 | 4 | `r00` with `r10` |
| 9 | 4 | `r20` with `r11` |
| 10 | 5 | `r10` with `r20` |

Short notation:

```text
tick 2: (r01,r02), (r11,r10), (r12,r22), (r20,r21)
tick 3: (r01,r11), (r10,r12)
tick 4: (r01,r12), (r00,r10), (r20,r11)
tick 5: (r10,r20)
```

## Rot(3) To Reg(3)

Source: Fig. A2. Single-qubit/reset-only ticks are omitted. The two interaction
layers below are the first two stabilizer-measurement steps of `Rot(3)`; after
these interactions, the `a..` ancillae become data atoms of `Reg(3)`.

| step | source tick | interacting atoms |
|---:|---:|---|
| 1 | 20 | `a01` with `r02` |
| 2 | 20 | `a11` with `r12` |
| 3 | 20 | `a00` with `r01` |
| 4 | 20 | `a10` with `r11` |
| 5 | 21 | `a01` with `r12` |
| 6 | 21 | `a11` with `r11` |
| 7 | 21 | `a10` with `r21` |
| 8 | 21 | `a00` with `r00` |

Short notation:

```text
tick 20: (a01,r02), (a11,r12), (a00,r01), (a10,r11)
tick 21: (a01,r12), (a11,r11), (a10,r21), (a00,r00)
```
