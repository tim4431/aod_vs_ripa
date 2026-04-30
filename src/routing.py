"""RoutingRequest: input to a scheduler.

A request is fully specified by

    (grid, src: list[Site], dst: list[Site], labeled: bool)

where `src` is the list of currently-occupied sites (one per atom) and
`dst` is the list of target sites (also one per atom; `len(src) ==
len(dst)`).

The `labeled` flag selects between the two routing problems:

* `labeled=False` (Case 1, set→set). Atoms are interchangeable; the
  scheduler may choose any bijection from `src` to `dst`. `dst` is
  conceptually a *set* — its order is irrelevant.
* `labeled=True` (Case 2, pairwise). Atoms are distinguishable; the
  scheduler must move the atom currently at `src[k]` to `dst[k]`.

`RoutingRequest` derives the initial `AtomConfig` from `src` (atom_ids
default to `0..M-1`, in `src` order). The grid lives on the request,
not on the AtomConfig — see `atom_config.py` for the rationale.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence
import random
import numpy as np

from .atom_config import AtomConfig, Grid

Site = tuple[int, int]


def stochastically_loaded_sites(N: int, atom_count: int, seed: int) -> list[Site]:
    """Return a list of sites with atoms placed stochastically."""
    if not 0 <= atom_count <= N * N:
        raise ValueError("atom_count must be between 0 and N*N")

    rng = random.Random(seed)
    sites = [(i, j) for i in range(N) for j in range(N)]
    rng.shuffle(sites)
    return sorted(sites[:atom_count])


def centered_square_targets(N: int, side: int) -> list[Site]:
    """Centered side x side target square."""
    if not 0 < side <= N:
        raise ValueError("target side must be between 1 and N")

    start = (N - side) // 2
    return [
        (i, j) for i in range(start, start + side) for j in range(start, start + side)
    ]


@dataclass
class RoutingRequest:
    grid: Grid
    src: Sequence[Site]
    dst: Sequence[Site]
    labeled: bool = False

    def __post_init__(self):
        self.src = [tuple(p) for p in self.src]
        self.dst = [tuple(p) for p in self.dst]
        if len(self.src) != len(self.dst):
            raise ValueError(
                f"src ({len(self.src)} sites) and dst ({len(self.dst)} sites) "
                "must have the same length"
            )
        if len(set(self.src)) != len(self.src):
            raise ValueError("src has duplicate sites")
        if len(set(self.dst)) != len(self.dst):
            raise ValueError("dst has duplicate sites")
        # Bound check (cheap; catches off-by-one mistakes early).
        N = self.grid.N
        for label, lst in (("src", self.src), ("dst", self.dst)):
            for i, j in lst:
                if not (0 <= i < N and 0 <= j < N):
                    raise ValueError(f"{label} site {(i, j)} is outside a {N}x{N} grid")

    @property
    def initial(self) -> AtomConfig:
        """Build the initial AtomConfig from `src` (atom_id k = src order)."""
        return AtomConfig(positions=np.asarray(self.src, dtype=int))

    def target_sites(self) -> set[Site]:
        """All target sites (set), regardless of labeled/unlabeled."""
        return set(self.dst)
