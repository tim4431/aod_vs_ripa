"""RoutingRequest: input to a scheduler.

* **Unlabeled** (Case 1): only the target *set* matters. Any source atom
  can fill any target. `targets: set[Site]`.
* **Labeled** (Case 2): atoms are distinguishable (encoded info). The
  request specifies a `pairing: dict[Site, Site]` from each source site
  to its target site. The scheduler must respect this assignment.

We address atoms by their initial site (an integer (i, j)) rather than
by an external label — the atom's location at t=0 is a unique key, and
this keeps the request free of separate identifier bookkeeping.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .atom_config import AtomConfig

Site = tuple[int, int]


@dataclass
class RoutingRequest:
    initial: AtomConfig
    targets: Optional[set[Site]] = None  # case 1
    pairing: Optional[dict[Site, Site]] = None  # case 2: src_site -> tgt_site

    def __post_init__(self):
        if (self.targets is None) == (self.pairing is None):
            raise ValueError("Provide exactly one of `targets` or `pairing`")
        if self.pairing is not None:
            occ = self.initial.occupied_sites()
            missing = set(self.pairing) - occ
            if missing:
                raise ValueError(f"pairing references unoccupied sites: {missing}")
            tgts = list(self.pairing.values())
            if len(set(tgts)) != len(tgts):
                raise ValueError("pairing has duplicate target sites")

    @property
    def labeled(self) -> bool:
        return self.pairing is not None

    def target_sites(self) -> set[Site]:
        """All target sites, regardless of labeled/unlabeled."""
        if self.targets is not None:
            return set(self.targets)
        return set(self.pairing.values())
