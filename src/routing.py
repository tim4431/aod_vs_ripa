"""RoutingRequest: the input to a scheduler.

Two flavors mirror the two routing problems:

* **Unlabeled** (Case 1): only the source set and target set matter. Any
  bijection source -> target is acceptable. The scheduler is free to pick
  the assignment.
* **Labeled** (Case 2): explicit (label, target) pairs. The scheduler
  must respect this assignment.

Both share the same `AtomConfig` as the starting state.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Optional

from .atoms import AtomConfig


Site = tuple[int, int]


@dataclass
class RoutingRequest:
    initial: AtomConfig
    # Unlabeled (Case 1): set of target sites; any source atom may fill any.
    targets: Optional[set[Site]] = None
    # Labeled (Case 2): label -> target site. Labels must match initial.labels.
    pairing: Optional[dict[Hashable, Site]] = None

    def __post_init__(self):
        if (self.targets is None) == (self.pairing is None):
            raise ValueError("Provide exactly one of `targets` or `pairing`")
        if self.pairing is not None and not self.initial.labeled:
            raise ValueError("Labeled routing requires labeled AtomConfig")
        if self.pairing is not None:
            missing = set(self.pairing) - set(self.initial.labels)
            if missing:
                raise ValueError(f"Pairing references unknown labels: {missing}")

    @property
    def labeled(self) -> bool:
        return self.pairing is not None

    def target_sites(self) -> set[Site]:
        """All target sites (set), regardless of labeled/unlabeled."""
        if self.targets is not None:
            return set(self.targets)
        return set(self.pairing.values())
