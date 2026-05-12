"""Structured interaction sequence for the code-cultivation example.

This module is the machine-readable companion to
`example/code_cultivation/code_cultivation_sequence.md`. It intentionally
contains only logical two-qubit interactions and no hardware movement
assumptions.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Interaction:
    first: str
    second: str
    note: str = ""


@dataclass(frozen=True)
class InteractionLayer:
    tick: int
    source: str
    interactions: tuple[Interaction, ...]


@dataclass(frozen=True)
class InteractionStage:
    name: str
    layers: tuple[InteractionLayer, ...]


ROT3_INITIALIZATION = InteractionStage(
    name="initialize_rot3",
    layers=(
        InteractionLayer(
            tick=2,
            source="Fig. A1",
            interactions=(
                Interaction("r01", "r02", "left vertical edge"),
                Interaction("r11", "r10", "central vertical edge"),
                Interaction("r12", "r22", "top horizontal edge"),
                Interaction("r20", "r21", "right vertical edge"),
            ),
        ),
        InteractionLayer(
            tick=3,
            source="Fig. A1",
            interactions=(
                Interaction("r01", "r11", "left-to-center edge"),
                Interaction("r10", "r12", "central long edge"),
            ),
        ),
        InteractionLayer(
            tick=4,
            source="Fig. A1",
            interactions=(
                Interaction("r01", "r12", "upper-left diagonal edge"),
                Interaction("r00", "r10", "lower-left horizontal edge"),
                Interaction("r20", "r11", "lower-right diagonal edge"),
            ),
        ),
        InteractionLayer(
            tick=5,
            source="Fig. A1",
            interactions=(Interaction("r10", "r20", "bottom horizontal edge"),),
        ),
    ),
)


ROT3_TO_REG3 = InteractionStage(
    name="rot3_to_reg3",
    layers=(
        InteractionLayer(
            tick=20,
            source="Fig. A2",
            interactions=(
                Interaction("a01", "r02", "upper-left diagonal edge"),
                Interaction("a11", "r12", "upper-right diagonal edge"),
                Interaction("a00", "r01", "lower-left diagonal edge"),
                Interaction("a10", "r11", "lower-right diagonal edge"),
            ),
        ),
        InteractionLayer(
            tick=21,
            source="Fig. A2",
            interactions=(
                Interaction("a01", "r12", "upper-middle edge"),
                Interaction("a11", "r11", "central edge"),
                Interaction("a10", "r21", "right-middle edge"),
                Interaction("a00", "r00", "lower-left edge"),
            ),
        ),
    ),
)


INTERACTION_STAGES = (ROT3_INITIALIZATION, ROT3_TO_REG3)
