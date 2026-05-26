"""AOD-only movement sketch for fold-transversal cultivation.

This example transcribes the two-qubit layers from arXiv:2509.05212,
Figs. A1 and A2. Single-qubit gates, resets, and measurements are kept in
the tick numbering but do not emit motion. Every CNOT-like entangling edge is
compiled as:

    move one atom through axis-aligned AOD waypoints -> dwell for CZ -> return

The quantum identity used here is the standard neutral-atom convention that a
CNOT is implemented as local single-qubit rotations around a CZ. Since this
script only studies atom transport, it only emits the CZ transport envelope.

This pass is deliberately AOD-only and uses one axis-aligned crossed-AOD set.
When a straight row/column route would collide with a resting atom, the router
tries a short orthogonal clearance move before crossing the crowded lane.

Usage:
    python example/code_cultivation/code_cultivation.py --no-render
    python example/code_cultivation/code_cultivation.py
    python example/code_cultivation/code_cultivation.py --demo
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.atom_config import AtomConfig, Grid, clean_position
from src.movement import AODStep, Step
from src.moving_sequence import MovingSequence
from src.segments import make_hold
from src.visualization import render_animation
from example.code_cultivation.code_cultivation_sequence import (
    INTERACTION_STAGES,
    ROT3_INITIALIZATION,
    ROT3_TO_REG3,
    Interaction,
    InteractionLayer,
    InteractionStage,
)


# The figures are layout diagrams, not calibrated hardware layouts. This pitch
# leaves enough room to visit a CZ site locally without running through a
# nearby resting atom.
LOGICAL_PITCH = 4.0
GRID_OFFSET = 1
GRID = Grid(N=11, d=3.0, rc=0.5)

GATE_SEPARATION = 1.0
LANE_CLEARANCE = 1.25
CZ_DWELL = 1.0e-6
PREFIX = "code_cultivation_aod"

POSITION_TOL = 1e-8


@dataclass(frozen=True)
class QubitSite:
    name: str
    logical_xy: tuple[float, float]
    role: str

    @property
    def pos(self) -> tuple[float, float]:
        return (
            GRID_OFFSET + LOGICAL_PITCH * self.logical_xy[0],
            GRID_OFFSET + LOGICAL_PITCH * self.logical_xy[1],
        )


@dataclass
class CZDwellStep(Step):
    """Hold the near pair long enough to represent the CZ pulse."""

    start_time: float
    atom_ids: tuple[int, ...]
    duration: float = CZ_DWELL
    label: str = ""
    channel: str = "aod"

    def apply(self, ensemble) -> None:
        if self.duration <= 0.0:
            return
        segments = []
        for atom_id in self.atom_ids:
            pos = ensemble.atomtraj_by_id(atom_id).resting_position_at(
                self.start_time
            )
            segments.append(
                (
                    atom_id,
                    make_hold(
                        pos,
                        self.start_time,
                        self.duration,
                        channel=self.channel,
                    ),
                )
            )
        ensemble.append_segments_batch(segments)

    def end_time(self, ensemble) -> float:
        return self.start_time + max(0.0, float(self.duration))


def derive_final_qubit_sites(
    stages: Sequence[InteractionStage],
) -> tuple[QubitSite, ...]:
    """Derive the needed final code positions from interaction labels."""

    labels = {
        label
        for stage in stages
        for layer in stage.layers
        for interaction in layer.interactions
        for label in (interaction.first, interaction.second)
    }
    return tuple(
        QubitSite(
            name=label,
            logical_xy=logical_position_from_label(label),
            role=role_from_label(label),
        )
        for label in sorted(labels, key=label_sort_key)
    )


def logical_position_from_label(label: str) -> tuple[float, float]:
    if len(label) != 3 or label[0] not in ("r", "a"):
        raise ValueError(f"unknown qubit label {label!r}")
    try:
        i = int(label[1])
        j = int(label[2])
    except ValueError as exc:
        raise ValueError(f"unknown qubit label {label!r}") from exc
    if label[0] == "r":
        return float(i), float(j)
    return i + 0.5, j + 0.5


def role_from_label(label: str) -> str:
    if label.startswith("r"):
        return "Rot(3) data"
    if label.startswith("a"):
        return "Rot(3) ancilla -> Reg(3) data"
    raise ValueError(f"unknown qubit label {label!r}")


def label_sort_key(label: str) -> tuple[int, int, int]:
    prefix_order = {"r": 0, "a": 1}
    if len(label) != 3 or label[0] not in prefix_order:
        raise ValueError(f"unknown qubit label {label!r}")
    return prefix_order[label[0]], int(label[1]), int(label[2])


def final_positions_by_label(
    stages: Sequence[InteractionStage],
) -> dict[str, tuple[float, float]]:
    return {site.name: site.pos for site in derive_final_qubit_sites(stages)}


ROT3_AND_REG3_SITES = derive_final_qubit_sites(INTERACTION_STAGES)


def initial_config() -> AtomConfig:
    positions = np.asarray([site.pos for site in ROT3_AND_REG3_SITES], dtype=float)
    atom_ids = np.arange(len(ROT3_AND_REG3_SITES), dtype=int)
    return AtomConfig(positions=positions, atom_ids=atom_ids)


def atom_id_by_name() -> dict[str, int]:
    return {site.name: k for k, site in enumerate(ROT3_AND_REG3_SITES)}


def build_sequence(stages: Sequence[InteractionStage]) -> MovingSequence:
    seq = MovingSequence(grid=GRID, initial=initial_config(), collision_dt=1e-7)
    ids = atom_id_by_name()

    for stage in stages:
        for layer in stage.layers:
            append_interaction_layer(
                seq,
                ids,
                layer,
                gate_separation=GATE_SEPARATION,
                dwell=CZ_DWELL,
            )
    return seq


def append_interaction_layer(
    seq: MovingSequence,
    ids: dict[str, int],
    layer: InteractionLayer,
    *,
    gate_separation: float,
    dwell: float,
) -> None:
    """Append single-mover AOD visit/dwell/return motion for one circuit tick."""

    if not layer.interactions:
        return

    seen: set[int] = set()
    for interaction in layer.interactions:
        atom_a = ids[interaction.first]
        atom_b = ids[interaction.second]
        if atom_a in seen or atom_b in seen:
            raise ValueError(
                "tick "
                f"{layer.tick} is not a disjoint two-qubit layer: {interaction}"
            )
        seen.update((atom_a, atom_b))

    for gate_index, interaction in enumerate(layer.interactions, start=1):
        label = (
            f"{layer.source} tick {layer.tick}.{gate_index} "
            f"{interaction.first}-{interaction.second}"
        )
        append_interaction(
            seq,
            ids,
            interaction,
            gate_separation=gate_separation,
            dwell=dwell,
            label=label,
        )


def append_interaction(
    seq: MovingSequence,
    ids: dict[str, int],
    interaction: Interaction,
    *,
    gate_separation: float,
    dwell: float,
    label: str,
) -> None:
    """Append one two-qubit gate as a single-mover AOD CZ visit."""

    mover = ids[interaction.first]
    fixed = ids[interaction.second]
    start_time = seq.next_start_time()
    home = seq.ensemble.atomtraj_by_id(mover).resting_position_at(start_time)
    fixed_pos = seq.ensemble.atomtraj_by_id(fixed).resting_position_at(start_time)
    gate_target = gate_target_for_pair(home, fixed_pos, gate_separation)

    forward_waypoints = append_aod_route(
        seq,
        mover,
        gate_target,
        f"{label}: move {interaction.first} to CZ",
    )
    seq.append(
        CZDwellStep(
            start_time=seq.next_start_time(),
            atom_ids=(mover, fixed),
            duration=dwell,
            label=f"{label}: CZ dwell",
        )
    )

    return_waypoints = tuple(reversed((home,) + forward_waypoints[:-1]))
    append_waypoint_path(
        seq,
        mover,
        return_waypoints,
        f"{label}: return {interaction.first}",
    )


def append_aod_route(
    seq: MovingSequence,
    atom_id: int,
    target: tuple[float, float],
    label: str,
) -> tuple[tuple[float, float], ...]:
    """Append the first collision-free axis-aligned route to `target`."""

    start = seq.ensemble.atomtraj_by_id(atom_id).resting_position_at(
        seq.next_start_time()
    )
    target = clean_position(target)
    if positions_close(start, target):
        return ()

    errors: list[str] = []
    for candidate in candidate_route_waypoints(start, target):
        waypoints = prune_axis_waypoints(start, candidate)
        if not waypoints:
            continue
        try:
            trial = clone_sequence(seq)
            append_waypoint_path(trial, atom_id, waypoints, label)
        except ValueError as exc:
            errors.append(str(exc))
            continue

        append_waypoint_path(seq, atom_id, waypoints, label)
        return waypoints

    detail = f"; last error: {errors[-1]}" if errors else ""
    raise RuntimeError(
        f"{label}: no collision-free axis-aligned AOD route "
        f"from {start} to {target}{detail}"
    )


def append_waypoint_path(
    seq: MovingSequence,
    atom_id: int,
    waypoints: Sequence[tuple[float, float]],
    label: str,
) -> tuple[tuple[float, float], ...]:
    """Append one-atom AOD moves along already chosen axis-aligned waypoints."""

    start = seq.ensemble.atomtraj_by_id(atom_id).resting_position_at(
        seq.next_start_time()
    )
    pruned = prune_axis_waypoints(start, waypoints)
    for index, waypoint in enumerate(pruned, start=1):
        append_aod_single_move(seq, atom_id, waypoint, f"{label}: leg {index}")
    return pruned


def append_aod_single_move(
    seq: MovingSequence,
    atom_id: int,
    target: tuple[float, float],
    label: str,
) -> None:
    """Append one legal one-atom move on the axis-aligned crossed AOD."""

    moves = ((atom_id, clean_position(target)),)
    step = make_aod_step(seq, moves)
    check_aod_step_affects_exactly(seq, step, moves)
    seq.append(step)


def make_aod_step(
    seq: MovingSequence,
    moves: Sequence[tuple[int, tuple[float, float]]],
) -> AODStep:
    start_time = seq.next_start_time()
    selected_axis_1: list[float] = []
    selected_axis_2: list[float] = []
    new_axis_1: list[float] = []
    new_axis_2: list[float] = []

    for atom_id, target in moves:
        current = seq.ensemble.atomtraj_by_id(atom_id).resting_position_at(start_time)
        old_1, old_2 = local_from_site(current)
        new_1, new_2 = local_from_site(target)
        selected_axis_1.append(old_1)
        selected_axis_2.append(old_2)
        new_axis_1.append(new_1)
        new_axis_2.append(new_2)

    ensure_unique_tones(selected_axis_1, "axis_1")
    ensure_unique_tones(selected_axis_2, "axis_2")

    return AODStep(
        start_time=start_time,
        selected_axis_1=tuple(selected_axis_1),
        selected_axis_2=tuple(selected_axis_2),
        new_axis_1=tuple(new_axis_1),
        new_axis_2=tuple(new_axis_2),
        match_tol=POSITION_TOL,
    )


def check_aod_step_affects_exactly(
    seq: MovingSequence,
    step: AODStep,
    moves: Sequence[tuple[int, tuple[float, float]]],
) -> None:
    expected = {int(atom_id): clean_position(target) for atom_id, target in moves}
    affected = list(step._affected(seq.ensemble))
    affected_ids = {int(atom_id) for atom_id, _, _ in affected}
    if affected_ids != set(expected):
        raise ValueError(
            "AOD batch would address the wrong atoms: "
            f"got {sorted(affected_ids)}, expected {sorted(expected)}"
        )
    for atom_id, _, actual_target in affected:
        if not positions_close(actual_target, expected[int(atom_id)]):
            raise ValueError(
                f"AOD batch target mismatch for atom {atom_id}: "
                f"got {actual_target}, expected {expected[int(atom_id)]}"
            )


def local_from_site(site: tuple[float, float]) -> tuple[float, float]:
    return float(site[0]), float(site[1])


def ensure_unique_tones(values: Sequence[float], axis_name: str) -> None:
    for idx, value in enumerate(values):
        for other in values[idx + 1:]:
            if abs(value - other) <= POSITION_TOL:
                raise ValueError(f"AOD {axis_name} tone collision in batch")


def positions_close(
    a: tuple[float, float],
    b: tuple[float, float],
    tol: float = POSITION_TOL,
) -> bool:
    return abs(a[0] - b[0]) <= tol and abs(a[1] - b[1]) <= tol


def gate_target_for_pair(
    mover_pos: tuple[float, float],
    fixed_pos: tuple[float, float],
    gate_separation: float,
) -> tuple[float, float]:
    """Choose a cardinal CZ site near the fixed atom for the moving atom."""

    dx = mover_pos[0] - fixed_pos[0]
    dy = mover_pos[1] - fixed_pos[1]
    distance = math.hypot(dx, dy)
    if distance <= 0.0:
        raise ValueError(f"cannot build a gate from coincident atoms at {mover_pos}")
    if distance <= gate_separation:
        return clean_position(mover_pos)

    if abs(dx) >= abs(dy) and abs(dx) > POSITION_TOL:
        sx = 1.0 if dx > 0.0 else -1.0
        return clean_position((fixed_pos[0] + sx * gate_separation, fixed_pos[1]))

    sy = 1.0 if dy > 0.0 else -1.0
    return clean_position((fixed_pos[0], fixed_pos[1] + sy * gate_separation))


def candidate_route_waypoints(
    start: tuple[float, float],
    target: tuple[float, float],
) -> Iterable[tuple[tuple[float, float], ...]]:
    """Yield Manhattan AOD routes, then orthogonal-clearance alternatives."""

    start = clean_position(start)
    target = clean_position(target)
    if positions_close(start, target):
        return

    same_x = abs(start[0] - target[0]) <= POSITION_TOL
    same_y = abs(start[1] - target[1]) <= POSITION_TOL
    if same_x or same_y:
        yield (target,)
    else:
        yield ((target[0], start[1]), target)
        yield ((start[0], target[1]), target)

    clearances = (LANE_CLEARANCE, LOGICAL_PITCH / 2.0, LOGICAL_PITCH)
    signs = (-1.0, 1.0)
    if not same_x:
        for clearance in clearances:
            for sign in signs:
                lane_y = start[1] + sign * clearance
                yield ((start[0], lane_y), (target[0], lane_y), target)
    if not same_y:
        for clearance in clearances:
            for sign in signs:
                lane_x = start[0] + sign * clearance
                yield ((lane_x, start[1]), (lane_x, target[1]), target)


def prune_axis_waypoints(
    start: tuple[float, float],
    waypoints: Sequence[tuple[float, float]],
) -> tuple[tuple[float, float], ...]:
    pruned: list[tuple[float, float]] = []
    current = clean_position(start)
    for waypoint in waypoints:
        waypoint = clean_position(waypoint)
        if positions_close(current, waypoint):
            continue
        if (
            abs(current[0] - waypoint[0]) > POSITION_TOL
            and abs(current[1] - waypoint[1]) > POSITION_TOL
        ):
            return ()
        pruned.append(waypoint)
        current = waypoint
    return tuple(pruned)


def clone_sequence(seq: MovingSequence) -> MovingSequence:
    trial = MovingSequence(
        grid=seq.grid,
        initial=seq.initial.copy(),
        collision_dt=seq.collision_dt,
    )
    for step in seq.steps:
        trial.append(step)
    return trial


def selected_stages(which: str) -> tuple[InteractionStage, ...]:
    if which == "rot3-init":
        return (ROT3_INITIALIZATION,)
    if which == "rot3-to-reg3":
        return (ROT3_TO_REG3,)
    return INTERACTION_STAGES


def iter_interactions(
    stages: Iterable[InteractionStage],
) -> Iterable[tuple[str, InteractionLayer, Interaction]]:
    for stage in stages:
        for layer in stage.layers:
            for interaction in layer.interactions:
                yield stage.name, layer, interaction


def print_summary(seq: MovingSequence, stages: Sequence[InteractionStage]) -> None:
    n_aod_steps = sum(isinstance(step, AODStep) for step in seq.steps)
    n_dwell_steps = sum(isinstance(step, CZDwellStep) for step in seq.steps)
    print("AOD-only cultivation movement sketch")
    print(f"atoms: {initial_config().n_atoms}")
    print(
        "two-qubit interactions compiled to CZ: "
        f"{sum(1 for _ in iter_interactions(stages))}"
    )
    print(f"AOD single-atom movement steps emitted: {n_aod_steps}")
    print(f"CZ dwell steps emitted: {n_dwell_steps}")
    print(f"total transport duration: {seq.total_duration() * 1e6:.3f} us")
    print()
    for stage in stages:
        print(stage.name)
        for layer in stage.layers:
            pairs = ", ".join(
                f"{interaction.first}-{interaction.second}"
                for interaction in layer.interactions
            )
            print(f"  tick {layer.tick:>2} ({layer.source}): {pairs}")


def assert_single_mover_sequence(seq: MovingSequence) -> None:
    """Guard against accidentally reintroducing multi-atom AOD moves."""

    replay = MovingSequence(
        grid=seq.grid,
        initial=seq.initial.copy(),
        collision_dt=1e-7,
    )
    for index, step in enumerate(seq.steps, start=1):
        if isinstance(step, AODStep):
            affected = list(step._affected(replay.ensemble))
            if len(affected) != 1:
                raise ValueError(
                    f"step {index} is not a single-mover AOD step: affects "
                    f"{len(affected)} atoms"
                )
        elif isinstance(step, CZDwellStep) and len(step.atom_ids) != 2:
            raise ValueError(
                f"step {index} is not pairwise: CZ dwell holds "
                f"{len(step.atom_ids)} atoms"
            )
        replay.append(step)


def render(seq: MovingSequence, out_path: Path, quality: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    colors = {}
    for atom_id, site in enumerate(ROT3_AND_REG3_SITES):
        if site.role.startswith("Rot(3) data"):
            colors[atom_id] = "#4c78a8"
        else:
            colors[atom_id] = "#f58518"
    render_animation(
        seq,
        out_path,
        view="demo",
        quality=quality,
        atom_colors=colors,
        show_routing_on_start=False,
        title="AOD-only Rot(3) initialization and Rot(3)->Reg(3) CZ transport",
    )
    print(f"wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build neutral-atom movement for code-cultivation two-qubit gates."
    )
    parser.add_argument(
        "--stage",
        choices=("all", "rot3-init", "rot3-to-reg3"),
        default="all",
        help="which paper stage to compile",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="render a high-quality GIF into demo/ instead of a quick render/ check",
    )
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="print the summary only and skip rendering",
    )
    args = parser.parse_args()

    stages = selected_stages(args.stage)
    seq = build_sequence(stages)
    report = seq.validate(dt=1e-7)
    if not report.ok:
        raise RuntimeError(report)
    assert_single_mover_sequence(seq)

    print_summary(seq, stages)
    if args.no_render:
        return

    if args.demo:
        out_path = ROOT / "demo" / f"{PREFIX}.gif"
        quality = "quality"
    else:
        out_path = ROOT / "render" / f"{PREFIX}.gif"
        quality = "speed"
    render(seq, out_path, quality)


if __name__ == "__main__":
    main()
