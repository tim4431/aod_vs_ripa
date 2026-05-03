"""Form the letters R, I, P, A in sequence on a RIPA array, with EOM tones.

A fixed set of 14 atoms repeatedly rearranges to spell each letter, holding
each shape between morphs so the EOM tone spectra/history clearly distinguish
hold (constant tones) from motion (tone sweeps).  Each 5x5 letter bitmap is
hand-designed to use exactly 14 dots, so the atom count is preserved and the
RIPA scheduler can route from one shape to the next without spawning or
destroying atoms.

Letter transitions are planned independently with
`UnlabeledRIPAPebbleAdvScheduler` (atoms are interchangeable, so each
transition picks the cheapest atom -> target assignment).  The per-transition
`MovingSequence`s are spliced onto a single master timeline by shifting their
`start_time`s; an inter-letter hold (longer than the moves themselves) leaves
the ensemble at rest between letters.

Renders with `view="detail"` so the row and col RIPA EOM tone spectra are
shown alongside the atom panel.

Usage:
    python example/forming_letters_demo.py            # quick check -> render/
    python example/forming_letters_demo.py --demo     # online quality -> demo/
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.atom_config import Grid
from src.atom_trajectory import AtomEnsemble
from src.movement import Step
from src.moving_sequence import MovingSequence
from src.routing import RoutingRequest, Site
from src.scheduler.heuristic.ripa_pebble_adv import UnlabeledRIPAPebbleAdvScheduler
from src.segments import make_hold
from src.visualization import render_animation

N = 11
GRID_SPACING_UM = 5.0
COLLISION_RADIUS_UM = 4.0
COLLISION_DT = 5e-6

# Where to place the 5x5 letter bitmap inside the NxN grid.
LETTER_OFFSET = (3, 3)

# Physics-time hold inserted before each transition.  Set well above a typical
# transition duration so each letter is held for noticeably longer than the
# motion takes; in the EOM tone view this reads as "constant tones, then a
# brief sweep, then constant tones".
LETTER_HOLD_S = 1e-4

PREFIX = "forming_letters_demo"

# 5x5 bitmaps. Each letter has exactly 14 dots so the atom count is preserved.
LETTERS: dict[str, list[str]] = {
    "R": [
        "XXXX.",
        "X...X",
        "XXXX.",
        "X.X..",
        "X..X.",
    ],
    "I": [
        "XXXXX",
        "X...X",
        "..X..",
        "..X..",
        "XXXXX",
    ],
    "P": [
        "XXXX.",
        "X...X",
        "XXXX.",
        "XX...",
        "XX...",
    ],
    "A": [
        ".XXX.",
        "X...X",
        "XXXXX",
        "X...X",
        "X...X",
    ],
}

LETTER_ORDER = ["R", "I", "P", "A"]


def letter_sites(name: str) -> list[Site]:
    """Bitmap (row, col) -> grid (i, j).

    The grid maps `i -> x` (rightward) and `j -> y` (upward). Bitmap column
    becomes horizontal position and bitmap row becomes vertical position
    (flipped so row 0 is at the top of the rendered plot).
    """
    rows = LETTERS[name]
    di, dj = LETTER_OFFSET
    nrows = len(rows)
    return sorted(
        (di + c, dj + (nrows - 1 - r))
        for r, row in enumerate(rows)
        for c, ch in enumerate(row)
        if ch == "X"
    )


def shifted_step(step: Step, offset: float) -> Step:
    """Translate a step (and its segments, if any) onto a later timeline."""
    if hasattr(step, "segments"):
        return replace(
            step,
            start_time=float(step.start_time) + offset,
            segments=tuple(
                replace(spec, start_time=float(spec.start_time) + offset)
                for spec in step.segments
            ),
        )
    return replace(step, start_time=float(step.start_time) + offset)


def plan_transition(
    grid: Grid,
    src: list[Site],
    dst: list[Site],
) -> MovingSequence:
    request = RoutingRequest(grid=grid, src=src, dst=dst, labeled=False)
    scheduler = UnlabeledRIPAPebbleAdvScheduler(request, collision_dt=COLLISION_DT)
    return scheduler.plan()


def fill_with_holds(ensemble: AtomEnsemble, tail_hold: float = 0.0) -> None:
    """Splice synthetic hold segments into every rest interval of every atom.

    `make_hold` produces a `start_pos == end_pos` segment with `channel=None`,
    which `_channels_for_segment` classifies as both row+col -- so each hold
    shows up in both EOM tone spectra/history panels at the atom's resting
    site frequency.  This is the visualization analogue of an active RIPA
    trap holding the atom in place during the gaps between moves.

    Fills (per atom): the initial gap before the first move, every gap
    between consecutive moves, and a tail hold of `tail_hold` past the last
    move so the final configuration is also held with traps.
    """
    target_total = ensemble.total_duration() + tail_hold
    for atom in ensemble.atomtrajs:
        new: list = []
        prev_end_time = 0.0
        prev_end_pos = tuple(atom.initial_pos)
        for seg in atom.segments:
            if seg.start_time > prev_end_time + 1e-12:
                new.append(
                    make_hold(prev_end_pos, prev_end_time, seg.start_time - prev_end_time)
                )
            new.append(seg)
            prev_end_time = seg.end_time
            prev_end_pos = seg.end_pos
        if target_total > prev_end_time + 1e-12:
            new.append(
                make_hold(prev_end_pos, prev_end_time, target_total - prev_end_time)
            )
        atom.segments[:] = new


def build_letters_sequence(
    grid: Grid,
    letters: list[str],
    hold: float,
) -> MovingSequence:
    src = letter_sites(letters[0])
    master = MovingSequence(
        grid=grid,
        initial=RoutingRequest(grid, src, src, labeled=False).initial,
        collision_dt=COLLISION_DT,
    )
    for prev, nxt in zip(letters, letters[1:]):
        site_by_atom = master.final_config().site_of_atom()
        current = [site_by_atom[atom_id] for atom_id in range(len(src))]
        stage = plan_transition(grid, current, letter_sites(nxt))
        offset = master.next_start_time() + hold
        for step in stage.steps:
            master.append(shifted_step(step, offset))
        print(
            f"  {prev} -> {nxt}: stage_steps={len(stage.steps)}, "
            f"stage_duration={stage.total_duration() * 1e6:.1f} us"
        )
    fill_with_holds(master.ensemble, tail_hold=hold)
    return master


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Form the letters R, I, P, A in sequence on a RIPA array."
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="render a high-quality GIF into demo/ instead of a quick render/ check",
    )
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="print the timing summary only and skip rendering",
    )
    args = parser.parse_args()

    counts = {name: len(letter_sites(name)) for name in LETTER_ORDER}
    if len(set(counts.values())) != 1:
        raise SystemExit(f"letter dot counts must all match, got {counts}")
    atoms = next(iter(counts.values()))

    grid = Grid(N=N, d=GRID_SPACING_UM, rc=COLLISION_RADIUS_UM)
    print(
        f"N={N}, atoms={atoms}, letters={'-> '.join(LETTER_ORDER)}, "
        f"hold_per_letter={LETTER_HOLD_S * 1e6:.0f} us"
    )

    sequence = build_letters_sequence(grid, LETTER_ORDER, hold=LETTER_HOLD_S)
    print(
        f"total_duration={sequence.total_duration() * 1e6:.1f} us, "
        f"steps={len(sequence.steps)}"
    )

    if args.no_render:
        return

    if args.demo:
        out_path = ROOT / "demo" / f"{PREFIX}.gif"
        quality = "quality"
    else:
        out_path = ROOT / "render" / f"{PREFIX}.gif"
        quality = "speed"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    render_animation(
        sequence,
        out_path,
        view="detail",
        quality=quality,
        time_dilation=1e4,
        hold_seconds=2.0,
        atom_colors={idx: "gray" for idx in range(atoms)},
        show_routing_on_start=False,
        title=f"forming letters: {' -> '.join(LETTER_ORDER)}",
    )
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
