"""Manual AOD movement script with CZ-beam visualization.

The SCRIPT below is a list of **sync batches**. Each batch is a list of
operations that fire at the same start_time via `MovingSequence.append_sync_batch`
(same pattern as `example/presentation_demo/async_l_to_line.py`); the batch
ends when its longest sub-step ends, and the next batch begins from there.

Op forms:
    ("mv", atom_name, (x, y))   # AOD single-atom move to absolute coords
    ("cz", atom_a, atom_b)      # CZ dwell on the (already adjacent) pair

The renderer draws, on top of the normal atom/trap layer, a dashed Rydberg-
blockade ring around each atom in an active CZ and a Gaussian "beam" glow
midway between them. The CZ dwell is only ~1 us, so the overlay carries a
QUINTIC ramp-in / ramp-out envelope of CZ_VIZ_RAMP so it stays on-screen
long enough to read.

Usage:
    python example/code_cultivation/code_cultivation_manual.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.atom_config import AtomConfig, Grid  # noqa: E402
from src.movement import AODStep, GateStep, RIPAStep, Step  # noqa: E402
from src.moving_sequence import MovingSequence  # noqa: E402
from src.visualization import render_animation  # noqa: E402


# ----------------------------------------------------------------------------- #
# Layout + timing
# ----------------------------------------------------------------------------- #

GRID = Grid(N=11, d=3.0, rc=0.5)

# Atom name -> initial (x, y) grid site. r* are Rot(3) data on a coarse grid;
# a* are ancilla atoms sitting in the plaquette centers.
INITIAL: dict[str, tuple[float, float]] = {
    "r00": (1.0, 1.0),  "r01": (1.0, 5.0),  "r02": (1.0, 9.0),
    "r10": (5.0, 1.0),  "r11": (5.0, 5.0),  "r12": (5.0, 9.0),
    "r20": (9.0, 1.0),  "r21": (9.0, 5.0),  "r22": (9.0, 9.0),
    "a00": (3.0, 3.0),  "a01": (3.0, 7.0),
    "a10": (7.0, 3.0),  "a11": (7.0, 7.0),
}

CZ_DWELL = 30.0e-6            # CZ pulse window; atoms hold still throughout (s)
CZ_VIZ_RAMP = 6.0e-6          # quintic ramp-in + ramp-out at each end of the dwell (s)
BLOCKADE_RADIUS_UM = 1.4       # dashed-ring radius around each CZ atom (um)


# ----------------------------------------------------------------------------- #
# SCRIPT: one batch per row; all ops in a row fire on the same sync_batch.
# ----------------------------------------------------------------------------- #

SCRIPT_RIPA: list[list[tuple]] = [
    # === Fig. A1 tick 2: 4 parallel CZs (single-leg movers) ===
    [("mv", "r01", (1, 8)),("mv", "r10", (5, 4)),
    ("mv", "r21", (9, 2)),("mv", "r12", (8, 9))],
    [("cz", "r01", "r02"), ("cz", "r11", "r10"),
     ("cz", "r12", "r22"), ("cz", "r20", "r21")],
    [("mv", "r01", (1, 5)), ("mv", "r11", (2, 5)),
     ("mv", "r10", (5, 8)),
     ("mv", "r21", (9, 5)), ("mv", "r12", (5, 9)),
    # === Fig. A1 tick 3: r01 short, r10 takes a clearance lane around r11.
    # We share batch 1 with both atoms; r10's longer route then trails on its own.
    ],
    [("cz", "r01", "r11"), ("cz", "r10", "r12")],
    [("mv", "r10", (5, 1)), ("mv", "r11", (5, 5)),
    # === Fig. A1 tick 4: 3 parallel CZs; r00 is single-leg, r01 + r20 are 2-leg.
    ("mv", "r01", (4, 5)),("mv", "r00", (4, 1)), ("mv", "r20", (6, 1)),
    ],

    [("mv", "r01", (4, 9)),("mv", "r20", (6, 5))],
    [("cz", "r01", "r12"), ("cz", "r00", "r10"), ("cz", "r20", "r11")],
    [("mv", "r01", (4, 5)),("mv", "r20", (6, 1))],
    [("mv", "r00", (1, 1)),("mv", "r01", (1, 5)),("mv", "r20", (9, 1)),
    # === Fig. A1 tick 5: lone CZ ===
     ("mv", "r10", (8, 1))],
    [("cz", "r10", "r20")],
    [("mv", "r10", (5, 1)),
    # === Fig. A2 tick 20: 4 parallel CZs (Rot(3) ancillas -> Reg(3) sites) ===
     ("mv", "a01", (2, 7)), ("mv", "a11", (6, 7)),
     ("mv", "a00", (2, 3)), ("mv", "a10", (6, 3))],
    [("mv", "a01", (2, 9)), ("mv", "a11", (6, 9)),
     ("mv", "a00", (2, 5)), ("mv", "a10", (6, 5))],
    [("cz", "a01", "r02"), ("cz", "a11", "r12"),
     ("cz", "a00", "r01"), ("cz", "a10", "r11")],
    [("mv", "a01", (4, 9)), ("mv", "a11", (6, 7)),
     ("mv", "a00", (2, 1)), ("mv", "a10", (8, 5)),
     ("mv", "r11", (5, 7))],


    # === Fig. A2 tick 21: 4 parallel CZs ===
    [("cz", "a01", "r12"), ("cz", "a11", "r11"),
     ("cz", "a10", "r21"), ("cz", "a00", "r00")],
    [("mv", "a01", (4, 7)), ("mv", "a11", (7, 7)),
     ("mv", "a00", (2, 3)), ("mv", "a10", (8, 3)),
     ("mv", "r11", (5, 5))],
    [("mv", "a01", (3, 7)), ("mv", "a00", (3, 3)),
     ("mv", "a10", (7, 3))],
]


SCRIPT_AOD: list[list[tuple]] = [
    # === Fig. A1 tick 2: 4 parallel CZs (single-leg movers) ===
    [("mv", "r01", (1, 8))],
    [("mv", "r11", (5, 2)), ("mv", "r21", (9, 2))],
    [("mv", "r12", (8, 9))],
    [("cz", "r01", "r02"), ("cz", "r11", "r10"),
     ("cz", "r12", "r22"), ("cz", "r20", "r21")],
    [("mv", "r01", (1, 5))],
    [("mv", "r11", (5, 5)), ("mv", "r21", (9, 5))],
    [("mv", "r12", (5, 9))],

    # === Fig. A1 tick 3: r01 short, r10 takes a clearance lane around r11.
    # We share batch 1 with both atoms; r10's longer route then trails on its own.
    [("mv", "r11", (2, 5))],
    [("mv", "r10", (5, 8))],
    [("cz", "r01", "r11"), ("cz", "r10", "r12")],
    [("mv", "r10", (5, 1))],
    [("mv", "r11", (5, 5))],

    # === Fig. A1 tick 4: 3 parallel CZs; r00 is single-leg, r01 + r20 are 2-leg.
    [("mv", "r01", (4, 5)),("mv", "r00", (4, 1))],
    [("mv", "r01", (4, 9))],
    [("mv", "r20", (6, 1))],
    [("mv", "r20", (6, 5))],
    [("cz", "r01", "r12"), ("cz", "r00", "r10"), ("cz", "r20", "r11")],
    [("mv", "r20", (6, 1))],
    [("mv", "r20", (9, 1))],
    [("mv", "r01", (4, 5))],
    [("mv", "r00", (1, 1)), ("mv", "r01", (1, 5))],

    # === Fig. A1 tick 5: lone CZ ===
    [("mv", "r10", (8, 1))],
    [("cz", "r10", "r20")],
    [("mv", "r10", (5, 1))],

    # === Fig. A2 tick 20: 4 parallel CZs (Rot(3) ancillas -> Reg(3) sites) ===
    [("mv", "a01", (2, 9)), ("mv", "a11", (6, 9)),
     ("mv", "a00", (2, 5)), ("mv", "a10", (6, 5))],
    [("cz", "a01", "r02"), ("cz", "a11", "r12"),
     ("cz", "a00", "r01"), ("cz", "a10", "r11")],
    [("mv", "a01", (3, 7)), ("mv", "a11", (7, 7)),
     ("mv", "a00", (3, 3)), ("mv", "a10", (7, 3))],

    # === Fig. A2 tick 21: 4 parallel CZs ===
    [("mv", "a01", (4, 8)), ("mv", "r11", (6, 6))],
    [("mv", "r00", (2, 2)), ("mv", "a10", (8, 4))],
    [("cz", "a01", "r12"), ("cz", "a11", "r11"),
     ("cz", "a10", "r21"), ("cz", "a00", "r00")],
    [("mv", "r00", (1, 1)), ("mv", "a10", (7, 3))],
    [("mv", "a01", (3, 7)), ("mv", "r11", (5, 5))],
]


# ----------------------------------------------------------------------------- #
# Build
# ----------------------------------------------------------------------------- #

def _build_aod_step(
    seq: MovingSequence,
    t0: float,
    name_to_id: dict[str, int],
    mv_ops: list[tuple],
) -> AODStep:
    """Pack a batch's "mv" ops into ONE multi-atom AODStep on the default
    (x, y) basis, then validate it against `AODStep._affected`.

    A real AOD shot moves the Cartesian product `selected_axis_1 x
    selected_axis_2` as a single tone sweep per axis. Two atoms sharing a
    source column must therefore share a target column (same for rows), and
    the product must not pick up any atom we did not intend to move.
    """
    id_to_name = {v: k for k, v in name_to_id.items()}
    sources: list[tuple[float, float]] = []
    targets: list[tuple[float, float]] = []
    ids: list[int] = []
    for _, name, target in mv_ops:
        aid = name_to_id[name]
        cur = seq.ensemble.atomtraj_by_id(aid).resting_position_at(t0)
        sources.append((float(cur[0]), float(cur[1])))
        targets.append((float(target[0]), float(target[1])))
        ids.append(aid)

    col_map: dict[float, float] = {}
    row_map: dict[float, float] = {}
    for (sx, sy), (tx, ty), (_, name, _) in zip(sources, targets, mv_ops):
        if sx in col_map and abs(col_map[sx] - tx) > 1e-8:
            raise ValueError(
                f"AOD batch at t={t0 * 1e6:.3f}us: source column {sx} maps to "
                f"both {col_map[sx]} and {tx} (atom {name!r}). One AOD shot "
                f"sweeps a whole column tone, so atoms sharing a column must "
                f"share a target column -- split this batch."
            )
        col_map[sx] = tx
        if sy in row_map and abs(row_map[sy] - ty) > 1e-8:
            raise ValueError(
                f"AOD batch at t={t0 * 1e6:.3f}us: source row {sy} maps to "
                f"both {row_map[sy]} and {ty} (atom {name!r}). One AOD shot "
                f"sweeps a whole row tone, so atoms sharing a row must share "
                f"a target row -- split this batch."
            )
        row_map[sy] = ty

    selected_axis_1 = tuple(sorted(col_map))
    new_axis_1 = tuple(col_map[c] for c in selected_axis_1)
    selected_axis_2 = tuple(sorted(row_map))
    new_axis_2 = tuple(row_map[r] for r in selected_axis_2)

    step = AODStep(
        start_time=t0,
        selected_axis_1=selected_axis_1,
        selected_axis_2=selected_axis_2,
        new_axis_1=new_axis_1,
        new_axis_2=new_axis_2,
        match_tol=1e-8,
    )

    affected = {aid: (old, new) for aid, old, new in step._affected(seq.ensemble)}
    expected_ids = set(ids)
    extras = set(affected) - expected_ids
    if extras:
        raise ValueError(
            f"AOD batch at t={t0 * 1e6:.3f}us: the Cartesian product "
            f"sel_1={selected_axis_1} x sel_2={selected_axis_2} also "
            f"addresses unintended atoms "
            f"{sorted(id_to_name[a] for a in extras)}. Pick different rows/"
            f"columns or move those atoms in the same shot."
        )
    missing = expected_ids - set(affected)
    if missing:
        raise ValueError(
            f"AOD batch at t={t0 * 1e6:.3f}us: requested atoms "
            f"{sorted(id_to_name[a] for a in missing)} are not at the "
            f"Cartesian product intersections -- check the source positions."
        )
    for aid, (sx, sy), (tx, ty) in zip(ids, sources, targets):
        got_new = affected[aid][1]
        if abs(got_new[0] - tx) > 1e-7 or abs(got_new[1] - ty) > 1e-7:
            raise ValueError(
                f"AOD batch at t={t0 * 1e6:.3f}us: atom {id_to_name[aid]!r} "
                f"is routed to {got_new} instead of {(tx, ty)} by this shot."
            )
    return step


def build_sequence() -> tuple[MovingSequence, dict[str, int]]:
    names = list(INITIAL)
    name_to_id = {name: idx for idx, name in enumerate(names)}
    positions = np.asarray([INITIAL[n] for n in names], dtype=float)
    initial = AtomConfig(positions=positions, atom_ids=np.arange(len(names)))
    seq = MovingSequence(grid=GRID, initial=initial, collision_dt=1e-7)

    for batch in SCRIPT_AOD:
        t0 = seq.next_start_time()
        mvs = [op for op in batch if op[0] == "mv"]
        czs = [op for op in batch if op[0] == "cz"]
        unknown = [op for op in batch if op[0] not in ("mv", "cz")]
        if unknown:
            raise ValueError(f"unknown op kind {unknown[0][0]!r}")

        steps: list[Step] = []
        if mvs:
            steps.append(_build_aod_step(seq, t0, name_to_id, mvs))
        for _, a, b in czs:
            steps.append(
                GateStep(
                    start_time=t0,
                    atom_ids=(name_to_id[a], name_to_id[b]),
                    duration=CZ_DWELL,
                    gate_type="CZ",
                    label=f"CZ {a}-{b}",
                )
            )
        seq.append_sync_batch(steps)
    return seq, name_to_id


def build_sequence_ripa() -> tuple[MovingSequence, dict[str, int]]:
    """Same SCRIPT pattern as `build_sequence`, but each "mv" op compiles to a
    single-atom RIPAStep on the row or col EOM channel (derived from the move
    direction). RIPA naturally moves any number of disjoint atoms in parallel
    -- they just share the sync_batch start_time, each on its own channel."""
    names = list(INITIAL)
    name_to_id = {name: idx for idx, name in enumerate(names)}
    positions = np.asarray([INITIAL[n] for n in names], dtype=float)
    initial = AtomConfig(positions=positions, atom_ids=np.arange(len(names)))
    seq = MovingSequence(grid=GRID, initial=initial, collision_dt=1e-7)

    for batch in SCRIPT_RIPA:
        t0 = seq.next_start_time()
        mvs = [op for op in batch if op[0] == "mv"]
        czs = [op for op in batch if op[0] == "cz"]
        unknown = [op for op in batch if op[0] not in ("mv", "cz")]
        if unknown:
            raise ValueError(f"unknown op kind {unknown[0][0]!r}")

        steps: list[Step] = []
        for _, name, target in mvs:
            aid = name_to_id[name]
            current = seq.ensemble.atomtraj_by_id(aid).resting_position_at(t0)
            di = float(target[0]) - float(current[0])
            dj = float(target[1]) - float(current[1])
            if abs(di) > 1e-9 and abs(dj) > 1e-9:
                raise ValueError(
                    f"RIPA move {name!r} at t={t0 * 1e6:.3f}us: "
                    f"{current} -> {target} crosses both row and col axes; "
                    f"RIPA needs a single-axis target per step"
                )
            channel = "row" if abs(di) > 1e-9 else "col"
            steps.append(
                RIPAStep(
                    start_time=t0,
                    atom_id=aid,
                    target=(float(target[0]), float(target[1])),
                    channel=channel,
                )
            )
        for _, a, b in czs:
            steps.append(
                GateStep(
                    start_time=t0,
                    atom_ids=(name_to_id[a], name_to_id[b]),
                    duration=CZ_DWELL,
                    gate_type="CZ",
                    label=f"CZ {a}-{b}",
                )
            )
        seq.append_sync_batch(steps)
    return seq, name_to_id


# ----------------------------------------------------------------------------- #
# Render -- delegated to src.visualization.render_animation (benchmark view).
# ----------------------------------------------------------------------------- #


def main() -> None:
    aod_seq, name_to_id = build_sequence()
    ripa_seq, _ = build_sequence_ripa()
    for label, seq in (("AOD", aod_seq), ("RIPA", ripa_seq)):
        report = seq.validate(dt=1e-7)
        if not report.ok:
            raise RuntimeError(f"{label}: {report}")

    print(f"atoms: {len(name_to_id)}")
    for label, seq, script in (
        ("AOD", aod_seq, SCRIPT_AOD),
        ("RIPA", ripa_seq, SCRIPT_RIPA),
    ):
        n_aod = sum(1 for s in seq.steps if isinstance(s, AODStep))
        n_ripa = sum(1 for s in seq.steps if isinstance(s, RIPAStep))
        n_cz = sum(1 for s in seq.steps if isinstance(s, GateStep))
        kind = f"AOD={n_aod}" if n_aod else f"RIPA={n_ripa}"
        print(
            f"  {label}: {len(script)} batches, {kind}, CZ={n_cz}, "
            f"duration={seq.total_duration() * 1e6:.3f} us"
        )

    atom_colors = {
        idx: ("#4c78a8" if name.startswith("r") else "#f58518")
        for name, idx in name_to_id.items()
    }
    out_path = ROOT / "render" / "code_cultivation_manual.gif"
    render_animation(
        # Dict order = panel order (left -> right): RIPA on the left, AOD right.
        {"RIPA": ripa_seq, "AOD": aod_seq},
        out_path,
        view="benchmark",
        quality="quality",
        time_dilation=8e3,
        hold_seconds=1.5,
        atom_colors=atom_colors,
        atom_scale=1.5,
        trap_scale=0.8,
        show_routing_on_start=False,
        panel_speedup={"AOD": 2.0},
        title="AOD vs RIPA - code-cultivation Rot(3) init + Rot(3)->Reg(3)",
    )
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
