"""Reusable scientific-drawing primitives that aren't tied to the
animation pipeline. Anything in here takes an `Axes` plus a few plain
inputs and just paints; it is callable from any plotting context.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from matplotlib.patches import Polygon, Rectangle, Wedge

from .atom_trajectory import AtomEnsemble

COLOR_CODE_PINK = "#f4cdd6"
COLOR_CODE_BLUE = "#cdd9f0"
Z_COLOR_CODE = 0.5  # below grid dots / atoms in `visualization.py`


def draw_color_code_pattern(
    ax: Any,
    ensemble: AtomEnsemble,
    *,
    alpha: float = 0.9,
    swap_colors: bool = False,
) -> None:
    """Overlay a rotated surface-code stabilizer pattern on a square patch.

    Reads the patch geometry from `ensemble.positions_at(0.0)` and
    requires a filled square of evenly-spaced atoms. Draws bulk
    checkerboard plaquettes plus eight outward-facing semicircular
    boundary lobes (top/bottom blue, left/right pink), placed so each
    lobe alternates with the bulk plaquette directly inside it.
    `swap_colors=True` applies the H^{otimes n} X<->Z swap by exchanging
    the two plaquette colors everywhere (used for the post-rotation
    state, where Hadamard has already flipped every qubit's basis).
    """
    bulk_even = COLOR_CODE_BLUE if swap_colors else COLOR_CODE_PINK
    bulk_odd = COLOR_CODE_PINK if swap_colors else COLOR_CODE_BLUE
    edge_blue = COLOR_CODE_PINK if swap_colors else COLOR_CODE_BLUE
    edge_pink = COLOR_CODE_BLUE if swap_colors else COLOR_CODE_PINK

    initial = ensemble.positions_at(0.0)
    rows = sorted({float(p[0]) for p in initial})
    cols = sorted({float(p[1]) for p in initial})
    if len(rows) < 2 or rows != cols:
        return
    grid = ensemble.grid
    n = len(rows)

    def xy(i: float, j: float) -> tuple[float, float]:
        a = grid.ij_to_xy(np.asarray([i, j], dtype=float))
        return float(a[0]), float(a[1])

    # Bulk: checkerboard rectangles between every 4 adjacent atoms.
    for ri in range(n - 1):
        for ci in range(n - 1):
            color = bulk_even if (ri + ci) % 2 == 0 else bulk_odd
            x0, y0 = xy(rows[ri], cols[ci])
            x1, y1 = xy(rows[ri + 1], cols[ci + 1])
            ax.add_patch(Rectangle(
                (min(x0, x1), min(y0, y1)),
                abs(x1 - x0), abs(y1 - y0),
                facecolor=color, edgecolor="none", alpha=alpha,
                zorder=Z_COLOR_CODE,
            ))

    # Boundary lobes: semicircular wedges. A lobe is placed at a
    # boundary pair iff the bulk plaquette directly inside has the
    # opposite color (the lobe extends the alternating pattern).
    def lobe(cx: float, cy: float, r: float, t1: float, t2: float, color: str) -> None:
        ax.add_patch(Wedge(
            center=(cx, cy), r=r, theta1=t1, theta2=t2,
            facecolor=color, edgecolor="none", alpha=alpha,
            zorder=Z_COLOR_CODE,
        ))

    for ri in range(n - 1):
        r = abs(xy(rows[ri + 1], 0.0)[0] - xy(rows[ri], 0.0)[0]) / 2.0
        cx_top, cy_top = xy(0.5 * (rows[ri] + rows[ri + 1]), cols[-1])
        cx_bot, cy_bot = xy(0.5 * (rows[ri] + rows[ri + 1]), cols[0])
        # Top: bulk plaquette (ri, n-2) pink iff (ri + n-2) even -> lobe present, blue.
        if (ri + n - 2) % 2 == 0:
            lobe(cx_top, cy_top, r, 0.0, 180.0, edge_blue)
        # Bottom: bulk plaquette (ri, 0) pink iff ri even -> lobe present, blue.
        if ri % 2 == 0:
            lobe(cx_bot, cy_bot, r, 180.0, 360.0, edge_blue)

    for ci in range(n - 1):
        r = abs(xy(0.0, cols[ci + 1])[1] - xy(0.0, cols[ci])[1]) / 2.0
        cx_right, cy_right = xy(rows[-1], 0.5 * (cols[ci] + cols[ci + 1]))
        cx_left, cy_left = xy(rows[0], 0.5 * (cols[ci] + cols[ci + 1]))
        # Right: bulk plaquette (n-2, ci) blue iff (n-2 + ci) odd -> lobe present, pink.
        if (n - 2 + ci) % 2 == 1:
            lobe(cx_right, cy_right, r, -90.0, 90.0, edge_pink)
        # Left: bulk plaquette (0, ci) blue iff ci odd -> lobe present, pink.
        if ci % 2 == 1:
            lobe(cx_left, cy_left, r, 90.0, 270.0, edge_pink)


def draw_reg3_color_code_pattern(
    ax: Any,
    ensemble: AtomEnsemble,
    *,
    pitch: float,
    alpha: float = 0.9,
    swap_colors: bool = False,
) -> None:
    """Overlay a Reg(3) rotated color-code stabilizer pattern on a square patch.

    `pitch` is the spacing (in grid units) between adjacent *data* atoms --
    the logical pitch of the rotated code, NOT the raw spacing of every atom
    in the ensemble. Pass this explicitly so dense layouts (data + ancilla
    atoms on a half-pitch sublattice) still draw the right-sized diamonds.

    Bounding box is taken from the most extreme atom positions in
    `ensemble.positions_at(0.0)`; the function assumes a square layout.
    The tiling places one 45-deg-rotated diamond at every logical data-atom
    site AND every plaquette midpoint, with diamond half-extent = pitch/2.
    Adjacent diamonds (atom <-> plaquette) share an edge so the two-color
    split is bipartite. Boundary atoms emit half-diamond triangles (one
    vertex dropped) or corner triangles (two vertices dropped, with the
    atom position itself as the third vertex).
    """
    initial = ensemble.positions_at(0.0)
    xs = sorted({float(p[0]) for p in initial})
    ys = sorted({float(p[1]) for p in initial})
    if not xs or not ys:
        return
    x_lo, x_hi = xs[0], xs[-1]
    y_lo, y_hi = ys[0], ys[-1]
    nx = int(round((x_hi - x_lo) / pitch)) + 1
    ny = int(round((y_hi - y_lo) / pitch)) + 1
    if nx < 2 or ny < 2 or nx != ny:
        return
    n = nx
    grid = ensemble.grid

    atom_color = COLOR_CODE_PINK if swap_colors else COLOR_CODE_BLUE
    plaquette_color = COLOR_CODE_BLUE if swap_colors else COLOR_CODE_PINK

    def xy(i: float, j: float) -> tuple[float, float]:
        a = grid.ij_to_xy(np.asarray([i, j], dtype=float))
        return float(a[0]), float(a[1])

    # Diamond vertex offsets in CW order from +y: top, right, bottom, left.
    r = 0.5 * pitch * grid.d
    offsets = ((0.0, +r), (+r, 0.0), (0.0, -r), (-r, 0.0))

    def atom_polygon(ri: int, ci: int) -> list[tuple[float, float]]:
        """Diamond at logical atom (ri, ci) with any off-grid vertices dropped.

        Interior atoms -> full diamond (4 verts). Edge atoms -> half-diamond
        triangle (3 verts). Corner atoms -> right triangle whose right-angle
        vertex is the atom itself (3 verts). Detection: a 2-long run of
        dropped vertices means we crossed a bbox corner, which coincides
        exactly with the atom position.
        """
        cx, cy = xy(x_lo + ri * pitch, y_lo + ci * pitch)
        dropped = (
            ci == n - 1,  # top vertex outside iff atom on top edge
            ri == n - 1,  # right
            ci == 0,      # bottom
            ri == 0,      # left
        )
        verts: list[tuple[float, float]] = []
        for k in range(4):
            if dropped[k]:
                if dropped[(k - 1) % 4] and not dropped[(k + 1) % 4]:
                    verts.append((cx, cy))
            else:
                dx, dy = offsets[k]
                verts.append((cx + dx, cy + dy))
        return verts

    for ri in range(n):
        for ci in range(n):
            verts = atom_polygon(ri, ci)
            if len(verts) >= 3:
                ax.add_patch(Polygon(
                    verts, closed=True,
                    facecolor=atom_color, edgecolor="none", alpha=alpha,
                    zorder=Z_COLOR_CODE,
                ))

    for ri in range(n - 1):
        for ci in range(n - 1):
            cx, cy = xy(
                x_lo + (ri + 0.5) * pitch, y_lo + (ci + 0.5) * pitch,
            )
            verts = [(cx + dx, cy + dy) for dx, dy in offsets]
            ax.add_patch(Polygon(
                verts, closed=True,
                facecolor=plaquette_color, edgecolor="none", alpha=alpha,
                zorder=Z_COLOR_CODE,
            ))
