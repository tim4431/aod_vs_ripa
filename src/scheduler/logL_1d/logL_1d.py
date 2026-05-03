"""
1D log(L)-depth AOD rearrangement planner.

Adapted from Wang et al., Nature Physics (2024),
https://www.nature.com/articles/s41567-024-02479-z.

Note: the pseudocode given in that paper does not actually work as
written. This is a corrected re-implementation of the divide-and-conquer
scheme.
"""


def aod_logL_1d_rearrangement(
    sites: list[int],
    src: list[int],
    order: list[int],
) -> list[tuple[list[int], list[int]]]:
    """
    Return raw AOD-native moves (no atom labels in output).

    sites:  all available integer trap sites.
    src:    initial occupied atom positions.
    order:  desired final left-to-right order, relative to the initial
            left-to-right order. E.g. src=[0,1,2,3,4,5], order=[5,4,3,2,1,0]
            reverses the initial ordering.

    Output: moves = [(from_sites, to_sites), ...]
    """
    sites = sorted(sites)
    src = sorted(src)
    n = len(src)

    if len(set(sites)) != len(sites):
        raise ValueError("sites contains duplicates.")
    if len(set(src)) != len(src):
        raise ValueError("src contains duplicates.")
    if not set(src).issubset(sites):
        raise ValueError("src must be a subset of sites.")
    if sorted(order) != list(range(n)):
        raise ValueError(f"order must be a permutation of 0..{n - 1}.")
    if len(sites) < (3*n)//2:
        raise ValueError(f"Need at least {(3*n)//2} sites for {n} atoms.")

    # Internal IDs only: ID i means the i-th atom in the initial ordering.
    pos = {i: src[i] for i in range(n)}
    moves: list[tuple[list[int], list[int]]] = []

    def add_move(ids: list[int], dst: list[int]) -> None:
        if not ids:
            return
        frm = [pos[i] for i in ids]
        if frm == dst:
            return
        moves.append((frm[:], dst[:]))
        for i, t in zip(ids, dst):
            pos[i] = t

    # Normalize: pack all atoms into the rightmost N traps to create a left buffer.
    ids0 = list(range(n))
    add_move(ids0, sites[-n:])

    # A block is (current_ids in current L-R order,
    #             target_ids in final L-R order,
    #             workspace_sites).
    blocks = [(ids0, list(order), sites)]

    while any(len(ids) > 1 for ids, _, _ in blocks):
        # Layer 1: move final-left-half atoms into the left buffer.
        split_blocks = []
        layer_ids: list[int] = []
        layer_dst: list[int] = []
        for ids, target, workspace in blocks:
            m = len(ids)
            if m <= 1:
                split_blocks.append((ids, target, workspace, None))
                continue
            m_left = m // 2
            rank = {i: r for r, i in enumerate(target)}
            left_ids = [i for i in ids if rank[i] < m_left]
            right_ids = [i for i in ids if rank[i] >= m_left]
            layer_ids += left_ids
            layer_dst += workspace[:m_left]
            split_blocks.append((ids, target, workspace, (left_ids, right_ids)))
        add_move(layer_ids, layer_dst)

        # Layer 2: compact each half into the right end of its sub-workspace.
        next_blocks = []
        layer_ids = []
        layer_dst = []
        for ids, target, workspace, split in split_blocks:
            if split is None:
                next_blocks.append((ids, target, workspace))
                continue
            left_ids, right_ids = split
            n_left, n_right = len(left_ids), len(right_ids)
            left_size = (3* n_left) // 2
            right_size = (3* n_right) // 2
            left_workspace = workspace[:left_size]
            right_workspace = workspace[left_size:left_size + right_size]
            layer_ids += left_ids + right_ids
            layer_dst += left_workspace[-n_left:] + right_workspace[-n_right:]
            next_blocks.append((left_ids, target[:n_left], left_workspace))
            next_blocks.append((right_ids, target[n_left:], right_workspace))
        add_move(layer_ids, layer_dst)

        blocks = next_blocks

    final_ids = [i for i, _ in sorted(pos.items(), key=lambda kv: kv[1])]
    assert final_ids == list(order), (final_ids, order)

    return moves


def apply_move_history(
    src: list[int],
    moves: list[tuple[list[int], list[int]]],
    labels: list[int] | None = None,
) -> list[dict[int, int]]:
    """
    Replay raw moves and return the atom configuration after every step.

    history[k] is a {site -> atom_label} dict after step k. If labels is None,
    atoms are labeled 0..N-1 by initial order.
    """
    src = sorted(src)
    if labels is None:
        labels = list(range(len(src)))
    if len(labels) != len(src):
        raise ValueError("labels must have the same length as src.")

    state = dict(zip(src, labels))
    history = [state.copy()]
    for frm, dst in moves:
        carried = [state.pop(site) for site in frm]
        for site, atom in zip(dst, carried):
            state[site] = atom
        history.append(state.copy())
    return history


def run_test(
    name: str,
    sites: list[int],
    src: list[int],
    order: list[int],
    labels: list[int] | None = None,
) -> None:
    print("=" * 72)
    print(name)
    print("=" * 72)

    moves = aod_logL_1d_rearrangement(sites, src, order)
    print("Raw move program:")
    for step, (frm, dst) in enumerate(moves):
        print(f"{step:02d}: {frm} -> {dst}")

    history = apply_move_history(src, moves, labels=labels)
    width = max(
        (len(str(a)) for state in history for a in state.values()),
        default=1,
    )
    print("\nInterpreted configurations:")
    for step, state in enumerate(history):
        row = " ".join(
            f"{str(state[s]) if s in state else '.':>{width}}"
            for s in sorted(sites)
        )
        print(f"{step:02d}: {row}")
    print()


if __name__ == "__main__":
    run_test(
        name="Example 1: reverse six atoms",
        sites=list(range(9)),
        src=[0, 1, 2, 3, 4, 5],
        order=[5, 4, 3, 2, 1, 0],
    )

    run_test(
        name="Example 2: sparse initial positions, reverse",
        sites=list(range(12)),
        src=[0, 2, 5, 6, 8, 10],
        order=[5, 4, 3, 2, 1, 0],
    )

    # Physical labels appear as 3 0 5 1 4 2; we want 0 1 2 3 4 5.
    initial = [3, 0, 5, 1, 4, 2]
    desired = [0, 1, 2, 3, 4, 5]
    run_test(
        name="Example 3: random physical order to sorted physical order",
        sites=list(range(9)),
        src=[0, 1, 2, 3, 4, 5],
        order=[initial.index(label) for label in desired],
        labels=initial,
    )
