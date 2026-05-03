def plan_aod_rearrangement(
    sites: list[int],
    src: list[int],
    order: list[int],
) -> list[tuple[list[int], list[int]]]:
    """
    Return only raw AOD-native moves.

    sites:
        All available integer trap sites.

    src:
        Initial occupied atom positions.

    order:
        Desired final left-to-right order, relative to the initial
        left-to-right order.

        Example:
            src = [0, 1, 2, 3, 4, 5]
            order = [5, 4, 3, 2, 1, 0]

        means reverse the initial ordering.

    Output:
        moves = [
            (from_sites, to_sites),
            ...
        ]

    No atom labels are stored in the output.
    """
    sites = sorted(sites)
    src = sorted(src)
    n = len(src)

    if len(set(sites)) != len(sites):
        raise ValueError("sites contains duplicates.")

    if len(set(src)) != len(src):
        raise ValueError("src contains duplicates.")

    if not set(src).issubset(set(sites)):
        raise ValueError("src must be a subset of sites.")

    if sorted(order) != list(range(n)):
        raise ValueError(f"order must be a permutation of 0..{n - 1}.")

    # Hard-coded trap requirement:
    # N atoms + floor(N/2) empty buffer traps.
    min_sites = n + n // 2

    if len(sites) < min_sites:
        raise ValueError(f"Need at least {min_sites} sites for {n} atoms.")

    # Internal IDs only. ID i means the i-th atom in the initial ordering.
    # These IDs are NOT returned in the move program.
    pos = {i: src[i] for i in range(n)}

    moves: list[tuple[list[int], list[int]]] = []

    def add_move(ids: list[int], dst: list[int]) -> None:
        """
        Add one AOD-native move.

        Internally, ids are only used to know which occupied sites are moved.
        The stored output is only (from_sites, to_sites).
        """
        if not ids:
            return

        frm = [pos[i] for i in ids]

        if frm != sorted(frm):
            raise RuntimeError(
                f"Selected atoms are not left-to-right: ids={ids}, sites={frm}"
            )

        if dst != sorted(dst):
            raise RuntimeError(f"Target sites are not left-to-right: {dst}")

        unmoved_sites = {
            p for i, p in pos.items()
            if i not in ids
        }

        if any(t in unmoved_sites for t in dst):
            raise RuntimeError(
                f"Move collides with unmoved atoms: {frm} -> {dst}"
            )

        if frm == dst:
            return

        moves.append((frm[:], dst[:]))

        for i, t in zip(ids, dst):
            pos[i] = t

    # ------------------------------------------------------------
    # Initial normalization:
    # move all atoms to the rightmost N traps.
    # This creates the left buffer.
    # ------------------------------------------------------------
    ids0 = list(range(n))
    add_move(ids0, sites[-n:])

    # A block is:
    #
    #     (current_ids, target_ids, workspace_sites)
    #
    # current_ids are in their current left-to-right order.
    # target_ids are their desired final left-to-right order.
    blocks = [
        (ids0, list(order), sites)
    ]

    while any(len(ids) > 1 for ids, _, _ in blocks):
        split_blocks = []

        # ========================================================
        # Layer 1:
        # Move final-left-half atoms into the left buffer.
        # ========================================================
        layer_ids = []
        layer_dst = []

        for ids, target, workspace in blocks:
            m = len(ids)

            if m <= 1:
                split_blocks.append((ids, target, workspace, None))
                continue

            m_left = m // 2

            rank = {
                i: r
                for r, i in enumerate(target)
            }

            left_ids = [
                i for i in ids
                if rank[i] < m_left
            ]

            right_ids = [
                i for i in ids
                if rank[i] >= m_left
            ]

            layer_ids += left_ids
            layer_dst += workspace[:m_left]

            split_blocks.append(
                (ids, target, workspace, (left_ids, right_ids))
            )

        add_move(layer_ids, layer_dst)

        # ========================================================
        # Layer 2:
        # Compact both halves into recursive workspaces.
        # ========================================================
        next_blocks = []

        layer_ids = []
        layer_dst = []

        for ids, target, workspace, split in split_blocks:
            if split is None:
                next_blocks.append((ids, target, workspace))
                continue

            left_ids, right_ids = split

            n_left = len(left_ids)
            n_right = len(right_ids)

            # Hard-coded child workspace sizes:
            # n + floor(n/2)
            left_size = n_left + n_left // 2
            right_size = n_right + n_right // 2

            left_workspace = workspace[:left_size]
            right_workspace = workspace[
                left_size:
                left_size + right_size
            ]

            left_dst = left_workspace[-n_left:]
            right_dst = right_workspace[-n_right:]

            layer_ids += left_ids + right_ids
            layer_dst += left_dst + right_dst

            next_blocks.append(
                (
                    left_ids,
                    target[:n_left],
                    left_workspace,
                )
            )

            next_blocks.append(
                (
                    right_ids,
                    target[n_left:],
                    right_workspace,
                )
            )

        add_move(layer_ids, layer_dst)

        blocks = next_blocks

    final_ids = [
        i for i, _ in sorted(pos.items(), key=lambda item: item[1])
    ]

    assert final_ids == list(order), (final_ids, order)

    return moves


def apply_move_history(
    sites: list[int],
    src: list[int],
    moves: list[tuple[list[int], list[int]]],
    labels: list[int] | None = None,
) -> list[dict[int, int]]:
    """
    Replay raw moves and return the atom configuration after every step.

    labels:
        Optional atom labels used only for display.

        If labels is None, atoms are labeled 0, 1, ..., N-1 by initial order.

    Output:
        history[k] is a dictionary:

            site -> atom_label

        after step k.
    """
    sites = sorted(sites)
    src = sorted(src)
    n = len(src)

    if labels is None:
        labels = list(range(n))

    if len(labels) != n:
        raise ValueError("labels must have the same length as src.")

    state = {
        site: label
        for site, label in zip(src, labels)
    }

    history = [state.copy()]

    for frm, dst in moves:
        if frm != sorted(frm):
            raise ValueError(f"Move source sites are not ordered: {frm}")

        if dst != sorted(dst):
            raise ValueError(f"Move target sites are not ordered: {dst}")

        if len(frm) != len(dst):
            raise ValueError(f"Move length mismatch: {frm} -> {dst}")

        if any(site not in state for site in frm):
            raise ValueError(f"Trying to move from an empty site: {frm}")

        carried_atoms = [
            state[site]
            for site in frm
        ]

        for site in frm:
            del state[site]

        if any(site in state for site in dst):
            raise ValueError(f"Move collides with unmoved atoms: {dst}")

        for site, atom in zip(dst, carried_atoms):
            state[site] = atom

        history.append(state.copy())

    return history


def draw_state(state: dict[int, int], sites: list[int]) -> str:
    width = max(
        1,
        max((len(str(atom)) for atom in state.values()), default=1),
    )

    return " ".join(
        f"{str(state[site]) if site in state else '.':>{width}}"
        for site in sorted(sites)
    )


def print_moves(moves: list[tuple[list[int], list[int]]]) -> None:
    for step, move in enumerate(moves):
        frm, dst = move
        print(f"{step:02d}: {frm} -> {dst}")


def print_history(
    history: list[dict[int, int]],
    sites: list[int],
) -> None:
    for step, state in enumerate(history):
        print(f"{step:02d}: {draw_state(state, sites)}")


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

    moves = plan_aod_rearrangement(sites, src, order)

    print("Raw move program:")
    print_moves(moves)

    print()
    print("Interpreted configurations:")
    history = apply_move_history(sites, src, moves, labels=labels)
    print_history(history, sites)

    final_state = history[-1]
    final_labels = [
        final_state[site]
        for site in sorted(final_state)
    ]

    print()


if __name__ == "__main__":
    # ------------------------------------------------------------
    # Example 1:
    # Six atoms initially occupying sites 0..5.
    # Reverse the initial order.
    # ------------------------------------------------------------
    run_test(
        name="Example 1: reverse six atoms",
        sites=list(range(9)),
        src=[0, 1, 2, 3, 4, 5],
        order=[5, 4, 3, 2, 1, 0],
    )

    # ------------------------------------------------------------
    # Example 2:
    # Six atoms start at sparse integer positions.
    # Reverse their initial left-to-right order.
    # ------------------------------------------------------------
    run_test(
        name="Example 2: sparse initial positions, reverse",
        sites=list(range(12)),
        src=[0, 2, 5, 6, 8, 10],
        order=[5, 4, 3, 2, 1, 0],
    )

    # ------------------------------------------------------------
    # Example 3:
    # Physical labels initially appear as:
    #
    #     3 0 5 1 4 2
    #
    # We want:
    #
    #     0 1 2 3 4 5
    #
    # The planner does not know physical labels.
    # It only sees the desired order relative to initial indices.
    # ------------------------------------------------------------
    initial_physical_labels = [3, 0, 5, 1, 4, 2]
    desired_physical_labels = [0, 1, 2, 3, 4, 5]

    order = [
        initial_physical_labels.index(label)
        for label in desired_physical_labels
    ]

    run_test(
        name="Example 3: random physical order to sorted physical order",
        sites=list(range(9)),
        src=[0, 1, 2, 3, 4, 5],
        order=order,
        labels=initial_physical_labels,
    )