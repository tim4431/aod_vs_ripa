from dataclasses import dataclass
from typing import Hashable


Atom = Hashable
Site = int


@dataclass
class Block:
    """
    One active recursive workspace.

    atoms:
        Current left-to-right atom order inside this workspace.

    target:
        Desired final left-to-right atom order for these atoms.

    sites:
        Integer trap sites assigned to this workspace.
    """
    atoms: list[Atom]
    target: list[Atom]
    sites: list[Site]


def draw_state(atom_position: dict[Atom, Site], sites: list[Site]) -> str:
    """Print atoms on integer traps, using '.' for empty traps."""
    atom_at_site = {p: a for a, p in atom_position.items()}
    width = max(1, max(len(str(a)) for a in atom_position))

    return " ".join(
        f"{str(atom_at_site[p]) if p in atom_at_site else '.':>{width}}"
        for p in sites
    )


def aod_native_move(
    atom_position: dict[Atom, Site],
    moving_atoms: list[Atom],
    target_sites: list[Site],
) -> bool:
    """
    Perform one AOD-native stretch/move.

    Constraint:
        The moving atoms must be selected in their current left-to-right order.
        The target sites must also be in left-to-right order.

    Therefore this move can translate/stretch/compress atoms,
    but it cannot permute their relative order.
    """
    if not moving_atoms:
        return False

    current_sites = [atom_position[a] for a in moving_atoms]

    if current_sites != sorted(current_sites):
        raise ValueError(
            f"Moving atoms are not selected left-to-right:\n"
            f"atoms = {moving_atoms}\n"
            f"sites = {current_sites}"
        )

    if target_sites != sorted(target_sites):
        raise ValueError(
            f"Target sites are not left-to-right ordered:\n"
            f"targets = {target_sites}"
        )

    unmoved_sites = {
        p for a, p in atom_position.items()
        if a not in moving_atoms
    }

    if any(p in unmoved_sites for p in target_sites):
        raise ValueError("AOD move collides with atoms that are not moving.")

    changed = any(atom_position[a] != p for a, p in zip(moving_atoms, target_sites))

    for atom, site in zip(moving_atoms, target_sites):
        atom_position[atom] = site

    return changed


def rearrange_1d_aod(
    start_order,
    final_order,
    *,
    sites=None,
    atom_position=None,
):
    """
    Arbitrary 1D AOD atom rearrangement by divide and conquer.

    Parameters
    ----------
    start_order:
        Initial left-to-right atom ordering.
        Example: "012345"

    final_order:
        Desired final left-to-right atom ordering.
        Example: "543210"

    sites:
        Available integer trap sites.
        If omitted, uses range(floor(3N/2)).

    atom_position:
        Optional explicit initial atom positions.
        Example: {"0": 0, "1": 2, "2": 5}

    Returns
    -------
    history:
        List of atom-position dictionaries: the initial state followed
        by one snapshot after each AOD-native move that changed positions.
    sites:
        The trap sites used (useful for rendering the history).
    """
    start_order = list(start_order)
    final_order = list(final_order)
    n_atoms = len(start_order)

    if set(start_order) != set(final_order):
        raise ValueError("start_order and final_order must contain the same atoms.")

    if len(set(start_order)) != n_atoms:
        raise ValueError("Atoms must have unique labels.")

    # Allocate traps: N occupied + floor(N/2) empty buffer.
    min_sites = (3 * n_atoms) // 2

    if sites is None:
        sites = list(range(min_sites))
    else:
        sites = list(sites)

    if len(sites) < min_sites:
        raise ValueError(
            f"Need at least {min_sites} traps for {n_atoms} atoms, "
            f"but only got {len(sites)}."
        )

    # Initialize atom positions.
    if atom_position is None:
        atom_position = {
            atom: sites[i]
            for i, atom in enumerate(start_order)
        }
    else:
        atom_position = dict(atom_position)
        start_order = sorted(start_order, key=lambda a: atom_position[a])

    history = [dict(atom_position)]

    def apply_layer(
        moving_atoms: list[Atom],
        target_sites: list[Site],
    ):
        """Apply one global AOD-native layer; record snapshot if it changed anything."""
        if aod_native_move(atom_position, moving_atoms, target_sites):
            history.append(dict(atom_position))

    # ------------------------------------------------------------
    # Initial normalization:
    #
    # Put all atoms into the rightmost N traps.
    #
    # This creates an empty buffer of size floor(N/2) on the left.
    # After this, every active block satisfies the invariant:
    #
    #     atoms occupy the rightmost n traps of their workspace
    #
    # ------------------------------------------------------------
    apply_layer(
        moving_atoms=start_order,
        target_sites=sites[-n_atoms:],
    )

    active_blocks = [
        Block(
            atoms=start_order,
            target=final_order,
            sites=sites,
        )
    ]

    # ------------------------------------------------------------
    # Main divide-and-conquer loop.
    #
    # Each round performs two native AOD layers:
    #
    #   Layer 1:
    #       Move atoms whose final rank belongs to the left half
    #       into the left buffer.
    #
    #   Layer 2:
    #       Compact the left-half atoms and right-half atoms into
    #       two smaller recursive workspaces.
    #
    # ------------------------------------------------------------
    while any(len(block.atoms) > 1 for block in active_blocks):
        split_blocks = []

        # ========================================================
        # Layer 1:
        # Move final-left-half atoms into the left buffer.
        # ========================================================
        layer_atoms = []
        layer_targets = []

        for block in active_blocks:
            n = len(block.atoms)

            if n <= 1:
                split_blocks.append((block, None))
                continue

            n_left = n // 2

            final_rank = {
                atom: rank
                for rank, atom in enumerate(block.target)
            }

            left_atoms = [
                atom for atom in block.atoms
                if final_rank[atom] < n_left
            ]

            right_atoms = [
                atom for atom in block.atoms
                if final_rank[atom] >= n_left
            ]

            left_buffer_sites = block.sites[:n_left]

            layer_atoms += left_atoms
            layer_targets += left_buffer_sites

            split_blocks.append((block, (left_atoms, right_atoms)))

        apply_layer(
            moving_atoms=layer_atoms,
            target_sites=layer_targets,
        )

        # ========================================================
        # Layer 2:
        # Compact both halves into their recursive workspaces.
        # ========================================================
        next_active_blocks = []

        layer_atoms = []
        layer_targets = []

        for block, split in split_blocks:
            if split is None:
                next_active_blocks.append(block)
                continue

            left_atoms, right_atoms = split

            n_left = len(left_atoms)
            n_right = len(right_atoms)

            left_workspace_size = (3 * n_left) // 2
            right_workspace_size = (3 * n_right) // 2

            left_workspace = block.sites[:left_workspace_size]
            right_workspace = block.sites[
                left_workspace_size:
                left_workspace_size + right_workspace_size
            ]

            # Preserve the invariant:
            # atoms occupy the rightmost traps of their workspace.
            left_atom_sites = left_workspace[-n_left:]
            right_atom_sites = right_workspace[-n_right:]

            layer_atoms += left_atoms + right_atoms
            layer_targets += left_atom_sites + right_atom_sites

            next_active_blocks.append(
                Block(
                    atoms=left_atoms,
                    target=block.target[:n_left],
                    sites=left_workspace,
                )
            )

            next_active_blocks.append(
                Block(
                    atoms=right_atoms,
                    target=block.target[n_left:],
                    sites=right_workspace,
                )
            )

        apply_layer(
            moving_atoms=layer_atoms,
            target_sites=layer_targets,
        )

        active_blocks = next_active_blocks

    # Check final left-to-right ordering.
    final_seen = [
        atom for atom, _ in sorted(
            atom_position.items(),
            key=lambda item: item[1],
        )
    ]

    assert final_seen == final_order, (final_seen, final_order)

    return history, sites


def print_history(history, sites):
    for step, atom_position in enumerate(history):
        print(f"{step:02d}: {draw_state(atom_position, sites)}")


if __name__ == "__main__":
    print("Reverse:")
    print_history(*rearrange_1d_aod("012345", "543210"))

    print("\nRandom to sorted:")
    print_history(*rearrange_1d_aod("305142", "012345"))