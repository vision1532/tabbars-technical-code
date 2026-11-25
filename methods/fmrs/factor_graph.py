"""Bipartite row--relation incidence graph with deterministic frontiers."""

from __future__ import annotations

from collections import defaultdict

from .relation_atoms import relation_atoms


def build_factor_graph(sections: list[dict]) -> dict:
    atoms, row_section = relation_atoms(sections)
    row_atoms: dict[int, list[int]] = {rid: [] for rid in row_section}
    incidence_edges = 0
    pairwise_witness_edges = 0
    for atom_id, atom in atoms.items():
        rows = atom["rows"]
        incidence_edges += len(rows)
        pairwise_witness_edges += len(rows) * (len(rows) - 1) // 2
        for rid in rows:
            row_atoms[rid].append(atom_id)
    return {"atoms": atoms, "row_atoms": {rid: tuple(ids) for rid, ids in row_atoms.items()},
            "row_section": row_section,
            "stats": {"rows": len(row_section), "relation_atoms": len(atoms),
                      "incidence_edges": incidence_edges,
                      "pairwise_witness_edges": pairwise_witness_edges}}


def projected_neighbors(graph: dict, rid: int) -> set[int]:
    neighbors = set()
    for atom_id in graph["row_atoms"].get(int(rid), ()):
        neighbors.update(graph["atoms"][atom_id]["rows"])
    neighbors.discard(int(rid))
    return neighbors


def frontier_with_witness(graph: dict, selected: tuple[int, ...]) -> list[tuple[int, int, int]]:
    """One real, lexicographically first (parent, atom, new-row) per frontier row."""
    selected_set = set(selected)
    first: dict[int, tuple[int, int, int]] = {}
    for parent in sorted(selected_set):
        for atom_id in graph["row_atoms"].get(parent, ()):
            for child in graph["atoms"][atom_id]["rows"]:
                if child not in selected_set:
                    first.setdefault(child, (parent, atom_id, child))
    return [first[child] for child in sorted(first)]
