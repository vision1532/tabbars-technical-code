"""Parser-equivalent factual relation atoms, without pairwise materialization.

The frozen MRS graph builder has no column-compatibility predicate: after the
v6 parser's value-frequency eligibility filter, any two rows containing the
same canonical value are adjacent. Column names are provenance only. Thus its
actual compatible-column-pair signature is the universal eligible pair; adding
a stricter schema signature would *remove* old edges, and a looser value rule
would introduce new edges.
"""

from __future__ import annotations

from collections import defaultdict

from run_longtablebench_search_oracle import normalize_cell_value, split_markdown_row

UNIVERSAL_COLUMN_PAIR = "parser_eligible:any_column_pair"


def relation_atoms(sections: list[dict]) -> tuple[dict[int, dict], dict[int, int]]:
    by_value: dict[str, dict[int, set[str]]] = defaultdict(lambda: defaultdict(set))
    row_section: dict[int, int] = {}
    for section in sections:
        headers = split_markdown_row(section.get("header_line", ""))
        for row in section["rows"]:
            rid = int(row["row_id"])
            row_section[rid] = int(section["section_idx"])
            allowed = set(row.get("connector_values", []))
            for index, cell in enumerate(split_markdown_row(row["line"])):
                value = normalize_cell_value(cell)
                if value in allowed:
                    column = headers[index] if index < len(headers) else f"column_{index}"
                    by_value[value][rid].add(column)
    atoms = {}
    for atom_id, (value, by_row) in enumerate(sorted(by_value.items())):
        if len(by_row) < 2:
            # The original builder only creates an edge for shared values.
            continue
        atoms[atom_id] = {"atom_id": atom_id,
                          "column_pair_signature": UNIVERSAL_COLUMN_PAIR,
                          "canonical_value": value,
                          "rows": tuple(sorted(by_row)),
                          "source_columns": {rid: tuple(sorted(names))
                                             for rid, names in by_row.items()}}
    return atoms, row_section
