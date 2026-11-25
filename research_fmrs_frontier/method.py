"""Specificity-first ordering at the frozen F-MRS frontier truncation point.

Only the order/witness of already reachable one-row successors changes.  The
factor graph, initial candidates, exact budget, J0, beam rule, evaluator cap,
and final QA remain the frozen F-MRS implementation.
"""

from __future__ import annotations

from unittest.mock import patch

import methods.fmrs.exact_budget_search as frozen_search
from methods.fmrs.factor_graph import frontier_with_witness as original_frontier
from methods.fmrs.pipeline import run_fmrs


def specificity_frontier(graph: dict, selected: tuple[int, ...]) -> list[tuple[int, int, int]]:
    """Rank legal next rows by the smallest relation-atom degree witnessing them.

    Select each child's least-generic real witness deterministically.  An atom
    with fewer member rows has higher specificity; ties use original row IDs.
    No question, qtype, gold, candidate score, or LLM output enters this key.
    """
    selected_set = set(selected)
    best: dict[int, tuple[int, int, int, int]] = {}
    for parent in sorted(selected_set):
        for atom_id in graph["row_atoms"].get(parent, ()):
            atom = graph["atoms"][atom_id]
            degree = len(atom["rows"])
            for child in atom["rows"]:
                if child in selected_set:
                    continue
                witness = (degree, parent, atom_id, child)
                if child not in best or witness < best[child]:
                    best[child] = witness
    return [(parent, atom_id, child)
            for _, parent, atom_id, child in sorted(best.values())]


def run_variant(example: dict, budget: float, *, max_qwen_candidate_evals: int,
                runtime) -> dict:
    with patch.object(frozen_search, "frontier_with_witness", specificity_frontier):
        result = run_fmrs(example, budget,
                          max_qwen_candidate_evals=max_qwen_candidate_evals,
                          runtime=runtime)
    result["method"] = "F-MRS specificity-first frontier (DEV-only)"
    return result


def validate_frontier(graph: dict, selected: tuple[int, ...]) -> None:
    """Assert exact successor-set preservation and factual witness validity."""
    old = original_frontier(graph, selected)
    new = specificity_frontier(graph, selected)
    if {x[2] for x in old} != {x[2] for x in new} or len(new) != len(old):
        raise RuntimeError("frontier successor set changed")
    for parent, atom_id, child in new:
        members = graph["atoms"][atom_id]["rows"]
        if parent not in selected or child in selected or parent not in members or child not in members:
            raise RuntimeError("invalid factual frontier witness")
