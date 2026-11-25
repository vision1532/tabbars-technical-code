"""Public F-MRS pipeline; frozen C0 modules are imported, never modified."""

from __future__ import annotations

import time

from methods.mrs.adjudicator import MRSRuntime, final_answer, load_runtime
from methods.mrs.budget import require_feasible, table_budget
from methods.mrs.proposal import lexical_proposal, parse_for_proposal

from .exact_budget_search import search
from .factor_graph import build_factor_graph


def run_fmrs(example: dict, budget: float, *, max_qwen_candidate_evals: int,
             runtime: MRSRuntime | None = None,
             model_path: str = "Qwen/Qwen3-8B") -> dict:
    if budget != .05:
        raise ValueError("this frozen F-MRS evaluation is defined only at B=.05")
    if runtime is None:
        runtime = load_runtime(model_path)
    safe = {key: example[key] for key in ("table", "question_text", "options", "instruction")}
    safe["meta"] = {}
    started = time.perf_counter()
    full_tokens, budget_tokens = table_budget(runtime.tokenizer, safe["table"], budget)
    prefix, sections, parsed_tokens = parse_for_proposal(safe, runtime.tokenizer)
    if parsed_tokens != full_tokens:
        raise RuntimeError("F-MRS parser/full-table token disagreement")
    proposal = lexical_proposal(safe["question_text"], safe["options"], sections)
    factor = build_factor_graph(sections)
    # The frozen C0 reachability mode uses lexical anchors rather than mixed
    # row relevance. This affects only the inherited J0 tie-break heuristic;
    # relation atoms and the factual frontier remain question-independent.
    for section in sections:
        for row in section["rows"]:
            row["combined_score"] = float(row["question_anchor_score"])
    found = search(safe, tokenizer=runtime.tokenizer, llm=runtime.llm,
                   sampling=runtime.sampling, prefix=prefix, sections=sections,
                   factor_graph=factor, budget_tokens=budget_tokens,
                   max_qwen_candidate_evals=max_qwen_candidate_evals)
    selected = found["selected_state"]
    context = selected.context if selected else ""
    final_tokens = require_feasible(runtime.tokenizer, context, budget_tokens)
    selected_cross_section_transition = any(
        factor["row_section"][step["parent_row"]] !=
        factor["row_section"][step["added_row"]]
        for step in found["selected_provenance"]
        if step["parent_row"] is not None)
    if selected is None:
        answer = {"prediction": ""}
        final_qa_latency = 0.
    else:
        answer_started = time.perf_counter()
        answer = final_answer(runtime.answerer, safe, context)
        final_qa_latency = time.perf_counter() - answer_started
    return {"method": "Factorized Minimal Relational Search",
            "prediction": answer.get("prediction", ""),
            "candidate_prediction": found["selected_candidate_prediction"],
            "selected_context": context,
            "selected_rows": list(selected.rows) if selected else [],
            "full_table_tokens": full_tokens,
            "budget_tokens": budget_tokens,
            "final_tokens": final_tokens,
            "rho": final_tokens / full_tokens if full_tokens else 0.,
            "proposal_seed_rows": proposal["seed_row_ids"],
            "relation_atom_count": factor["stats"]["relation_atoms"],
            "incidence_edge_count": factor["stats"]["incidence_edges"],
            "evaluated_candidates": found["evaluated"],
            "max_qwen_candidate_evals": max_qwen_candidate_evals,
            "num_qwen_evaluated_candidates": found["trace"]["candidate_contexts_sent"],
            "qwen_batches": found["trace"]["qwen_candidate_requests"],
            "candidate_batch_sizes": found["trace"]["candidate_batch_sizes"],
            "candidate_input_tokens": found["trace"]["candidate_input_tokens"],
            "candidate_output_tokens": found["trace"]["candidate_output_tokens"],
            "candidate_qwen_wall_sec": found["trace"]["candidate_qwen_wall_sec"],
            "raw_successors_generated": found["counters"]["raw_successors_generated"],
            "initial_candidates_generated": found["counters"]["initial_candidates_generated"],
            "generated_candidates": found["counters"]["initial_candidates_generated"] +
                                    found["counters"]["raw_successors_generated"],
            "frontier_expansions": found["counters"]["frontier_expansions"],
            "duplicate_rowsets_removed": found["counters"]["duplicate_rowsets_removed"],
            "duplicate_serializations_removed": found["counters"]["duplicate_serializations_removed"],
            "budget_rejected_states": found["counters"]["budget_rejected_states"],
            "unique_feasible_states": found["counters"]["unique_feasible_states"],
            "search_depth_reached": found["search_depth_reached"],
            "termination_reason": found["termination_reason"],
            "selected_structural_provenance": found["selected_provenance"],
            "selected_cross_section_transition": selected_cross_section_transition,
            "pruning_latency_sec": found["search_latency_sec"],
            "final_qa_latency_sec": final_qa_latency,
            "total_latency_sec": time.perf_counter() - started,
            "final_budget_valid": final_tokens <= budget_tokens,
            "candidate_budget_violations": 0}
