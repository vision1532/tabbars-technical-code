"""Compute-matched, exact-budget relational state search over factor incidences."""

from __future__ import annotations

import time

import run_merged_jsonl_graph_search_v6 as v6
from research_v6_localization.run_v6_ablation import CountingLLM, candidate_score_without_graph

from methods.mrs.adjudicator import evaluate_frozen_candidates
from methods.mrs.budget import exact_tokens
from methods.mrs.candidate_search import search_candidates

from .factor_graph import frontier_with_witness
from .search_state import SearchState, context_hash, row_set_hash

BEAM_WIDTH = 4  # The frozen C0 beam width.
STAGE_LIMIT = 24  # Frozen C0 expansion rank limit: max(beam*2, expand_topk*3).


def _initial_candidates(example: dict, tokenizer) -> list[dict]:
    """Stop the actual C0 initializer at its first (deterministic) eval stage.

    The stage judge never sees Qwen or gold. Its only role is to stop the C0
    adapter immediately after the original seed construction and exact gate.
    """
    def stop_after_initial_stage(*_args):
        return {"selected_candidate": 1, "sufficient": True, "parse_valid": True}

    search = search_candidates(example, .05, tokenizer, None, None,
                               adjudication="deterministic", construction_mode="C0",
                               stage_judge=stop_after_initial_stage)
    stages = [stage for stage in search["trace"]["stages"]
              if stage["stage"].startswith("candidate_eval_")]
    if len(stages) > 1:
        raise RuntimeError("C0 initializer unexpectedly passed its first eval stage")
    return stages[0]["candidates"] if stages else []


def _new_state(rows: tuple[int, ...], prefix: str, sections: list[dict], tokenizer,
               row_index: dict, section_index: dict, connector_index: dict,
               budget_tokens: int, *, parent: SearchState | None = None,
               witness: tuple[int, int, int] | None = None,
               initial_heuristic: float | None = None,
               initial_estimated: int | None = None) -> SearchState:
    context = v6.render_pruned_table(prefix, sections, set(rows))
    cost = exact_tokens(tokenizer, context)
    if cost > budget_tokens:
        raise ValueError("infeasible state passed to _new_state")
    fixed = v6.build_section_fixed_cost(sections)
    estimated = (initial_estimated if initial_estimated is not None else
                 v6.estimate_row_set_token_cost(set(rows), row_index, fixed))
    heuristic = (initial_heuristic if initial_heuristic is not None else
                 candidate_score_without_graph(set(rows), row_index, section_index,
                                               connector_index, .05, budget_tokens,
                                               token_cost=estimated,
                                               keep_section=False, keep_utilization=False))
    return SearchState(rows=rows, context=context, exact_tokens=cost,
                       estimated_tokens=estimated, heuristic_score=float(heuristic),
                       state_hash=row_set_hash(rows),
                       serialized_context_hash=context_hash(context),
                       parent_state_hash=parent.state_hash if parent else None,
                       added_row=witness[2] if witness else None,
                       transition_relation_atom=witness[1] if witness else None,
                       parent_row=witness[0] if witness else None,
                       depth=parent.depth + 1 if parent else 0)


def _rank_key(candidate) -> tuple:
    return v6.candidate_sort_key(candidate, "blind")


def search(example: dict, *, tokenizer, llm, sampling, prefix: str,
           sections: list[dict], factor_graph: dict, budget_tokens: int,
           max_qwen_candidate_evals: int) -> dict:
    """No gold, qtype, semantic graph scores, or extra LLM beyond frozen J0."""
    if max_qwen_candidate_evals < 0:
        raise ValueError("negative C0 evaluator cap")
    started = time.perf_counter()
    trace = {"qwen_candidate_requests": 0, "candidate_batch_sizes": [],
             "candidate_contexts_sent": 0, "candidate_input_tokens": 0,
             "candidate_output_tokens": 0, "candidate_qwen_wall_sec": 0.0}
    row_index = v6.build_row_index(sections)
    section_index = v6.build_section_index(sections)
    connector_index = v6.build_connector_index(row_index)
    seen_rowsets: set[str] = set()
    seen_serializations: set[str] = set()
    states_by_hash: dict[str, SearchState] = {}
    evaluated: list[dict] = []
    counters = {"raw_successors_generated": 0, "frontier_expansions": 0,
                "initial_candidates_generated": 0,
                "duplicate_rowsets_removed": 0,
                "duplicate_serializations_removed": 0,
                "budget_rejected_states": 0,
                "unique_feasible_states": 0}
    initial = []
    initial_raw = _initial_candidates(example, tokenizer)
    counters["initial_candidates_generated"] = len(initial_raw)
    for item in initial_raw:
        rows = tuple(sorted(map(int, item["rows"])))
        context = v6.render_pruned_table(prefix, sections, set(rows))
        cost = exact_tokens(tokenizer, context)
        if cost > budget_tokens:
            raise RuntimeError("C0 initializer emitted an exact-over-budget candidate")
        row_hash = row_set_hash(rows)
        serialization_hash = context_hash(context)
        if row_hash in seen_rowsets:
            counters["duplicate_rowsets_removed"] += 1
            continue
        seen_rowsets.add(row_hash)
        if serialization_hash in seen_serializations:
            counters["duplicate_serializations_removed"] += 1
            continue
        seen_serializations.add(serialization_hash)
        state = _new_state(rows, prefix, sections, tokenizer, row_index,
                           section_index, connector_index, budget_tokens,
                           initial_heuristic=float(item["heuristic"]),
                           initial_estimated=int(item["estimated_tokens"]))
        states_by_hash[state.state_hash] = state
        initial.append(state)
        counters["unique_feasible_states"] += 1

    remaining = max_qwen_candidate_evals
    if remaining == 0:
        return {"selected_state": None, "selected_candidate_prediction": "",
                "evaluated": [], "trace": trace, "counters": counters,
                "search_depth_reached": 0, "termination_reason": "ZERO_EVALUATOR_QUOTA",
                "search_latency_sec": time.perf_counter() - started,
                "selected_provenance": []}

    counting_llm = CountingLLM(llm, tokenizer, trace)
    best: tuple[SearchState, object] | None = None
    beam: list[tuple[SearchState, object]] = []
    depth_reached = 0

    def evaluate(states: list[SearchState]) -> list[tuple[SearchState, object]]:
        if not states:
            return []
        # This second exact gate sits immediately at the unchanged J0 boundary.
        for state in states:
            rendered = v6.render_pruned_table(prefix, sections, set(state.rows))
            if exact_tokens(tokenizer, rendered) > budget_tokens:
                raise RuntimeError("candidate reached frozen Qwen above exact budget")
        candidates = evaluate_frozen_candidates(
            v6.evaluate_candidate_sets, counting_llm, sampling, example.get("instruction", ""),
            example.get("question_text", ""), example.get("options", ""),
            prefix, sections, [set(state.rows) for state in states],
            heuristic_scores=[state.heuristic_score for state in states],
            selection_mode="blind", completion=None)
        if len(candidates) != len(states):
            raise RuntimeError("J0 returned a mismatched candidate batch")
        for state, candidate in zip(states, candidates):
            evaluated.append({"rows": list(state.rows), "exact_tokens": state.exact_tokens,
                              "estimated_tokens": candidate.token_cost,
                              "heuristic_score": candidate.heuristic_score,
                              "j0_prediction": candidate.predicted,
                              "depth": state.depth,
                              "state_hash": state.state_hash,
                              "context_hash": state.serialized_context_hash,
                              "provenance": state.provenance()})
        return list(zip(states, candidates))

    first = evaluate(initial[:remaining])
    remaining -= len(first)
    if first:
        beam = sorted(first, key=lambda pair: _rank_key(pair[1]), reverse=True)[:BEAM_WIDTH]
        best = beam[0]
    if not beam:
        termination = "NO_FEASIBLE_INITIAL_STATE"
    else:
        termination = "EVALUATOR_CAP"
        while remaining > 0:
            successors: list[SearchState] = []
            for parent, _ in beam:
                for witness in frontier_with_witness(factor_graph, parent.rows):
                    counters["raw_successors_generated"] += 1
                    counters["frontier_expansions"] += 1
                    rows = tuple(sorted((*parent.rows, witness[2])))
                    row_hash = row_set_hash(rows)
                    if row_hash in seen_rowsets:
                        counters["duplicate_rowsets_removed"] += 1
                        continue
                    seen_rowsets.add(row_hash)
                    context = v6.render_pruned_table(prefix, sections, set(rows))
                    cost = exact_tokens(tokenizer, context)
                    if cost > budget_tokens:
                        counters["budget_rejected_states"] += 1
                        continue
                    serialization_hash = context_hash(context)
                    if serialization_hash in seen_serializations:
                        counters["duplicate_serializations_removed"] += 1
                        continue
                    seen_serializations.add(serialization_hash)
                    state = _new_state(rows, prefix, sections, tokenizer,
                                       row_index, section_index, connector_index,
                                       budget_tokens, parent=parent, witness=witness)
                    states_by_hash[state.state_hash] = state
                    successors.append(state)
                    counters["unique_feasible_states"] += 1
            if not successors:
                termination = "NO_NEW_FEASIBLE_STATE"
                break
            # Fixed structural order, inherited 24-candidate stage width, and
            # exact remaining C0 quota. No score/gold-dependent quota refill.
            batch = successors[:min(STAGE_LIMIT, remaining)]
            ranked = evaluate(batch)
            remaining -= len(ranked)
            depth_reached = max(depth_reached, *(state.depth for state, _ in ranked))
            beam = sorted(ranked, key=lambda pair: _rank_key(pair[1]), reverse=True)[:BEAM_WIDTH]
            for pair in ranked:
                if best is None or _rank_key(pair[1]) > _rank_key(best[1]):
                    best = pair
            if not beam:
                termination = "NO_NEW_FEASIBLE_STATE"
                break
    selected = best[0] if best else None
    lineage = []
    node = selected
    while node is not None:
        lineage.append(node.provenance())
        node = states_by_hash.get(node.parent_state_hash) if node.parent_state_hash else None
    lineage.reverse()
    if (trace["candidate_contexts_sent"] != len(evaluated) or
            trace["candidate_contexts_sent"] > max_qwen_candidate_evals or
            sum(trace["candidate_batch_sizes"]) != len(evaluated)):
        raise RuntimeError("F-MRS evaluator quota or trace mismatch")
    return {"selected_state": selected,
            "selected_candidate_prediction": best[1].predicted if best else "",
            "evaluated": evaluated, "trace": trace, "counters": counters,
            "search_depth_reached": depth_reached,
            "termination_reason": termination,
            "search_latency_sec": time.perf_counter() - started,
            "selected_provenance": lineage}
