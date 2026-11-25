"""Exact-budget adapter over the validated v6 candidate generator and search.

Only the original parser, seed/neighbor generator, candidate evaluator, and
beam loop make selection decisions. The patches below reproduce the strict
M0_LEXICAL ablation; additional wrappers record decisions and reject rendered
tables above the budget before either evaluation or beam admission.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from pathlib import Path
import sys
import time
from typing import Callable
from unittest.mock import patch

import run_longtablebench_search_oracle as oracle
import run_merged_jsonl_graph_search_v6 as v6

from .adjudicator import evaluate_frozen_candidates
from .budget import exact_tokens, table_budget
from .evidence_bundles import BundleFactory

_ROOT = Path(__file__).resolve().parents[2]
# The frozen experimental adapter uses module-level imports from its directory.
for _path in (_ROOT / "research_v6_localization", _ROOT / "research_stable_graph"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from run_v6_ablation import CountingLLM, candidate_score_without_graph, row_score_grounding_only  # noqa: E402


class _SufficientStop(Exception):
    """Stop the unchanged beam loop after a judge accepts current evidence."""


def _safe_example(example: dict) -> dict:
    safe = dict(example)
    safe.pop("completion", None)
    safe["meta"] = {key: value for key, value in example.get("meta", {}).items()
                    if key not in {"rightIdx", "qtype"}}
    return safe


def search_candidates(example: dict, budget: float, tokenizer, llm, sampling,
                      *, proposal_mode: str = "lexical", graph_mode: str = "reachability",
                      adjudication: str = "qwen", replay: dict[tuple[int, ...], str] | None = None,
                      stage_judge: Callable | None = None,
                      construction_mode: str = "C0",
                      candidate_cache: dict[tuple[int, ...], str] | None = None) -> dict:
    if construction_mode not in {"C0", "C1", "C2"}:
        raise ValueError(construction_mode)
    if proposal_mode not in {"lexical", "none"}:
        raise ValueError(proposal_mode)
    if graph_mode not in {"reachability", "legacy_weighted"}:
        raise ValueError(graph_mode)
    if adjudication not in {"qwen", "deterministic"}:
        raise ValueError(adjudication)
    if adjudication == "qwen" and llm is None and replay is None and candidate_cache is None:
        raise ValueError("Qwen adjudication needs a loaded LLM or exact candidate replay")
    safe = _safe_example(example)
    full_tokens, budget_tokens = table_budget(tokenizer, str(safe["table"]), budget)
    trace = {"generated_candidate_sets": 0, "unique_generated_candidate_sets": 0,
             "candidate_contexts_considered": 0, "candidate_replay_hits": 0,
             "candidate_contexts_sent": 0, "unique_candidate_sets_sent": 0,
             "candidate_contexts_cached": 0,
             "qwen_candidate_requests": 0, "candidate_batch_sizes": [],
             "candidate_input_tokens": 0, "candidate_output_tokens": 0,
             "candidate_qwen_wall_sec": 0.0, "stages": [],
             "exact_candidates_checked": 0, "exact_candidates_rejected": 0,
             "judge_rounds": [], "construction_mode": construction_mode,
             "bundle_proposals": 0, "bundle_provenance": {}}
    seen_generated: set[tuple[int, ...]] = set()
    seen_sent: set[tuple[int, ...]] = set()
    state: dict = {}
    bundle_keys: set[tuple[int, ...]] = set()
    stage = {"rank": 0, "eval": 0, "neighbor": 0}
    judge_state: dict = {"beam": [], "selected": None, "last": None}
    if budget_tokens == 0:
        trace["search_wall_sec"] = 0.0
        return {"pruned_table": "", "best_selected_row_ids": [], "exact_pruned_tokens": 0,
                "exact_budget_tokens": 0, "original_table_tokens": full_tokens,
                "search_status": "no_feasible_candidate", "trace": trace,
                "termination_reason": "NO_FEASIBLE_CANDIDATE",
                "legacy_candidate_prediction": ""}

    original_sections = v6.build_sections_from_markdown_table
    original_seeds = v6.build_seed_sets
    original_neighbors = v6.generate_neighbor_sets
    original_rank = v6.select_top_candidate_sets
    original_eval = v6.evaluate_candidate_sets

    def note_sets(sets):
        trace["generated_candidate_sets"] += len(sets)
        seen_generated.update(tuple(sorted(rows)) for rows in sets)
        trace["unique_generated_candidate_sets"] = len(seen_generated)
        return sets

    def sections_patch(table_text, question_text, tok):
        prefix, sections, count = original_sections(
            table_text, question_text if proposal_mode == "lexical" else "", tok)
        if graph_mode == "reachability":
            for section in sections:
                for row in section["rows"]:
                    row["combined_score"] = float(row["question_anchor_score"])
        state.update(prefix=prefix, sections=sections)
        if construction_mode != "C0":
            state["bundle_factory"] = BundleFactory(
                question=str(safe.get("question_text", "")),
                options=str(safe.get("options", "")), prefix=prefix,
                sections=sections, tokenizer=tokenizer, budget_tokens=budget_tokens)
        if count != full_tokens:
            raise RuntimeError("parser and exact budget disagree on full-table tokens")
        return prefix, sections, count

    def seeds_patch(*args, **kwargs):
        base = original_seeds(*args, **kwargs)
        additions = []
        if construction_mode != "C0":
            additions = state["bundle_factory"].propose(set(), construction_mode)
            base_keys = {tuple(sorted(rows)) for rows in base}
            bundle_keys.update(tuple(sorted(rows)) for rows in additions
                               if tuple(sorted(rows)) not in base_keys)
            trace["bundle_proposals"] += len(additions)
        sets = note_sets(base + additions)
        trace["stages"].append({"stage": "raw_seeds", "sets": [sorted(s) for s in sets]})
        return sets

    def neighbors_patch(*args, **kwargs):
        base = original_neighbors(*args, **kwargs)
        additions = []
        if construction_mode != "C0":
            additions = state["bundle_factory"].propose(set(args[0]), construction_mode)
            base_keys = {tuple(sorted(rows)) for rows in base}
            bundle_keys.update(tuple(sorted(rows)) for rows in additions
                               if tuple(sorted(rows)) not in base_keys)
            trace["bundle_proposals"] += len(additions)
        sets = note_sets(base + additions)
        stage["neighbor"] += 1
        trace["stages"].append({"stage": f"neighbor_call_{stage['neighbor']}",
                                "parent": sorted(args[0]), "sets": [sorted(s) for s in sets]})
        return sets

    def rank_patch(candidate_sets, row_index, section_index, connector_index,
                   keep_ratio, estimated_budget, section_fixed_cost, limit):
        if estimated_budget != budget_tokens:
            raise RuntimeError("search budget differs from exact table budget")
        base_sets = [rows for rows in candidate_sets
                     if tuple(sorted(rows)) not in bundle_keys]
        bundle_sets = [rows for rows in candidate_sets
                       if tuple(sorted(rows)) in bundle_keys]
        ordered = original_rank(base_sets, row_index, section_index,
                                connector_index, keep_ratio, estimated_budget,
                                section_fixed_cost, limit=max(1, len(base_sets)))
        bundle_ordered = (original_rank(bundle_sets, row_index, section_index,
                                       connector_index, keep_ratio, estimated_budget,
                                       section_fixed_cost, limit=max(1, len(bundle_sets)))
                          if bundle_sets else [])
        accepted = []
        for rows, estimated_cost, heuristic in ordered:
            rendered = v6.render_pruned_table(state["prefix"], state["sections"], rows)
            exact_cost = exact_tokens(tokenizer, rendered)
            trace["exact_candidates_checked"] += 1
            if exact_cost > budget_tokens:
                trace["exact_candidates_rejected"] += 1
                continue
            accepted.append((rows, estimated_cost, heuristic))
            if len(accepted) >= limit:
                break
        # A fixed 16-context construction channel prevents the original
        # anchor-only ranking cutoff from erasing every feasible joint bundle.
        # Both channels use the *same frozen heuristic* and the beam/J0 key
        # remain unchanged. C0 never enters this branch.
        if construction_mode != "C0":
            seen_accepted = {tuple(sorted(rows)) for rows, _, _ in accepted}
            for rows, estimated_cost, heuristic in bundle_ordered:
                key = tuple(sorted(rows))
                if key in seen_accepted:
                    continue
                rendered = v6.render_pruned_table(state["prefix"], state["sections"], rows)
                exact_cost = exact_tokens(tokenizer, rendered)
                trace["exact_candidates_checked"] += 1
                if exact_cost > budget_tokens:
                    trace["exact_candidates_rejected"] += 1
                    continue
                accepted.append((rows, estimated_cost, heuristic))
                seen_accepted.add(key)
                if len(accepted) >= limit + 16:
                    break
        stage["rank"] += 1
        trace["stages"].append({"stage": f"rank_{stage['rank']}",
                                "sets": [{"rows": sorted(rows), "estimated_tokens": cost,
                                          "heuristic": score}
                                         for rows, cost, score in accepted]})
        return accepted

    def eval_patch(*args, **kwargs):
        sets = args[7] if len(args) > 7 else kwargs["selected_row_sets"]
        scores = kwargs.get("heuristic_scores") or [0.0] * len(sets)
        # The ranking gate is authoritative, and this second check protects
        # the model boundary if the legacy beam ever calls evaluation directly.
        exact_costs = [exact_tokens(tokenizer,
            v6.render_pruned_table(state["prefix"], state["sections"], rows))
                       for rows in sets]
        if any(cost > budget_tokens for cost in exact_costs):
            raise RuntimeError("candidate reached adjudication above exact table budget")
        trace["candidate_contexts_considered"] += len(sets)
        stage["eval"] += 1
        if adjudication == "deterministic" or replay is not None:
            sections = args[6]
            row_index = oracle.build_row_index(sections)
            fixed = oracle.build_section_fixed_cost(sections)
            candidates = []
            for rows, score in zip(sets, scores):
                key = tuple(sorted(rows))
                if adjudication == "deterministic":
                    prediction = ""
                else:
                    if key not in replay:
                        raise RuntimeError(f"candidate replay miss: {key}")
                    prediction = replay[key]
                    trace["candidate_replay_hits"] += 1
                candidates.append(oracle.Candidate(set(rows), float(score),
                    oracle.estimate_row_set_token_cost(set(rows), row_index, fixed),
                    "", prediction, 0.0 if replay is not None else -1.0))
        elif candidate_cache is not None:
            row_index = oracle.build_row_index(args[6])
            fixed = oracle.build_section_fixed_cost(args[6])
            candidates = [None] * len(sets)
            misses = []
            miss_positions = []
            for i, (rows, score) in enumerate(zip(sets, scores)):
                key = tuple(sorted(rows))
                if key in candidate_cache:
                    candidates[i] = oracle.Candidate(set(rows), float(score),
                        oracle.estimate_row_set_token_cost(set(rows), row_index, fixed),
                        "", candidate_cache[key], -1.0)
                    trace["candidate_contexts_cached"] += 1
                else:
                    misses.append(rows)
                    miss_positions.append(i)
                    seen_sent.add(key)
            trace["unique_candidate_sets_sent"] = len(seen_sent)
            if misses:
                miss_args = list(args)
                miss_args[7] = misses
                miss_kwargs = dict(kwargs)
                miss_kwargs["heuristic_scores"] = [scores[i] for i in miss_positions]
                fresh = evaluate_frozen_candidates(original_eval, *miss_args, **miss_kwargs)
                for i, candidate in zip(miss_positions, fresh):
                    candidates[i] = candidate
            if any(candidate is None for candidate in candidates):
                raise RuntimeError("candidate cache merge incomplete")
        else:
            seen_sent.update(tuple(sorted(rows)) for rows in sets)
            trace["unique_candidate_sets_sent"] = len(seen_sent)
            candidates = evaluate_frozen_candidates(original_eval, *args, **kwargs)
        trace["stages"].append({"stage": f"candidate_eval_{stage['eval']}",
                                "candidates": [{"rows": sorted(c.selected_row_ids),
                                                "heuristic": c.heuristic_score,
                                                "prediction": c.predicted,
                                                "estimated_tokens": c.token_cost,
                                                "exact_tokens": exact_costs[i],
                                                "bundle_provenance": (state["bundle_factory"].provenance.get(
                                                    tuple(sorted(c.selected_row_ids)), [])
                                                    if construction_mode != "C0" else [])}
                                               for i, c in enumerate(candidates)]})
        if stage_judge is not None and candidates:
            # With the deterministic candidate adapter, this is exactly the
            # frozen beam that v6 will retain after the evaluator returns.
            pool = judge_state["beam"] + candidates
            beam = sorted(pool, key=lambda c: v6.candidate_sort_key(c, "blind"), reverse=True)[:search_args.beam_size]
            outcome = stage_judge(stage["eval"], beam, state["prefix"], state["sections"], budget_tokens)
            chosen = int(outcome["selected_candidate"])
            if not 1 <= chosen <= len(beam):
                raise RuntimeError("judge adapter returned candidate outside current beam")
            judge_state.update(beam=beam, selected=beam[chosen - 1], last=outcome)
            trace["judge_rounds"].append(outcome)
            if outcome["sufficient"]:
                raise _SufficientStop
        return candidates

    search_args = argparse.Namespace(compression_ratio=1.0 - budget, selection_mode="blind",
                                     seed_topk=6, beam_size=4, expand_topk=8, max_expansions=2)
    started = time.perf_counter()
    with ExitStack() as stack:
        stack.enter_context(patch.object(v6, "build_sections_from_markdown_table", sections_patch))
        stack.enter_context(patch.object(v6, "build_seed_sets", seeds_patch))
        stack.enter_context(patch.object(v6, "generate_neighbor_sets", neighbors_patch))
        stack.enter_context(patch.object(v6, "select_top_candidate_sets", rank_patch))
        stack.enter_context(patch.object(v6, "evaluate_candidate_sets", eval_patch))
        if graph_mode == "reachability":
            stack.enter_context(patch.object(oracle, "marginal_gain", row_score_grounding_only))
            stack.enter_context(patch.object(v6, "marginal_gain", row_score_grounding_only))
            stack.enter_context(patch.object(oracle, "candidate_heuristic_score",
                lambda *a, **kw: candidate_score_without_graph(
                    *a, **kw, keep_section=False, keep_utilization=False)))
            stack.enter_context(patch.object(oracle, "connector_bundle_rows", lambda *a, **kw: []))
        effective_llm = CountingLLM(llm, tokenizer, trace) if llm and adjudication == "qwen" and replay is None else None
        try:
            result = v6.run_example(effective_llm, sampling, tokenizer, safe, search_args)
        except _SufficientStop:
            result = {"original_table_tokens": full_tokens, "search_status": "ok"}
        except RuntimeError as exc:
            if "empty beam after search" not in str(exc):
                raise
            message = str(exc)
            trace["search_wall_sec"] = time.perf_counter() - started
            result = {"pruned_table": "", "best_selected_row_ids": [],
                      "original_table_tokens": full_tokens, "exact_pruned_tokens": 0,
                      "exact_budget_tokens": budget_tokens,
                      "search_status": "no_feasible_candidate", "search_failure": message,
                      "termination_reason": "NO_FEASIBLE_CANDIDATE",
                      "legacy_candidate_prediction": ""}
    if stage_judge is not None and judge_state["selected"] is not None:
        selected = judge_state["selected"]
        result["best_selected_row_ids"] = sorted(selected.selected_row_ids)
        result["pruned_table"] = v6.render_pruned_table(
            state["prefix"], state["sections"], selected.selected_row_ids)
        result["judge_selection"] = judge_state["last"]
        if judge_state["last"]["sufficient"]:
            result["termination_reason"] = "SUFFICIENT"
        elif not judge_state["last"].get("parse_valid", False):
            result["termination_reason"] = "JUDGE_PARSE_FAILURE"
        else:
            result["termination_reason"] = "SEARCH_CAP_INSUFFICIENT"
    trace["search_wall_sec"] = time.perf_counter() - started
    if construction_mode != "C0" and "bundle_factory" in state:
        trace["bundle_provenance"] = {
            ",".join(map(str, key)): entries
            for key, entries in state["bundle_factory"].provenance.items()}
    result.pop("raw_output", None)
    result["legacy_candidate_prediction"] = result.pop("predicted", result.get("legacy_candidate_prediction", ""))
    result["trace"] = trace
    result["exact_budget_tokens"] = budget_tokens
    result["exact_pruned_tokens"] = exact_tokens(tokenizer, result["pruned_table"])
    if result["exact_pruned_tokens"] > budget_tokens:
        raise RuntimeError("selected context violates exact budget")
    return result
