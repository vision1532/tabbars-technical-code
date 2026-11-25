"""Frozen table rendering, lexical initialization, and candidate evaluation helpers."""

from __future__ import annotations

import re
import time
from typing import Any

import torch
from transformers import AutoTokenizer

try:
    from vllm import LLM, SamplingParams
except ImportError:
    LLM = None
    SamplingParams = None


from run_longtablebench_search_oracle import (
    Candidate,
    build_connector_index,
    build_row_index,
    build_section_fixed_cost,
    build_section_index,
    build_seed_sets,
    estimate_row_set_token_cost,
    estimate_text_tokens,
    extract_question_terms,
    generate_neighbor_sets,
    is_ministral_model,
    make_budget_feasible_row_set,
    marginal_gain,
    normalize_cell_value,
    normalize_for_match,
    select_top_candidate_sets,
    budget_regime_config,
)


ANSWER_PATTERNS = [
    re.compile(r"[Aa]nswer\s*[:：]\s*([A-D])\b"),
    re.compile(r"(?:the\s+)?correct\s+answer\s+(?:is|would\s+be)\s*[:：]?\s*([A-D])\b", re.IGNORECASE),
    re.compile(r"final\s+(?:choice|answer)\s*[:：]?\s*([A-D])\b", re.IGNORECASE),
    re.compile(r"\boption\s*([A-D])\b", re.IGNORECASE),
]


TAIL_PATTERNS = [
    re.compile(r"\(([A-D])\)"),
    re.compile(r"\b([A-D])\)"),
]


def extract_answer(text: str) -> str | None:
    for pattern in ANSWER_PATTERNS:
        matches = list(pattern.finditer(text))
        if matches:
            return matches[-1].group(1).upper()
    tail = text[-300:]
    for pattern in TAIL_PATTERNS:
        matches = list(pattern.finditer(tail))
        if matches:
            return matches[-1].group(1).upper()
    match = re.search(r"\b([A-D])\b$", text.strip())
    if match:
        return match.group(1).upper()
    return None


def count_tokens(tokenizer: AutoTokenizer, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def build_sections_from_markdown_table(
    table_text: str,
    question_text: str,
    tokenizer: AutoTokenizer,
) -> tuple[str, list[dict[str, Any]], int]:
    question_terms = extract_question_terms(question_text)
    sections: list[dict[str, Any]] = []
    global_row_id = 0
    original_table_tokens = estimate_text_tokens(tokenizer, table_text)
    lines = table_text.splitlines()
    title_indices = [idx for idx, line in enumerate(lines) if line.startswith("## ")]
    if title_indices:
        table_prefix = "\n".join(lines[: title_indices[0]]).rstrip()
    else:
        table_prefix = table_text.strip()

    for sec_pos, title_idx in enumerate(title_indices):
        next_title_idx = title_indices[sec_pos + 1] if sec_pos + 1 < len(title_indices) else len(lines)
        title_line = lines[title_idx].rstrip("\n")
        table_name = title_line[3:].strip()
        sec_idx = len(sections)
        section_rows: list[dict[str, Any]] = []
        header_line = ""
        delimiter_line = ""
        header_found = False
        delimiter_found = False

        for idx in range(title_idx + 1, next_title_idx):
            line = lines[idx].rstrip("\n")
            if not line.strip():
                continue
            if not line.lstrip().startswith("|"):
                continue
            if not header_found:
                header_line = line
                header_found = True
                continue
            if not delimiter_found:
                delimiter_line = line
                delimiter_found = True
                continue
            normalized_values = [
                normalize_cell_value(part.strip())
                for part in line.strip().strip("|").split("|")
            ]
            normalized_row_text = normalize_for_match(" ".join(normalized_values))
            exact_term_hits = sum(1 for term in question_terms if term in normalized_row_text)
            anchor_mass = float(exact_term_hits)
            combined_score = anchor_mass + 0.25 * len([v for v in normalized_values if v])
            row_obj = {
                "row_id": global_row_id,
                "section_idx": sec_idx,
                "table_name": table_name,
                "line": line,
                "token_len": estimate_text_tokens(tokenizer, line),
                "question_anchor_score": anchor_mass,
                "combined_score": combined_score,
                "repair_score": 0.0,
                "neighbor_ids": [],
                "connector_values": [v for v in normalized_values if 2 <= len(v) <= 64],
            }
            section_rows.append(row_obj)
            global_row_id += 1

        sections.append(
            {
                "section_idx": sec_idx,
                "title": table_name,
                "title_line": title_line,
                "header_line": header_line,
                "delimiter_line": delimiter_line,
                "fixed_token_len": estimate_text_tokens(
                    tokenizer, "\n".join([title_line, header_line, delimiter_line])
                ),
                "score": 0.0,
                "anchor_mass": 0.0,
                "bridge_strength": 0.0,
                "rows": section_rows,
            }
        )

    all_rows = [(int(row["row_id"]), row) for sec in sections for row in sec["rows"]]
    connector_index: dict[str, set[int]] = {}
    for rid, row in all_rows:
        for value in row["connector_values"]:
            connector_index.setdefault(value, set()).add(rid)

    row_to_section = {int(row["row_id"]): int(sec["section_idx"]) for sec in sections for row in sec["rows"]}

    for _, row in all_rows:
        neighbors: set[int] = set()
        bridge_sections: set[int] = set()
        filtered_connectors = []
        for value in row["connector_values"]:
            linked = connector_index.get(value, set())
            if len(linked) <= 1 or len(linked) > 24:
                continue
            filtered_connectors.append(value)
            for other in linked:
                if other != row["row_id"]:
                    neighbors.add(other)
        for other_id in neighbors:
            bridge_sections.add(row_to_section.get(int(other_id), -1))
        bridge_sections.discard(-1)
        row["connector_values"] = filtered_connectors
        row["neighbor_ids"] = sorted(neighbors)
        row["repair_score"] = float(len(bridge_sections))

    for sec in sections:
        sec["anchor_mass"] = float(sum(float(r["question_anchor_score"]) for r in sec["rows"]))
        sec["bridge_strength"] = float(sum(len(r["neighbor_ids"]) for r in sec["rows"]))
        sec["score"] = sec["anchor_mass"] + 0.2 * sec["bridge_strength"]

    return table_prefix, sections, original_table_tokens


def render_pruned_table(
    table_prefix: str,
    sections: list[dict[str, Any]],
    selected_row_ids: set[int],
) -> str:
    blocks = []
    for sec in sections:
        chosen = [row for row in sec.get("rows", []) if int(row["row_id"]) in selected_row_ids]
        if not chosen:
            continue
        lines = [sec["title_line"], sec["header_line"], sec["delimiter_line"]]
        lines.extend(row["line"] for row in chosen)
        blocks.append("\n".join(lines))
    body = "\n\n".join(blocks)
    if table_prefix and body:
        return f"{table_prefix}\n\n{body}"
    return body or table_prefix


def build_prompt(instruction: str, table_text: str, question_text: str, options: str) -> str:
    sep = "\n\n"
    return instruction + sep + table_text + sep + question_text + sep + options


def candidate_sort_key(candidate: Candidate, selection_mode: str) -> tuple[float, ...]:
    has_prediction = 1.0 if candidate.predicted else 0.0
    if selection_mode == "oracle":
        return (candidate.raw_f1, has_prediction, candidate.heuristic_score, -candidate.token_cost)
    return (has_prediction, candidate.heuristic_score, -candidate.token_cost)


def evaluate_candidate_sets(
    llm: LLM,
    sampling_params: SamplingParams,
    instruction: str,
    question_text: str,
    options: str,
    table_prefix: str,
    sections: list[dict[str, Any]],
    selected_row_sets: list[set[int]],
    heuristic_scores: list[float] | None = None,
    selection_mode: str = "oracle",
    completion: str | None = None,
) -> list[Candidate]:
    prompts = []
    for rows in selected_row_sets:
        pruned_table = render_pruned_table(table_prefix, sections, rows)
        prompts.append(build_prompt(instruction, pruned_table, question_text, options))
    outputs = llm.generate(prompts, sampling_params)
    candidates: list[Candidate] = []
    for idx, rows in enumerate(selected_row_sets):
        raw = outputs[idx].outputs[0].text if outputs[idx].outputs else ""
        pred = extract_answer(raw)
        token_cost = sum(
            int(sec.get("fixed_token_len", 0)) + sum(int(r.get("token_len", 0)) for r in sec.get("rows", []) if int(r["row_id"]) in rows)
            for sec in sections
            if any(int(r["row_id"]) in rows for r in sec.get("rows", []))
        )
        score = 0.0
        if selection_mode == "oracle" and completion is not None:
            score = 1.0 if pred and pred == completion.strip().upper() else 0.0
        candidates.append(
            Candidate(
                selected_row_ids=set(rows),
                heuristic_score=0.0 if heuristic_scores is None else float(heuristic_scores[idx]),
                token_cost=token_cost,
                raw_output=raw,
                predicted=pred or "",
                raw_f1=score,
            )
        )
    return candidates


def run_example(
    llm: LLM,
    sampling_params: SamplingParams,
    tokenizer: AutoTokenizer,
    example: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    search_start_time = time.time()
    instruction = example.get("instruction", "")
    table = example.get("table", "")
    question_text = example.get("question_text", "")
    options = example.get("options", "")
    completion = example.get("completion", "").strip().upper()
    selection_mode = str(args.selection_mode)

    table_prefix, sections, original_table_tokens = build_sections_from_markdown_table(
        table,
        question_text + "\n" + options,
        tokenizer,
    )
    row_index = build_row_index(sections)
    connector_index = build_connector_index(row_index)
    section_index = build_section_index(sections)
    section_fixed_cost = build_section_fixed_cost(sections)
    compression_ratio = float(args.compression_ratio)
    keep_ratio = max(0.0, min(1.0, 1.0 - compression_ratio))
    budget_tokens = max(1, int(original_table_tokens * keep_ratio))
    regime_cfg = budget_regime_config(keep_ratio)
    meta = example.get("meta", {})
    sample_label = (
        f"{meta.get('scale', 'unknown')}/db{meta.get('dbIdx', 'na')}/"
        f"sample{meta.get('sampleIdx', 'na')}/q{meta.get('questionIdx', 'na')}"
    )
    print(
        f"[merged_v6][sample] start={sample_label} rows={len(row_index)} sections={len(section_index)} "
        f"budget={budget_tokens} keep_ratio={keep_ratio:.4f}",
        flush=True,
    )

    seed_sets = build_seed_sets(
        set(),
        row_index,
        section_index,
        connector_index,
        keep_ratio,
        args.seed_topk,
        budget_tokens,
        section_fixed_cost,
    )
    print(f"[merged_v6][sample] raw_seed_sets={len(seed_sets)}", flush=True)
    if not seed_sets and row_index:
        fallback = sorted(
            row_index.keys(),
            key=lambda rid: marginal_gain(rid, set(), row_index, section_index, connector_index, keep_ratio),
            reverse=True,
        )[: max(1, args.seed_topk)]
        seed_sets = [
            make_budget_feasible_row_set([int(rid)], row_index, budget_tokens, section_fixed_cost)
            for rid in fallback
        ]
        seed_sets = [rows for rows in seed_sets if rows]
        print(f"[merged_v6][sample] fallback_seed_sets={len(seed_sets)}", flush=True)

    seen: set[tuple[int, ...]] = set()
    unique_seed_sets: list[set[int]] = []
    for rows in seed_sets:
        key = tuple(sorted(rows))
        if key in seen:
            continue
        token_cost = estimate_row_set_token_cost(rows, row_index, section_fixed_cost)
        if token_cost > budget_tokens:
            continue
        seen.add(key)
        unique_seed_sets.append(rows)
    print(f"[merged_v6][sample] feasible_seed_sets={len(unique_seed_sets)}", flush=True)

    ranked_seed_sets = select_top_candidate_sets(
        unique_seed_sets,
        row_index,
        section_index,
        connector_index,
        keep_ratio,
        budget_tokens,
        section_fixed_cost,
        limit=max(args.beam_size * int(regime_cfg["seed_batch_multiplier"]), args.seed_topk * 2),
    )
    print(f"[merged_v6][sample] ranked_seed_sets={len(ranked_seed_sets)}", flush=True)
    beam = evaluate_candidate_sets(
        llm,
        sampling_params,
        instruction,
        question_text,
        options,
        table_prefix,
        sections,
        [rows for rows, _, _ in ranked_seed_sets],
        heuristic_scores=[heur for _, _, heur in ranked_seed_sets],
        selection_mode=selection_mode,
        completion=completion if selection_mode == "oracle" else None,
    )
    beam.sort(key=lambda c: candidate_sort_key(c, selection_mode), reverse=True)
    beam = beam[: args.beam_size]
    print(f"[merged_v6][sample] beam_after_seed={len(beam)}", flush=True)

    for expansion_idx in range(args.max_expansions):
        if selection_mode == "oracle" and beam and beam[0].raw_f1 >= 0.999:
            break
        frontier_pool: list[Candidate] = list(beam)
        queued_neighbors: list[set[int]] = []
        for cand in beam:
            neighbors = generate_neighbor_sets(
                cand.selected_row_ids,
                row_index,
                section_index,
                connector_index,
                keep_ratio,
                add_topk=args.expand_topk,
                drop_topk=max(1, args.expand_topk // 2),
                budget_tokens=budget_tokens,
                section_fixed_cost=section_fixed_cost,
            )
            for rows in neighbors:
                key = tuple(sorted(rows))
                if key in seen:
                    continue
                token_cost = estimate_row_set_token_cost(rows, row_index, section_fixed_cost)
                if token_cost > budget_tokens:
                    continue
                seen.add(key)
                queued_neighbors.append(rows)
        print(
            f"[merged_v6][sample] expansion={expansion_idx} queued_neighbors={len(queued_neighbors)}",
            flush=True,
        )

        ranked_neighbors = select_top_candidate_sets(
            queued_neighbors,
            row_index,
            section_index,
            connector_index,
            keep_ratio,
            budget_tokens,
            section_fixed_cost,
            limit=max(args.beam_size * int(regime_cfg["expand_batch_multiplier"]), args.expand_topk * 3),
        )
        print(
            f"[merged_v6][sample] expansion={expansion_idx} ranked_neighbors={len(ranked_neighbors)}",
            flush=True,
        )
        if ranked_neighbors:
            frontier_pool.extend(
                evaluate_candidate_sets(
                    llm,
                    sampling_params,
                    instruction,
                    question_text,
                    options,
                    table_prefix,
                    sections,
                    [rows for rows, _, _ in ranked_neighbors],
                    heuristic_scores=[heur for _, _, heur in ranked_neighbors],
                    selection_mode=selection_mode,
                    completion=completion if selection_mode == "oracle" else None,
                )
            )
        frontier_pool.sort(key=lambda c: candidate_sort_key(c, selection_mode), reverse=True)
        beam = frontier_pool[: args.beam_size]
        print(f"[merged_v6][sample] expansion={expansion_idx} beam={len(beam)}", flush=True)

    if not beam:
        raise RuntimeError(
            f"empty beam after search for {sample_label}: "
            f"raw_seed_sets={len(seed_sets)} feasible_seed_sets={len(unique_seed_sets)} "
            f"seen={len(seen)} budget={budget_tokens} keep_ratio={keep_ratio:.4f}"
        )
    best = beam[0]
    search_time_seconds = time.time() - search_start_time
    original_prompt = build_prompt(instruction, table, question_text, options)
    pruned_table = render_pruned_table(table_prefix, sections, best.selected_row_ids)
    pruned_prompt = build_prompt(instruction, pruned_table, question_text, options)
    return {
        "predicted": best.predicted,
        "raw_output": best.raw_output,
        "search_mode": "training_free_budget_conditioned_graph_search_v6",
        "selection_mode": selection_mode,
        "search_selection_score": candidate_sort_key(best, selection_mode)[0],
        "compression_ratio": compression_ratio,
        "budget_keep_ratio": keep_ratio,
        "search_budget_tokens": budget_tokens,
        "original_table_tokens": original_table_tokens,
        "pruned_table_tokens": best.token_cost,
        "pruned_table": pruned_table,
        "original_prompt_tokens": count_tokens(tokenizer, original_prompt),
        "pruned_prompt_tokens": count_tokens(tokenizer, pruned_prompt),
        "best_selected_row_ids": sorted(best.selected_row_ids),
        "num_candidate_evals": len(seen),
        "search_time_seconds": search_time_seconds,
        "meta": example.get("meta", {}),
    }
