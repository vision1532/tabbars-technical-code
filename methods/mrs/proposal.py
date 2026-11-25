"""Observed lexical doorways from the unchanged v6 matcher."""

from __future__ import annotations

from run_longtablebench_search_oracle import extract_question_terms, normalize_cell_value, normalize_for_match
from run_merged_jsonl_graph_search_v6 import build_sections_from_markdown_table


def parse_for_proposal(example: dict, tokenizer, *, enabled: bool = True):
    question = str(example.get("question_text", ""))
    options = str(example.get("options", ""))
    query = question + "\n" + options if enabled else ""
    return build_sections_from_markdown_table(str(example.get("table", "")), query, tokenizer)


def lexical_proposal(question: str, options: str, sections: list[dict], *, enabled: bool = True) -> dict:
    question_terms = set(extract_question_terms(question)) if enabled else set()
    option_terms = set(extract_question_terms(options)) if enabled else set()
    # The v6 parser matches its sorted union against the whole normalized row.
    all_terms = set(extract_question_terms(question + "\n" + options)) if enabled else set()
    seeds = []
    for section in sections:
        for row in section["rows"]:
            normalized = normalize_for_match(" ".join(
                normalize_cell_value(part.strip()) for part in row["line"].strip().strip("|").split("|")))
            terms = sorted(term for term in all_terms if term in normalized)
            if bool(terms) != (float(row["question_anchor_score"]) > 0):
                raise RuntimeError(f"lexical provenance disagrees with v6 matcher on row {row['row_id']}")
            if terms:
                seeds.append({
                    "row_id": int(row["row_id"]),
                    "matched_terms": terms,
                    "question_terms": sorted(set(terms) & question_terms),
                    "option_terms": sorted(set(terms) & option_terms),
                })
    return {"seed_row_ids": [s["row_id"] for s in seeds], "seeds": seeds,
            "num_lexical_seeds": len(seeds)}
