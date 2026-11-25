"""Frozen table parsing and row-set scoring helpers."""

from __future__ import annotations

from dataclasses import dataclass
import re

import pandas as pd
from transformers import AutoTokenizer


def is_ministral_model(model_path: str) -> bool:
    m = model_path.lower()
    return "ministral" in m or "mistral-small-3" in m


def estimate_text_tokens(tokenizer: AutoTokenizer, text: str) -> int:
    return max(1, len(tokenizer.encode(text, add_special_tokens=False)))


def normalize_cell_value(value: object) -> str:
    text = "" if value is None or (isinstance(value, float) and pd.isna(value)) else str(value)
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def normalize_for_match(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def extract_question_terms(question_text: str) -> list[str]:
    stop = {
        "the", "a", "an", "is", "are", "was", "were", "what", "which", "who", "whom", "whose",
        "where", "when", "why", "how", "and", "or", "of", "to", "in", "on", "for", "with",
        "there", "exists", "given", "table", "please", "determine", "whether", "output",
        "full", "row", "python", "dict", "list", "string", "strings", "otherwise",
    }
    terms = []
    for tok in re.findall(r"[A-Za-z0-9_]+", question_text.lower()):
        if len(tok) >= 2 and tok not in stop:
            terms.append(tok)
    return sorted(set(terms))


def split_markdown_row(line: str) -> list[str]:
    return [part.strip() for part in line.strip().strip("|").split("|")]


def extract_answer(text: str) -> str:
    import re
    if not text:
        return ""
    code_matches = re.findall(r"```(?:[^\n`]*)\n?([\s\S]*?)```", text)
    if code_matches:
        content = code_matches[-1].strip()
        content = re.sub(r"^\s*(?:list\s*\[\s*str\s*\]|\[\s*str\s*\]|dict|list)\s*:\s*", "", content, flags=re.I)
        content = re.sub(r"^\s*```+\s*", "", content)
        return content.strip()
    return text.strip()


@dataclass
class Candidate:
    selected_row_ids: set[int]
    heuristic_score: float
    token_cost: int
    raw_output: str = ""
    predicted: str = ""
    raw_f1: float = -1.0


def build_row_index(sections: list[dict]) -> dict[int, dict]:
    out: dict[int, dict] = {}
    for sec in sections:
        for row in sec.get("rows", []):
            out[int(row["row_id"])] = row
    return out


def build_connector_index(row_index: dict[int, dict]) -> dict[str, set[int]]:
    out: dict[str, set[int]] = {}
    for rid, row in row_index.items():
        for value in row.get("connector_values", []):
            key = str(value).strip().lower()
            if key:
                out.setdefault(key, set()).add(rid)
    return out


def build_section_index(sections: list[dict]) -> dict[int, set[int]]:
    out: dict[int, set[int]] = {}
    for sec in sections:
        sec_idx = int(sec.get("section_idx", -1))
        out[sec_idx] = {int(row["row_id"]) for row in sec.get("rows", [])}
    return out


def build_section_fixed_cost(sections: list[dict]) -> dict[int, int]:
    out: dict[int, int] = {}
    for sec in sections:
        sec_idx = int(sec.get("section_idx", -1))
        out[sec_idx] = int(sec.get("fixed_token_len", 0))
    return out


def estimate_row_set_token_cost(
    selected_row_ids: set[int],
    row_index: dict[int, dict],
    section_fixed_cost: dict[int, int],
) -> int:
    if not selected_row_ids:
        return 0
    cost = 0
    covered_sections: set[int] = set()
    for rid in selected_row_ids:
        row = row_index[rid]
        sec_idx = int(row.get("section_idx", -1))
        if sec_idx not in covered_sections:
            covered_sections.add(sec_idx)
            cost += int(section_fixed_cost.get(sec_idx, 0))
        cost += int(row.get("token_len", 0))
    return cost


def regime_from_ratio(keep_ratio: float) -> str:
    if keep_ratio <= 0.08:
        return "low"
    if keep_ratio <= 0.18:
        return "mid"
    return "high"


def budget_regime_config(keep_ratio: float) -> dict[str, float | int]:
    regime = regime_from_ratio(keep_ratio)
    if regime == "low":
        return {
            "frontier_topk": 4,
            "completion_steps": 1,
            "completion_frontier_divisor": 2,
            "section_bundle_topk": 1,
            "connector_bundle_topk": 1,
            "section_seed_topk": 1,
            "section_seed_bundle_size": 2,
            "swap_section_bundle_topk": 1,
            "seed_batch_multiplier": 2,
            "expand_batch_multiplier": 2,
            "drop_topk_limit": 2,
            "utilization_target": 0.72,
            "underfill_bonus": 0.18,
            "overfill_penalty": 1.35,
            "section_penalty": 0.18,
            "connector_bonus": 0.65,
            "neighbor_bonus": 0.04,
        }
    if regime == "mid":
        return {
            "frontier_topk": 6,
            "completion_steps": 2,
            "completion_frontier_divisor": 2,
            "section_bundle_topk": 2,
            "connector_bundle_topk": 2,
            "section_seed_topk": 2,
            "section_seed_bundle_size": 4,
            "swap_section_bundle_topk": 2,
            "seed_batch_multiplier": 3,
            "expand_batch_multiplier": 3,
            "drop_topk_limit": 2,
            "utilization_target": 0.82,
            "underfill_bonus": 0.24,
            "overfill_penalty": 1.10,
            "section_penalty": 0.10,
            "connector_bonus": 0.85,
            "neighbor_bonus": 0.06,
        }
    return {
        "frontier_topk": 8,
        "completion_steps": 3,
        "completion_frontier_divisor": 1,
        "section_bundle_topk": 2,
        "connector_bundle_topk": 2,
        "section_seed_topk": 3,
        "section_seed_bundle_size": 5,
        "swap_section_bundle_topk": 2,
        "seed_batch_multiplier": 4,
        "expand_batch_multiplier": 4,
        "drop_topk_limit": 2,
        "utilization_target": 0.88,
        "underfill_bonus": 0.30,
        "overfill_penalty": 0.90,
        "section_penalty": 0.06,
        "connector_bonus": 0.95,
        "neighbor_bonus": 0.07,
    }


def feature_state(
    selected_row_ids: set[int],
    row_index: dict[int, dict],
    section_index: dict[int, set[int]],
    connector_index: dict[str, set[int]],
) -> dict[str, float | set[str] | set[int]]:
    total_anchor = sum(float(r.get("question_anchor_score", 0.0)) for r in row_index.values()) + 1e-8
    selected_anchor = sum(float(row_index[rid].get("question_anchor_score", 0.0)) for rid in selected_row_ids)

    total_sections = max(1, len(section_index))
    covered_sections = set()
    selected_connectors: set[str] = set()
    bridged_neighbors = 0
    for rid in selected_row_ids:
        row = row_index[rid]
        covered_sections.add(int(row.get("section_idx", -1)))
        selected_connectors |= {str(v).strip().lower() for v in row.get("connector_values", []) if str(v).strip()}
        bridged_neighbors += len(row.get("neighbor_ids", []))

    total_connectors = max(1, len(connector_index))
    selected_connector_ratio = len(selected_connectors) / total_connectors
    section_ratio = len(covered_sections) / total_sections
    anchor_ratio = selected_anchor / total_anchor

    return {
        "anchor_ratio": anchor_ratio,
        "section_ratio": section_ratio,
        "connector_ratio": selected_connector_ratio,
        "covered_sections": covered_sections,
        "selected_connectors": selected_connectors,
        "neighbor_mass": float(bridged_neighbors),
    }


def dynamic_weights(keep_ratio: float, state: dict[str, float | set[str] | set[int]]) -> dict[str, float]:
    regime = regime_from_ratio(keep_ratio)
    anchor_gap = 1.0 - float(state["anchor_ratio"])
    section_gap = 1.0 - float(state["section_ratio"])
    connector_gap = 1.0 - float(state["connector_ratio"])
    if regime == "low":
        base = {
            "anchor": 0.50 + 0.50 * anchor_gap,
            "connector": 0.20 + 0.35 * connector_gap,
            "section": 0.10 + 0.15 * section_gap,
            "repair": 0.20,
            "cost_exp": 1.35,
        }
    elif regime == "mid":
        base = {
            "anchor": 0.35 + 0.35 * anchor_gap,
            "connector": 0.25 + 0.30 * connector_gap,
            "section": 0.20 + 0.25 * section_gap,
            "repair": 0.25,
            "cost_exp": 1.10,
        }
    else:
        base = {
            "anchor": 0.20 + 0.25 * anchor_gap,
            "connector": 0.25 + 0.25 * connector_gap,
            "section": 0.30 + 0.35 * section_gap,
            "repair": 0.30,
            "cost_exp": 0.95,
        }
    return base


def marginal_gain(
    rid: int,
    selected_row_ids: set[int],
    row_index: dict[int, dict],
    section_index: dict[int, set[int]],
    connector_index: dict[str, set[int]],
    keep_ratio: float,
) -> float:
    row = row_index[rid]
    state = feature_state(selected_row_ids, row_index, section_index, connector_index)
    weights = dynamic_weights(keep_ratio, state)

    anchor_gain = float(row.get("question_anchor_score", 0.0)) + 0.4 * float(row.get("combined_score", 0.0))
    section_gain = 1.0 if int(row.get("section_idx", -1)) not in state["covered_sections"] else 0.0
    new_connectors = {
        str(v).strip().lower() for v in row.get("connector_values", []) if str(v).strip()
    } - set(state["selected_connectors"])
    connector_gain = float(len(new_connectors))
    neighbor_gain = 0.0
    for nbr in row.get("neighbor_ids", []):
        nbr = int(nbr)
        if nbr not in selected_row_ids:
            neighbor_gain += 1.0
    repair_gain = neighbor_gain + 0.5 * float(row.get("repair_score", 0.0))

    token_cost = max(1.0, float(row.get("token_len", 1)))
    score = (
        weights["anchor"] * anchor_gain
        + weights["connector"] * connector_gain
        + weights["section"] * section_gain
        + weights["repair"] * repair_gain
    ) / (token_cost ** float(weights["cost_exp"]))
    return score


def candidate_heuristic_score(
    selected_row_ids: set[int],
    row_index: dict[int, dict],
    section_index: dict[int, set[int]],
    connector_index: dict[str, set[int]],
    keep_ratio: float,
    budget_tokens: int,
    token_cost: int | None = None,
) -> float:
    if not selected_row_ids:
        return float("-inf")
    regime_cfg = budget_regime_config(keep_ratio)
    state = feature_state(selected_row_ids, row_index, section_index, connector_index)
    if token_cost is None:
        token_cost = sum(int(row_index[rid].get("token_len", 0)) for rid in selected_row_ids)
    anchor_mass = sum(float(row_index[rid].get("question_anchor_score", 0.0)) for rid in selected_row_ids)
    combined_mass = sum(float(row_index[rid].get("combined_score", 0.0)) for rid in selected_row_ids)
    repair_mass = sum(float(row_index[rid].get("repair_score", 0.0)) for rid in selected_row_ids)
    utilization = min(1.0, token_cost / max(1.0, float(budget_tokens)))
    utilization_gap = abs(utilization - float(regime_cfg["utilization_target"]))
    coverage_bonus = (
        1.8 * float(state["anchor_ratio"])
        + (0.9 - float(regime_cfg["section_penalty"])) * float(state["section_ratio"])
        + float(regime_cfg["connector_bonus"]) * float(state["connector_ratio"])
    )
    completeness_bonus = float(regime_cfg["underfill_bonus"]) * utilization - float(regime_cfg["overfill_penalty"]) * utilization_gap
    return (
        1.3 * anchor_mass
        + 0.35 * combined_mass
        + 0.25 * repair_mass
        + float(regime_cfg["neighbor_bonus"]) * float(state["neighbor_mass"])
        + coverage_bonus
        + completeness_bonus
    ) / (max(1.0, float(token_cost)) ** (0.22 if keep_ratio >= 0.5 else 0.35))


def greedy_complete_row_set(
    initial_rows: set[int],
    row_index: dict[int, dict],
    section_index: dict[int, set[int]],
    connector_index: dict[str, set[int]],
    keep_ratio: float,
    budget_tokens: int,
    section_fixed_cost: dict[int, int],
    frontier_topk: int,
    max_steps: int = 4,
) -> set[int]:
    regime_cfg = budget_regime_config(keep_ratio)
    current = set(initial_rows)
    capped_steps = min(max_steps, int(regime_cfg["completion_steps"]))
    capped_frontier_topk = max(1, min(frontier_topk, int(regime_cfg["frontier_topk"])))
    for _ in range(capped_steps):
        frontier = frontier_rows(
            current,
            row_index,
            section_index,
            connector_index,
            keep_ratio,
            capped_frontier_topk,
        )
        best_rid = None
        best_score = float("-inf")
        for rid in frontier:
            if rid in current:
                continue
            nxt = set(current)
            nxt.add(int(rid))
            token_cost = estimate_row_set_token_cost(nxt, row_index, section_fixed_cost)
            if token_cost > budget_tokens:
                continue
            score = candidate_heuristic_score(
                nxt,
                row_index,
                section_index,
                connector_index,
                keep_ratio,
                budget_tokens,
                token_cost=token_cost,
            )
            if score > best_score:
                best_score = score
                best_rid = int(rid)
        if best_rid is None:
            break
        current.add(best_rid)
    return current


def make_budget_feasible_row_set(
    ordered_row_ids: list[int],
    row_index: dict[int, dict],
    budget_tokens: int,
    section_fixed_cost: dict[int, int],
) -> set[int]:
    current: set[int] = set()
    for rid in ordered_row_ids:
        rid = int(rid)
        if rid not in row_index or rid in current:
            continue
        nxt = set(current)
        nxt.add(rid)
        if estimate_row_set_token_cost(nxt, row_index, section_fixed_cost) <= budget_tokens:
            current.add(rid)
    return current


def select_top_candidate_sets(
    candidate_sets: list[set[int]],
    row_index: dict[int, dict],
    section_index: dict[int, set[int]],
    connector_index: dict[str, set[int]],
    keep_ratio: float,
    budget_tokens: int,
    section_fixed_cost: dict[int, int],
    limit: int,
) -> list[tuple[set[int], int, float]]:
    ranked: list[tuple[float, int, tuple[int, ...], set[int]]] = []
    seen_keys: set[tuple[int, ...]] = set()
    for rows in candidate_sets:
        key = tuple(sorted(rows))
        if key in seen_keys:
            continue
        seen_keys.add(key)
        token_cost = estimate_row_set_token_cost(rows, row_index, section_fixed_cost)
        if token_cost > budget_tokens:
            continue
        heuristic = candidate_heuristic_score(
            rows,
            row_index,
            section_index,
            connector_index,
            keep_ratio,
            budget_tokens,
            token_cost=token_cost,
        )
        ranked.append((heuristic, token_cost, key, rows))
    ranked.sort(key=lambda item: (item[0], -item[1]), reverse=True)
    out: list[tuple[set[int], int, float]] = []
    for heuristic, token_cost, _, rows in ranked[: max(1, limit)]:
        out.append((rows, token_cost, heuristic))
    return out


def build_seed_sets(
    base_selected_ids: set[int],
    row_index: dict[int, dict],
    section_index: dict[int, set[int]],
    connector_index: dict[str, set[int]],
    keep_ratio: float,
    seed_topk: int,
    budget_tokens: int,
    section_fixed_cost: dict[int, int],
) -> list[set[int]]:
    regime_cfg = budget_regime_config(keep_ratio)
    seeds: list[set[int]] = []
    if base_selected_ids:
        base_seed = make_budget_feasible_row_set(
            sorted(base_selected_ids),
            row_index,
            budget_tokens,
            section_fixed_cost,
        )
        if base_seed:
            seeds.append(base_seed)
    scored = []
    for rid in row_index:
        score = marginal_gain(rid, set(), row_index, section_index, connector_index, keep_ratio)
        scored.append((score, rid))
    scored.sort(reverse=True)
    for _, rid in scored[:seed_topk]:
        ordered_seed = [int(rid)]
        row = row_index[rid]
        for nbr in row.get("neighbor_ids", [])[:2]:
            nbr = int(nbr)
            if nbr in row_index:
                ordered_seed.append(nbr)
        sec_idx = int(row.get("section_idx", -1))
        for other in sorted(section_index.get(sec_idx, set()))[:2]:
            ordered_seed.append(int(other))
        for value in row.get("connector_values", [])[:1]:
            linked = sorted(connector_index.get(str(value).strip().lower(), set()))
            for other in linked[:2]:
                ordered_seed.append(int(other))
        seed = make_budget_feasible_row_set(
            ordered_seed,
            row_index,
            budget_tokens,
            section_fixed_cost,
        )
        if not seed:
            continue
        seeds.append(seed)
        completed_seed = greedy_complete_row_set(
            seed,
            row_index,
            section_index,
            connector_index,
            keep_ratio,
            budget_tokens,
            section_fixed_cost,
            frontier_topk=max(1, min(int(regime_cfg["frontier_topk"]), max(2, seed_topk // max(1, int(regime_cfg["completion_frontier_divisor"]))))),
            max_steps=int(regime_cfg["completion_steps"]),
        )
        if completed_seed != seed:
            seeds.append(completed_seed)

    if len(row_index) >= 600:
        column_groups: dict[str, list[int]] = {}
        for rid, row in row_index.items():
            for signature in row.get("column_coverage_signature", set()):
                column_groups.setdefault(str(signature), []).append(int(rid))
        column_ranked: list[tuple[float, str, list[int]]] = []
        for signature, rids in column_groups.items():
            group_score = sum(
                float(row_index[rid].get("column_relevance_score", 0.0))
                + 0.5 * float(row_index[rid].get("question_anchor_score", 0.0))
                for rid in rids
            )
            column_ranked.append((group_score, signature, rids))
        column_ranked.sort(reverse=True)
        for _, _, rids in column_ranked[: max(1, min(seed_topk // 2, int(regime_cfg["section_seed_topk"])))]:
            ranked_rows = sorted(
                rids,
                key=lambda rid: (
                    float(row_index[rid].get("question_anchor_score", 0.0)),
                    float(row_index[rid].get("combined_score", 0.0)),
                    -float(row_index[rid].get("token_len", 1.0)),
                ),
                reverse=True,
            )
            seed = make_budget_feasible_row_set(
                ranked_rows[: max(2, min(int(regime_cfg["section_seed_bundle_size"]), len(ranked_rows)))],
                row_index,
                budget_tokens,
                section_fixed_cost,
            )
            if seed:
                seeds.append(seed)
                completed_seed = greedy_complete_row_set(
                    seed,
                    row_index,
                    section_index,
                    connector_index,
                    keep_ratio,
                    budget_tokens,
                    section_fixed_cost,
                    frontier_topk=max(1, min(int(regime_cfg["frontier_topk"]), max(2, seed_topk // max(1, int(regime_cfg["completion_frontier_divisor"]))))),
                    max_steps=1,
                )
                if completed_seed != seed:
                    seeds.append(completed_seed)

    # Add section-level bundle seeds so the search is not forced to start
    # from single-row neighborhoods only.
    section_scores = []
    for sec_idx, rids in section_index.items():
        sec_score = sum(float(row_index[rid].get("combined_score", 0.0)) for rid in rids)
        section_scores.append((sec_score, sec_idx))
    section_scores.sort(reverse=True)
    for _, sec_idx in section_scores[: max(1, min(seed_topk // 2, int(regime_cfg["section_seed_topk"]))) ]:
        ranked = sorted(
            section_index.get(sec_idx, set()),
            key=lambda rid: (
                float(row_index[rid].get("question_anchor_score", 0.0)),
                float(row_index[rid].get("combined_score", 0.0)),
                -float(row_index[rid].get("token_len", 1.0)),
            ),
            reverse=True,
        )
        bundle = make_budget_feasible_row_set(
            [int(rid) for rid in ranked[: min(int(regime_cfg["section_seed_bundle_size"]), len(ranked))]],
            row_index,
            budget_tokens,
            section_fixed_cost,
        )
        if bundle:
            seeds.append(bundle)
            completed_bundle = greedy_complete_row_set(
                bundle,
                row_index,
                section_index,
                connector_index,
                keep_ratio,
                budget_tokens,
                section_fixed_cost,
                frontier_topk=max(1, min(int(regime_cfg["frontier_topk"]), max(2, seed_topk // max(1, int(regime_cfg["completion_frontier_divisor"]))))),
                max_steps=max(1, int(regime_cfg["completion_steps"]) - 1),
            )
            if completed_bundle != bundle:
                seeds.append(completed_bundle)
    return seeds


def frontier_rows(
    selected_row_ids: set[int],
    row_index: dict[int, dict],
    section_index: dict[int, set[int]],
    connector_index: dict[str, set[int]],
    keep_ratio: float,
    topk: int,
) -> list[int]:
    candidates: set[int] = set()
    if not selected_row_ids:
        candidates = set(row_index.keys())
    else:
        for rid in selected_row_ids:
            row = row_index[rid]
            for nbr in row.get("neighbor_ids", []):
                nbr = int(nbr)
                if nbr in row_index and nbr not in selected_row_ids:
                    candidates.add(nbr)
            for value in row.get("connector_values", []):
                for nbr in connector_index.get(str(value).strip().lower(), set()):
                    if nbr not in selected_row_ids:
                        candidates.add(nbr)
        if not candidates:
            candidates = set(row_index.keys()) - set(selected_row_ids)
    scored = [
        (marginal_gain(rid, selected_row_ids, row_index, section_index, connector_index, keep_ratio), rid)
        for rid in candidates
    ]
    scored.sort(reverse=True)
    return [rid for _, rid in scored[:topk]]


def top_rows_for_section(
    sec_idx: int,
    selected_row_ids: set[int],
    row_index: dict[int, dict],
    section_index: dict[int, set[int]],
    connector_index: dict[str, set[int]],
    keep_ratio: float,
    topk: int,
) -> list[int]:
    rids = [rid for rid in section_index.get(sec_idx, set()) if rid not in selected_row_ids]
    scored = [
        (marginal_gain(rid, selected_row_ids, row_index, section_index, connector_index, keep_ratio), rid)
        for rid in rids
    ]
    scored.sort(reverse=True)
    return [rid for _, rid in scored[:topk]]


def connector_bundle_rows(
    seed_rid: int,
    selected_row_ids: set[int],
    row_index: dict[int, dict],
    connector_index: dict[str, set[int]],
    max_extra: int,
) -> list[int]:
    out: list[int] = []
    seen: set[int] = set()
    for value in row_index[seed_rid].get("connector_values", []):
        for rid in sorted(connector_index.get(str(value).strip().lower(), set())):
            if rid == seed_rid or rid in selected_row_ids or rid in seen:
                continue
            seen.add(rid)
            out.append(rid)
            if len(out) >= max_extra:
                return out
    return out


def removable_rows(
    selected_row_ids: set[int],
    row_index: dict[int, dict],
    section_index: dict[int, set[int]],
    connector_index: dict[str, set[int]],
    keep_ratio: float,
    topk: int,
) -> list[int]:
    if not selected_row_ids:
        return []
    scored = []
    current_score = 0.0
    for rid in selected_row_ids:
        current_score += marginal_gain(rid, selected_row_ids - {rid}, row_index, section_index, connector_index, keep_ratio)
    for rid in selected_row_ids:
        drop_penalty = marginal_gain(rid, selected_row_ids - {rid}, row_index, section_index, connector_index, keep_ratio)
        scored.append((drop_penalty, rid))
    scored.sort()
    return [rid for _, rid in scored[:topk]]


def generate_neighbor_sets(
    cand_rows: set[int],
    row_index: dict[int, dict],
    section_index: dict[int, set[int]],
    connector_index: dict[str, set[int]],
    keep_ratio: float,
    add_topk: int,
    drop_topk: int,
    budget_tokens: int,
    section_fixed_cost: dict[int, int],
) -> list[set[int]]:
    regime_cfg = budget_regime_config(keep_ratio)
    neighbors: list[set[int]] = []

    effective_add_topk = min(add_topk, int(regime_cfg["frontier_topk"]))
    effective_drop_topk = min(drop_topk, int(regime_cfg["drop_topk_limit"]))
    for rid in frontier_rows(cand_rows, row_index, section_index, connector_index, keep_ratio, effective_add_topk):
        nxt = set(cand_rows)
        nxt.add(int(rid))
        neighbors.append(nxt)

        # Section bundle: add the frontier row together with a few strong
        # same-section rows to improve evidence completeness in one step.
        sec_idx = int(row_index[rid].get("section_idx", -1))
        section_bundle = set(nxt)
        for other in top_rows_for_section(
            sec_idx,
            section_bundle,
            row_index,
            section_index,
            connector_index,
            keep_ratio,
            topk=int(regime_cfg["section_bundle_topk"]),
        ):
            section_bundle.add(int(other))
        if section_bundle != nxt:
            neighbors.append(section_bundle)

        # Connector bundle: close a local bridge around the added row.
        connector_bundle = set(nxt)
        for other in connector_bundle_rows(rid, connector_bundle, row_index, connector_index, max_extra=int(regime_cfg["connector_bundle_topk"])):
            connector_bundle.add(int(other))
        if connector_bundle != nxt:
            neighbors.append(connector_bundle)

    for rid in removable_rows(cand_rows, row_index, section_index, connector_index, keep_ratio, effective_drop_topk):
        nxt = set(cand_rows)
        nxt.discard(int(rid))
        if nxt:
            neighbors.append(nxt)

    frontier = frontier_rows(cand_rows, row_index, section_index, connector_index, keep_ratio, effective_add_topk)
    removable = removable_rows(cand_rows, row_index, section_index, connector_index, keep_ratio, effective_drop_topk)
    for drop_rid in removable:
        reduced = set(cand_rows)
        reduced.discard(int(drop_rid))
        for add_rid in frontier:
            if add_rid in reduced:
                continue
            nxt = set(reduced)
            nxt.add(int(add_rid))
            if nxt:
                neighbors.append(nxt)

            sec_idx = int(row_index[add_rid].get("section_idx", -1))
            swap_bundle = set(nxt)
            for other in top_rows_for_section(
                sec_idx,
                swap_bundle,
                row_index,
                section_index,
                connector_index,
                keep_ratio,
                topk=int(regime_cfg["swap_section_bundle_topk"]),
            ):
                swap_bundle.add(int(other))
            if swap_bundle != nxt:
                neighbors.append(swap_bundle)

    dedup: list[set[int]] = []
    seen_local: set[tuple[int, ...]] = set()
    for rows in neighbors:
        key = tuple(sorted(rows))
        if key in seen_local:
            continue
        seen_local.add(key)
        dedup.append(rows)
        completed = greedy_complete_row_set(
            rows,
            row_index,
            section_index,
            connector_index,
            keep_ratio,
            budget_tokens,
            section_fixed_cost,
            frontier_topk=max(1, min(int(regime_cfg["frontier_topk"]), max(2, add_topk // max(1, int(regime_cfg["completion_frontier_divisor"]))))),
            max_steps=int(regime_cfg["completion_steps"]),
        )
        completed_key = tuple(sorted(completed))
        if completed and completed_key not in seen_local:
            seen_local.add(completed_key)
            dedup.append(completed)
    return dedup
