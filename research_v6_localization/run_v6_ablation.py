"""Frozen row scoring and candidate token accounting."""

from __future__ import annotations

import time

import run_longtablebench_search_oracle as oracle


def row_score_grounding_only(rid, selected, row_index, section_index, connector_index, keep_ratio):
    row = row_index[rid]
    anchor = float(row.get("question_anchor_score", 0.0))
    combined = float(row.get("combined_score", 0.0))
    return (anchor + .4 * combined) / max(1., float(row.get("token_len", 1))) ** 1.35


def candidate_score_without_graph(rows, row_index, section_index, connector_index,
                                  keep_ratio, budget_tokens, token_cost=None, *, keep_section: bool,
                                  keep_utilization: bool = True):
    if not rows:
        return float("-inf")
    state = oracle.feature_state(rows, row_index, section_index, connector_index)
    if token_cost is None:
        token_cost = sum(int(row_index[r]["token_len"]) for r in rows)
    regime = oracle.budget_regime_config(keep_ratio)
    anchor_mass = sum(float(row_index[r].get("question_anchor_score", 0.0)) for r in rows)
    combined_mass = sum(float(row_index[r].get("combined_score", 0.0)) for r in rows)
    utilization = min(1., token_cost / max(1., float(budget_tokens)))
    numerator = 1.3 * anchor_mass + .35 * combined_mass + 1.8 * float(state["anchor_ratio"])
    if keep_utilization:
        numerator += (float(regime["underfill_bonus"]) * utilization
                      - float(regime["overfill_penalty"]) *
                      abs(utilization - float(regime["utilization_target"])))
    if keep_section:
        numerator += (.9 - float(regime["section_penalty"])) * float(state["section_ratio"])
    return numerator / max(1., float(token_cost)) ** .35


class CountingLLM:
    def __init__(self, llm, tokenizer, trace):
        self.llm, self.tokenizer, self.trace = llm, tokenizer, trace

    def generate(self, prompts, sampling_params):
        if isinstance(prompts, str):
            prompts = [prompts]
        started = time.perf_counter()
        outputs = self.llm.generate(prompts, sampling_params, use_tqdm=False)
        self.trace["qwen_candidate_requests"] += 1
        self.trace["candidate_batch_sizes"].append(len(prompts))
        self.trace["candidate_contexts_sent"] += len(prompts)
        self.trace["candidate_input_tokens"] += sum(
            len(getattr(item, "prompt_token_ids", None) or self.tokenizer.encode(prompt, add_special_tokens=False))
            for prompt, item in zip(prompts, outputs))
        self.trace["candidate_output_tokens"] += sum(
            len(item.outputs[0].token_ids) if item.outputs else 0 for item in outputs)
        self.trace["candidate_qwen_wall_sec"] += time.perf_counter() - started
        return outputs
