"""One gold-blind LongTableBench J0/final-QA interface shared by M0/M1/M2.

The prompt text/turn-history structure is produced by the installed official
LongTableBench prompt module with only its table-loader result substituted by
the selected pruned context. No dataset answer is exposed to this module.
"""

from __future__ import annotations

import ast
import time
from unittest.mock import patch

import torch
from transformers import AutoTokenizer

import run_longtablebench_search_oracle as ltb_v6
import run_merged_jsonl_graph_search_v6 as v6
from eval import prompt as official_prompt
from methods.mrs.adjudicator import MRSRuntime, SharedTQABenchAnswerer


# Qwen3-8B config.json declares max_position_embeddings=40960. All three
# LongTableBench arms use this identical native context window.
MAX_MODEL_LEN = 40960
J0_MAX_NEW_TOKENS = 512
QA_MAX_NEW_TOKENS = 1024


class CandidatePromptTooLong(RuntimeError):
    """Fail closed before sending an over-context candidate to the model."""


def messages_for_turn(sample: dict, mode: str, turn_index: int,
                      previous_raw_outputs: list[str], table_context: str) -> list[dict]:
    questions = sample["questions"]
    if not 0 <= turn_index < len(questions):
        raise ValueError("turn index outside conversation")
    if len(previous_raw_outputs) != turn_index:
        raise ValueError("history must contain exactly prior model outputs")
    if any(not isinstance(q, str) or not q.strip() for q in questions[:turn_index + 1]):
        raise ValueError("empty or missing question")
    safe = {"question": questions[turn_index], "evidence": sample.get("evidence", "") or ""}
    if mode == "multi":
        safe["QA"] = [{"question": q, "output_text": previous_raw_outputs[index]
                       if index < turn_index else ""}
                      for index, q in enumerate(questions[:turn_index + 1])]
    with patch.object(official_prompt, "load_tables", lambda *_args: table_context):
        if mode == "single":
            messages = official_prompt.build_instruction_prompt_chat(
                safe, "single", "markdown", "unused")
        elif mode == "multi":
            messages = official_prompt.build_instruction_prompt_chat_multi(
                safe, "multi", "markdown", "unused", turn_index + 1)
        else:
            raise ValueError(mode)
    if not isinstance(messages, list) or not messages:
        raise RuntimeError("official prompt produced no messages")
    return messages


def render_chat_prompt(tokenizer, messages: list[dict]) -> str:
    return tokenizer.apply_chat_template(messages, tokenize=False,
                                         add_generation_prompt=True,
                                         enable_thinking=False)


def parse_candidate_output(raw: str) -> str:
    """Blind format-validity signal only; does not inspect answer correctness."""
    extracted = ltb_v6.extract_answer(raw)
    if not extracted:
        return ""
    try:
        value = ast.literal_eval(extracted)
    except (SyntaxError, ValueError, TypeError, MemoryError, RecursionError):
        return ""
    if isinstance(value, dict):
        return extracted if value else ""
    if isinstance(value, list):
        return extracted if all(isinstance(item, (str, int, float)) for item in value) else ""
    if isinstance(value, str) and value.strip().casefold() == "unsupported":
        return extracted
    return ""


def make_blind_j0_evaluator(sample: dict, mode: str, turn_index: int,
                            previous_raw_outputs: list[str], tokenizer):
    """V6-compatible signature; no gold/answers argument or metric call."""
    def evaluate(llm, sampling, instruction, question_text, options,
                 table_prefix, sections, selected_row_sets,
                 heuristic_scores=None, selection_mode="blind", completion=None):
        if selection_mode != "blind" or completion is not None:
            raise RuntimeError("LongTableBench J0 must be gold-blind")
        prompts = []
        contexts = []
        for rows in selected_row_sets:
            context = v6.render_pruned_table(table_prefix, sections, rows)
            contexts.append(context)
            messages = messages_for_turn(sample, mode, turn_index,
                                         previous_raw_outputs, context)
            prompt = render_chat_prompt(tokenizer, messages)
            prompt_tokens = len(tokenizer.encode(prompt, add_special_tokens=False))
            if prompt_tokens + J0_MAX_NEW_TOKENS > MAX_MODEL_LEN:
                raise CandidatePromptTooLong(
                    f"LongTableBench J0 input OOC: {prompt_tokens}+"
                    f"{J0_MAX_NEW_TOKENS}>{MAX_MODEL_LEN}; "
                    f"candidate_index={len(prompts)}")
            prompts.append(prompt)
        outputs = llm.generate(prompts, sampling)
        if len(outputs) != len(selected_row_sets):
            raise RuntimeError("J0 output count mismatch")
        candidates = []
        for index, rows in enumerate(selected_row_sets):
            raw = outputs[index].outputs[0].text if outputs[index].outputs else ""
            estimated = sum(
                int(section.get("fixed_token_len", 0)) +
                sum(int(row.get("token_len", 0)) for row in section["rows"]
                    if int(row["row_id"]) in rows)
                for section in sections
                if any(int(row["row_id"]) in rows for row in section["rows"]))
            candidates.append(v6.Candidate(
                selected_row_ids=set(rows),
                heuristic_score=(0.0 if heuristic_scores is None else
                                 float(heuristic_scores[index])),
                token_cost=estimated, raw_output=raw,
                predicted=parse_candidate_output(raw), raw_f1=-1.0))
        return candidates
    return evaluate


class LongTableBenchAnswerer(SharedTQABenchAnswerer):
    def answer_turn(self, sample: dict, mode: str, turn_index: int,
                    previous_raw_outputs: list[str], context: str) -> dict:
        messages = messages_for_turn(sample, mode, turn_index,
                                     previous_raw_outputs, context)
        prompt = render_chat_prompt(self.tokenizer, messages)
        encoded = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        input_tokens = int(encoded.input_ids.shape[1])
        if input_tokens + QA_MAX_NEW_TOKENS > MAX_MODEL_LEN:
            return {"prediction": "", "raw_output": "", "input_tokens": input_tokens,
                    "output_tokens": 0, "latency_sec": 0.0,
                    "model_input_ooc": True}
        started = time.perf_counter()
        with torch.inference_mode():
            output = self.model.generate(
                encoded.input_ids.to(self.device),
                attention_mask=encoded.attention_mask.to(self.device),
                max_new_tokens=QA_MAX_NEW_TOKENS, do_sample=False,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id)
        continuation = output[0, input_tokens:]
        raw = self.tokenizer.decode(continuation, skip_special_tokens=True)
        return {"prediction": raw, "raw_output": raw,
                "input_tokens": input_tokens,
                "output_tokens": int(continuation.shape[0]),
                "latency_sec": time.perf_counter() - started,
                "model_input_ooc": False,
                "parse_valid": bool(parse_candidate_output(raw))}


def load_ltb_runtime(model_path: str, *, gpu_memory_utilization: float = .35) -> MRSRuntime:
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if v6.LLM is None or v6.SamplingParams is None:
        raise ImportError("vLLM is required")
    llm = v6.LLM(model=model_path, tokenizer=model_path, trust_remote_code=True,
                 tensor_parallel_size=1, gpu_memory_utilization=gpu_memory_utilization,
                 max_model_len=MAX_MODEL_LEN)
    sampling = v6.SamplingParams(temperature=0., top_p=1.,
                                  max_tokens=J0_MAX_NEW_TOKENS)
    answerer = LongTableBenchAnswerer(model_path, device="cuda:0")
    return MRSRuntime(tokenizer, llm, sampling, answerer)
