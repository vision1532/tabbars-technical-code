"""Frozen table question-answering runtime."""

from __future__ import annotations

import json
import re
import time
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_DEFAULT = "Qwen/Qwen3-8B"


SYSTEM = "Select the correct answer. Return only one letter: A, B, C, or D. Do not provide explanation."


MAX_ANSWER_TOKENS = 16


MAX_PROJECTION_TOKENS = 96


class SharedTQABenchAnswerer:
    def __init__(self, model_path: str = MODEL_DEFAULT, device: str = "cuda:0") -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map={"": device},
            trust_remote_code=True, low_cpu_mem_usage=True).eval()
        self.device = device

    def generate(self, system: str, user: str, max_new_tokens: int) -> dict[str, Any]:
        prompt = self.tokenizer.apply_chat_template(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
        encoded = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        input_ids = encoded.input_ids.to(self.device)
        attention_mask = encoded.attention_mask.to(self.device)
        started = time.perf_counter()
        with torch.inference_mode():
            output = self.model.generate(input_ids, attention_mask=attention_mask,
                max_new_tokens=max_new_tokens, do_sample=False,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id)
        latency = time.perf_counter() - started
        continuation = output[0, input_ids.shape[1]:]
        raw = self.tokenizer.decode(continuation, skip_special_tokens=True)
        return {"raw_output": raw, "input_tokens": int(input_ids.shape[1]),
                "output_tokens": int(continuation.shape[0]), "latency_sec": latency,
                "hit_token_cap": int(continuation.shape[0]) >= max_new_tokens,
                "chat_prompt": prompt}

    def answer(self, ex: dict[str, Any], table_text: str) -> dict[str, Any]:
        # The dataset's legacy instruction explicitly requests step-by-step reasoning;
        # it conflicts with this shared short-answer evaluator and is not included.
        user = "\n\n".join((table_text, str(ex.get("question_text", "")),
                             str(ex.get("options", "")), SYSTEM))
        result = self.generate(SYSTEM, user, MAX_ANSWER_TOKENS)
        raw = result["raw_output"].strip()
        match = re.fullmatch(r"(?:\*\*)?([ABCD])(?:\*\*)?[.\s]*", raw, re.IGNORECASE)
        result["prediction"] = match.group(1).upper() if match else ""
        result["parse_failure"] = not bool(match)
        result.pop("chat_prompt")
        return result

    def project(self, ex: dict[str, Any], table: Any) -> dict[str, Any]:
        columns = [(section.section_id, section.table_name, col) for section in table.sections for col in section.columns]
        if not columns:
            return {"projection_success": False, "reason": "no_columns", "ids": [], "columns": [], "raw_output": ""}
        enumeration = "\n".join(f"{i}: {name}.{col}" for i, (_, name, col) in enumerate(columns))
        user = ("Question: " + str(ex.get("question_text", "")) + "\nOptions: " + str(ex.get("options", "")) +
                "\nSelect relevant columns by integer ID only. Return one JSON integer list such as [0, 2].\n" + enumeration)
        try:
            result = self.generate("Return only one JSON list of integer IDs. No explanation.", user, MAX_PROJECTION_TOKENS)
            raw = result["raw_output"].strip()
            # Accept precisely one list, optionally surrounded by a markdown code fence.
            raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.IGNORECASE).strip()
            parsed = json.loads(raw)
            if not isinstance(parsed, list) or any(type(v) is not int for v in parsed):
                raise ValueError("not a strict integer list")
            ids = sorted({v for v in parsed if 0 <= v < len(columns)})
            if not ids:
                raise ValueError("no valid column IDs")
            result.update({"projection_success": True, "reason": "ok", "ids": ids,
                           "columns": [columns[v] for v in ids]})
        except Exception as exc:
            result = locals().get("result", {"raw_output": "", "latency_sec": 0, "input_tokens": 0,
                                              "output_tokens": 0, "hit_token_cap": False})
            result.update({"projection_success": False, "reason": str(exc), "ids": [], "columns": []})
        result.pop("chat_prompt", None)
        return result
