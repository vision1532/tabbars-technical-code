"""Frozen candidate and final-QA model interfaces."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

from transformers import AutoTokenizer
import run_merged_jsonl_graph_search_v6 as v6

_STABLE = Path(__file__).resolve().parents[2] / "research_stable_graph"
if str(_STABLE) not in sys.path:
    sys.path.insert(0, str(_STABLE))
from v0_plus import SharedTQABenchAnswerer  # noqa: E402


@dataclass
class MRSRuntime:
    tokenizer: object
    llm: object | None
    sampling: object | None
    answerer: object


def evaluate_frozen_candidates(evaluator, *args, **kwargs):
    """Pass candidate contexts to the original blind v6 Qwen evaluator.

    The evaluator's prompt, sampling settings, and candidate construction are
    retained; this boundary rejects any oracle label or oracle sorting mode.
    """
    if kwargs.get("selection_mode") != "blind" or kwargs.get("completion") is not None:
        raise ValueError("MRS candidate adjudication must be blind")
    return evaluator(*args, **kwargs)


def load_runtime(model_path: str, *, adjudication: str = "qwen",
                 gpu_memory_utilization: float = .35,
                 max_model_len: int = 16384) -> MRSRuntime:
    """Use the same single-GPU Qwen settings as strict M0_LEXICAL."""
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    llm = sampling = None
    if adjudication == "qwen":
        if v6.LLM is None or v6.SamplingParams is None:
            raise ImportError("vLLM is required for Qwen candidate adjudication")
        llm = v6.LLM(model=model_path, tokenizer=model_path, trust_remote_code=True,
                     tensor_parallel_size=1, gpu_memory_utilization=gpu_memory_utilization,
                     max_model_len=max_model_len)
        sampling = v6.SamplingParams(temperature=0., top_p=1., max_tokens=512)
    answerer = SharedTQABenchAnswerer(model_path, device="cuda:0")
    return MRSRuntime(tokenizer, llm, sampling, answerer)


def final_answer(answerer, example: dict, context: str) -> dict:
    """The answerer receives no label, even when the source record has one."""
    safe = dict(example)
    safe.pop("completion", None)
    safe["meta"] = {k: v for k, v in example.get("meta", {}).items()
                    if k not in {"rightIdx", "qtype"}}
    return answerer.answer(safe, context)
