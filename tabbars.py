"""Run frozen TabBARS on one question using a caller-supplied evaluation cap."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

import run_merged_jsonl_graph_search_v6 as v6
from methods.mrs.adjudicator import load_runtime
from research_fmrs_sf_final.budget_adapter import run_fmrs_sf_at_budget


def run_tqa(record, *, runtime, budget: float, cap: int) -> dict:
    example = {
        "table": record["table"],
        "question_text": record["question_text"],
        "options": record.get("options", ""),
        "instruction": record.get("instruction", ""),
    }
    return run_fmrs_sf_at_budget(example, budget,
                                  max_qwen_candidate_evals=cap, runtime=runtime)


def run_ltb_turn(sample, table: str, *, mode: str, turn: int, history: list[str],
                 runtime, budget: float, cap: int, model_max_length: int) -> dict:
    from research_ir_fmrs import ltb_interface

    if not 0 <= turn < len(sample["questions"]) or len(history) != turn:
        raise ValueError("turn and prior model outputs must match the conversation")
    safe_sample = {
        "questions": sample["questions"],
        "evidence": sample.get("evidence", ""),
    }
    example = {"table": table, "question_text": safe_sample["questions"][turn],
               "options": "", "instruction": ""}
    evaluator = ltb_interface.make_blind_j0_evaluator(
        safe_sample, mode, turn, history, runtime.tokenizer)

    def answer(_example, context):
        return runtime.answerer.answer_turn(safe_sample, mode, turn, history, context)

    turn_runtime = SimpleNamespace(tokenizer=runtime.tokenizer, llm=runtime.llm,
                                   sampling=runtime.sampling,
                                   answerer=SimpleNamespace(answer=answer))
    with patch.object(ltb_interface, "MAX_MODEL_LEN", model_max_length), \
         patch.object(v6, "evaluate_candidate_sets", evaluator):
        return run_fmrs_sf_at_budget(example, budget,
                                      max_qwen_candidate_evals=cap, runtime=turn_runtime)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark", choices=("tqa", "ltb"))
    parser.add_argument("input", type=Path,
                        help="JSON object with table/question_text (TQA) or questions/db_path/db_id (LTB)")
    parser.add_argument("--model", required=True, help="Local model path or model ID")
    parser.add_argument("--budget", type=float, choices=(.05, .10, .50), required=True)
    parser.add_argument("--cap", type=int, required=True, help="Candidate model evaluation limit")
    parser.add_argument("--model-max-length", type=int, default=16384)
    parser.add_argument("--gpu-memory-utilization", type=float, default=.35)
    parser.add_argument("--ltb-root", type=Path,
                        help="LongTableBench checkout containing eval/ and datasets/")
    parser.add_argument("--mode", choices=("single", "multi"), default="single")
    parser.add_argument("--turn", type=int, default=0, help="Zero-based LongTableBench turn index")
    parser.add_argument("--history", type=Path, help="JSON list of prior raw model outputs")
    args = parser.parse_args()
    if args.cap < 0 or args.model_max_length <= 0:
        parser.error("cap must be nonnegative and model-max-length must be positive")
    if args.benchmark == "ltb" and args.ltb_root is None:
        parser.error("--ltb-root is required for LongTableBench")
    record = json.loads(args.input.read_text(encoding="utf-8"))

    if args.benchmark == "tqa":
        runtime = load_runtime(args.model, max_model_len=args.model_max_length,
                               gpu_memory_utilization=args.gpu_memory_utilization)
        result = run_tqa(record, runtime=runtime, budget=args.budget, cap=args.cap)
    else:
        root = args.ltb_root.resolve()
        sys.path.append(str(root))
        from research_ir_fmrs import ltb_adapter, ltb_interface

        history = json.loads(args.history.read_text(encoding="utf-8")) if args.history else []
        if not isinstance(history, list) or any(not isinstance(value, str) for value in history):
            parser.error("history must be a JSON list of strings")
        with patch.object(ltb_adapter, "TABLES", root / "datasets" / "tables"):
            table = ltb_adapter.load_table(record, args.mode)
        if table.error:
            parser.error(table.error)
        with patch.object(ltb_interface, "MAX_MODEL_LEN", args.model_max_length):
            runtime = ltb_interface.load_ltb_runtime(
                args.model, gpu_memory_utilization=args.gpu_memory_utilization)
        result = run_ltb_turn(record, table.markdown, mode=args.mode, turn=args.turn,
                              history=history, runtime=runtime, budget=args.budget,
                              cap=args.cap, model_max_length=args.model_max_length)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
