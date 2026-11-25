"""Source-bound budget-only lift of the frozen F-MRS-SF search.

This module does not edit the source F-MRS/F-MRS-SF packages. It compiles the
three frozen functions with their sole literal budget ratio replaced, as in
the independently validated B=.10 adapter. Any source drift fails closed.
"""

from __future__ import annotations

import ast
from hashlib import sha256
import inspect
import math
from pathlib import Path
import textwrap
from unittest.mock import patch

from methods.fmrs import exact_budget_search, pipeline
from methods.mrs import candidate_search
from research_fmrs_frontier.method import specificity_frontier


PINNED_PIPELINE_SHA256 = "1ac8e38187a623d97784efc513cb98dafc4e81a58676522a44f98d481929797d"
PINNED_SEARCH_SHA256 = "130b63e8b42ca0dd58656fb57ddef86039ab5adefdc37d3af2c1d82839740b9e"
PIPELINE_PATH = Path(pipeline.__file__).resolve()
SEARCH_PATH = Path(exact_budget_search.__file__).resolve()
_ORIGINAL_V6_RUN_EXAMPLE = candidate_search.v6.run_example


def _compile_at_budget(original, path: Path, expected_hash: str, budget: float):
    if sha256(path.read_bytes()).hexdigest() != expected_hash:
        raise RuntimeError(f"frozen source changed: {path}")
    module = ast.parse(textwrap.dedent(inspect.getsource(original)))
    if len(module.body) != 1 or not isinstance(module.body[0], ast.FunctionDef):
        raise RuntimeError("unexpected frozen function syntax")
    function = module.body[0]
    if function.name != original.__name__:
        raise RuntimeError("frozen function identity changed")
    matches = [node for node in ast.walk(function)
               if isinstance(node, ast.Constant) and type(node.value) is float
               and node.value == .05]
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one budget literal in {function.name}")
    matches[0].value = budget
    ast.fix_missing_locations(module)
    namespace = dict(original.__globals__)
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[function.name]


def run_fmrs_sf_at_budget(example: dict, budget: float, *,
                          max_qwen_candidate_evals: int, runtime) -> dict:
    if budget not in (.05, .10, .50):
        raise ValueError("only preregistered B=.05/.10/.50 are supported")
    if budget == .05:
        # Preserve the exact already-running B=.05 route.
        from research_fmrs_frontier.method import run_variant
        return run_variant(example, .05,
                           max_qwen_candidate_evals=max_qwen_candidate_evals,
                           runtime=runtime)
    run_fmrs = _compile_at_budget(pipeline.run_fmrs, PIPELINE_PATH,
                                  PINNED_PIPELINE_SHA256, budget)
    initial = _compile_at_budget(exact_budget_search._initial_candidates,
                                 SEARCH_PATH, PINNED_SEARCH_SHA256, budget)
    new_state = _compile_at_budget(exact_budget_search._new_state,
                                   SEARCH_PATH, PINNED_SEARCH_SHA256, budget)

    def v6_budget_transport(*args, **kwargs):
        if len(args) != 5 or kwargs:
            raise RuntimeError("unexpected frozen C0 initializer call shape")
        search_args = args[4]
        if search_args.compression_ratio != 1.0 - budget:
            raise RuntimeError("unexpected frozen compression ratio")
        # The C0 entry uses 1-(1-B) in floating point. Nudge only its
        # transport value so exact multiples cannot get an off-by-one K.
        search_args.compression_ratio = math.nextafter(
            search_args.compression_ratio, 0.0)
        return _ORIGINAL_V6_RUN_EXAMPLE(*args)

    with patch.object(exact_budget_search, "_initial_candidates", initial), \
         patch.object(exact_budget_search, "_new_state", new_state), \
         patch.object(exact_budget_search, "frontier_with_witness",
                      specificity_frontier), \
         patch.object(candidate_search.v6, "run_example", v6_budget_transport):
        result = run_fmrs(example, budget,
                          max_qwen_candidate_evals=max_qwen_candidate_evals,
                          runtime=runtime)
    result["method"] = "F-MRS-SF"
    return result
