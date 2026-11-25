"""Exact table-text budget checks shared by search and reporting."""

from __future__ import annotations

import math


def exact_tokens(tokenizer, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def table_budget(tokenizer, full_table: str, keep_ratio: float) -> tuple[int, int]:
    if not 0.0 < keep_ratio <= 1.0:
        raise ValueError("budget must be a keep ratio in (0, 1]")
    full = exact_tokens(tokenizer, full_table)
    return full, math.floor(keep_ratio * full)


def require_feasible(tokenizer, rendered: str, budget_tokens: int) -> int:
    cost = exact_tokens(tokenizer, rendered)
    if cost > budget_tokens:
        raise RuntimeError(f"exact rendered table exceeds budget: {cost} > {budget_tokens}")
    return cost
