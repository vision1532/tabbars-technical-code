"""Budget-feasible table contexts are states, independent of traversal history."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json


def row_set_hash(rows: tuple[int, ...]) -> str:
    return sha256(json.dumps(rows, separators=(",", ":")).encode()).hexdigest()


def context_hash(context: str) -> str:
    return sha256(context.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class SearchState:
    rows: tuple[int, ...]
    context: str
    exact_tokens: int
    estimated_tokens: int
    heuristic_score: float
    state_hash: str
    serialized_context_hash: str
    parent_state_hash: str | None
    added_row: int | None
    transition_relation_atom: int | None
    parent_row: int | None
    depth: int

    def provenance(self) -> dict:
        return {"parent_state": self.parent_state_hash, "added_row": self.added_row,
                "parent_row": self.parent_row,
                "transition_relation_atom": self.transition_relation_atom,
                "state_hash": self.state_hash, "depth": self.depth}
