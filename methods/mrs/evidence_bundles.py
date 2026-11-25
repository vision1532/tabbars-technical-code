"""Bounded, label-free structural bundle proposals for the C1/C2 diagnostic.

This module does not score candidates or change the frozen lexical parser or
factual graph. All proposals are checked against the *rendered* token budget
before being handed to the existing candidate ranker.
"""

from __future__ import annotations

import heapq

import run_longtablebench_search_oracle as oracle
import run_merged_jsonl_graph_search_v6 as v6

from .budget import exact_tokens
from .proposal import lexical_proposal


def _key(rows):
    return tuple(sorted(int(r) for r in rows))


class BundleFactory:
    def __init__(self, *, question: str, options: str, prefix: str,
                 sections: list[dict], tokenizer, budget_tokens: int):
        self.prefix, self.sections = prefix, sections
        self.tokenizer, self.budget_tokens = tokenizer, budget_tokens
        self.rows = oracle.build_row_index(sections)
        self.connectors = oracle.build_connector_index(self.rows)
        self.provenance: dict[tuple[int, ...], list[dict]] = {}
        # Consume the exact frozen proposal's matched_terms. In particular,
        # its normalized substring semantics must not be replaced by token
        # equality while forming relation/set groups.
        proposal = lexical_proposal(question, options, sections)
        self.term_rows: dict[str, list[int]] = {}
        for seed in proposal["seeds"]:
            for term in seed["matched_terms"]:
                self.term_rows.setdefault(term, []).append(seed["row_id"])
        max_df = max(8, len(self.rows) // 20)
        ranked_terms = sorted((len(rids), term) for term, rids in self.term_rows.items()
                              if 0 < len(rids) <= max_df)
        anchors = []
        for _, term in ranked_terms:
            rid = min(self.term_rows[term],
                      key=lambda r: (-float(self.rows[r]["question_anchor_score"]), r))
            if rid not in [item[0] for item in anchors]:
                anchors.append((rid, term))
        self.anchors = anchors[:12]

    def _cost(self, rows: set[int]) -> int:
        return exact_tokens(self.tokenizer,
                            v6.render_pruned_table(self.prefix, self.sections, rows))

    def _admit(self, parent: set[int], members: set[int], kind: str,
               source: dict, output: list[set[int]]) -> None:
        child = parent | members
        if not child or child == parent:
            return
        cost = self._cost(child)
        if cost > self.budget_tokens:
            return
        record = {"bundle_type": kind, "member_rows": sorted(members),
                  "source": source, "incremental_token_cost": cost - self._cost(parent),
                  "candidate_exact_tokens": cost, "parent_rows": sorted(parent)}
        key = _key(child)
        if record not in self.provenance.setdefault(key, []):
            self.provenance[key].append(record)
        output.append(child)

    def _cross_section_key(self, rid: int) -> tuple[str, set[int]] | None:
        base_sec = int(self.rows[rid]["section_idx"])
        choices = []
        for value in self.rows[rid].get("connector_values", []):
            value = str(value).strip().lower()
            group = self.connectors.get(value, set())
            if (len(group) >= 2 and rid in group
                    and any(int(self.rows[r]["section_idx"]) != base_sec for r in group)):
                choices.append((len(group), sum(int(self.rows[r]["token_len"]) for r in group),
                                value, group))
        if not choices:
            return None
        _, _, value, group = min(choices)
        return value, set(group)

    def _shortest_path(self, starts: set[int], target: int) -> list[int] | None:
        """Dijkstra with row-token entry cost and stable tie breaking."""
        if target in starts:
            return [target]
        heap = [(0, (rid,), rid) for rid in sorted(starts)]
        heapq.heapify(heap)
        best = {rid: (0, (rid,)) for rid in starts}
        while heap:
            cost, path, rid = heapq.heappop(heap)
            if best.get(rid) != (cost, path):
                continue
            if rid == target:
                return list(path)
            for nbr in self.rows[rid].get("neighbor_ids", []):
                nbr = int(nbr)
                if nbr not in self.rows or nbr in path:
                    continue
                proposal = (cost + int(self.rows[nbr]["token_len"]), path + (nbr,))
                if nbr not in best or proposal < best[nbr]:
                    best[nbr] = proposal
                    heapq.heappush(heap, (*proposal, nbr))
        return None

    def propose(self, parent: set[int], mode: str) -> list[set[int]]:
        if mode not in {"C1", "C2"}:
            raise ValueError(mode)
        output: list[set[int]] = []
        # A canonical shared value is atomic: no partial key group is silently
        # admitted. Prefer links that cross table sections (e.g. entity↔fact).
        roots = sorted(parent) if parent else [rid for rid, _ in self.anchors]
        for rid in roots[:12]:
            keyed = self._cross_section_key(rid)
            if keyed:
                value, group = keyed
                self._admit(parent, group, "shared_key",
                            {"shared_value": value, "anchor_row": rid}, output)

        # Connect a current grounded region to another rare-term grounded
        # region. At the seed stage, start from each of the first two distinct
        # lexical anchors, including their cross-section factual-key closure.
        origins = [parent] if parent else [{rid} for rid, _ in self.anchors[:2]]
        for origin in origins:
            if not origin:
                continue
            source_row = min(origin)
            source_key = self._cross_section_key(source_row)
            base = set(origin) | (source_key[1] if source_key else set())
            for target, term in self.anchors[:12]:
                if target in base:
                    continue
                path = self._shortest_path(base, target)
                if path is None:
                    continue
                target_key = self._cross_section_key(target)
                members = set(base) | set(path) | (target_key[1] if target_key else set())
                self._admit(parent, members, "path",
                            {"path": path, "target_lexical_term": term,
                             "target_row": target,
                             "endpoint_shared_values": [item[0] for item in
                                                        (source_key, target_key) if item]}, output)

        if mode == "C2":
            # Same existing lexical condition in a single table/relation
            # region. No qtype or answer-specific operand rules are used.
            groups = []
            for term, ids in self.term_rows.items():
                by_section: dict[int, set[int]] = {}
                for rid in ids:
                    by_section.setdefault(int(self.rows[rid]["section_idx"]), set()).add(rid)
                for sec, group in by_section.items():
                    if len(group) >= 2:
                        groups.append((len(group), term, sec, group))
            for _, term, sec, group in sorted(groups)[:32]:
                self._admit(parent, group, "relation_set",
                            {"lexical_condition": term, "section_idx": sec}, output)
        return list({_key(rows): rows for rows in output}.values())
