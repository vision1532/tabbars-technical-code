"""Gold-blind, reversible CSV-to-Markdown input adapter for LongTableBench.

Single-turn follows the official first-highlighted-table scope; multi-turn uses
all CSVs in the database. Table file order is sorted for deterministic IDs.
Cell boundaries are preserved by escaping Markdown delimiters and newlines.
The existing proposal, graph, renderer, and exact tokenizer remain unchanged.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path


ROOT = Path("LongTableBench")
QUESTIONS = ROOT / "datasets" / "questions"
TABLES = ROOT / "datasets" / "tables"
SUBSETS = (("8K-S", "8k", "single"), ("8K-M", "8k", "multi"),
           ("32K-S", "32k", "single"), ("32K-M", "32k", "multi"),
           ("Inf-S", "inf", "single"), ("Inf-M", "inf", "multi"))


def checksum(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def encode_cell(value: str) -> str:
    """Bijective within text cells; HTML numeric entities have no `|`/newline."""
    return (value.replace("&", "&amp;").replace("|", "&#124;")
            .replace("\r", "&#13;").replace("\n", "&#10;"))


def decode_cell(value: str) -> str:
    # Decode only this adapter's four entities, once; no generic HTML unescape.
    return (value.replace("&#10;", "\n").replace("&#13;", "\r")
            .replace("&#124;", "|").replace("&amp;", "&"))


@dataclass(frozen=True)
class TableInput:
    markdown: str
    n_rows: int
    n_sections: int
    csv_paths: tuple[str, ...]
    csv_sha256: tuple[str, ...]
    error: str | None


def _safe_directory(sample: dict) -> Path:
    db_path = sample.get("db_path")
    db_id = sample.get("db_id")
    if not isinstance(db_path, str) or not isinstance(db_id, str):
        raise ValueError("missing_db_path_or_id")
    directory = (TABLES / db_path / db_id).resolve()
    if not directory.is_relative_to(TABLES.resolve()):
        raise ValueError("database_path_escape")
    if not directory.is_dir():
        raise ValueError("missing_database_directory")
    return directory


def _csv_paths(sample: dict, mode: str) -> list[Path]:
    directory = _safe_directory(sample)
    files = sorted((p for p in directory.iterdir() if p.is_file() and
                    p.suffix.lower() == ".csv"), key=lambda p: (p.name.casefold(), p.name))
    if mode == "multi":
        if not files:
            raise ValueError("no_csv_tables")
        return files
    if mode != "single":
        raise ValueError("invalid_content_mode")
    highlights = sample.get("highlighted_table")
    if not isinstance(highlights, list) or not highlights or not isinstance(highlights[0], str):
        raise ValueError("missing_highlighted_table")
    name = highlights[0]
    selected = [path for path in files if path.name.casefold() == f"{name}.csv".casefold()]
    if len(selected) != 1:
        raise ValueError("missing_or_ambiguous_highlighted_csv")
    return selected


def _read_csv(path: Path) -> tuple[list[str], list[list[str]]]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.reader(stream, strict=True)
            header = next(reader, None)
            if header is None or not header:
                raise ValueError("empty_csv_header")
            rows = []
            for row in reader:
                if len(row) != len(header):
                    raise ValueError("ragged_csv_row")
                rows.append(row)
    except UnicodeDecodeError as exc:
        raise ValueError("csv_encoding_error") from exc
    except csv.Error as exc:
        raise ValueError("csv_parse_error") from exc
    return header, rows


def load_table(sample: dict, mode: str) -> TableInput:
    try:
        paths = _csv_paths(sample, mode)
    except ValueError as exc:
        return TableInput("", 0, 0, (), (), str(exc))
    blocks = []
    total_rows = 0
    hashes = []
    for path in paths:
        title = path.stem
        if not title or any(ch in title for ch in ("\n", "\r", "|")):
            return TableInput("", total_rows, len(paths), tuple(map(str, paths)),
                              tuple(hashes), "invalid_table_title")
        try:
            header, rows = _read_csv(path)
        except ValueError as exc:
            return TableInput("", total_rows, len(paths), tuple(map(str, paths)),
                              tuple(hashes), str(exc))
        hashes.append(checksum(path))
        lines = [f"## {title}",
                 "| " + " | ".join(encode_cell(value) for value in header) + " |",
                 "| " + " | ".join("---" for _ in header) + " |"]
        for row in rows:
            lines.append("| " + " | ".join(encode_cell(value) for value in row) + " |")
        blocks.append("\n".join(lines))
        total_rows += len(rows)
    if total_rows == 0:
        return TableInput("", 0, len(paths), tuple(map(str, paths)),
                          tuple(hashes), "no_valid_rows")
    return TableInput("\n\n".join(blocks), total_rows, len(paths),
                      tuple(map(str, paths)), tuple(hashes), None)


def safe_questions(source: Path, mode: str) -> list[dict]:
    """Read source and immediately discard every answer/gold field."""
    records = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError("question_source_not_list")
    safe = []
    for index, sample in enumerate(records):
        if not isinstance(sample, dict):
            raise ValueError(f"non_object_conversation:{index}")
        base = {key: sample.get(key) for key in (
            "question_id", "db_path", "db_id", "highlighted_table", "evidence")}
        if mode == "single":
            base["questions"] = [sample.get("question")]
        else:
            rounds = sample.get("QA")
            base["questions"] = [entry.get("question") if isinstance(entry, dict) else None
                                 for entry in rounds] if isinstance(rounds, list) else []
        base["sample_index"] = index
        if any(key in base for key in ("answer", "completion", "gold")):
            raise RuntimeError("gold field escaped safe adapter")
        safe.append(base)
    return safe
