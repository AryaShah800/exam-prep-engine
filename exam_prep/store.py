"""Small JSONL helpers. Every step writes plain JSONL so you can open it in any editor."""
import json
from pathlib import Path
from typing import Iterable, TypeVar
from pydantic import BaseModel

M = TypeVar("M", bound=BaseModel)


def write_jsonl(path: Path, rows: Iterable[BaseModel]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(row.model_dump_json() + "\n")
            n += 1
    return n


def append_jsonl(path: Path, rows: Iterable[BaseModel]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(row.model_dump_json() + "\n")
            n += 1
    return n


def read_jsonl(path: Path, model: type[M]) -> list[M]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [model.model_validate(json.loads(line)) for line in f if line.strip()]
