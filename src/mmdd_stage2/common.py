"""Small JSON/JSONL helpers shared by the Stage-2 steps."""
from __future__ import annotations

import gzip
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Iterator


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    """Atomic write, so an interrupted step never leaves a half-written output that resume trusts."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=1, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(temporary, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    os.replace(temporary, path)


def digest(value: Any) -> str:
    """Stable content id (pair ids, task ids, bridge ids)."""
    text = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(text.encode()).hexdigest()
