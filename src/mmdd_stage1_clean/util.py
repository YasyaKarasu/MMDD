"""Deterministic I/O, hashing and command receipts for CLEAN-R1."""
from __future__ import annotations

import hashlib
import heapq
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

# --------------------------------------------------------------------------
# hashing
# --------------------------------------------------------------------------


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_digest(*parts: Any) -> str:
    payload = json.dumps(
        parts, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
    )
    return sha256_text(payload)


def digest_jsonl(records: Iterable[dict[str, Any]]) -> str:
    """Order-sensitive digest of a JSONL stream (used for semantic hashes)."""
    digest = hashlib.sha256()
    for record in records:
        digest.update(
            json.dumps(
                record, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        )
        digest.update(b"\n")
    return digest.hexdigest()


# --------------------------------------------------------------------------
# deterministic ordering (spec section 3)
# --------------------------------------------------------------------------


def stable_key(namespace: str, value: str) -> str:
    return sha256_text(namespace + "\0" + value)


def stable_order(values: Iterable[str], namespace: str) -> list[str]:
    """Sort by SHA256(namespace NUL id) then by the original id (UTF-8 bytes)."""
    return sorted(set(values), key=lambda v: (stable_key(namespace, v), v.encode("utf-8")))


def byte_order(values: Iterable[str]) -> list[str]:
    return sorted(set(values), key=lambda v: v.encode("utf-8"))


# --------------------------------------------------------------------------
# jsonl
# --------------------------------------------------------------------------


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"JSONL record is not an object: {path}:{number}")
            yield value


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    count = 0
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(
                json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
            )
            count += 1
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return count


def write_json(path: Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


# --------------------------------------------------------------------------
# misc
# --------------------------------------------------------------------------


def utcnow() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def file_metadata(path: Path, root: Path) -> dict[str, Any]:
    path = Path(path)
    try:
        relative = path.relative_to(root).as_posix()
    except ValueError:
        relative = path.as_posix()
    stat = path.stat()
    return {
        "path": relative,
        "bytes": stat.st_size,
        "sha256": sha256_path(path),
    }


def git_commit(repo_root: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def hardware() -> dict[str, Any]:
    info: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
    }
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda"] = torch.version.cuda
        info["cuda_available"] = bool(torch.cuda.is_available())
        info["gpu_count"] = torch.cuda.device_count() if torch.cuda.is_available() else 0
        info["gpu_names"] = [
            torch.cuda.get_device_name(i) for i in range(info["gpu_count"])
        ]
        info["cudnn"] = torch.backends.cudnn.version()
    except Exception as error:  # pragma: no cover - diagnostics only
        info["torch_error"] = repr(error)
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        info["driver"] = out.splitlines()[0] if out else None
    except (OSError, subprocess.CalledProcessError, IndexError):
        info["driver"] = None
    return info


class CommandReceipt:
    """Records one CLI invocation; always written, success or failure."""

    def __init__(self, command: str, argv: list[str], cwd: Path, log_path: Path):
        self.record: dict[str, Any] = {
            "command": command,
            "argv": argv,
            "cwd": str(cwd),
            "start_utc": utcnow(),
            "start_epoch": time.time(),
            "hardware": hardware(),
        }
        self.log_path = log_path

    def finish(self, exit_code: int, **extra: Any) -> dict[str, Any]:
        self.record.update(extra)
        self.record["exit_code"] = exit_code
        self.record["end_utc"] = utcnow()
        self.record["duration_seconds"] = time.time() - self.record["start_epoch"]
        self.record.pop("start_epoch", None)
        path = Path(self.log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(self.record, ensure_ascii=False, sort_keys=True) + "\n"
            )
        return self.record


def log_line(message: str) -> None:
    print(f"[{utcnow()}] {message}", flush=True)


def format_exception() -> str:
    return traceback.format_exc()


def _digest_records(ids, encoded, namespace, blocked):
    """(full 32-byte digest, byte-order rank) for every id not in ``blocked``.

    The digest is ``SHA256(namespace + NUL)`` copied once and then extended with
    the id's UTF-8 bytes, which is the same byte stream as hashing the
    concatenation directly, so no work is repeated per id beyond the hash itself.
    The entire 32-byte digest is compared, so this is the literal ordering rule and
    not a truncated or floating-point approximation of it.
    """
    template = hashlib.sha256((namespace + "\0").encode("utf-8"))
    copy_prefix = template.copy
    for rank, (object_id, object_bytes) in enumerate(zip(ids, encoded)):
        if object_id in blocked:
            continue
        digest = copy_prefix()
        digest.update(object_bytes)
        yield digest.digest(), rank


def stable_order_topk(
    values: Iterable[str],
    namespace: str,
    k: int,
    *,
    blocked: Iterable[str] = (),
    corpus: "SamplingCorpus | None" = None,
) -> list[str]:
    """The first ``k`` items of :func:`stable_order`, as a partial selection.

    Exactly ``stable_order(values, namespace)[:k]``, computed by keeping the ``k``
    smallest (digest, id-rank) records instead of ordering everything.  ``values``
    is only read when no pre-encoded ``corpus`` is supplied, so the hot path avoids
    re-encoding a few hundred thousand ids on every call.
    """
    if k <= 0:
        return []
    if corpus is not None:
        blocked_set = frozenset(blocked)
        chosen = heapq.nsmallest(
            k, _digest_records(corpus.ids, corpus.encoded, namespace, blocked_set)
        )
        return [corpus.ids[rank] for _, rank in chosen]
    unique = sorted(set(values), key=lambda v: v.encode("utf-8"))
    if len(unique) <= k:
        return stable_order(unique, namespace)
    encoded = [value.encode("utf-8") for value in unique]
    chosen = heapq.nsmallest(k, _digest_records(unique, encoded, namespace, _EMPTY_BLOCKED))
    return [unique[rank] for _, rank in chosen]


_EMPTY_BLOCKED: frozenset[str] = frozenset()


_CORPUS_CACHE: dict[int, "SamplingCorpus"] = {}


@dataclass(frozen=True)
class SamplingCorpus:
    """Byte-ordered ids with their encodings, built once per destination corpus."""

    ids: tuple[str, ...]
    encoded: tuple[bytes, ...]
    members: frozenset[str]
    fingerprint: str
    # Identity of the sequence this was built from, so a recycled ``id()`` cannot
    # hand back another corpus's index.
    source: Any = None

    @classmethod
    def build(cls, values: Iterable[str], *, cache: bool = True) -> "SamplingCorpus":
        """Byte-ordered, pre-encoded view of one destination corpus.

        Sorting and encoding a few hundred thousand ids costs more than the
        sampling itself, and the same corpus is requested once per packet per
        query, so the result is memoised on the identity of the sequence.
        """
        if cache:
            key = id(values)
            cached = _CORPUS_CACHE.get(key)
            if cached is not None and cached.source is values:
                return cached
        unique = set(values)
        if any(not isinstance(value, str) for value in unique):
            raise TypeError("Canonical corpus ids must be strings")
        ids = tuple(sorted(unique, key=lambda v: v.encode("utf-8")))
        encoded = tuple(value.encode("utf-8") for value in ids)
        digest = hashlib.sha256()
        for payload in encoded:
            # Length framing applies to this cache fingerprint only, never to the
            # sampling order, so that a corpus cannot be confused with another one
            # whose ids merely concatenate to the same bytes.
            digest.update(len(payload).to_bytes(8, "big"))
            digest.update(payload)
        built = cls(ids, encoded, frozenset(ids), digest.hexdigest(), values)
        if cache:
            _CORPUS_CACHE[id(values)] = built
        return built

    def legal_count(self, blocked: frozenset[str]) -> int:
        if not blocked:
            return len(self.ids)
        return len(self.ids) - sum(1 for value in blocked if value in self.members)

    def legal_ids(self, blocked: frozenset[str]) -> list[str]:
        if not blocked:
            return list(self.ids)
        return [value for value in self.ids if value not in blocked]
