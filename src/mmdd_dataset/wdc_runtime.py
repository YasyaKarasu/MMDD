from __future__ import annotations

import hashlib
import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, TypeVar


T = TypeVar("T")
R = TypeVar("R")


@dataclass(frozen=True)
class ShardInfo:
    artifact: str
    path: str
    records: int
    bytes: int
    sha256: str


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_digest(*parts: Any) -> str:
    payload = json.dumps(
        parts,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


class AtomicJsonlShard:
    """A JSONL shard is visible only after a durable atomic commit."""

    def __init__(self, path: Path, *, artifact: str, root: Path) -> None:
        self.path = path
        self.root = root
        self.artifact = artifact
        self.temporary_path = path.with_name(path.name + ".tmp")
        path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.temporary_path.open("w", encoding="utf-8")
        self._records = 0

    def write(self, record: dict[str, Any]) -> None:
        if self._handle.closed:
            raise RuntimeError("cannot write to a closed shard")
        self._handle.write(
            json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
        )
        self._records += 1

    def commit(self) -> ShardInfo:
        self._handle.flush()
        os.fsync(self._handle.fileno())
        self._handle.close()
        size = self.temporary_path.stat().st_size
        digest = sha256_path(self.temporary_path)
        os.replace(self.temporary_path, self.path)
        return ShardInfo(
            artifact=self.artifact,
            path=self.path.relative_to(self.root).as_posix(),
            records=self._records,
            bytes=size,
            sha256=digest,
        )

    def abort(self) -> None:
        if not self._handle.closed:
            self._handle.close()


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"JSONL record is not an object: {path}:{line_number}")
            yield value


def valid_shard(root: Path, value: dict[str, Any]) -> bool:
    path = root / str(value.get("path", ""))
    if not path.is_file() or path.name.endswith(".tmp"):
        return False
    try:
        return (
            path.stat().st_size == int(value["bytes"])
            and sha256_path(path) == value["sha256"]
        )
    except (KeyError, OSError, TypeError, ValueError):
        return False


def stage_manifest(
    stage: str,
    *,
    parameters: dict[str, Any],
    random_seed: int,
    input_fingerprint: str,
) -> dict[str, Any]:
    return {
        "format": "mmdd_wdc_stage_v1",
        "stage": stage,
        "parameters": parameters,
        "random_seed": random_seed,
        "input_fingerprint": input_fingerprint,
        "parameter_fingerprint": stable_digest(parameters, random_seed),
        "outputs": {},
        "counts": {},
        "complete": False,
    }


def load_stage_manifest(
    root: Path,
    stage: str,
    *,
    parameters: dict[str, Any],
    random_seed: int,
    input_fingerprint: str,
    shard_root: Path | None = None,
) -> dict[str, Any]:
    path = root / "manifest.json"
    expected = stage_manifest(
        stage,
        parameters=parameters,
        random_seed=random_seed,
        input_fingerprint=input_fingerprint,
    )
    if not path.exists():
        return expected
    current = json.loads(path.read_text(encoding="utf-8"))
    identity_fields = (
        "format",
        "stage",
        "random_seed",
        "input_fingerprint",
        "parameter_fingerprint",
    )
    if any(current.get(field) != expected[field] for field in identity_fields):
        raise ValueError(
            f"stage configuration changed for {stage}; use a different work directory"
        )
    original_outputs = current.get("outputs", {})
    outputs: dict[str, list[dict[str, Any]]] = {}
    validation_root = root if shard_root is None else shard_root
    for artifact, shards in original_outputs.items():
        outputs[artifact] = [
            shard for shard in shards if valid_shard(validation_root, shard)
        ]
    current["outputs"] = outputs
    if current.get("complete") and outputs != original_outputs:
        current["complete"] = False
    return current


def publish_stage_manifest(root: Path, manifest: dict[str, Any]) -> None:
    atomic_write_json(root / "manifest.json", manifest)


def add_stage_shard(
    root: Path,
    manifest: dict[str, Any],
    shard: ShardInfo,
) -> None:
    outputs = manifest.setdefault("outputs", {})
    records = outputs.setdefault(shard.artifact, [])
    payload = asdict(shard)
    for index, existing in enumerate(records):
        if existing["path"] == shard.path:
            records[index] = payload
            break
    else:
        records.append(payload)
    records.sort(key=lambda item: item["path"])
    manifest["counts"][shard.artifact] = sum(
        int(item["records"]) for item in records
    )
    publish_stage_manifest(root, manifest)


def manifest_shards(
    manifest_path: Path,
    artifact: str,
    *,
    root: Path | None = None,
) -> tuple[Path, ...]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("complete") is not True:
        raise ValueError(f"stage is incomplete: {manifest_path}")
    shard_root = manifest_path.parent if root is None else root
    records = payload.get("outputs", {}).get(artifact, [])
    if not all(valid_shard(shard_root, record) for record in records):
        raise ValueError(f"artifact has an invalid shard: {artifact}")
    return tuple(shard_root / record["path"] for record in records)


def iter_manifest_artifact(
    manifest_path: Path,
    artifact: str,
    *,
    root: Path | None = None,
) -> Iterator[dict[str, Any]]:
    for path in manifest_shards(manifest_path, artifact, root=root):
        yield from iter_jsonl(path)


def check_disk_space(path: Path, estimated_bytes: int, reserve_bytes: int) -> None:
    probe = path.resolve()
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    free = shutil.disk_usage(probe).free
    required = max(0, estimated_bytes) + max(0, reserve_bytes)
    if free < required:
        raise OSError(
            f"insufficient disk space for {path}: free={free}, required={required}"
        )


def bounded_map(
    function: Callable[[T], R],
    values: Iterable[T],
    *,
    workers: int,
    max_pending: int | None = None,
) -> Iterator[R]:
    """Map in deterministic input order with a bounded number of futures."""
    worker_count = max(1, int(workers))
    batch_size = max(worker_count, int(max_pending or worker_count * 2))
    iterator = iter(values)
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        while True:
            batch: list[T] = []
            for _ in range(batch_size):
                try:
                    batch.append(next(iterator))
                except StopIteration:
                    break
            if not batch:
                return
            futures = [executor.submit(function, value) for value in batch]
            for future in futures:
                yield future.result()


def input_tree_fingerprint(paths: Iterable[Path], root: Path) -> str:
    digest = hashlib.sha256()
    for path in paths:
        stat = path.stat()
        try:
            relative = path.relative_to(root).as_posix()
        except ValueError:
            relative = path.name
        digest.update(
            json.dumps(
                [relative, stat.st_size, stat.st_mtime_ns],
                separators=(",", ":"),
            ).encode("utf-8")
        )
    return digest.hexdigest()


def iter_dataset_artifact(output_dir: Path, artifact: str) -> Iterator[dict[str, Any]]:
    """Read a final dataset using only paths relative to its own manifest."""
    manifest_path = output_dir / "dataset_manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("complete") is not True:
        raise ValueError(f"dataset is incomplete: {output_dir}")
    shards = payload.get("artifacts", {}).get(artifact, {}).get("shards", [])
    if not all(valid_shard(output_dir, shard) for shard in shards):
        raise ValueError(f"dataset artifact has an invalid shard: {artifact}")
    for shard in shards:
        yield from iter_jsonl(output_dir / shard["path"])
