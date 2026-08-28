from __future__ import annotations

import argparse
import json
import os
import shutil
import sqlite3
import threading
import time
from dataclasses import asdict
from pathlib import Path, PurePosixPath
from typing import Any

from mmdd_progress import progress

from .wdc_evidence import normalize_public_url
from .wdc_runtime import (
    AtomicJsonlShard,
    ShardInfo,
    atomic_write_json,
    check_disk_space,
    iter_jsonl,
    sha256_path,
    valid_shard,
)


CACHE_FORMAT = "mmdd_wdc_legacy_evidence_cache_v1"
OLD_NETWORK_FORMAT = "wdc200k-network-fetch-v1"
NETWORK_LOCATIONS = {
    "page": Path("page_jobs/network"),
    "image": Path("image_jobs/network"),
}


def _safe_relative_path(value: Any) -> Path:
    raw = str(value or "")
    pure = PurePosixPath(raw)
    if not raw or pure.is_absolute() or ".." in pure.parts:
        raise ValueError(f"unsafe relative cache path: {raw!r}")
    return Path(*pure.parts)


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON document is not an object: {path}")
    return value


def _old_network_manifest(work_dir: Path, kind: str) -> tuple[Path, dict[str, Any]]:
    root = work_dir / NETWORK_LOCATIONS[kind]
    path = root / "network-manifest.json"
    manifest = _load_json(path)
    if (
        manifest.get("schema_version") != OLD_NETWORK_FORMAT
        or manifest.get("stage") != "wdc200k_network_fetch"
        or manifest.get("complete") is not True
    ):
        raise ValueError(f"legacy {kind} network stage is not complete: {path}")
    shards = manifest.get("completed_shards")
    if not isinstance(shards, list) or not shards:
        raise ValueError(f"legacy {kind} manifest has no completed shards: {path}")
    for shard in shards:
        _safe_relative_path(shard.get("path"))
    return root, manifest


def _declared_shards(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    return sorted(manifest["completed_shards"], key=lambda item: str(item["path"]))


def inspect_legacy_evidence(work_dir: Path, cache_dir: Path) -> dict[str, Any]:
    work_dir = work_dir.resolve()
    page_root, page_manifest = _old_network_manifest(work_dir, "page")
    image_root, image_manifest = _old_network_manifest(work_dir, "image")
    del page_root, image_root
    page_bytes = sum(int(item["bytes"]) for item in _declared_shards(page_manifest))
    image_outcome_bytes = sum(
        int(item["bytes"]) for item in _declared_shards(image_manifest)
    )
    image_bytes_upper_bound = 0
    for shard in _declared_shards(image_manifest):
        path = work_dir / NETWORK_LOCATIONS["image"] / _safe_relative_path(shard["path"])
        for record in iter_jsonl(path):
            if record.get("status") == "success":
                image_bytes_upper_bound += int(record.get("bytes") or 0)
    cache_parent = cache_dir.resolve().parent
    probe = cache_parent
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    return {
        "work_directory_name": work_dir.name,
        "cache_dir": str(cache_dir),
        "page_outcomes": dict(page_manifest["counts"]),
        "image_outcomes": dict(image_manifest["counts"]),
        "page_shards": len(_declared_shards(page_manifest)),
        "image_shards": len(_declared_shards(image_manifest)),
        "outcome_bytes": page_bytes + image_outcome_bytes,
        "image_bytes_upper_bound": image_bytes_upper_bound,
        "estimated_copy_bytes": page_bytes + image_outcome_bytes + image_bytes_upper_bound,
        "same_filesystem_as_work": work_dir.stat().st_dev == probe.stat().st_dev,
    }


def _validate_source_shard(root: Path, shard: dict[str, Any]) -> Path:
    if not valid_shard(root, shard):
        raise ValueError(f"legacy outcome shard failed validation: {shard.get('path')}")
    return root / _safe_relative_path(shard["path"])


def _publish_file(
    source: Path,
    target: Path,
    *,
    expected_bytes: int,
    expected_sha256: str,
    copy_mode: str,
) -> None:
    if target.exists():
        if (
            target.is_file()
            and target.stat().st_size == expected_bytes
            and sha256_path(target) == expected_sha256
        ):
            return
        raise ValueError(f"existing cache file does not match source: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    temporary.unlink(missing_ok=True)
    use_link = copy_mode == "hardlink" or (
        copy_mode == "auto" and source.stat().st_dev == target.parent.stat().st_dev
    )
    if use_link:
        os.link(source, temporary)
    else:
        with source.open("rb") as source_handle, temporary.open("wb") as target_handle:
            shutil.copyfileobj(source_handle, target_handle, length=1024 * 1024)
            target_handle.flush()
            os.fsync(target_handle.fileno())
        if (
            temporary.stat().st_size != expected_bytes
            or sha256_path(temporary) != expected_sha256
        ):
            temporary.unlink(missing_ok=True)
            raise ValueError(f"copied cache file failed validation: {target}")
    os.replace(temporary, target)


def _image_relative_path(record: dict[str, Any]) -> Path:
    relative = _safe_relative_path(record.get("relative_path"))
    if len(relative.parts) < 2 or relative.parts[0] != "images":
        raise ValueError(f"legacy image path is not under images/: {relative}")
    return relative


def _source_image_path(record: dict[str, Any], image_root: Path | None) -> Path:
    relative = _image_relative_path(record)
    if image_root is not None:
        candidate = image_root / relative
    else:
        candidate = Path(str(record.get("local_path") or ""))
    if not candidate.is_file():
        raise ValueError(f"legacy image file is missing: {candidate}")
    return candidate


def _copy_page_shards(
    source_root: Path,
    source_manifest: dict[str, Any],
    staging: Path,
    *,
    copy_mode: str,
) -> list[ShardInfo]:
    output: list[ShardInfo] = []
    declared = _declared_shards(source_manifest)
    for shard in progress(
        declared, desc="Copy page evidence", unit="shard", leave=False
    ):
        source = _validate_source_shard(source_root, shard)
        target = staging / "page_outcomes" / source.name
        _publish_file(
            source,
            target,
            expected_bytes=int(shard["bytes"]),
            expected_sha256=str(shard["sha256"]),
            copy_mode=copy_mode,
        )
        output.append(
            ShardInfo(
                artifact="page_outcomes",
                path=target.relative_to(staging).as_posix(),
                records=int(shard["records"]),
                bytes=int(shard["bytes"]),
                sha256=str(shard["sha256"]),
            )
        )
    return output


def _rewrite_image_shards(
    source_root: Path,
    source_manifest: dict[str, Any],
    staging: Path,
    *,
    image_root: Path | None,
    copy_mode: str,
) -> list[ShardInfo]:
    output: list[ShardInfo] = []
    declared = _declared_shards(source_manifest)
    for index, shard in progress(
        enumerate(declared),
        total=len(declared),
        desc="Copy image evidence",
        unit="shard",
    ):
        source = _validate_source_shard(source_root, shard)
        writer = AtomicJsonlShard(
            staging / "image_outcomes" / f"part-{index:05d}.jsonl",
            artifact="image_outcomes",
            root=staging,
        )
        try:
            for record in iter_jsonl(source):
                normalized = dict(record)
                if normalized.get("status") == "success":
                    relative = _image_relative_path(normalized)
                    image_source = _source_image_path(normalized, image_root)
                    expected_bytes = int(normalized["bytes"])
                    expected_sha256 = str(normalized["sha256"])
                    if (
                        image_source.stat().st_size != expected_bytes
                        or sha256_path(image_source) != expected_sha256
                    ):
                        raise ValueError(f"legacy image checksum mismatch: {image_source}")
                    _publish_file(
                        image_source,
                        staging / relative,
                        expected_bytes=expected_bytes,
                        expected_sha256=expected_sha256,
                        copy_mode=copy_mode,
                    )
                    normalized["local_path"] = relative.as_posix()
                    normalized["relative_path"] = relative.as_posix()
                writer.write(normalized)
            output.append(writer.commit())
        except BaseException:
            writer.abort()
            raise
    return output


def _remove_sqlite(path: Path) -> None:
    for candidate in (path, Path(str(path) + "-wal"), Path(str(path) + "-shm")):
        candidate.unlink(missing_ok=True)


def _index_url(kind: str, record: dict[str, Any]) -> str | None:
    if record.get("status") != "success":
        return None
    value = record.get("page_url") if kind == "page" else record.get("image_url")
    return normalize_public_url(value)


def _build_lookup(staging: Path, shards: dict[str, list[ShardInfo]]) -> tuple[ShardInfo, dict[str, int]]:
    path = staging / "lookup.sqlite3"
    _remove_sqlite(path)
    connection = sqlite3.connect(path)
    try:
        connection.execute("PRAGMA journal_mode=DELETE")
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute(
            """
            CREATE TABLE evidence (
                kind TEXT NOT NULL,
                url TEXT NOT NULL,
                shard_path TEXT NOT NULL,
                byte_offset INTEGER NOT NULL,
                byte_length INTEGER NOT NULL,
                PRIMARY KEY (kind, url)
            ) WITHOUT ROWID
            """
        )
        connection.execute(
            """
            CREATE TABLE image_files (
                path TEXT PRIMARY KEY,
                bytes INTEGER NOT NULL,
                sha256 TEXT NOT NULL
            ) WITHOUT ROWID
            """
        )
        unsafe = 0
        artifacts = (("page", "page_outcomes"), ("image", "image_outcomes"))
        for kind, artifact in progress(
            artifacts, desc="Index evidence", unit="type", leave=False
        ):
            for shard in progress(
                shards[artifact],
                desc=f"Index {kind} evidence",
                unit="shard",
                leave=False,
            ):
                shard_path = staging / shard.path
                with shard_path.open("rb") as handle:
                    while True:
                        offset = handle.tell()
                        line = handle.readline()
                        if not line:
                            break
                        record = json.loads(line)
                        url = _index_url(kind, record)
                        if record.get("status") == "success" and url is None:
                            unsafe += 1
                            continue
                        if url is None:
                            continue
                        connection.execute(
                            "INSERT OR IGNORE INTO evidence VALUES (?, ?, ?, ?, ?)",
                            (kind, url, shard.path, offset, len(line)),
                        )
                        if kind == "image":
                            relative = _image_relative_path(record).as_posix()
                            connection.execute(
                                "INSERT OR IGNORE INTO image_files VALUES (?, ?, ?)",
                                (relative, int(record["bytes"]), str(record["sha256"])),
                            )
                connection.commit()
        counts = {
            "indexed_page_success": int(
                connection.execute(
                    "SELECT COUNT(*) FROM evidence WHERE kind = 'page'"
                ).fetchone()[0]
            ),
            "indexed_image_success": int(
                connection.execute(
                    "SELECT COUNT(*) FROM evidence WHERE kind = 'image'"
                ).fetchone()[0]
            ),
            "unique_image_files": int(
                connection.execute("SELECT COUNT(*) FROM image_files").fetchone()[0]
            ),
            "unique_image_bytes": int(
                connection.execute("SELECT COALESCE(SUM(bytes), 0) FROM image_files").fetchone()[0]
            ),
            "unsafe_success_records": unsafe,
        }
    finally:
        connection.close()
    return (
        ShardInfo(
            artifact="lookup",
            path=path.relative_to(staging).as_posix(),
            records=counts["indexed_page_success"] + counts["indexed_image_success"],
            bytes=path.stat().st_size,
            sha256=sha256_path(path),
        ),
        counts,
    )


def _source_identity(work_dir: Path, kind: str, manifest: dict[str, Any]) -> dict[str, Any]:
    manifest_path = work_dir / NETWORK_LOCATIONS[kind] / "network-manifest.json"
    return {
        "schema_version": manifest["schema_version"],
        "policy_fingerprint": manifest["policy_fingerprint"],
        "manifest_sha256": sha256_path(manifest_path),
        "counts": dict(manifest["counts"]),
    }


def consolidate_legacy_evidence(
    work_dir: Path,
    cache_dir: Path,
    *,
    image_root: Path | None = None,
    copy_mode: str = "auto",
    reserve_bytes: int = 1_000_000_000,
) -> dict[str, Any]:
    if copy_mode not in {"auto", "hardlink", "copy"}:
        raise ValueError("copy_mode must be auto, hardlink, or copy")
    if work_dir.is_symlink():
        raise ValueError("work directory must be an existing non-symlink directory")
    work_dir = work_dir.resolve()
    cache_dir = cache_dir.resolve()
    if not work_dir.is_dir():
        raise ValueError("work directory must be an existing non-symlink directory")
    if _is_within(cache_dir, work_dir) or _is_within(work_dir, cache_dir):
        raise ValueError("work and evidence cache directories must not contain each other")
    if cache_dir.exists():
        manifest = verify_evidence_cache(cache_dir, verify_images=False)
        if manifest["source"]["work_directory_name"] != work_dir.name:
            raise ValueError("existing cache belongs to a different work directory")
        return manifest

    plan = inspect_legacy_evidence(work_dir, cache_dir)
    cache_dir.parent.mkdir(parents=True, exist_ok=True)
    check_disk_space(cache_dir.parent, int(plan["estimated_copy_bytes"]), reserve_bytes)
    staging = cache_dir.with_name("." + cache_dir.name + ".tmp")
    staging.mkdir(parents=True, exist_ok=True)
    marker_path = staging / "staging.json"
    if marker_path.exists():
        marker = _load_json(marker_path)
        if (
            marker.get("format") != CACHE_FORMAT
            or marker.get("work_directory_name") != work_dir.name
        ):
            raise ValueError(f"staging directory belongs to another operation: {staging}")
    else:
        atomic_write_json(
            marker_path,
            {"format": CACHE_FORMAT, "work_directory_name": work_dir.name},
        )

    page_root, page_manifest = _old_network_manifest(work_dir, "page")
    image_source_root, image_manifest = _old_network_manifest(work_dir, "image")
    page_shards = _copy_page_shards(
        page_root, page_manifest, staging, copy_mode=copy_mode
    )
    image_shards = _rewrite_image_shards(
        image_source_root,
        image_manifest,
        staging,
        image_root=image_root.resolve() if image_root else None,
        copy_mode=copy_mode,
    )
    shard_groups = {
        "page_outcomes": page_shards,
        "image_outcomes": image_shards,
    }
    lookup, indexed_counts = _build_lookup(staging, shard_groups)
    manifest = {
        "format": CACHE_FORMAT,
        "complete": True,
        "source": {
            "work_directory_name": work_dir.name,
            "page": _source_identity(work_dir, "page", page_manifest),
            "image": _source_identity(work_dir, "image", image_manifest),
        },
        "artifacts": {
            artifact: {
                "total_records": sum(item.records for item in shards),
                "shards": [asdict(item) for item in shards],
            }
            for artifact, shards in shard_groups.items()
        },
        "lookup": asdict(lookup),
        "images": {
            "directory": "images",
            "files": indexed_counts["unique_image_files"],
            "bytes": indexed_counts["unique_image_bytes"],
        },
        "counts": indexed_counts,
        "note": "All cache paths are relative; only successful URL outcomes are indexed for reuse.",
    }
    atomic_write_json(staging / "cache_manifest.json", manifest)
    verify_evidence_cache(staging, verify_images=False)
    os.replace(staging, cache_dir)
    return manifest


def _cache_manifest(cache_dir: Path) -> dict[str, Any]:
    manifest = _load_json(cache_dir / "cache_manifest.json")
    if manifest.get("format") != CACHE_FORMAT or manifest.get("complete") is not True:
        raise ValueError(f"evidence cache is not complete: {cache_dir}")
    return manifest


def _manifest_records(manifest: dict[str, Any], artifact: str) -> list[dict[str, Any]]:
    value = manifest.get("artifacts", {}).get(artifact, {}).get("shards")
    if not isinstance(value, list):
        raise ValueError(f"evidence cache is missing {artifact} shards")
    return value


def verify_evidence_cache(
    cache_dir: Path, *, verify_images: bool = True
) -> dict[str, Any]:
    cache_dir = cache_dir.resolve()
    manifest = _cache_manifest(cache_dir)
    for artifact in ("page_outcomes", "image_outcomes"):
        records = _manifest_records(manifest, artifact)
        if not all(valid_shard(cache_dir, record) for record in records):
            raise ValueError(f"evidence cache has an invalid {artifact} shard")
        expected = int(manifest["artifacts"][artifact]["total_records"])
        if sum(int(record["records"]) for record in records) != expected:
            raise ValueError(f"evidence cache has inconsistent {artifact} counts")
    lookup = manifest.get("lookup")
    if not isinstance(lookup, dict) or not valid_shard(cache_dir, lookup):
        raise ValueError("evidence cache lookup index failed validation")
    database = cache_dir / _safe_relative_path(lookup["path"])
    connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    try:
        page_count = int(
            connection.execute(
                "SELECT COUNT(*) FROM evidence WHERE kind = 'page'"
            ).fetchone()[0]
        )
        image_count = int(
            connection.execute(
                "SELECT COUNT(*) FROM evidence WHERE kind = 'image'"
            ).fetchone()[0]
        )
        if page_count != int(manifest["counts"]["indexed_page_success"]):
            raise ValueError("evidence cache page index count mismatch")
        if image_count != int(manifest["counts"]["indexed_image_success"]):
            raise ValueError("evidence cache image index count mismatch")
        image_rows = connection.execute(
            "SELECT path, bytes, sha256 FROM image_files ORDER BY path"
        )
        if verify_images:
            for relative, size, digest in progress(
                image_rows,
                total=int(manifest["counts"]["unique_image_files"]),
                desc="Verify cached images",
                unit="image",
                leave=False,
            ):
                record = {
                    "path": _safe_relative_path(relative).as_posix(),
                    "bytes": int(size),
                    "sha256": str(digest),
                }
                if not valid_shard(cache_dir, record):
                    raise ValueError(f"cached image failed validation: {relative}")
    finally:
        connection.close()
    return manifest


def evidence_cache_identity(cache_dir: Path) -> str:
    cache_dir = cache_dir.resolve()
    manifest = _cache_manifest(cache_dir)
    lookup = manifest.get("lookup")
    if not isinstance(lookup, dict) or not valid_shard(cache_dir, lookup):
        raise ValueError("evidence cache lookup index failed validation")
    return sha256_path(cache_dir / "cache_manifest.json")


class EvidenceReuseCache:
    """Read successful legacy outcomes through a disk-backed URL index."""

    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = cache_dir.resolve()
        manifest = _cache_manifest(self.cache_dir)
        lookup = manifest.get("lookup")
        if not isinstance(lookup, dict) or not valid_shard(self.cache_dir, lookup):
            raise ValueError("evidence cache lookup index failed validation")
        self.database = self.cache_dir / _safe_relative_path(lookup["path"])
        self._local = threading.local()

    def _connect(self) -> sqlite3.Connection:
        connection = getattr(self._local, "connection", None)
        if connection is None:
            connection = sqlite3.connect(
                f"file:{self.database}?mode=ro", uri=True, check_same_thread=False
            )
            self._local.connection = connection
        return connection

    def get(self, kind: str, url: str) -> dict[str, Any] | None:
        normalized = normalize_public_url(url)
        if normalized is None or kind not in {"page", "image"}:
            return None
        row = self._connect().execute(
            "SELECT shard_path, byte_offset, byte_length FROM evidence WHERE kind = ? AND url = ?",
            (kind, normalized),
        ).fetchone()
        if row is None:
            return None
        shard = self.cache_dir / _safe_relative_path(row[0])
        with shard.open("rb") as handle:
            handle.seek(int(row[1]))
            raw = handle.read(int(row[2]))
        record = json.loads(raw)
        if record.get("status") != "success":
            return None
        if kind == "page":
            return {
                "status": "success",
                "url": record.get("final_url") or record["page_url"],
                "text": record.get("text") or "",
                "image_urls": record.get("image_urls") or [],
                "bytes": 0,
                "cache_source": CACHE_FORMAT,
            }
        relative = _image_relative_path(record)
        image_path = self.cache_dir / relative
        if not image_path.is_file() or image_path.stat().st_size != int(record["bytes"]):
            return None
        return {
            "status": "success",
            "url": record.get("final_url") or record["image_url"],
            "cache_path": str(image_path),
            "content_type": record.get("mime_type") or "application/octet-stream",
            "bytes": int(record["bytes"]),
            "cache_source": CACHE_FORMAT,
        }


def delete_legacy_work(
    work_dir: Path,
    cache_dir: Path,
    *,
    confirmation: str,
) -> dict[str, Any]:
    if work_dir.is_symlink():
        raise ValueError("refusing to delete a symlinked work directory")
    work_dir = work_dir.resolve()
    cache_dir = cache_dir.resolve()
    if not work_dir.is_dir() or not work_dir.name.startswith("work_wdc_"):
        raise ValueError("cleanup is restricted to work_wdc_* directories")
    if confirmation != work_dir.name:
        raise ValueError("confirmation must exactly match the work directory name")
    if _is_within(cache_dir, work_dir) or _is_within(work_dir, cache_dir):
        raise ValueError("work and evidence cache directories must not contain each other")
    manifest = verify_evidence_cache(cache_dir, verify_images=True)
    if manifest["source"]["work_directory_name"] != work_dir.name:
        raise ValueError("evidence cache was built from a different work directory")
    removed_name = work_dir.name
    shutil.rmtree(work_dir)
    receipt = {
        "format": "mmdd_wdc_work_cleanup_receipt_v1",
        "work_directory_name": removed_name,
        "cache_manifest_sha256": sha256_path(cache_dir / "cache_manifest.json"),
        "removed_at_unix": time.time(),
    }
    atomic_write_json(cache_dir / "cleanup_receipt.json", receipt)
    return receipt


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Preserve legacy WDC page/image evidence before deleting a work directory."
    )
    result.add_argument("command", choices=("inspect", "consolidate", "verify", "delete-work"))
    result.add_argument("--work-dir")
    result.add_argument("--cache-dir", required=True)
    result.add_argument("--image-root")
    result.add_argument("--copy-mode", choices=("auto", "hardlink", "copy"), default="auto")
    result.add_argument("--min-free-gb", type=float, default=1.0)
    result.add_argument("--confirm-delete-work")
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    cache_dir = Path(args.cache_dir)
    if args.command == "verify":
        payload = verify_evidence_cache(cache_dir, verify_images=True)
    else:
        if not args.work_dir:
            raise SystemExit("--work-dir is required for this command")
        work_dir = Path(args.work_dir)
        if args.command == "inspect":
            payload = inspect_legacy_evidence(work_dir, cache_dir)
        elif args.command == "consolidate":
            payload = consolidate_legacy_evidence(
                work_dir,
                cache_dir,
                image_root=Path(args.image_root) if args.image_root else None,
                copy_mode=args.copy_mode,
                reserve_bytes=int(args.min_free_gb * 1024**3),
            )
        else:
            payload = delete_legacy_work(
                work_dir,
                cache_dir,
                confirmation=args.confirm_delete_work or "",
            )
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0
