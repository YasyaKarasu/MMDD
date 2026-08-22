from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mmdd_dataset.wdc_evidence import ResultCache, execute_tasks  # noqa: E402
from mmdd_dataset.wdc_evidence_cache import (  # noqa: E402
    CACHE_FORMAT,
    EvidenceReuseCache,
    consolidate_legacy_evidence,
    delete_legacy_work,
    inspect_legacy_evidence,
    verify_evidence_cache,
)
from mmdd_dataset.wdc_runtime import sha256_path  # noqa: E402


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
            for record in records
        ),
        encoding="utf-8",
    )
    return {
        "path": path.relative_to(path.parents[1]).as_posix(),
        "records": len(records),
        "bytes": path.stat().st_size,
        "sha256": sha256_path(path),
    }


def _write_network_manifest(
    root: Path,
    shard: dict[str, Any],
    *,
    success: int,
    terminal: int,
    policy: str,
) -> None:
    (root / "network-manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "wdc200k-network-fetch-v1",
                "stage": "wdc200k_network_fetch",
                "complete": True,
                "policy_fingerprint": policy,
                "counts": {
                    "unique": success + terminal,
                    "success": success,
                    "terminal": terminal,
                    "pending": 0,
                    "leased": 0,
                },
                "completed_shards": [shard],
            }
        ),
        encoding="utf-8",
    )


def _legacy_fixture(tmp_path: Path) -> tuple[Path, Path, bytes]:
    work = tmp_path / "work_wdc_synthetic"
    image_root = tmp_path / "old-cache"
    image_bytes = b"synthetic-image-bytes"
    image_digest = hashlib.sha256(image_bytes).hexdigest()
    image_path = image_root / "images" / f"image_{image_digest}.png"
    image_path.parent.mkdir(parents=True)
    image_path.write_bytes(image_bytes)

    page_root = work / "page_jobs" / "network"
    page_shard = _write_jsonl(
        page_root / "outcomes" / "part-00000.jsonl",
        [
            {
                "policy_fingerprint": "page-policy",
                "url_key": "page-success",
                "page_url": "https://example.com/item",
                "status": "success",
                "final_url": "https://example.com/item",
                "text": "cached page text",
                "image_urls": ["https://example.com/image.png"],
                "error_class": None,
                "http_status": 200,
                "payload_sha256": "synthetic",
                "affected_reference_count": 1,
            },
            {
                "policy_fingerprint": "page-policy",
                "url_key": "page-terminal",
                "page_url": "https://example.com/retry",
                "status": "terminal",
                "final_url": None,
                "text": None,
                "image_urls": [],
                "error_class": "timeout",
                "http_status": None,
                "payload_sha256": "synthetic-terminal",
                "affected_reference_count": 1,
            },
        ],
    )
    _write_network_manifest(
        page_root, page_shard, success=1, terminal=1, policy="page-policy"
    )

    image_network_root = work / "image_jobs" / "network"
    image_shard = _write_jsonl(
        image_network_root / "outcomes" / "part-00000.jsonl",
        [
            {
                "asset_id": "asset-image-1",
                "asset_type": "image",
                "bytes": len(image_bytes),
                "downloaded": True,
                "entity_id": "entity-1",
                "file_name": image_path.name,
                "final_url": "https://example.com/image.png",
                "height": 1,
                "image_url": "https://example.com/image.png",
                "local_path": str(image_path),
                "mime_type": "image/png",
                "original_url": "https://example.com/image.png",
                "page_url": "https://example.com/item",
                "policy_fingerprint": "image-policy",
                "relative_path": f"images/{image_path.name}",
                "sha256": image_digest,
                "source": "wdc_page_image",
                "status": "success",
                "width": 1,
            },
            {
                "error_class": "download_failed",
                "image_url": "https://example.com/retry.png",
                "original_url": "https://example.com/retry.png",
                "policy_fingerprint": "image-policy",
                "status": "terminal",
            },
        ],
    )
    _write_network_manifest(
        image_network_root,
        image_shard,
        success=1,
        terminal=1,
        policy="image-policy",
    )
    return work, image_root, image_bytes


def test_inspect_is_read_only_and_consolidation_is_self_contained(tmp_path: Path) -> None:
    work, image_root, image_bytes = _legacy_fixture(tmp_path)
    cache = tmp_path / "cache" / "wdc-evidence"

    plan = inspect_legacy_evidence(work, cache)

    assert plan["page_outcomes"]["success"] == 1
    assert plan["image_outcomes"]["success"] == 1
    assert not cache.exists()

    manifest = consolidate_legacy_evidence(
        work,
        cache,
        image_root=image_root,
        copy_mode="copy",
        reserve_bytes=0,
    )

    assert manifest["format"] == CACHE_FORMAT
    assert work.is_dir()
    assert verify_evidence_cache(cache)["complete"] is True
    manifest_text = (cache / "cache_manifest.json").read_text(encoding="utf-8")
    assert str(work) not in manifest_text
    assert str(image_root) not in manifest_text
    assert all(
        not Path(shard["path"]).is_absolute()
        for artifact in manifest["artifacts"].values()
        for shard in artifact["shards"]
    )
    cached_images = list((cache / "images").iterdir())
    assert len(cached_images) == 1
    assert cached_images[0].read_bytes() == image_bytes
    image_record = json.loads(
        (cache / "image_outcomes" / "part-00000.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()[0]
    )
    assert image_record["local_path"].startswith("images/")
    assert not Path(image_record["local_path"]).is_absolute()


def test_consolidated_cache_reuses_successes_but_not_terminal_failures(
    tmp_path: Path,
) -> None:
    work, image_root, _ = _legacy_fixture(tmp_path)
    cache_dir = tmp_path / "cache" / "wdc-evidence"
    consolidate_legacy_evidence(
        work,
        cache_dir,
        image_root=image_root,
        copy_mode="copy",
        reserve_bytes=0,
    )
    reuse_cache = EvidenceReuseCache(cache_dir)
    calls: list[str] = []

    def fetch(url: str) -> dict[str, Any]:
        calls.append(url)
        return {"status": "success", "url": url, "text": "fresh"}

    tasks = [
        {"task_id": "cached", "entity_id": "one", "url": "https://example.com/item"},
        {"task_id": "retry", "entity_id": "two", "url": "https://example.com/retry"},
    ]
    outcomes = execute_tasks(
        tasks,
        kind="page",
        cache=ResultCache(tmp_path / "new-cache.sqlite3"),
        policy="new-policy",
        workers=2,
        fetch=fetch,
        reuse=reuse_cache.get,
    )

    assert calls == ["https://example.com/retry"]
    assert outcomes[0]["text"] == "cached page text"
    assert outcomes[0]["cache_source"] == CACHE_FORMAT
    assert outcomes[1]["text"] == "fresh"
    image = reuse_cache.get("image", "https://example.com/image.png")
    assert image is not None
    assert Path(image["cache_path"]).is_file()


def test_invalid_source_shard_never_publishes_cache_or_deletes_work(tmp_path: Path) -> None:
    work, image_root, _ = _legacy_fixture(tmp_path)
    cache = tmp_path / "cache" / "wdc-evidence"
    page_manifest = work / "page_jobs" / "network" / "network-manifest.json"
    payload = json.loads(page_manifest.read_text(encoding="utf-8"))
    payload["completed_shards"][0]["sha256"] = "0" * 64
    page_manifest.write_text(json.dumps(payload), encoding="utf-8")
    temporary = work / "page_jobs" / "network" / "outcomes" / "part-99999.jsonl.tmp"
    temporary.write_text('{"status":"success"}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="failed validation"):
        consolidate_legacy_evidence(
            work,
            cache,
            image_root=image_root,
            copy_mode="copy",
            reserve_bytes=0,
        )

    assert work.is_dir()
    assert not cache.exists()


def test_delete_work_requires_exact_confirmation_and_validated_cache(tmp_path: Path) -> None:
    work, image_root, _ = _legacy_fixture(tmp_path)
    cache = tmp_path / "cache" / "wdc-evidence"
    consolidate_legacy_evidence(
        work,
        cache,
        image_root=image_root,
        copy_mode="copy",
        reserve_bytes=0,
    )

    with pytest.raises(ValueError, match="confirmation"):
        delete_legacy_work(work, cache, confirmation="wrong-directory")
    assert work.is_dir()

    receipt = delete_legacy_work(work, cache, confirmation=work.name)

    assert not work.exists()
    assert cache.is_dir()
    assert receipt["work_directory_name"] == "work_wdc_synthetic"
    assert (cache / "cleanup_receipt.json").is_file()
    reader = EvidenceReuseCache(cache)
    assert reader.get("page", "https://example.com/item")["text"] == "cached page text"
