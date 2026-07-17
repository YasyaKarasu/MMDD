import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from wdc200k_io import (  # noqa: E402
    AtomicJsonlShard,
    StageFingerprint,
    StageManifest,
    SqliteJobStore,
    external_unique_jsonl,
    validate_completed_shard,
)


def test_module_supports_package_import() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from scripts.wdc200k_io import StageFingerprint",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_atomic_shard_is_visible_only_after_commit(tmp_path: Path) -> None:
    shard = AtomicJsonlShard(tmp_path / "part-00000.jsonl")
    shard.write({"id": "a"})
    assert not shard.path.exists()
    record = shard.commit()
    assert shard.path.exists()
    assert record.records == 1
    assert validate_completed_shard(record, root=tmp_path)


def test_job_store_does_not_reclaim_terminal_outcomes(tmp_path: Path) -> None:
    store = SqliteJobStore(tmp_path / "jobs.sqlite3")
    store.enqueue("page", "url-1", {"url": "https://example.test/a"})
    claimed = store.claim("page", limit=1, owner="worker-1")
    assert [job.job_id for job in claimed] == ["url-1"]
    store.finish("url-1", status="terminal", result={"error": "timeout"})
    assert store.claim("page", limit=1, owner="worker-2") == []


def test_completed_shard_validation_detects_content_changes(tmp_path: Path) -> None:
    shard = AtomicJsonlShard(tmp_path / "part-00000.jsonl")
    shard.write({"id": "a"})
    record = shard.commit()

    shard.path.write_text('{"id":"b"}\n', encoding="utf-8")

    assert not validate_completed_shard(record, root=tmp_path)


def test_stage_manifest_persists_progress_and_completion(tmp_path: Path) -> None:
    shard = AtomicJsonlShard(tmp_path / "part-00000.jsonl")
    shard.write({"id": "a"})
    completed_shard = shard.commit()
    fingerprint = StageFingerprint(
        stage="extract",
        input_fingerprint="input-v1",
        parameter_fingerprint="parameters-v1",
    )
    manifest_path = tmp_path / "work" / "extract-manifest.json"

    manifest = StageManifest(manifest_path, fingerprint)
    manifest.record_shard(completed_shard)

    progress = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert progress == {
        "stage": "extract",
        "input_fingerprint": "input-v1",
        "parameter_fingerprint": "parameters-v1",
        "completed_shards": [
            {
                "path": "part-00000.jsonl",
                "records": 1,
                "bytes": completed_shard.bytes,
                "sha256": completed_shard.sha256,
            }
        ],
        "totals": {
            "shards": 1,
            "records": 1,
            "bytes": completed_shard.bytes,
        },
        "complete": False,
    }
    assert not manifest_path.with_suffix(".json.tmp").exists()

    resumed = StageManifest(manifest_path, fingerprint)
    assert resumed.completed_shards == [completed_shard]
    resumed.mark_complete()
    assert json.loads(manifest_path.read_text(encoding="utf-8"))["complete"] is True


def test_stage_manifest_rejects_a_different_fingerprint(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    original = StageFingerprint("extract", "input-v1", "parameters-v1")
    StageManifest(manifest_path, original)

    changed = StageFingerprint("extract", "input-v2", "parameters-v1")

    try:
        StageManifest(manifest_path, changed)
    except ValueError as error:
        assert "fingerprint" in str(error)
    else:
        raise AssertionError("mismatched fingerprints must not resume")


def test_job_store_reclaims_expired_and_retryable_jobs(tmp_path: Path) -> None:
    store = SqliteJobStore(tmp_path / "jobs.sqlite3")
    store.enqueue("expiring-page", "expired", {"attempt": 1})
    store.enqueue("page", "retry", {"attempt": 1})

    assert [
        job.job_id
        for job in store.claim(
            "expiring-page",
            limit=1,
            owner="worker-1",
            lease_seconds=-1.0,
        )
    ] == ["expired"]
    retry_job = store.claim("expiring-page", limit=1, owner="worker-1")[0]
    assert retry_job.job_id == "expired"
    store.finish("expired", status="success", result={"ok": True})
    claimed_retry = store.claim("page", limit=1, owner="worker-1")
    assert [job.job_id for job in claimed_retry] == ["retry"]
    store.finish("retry", status="retryable", result={"error": "busy"})

    assert [
        job.job_id
        for job in store.claim("page", limit=1, owner="worker-2")
    ] == ["retry"]


def test_enqueue_does_not_reset_a_successful_job(tmp_path: Path) -> None:
    store = SqliteJobStore(tmp_path / "jobs.sqlite3")
    store.enqueue("page", "url-1", {"version": 1})
    store.claim("page", limit=1, owner="worker-1")
    store.finish("url-1", status="success", result={"ok": True})

    store.enqueue("page", "url-1", {"version": 2})

    assert store.claim("page", limit=1, owner="worker-2") == []


def test_external_unique_jsonl_keeps_first_record_across_runs(
    tmp_path: Path,
) -> None:
    first_input = tmp_path / "input-1.jsonl"
    second_input = tmp_path / "input-2.jsonl"
    first_input.write_text(
        "\n".join(
            [
                json.dumps({"id": "b", "value": "first b"}),
                json.dumps({"id": "a", "value": "first a"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    second_input.write_text(
        "\n".join(
            [
                json.dumps({"id": "a", "value": "later a"}),
                json.dumps({"id": "c", "value": "first c"}),
                json.dumps({"id": "b", "value": "later b"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    output_path = tmp_path / "final" / "unique.jsonl"

    completed = external_unique_jsonl(
        [first_input, second_input],
        output_path,
        key_fn=lambda record: record["id"],
        chunk_records=2,
    )

    output = [
        json.loads(line)
        for line in output_path.read_text(encoding="utf-8").splitlines()
    ]
    assert output == [
        {"id": "a", "value": "first a"},
        {"id": "b", "value": "first b"},
        {"id": "c", "value": "first c"},
    ]
    assert completed.records == 3
    assert validate_completed_shard(completed, root=output_path.parent)
