import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, TextIO

import pytest

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
    store.finish(
        "url-1",
        status="terminal",
        result={"error": "timeout"},
        owner="worker-1",
    )
    assert store.claim("page", limit=1, owner="worker-2") == []


def test_job_store_does_not_change_success_to_retryable(tmp_path: Path) -> None:
    store = SqliteJobStore(tmp_path / "jobs.sqlite3")
    store.enqueue("page", "url-1", {"url": "https://example.test/a"})
    store.claim("page", limit=1, owner="worker-1")
    store.finish("url-1", status="success", owner="worker-1")

    with pytest.raises(RuntimeError, match="active lease"):
        store.finish("url-1", status="retryable", owner="worker-1")

    assert store.claim("page", limit=1, owner="worker-2") == []


def test_job_store_does_not_change_terminal_to_retryable(tmp_path: Path) -> None:
    store = SqliteJobStore(tmp_path / "jobs.sqlite3")
    store.enqueue("page", "url-1", {"url": "https://example.test/a"})
    store.claim("page", limit=1, owner="worker-1")
    store.finish("url-1", status="terminal", owner="worker-1")

    with pytest.raises(RuntimeError, match="active lease"):
        store.finish("url-1", status="retryable", owner="worker-1")

    assert store.claim("page", limit=1, owner="worker-2") == []


def test_job_store_rejects_a_stale_worker_after_reclaim(tmp_path: Path) -> None:
    store = SqliteJobStore(tmp_path / "jobs.sqlite3")
    store.enqueue("page", "url-1", {"url": "https://example.test/a"})
    store.claim("page", limit=1, owner="worker-1", lease_seconds=-1.0)
    reclaimed = store.claim("page", limit=1, owner="worker-2")
    assert [job.job_id for job in reclaimed] == ["url-1"]

    with pytest.raises(RuntimeError, match="active lease"):
        store.finish("url-1", status="success", owner="worker-1")

    store.finish("url-1", status="success", owner="worker-2")
    assert store.claim("page", limit=1, owner="worker-3") == []


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


def test_stage_manifest_records_an_identical_shard_once(tmp_path: Path) -> None:
    shard = AtomicJsonlShard(tmp_path / "part-00000.jsonl")
    shard.write({"id": "a"})
    completed_shard = shard.commit()
    manifest = StageManifest(
        tmp_path / "manifest.json",
        StageFingerprint("extract", "input-v1", "parameters-v1"),
    )

    manifest.record_shard(completed_shard)
    manifest.record_shard(completed_shard)

    assert manifest.completed_shards == [completed_shard]
    payload = json.loads(manifest.path.read_text(encoding="utf-8"))
    assert payload["totals"]["shards"] == 1


def test_stage_manifest_rejects_a_conflicting_same_path_shard(
    tmp_path: Path,
) -> None:
    shard = AtomicJsonlShard(tmp_path / "part-00000.jsonl")
    shard.write({"id": "a"})
    completed_shard = shard.commit()
    manifest = StageManifest(
        tmp_path / "manifest.json",
        StageFingerprint("extract", "input-v1", "parameters-v1"),
    )
    manifest.record_shard(completed_shard)
    conflicting = replace(completed_shard, sha256="0" * 64)

    with pytest.raises(ValueError, match="conflicting completed shard"):
        manifest.record_shard(conflicting)

    assert manifest.completed_shards == [completed_shard]


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
    store.finish(
        "expired",
        status="success",
        result={"ok": True},
        owner="worker-1",
    )
    claimed_retry = store.claim("page", limit=1, owner="worker-1")
    assert [job.job_id for job in claimed_retry] == ["retry"]
    store.finish(
        "retry",
        status="retryable",
        result={"error": "busy"},
        owner="worker-1",
    )

    assert [
        job.job_id
        for job in store.claim("page", limit=1, owner="worker-2")
    ] == ["retry"]


def test_enqueue_does_not_reset_a_successful_job(tmp_path: Path) -> None:
    store = SqliteJobStore(tmp_path / "jobs.sqlite3")
    store.enqueue("page", "url-1", {"version": 1})
    store.claim("page", limit=1, owner="worker-1")
    store.finish(
        "url-1",
        status="success",
        result={"ok": True},
        owner="worker-1",
    )

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


def test_external_unique_jsonl_bounds_open_runs_by_merge_fan_in(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = tmp_path / "input.jsonl"
    records = [
        {"id": f"id-{index % 5:02d}", "value": index}
        for index in reversed(range(20))
    ]
    input_path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    original_open = Path.open
    active_run_handles = 0
    maximum_run_handles = 0

    class TrackedRunHandle:
        def __init__(self, handle: TextIO) -> None:
            self._handle = handle
            self._closed = False

        def __iter__(self) -> "TrackedRunHandle":
            return self

        def __next__(self) -> str:
            return next(self._handle)

        def __enter__(self) -> "TrackedRunHandle":
            return self

        def __exit__(self, *_args: Any) -> None:
            self.close()

        def close(self) -> None:
            nonlocal active_run_handles
            if not self._closed:
                self._handle.close()
                self._closed = True
                active_run_handles -= 1

        def __getattr__(self, name: str) -> Any:
            return getattr(self._handle, name)

    def tracked_open(path: Path, *args: Any, **kwargs: Any) -> TextIO:
        nonlocal active_run_handles, maximum_run_handles
        handle = original_open(path, *args, **kwargs)
        mode = str(args[0] if args else kwargs.get("mode", "r"))
        is_temporary_run = (
            mode == "r"
            and (
                path.name.startswith("run-")
                or path.name.startswith("merge-")
            )
        )
        if not is_temporary_run:
            return handle
        active_run_handles += 1
        maximum_run_handles = max(maximum_run_handles, active_run_handles)
        return TrackedRunHandle(handle)  # type: ignore[return-value]

    monkeypatch.setattr(Path, "open", tracked_open)
    output_path = tmp_path / "unique.jsonl"

    completed = external_unique_jsonl(
        [input_path],
        output_path,
        key_fn=lambda record: record["id"],
        chunk_records=1,
        merge_fan_in=3,
    )

    assert completed.records == 5
    output = [
        json.loads(line)
        for line in output_path.read_text(encoding="utf-8").splitlines()
    ]
    assert output == [
        {"id": "id-00", "value": 15},
        {"id": "id-01", "value": 16},
        {"id": "id-02", "value": 17},
        {"id": "id-03", "value": 18},
        {"id": "id-04", "value": 19},
    ]
    assert maximum_run_handles <= 3
    assert active_run_handles == 0
