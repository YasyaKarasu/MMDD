import json
import sqlite3
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, TextIO

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import wdc200k_io as wdc200k_io_module  # noqa: E402
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


def test_atomic_shard_guard_runs_before_open_write_and_commit(
    tmp_path: Path,
) -> None:
    calls: list[tuple[Path, int]] = []
    target = tmp_path / "target" / "part.jsonl"
    shard = AtomicJsonlShard(
        target,
        pre_write_guard=lambda path, size=0: calls.append((path, size)),
    )
    shard.write({"id": "a"})
    shard.commit()

    assert calls[0] == (target, 0)
    assert any(path == target and size > 0 for path, size in calls)
    assert calls[-1] == (target, 0)


def test_atomic_shard_guard_failure_before_commit_preserves_checkpoint(
    tmp_path: Path,
) -> None:
    target = tmp_path / "part.jsonl"
    estimated_checks = 0

    def guard(_path: Path, estimated_bytes: int = 0) -> None:
        nonlocal estimated_checks
        if estimated_bytes:
            estimated_checks += 1
            if estimated_checks == 2:
                raise OSError("disk reserve")

    checkpoint = AtomicJsonlShard(
        tmp_path / "part-00000.jsonl",
        pre_write_guard=guard,
        guard_interval_bytes=20,
    )
    checkpoint.write({"id": "checkpoint"})
    committed = checkpoint.commit()
    shard = AtomicJsonlShard(
        target,
        pre_write_guard=guard,
        guard_interval_bytes=20,
    )
    with pytest.raises(OSError, match="disk reserve"):
        shard.write({"id": "next"})
    shard.abort()

    assert validate_completed_shard(committed, tmp_path)
    assert not target.exists()


def test_job_store_initialization_commit_uses_live_guard(
    tmp_path: Path,
) -> None:
    path = tmp_path / "jobs.sqlite3"
    zero_checks = 0

    def reject_commit(_path: Path, estimated_bytes: int = 0) -> None:
        nonlocal zero_checks
        if estimated_bytes == 0:
            zero_checks += 1
            if zero_checks == 2:
                raise OSError("job init reserve exhausted")

    with pytest.raises(OSError, match="job init reserve"):
        SqliteJobStore(path, pre_write_guard=reject_commit)

    with sqlite3.connect(path) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE name = 'jobs'"
        ).fetchone() == (0,)


def test_atomic_shard_amortizes_guard_checks_by_byte_window(
    tmp_path: Path,
) -> None:
    calls: list[int] = []
    shard = AtomicJsonlShard(
        tmp_path / "part.jsonl",
        pre_write_guard=lambda _path, size=0: calls.append(size),
        guard_interval_bytes=30,
    )
    for index in range(5):
        shard.write({"id": index})
    shard.commit()

    positive_write_checks = calls[1:-1]
    assert positive_write_checks == [30, 30]
    assert len(positive_write_checks) < 5


def test_commit_gate_always_rechecks_with_write_window_remaining(
    tmp_path: Path,
) -> None:
    calls: list[int] = []
    tracker = wdc200k_io_module.GuardedWriteTracker(
        tmp_path / "jobs.sqlite3",
        lambda _path, size=0: calls.append(size),
        interval_bytes=16,
    )

    tracker.before_write(1)
    tracker.before_commit(0)
    assert calls == [0, 16, 0]

    tracker.before_write(15)
    tracker.before_commit(0)
    assert calls == [0, 16, 0, 0]


def test_sqlite_enqueue_commit_rechecks_with_window_remaining(
    tmp_path: Path,
) -> None:
    zero_checks = 0

    def reject_commit(_path: Path, estimated: int = 0) -> None:
        nonlocal zero_checks
        if estimated == 0:
            zero_checks += 1
            if zero_checks == 3:
                raise OSError("live commit reserve exhausted")

    path = tmp_path / "jobs.sqlite3"
    store = SqliteJobStore(
        path,
        pre_write_guard=reject_commit,
        guard_interval_bytes=64 * 1024 * 1024,
    )
    with pytest.raises(OSError, match="live commit reserve"):
        store.enqueue("page", "one", {"payload": "x"})

    with sqlite3.connect(path) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM jobs"
        ).fetchone() == (0,)


def test_atomic_commit_does_not_reserve_temporary_bytes_twice(
    tmp_path: Path,
) -> None:
    calls: list[int] = []
    commit_phase = False
    reserve = 1_000
    free = reserve + 1

    def guard(_path: Path, estimated: int = 0) -> None:
        calls.append(estimated)
        if commit_phase and free < reserve + estimated:
            raise OSError("double-counted temporary bytes")

    shard = AtomicJsonlShard(
        tmp_path / "part.jsonl",
        pre_write_guard=guard,
        guard_interval_bytes=64,
    )
    shard.write({"payload": "x" * 200})

    commit_phase = True
    completed = shard.commit()

    assert completed.bytes > 64
    assert calls[-1] == 0


def test_external_merge_guard_fails_midstream_without_touching_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_paths = []
    for index in range(2):
        path = tmp_path / f"run-{index}.jsonl"
        path.write_text(
            "".join(
                json.dumps([str(value), value, {"id": value}]) + "\n"
                for value in range(index, 12, 2)
            ),
            encoding="utf-8",
        )
        run_paths.append(path)
    calls = 0
    monkeypatch.setattr(
        wdc200k_io_module.GuardedWriteTracker,
        "DEFAULT_INTERVAL_BYTES",
        80,
    )

    def guard(_path: Path, estimated: int = 0) -> None:
        nonlocal calls
        if estimated:
            calls += 1
            if calls == 2:
                raise OSError("mid-merge reserve")

    with pytest.raises(OSError, match="mid-merge reserve"):
        wdc200k_io_module._merge_run_group(
            run_paths,
            tmp_path / "merged.jsonl",
            guard,
        )

    assert all(path.is_file() for path in run_paths)


def test_sqlite_job_store_guard_fails_before_next_enqueue_transaction(
    tmp_path: Path,
) -> None:
    checks = 0

    def guard(_path: Path, estimated: int = 0) -> None:
        nonlocal checks
        if estimated:
            checks += 1
            if checks == 3:
                raise OSError("job reserve")

    path = tmp_path / "jobs.sqlite3"
    store = SqliteJobStore(
        path,
        pre_write_guard=guard,
        guard_interval_bytes=5_000,
    )
    store.enqueue("page", "one", {"payload": "x"})
    with pytest.raises(OSError, match="job reserve"):
        store.enqueue("page", "two", {"payload": "y"})

    with sqlite3.connect(path) as connection:
        assert connection.execute(
            "SELECT job_id FROM jobs ORDER BY job_id"
        ).fetchall() == [("one",)]


def test_sqlite_job_store_claim_commit_guard_rolls_back_lease(
    tmp_path: Path,
) -> None:
    path = tmp_path / "jobs.sqlite3"
    SqliteJobStore(path).enqueue("page", "one", {"url": "https://e.test"})
    zero_checks = 0

    def reject_commit(_path: Path, estimated: int = 0) -> None:
        nonlocal zero_checks
        if estimated == 0:
            zero_checks += 1
            if zero_checks == 3:
                raise OSError("claim commit reserve exhausted")

    guarded = SqliteJobStore(
        path,
        pre_write_guard=reject_commit,
        guard_interval_bytes=1,
    )
    with pytest.raises(OSError, match="claim commit reserve"):
        guarded.claim("page", limit=1, owner="worker")

    with sqlite3.connect(path) as connection:
        assert connection.execute(
            "SELECT status, owner FROM jobs WHERE job_id = 'one'"
        ).fetchone() == ("pending", None)
    assert len(
        SqliteJobStore(path).claim("page", limit=1, owner="worker")
    ) == 1


def test_sqlite_job_store_finish_commit_guard_preserves_lease(
    tmp_path: Path,
) -> None:
    path = tmp_path / "jobs.sqlite3"
    store = SqliteJobStore(path)
    store.enqueue("page", "one", {"url": "https://e.test"})
    claimed = store.claim("page", limit=1, owner="worker")[0]
    zero_checks = 0

    def reject_commit(_path: Path, estimated: int = 0) -> None:
        nonlocal zero_checks
        if estimated == 0:
            zero_checks += 1
            if zero_checks == 3:
                raise OSError("finish commit reserve exhausted")

    guarded = SqliteJobStore(
        path,
        pre_write_guard=reject_commit,
        guard_interval_bytes=1,
    )
    with pytest.raises(OSError, match="finish commit reserve"):
        guarded.finish(
            "one",
            status="success",
            owner="worker",
            lease_id=claimed.lease_id,
        )

    with sqlite3.connect(path) as connection:
        assert connection.execute(
            "SELECT status, owner FROM jobs WHERE job_id = 'one'"
        ).fetchone() == ("leased", "worker")
    store.finish(
        "one",
        status="success",
        owner="worker",
        lease_id=claimed.lease_id,
    )


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
        lease_id=claimed[0].lease_id,
    )
    assert store.claim("page", limit=1, owner="worker-2") == []


def test_job_store_does_not_change_success_to_retryable(tmp_path: Path) -> None:
    store = SqliteJobStore(tmp_path / "jobs.sqlite3")
    store.enqueue("page", "url-1", {"url": "https://example.test/a"})
    claimed = store.claim("page", limit=1, owner="worker-1")
    store.finish(
        "url-1",
        status="success",
        owner="worker-1",
        lease_id=claimed[0].lease_id,
    )

    with pytest.raises(RuntimeError, match="active lease"):
        store.finish(
            "url-1",
            status="retryable",
            owner="worker-1",
            lease_id=claimed[0].lease_id,
        )

    assert store.claim("page", limit=1, owner="worker-2") == []


def test_job_store_does_not_change_terminal_to_retryable(tmp_path: Path) -> None:
    store = SqliteJobStore(tmp_path / "jobs.sqlite3")
    store.enqueue("page", "url-1", {"url": "https://example.test/a"})
    claimed = store.claim("page", limit=1, owner="worker-1")
    store.finish(
        "url-1",
        status="terminal",
        owner="worker-1",
        lease_id=claimed[0].lease_id,
    )

    with pytest.raises(RuntimeError, match="active lease"):
        store.finish(
            "url-1",
            status="retryable",
            owner="worker-1",
            lease_id=claimed[0].lease_id,
        )

    assert store.claim("page", limit=1, owner="worker-2") == []


def test_job_store_rejects_a_stale_worker_after_reclaim(tmp_path: Path) -> None:
    store = SqliteJobStore(tmp_path / "jobs.sqlite3")
    store.enqueue("page", "url-1", {"url": "https://example.test/a"})
    original = store.claim(
        "page",
        limit=1,
        owner="worker-1",
        lease_seconds=-1.0,
    )
    reclaimed = store.claim("page", limit=1, owner="worker-2")
    assert [job.job_id for job in reclaimed] == ["url-1"]

    with pytest.raises(RuntimeError, match="active lease"):
        store.finish(
            "url-1",
            status="success",
            owner="worker-1",
            lease_id=original[0].lease_id,
        )

    store.finish(
        "url-1",
        status="success",
        owner="worker-2",
        lease_id=reclaimed[0].lease_id,
    )
    assert store.claim("page", limit=1, owner="worker-3") == []


def test_job_store_fences_same_owner_lease_generations(tmp_path: Path) -> None:
    store = SqliteJobStore(tmp_path / "jobs.sqlite3")
    store.enqueue("page", "url-1", {"url": "https://example.test/a"})
    original = store.claim(
        "page",
        limit=1,
        owner="worker-1",
        lease_seconds=-1.0,
    )[0]
    reclaimed = store.claim("page", limit=1, owner="worker-1")[0]

    assert original.lease_id != reclaimed.lease_id
    with pytest.raises(RuntimeError, match="active lease"):
        store.finish(
            "url-1",
            status="success",
            owner="worker-1",
            lease_id=original.lease_id,
        )

    store.finish(
        "url-1",
        status="success",
        owner="worker-1",
        lease_id=reclaimed.lease_id,
    )


def test_job_store_releases_only_current_owner_leases_for_kind(
    tmp_path: Path,
) -> None:
    store = SqliteJobStore(tmp_path / "jobs.sqlite3")
    for job_id in ("finished", "interrupted", "foreign"):
        store.enqueue("image-kind", job_id, {"url": job_id})
    store.enqueue("other-kind", "other", {"url": "other"})
    current = store.claim(
        "image-kind",
        limit=2,
        owner="current-execution",
        lease_seconds=3600,
    )
    foreign = store.claim(
        "image-kind",
        limit=1,
        owner="foreign-execution",
        lease_seconds=3600,
    )[0]
    other_kind = store.claim(
        "other-kind",
        limit=1,
        owner="current-execution",
        lease_seconds=3600,
    )[0]
    store.finish(
        current[0].job_id,
        status="success",
        owner="current-execution",
        lease_id=current[0].lease_id,
    )

    assert (
        store.release_owner_leases("image-kind", owner="current-execution")
        == 1
    )
    reclaimed = store.claim(
        "image-kind",
        limit=1,
        owner="resumed-execution",
        lease_seconds=3600,
    )

    assert [job.job_id for job in reclaimed] == [current[1].job_id]
    with sqlite3.connect(store.path) as connection:
        assert connection.execute(
            "SELECT status, owner, lease_id FROM jobs WHERE job_id = ?",
            (current[0].job_id,),
        ).fetchone() == ("success", None, None)
        assert connection.execute(
            "SELECT status, owner, lease_id FROM jobs WHERE job_id = ?",
            (foreign.job_id,),
        ).fetchone() == (
            "leased",
            "foreign-execution",
            foreign.lease_id,
        )
        assert connection.execute(
            "SELECT status, owner, lease_id FROM jobs WHERE job_id = ?",
            (other_kind.job_id,),
        ).fetchone() == (
            "leased",
            "current-execution",
            other_kind.lease_id,
        )


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

    original = store.claim(
        "expiring-page",
        limit=1,
        owner="worker-1",
        lease_seconds=-1.0,
    )
    assert [job.job_id for job in original] == ["expired"]
    retry_job = store.claim("expiring-page", limit=1, owner="worker-1")[0]
    assert retry_job.job_id == "expired"
    store.finish(
        "expired",
        status="success",
        result={"ok": True},
        owner="worker-1",
        lease_id=retry_job.lease_id,
    )
    claimed_retry = store.claim("page", limit=1, owner="worker-1")
    assert [job.job_id for job in claimed_retry] == ["retry"]
    store.finish(
        "retry",
        status="retryable",
        result={"error": "busy"},
        owner="worker-1",
        lease_id=claimed_retry[0].lease_id,
    )

    assert [
        job.job_id
        for job in store.claim("page", limit=1, owner="worker-2")
    ] == ["retry"]


def test_enqueue_does_not_reset_a_successful_job(tmp_path: Path) -> None:
    store = SqliteJobStore(tmp_path / "jobs.sqlite3")
    store.enqueue("page", "url-1", {"version": 1})
    claimed = store.claim("page", limit=1, owner="worker-1")
    store.finish(
        "url-1",
        status="success",
        result={"ok": True},
        owner="worker-1",
        lease_id=claimed[0].lease_id,
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


def test_external_unique_jsonl_bounds_pending_run_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = tmp_path / "input.jsonl"
    input_path.write_text(
        "".join(
            json.dumps({"id": f"id-{index:03d}"}) + "\n"
            for index in range(243)
        ),
        encoding="utf-8",
    )
    maximum_pending_paths = 0
    original_add = wdc200k_io_module._RunAccumulator.add

    def tracked_add(
        accumulator: Any,
        run_path: Path,
    ) -> None:
        nonlocal maximum_pending_paths
        original_add(accumulator, run_path)
        maximum_pending_paths = max(
            maximum_pending_paths,
            accumulator.pending_path_count,
        )

    monkeypatch.setattr(
        wdc200k_io_module._RunAccumulator,
        "add",
        tracked_add,
    )

    completed = external_unique_jsonl(
        [input_path],
        tmp_path / "unique.jsonl",
        key_fn=lambda record: record["id"],
        chunk_records=1,
        merge_fan_in=3,
    )

    assert completed.records == 243
    assert maximum_pending_paths <= 12
