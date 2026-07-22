import argparse
import hashlib
import json
import sqlite3
import sys
import threading
import time
import traceback
from collections import Counter
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from build_mm_joinability_dataset import (
    ExtractionCache,
    TransientModelEndpointError,
)
from wdc200k_io import AtomicJsonlShard, SqliteJobStore
import wdc200k_models as models
from wdc200k_models import (
    AssetStageBarrier,
    MODEL_PARSER_SCHEMA_VERSION,
    ModelStageAuthority,
    adapt_model_tasks_from_manifests,
    enqueue_model_tasks,
    enqueue_model_tasks_from_manifest,
    iter_assets_from_materialization_manifest,
    marker_matches,
    model_adapter_input_fingerprint,
    run_model_stage,
    validate_adapted_model_tasks,
    validate_model_stage,
    validate_model_stage_for_adapter,
    write_model_done_marker,
    write_model_start_marker,
)


def model_args(
    *,
    text_model_name: str = "text-v1",
    image_model_name: str = "image-v1",
) -> argparse.Namespace:
    return argparse.Namespace(
        text_model_name=text_model_name,
        image_model_name=image_model_name,
    )


def test_model_schema_initialization_commit_uses_live_guard(
    tmp_path: Path,
) -> None:
    path = tmp_path / "models.sqlite3"
    SqliteJobStore(path)
    zero_checks = 0

    def reject_commit(_path: Path, estimated_bytes: int = 0) -> None:
        nonlocal zero_checks
        if estimated_bytes == 0:
            zero_checks += 1
            if zero_checks == 2:
                raise OSError("model schema reserve exhausted")

    with pytest.raises(OSError, match="model schema reserve"):
        models._initialize_tables(
            path,
            pre_write_guard=reject_commit,
        )

    with sqlite3.connect(path) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM sqlite_master "
            "WHERE name = 'model_jobsets'"
        ).fetchone() == (0,)


def test_model_jobset_initial_commit_uses_store_live_guard(
    tmp_path: Path,
) -> None:
    path = tmp_path / "models.sqlite3"
    zero_checks = 0

    def reject_jobset_commit(
        _path: Path,
        estimated_bytes: int = 0,
    ) -> None:
        nonlocal zero_checks
        if estimated_bytes == 0:
            zero_checks += 1
            if zero_checks == 5:
                raise OSError("model jobset reserve exhausted")

    store = SqliteJobStore(path, pre_write_guard=reject_jobset_commit)
    with pytest.raises(OSError, match="model jobset reserve"):
        enqueue_model_tasks(
            [asset("guarded")],
            store,
            args=model_args(),
        )

    with sqlite3.connect(path) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM model_jobsets"
        ).fetchone() == (0,)


def asset(
    asset_id: str,
    asset_type: str = "text",
    *,
    content: str | None = None,
    sha256: str | None = None,
) -> dict:
    record = {
        "asset_id": asset_id,
        "asset_type": asset_type,
        "entity_id": f"entity-{asset_id}",
        "entity_wiki_title": f"Entity {asset_id}",
        "entity_text": f"Entity {asset_id}",
        "candidate_attribute_names": ["State"],
        "source_table_id": "table-1",
        "source_row_id": 0,
        "entity_column_index": 0,
        "entity_column_name": "Name",
    }
    if asset_type == "text":
        record["content"] = content or f"{asset_id} is in Alabama."
    else:
        record.update(
            {
                "image_url": f"https://images.test/{asset_id}.jpg",
                "sha256": sha256 or asset_id * 8,
            }
        )
    return record


def task5_fingerprint(
    *,
    input_fingerprint: str = "assets-v1",
) -> dict:
    return {
        "input_fingerprint": input_fingerprint,
        "schema_version": "wdc200k-asset-materialization-v1",
        "planning_manifest_sha256": "1" * 64,
        "unique_job_manifest_sha256": "2" * 64,
        "unique_job_sha256": "3" * 64,
        "image_fetch_manifest_sha256": "4" * 64,
        "image_policy_fingerprint": "image-policy-v1",
        "image_outcome_digest": "5" * 64,
        "image_outcome_count": 1,
        "image_outcome_url_key_digest": "6" * 64,
        "attempts_per_entity": 3,
        "retained_per_entity": 3,
        "text_asset_chunk_chars": 800,
        "min_text_asset_chunk_chars": 120,
        "max_text_asset_chunks_per_entity": 3,
        "records_per_shard": 10_000,
    }


def task5_barrier(
    *,
    fingerprint: dict | None = None,
    bridge_assets: int = 1,
    table_asset_links: int = 1,
) -> AssetStageBarrier:
    return AssetStageBarrier(
        fingerprint=fingerprint or task5_fingerprint(),
        bridge_assets=bridge_assets,
        table_asset_links=table_asset_links,
    )


def write_strict_upstream_barriers(
    root: Path,
) -> tuple[Path, Path]:
    from dataclasses import asdict
    from wdc200k_io import AtomicJsonlShard

    network_root = root / "network"
    network_writer = AtomicJsonlShard(
        network_root / "outcomes" / "part-00000.jsonl"
    )
    network_writer.write({"status": "success", "url": "https://example.test"})
    network_shard = asdict(network_writer.commit())
    network_shard["path"] = "outcomes/part-00000.jsonl"
    network = network_root / "network-manifest.json"
    network.write_text(
        json.dumps(
            {
                "stage": "wdc200k_network_fetch",
                "schema_version": "wdc200k-network-fetch-v1",
                "policy_fingerprint": "network-policy-v1",
                "counts": {
                    "unique": 1,
                    "success": 1,
                    "terminal": 0,
                    "pending": 0,
                    "leased": 0,
                },
                "completed_shards": [network_shard],
                "complete": True,
            }
        ),
        encoding="utf-8",
    )

    assets_root = root / "assets"
    asset_writer = AtomicJsonlShard(
        assets_root / "bridge_assets" / "part-00000.jsonl"
    )
    asset_writer.write(asset("upstream"))
    asset_shard = asdict(asset_writer.commit())
    asset_shard["path"] = "bridge_assets/part-00000.jsonl"
    link_writer = AtomicJsonlShard(
        assets_root / "table_asset_links" / "part-00000.jsonl"
    )
    link_writer.write(
        {
            "source_table_id": "table-1",
            "row_id": 0,
            "entity_id": "entity-upstream",
            "asset_ids": ["upstream"],
        }
    )
    link_shard = asdict(link_writer.commit())
    link_shard["path"] = "table_asset_links/part-00000.jsonl"
    assets = assets_root / "asset-materialization-manifest.json"
    assets.write_text(
        json.dumps(
            {
                "stage": "wdc200k_asset_materialization",
                "fingerprint": task5_fingerprint(),
                "bridge_asset_shards": [asset_shard],
                "table_asset_link_shards": [link_shard],
                "complete": True,
            }
        ),
        encoding="utf-8",
    )
    return network, assets


def write_task5_image_manifest(
    root: Path,
    *,
    content: bytes = b"FIRST",
    declared_sha256: str | None = None,
    create_file: bool = True,
) -> tuple[Path, AssetStageBarrier, Path]:
    from dataclasses import asdict
    from wdc200k_io import AtomicJsonlShard

    assets_root = root / "assets"
    content_sha256 = hashlib.sha256(content).hexdigest()
    image_path = (
        assets_root / "images" / f"image_{content_sha256}.jpg"
    )
    image_path.parent.mkdir(parents=True, exist_ok=True)
    if create_file:
        image_path.write_bytes(content)
    image_record = asset("verified-image", "image")
    image_record.update(
        {
            "local_path": str(image_path),
            "file_name": image_path.name,
            "relative_path": f"images/{image_path.name}",
            "sha256": declared_sha256 or content_sha256,
        }
    )
    asset_writer = AtomicJsonlShard(
        assets_root / "bridge_assets" / "part-00000.jsonl"
    )
    asset_writer.write(image_record)
    asset_shard = asdict(asset_writer.commit())
    asset_shard["path"] = "bridge_assets/part-00000.jsonl"
    link_writer = AtomicJsonlShard(
        assets_root / "table_asset_links" / "part-00000.jsonl"
    )
    link_writer.write(
        {
            "source_table_id": "table-1",
            "row_id": 0,
            "entity_id": "entity-verified-image",
            "asset_ids": ["verified-image"],
        }
    )
    link_shard = asdict(link_writer.commit())
    link_shard["path"] = "table_asset_links/part-00000.jsonl"
    manifest = assets_root / "asset-materialization-manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "stage": "wdc200k_asset_materialization",
                "fingerprint": task5_fingerprint(),
                "bridge_asset_shards": [asset_shard],
                "table_asset_link_shards": [link_shard],
                "complete": True,
            }
        ),
        encoding="utf-8",
    )
    return manifest, task5_barrier(), image_path


def write_strict_ready_marker(
    path: Path,
    jobset,
    *,
    run_fingerprint: str,
    start_fingerprint: str,
) -> None:
    path.write_text(
        json.dumps(
            {
                "stage": "wdc200k_model_ready",
                "schema_version": "wdc200k-model-markers-v1",
                "status": "vllm_servers_ready",
                "model_kind": "text+image",
                "run_fingerprint": run_fingerprint,
                "text_jobset_fingerprint": jobset.text_fingerprint,
                "image_jobset_fingerprint": jobset.image_fingerprint,
                "text_task_count": jobset.text_tasks,
                "image_task_count": jobset.image_tasks,
                "start_fingerprint": start_fingerprint,
                "timestamp": 0.0,
            }
        ),
        encoding="utf-8",
    )


class CountingExtractor:
    def __init__(self, *, fail: bool = False, delay: float = 0.0) -> None:
        self.asset_ids: list[str] = []
        self.fail = fail
        self.delay = delay
        self.lock = threading.Lock()

    def extract(self, current_asset, _entity, _candidate_attributes):
        with self.lock:
            self.asset_ids.append(current_asset["asset_id"])
        if self.delay:
            time.sleep(self.delay)
        if self.fail:
            raise RuntimeError("model exploded")
        return {
            "attributes": [
                {
                    "name": "State",
                    "value": "Alabama",
                    "evidence": "Alabama",
                    "connection_evidence": "The entity name is visible.",
                }
            ],
            "raw_response": '{"attributes":[]}',
            "error": "",
        }


class EndpointAwareExtractor(CountingExtractor):
    def __init__(
        self,
        *,
        readiness_error: BaseException | None = None,
        transient_asset_ids: set[str] | None = None,
        transient_error: str = "model endpoint request failed: HTTP 503",
    ) -> None:
        super().__init__()
        self.readiness_error = readiness_error
        self.transient_asset_ids = transient_asset_ids or set()
        self.transient_error = transient_error
        self.readiness_calls: list[tuple[set[str], float]] = []

    def ensure_endpoints_ready(self, *, modalities, timeout_seconds):
        self.readiness_calls.append((set(modalities), timeout_seconds))
        if self.readiness_error is not None:
            raise self.readiness_error

    def extract(self, current_asset, entity, candidate_attributes):
        if current_asset["asset_id"] in self.transient_asset_ids:
            with self.lock:
                self.asset_ids.append(current_asset["asset_id"])
            raise TransientModelEndpointError(
                self.transient_error
            )
        return super().extract(current_asset, entity, candidate_attributes)


def _job_rows(store: SqliteJobStore) -> list[tuple[str, str]]:
    with sqlite3.connect(store.path) as connection:
        return connection.execute(
            "SELECT job_id, status FROM jobs ORDER BY job_id"
        ).fetchall()


def _claim_order(store: SqliteJobStore) -> list[tuple[str, str]]:
    with sqlite3.connect(store.path) as connection:
        rows = connection.execute(
            "SELECT job_id, payload_json FROM jobs ORDER BY updated_at, job_id"
        ).fetchall()
    return [
        (job_id, json.loads(payload)["asset"]["asset_id"])
        for job_id, payload in rows
    ]


def _insert_model_result(
    store: SqliteJobStore,
    job_id: str,
    *,
    committed: int,
) -> None:
    with sqlite3.connect(store.path) as connection:
        payload = json.loads(
            connection.execute(
                "SELECT payload_json FROM jobs WHERE job_id = ?",
                (job_id,),
            ).fetchone()[0]
        )
        record_json = json.dumps({"stale": True})
        connection.execute(
            """
            INSERT INTO model_results (
                job_id, jobset_fingerprint, modality, status,
                record_json, record_sha256, commit_owner,
                commit_lease_id, commit_lease_expires,
                committed, updated_at
            ) VALUES (?, ?, ?, 'success', ?, ?, 'old-owner',
                      'old-lease', ?, ?, ?)
            """,
            (
                job_id,
                payload["jobset_fingerprint"],
                payload["modality"],
                record_json,
                hashlib.sha256(record_json.encode()).hexdigest(),
                time.time() - 10,
                committed,
                time.time() - 20,
            ),
        )


def test_model_endpoint_preflight_fails_before_claim(tmp_path: Path) -> None:
    store = SqliteJobStore(tmp_path / "models.sqlite3")
    jobset = enqueue_model_tasks(
        [asset("a")],
        store,
        args=model_args(),
        input_fingerprint="assets-v1",
    )
    extractor = EndpointAwareExtractor(
        readiness_error=TransientModelEndpointError(
            "model endpoint readiness failed: HTTP 503"
        )
    )

    with pytest.raises(TransientModelEndpointError, match="HTTP 503"):
        run_model_stage(
            store,
            extractor,
            jobset=jobset,
            endpoint_ready_timeout_seconds=2.5,
            output_root=tmp_path / "outputs",
        )

    assert extractor.readiness_calls == [({"text"}, 2.5)]
    assert extractor.asset_ids == []
    assert _job_rows(store) == [(jobset.jobs[0].job_id, "pending")]
    with sqlite3.connect(store.path) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM model_results"
        ).fetchone() == (0,)


def test_model_endpoint_preflight_only_probes_claimable_modalities(
    tmp_path: Path,
) -> None:
    store = SqliteJobStore(tmp_path / "models.sqlite3")
    jobset = enqueue_model_tasks(
        [asset("text", "text"), asset("image", "image")],
        store,
        args=model_args(),
        input_fingerprint="assets-v1",
    )
    first = run_model_stage(
        store,
        CountingExtractor(),
        jobset=jobset,
        stop_after=1,
        output_root=tmp_path / "outputs",
    )
    assert first.complete is False

    extractor = EndpointAwareExtractor()
    run_model_stage(
        store,
        extractor,
        jobset=jobset,
        endpoint_ready_timeout_seconds=1.0,
        output_root=tmp_path / "outputs",
    )

    assert extractor.readiness_calls == [({"image"}, 1.0)]
    assert extractor.asset_ids == ["image"]


def test_model_endpoint_preflight_includes_expired_leased_modality(
    tmp_path: Path,
) -> None:
    store = SqliteJobStore(tmp_path / "models.sqlite3")
    jobset = enqueue_model_tasks(
        [asset("image", "image")],
        store,
        args=model_args(),
        input_fingerprint="assets-v1",
    )
    with sqlite3.connect(store.path) as connection:
        connection.execute(
            """
            UPDATE jobs SET status = 'leased', owner = 'dead',
                lease_id = 'expired', lease_expires = ?
            WHERE job_id = ?
            """,
            (time.time() - 1, jobset.jobs[0].job_id),
        )
    extractor = EndpointAwareExtractor(
        readiness_error=TransientModelEndpointError("image unavailable")
    )

    with pytest.raises(TransientModelEndpointError, match="unavailable"):
        run_model_stage(
            store,
            extractor,
            jobset=jobset,
            output_root=tmp_path / "outputs",
        )

    assert extractor.readiness_calls == [({"image"}, 0.0)]
    assert _job_rows(store) == [(jobset.jobs[0].job_id, "leased")]


def test_model_endpoint_preflight_skips_unexpired_foreign_lease(
    tmp_path: Path,
) -> None:
    store = SqliteJobStore(tmp_path / "models.sqlite3")
    jobset = enqueue_model_tasks(
        [asset("image", "image")],
        store,
        args=model_args(),
        input_fingerprint="assets-v1",
    )
    lease_expires = time.time() + 60
    with sqlite3.connect(store.path) as connection:
        connection.execute(
            """
            UPDATE jobs SET status = 'leased', owner = 'other-worker',
                lease_id = 'active-lease', lease_expires = ?
            WHERE job_id = ?
            """,
            (lease_expires, jobset.jobs[0].job_id),
        )
    extractor = EndpointAwareExtractor(
        readiness_error=AssertionError("readiness must not run")
    )

    result = run_model_stage(
        store,
        extractor,
        jobset=jobset,
        output_root=tmp_path / "outputs",
    )

    assert result.complete is False
    assert result.leased == 1
    assert extractor.readiness_calls == []
    assert extractor.asset_ids == []
    with sqlite3.connect(store.path) as connection:
        assert connection.execute(
            """
            SELECT status, owner, lease_id, lease_expires
            FROM jobs WHERE job_id = ?
            """,
            (jobset.jobs[0].job_id,),
        ).fetchone() == (
            "leased",
            "other-worker",
            "active-lease",
            lease_expires,
        )


def test_model_endpoint_preflight_rechecks_modality_that_becomes_claimable(
    tmp_path: Path,
) -> None:
    store = SqliteJobStore(tmp_path / "models.sqlite3")
    jobset = enqueue_model_tasks(
        [asset("text", "text"), asset("image", "image")],
        store,
        args=model_args(),
        input_fingerprint="assets-v1",
    )
    image_job = next(job for job in jobset.jobs if job.modality == "image")
    with sqlite3.connect(store.path) as connection:
        connection.execute(
            """
            UPDATE jobs SET status = 'leased', owner = 'other-worker',
                lease_id = 'active-image', lease_expires = ?
            WHERE job_id = ?
            """,
            (time.time() + 60, image_job.job_id),
        )

    expired = False

    def expire_image_after_text(_job_id: str, record: dict) -> None:
        nonlocal expired
        if record["modality"] != "text" or expired:
            return
        expired = True
        with sqlite3.connect(store.path) as connection:
            connection.execute(
                "UPDATE jobs SET lease_expires = ? WHERE job_id = ?",
                (time.time() - 1, image_job.job_id),
            )

    extractor = EndpointAwareExtractor()
    result = run_model_stage(
        store,
        extractor,
        jobset=jobset,
        after_result_write=expire_image_after_text,
        output_root=tmp_path / "outputs",
    )

    assert result.complete is True
    assert extractor.readiness_calls == [
        ({"text"}, 0.0),
        ({"image"}, 0.0),
    ]
    assert extractor.asset_ids == ["text", "image"]


def test_transient_model_error_is_retryable_without_durable_result(
    tmp_path: Path,
) -> None:
    store = SqliteJobStore(tmp_path / "models.sqlite3")
    jobset = enqueue_model_tasks(
        [asset("a")],
        store,
        args=model_args(),
        input_fingerprint="assets-v1",
    )
    extractor = EndpointAwareExtractor(
        transient_asset_ids={"a"},
        transient_error="request used Authorization: Bearer SUPERSECRET",
    )

    with pytest.raises(RuntimeError, match="transient model endpoint") as caught:
        run_model_stage(
            store,
            extractor,
            jobset=jobset,
            output_root=tmp_path / "outputs",
        )

    assert "SUPERSECRET" not in str(caught.value)
    assert "SUPERSECRET" not in "".join(
        traceback.format_exception(
            type(caught.value),
            caught.value,
            caught.value.__traceback__,
        )
    )
    assert _job_rows(store) == [(jobset.jobs[0].job_id, "retryable")]
    with sqlite3.connect(store.path) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM model_results"
        ).fetchone() == (0,)
        assert connection.execute(
            "SELECT COUNT(*) FROM model_call_cache"
        ).fetchone() == (0,)


def test_transient_retry_clears_only_stale_uncommitted_prepared_result(
    tmp_path: Path,
) -> None:
    store = SqliteJobStore(tmp_path / "models.sqlite3")
    jobset = enqueue_model_tasks(
        [asset("stale"), asset("committed")],
        store,
        args=model_args(),
        input_fingerprint="assets-v1",
    )
    jobs_by_asset = {
        asset_id: job_id for job_id, asset_id in _claim_order(store)
    }
    _insert_model_result(
        store,
        jobs_by_asset["stale"],
        committed=0,
    )
    _insert_model_result(
        store,
        jobs_by_asset["committed"],
        committed=1,
    )
    extractor = EndpointAwareExtractor(
        transient_asset_ids={"stale", "committed"}
    )

    with pytest.raises(RuntimeError, match="transient model endpoint"):
        run_model_stage(
            store,
            extractor,
            jobset=jobset,
            group_size=2,
            output_root=tmp_path / "outputs",
        )

    assert all(status == "retryable" for _, status in _job_rows(store))
    with sqlite3.connect(store.path) as connection:
        rows = connection.execute(
            "SELECT job_id, committed FROM model_results ORDER BY job_id"
        ).fetchall()
    assert rows == [(jobs_by_asset["committed"], 1)]


def test_transient_lease_loss_does_not_abort_other_group_callbacks(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import wdc200k_models

    store = SqliteJobStore(tmp_path / "models.sqlite3")
    jobset = enqueue_model_tasks(
        [asset("transient"), asset("success")],
        store,
        args=model_args(),
        input_fingerprint="assets-v1",
    )
    jobs_by_asset = {
        asset_id: job_id for job_id, asset_id in _claim_order(store)
    }

    def lose_transient_then_deliver_success(**kwargs):
        tasks = sorted(
            kwargs["tasks"],
            key=lambda task: task.asset["asset_id"] != "transient",
        )
        records = {}
        for task in tasks:
            if task.asset["asset_id"] == "transient":
                with sqlite3.connect(store.path) as connection:
                    connection.execute(
                        """
                        UPDATE jobs SET owner = 'new-owner',
                            lease_id = 'new-lease', lease_expires = ?
                        WHERE job_id = ?
                        """,
                        (time.time() + 60, jobs_by_asset["transient"]),
                    )
                record = {
                    "attributes": [],
                    "raw_response": "",
                    "error": "model endpoint temporarily unavailable",
                    "error_class": "model_endpoint_transient",
                }
            else:
                record = {
                    "attributes": [],
                    "raw_response": '{"attributes":[]}',
                    "error": "",
                }
            records[task.cache_key] = record
            kwargs["on_record"](task.cache_key, record)
        return records

    monkeypatch.setattr(
        wdc200k_models,
        "run_extraction_task_group",
        lose_transient_then_deliver_success,
    )

    with pytest.raises(RuntimeError, match="transient model endpoint"):
        run_model_stage(
            store,
            CountingExtractor(),
            jobset=jobset,
            group_size=2,
            output_root=tmp_path / "outputs",
        )

    assert dict(_job_rows(store)) == {
        jobs_by_asset["transient"]: "leased",
        jobs_by_asset["success"]: "success",
    }
    with sqlite3.connect(store.path) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM model_results WHERE committed = 1"
        ).fetchone() == (1,)


def test_fenced_retry_mismatch_returns_false_before_write_guard(
    tmp_path: Path,
) -> None:
    store = SqliteJobStore(tmp_path / "models.sqlite3")
    jobset = enqueue_model_tasks(
        [asset("stale")],
        store,
        args=model_args(),
        input_fingerprint="assets-v1",
    )
    claimed = store.claim(
        jobset.text_kind,
        limit=1,
        owner="old-owner",
        lease_seconds=60,
    )[0]
    with sqlite3.connect(store.path) as connection:
        connection.execute(
            """
            UPDATE jobs SET owner = 'new-owner', lease_id = 'new-lease'
            WHERE job_id = ?
            """,
            (claimed.job_id,),
        )

    class RejectWrites:
        def before_write(self, _estimated_bytes: int) -> None:
            raise AssertionError("stale fence must not reserve a write")

        def before_commit(self, _estimated_bytes: int) -> None:
            raise AssertionError("stale fence must not commit")

    assert models._fenced_retry_model_job(
        store.path,
        job=claimed,
        expected_kind=claimed.kind,
        record={"error_class": "model_endpoint_transient"},
        write_tracker=RejectWrites(),
    ) is False


def test_transient_group_commits_success_stops_and_healthy_resume_succeeds(
    tmp_path: Path,
) -> None:
    store = SqliteJobStore(tmp_path / "models.sqlite3")
    jobset = enqueue_model_tasks(
        [asset("a"), asset("b"), asset("c")],
        store,
        args=model_args(),
        input_fingerprint="assets-v1",
    )
    claim_order = _claim_order(store)
    first_success = claim_order[0]
    transient = claim_order[1]
    not_claimed = claim_order[2]
    failing = EndpointAwareExtractor(transient_asset_ids={transient[1]})

    with pytest.raises(RuntimeError, match="transient model endpoint"):
        run_model_stage(
            store,
            failing,
            jobset=jobset,
            group_size=2,
            workers=2,
            output_root=tmp_path / "outputs",
        )

    assert set(failing.asset_ids) == {first_success[1], transient[1]}
    assert dict(_job_rows(store)) == {
        first_success[0]: "success",
        transient[0]: "retryable",
        not_claimed[0]: "pending",
    }
    with sqlite3.connect(store.path) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM model_results"
        ).fetchone() == (1,)
        assert connection.execute(
            "SELECT COUNT(*) FROM model_call_cache"
        ).fetchone() == (1,)

    healthy = EndpointAwareExtractor()
    result = run_model_stage(
        store,
        healthy,
        jobset=jobset,
        group_size=2,
        output_root=tmp_path / "outputs",
    )

    assert result.complete is True
    assert set(healthy.asset_ids) == {transient[1], not_claimed[1]}
    assert all(status == "success" for _job_id, status in _job_rows(store))


def test_enqueue_staging_database_uses_guarded_controlled_directory(
    tmp_path: Path,
) -> None:
    staging_dir = tmp_path / "work" / "model-enqueue"
    calls: list[tuple[Path, int]] = []
    store = SqliteJobStore(tmp_path / "models.sqlite3")

    jobset = enqueue_model_tasks(
        [asset("a"), asset("b")],
        store,
        args=model_args(),
        staging_dir=staging_dir,
        pre_write_guard=lambda path, size=0: calls.append(
            (Path(path), size)
        ),
    )

    staging_calls = [
        (path, size)
        for path, size in calls
        if path.parent == staging_dir
    ]
    assert jobset.total_tasks == 2
    assert any(size > 0 for _path, size in staging_calls)
    assert any(size == 0 for _path, size in staging_calls)
    assert not staging_dir.exists() or list(staging_dir.iterdir()) == []


def test_enqueue_staging_database_is_cleaned_when_initial_guard_fails(
    tmp_path: Path,
) -> None:
    staging_dir = tmp_path / "work" / "model-enqueue"

    def reject_staging(_path: Path, estimated_bytes: int = 0) -> None:
        if estimated_bytes > 0:
            raise RuntimeError("synthetic staging reserve exhausted")

    with pytest.raises(RuntimeError, match="staging reserve"):
        enqueue_model_tasks(
            [asset("a")],
            SqliteJobStore(tmp_path / "models.sqlite3"),
            args=model_args(),
            staging_dir=staging_dir,
            pre_write_guard=reject_staging,
        )

    assert not staging_dir.exists() or list(staging_dir.iterdir()) == []


def test_model_result_writes_are_amortized_but_commits_recheck_live_disk(
    tmp_path: Path,
) -> None:
    store_path = tmp_path / "models.sqlite3"
    store = SqliteJobStore(store_path)
    records = [asset(f"asset-{index}") for index in range(40)]
    jobset = enqueue_model_tasks(records, store, args=model_args())
    calls: list[tuple[Path, int]] = []

    result = run_model_stage(
        store,
        CountingExtractor(),
        jobset=jobset,
        group_size=len(records),
        workers=4,
        output_root=tmp_path / "outputs",
        pre_write_guard=lambda path, size=0: calls.append(
            (Path(path), size)
        ),
    )

    store_calls = [call for call in calls if call[0] == store_path]
    assert result.success == len(records)
    assert any(size >= 64 * 1024 * 1024 for _path, size in store_calls)
    assert sum(size == 0 for _path, size in store_calls) >= len(records)


def test_manifest_validation_uses_guarded_controlled_membership_database(
    tmp_path: Path,
) -> None:
    store = SqliteJobStore(tmp_path / "work" / "models.sqlite3")
    jobset = enqueue_model_tasks(
        [asset("a")],
        store,
        args=model_args(),
    )
    result = run_model_stage(
        store,
        CountingExtractor(),
        jobset=jobset,
        output_root=tmp_path / "work" / "outputs",
    )
    calls: list[tuple[Path, int]] = []

    loaded = models._load_valid_manifest(
        result.manifest_path,
        output_root=result.manifest_path.parent,
        jobset=jobset,
        pre_write_guard=lambda path, size=0: calls.append(
            (Path(path), size)
        ),
    )

    validation_dir = store.path.parent / ".model-validation-staging"
    assert loaded is not None
    assert calls
    assert all(
        path == validation_dir or validation_dir in path.parents
        for path, _size in calls
    )
    assert any(size > 0 for _path, size in calls)
    assert any(size == 0 for _path, size in calls)
    assert list(validation_dir.iterdir()) == []
    with sqlite3.connect(store.path) as connection:
        assert connection.execute(
            """
            SELECT COUNT(*) FROM sqlite_master
            WHERE type = 'table' AND name LIKE 'observed_outputs_%'
            """
        ).fetchone() == (0,)


def test_adapter_and_model_validators_bind_exact_task_membership(
    tmp_path: Path,
) -> None:
    structural_manifest = tmp_path / "structural.json"
    finalized_manifest = tmp_path / "final.json"
    assets_manifest = tmp_path / "assets.json"
    structural_manifest.write_text('{"complete":true}', encoding="utf-8")
    finalized_manifest.write_text('{"complete":true}', encoding="utf-8")
    assets_manifest.write_text('{"complete":true}', encoding="utf-8")
    adapter_input = model_adapter_input_fingerprint(
        [
            hashlib.sha256(
                structural_manifest.read_bytes()
            ).hexdigest()
        ],
        finalized_selection_manifest=finalized_manifest,
        assets_manifest=assets_manifest,
    )
    adapter_root = tmp_path / "adapter"
    task_path = adapter_root / "tasks" / "part-00000.jsonl"
    task_writer = AtomicJsonlShard(task_path)
    task_writer.write(asset("adapter"))
    task_shard = task_writer.commit()
    relative_task_shard = {
        "path": task_path.relative_to(adapter_root).as_posix(),
        "records": task_shard.records,
        "bytes": task_shard.bytes,
        "sha256": task_shard.sha256,
    }
    adapter_manifest = (
        adapter_root / "model-task-adapter-manifest.json"
    )
    adapter_manifest.write_text(
        json.dumps(
            {
                "stage": "wdc200k_model_task_adapter",
                "schema_version": models.MODEL_QUEUE_SCHEMA_VERSION,
                "input_fingerprint": adapter_input,
                "task_shards": [relative_task_shard],
                "error_shards": [],
                "counts": {"tasks": 1, "errors": 0},
                "complete": True,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    adapted = models.AdaptedModelTasks(
        output_root=adapter_root,
        task_paths=(task_path,),
        error_paths=(),
        manifest_path=adapter_manifest,
        input_fingerprint=adapter_input,
        tasks=1,
        errors=0,
    )
    validated = validate_adapted_model_tasks(
        adapted,
        expected_input_fingerprint=adapter_input,
    )
    args = model_args()
    authority = ModelStageAuthority.current(args)
    store = SqliteJobStore(tmp_path / "models.sqlite3")
    jobset = enqueue_model_tasks(
        [asset("adapter")],
        store,
        args=args,
        input_fingerprint=adapter_input,
        text_input_fingerprint=adapter_input,
        image_input_fingerprint=adapter_input,
    )
    result = run_model_stage(
        store,
        CountingExtractor(),
        jobset=jobset,
        output_root=tmp_path / "model-output",
    )

    assert validated == adapted
    assert validate_model_stage_for_adapter(
        result,
        adapted,
        args=args,
        authority=authority,
        validation_store_path=tmp_path / "validation-models.sqlite3",
    )

    foreign_adapter = models.AdaptedModelTasks(
        **{
            **adapted.__dict__,
            "input_fingerprint": "foreign-adapter",
        }
    )
    with pytest.raises(ValueError, match="adapter"):
        validate_model_stage_for_adapter(
            result,
            foreign_adapter,
            args=args,
            authority=authority,
            validation_store_path=tmp_path / "foreign-models.sqlite3",
        )

    foreign_runs = [
        (
            model_args(),
            {
                "prompt_version": "foreign-prompt-v0",
            },
        ),
        (
            model_args(text_model_name="foreign-text-model"),
            {},
        ),
        (
            model_args(),
            {
                "policy_fingerprint": "foreign-model-policy-v0",
            },
        ),
    ]
    for index, (foreign_args, options) in enumerate(foreign_runs):
        foreign_store = SqliteJobStore(
            tmp_path / f"foreign-authority-{index}.sqlite3"
        )
        foreign_jobset = enqueue_model_tasks(
            [asset("adapter")],
            foreign_store,
            args=foreign_args,
            input_fingerprint=adapter_input,
            text_input_fingerprint=adapter_input,
            image_input_fingerprint=adapter_input,
            **options,
        )
        foreign_result = run_model_stage(
            foreign_store,
            CountingExtractor(),
            jobset=foreign_jobset,
            output_root=tmp_path / f"foreign-output-{index}",
        )
        with pytest.raises(ValueError, match="model stage authority"):
            validate_model_stage_for_adapter(
                foreign_result,
                adapted,
                args=args,
                authority=authority,
                validation_store_path=(
                    tmp_path / f"foreign-validation-{index}.sqlite3"
                ),
            )

    foreign_policy_authority = ModelStageAuthority(
        text_model_identity=authority.text_model_identity,
        image_model_identity=authority.image_model_identity,
        policy_fingerprint="foreign-model-policy-v0",
    )
    with pytest.raises(ValueError, match="model stage authority policy"):
        validate_model_stage_for_adapter(
            foreign_result,
            adapted,
            args=args,
            authority=foreign_policy_authority,
            validation_store_path=(
                tmp_path / "foreign-policy-authority.sqlite3"
            ),
        )


def test_model_stage_resumes_without_repeating_success(tmp_path: Path) -> None:
    extractor = CountingExtractor()
    store = SqliteJobStore(tmp_path / "models.sqlite3")
    jobset = enqueue_model_tasks(
        [asset("a"), asset("b", "image")],
        store,
    )
    text_done = tmp_path / "text-done.json"
    image_done = tmp_path / "image-done.json"
    network, assets_manifest = write_strict_upstream_barriers(tmp_path)
    start = tmp_path / "start.json"
    ready = tmp_path / "ready.json"
    start_fingerprint = write_model_start_marker(
        start,
        jobset,
        network_manifests=[network],
        assets_manifest=assets_manifest,
        assets_barrier=task5_barrier(),
        run_fingerprint="run-v1",
    )
    write_strict_ready_marker(
        ready,
        jobset,
        run_fingerprint="run-v1",
        start_fingerprint=start_fingerprint,
    )

    first = run_model_stage(
        store,
        extractor,
        stop_after=1,
        output_root=tmp_path / "outputs",
        text_done_marker=text_done,
        image_done_marker=image_done,
        start_marker=start,
        ready_marker=ready,
        network_manifests=[network],
        assets_manifest=assets_manifest,
        assets_barrier=task5_barrier(),
        run_fingerprint="run-v1",
    )
    assert first.complete is False
    assert text_done.exists()
    assert not image_done.exists()
    text_done_payload = json.loads(text_done.read_text(encoding="utf-8"))
    assert text_done_payload["stage"] == "wdc200k_model_done"
    assert text_done_payload["schema_version"] == (
        "wdc200k-model-markers-v1"
    )
    assert text_done_payload["start_fingerprint"] == start_fingerprint
    second = run_model_stage(
        store,
        extractor,
        output_root=tmp_path / "outputs",
        text_done_marker=text_done,
        image_done_marker=image_done,
        start_marker=start,
        ready_marker=ready,
        network_manifests=[network],
        assets_manifest=assets_manifest,
        assets_barrier=task5_barrier(),
        run_fingerprint="run-v1",
    )

    assert second.complete is True
    assert image_done.exists()
    assert json.loads(image_done.read_text(encoding="utf-8"))[
        "start_fingerprint"
    ] == start_fingerprint
    assert Counter(extractor.asset_ids) == {"a": 1, "b": 1}
    assert validate_model_stage(second)
    mtimes = {
        path: path.stat().st_mtime_ns
        for path in (
            second.manifest_path,
            *second.extraction_paths,
            *second.error_paths,
        )
    }

    third = run_model_stage(
        store,
        extractor,
        output_root=tmp_path / "outputs",
    )
    assert third.complete is True
    assert {
        path: path.stat().st_mtime_ns for path in mtimes
    } == mtimes


def test_resume_manifest_rejects_corrupt_counts(tmp_path: Path) -> None:
    store = SqliteJobStore(tmp_path / "models.sqlite3")
    jobset = enqueue_model_tasks(
        [asset("a")],
        store,
        args=model_args(),
        input_fingerprint="assets-v1",
    )
    result = run_model_stage(
        store,
        CountingExtractor(),
        jobset=jobset,
        output_root=tmp_path / "outputs",
    )
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    manifest["counts"]["success"] += 1
    result.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    assert validate_model_stage(result) is False
    with pytest.raises(ValueError, match="count"):
        run_model_stage(
            store,
            CountingExtractor(),
            jobset=jobset,
            output_root=tmp_path / "outputs",
        )


def test_resume_manifest_rejects_record_provenance_even_with_new_checksum(
    tmp_path: Path,
) -> None:
    store = SqliteJobStore(tmp_path / "models.sqlite3")
    jobset = enqueue_model_tasks(
        [asset("a")],
        store,
        args=model_args(),
        input_fingerprint="assets-v1",
    )
    result = run_model_stage(
        store,
        CountingExtractor(),
        jobset=jobset,
        output_root=tmp_path / "outputs",
    )
    path = result.extraction_paths[0]
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
    ]
    records[0]["jobset_fingerprint"] = "foreign-jobset"
    encoded = "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
        for record in records
    ).encode("utf-8")
    path.write_bytes(encoded)
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    shard = manifest["extraction_shards"][0]
    shard["bytes"] = len(encoded)
    shard["sha256"] = hashlib.sha256(encoded).hexdigest()
    result.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    assert validate_model_stage(result) is False


@pytest.mark.parametrize(
    ("field", "forged_value"),
    [
        ("model_call_key", "f" * 64),
        ("cache_key", "forged-cache-key"),
        ("asset_fingerprint", "e" * 64),
        ("entity_prompt_fingerprint", "d" * 64),
        ("candidate_attribute_names", ["Forged"]),
    ],
)
def test_resume_manifest_rejects_exact_payload_field_forgery(
    tmp_path: Path,
    field: str,
    forged_value: object,
) -> None:
    store = SqliteJobStore(tmp_path / "models.sqlite3")
    jobset = enqueue_model_tasks(
        [asset("a")],
        store,
        args=model_args(),
        input_fingerprint="assets-v1",
    )
    result = run_model_stage(
        store,
        CountingExtractor(),
        jobset=jobset,
        output_root=tmp_path / "outputs",
    )
    path = result.extraction_paths[0]
    record = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    assert record["job_kind"] == jobset.text_kind
    assert len(record["payload_sha256"]) == 64
    record[field] = forged_value
    encoded = (
        json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
    ).encode("utf-8")
    path.write_bytes(encoded)
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    shard = manifest["extraction_shards"][0]
    shard["bytes"] = len(encoded)
    shard["sha256"] = hashlib.sha256(encoded).hexdigest()
    result.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    assert validate_model_stage(result) is False


def test_resume_manifest_rejects_tampered_durable_payload_hash(
    tmp_path: Path,
) -> None:
    store = SqliteJobStore(tmp_path / "models.sqlite3")
    jobset = enqueue_model_tasks(
        [asset("a")],
        store,
        args=model_args(),
        input_fingerprint="assets-v1",
    )
    result = run_model_stage(
        store,
        CountingExtractor(),
        jobset=jobset,
        output_root=tmp_path / "outputs",
    )
    with sqlite3.connect(store.path) as connection:
        payload = json.loads(
            connection.execute(
                "SELECT payload_json FROM jobs WHERE job_id = ?",
                (jobset.jobs[0].job_id,),
            ).fetchone()[0]
        )
        payload["entity"]["cell_text"] = "tampered after enqueue"
        connection.execute(
            "UPDATE jobs SET payload_json = ? WHERE job_id = ?",
            (
                json.dumps(
                    payload,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                ),
                jobset.jobs[0].job_id,
            ),
        )

    assert validate_model_stage(result) is False


def test_resume_manifest_rejects_duplicate_substitution_with_new_checksum(
    tmp_path: Path,
) -> None:
    store = SqliteJobStore(tmp_path / "models.sqlite3")
    jobset = enqueue_model_tasks(
        [asset("a"), asset("b")],
        store,
        args=model_args(),
        input_fingerprint="assets-v1",
    )
    result = run_model_stage(
        store,
        CountingExtractor(),
        jobset=jobset,
        output_root=tmp_path / "outputs",
    )
    path = result.extraction_paths[0]
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
    ]
    assert len(records) == 2
    records[1] = dict(records[0])
    encoded = "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
        for record in records
    ).encode("utf-8")
    path.write_bytes(encoded)
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    shard = manifest["extraction_shards"][0]
    shard["bytes"] = len(encoded)
    shard["sha256"] = hashlib.sha256(encoded).hexdigest()
    result.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    assert validate_model_stage(result) is False


def test_result_written_before_finish_repairs_without_model_call(
    tmp_path: Path,
) -> None:
    extractor = CountingExtractor()
    store = SqliteJobStore(tmp_path / "models.sqlite3")
    jobset = enqueue_model_tasks(
        [asset("a")],
        store,
        args=model_args(),
        input_fingerprint="assets-v1",
    )
    crashed = False

    def crash_once(_job_id: str, _record: dict) -> None:
        nonlocal crashed
        if not crashed:
            crashed = True
            raise RuntimeError("simulated result-before-finish crash")

    with pytest.raises(RuntimeError, match="simulated"):
        run_model_stage(
            store,
            extractor,
            jobset=jobset,
            output_root=tmp_path / "outputs",
            after_result_write=crash_once,
            lease_seconds=0.08,
            heartbeat_seconds=0.02,
        )

    time.sleep(0.1)
    resumed = run_model_stage(
        store,
        extractor,
        jobset=jobset,
        output_root=tmp_path / "outputs",
    )
    assert resumed.complete is True
    assert extractor.asset_ids == ["a"]


def test_first_model_result_commit_guard_rolls_back_prepared_record(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    extractor = CountingExtractor()
    store = SqliteJobStore(tmp_path / "models.sqlite3")
    jobset = enqueue_model_tasks(
        [asset("a")],
        store,
        args=model_args(),
    )
    monkeypatch.setattr(
        models.GuardedWriteTracker,
        "DEFAULT_INTERVAL_BYTES",
        1,
    )
    zero_checks = 0

    def reject_first_commit(
        path: Path,
        estimated_bytes: int = 0,
    ) -> None:
        nonlocal zero_checks
        if Path(path) != store.path:
            return
        if estimated_bytes == 0:
            zero_checks += 1
        if zero_checks == 5:
            raise OSError("prepared result commit reserve exhausted")

    with pytest.raises(OSError, match="prepared result commit reserve"):
        run_model_stage(
            store,
            extractor,
            jobset=jobset,
            output_root=tmp_path / "outputs",
            pre_write_guard=reject_first_commit,
            lease_seconds=0.08,
            heartbeat_seconds=0.02,
        )

    with sqlite3.connect(store.path) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM model_results"
        ).fetchone() == (0,)
        assert connection.execute(
            "SELECT status FROM jobs WHERE job_id = ?",
            (jobset.jobs[0].job_id,),
        ).fetchone() == ("leased",)


def test_second_model_result_commit_guard_leaves_repairable_record(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    extractor = CountingExtractor()
    store = SqliteJobStore(tmp_path / "models.sqlite3")
    jobset = enqueue_model_tasks(
        [asset("a")],
        store,
        args=model_args(),
    )
    monkeypatch.setattr(
        models.GuardedWriteTracker,
        "DEFAULT_INTERVAL_BYTES",
        1,
    )
    store_commit_checks = 0

    def reject_second_commit(
        path: Path,
        estimated_bytes: int = 0,
    ) -> None:
        nonlocal store_commit_checks
        if Path(path) != store.path or estimated_bytes != 0:
            return
        with sqlite3.connect(store.path) as connection:
            prepared = connection.execute(
                """
                SELECT COUNT(*) FROM model_results
                WHERE committed = 0
                """
            ).fetchone()[0]
        if prepared:
            store_commit_checks += 1
            raise OSError("final model result commit reserve exhausted")

    with pytest.raises(OSError, match="final model result commit reserve"):
        run_model_stage(
            store,
            extractor,
            jobset=jobset,
            output_root=tmp_path / "outputs",
            pre_write_guard=reject_second_commit,
            lease_seconds=0.08,
            heartbeat_seconds=0.02,
        )
    assert store_commit_checks == 1
    with sqlite3.connect(store.path) as connection:
        assert connection.execute(
            "SELECT committed FROM model_results WHERE job_id = ?",
            (jobset.jobs[0].job_id,),
        ).fetchone() == (0,)
        assert connection.execute(
            "SELECT status FROM jobs WHERE job_id = ?",
            (jobset.jobs[0].job_id,),
        ).fetchone() == ("leased",)
    time.sleep(0.1)
    resumed = run_model_stage(
        store,
        extractor,
        jobset=jobset,
        output_root=tmp_path / "outputs",
    )
    assert resumed.complete is True
    assert extractor.asset_ids == ["a"]


def test_durable_result_repair_guard_failure_rolls_back_and_resumes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    extractor = CountingExtractor()
    store = SqliteJobStore(tmp_path / "models.sqlite3")
    jobset = enqueue_model_tasks(
        [asset("a")],
        store,
        args=model_args(),
        input_fingerprint="assets-v1",
    )

    def crash_after_prepare(_job_id: str, _record: dict) -> None:
        raise RuntimeError("simulated prepared-result crash")

    with pytest.raises(RuntimeError, match="prepared-result"):
        run_model_stage(
            store,
            extractor,
            jobset=jobset,
            output_root=tmp_path / "outputs",
            after_result_write=crash_after_prepare,
            lease_seconds=0.08,
            heartbeat_seconds=0.02,
        )
    time.sleep(0.1)
    monkeypatch.setattr(
        models.GuardedWriteTracker,
        "DEFAULT_INTERVAL_BYTES",
        1,
    )
    zero_checks = 0

    def reject_repair_commit(
        _path: Path,
        estimated_bytes: int = 0,
    ) -> None:
        nonlocal zero_checks
        if estimated_bytes == 0:
            zero_checks += 1
            if zero_checks == 2:
                raise OSError("durable repair commit reserve exhausted")

    with pytest.raises(OSError, match="repair commit reserve"):
        run_model_stage(
            store,
            extractor,
            jobset=jobset,
            output_root=tmp_path / "outputs",
            pre_write_guard=reject_repair_commit,
        )
    with sqlite3.connect(store.path) as connection:
        assert connection.execute(
            "SELECT status FROM jobs WHERE job_id = ?",
            (jobset.jobs[0].job_id,),
        ).fetchone() == ("leased",)
        assert connection.execute(
            "SELECT committed FROM model_results WHERE job_id = ?",
            (jobset.jobs[0].job_id,),
        ).fetchone() == (0,)

    resumed = run_model_stage(
        store,
        extractor,
        jobset=jobset,
        output_root=tmp_path / "outputs",
    )
    assert resumed.complete is True
    assert extractor.asset_ids == ["a"]


def test_cache_written_before_result_repairs_without_model_call(
    tmp_path: Path,
) -> None:
    extractor = CountingExtractor()
    store = SqliteJobStore(tmp_path / "models.sqlite3")
    jobset = enqueue_model_tasks(
        [asset("a")],
        store,
        args=model_args(),
        input_fingerprint="assets-v1",
    )

    def crash_after_cache(_job_id: str, _record: dict) -> None:
        raise RuntimeError("simulated cache-before-result crash")

    with pytest.raises(RuntimeError, match="cache-before-result"):
        run_model_stage(
            store,
            extractor,
            jobset=jobset,
            output_root=tmp_path / "outputs",
            after_cache_write=crash_after_cache,
            lease_seconds=0.08,
            heartbeat_seconds=0.02,
        )

    time.sleep(0.1)
    resumed = run_model_stage(
        store,
        extractor,
        jobset=jobset,
        output_root=tmp_path / "outputs",
    )
    assert resumed.complete is True
    assert extractor.asset_ids == ["a"]


def test_existing_extraction_cache_repairs_job_without_model_call(
    tmp_path: Path,
) -> None:
    store = SqliteJobStore(tmp_path / "models.sqlite3")
    jobset = enqueue_model_tasks(
        [asset("a")],
        store,
        args=model_args(),
        input_fingerprint="assets-v1",
    )
    cache = ExtractionCache(tmp_path / "legacy-cache.jsonl")
    cache.put(
        jobset.jobs[0].cache_key,
        {
            "cache_key": jobset.jobs[0].cache_key,
            "model_call_key": jobset.jobs[0].model_call_key,
            "prompt_version": jobset.prompt_version,
            "model_identity": "text-v1",
            "asset_fingerprint": jobset.jobs[0].asset_fingerprint,
            "entity_prompt_fingerprint": (
                jobset.jobs[0].entity_prompt_fingerprint
            ),
            "entity_id": "entity-a",
            "entity_text": "Entity a",
            "entity_wiki_title": "Entity a",
            "asset_id": "a",
            "asset_type": "text",
            "modality": "text",
            "policy_fingerprint": "existing-extraction-semantics-v1",
            "parser_schema_version": MODEL_PARSER_SCHEMA_VERSION,
            "candidate_attribute_names": ["State"],
            "attributes": [],
            "raw_response": '{"attributes":[]}',
            "error": "",
        },
    )
    extractor = CountingExtractor()

    result = run_model_stage(
        store,
        extractor,
        jobset=jobset,
        cache=cache,
        output_root=tmp_path / "outputs",
    )

    assert result.complete is True
    assert extractor.asset_ids == []


def test_terminal_model_error_is_durable_and_not_retried(
    tmp_path: Path,
) -> None:
    extractor = CountingExtractor(fail=True)
    store = SqliteJobStore(tmp_path / "models.sqlite3")
    jobset = enqueue_model_tasks(
        [asset("a")],
        store,
        args=model_args(),
        input_fingerprint="assets-v1",
    )

    first = run_model_stage(
        store,
        extractor,
        jobset=jobset,
        output_root=tmp_path / "outputs",
    )
    second = run_model_stage(
        store,
        extractor,
        jobset=jobset,
        output_root=tmp_path / "outputs",
    )

    assert first.terminal == second.terminal == 1
    assert extractor.asset_ids == ["a"]
    errors = [
        json.loads(line)
        for path in second.error_paths
        for line in path.read_text(encoding="utf-8").splitlines()
    ]
    assert errors[0]["error"] == "model exploded"
    assert set(errors[0]) >= {
        "cache_key",
        "prompt_version",
        "entity_id",
        "asset_id",
        "asset_type",
        "candidate_attribute_names",
        "attributes",
        "raw_response",
        "error",
    }


def test_missing_extractor_does_not_terminally_poison_uncached_job(
    tmp_path: Path,
) -> None:
    store = SqliteJobStore(tmp_path / "models.sqlite3")
    jobset = enqueue_model_tasks(
        [asset("a")],
        store,
        args=model_args(),
        input_fingerprint="assets-v1",
    )

    with pytest.raises(RuntimeError, match="extractor"):
        run_model_stage(store, None, jobset=jobset)

    result = run_model_stage(
        store,
        CountingExtractor(),
        jobset=jobset,
        output_root=tmp_path / "outputs",
    )
    assert result.complete is True
    assert result.terminal == 0


def test_text_and_image_jobsets_change_independently(tmp_path: Path) -> None:
    store = SqliteJobStore(tmp_path / "models.sqlite3")
    first = enqueue_model_tasks(
        [asset("a"), asset("b", "image")],
        store,
        args=model_args(),
        input_fingerprint="assets-v1",
    )
    text_changed = enqueue_model_tasks(
        [asset("a"), asset("b", "image")],
        store,
        args=model_args(text_model_name="text-v2"),
        input_fingerprint="assets-v1",
    )
    image_changed = enqueue_model_tasks(
        [asset("a"), asset("b", "image", sha256="changed")],
        store,
        args=model_args(),
        input_fingerprint="assets-v2",
    )

    assert first.text_kind != text_changed.text_kind
    assert first.image_kind == text_changed.image_kind
    assert first.text_kind != image_changed.text_kind
    assert first.image_kind != image_changed.image_kind


def test_jobsets_v1_v2_v1_are_physically_isolated_in_one_store(
    tmp_path: Path,
) -> None:
    store = SqliteJobStore(tmp_path / "models.sqlite3")
    extractor = CountingExtractor()
    first = enqueue_model_tasks(
        [asset("a")],
        store,
        args=model_args(),
        input_fingerprint="assets-v1",
    )
    first_result = run_model_stage(
        store,
        extractor,
        jobset=first,
        output_root=tmp_path / "outputs",
    )
    second = enqueue_model_tasks(
        [asset("a")],
        store,
        args=model_args(),
        input_fingerprint="assets-v2",
    )
    second_result = run_model_stage(
        store,
        extractor,
        jobset=second,
        output_root=tmp_path / "outputs",
    )
    resumed_first = enqueue_model_tasks(
        [asset("a")],
        store,
        args=model_args(),
        input_fingerprint="assets-v1",
    )
    resumed_result = run_model_stage(
        store,
        extractor,
        jobset=resumed_first,
        output_root=tmp_path / "outputs",
    )

    assert first.text_kind != second.text_kind
    assert first.jobs[0].job_id != second.jobs[0].job_id
    assert resumed_first.text_kind == first.text_kind
    assert extractor.asset_ids == ["a"]
    assert first_result.manifest_path != second_result.manifest_path
    assert resumed_result.manifest_path == first_result.manifest_path


@pytest.mark.parametrize(
    ("first_ids", "second_ids"),
    [
        (["a", "b"], ["a"]),
        (["a"], ["a", "b"]),
        (["a", "b"], ["a", "c"]),
    ],
)
def test_completed_jobset_rejects_exact_membership_changes_without_mutation(
    tmp_path: Path,
    first_ids: list[str],
    second_ids: list[str],
) -> None:
    store = SqliteJobStore(tmp_path / "models.sqlite3")
    first = enqueue_model_tasks(
        [asset(value) for value in first_ids],
        store,
        args=model_args(),
        input_fingerprint="same-input",
    )
    result = run_model_stage(
        store,
        CountingExtractor(),
        jobset=first,
        output_root=tmp_path / "outputs",
    )
    mtimes = {
        path: path.stat().st_mtime_ns
        for path in (result.manifest_path, *result.extraction_paths)
    }

    with pytest.raises(ValueError, match="membership"):
        enqueue_model_tasks(
            [asset(value) for value in second_ids],
            store,
            args=model_args(),
            input_fingerprint="same-input",
        )

    resumed = run_model_stage(
        store,
        CountingExtractor(),
        jobset=first,
        output_root=tmp_path / "outputs",
    )
    assert validate_model_stage(resumed)
    assert {
        path: path.stat().st_mtime_ns for path in mtimes
    } == mtimes
    with sqlite3.connect(store.path) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM jobs WHERE kind = ?",
            (first.text_kind,),
        ).fetchone()[0] == len(first_ids)


def test_completed_jobset_accepts_reordered_duplicate_identical_members(
    tmp_path: Path,
) -> None:
    store = SqliteJobStore(tmp_path / "models.sqlite3")
    first = enqueue_model_tasks(
        [asset("a"), asset("b")],
        store,
        args=model_args(),
        input_fingerprint="same-input",
    )

    resumed = enqueue_model_tasks(
        [asset("b"), asset("a"), asset("a")],
        store,
        args=model_args(),
        input_fingerprint="same-input",
    )

    assert resumed.text_tasks == first.text_tasks == 2
    with sqlite3.connect(store.path) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM jobs WHERE kind = ?",
            (first.text_kind,),
        ).fetchone()[0] == 2


def test_incomplete_jobset_rejects_resume_missing_an_existing_member(
    tmp_path: Path,
) -> None:
    store = SqliteJobStore(tmp_path / "models.sqlite3")
    first = enqueue_model_tasks(
        [asset("a"), asset("b")],
        store,
        args=model_args(),
        input_fingerprint="same-input",
    )
    with sqlite3.connect(store.path) as connection:
        connection.execute(
            """
            UPDATE model_jobsets SET enqueue_complete = 0
            WHERE fingerprint = ?
            """,
            (first.text_fingerprint,),
        )
    with sqlite3.connect(store.path) as connection:
        before_members = connection.execute(
            """
            SELECT job_id, cache_key, asset_fingerprint, payload_sha256
            FROM model_job_members
            WHERE jobset_fingerprint = ? ORDER BY job_id
            """,
            (first.text_fingerprint,),
        ).fetchall()

    with pytest.raises(ValueError, match="membership"):
        enqueue_model_tasks(
            [asset("a")],
            store,
            args=model_args(),
            input_fingerprint="same-input",
        )

    with sqlite3.connect(store.path) as connection:
        after_members = connection.execute(
            """
            SELECT job_id, cache_key, asset_fingerprint, payload_sha256
            FROM model_job_members
            WHERE jobset_fingerprint = ? ORDER BY job_id
            """,
            (first.text_fingerprint,),
        ).fetchall()
        enqueue_complete = connection.execute(
            """
            SELECT enqueue_complete FROM model_jobsets
            WHERE fingerprint = ?
            """,
            (first.text_fingerprint,),
        ).fetchone()[0]
    assert after_members == before_members
    assert enqueue_complete == 0


def test_entity_prompt_alias_changes_full_call_identity_and_can_revert(
    tmp_path: Path,
) -> None:
    store = SqliteJobStore(tmp_path / "models.sqlite3")

    def task(alias: str) -> dict:
        return {
            "extraction_task": {
                "entity": {
                    "entity_id": "entity-shared",
                    "wiki_title": alias,
                    "cell_text": alias,
                    "context": ["context", alias],
                    "entity_column_index": 0,
                    "entity_column_name": "Name",
                },
                "asset": {
                    "asset_id": "shared-asset",
                    "asset_type": "text",
                    "content": "shared bytes",
                },
                "candidate_attribute_names": ["State", "Year"],
            }
        }

    jobsets = [
        enqueue_model_tasks(
            [task(alias)],
            store,
            args=model_args(),
            input_fingerprint=f"input-{index}",
        )
        for index, alias in enumerate(
            ("Alias One", "Alias Two", "Alias One")
        )
    ]

    assert (
        jobsets[0].jobs[0].model_call_key
        != jobsets[1].jobs[0].model_call_key
    )
    assert (
        jobsets[0].jobs[0].model_call_key
        == jobsets[2].jobs[0].model_call_key
    )
    extractor = CountingExtractor()
    records = []
    for jobset in jobsets:
        result = run_model_stage(
            store,
            extractor,
            jobset=jobset,
            output_root=tmp_path / "outputs",
        )
        records.append(
            json.loads(
                result.extraction_paths[0].read_text(
                    encoding="utf-8"
                ).splitlines()[0]
            )
        )
    assert extractor.asset_ids == ["shared-asset", "shared-asset"]
    assert (
        records[0]["entity_prompt_fingerprint"]
        != records[1]["entity_prompt_fingerprint"]
    )
    assert (
        records[0]["entity_prompt_fingerprint"]
        == records[2]["entity_prompt_fingerprint"]
    )


def test_expanded_shrunk_and_disjoint_jobsets_publish_exact_manifests(
    tmp_path: Path,
) -> None:
    store = SqliteJobStore(tmp_path / "models.sqlite3")
    extractor = CountingExtractor()
    variants = (
        ("expanded", [asset("a"), asset("b"), asset("c")]),
        ("shrunk", [asset("a")]),
        ("disjoint", [asset("x"), asset("y", "image")]),
    )
    results = []
    expected_tasks = []
    for fingerprint, tasks in variants:
        jobset = enqueue_model_tasks(
            tasks,
            store,
            args=model_args(),
            input_fingerprint=fingerprint,
        )
        expected_tasks.append(jobset.total_tasks)
        results.append(
            run_model_stage(
                store,
                extractor,
                jobset=jobset,
                output_root=tmp_path / "outputs",
            )
        )

    assert expected_tasks == [3, 1, 2]
    assert len({result.manifest_path for result in results}) == 3
    assert all(validate_model_stage(result) for result in results)
    assert [
        result.success + result.terminal for result in results
    ] == expected_tasks
    with sqlite3.connect(store.path) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM jobs"
        ).fetchone()[0] == sum(expected_tasks)
    assert Counter(extractor.asset_ids) == {
        "a": 1,
        "b": 1,
        "c": 1,
        "x": 1,
        "y": 1,
    }


def test_model_call_cache_isolated_by_full_payload_provenance(
    tmp_path: Path,
) -> None:
    store = SqliteJobStore(tmp_path / "models.sqlite3")
    extractor = CountingExtractor()
    variants = [
        (
            asset("a", content="old bytes"),
            model_args(),
            "input-content-old",
            "prompt-p1",
            "policy-p1",
        ),
        (
            asset("a", content="new bytes"),
            model_args(),
            "input-content-new",
            "prompt-p1",
            "policy-p1",
        ),
        (
            asset("a", content="old bytes"),
            model_args(),
            "input-prompt-p2",
            "prompt-p2",
            "policy-p1",
        ),
        (
            asset("a", content="old bytes"),
            model_args(text_model_name="text-v2"),
            "input-model-v2",
            "prompt-p1",
            "policy-p1",
        ),
        (
            {**asset("a", content="old bytes"), "candidate_attribute_names": ["Year"]},
            model_args(),
            "input-candidates-year",
            "prompt-p1",
            "policy-p1",
        ),
        (
            asset("a", content="old bytes"),
            model_args(),
            "input-policy-p2",
            "prompt-p1",
            "policy-p2",
        ),
    ]
    records = []
    for current, args, input_fp, prompt, policy in variants:
        jobset = enqueue_model_tasks(
            [current],
            store,
            args=args,
            input_fingerprint=input_fp,
            prompt_version=prompt,
            policy_fingerprint=policy,
        )
        result = run_model_stage(
            store,
            extractor,
            jobset=jobset,
            output_root=tmp_path / "outputs",
        )
        records.append(
            json.loads(
                result.extraction_paths[0].read_text(
                    encoding="utf-8"
                ).splitlines()[0]
            )
        )

    assert len(extractor.asset_ids) == len(variants)
    assert len({record["model_call_key"] for record in records}) == len(
        variants
    )
    for record, (_asset, args, _input, prompt, policy) in zip(
        records,
        variants,
    ):
        assert record["prompt_version"] == prompt
        assert record["model_identity"] == args.text_model_name
        assert record["policy_fingerprint"] == policy
        assert record["asset_fingerprint"]


def test_legacy_cache_requires_complete_matching_provenance(
    tmp_path: Path,
) -> None:
    store = SqliteJobStore(tmp_path / "models.sqlite3")
    jobset = enqueue_model_tasks(
        [asset("a")],
        store,
        args=model_args(),
        input_fingerprint="assets-v1",
    )
    incomplete = ExtractionCache(tmp_path / "legacy-cache.jsonl")
    incomplete.put(
        jobset.jobs[0].cache_key,
        {
            "cache_key": jobset.jobs[0].cache_key,
            "asset_id": "a",
            "entity_id": "entity-a",
            "attributes": [],
            "error": "",
        },
    )
    extractor = CountingExtractor()

    run_model_stage(
        store,
        extractor,
        jobset=jobset,
        cache=incomplete,
        output_root=tmp_path / "outputs",
    )

    assert extractor.asset_ids == ["a"]


def test_matching_legacy_cache_error_is_committed_as_terminal(
    tmp_path: Path,
) -> None:
    store = SqliteJobStore(tmp_path / "models.sqlite3")
    jobset = enqueue_model_tasks(
        [asset("a")],
        store,
        args=model_args(),
        input_fingerprint="assets-v1",
    )
    with sqlite3.connect(store.path) as connection:
        payload = json.loads(
            connection.execute(
                "SELECT payload_json FROM jobs WHERE job_id = ?",
                (jobset.jobs[0].job_id,),
            ).fetchone()[0]
        )
    record = {
        "cache_key": payload["cache_key"],
        "model_call_key": payload["model_call_key"],
        "prompt_version": payload["prompt_version"],
        "model_identity": payload["model_identity"],
        "asset_fingerprint": payload["asset_fingerprint"],
        "entity_prompt_fingerprint": payload[
            "entity_prompt_fingerprint"
        ],
        "asset_type": payload["modality"],
        "modality": payload["modality"],
        "policy_fingerprint": payload["policy_fingerprint"],
        "parser_schema_version": payload["parser_schema_version"],
        "candidate_attribute_names": payload[
            "candidate_attribute_names"
        ],
        "attributes": [],
        "raw_response": "",
        "error": "cached model failure",
    }
    cache = ExtractionCache(tmp_path / "legacy-error-cache.jsonl")
    cache.put(payload["cache_key"], record)

    result = run_model_stage(
        store,
        None,
        jobset=jobset,
        cache=cache,
        output_root=tmp_path / "outputs",
    )

    assert result.success == 0
    assert result.terminal == 1
    terminal = json.loads(
        result.error_paths[0].read_text(encoding="utf-8").splitlines()[0]
    )
    assert terminal["error"] == "cached model failure"


def test_duplicate_extraction_key_across_tables_enqueues_one_model_call(
    tmp_path: Path,
) -> None:
    first = asset("a")
    second = {**first, "source_table_id": "table-2", "source_row_id": 9}
    store = SqliteJobStore(tmp_path / "models.sqlite3")

    jobset = enqueue_model_tasks(
        [first, second],
        store,
        args=model_args(),
        input_fingerprint="assets-v1",
    )
    extractor = CountingExtractor()
    result = run_model_stage(
        store,
        extractor,
        jobset=jobset,
        output_root=tmp_path / "outputs",
    )

    assert result.text_total == 1
    assert extractor.asset_ids == ["a"]


def test_model_stage_never_exceeds_bounded_group(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import wdc200k_models

    observed: list[int] = []
    authoritative = wdc200k_models.run_extraction_task_group

    def recording_group(**kwargs):
        observed.append(len(kwargs["tasks"]))
        return authoritative(**kwargs)

    monkeypatch.setattr(
        wdc200k_models,
        "run_extraction_task_group",
        recording_group,
    )
    store = SqliteJobStore(tmp_path / "models.sqlite3")
    jobset = enqueue_model_tasks(
        [asset(str(index)) for index in range(7)],
        store,
        args=model_args(),
        input_fingerprint="assets-v1",
    )

    run_model_stage(
        store,
        CountingExtractor(),
        jobset=jobset,
        group_size=2,
        output_root=tmp_path / "outputs",
    )

    assert observed
    assert max(observed) <= 2


def test_heartbeat_prevents_second_worker_from_repeating_model_call(
    tmp_path: Path,
) -> None:
    extractor = CountingExtractor(delay=0.35)
    store = SqliteJobStore(tmp_path / "models.sqlite3")
    jobset = enqueue_model_tasks(
        [asset("a")],
        store,
        args=model_args(),
        input_fingerprint="assets-v1",
    )
    errors: list[BaseException] = []

    def worker(owner: str) -> None:
        try:
            run_model_stage(
                store,
                extractor,
                jobset=jobset,
                owner=owner,
                lease_seconds=0.12,
                heartbeat_seconds=0.03,
                output_root=tmp_path / "outputs",
            )
        except BaseException as error:
            errors.append(error)

    first = threading.Thread(target=worker, args=("worker-1",))
    second = threading.Thread(target=worker, args=("worker-2",))
    first.start()
    time.sleep(0.18)
    second.start()
    first.join(timeout=3)
    second.join(timeout=3)

    assert not errors
    assert extractor.asset_ids == ["a"]


def test_heartbeat_zero_row_discards_late_result_and_shared_cache(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import wdc200k_models

    store = SqliteJobStore(tmp_path / "models.sqlite3")
    jobset = enqueue_model_tasks(
        [asset("a")],
        store,
        args=model_args(),
        input_fingerprint="assets-v1",
    )
    monkeypatch.setattr(
        wdc200k_models,
        "_extend_leases",
        lambda *_args, **_kwargs: 0,
    )

    result = run_model_stage(
        store,
        CountingExtractor(delay=0.12),
        jobset=jobset,
        lease_seconds=1,
        heartbeat_seconds=0.02,
        output_root=tmp_path / "outputs",
    )

    assert result.complete is False
    with sqlite3.connect(store.path) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM model_results"
        ).fetchone()[0] == 0
        assert connection.execute(
            "SELECT COUNT(*) FROM model_call_cache"
        ).fetchone()[0] == 0


def test_stolen_lease_fences_old_worker_result_and_cache(
    tmp_path: Path,
) -> None:
    import sqlite3

    started = threading.Event()
    release = threading.Event()

    class BlockingExtractor(CountingExtractor):
        def extract(self, current_asset, entity, candidates):
            started.set()
            release.wait(timeout=2)
            return super().extract(current_asset, entity, candidates)

    store = SqliteJobStore(tmp_path / "models.sqlite3")
    jobset = enqueue_model_tasks(
        [asset("a")],
        store,
        args=model_args(),
        input_fingerprint="assets-v1",
    )
    failures: list[BaseException] = []
    results = []

    def run_old() -> None:
        try:
            results.append(
                run_model_stage(
                    store,
                    BlockingExtractor(),
                    jobset=jobset,
                    owner="old-owner",
                    lease_seconds=5,
                    heartbeat_seconds=1,
                    output_root=tmp_path / "outputs",
                )
            )
        except BaseException as error:
            failures.append(error)

    worker = threading.Thread(target=run_old)
    worker.start()
    assert started.wait(timeout=1)
    with sqlite3.connect(store.path) as connection:
        connection.execute(
            """
            UPDATE jobs
            SET owner = 'new-owner', lease_id = 'new-lease',
                lease_expires = ?
            WHERE job_id = ?
            """,
            (time.time() + 5, jobset.jobs[0].job_id),
        )
    release.set()
    worker.join(timeout=2)

    assert not worker.is_alive()
    assert not failures
    assert results and results[0].complete is False
    with sqlite3.connect(store.path) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM model_results"
        ).fetchone()[0] == 0
        assert connection.execute(
            "SELECT COUNT(*) FROM model_call_cache"
        ).fetchone()[0] == 0


def test_stolen_prepared_result_is_not_visible_as_shared_cache(
    tmp_path: Path,
) -> None:
    prepared = threading.Event()
    release = threading.Event()
    store = SqliteJobStore(tmp_path / "models.sqlite3")
    jobset = enqueue_model_tasks(
        [asset("a")],
        store,
        args=model_args(),
        input_fingerprint="assets-v1",
    )

    def pause_after_prepare(_job_id: str, _record: dict) -> None:
        prepared.set()
        release.wait(timeout=2)

    old_extractor = CountingExtractor()
    old_result = []

    def old_worker() -> None:
        old_result.append(
            run_model_stage(
                store,
                old_extractor,
                jobset=jobset,
                owner="old-owner",
                lease_seconds=5,
                heartbeat_seconds=1,
                output_root=tmp_path / "outputs",
                after_result_write=pause_after_prepare,
            )
        )

    worker = threading.Thread(target=old_worker)
    worker.start()
    assert prepared.wait(timeout=1)
    with sqlite3.connect(store.path) as connection:
        connection.execute(
            """
            UPDATE jobs
            SET owner = 'new-owner', lease_id = 'new-lease',
                lease_expires = ?
            WHERE job_id = ?
            """,
            (time.time() + 5, jobset.jobs[0].job_id),
        )
    release.set()
    worker.join(timeout=2)
    assert not worker.is_alive()
    assert old_result and old_result[0].complete is False
    with sqlite3.connect(store.path) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM model_call_cache"
        ).fetchone()[0] == 0
        connection.execute(
            """
            UPDATE jobs
            SET status = 'pending', owner = NULL,
                lease_id = NULL, lease_expires = NULL
            WHERE job_id = ?
            """,
            (jobset.jobs[0].job_id,),
        )

    new_extractor = CountingExtractor()
    completed = run_model_stage(
        store,
        new_extractor,
        jobset=jobset,
        owner="new-owner",
        output_root=tmp_path / "outputs",
    )

    assert completed.complete is True
    assert new_extractor.asset_ids == ["a"]


def test_start_marker_requires_complete_upstream_manifests_and_is_fenced(
    tmp_path: Path,
) -> None:
    network, assets = write_strict_upstream_barriers(tmp_path)
    marker = tmp_path / "start.json"
    store = SqliteJobStore(tmp_path / "models.sqlite3")
    jobset = enqueue_model_tasks(
        [asset("a"), asset("b", "image")],
        store,
        args=model_args(),
        input_fingerprint="assets-v1",
    )

    faux_assets = tmp_path / "faux-assets.json"
    faux_assets.write_text('{"complete": true}', encoding="utf-8")
    with pytest.raises(ValueError, match="Task-5"):
        write_model_start_marker(
            marker,
            jobset,
            network_manifests=[network],
            assets_manifest=faux_assets,
            assets_barrier=task5_barrier(),
            run_fingerprint="run-v1",
        )
    assert not marker.exists()

    write_model_start_marker(
        marker,
        jobset,
        network_manifests=[network],
        assets_manifest=assets,
        assets_barrier=task5_barrier(),
        run_fingerprint="run-v1",
    )
    payload = json.loads(marker.read_text(encoding="utf-8"))
    assert payload["stage"] == "wdc200k_model_start"
    assert payload["schema_version"] == "wdc200k-model-markers-v1"
    assert len(payload["start_fingerprint"]) == 64
    assert payload["text_task_count"] == 1
    assert payload["image_task_count"] == 1
    assert payload["run_fingerprint"] == "run-v1"
    assert payload["text_jobset_fingerprint"] == jobset.text_fingerprint
    assert payload["image_jobset_fingerprint"] == jobset.image_fingerprint


def test_model_start_marker_guard_failure_preserves_existing_marker(
    tmp_path: Path,
) -> None:
    network, assets = write_strict_upstream_barriers(tmp_path)
    marker = tmp_path / "start.json"
    marker.write_text('{"old": true}\n', encoding="utf-8")
    store = SqliteJobStore(tmp_path / "models.sqlite3")
    jobset = enqueue_model_tasks(
        [asset("a")],
        store,
        args=model_args(),
    )
    calls = 0

    def fail_commit(_path: Path, _estimated: int) -> None:
        nonlocal calls
        calls += 1
        if calls == 3:
            raise RuntimeError("reserve exhausted")

    with pytest.raises(RuntimeError, match="reserve exhausted"):
        write_model_start_marker(
            marker,
            jobset,
            network_manifests=[network],
            assets_manifest=assets,
            assets_barrier=task5_barrier(),
            run_fingerprint="run-v1",
            pre_write_guard=fail_commit,
        )
    assert json.loads(marker.read_text(encoding="utf-8")) == {"old": True}
    assert not marker.with_suffix(".json.tmp").exists()


def test_model_done_marker_guard_failure_preserves_existing_marker(
    tmp_path: Path,
) -> None:
    marker = tmp_path / "done.json"
    marker.write_text('{"old": true}\n', encoding="utf-8")
    calls = 0

    def fail_commit(_path: Path, _estimated: int) -> None:
        nonlocal calls
        calls += 1
        if calls == 3:
            raise RuntimeError("reserve exhausted")

    with pytest.raises(RuntimeError, match="reserve exhausted"):
        write_model_done_marker(
            marker,
            model_kind="text",
            task_count=1,
            jobset_fingerprint="text-v1",
            run_fingerprint="run-v1",
            text_jobset_fingerprint="text-v1",
            image_jobset_fingerprint="image-v1",
            text_task_count=1,
            image_task_count=1,
            start_fingerprint="a" * 64,
            pre_write_guard=fail_commit,
        )
    assert json.loads(marker.read_text(encoding="utf-8")) == {"old": True}
    assert not marker.with_suffix(".json.tmp").exists()


def test_model_stage_owns_start_ready_marker_handshake(
    tmp_path: Path,
) -> None:
    network, assets_manifest = write_strict_upstream_barriers(tmp_path)
    start = tmp_path / "start.json"
    ready = tmp_path / "ready.json"
    store = SqliteJobStore(tmp_path / "models.sqlite3")
    jobset = enqueue_model_tasks(
        [asset("a")],
        store,
        args=model_args(),
        input_fingerprint="assets-v1",
    )
    start_fingerprint = write_model_start_marker(
        start,
        jobset,
        network_manifests=[network],
        assets_manifest=assets_manifest,
        assets_barrier=task5_barrier(),
        run_fingerprint="run-v1",
    )
    write_strict_ready_marker(
        ready,
        jobset,
        run_fingerprint="run-v1",
        start_fingerprint=start_fingerprint,
    )

    result = run_model_stage(
        store,
        CountingExtractor(),
        jobset=jobset,
        output_root=tmp_path / "outputs",
        start_marker=start,
        ready_marker=ready,
        network_manifests=[network],
        assets_manifest=assets_manifest,
        assets_barrier=task5_barrier(),
        run_fingerprint="run-v1",
        ready_timeout_seconds=1,
    )

    assert result.complete is True
    assert json.loads(start.read_text(encoding="utf-8"))[
        "run_fingerprint"
    ] == "run-v1"


def test_ready_marker_rejects_stale_modality_jobsets(
    tmp_path: Path,
) -> None:
    ready = tmp_path / "ready.json"
    ready.write_text(
        json.dumps(
            {
                "run_fingerprint": "run-v1",
                "text_jobset_fingerprint": "old-text",
                "image_jobset_fingerprint": "image-v1",
            }
        ),
        encoding="utf-8",
    )

    assert not marker_matches(
        ready,
        run_fingerprint="run-v1",
        text_jobset_fingerprint="text-v1",
        image_jobset_fingerprint="image-v1",
    )


def test_ready_marker_requires_exact_status_and_kind(tmp_path: Path) -> None:
    ready = tmp_path / "ready.json"
    payload = {
        "status": "wrong_ready_state",
        "model_kind": "text",
        "run_fingerprint": "run-v1",
        "text_jobset_fingerprint": "text-v1",
        "image_jobset_fingerprint": "image-v1",
    }
    ready.write_text(json.dumps(payload), encoding="utf-8")

    assert not marker_matches(
        ready,
        expected_status="vllm_servers_ready",
        run_fingerprint="run-v1",
        text_jobset_fingerprint="text-v1",
        image_jobset_fingerprint="image-v1",
    )


def test_model_markers_require_schema_stage_and_start_identity(
    tmp_path: Path,
) -> None:
    ready = tmp_path / "ready.json"
    base = {
        "status": "vllm_servers_ready",
        "model_kind": "text+image",
        "run_fingerprint": "run-v1",
        "text_jobset_fingerprint": "text-v1",
        "image_jobset_fingerprint": "image-v1",
        "text_task_count": 2,
        "image_task_count": 3,
        "start_fingerprint": "a" * 64,
        "timestamp": 0.0,
    }
    ready.write_text(json.dumps(base), encoding="utf-8")
    assert not marker_matches(
        ready,
        expected_stage="wdc200k_model_ready",
        expected_status="vllm_servers_ready",
        run_fingerprint="run-v1",
        text_jobset_fingerprint="text-v1",
        image_jobset_fingerprint="image-v1",
        text_task_count=2,
        image_task_count=3,
        start_fingerprint="a" * 64,
    )

    ready.write_text(
        json.dumps(
            {
                **base,
                "stage": "wdc200k_model_ready",
                "schema_version": "wdc200k-model-markers-v1",
            }
        ),
        encoding="utf-8",
    )
    assert marker_matches(
        ready,
        expected_stage="wdc200k_model_ready",
        expected_status="vllm_servers_ready",
        run_fingerprint="run-v1",
        text_jobset_fingerprint="text-v1",
        image_jobset_fingerprint="image-v1",
        text_task_count=2,
        image_task_count=3,
        start_fingerprint="a" * 64,
    )
    assert not marker_matches(
        ready,
        expected_stage="wdc200k_model_ready",
        expected_status="vllm_servers_ready",
        run_fingerprint="run-v1",
        text_jobset_fingerprint="text-v1",
        image_jobset_fingerprint="image-v1",
        text_task_count=2,
        image_task_count=3,
        start_fingerprint="b" * 64,
    )


def test_faux_complete_network_barrier_is_rejected(tmp_path: Path) -> None:
    _, assets = write_strict_upstream_barriers(tmp_path)
    network = tmp_path / "faux-network.json"
    network.write_text('{"complete": true}', encoding="utf-8")
    store = SqliteJobStore(tmp_path / "models.sqlite3")
    jobset = enqueue_model_tasks([asset("a")], store, args=model_args())

    with pytest.raises(ValueError, match="network"):
        write_model_start_marker(
            tmp_path / "start.json",
            jobset,
            network_manifests=[network],
            assets_manifest=assets,
            assets_barrier=task5_barrier(),
            run_fingerprint="run-v1",
        )


def test_minimal_task5_fingerprint_is_rejected_by_start_barrier(
    tmp_path: Path,
) -> None:
    network, assets = write_strict_upstream_barriers(tmp_path)
    payload = json.loads(assets.read_text(encoding="utf-8"))
    payload["fingerprint"] = {
        "schema_version": "wdc200k-asset-materialization-v1"
    }
    assets.write_text(json.dumps(payload), encoding="utf-8")
    store = SqliteJobStore(tmp_path / "models.sqlite3")
    jobset = enqueue_model_tasks([asset("a")], store, args=model_args())

    with pytest.raises(ValueError, match="fingerprint"):
        write_model_start_marker(
            tmp_path / "start.json",
            jobset,
            network_manifests=[network],
            assets_manifest=assets,
            assets_barrier=task5_barrier(),
            run_fingerprint="run-v1",
        )


def test_task5_barrier_rejects_forged_fingerprint_and_wrong_counts(
    tmp_path: Path,
) -> None:
    network, assets = write_strict_upstream_barriers(tmp_path)
    store = SqliteJobStore(tmp_path / "models.sqlite3")
    jobset = enqueue_model_tasks([asset("a")], store, args=model_args())
    forged = task5_fingerprint()
    forged["planning_manifest_sha256"] = "f" * 64

    with pytest.raises(ValueError, match="fingerprint"):
        write_model_start_marker(
            tmp_path / "forged-start.json",
            jobset,
            network_manifests=[network],
            assets_manifest=assets,
            assets_barrier=task5_barrier(fingerprint=forged),
            run_fingerprint="run-v1",
        )
    with pytest.raises(ValueError, match="count"):
        write_model_start_marker(
            tmp_path / "wrong-count-start.json",
            jobset,
            network_manifests=[network],
            assets_manifest=assets,
            assets_barrier=task5_barrier(bridge_assets=2),
            run_fingerprint="run-v1",
        )


def test_task5_manifest_is_checksum_validated_before_streaming_assets(
    tmp_path: Path,
) -> None:
    from wdc200k_io import AtomicJsonlShard

    root = tmp_path / "assets"
    writer = AtomicJsonlShard(root / "bridge_assets" / "part-00000.jsonl")
    writer.write(asset("a"))
    writer.write({**asset("bad"), "status": "terminal"})
    completed = writer.commit()
    relative_path = (
        root / "bridge_assets" / completed.path
    ).relative_to(root).as_posix()
    link_writer = AtomicJsonlShard(
        root / "table_asset_links" / "part-00000.jsonl"
    )
    link_writer.write(
        {
            "source_table_id": "table-1",
            "row_id": 0,
            "entity_id": "entity-a",
            "asset_ids": ["a"],
        }
    )
    link_completed = link_writer.commit()
    link_relative = (
        root / "table_asset_links" / link_completed.path
    ).relative_to(root).as_posix()
    manifest = root / "asset-materialization-manifest.json"
    manifest.write_text(
        json.dumps(
                {
                    "stage": "wdc200k_asset_materialization",
                    "fingerprint": task5_fingerprint(),
                    "complete": True,
                    "bridge_asset_shards": [
                    {
                        "path": relative_path,
                        "records": completed.records,
                        "bytes": completed.bytes,
                            "sha256": completed.sha256,
                        }
                    ],
                    "table_asset_link_shards": [
                        {
                            "path": link_relative,
                            "records": link_completed.records,
                            "bytes": link_completed.bytes,
                            "sha256": link_completed.sha256,
                        }
                    ],
                }
        ),
        encoding="utf-8",
    )

    assert [
        record["asset_id"]
        for record in iter_assets_from_materialization_manifest(
            manifest,
            assets_barrier=task5_barrier(bridge_assets=2),
        )
    ] == ["a"]

    (root / relative_path).write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="validation"):
        list(
            iter_assets_from_materialization_manifest(
                manifest,
                assets_barrier=task5_barrier(bridge_assets=2),
            )
        )

    network, _ = write_strict_upstream_barriers(tmp_path / "strict")
    store = SqliteJobStore(tmp_path / "models.sqlite3")
    jobset = enqueue_model_tasks(
        [asset("a")],
        store,
        args=model_args(),
        input_fingerprint="assets-v1",
    )
    with pytest.raises(ValueError, match="validation"):
        write_model_start_marker(
            tmp_path / "start.json",
            jobset,
            network_manifests=[network],
            assets_manifest=manifest,
            assets_barrier=task5_barrier(bridge_assets=2),
            run_fingerprint="run-v1",
        )


def test_enqueue_from_task5_manifest_requires_explicit_task_factory(
    tmp_path: Path,
) -> None:
    from wdc200k_io import AtomicJsonlShard

    root = tmp_path / "assets"
    writer = AtomicJsonlShard(root / "bridge_assets" / "part-00000.jsonl")
    writer.write(asset("a"))
    writer.write({**asset("bad"), "status": "terminal"})
    completed = writer.commit()
    relative = (
        root / "bridge_assets" / completed.path
    ).relative_to(root).as_posix()
    manifest = root / "asset-materialization-manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "stage": "wdc200k_asset_materialization",
                "complete": True,
                "bridge_asset_shards": [
                    {
                        "path": relative,
                        "records": completed.records,
                        "bytes": completed.bytes,
                        "sha256": completed.sha256,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    store = SqliteJobStore(tmp_path / "models.sqlite3")

    with pytest.raises(ValueError, match="task_factory"):
        enqueue_model_tasks_from_manifest(
            manifest,
            store,
            args=model_args(),
            input_fingerprint="assets-v1",
        )


def test_enqueue_from_strict_task5_manifest_with_factory(
    tmp_path: Path,
) -> None:
    _network, manifest = write_strict_upstream_barriers(tmp_path)
    store = SqliteJobStore(tmp_path / "models.sqlite3")

    jobset = enqueue_model_tasks_from_manifest(
        manifest,
        store,
        args=model_args(),
        assets_barrier=task5_barrier(),
        input_fingerprint="assets-v1",
        task_factory=lambda current_asset: current_asset,
    )

    assert jobset.text_tasks == 1
    assert jobset.image_tasks == 0


@pytest.mark.parametrize(
    ("declared_sha256", "create_file"),
    [
        ("f" * 64, True),
        (None, False),
    ],
)
def test_task5_image_enqueue_rejects_missing_or_mismatched_bytes(
    tmp_path: Path,
    declared_sha256: str | None,
    create_file: bool,
) -> None:
    manifest, barrier, _image_path = write_task5_image_manifest(
        tmp_path,
        declared_sha256=declared_sha256,
        create_file=create_file,
    )

    with pytest.raises(
        ValueError,
        match="image.*(missing|hash|content-addressed)",
    ):
        enqueue_model_tasks_from_manifest(
            manifest,
            SqliteJobStore(tmp_path / "models.sqlite3"),
            args=model_args(),
            assets_barrier=barrier,
            input_fingerprint="task5-images",
            task_factory=lambda current_asset: current_asset,
        )


def test_task5_image_bytes_are_fingerprinted_and_rechecked_before_model_call(
    tmp_path: Path,
) -> None:
    manifest, barrier, image_path = write_task5_image_manifest(tmp_path)
    store = SqliteJobStore(tmp_path / "models.sqlite3")
    jobset = enqueue_model_tasks_from_manifest(
        manifest,
        store,
        args=model_args(),
        assets_barrier=barrier,
        input_fingerprint="task5-images",
        task_factory=lambda current_asset: current_asset,
    )
    expected = hashlib.sha256(b"FIRST").hexdigest()
    with sqlite3.connect(store.path) as connection:
        payload = json.loads(
            connection.execute(
                "SELECT payload_json FROM jobs WHERE job_id = ?",
                (jobset.jobs[0].job_id,),
            ).fetchone()[0]
        )
    assert payload["asset"]["verified_content_sha256"] == expected
    assert jobset.jobs[0].asset_fingerprint == hashlib.sha256(
        json.dumps(
            payload["asset"],
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()

    image_path.write_bytes(b"SECOND")
    extractor = CountingExtractor()
    result = run_model_stage(
        store,
        extractor,
        jobset=jobset,
        output_root=tmp_path / "outputs",
    )

    assert extractor.asset_ids == []
    assert result.success == 0
    assert result.terminal == 1
    terminal = json.loads(
        result.error_paths[0].read_text(encoding="utf-8").splitlines()[0]
    )
    assert "image content hash changed" in terminal["error"]
    with pytest.raises(ValueError, match="image.*hash"):
        enqueue_model_tasks_from_manifest(
            manifest,
            SqliteJobStore(tmp_path / "second.sqlite3"),
            args=model_args(),
            assets_barrier=barrier,
            input_fingerprint="same-json-second-bytes",
            task_factory=lambda current_asset: current_asset,
        )


def test_task3_task5_adapter_builds_real_non_empty_candidate_tasks(
    tmp_path: Path,
) -> None:
    from dataclasses import asdict
    from stage1_io import stable_hash
    from wdc200k_io import AtomicJsonlShard

    structural_root = tmp_path / "structural"
    source = {
        "source_table_id": "source-1",
        "source_file": "Thing/Thing_test.json.gz",
        "page_title": "Thing",
        "caption": "",
        "section_title": "",
        "num_rows": 1,
        "num_cols": 2,
        "columns": [
            {"column_index": 0, "column_name": "name"},
            {"column_index": 1, "column_name": "State"},
        ],
        "rows": [
            {
                "row_id": 0,
                "cells": [
                    {
                        "column_index": 0,
                        "column_name": "name",
                        "text": "Alpha",
                        "wiki_title": "wdc_alpha",
                    },
                    {
                        "column_index": 1,
                        "column_name": "State",
                        "text": "",
                        "wiki_title": None,
                    },
                ],
            }
        ],
        "provenance_builder": "build_wdc_mm_joinability_dataset.py",
        "metadata": {
            "candidate_entity_columns": [0],
            "column_profiles": [
                {
                    "column_index": 0,
                    "wiki_link_ratio": 1.0,
                    "non_empty_ratio": 1.0,
                },
                {
                    "column_index": 1,
                    "wiki_link_ratio": 0.0,
                    "non_empty_ratio": 1.0,
                },
            ],
        },
    }
    entity_record = {
        "entity_id": "entity-alpha",
        "wiki_title": "wdc_alpha",
        "display_texts": ["Alpha"],
        "context_terms": ["State"],
        "appears_in": [
            {
                "source_table_id": "source-1",
                "query_view_id": None,
                "row_id": 0,
                "column_index": 0,
                "column_name": "name",
            }
        ],
        "page_url": "https://example.test/alpha",
        "image_urls": [],
    }
    shard_records = {
        "source_tables/part-00000.jsonl": [source],
        "entities/part-00000.jsonl": [entity_record],
        "page_refs/part-00000.jsonl": [
            {
                "entity_id": "entity-alpha",
                "page_url": "https://example.test/alpha",
            }
        ],
        "direct_image_refs/part-00000.jsonl": [],
        "structural_failures/part-00000.jsonl": [],
        "selection/validated-00000.jsonl": [
            {"relative_path": "Thing/test.json.gz", "rows": 1}
        ],
    }
    completed = []
    for relative, records in shard_records.items():
        writer = AtomicJsonlShard(structural_root / relative)
        for record in records:
            writer.write(record)
        item = asdict(writer.commit())
        item["path"] = relative
        completed.append(item)
    structural_manifest = (
        structural_root / "stage_manifests" / "structural-00000.json"
    )
    structural_manifest.parent.mkdir(parents=True, exist_ok=True)
    structural_manifest.write_text(
        json.dumps(
                {
                    "stage": "wdc200k_structural",
                    "schema_version": "wdc200k-structural-v2",
                    "input_fingerprint": "selection-v1",
                "parameter_fingerprint": "structural-v2",
                "completed_shards": completed,
                "complete": True,
            }
        ),
        encoding="utf-8",
    )
    final_writer = AtomicJsonlShard(
        structural_root
        / "selection"
        / "validated-selected-tables.jsonl"
    )
    final_writer.write(
        {"relative_path": "Thing/test.json.gz", "rows": 1}
    )
    final_completed = asdict(final_writer.commit())
    final_completed["path"] = (
        "selection/validated-selected-tables.jsonl"
    )
    final_manifest = (
        structural_root
        / "stage_manifests"
        / "validated-selection-global.json"
    )
    structural_manifest_sha = hashlib.sha256(
        structural_manifest.read_bytes()
    ).hexdigest()
    final_manifest.write_text(
        json.dumps(
                {
                    "stage": "wdc200k_validated_selection",
                    "schema_version": "wdc200k-structural-v2",
                    "input_fingerprint": stable_hash(
                    "wdc200k-structural-v2",
                    (
                        f"{structural_manifest.resolve()}:"
                        f"{structural_manifest_sha}"
                    ),
                    length=40,
                ),
                "parameter_fingerprint": stable_hash(
                    "validated-selection-global-v1",
                    1,
                    length=40,
                ),
                "completed_shards": [final_completed],
                "complete": True,
            }
        ),
        encoding="utf-8",
    )
    structural_key = structural_manifest.resolve().as_posix()
    structural_barrier = models.StructuralStageBarrier(
        schema_version="wdc200k-structural-v2",
        manifest_count=1,
        manifest_sha256={
            structural_key: structural_manifest_sha,
        },
        input_fingerprints={
            structural_key: "selection-v1",
        },
        parameter_fingerprints={
            structural_key: "structural-v2",
        },
        final_manifest_sha256=hashlib.sha256(
            final_manifest.read_bytes()
        ).hexdigest(),
        final_selection=final_completed,
    )

    assets_root = tmp_path / "assets"
    asset_writer = AtomicJsonlShard(
        assets_root / "bridge_assets" / "part-00000.jsonl"
    )
    asset_writer.write(asset("asset-alpha"))
    asset_completed = asdict(asset_writer.commit())
    asset_completed["path"] = "bridge_assets/part-00000.jsonl"
    link_writer = AtomicJsonlShard(
        assets_root / "table_asset_links" / "part-00000.jsonl"
    )
    link_writer.write(
        {
            "source_table_id": "source-1",
            "row_id": 0,
            "entity_id": "entity-alpha",
            "asset_ids": ["asset-alpha"],
        }
    )
    link_completed = asdict(link_writer.commit())
    link_completed["path"] = "table_asset_links/part-00000.jsonl"
    assets_manifest = assets_root / "asset-materialization-manifest.json"
    assets_manifest.write_text(
        json.dumps(
            {
                "stage": "wdc200k_asset_materialization",
                "fingerprint": task5_fingerprint(),
                "bridge_asset_shards": [asset_completed],
                "table_asset_link_shards": [link_completed],
                "complete": True,
            }
        ),
        encoding="utf-8",
    )

    adapted = adapt_model_tasks_from_manifests(
        structural_output_root=structural_root,
        structural_manifests=[structural_manifest],
        finalized_selection_manifest=final_manifest,
        structural_barrier=structural_barrier,
        assets_manifest=assets_manifest,
        assets_barrier=task5_barrier(),
        output_root=tmp_path / "adapted",
        args=model_args(),
    )
    records = [
        json.loads(line)
        for path in adapted.task_paths
        for line in path.read_text(encoding="utf-8").splitlines()
    ]

    assert adapted.tasks == 1
    assert adapted.errors == 0
    assert records[0]["extraction_task"]["candidate_attribute_names"] == ["State"]
    assert (
        records[0]["extraction_task"]["entity"]["entity_id"]
        == "entity-alpha"
    )

    from dataclasses import replace

    with pytest.raises(ValueError, match="structural.*schema"):
        adapt_model_tasks_from_manifests(
            structural_output_root=structural_root,
            structural_manifests=[structural_manifest],
            finalized_selection_manifest=final_manifest,
            structural_barrier=replace(
                structural_barrier,
                schema_version="arbitrary-structural-v2",
            ),
            assets_manifest=assets_manifest,
            assets_barrier=task5_barrier(),
            output_root=tmp_path / "bad-schema-adapted",
            args=model_args(),
        )

    corrupted = json.loads(final_manifest.read_text(encoding="utf-8"))
    corrupted["input_fingerprint"] = "foreign-structural-set"
    final_manifest.write_text(json.dumps(corrupted), encoding="utf-8")
    with pytest.raises(ValueError, match="(fingerprint|checksum)"):
        adapt_model_tasks_from_manifests(
            structural_output_root=structural_root,
            structural_manifests=[structural_manifest],
            finalized_selection_manifest=final_manifest,
            structural_barrier=structural_barrier,
            assets_manifest=assets_manifest,
            assets_barrier=task5_barrier(),
            output_root=tmp_path / "corrupt-adapted",
            args=model_args(),
        )
