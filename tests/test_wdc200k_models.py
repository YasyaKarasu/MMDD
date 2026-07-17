import argparse
import json
import sys
import threading
import time
from collections import Counter
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from build_mm_joinability_dataset import ExtractionCache
from wdc200k_io import SqliteJobStore
from wdc200k_models import (
    enqueue_model_tasks,
    enqueue_model_tasks_from_manifest,
    iter_assets_from_materialization_manifest,
    marker_matches,
    run_model_stage,
    validate_model_stage,
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


def test_model_stage_resumes_without_repeating_success(tmp_path: Path) -> None:
    extractor = CountingExtractor()
    store = SqliteJobStore(tmp_path / "models.sqlite3")
    jobset = enqueue_model_tasks(
        [asset("a"), asset("b", "image")],
        store,
    )
    text_done = tmp_path / "text-done.json"
    image_done = tmp_path / "image-done.json"

    first = run_model_stage(
        store,
        extractor,
        stop_after=1,
        output_root=tmp_path / "outputs",
        text_done_marker=text_done,
        image_done_marker=image_done,
        run_fingerprint="run-v1",
    )
    assert first.complete is False
    assert text_done.exists()
    assert not image_done.exists()
    second = run_model_stage(
        store,
        extractor,
        output_root=tmp_path / "outputs",
        text_done_marker=text_done,
        image_done_marker=image_done,
        run_fingerprint="run-v1",
    )

    assert second.complete is True
    assert image_done.exists()
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
        )

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
        )

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
            "prompt_version": jobset.prompt_version,
            "entity_id": "entity-a",
            "entity_text": "Entity a",
            "entity_wiki_title": "Entity a",
            "asset_id": "a",
            "asset_type": "text",
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


def test_start_marker_requires_complete_upstream_manifests_and_is_fenced(
    tmp_path: Path,
) -> None:
    network = tmp_path / "network.json"
    assets = tmp_path / "assets.json"
    marker = tmp_path / "start.json"
    network.write_text('{"complete": true}', encoding="utf-8")
    assets.write_text('{"complete": false}', encoding="utf-8")
    store = SqliteJobStore(tmp_path / "models.sqlite3")
    jobset = enqueue_model_tasks(
        [asset("a"), asset("b", "image")],
        store,
        args=model_args(),
        input_fingerprint="assets-v1",
    )

    with pytest.raises(ValueError, match="complete"):
        write_model_start_marker(
            marker,
            jobset,
            network_manifests=[network],
            assets_manifest=assets,
            run_fingerprint="run-v1",
        )
    assert not marker.exists()

    assets.write_text('{"complete": true}', encoding="utf-8")
    write_model_start_marker(
        marker,
        jobset,
        network_manifests=[network],
        assets_manifest=assets,
        run_fingerprint="run-v1",
    )
    payload = json.loads(marker.read_text(encoding="utf-8"))
    assert payload["text_task_count"] == 1
    assert payload["image_task_count"] == 1
    assert payload["run_fingerprint"] == "run-v1"
    assert payload["text_jobset_fingerprint"] == jobset.text_fingerprint
    assert payload["image_jobset_fingerprint"] == jobset.image_fingerprint


def test_model_stage_owns_start_ready_marker_handshake(
    tmp_path: Path,
) -> None:
    network = tmp_path / "network.json"
    assets_manifest = tmp_path / "assets.json"
    start = tmp_path / "start.json"
    ready = tmp_path / "ready.json"
    network.write_text('{"complete": true}', encoding="utf-8")
    assets_manifest.write_text('{"complete": true}', encoding="utf-8")
    store = SqliteJobStore(tmp_path / "models.sqlite3")
    jobset = enqueue_model_tasks(
        [asset("a")],
        store,
        args=model_args(),
        input_fingerprint="assets-v1",
    )
    ready.write_text(
        json.dumps(
            {
                "status": "vllm_servers_ready",
                "run_fingerprint": "run-v1",
                "text_jobset_fingerprint": jobset.text_fingerprint,
                "image_jobset_fingerprint": jobset.image_fingerprint,
            }
        ),
        encoding="utf-8",
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
        run_fingerprint="run-v1",
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
    manifest = root / "asset-materialization-manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "stage": "wdc200k_asset_materialization",
                "complete": True,
                "bridge_asset_shards": [
                    {
                        "path": relative_path,
                        "records": completed.records,
                        "bytes": completed.bytes,
                        "sha256": completed.sha256,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    assert [
        record["asset_id"]
        for record in iter_assets_from_materialization_manifest(manifest)
    ] == ["a"]

    (root / relative_path).write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="validation"):
        list(iter_assets_from_materialization_manifest(manifest))

    network = tmp_path / "network.json"
    network.write_text('{"complete": true}', encoding="utf-8")
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
            run_fingerprint="run-v1",
        )


def test_enqueue_from_task5_manifest_filters_non_success_assets(
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

    jobset = enqueue_model_tasks_from_manifest(
        manifest,
        store,
        args=model_args(),
        input_fingerprint="assets-v1",
    )

    assert jobset.text_tasks == 1
    assert jobset.image_tasks == 0
