import argparse
import hashlib
import json
import sqlite3
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
    MODEL_PARSER_SCHEMA_VERSION,
    adapt_model_tasks_from_manifests,
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
                "fingerprint": {
                    "schema_version": "wdc200k-asset-materialization-v1"
                },
                "bridge_asset_shards": [asset_shard],
                "table_asset_link_shards": [link_shard],
                "complete": True,
            }
        ),
        encoding="utf-8",
    )
    return network, assets


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
            run_fingerprint="run-v1",
        )
    assert not marker.exists()

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
                    "fingerprint": {
                        "schema_version": "wdc200k-asset-materialization-v1"
                    },
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
        for record in iter_assets_from_materialization_manifest(manifest)
    ] == ["a"]

    (root / relative_path).write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="validation"):
        list(iter_assets_from_materialization_manifest(manifest))

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


def test_task3_task5_adapter_builds_real_non_empty_candidate_tasks(
    tmp_path: Path,
) -> None:
    from dataclasses import asdict
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
    final_manifest.write_text(
        json.dumps(
            {
                "stage": "wdc200k_validated_selection",
                "input_fingerprint": "structural-v2",
                "parameter_fingerprint": "validated-selection-global-v1",
                "completed_shards": [final_completed],
                "complete": True,
            }
        ),
        encoding="utf-8",
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
                "fingerprint": {
                    "schema_version": "wdc200k-asset-materialization-v1"
                },
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
        assets_manifest=assets_manifest,
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
