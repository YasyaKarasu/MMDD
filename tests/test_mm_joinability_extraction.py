import argparse
import json
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts_old"))

import build_mm_joinability_dataset as joinability_dataset
import run_mm_joinability_dynamic_vllm as dynamic_vllm_runner
from build_mm_joinability_dataset import (
    ExtractionCache,
    ExtractionTask,
    ModelConcurrencyState,
    ModelAutoCheckStats,
    LocalAttributeExtractor,
    apply_model_auto_check,
    build_bridge_assets_parallel,
    extraction_row_attributes,
    extraction_cache_key,
    image_data_url,
    normalize_extracted_attributes,
    project_selected_rows,
    precompute_extraction_task_groups,
    reparse_extraction_record,
    recovery_column_profile,
    required_recovered_row_count,
    resolve_extraction_tasks,
    safe_json_object,
    select_query_source_rows,
    select_query_source_row_views,
    select_best_qualified_column,
    tasks_requiring_model_analysis,
    values_match,
)
from build_mm_table_dataset import ShardedJsonlWriter
from run_mm_joinability_dynamic_vllm import (
    VllmServerSpec,
    build_builder_command,
    default_vllm_extra_args,
    main as dynamic_vllm_main,
    parse_args as parse_dynamic_vllm_args,
    read_pending_model_task_count,
    start_server,
    wait_for_server,
)


def test_prompt_version_preserves_stage_one_analysis_cache():
    import build_mm_joinability_dataset as joinability_dataset

    assert joinability_dataset.PROMPT_VERSION == (
        "entity_attribute_extraction_v5_batched_leave_one_out"
    )


def test_model_analysis_progress_registers_new_keys_once() -> None:
    progress = joinability_dataset.ModelAnalysisProgress(
        total=0, cached_keys=set(), enabled=False
    )

    assert progress.register({"cached", "model"}) == 2
    assert progress.register({"model", "later"}) == 1
    progress.mark("cached", "cached")
    progress.mark("model", "model")

    assert progress.total == 3
    assert progress.cached == 1
    assert progress.model == 1
    assert progress.planned_keys == {"later"}
    assert progress.completed_keys == {"cached", "model"}
    assert progress.register({"cached", "model"}) == 0
    assert progress.total == 3

    preallocated = joinability_dataset.ModelAnalysisProgress(
        total=3, cached_keys={"cached"}, enabled=False
    )
    assert preallocated.register({"first", "second"}) == 2
    assert preallocated.total == 3
    assert preallocated.planned_keys == {"first", "second"}
    assert preallocated.completed_keys == {"cached"}


def test_model_analysis_progress_refreshes_dynamic_tqdm_total(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ProgressBar:
        def __init__(self, total: int) -> None:
            self.total = total
            self.refreshes = 0
            self.closes = 0

        def set_postfix(self, *, refresh: bool = True, **_postfix: int) -> None:
            if refresh:
                self.refresh()

        def refresh(self) -> None:
            self.refreshes += 1

        def update(self, _amount: int = 1) -> None:
            pass

        def close(self) -> None:
            self.closes += 1

    bars: list[ProgressBar] = []

    def fake_tqdm(*, total: int, **_kwargs: object) -> ProgressBar:
        bar = ProgressBar(total)
        bars.append(bar)
        return bar

    monkeypatch.setattr(joinability_dataset, "tqdm", fake_tqdm)
    progress = joinability_dataset.ModelAnalysisProgress(
        total=0, cached_keys=set(), enabled=True
    )

    refreshes_before_register = bars[0].refreshes
    progress.register({"first", "second"})
    progress.close()
    progress.close()

    assert bars[0].total == 2
    assert bars[0].refreshes - refreshes_before_register == 1
    assert bars[0].closes == 1


def test_select_best_qualified_column_uses_highest_recovery_ratio():
    qualified = [
        {"column_index": 1, "recovered_value_ratio": 0.75},
        {"column_index": 2, "recovered_value_ratio": 1.0},
        {"column_index": 3, "recovered_value_ratio": 0.8},
    ]

    assert select_best_qualified_column(qualified) == [qualified[1]]


def build_multi_attribute_join_records(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    context_column_names: list[str],
    row_count: int = 5,
    split: str = "train",
):
    column_names = ["Entity", "Bridge B", "Bridge C", *context_column_names]
    rows = []
    expected_values: dict[int, dict[str, str]] = {}
    assets = {}
    entity_to_assets = {}
    wiki_to_entity_id = {}
    for row_index in range(row_count):
        wiki_title = f"Entity {row_index}"
        entity_id = f"entity-{row_index}"
        asset_id = f"asset-{row_index}"
        values = {
            column_name: f"{column_name} value {row_index}"
            for column_name in column_names[1:]
        }
        expected_values[row_index] = values
        rows.append(
            {
                "row_id": row_index,
                "cells": [
                    {
                        "column_index": 0,
                        "column_name": "Entity",
                        "text": wiki_title,
                        "wiki_title": wiki_title,
                    },
                    *[
                        {
                            "column_index": column_index,
                            "column_name": column_name,
                            "text": values[column_name],
                            "wiki_title": None,
                        }
                        for column_index, column_name in enumerate(
                            column_names[1:], start=1
                        )
                    ],
                ],
            }
        )
        assets[asset_id] = {
            "asset_id": asset_id,
            "asset_type": "text",
            "entity_id": entity_id,
            "content": f"{wiki_title} bridge evidence",
        }
        entity_to_assets[entity_id] = [asset_id]
        wiki_to_entity_id[wiki_title] = entity_id

    source_table = {
        "source_table_id": "multi-attribute-source",
        "page_title": "Multi attribute source",
        "columns": [
            {"column_index": index, "column_name": column_name}
            for index, column_name in enumerate(column_names)
        ],
        "rows": rows,
        "metadata": {"candidate_entity_columns": [0]},
    }
    args = joinability_dataset.parse_args(
        [
            "--input_dir",
            str(tmp_path),
            "--output_dir",
            str(tmp_path / "output"),
            "--query_rows_per_table",
            "5",
            "--min_rows_per_output_table",
            "5",
            "--min_recovered_value_ratio",
            "0.5",
            "--max_query_tables_per_source_table",
            "0",
        ]
    )

    def fake_resolve_extraction_tasks(**kwargs: object):
        tasks = kwargs["tasks"]
        return [
            (
                task,
                {
                    "cache_key": task.cache_key,
                    "attributes": [
                        {
                            "name": column_name,
                            "value": expected_values[task.source_row_id][column_name],
                            "evidence": "synthetic",
                            "connection_evidence": "synthetic",
                        }
                        for column_name in ("Bridge B", "Bridge C")
                    ],
                    "raw_response": "",
                    "error": "",
                },
            )
            for task in tasks
        ]

    monkeypatch.setattr(
        joinability_dataset,
        "resolve_extraction_tasks",
        fake_resolve_extraction_tasks,
    )
    extraction_writer = joinability_dataset.ListRecordWriter()
    recovery_writer = joinability_dataset.ListRecordWriter()
    query_tables, target_tables, qrels, decision = (
        joinability_dataset.build_table_join_records(
            source_table=source_table,
            split=split,
            assets=assets,
            entity_to_assets=entity_to_assets,
            wiki_to_entity_id=wiki_to_entity_id,
            extractor=None,
            cache=ExtractionCache(tmp_path / "model-cache.jsonl"),
            progress=None,
            concurrency_state=ModelConcurrencyState(
                text_workers=1, image_workers=1
            ),
            extraction_writer=extraction_writer,
            recovery_writer=recovery_writer,
            args=args,
        )
    )
    return query_tables, target_tables, qrels, decision, recovery_writer.records


def test_multi_attribute_query_merges_distinct_positive_targets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    query_tables, target_tables, qrels, decision, recovery_records = (
        build_multi_attribute_join_records(
            tmp_path,
            monkeypatch,
            context_column_names=["Query X", "Query Y", "Target Z"],
        )
    )

    assert decision["reason"] == "queryable"
    assert len(query_tables) == 1
    assert len(target_tables) == 2
    assert len(qrels) == 2
    assert {qrel["query_table_id"] for qrel in qrels} == {
        query_tables[0]["table_id"]
    }
    assert set(query_tables[0]["target_table_ids"]) == {
        target["table_id"] for target in target_tables
    }
    assert {target["join_col_name"] for target in target_tables} == {
        "Bridge B",
        "Bridge C",
    }
    assert all(
        not ({1, 2} & set(query["source_column_indices"]))
        for query in query_tables
    )
    assert all(
        not (
            set(query["source_column_indices"])
            & set(target["source_column_indices"])
        )
        for query in query_tables
        for target in target_tables
    )
    assert {record["target_table_id"] for record in recovery_records} == {
        target["table_id"] for target in target_tables
    }


def build_query_auto_check_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    recovered_rows: set[int],
    supported_rows: set[int],
    apply_query_auto_check: bool = True,
    finalize_query_recoveries: bool = False,
):
    row_count = 6
    source_table = {
        "source_table_id": "query-auto-check-source",
        "columns": [
            {"column_index": 0, "column_name": "Entity"},
            {"column_index": 1, "column_name": "Bridge"},
            {"column_index": 2, "column_name": "Context A"},
            {"column_index": 3, "column_name": "Context B"},
        ],
        "rows": [
            {
                "row_id": index,
                "cells": [
                    {
                        "column_index": 0,
                        "column_name": "Entity",
                        "text": f"Entity {index}",
                        "wiki_title": f"Entity {index}",
                    },
                    {
                        "column_index": 1,
                        "column_name": "Bridge",
                        "text": f"Bridge {index}",
                    },
                    {
                        "column_index": 2,
                        "column_name": "Context A",
                        "text": f"Context A {index}",
                    },
                    {
                        "column_index": 3,
                        "column_name": "Context B",
                        "text": f"Context B {index}",
                    },
                ],
            }
            for index in range(row_count)
        ],
        "metadata": {"candidate_entity_columns": [0]},
    }
    assets = {
        f"asset-{index}": {
            "asset_id": f"asset-{index}",
            "asset_type": "text",
            "content": f"Entity {index} has Bridge {index}.",
        }
        for index in range(row_count)
    }
    entity_to_assets = {
        f"entity-{index}": [f"asset-{index}"]
        for index in range(row_count)
    }
    wiki_to_entity_id = {
        f"Entity {index}": f"entity-{index}"
        for index in range(row_count)
    }

    def fake_resolve_extraction_tasks(**kwargs):
        return [
            (
                task,
                {
                    "cache_key": task.cache_key,
                    "attributes": (
                        [
                            {
                                "name": "Bridge",
                                "value": f"Bridge {task.source_row_id}",
                            }
                        ]
                        if task.source_row_id in recovered_rows
                        else []
                    ),
                    "error": "",
                },
            )
            for task in kwargs["tasks"]
        ]

    monkeypatch.setattr(
        joinability_dataset,
        "resolve_extraction_tasks",
        fake_resolve_extraction_tasks,
    )

    class Extractor:
        auto_check_enabled = True
        auto_check_luna_reviewer = None
        auto_check_terra_reviewer = None

        def __init__(self):
            self.calls: list[int] = []
            self.model_auto_check_stats = ModelAutoCheckStats()

        def review_auto_check_attribute(self, *, task, claimed_value, **_kwargs):
            visible_names = {
                item["name"]
                for item in task.entity.get("row_attributes") or []
            }
            assert "Entity" in visible_names
            assert "Bridge" not in visible_names
            assert "entity_url" in visible_names
            context_names = {
                name for name in visible_names if name.startswith("Context ")
            }
            assert len(context_names) == 1
            assert len(visible_names) == 3
            self.calls.append(task.source_row_id)
            supported = task.source_row_id in supported_rows
            return {
                "extracted_value": claimed_value if supported else "",
                "verdict": "supported" if supported else "insufficient",
                "comparison": (
                    "normalized_values_match" if supported else "empty_extraction"
                ),
                "decision_source": "primary_local",
                "review_complete": True,
                "error_code": "",
            }

    args = joinability_dataset.parse_args(
        [
            "--input_dir",
            str(tmp_path),
            "--output_dir",
            str(tmp_path / "output"),
            "--query_rows_per_table",
            "5",
            "--min_rows_per_output_table",
            "5",
            "--min_recovered_value_ratio",
            "0.6",
            "--explicit_join_fallback_ratio",
            "0",
        ]
    )
    extractor = Extractor()
    recovery_writer = joinability_dataset.ListRecordWriter()
    records = joinability_dataset.build_table_join_records(
        source_table=source_table,
        split="test",
        assets=assets,
        entity_to_assets=entity_to_assets,
        wiki_to_entity_id=wiki_to_entity_id,
        extractor=extractor,
        cache=ExtractionCache(tmp_path / "model-cache.jsonl"),
        progress=None,
        concurrency_state=ModelConcurrencyState(text_workers=1, image_workers=1),
        extraction_writer=joinability_dataset.ListRecordWriter(),
        recovery_writer=recovery_writer,
        args=args,
        query_auto_check_cache=ExtractionCache(
            tmp_path / "query-auto-check-cache.jsonl"
        ),
        apply_query_auto_check=apply_query_auto_check,
        finalize_query_recoveries=finalize_query_recoveries,
    )
    return records, recovery_writer.records, extractor.calls


def test_auto_check_runs_only_for_recoveries_in_selected_query_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (queries, _targets, _qrels, decision), recoveries, calls = (
        build_query_auto_check_fixture(
            tmp_path,
            monkeypatch,
            recovered_rows=set(range(6)),
            supported_rows={0, 1, 2},
        )
    )

    assert decision["reason"] == "queryable"
    assert len(queries) == 1
    assert sorted(calls) == [0, 1, 2]
    assert {record["source_row_id"] for record in recoveries} == {0, 1, 2}
    assert all("auto_check" in record for record in recoveries)


def test_auto_check_skips_recoveries_when_no_query_can_be_formed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (queries, _targets, _qrels, decision), recoveries, calls = (
        build_query_auto_check_fixture(
            tmp_path,
            monkeypatch,
            recovered_rows={0, 1},
            supported_rows={0, 1},
        )
    )

    assert queries == []
    assert decision["reason"] == "no_column_met_recovered_value_ratio"
    assert recoveries == []
    assert calls == []


def test_final_query_materialization_rejects_incomplete_evidence_checks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(
        RuntimeError,
        match="final query evidence auto-check is incomplete",
    ):
        build_query_auto_check_fixture(
            tmp_path,
            monkeypatch,
            recovered_rows=set(range(6)),
            supported_rows=set(range(6)),
            apply_query_auto_check=False,
            finalize_query_recoveries=True,
        )


def test_query_recovery_auto_check_reuses_completed_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class ProgressBar:
        def __init__(self, total: int) -> None:
            self.total = total
            self.updates = 0
            self.closes = 0

        def update(self, amount: int = 1) -> None:
            self.updates += amount

        def close(self) -> None:
            self.closes += 1

    bars: list[ProgressBar] = []

    def fake_tqdm(*, total: int, **kwargs: object) -> ProgressBar:
        assert kwargs["desc"] == "Query recovery eligibility check"
        assert kwargs["unit"] == "recovery"
        assert kwargs["dynamic_ncols"] is True
        assert kwargs["disable"] is False
        bar = ProgressBar(total)
        bars.append(bar)
        return bar

    monkeypatch.setattr(joinability_dataset, "tqdm", fake_tqdm)
    first, _recoveries, first_calls = build_query_auto_check_fixture(
        tmp_path,
        monkeypatch,
        recovered_rows=set(range(6)),
        supported_rows=set(range(6)),
    )
    second, _recoveries, second_calls = build_query_auto_check_fixture(
        tmp_path,
        monkeypatch,
        recovered_rows=set(range(6)),
        supported_rows=set(range(6)),
    )

    assert first[0] and second[0]
    assert sorted(first_calls) == [0, 1, 2]
    assert second_calls == []
    assert len(bars) == 1
    assert bars[0].total == 3
    assert bars[0].updates == 3
    assert bars[0].closes == 1


def _make_query_recovery_candidate(
    *,
    cache_key: str = "extraction-cache-key",
    asset_id: str = "evidence",
) -> joinability_dataset.QueryRecoveryCandidate:
    task = ExtractionTask(
        order=0,
        cache_key=cache_key,
        source_table_id="source",
        source_row_id=0,
        entity_column_index=0,
        entity_column_name="Entity",
        entity={
            "entity_id": "entity",
            "wiki_title": "Entity",
            "cell_text": "Entity",
            "row_attributes": [
                {"name": "Entity", "value": "Entity", "is_entity": True},
            ],
        },
        asset={
            "asset_id": asset_id,
            "asset_type": "text",
            "content": "Evidence",
        },
        candidate_attribute_names=["State"],
    )
    return joinability_dataset.QueryRecoveryCandidate(
        task=task,
        extraction={"attributes": [{"name": "State", "value": "Alabama"}]},
        recovery={
            "source_table_id": "source",
            "source_row_id": 0,
            "recovered_attribute": {
                "column_index": 1,
                "column_name": "State",
                "value": "Alabama",
                "model_value": "Alabama",
                "hidden_in_query": True,
            },
            "evidence": {"asset_id": asset_id, "asset_type": "text"},
        },
    )


def _completed_query_recovery_record(cache_key: str) -> dict[str, object]:
    candidate = _make_query_recovery_candidate()
    return {
        "cache_key": cache_key,
        "extraction_cache_key": "extraction-cache-key",
        "review_policy": (
            joinability_dataset.AUTO_CHECK_REVIEW_POLICY_CASCADE
        ),
        "query_row_attributes": candidate.task.entity["row_attributes"],
        "attribute_name": "State",
        "claimed_value": "Alabama",
        "evidence_identity": (
            joinability_dataset.query_recovery_remote_evidence_identity(
                candidate
            )
        ),
        "schema_version": joinability_dataset.MODEL_AUTO_CHECK_SCHEMA_VERSION,
        "supported": True,
        "auto_check": {
            "schema_version": joinability_dataset.MODEL_AUTO_CHECK_SCHEMA_VERSION,
            "review_policy": (
                joinability_dataset.AUTO_CHECK_REVIEW_POLICY_CASCADE
            ),
            "reviewed_attributes": 1,
            "reviews": [
                {
                    "review_complete": True,
                    "error_code": "",
                    "verdict": "supported",
                    "final_judge_model": "grok-4.5",
                }
            ],
        },
    }


def test_repaired_extraction_reuses_unchanged_auto_check_key(
    tmp_path: Path,
) -> None:
    repaired, changed = reparse_extraction_record(
        {
            "cache_key": "extraction-cache-key",
            "attributes": [],
            "raw_response": (
                '{"attributes":[{"name":"State","value":"Alabama"}]'
            ),
            "error": "",
        },
        ["State"],
    )
    candidate = _make_query_recovery_candidate(
        cache_key=str(repaired["cache_key"])
    )
    extractor = SimpleNamespace(auto_check_luna_reviewer=object())
    key = joinability_dataset.query_recovery_auto_check_key(
        candidate,
        extractor,
    )
    cache = ExtractionCache(tmp_path / "query-auto-check-cache.jsonl")
    cached_record = _completed_query_recovery_record(key)
    cache.put(key, cached_record)

    assert changed is True
    assert joinability_dataset.query_recovery_cached_check(
        key,
        cache,
        candidate,
        extractor=extractor,
    ) == cached_record


def test_query_recovery_auto_check_key_ignores_reviewer_pool_identity() -> None:
    candidate = _make_query_recovery_candidate()
    first_pool = SimpleNamespace(
        auto_check_luna_reviewer=SimpleNamespace(identity={"model": "gpt-5.6-luna"}),
        auto_check_terra_reviewer=SimpleNamespace(identity={"model": "grok-4.5"}),
    )
    changed_pool = SimpleNamespace(
        auto_check_luna_reviewer=SimpleNamespace(identity={"model": "other-luna"}),
        auto_check_terra_reviewer=SimpleNamespace(
            identity={"model": "claude-sonnet-5"}
        ),
    )

    assert joinability_dataset.query_recovery_auto_check_key(
        candidate, first_pool
    ) == joinability_dataset.query_recovery_auto_check_key(candidate, changed_pool)


def test_query_recovery_cache_key_separates_local_and_cascade_policies() -> None:
    candidate = _make_query_recovery_candidate()
    local_only = SimpleNamespace(auto_check_luna_reviewer=None)
    cascade = SimpleNamespace(auto_check_luna_reviewer=object())

    assert joinability_dataset.query_recovery_auto_check_key(
        candidate, local_only
    ) != joinability_dataset.query_recovery_auto_check_key(candidate, cascade)


def test_remote_review_cache_survives_local_model_change(
    tmp_path: Path,
) -> None:
    class Extractor:
        auto_check_enabled = True
        auto_check_luna_reviewer = object()
        auto_check_terra_reviewer = object()
        auto_check_parallelism = 1
        model_auto_check_stats = ModelAutoCheckStats()

        def review_auto_check_attribute(self, **_kwargs):
            raise AssertionError("remote-reviewed evidence must come from cache")

    old_candidate = _make_query_recovery_candidate(cache_key="old-local-model")
    new_candidate = _make_query_recovery_candidate(cache_key="new-local-model")
    old_key = joinability_dataset.query_recovery_auto_check_key(
        old_candidate, None
    )
    new_key = joinability_dataset.query_recovery_auto_check_key(
        new_candidate, Extractor()
    )
    assert old_key != new_key
    assert joinability_dataset.query_recovery_remote_evidence_key(
        old_candidate
    ) == joinability_dataset.query_recovery_remote_evidence_key(new_candidate)

    legacy_record = _completed_query_recovery_record(old_key)
    legacy_record["extraction_cache_key"] = old_candidate.task.cache_key
    cache_path = tmp_path / "query-auto-check-cache.jsonl"
    cache_path.write_text(json.dumps(legacy_record) + "\n", encoding="utf-8")
    cache = ExtractionCache(
        cache_path,
        record_key_alias=joinability_dataset.query_recovery_auto_check_record_key,
    )

    results = joinability_dataset.resolve_query_recovery_auto_checks(
        candidates=[new_candidate],
        extractor=Extractor(),
        cache=cache,
        args=argparse.Namespace(model_progress=False),
        required_recovered_rows=1,
        source_row_order=[0],
    )

    assert results == {new_key: legacy_record}


def test_remote_evidence_cache_reuses_previous_policy_model_stages(
    tmp_path: Path,
) -> None:
    candidate = _make_query_recovery_candidate()
    identity = joinability_dataset.query_recovery_remote_evidence_identity(candidate)
    identity_without_policy = dict(identity)
    identity_without_policy.pop("review_policy")
    legacy_key = joinability_dataset._query_recovery_evidence_identity_key(
        identity_without_policy
    )
    legacy_record = _completed_query_recovery_record(legacy_key)
    legacy_record.pop("review_policy")
    legacy_record["evidence_identity"] = identity_without_policy
    legacy_record["supported"] = True
    legacy_record["auto_check"].pop("review_policy")
    legacy_record["auto_check"]["reviews"] = [
        {
            "attribute_name": "State",
            "claimed_value": "Alabama",
            "extracted_value": "Georgia",
            "verdict": "supported",
            "comparison": "stale_policy_decision",
            "decision_source": "local_luna_consensus",
            "review_complete": True,
            "error_code": "",
            "primary_extracted_value": "Georgia",
            "primary_verdict": "contradicted",
            "primary_comparison": "extracted_value_mismatch",
            "primary_error_code": "",
            "luna_triggered": True,
            "luna_extracted_value": "Georgia",
            "luna_verdict": "contradicted",
            "luna_comparison": "extracted_value_mismatch",
            "luna_agrees_with_local": True,
            "luna_error_code": "",
            "terra_triggered": False,
            "terra_extracted_value": None,
            "terra_verdict": None,
            "terra_comparison": None,
            "terra_error_code": "",
            "secondary_triggered": True,
            "secondary_extracted_value": "Georgia",
            "secondary_verdict": "contradicted",
        }
    ]
    cache_path = tmp_path / "query-auto-check-cache.jsonl"
    cache_path.write_text(json.dumps(legacy_record) + "\n", encoding="utf-8")
    cache = ExtractionCache(
        cache_path,
        record_key_alias=joinability_dataset.query_recovery_auto_check_record_key,
    )
    current_key = joinability_dataset.query_recovery_auto_check_key(
        candidate,
        SimpleNamespace(auto_check_luna_reviewer=object()),
    )
    extractor = LocalAttributeExtractor(_extractor_args())

    class Reviewer:
        def __init__(self) -> None:
            self.calls = 0

        def extract_batches(self, _batches):
            self.calls += 1
            raise AssertionError("completed model stages must be reused")

    luna = Reviewer()
    terra = Reviewer()
    extractor.auto_check_luna_reviewer = luna
    extractor.auto_check_terra_reviewer = terra

    assert joinability_dataset.query_recovery_cached_check(
        current_key,
        cache,
        candidate,
        extractor=extractor,
    ) is None
    results = joinability_dataset.resolve_query_recovery_auto_checks(
        candidates=[candidate],
        extractor=extractor,
        cache=cache,
        args=argparse.Namespace(model_progress=False),
        required_recovered_rows=1,
        source_row_order=[0],
    )

    assert luna.calls == 0
    assert terra.calls == 0
    migrated = results[current_key]
    assert migrated["review_policy"] == (
        joinability_dataset.AUTO_CHECK_REVIEW_POLICY_CASCADE
    )
    assert migrated["supported"] is False
    review = migrated["auto_check"]["reviews"][0]
    assert review["verdict"] == "contradicted"
    assert review["comparison"] == "extracted_value_mismatch"
    assert review["decision_source"] == "local_luna_consensus"


def test_local_only_review_cache_remains_local_model_specific(
    tmp_path: Path,
) -> None:
    old_candidate = _make_query_recovery_candidate(cache_key="old-local-model")
    new_candidate = _make_query_recovery_candidate(cache_key="new-local-model")
    old_key = joinability_dataset.query_recovery_auto_check_key(
        old_candidate, None
    )
    local_record = _completed_query_recovery_record(old_key)
    local_record["extraction_cache_key"] = old_candidate.task.cache_key
    review = local_record["auto_check"]["reviews"][0]
    review["final_judge_model"] = None
    review["decision_source"] = "primary_local"
    cache_path = tmp_path / "query-auto-check-cache.jsonl"
    cache_path.write_text(json.dumps(local_record) + "\n", encoding="utf-8")
    cache = ExtractionCache(
        cache_path,
        record_key_alias=joinability_dataset.query_recovery_auto_check_record_key,
    )
    new_key = joinability_dataset.query_recovery_auto_check_key(
        new_candidate, None
    )

    assert joinability_dataset.query_recovery_cached_check(
        new_key,
        cache,
        new_candidate,
        extractor=None,
    ) is None


def test_remote_cache_alias_is_available_only_to_cascade_policy(
    tmp_path: Path,
) -> None:
    cache_path = tmp_path / "query-auto-check-cache.jsonl"
    legacy_key = "legacy-reviewer-pool-dependent-key"
    legacy_record = _completed_query_recovery_record(legacy_key)
    cache_path.write_text(json.dumps(legacy_record) + "\n", encoding="utf-8")

    cache = ExtractionCache(
        cache_path,
        record_key_alias=joinability_dataset.query_recovery_auto_check_record_key,
    )
    candidate = _make_query_recovery_candidate()
    cascade = SimpleNamespace(auto_check_luna_reviewer=object())
    cascade_key = joinability_dataset.query_recovery_auto_check_key(
        candidate,
        cascade,
    )
    local_key = joinability_dataset.query_recovery_auto_check_key(candidate, None)
    remote_key = joinability_dataset.query_recovery_remote_evidence_key(candidate)

    assert cache.get(legacy_key) == legacy_record
    assert cache.get(remote_key) == legacy_record
    assert joinability_dataset.query_recovery_cached_check(
        cascade_key,
        cache,
        candidate,
        extractor=cascade,
    ) == legacy_record
    assert joinability_dataset.query_recovery_cached_check(
        local_key,
        cache,
        candidate,
        extractor=None,
    ) is None


def test_query_recovery_cache_rejects_incomplete_transient_review(
    tmp_path: Path,
) -> None:
    candidate = _make_query_recovery_candidate()
    extractor = SimpleNamespace(auto_check_luna_reviewer=object())
    key = joinability_dataset.query_recovery_auto_check_key(
        candidate,
        extractor,
    )
    record = _completed_query_recovery_record(key)
    review = record["auto_check"]["reviews"][0]
    review["review_complete"] = False
    review["error_code"] = "model_review_failed:invalid_json"
    cache = ExtractionCache(tmp_path / "query-auto-check-cache.jsonl")
    cache.put_transient(key, record)

    assert joinability_dataset.query_recovery_cached_check(
        key,
        cache,
        candidate,
        extractor=extractor,
    ) is None


def test_finalize_query_recovery_uses_one_cache_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = _make_query_recovery_candidate()
    extractor = SimpleNamespace(
        auto_check_enabled=True,
        auto_check_luna_reviewer=None,
    )
    key = joinability_dataset.query_recovery_auto_check_key(
        candidate,
        extractor,
    )
    record = _completed_query_recovery_record(key)
    record["review_policy"] = (
        joinability_dataset.AUTO_CHECK_REVIEW_POLICY_LOCAL
    )
    record["auto_check"]["review_policy"] = (
        joinability_dataset.AUTO_CHECK_REVIEW_POLICY_LOCAL
    )
    cache = ExtractionCache(tmp_path / "query-auto-check-cache.jsonl")
    cache.put(key, record)
    plan = joinability_dataset.QueryRecoveryAutoCheckPlan(
        query_key="query",
        required_recovered_rows=1,
        source_row_order=(0,),
        candidates=(candidate,),
    )
    rounds: list[bool] = []
    acceptance_caches: list[object] = []

    def run_round(**kwargs):
        rounds.append(bool(kwargs.get("exhaustive")))
        return {}

    def is_supported(_plan, _extractor, acceptance_cache):
        acceptance_caches.append(acceptance_cache)
        assert acceptance_cache.get(key) == record
        return True

    monkeypatch.setattr(
        joinability_dataset,
        "run_query_recovery_auto_check_round",
        run_round,
    )
    monkeypatch.setattr(
        joinability_dataset,
        "query_recovery_plan_is_supported",
        is_supported,
    )

    accepted = joinability_dataset.finalize_query_recovery_auto_checks(
        plans=[plan],
        extractor=extractor,
        cache=cache,
        args=argparse.Namespace(),
    )

    assert accepted == [plan]
    assert rounds == [False, True]
    assert len(acceptance_caches) == 1
    assert acceptance_caches[0] is not cache


def test_changed_reviewer_pool_reuses_completed_legacy_recovery(
    tmp_path: Path,
) -> None:
    class Extractor:
        auto_check_enabled = True
        auto_check_luna_reviewer = SimpleNamespace(identity={"model": "new-luna"})
        auto_check_terra_reviewer = SimpleNamespace(identity={"model": "gemini-3.0-pro"})
        model_auto_check_stats = ModelAutoCheckStats()

        def review_auto_check_attribute(self, **_kwargs):
            raise AssertionError("a completed recovery must not be judged again")

    cache_path = tmp_path / "query-auto-check-cache.jsonl"
    legacy_record = _completed_query_recovery_record("old-pool-key")
    cache_path.write_text(json.dumps(legacy_record) + "\n", encoding="utf-8")
    cache = ExtractionCache(
        cache_path,
        record_key_alias=joinability_dataset.query_recovery_auto_check_record_key,
    )
    candidate = _make_query_recovery_candidate()
    canonical_key = joinability_dataset.query_recovery_auto_check_key(
        candidate, Extractor()
    )

    results = joinability_dataset.resolve_query_recovery_auto_checks(
        candidates=[candidate],
        extractor=Extractor(),
        cache=cache,
        args=argparse.Namespace(model_progress=False),
        required_recovered_rows=1,
        source_row_order=[0],
    )

    assert results == {canonical_key: legacy_record}


def test_query_recovery_auto_check_short_circuits_evidence_and_rows(
    tmp_path: Path,
) -> None:
    calls: list[tuple[int, str]] = []

    class Extractor:
        auto_check_enabled = True
        auto_check_luna_reviewer = None
        auto_check_terra_reviewer = None
        auto_check_parallelism = 4

        def __init__(self) -> None:
            self.model_auto_check_stats = ModelAutoCheckStats()

        def review_auto_check_attribute(self, *, task, claimed_value, **_kwargs):
            asset_id = str(task.asset["asset_id"])
            calls.append((task.source_row_id, asset_id))
            supported = asset_id in {"row-0-second", "row-1-first"}
            return {
                "extracted_value": claimed_value if supported else "",
                "verdict": "supported" if supported else "insufficient",
                "comparison": (
                    "normalized_values_match" if supported else "empty_extraction"
                ),
                "decision_source": "primary_local",
                "review_complete": True,
                "error_code": "",
            }

    def candidate(row_id: int, asset_id: str):
        task = ExtractionTask(
            order=0,
            cache_key=f"cache-{asset_id}",
            source_table_id="source",
            source_row_id=row_id,
            entity_column_index=0,
            entity_column_name="Entity",
            entity={
                "entity_id": f"entity-{row_id}",
                "wiki_title": f"Entity {row_id}",
                "cell_text": f"Entity {row_id}",
                "row_attributes": [
                    {
                        "name": "Entity",
                        "value": f"Entity {row_id}",
                        "is_entity": True,
                    },
                    {"name": "State", "value": "Alabama", "is_entity": False},
                ],
            },
            asset={
                "asset_id": asset_id,
                "asset_type": "text",
                "content": "Evidence",
            },
            candidate_attribute_names=["State"],
        )
        recovery = {
            "source_table_id": "source",
            "source_row_id": row_id,
            "recovered_attribute": {
                "column_index": 1,
                "column_name": "State",
                "value": "Alabama",
                "model_value": "Alabama",
                "hidden_in_query": True,
            },
            "evidence": {"asset_id": asset_id, "asset_type": "text"},
        }
        return joinability_dataset.QueryRecoveryCandidate(
            task=task,
            extraction={"attributes": [{"name": "State", "value": "Alabama"}]},
            recovery=recovery,
        )

    candidates = [
        candidate(0, "row-0-first"),
        candidate(0, "row-0-second"),
        candidate(0, "row-0-third"),
        candidate(1, "row-1-first"),
        candidate(1, "row-1-second"),
        candidate(2, "row-2-first"),
    ]
    extractor = Extractor()
    results = joinability_dataset.resolve_query_recovery_auto_checks(
        candidates=candidates,
        extractor=extractor,
        cache=ExtractionCache(tmp_path / "query-auto-check-cache.jsonl"),
        args=argparse.Namespace(model_progress=False),
        required_recovered_rows=2,
        source_row_order=[0, 1, 2],
    )

    assert calls == [
        (0, "row-0-first"),
        (0, "row-0-second"),
        (1, "row-1-first"),
    ]
    assert len(results) == 3
    assert sum(bool(result.get("supported")) for result in results.values()) == 2

    exhaustive_results = (
        joinability_dataset.resolve_query_recovery_auto_checks(
            candidates=candidates,
            extractor=extractor,
            cache=ExtractionCache(
                tmp_path / "query-auto-check-cache.jsonl"
            ),
            args=argparse.Namespace(model_progress=False),
            required_recovered_rows=2,
            source_row_order=[0, 1, 2],
            exhaustive=True,
        )
    )

    assert calls == [
        (0, "row-0-first"),
        (0, "row-0-second"),
        (1, "row-1-first"),
        (0, "row-0-third"),
        (1, "row-1-second"),
        (2, "row-2-first"),
    ]
    assert len(exhaustive_results) == 6


def test_query_recovery_auto_check_prefers_cached_supported_evidence(
    tmp_path: Path,
) -> None:
    extractor_calls: list[str] = []

    class Extractor:
        auto_check_enabled = True
        auto_check_luna_reviewer = None
        auto_check_terra_reviewer = None
        model_auto_check_stats = ModelAutoCheckStats()

        def review_auto_check_attribute(self, *, task, **_kwargs):
            extractor_calls.append(str(task.asset["asset_id"]))
            raise AssertionError("uncached earlier evidence must not be checked")

    def candidate(asset_id: str):
        task = ExtractionTask(
            order=0,
            cache_key=f"cache-{asset_id}",
            source_table_id="source",
            source_row_id=0,
            entity_column_index=0,
            entity_column_name="Entity",
            entity={
                "entity_id": "entity",
                "wiki_title": "Entity",
                "cell_text": "Entity",
                "row_attributes": [
                    {"name": "Entity", "value": "Entity", "is_entity": True},
                    {"name": "State", "value": "Alabama", "is_entity": False},
                ],
            },
            asset={"asset_id": asset_id, "asset_type": "text", "content": ""},
            candidate_attribute_names=["State"],
        )
        return joinability_dataset.QueryRecoveryCandidate(
            task=task,
            extraction={"attributes": [{"name": "State", "value": "Alabama"}]},
            recovery={
                "source_table_id": "source",
                "source_row_id": 0,
                "recovered_attribute": {
                    "column_index": 1,
                    "column_name": "State",
                    "value": "Alabama",
                    "model_value": "Alabama",
                    "hidden_in_query": True,
                },
                "evidence": {"asset_id": asset_id, "asset_type": "text"},
            },
        )

    extractor = Extractor()
    first = candidate("first")
    cached = candidate("cached")
    cache = ExtractionCache(tmp_path / "query-auto-check-cache.jsonl")
    cached_key = joinability_dataset.query_recovery_auto_check_key(
        cached, extractor
    )
    cache.put(
        cached_key,
        {
            "cache_key": cached_key,
            "supported": True,
            "auto_check": {
                "schema_version": joinability_dataset.MODEL_AUTO_CHECK_SCHEMA_VERSION,
                "reviewed_attributes": 1,
                "reviews": [
                    {
                        "review_complete": True,
                        "error_code": "",
                        "verdict": "supported",
                    }
                ],
            },
        },
    )

    results = joinability_dataset.resolve_query_recovery_auto_checks(
        candidates=[first, cached],
        extractor=extractor,
        cache=cache,
        args=argparse.Namespace(model_progress=False),
        required_recovered_rows=1,
        source_row_order=[0],
    )

    assert extractor_calls == []
    assert results == {cached_key: cache.get(cached_key)}


def test_query_recovery_external_wait_does_not_block_local_checks(
    tmp_path: Path,
) -> None:
    external_started = threading.Event()
    release_external = threading.Event()
    second_local_finished = threading.Event()

    class Extractor:
        auto_check_enabled = True
        auto_check_parallelism = 2
        auto_check_luna_reviewer = object()
        auto_check_terra_reviewer = None
        model_auto_check_stats = ModelAutoCheckStats()

        def review_auto_check_attribute(
            self,
            *,
            task,
            claimed_value,
            defer_remote=False,
            **_kwargs,
        ):
            asset_id = str(task.asset["asset_id"])
            if asset_id == "second-local":
                second_local_finished.set()
                return {
                    "extracted_value": claimed_value,
                    "verdict": "supported",
                    "comparison": "normalized_values_match",
                    "decision_source": "primary_local",
                    "review_complete": True,
                    "error_code": "",
                }
            assert defer_remote is True
            return {
                "extracted_value": "",
                "verdict": "insufficient",
                "comparison": "empty_extraction",
                "decision_source": "remote_review_pending",
                "review_complete": False,
                "error_code": "",
                "primary_extracted_value": "",
                "primary_verdict": "insufficient",
                "primary_comparison": "empty_extraction",
            }

        def complete_auto_check_attribute_review(
            self,
            *,
            claimed_value,
            **_kwargs,
        ):
            external_started.set()
            assert release_external.wait(timeout=2.0)
            return {
                "extracted_value": claimed_value,
                "verdict": "supported",
                "comparison": "normalized_values_match",
                "decision_source": "luna_recovery",
                "review_complete": True,
                "error_code": "",
            }

    first = _make_query_recovery_candidate(
        cache_key="first-cache",
        asset_id="external-wait",
    )
    second = _make_query_recovery_candidate(
        cache_key="second-cache",
        asset_id="second-local",
    )
    plans = [
        joinability_dataset.QueryRecoveryAutoCheckPlan(
            query_key="first",
            required_recovered_rows=1,
            source_row_order=(0,),
            candidates=(first,),
        ),
        joinability_dataset.QueryRecoveryAutoCheckPlan(
            query_key="second",
            required_recovered_rows=1,
            source_row_order=(0,),
            candidates=(second,),
        ),
    ]
    errors: list[BaseException] = []

    def resolve() -> None:
        try:
            joinability_dataset.resolve_query_recovery_auto_check_plans(
                plans=plans,
                extractor=Extractor(),
                cache=ExtractionCache(tmp_path / "query-auto-check-cache.jsonl"),
                args=argparse.Namespace(model_progress=False),
                concurrency_state=ModelConcurrencyState(
                    text_workers=1,
                    image_workers=1,
                ),
            )
        except BaseException as error:  # pragma: no cover - asserted below.
            errors.append(error)

    resolver = threading.Thread(target=resolve)
    resolver.start()
    assert external_started.wait(timeout=2.0)
    assert second_local_finished.wait(timeout=1.0)
    release_external.set()
    resolver.join(timeout=2.0)

    assert not resolver.is_alive()
    assert errors == []


def test_query_recovery_image_checks_use_local_and_remote_pools() -> None:
    first_wave = threading.Barrier(2, timeout=2.0)
    calls: list[str] = []
    calls_lock = threading.Lock()

    class Extractor:
        auto_check_enabled = True
        auto_check_luna_reviewer = None
        auto_check_terra_reviewer = None
        model_auto_check_stats = ModelAutoCheckStats()

        def review_auto_check_attribute(
            self,
            *,
            claimed_value,
            endpoint_pool,
            **_kwargs,
        ):
            with calls_lock:
                calls.append(endpoint_pool)
            first_wave.wait()
            return {
                "extracted_value": claimed_value,
                "verdict": "supported",
                "comparison": "normalized_values_match",
                "decision_source": "primary_local",
                "review_complete": True,
                "error_code": "",
            }

    def image_candidate(cache_key: str, asset_id: str):
        text_candidate = _make_query_recovery_candidate(
            cache_key=cache_key,
            asset_id=asset_id,
        )
        text_candidate.task.asset["asset_type"] = "image"
        text_candidate.recovery["evidence"]["asset_type"] = "image"
        return text_candidate

    scheduler = joinability_dataset.QueryRecoveryLocalCheckScheduler(
        extractor=Extractor(),
        state=ModelConcurrencyState(
            text_workers=1,
            image_workers=1,
            remote_image_workers=1,
        ),
    )
    try:
        futures = [
            scheduler.submit(image_candidate("image-1", "image-local")),
            scheduler.submit(image_candidate("image-2", "image-remote")),
        ]
        assert all(future.result(timeout=2.0)["supported"] for future in futures)
    finally:
        scheduler.close()

    assert sorted(calls) == ["local", "remote"]


def test_identical_visible_multi_attribute_query_keeps_all_positive_targets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    query_tables, target_tables, qrels, decision, _recovery_records = (
        build_multi_attribute_join_records(
            tmp_path,
            monkeypatch,
            context_column_names=["Query X", "Target Z"],
        )
    )

    assert decision["reason"] == "queryable"
    assert len(query_tables) == 1
    assert len(target_tables) == 2
    assert len(qrels) == 2
    assert {qrel["query_table_id"] for qrel in qrels} == {
        query_tables[0]["table_id"]
    }
    assert {
        item["column_name"] for item in query_tables[0]["hidden_attributes"]
    } == {"Bridge B", "Bridge C"}
    assert set(query_tables[0]["target_table_ids"]) == {
        target["table_id"] for target in target_tables
    }
    assert {target["join_col_name"] for target in target_tables} == {
        "Bridge B",
        "Bridge C",
    }


def test_implicit_query_uniqueness_validator_allows_multiple_targets() -> None:
    qrels = [
        {
            "query_table_id": "query_same",
            "target_table_id": "target_a",
            "join_attribute": {"source_column_index": 1},
            "reason": "model_recoverable_join_column",
        },
        {
            "query_table_id": "query_same",
            "target_table_id": "target_b",
            "join_attribute": {"source_column_index": 2},
            "reason": "model_recoverable_join_column",
        },
    ]

    assert joinability_dataset.validate_implicit_query_uniqueness(qrels) == 1

    with pytest.raises(ValueError, match="duplicate implicit query qrel"):
        joinability_dataset.validate_implicit_query_uniqueness([*qrels, qrels[0]])


def test_implicit_query_uniqueness_validator_requires_one_qrel_per_query() -> None:
    qrels = [
        {
            "query_table_id": "query_one",
            "target_table_id": "target_one",
            "join_attribute": {"source_column_index": 1},
            "reason": "model_recoverable_join_column",
        }
    ]

    with pytest.raises(ValueError, match="count mismatch"):
        joinability_dataset.validate_implicit_query_uniqueness(
            qrels,
            expected_query_count=2,
        )


def test_multi_attribute_split_demotes_weakest_join_when_context_is_narrow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    query_tables, target_tables, qrels, decision, _recovery_records = (
        build_multi_attribute_join_records(
            tmp_path,
            monkeypatch,
            context_column_names=["Only Context"],
        )
    )

    assert decision["reason"] == "queryable"
    assert len(query_tables) == 1
    assert len(target_tables) == len(qrels) == 1
    query = query_tables[0]
    target = target_tables[0]
    assert target["join_col_name"] == "Bridge B"
    assert set(query["source_column_indices"]) | set(
        target["source_column_indices"]
    ) == {0, 1, 2, 3}
    assert set(query["source_column_indices"]) & set(
        target["source_column_indices"]
    ) == set()
    assert len(query["source_column_indices"]) == 2
    assert len(target["source_column_indices"]) == 2


def test_multi_attribute_context_layout_demotes_join_columns_to_reach_floor() -> None:
    source_table = {
        "source_table_id": "context-floor-source",
        "columns": [
            {"column_index": index, "column_name": name}
            for index, name in enumerate(
                ["Entity", "Strong Bridge", "Middle Bridge", "Weak Bridge"]
            )
        ],
        "rows": [],
    }
    qualified = [
        {"column_index": 1, "recovered_value_ratio": 1.0},
        {"column_index": 2, "recovered_value_ratio": 0.8},
        {"column_index": 3, "recovered_value_ratio": 0.6},
    ]

    layouts = joinability_dataset.multi_attribute_context_layout(
        source_table=source_table,
        entity_col=0,
        qualified_cols=qualified,
        args=SimpleNamespace(seed=13, max_query_tables_per_source_table=0),
    )

    assert len(layouts) == 1
    emitted, query_context, target_context = layouts[0]
    assert emitted["column_index"] == 1
    assert set(query_context) | set(target_context) == {2, 3}
    assert set(query_context).isdisjoint(target_context)
    assert len(query_context) == len(target_context) == 1


def test_multi_attribute_context_layout_rejects_unachievable_floor() -> None:
    source_table = {
        "source_table_id": "insufficient-context-source",
        "columns": [
            {"column_index": index, "column_name": name}
            for index, name in enumerate(["Entity", "Bridge A", "Bridge B"])
        ],
        "rows": [],
    }
    qualified = [
        {"column_index": 1, "recovered_value_ratio": 1.0},
        {"column_index": 2, "recovered_value_ratio": 0.8},
    ]

    assert joinability_dataset.multi_attribute_context_layout(
        source_table=source_table,
        entity_col=0,
        qualified_cols=qualified,
        args=SimpleNamespace(seed=13, max_query_tables_per_source_table=0),
    ) == []


def build_rejected_table_with_explicit_join(
    tmp_path: Path,
    *,
    ratio: float,
    mode: str = "ratio",
):
    column_names = ["Entity", "City", "Country", "Score"]
    source_table = {
        "source_table_id": "explicit-fallback-source",
        "page_title": "Explicit fallback source",
        "columns": [
            {"column_index": index, "column_name": column_name}
            for index, column_name in enumerate(column_names)
        ],
        "rows": [
            {
                "row_id": row_index,
                "cells": [
                    {
                        "column_index": 0,
                        "column_name": "Entity",
                        "text": f"Entity {row_index}",
                        "wiki_title": f"Entity {row_index}",
                    },
                    {
                        "column_index": 1,
                        "column_name": "City",
                        "text": f"City {row_index}",
                    },
                    {
                        "column_index": 2,
                        "column_name": "Country",
                        "text": f"Country {row_index}",
                    },
                    {
                        "column_index": 3,
                        "column_name": "Score",
                        "text": str(row_index + 10),
                    },
                ],
            }
            for row_index in range(5)
        ],
        # With no assets, none of these attributes can pass multimodal recovery.
        "metadata": {"candidate_entity_columns": [0]},
    }
    args = joinability_dataset.parse_args(
        [
            "--input_dir",
            str(tmp_path),
            "--output_dir",
            str(tmp_path / "output"),
            "--query_rows_per_table",
            "3",
            "--min_rows_per_output_table",
            "3",
            "--explicit_join_fallback_mode",
            mode,
            "--explicit_join_fallback_ratio",
            str(ratio),
        ]
    )
    return joinability_dataset.build_table_join_records(
        source_table=source_table,
        split="train",
        assets={},
        entity_to_assets={},
        wiki_to_entity_id={},
        extractor=None,
        cache=ExtractionCache(tmp_path / "model-cache.jsonl"),
        progress=None,
        concurrency_state=ModelConcurrencyState(text_workers=1, image_workers=1),
        extraction_writer=joinability_dataset.ListRecordWriter(),
        recovery_writer=joinability_dataset.ListRecordWriter(),
        args=args,
    )


def test_multimodal_rejection_can_become_visible_join_pair(tmp_path: Path) -> None:
    query_tables, target_tables, qrels, decision = (
        build_rejected_table_with_explicit_join(tmp_path, ratio=1.0)
    )

    assert decision["reason"] == "explicit_join_fallback"
    assert decision["rejected_multimodal_reason"] == "no_column_met_recovered_value_ratio"
    assert len(query_tables) == len(target_tables) == len(qrels) == 1
    query = query_tables[0]
    target = target_tables[0]
    join_col = decision["join_column_index"]
    assert join_col != 0
    assert join_col in query["source_column_indices"]
    assert join_col in target["source_column_indices"]
    assert set(query["source_column_indices"]) & set(
        target["source_column_indices"]
    ) == {join_col}
    assert query["hidden_attributes"] == []
    assert query["construction_type"] == "explicit_visible_join"
    assert target["construction_type"] == "explicit_visible_join"
    assert len(query["rows"]) == 3
    assert len(target["rows"]) == 5
    assert qrels[0]["join_attribute"]["hidden_in_query"] is False
    assert qrels[0]["reason"] == "explicit_visible_join_column"


def test_explicit_candidates_from_one_source_are_sibling_disjoint(
    tmp_path: Path,
) -> None:
    # Enumerate every query-level explicit candidate for one rejected source.
    source = {
        "source_table_id": "explicit-fallback-source",
        "page_title": "Explicit fallback source",
        "columns": [
            {"column_index": index, "column_name": name}
            for index, name in enumerate(
                ["Entity", "City", "Country", "Score", "Context A", "Context B"]
            )
        ],
        "rows": [
            {
                "row_id": index,
                "cells": [
                    {
                        "column_index": 0,
                        "column_name": "Entity",
                        "text": f"Entity {index}" if index >= 2 else "",
                        "wiki_title": f"Entity {index}" if index >= 2 else "",
                    },
                    *[
                        {
                            "column_index": column,
                            "column_name": [
                                "Entity",
                                "City",
                                "Country",
                                "Score",
                                "Context A",
                                "Context B",
                            ][column],
                            "text": f"value-{column}-{index}",
                        }
                        for column in (1, 2, 3)
                    ]
                    + [
                        {
                            "column_index": column,
                            "column_name": [
                                "Entity",
                                "City",
                                "Country",
                                "Score",
                                "Context A",
                                "Context B",
                            ][column],
                            "text": f"context-{column}-{index}" if index <= 2 else "",
                        }
                        for column in (4, 5)
                    ],
                ],
            }
            for index in range(5)
        ],
        "metadata": {"candidate_entity_columns": [0]},
    }
    args = joinability_dataset.parse_args(
        [
            "--input_dir",
            str(tmp_path),
            "--output_dir",
            str(tmp_path / "output-candidates"),
            "--query_rows_per_table",
            "3",
            "--min_rows_per_output_table",
            "3",
            "--max_target_context_attrs",
            "1",
            "--explicit_join_fallback_mode",
            "match_implicit",
        ]
    )
    candidates = joinability_dataset.build_explicit_join_fallback_candidates(
        source_table=source,
        split="train",
        entity_col=0,
        rejected_multimodal_reason="no_column_met_recovered_value_ratio",
        args=args,
        force=True,
    )
    assert len(candidates) == 3
    records = [
        joinability_dataset.materialize_balanced_explicit_join_candidate(
            source_table=source,
            split="train",
            candidate_decision=candidate,
            args=args,
        )
        for candidate in candidates
    ]
    for index, (queries, targets, _qrels, _decision) in enumerate(records):
        assert len(queries) == len(targets) == 1
        own_join = candidates[index]["join_column_index"]
        assert set(queries[0]["source_column_indices"]) & set(
            targets[0]["source_column_indices"]
        ) == {own_join}
        for other_index, (_other_queries, other_targets, _q, _d) in enumerate(records):
            if index == other_index:
                continue
            assert not (
                set(queries[0]["source_column_indices"])
                & set(other_targets[0]["source_column_indices"])
            )
    selected, _counts = joinability_dataset.select_balanced_explicit_join_candidates(
        candidate_splits={candidate["candidate_id"]: "train" for candidate in candidates},
        implicit_query_counts={"train": 2, "dev": 0, "test": 0},
        args=argparse.Namespace(seed=13),
    )
    assert len(selected) == 2


def test_visible_join_fallback_ratio_zero_keeps_rejection_raw(tmp_path: Path) -> None:
    query_tables, data_lake_tables, qrels, decision = (
        build_rejected_table_with_explicit_join(tmp_path, ratio=0.0)
    )

    assert query_tables == []
    assert qrels == []
    assert decision["reason"] == "no_column_met_recovered_value_ratio"
    assert data_lake_tables[0]["role"] == "raw_data_lake_table"


def test_match_implicit_defers_viable_explicit_candidate(
    tmp_path: Path,
) -> None:
    query_tables, data_lake_tables, qrels, decision = (
        build_rejected_table_with_explicit_join(
            tmp_path,
            ratio=0.0,
            mode="match_implicit",
        )
    )

    assert query_tables == []
    assert qrels == []
    assert data_lake_tables[0]["role"] == "raw_data_lake_table"
    assert decision["reason"] == "no_column_met_recovered_value_ratio"
    assert decision["explicit_join_candidate"]["reason"] == (
        "explicit_join_fallback"
    )


def test_balanced_explicit_candidate_selection_matches_each_split() -> None:
    args = argparse.Namespace(seed=13)
    candidates = {
        "train-a": "train",
        "train-b": "train",
        "train-c": "train",
        "dev-a": "dev",
        "dev-b": "dev",
        "test-a": "test",
    }

    selected, counts = (
        joinability_dataset.select_balanced_explicit_join_candidates(
            candidate_splits=candidates,
            implicit_query_counts={"train": 2, "dev": 1, "test": 1},
            args=args,
        )
    )

    assert counts == {"train": 3, "dev": 2, "test": 1}
    assert sum(candidates[item] == "train" for item in selected) == 2
    assert sum(candidates[item] == "dev" for item in selected) == 1
    assert sum(candidates[item] == "test" for item in selected) == 1
    repeated, _ = joinability_dataset.select_balanced_explicit_join_candidates(
        candidate_splits=dict(reversed(list(candidates.items()))),
        implicit_query_counts={"train": 2, "dev": 1, "test": 1},
        args=args,
    )
    assert repeated == selected


def test_balanced_explicit_candidate_selection_rejects_shortfall() -> None:
    with pytest.raises(ValueError, match="required=2, available=1"):
        joinability_dataset.select_balanced_explicit_join_candidates(
            candidate_splits={"train-a": "train"},
            implicit_query_counts={"train": 2, "dev": 0, "test": 0},
            args=argparse.Namespace(seed=13),
        )


def test_required_recovered_rows_caps_denominator_at_query_size():
    assert required_recovered_row_count(4, 5, 0.6) == 3
    assert required_recovered_row_count(5, 5, 0.6) == 3
    assert required_recovered_row_count(20, 5, 0.6) == 3
    assert required_recovered_row_count(100, 100, 0.07) == 7


def test_recovery_profile_counts_empty_attribute_rows_as_failures():
    profile = recovery_column_profile(
        valid_source_rows={0, 1, 2, 3, 4, 5},
        recovered_source_rows={0, 1, 2},
        query_rows_per_table=5,
        min_recovery_denominator=2,
        min_ratio=0.6,
    )

    assert profile == {
        "eligible_rows": 6,
        "valid_entity_rows": 6,
        "recovered_rows": 3,
        "required_recovered_rows": 3,
        "recovered_value_ratio": 0.5,
    }


def test_select_query_rows_uses_recoveries_then_failures():
    assert select_query_source_rows(
        source_row_order=[0, 1, 2, 3, 4, 5],
        recovered_source_rows={0, 2, 4},
        query_rows_per_table=5,
        required_recovered_rows=3,
    ) == [0, 2, 4, 1, 3]


def test_select_query_rows_prefers_extra_recoveries_over_failures():
    assert select_query_source_rows(
        source_row_order=[0, 1, 2, 3, 4, 5],
        recovered_source_rows={0, 1, 2, 3, 4},
        query_rows_per_table=5,
        required_recovered_rows=3,
    ) == [0, 1, 2, 3, 4]


def test_select_query_row_views_are_disjoint_and_adapt_to_available_evidence():
    views = select_query_source_row_views(
        source_row_order=list(range(25)),
        recovered_source_rows=set(range(10)),
        query_rows_per_table=5,
        required_recovered_rows=2,
        max_views=5,
    )

    assert len(views) == 5
    assert all(len(view) == 5 for view in views)
    assert len({row for view in views for row in view}) == 25
    assert all(len(set(view) & set(range(10))) >= 2 for view in views)


def test_train_emits_multiple_disjoint_row_views_but_dev_stays_canonical(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    train_queries, train_targets, train_qrels, decision, _recoveries = (
        build_multi_attribute_join_records(
            tmp_path,
            monkeypatch,
            context_column_names=["Query X", "Target Z"],
            row_count=20,
            split="train",
        )
    )
    dev_queries, dev_targets, dev_qrels, _dev_decision, _dev_recoveries = (
        build_multi_attribute_join_records(
            tmp_path,
            monkeypatch,
            context_column_names=["Query X", "Target Z"],
            row_count=20,
            split="dev",
        )
    )

    assert len(train_queries) == 4
    assert len(train_targets) == 2
    assert len(train_qrels) == 8
    assert len({qrel["query_table_id"] for qrel in train_qrels}) == 4
    assert len({qrel["target_table_id"] for qrel in train_qrels}) == 2
    assert {item["row_views"] for item in decision["qualified_columns"]} == {4}
    train_row_sets = [set(query["source_row_indices"]) for query in train_queries]
    assert all(
        left.isdisjoint(right)
        for index, left in enumerate(train_row_sets)
        for right in train_row_sets[index + 1 :]
    )
    assert len(dev_queries) == 1
    assert len(dev_targets) == 2
    assert len(dev_qrels) == 2


@pytest.mark.parametrize(
    ("predicted", "expected", "attribute_name"),
    [
        ("NBC", "NBC", "Network"),
        ("1,000", "1000", "Population"),
        ("Birmingham", "Olympic Stadium in Birmingham", "Venue"),
        ("2001", "August 2001", "Year"),
        ("Stade-5 Juillet", "Stade 5 Juillet", "Venue"),
    ],
)
def test_values_match_accepts_safe_exact_numeric_and_phrase_matches(
    predicted: str, expected: str, attribute_name: str
) -> None:
    assert values_match(predicted, expected, attribute_name=attribute_name)


@pytest.mark.parametrize(
    ("predicted", "expected", "attribute_name"),
    [
        ("a", "Canada", "Country"),
        ("1", "Test 1152", "Test No."),
        ("US", "Russia", "Country"),
        ("7", "6.97", "Mile"),
        ("GT2", "LMGT2", "Class"),
    ],
)
def test_values_match_rejects_unsafe_short_or_partial_numeric_matches(
    predicted: str, expected: str, attribute_name: str
) -> None:
    assert not values_match(predicted, expected, attribute_name=attribute_name)


def test_values_match_accepts_station_type_suffix_only_with_name_context() -> None:
    assert values_match(
        "観音駅",
        "観音",
        attribute_name="Japanese",
        entity_column_name="Station",
    )
    assert values_match(
        "Kannon Station",
        "Kannon",
        attribute_name="English name",
        entity_column_name="Station",
    )
    assert not values_match(
        "観音駅",
        "観音",
        attribute_name="Japanese",
    )
    assert not values_match(
        "観音駅",
        "観音",
        attribute_name="Location",
        entity_column_name="Station",
    )


def test_query_rows_per_table_defaults_to_five(tmp_path):
    args = joinability_dataset.parse_args(
        ["--input_dir", str(tmp_path), "--output_dir", str(tmp_path / "out")]
    )

    assert args.query_rows_per_table == 5
    assert args.max_train_query_row_views_per_join == 5
    assert args.explicit_join_fallback_mode == "ratio"
    assert args.explicit_join_fallback_ratio == 0.2
    assert args.image_model_name == "Qwen3-VL-8B-Instruct"
    assert args.max_scanned_files is None
    assert args.auto_check_secondary_openai is True
    assert args.auto_check_openai_model == "gpt-5.6-luna"
    assert args.auto_check_openai_reasoning_effort == "none"
    assert args.auto_check_terra_model == "gpt-5.6-terra"


def test_auto_check_json_profiles_use_split_connections_and_shared_concurrency(
    tmp_path: Path,
) -> None:
    config_file = tmp_path / "auto-check.json"
    config_file.write_text(
        json.dumps(
            {
                "version": 1,
                "profiles": {
                    "luna_only": {
                        "initial": {
                            "api_key": "fake-luna",
                            "base_url": "https://luna.example.test/v1",
                            "model": "gpt-5.6-luna",
                        },
                        "final_judge": None,
                    },
                    "mixed_api": {
                        "initial": {
                            "api_key": "fake-mixed-initial",
                            "base_url": (
                                "https://mixed-initial.example.test/v1"
                            ),
                            "model": "gpt-5.6-luna",
                        },
                        "final_judge": {
                            "api_key": "fake-mixed-final",
                            "base_url": (
                                "https://mixed-final.example.test/v1"
                            ),
                            "model": "grok-4.5",
                        },
                    },
                    "other_api": {
                        "initial": {
                            "api_key": "fake-other-initial",
                            "base_url": (
                                "https://other-initial.example.test/v1"
                            ),
                            "model": "gpt-5.6-luna",
                        },
                        "final_judge": {
                            "api_key": "fake-other-final",
                            "base_url": (
                                "https://other-final.example.test/v1"
                            ),
                            "model": "gemini-3.0-pro",
                        },
                    },
                    "final_only_api": {
                        "initial": None,
                        "final_judge": {
                            "api_key": "fake-final-only",
                            "base_url": (
                                "https://final-only.example.test/v1"
                            ),
                            "model": "claude-sonnet-5",
                        },
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    config_file.chmod(0o600)
    args = joinability_dataset.parse_args(
        [
            "--input_dir",
            str(tmp_path / "input"),
            "--output_dir",
            str(tmp_path / "output"),
            "--auto_check_api_config_file",
            str(config_file),
        ]
    )

    initial_pool, final_pool = (
        joinability_dataset.prepare_model_auto_check_reviewers(args)
    )

    assert isinstance(
        initial_pool,
        joinability_dataset.StableAutoCheckReviewerPool,
    )
    assert isinstance(final_pool, joinability_dataset.StableAutoCheckReviewerPool)
    assert [name for name, _reviewer in initial_pool.reviewers] == [
        "luna_only",
        "mixed_api",
        "other_api",
    ]
    assert [name for name, _reviewer in final_pool.reviewers] == [
        "mixed_api",
        "other_api",
        "final_only_api",
    ]
    assert [reviewer.identity["model"] for _name, reviewer in final_pool.reviewers] == [
        "grok-4.5",
        "gemini-3.0-pro",
        "claude-sonnet-5",
    ]
    initial_by_name = dict(initial_pool.reviewers)
    final_by_name = dict(final_pool.reviewers)
    assert (
        initial_by_name["mixed_api"].client.request_controller
        is final_by_name["mixed_api"].client.request_controller
    )
    assert initial_by_name["mixed_api"].client.api_key == "fake-mixed-initial"
    assert initial_by_name["mixed_api"].client.api_base_url == (
        "https://mixed-initial.example.test/v1"
    )
    assert final_by_name["mixed_api"].client.api_key == "fake-mixed-final"
    assert final_by_name["mixed_api"].client.api_base_url == (
        "https://mixed-final.example.test/v1"
    )
    assert final_by_name["final_only_api"].client.api_key == (
        "fake-final-only"
    )
    assert final_by_name["final_only_api"].client.api_base_url == (
        "https://final-only.example.test/v1"
    )
    for reviewer in (
        initial_by_name["mixed_api"],
        final_by_name["mixed_api"],
    ):
        identity_json = json.dumps(reviewer.identity, sort_keys=True)
        assert "fake-mixed" not in identity_json
        assert "mixed-initial.example.test" not in identity_json
        assert "mixed-final.example.test" not in identity_json
    assert initial_by_name["mixed_api"].client.request_controller.summary()[
        "current_inflight_limit"
    ] == 5
    assert initial_by_name["mixed_api"].client.max_retries == 0
    assert final_by_name["mixed_api"].client.max_retries == 0


def test_balanced_auto_check_pool_round_robins_equal_providers() -> None:
    class Controller:
        max_inflight = 5

        @staticmethod
        def summary() -> dict[str, int]:
            return {"max_inflight": 5, "current_inflight_limit": 5}

    class Reviewer:
        def __init__(self, name: str) -> None:
            self.name = name
            self.identity = {"model": f"model-{name}"}

        def extract_batches(self, batches):
            return {
                batch["query_table_id"]: [{"extracted_value": self.name}]
                for batch in batches
            }

    names = ["provider-a", "provider-b", "provider-c"]
    load_balancer = joinability_dataset.AutoCheckProviderLoadBalancer(
        {name: Controller() for name in names}
    )
    pool = joinability_dataset.BalancedAutoCheckReviewerPool(
        [(name, Reviewer(name)) for name in names],
        role="final",
        load_balancer=load_balancer,
    )
    selected: list[str] = []

    for index in range(6):
        values = pool.extract_batch(
            {"query_table_id": f"query-{index}"},
            on_selected=lambda identity: selected.append(identity["api_profile"]),
        )
        assert values is not None

    assert selected == names * 2


def test_balanced_auto_check_pool_routes_around_busy_provider() -> None:
    class Controller:
        max_inflight = 1

        @staticmethod
        def summary() -> dict[str, int]:
            return {"max_inflight": 1, "current_inflight_limit": 1}

    slow_started = threading.Event()
    release_slow = threading.Event()

    class Reviewer:
        def __init__(self, name: str, *, slow: bool = False) -> None:
            self.name = name
            self.slow = slow
            self.identity = {"model": f"model-{name}"}

        def extract_batches(self, batches):
            if self.slow:
                slow_started.set()
                assert release_slow.wait(timeout=5)
            return {
                batch["query_table_id"]: [{"extracted_value": self.name}]
                for batch in batches
            }

    load_balancer = joinability_dataset.AutoCheckProviderLoadBalancer(
        {"slow": Controller(), "fast": Controller()}
    )
    pool = joinability_dataset.BalancedAutoCheckReviewerPool(
        [
            ("slow", Reviewer("slow", slow=True)),
            ("fast", Reviewer("fast")),
        ],
        role="initial",
        load_balancer=load_balancer,
    )
    selected: list[str] = []

    with ThreadPoolExecutor(max_workers=2) as executor:
        slow_future = executor.submit(
            pool.extract_batch,
            {"query_table_id": "slow-query"},
            on_selected=lambda identity: selected.append(identity["api_profile"]),
        )
        assert slow_started.wait(timeout=5)
        fast_future = executor.submit(
            pool.extract_batch,
            {"query_table_id": "fast-query"},
            on_selected=lambda identity: selected.append(identity["api_profile"]),
        )
        assert fast_future.result(timeout=5) == [{"extracted_value": "fast"}]
        release_slow.set()
        assert slow_future.result(timeout=5) == [{"extracted_value": "slow"}]

    assert selected == ["slow", "fast"]


def test_balanced_auto_check_pool_fails_over_and_cools_failed_provider() -> None:
    class Controller:
        max_inflight = 1

        @staticmethod
        def summary() -> dict[str, int]:
            return {"max_inflight": 1, "current_inflight_limit": 1}

    class Reviewer:
        def __init__(self, name: str, *, fail: bool = False) -> None:
            self.name = name
            self.fail = fail
            self.calls = 0
            self.identity = {"model": f"model-{name}"}

        def extract_batches(self, batches):
            self.calls += 1
            if self.fail:
                raise joinability_dataset.TransientModelEndpointError(
                    f"{self.name} failed"
                )
            return {
                batch["query_table_id"]: [{"extracted_value": self.name}]
                for batch in batches
            }

    failed = Reviewer("failed", fail=True)
    healthy = Reviewer("healthy")
    load_balancer = joinability_dataset.AutoCheckProviderLoadBalancer(
        {"failed": Controller(), "healthy": Controller()},
        failure_cooldown_seconds=60,
        max_failure_cooldown_seconds=60,
    )
    pool = joinability_dataset.BalancedAutoCheckReviewerPool(
        [("failed", failed), ("healthy", healthy)],
        role="final",
        load_balancer=load_balancer,
    )
    selected: list[str] = []

    first = pool.extract_batch(
        {"query_table_id": "first"},
        on_selected=lambda identity: selected.append(identity["api_profile"]),
    )
    second = pool.extract_batch(
        {"query_table_id": "second"},
        on_selected=lambda identity: selected.append(identity["api_profile"]),
    )

    assert first == [{"extracted_value": "healthy"}]
    assert second == [{"extracted_value": "healthy"}]
    assert selected == ["failed", "healthy", "healthy"]
    assert failed.calls == 1
    assert healthy.calls == 2


def test_balanced_auto_check_pool_shares_provider_capacity_across_roles() -> None:
    class Controller:
        max_inflight = 1

        @staticmethod
        def summary() -> dict[str, int]:
            return {"max_inflight": 1, "current_inflight_limit": 1}

    initial_started = threading.Event()
    release_initial = threading.Event()
    final_started = threading.Event()

    class Reviewer:
        def __init__(self, name: str, *, wait: bool = False) -> None:
            self.name = name
            self.wait = wait
            self.identity = {"model": name}

        def extract_batches(self, batches):
            if self.wait:
                initial_started.set()
                assert release_initial.wait(timeout=5)
            else:
                final_started.set()
            return {
                batch["query_table_id"]: [{"extracted_value": self.name}]
                for batch in batches
            }

    load_balancer = joinability_dataset.AutoCheckProviderLoadBalancer(
        {"shared": Controller()}
    )
    initial_pool = joinability_dataset.BalancedAutoCheckReviewerPool(
        [("shared", Reviewer("initial", wait=True))],
        role="initial",
        load_balancer=load_balancer,
    )
    final_pool = joinability_dataset.BalancedAutoCheckReviewerPool(
        [("shared", Reviewer("final"))],
        role="final",
        load_balancer=load_balancer,
    )

    with ThreadPoolExecutor(max_workers=2) as executor:
        initial_future = executor.submit(
            initial_pool.extract_batch,
            {"query_table_id": "initial-query"},
        )
        assert initial_started.wait(timeout=5)
        final_future = executor.submit(
            final_pool.extract_batch,
            {"query_table_id": "final-query"},
        )
        assert not final_started.wait(timeout=0.1)
        release_initial.set()
        assert initial_future.result(timeout=5) == [
            {"extracted_value": "initial"}
        ]
        assert final_future.result(timeout=5) == [{"extracted_value": "final"}]


def test_auto_check_json_profiles_require_an_initial_reviewer(
    tmp_path: Path,
) -> None:
    config_file = tmp_path / "auto-check.json"
    config_file.write_text(
        json.dumps(
            {
                "version": 1,
                "profiles": {
                    "final_only": {
                        "initial": None,
                        "final_judge": {
                            "api_key": "fake-final-only",
                            "base_url": "https://final-only.example.test/v1",
                            "model": "grok-4.5",
                        },
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    config_file.chmod(0o600)
    args = joinability_dataset.parse_args(
        [
            "--input_dir",
            str(tmp_path / "input"),
            "--output_dir",
            str(tmp_path / "output"),
            "--auto_check_api_config_file",
            str(config_file),
        ]
    )

    with pytest.raises(ValueError, match="at least one initial reviewer"):
        joinability_dataset.prepare_model_auto_check_reviewers(args)


def test_auto_check_default_json_config_is_discovered(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_file = tmp_path / ".auto_check_apis.json"
    config_file.write_text(
        json.dumps(
            {
                "version": 1,
                "profiles": {
                    "default_api": {
                        "response": True,
                        "initial": {
                            "api_key": "default-key",
                            "base_url": "https://default.example.test/v1",
                            "model": "gpt-5.6-luna",
                        },
                        "final_judge": None,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    config_file.chmod(0o600)
    monkeypatch.chdir(tmp_path)
    args = joinability_dataset.parse_args(
        [
            "--input_dir",
            str(tmp_path / "input"),
            "--output_dir",
            str(tmp_path / "output"),
        ]
    )

    initial_pool, final_pool = (
        joinability_dataset.prepare_model_auto_check_reviewers(args)
    )

    assert [name for name, _reviewer in initial_pool.reviewers] == [
        "default_api"
    ]
    reviewer = dict(initial_pool.reviewers)["default_api"]
    assert reviewer.client.use_responses is True
    assert reviewer.client.portable_chat_completions is False
    assert reviewer.identity["api"] == "responses"
    assert final_pool is None


def test_auto_check_json_config_hot_adds_providers_and_reuses_unchanged_clients(
    tmp_path: Path,
) -> None:
    config_file = tmp_path / "auto-check.json"

    def write_profiles(profiles: dict[str, object]) -> None:
        config_file.write_text(
            json.dumps({"version": 1, "profiles": profiles}),
            encoding="utf-8",
        )
        config_file.chmod(0o600)

    primary = {
        "initial": {
            "api_key": "fake-primary-key",
            "base_url": "https://primary.example.test/v1",
            "model": "gpt-5.6-luna",
        },
        "final_judge": None,
    }
    write_profiles({"primary": primary})
    args = joinability_dataset.parse_args(
        [
            "--input_dir",
            str(tmp_path / "input"),
            "--output_dir",
            str(tmp_path / "output"),
            "--auto_check_api_config_file",
            str(config_file),
        ]
    )
    initial_pool, final_pool = (
        joinability_dataset.prepare_model_auto_check_reviewers(args)
    )
    original_reviewer = dict(initial_pool.reviewers)["primary"]
    assert final_pool is None

    write_profiles(
        {
            "primary": primary,
            "new_gateway": {
                "initial": {
                    "api_key": "fake-new-initial-key",
                    "base_url": "https://new-initial.example.test/v1",
                    "model": "gpt-5.6-luna",
                },
                "final_judge": {
                    "api_key": "fake-new-final-key",
                    "base_url": "https://new-final.example.test/v1",
                    "model": "grok-4.5",
                },
            },
        }
    )

    assert initial_pool.reload_if_changed()
    assert [name for name, _reviewer in initial_pool.reviewers] == [
        "primary",
        "new_gateway",
    ]
    assert dict(initial_pool.reviewers)["primary"] is original_reviewer
    dynamic_final_pool = initial_pool.companion_final_pool
    assert [name for name, _reviewer in dynamic_final_pool.reviewers] == [
        "new_gateway"
    ]
    new_initial = dict(initial_pool.reviewers)["new_gateway"]
    new_final = dict(dynamic_final_pool.reviewers)["new_gateway"]
    assert (
        new_initial.client.request_controller
        is new_final.client.request_controller
    )


def test_auto_check_json_config_keeps_last_valid_profiles_after_bad_update(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    config_file = tmp_path / "auto-check.json"

    def write_profiles(names: list[str]) -> None:
        profiles = {
            name: {
                "initial": {
                    "api_key": f"fake-{name}-key",
                    "base_url": f"https://{name}.example.test/v1",
                    "model": "gpt-5.6-luna",
                },
                "final_judge": None,
            }
            for name in names
        }
        config_file.write_text(
            json.dumps({"version": 1, "profiles": profiles}),
            encoding="utf-8",
        )
        config_file.chmod(0o600)

    write_profiles(["primary"])
    args = joinability_dataset.parse_args(
        [
            "--input_dir",
            str(tmp_path / "input"),
            "--output_dir",
            str(tmp_path / "output"),
            "--auto_check_api_config_file",
            str(config_file),
        ]
    )
    initial_pool, _final_pool = (
        joinability_dataset.prepare_model_auto_check_reviewers(args)
    )

    config_file.write_text('{"version": 1, "profiles":', encoding="utf-8")
    with caplog.at_level("WARNING"):
        assert not initial_pool.reload_if_changed()
        assert not initial_pool.reload_if_changed()

    assert [name for name, _reviewer in initial_pool.reviewers] == ["primary"]
    warnings = [
        record.message
        for record in caplog.records
        if "Ignoring updated auto-check API config" in record.message
    ]
    assert len(warnings) == 1
    assert "continuing with the last valid configuration" in warnings[0]
    assert "fake-primary-key" not in warnings[0]

    write_profiles(["primary", "recovered_gateway"])
    assert initial_pool.reload_if_changed()
    assert [name for name, _reviewer in initial_pool.reviewers] == [
        "primary",
        "recovered_gateway",
    ]


def test_auto_check_explicit_json_and_dotenv_are_mutually_exclusive(
    tmp_path: Path,
) -> None:
    args = joinability_dataset.parse_args(
        [
            "--input_dir",
            str(tmp_path / "input"),
            "--output_dir",
            str(tmp_path / "output"),
            "--auto_check_api_config_file",
            str(tmp_path / "profiles.json"),
            "--auto_check_openai_env_file",
            str(tmp_path / "profiles.env"),
        ]
    )

    with pytest.raises(ValueError, match="mutually exclusive"):
        joinability_dataset.prepare_model_auto_check_reviewers(args)


def test_all_none_profiles_leave_conflicts_incomplete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment_file = tmp_path / "profiles.env"
    environment_file.write_text(
        "MMDD_AUTO_CHECK_API_PROFILES=luna_only\n"
        "MMDD_AUTO_CHECK_API_LUNA_ONLY_API_KEY=fake-luna\n"
        "MMDD_AUTO_CHECK_API_LUNA_ONLY_BASE_URL=https://luna.example.test/v1\n"
        "MMDD_AUTO_CHECK_API_LUNA_ONLY_MODEL=gpt-5.6-luna\n"
        "MMDD_AUTO_CHECK_API_LUNA_ONLY_FINAL_JUDGE_MODEL=none\n",
        encoding="utf-8",
    )
    environment_file.chmod(0o600)
    for name in (
        "MMDD_AUTO_CHECK_API_PROFILES",
        "MMDD_AUTO_CHECK_API_LUNA_ONLY_API_KEY",
        "MMDD_AUTO_CHECK_API_LUNA_ONLY_BASE_URL",
        "MMDD_AUTO_CHECK_API_LUNA_ONLY_MODEL",
        "MMDD_AUTO_CHECK_API_LUNA_ONLY_FINAL_JUDGE_MODEL",
        "MMDD_AUTO_CHECK_API_LUNA_ONLY_MAX_CONCURRENCY",
    ):
        monkeypatch.delenv(name, raising=False)
    args = joinability_dataset.parse_args(
        [
            "--input_dir",
            str(tmp_path / "input"),
            "--output_dir",
            str(tmp_path / "output"),
            "--auto_check_openai_env_file",
            str(environment_file),
        ]
    )
    initial_pool, final_pool = (
        joinability_dataset.prepare_model_auto_check_reviewers(args)
    )
    extractor = LocalAttributeExtractor(_extractor_args())
    extractor.auto_check_luna_reviewer = initial_pool
    extractor.auto_check_terra_reviewer = final_pool
    monkeypatch.setattr(
        initial_pool,
        "extract_batch",
        lambda _batch, **_kwargs: [{"extracted_value": "Alabama"}],
    )
    monkeypatch.setattr(
        extractor,
        "extract_auto_check_value",
        lambda **_kwargs: "Georgia",
    )

    review = extractor.review_auto_check_attribute(
        task=_auto_check_task(),
        attribute_name="State",
        claimed_value="Alabama",
    )

    assert final_pool is None
    assert review["verdict"] == "insufficient"
    assert review["review_complete"] is False
    assert review["decision_source"] == "final_judge_incomplete"
    assert review["error_code"] == (
        "model_review_failed:final_judge_not_configured"
    )


def test_dynamic_vllm_runner_defaults_to_instruct_image_model(tmp_path: Path) -> None:
    args, passthrough = parse_dynamic_vllm_args(
        [
            "--input_dir",
            str(tmp_path),
            "--output_dir",
            str(tmp_path / "out"),
            "--text_model_path",
            "/models/text",
            "--image_model_path",
            "/models/image",
        ]
    )

    assert args.image_model_name == "Qwen3-VL-8B-Instruct"
    assert passthrough == []


@pytest.mark.parametrize("value", ["-0.01", "1.01"])
def test_explicit_join_fallback_ratio_must_be_a_probability(
    tmp_path: Path, value: str
) -> None:
    args = joinability_dataset.parse_args(
        [
            "--input_dir",
            str(tmp_path),
            "--output_dir",
            str(tmp_path / "out"),
            "--explicit_join_fallback_ratio",
            value,
        ]
    )

    with pytest.raises(ValueError, match=r"within \[0, 1\]"):
        joinability_dataset.configured_explicit_join_fallback_ratio(args)


def test_query_rows_per_table_rejects_contradictory_output_minimum(tmp_path):
    args = joinability_dataset.parse_args(
        [
            "--input_dir",
            str(tmp_path),
            "--output_dir",
            str(tmp_path / "out"),
            "--query_rows_per_table",
            "4",
            "--min_rows_per_output_table",
            "5",
        ]
    )

    with pytest.raises(ValueError, match="min_rows_per_output_table"):
        joinability_dataset.configured_query_rows_per_table(args)


@pytest.mark.parametrize("value", ["0", "-1"])
def test_query_rows_per_table_must_be_positive(tmp_path, value):
    args = joinability_dataset.parse_args(
        [
            "--input_dir",
            str(tmp_path),
            "--output_dir",
            str(tmp_path / "out"),
            "--query_rows_per_table",
            value,
        ]
    )

    with pytest.raises(ValueError, match="must be positive"):
        joinability_dataset.configured_query_rows_per_table(args)


def test_query_rows_per_table_accepts_explicit_override(tmp_path):
    args = joinability_dataset.parse_args(
        [
            "--input_dir",
            str(tmp_path),
            "--output_dir",
            str(tmp_path / "out"),
            "--query_rows_per_table",
            "3",
        ]
    )

    assert joinability_dataset.configured_query_rows_per_table(args) == 3


@pytest.mark.parametrize(("value", "expected"), [("0", 0), ("7", 7)])
def test_max_train_query_row_views_accepts_unbounded_or_explicit_cap(
    tmp_path: Path, value: str, expected: int
) -> None:
    args = joinability_dataset.parse_args(
        [
            "--input_dir",
            str(tmp_path),
            "--output_dir",
            str(tmp_path / "out"),
            "--max_train_query_row_views_per_join",
            value,
        ]
    )

    assert (
        joinability_dataset.configured_max_train_query_row_views_per_join(args)
        == expected
    )


def test_max_train_query_row_views_rejects_negative_cap(tmp_path: Path) -> None:
    args = joinability_dataset.parse_args(
        [
            "--input_dir",
            str(tmp_path),
            "--output_dir",
            str(tmp_path / "out"),
            "--max_train_query_row_views_per_join",
            "-1",
        ]
    )

    with pytest.raises(ValueError, match="must be non-negative"):
        joinability_dataset.configured_max_train_query_row_views_per_join(args)


def test_safe_json_object_uses_final_attributes_json_after_thinking_text():
    raw = """
    Thinking Process:
    Keep this short.
    Example schema: {"attributes":[{"name":"<attribute>","value":"<value>","evidence":"<quote>"}]}

    Final answer:
    {"attributes":[{"name":"State","value":"Alabama","evidence":"BAMA"}]}
    """

    payload = safe_json_object(raw)
    attrs = normalize_extracted_attributes(payload, ["State"])

    assert attrs == [{"name": "State", "value": "Alabama"}]


def test_safe_json_object_repairs_truncated_attributes_object():
    raw = (
        '{"attributes":[{"name":"Rank","value":"10"},'
        '{"name":"Season","value":"1856"},'
        '{"name":"Fatalities","value":"400"}]'
    )

    payload = safe_json_object(raw)

    assert normalize_extracted_attributes(
        payload,
        ["Rank", "Season", "Fatalities"],
    ) == [
        {"name": "Rank", "value": "10"},
        {"name": "Season", "value": "1856"},
        {"name": "Fatalities", "value": "400"},
    ]
    assert joinability_dataset.parse_json_object(raw).method == "json_repair"


def test_safe_json_object_keeps_valid_empty_attributes_as_native_json():
    parsed = joinability_dataset.parse_json_object('{"attributes":[]}')

    assert parsed.payload == {"attributes": []}
    assert parsed.method == "json"


def test_normalize_extracted_attributes_drops_placeholders_and_non_candidates():
    payload = {
        "attributes": [
            {"name": "<attribute>", "value": "<value>", "evidence": "<quote>"},
            {"name": "Wrong", "value": "Ignored", "evidence": "outside candidate list"},
            {"name": "Year", "value": "2008", "evidence": "October 20, 2008"},
        ]
    }

    assert normalize_extracted_attributes(payload, ["Year"]) == [
        {"name": "Year", "value": "2008"}
    ]


def test_normalize_extracted_attributes_discards_explanatory_fields():
    payload = {
        "attributes": [
            {"name": "State", "value": "Alabama", "evidence": "BAMA"},
            {
                "name": "State",
                "value": "Alabama",
                "evidence": "BAMA",
                "connection_evidence": "The jersey identifies the Alabama team.",
            },
        ]
    }

    assert normalize_extracted_attributes(payload, ["State"]) == [
        {"name": "State", "value": "Alabama"},
        {"name": "State", "value": "Alabama"},
    ]


def test_chat_payload_disables_qwen_thinking_without_prompt_text(monkeypatch):
    captured = []
    responses = []

    class Response:
        def __init__(self):
            self.closed = False

        def raise_for_status(self):
            return None

        def json(self):
            return {"choices": [{"message": {"content": '{"attributes":[]}'}}]}

        def close(self):
            self.closed = True

    def fake_post(url, headers, json, timeout):
        captured.append(json)
        response = Response()
        responses.append(response)
        return response

    monkeypatch.setattr("build_mm_joinability_dataset.requests.post", fake_post)
    extractor = LocalAttributeExtractor(
        argparse.Namespace(
            text_model_base_url="http://localhost:8001/v1",
            text_model_name="Qwen3.5-9B",
            text_model_api_key=None,
            image_model_base_url="http://localhost:8000/v1",
            image_model_name="Qwen3-VL-8B-Thinking",
            image_model_api_key=None,
            model_timeout_seconds=120.0,
            model_temperature=0.0,
            model_max_tokens=1024,
            disable_thinking=True,
            model_max_retries=0,
            model_retry_sleep_seconds=0.0,
        )
    )

    prompt = extractor.extraction_prompt(
        entity_text="Alpha",
        row_attributes=[
            {"name": "Name", "value": "Alpha", "is_entity": True},
            {"name": "State", "value": "Texas", "is_entity": False},
            {"name": "Founded", "value": "1901", "is_entity": False},
        ],
        candidate_attributes=["State", "Founded"],
    )
    extractor.chat(base_url="http://localhost:8001/v1", model="Qwen3.5-9B", api_key=None, messages=[])
    extractor.chat(
        model_kind="image",
        base_url="http://localhost:8000/v1",
        model="Qwen3.5-9B",
        api_key=None,
        messages=[],
    )

    assert "Thinking Process" not in prompt
    assert "Wikipedia" not in prompt
    assert "Name [ENTITY; NEVER MASK]: Alpha" in prompt
    assert "State: Texas" in prompt
    assert "Founded: 1901" in prompt
    assert "Perform a separate leave-one-attribute-out test" in prompt
    assert "same request does not imply that they are related" in prompt
    assert "ENTITY is always visible" in prompt
    assert "candidate's displayed table value" in prompt
    assert "pretrained, memorized, and outside knowledge" not in prompt
    assert '"name":"<one candidate attribute name>","value":"<extracted value>"' in prompt
    assert "connection_evidence" not in prompt
    assert "Do not rely on Wikipedia page provenance" not in prompt
    assert len(captured) == 2
    assert all(response.closed for response in responses)
    assert all(
        payload["chat_template_kwargs"] == {"enable_thinking": False}
        for payload in captured
    )


def test_chat_closes_response_after_http_failure(monkeypatch):
    class Response:
        status_code = 503
        text = "temporarily unavailable"

        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    response = Response()

    def fake_post(url, headers, json, timeout):
        return response

    monkeypatch.setattr("build_mm_joinability_dataset.requests.post", fake_post)
    extractor = LocalAttributeExtractor(_extractor_args(model_max_retries=0))

    with pytest.raises(joinability_dataset.TransientModelEndpointError):
        extractor.chat(
            base_url="http://localhost:8001/v1",
            model="Qwen3.5-9B",
            api_key=None,
            messages=[],
        )

    assert response.closed is True


def test_extraction_prompt_requires_short_empty_json_response():
    extractor = LocalAttributeExtractor(_extractor_args())

    prompt = extractor.extraction_prompt(
        entity_text="Alpha",
        row_attributes=[
            {"name": "Name", "value": "Alpha", "is_entity": True},
            {"name": "State", "value": "Texas", "is_entity": False},
        ],
        candidate_attributes=["State"],
    )

    assert 'If no candidate attribute can be recovered, return exactly {"attributes":[]}' in prompt
    assert "Do not output evidence, explanations, rationale, confidence" in prompt


def test_image_chat_uses_default_visual_token_limit(monkeypatch):
    captured = {}

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"choices": [{"message": {"content": '{"attributes":[]}'}}]}

    def fake_post(url, headers, json, timeout):
        captured["json"] = json
        return Response()

    monkeypatch.setattr("build_mm_joinability_dataset.requests.post", fake_post)
    extractor = LocalAttributeExtractor(_extractor_args(model_max_tokens=1024))

    extractor.chat(
        model_kind="image",
        base_url="http://localhost:8000/v1",
        model="Qwen3-VL-8B-Thinking",
        api_key=None,
        messages=[],
    )

    assert captured["json"]["max_tokens"] == 384


def test_chat_records_model_request_time_and_token_usage_by_kind(monkeypatch):
    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "choices": [{"message": {"content": '{"attributes":[]}'}}],
                "usage": {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10},
            }

    def fake_post(url, headers, json, timeout):
        return Response()

    ticks = iter([100.0, 101.25, 200.0, 200.75])
    monkeypatch.setattr("build_mm_joinability_dataset.requests.post", fake_post)
    monkeypatch.setattr("build_mm_joinability_dataset.time.perf_counter", lambda: next(ticks))
    extractor = LocalAttributeExtractor(
        argparse.Namespace(
            text_model_base_url="http://localhost:8001/v1",
            text_model_name="Qwen3.5-9B",
            text_model_api_key=None,
            image_model_base_url="http://localhost:8000/v1",
            image_model_name="Qwen3-VL-8B-Thinking",
            image_model_api_key=None,
            model_timeout_seconds=120.0,
            model_temperature=0.0,
            model_max_tokens=1024,
            disable_thinking=True,
            model_max_retries=0,
            model_retry_sleep_seconds=0.0,
        )
    )

    extractor.chat(
        model_kind="text",
        base_url="http://localhost:8001/v1",
        model="Qwen3.5-9B",
        api_key=None,
        messages=[],
    )
    extractor.chat(
        model_kind="image",
        base_url="http://localhost:8000/v1",
        model="Qwen3-VL-8B-Thinking",
        api_key=None,
        messages=[],
    )

    summary = extractor.model_call_stats.summary()

    assert summary["text"]["requests"] == 1
    assert summary["text"]["elapsed_seconds"] == 1.25
    assert summary["text"]["prompt_tokens"] == 7
    assert summary["text"]["completion_tokens"] == 3
    assert summary["text"]["total_tokens"] == 10
    assert summary["text"]["responses_with_usage"] == 1
    assert summary["image"]["requests"] == 1
    assert summary["image"]["elapsed_seconds"] == 0.75
    assert summary["image"]["total_tokens"] == 10


def test_image_extraction_rejects_remote_url_to_hide_asset_provenance(monkeypatch):
    captured = {}

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"choices": [{"message": {"content": '{"attributes":[]}'}}]}

    def fake_post(url, headers, json, timeout):
        captured["json"] = json
        return Response()

    monkeypatch.setattr("build_mm_joinability_dataset.requests.post", fake_post)
    extractor = LocalAttributeExtractor(_extractor_args())

    with pytest.raises(ValueError, match="no usable local image or data URL"):
        extractor.extract(
            asset={
                "asset_id": "img_1",
                "asset_type": "image",
                "image_url": "https://upload.wikimedia.org/example/large-image.jpg",
            },
            entity={
                "cell_text": "Alpha",
                "wiki_title": "SECRET_WIKIPEDIA_TITLE",
            },
            candidate_attributes=["State"],
        )

    assert "json" not in captured


def test_image_extraction_retries_context_length_error_with_resized_local_image(monkeypatch, tmp_path):
    from PIL import Image

    image_path = tmp_path / "large.jpg"
    image = Image.new("RGB", (1000, 1000))
    pixels = image.load()
    for y in range(1000):
        for x in range(1000):
            pixels[x, y] = ((x * 17) % 256, (y * 31) % 256, ((x + y) * 13) % 256)
    image.save(image_path, format="JPEG", quality=95)
    sent_urls = []

    class Response:
        def __init__(self, status_code, text="", content='{"attributes":[]}'):
            self.status_code = status_code
            self.text = text
            self._content = content

        def raise_for_status(self):
            return None

        def json(self):
            return {"choices": [{"message": {"content": self._content}}]}

    def fake_post(url, headers, json, timeout):
        sent_urls.append(json["messages"][1]["content"][1]["image_url"]["url"])
        if len(sent_urls) == 1:
            return Response(
                400,
                '{"error":{"message":"Input length (9000) exceeds model\'s maximum context length (4096)."}}',
            )
        return Response(200)

    monkeypatch.setattr("build_mm_joinability_dataset.requests.post", fake_post)
    extractor = LocalAttributeExtractor(_extractor_args(model_max_retries=0))

    result = extractor.extract(
        asset={
            "asset_id": "img_1",
            "asset_type": "image",
            "local_path": str(image_path),
        },
        entity={
            "entity_id": "ent_1",
            "cell_text": "Alpha",
            "wiki_title": "Alpha",
        },
        candidate_attributes=["State"],
    )

    assert result["error"] == ""
    assert len(sent_urls) == 2
    assert sent_urls[0].startswith("data:image/")
    assert sent_urls[1].startswith("data:image/")
    assert len(sent_urls[1]) < len(sent_urls[0])


def test_image_extraction_resizes_local_image_before_first_request(monkeypatch, tmp_path):
    from PIL import Image

    image_path = tmp_path / "large.jpg"
    image = Image.new("RGB", (1000, 1000), color=(20, 80, 140))
    image.save(image_path, format="JPEG", quality=95)
    sent_urls = []

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"choices": [{"message": {"content": '{"attributes":[]}'}}]}

    def fake_post(url, headers, json, timeout):
        sent_urls.append(json["messages"][1]["content"][1]["image_url"]["url"])
        return Response()

    monkeypatch.setattr("build_mm_joinability_dataset.requests.post", fake_post)
    extractor = LocalAttributeExtractor(_extractor_args())

    extractor.extract(
        asset={
            "asset_id": "img_1",
            "asset_type": "image",
            "local_path": str(image_path),
        },
        entity={
            "entity_id": "ent_1",
            "cell_text": "Alpha",
            "wiki_title": "Alpha",
        },
        candidate_attributes=["State"],
    )

    original_url = image_data_url(image_path)
    assert len(sent_urls) == 1
    assert sent_urls[0].startswith("data:image/")
    assert len(sent_urls[0]) < len(original_url)


def test_joinability_projected_rows_sanitize_cell_urls_for_model_tables():
    source_table = {
        "source_table_id": "src",
        "columns": [
            {"column_index": 0, "column_name": "Entity"},
            {"column_index": 1, "column_name": "Image Source"},
        ],
        "rows": [
            {
                "row_id": 0,
                "cells": [
                    {"column_index": 0, "column_name": "Entity", "text": "Alpha"},
                    {"column_index": 1, "column_name": "Image Source", "text": "https://example.com/" + "x" * 200},
                ],
            },
            {
                "row_id": 1,
                "cells": [
                    {"column_index": 0, "column_name": "Entity", "text": "Beta"},
                    {
                        "column_index": 1,
                        "column_name": "Image Source",
                        "text": "shown at https://example.org/image.png?cache=" + "y" * 200 + " source",
                    },
                ],
            },
        ],
    }

    rows, _source_rows = project_selected_rows(
        source_table,
        [0, 1],
        {0, 1},
        min_required_cols=1,
    )

    assert rows[0]["cells"][1]["text"] == "[url]"
    assert rows[1]["cells"][1]["text"] == "shown at source"


def test_table_record_accepts_source_provenance_builder():
    record = joinability_dataset.table_record(
        table_id="table_1",
        role="query_table",
        split="train",
        source_table={
            "source_table_id": "source_1",
            "columns": [],
            "provenance_builder": "wdc",
        },
        column_indices=[],
        rows=[],
        source_row_indices=[],
        extra={},
    )

    assert record["provenance"]["builder"] == "wdc"


@pytest.mark.parametrize(
    "source_overrides",
    [
        {},
        {"provenance_builder": ""},
        {"provenance_builder": "   "},
    ],
    ids=["missing", "empty", "whitespace"],
)
def test_table_record_defaults_dataset_provenance_builder(source_overrides):
    record = joinability_dataset.table_record(
        table_id="table_1",
        role="query_table",
        split="train",
        source_table={
            "source_table_id": "source_1",
            "columns": [],
            **source_overrides,
        },
        column_indices=[],
        rows=[],
        source_row_indices=[],
        extra={},
    )

    assert record["provenance"]["builder"] == "build_mm_joinability_dataset.py"


def test_old_cache_key_is_preserved_when_disabling_thinking():
    base = dict(
        asset_id="asset_1",
        entity_id="entity_1",
        candidate_attribute_names=["State"],
        asset_type="text",
    )
    args_disabled = argparse.Namespace(text_model_name="Qwen3.5-9B", image_model_name="Qwen3-VL-8B-Thinking")
    args_enabled = argparse.Namespace(text_model_name="Qwen3.5-9B", image_model_name="Qwen3-VL-8B-Thinking")

    assert extraction_cache_key(**base, args=args_disabled) == extraction_cache_key(**base, args=args_enabled)


def test_extraction_cache_key_includes_batched_row_context():
    args = argparse.Namespace(
        text_model_name="text-model",
        image_model_name="image-model",
    )
    base = dict(
        asset_id="asset_1",
        entity_id="entity_1",
        candidate_attribute_names=["State"],
        asset_type="text",
        args=args,
    )

    texas_key = extraction_cache_key(
        **base,
        row_attributes=[
            {"name": "Name", "value": "Alpha", "is_entity": True},
            {"name": "State", "value": "Texas", "is_entity": False},
        ],
    )
    ohio_key = extraction_cache_key(
        **base,
        row_attributes=[
            {"name": "Name", "value": "Alpha", "is_entity": True},
            {"name": "State", "value": "Ohio", "is_entity": False},
        ],
    )

    assert texas_key != ohio_key


def test_extraction_row_attributes_use_only_sanitized_cell_content():
    table = {
        "columns": [
            {"column_index": 0, "column_name": "Name"},
            {"column_index": 1, "column_name": "State"},
            {"column_index": 2, "column_name": "Reference"},
        ]
    }
    row = {
        "cells": [
            {
                "column_index": 0,
                "text": "Alpha",
                "wiki_title": "SECRET_WIKIPEDIA_TITLE",
            },
            {"column_index": 1, "text": "Texas"},
            {
                "column_index": 2,
                "text": "https://private.example/entity/alpha",
            },
        ]
    }

    attributes = extraction_row_attributes(table, row, entity_col=0)

    assert attributes == [
        {"name": "Name", "value": "Alpha", "is_entity": True},
        {"name": "State", "value": "Texas", "is_entity": False},
        {"name": "Reference", "value": "[url]", "is_entity": False},
    ]
    assert "SECRET_WIKIPEDIA_TITLE" not in json.dumps(attributes)


def test_collect_extraction_task_keeps_one_call_with_full_row_context():
    source_table = {
        "source_table_id": "source-1",
        "columns": [
            {"column_index": 0, "column_name": "Name"},
            {"column_index": 1, "column_name": "State"},
            {"column_index": 2, "column_name": "Founded"},
        ],
        "rows": [
            {
                "row_id": 0,
                "cells": [
                    {
                        "column_index": 0,
                        "text": "Alpha",
                        "wiki_title": "SECRET_WIKIPEDIA_TITLE",
                    },
                    {"column_index": 1, "text": "Texas"},
                    {"column_index": 2, "text": "1901"},
                ],
            }
        ],
        "metadata": {
            "candidate_entity_columns": [0],
            "column_profiles": [
                {"column_index": 0, "non_empty_ratio": 1.0},
                {"column_index": 1, "non_empty_ratio": 1.0},
                {"column_index": 2, "non_empty_ratio": 1.0},
            ],
        },
    }
    args = argparse.Namespace(
        query_rows_per_table=1,
        min_rows_per_output_table=1,
        min_column_non_empty_ratio=0.5,
        text_model_name="text-model",
        image_model_name="image-model",
    )

    tasks = joinability_dataset.collect_table_extraction_tasks(
        source_table=source_table,
        assets={
            "asset-1": {
                "asset_id": "asset-1",
                "asset_type": "text",
                "content": "Alpha is in Texas and was founded in 1901.",
            }
        },
        entity_to_assets={"entity-1": ["asset-1"]},
        wiki_to_entity_id={"SECRET_WIKIPEDIA_TITLE": "entity-1"},
        args=args,
    )

    assert len(tasks) == 1
    assert tasks[0].candidate_attribute_names == ["State", "Founded"]
    assert tasks[0].entity["row_attributes"] == [
        {"name": "Name", "value": "Alpha", "is_entity": True},
        {"name": "State", "value": "Texas", "is_entity": False},
        {"name": "Founded", "value": "1901", "is_entity": False},
    ]


def test_reparse_extraction_record_updates_old_cached_raw_response():
    cached = {
        "attributes": [{"name": "<attribute>", "value": "<value>", "evidence": "<quote>"}],
        "raw_response": (
            'Thinking Process... schema {"attributes":[{"name":"<attribute>","value":"<value>","evidence":"<quote>"}]} '
            'final {"attributes":[{"name":"State","value":"Alabama","evidence":"BAMA"}]}'
        ),
    }

    updated, changed = reparse_extraction_record(cached, ["State"])

    assert changed
    assert updated["attributes"] == [{"name": "State", "value": "Alabama"}]


def test_reparse_extraction_record_repairs_cache_without_changing_key():
    cached = {
        "cache_key": "stable-extraction-key",
        "attributes": [],
        "raw_response": (
            '{"attributes":[{"name":"Season","value":"1856"}]'
        ),
        "error": "",
    }

    updated, changed = reparse_extraction_record(cached, ["Season"])

    assert changed is True
    assert updated["cache_key"] == "stable-extraction-key"
    assert updated["attributes"] == [{"name": "Season", "value": "1856"}]
    assert updated["raw_response_parse_method"] == "json_repair"


def test_reparse_extraction_record_does_not_rewrite_valid_empty_result():
    cached = {
        "cache_key": "legitimate-empty",
        "attributes": [],
        "raw_response": '{"attributes":[]}',
        "error": "",
    }

    updated, changed = reparse_extraction_record(cached, ["Season"])

    assert changed is False
    assert updated is cached


def _task(asset_type: str, suffix: str = "1") -> ExtractionTask:
    return ExtractionTask(
        order=0,
        cache_key=f"cache_{asset_type}_{suffix}",
        source_table_id="src",
        source_row_id=0,
        entity_column_index=0,
        entity_column_name="Entity",
        entity={"entity_id": f"ent_{suffix}", "wiki_title": f"Entity {suffix}", "cell_text": f"Entity {suffix}"},
        asset={"asset_id": f"asset_{asset_type}_{suffix}", "asset_type": asset_type, "content": "State: Alabama"},
        candidate_attribute_names=["State"],
    )


def _auto_check_task() -> ExtractionTask:
    return ExtractionTask(
        order=0,
        cache_key="cache_text_auto_check",
        source_table_id="src",
        source_row_id=7,
        entity_column_index=0,
        entity_column_name="Entity",
        entity={
            "entity_id": "ent_auto_check",
            "wiki_title": "Alpha",
            "cell_text": "Alpha",
            "row_attributes": [
                {"name": "Entity", "value": "Alpha", "is_entity": True},
                {"name": "State", "value": "Alabama", "is_entity": False},
                {"name": "Founded", "value": "1901", "is_entity": False},
            ],
        },
        asset={
            "asset_id": "asset_text_auto_check",
            "asset_type": "text",
            "content": "Alpha is in Alabama.",
        },
        candidate_attribute_names=["State", "Founded"],
    )


def test_post_analysis_auto_check_keeps_only_supported_attributes():
    class Checker:
        auto_check_enabled = True

        def __init__(self):
            self.model_auto_check_stats = ModelAutoCheckStats()

        def extract_auto_check_value(self, *, attribute_name, **_kwargs):
            return "Alabama" if attribute_name == "State" else ""

    record = apply_model_auto_check(
        extractor=Checker(),
        task=_auto_check_task(),
        record={
            "attributes": [
                {"name": "State", "value": "Alabama"},
                {"name": "Founded", "value": "1901"},
            ],
            "error": "",
        },
    )

    assert record["model_attributes"] == [
        {"name": "State", "value": "Alabama"},
        {"name": "Founded", "value": "1901"},
    ]
    assert record["attributes"] == [{"name": "State", "value": "Alabama"}]
    assert record["auto_check"]["supported_attributes"] == 1
    assert record["auto_check"]["filtered_attributes"] == 1
    assert [review["verdict"] for review in record["auto_check"]["reviews"]] == [
        "supported",
        "insufficient",
    ]


def test_post_analysis_auto_check_fails_closed_on_checker_error():
    class BrokenChecker:
        auto_check_enabled = True

        def extract_auto_check_value(self, **_kwargs):
            raise RuntimeError("synthetic checker outage")

    record = apply_model_auto_check(
        extractor=BrokenChecker(),
        task=_auto_check_task(),
        record={
            "attributes": [{"name": "State", "value": "Alabama"}],
            "error": "",
        },
    )

    assert record["attributes"] == []
    assert record["auto_check"]["reviews"][0]["comparison"] == (
        "auto_check_failed"
    )
    assert record["auto_check"]["reviews"][0]["error_code"] == "RuntimeError"


def test_post_analysis_auto_check_accepts_supported_terra_adjudication():
    class Checker:
        auto_check_enabled = True

        def __init__(self):
            self.model_auto_check_stats = ModelAutoCheckStats()

        def review_auto_check_attribute(self, **_kwargs):
            return {
                "extracted_value": "Alabama",
                "verdict": "supported",
                "comparison": "normalized_values_match",
                "decision_source": "terra_adjudication",
                "review_complete": True,
                "error_code": "",
                "primary_extracted_value": "Georgia",
                "primary_verdict": "contradicted",
                "primary_comparison": "extracted_value_mismatch",
                "primary_error_code": "",
                "luna_triggered": True,
                "luna_extracted_value": "Alabama",
                "luna_verdict": "supported",
                "luna_comparison": "normalized_values_match",
                "luna_agrees_with_local": False,
                "luna_error_code": "",
                "terra_triggered": True,
                "terra_extracted_value": "Alabama",
                "terra_verdict": "supported",
                "terra_comparison": "normalized_values_match",
                "terra_error_code": "",
            }

    checker = Checker()
    record = apply_model_auto_check(
        extractor=checker,
        task=_auto_check_task(),
        record={
            "attributes": [{"name": "State", "value": "Alabama"}],
            "error": "",
        },
    )

    assert record["attributes"] == [{"name": "State", "value": "Alabama"}]
    review = record["auto_check"]["reviews"][0]
    assert review["decision_source"] == "terra_adjudication"
    assert review["primary_verdict"] == "contradicted"
    assert review["luna_triggered"] is True
    assert review["terra_triggered"] is True
    assert checker.model_auto_check_stats.summary()["terra_supported"] == 1


def test_post_analysis_auto_check_keeps_source_value_for_equivalent_full_name():
    class Checker:
        auto_check_enabled = True

        def review_auto_check_attribute(self, **_kwargs):
            return {
                "extracted_value": "観音駅",
                "verdict": "supported",
                "comparison": "normalized_values_match",
                "decision_source": "terra_adjudication",
                "review_complete": True,
                "error_code": "",
            }

    task = ExtractionTask(
        order=0,
        cache_key="cache_station",
        source_table_id="src",
        source_row_id=0,
        entity_column_index=0,
        entity_column_name="Station",
        entity={
            "cell_text": "Kannon",
            "row_attributes": [
                {"name": "Station", "value": "Kannon", "is_entity": True},
                {"name": "Japanese", "value": "観音", "is_entity": False},
            ],
        },
        asset={"asset_id": "asset_station", "asset_type": "image"},
        candidate_attribute_names=["Japanese"],
    )

    record = apply_model_auto_check(
        extractor=Checker(),
        task=task,
        record={
            "attributes": [{"name": "Japanese", "value": "観音"}],
            "error": "",
        },
    )

    assert record["attributes"] == [{"name": "Japanese", "value": "観音"}]
    assert record["auto_check"]["reviews"][0]["extracted_value"] == "観音駅"


def test_local_auto_check_escalates_luna_disagreement_to_terra(monkeypatch):
    extractor = LocalAttributeExtractor(_extractor_args())

    class Reviewer:
        def __init__(self, value):
            self.value = value
            self.calls = 0

        def extract_batches(self, batches):
            self.calls += 1
            return {
                batches[0]["query_table_id"]: [
                    {"extracted_value": self.value}
                ]
            }

    luna = Reviewer("Alabama")
    terra = Reviewer("Alabama")
    extractor.auto_check_luna_reviewer = luna
    extractor.auto_check_terra_reviewer = terra
    monkeypatch.setattr(
        extractor,
        "extract_auto_check_value",
        lambda **_kwargs: "Georgia",
    )

    review = extractor.review_auto_check_attribute(
        task=_auto_check_task(),
        attribute_name="State",
        claimed_value="Alabama",
    )

    assert review["verdict"] == "supported"
    assert review["decision_source"] == "terra_adjudication"
    assert review["primary_verdict"] == "contradicted"
    assert review["luna_extracted_value"] == "Alabama"
    assert review["luna_agrees_with_local"] is False
    assert review["terra_extracted_value"] == "Alabama"
    assert luna.calls == 1
    assert terra.calls == 1


def test_supported_local_auto_check_still_runs_luna(monkeypatch):
    extractor = LocalAttributeExtractor(_extractor_args())

    class Reviewer:
        def __init__(self, value):
            self.value = value
            self.calls = 0

        def extract_batches(self, batches):
            self.calls += 1
            return {
                batches[0]["query_table_id"]: [
                    {"extracted_value": self.value}
                ]
            }

    luna = Reviewer("Alabama")
    terra = Reviewer("Georgia")
    extractor.auto_check_luna_reviewer = luna
    extractor.auto_check_terra_reviewer = terra
    monkeypatch.setattr(
        extractor,
        "extract_auto_check_value",
        lambda **_kwargs: "Alabama",
    )

    review = extractor.review_auto_check_attribute(
        task=_auto_check_task(),
        attribute_name="State",
        claimed_value="Alabama",
    )

    assert review["verdict"] == "supported"
    assert review["decision_source"] == "local_luna_consensus"
    assert review["luna_agrees_with_local"] is True
    assert luna.calls == 1
    assert terra.calls == 0


def test_failed_local_auto_check_still_runs_luna_and_terra(monkeypatch):
    extractor = LocalAttributeExtractor(_extractor_args())

    class Reviewer:
        def __init__(self, value):
            self.value = value
            self.calls = 0

        def extract_batches(self, batches):
            self.calls += 1
            return {
                batches[0]["query_table_id"]: [
                    {"extracted_value": self.value}
                ]
            }

    luna = Reviewer("Alabama")
    terra = Reviewer("Alabama")
    extractor.auto_check_luna_reviewer = luna
    extractor.auto_check_terra_reviewer = terra
    monkeypatch.setattr(
        extractor,
        "extract_auto_check_value",
        lambda **_kwargs: (_ for _ in ()).throw(
            ValueError("invalid local JSON")
        ),
    )

    review = extractor.review_auto_check_attribute(
        task=_auto_check_task(),
        attribute_name="State",
        claimed_value="Alabama",
    )

    assert review["primary_error_code"]
    assert review["luna_triggered"] is True
    assert review["luna_agrees_with_local"] is False
    assert review["terra_triggered"] is True
    assert review["decision_source"] == "terra_adjudication"
    assert review["verdict"] == "supported"
    assert review["review_complete"] is True
    assert luna.calls == 1
    assert terra.calls == 1


def test_supported_local_luna_disagreement_runs_final_judge(monkeypatch):
    extractor = LocalAttributeExtractor(_extractor_args())

    class Reviewer:
        def __init__(self, value):
            self.value = value
            self.calls = 0

        def extract_batches(self, batches):
            self.calls += 1
            return {
                batches[0]["query_table_id"]: [
                    {"extracted_value": self.value}
                ]
            }

    luna = Reviewer("Georgia")
    terra = Reviewer("Alabama")
    extractor.auto_check_luna_reviewer = luna
    extractor.auto_check_terra_reviewer = terra
    monkeypatch.setattr(
        extractor,
        "extract_auto_check_value",
        lambda **_kwargs: "Alabama",
    )

    review = extractor.review_auto_check_attribute(
        task=_auto_check_task(),
        attribute_name="State",
        claimed_value="Alabama",
    )

    assert review["primary_verdict"] == "supported"
    assert review["luna_agrees_with_local"] is False
    assert review["decision_source"] == "terra_adjudication"
    assert review["verdict"] == "supported"
    assert luna.calls == 1
    assert terra.calls == 1


def test_supported_deferred_auto_check_waits_for_luna(monkeypatch):
    extractor = LocalAttributeExtractor(_extractor_args())

    class Reviewer:
        def __init__(self, value):
            self.value = value
            self.calls = 0

        def extract_batches(self, batches):
            self.calls += 1
            return {
                batches[0]["query_table_id"]: [
                    {"extracted_value": self.value}
                ]
            }

    luna = Reviewer("Alabama")
    terra = Reviewer("Georgia")
    extractor.auto_check_luna_reviewer = luna
    extractor.auto_check_terra_reviewer = terra
    monkeypatch.setattr(
        extractor,
        "extract_auto_check_value",
        lambda **_kwargs: "Alabama",
    )

    pending = extractor.review_auto_check_attribute(
        task=_auto_check_task(),
        attribute_name="State",
        claimed_value="Alabama",
        defer_remote=True,
    )
    assert pending["decision_source"] == "remote_review_pending"
    assert pending["review_complete"] is False
    assert luna.calls == 0

    completed = extractor.complete_auto_check_attribute_review(
        task=_auto_check_task(),
        attribute_name="State",
        claimed_value="Alabama",
        local_review=pending,
    )
    assert completed["decision_source"] == "local_luna_consensus"
    assert completed["verdict"] == "supported"
    assert luna.calls == 1
    assert terra.calls == 0


def test_local_only_auto_check_keeps_primary_result(monkeypatch):
    extractor = LocalAttributeExtractor(_extractor_args())
    extractor.auto_check_luna_reviewer = None
    extractor.auto_check_terra_reviewer = None
    monkeypatch.setattr(
        extractor,
        "extract_auto_check_value",
        lambda **_kwargs: "Alabama",
    )

    review = extractor.review_auto_check_attribute(
        task=_auto_check_task(),
        attribute_name="State",
        claimed_value="Alabama",
    )

    assert review["verdict"] == "supported"
    assert review["decision_source"] == "primary_local"
    assert review["luna_triggered"] is False
    assert review["terra_triggered"] is False


def test_cascade_reuses_previous_local_result_and_only_adds_luna(monkeypatch):
    extractor = LocalAttributeExtractor(_extractor_args())

    class Reviewer:
        def __init__(self, value):
            self.value = value
            self.calls = 0

        def extract_batches(self, batches):
            self.calls += 1
            return {
                batches[0]["query_table_id"]: [
                    {"extracted_value": self.value}
                ]
            }

    luna = Reviewer("Alabama")
    terra = Reviewer("Georgia")
    extractor.auto_check_luna_reviewer = luna
    extractor.auto_check_terra_reviewer = terra
    monkeypatch.setattr(
        extractor,
        "extract_auto_check_value",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("the cached primary result must be reused")
        ),
    )
    previous = {
        "attribute_name": "State",
        "claimed_value": "Alabama",
        "extracted_value": "Alabama",
        "verdict": "supported",
        "comparison": "normalized_values_match",
        "decision_source": "primary_local",
        "review_complete": True,
        "error_code": "",
        "primary_extracted_value": "Alabama",
        "primary_verdict": "supported",
        "primary_comparison": "normalized_values_match",
        "primary_error_code": "",
        "luna_triggered": False,
        "luna_extracted_value": None,
        "luna_error_code": "",
    }

    review = extractor.complete_auto_check_attribute_review(
        task=_auto_check_task(),
        attribute_name="State",
        claimed_value="Alabama",
        local_review=previous,
    )

    assert review["decision_source"] == "local_luna_consensus"
    assert review["verdict"] == "supported"
    assert luna.calls == 1
    assert terra.calls == 0


def test_cascade_reuses_previous_luna_and_terra_results(monkeypatch):
    extractor = LocalAttributeExtractor(_extractor_args())

    class Reviewer:
        def __init__(self) -> None:
            self.calls = 0

        def extract_batches(self, _batches):
            self.calls += 1
            raise AssertionError("the cached remote result must be reused")

    luna = Reviewer()
    terra = Reviewer()
    extractor.auto_check_luna_reviewer = luna
    extractor.auto_check_terra_reviewer = terra
    monkeypatch.setattr(
        extractor,
        "extract_auto_check_value",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("the cached primary result must be reused")
        ),
    )
    previous = {
        "primary_extracted_value": "Alabama",
        "primary_verdict": "supported",
        "primary_comparison": "normalized_values_match",
        "primary_error_code": "",
        "luna_triggered": True,
        "luna_extracted_value": "Georgia",
        "luna_verdict": "contradicted",
        "luna_comparison": "extracted_value_mismatch",
        "luna_error_code": "",
        "terra_triggered": True,
        "terra_extracted_value": "Alabama",
        "terra_verdict": "supported",
        "terra_comparison": "normalized_values_match",
        "terra_error_code": "",
        "review_complete": True,
        "error_code": "",
    }

    review = extractor.complete_auto_check_attribute_review(
        task=_auto_check_task(),
        attribute_name="State",
        claimed_value="Alabama",
        local_review=previous,
    )

    assert review["decision_source"] == "terra_adjudication"
    assert review["verdict"] == "supported"
    assert luna.calls == 0
    assert terra.calls == 0


def test_local_and_luna_matching_mismatch_skips_terra(monkeypatch):
    extractor = LocalAttributeExtractor(_extractor_args())

    class Reviewer:
        def __init__(self, value):
            self.value = value
            self.calls = 0

        def extract_batches(self, batches):
            self.calls += 1
            return {
                batches[0]["query_table_id"]: [
                    {"extracted_value": self.value}
                ]
            }

    luna = Reviewer("Georgia")
    terra = Reviewer("Alabama")
    extractor.auto_check_luna_reviewer = luna
    extractor.auto_check_terra_reviewer = terra
    monkeypatch.setattr(
        extractor,
        "extract_auto_check_value",
        lambda **_kwargs: "Georgia",
    )

    review = extractor.review_auto_check_attribute(
        task=_auto_check_task(),
        attribute_name="State",
        claimed_value="Alabama",
    )

    assert review["verdict"] == "contradicted"
    assert review["decision_source"] == "local_luna_consensus"
    assert review["luna_agrees_with_local"] is True
    assert luna.calls == 1
    assert terra.calls == 0


def test_deferred_auto_check_does_not_repeat_local_inference(monkeypatch):
    extractor = LocalAttributeExtractor(_extractor_args())

    class Reviewer:
        def __init__(self, value):
            self.value = value
            self.calls = 0

        def extract_batches(self, batches):
            self.calls += 1
            return {
                batches[0]["query_table_id"]: [
                    {"extracted_value": self.value}
                ]
            }

    local_calls = 0

    def extract_local(**_kwargs):
        nonlocal local_calls
        local_calls += 1
        return "Georgia"

    luna = Reviewer("Alabama")
    terra = Reviewer("Alabama")
    extractor.auto_check_luna_reviewer = luna
    extractor.auto_check_terra_reviewer = terra
    monkeypatch.setattr(extractor, "extract_auto_check_value", extract_local)

    pending = apply_model_auto_check(
        extractor=extractor,
        task=_auto_check_task(),
        record={
            "attributes": [{"name": "State", "value": "Alabama"}],
            "error": "",
        },
        defer_remote=True,
    )

    assert local_calls == 1
    assert luna.calls == 0
    assert terra.calls == 0
    review = pending["auto_check"]["reviews"][0]
    assert review["decision_source"] == "remote_review_pending"
    assert review["review_complete"] is False

    completed = joinability_dataset.complete_deferred_model_auto_check(
        extractor=extractor,
        task=_auto_check_task(),
        record=pending,
    )

    assert local_calls == 1
    assert luna.calls == 1
    assert terra.calls == 1
    assert completed["attributes"] == [
        {"name": "State", "value": "Alabama"}
    ]


def test_resolve_finishes_raw_extraction_without_starting_openai(tmp_path):
    luna_calls = 0
    local_done = threading.Event()

    class Reviewer:
        def extract_batches(self, batches):
            nonlocal luna_calls
            luna_calls += 1
            return {
                batches[0]["query_table_id"]: [
                    {"extracted_value": "Georgia"}
                ]
            }

    class Extractor:
        auto_check_enabled = True
        model_auto_check_stats = ModelAutoCheckStats()
        auto_check_luna_reviewer = Reviewer()
        auto_check_terra_reviewer = None

        def extract(self, *_args, **_kwargs):
            return {
                "attributes": [{"name": "State", "value": "Alabama"}],
                "raw_response": '{"attributes":[]}',
                "error": "",
            }

        def extract_auto_check_value(self, **_kwargs):
            return "Georgia"

        review_auto_check_attribute = (
            LocalAttributeExtractor.review_auto_check_attribute
        )
        complete_auto_check_attribute_review = (
            LocalAttributeExtractor.complete_auto_check_attribute_review
        )
        _auto_check_review_batch = LocalAttributeExtractor._auto_check_review_batch

    task = _auto_check_task()
    cache = ExtractionCache(tmp_path / "model_cache.jsonl")
    result = resolve_extraction_tasks(
        extractor=Extractor(),
        cache=cache,
        tasks=[task],
        args=_parallel_args(auto_check_openai_max_inflight=1),
        state=ModelConcurrencyState(text_workers=1, image_workers=1),
        on_local_phase_done=local_done.set,
    )

    assert local_done.is_set()
    assert luna_calls == 0
    assert result[0][1]["attributes"] == [
        {"name": "State", "value": "Alabama"}
    ]
    assert "auto_check" not in result[0][1]
    assert cache.get(task.cache_key) is not None


def _parallel_args(**overrides):
    values = {
        "text_model_workers": 1,
        "image_model_workers": 1,
        "cache_failed_model_outputs": False,
        "model_attribute_errors_path": "",
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def _extractor_args(**overrides):
    values = {
        "text_model_base_url": "http://localhost:8001/v1",
        "text_model_base_urls": None,
        "text_model_base_urls_file": None,
        "text_model_name": "Qwen3.5-9B",
        "text_model_api_key": None,
        "image_model_base_url": "http://localhost:8000/v1",
        "image_model_base_urls": None,
        "image_model_base_urls_file": None,
        "image_model_name": "Qwen3-VL-8B-Thinking",
        "image_model_api_key": None,
        "model_timeout_seconds": 120.0,
        "model_temperature": 0.0,
        "model_max_tokens": 1024,
        "disable_thinking": True,
        "model_max_retries": 0,
        "model_retry_sleep_seconds": 0.0,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_local_attribute_extractor_uses_one_model_endpoint_config(tmp_path):
    config_path = tmp_path / "model-endpoints.json"
    config_path.write_text(
        json.dumps(
            {
                "schema_version": "mmdd-model-endpoints-v1",
                "served_model_name": "Qwen3.5-9B",
                "endpoints": [
                    {
                        "endpoint_id": "local-text",
                        "base_url": "http://127.0.0.1:8001/v1",
                        "pool": "local",
                        "max_inflight": {
                            "text": 2,
                            "image": 0,
                            "total": 2,
                        },
                    },
                    {
                        "endpoint_id": "local-image",
                        "base_url": "http://127.0.0.1:8000/v1",
                        "pool": "local",
                        "max_inflight": {
                            "text": 0,
                            "image": 1,
                            "total": 1,
                        },
                    },
                    {
                        "endpoint_id": "remote-mm",
                        "base_url": "http://127.0.0.1:18011/v1",
                        "pool": "remote",
                        "max_inflight": {
                            "text": 4,
                            "image": 2,
                            "total": 5,
                        },
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    extractor = LocalAttributeExtractor(
        _extractor_args(
            model_endpoint_config=str(config_path),
            remote_text_model_workers=4,
            remote_image_model_workers=2,
        )
    )

    assert extractor.text_model_name == "Qwen3.5-9B"
    assert extractor.image_model_name == "Qwen3.5-9B"
    assert extractor.current_text_model_base_urls() == [
        "http://127.0.0.1:8001/v1"
    ]
    assert extractor.current_image_model_base_urls() == [
        "http://127.0.0.1:8000/v1"
    ]
    assert extractor.current_remote_text_model_base_urls() == [
        "http://127.0.0.1:18011/v1"
    ]
    assert extractor.current_remote_image_model_base_urls() == [
        "http://127.0.0.1:18011/v1"
    ]


def test_builder_auto_check_uses_existing_blind_single_attribute_prompt(
    monkeypatch,
):
    extractor = LocalAttributeExtractor(_extractor_args())
    calls = []

    def chat(**kwargs):
        calls.append(kwargs)
        return '{"extracted_value":"Alabama"}'

    monkeypatch.setattr(extractor, "chat", chat)

    task = _auto_check_task()
    task.asset["content"] = "Alpha has a state listed in the material."
    value = extractor.extract_auto_check_value(
        task=task,
        attribute_name="State",
        claimed_value="Alabama",
    )

    assert value == "Alabama"
    rendered = json.dumps(calls[0]["messages"], ensure_ascii=False)
    assert "Alabama" not in rendered
    assert "Alpha" in rendered
    assert "1901" in rendered
    assert calls[0]["response_schema"]["required"] == ["extracted_value"]


def test_endpoint_pools_add_urls_from_runtime_files(tmp_path):
    text_endpoint_file = tmp_path / "text_endpoints.txt"
    endpoint_file = tmp_path / "image_endpoints.txt"
    text_endpoint_file.write_text("http://localhost:8001/v1\n", encoding="utf-8")
    endpoint_file.write_text("http://localhost:8000/v1\n", encoding="utf-8")
    extractor = LocalAttributeExtractor(
        _extractor_args(
            image_model_base_url="http://localhost:8000/v1",
            text_model_base_urls_file=str(text_endpoint_file),
            image_model_base_urls_file=str(endpoint_file),
        )
    )

    assert extractor.next_text_model_base_url() == "http://localhost:8001/v1"
    assert extractor.next_image_model_base_url() == "http://localhost:8000/v1"

    text_endpoint_file.write_text(
        "http://localhost:8001/v1\nhttp://localhost:8003/v1\n",
        encoding="utf-8",
    )
    endpoint_file.write_text(
        "http://localhost:8000/v1\nhttp://localhost:8002/v1\n",
        encoding="utf-8",
    )

    text_urls = [extractor.next_text_model_base_url() for _ in range(4)]
    urls = [extractor.next_image_model_base_url() for _ in range(4)]
    assert text_urls == [
        "http://localhost:8003/v1",
        "http://localhost:8001/v1",
        "http://localhost:8003/v1",
        "http://localhost:8001/v1",
    ]
    assert urls == [
        "http://localhost:8002/v1",
        "http://localhost:8000/v1",
        "http://localhost:8002/v1",
        "http://localhost:8000/v1",
    ]


def test_runtime_endpoint_file_is_authoritative_when_service_is_removed(tmp_path):
    text_endpoint_file = tmp_path / "text_endpoints.txt"
    text_endpoint_file.write_text("http://localhost:8003/v1\n", encoding="utf-8")
    extractor = LocalAttributeExtractor(
        _extractor_args(text_model_base_urls_file=str(text_endpoint_file))
    )

    assert [extractor.next_text_model_base_url() for _ in range(3)] == [
        "http://localhost:8003/v1",
        "http://localhost:8003/v1",
        "http://localhost:8003/v1",
    ]


def test_remote_endpoint_pool_routes_separately_with_its_own_key(monkeypatch):
    extractor = LocalAttributeExtractor(
        _extractor_args(
            text_model_base_url="http://localhost:8001/v1",
            text_model_api_key="local-key",
            remote_text_model_base_url="http://remote.example:18001/v1",
            remote_text_model_api_key="remote-key",
            remote_text_model_workers=4,
        )
    )
    calls = []

    def chat(**kwargs):
        calls.append(kwargs)
        return '{"attributes":[]}'

    monkeypatch.setattr(extractor, "chat", chat)
    task = _task("text", "remote-route")

    extractor.extract(
        task.asset,
        task.entity,
        task.candidate_attribute_names,
    )
    extractor.extract_from_pool(
        task.asset,
        task.entity,
        task.candidate_attribute_names,
        endpoint_pool="remote",
    )

    assert [call["base_url"] for call in calls] == [
        "http://localhost:8001/v1",
        "http://remote.example:18001/v1",
    ]
    assert [call["api_key"] for call in calls] == ["local-key", "remote-key"]


def test_remote_workers_require_a_remote_endpoint():
    with pytest.raises(ValueError, match="remote text workers require"):
        LocalAttributeExtractor(
            _extractor_args(remote_text_model_workers=2)
        )


def test_dynamic_remote_workers_leave_tasks_for_local_pool_until_routed():
    calls: list[str] = []

    class Extractor:
        abort_on_transient_error = False

        def wait_for_endpoint_pool(
            self,
            model_kind: str,
            endpoint_pool: str,
            timeout_seconds: float,
        ) -> bool:
            assert model_kind == "text"
            assert endpoint_pool == "remote"
            time.sleep(min(timeout_seconds, 0.01))
            return False

        def extract(self, *_args):
            calls.append("local")
            return {"attributes": [], "raw_response": ""}

    tasks = [_task("text", str(index)) for index in range(4)]
    records = joinability_dataset.run_extraction_kind_distributed(
        extractor=Extractor(),
        tasks=tasks,
        model_kind="text",
        state=ModelConcurrencyState(
            text_workers=1,
            image_workers=1,
            remote_text_workers=2,
        ),
    )

    assert set(records) == {task.cache_key for task in tasks}
    assert calls == ["local"] * len(tasks)
    assert all(not record.get("error") for record in records.values())


def test_entitables_remote_borrower_returns_lease_before_acknowledging_wdc(
    tmp_path,
    monkeypatch,
):
    events: list[str] = []
    controller_started = threading.Event()

    class FakeController:
        def __init__(self, _config, _scheduler):
            pass

        def start(self):
            events.append("controller:start")
            controller_started.set()

        def request_stop(self):
            events.append("controller:stop-requested")

        def close(self, **_kwargs):
            events.append("controller:close")

    monkeypatch.setattr(
        joinability_dataset,
        "RemoteLayoutController",
        FakeController,
    )
    scheduler = joinability_dataset.RoutingScheduler(
        tmp_path / "routing.json",
        load_existing=False,
    )
    borrower = joinability_dataset.EntiTablesRemoteGpuBorrower(
        controller_config=SimpleNamespace(request_timeout_seconds=0.1),
        scheduler=scheduler,
        workload=joinability_dataset._RoundRemoteWorkload(
            {"text": 1, "image": 1}
        ),
        coordination_dir=tmp_path / "priority",
        coordination_poll_seconds=0.01,
        drain_timeout_seconds=1.0,
    )
    owner = joinability_dataset.gpu_priority.PriorityGpuOwner(
        tmp_path / "priority",
        gpu_ids=("remote:primary_image", "remote:switchable"),
        owner="wdc-remote",
        reclaim_timeout_seconds=2.0,
        borrower_stale_seconds=1.0,
        unregistered_grace_seconds=0.1,
        poll_seconds=0.01,
    )

    borrower.start()
    try:
        assert controller_started.wait(timeout=1.0)
        request = owner.request_gpus(reason="wdc_model_stage")
        acknowledgement = joinability_dataset.gpu_priority.read_json(
            owner.paths.acknowledgement
        )
        assert acknowledgement is not None
        assert acknowledgement["generation"] == request.generation
        assert acknowledgement["sequence"] == request.sequence
        assert events.index("controller:close") >= events.index(
            "controller:start"
        )
    finally:
        borrower.close()


def test_entitables_remote_borrower_keeps_status_when_shutdown_is_unsafe(
    tmp_path,
    monkeypatch,
):
    scheduler = joinability_dataset.RoutingScheduler(
        tmp_path / "routing.json",
        load_existing=False,
    )
    borrower = joinability_dataset.EntiTablesRemoteGpuBorrower(
        controller_config=SimpleNamespace(request_timeout_seconds=0.1),
        scheduler=scheduler,
        workload=joinability_dataset._RoundRemoteWorkload(
            {"text": 1, "image": 0}
        ),
        coordination_dir=tmp_path / "priority",
        coordination_poll_seconds=0.01,
        drain_timeout_seconds=0.1,
    )
    borrower._heartbeat("draining", None)
    monkeypatch.setattr(borrower, "_stop_controller", lambda: False)
    borrower._stop.set()

    borrower._supervise()

    assert borrower.coordination_paths is not None
    assert borrower.coordination_paths.borrower.exists()
    with pytest.raises(RuntimeError, match="without releasing"):
        borrower.close()


def test_dynamic_vllm_builder_command_enables_text_precompute_and_endpoint_file(tmp_path):
    text_server = VllmServerSpec(
        role="text",
        model_path="/models/text",
        served_model_name="Qwen3.5-9B",
        gpu="1",
        port=8001,
        extra_args=[],
    )
    image_server = VllmServerSpec(
        role="image-primary",
        model_path="/models/vl",
        served_model_name="Qwen3-VL-8B-Thinking",
        gpu="0",
        port=8000,
        extra_args=[],
    )

    command = build_builder_command(
        python_executable="/usr/bin/python",
        builder_script=Path("/repo/scripts/build_mm_joinability_dataset.py"),
        input_dir=Path("/data/input"),
        output_dir=Path("/data/output"),
        text_server=text_server,
        primary_image_server=image_server,
        text_endpoints_file=tmp_path / "text_endpoints.txt",
        image_endpoints_file=tmp_path / "image_endpoints.txt",
        model_start_marker=tmp_path / "model_start.json",
        model_ready_marker=tmp_path / "model_ready.json",
        text_done_marker=tmp_path / "text_done.json",
        image_done_marker=tmp_path / "image_done.json",
        model_round_control_dir=tmp_path / "round-control",
        model_round_run_id="test-run",
        passthrough_args=["--max_source_tables", "10"],
    )

    assert "--precompute_model_cache" in command
    assert "--model_start_marker" in command
    assert "--model_ready_marker" in command
    assert "--model_text_done_marker" in command
    assert "--model_image_done_marker" in command
    assert "--model_round_control_dir" in command
    assert "--text_model_base_urls_file" in command
    assert "--image_model_base_urls_file" in command
    assert "http://127.0.0.1:8001/v1" in command
    assert "http://127.0.0.1:8000/v1" in command
    assert command[-2:] == ["--max_source_tables", "10"]


def test_round_runner_reallocates_text_gpu_then_restores_it_next_round(
    tmp_path, monkeypatch
):
    control_dir = tmp_path / "rounds"
    control_dir.mkdir()
    text_endpoints = tmp_path / "text_endpoints.txt"
    image_endpoints = tmp_path / "image_endpoints.txt"
    text_server = VllmServerSpec("text", "/text", "text", "1", 8001, [])
    primary_image = VllmServerSpec(
        "image-primary", "/image", "image", "0", 8000, []
    )
    secondary_image = VllmServerSpec(
        "image-secondary", "/image", "image", "1", 8002, []
    )
    dynamic_vllm_runner.write_endpoint_file(text_endpoints, [])
    dynamic_vllm_runner.write_endpoint_file(image_endpoints, [])
    events: list[str] = []
    run_id = "test-run"

    class Process:
        def __init__(self, role: str):
            self.role = role

    def start(spec, **_kwargs):
        events.append(f"start:{spec.role}")
        return Process(spec.role)

    def stop(process):
        if process is not None:
            events.append(f"stop:{process.role}")

    monkeypatch.setattr(dynamic_vllm_runner, "start_and_wait_server", start)
    monkeypatch.setattr(dynamic_vllm_runner, "stop_process", stop)
    monkeypatch.setattr(dynamic_vllm_runner.time, "sleep", lambda _seconds: None)

    class Owner:
        def request_gpus(self, *, reason):
            events.append(f"request:{reason}")

        def release_gpus(self, *, reason):
            events.append(f"release:{reason}")

    owner = Owner()

    class Builder:
        def poll(self):
            start0 = dynamic_vllm_runner.round_event_path(
                control_dir, 0, "start"
            )
            ready0 = dynamic_vllm_runner.round_event_path(
                control_dir, 0, "ready"
            )
            text_done0 = dynamic_vllm_runner.round_event_path(
                control_dir, 0, "text.done"
            )
            done0 = dynamic_vllm_runner.round_event_path(control_dir, 0, "done")
            start1 = dynamic_vllm_runner.round_event_path(
                control_dir, 1, "start"
            )
            ready1 = dynamic_vllm_runner.round_event_path(
                control_dir, 1, "ready"
            )
            done1 = dynamic_vllm_runner.round_event_path(control_dir, 1, "done")
            if not start0.exists():
                dynamic_vllm_runner.write_atomic_json(
                    start0,
                    {
                        "status": "model_round_start",
                        "round_id": 0,
                        "run_id": run_id,
                        "text_task_count": 1,
                        "image_task_count": 10,
                    },
                )
            elif ready0.exists() and not text_done0.exists():
                dynamic_vllm_runner.write_atomic_json(
                    text_done0,
                    {
                        "status": "text_round_tasks_completed",
                        "round_id": 0,
                        "run_id": run_id,
                    },
                )
            elif (
                secondary_image.base_url in image_endpoints.read_text()
                and not done0.exists()
            ):
                dynamic_vllm_runner.write_atomic_json(
                    dynamic_vllm_runner.round_event_path(
                        control_dir, 0, "image.done"
                    ),
                    {
                        "status": "image_round_tasks_completed",
                        "round_id": 0,
                        "run_id": run_id,
                    },
                )
                dynamic_vllm_runner.write_atomic_json(
                    done0,
                    {
                        "status": "model_round_completed",
                        "round_id": 0,
                        "run_id": run_id,
                    },
                )
                dynamic_vllm_runner.write_atomic_json(
                    start1,
                    {
                        "status": "model_round_start",
                        "round_id": 1,
                        "run_id": run_id,
                        "text_task_count": 1,
                        "image_task_count": 1,
                    },
                )
            elif ready1.exists() and not done1.exists():
                assert text_endpoints.read_text().strip() == text_server.base_url
                assert image_endpoints.read_text().strip() == primary_image.base_url
                dynamic_vllm_runner.write_atomic_json(
                    done1,
                    {
                        "status": "model_round_completed",
                        "round_id": 1,
                        "run_id": run_id,
                    },
                )
                return 0
            return None

    code = dynamic_vllm_runner.run_round_service_loop(
        builder=Builder(),
        control_dir=control_dir,
        run_id=run_id,
        runtime_dir=tmp_path,
        text_server=text_server,
        primary_image_server=primary_image,
        secondary_image_server=secondary_image,
        text_endpoints_file=text_endpoints,
        image_endpoints_file=image_endpoints,
        text_process=None,
        primary_image_process=None,
        secondary_image_process=None,
        server_start_timeout_seconds=10,
        round_timeout_seconds=10,
        gpu_priority_owner=owner,
        poll_seconds=0,
    )

    assert code == 0
    assert events == [
        "request:model_round_0",
        "start:image-primary",
        "start:text",
        "stop:text",
        "start:image-secondary",
        "stop:image-primary",
        "stop:image-secondary",
        "release:after_model_round_0",
        "request:model_round_1",
        "start:image-primary",
        "start:text",
        "stop:text",
        "stop:image-primary",
        "release:round_service_loop_stopped",
    ]
    assert text_endpoints.read_text() == "\n"
    assert image_endpoints.read_text() == "\n"


def test_round_runner_treats_zero_text_as_done_and_avoids_late_endpoint_publish(
    tmp_path, monkeypatch
):
    control_dir = tmp_path / "rounds"
    control_dir.mkdir()
    run_id = "zero-text-run"
    text_endpoints = tmp_path / "text_endpoints.txt"
    image_endpoints = tmp_path / "image_endpoints.txt"
    text_server = VllmServerSpec("text", "/text", "text", "1", 8001, [])
    primary_image = VllmServerSpec(
        "image-primary", "/image", "image", "0", 8000, []
    )
    secondary_image = VllmServerSpec(
        "image-secondary", "/image", "image", "1", 8002, []
    )
    dynamic_vllm_runner.write_endpoint_file(text_endpoints, [])
    dynamic_vllm_runner.write_endpoint_file(image_endpoints, [])
    events: list[str] = []

    class Process:
        def __init__(self, role):
            self.role = role

    def start(spec, **_kwargs):
        events.append(f"start:{spec.role}")
        dynamic_vllm_runner.write_atomic_json(
            dynamic_vllm_runner.round_event_path(
                control_dir, 0, "image.done"
            ),
            {
                "status": "image_round_tasks_completed",
                "round_id": 0,
                "run_id": run_id,
            },
        )
        dynamic_vllm_runner.write_atomic_json(
            dynamic_vllm_runner.round_event_path(control_dir, 0, "done"),
            {
                "status": "model_round_completed",
                "round_id": 0,
                "run_id": run_id,
            },
        )
        return Process(spec.role)

    monkeypatch.setattr(dynamic_vllm_runner, "start_and_wait_server", start)
    monkeypatch.setattr(
        dynamic_vllm_runner,
        "stop_process",
        lambda process: events.append(f"stop:{process.role}")
        if process is not None
        else None,
    )
    monkeypatch.setattr(dynamic_vllm_runner.time, "sleep", lambda _seconds: None)

    class Builder:
        def poll(self):
            start_path = dynamic_vllm_runner.round_event_path(
                control_dir, 0, "start"
            )
            done_path = dynamic_vllm_runner.round_event_path(
                control_dir, 0, "done"
            )
            if done_path.exists():
                return 0
            if not start_path.exists():
                dynamic_vllm_runner.write_atomic_json(
                    start_path,
                    {
                        "status": "model_round_start",
                        "round_id": 0,
                        "run_id": run_id,
                        "text_task_count": 0,
                        "image_task_count": 5,
                    },
                )
            return None

    code = dynamic_vllm_runner.run_round_service_loop(
        builder=Builder(),
        control_dir=control_dir,
        run_id=run_id,
        runtime_dir=tmp_path,
        text_server=text_server,
        primary_image_server=primary_image,
        secondary_image_server=secondary_image,
        text_endpoints_file=text_endpoints,
        image_endpoints_file=image_endpoints,
        text_process=None,
        primary_image_process=None,
        secondary_image_process=None,
        server_start_timeout_seconds=10,
        round_timeout_seconds=10,
        poll_seconds=0,
    )

    assert code == 0
    assert events == [
        "start:image-primary",
        "start:image-secondary",
        "stop:image-primary",
        "stop:image-secondary",
    ]
    assert image_endpoints.read_text() == "\n"


def test_start_round_services_uses_one_image_server_for_one_task(
    tmp_path, monkeypatch
):
    text_endpoints = tmp_path / "text_endpoints.txt"
    image_endpoints = tmp_path / "image_endpoints.txt"
    text_server = VllmServerSpec("text", "/text", "text", "1", 8001, [])
    primary_image = VllmServerSpec(
        "image-primary", "/image", "image", "0", 8000, []
    )
    secondary_image = VllmServerSpec(
        "image-secondary", "/image", "image", "1", 8002, []
    )
    processes = dynamic_vllm_runner.RoundServiceProcesses(
        text=None,
        primary_image=None,
        secondary_image=None,
    )
    events: list[str] = []

    class Process:
        def __init__(self, role: str):
            self.role = role

    def start(spec, **_kwargs):
        events.append(f"start:{spec.role}")
        return Process(spec.role)

    monkeypatch.setattr(dynamic_vllm_runner, "start_and_wait_server", start)

    dynamic_vllm_runner.start_round_services(
        round_id=0,
        text_task_count=0,
        image_task_count=1,
        runtime_dir=tmp_path,
        text_server=text_server,
        primary_image_server=primary_image,
        secondary_image_server=secondary_image,
        text_endpoints_file=text_endpoints,
        image_endpoints_file=image_endpoints,
        processes=processes,
        server_start_timeout_seconds=10,
        gpu_priority_owner=None,
    )

    assert events == ["start:image-primary"]
    assert processes.primary_image is not None
    assert processes.secondary_image is None
    assert text_endpoints.read_text() == "\n"
    assert image_endpoints.read_text().strip() == primary_image.base_url


@pytest.mark.parametrize("failure_mode", ["endpoint", "timeout"])
def test_round_runner_cleans_secondary_image_owned_when_loop_fails(
    tmp_path, monkeypatch, failure_mode
):
    control_dir = tmp_path / "rounds"
    control_dir.mkdir()
    run_id = f"failure-{failure_mode}"
    text_endpoints = tmp_path / "text_endpoints.txt"
    image_endpoints = tmp_path / "image_endpoints.txt"
    text_server = VllmServerSpec("text", "/text", "text", "1", 8001, [])
    primary_image = VllmServerSpec(
        "image-primary", "/image", "image", "0", 8000, []
    )
    secondary_image = VllmServerSpec(
        "image-secondary", "/image", "image", "1", 8002, []
    )
    dynamic_vllm_runner.write_endpoint_file(text_endpoints, [])
    dynamic_vllm_runner.write_endpoint_file(image_endpoints, [])
    events: list[str] = []
    secondary_started = False

    class Process:
        def __init__(self, role):
            self.role = role

    def start(spec, **_kwargs):
        nonlocal secondary_started
        if spec.role == "image-secondary":
            secondary_started = True
        events.append(f"start:{spec.role}")
        return Process(spec.role)

    original_write_endpoints = dynamic_vllm_runner.write_endpoint_file

    def write_endpoints(path, urls):
        values = list(urls)
        if failure_mode == "endpoint" and secondary_image.base_url in values:
            raise OSError("injected endpoint replace failure")
        original_write_endpoints(path, values)

    clock = 0.0

    def fake_time():
        nonlocal clock
        if secondary_started and failure_mode == "timeout":
            clock += 10.0
        return clock

    monkeypatch.setattr(dynamic_vllm_runner, "start_and_wait_server", start)
    monkeypatch.setattr(dynamic_vllm_runner, "write_endpoint_file", write_endpoints)
    monkeypatch.setattr(
        dynamic_vllm_runner,
        "stop_process",
        lambda process: events.append(f"stop:{process.role}")
        if process is not None
        else None,
    )
    monkeypatch.setattr(dynamic_vllm_runner.time, "time", fake_time)
    monkeypatch.setattr(dynamic_vllm_runner.time, "sleep", lambda _seconds: None)

    class Builder:
        def poll(self):
            start_path = dynamic_vllm_runner.round_event_path(
                control_dir, 0, "start"
            )
            ready_path = dynamic_vllm_runner.round_event_path(
                control_dir, 0, "ready"
            )
            text_done_path = dynamic_vllm_runner.round_event_path(
                control_dir, 0, "text.done"
            )
            if not start_path.exists():
                dynamic_vllm_runner.write_atomic_json(
                    start_path,
                    {
                        "status": "model_round_start",
                        "round_id": 0,
                        "run_id": run_id,
                        "text_task_count": 1,
                        "image_task_count": 5,
                    },
                )
            elif ready_path.exists() and not text_done_path.exists():
                dynamic_vllm_runner.write_atomic_json(
                    text_done_path,
                    {
                        "status": "text_round_tasks_completed",
                        "round_id": 0,
                        "run_id": run_id,
                    },
                )
            return None

    expected_error = OSError if failure_mode == "endpoint" else RuntimeError
    with pytest.raises(expected_error):
        dynamic_vllm_runner.run_round_service_loop(
            builder=Builder(),
            control_dir=control_dir,
            run_id=run_id,
            runtime_dir=tmp_path,
            text_server=text_server,
            primary_image_server=primary_image,
            secondary_image_server=secondary_image,
            text_endpoints_file=text_endpoints,
            image_endpoints_file=image_endpoints,
            text_process=None,
            primary_image_process=None,
            secondary_image_process=None,
            server_start_timeout_seconds=10,
            round_timeout_seconds=5,
            poll_seconds=0,
        )

    assert events == [
        "start:image-primary",
        "start:text",
        "stop:text",
        "start:image-secondary",
        "stop:image-primary",
        "stop:image-secondary",
    ]


def test_round_runner_does_not_release_priority_after_incomplete_gpu_cleanup(
    tmp_path,
    monkeypatch,
):
    text_endpoints = tmp_path / "text_endpoints.txt"
    image_endpoints = tmp_path / "image_endpoints.txt"
    releases = []

    class Process:
        role = "text"

    class Builder:
        def poll(self):
            return 0

    class Owner:
        def release_gpus(self, *, reason):
            releases.append(reason)

    monkeypatch.setattr(
        dynamic_vllm_runner,
        "stop_process",
        lambda _process: (_ for _ in ()).throw(
            RuntimeError("process group remains alive")
        ),
    )

    with pytest.raises(RuntimeError, match="failed to stop every"):
        dynamic_vllm_runner.run_round_service_loop(
            builder=Builder(),
            control_dir=tmp_path / "rounds",
            run_id="cleanup-failure",
            runtime_dir=tmp_path,
            text_server=VllmServerSpec(
                "text", "/text", "text", "1", 8001, []
            ),
            primary_image_server=VllmServerSpec(
                "image-primary", "/image", "image", "0", 8000, []
            ),
            secondary_image_server=VllmServerSpec(
                "image-secondary", "/image", "image", "1", 8002, []
            ),
            text_endpoints_file=text_endpoints,
            image_endpoints_file=image_endpoints,
            text_process=Process(),
            primary_image_process=None,
            secondary_image_process=None,
            server_start_timeout_seconds=10,
            round_timeout_seconds=10,
            gpu_priority_owner=Owner(),
        )

    assert releases == []


def test_dynamic_vllm_main_defers_round_process_ownership_to_loop(
    tmp_path, monkeypatch
):
    stopped: list[str] = []

    class Process:
        def __init__(self, role):
            self.role = role
            self.pid = 12345

        def poll(self):
            return None

        def wait(self, timeout=None):
            return 0

    builder_process = Process("builder")
    def popen(command, **_kwargs):
        marker = Path(command[command.index("--model_start_marker") + 1])
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(
            json.dumps(
                {
                    "round_mode": True,
                    "runner_startup_task_count": 1,
                }
            ),
            encoding="utf-8",
        )
        return builder_process

    def start(spec, **_kwargs):
        pytest.fail(f"main must defer {spec.role} startup to the round loop")

    def fail_round_loop(**kwargs):
        assert kwargs["text_process"] is None
        assert kwargs["primary_image_process"] is None
        assert kwargs["secondary_image_process"] is None
        raise RuntimeError("injected round loop failure")

    monkeypatch.setattr(dynamic_vllm_runner.subprocess, "Popen", popen)
    monkeypatch.setattr(dynamic_vllm_runner, "start_and_wait_server", start)
    monkeypatch.setattr(dynamic_vllm_runner, "run_round_service_loop", fail_round_loop)
    monkeypatch.setattr(
        dynamic_vllm_runner,
        "stop_process",
        lambda process: stopped.append(process.role) if process is not None else None,
    )

    with pytest.raises(RuntimeError, match="injected round loop failure"):
        dynamic_vllm_main(
            [
                "--input_dir",
                str(tmp_path / "input"),
                "--output_dir",
                str(tmp_path / "output"),
                "--text_model_path",
                "/models/text",
                "--image_model_path",
                "/models/image",
            ]
        )

    assert stopped.count("builder") == 1


def test_dynamic_vllm_defaults_limit_startup_kv_cache_memory():
    args = argparse.Namespace(
        vllm_dtype="bfloat16",
        vllm_max_model_len=8192,
        vllm_gpu_memory_utilization=0.9,
        vllm_max_num_batched_tokens=1024,
        vllm_max_num_seqs=8,
        vllm_mm_processor_cache_gb=0,
        no_default_vllm_memory_args=False,
    )

    assert default_vllm_extra_args(args) == [
        "--trust-remote-code",
        "--dtype",
        "bfloat16",
        "--max-model-len",
        "8192",
        "--gpu-memory-utilization",
        "0.9",
        "--enforce-eager",
        "--skip-mm-profiling",
        "--mm-processor-cache-gb",
        "0",
        "--max-num-batched-tokens",
        "1024",
        "--max-num-seqs",
        "8",
    ]


def test_dynamic_vllm_unified_model_has_role_specific_local_profiles():
    args, passthrough = parse_dynamic_vllm_args(
        [
            "--input_dir",
            "/data/input",
            "--output_dir",
            "/data/output",
            "--model_path",
            "/models/Qwen3.5-9B",
            "--model_name",
            "Qwen3.5-9B",
            "--text_language_model_only",
            "--text_vllm_gpu_memory_utilization",
            "0.9",
            "--image_vllm_gpu_memory_utilization",
            "0.92",
            "--text_vllm_max_num_seqs",
            "12",
            "--image_vllm_max_num_seqs",
            "4",
        ]
    )

    assert passthrough == []
    assert args.text_model_path == "/models/Qwen3.5-9B"
    assert args.image_model_path == "/models/Qwen3.5-9B"
    assert args.text_model_name == args.image_model_name == "Qwen3.5-9B"
    text_args = default_vllm_extra_args(args, "text")
    image_args = default_vllm_extra_args(args, "image")
    assert text_args[text_args.index("--gpu-memory-utilization") + 1] == "0.9"
    assert image_args[image_args.index("--gpu-memory-utilization") + 1] == "0.92"
    assert text_args[text_args.index("--max-num-seqs") + 1] == "12"
    assert image_args[image_args.index("--max-num-seqs") + 1] == "4"
    assert "--language-model-only" in text_args
    assert "--language-model-only" not in image_args


def test_dynamic_vllm_rejects_thinking_for_either_modality(capsys):
    with pytest.raises(SystemExit):
        parse_dynamic_vllm_args(
            [
                "--input_dir",
                "/data/input",
                "--output_dir",
                "/data/output",
                "--model_path",
                "/models/Qwen3.5-9B",
                "--enable_thinking",
            ]
        )

    assert "thinking mode is always disabled" in capsys.readouterr().err


def test_dynamic_vllm_default_context_length_is_larger_for_vl_images():
    args, _passthrough = parse_dynamic_vllm_args(
        [
            "--input_dir",
            "/data/input",
            "--output_dir",
            "/data/output",
            "--text_model_path",
            "/models/text",
            "--image_model_path",
            "/models/vl",
        ]
    )

    assert args.vllm_max_model_len == 8192
    assert args.model_start_timeout_seconds is None
    assert args.forwarded_signal_grace_seconds == 30.0
    assert args.server_start_attempts == 5
    assert args.server_start_retry_backoff_seconds == 15.0
    assert args.cuda_readiness_timeout_seconds == 30.0


def test_dynamic_vllm_remote_worker_counts_are_modality_specific():
    args, passthrough = parse_dynamic_vllm_args(
        [
            "--input_dir",
            "/data/input",
            "--output_dir",
            "/data/output",
            "--text_model_path",
            "/models/text",
            "--image_model_path",
            "/models/vl",
            "--remote_layout_control_url",
            "http://127.0.0.1:18999",
            "--remote_layout_control_token_file",
            "/run/layout-token",
            "--remote_layout_primary_image_url",
            "http://127.0.0.1:18000/v1",
            "--remote_layout_switchable_url",
            "http://127.0.0.1:18001/v1",
            "--remote_layout_coordination_dir",
            "/run/remote-priority",
            "--remote_text_model_workers",
            "32",
            "--remote_image_model_workers",
            "64",
        ]
    )

    assert args.remote_text_model_workers == 32
    assert args.remote_image_model_workers == 64
    assert passthrough == []
    assert dynamic_vllm_runner.with_remote_model_workers(
        [],
        text_workers=args.remote_text_model_workers,
        image_workers=args.remote_image_model_workers,
    ) == [
        "--remote_text_model_workers",
        "32",
        "--remote_image_model_workers",
        "64",
    ]


def test_dynamic_vllm_static_remote_workers_are_forwarded_without_layout():
    args, passthrough = parse_dynamic_vllm_args(
        [
            "--input_dir",
            "/data/input",
            "--output_dir",
            "/data/output",
            "--text_model_path",
            "/models/text",
            "--image_model_path",
            "/models/vl",
            "--remote_image_model_base_url",
            "http://127.0.0.1:18000/v1",
            "--remote_image_model_workers",
            "32",
        ]
    )

    assert args.remote_layout_control_url is None
    assert args.remote_image_model_workers == 32
    assert "--remote_image_model_base_url" in passthrough
    forwarded = dynamic_vllm_runner.with_remote_model_workers(
        passthrough,
        text_workers=args.remote_text_model_workers,
        image_workers=args.remote_image_model_workers,
    )
    assert forwarded[-4:] == [
        "--remote_text_model_workers",
        "0",
        "--remote_image_model_workers",
        "32",
    ]


def test_dynamic_vllm_local_worker_counts_are_modality_specific():
    args, passthrough = parse_dynamic_vllm_args(
        [
            "--input_dir",
            "/data/input",
            "--output_dir",
            "/data/output",
            "--text_model_path",
            "/models/text",
            "--image_model_path",
            "/models/vl",
            "--text_model_workers",
            "4",
            "--image_model_workers",
            "8",
        ]
    )

    assert args.text_model_workers == 4
    assert args.image_model_workers == 8
    assert passthrough == []
    assert dynamic_vllm_runner.with_model_workers(
        [],
        text_workers=args.text_model_workers,
        image_workers=args.image_model_workers,
    ) == [
        "--text_model_workers",
        "4",
        "--image_model_workers",
        "8",
    ]


@pytest.mark.parametrize(
    ("combined_option", "replacement"),
    [
        ("--dynamic_model_workers", "--text_model_workers"),
        ("--remote_dynamic_model_workers", "--remote_text_model_workers"),
    ],
)
def test_dynamic_vllm_rejects_combined_worker_counts(
    combined_option,
    replacement,
    capsys,
):
    with pytest.raises(SystemExit):
        parse_dynamic_vllm_args(
            [
                "--input_dir",
                "/data/input",
                "--output_dir",
                "/data/output",
                "--text_model_path",
                "/models/text",
                "--image_model_path",
                "/models/vl",
                combined_option,
                "2",
            ]
        )

    assert replacement in capsys.readouterr().err


def test_dynamic_vllm_forwards_signal_to_every_live_process_group(
    monkeypatch,
):
    class FakeProcess:
        def __init__(self, pid, return_code):
            self.pid = pid
            self.return_code = return_code

        def poll(self):
            return self.return_code

    forwarded = []
    monkeypatch.setattr(
        dynamic_vllm_runner.os,
        "killpg",
        lambda pid, signum: forwarded.append((pid, signum)),
    )

    dynamic_vllm_runner.forward_signal_to_live_process_groups(
        signal.SIGTERM,
        [
            FakeProcess(101, None),
            FakeProcess(102, 0),
            None,
            FakeProcess(103, None),
        ],
    )

    assert forwarded == [
        (101, signal.SIGTERM),
        (103, signal.SIGTERM),
    ]


def test_dynamic_vllm_forwarded_signal_grace_skips_absent_or_exited_process():
    class ExitedProcess:
        def poll(self):
            return 0

        def wait(self, **_kwargs):
            pytest.fail("already-exited processes must not be waited again")

    assert dynamic_vllm_runner.wait_for_forwarded_process_exit(
        None,
        timeout_seconds=30.0,
    )
    assert dynamic_vllm_runner.wait_for_forwarded_process_exit(
        ExitedProcess(),
        timeout_seconds=30.0,
    )


def test_dynamic_vllm_cleanup_continues_after_one_stop_failure(monkeypatch):
    processes = [object(), object(), object()]
    attempted = []

    def fake_stop(process):
        attempted.append(process)
        if process is processes[0]:
            raise RuntimeError("first process would not stop")

    monkeypatch.setattr(dynamic_vllm_runner, "stop_process", fake_stop)

    errors = dynamic_vllm_runner.stop_processes_best_effort(processes)

    assert attempted == processes
    assert len(errors) == 1
    assert errors[0][0] is processes[0]
    assert str(errors[0][1]) == "first process would not stop"


def test_start_server_persists_vllm_output_and_closes_parent_handle(monkeypatch, tmp_path):
    captured = {}

    class FakePopen:
        def __init__(self, command, **kwargs):
            captured["command"] = command
            captured["env"] = kwargs["env"]
            captured["stdout"] = kwargs["stdout"]
            captured["stderr"] = kwargs["stderr"]
            captured["text"] = kwargs["text"]
            captured["start_new_session"] = kwargs["start_new_session"]

    monkeypatch.setenv("VLLM_API_KEY", "remote-only-secret")
    monkeypatch.setenv("MMDD_REMOTE_IMAGE_MODEL_API_KEY", "remote-image-secret")
    monkeypatch.setattr("run_mm_joinability_dynamic_vllm.subprocess.Popen", FakePopen)
    spec = VllmServerSpec(
        role="text",
        model_path="/models/text",
        served_model_name="Qwen3.5-9B",
        gpu="1",
        port=8001,
        extra_args=[],
    )

    log_path = tmp_path / "runtime" / "text.log"
    start_server(spec, log_path=log_path)

    assert captured["command"] == spec.command()
    assert "VLLM_API_KEY" not in captured["env"]
    assert captured["env"]["MMDD_REMOTE_IMAGE_MODEL_API_KEY"] == (
        "remote-image-secret"
    )
    assert Path(captured["stdout"].name) == log_path
    assert captured["stdout"].closed
    assert captured["stderr"] == subprocess.STDOUT
    assert captured["text"] is True
    assert captured["start_new_session"] is True


def test_wait_for_server_fails_immediately_when_process_exits(monkeypatch, tmp_path):
    log_path = tmp_path / "image-primary.log"
    log_path.write_text("first line\nfatal: CUDA initialization failed\n", encoding="utf-8")

    class ExitedProcess:
        def poll(self):
            return 7

    monkeypatch.setattr(
        "run_mm_joinability_dynamic_vllm.requests.get",
        lambda *_args, **_kwargs: pytest.fail("health endpoint must not be polled after process exit"),
    )

    with pytest.raises(RuntimeError) as exc_info:
        wait_for_server(
            "http://127.0.0.1:8000/v1",
            process=ExitedProcess(),
            role="image-primary",
            log_path=log_path,
            expected_model_name="Qwen3-VL",
            timeout_seconds=900,
            poll_seconds=0,
        )

    message = str(exc_info.value)
    assert "image-primary" in message
    assert "exit code 7" in message
    assert str(log_path) in message
    assert "fatal: CUDA initialization failed" in message


def test_start_and_wait_server_retries_one_early_exit(monkeypatch, tmp_path):
    processes = [object(), object()]
    started_log_paths = []
    waited = []
    stopped = []

    def fake_start_server(_spec, *, log_path):
        started_log_paths.append(log_path)
        return processes[len(started_log_paths) - 1]

    def fake_wait_for_server(_base_url, *, process, log_path, **_kwargs):
        waited.append((process, log_path))
        if process is processes[0]:
            raise dynamic_vllm_runner.ServerExitedBeforeHealthy(
                "first attempt exited"
            )

    monkeypatch.setattr(dynamic_vllm_runner, "start_server", fake_start_server)
    monkeypatch.setattr(dynamic_vllm_runner, "wait_for_server", fake_wait_for_server)
    monkeypatch.setattr(
        dynamic_vllm_runner, "stop_process", lambda process: stopped.append(process)
    )
    spec = VllmServerSpec(
        role="image-primary",
        model_path="/models/vl",
        served_model_name="Qwen3-VL",
        gpu="0",
        port=8000,
        extra_args=[],
    )

    process = dynamic_vllm_runner.start_and_wait_server(
        spec,
        runtime_dir=tmp_path,
        timeout_seconds=900,
    )

    assert process is processes[1]
    assert started_log_paths == [
        tmp_path / "image-primary.attempt-1.log",
        tmp_path / "image-primary.attempt-2.log",
    ]
    assert waited == list(zip(processes, started_log_paths))
    assert stopped == [processes[0]]


def test_start_and_wait_server_waits_for_cuda_readiness_to_recover(
    monkeypatch, tmp_path
):
    process = object()
    readiness_results = iter([False, True])
    readiness_logs = []
    server_logs = []
    sleeps = []

    def probe(_gpu, *, log_path, timeout_seconds):
        readiness_logs.append((log_path, timeout_seconds))
        return next(readiness_results)

    monkeypatch.setattr(dynamic_vllm_runner, "run_cuda_readiness_probe", probe)
    monkeypatch.setattr(
        dynamic_vllm_runner,
        "start_server",
        lambda _spec, *, log_path: server_logs.append(log_path) or process,
    )
    monkeypatch.setattr(
        dynamic_vllm_runner, "wait_for_server", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(dynamic_vllm_runner.time, "sleep", sleeps.append)
    spec = VllmServerSpec(
        role="image-primary",
        model_path="/models/vl",
        served_model_name="Qwen3-VL",
        gpu="0",
        port=8000,
        extra_args=[],
        startup_attempts=3,
        startup_retry_backoff_seconds=4,
        cuda_readiness_timeout_seconds=30,
    )

    assert (
        dynamic_vllm_runner.start_and_wait_server(
            spec,
            runtime_dir=tmp_path,
            timeout_seconds=900,
        )
        is process
    )
    assert readiness_logs == [
        (tmp_path / "image-primary.cuda-readiness.attempt-1.log", 30),
        (tmp_path / "image-primary.cuda-readiness.attempt-2.log", 30),
    ]
    assert server_logs == [tmp_path / "image-primary.attempt-2.log"]
    assert sleeps == [4]


def test_start_and_wait_server_uses_linear_backoff_for_early_exits(
    monkeypatch, tmp_path
):
    processes = [object(), object(), object()]
    starts = []
    stopped = []
    sleeps = []

    def start(_spec, *, log_path):
        starts.append(log_path)
        return processes[len(starts) - 1]

    def wait(_base_url, *, process, **_kwargs):
        if process is not processes[-1]:
            raise dynamic_vllm_runner.ServerExitedBeforeHealthy("early exit")

    monkeypatch.setattr(dynamic_vllm_runner, "start_server", start)
    monkeypatch.setattr(dynamic_vllm_runner, "wait_for_server", wait)
    monkeypatch.setattr(
        dynamic_vllm_runner, "stop_process", lambda process: stopped.append(process)
    )
    monkeypatch.setattr(dynamic_vllm_runner.time, "sleep", sleeps.append)
    spec = VllmServerSpec(
        role="text",
        model_path="/models/text",
        served_model_name="Qwen3.5-9B",
        gpu="1",
        port=8001,
        extra_args=[],
        startup_attempts=3,
        startup_retry_backoff_seconds=2,
    )

    assert (
        dynamic_vllm_runner.start_and_wait_server(
            spec,
            runtime_dir=tmp_path,
            timeout_seconds=900,
        )
        is processes[-1]
    )
    assert starts == [
        tmp_path / "text.attempt-1.log",
        tmp_path / "text.attempt-2.log",
        tmp_path / "text.attempt-3.log",
    ]
    assert stopped == processes[:-1]
    assert sleeps == [2, 4]


def test_cuda_readiness_probe_uses_fresh_target_gpu_process(
    monkeypatch, tmp_path
):
    captured = {}

    class Result:
        returncode = 0

    def run(command, **kwargs):
        captured["command"] = command
        captured.update(kwargs)
        kwargs["stdout"].write("NVIDIA GeForce RTX 4090\n")
        return Result()

    monkeypatch.setattr(dynamic_vllm_runner.subprocess, "run", run)
    log_path = tmp_path / "cuda-readiness.log"

    assert dynamic_vllm_runner.run_cuda_readiness_probe(
        "1",
        log_path=log_path,
        timeout_seconds=30,
    )
    assert captured["command"][0] == sys.executable
    assert "torch.cuda.init()" in captured["command"][2]
    assert captured["env"]["CUDA_VISIBLE_DEVICES"] == "1"
    assert captured["start_new_session"] is True
    assert captured["timeout"] == 30
    assert log_path.read_text(encoding="utf-8") == "NVIDIA GeForce RTX 4090\n"


def test_start_and_wait_server_cleans_second_failed_attempt(monkeypatch, tmp_path):
    processes = [object(), object()]
    stopped = []

    monkeypatch.setattr(
        dynamic_vllm_runner,
        "start_server",
        lambda _spec, *, log_path: processes[0]
        if log_path.name.endswith("attempt-1.log")
        else processes[1],
    )
    monkeypatch.setattr(
        dynamic_vllm_runner,
        "wait_for_server",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            dynamic_vllm_runner.ServerExitedBeforeHealthy("exited")
        ),
    )
    monkeypatch.setattr(
        dynamic_vllm_runner, "stop_process", lambda process: stopped.append(process)
    )
    spec = VllmServerSpec(
        role="image-primary",
        model_path="/models/vl",
        served_model_name="Qwen3-VL",
        gpu="0",
        port=8000,
        extra_args=[],
    )

    with pytest.raises(dynamic_vllm_runner.ServerExitedBeforeHealthy):
        dynamic_vllm_runner.start_and_wait_server(
            spec, runtime_dir=tmp_path, timeout_seconds=900
        )

    assert stopped == processes


def test_wait_for_server_checks_process_once_more_at_deadline(tmp_path):
    log_path = tmp_path / "text.log"
    log_path.write_text("leader exited", encoding="utf-8")

    class ExitedProcess:
        def poll(self):
            return 11

    with pytest.raises(
        dynamic_vllm_runner.ServerExitedBeforeHealthy, match="exit code 11"
    ):
        wait_for_server(
            "http://127.0.0.1:8001/v1",
            process=ExitedProcess(),
            role="text",
            log_path=log_path,
            expected_model_name="Qwen3.5-9B",
            timeout_seconds=0,
            poll_seconds=0,
        )


def test_wait_for_server_rechecks_process_after_http_success(monkeypatch, tmp_path):
    log_path = tmp_path / "text.log"
    log_path.write_text("exited after response", encoding="utf-8")
    polls = iter([None, 12])

    class Process:
        def poll(self):
            return next(polls)

    class Response:
        status_code = 200

        def json(self):
            return {"data": [{"id": "Qwen3.5-9B"}]}

    monkeypatch.setattr(dynamic_vllm_runner.requests, "get", lambda *_a, **_k: Response())

    with pytest.raises(
        dynamic_vllm_runner.ServerExitedBeforeHealthy, match="exit code 12"
    ):
        wait_for_server(
            "http://127.0.0.1:8001/v1",
            process=Process(),
            role="text",
            log_path=log_path,
            expected_model_name="Qwen3.5-9B",
            timeout_seconds=10,
            poll_seconds=0,
        )


@pytest.mark.parametrize(
    ("status_code", "model_id"),
    [(204, "Qwen3.5-9B"), (200, "stale-model")],
)
def test_wait_for_server_rejects_non_200_or_wrong_model(
    monkeypatch, tmp_path, status_code, model_id
):
    times = iter([0.0, 0.0, 2.0])

    class Process:
        def poll(self):
            return None

    class Response:
        def __init__(self):
            self.status_code = status_code

        def json(self):
            return {"data": [{"id": model_id}]}

    monkeypatch.setattr(dynamic_vllm_runner.time, "time", lambda: next(times))
    monkeypatch.setattr(dynamic_vllm_runner.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(dynamic_vllm_runner.requests, "get", lambda *_a, **_k: Response())

    with pytest.raises(RuntimeError, match="Timed out waiting"):
        wait_for_server(
            "http://127.0.0.1:8001/v1",
            process=Process(),
            role="text",
            log_path=tmp_path / "text.log",
            expected_model_name="Qwen3.5-9B",
            timeout_seconds=1,
            poll_seconds=0,
        )


def test_stop_process_cleans_group_when_leader_already_exited(monkeypatch):
    signals = []
    waits = []
    group_alive = True

    class Process:
        pid = 123

        def poll(self):
            return 7

        def wait(self, timeout=None):
            waits.append(timeout)
            return 7

    def kill_group(pid, sig):
        nonlocal group_alive
        if sig == 0:
            if group_alive:
                return
            raise ProcessLookupError()
        signals.append((pid, sig))
        if sig == signal.SIGKILL:
            group_alive = False

    monkeypatch.setattr(dynamic_vllm_runner.os, "killpg", kill_group)

    dynamic_vllm_runner.stop_process(Process(), timeout_seconds=0.01)

    assert signals == [(123, signal.SIGTERM), (123, signal.SIGKILL)]
    assert waits


def test_stop_process_reaps_leader_after_killpg_race(monkeypatch):
    waits = []

    class Process:
        pid = 456

        def poll(self):
            return None

        def wait(self, timeout=None):
            waits.append(timeout)
            return 0

    monkeypatch.setattr(
        dynamic_vllm_runner.os,
        "killpg",
        lambda _pid, _sig: (_ for _ in ()).throw(ProcessLookupError()),
    )

    dynamic_vllm_runner.stop_process(Process(), timeout_seconds=0.01)

    assert waits == [0.01]


def test_cleanup_processes_continues_after_one_stop_failure(monkeypatch):
    processes = [object(), object(), object()]
    stopped = []

    def fake_stop(process):
        stopped.append(process)
        if process is processes[0]:
            raise RuntimeError("first cleanup failed")

    monkeypatch.setattr(dynamic_vllm_runner, "stop_process", fake_stop)

    dynamic_vllm_runner.cleanup_processes(processes)

    assert stopped == processes


def test_start_and_wait_server_does_not_retry_health_timeout(monkeypatch, tmp_path):
    starts = []
    stopped = []
    process = object()

    def fake_start_server(_spec, *, log_path):
        starts.append(log_path)
        return process

    monkeypatch.setattr(dynamic_vllm_runner, "start_server", fake_start_server)
    monkeypatch.setattr(
        dynamic_vllm_runner,
        "wait_for_server",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("Timed out waiting for health endpoint")
        ),
    )
    monkeypatch.setattr(
        dynamic_vllm_runner, "stop_process", lambda proc: stopped.append(proc)
    )
    spec = VllmServerSpec(
        role="text",
        model_path="/models/text",
        served_model_name="Qwen3.5-9B",
        gpu="1",
        port=8001,
        extra_args=[],
    )

    with pytest.raises(RuntimeError, match="Timed out waiting"):
        dynamic_vllm_runner.start_and_wait_server(
            spec,
            runtime_dir=tmp_path,
            timeout_seconds=0,
        )

    assert starts == [tmp_path / "text.attempt-1.log"]
    assert stopped == [process]


@pytest.mark.parametrize("interruption", [KeyboardInterrupt(), SystemExit(23)])
def test_start_and_wait_server_cleans_process_and_reraises_base_exception(
    monkeypatch, tmp_path, interruption
):
    process = object()
    starts = []
    stopped = []

    def fake_start_server(_spec, *, log_path):
        starts.append(log_path)
        return process

    monkeypatch.setattr(dynamic_vllm_runner, "start_server", fake_start_server)
    monkeypatch.setattr(
        dynamic_vllm_runner,
        "wait_for_server",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(interruption),
    )
    monkeypatch.setattr(
        dynamic_vllm_runner, "stop_process", lambda proc: stopped.append(proc)
    )
    spec = VllmServerSpec(
        role="text",
        model_path="/models/text",
        served_model_name="Qwen3.5-9B",
        gpu="1",
        port=8001,
        extra_args=[],
    )

    with pytest.raises(type(interruption)) as exc_info:
        dynamic_vllm_runner.start_and_wait_server(
            spec, runtime_dir=tmp_path, timeout_seconds=900
        )

    assert exc_info.value is interruption
    assert starts == [tmp_path / "text.attempt-1.log"]
    assert stopped == [process]


def test_dynamic_vllm_delays_server_start_until_builder_requests_models(monkeypatch, tmp_path):
    events = []

    class FakePopen:
        def __init__(self, command, **kwargs):
            self.command = command
            self.pid = 12345
            self._poll = None
            if command[0] == "/usr/bin/python":
                events.append("builder_started")
                marker = Path(command[command.index("--model_start_marker") + 1])
                marker.parent.mkdir(parents=True, exist_ok=True)
                marker.write_text("{}", encoding="utf-8")
                Path(command[command.index("--model_text_done_marker") + 1]).write_text("{}", encoding="utf-8")
                Path(command[command.index("--model_image_done_marker") + 1]).write_text("{}", encoding="utf-8")
            else:
                events.append(f"server_started:{command[command.index('--served-model-name') + 1]}")

        def poll(self):
            return self._poll

        def wait(self, timeout=None):
            self._poll = 0
            return 0

    monkeypatch.setattr("run_mm_joinability_dynamic_vllm.subprocess.Popen", FakePopen)
    monkeypatch.setattr("run_mm_joinability_dynamic_vllm.wait_for_server", lambda *args, **kwargs: events.append("server_ready"))
    monkeypatch.setattr(
        "run_mm_joinability_dynamic_vllm.run_cuda_readiness_probe",
        lambda *_args, **_kwargs: True,
    )

    code = dynamic_vllm_main(
        [
            "--input_dir",
            str(tmp_path / "input"),
            "--output_dir",
            str(tmp_path / "output"),
            "--text_model_path",
            "/models/text",
            "--image_model_path",
            "/models/vl",
            "--python_executable",
            "/usr/bin/python",
        ]
    )

    assert code == 0
    assert events[:5] == [
        "builder_started",
        "server_started:Qwen3.5-9B",
        "server_ready",
        "server_started:Qwen3-VL-8B-Instruct",
        "server_ready",
    ]


def test_dynamic_vllm_skips_server_start_when_builder_has_no_pending_model_tasks(monkeypatch, tmp_path):
    events = []

    class FakePopen:
        def __init__(self, command, **kwargs):
            self.command = command
            self.pid = 12345
            self._poll = None
            if command[0] == "/usr/bin/python":
                events.append("builder_started")
                marker = Path(command[command.index("--model_start_marker") + 1])
                marker.parent.mkdir(parents=True, exist_ok=True)
                marker.write_text(
                    '{"text_task_count": 0, "image_task_count": 0}',
                    encoding="utf-8",
                )
            else:
                events.append("server_started")

        def poll(self):
            return self._poll

        def wait(self, timeout=None):
            self._poll = 0
            return 0

    monkeypatch.setattr("run_mm_joinability_dynamic_vllm.subprocess.Popen", FakePopen)

    code = dynamic_vllm_main(
        [
            "--input_dir",
            str(tmp_path / "input"),
            "--output_dir",
            str(tmp_path / "output"),
            "--text_model_path",
            "/models/text",
            "--image_model_path",
            "/models/vl",
            "--python_executable",
            "/usr/bin/python",
        ]
    )

    assert code == 0
    assert events == ["builder_started"]


def test_dynamic_vllm_defers_round_mode_servers_to_round_handshake(
    monkeypatch, tmp_path
):
    events = []
    round_loops = []

    class FakePopen:
        def __init__(self, command, **kwargs):
            self.command = command
            self.pid = 12345
            self._poll = None
            if command[0] == "/usr/bin/python":
                events.append("builder_started")
                marker = Path(command[command.index("--model_start_marker") + 1])
                marker.parent.mkdir(parents=True, exist_ok=True)
                marker.write_text(
                    json.dumps(
                        {
                            "text_task_count": 0,
                            "image_task_count": 0,
                            "round_mode": True,
                            "runner_startup_task_count": 1,
                        }
                    ),
                    encoding="utf-8",
                )
            else:
                events.append("server_started")

        def poll(self):
            return self._poll

        def wait(self, timeout=None):
            self._poll = 0
            return 0

    monkeypatch.setattr("run_mm_joinability_dynamic_vllm.subprocess.Popen", FakePopen)
    monkeypatch.setattr("run_mm_joinability_dynamic_vllm.wait_for_server", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        "run_mm_joinability_dynamic_vllm.run_round_service_loop",
        lambda **kwargs: round_loops.append(kwargs) or 0,
    )

    code = dynamic_vllm_main(
        [
            "--input_dir",
            str(tmp_path / "input"),
            "--output_dir",
            str(tmp_path / "output"),
            "--text_model_path",
            "/models/text",
            "--image_model_path",
            "/models/vl",
            "--python_executable",
            "/usr/bin/python",
        ]
    )

    assert code == 0
    assert events == ["builder_started"]
    assert len(round_loops) == 1
    assert round_loops[0]["text_process"] is None
    assert round_loops[0]["primary_image_process"] is None


def test_dynamic_vllm_skips_servers_for_zero_runner_startup_count_in_round_mode(
    monkeypatch, tmp_path
):
    events = []

    class FakePopen:
        def __init__(self, command, **kwargs):
            self.command = command
            self.pid = 12345
            self._poll = None
            if command[0] == "/usr/bin/python":
                events.append("builder_started")
                self.ready_marker = Path(
                    command[command.index("--model_ready_marker") + 1]
                )
                marker = Path(command[command.index("--model_start_marker") + 1])
                marker.parent.mkdir(parents=True, exist_ok=True)
                marker.write_text(
                    json.dumps(
                        {
                            "text_task_count": 0,
                            "image_task_count": 0,
                            "round_mode": True,
                            "runner_startup_task_count": 0,
                        }
                    ),
                    encoding="utf-8",
                )
            else:
                events.append("server_started")

        def poll(self):
            return self._poll

        def wait(self, timeout=None):
            assert self.ready_marker.exists(), "builder remains blocked without ready marker"
            self._poll = 0
            return 0

    monkeypatch.setattr("run_mm_joinability_dynamic_vllm.subprocess.Popen", FakePopen)
    monkeypatch.setattr(
        "run_mm_joinability_dynamic_vllm.start_server",
        lambda _server: pytest.fail("zero-target round must not start model servers"),
    )

    code = dynamic_vllm_main(
        [
            "--input_dir",
            str(tmp_path / "input"),
            "--output_dir",
            str(tmp_path / "output"),
            "--text_model_path",
            "/models/text",
            "--image_model_path",
            "/models/vl",
            "--python_executable",
            "/usr/bin/python",
        ]
    )

    assert code == 0
    assert events == ["builder_started"]


def test_model_start_marker_can_signal_round_mode(tmp_path):
    marker = tmp_path / "model_start.json"

    joinability_dataset.write_model_start_marker(
        str(marker), text_task_count=0, image_task_count=0, round_mode=True
    )

    assert '"round_mode": true' in marker.read_text(encoding="utf-8")


def test_round_mode_start_marker_requests_services_without_inflating_actual_counts(
    tmp_path,
):
    marker = tmp_path / "model_start.json"

    joinability_dataset.write_model_start_marker(
        str(marker),
        text_task_count=0,
        image_task_count=0,
        round_mode=True,
        round_mode_requires_services=True,
    )

    payload = json.loads(marker.read_text(encoding="utf-8"))
    assert payload["runner_startup_task_count"] > 0
    assert payload["text_task_count"] == 0
    assert payload["image_task_count"] == 0
    assert read_pending_model_task_count(marker) == payload["runner_startup_task_count"]


def test_candidate_rounds_defer_done_markers_until_accumulated_work_finishes(
    tmp_path, monkeypatch
):
    class FakeExtractor:
        def extract(self, asset, entity, candidate_attribute_names):
            return {
                "attributes": [],
                "raw_response": '{"attributes":[]}',
                "error": "",
            }

    args = _parallel_args(
        model_text_done_marker=str(tmp_path / "text_done.json"),
        model_image_done_marker=str(tmp_path / "image_done.json"),
    )
    cache = ExtractionCache(tmp_path / "model_cache.jsonl")
    state = ModelConcurrencyState(text_workers=1, image_workers=1)
    accumulated = {"text": 0, "image": 0}
    marker_writes = []
    write_model_done_marker = joinability_dataset.write_model_done_marker

    def record_done_marker(path_value, *, model_kind, task_count):
        marker_writes.append((model_kind, task_count))
        write_model_done_marker(
            path_value, model_kind=model_kind, task_count=task_count
        )

    monkeypatch.setattr(
        joinability_dataset, "write_model_done_marker", record_done_marker
    )

    for round_index in range(2):
        counts = precompute_extraction_task_groups(
            extractor=FakeExtractor(),
            cache=cache,
            tasks_by_kind={
                "text": [_task("text", f"round_{round_index}")],
                "image": [_task("image", f"round_{round_index}")],
            },
            args=args,
            state=state,
            progress=None,
            write_done_markers=False,
        )
        for kind in accumulated:
            accumulated[kind] += counts[kind]
        assert marker_writes == []
        assert not (tmp_path / "text_done.json").exists()
        assert not (tmp_path / "image_done.json").exists()

    joinability_dataset.write_done_markers_after_selection(
        args,
        text_task_count=accumulated["text"],
        image_task_count=accumulated["image"],
    )

    assert marker_writes == [("text", 2), ("image", 2)]
    text_payload = json.loads((tmp_path / "text_done.json").read_text(encoding="utf-8"))
    image_payload = json.loads((tmp_path / "image_done.json").read_text(encoding="utf-8"))
    assert text_payload["task_count"] == 2
    assert image_payload["task_count"] == 2


def test_precompute_task_groups_write_each_modality_done_marker_independently(tmp_path):
    release_text = threading.Event()
    text_started = threading.Event()

    class FakeExtractor:
        def extract(self, asset, entity, candidate_attribute_names):
            if asset["asset_type"] == "text":
                text_started.set()
                release_text.wait(timeout=2.0)
            attrs = [{"name": "State", "value": "Alabama", "evidence": asset["asset_type"]}]
            return {"attributes": attrs, "raw_response": '{"attributes":[]}', "error": ""}

    args = _parallel_args(
        model_text_done_marker=str(tmp_path / "text_done.json"),
        model_image_done_marker=str(tmp_path / "image_done.json"),
        reparse_cached_model_outputs=True,
    )
    cache = ExtractionCache(tmp_path / "model_cache.jsonl")
    state = ModelConcurrencyState(text_workers=1, image_workers=1)
    tasks_by_kind = {
        "text": [_task("text", "1")],
        "image": [_task("image", "2")],
    }

    worker = threading.Thread(
        target=precompute_extraction_task_groups,
        kwargs={
            "extractor": FakeExtractor(),
            "cache": cache,
            "tasks_by_kind": tasks_by_kind,
            "args": args,
            "state": state,
            "progress": None,
        },
    )
    worker.start()
    assert text_started.wait(timeout=2.0)

    image_done = tmp_path / "image_done.json"
    for _ in range(20):
        if image_done.exists():
            break
        threading.Event().wait(0.05)

    assert image_done.exists()
    assert not (tmp_path / "text_done.json").exists()

    release_text.set()
    worker.join(timeout=2.0)

    assert not worker.is_alive()
    assert (tmp_path / "text_done.json").exists()


def test_precompute_task_groups_use_generation_scoped_round_handshake(
    tmp_path, monkeypatch
):
    control_dir = tmp_path / "round-control"
    args = _parallel_args()
    args.model_round_control_dir = str(control_dir)
    text_finished = threading.Event()
    responder_errors: list[str] = []

    def acknowledge_round() -> None:
        deadline = time.time() + 2
        start_path = control_dir / "round-000000.start.json"
        while not start_path.exists() and time.time() < deadline:
            time.sleep(0.01)
        if not start_path.exists():
            responder_errors.append("round start was not written")
            return
        payload = json.loads(start_path.read_text(encoding="utf-8"))
        assert payload["round_id"] == 0
        assert payload["text_task_count"] == 1
        assert payload["image_task_count"] == 1
        (control_dir / "round-000000.ready.json").write_text(
            json.dumps(
                {
                    "status": "model_round_services_ready",
                    "round_id": 0,
                    "run_id": "",
                }
            ),
            encoding="utf-8",
        )

    responder = threading.Thread(target=acknowledge_round)
    responder.start()

    def fake_resolve(**kwargs):
        tasks = kwargs["tasks"]
        kind = tasks[0].asset["asset_type"]
        if kind == "text":
            text_finished.set()
            return []
        assert text_finished.wait(timeout=1)
        deadline = time.time() + 2
        text_done = control_dir / "round-000000.text.done.json"
        while not text_done.exists() and time.time() < deadline:
            time.sleep(0.01)
        assert text_done.exists()
        return []

    monkeypatch.setattr(joinability_dataset, "resolve_extraction_tasks", fake_resolve)

    counts = precompute_extraction_task_groups(
        extractor=object(),
        cache=object(),
        tasks_by_kind={"text": [_task("text", "1")], "image": [_task("image", "1")]},
        args=args,
        state=ModelConcurrencyState(text_workers=1, image_workers=1),
        write_done_markers=False,
    )
    responder.join(timeout=2)

    assert responder_errors == []
    assert counts == {"text": 1, "image": 1}
    assert (control_dir / "round-000000.text.done.json").exists()
    assert (control_dir / "round-000000.image.done.json").exists()
    assert (control_dir / "round-000000.done.json").exists()


def test_precompute_task_groups_immediately_marks_empty_modality_done(tmp_path):
    image_started = threading.Event()
    release_image = threading.Event()

    class FakeExtractor:
        def extract(self, asset, entity, candidate_attribute_names):
            image_started.set()
            release_image.wait(timeout=2.0)
            return {"attributes": [], "raw_response": '{"attributes":[]}', "error": ""}

    args = _parallel_args(
        model_text_done_marker=str(tmp_path / "text_done.json"),
        model_image_done_marker=str(tmp_path / "image_done.json"),
    )
    worker = threading.Thread(
        target=precompute_extraction_task_groups,
        kwargs={
            "extractor": FakeExtractor(),
            "cache": ExtractionCache(tmp_path / "model_cache.jsonl"),
            "tasks_by_kind": {"text": [], "image": [_task("image", "pending")]},
            "args": args,
            "state": ModelConcurrencyState(text_workers=1, image_workers=1),
            "progress": None,
        },
    )
    worker.start()
    assert image_started.wait(timeout=2.0)

    assert (tmp_path / "text_done.json").exists()
    assert not (tmp_path / "image_done.json").exists()

    release_image.set()
    worker.join(timeout=2.0)
    assert not worker.is_alive()


def test_tasks_requiring_model_analysis_excludes_reusable_cached_tasks(tmp_path):
    tasks = [_task("text", "cached"), _task("image", "missing")]
    cache = ExtractionCache(tmp_path / "model_cache.jsonl")
    cache.put(
        tasks[0].cache_key,
        {
            "cache_key": tasks[0].cache_key,
            "attributes": [{"name": "State", "value": "Alabama", "evidence": "cached"}],
            "raw_response": '{"attributes":[]}',
            "error": "",
        },
    )

    pending = tasks_requiring_model_analysis(tasks, cache, _parallel_args())

    assert pending == [tasks[1]]


def test_tasks_requiring_model_analysis_ignores_legacy_checker_state(tmp_path):
    task = _task("text", "checker-failed")
    cache = ExtractionCache(tmp_path / "model_cache.jsonl")
    cache.put(
        task.cache_key,
        {
            "cache_key": task.cache_key,
            "attributes": [],
            "model_attributes": [{"name": "State", "value": "Alabama"}],
            "raw_response": '{"attributes":[{"name":"State","value":"Alabama"}]}',
            "error": "",
            "auto_check": {
                "schema_version": joinability_dataset.MODEL_AUTO_CHECK_SCHEMA_VERSION,
                "reviewed_attributes": 1,
                "reviews": [
                    {
                        "verdict": "insufficient",
                        "review_complete": False,
                        "error_code": "model_review_failed:response_invalid_json",
                        "decision_source": "primary_provisional_secondary_failed",
                    }
                ],
            },
        },
    )

    assert tasks_requiring_model_analysis(
        [task],
        cache,
        _parallel_args(
            reparse_cached_model_outputs=False,
            refresh_invalid_model_cache=True,
        ),
    ) == []


def test_tasks_requiring_model_analysis_reuses_completed_secondary_review(tmp_path):
    task = _task("text", "secondary-complete")
    cache = ExtractionCache(tmp_path / "model_cache.jsonl")
    cache.put(
        task.cache_key,
        {
            "cache_key": task.cache_key,
            "attributes": [{"name": "State", "value": "Alabama"}],
            "model_attributes": [{"name": "State", "value": "Alabama"}],
            "raw_response": '{"attributes":[{"name":"State","value":"Alabama"}]}',
            "error": "",
            "auto_check": {
                "schema_version": joinability_dataset.MODEL_AUTO_CHECK_SCHEMA_VERSION,
                "reviewed_attributes": 1,
                "reviews": [
                    {
                        "verdict": "supported",
                        "review_complete": True,
                        "error_code": "",
                        "primary_error_code": "LocalModelHTTPError",
                        "decision_source": "secondary_openai",
                    }
                ],
            },
        },
    )

    assert tasks_requiring_model_analysis(
        [task], cache, _parallel_args(reparse_cached_model_outputs=False)
    ) == []


def test_resolve_extraction_tasks_reuses_legacy_analysis_without_checker(
    tmp_path,
):
    class CheckerExtractor:
        auto_check_enabled = True
        model_auto_check_stats = joinability_dataset.ModelAutoCheckStats()

        def __init__(self):
            self.analysis_calls = 0
            self.checker_calls = 0

        def extract(self, *_args, **_kwargs):
            self.analysis_calls += 1
            raise AssertionError("legacy cache upgrade must not rerun analysis")

        def review_auto_check_attribute(self, **_kwargs):
            self.checker_calls += 1
            return {
                "extracted_value": "Alabama",
                "verdict": "supported",
                "comparison": "normalized_values_match",
                "decision_source": "primary_local",
                "review_complete": True,
                "error_code": "",
            }

    task = _task("text", "legacy-auto-check-upgrade")
    task.entity["row_attributes"] = [
        {"name": "Entity", "value": "Entity", "is_entity": True},
        {"name": "State", "value": "Alabama", "is_entity": False},
    ]
    cache = ExtractionCache(tmp_path / "model_cache.jsonl")
    cache.put(
        task.cache_key,
        {
            "cache_key": task.cache_key,
            "prompt_version": joinability_dataset.PROMPT_VERSION,
            "attributes": [{"name": "State", "value": "Alabama"}],
            "raw_response": (
                '{"attributes":[{"name":"State","value":"Alabama"}]}'
            ),
            "error": "",
        },
    )
    extractor = CheckerExtractor()
    progress = joinability_dataset.ModelAnalysisProgress(
        total=1, cached_keys=set(), enabled=False
    )
    progress.register([task.cache_key])

    assert tasks_requiring_model_analysis(
        [task],
        cache,
        _parallel_args(reparse_cached_model_outputs=False),
        extractor=extractor,
    ) == []
    records = resolve_extraction_tasks(
        extractor=extractor,
        cache=cache,
        tasks=[task],
        args=_parallel_args(reparse_cached_model_outputs=False),
        state=ModelConcurrencyState(text_workers=1, image_workers=1),
        progress=progress,
    )

    assert extractor.analysis_calls == 0
    assert extractor.checker_calls == 0
    assert records[0][1]["attributes"] == [
        {"name": "State", "value": "Alabama"}
    ]
    assert "auto_check" not in records[0][1]
    assert progress.cached == 1
    assert progress.model == 0


def test_tasks_requiring_model_analysis_immediately_marks_only_reusable_records(
    tmp_path,
):
    cached = _task("text", "cached")
    transient = _task("image", "transient")
    transient_error = _task("text", "transient-error")
    retry_error = _task("text", "retry-error")
    refresh_invalid = _task("image", "refresh-invalid")
    missing = _task("text", "missing")
    cache = ExtractionCache(tmp_path / "model_cache.jsonl")
    cache.put(
        cached.cache_key,
        {
            "cache_key": cached.cache_key,
            "attributes": [{"name": "State", "value": "Alabama"}],
            "raw_response": '{"attributes":[]}',
            "error": "",
        },
    )
    cache.put_transient(
        transient.cache_key,
        {
            "cache_key": transient.cache_key,
            "attributes": [{"name": "State", "value": "Alabama"}],
            "error": "",
        },
    )
    cache.put_transient(
        transient_error.cache_key,
        {
            "cache_key": transient_error.cache_key,
            "attributes": [],
            "error": "already failed this run",
        },
    )
    cache.put(
        retry_error.cache_key,
        {
            "cache_key": retry_error.cache_key,
            "attributes": [],
            "raw_response": "",
            "error": "retry me",
        },
    )
    cache.put(
        refresh_invalid.cache_key,
        {
            "cache_key": refresh_invalid.cache_key,
            "attributes": [],
            "raw_response": '{"attributes":[]}',
            "error": "",
        },
    )
    progress = joinability_dataset.ModelAnalysisProgress(
        total=6,
        cached_keys=set(),
        enabled=False,
    )
    progress.register(
        task.cache_key
        for task in (
            cached,
            transient,
            transient_error,
            retry_error,
            refresh_invalid,
            missing,
        )
    )
    args = _parallel_args(
        reparse_cached_model_outputs=False,
        refresh_invalid_model_cache=True,
    )

    pending = tasks_requiring_model_analysis(
        [
            cached,
            cached,
            transient,
            transient_error,
            retry_error,
            refresh_invalid,
            missing,
        ],
        cache,
        args,
        progress=progress,
    )

    assert pending == [retry_error, refresh_invalid, missing]
    assert progress.cached == 1
    assert progress.model == 2
    assert progress.errors == 1
    assert progress.completed_keys == {
        cached.cache_key,
        transient.cache_key,
        transient_error.cache_key,
    }
    assert retry_error.cache_key in progress.planned_keys
    assert refresh_invalid.cache_key in progress.planned_keys
    assert missing.cache_key in progress.planned_keys

    resolve_extraction_tasks(
        extractor=None,
        cache=cache,
        tasks=[cached, transient, transient_error],
        args=args,
        state=ModelConcurrencyState(text_workers=1, image_workers=1),
        progress=progress,
    )
    assert progress.cached == 1
    assert progress.model == 2
    assert progress.errors == 1


def test_resolve_extraction_tasks_runs_text_and_image_pools_concurrently(tmp_path):
    barrier = threading.Barrier(2, timeout=2.0)

    class FakeExtractor:
        def extract(self, asset, entity, candidate_attribute_names):
            barrier.wait()
            attrs = [{"name": "State", "value": "Alabama", "evidence": asset["asset_type"]}]
            return {"attributes": attrs, "raw_response": '{"attributes":[]}', "error": ""}

    cache = ExtractionCache(tmp_path / "model_cache.jsonl")
    tasks = [_task("text", "1"), _task("image", "2")]
    state = ModelConcurrencyState(text_workers=1, image_workers=1)

    records = resolve_extraction_tasks(
        extractor=FakeExtractor(),
        cache=cache,
        tasks=tasks,
        args=_parallel_args(),
        state=state,
        progress=None,
    )

    assert [record["asset_type"] for _task_item, record in records] == ["text", "image"]
    assert all(not record["error"] for _task_item, record in records)


def test_local_and_remote_text_pools_run_with_independent_worker_limits(tmp_path):
    first_wave = threading.Barrier(4, timeout=2.0)

    class PoolCountingExtractor:
        def __init__(self):
            self.active = {"local": 0, "remote": 0}
            self.maximum = {"local": 0, "remote": 0}
            self.calls = {"local": 0, "remote": 0}
            self.lock = threading.Lock()

        def _extract(self, endpoint_pool):
            with self.lock:
                self.active[endpoint_pool] += 1
                self.calls[endpoint_pool] += 1
                self.maximum[endpoint_pool] = max(
                    self.maximum[endpoint_pool],
                    self.active[endpoint_pool],
                )
                call_number = sum(self.calls.values())
            try:
                if call_number <= 4:
                    first_wave.wait()
                return {
                    "attributes": [],
                    "raw_response": '{"attributes":[]}',
                    "error": "",
                }
            finally:
                with self.lock:
                    self.active[endpoint_pool] -= 1

        def extract(self, asset, entity, candidate_attribute_names):
            return self._extract("local")

        def extract_from_pool(
            self,
            asset,
            entity,
            candidate_attribute_names,
            *,
            endpoint_pool,
        ):
            return self._extract(endpoint_pool)

    extractor = PoolCountingExtractor()
    state = ModelConcurrencyState(
        text_workers=1,
        image_workers=1,
        remote_text_workers=3,
    )

    records = resolve_extraction_tasks(
        extractor=extractor,
        cache=ExtractionCache(tmp_path / "model_cache.jsonl"),
        tasks=[_task("text", str(index)) for index in range(12)],
        args=_parallel_args(
            text_model_workers=1,
            remote_text_model_workers=3,
        ),
        state=state,
        progress=None,
    )

    assert len(records) == 12
    assert extractor.maximum == {"local": 1, "remote": 3}
    assert extractor.calls["local"] > 0
    assert extractor.calls["remote"] > 0
    assert state.text_workers == 1
    assert state.remote_text_workers == 3


def test_remote_oom_downgrade_does_not_change_local_concurrency():
    state = ModelConcurrencyState(
        text_workers=2,
        image_workers=3,
        remote_text_workers=8,
        remote_image_workers=6,
    )

    assert state.downgrade_after_oom(
        "image",
        attempted_workers=6,
        endpoint_pool="remote",
    ) == 3
    assert state.image_workers == 3
    assert state.remote_image_workers == 3
    assert state.image_oom_downgrades == 0
    assert state.remote_image_oom_downgrades == 1


def test_resolve_extraction_tasks_marks_progress_as_each_model_task_finishes(tmp_path):
    slow_started = threading.Event()
    release_slow = threading.Event()
    quick_marked = threading.Event()

    class FakeExtractor:
        def extract(self, asset, entity, candidate_attribute_names):
            if entity["entity_id"] == "ent_1":
                slow_started.set()
                release_slow.wait(timeout=2.0)
            attrs = [{"name": "State", "value": "Alabama", "evidence": entity["entity_id"]}]
            return {"attributes": attrs, "raw_response": '{"attributes":[]}', "error": ""}

    class FakeProgress:
        def mark(self, cache_key, status):
            if cache_key == "cache_text_2" and status == "model":
                quick_marked.set()

    cache = ExtractionCache(tmp_path / "model_cache.jsonl")
    tasks = [_task("text", "1"), _task("text", "2")]
    state = ModelConcurrencyState(text_workers=2, image_workers=1)

    worker = threading.Thread(
        target=resolve_extraction_tasks,
        kwargs={
            "extractor": FakeExtractor(),
            "cache": cache,
            "tasks": tasks,
            "args": _parallel_args(text_model_workers=2),
            "state": state,
            "progress": FakeProgress(),
        },
    )
    worker.start()
    assert slow_started.wait(timeout=2.0)

    assert quick_marked.wait(timeout=2.0)
    assert worker.is_alive()

    release_slow.set()
    worker.join(timeout=2.0)

    assert not worker.is_alive()


def test_resolve_extraction_tasks_downgrades_after_oom_and_retries_serially(tmp_path):
    active = 0
    calls = 0
    active_lock = threading.Lock()
    first_wave = threading.Barrier(2, timeout=2.0)

    class FakeExtractor:
        def extract(self, asset, entity, candidate_attribute_names):
            nonlocal active, calls
            with active_lock:
                active += 1
                calls += 1
                current = active
                call_number = calls
            try:
                if call_number <= 2:
                    first_wave.wait()
                if current > 1:
                    raise RuntimeError("CUDA out of memory")
                attrs = [{"name": "State", "value": "Alabama", "evidence": "ok"}]
                return {"attributes": attrs, "raw_response": '{"attributes":[]}', "error": ""}
            finally:
                with active_lock:
                    active -= 1

    cache = ExtractionCache(tmp_path / "model_cache.jsonl")
    tasks = [_task("image", "1"), _task("image", "2")]
    state = ModelConcurrencyState(text_workers=1, image_workers=2)

    records = resolve_extraction_tasks(
        extractor=FakeExtractor(),
        cache=cache,
        tasks=tasks,
        args=_parallel_args(image_model_workers=2),
        state=state,
        progress=None,
    )

    assert state.image_workers == 1
    assert all(not record["error"] for _task_item, record in records)
    assert cache.get("cache_image_1") is not None
    assert cache.get("cache_image_2") is not None


def test_resolve_extraction_tasks_recovers_workers_additively_after_halving_on_oom(tmp_path):
    class ActiveCountingExtractor:
        def __init__(self, *, barrier_parties: int, oom_above_active: int | None = None):
            self.active = 0
            self.calls = 0
            self.max_active = 0
            self.lock = threading.Lock()
            self.barrier = threading.Barrier(barrier_parties, timeout=2.0)
            self.barrier_parties = barrier_parties
            self.oom_above_active = oom_above_active

        def extract(self, asset, entity, candidate_attribute_names):
            with self.lock:
                self.active += 1
                self.calls += 1
                current_active = self.active
                call_number = self.calls
                self.max_active = max(self.max_active, current_active)
            try:
                if call_number <= self.barrier_parties:
                    self.barrier.wait()
                if self.oom_above_active is not None and current_active > self.oom_above_active:
                    raise RuntimeError("CUDA out of memory")
                attrs = [{"name": "State", "value": "Alabama", "evidence": entity["entity_id"]}]
                return {"attributes": attrs, "raw_response": '{"attributes":[]}', "error": ""}
            finally:
                with self.lock:
                    self.active -= 1

    cache = ExtractionCache(tmp_path / "model_cache.jsonl")
    state = ModelConcurrencyState(text_workers=1, image_workers=4)

    oom_extractor = ActiveCountingExtractor(barrier_parties=4, oom_above_active=2)
    records = resolve_extraction_tasks(
        extractor=oom_extractor,
        cache=cache,
        tasks=[_task("image", f"oom_{idx}") for idx in range(4)],
        args=_parallel_args(image_model_workers=4),
        state=state,
        progress=None,
    )

    assert oom_extractor.max_active == 4
    assert state.image_workers == 2
    assert state.image_oom_downgrades == 1
    assert all(not record["error"] for _task_item, record in records)

    first_recovery = ActiveCountingExtractor(barrier_parties=2)
    resolve_extraction_tasks(
        extractor=first_recovery,
        cache=cache,
        tasks=[_task("image", f"recover_a_{idx}") for idx in range(2)],
        args=_parallel_args(image_model_workers=4),
        state=state,
        progress=None,
    )

    assert first_recovery.max_active == 2
    assert state.image_workers == 3

    second_recovery = ActiveCountingExtractor(barrier_parties=3)
    resolve_extraction_tasks(
        extractor=second_recovery,
        cache=cache,
        tasks=[_task("image", f"recover_b_{idx}") for idx in range(3)],
        args=_parallel_args(image_model_workers=4),
        state=state,
        progress=None,
    )

    assert second_recovery.max_active == 3
    assert state.image_workers == 4


def test_resolve_extraction_tasks_does_not_cache_failed_model_outputs(tmp_path):
    class FakeExtractor:
        def extract(self, asset, entity, candidate_attribute_names):
            raise RuntimeError("CUDA out of memory")

    cache = ExtractionCache(tmp_path / "model_cache.jsonl")
    task = _task("image", "1")
    state = ModelConcurrencyState(text_workers=1, image_workers=1)

    records = resolve_extraction_tasks(
        extractor=FakeExtractor(),
        cache=cache,
        tasks=[task],
        args=_parallel_args(),
        state=state,
        progress=None,
    )

    assert records[0][1]["error"]
    assert cache.get(task.cache_key) is None
    assert not (tmp_path / "model_cache.jsonl").exists() or not (tmp_path / "model_cache.jsonl").read_text(encoding="utf-8").strip()


def test_failed_precompute_result_is_reused_during_table_evaluation_only(tmp_path):
    calls = 0

    class FakeExtractor:
        def extract(self, asset, entity, candidate_attribute_names):
            nonlocal calls
            calls += 1
            raise RuntimeError("model unavailable")

    args = joinability_dataset.parse_args(
        [
            "--input_dir",
            str(tmp_path),
            "--output_dir",
            str(tmp_path / "output"),
            "--query_rows_per_table",
            "1",
            "--min_rows_per_output_table",
            "1",
        ]
    )
    args.model_attribute_errors_path = str(tmp_path / "model_attribute_errors.jsonl")
    source_table = {
        "source_table_id": "src",
        "columns": [
            {"column_index": 0, "column_name": "Entity"},
            {"column_index": 1, "column_name": "State"},
        ],
        "rows": [
            {
                "row_id": 0,
                "cells": [
                    {
                        "column_index": 0,
                        "column_name": "Entity",
                        "text": "Alpha",
                        "wiki_title": "Alpha",
                    },
                    {"column_index": 1, "column_name": "State", "text": "Alabama"},
                ],
            }
        ],
        "metadata": {"candidate_entity_columns": [0]},
    }
    assets = {
        "asset_text": {
            "asset_id": "asset_text",
            "asset_type": "text",
            "content": "Alpha is in Alabama.",
        }
    }
    entity_to_assets = {"entity_alpha": ["asset_text"]}
    wiki_to_entity_id = {"Alpha": "entity_alpha"}
    tasks = joinability_dataset.collect_table_extraction_tasks(
        source_table=source_table,
        assets=assets,
        entity_to_assets=entity_to_assets,
        wiki_to_entity_id=wiki_to_entity_id,
        args=args,
    )
    cache_path = tmp_path / "model_attribute_extractions.jsonl"
    cache = ExtractionCache(cache_path)
    state = ModelConcurrencyState(text_workers=1, image_workers=1)
    progress = joinability_dataset.ModelAnalysisProgress(
        total=1, cached_keys=set(), enabled=False
    )

    counts = precompute_extraction_task_groups(
        extractor=FakeExtractor(),
        cache=cache,
        tasks_by_kind={"text": tasks},
        args=args,
        state=state,
        progress=progress,
        write_done_markers=False,
    )
    joinability_dataset.build_table_join_records(
        source_table=source_table,
        split="candidate",
        assets=assets,
        entity_to_assets=entity_to_assets,
        wiki_to_entity_id=wiki_to_entity_id,
        extractor=FakeExtractor(),
        cache=cache,
        progress=progress,
        concurrency_state=state,
        extraction_writer=joinability_dataset.ListRecordWriter(),
        recovery_writer=joinability_dataset.ListRecordWriter(),
        args=args,
    )

    assert counts == {"text": 1}
    assert calls == 1
    assert progress.model == 1
    assert progress.errors == 1
    assert not cache_path.exists() or not cache_path.read_text(encoding="utf-8").strip()
    assert len(
        (tmp_path / "model_attribute_errors.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ) == 1
    assert tasks_requiring_model_analysis(
        tasks, ExtractionCache(cache_path), args
    ) == tasks


def test_resolve_extraction_tasks_writes_failed_model_outputs_to_error_log(tmp_path):
    class FakeExtractor:
        def extract(self, asset, entity, candidate_attribute_names):
            raise RuntimeError("Input length (6749) exceeds model's maximum context length (4096).")

    cache = ExtractionCache(tmp_path / "model_cache.jsonl")
    task = _task("image", "1")
    state = ModelConcurrencyState(text_workers=1, image_workers=1)
    error_log = tmp_path / "model_attribute_errors.jsonl"

    records = resolve_extraction_tasks(
        extractor=FakeExtractor(),
        cache=cache,
        tasks=[task],
        args=_parallel_args(model_attribute_errors_path=str(error_log)),
        state=state,
        progress=None,
    )

    assert records[0][1]["error"]
    assert cache.get(task.cache_key) is None
    lines = [line for line in error_log.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) == 1
    assert "Input length (6749)" in lines[0]


def test_resolve_extraction_tasks_persists_raw_result_without_running_checker(tmp_path):
    class FakeExtractor:
        auto_check_enabled = True
        model_auto_check_stats = joinability_dataset.ModelAutoCheckStats()

        def extract(self, asset, entity, candidate_attribute_names):
            return {
                "attributes": [{"name": "State", "value": "Alabama"}],
                "raw_response": (
                    '{"attributes":[{"name":"State","value":"Alabama"}]}'
                ),
                "error": "",
            }

        def review_auto_check_attribute(self, **_kwargs):
            raise RuntimeError("secondary checker unavailable")

    cache = ExtractionCache(tmp_path / "model_cache.jsonl")
    task = _auto_check_task()

    records = resolve_extraction_tasks(
        extractor=FakeExtractor(),
        cache=cache,
        tasks=[task],
        args=_parallel_args(),
        state=ModelConcurrencyState(text_workers=1, image_workers=1),
        progress=None,
    )

    assert records[0][1]["attributes"] == [
        {"name": "State", "value": "Alabama"}
    ]
    assert "auto_check" not in records[0][1]
    assert cache.get(task.cache_key) is not None
    assert cache.get_transient(task.cache_key) is None


def test_parallel_bridge_asset_builder_fetches_entities_concurrently(tmp_path):
    barrier = threading.Barrier(2, timeout=2.0)

    class FakeWikipediaClient:
        api_failures = 0

        def get_page(self, wiki_title):
            barrier.wait()
            return {
                "extract": f"{wiki_title} State: Alabama.",
                "canonicalurl": f"https://example.test/wiki/{wiki_title}",
                "images": [],
            }

    entities = [
        {"entity_id": "ent_alpha", "wiki_title": "Alpha", "display_texts": ["Alpha"], "context_terms": ["State"]},
        {"entity_id": "ent_beta", "wiki_title": "Beta", "display_texts": ["Beta"], "context_terms": ["State"]},
    ]
    writer = ShardedJsonlWriter(tmp_path / "bridge_assets", max_records_per_shard=10)

    with writer:
        entity_to_assets, api_failures, text_count, image_count = build_bridge_assets_parallel(
            entities=entities,
            max_entities=None,
            max_images_per_entity=0,
            text_asset_chunk_chars=200,
            min_text_asset_chunk_chars=20,
            max_text_asset_chunks_per_entity=1,
            wikipedia_client_factory=FakeWikipediaClient,
            asset_writer=writer,
            flush_every_records=10,
            workers=2,
        )

    assert api_failures == 0
    assert text_count == 2
    assert image_count == 0
    assert sorted(entity_to_assets) == ["ent_alpha", "ent_beta"]
