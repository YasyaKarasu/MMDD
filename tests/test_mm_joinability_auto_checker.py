import json
import re
import sqlite3
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from mm_joinability_dataset_auto_checker import (
    AutoCheckConfig,
    parse_model_extractions,
    review_messages,
    run_auto_check,
    validate_local_base_url,
)
from stage1_io import write_jsonl


def _table_record(table_id: str, role: str, entity: str, year: str) -> dict:
    columns = (
        [{"column_index": 0, "column_name": "Entity"}]
        if role == "query"
        else [{"column_index": 0, "column_name": "Year"}]
    )
    value = entity if role == "query" else year
    return {
        "table_id": table_id,
        "role": role,
        "split": "test",
        "source_table_id": f"source_{entity}",
        "page_title": entity,
        "caption": "synthetic auto-check fixture",
        "columns": columns,
        "rows": [
            {
                "row_id": 0,
                "source_row_id": 0,
                "cells": [
                    {
                        "column_index": 0,
                        "column_name": columns[0]["column_name"],
                        "text": value,
                    }
                ],
            }
        ],
    }


def _write_auto_check_dataset(
    root: Path,
    query_count: int = 10,
    evidence_per_query: int = 1,
) -> None:
    artifact_records: dict[str, list[dict]] = {
        "query_tables": [],
        "data_lake_tables": [],
        "bridge_assets": [],
        "evidence_recoveries": [],
    }
    qrels = []
    for index in range(query_count):
        query_id = f"query_{index:03d}"
        target_id = f"target_{index:03d}"
        entity = f"Entity {index}"
        year = str(2000 + index)
        artifact_records["query_tables"].append(
            _table_record(query_id, "query", entity, year)
        )
        artifact_records["data_lake_tables"].append(
            _table_record(target_id, "target_data_lake_table", entity, year)
        )
        for evidence_index in range(evidence_per_query):
            asset_id = f"asset_{index:03d}_{evidence_index:02d}"
            artifact_records["bridge_assets"].append(
                {
                    "asset_id": asset_id,
                    "asset_type": "text",
                    "entity_wiki_title": entity,
                    "content": f"{entity} was founded in {year}.",
                }
            )
            artifact_records["evidence_recoveries"].append(
                {
                    "recovery_id": f"recovery_{index:03d}_{evidence_index:02d}",
                    "path_id": f"path_{index:03d}_{evidence_index:02d}",
                    "query_table_id": query_id,
                    "target_table_id": target_id,
                    "query_row_id": 0,
                    "target_row_ids": [0],
                    "source_table_id": f"source_{entity}",
                    "split": "test",
                    "query_entity": {
                        "cell_text": entity,
                        "wiki_title": entity,
                        "entity_column_name": "Entity",
                        "row_attributes": [
                            {"name": "Entity", "value": entity, "is_entity": True},
                            {"name": "Year", "value": year, "is_entity": False},
                            {"name": "Category", "value": "Synthetic", "is_entity": False},
                        ],
                    },
                    "recovered_attribute": {
                        "column_name": "Year",
                        "value": year,
                    },
                    "evidence": {
                        "asset_id": asset_id,
                        "asset_type": "text",
                        "content_snippet": f"{entity} was founded in {year}.",
                    },
                }
            )
        qrels.append(
            {
                "query_table_id": query_id,
                "target_table_id": target_id,
                "data_lake_table_id": target_id,
                "rel": 3,
                "split": "test",
                "source_table_id": f"source_{entity}",
                "join_attribute": {"column_name": "Year"},
                "reason": "model_recoverable_join_column",
            }
        )

    manifest_artifacts = {}
    for artifact, records in artifact_records.items():
        directory = root / artifact
        directory.mkdir(parents=True)
        relative = f"{artifact}/part-00000.jsonl"
        write_jsonl(root / relative, records)
        manifest_artifacts[artifact] = {
            "shards": [{"path": relative, "records": len(records)}]
        }
    write_jsonl(root / "qrels.jsonl", qrels)
    (root / "stats.json").write_text("{}", encoding="utf-8")
    (root / "dataset_manifest.json").write_text(
        json.dumps(
            {
                "format": "sharded_jsonl",
                "artifacts": manifest_artifacts,
                "single_files": {"qrels": "qrels.jsonl", "stats": "stats.json"},
            }
        ),
        encoding="utf-8",
    )


class FakeExtractor:
    def __init__(
        self,
        identity_version: str = "v1",
        value_override: str | None = None,
    ) -> None:
        self.identity = {"provider": "fake", "version": identity_version}
        self.calls: list[list[str]] = []
        self.value_override = value_override

    def extract_batches(self, batches: list[dict]) -> dict[str, list[dict[str, str]]]:
        assert len(batches) == 1
        assert len(batches[0]["items"]) == 1
        item = batches[0]["items"][0]
        masked_names = {cell["name"] for cell in item["masked_row"]}
        assert item["attribute"]["name"] not in masked_names
        assert masked_names == {"Entity"}
        assert "Category" not in masked_names
        self.calls.append([batch["query_table_id"] for batch in batches])
        return {
            batch["query_table_id"]: [
                {
                    "review_id": item["review_id"],
                    "attribute_name": item["attribute"]["name"],
                    "extracted_value": self.value_override
                    if self.value_override is not None
                    else re.search(
                        r"founded in (\d{4})",
                        item["evidence"]["content"],
                    ).group(1),
                }
                for item in batch["items"]
            ]
            for batch in batches
        }


class BrokenExtractor:
    def __init__(self, identity_version: str) -> None:
        self.identity = {"provider": "fake", "version": identity_version}
        self.calls = 0

    def extract_batches(self, _batches: list[dict]) -> dict:
        self.calls += 1
        raise RuntimeError("synthetic model failure")


def _config(root: Path, sample_rate: float) -> AutoCheckConfig:
    return AutoCheckConfig(
        output_dir=root,
        cache_path=root / "cache" / "auto_checker.sqlite3",
        report_dir=root / "reports",
        sample_rate=sample_rate,
        seed=13,
        workers=2,
        max_asset_chars=1000,
        index_path=root / "auto_checker_index.sqlite3",
        progress_every=0,
    )


def test_increasing_sample_reuses_query_cache_and_writes_reports(tmp_path: Path) -> None:
    _write_auto_check_dataset(tmp_path)
    first_reviewer = FakeExtractor()
    first = run_auto_check(_config(tmp_path, 0.10), first_reviewer)

    assert first["sample"]["population_queries"] == 10
    assert first["sample"]["sampled_queries"] == 1
    assert first["path_judgments"]["supported"] == 1
    assert first["execution"]["cache_hit_attribute_reviews"] == 0
    assert len(first_reviewer.calls) == 1

    second_reviewer = FakeExtractor()
    second = run_auto_check(_config(tmp_path, 0.20), second_reviewer)

    assert second["sample"]["sampled_queries"] == 2
    assert second["execution"]["cache_hit_attribute_reviews"] == 1
    assert second["execution"]["model_extracted_attributes"] == 1
    assert second["path_judgments"]["supported_rate"] == 1.0
    assert sum(len(call) for call in second_reviewer.calls) == 1
    path_reviews = [
        json.loads(line)
        for line in Path(second["artifacts"]["path_reviews"])
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(path_reviews) == 2
    assert {row["cache_hit"] for row in path_reviews} == {False, True}
    assert all(row["recommended_action"] == "keep_recovery" for row in path_reviews)

    with sqlite3.connect(tmp_path / "cache" / "auto_checker.sqlite3") as connection:
        assert connection.execute("SELECT COUNT(*) FROM attribute_reviews").fetchone()[0] == 2
        assert connection.execute("SELECT COUNT(*) FROM audit_runs").fetchone()[0] == 2


def test_model_identity_change_invalidates_cache(tmp_path: Path) -> None:
    _write_auto_check_dataset(tmp_path, query_count=2)
    run_auto_check(_config(tmp_path, 0.5), FakeExtractor("v1"))
    changed = FakeExtractor("v2")

    summary = run_auto_check(_config(tmp_path, 0.5), changed)

    assert summary["execution"]["cache_hit_attribute_reviews"] == 0
    assert sum(len(call) for call in changed.calls) == 1


def test_review_prompt_omits_target_rows_and_parser_requires_exact_ids() -> None:
    batch = {
        "query_table_id": "query-1",
        "target_table_id": "secret-target",
        "query_row_ids": ["0"],
        "items": [
            {
                "review_id": "recovery-1",
                "query_row_id": "0",
                "masked_row": [
                    {"name": "Entity", "value": "Alpha", "is_entity": True}
                ],
                "query_entity": {"cell_text": "Alpha"},
                "attribute": {"name": "Country", "value": "France"},
                "evidence": {
                    "asset_id": "text-1",
                    "asset_type": "text",
                    "title": "SECRET EVIDENCE TITLE",
                    "source": "SECRET EVIDENCE SOURCE",
                    "content": "Alpha has a country stated in the source.",
                },
                "image_path": "",
            }
        ],
    }
    rendered = json.dumps(review_messages([batch]), ensure_ascii=False)
    assert "secret-target" not in rendered
    assert "France" not in rendered
    assert "query-1" not in rendered
    assert "recovery-1" not in rendered
    assert "text-1" not in rendered
    assert "SECRET EVIDENCE TITLE" not in rendered
    assert "SECRET EVIDENCE SOURCE" not in rendered
    assert "is_entity" not in rendered
    assert "Country" in rendered
    assert "evidence may be unrelated to the entity" in rendered
    assert "merely assuming the entity-evidence relationship" in rendered
    assert "pretrained, memorized, and outside knowledge as unavailable" in rendered
    assert "leaves multiple candidates" in rendered

    response = json.dumps({"extracted_value": "France"})
    assert parse_model_extractions(response, [batch])["query-1"][0][
        "extracted_value"
    ] == "France"

    missing = json.dumps({})
    with pytest.raises(ValueError, match="missing extracted_value"):
        parse_model_extractions(missing, [batch])

    with pytest.raises(ValueError, match="exactly one evidence"):
        review_messages([batch, {**batch, "query_table_id": "query-2"}])


def test_each_evidence_is_a_separate_model_request(tmp_path: Path) -> None:
    _write_auto_check_dataset(tmp_path, query_count=1, evidence_per_query=2)
    reviewer = FakeExtractor()

    summary = run_auto_check(_config(tmp_path, 1.0), reviewer)

    assert summary["sample"]["sampled_queries"] == 1
    assert summary["sample"]["planned_recovery_paths"] == 2
    assert summary["execution"]["model_calls"] == 2
    assert summary["execution"]["attributes_per_request"] == 1
    assert len(reviewer.calls) == 2


def test_local_comparison_rejects_different_extracted_value(tmp_path: Path) -> None:
    _write_auto_check_dataset(tmp_path, query_count=1)

    summary = run_auto_check(
        _config(tmp_path, 1.0),
        FakeExtractor(value_override="1999"),
    )

    assert summary["path_judgments"]["contradicted"] == 1
    row = json.loads(
        Path(summary["artifacts"]["path_reviews"])
        .read_text(encoding="utf-8")
        .strip()
    )
    assert row["claimed_value"] == "2000"
    assert row["extracted_value"] == "1999"
    assert row["comparison"] == "extracted_value_mismatch"
    assert row["recommended_action"] == "drop_recovery"


def test_local_mismatch_runs_luna_and_disagreement_runs_terra(
    tmp_path: Path,
) -> None:
    _write_auto_check_dataset(tmp_path, query_count=2)
    primary = FakeExtractor(identity_version="local", value_override="1999")
    luna = FakeExtractor(identity_version="luna")
    terra = FakeExtractor(identity_version="terra")

    summary = run_auto_check(
        _config(tmp_path, 1.0),
        primary,
        luna,
        terra,
    )

    assert len(primary.calls) == 2
    assert len(luna.calls) == 2
    assert len(terra.calls) == 2
    assert summary["execution"]["secondary_candidates"] == 2
    assert summary["execution"]["secondary_model_calls"] == 2
    assert summary["execution"]["secondary_max_concurrency"] == 5
    assert summary["execution"]["terra_candidates"] == 2
    assert summary["execution"]["terra_model_calls"] == 2
    assert summary["path_judgments"]["supported"] == 2
    rows = [
        json.loads(line)
        for line in Path(summary["artifacts"]["path_reviews"])
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert {row["primary_verdict"] for row in rows} == {"contradicted"}
    assert {row["luna_verdict"] for row in rows} == {"supported"}
    assert {row["luna_agrees_with_local"] for row in rows} == {False}
    assert {row["terra_verdict"] for row in rows} == {"supported"}
    assert {row["decision_source"] for row in rows} == {"terra_adjudication"}
    assert all(row["review_complete"] for row in rows)


def test_supported_local_extraction_skips_luna_and_terra(tmp_path: Path) -> None:
    _write_auto_check_dataset(tmp_path, query_count=1)
    primary = FakeExtractor(identity_version="local")
    luna = FakeExtractor(identity_version="luna", value_override="1999")
    terra = FakeExtractor(identity_version="terra", value_override="1999")

    summary = run_auto_check(
        _config(tmp_path, 1.0),
        primary,
        luna,
        terra,
    )

    assert len(primary.calls) == 1
    assert luna.calls == []
    assert terra.calls == []
    assert summary["execution"]["secondary_candidates"] == 0
    assert summary["execution"]["secondary_model_calls"] == 0
    assert summary["execution"]["terra_model_calls"] == 0
    assert summary["path_judgments"]["supported"] == 1


def test_luna_and_terra_caches_are_reused_after_local_mismatch(tmp_path: Path) -> None:
    _write_auto_check_dataset(tmp_path, query_count=1)
    config = _config(tmp_path, 1.0)
    run_auto_check(
        config,
        FakeExtractor(identity_version="local", value_override="1999"),
        FakeExtractor(identity_version="luna"),
        FakeExtractor(identity_version="terra"),
    )
    primary = FakeExtractor(identity_version="local", value_override="1999")
    luna = FakeExtractor(identity_version="luna")
    terra = FakeExtractor(identity_version="terra")

    summary = run_auto_check(config, primary, luna, terra)

    assert primary.calls == []
    assert luna.calls == []
    assert terra.calls == []
    assert summary["execution"]["primary_cache_hits"] == 1
    assert summary["execution"]["secondary_cache_hits"] == 1
    assert summary["execution"]["terra_cache_hits"] == 1
    assert summary["execution"]["model_calls"] == 0
    assert summary["path_judgments"]["supported"] == 1


def test_luna_concurrency_is_hard_limited_to_five(tmp_path: Path) -> None:
    _write_auto_check_dataset(tmp_path, query_count=1)
    config = _config(tmp_path, 1.0)
    config = AutoCheckConfig(**{**config.__dict__, "secondary_workers": 6})

    with pytest.raises(ValueError, match="between 1 and 5"):
        run_auto_check(config, FakeExtractor(), FakeExtractor("luna"))


def test_matching_empty_local_and_luna_results_skip_terra(tmp_path: Path) -> None:
    _write_auto_check_dataset(tmp_path, query_count=1)
    primary = FakeExtractor(identity_version="local", value_override="")
    luna = FakeExtractor(identity_version="luna", value_override="")
    terra = FakeExtractor(identity_version="terra")

    summary = run_auto_check(_config(tmp_path, 1.0), primary, luna, terra)

    assert len(luna.calls) == 1
    assert terra.calls == []
    assert summary["execution"]["terra_candidates"] == 0
    assert summary["path_judgments"]["insufficient"] == 1
    row = json.loads(
        Path(summary["artifacts"]["path_reviews"]).read_text(encoding="utf-8")
    )
    assert row["luna_agrees_with_local"] is True
    assert row["decision_source"] == "local_luna_consensus"


def test_any_local_luna_result_disagreement_runs_terra(tmp_path: Path) -> None:
    _write_auto_check_dataset(tmp_path, query_count=1)
    primary = FakeExtractor(identity_version="local", value_override="1999")
    luna = FakeExtractor(identity_version="luna", value_override="")
    terra = FakeExtractor(identity_version="terra")

    summary = run_auto_check(_config(tmp_path, 1.0), primary, luna, terra)

    assert len(luna.calls) == 1
    assert len(terra.calls) == 1
    assert summary["path_judgments"]["supported"] == 1


def test_terra_failure_is_incomplete_and_fail_closed(tmp_path: Path) -> None:
    _write_auto_check_dataset(tmp_path, query_count=1)
    primary = FakeExtractor(identity_version="local", value_override="")
    luna = FakeExtractor(identity_version="luna")
    terra = BrokenExtractor("terra")

    summary = run_auto_check(_config(tmp_path, 1.0), primary, luna, terra)

    assert terra.calls == 1
    assert summary["execution"]["complete"] is False
    assert summary["execution"]["failed_attribute_reviews"] == 1
    assert summary["path_judgments"]["reviewed"] == 0
    row = json.loads(
        Path(summary["artifacts"]["path_reviews"]).read_text(encoding="utf-8")
    )
    assert row["review_complete"] is False
    assert row["decision_source"] == "terra_adjudication_incomplete"
    assert row["recommended_action"] == "manual_review_checker_incomplete"


def test_local_base_url_validation() -> None:
    assert validate_local_base_url("http://127.0.0.1:8001/v1/") == (
        "http://127.0.0.1:8001/v1"
    )
    with pytest.raises(ValueError, match="without credentials"):
        validate_local_base_url("http://user:secret@127.0.0.1:8001/v1")
