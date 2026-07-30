import argparse
import json
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import build_mm_joinability_dataset as joinability_dataset
import run_mm_joinability_dynamic_vllm as dynamic_vllm_runner
from build_mm_joinability_dataset import (
    ExtractionCache,
    ExtractionTask,
    ModelConcurrencyState,
    LocalAttributeExtractor,
    build_bridge_assets_parallel,
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
    select_best_qualified_column,
    tasks_requiring_model_analysis,
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


def test_prompt_version_invalidates_cache_after_image_prompt_changes():
    import build_mm_joinability_dataset as joinability_dataset

    assert joinability_dataset.PROMPT_VERSION == "entity_attribute_extraction_v3_short_empty_precompressed_image"


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
):
    column_names = ["Entity", "Bridge B", "Bridge C", *context_column_names]
    rows = []
    expected_values: dict[int, dict[str, str]] = {}
    assets = {}
    entity_to_assets = {}
    wiki_to_entity_id = {}
    for row_index in range(5):
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
            split="train",
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


def test_multi_attribute_queries_use_globally_disjoint_context_sides(
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
    assert len(query_tables) == 2
    assert len(target_tables) == 2
    assert len(qrels) == 2
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


def test_identical_visible_multi_attribute_queries_merge_with_multiple_qrels(
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
    assert {item["column_name"] for item in query_tables[0]["hidden_attributes"]} == {
        "Bridge B",
        "Bridge C",
    }
    assert set(query_tables[0]["target_table_ids"]) == {
        target["table_id"] for target in target_tables
    }


def test_multi_attribute_split_falls_back_to_best_column_when_context_is_too_narrow(
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
    assert len(query_tables) == len(target_tables) == len(qrels) == 1


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


def test_query_rows_per_table_defaults_to_five(tmp_path):
    args = joinability_dataset.parse_args(
        ["--input_dir", str(tmp_path), "--output_dir", str(tmp_path / "out")]
    )

    assert args.query_rows_per_table == 5


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

    assert attrs == [{"name": "State", "value": "Alabama", "evidence": "BAMA"}]


def test_normalize_extracted_attributes_drops_placeholders_and_non_candidates():
    payload = {
        "attributes": [
            {"name": "<attribute>", "value": "<value>", "evidence": "<quote>"},
            {"name": "Wrong", "value": "Ignored", "evidence": "outside candidate list"},
            {"name": "Year", "value": "2008", "evidence": "October 20, 2008"},
        ]
    }

    assert normalize_extracted_attributes(payload, ["Year"]) == [
        {"name": "Year", "value": "2008", "evidence": "October 20, 2008"}
    ]


def test_normalize_extracted_attributes_can_require_entity_connection_evidence():
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

    assert normalize_extracted_attributes(payload, ["State"], require_connection_evidence=True) == [
        {
            "name": "State",
            "value": "Alabama",
            "evidence": "BAMA",
            "connection_evidence": "The jersey identifies the Alabama team.",
        }
    ]


def test_chat_payload_disables_qwen_thinking_without_prompt_text(monkeypatch):
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
        entity_wiki_title="Alpha",
        candidate_attributes=["State"],
    )
    extractor.chat(base_url="http://localhost:8001/v1", model="Qwen3.5-9B", api_key=None, messages=[])

    assert "Thinking Process" not in prompt
    assert "connection_evidence" in prompt
    assert "Do not rely on Wikipedia page provenance" in prompt
    assert captured["json"]["chat_template_kwargs"] == {"enable_thinking": False}


def test_extraction_prompt_requires_short_empty_json_response():
    extractor = LocalAttributeExtractor(_extractor_args())

    prompt = extractor.extraction_prompt(
        entity_text="Alpha",
        entity_wiki_title="Alpha",
        candidate_attributes=["State"],
    )

    assert 'If no candidate attribute is directly supported, return exactly {"attributes":[]}' in prompt


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


def test_image_extraction_keeps_image_url_but_sanitizes_entity_cell_url(monkeypatch):
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

    image_url = "https://upload.wikimedia.org/example/large-image.jpg"
    extractor.extract(
        asset={
            "asset_id": "img_1",
            "asset_type": "image",
            "image_url": image_url,
        },
        entity={
            "cell_text": "https://example.com/entity/" + "a" * 200,
            "wiki_title": "Alpha",
        },
        candidate_attributes=["State"],
    )

    user_content = captured["json"]["messages"][1]["content"]
    assert user_content[0]["type"] == "text"
    assert "Entity display text: [url]" in user_content[0]["text"]
    assert "example.com/entity" not in user_content[0]["text"]
    assert user_content[1] == {"type": "image_url", "image_url": {"url": image_url}}


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
    assert updated["attributes"] == [{"name": "State", "value": "Alabama", "evidence": "BAMA"}]


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
            captured["stdout"] = kwargs["stdout"]
            captured["stderr"] = kwargs["stderr"]
            captured["text"] = kwargs["text"]
            captured["start_new_session"] = kwargs["start_new_session"]

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
        "server_started:Qwen3-VL-8B-Thinking",
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
