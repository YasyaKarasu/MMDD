import argparse
import hashlib
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
    marker_matches_run,
    parse_args as parse_dynamic_vllm_args,
    start_server,
    write_endpoint_file,
    write_ready_marker,
    wait_for_any_marker_or_builder_exit,
    wait_for_marker_or_builder_exit,
)


def strict_start_marker(
    *,
    run_fingerprint: str,
    text_tasks: int,
    image_tasks: int,
    text_jobset: str = "text-v1",
    image_jobset: str = "image-v1",
) -> dict:
    payload = {
        "stage": "wdc200k_model_start",
        "schema_version": "wdc200k-model-markers-v1",
        "status": "model_cache_ready_to_start",
        "run_fingerprint": run_fingerprint,
        "text_jobset_fingerprint": text_jobset,
        "image_jobset_fingerprint": image_jobset,
        "text_task_count": text_tasks,
        "image_task_count": image_tasks,
        "upstream_manifests": [],
    }
    payload["start_fingerprint"] = hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    payload["timestamp"] = 0.0
    return payload


def strict_done_marker(
    *,
    model_kind: str,
    run_fingerprint: str,
    start_fingerprint: str,
    text_tasks: int = 1,
    image_tasks: int = 1,
    text_jobset: str = "text-v1",
    image_jobset: str = "image-v1",
) -> dict:
    task_count = text_tasks if model_kind == "text" else image_tasks
    jobset = text_jobset if model_kind == "text" else image_jobset
    return {
        "stage": "wdc200k_model_done",
        "schema_version": "wdc200k-model-markers-v1",
        "status": f"{model_kind}_model_cache_precomputed",
        "model_kind": model_kind,
        "task_count": task_count,
        f"{model_kind}_task_count": task_count,
        "jobset_fingerprint": jobset,
        "run_fingerprint": run_fingerprint,
        "text_jobset_fingerprint": text_jobset,
        "image_jobset_fingerprint": image_jobset,
        "text_task_count": text_tasks,
        "image_task_count": image_tasks,
        "start_fingerprint": start_fingerprint,
        "timestamp": 0.0,
    }


def test_prompt_version_invalidates_cache_after_image_prompt_changes():
    import build_mm_joinability_dataset as joinability_dataset

    assert joinability_dataset.PROMPT_VERSION == "entity_attribute_extraction_v3_short_empty_precompressed_image"


def test_select_best_qualified_column_uses_highest_recovery_ratio():
    qualified = [
        {"column_index": 1, "recovered_value_ratio": 0.75},
        {"column_index": 2, "recovered_value_ratio": 1.0},
        {"column_index": 3, "recovered_value_ratio": 0.8},
    ]

    assert select_best_qualified_column(qualified) == [qualified[1]]


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


def test_select_query_rows_uses_recovery_quota_then_failures():
    assert select_query_source_rows(
        source_row_order=[0, 1, 2, 3, 4, 5],
        recovered_source_rows={0, 2, 4},
        query_rows_per_table=5,
        required_recovered_rows=3,
    ) == [0, 2, 4, 1, 3]


def test_select_query_rows_uses_extra_recoveries_when_failures_are_exhausted():
    assert select_query_source_rows(
        source_row_order=[0, 1, 2, 3, 4, 5],
        recovered_source_rows={0, 1, 2, 3, 4},
        query_rows_per_table=5,
        required_recovered_rows=3,
    ) == [0, 1, 2, 5, 3]


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


def test_remote_text_and_image_requests_keep_endpoint_model_and_key_separate(monkeypatch):
    calls = []

    class Response:
        status_code = 200
        text = ""

        def raise_for_status(self):
            return None

        def json(self):
            return {"choices": [{"message": {"content": '{"attributes":[]}'}}]}

    def fake_post(url, headers, json, timeout):
        calls.append((url, headers, json))
        return Response()

    monkeypatch.setattr("build_mm_joinability_dataset.requests.post", fake_post)
    extractor = LocalAttributeExtractor(
        _extractor_args(
            text_model_base_url="https://text.example.test/openai/v1/",
            text_model_name="served-text",
            text_model_api_key="text-secret",
            image_model_base_url="https://image.example.test/openai/v1/",
            image_model_name="served-image",
            image_model_api_key="image-secret",
        )
    )

    extractor.extract(
        {"asset_id": "txt", "asset_type": "text", "content": "evidence"},
        {"cell_text": "Alpha", "wiki_title": "Alpha"},
        ["State"],
    )
    extractor.extract(
        {"asset_id": "img", "asset_type": "image", "image_url": "https://assets.test/a.jpg"},
        {"cell_text": "Alpha", "wiki_title": "Alpha"},
        ["State"],
    )

    assert [(url, headers["Authorization"], payload["model"]) for url, headers, payload in calls] == [
        ("https://text.example.test/openai/v1/chat/completions", "Bearer text-secret", "served-text"),
        ("https://image.example.test/openai/v1/chat/completions", "Bearer image-secret", "served-image"),
    ]


def test_endpoint_readiness_checks_every_configured_url_with_modality_credentials(monkeypatch):
    calls = []

    class Response:
        status_code = 200
        text = ""

        def raise_for_status(self):
            return None

        def json(self):
            model = "served-text" if "text" in self.url else "served-image"
            return {"data": [{"id": model}]}

    def fake_get(url, headers, timeout):
        response = Response()
        response.url = url
        calls.append((url, headers.get("Authorization")))
        return response

    monkeypatch.setattr("build_mm_joinability_dataset.requests.get", fake_get)
    extractor = LocalAttributeExtractor(
        _extractor_args(
            text_model_base_url="https://text-a.test/v1/",
            text_model_base_urls=["https://text-b.test/v1"],
            text_model_name="served-text",
            text_model_api_key="text-key",
            image_model_base_url="https://image-a.test/v1/",
            image_model_base_urls=["https://image-b.test/v1"],
            image_model_name="served-image",
            image_model_api_key="image-key",
        )
    )

    extractor.ensure_endpoints_ready({"text", "image"}, timeout_seconds=0)

    assert calls == [
        ("https://text-a.test/v1/models", "Bearer text-key"),
        ("https://text-b.test/v1/models", "Bearer text-key"),
        ("https://image-a.test/v1/models", "Bearer image-key"),
        ("https://image-b.test/v1/models", "Bearer image-key"),
    ]


def test_endpoint_readiness_uses_explicit_poll_interval_not_chat_retry_sleep(monkeypatch):
    attempts = 0
    sleeps = []
    now = 100.0

    class Response:
        text = ""

        def __init__(self, status_code):
            self.status_code = status_code

        def json(self):
            return {"data": [{"id": "served-text"}]}

    def fake_get(url, headers, timeout):
        nonlocal attempts
        attempts += 1
        return Response(503 if attempts == 1 else 200)

    def fake_sleep(seconds):
        nonlocal now
        sleeps.append(seconds)
        now += seconds

    monkeypatch.setattr("build_mm_joinability_dataset.requests.get", fake_get)
    monkeypatch.setattr("build_mm_joinability_dataset.time.monotonic", lambda: now)
    monkeypatch.setattr("build_mm_joinability_dataset.time.sleep", fake_sleep)
    extractor = LocalAttributeExtractor(
        _extractor_args(
            text_model_base_url="https://text.test/v1",
            text_model_name="served-text",
            model_retry_sleep_seconds=99.0,
        )
    )

    extractor.ensure_endpoints_ready({"text"}, timeout_seconds=10.0, poll_seconds=0.25)

    assert attempts == 2
    assert sleeps == [0.25]


@pytest.mark.parametrize("poll_seconds", [-0.1, float("inf"), float("nan")])
def test_endpoint_readiness_rejects_invalid_poll_seconds(poll_seconds):
    extractor = LocalAttributeExtractor(_extractor_args())

    with pytest.raises(ValueError, match="poll_seconds"):
        extractor.ensure_endpoints_ready({"text"}, timeout_seconds=0, poll_seconds=poll_seconds)


def test_endpoint_readiness_bounds_each_get_by_remaining_deadline(monkeypatch):
    request_timeouts = []
    now = 10.0

    class Response:
        status_code = 200
        text = ""

        def json(self):
            return {"data": [{"id": "served-text"}]}

    def fake_get(url, headers, timeout):
        nonlocal now
        request_timeouts.append(timeout)
        now += 0.4
        return Response()

    monkeypatch.setattr("build_mm_joinability_dataset.requests.get", fake_get)
    monkeypatch.setattr("build_mm_joinability_dataset.time.monotonic", lambda: now)
    extractor = LocalAttributeExtractor(
        _extractor_args(
            text_model_base_url="https://text-a.test/v1",
            text_model_base_urls=["https://text-b.test/v1"],
            text_model_name="served-text",
            model_timeout_seconds=120.0,
        )
    )

    extractor.ensure_endpoints_ready({"text"}, timeout_seconds=1.0)

    assert request_timeouts == pytest.approx([1.0, 0.6])


def test_zero_readiness_timeout_still_uses_finite_positive_get_timeout(monkeypatch):
    request_timeouts = []

    class Response:
        status_code = 200
        text = ""

        def json(self):
            return {"data": [{"id": "served-text"}]}

    def fake_get(url, headers, timeout):
        request_timeouts.append(timeout)
        return Response()

    monkeypatch.setattr("build_mm_joinability_dataset.requests.get", fake_get)
    extractor = LocalAttributeExtractor(
        _extractor_args(
            text_model_base_url="https://text.test/v1",
            text_model_name="served-text",
            model_timeout_seconds=120.0,
        )
    )

    extractor.ensure_endpoints_ready({"text"}, timeout_seconds=0)

    assert len(request_timeouts) == 1
    assert 0 < request_timeouts[0] <= 1.0


@pytest.mark.parametrize(
    ("response_status", "payload", "raised", "expected_type"),
    [
        (200, {"data": [{"id": "wrong-model"}]}, None, RuntimeError),
        (401, {}, None, RuntimeError),
        (404, {}, None, RuntimeError),
        (500, {}, None, "transient"),
        (503, {}, None, "transient"),
        (200, ValueError("bad json"), None, RuntimeError),
        (None, None, "connection", "transient"),
        (None, None, "timeout", "transient"),
    ],
)
def test_endpoint_readiness_errors_are_classified_and_do_not_leak_keys(
    monkeypatch, response_status, payload, raised, expected_type
):
    secret = "never-show-this-key"

    class Response:
        status_code = response_status
        text = secret

        def raise_for_status(self):
            return None

        def json(self):
            if isinstance(payload, Exception):
                raise payload
            return payload

    def fake_get(url, headers, timeout):
        if raised == "connection":
            raise joinability_dataset.requests.exceptions.ConnectionError(secret)
        if raised == "timeout":
            raise joinability_dataset.requests.exceptions.Timeout(secret)
        return Response()

    monkeypatch.setattr("build_mm_joinability_dataset.requests.get", fake_get)
    extractor = LocalAttributeExtractor(
        _extractor_args(
            text_model_base_url="https://remote.test/v1/",
            text_model_name="served-text",
            text_model_api_key=secret,
        )
    )
    if expected_type == "transient":
        expected_type = joinability_dataset.TransientModelEndpointError

    with pytest.raises(expected_type) as caught:
        extractor.ensure_endpoints_ready({"text"}, timeout_seconds=0)

    assert type(caught.value) is expected_type
    assert "text" in str(caught.value)
    assert "https://remote.test/v1" in str(caught.value)
    assert secret not in str(caught.value)


def test_model_api_key_precedence(monkeypatch):
    monkeypatch.setenv("MMDD_TEXT_MODEL_API_KEY", "text-env")
    monkeypatch.setenv("MMDD_IMAGE_MODEL_API_KEY", "image-env")
    monkeypatch.setenv("VLLM_API_KEY", "shared-env")
    explicit = LocalAttributeExtractor(
        _extractor_args(text_model_api_key="text-cli", image_model_api_key="image-cli")
    )
    modality_env = LocalAttributeExtractor(_extractor_args())
    monkeypatch.delenv("MMDD_TEXT_MODEL_API_KEY")
    monkeypatch.delenv("MMDD_IMAGE_MODEL_API_KEY")
    shared_env = LocalAttributeExtractor(_extractor_args())
    monkeypatch.delenv("VLLM_API_KEY")
    no_key = LocalAttributeExtractor(_extractor_args())

    assert (explicit.text_model_api_key, explicit.image_model_api_key) == ("text-cli", "image-cli")
    assert (modality_env.text_model_api_key, modality_env.image_model_api_key) == ("text-env", "image-env")
    assert (shared_env.text_model_api_key, shared_env.image_model_api_key) == ("shared-env", "shared-env")
    assert (no_key.text_model_api_key, no_key.image_model_api_key) == (None, None)


@pytest.mark.parametrize("failure", [429, 503, "connection", "timeout"])
def test_chat_raises_typed_transient_error_after_retries(monkeypatch, failure):
    attempts = 0

    class Response:
        status_code = failure
        text = "temporarily unavailable"

        def raise_for_status(self):
            return None

    def fake_post(url, headers, json, timeout):
        nonlocal attempts
        attempts += 1
        if failure == "connection":
            raise joinability_dataset.requests.exceptions.ConnectionError("offline")
        if failure == "timeout":
            raise joinability_dataset.requests.exceptions.Timeout("slow")
        return Response()

    monkeypatch.setattr("build_mm_joinability_dataset.requests.post", fake_post)
    extractor = LocalAttributeExtractor(_extractor_args(model_max_retries=1))

    with pytest.raises(joinability_dataset.TransientModelEndpointError):
        extractor.chat(
            base_url="https://text.test/v1",
            model="served-text",
            api_key=None,
            messages=[],
            model_kind="text",
        )

    assert attempts == 2


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
        runtime_dir=tmp_path / "runtime",
        text_server=text_server,
        primary_image_server=image_server,
        text_endpoints_file=tmp_path / "text_endpoints.txt",
        image_endpoints_file=tmp_path / "image_endpoints.txt",
        model_start_marker=tmp_path / "model_start.json",
        model_ready_marker=tmp_path / "model_ready.json",
        text_done_marker=tmp_path / "text_done.json",
        image_done_marker=tmp_path / "image_done.json",
        passthrough_args=[
            "--max_source_tables",
            "10",
            "--run_fingerprint",
            "run-v1",
        ],
    )

    assert "--precompute_model_cache" in command
    assert "--model_start_marker" in command
    assert "--model_ready_marker" in command
    assert "--model_text_done_marker" in command
    assert "--model_image_done_marker" in command
    assert "--text_model_base_urls_file" in command
    assert "--image_model_base_urls_file" in command
    assert command[command.index("--runtime_dir") + 1] == str(
        tmp_path / "runtime"
    )
    assert "http://127.0.0.1:8001/v1" in command
    assert "http://127.0.0.1:8000/v1" in command
    assert command[-4:] == [
        "--max_source_tables",
        "10",
        "--run_fingerprint",
        "run-v1",
    ]


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


def test_dynamic_vllm_forwarded_signal_grace_defaults_to_thirty_seconds():
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

    assert args.forwarded_signal_grace_seconds == 30.0


def test_dynamic_vllm_accepts_staged_run_fingerprint():
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
            "--run_fingerprint",
            "wdc-run-v1",
        ]
    )

    assert args.run_fingerprint == "wdc-run-v1"
    assert "--run_fingerprint" not in passthrough


def _capture_dynamic_builder_launch(
    monkeypatch,
    *,
    tmp_path: Path,
    runtime_dir: Path | None = None,
    work_dir: Path | None = None,
) -> tuple[list[str], dict[str, object]]:
    captured: dict[str, object] = {}

    class FakePopen:
        def __init__(self, command, **kwargs):
            self.pid = 12345
            self._poll = None
            captured["command"] = command
            captured["kwargs"] = kwargs
            marker = Path(command[command.index("--model_start_marker") + 1])
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text(
                json.dumps(
                    strict_start_marker(
                        run_fingerprint="",
                        text_tasks=0,
                        image_tasks=0,
                    )
                ),
                encoding="utf-8",
            )

        def poll(self):
            return self._poll

        def wait(self, timeout=None):
            self._poll = 0
            return 0

    monkeypatch.setattr(
        "run_mm_joinability_dynamic_vllm.subprocess.Popen",
        FakePopen,
    )
    argv = [
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
        "--builder_script",
        "/repo/scripts/build_wdc200k_mm_joinability_dataset.py",
    ]
    if runtime_dir is not None:
        argv.extend(["--runtime_dir", str(runtime_dir)])
    if work_dir is not None:
        argv.extend(["--work_dir", str(work_dir)])

    assert dynamic_vllm_main(argv) == 0
    return captured["command"], captured["kwargs"]


def test_dynamic_vllm_default_runtime_is_work_runtime(
    monkeypatch,
    tmp_path,
):
    command, popen_kwargs = _capture_dynamic_builder_launch(
        monkeypatch,
        tmp_path=tmp_path,
    )
    output_dir = tmp_path / "output"
    runtime_dir = tmp_path / "work_wdc_200k" / "runtime"
    runtime_options = {
        "--text_model_base_urls_file": "text_endpoints.txt",
        "--image_model_base_urls_file": "image_endpoints.txt",
        "--model_start_marker": "model_start.json",
        "--model_ready_marker": "model_ready.json",
        "--model_text_done_marker": "text_done.json",
        "--model_image_done_marker": "image_done.json",
    }

    assert command[1] == "/repo/scripts/build_wdc200k_mm_joinability_dataset.py"
    assert command[command.index("--runtime_dir") + 1] == str(runtime_dir)
    for option, filename in runtime_options.items():
        path = Path(command[command.index(option) + 1])
        assert path == runtime_dir / filename
        assert not path.is_relative_to(output_dir)
    assert popen_kwargs.get("stdout") is None
    assert popen_kwargs["start_new_session"] is True


def test_dynamic_vllm_default_builder_is_wdc200k() -> None:
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
        ]
    )

    assert Path(args.builder_script).name == (
        "build_wdc200k_mm_joinability_dataset.py"
    )
    assert passthrough == []


def test_dynamic_vllm_help_names_staged_wdc200k_builder() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "run_mm_joinability_dynamic_vllm.py"),
            "--help",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "build_wdc200k_mm_joinability_dataset.py" in completed.stdout
    assert "run build_mm_joinability_dataset.py" not in completed.stdout


def test_dynamic_vllm_preserves_required_explicit_runtime_dir(
    monkeypatch,
    tmp_path,
):
    work_dir = tmp_path / "explicit-work"
    explicit_runtime = work_dir / "runtime"
    command, _popen_kwargs = _capture_dynamic_builder_launch(
        monkeypatch,
        tmp_path=tmp_path,
        runtime_dir=explicit_runtime,
        work_dir=work_dir,
    )

    assert command[command.index("--runtime_dir") + 1] == str(
        explicit_runtime
    )
    for option in (
        "--text_model_base_urls_file",
        "--image_model_base_urls_file",
        "--model_start_marker",
        "--model_ready_marker",
        "--model_text_done_marker",
        "--model_image_done_marker",
    ):
        assert Path(command[command.index(option) + 1]).parent == explicit_runtime


def test_dynamic_vllm_rejects_runtime_outside_work_before_any_write(
    tmp_path,
):
    runtime = tmp_path / "outside-runtime"
    with pytest.raises(ValueError, match="equal work_dir/runtime"):
        dynamic_vllm_main(
            [
                "--input_dir",
                str(tmp_path / "input"),
                "--output_dir",
                str(tmp_path / "output"),
                "--work_dir",
                str(tmp_path / "work"),
                "--runtime_dir",
                str(runtime),
                "--text_model_path",
                "/models/text",
                "--image_model_path",
                "/models/vl",
            ]
        )
    assert not runtime.exists()


@pytest.mark.parametrize("grace_value", ["-0.01", "nan", "inf", "-inf"])
def test_dynamic_vllm_forwarded_signal_grace_rejects_invalid_before_writes_or_spawns(
    monkeypatch,
    tmp_path,
    grace_value,
):
    import run_mm_joinability_dynamic_vllm as runner

    side_effects = []
    runtime = tmp_path / "work_wdc_200k" / "runtime"

    def record_write(*_args, **_kwargs):
        side_effects.append("write")

    class ForbiddenPopen:
        def __init__(self, *_args, **_kwargs):
            side_effects.append("spawn")
            raise AssertionError("process spawned before grace validation")

    monkeypatch.setattr(runner, "write_endpoint_file", record_write)
    monkeypatch.setattr(runner.subprocess, "Popen", ForbiddenPopen)

    with pytest.raises(ValueError, match="must be"):
        dynamic_vllm_main(
            [
                "--input_dir",
                str(tmp_path / "input"),
                "--output_dir",
                str(tmp_path / "output"),
                "--text_model_path",
                "/models/text",
                "--image_model_path",
                "/models/vl",
                f"--forwarded_signal_grace_seconds={grace_value}",
            ]
        )

    assert side_effects == []
    assert not runtime.exists()


@pytest.mark.parametrize(
    "option",
    [
        "--runtime_dir=escape",
        "--text_model_base_url=http://escape/v1",
        "--text_model_base_urls_file=escape",
        "--text_model_name=escape",
        "--image_model_base_url=http://escape/v1",
        "--image_model_name=escape",
        "--run_fingerprint=escape",
        "--model_start_marker=escape",
    ],
)
def test_dynamic_vllm_rejects_reserved_builder_passthrough(
    tmp_path,
    option,
):
    with pytest.raises(ValueError, match="controlled by"):
        dynamic_vllm_main(
            [
                "--input_dir",
                str(tmp_path / "input"),
                "--output_dir",
                str(tmp_path / "output"),
                "--text_model_path",
                "/models/text",
                "--image_model_path",
                "/models/vl",
                "--",
                option,
            ]
        )


def test_endpoint_guard_failure_preserves_existing_file(tmp_path):
    path = tmp_path / "endpoints.txt"
    path.write_text("old\n", encoding="utf-8")
    calls = 0

    def fail_commit(_path, _estimated):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise RuntimeError("reserve exhausted")

    with pytest.raises(RuntimeError, match="reserve exhausted"):
        write_endpoint_file(
            path,
            ["http://new.example/v1"],
            pre_write_guard=fail_commit,
        )
    assert path.read_text(encoding="utf-8") == "old\n"
    assert not path.with_suffix(".txt.tmp").exists()


def test_ready_marker_guard_failure_preserves_existing_marker(tmp_path):
    path = tmp_path / "ready.json"
    path.write_text('{"old": true}\n', encoding="utf-8")
    calls = 0

    def fail_commit(_path, _estimated):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise RuntimeError("reserve exhausted")

    with pytest.raises(RuntimeError, match="reserve exhausted"):
        write_ready_marker(
            path,
            run_fingerprint="run-v1",
            text_jobset_fingerprint="text-v1",
            image_jobset_fingerprint="image-v1",
            text_task_count=1,
            image_task_count=1,
            start_fingerprint="a" * 64,
            pre_write_guard=fail_commit,
        )
    assert json.loads(path.read_text(encoding="utf-8")) == {"old": True}
    assert not path.with_suffix(".json.tmp").exists()


def test_real_builder_and_default_runner_share_strict_marker_contract(
    tmp_path,
):
    import run_mm_joinability_dynamic_vllm as runner

    args = joinability_dataset.parse_args(
        [
            "--input_dir",
            str(tmp_path / "input"),
            "--output_dir",
            str(tmp_path / "output"),
            "--run_fingerprint",
            "run-real-v1",
            "--text_model_name",
            "text-real-v1",
            "--image_model_name",
            "image-real-v1",
        ]
    )
    tasks_by_kind = {
        "text": [_task("text", "one")],
        "image": [_task("image", "two")],
    }
    context = joinability_dataset.build_model_marker_context(
        args=args,
        tasks_by_kind=tasks_by_kind,
        upstream_identities=[
            {
                "path": "/data/source-00000.jsonl",
                "sha256": "a" * 64,
            }
        ],
    )
    changed_context = joinability_dataset.build_model_marker_context(
        args=args,
        tasks_by_kind={
            **tasks_by_kind,
            "text": [_task("text", "changed")],
        },
        upstream_identities=[
            {
                "path": "/data/source-00000.jsonl",
                "sha256": "a" * 64,
            }
        ],
    )
    assert (
        changed_context.text_jobset_fingerprint
        != context.text_jobset_fingerprint
    )

    start = tmp_path / "start.json"
    ready = tmp_path / "ready.json"
    text_done = tmp_path / "text-done.json"
    image_done = tmp_path / "image-done.json"
    joinability_dataset.write_model_start_marker(
        str(start),
        context=context,
    )
    assert runner.marker_matches_run(
        start,
        "run-real-v1",
        expected_status="model_cache_ready_to_start",
    )
    assert runner.read_pending_model_task_count(start) == 2

    runner.write_ready_marker(
        ready,
        run_fingerprint=context.run_fingerprint,
        text_jobset_fingerprint=context.text_jobset_fingerprint,
        image_jobset_fingerprint=context.image_jobset_fingerprint,
        text_task_count=context.text_task_count,
        image_task_count=context.image_task_count,
        start_fingerprint=context.start_fingerprint,
    )
    assert joinability_dataset.model_ready_marker_matches(
        str(ready),
        context=context,
    )

    for kind, path in (("text", text_done), ("image", image_done)):
        joinability_dataset.write_model_done_marker(
            str(path),
            model_kind=kind,
            task_count=1,
            context=context,
        )
        assert runner.marker_matches_run(
            path,
            context.run_fingerprint,
            context.fingerprint_for(kind),
            expected_status=f"{kind}_model_cache_precomputed",
            expected_model_kind=kind,
            expected_task_count=1,
            expected_text_jobset_fingerprint=(
                context.text_jobset_fingerprint
            ),
            expected_image_jobset_fingerprint=(
                context.image_jobset_fingerprint
            ),
            expected_text_task_count=context.text_task_count,
            expected_image_task_count=context.image_task_count,
            expected_start_fingerprint=context.start_fingerprint,
        )
    assert not list(tmp_path.glob("*.tmp"))


def test_start_server_passes_vllm_output_through_to_tmux(monkeypatch):
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

    start_server(spec)

    assert captured["command"] == spec.command()
    assert captured["stdout"] is None
    assert captured["stderr"] is None
    assert captured["text"] is True
    assert captured["start_new_session"] is True


def test_dynamic_marker_wait_ignores_stale_run_fingerprint(tmp_path):
    marker = tmp_path / "start.json"
    marker.write_text(
        json.dumps(
            strict_start_marker(
                run_fingerprint="stale",
                text_tasks=1,
                image_tasks=1,
            )
        ),
        encoding="utf-8",
    )

    class RunningBuilder:
        @staticmethod
        def poll():
            return None

    def publish_current():
        threading.Event().wait(0.05)
        marker.write_text(
            json.dumps(
                strict_start_marker(
                    run_fingerprint="current",
                    text_tasks=1,
                    image_tasks=1,
                )
            ),
            encoding="utf-8",
        )

    publisher = threading.Thread(target=publish_current)
    publisher.start()
    wait_for_marker_or_builder_exit(
        marker=marker,
        builder=RunningBuilder(),
        timeout_seconds=1,
        poll_seconds=0.01,
        expected_run_fingerprint="current",
    )
    publisher.join(timeout=1)

    assert not publisher.is_alive()


def test_dynamic_done_wait_ignores_stale_marker(tmp_path):
    marker = tmp_path / "text-done.json"
    start_fingerprint = "a" * 64
    marker.write_text(
        json.dumps(
            strict_done_marker(
                model_kind="text",
                run_fingerprint="stale",
                start_fingerprint=start_fingerprint,
            )
        ),
        encoding="utf-8",
    )

    class RunningBuilder:
        @staticmethod
        def poll():
            return None

    def publish_current():
        threading.Event().wait(0.05)
        marker.write_text(
            json.dumps(
                strict_done_marker(
                    model_kind="text",
                    run_fingerprint="current",
                    start_fingerprint=start_fingerprint,
                )
            ),
            encoding="utf-8",
        )

    publisher = threading.Thread(target=publish_current)
    publisher.start()
    completed = wait_for_any_marker_or_builder_exit(
        markers={"text": marker},
        builder=RunningBuilder(),
        timeout_seconds=1,
        poll_seconds=0.01,
        expected_run_fingerprint="current",
    )
    publisher.join(timeout=1)

    assert completed == {"text"}


def test_dynamic_done_wait_ignores_stale_jobset_fingerprint(tmp_path):
    marker = tmp_path / "image-done.json"
    start_fingerprint = "a" * 64
    marker.write_text(
        json.dumps(
            strict_done_marker(
                model_kind="image",
                run_fingerprint="current",
                start_fingerprint=start_fingerprint,
                image_jobset="old",
            )
        ),
        encoding="utf-8",
    )

    class RunningBuilder:
        @staticmethod
        def poll():
            return None

    def publish_current():
        threading.Event().wait(0.05)
        marker.write_text(
            json.dumps(
                strict_done_marker(
                    model_kind="image",
                    run_fingerprint="current",
                    start_fingerprint=start_fingerprint,
                    image_jobset="image-v2",
                )
            ),
            encoding="utf-8",
        )

    publisher = threading.Thread(target=publish_current)
    publisher.start()
    completed = wait_for_any_marker_or_builder_exit(
        markers={"image": marker},
        builder=RunningBuilder(),
        timeout_seconds=1,
        poll_seconds=0.01,
        expected_run_fingerprint="current",
        expected_jobset_fingerprints={"image": "image-v2"},
    )
    publisher.join(timeout=1)

    assert completed == {"image"}


def test_dynamic_done_marker_requires_exact_status_kind_and_count(tmp_path):
    marker = tmp_path / "text-done.json"
    marker.write_text(
        '{"status":"image_model_cache_precomputed","model_kind":"image",'
        '"task_count":2,"run_fingerprint":"run-v1",'
        '"jobset_fingerprint":"text-v1"}',
        encoding="utf-8",
    )

    assert not marker_matches_run(
        marker,
        "run-v1",
        "text-v1",
        expected_status="text_model_cache_precomputed",
        expected_model_kind="text",
        expected_task_count=1,
    )


def test_dynamic_markers_reject_legacy_and_ready_binds_start(
    tmp_path,
):
    import run_mm_joinability_dynamic_vllm as runner

    legacy = tmp_path / "legacy.json"
    legacy.write_text(
        json.dumps(
            {
                "status": "text_model_cache_precomputed",
                "model_kind": "text",
                "task_count": 1,
                "run_fingerprint": "run-v1",
                "jobset_fingerprint": "text-v1",
            }
        ),
        encoding="utf-8",
    )
    assert not marker_matches_run(
        legacy,
        "run-v1",
        "text-v1",
        expected_status="text_model_cache_precomputed",
        expected_model_kind="text",
        expected_task_count=1,
    )

    ready = tmp_path / "ready.json"
    runner.write_ready_marker(
        ready,
        run_fingerprint="run-v1",
        text_jobset_fingerprint="text-v1",
        image_jobset_fingerprint="image-v1",
        text_task_count=2,
        image_task_count=3,
        start_fingerprint="a" * 64,
    )
    payload = json.loads(ready.read_text(encoding="utf-8"))
    assert payload["stage"] == "wdc200k_model_ready"
    assert payload["schema_version"] == "wdc200k-model-markers-v1"
    assert payload["model_kind"] == "text+image"
    assert payload["start_fingerprint"] == "a" * 64
    assert not ready.with_suffix(".json.tmp").exists()


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
                start_payload = strict_start_marker(
                    run_fingerprint="",
                    text_tasks=1,
                    image_tasks=1,
                )
                marker.write_text(
                    json.dumps(start_payload),
                    encoding="utf-8",
                )
                Path(
                    command[command.index("--model_text_done_marker") + 1]
                ).write_text(
                    json.dumps(
                        strict_done_marker(
                            model_kind="text",
                            run_fingerprint="",
                            start_fingerprint=start_payload[
                                "start_fingerprint"
                            ],
                        )
                    ),
                    encoding="utf-8",
                )
                Path(
                    command[command.index("--model_image_done_marker") + 1]
                ).write_text(
                    json.dumps(
                        strict_done_marker(
                            model_kind="image",
                            run_fingerprint="",
                            start_fingerprint=start_payload[
                                "start_fingerprint"
                            ],
                        )
                    ),
                    encoding="utf-8",
                )
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
    assert events[0] == "builder_started"
    assert events[1].startswith("server_started:")


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
                    json.dumps(
                        strict_start_marker(
                            run_fingerprint="",
                            text_tasks=0,
                            image_tasks=0,
                        )
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


def test_dynamic_vllm_signal_path_stops_builder_and_all_started_models(
    monkeypatch,
    tmp_path,
):
    import run_mm_joinability_dynamic_vllm as runner

    started = []
    stopped = []

    class FakePopen:
        def __init__(self, command, **_kwargs):
            self.command = command
            self.pid = 12345 + len(started)
            self._poll = None
            self.role = (
                "builder"
                if command[0] == "/usr/bin/python"
                else command[command.index("--served-model-name") + 1]
            )
            started.append(self)
            if self.role == "builder":
                marker = Path(
                    command[command.index("--model_start_marker") + 1]
                )
                marker.parent.mkdir(parents=True, exist_ok=True)
                marker.write_text(
                    json.dumps(
                        strict_start_marker(
                            run_fingerprint="",
                            text_tasks=1,
                            image_tasks=1,
                        )
                    ),
                    encoding="utf-8",
                )

        def poll(self):
            return self._poll

    monkeypatch.setattr(runner.subprocess, "Popen", FakePopen)
    monkeypatch.setattr(
        runner,
        "wait_for_server",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        runner,
        "wait_for_any_marker_or_builder_exit",
        lambda **_kwargs: (_ for _ in ()).throw(KeyboardInterrupt()),
    )
    monkeypatch.setattr(
        runner,
        "stop_process",
        lambda proc, **_kwargs: (
            stopped.append(proc.role) if proc is not None else None
        ),
    )

    with pytest.raises(KeyboardInterrupt):
        dynamic_vllm_main(
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

    assert set(stopped) == {
        "builder",
        "Qwen3.5-9B",
        "Qwen3-VL-8B-Thinking",
    }


def test_dynamic_vllm_masks_signal_handlers_during_best_effort_cleanup(
    monkeypatch,
    tmp_path,
):
    import run_mm_joinability_dynamic_vllm as runner

    started = []
    stopped = []
    events = []
    current_handlers = {}

    class FakePopen:
        def __init__(self, command, **_kwargs):
            self.command = command
            self.pid = 12345 + len(started)
            self._poll = None
            self.role = (
                "builder"
                if command[0] == "/usr/bin/python"
                else command[command.index("--served-model-name") + 1]
            )
            started.append(self)
            if self.role == "builder":
                marker = Path(
                    command[command.index("--model_start_marker") + 1]
                )
                marker.parent.mkdir(parents=True, exist_ok=True)
                marker.write_text(
                    json.dumps(
                        strict_start_marker(
                            run_fingerprint="",
                            text_tasks=1,
                            image_tasks=1,
                        )
                    ),
                    encoding="utf-8",
                )

        def poll(self):
            return self._poll

        def wait(self, timeout=None):
            raise subprocess.TimeoutExpired(self.command, timeout)

    def previous_handler(signum, _frame):
        events.append(("previous_handler", signum))

    def fake_signal(signum, handler):
        previous = current_handlers.get(signum, previous_handler)
        current_handlers[signum] = handler
        if handler == signal.SIG_IGN:
            events.append(("mask", signum))
        elif handler is previous_handler:
            events.append(("restore", signum))
        return previous

    def fake_stop(process, **_kwargs):
        stopped.append(process.role)
        events.append(("stop", process.role))
        if len(stopped) == 1:
            handler = current_handlers[signal.SIGTERM]
            if handler == signal.SIG_IGN:
                events.append(("ignored", signal.SIGTERM))
            else:
                handler(signal.SIGTERM, None)

    monkeypatch.setattr(runner.subprocess, "Popen", FakePopen)
    monkeypatch.setattr(runner.signal, "signal", fake_signal)
    monkeypatch.setattr(
        runner,
        "wait_for_server",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        runner,
        "wait_for_any_marker_or_builder_exit",
        lambda **_kwargs: (_ for _ in ()).throw(
            runner.ForwardedSignal(signal.SIGTERM)
        ),
    )
    monkeypatch.setattr(runner, "stop_process", fake_stop)

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

    assert code == 128 + signal.SIGTERM
    assert events.index(("mask", signal.SIGTERM)) < events.index(
        ("stop", "builder")
    )
    assert ("ignored", signal.SIGTERM) in events
    assert events.index(("restore", signal.SIGTERM)) > max(
        index
        for index, event in enumerate(events)
        if event[0] == "stop"
    )
    assert set(stopped) == {
        "builder",
        "Qwen3.5-9B",
        "Qwen3-VL-8B-Thinking",
    }


def test_dynamic_vllm_forwards_signal_to_every_live_process_group():
    import run_mm_joinability_dynamic_vllm as runner

    processes = [
        subprocess.Popen(
            [
                sys.executable,
                "-c",
                "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_DFL); time.sleep(30)",
            ],
            start_new_session=True,
        )
        for _index in range(2)
    ]
    try:
        runner.forward_signal_to_live_process_groups(signal.SIGTERM, processes)

        assert [process.wait(timeout=3) for process in processes] == [
            -signal.SIGTERM,
            -signal.SIGTERM,
        ]
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
            process.wait(timeout=3)


def _wait_for_test_path(path: Path, *, timeout_seconds: float = 3.0) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if path.exists():
            return
        time.sleep(0.01)
    raise AssertionError(f"timed out waiting for test path: {path}")


def test_dynamic_vllm_waits_for_signalled_builder_cleanup_without_fallback_signal(
    monkeypatch,
    tmp_path,
):
    import run_mm_joinability_dynamic_vllm as runner

    ready = tmp_path / "ready"
    received = tmp_path / "received"
    release = tmp_path / "release"
    cleaned = tmp_path / "cleaned"
    fallback = tmp_path / "fallback"
    child_code = "\n".join(
        [
            "import os, signal, sys, time",
            "from pathlib import Path",
            "ready, received, release, cleaned, fallback = map(Path, sys.argv[1:])",
            "def handle_sigterm(_signum, _frame):",
            "    fallback.write_text('sigterm', encoding='utf-8')",
            "    os._exit(143)",
            "def handle_sigint(_signum, _frame):",
            "    received.write_text('sigint', encoding='utf-8')",
            "    while not release.exists():",
            "        time.sleep(0.01)",
            "    time.sleep(0.1)",
            "    cleaned.write_text('complete', encoding='utf-8')",
            "    os._exit(0)",
            "signal.signal(signal.SIGTERM, handle_sigterm)",
            "signal.signal(signal.SIGINT, handle_sigint)",
            "ready.write_text('ready', encoding='utf-8')",
            "while True:",
            "    time.sleep(1)",
        ]
    )
    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            child_code,
            str(ready),
            str(received),
            str(release),
            str(cleaned),
            str(fallback),
        ],
        start_new_session=True,
    )
    real_killpg = runner.os.killpg
    sent_signals = []

    def tracking_killpg(pid, signum):
        sent_signals.append(signum)
        real_killpg(pid, signum)

    monkeypatch.setattr(runner.os, "killpg", tracking_killpg)
    try:
        _wait_for_test_path(ready)
        runner.forward_signal_to_live_process_groups(signal.SIGINT, [process])
        _wait_for_test_path(received)
        assert process.poll() is None

        release.write_text("release", encoding="utf-8")
        assert runner.wait_for_forwarded_process_exit(
            process,
            timeout_seconds=2.0,
        )
        runner.stop_process(process, timeout_seconds=0.1)

        assert process.returncode == 0
        assert cleaned.read_text(encoding="utf-8") == "complete"
        assert not fallback.exists()
        assert sent_signals == [signal.SIGINT]
    finally:
        if process.poll() is None:
            process.kill()
        process.wait(timeout=3)


def test_dynamic_vllm_forwarded_signal_grace_skips_absent_or_exited_process():
    import run_mm_joinability_dynamic_vllm as runner

    class ExitedProcess:
        def poll(self):
            return 17

        def wait(self, timeout=None):
            raise AssertionError(f"unexpected wait with timeout {timeout}")

    exited = ExitedProcess()

    assert runner.wait_for_forwarded_process_exit(
        None,
        timeout_seconds=1.0,
    )
    assert runner.wait_for_forwarded_process_exit(
        exited,
        timeout_seconds=1.0,
    )
    runner.stop_process(exited)


def _run_dynamic_forwarded_signal_grace_main(
    monkeypatch,
    tmp_path: Path,
    *,
    builder_grace_times_out: bool,
    builder_already_exited: bool = False,
    second_signal_during_grace: int | None = None,
    inject_signal_during_mask: int | None = None,
):
    import run_mm_joinability_dynamic_vllm as runner

    events = []
    started = []
    by_pid = {}
    installed_handlers = {}
    initial_mask = {signal.SIGUSR1}
    blocked_signals = set(initial_mask)
    mask_install_count = 0

    class FakePopen:
        def __init__(self, command, **_kwargs):
            self.command = command
            self.pid = 12345 + len(started)
            self._poll = None
            self.wait_timeouts = []
            self.role = (
                "builder"
                if command[0] == "/usr/bin/python"
                else command[command.index("--served-model-name") + 1]
            )
            started.append(self)
            by_pid[self.pid] = self
            if self.role == "builder":
                marker = Path(
                    command[command.index("--model_start_marker") + 1]
                )
                marker.parent.mkdir(parents=True, exist_ok=True)
                marker.write_text(
                    json.dumps(
                        strict_start_marker(
                            run_fingerprint="",
                            text_tasks=1,
                            image_tasks=1,
                        )
                    ),
                    encoding="utf-8",
                )

        def poll(self):
            return self._poll

        def wait(self, timeout=None):
            self.wait_timeouts.append(timeout)
            if self.role == "builder" and timeout == 7.5:
                events.append("builder_grace_wait")
                if second_signal_during_grace is not None:
                    installed_handlers[second_signal_during_grace](
                        second_signal_during_grace,
                        None,
                    )
                if builder_grace_times_out:
                    raise subprocess.TimeoutExpired(self.command, timeout)
            self._poll = 0
            return 0

    def fake_signal(signum, handler):
        nonlocal mask_install_count
        installed_handlers[signum] = handler
        if handler == signal.SIG_IGN:
            mask_install_count += 1
            events.append(("mask", signum))
            if mask_install_count == 2 and inject_signal_during_mask is not None:
                if inject_signal_during_mask in blocked_signals:
                    events.append(("deferred", inject_signal_during_mask))
                else:
                    installed_handlers[inject_signal_during_mask](
                        inject_signal_during_mask,
                        None,
                    )
        return signal.SIG_DFL

    def fake_pthread_sigmask(how, signals):
        requested = frozenset(signals)
        previous = frozenset(blocked_signals)
        events.append(("sigmask", how, requested))
        if how == signal.SIG_BLOCK:
            blocked_signals.update(requested)
        elif how == signal.SIG_SETMASK:
            blocked_signals.clear()
            blocked_signals.update(requested)
        return previous

    def fake_forward_signal(signum, processes):
        assert [process.role for process in processes if process is not None]
        events.append(("forward", signum))
        if builder_already_exited and signum == signal.SIGINT:
            builder = next(
                process for process in started if process.role == "builder"
            )
            builder._poll = 0

    def raise_forwarded_signal(**_kwargs):
        installed_handlers[signal.SIGINT](signal.SIGINT, None)

    def fake_killpg(pid, signum):
        events.append(("kill", by_pid[pid].role, signum))

    original_best_effort = runner.stop_processes_best_effort

    def recording_best_effort(processes):
        events.append("best_effort_stop")
        return original_best_effort(processes)

    monkeypatch.setattr(runner.subprocess, "Popen", FakePopen)
    monkeypatch.setattr(runner.signal, "signal", fake_signal)
    monkeypatch.setattr(runner.signal, "pthread_sigmask", fake_pthread_sigmask)
    monkeypatch.setattr(
        runner,
        "forward_signal_to_live_process_groups",
        fake_forward_signal,
    )
    monkeypatch.setattr(runner, "restore_signal_handlers", lambda _handlers: None)
    monkeypatch.setattr(runner, "wait_for_server", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        runner,
        "wait_for_any_marker_or_builder_exit",
        raise_forwarded_signal,
    )
    monkeypatch.setattr(runner.os, "killpg", fake_killpg)
    monkeypatch.setattr(
        runner,
        "stop_processes_best_effort",
        recording_best_effort,
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
            "--forwarded_signal_grace_seconds",
            "7.5",
        ]
    )
    builder = next(process for process in started if process.role == "builder")
    return code, events, builder, installed_handlers


def test_dynamic_vllm_forwarded_signal_grace_waits_before_best_effort_stop(
    monkeypatch,
    tmp_path,
):
    code, events, builder, _handlers = _run_dynamic_forwarded_signal_grace_main(
        monkeypatch,
        tmp_path,
        builder_grace_times_out=False,
    )

    assert code == 128 + signal.SIGINT
    assert events.index(("forward", signal.SIGINT)) < events.index(
        "builder_grace_wait"
    )
    assert events.index("builder_grace_wait") < events.index("best_effort_stop")
    mask_indices = [
        index
        for index, event in enumerate(events)
        if event[0] == "mask"
    ]
    assert len(mask_indices) == 6
    assert events.index("builder_grace_wait") < mask_indices[0]
    assert mask_indices[-1] < events.index("best_effort_stop")
    assert builder.wait_timeouts == [7.5]
    assert ("kill", "builder", signal.SIGTERM) not in events


def test_dynamic_vllm_forwarded_signal_grace_timeout_falls_back_to_existing_stop(
    monkeypatch,
    tmp_path,
):
    code, events, builder, _handlers = _run_dynamic_forwarded_signal_grace_main(
        monkeypatch,
        tmp_path,
        builder_grace_times_out=True,
    )

    assert code == 128 + signal.SIGINT
    assert events.index(("forward", signal.SIGINT)) < events.index(
        "builder_grace_wait"
    )
    assert events.index("builder_grace_wait") < events.index("best_effort_stop")
    assert events.index("best_effort_stop") < events.index(
        ("kill", "builder", signal.SIGTERM)
    )
    assert builder.wait_timeouts == [7.5, 30.0]


def test_dynamic_vllm_forwarded_signal_grace_skips_already_exited_builder_in_main(
    monkeypatch,
    tmp_path,
):
    code, events, builder, _handlers = _run_dynamic_forwarded_signal_grace_main(
        monkeypatch,
        tmp_path,
        builder_grace_times_out=False,
        builder_already_exited=True,
    )

    assert code == 128 + signal.SIGINT
    assert events.index(("forward", signal.SIGINT)) < events.index(
        "best_effort_stop"
    )
    assert "builder_grace_wait" not in events
    assert builder.wait_timeouts == []
    assert ("kill", "builder", signal.SIGTERM) not in events


def test_dynamic_vllm_second_forwarded_signal_interrupts_grace_and_keeps_first_exit_code(
    monkeypatch,
    tmp_path,
):
    code, events, builder, installed_handlers = (
        _run_dynamic_forwarded_signal_grace_main(
            monkeypatch,
            tmp_path,
            builder_grace_times_out=False,
            second_signal_during_grace=signal.SIGTERM,
        )
    )

    assert code == 128 + signal.SIGINT
    assert [event for event in events if event[0] == "forward"] == [
        ("forward", signal.SIGINT),
        ("forward", signal.SIGTERM),
    ]
    assert events.index("builder_grace_wait") < events.index(
        ("forward", signal.SIGTERM)
    )
    first_mask = next(event for event in events if event[0] == "mask")
    assert events.index(first_mask) < events.index(("forward", signal.SIGTERM))
    assert events.index(("forward", signal.SIGTERM)) < events.index(
        "best_effort_stop"
    )
    assert builder.wait_timeouts == [7.5, 30.0]
    assert set(installed_handlers.values()) == {signal.SIG_IGN}


def test_dynamic_vllm_recursive_signal_delivery_forwards_only_root_signal(
    monkeypatch,
):
    import run_mm_joinability_dynamic_vllm as runner

    installed = {}
    forwarded = []

    def fake_signal(signum, handler):
        installed[signum] = handler
        return signal.SIG_DFL

    def recursive_forward(signum, processes):
        forwarded.append((signum, tuple(processes)))
        if len(forwarded) == 1:
            installed[signum](signum, None)

    monkeypatch.setattr(runner.signal, "signal", fake_signal)
    monkeypatch.setattr(
        runner,
        "forward_signal_to_live_process_groups",
        recursive_forward,
    )
    live_processes = (object(),)
    runner.install_process_group_signal_handlers(lambda: live_processes)

    with pytest.raises(runner.ForwardedSignal) as raised:
        installed[signal.SIGINT](signal.SIGINT, None)

    assert raised.value.signum == signal.SIGINT
    assert forwarded == [(signal.SIGINT, live_processes)]


def test_dynamic_vllm_atomic_mask_defers_delivery_during_handler_install(
    monkeypatch,
    tmp_path,
):
    code, events, _builder, _handlers = (
        _run_dynamic_forwarded_signal_grace_main(
            monkeypatch,
            tmp_path,
            builder_grace_times_out=False,
            inject_signal_during_mask=signal.SIGINT,
        )
    )
    managed = frozenset(
        (signal.SIGTERM, signal.SIGHUP, signal.SIGINT)
    )
    prior_mask = frozenset({signal.SIGUSR1})
    block_event = ("sigmask", signal.SIG_BLOCK, managed)
    restore_event = ("sigmask", signal.SIG_SETMASK, prior_mask)
    block_index = events.index(block_event)
    restore_index = events.index(restore_event)
    mask_indices = [
        index
        for index, event in enumerate(events)
        if event[0] == "mask" and block_index < index < restore_index
    ]

    assert code == 128 + signal.SIGINT
    assert [event for event in events if event[0] == "forward"] == [
        ("forward", signal.SIGINT)
    ]
    assert ("deferred", signal.SIGINT) in events
    assert block_index < mask_indices[0]
    assert mask_indices[-1] < restore_index


def test_dynamic_vllm_atomic_mask_restores_previous_mask_after_install_error(
    monkeypatch,
):
    import run_mm_joinability_dynamic_vllm as runner

    managed = frozenset(
        (signal.SIGTERM, signal.SIGHUP, signal.SIGINT)
    )
    prior_mask = frozenset({signal.SIGUSR1})
    mask_calls = []

    def fake_pthread_sigmask(how, signals):
        requested = frozenset(signals)
        mask_calls.append((how, requested))
        if how == signal.SIG_BLOCK:
            return prior_mask
        return managed

    def fail_second_install(signum, _handler):
        if signum == signal.SIGHUP:
            raise RuntimeError("signal install failed")
        return signal.SIG_DFL

    monkeypatch.setattr(runner.signal, "pthread_sigmask", fake_pthread_sigmask)
    monkeypatch.setattr(runner.signal, "signal", fail_second_install)

    with pytest.raises(RuntimeError, match="signal install failed"):
        runner.mask_process_group_signals_for_cleanup()

    assert mask_calls == [
        (signal.SIG_BLOCK, managed),
        (signal.SIG_SETMASK, prior_mask),
    ]


def test_dynamic_vllm_installs_explicit_signal_handlers_and_restores_them(
    monkeypatch,
):
    import run_mm_joinability_dynamic_vllm as runner

    registrations = []
    forwarded = []

    def fake_signal(signum, handler):
        registrations.append((signum, handler))
        return f"previous-{signum}"

    monkeypatch.setattr(runner.signal, "signal", fake_signal)
    monkeypatch.setattr(
        runner,
        "forward_signal_to_live_process_groups",
        lambda signum, processes: forwarded.append(
            (signum, tuple(processes))
        ),
    )
    live_processes = (object(), object())

    previous = runner.install_process_group_signal_handlers(
        lambda: live_processes
    )
    installed = {
        signum: handler
        for signum, handler in registrations
    }

    assert set(installed) == {
        signal.SIGTERM,
        signal.SIGHUP,
        signal.SIGINT,
    }
    with pytest.raises(runner.ForwardedSignal) as raised:
        installed[signal.SIGHUP](signal.SIGHUP, None)
    assert raised.value.signum == signal.SIGHUP
    assert forwarded == [(signal.SIGHUP, live_processes)]

    runner.restore_signal_handlers(previous)

    assert registrations[-3:] == [
        (signum, f"previous-{signum}")
        for signum in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT)
    ]


def test_dynamic_vllm_cleanup_continues_after_one_stop_failure(monkeypatch):
    import run_mm_joinability_dynamic_vllm as runner

    attempted: list[int] = []

    class FakeProcess:
        def __init__(self, pid: int):
            self.pid = pid

    processes = [FakeProcess(101), FakeProcess(102), FakeProcess(103)]

    def fake_stop(process):
        attempted.append(process.pid)
        if process.pid == 101:
            raise RuntimeError("first process would not stop")

    monkeypatch.setattr(runner, "stop_process", fake_stop)

    errors = runner.stop_processes_best_effort(processes)

    assert attempted == [101, 102, 103]
    assert len(errors) == 1
    assert errors[0][0] is processes[0]
    assert str(errors[0][1]) == "first process would not stop"


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
    marker_context = joinability_dataset.build_model_marker_context(
        args=args,
        tasks_by_kind=tasks_by_kind,
        upstream_identities=[],
    )

    worker = threading.Thread(
        target=precompute_extraction_task_groups,
        kwargs={
            "extractor": FakeExtractor(),
            "cache": cache,
            "tasks_by_kind": tasks_by_kind,
            "args": args,
            "state": state,
            "progress": None,
            "marker_context": marker_context,
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
    tasks_by_kind = {
        "text": [],
        "image": [_task("image", "pending")],
    }
    marker_context = joinability_dataset.build_model_marker_context(
        args=args,
        tasks_by_kind=tasks_by_kind,
        upstream_identities=[],
    )
    worker = threading.Thread(
        target=precompute_extraction_task_groups,
        kwargs={
            "extractor": FakeExtractor(),
            "cache": ExtractionCache(tmp_path / "model_cache.jsonl"),
            "tasks_by_kind": tasks_by_kind,
            "args": args,
            "state": ModelConcurrencyState(text_workers=1, image_workers=1),
            "progress": None,
            "marker_context": marker_context,
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
