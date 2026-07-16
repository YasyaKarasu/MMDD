import argparse
import subprocess
import sys
import threading
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
    parse_args as parse_dynamic_vllm_args,
    start_server,
)


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
        passthrough_args=["--max_source_tables", "10"],
    )

    assert "--precompute_model_cache" in command
    assert "--model_start_marker" in command
    assert "--model_ready_marker" in command
    assert "--model_text_done_marker" in command
    assert "--model_image_done_marker" in command
    assert "--text_model_base_urls_file" in command
    assert "--image_model_base_urls_file" in command
    assert "http://127.0.0.1:8001/v1" in command
    assert "http://127.0.0.1:8000/v1" in command
    assert command[-2:] == ["--max_source_tables", "10"]


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


def test_start_server_discards_vllm_output_by_default(monkeypatch):
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
    assert captured["stdout"] == subprocess.DEVNULL
    assert captured["stderr"] == subprocess.DEVNULL
    assert captured["text"] is True
    assert captured["start_new_session"] is True


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
