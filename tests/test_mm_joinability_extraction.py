import argparse
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from build_mm_joinability_dataset import (
    ExtractionCache,
    ExtractionTask,
    ModelConcurrencyState,
    LocalAttributeExtractor,
    build_bridge_assets_parallel,
    extraction_cache_key,
    normalize_extracted_attributes,
    reparse_extraction_record,
    resolve_extraction_tasks,
    safe_json_object,
)
from build_mm_table_dataset import ShardedJsonlWriter


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
    }
    values.update(overrides)
    return argparse.Namespace(**values)


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
