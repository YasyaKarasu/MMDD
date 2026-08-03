from __future__ import annotations

import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import openai_attribute_extractor as openai_extractor  # noqa: E402
from build_wdc200k_mm_joinability_dataset_openai import (  # noqa: E402
    parse_args,
    prepare_openai_run,
)
from openai_attribute_extractor import (  # noqa: E402
    ATTRIBUTE_EXTRACTION_SCHEMA,
    OpenAIAttributeExtractor,
    OpenAIRequestController,
    current_process_usage,
    summarize_usage_journal,
)


class FakeResponse:
    def __init__(
        self,
        payload: dict[str, Any],
        *,
        status_code: int = 200,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.payload = payload
        self.status_code = status_code
        self.headers = headers or {}
        self.text = json.dumps(payload)

    def json(self) -> dict[str, Any]:
        return self.payload


def response_payload() -> dict[str, Any]:
    return {
        "id": "chatcmpl_test",
        "choices": [
            {
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": json.dumps(
                        {
                            "attributes": [
                                {
                                    "name": "State",
                                    "value": "Texas",
                                }
                            ]
                        }
                    ),
                },
            }
        ],
        "usage": {
            "prompt_tokens": 120,
            "completion_tokens": 30,
            "total_tokens": 150,
            "prompt_tokens_details": {"cached_tokens": 20},
            "completion_tokens_details": {"reasoning_tokens": 4},
        },
    }


def write_openai_env(path: Path) -> None:
    path.write_text(
        "OPENAI_API_KEY=wdc-secret-from-file\n"
        "OPENAI_BASE_URL=https://wdc-file-gateway.example.test/v1\n",
        encoding="utf-8",
    )
    path.chmod(0o600)


@pytest.fixture(autouse=True)
def isolate_default_openai_env_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    for name in ("OPENAI_API_KEY", "OPENAI_BASE_URL"):
        if name in os.environ:
            monkeypatch.setenv(name, os.environ[name])
        else:
            monkeypatch.delenv(name, raising=False)


def test_openai_chat_payload_converts_image_and_uses_strict_schema(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    captured: dict[str, Any] = {}

    def fake_post(url: str, **kwargs: Any) -> FakeResponse:
        captured["url"] = url
        captured.update(kwargs)
        return FakeResponse(response_payload())

    monkeypatch.setattr(openai_extractor.requests, "post", fake_post)
    journal = tmp_path / "usage.jsonl"
    extractor = OpenAIAttributeExtractor(
        api_key="secret-test-key",
        model="gpt-test",
        max_output_tokens=512,
        reasoning_effort="none",
        verbosity="low",
        image_detail="high",
        usage_journal_path=journal,
    )
    result = extractor.chat(
        base_url="https://api.openai.com/v1",
        model="gpt-test",
        api_key=None,
        model_kind="image",
        messages=[
            {"role": "system", "content": "Extract attributes."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Inspect this image."},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,AAAA"},
                    },
                ],
            },
        ],
    )

    assert json.loads(result)["attributes"][0]["value"] == "Texas"
    assert captured["url"] == "https://api.openai.com/v1/chat/completions"
    assert captured["headers"]["Authorization"] == "Bearer secret-test-key"
    request = captured["json"]
    assert request["store"] is False
    assert request["stream"] is False
    assert captured["stream"] is False
    assert captured["allow_redirects"] is False
    assert request["max_completion_tokens"] == 512
    assert request["reasoning_effort"] == "none"
    assert request["verbosity"] == "low"
    assert request["response_format"] == {
        "type": "json_schema",
        "json_schema": {
            "name": "attribute_extraction",
            "strict": True,
            "schema": ATTRIBUTE_EXTRACTION_SCHEMA,
        },
    }
    image_input = request["messages"][1]["content"][1]
    assert image_input == {
        "type": "image_url",
        "image_url": {
            "url": "data:image/png;base64,AAAA",
            "detail": "high",
        },
    }
    assert "secret-test-key" not in journal.read_text(encoding="utf-8")

    current = current_process_usage(extractor)["image"]
    assert current["input_tokens"] == 120
    assert current["output_tokens"] == 30
    assert current["average_total_tokens"] == 150.0
    cumulative = summarize_usage_journal(journal)
    assert cumulative["image"]["cached_input_tokens"] == 20
    assert cumulative["all"]["reasoning_output_tokens"] == 4
    assert cumulative["all"]["average_input_tokens"] == 120.0


def test_openai_extractor_reuses_existing_prompt_and_parser(
    monkeypatch: Any,
) -> None:
    captured: dict[str, Any] = {}

    def fake_post(_url: str, **kwargs: Any) -> FakeResponse:
        captured.update(kwargs)
        return FakeResponse(response_payload())

    monkeypatch.setattr(openai_extractor.requests, "post", fake_post)
    extractor = OpenAIAttributeExtractor(
        api_key="test",
        model="gpt-test",
        max_retries=0,
    )
    result = extractor.extract(
        {
            "asset_id": "text-1",
            "asset_type": "text",
            "content": "Alpha is located in Texas.",
        },
        {
            "entity_id": "entity-1",
            "cell_text": "Alpha",
            "wiki_title": "SECRET_WIKIPEDIA_TITLE",
            "row_attributes": [
                {"name": "Name", "value": "Alpha", "is_entity": True},
                {"name": "State", "value": "Texas", "is_entity": False},
                {"name": "Category", "value": "City", "is_entity": False},
            ],
        },
        ["State"],
    )

    assert result["error"] == ""
    assert result["attributes"] == [
        {
            "name": "State",
            "value": "Texas",
        }
    ]
    user_text = captured["json"]["messages"][1]["content"]
    assert "Name [ENTITY; NEVER MASK]: Alpha" in user_text
    assert "State: Texas" in user_text
    assert "Category: City" in user_text
    assert "SECRET_WIKIPEDIA_TITLE" not in user_text
    assert "Wikipedia" not in user_text
    assert "connection_evidence" not in user_text
    assert "separate leave-one-attribute-out test" in user_text
    assert "same request does not imply that they are related" in user_text


def test_openai_transient_rate_limit_is_retried(
    monkeypatch: Any,
) -> None:
    responses = [
        FakeResponse(
            {"error": {"message": "rate limited"}},
            status_code=429,
        ),
        FakeResponse(response_payload()),
    ]

    def fake_post(_url: str, **_kwargs: Any) -> FakeResponse:
        return responses.pop(0)

    monkeypatch.setattr(openai_extractor.requests, "post", fake_post)
    extractor = OpenAIAttributeExtractor(
        api_key="test",
        model="gpt-test",
        max_retries=1,
        retry_sleep_seconds=0,
    )

    extractor.chat(
        base_url="https://api.openai.com/v1",
        model="gpt-test",
        api_key=None,
        messages=[{"role": "user", "content": "Extract."}],
    )

    stats = current_process_usage(extractor)["text"]
    assert stats["requests"] == 2
    assert stats["failed_requests"] == 1
    assert stats["total_tokens"] == 150


def test_openai_retry_honors_retry_after_header(
    monkeypatch: Any,
) -> None:
    responses = [
        FakeResponse(
            {"error": {"message": "rate limited"}},
            status_code=429,
            headers={"Retry-After": "3"},
        ),
        FakeResponse(response_payload()),
    ]

    def fake_post(_url: str, **_kwargs: Any) -> FakeResponse:
        return responses.pop(0)

    monkeypatch.setattr(openai_extractor.requests, "post", fake_post)
    extractor = OpenAIAttributeExtractor(
        api_key="test",
        model="gpt-test",
        max_retries=1,
        retry_sleep_seconds=0,
        retry_max_seconds=10,
    )
    deferred: list[float] = []
    monkeypatch.setattr(
        extractor.request_controller,
        "defer",
        deferred.append,
    )

    extractor.chat(
        base_url="https://api.openai.com/v1",
        model="gpt-test",
        api_key=None,
        messages=[{"role": "user", "content": "Extract."}],
    )

    assert deferred == [3.0]


def test_openai_request_controller_paces_shared_rpm_and_tpm() -> None:
    now = [0.0]
    sleeps: list[float] = []

    def monotonic() -> float:
        return now[0]

    def sleep(seconds: float) -> None:
        sleeps.append(seconds)
        now[0] += seconds

    controller = OpenAIRequestController(
        max_inflight=2,
        requests_per_minute=60,
        tokens_per_minute=600,
        monotonic=monotonic,
        sleep=sleep,
    )

    controller.wait_for_budget(10)
    controller.wait_for_budget(10)

    assert sleeps == [1.0]


def test_openai_max_inflight_is_shared_across_modalities(
    monkeypatch: Any,
) -> None:
    state_lock = threading.Lock()
    release = threading.Event()
    limit_reached = threading.Event()
    inflight = 0
    maximum_inflight = 0

    def fake_post(_url: str, **_kwargs: Any) -> FakeResponse:
        nonlocal inflight, maximum_inflight
        with state_lock:
            inflight += 1
            maximum_inflight = max(maximum_inflight, inflight)
            if inflight == 2:
                limit_reached.set()
        try:
            assert release.wait(timeout=5)
            return FakeResponse(response_payload())
        finally:
            with state_lock:
                inflight -= 1

    monkeypatch.setattr(openai_extractor.requests, "post", fake_post)
    extractor = OpenAIAttributeExtractor(
        api_key="test",
        model="gpt-test",
        max_retries=0,
        max_inflight=2,
    )

    def run_call(model_kind: str) -> str:
        return extractor.chat(
            base_url="https://api.openai.com/v1",
            model="gpt-test",
            api_key=None,
            model_kind=model_kind,
            messages=[{"role": "user", "content": "Extract."}],
        )

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [
            pool.submit(run_call, "text" if index % 2 == 0 else "image")
            for index in range(4)
        ]
        assert limit_reached.wait(timeout=5)
        release.set()
        for future in futures:
            assert future.result()

    assert maximum_inflight == 2


def test_openai_base_url_rejects_embedded_credentials(
    tmp_path: Path,
) -> None:
    args = parse_args(
        [
            "--input_dir",
            str(tmp_path / "input"),
            "--output_dir",
            str(tmp_path / "output"),
            "--openai_base_url",
            "https://secret@example.test/v1",
        ]
    )

    with pytest.raises(ValueError, match="without credentials"):
        prepare_openai_run(args)


def test_openai_base_url_uses_environment_and_cli_wins(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_BASE_URL", "https://gateway.example.test/v1")
    common = [
        "--input_dir",
        str(tmp_path / "input"),
        "--output_dir",
        str(tmp_path / "output"),
    ]

    from_environment = parse_args(common)
    from_cli = parse_args(
        [
            *common,
            "--openai_base_url",
            "https://override.example.test/v1",
        ]
    )

    assert (
        from_environment.openai_base_url
        == "https://gateway.example.test/v1"
    )
    assert from_cli.openai_base_url == "https://override.example.test/v1"


def test_wdc_openai_env_file_is_loaded_before_full_argument_parsing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment_file = tmp_path / ".env.openai"
    write_openai_env(environment_file)
    monkeypatch.setenv("OPENAI_API_KEY", "stale-wdc-shell-key")
    monkeypatch.setenv(
        "OPENAI_BASE_URL",
        "https://stale-wdc-gateway.example.test/v1",
    )
    args = parse_args(
        [
            "--input_dir",
            str(tmp_path / "input"),
            "--output_dir",
            str(tmp_path / "output"),
            "--cache_dir",
            str(tmp_path / "cache"),
            "--openai_model",
            "gpt-test",
            "--openai_env_file",
            str(environment_file),
        ]
    )

    openai_run = prepare_openai_run(args)

    assert args.openai_base_url == (
        "https://wdc-file-gateway.example.test/v1"
    )
    assert openai_run.extractor.api_key == "wdc-secret-from-file"
    config_text = (
        openai_run.config.work_dir / "openai_run_config.json"
    ).read_text(encoding="utf-8")
    assert "wdc-secret-from-file" not in config_text
    assert "stale-wdc-shell-key" not in config_text


def test_wdc_default_openai_env_file_is_loaded_and_cli_wins(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    write_openai_env(tmp_path / ".env.openai")
    monkeypatch.setenv("OPENAI_API_KEY", "stale-wdc-shell-key")
    monkeypatch.setenv(
        "OPENAI_BASE_URL",
        "https://stale-wdc-gateway.example.test/v1",
    )
    common = [
        "--input_dir",
        str(tmp_path / "input"),
        "--output_dir",
        str(tmp_path / "output"),
        "--openai_model",
        "gpt-test",
    ]

    from_file = prepare_openai_run(parse_args(common))
    from_cli = prepare_openai_run(
        parse_args(
            [
                *common,
                "--openai_base_url",
                "https://override.example.test/v1",
            ]
        )
    )

    assert from_file.config.text_model_base_url == (
        "https://wdc-file-gateway.example.test/v1"
    )
    assert from_file.extractor.api_key == "wdc-secret-from-file"
    assert from_cli.config.text_model_base_url == (
        "https://override.example.test/v1"
    )


def test_legacy_openai_api_base_url_alias_is_preserved(
    tmp_path: Path,
) -> None:
    args = parse_args(
        [
            "--input_dir",
            str(tmp_path / "input"),
            "--output_dir",
            str(tmp_path / "output"),
            "--openai_api_base_url",
            "https://legacy.example.test/v1",
        ]
    )

    assert args.openai_base_url == "https://legacy.example.test/v1"


def test_openai_work_is_fingerprinted_but_download_cache_is_shared(
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    cache_dir = tmp_path / "shared_download_cache"
    common = [
        "--input_dir",
        str(input_dir),
        "--output_dir",
        str(output_dir),
        "--cache_dir",
        str(cache_dir),
        "--openai_model",
        "gpt-test",
    ]

    first = prepare_openai_run(parse_args(common))
    second = prepare_openai_run(
        parse_args([*common, "--openai_reasoning_effort", "low"])
    )

    assert first.config.cache_dir == cache_dir.resolve()
    assert second.config.cache_dir == cache_dir.resolve()
    assert first.config.work_dir != second.config.work_dir
    assert "openai_model_runs" in first.config.work_dir.parts
    assert first.config.work_dir != (
        output_dir.parent / "work_wdc_200k"
    ).resolve()
    assert first.config.text_model_name.startswith(
        "openai-chat:gpt-test:"
    )
    assert "Qwen" not in first.config.text_model_name
    run_config = json.loads(
        (first.config.work_dir / "openai_run_config.json").read_text(
            encoding="utf-8"
        )
    )
    assert run_config["download_cache_dir"] == str(cache_dir.resolve())
    assert run_config["inference_work_dir"] == str(first.config.work_dir)
    assert "api_key" not in json.dumps(run_config)


def test_same_openai_identity_resumes_the_same_inference_work_dir(
    tmp_path: Path,
) -> None:
    argv = [
        "--input_dir",
        str(tmp_path / "input"),
        "--output_dir",
        str(tmp_path / "output"),
        "--cache_dir",
        str(tmp_path / "cache"),
        "--openai_model",
        "gpt-test",
    ]

    first = prepare_openai_run(parse_args(argv))
    resumed = prepare_openai_run(parse_args(argv))

    assert first.inference_fingerprint == resumed.inference_fingerprint
    assert first.config.work_dir == resumed.config.work_dir
    assert first.usage_journal_path == resumed.usage_journal_path
