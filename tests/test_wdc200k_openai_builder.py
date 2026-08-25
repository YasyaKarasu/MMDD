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
SCRIPTS = ROOT / "scripts_old"
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
    load_openai_auto_check_api_config,
    load_openai_compatible_api_profiles,
    load_openai_environment_file,
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


def responses_api_payload() -> dict[str, Any]:
    return {
        "id": "resp_test",
        "status": "completed",
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": json.dumps(
                            {
                                "attributes": [
                                    {"name": "State", "value": "Texas"}
                                ]
                            }
                        ),
                    }
                ],
            }
        ],
        "usage": {
            "input_tokens": 120,
            "output_tokens": 30,
            "total_tokens": 150,
            "input_tokens_details": {"cached_tokens": 20},
            "output_tokens_details": {"reasoning_tokens": 0},
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


def test_portable_chat_payload_sends_reasoning_effort() -> None:
    extractor = OpenAIAttributeExtractor(
        api_key="fake-key",
        model="vendor-model",
        max_output_tokens=256,
        reasoning_effort="medium",
        verbosity="high",
        portable_chat_completions=True,
    )

    payload = extractor.request_payload(
        model="vendor-model",
        messages=[{"role": "user", "content": "Extract."}],
    )

    assert payload["max_tokens"] == 256
    assert "max_completion_tokens" not in payload
    assert payload["reasoning_effort"] == "medium"
    assert "verbosity" not in payload
    assert "store" not in payload


def test_portable_chat_payload_can_omit_reasoning_effort() -> None:
    extractor = OpenAIAttributeExtractor(
        api_key="fake-key",
        model="vendor-model",
        max_output_tokens=256,
        reasoning_effort="omit",
        portable_chat_completions=True,
    )

    payload = extractor.request_payload(
        model="vendor-model",
        messages=[{"role": "user", "content": "Extract."}],
    )

    assert "reasoning_effort" not in payload


def test_responses_payload_converts_image_and_uses_strict_schema(
    monkeypatch: Any,
) -> None:
    captured: dict[str, Any] = {}

    def fake_post(url: str, **kwargs: Any) -> FakeResponse:
        captured["url"] = url
        captured.update(kwargs)
        return FakeResponse(responses_api_payload())

    monkeypatch.setattr(openai_extractor.requests, "post", fake_post)
    extractor = OpenAIAttributeExtractor(
        api_key="fake-key",
        model="gpt-test",
        max_output_tokens=512,
        reasoning_effort="none",
        verbosity="low",
        image_detail="high",
        use_responses=True,
    )

    result = extractor.chat(
        base_url="https://api.example.test/v1",
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
    assert captured["url"] == "https://api.example.test/v1/responses"
    request = captured["json"]
    assert "messages" not in request
    assert "response_format" not in request
    assert request["max_output_tokens"] == 512
    assert request["reasoning"] == {"effort": "none"}
    assert request["text"] == {
        "format": {
            "type": "json_schema",
            "name": "attribute_extraction",
            "strict": True,
            "schema": ATTRIBUTE_EXTRACTION_SCHEMA,
        },
        "verbosity": "low",
    }
    assert request["input"][1]["content"][1] == {
        "type": "input_image",
        "image_url": "data:image/png;base64,AAAA",
        "detail": "high",
    }


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
    assert "pretrained, memorized, and outside knowledge" not in user_text
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


def test_auto_check_api_profiles_support_arbitrary_models_and_no_final_judge(
    tmp_path: Path,
) -> None:
    environment_file = tmp_path / "profiles.env"
    environment_file.write_text(
        "MMDD_AUTO_CHECK_API_PROFILES=gateway_a,gateway_b,gateway_c\n"
        "MMDD_AUTO_CHECK_API_GATEWAY_A_API_KEY=fake-a\n"
        "MMDD_AUTO_CHECK_API_GATEWAY_A_BASE_URL=https://a.example.test/v1\n"
        "MMDD_AUTO_CHECK_API_GATEWAY_A_MODEL=gpt-5.6-luna\n"
        "MMDD_AUTO_CHECK_API_GATEWAY_A_FINAL_JUDGE_MODEL=none\n"
        "MMDD_AUTO_CHECK_API_GATEWAY_A_MAX_CONCURRENCY=8\n"
        "MMDD_AUTO_CHECK_API_GATEWAY_B_API_KEY=fake-b\n"
        "MMDD_AUTO_CHECK_API_GATEWAY_B_BASE_URL=https://b.example.test/v1\n"
        "MMDD_AUTO_CHECK_API_GATEWAY_B_MODEL=gpt-5.6-luna\n"
        "MMDD_AUTO_CHECK_API_GATEWAY_B_FINAL_JUDGE_MODEL=grok-4.5\n"
        "MMDD_AUTO_CHECK_API_GATEWAY_C_API_KEY=fake-c\n"
        "MMDD_AUTO_CHECK_API_GATEWAY_C_BASE_URL=https://c.example.test/v1\n"
        "MMDD_AUTO_CHECK_API_GATEWAY_C_MODEL=gpt-5.6-luna\n"
        "MMDD_AUTO_CHECK_API_GATEWAY_C_FINAL_JUDGE_MODEL=claude-sonnet-5\n",
        encoding="utf-8",
    )
    environment_file.chmod(0o600)
    environ: dict[str, str] = {}

    loaded = load_openai_environment_file(environment_file, environ=environ)
    profiles = load_openai_compatible_api_profiles(environ=environ)

    assert len(loaded) == 14
    assert [profile.name for profile in profiles] == [
        "gateway_a",
        "gateway_b",
        "gateway_c",
    ]
    assert profiles[0].final_judge_model is None
    assert profiles[0].final_judge_api_key is None
    assert profiles[0].final_judge_api_base_url is None
    assert profiles[0].max_concurrency == 8
    assert profiles[1].final_judge_model == "grok-4.5"
    assert profiles[1].initial_api_key == "fake-b"
    assert profiles[1].final_judge_api_key == "fake-b"
    assert profiles[1].initial_api_base_url == "https://b.example.test/v1"
    assert profiles[1].final_judge_api_base_url == (
        "https://b.example.test/v1"
    )
    assert profiles[2].final_judge_model == "claude-sonnet-5"


def test_auto_check_json_config_supports_nested_stage_connections(
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
                            "api_key": "luna-only-key",
                            "base_url": "https://luna.example.test/v1",
                            "model": "gpt-5.6-luna",
                        },
                        "final_judge": None,
                        "max_concurrency": 8,
                    },
                    "mixed_api": {
                        "response": True,
                        "initial": {
                            "api_key": "initial-key",
                            "base_url": "https://initial.example.test/v1",
                            "model": "gpt-5.6-luna",
                        },
                        "final_judge": {
                            "api_key": "final-key",
                            "base_url": "https://final.example.test/v1",
                            "model": "grok-4.5",
                            "response": False,
                        },
                        "max_concurrency": 20,
                    },
                    "final_only": {
                        "initial": None,
                        "final_judge": {
                            "api_key": "final-only-key",
                            "base_url": "https://final-only.example.test/v1",
                            "model": "gemini-3.0-pro",
                        },
                        "max_concurrency": 12,
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    config_file.chmod(0o600)

    profiles = load_openai_auto_check_api_config(config_file)

    assert [profile.name for profile in profiles] == [
        "luna_only",
        "mixed_api",
        "final_only",
    ]
    assert profiles[0].final_judge_model is None
    assert profiles[0].max_concurrency == 8
    assert profiles[0].initial_use_responses is False
    assert profiles[1].initial_api_key == "initial-key"
    assert profiles[1].initial_api_base_url == (
        "https://initial.example.test/v1"
    )
    assert profiles[1].final_judge_api_key == "final-key"
    assert profiles[1].final_judge_api_base_url == (
        "https://final.example.test/v1"
    )
    assert profiles[1].final_judge_model == "grok-4.5"
    assert profiles[1].initial_use_responses is True
    assert profiles[1].final_judge_use_responses is False
    assert profiles[2].model is None
    assert profiles[2].initial_api_key is None
    assert profiles[2].initial_api_base_url is None
    assert profiles[2].final_judge_api_key == "final-only-key"
    assert profiles[2].final_judge_api_base_url == (
        "https://final-only.example.test/v1"
    )
    assert profiles[2].final_judge_model == "gemini-3.0-pro"


@pytest.mark.parametrize(
    ("config", "expected_error"),
    [
        ({"version": 2, "profiles": {}}, "version must be 1"),
        (
            {
                "version": 1,
                "profiles": {
                    "api_a": {
                        "initial": {
                            "api_key": "fake-key",
                            "base_url": "https://api.example.test/v1",
                            "model": "gpt-5.6-luna",
                            "unexpected": True,
                        },
                        "final_judge": None,
                    }
                },
            },
            "unsupported fields",
        ),
        (
            {
                "version": 1,
                "profiles": {
                    "api_a": {
                        "initial": {
                            "api_key": "fake-key",
                            "base_url": "https://api.example.test/v1",
                        },
                        "final_judge": None,
                    }
                },
            },
            "initial.model",
        ),
        (
            {
                "version": 1,
                "profiles": {
                    "api_a": {
                        "initial": {
                            "api_key": "fake-key",
                            "base_url": "https://api.example.test/v1",
                            "model": "gpt-5.6-luna",
                        },
                        "final_judge": None,
                        "max_concurrency": 4,
                    }
                },
            },
            "integer >= 5",
        ),
        (
            {
                "version": 1,
                "profiles": {
                    "api_a": {
                        "initial": {
                            "api_key": "fake-key",
                            "base_url": "https://api.example.test/v1",
                            "model": "gpt-5.6-luna",
                            "response": "true",
                        },
                        "final_judge": None,
                    }
                },
            },
            "response must be a boolean",
        ),
        (
            {
                "version": 1,
                "profiles": {
                    "api_a": {
                        "initial": None,
                        "final_judge": None,
                    }
                },
            },
            "must configure initial, final_judge, or both",
        ),
    ],
)
def test_auto_check_json_config_rejects_invalid_structure(
    tmp_path: Path,
    config: dict[str, Any],
    expected_error: str,
) -> None:
    config_file = tmp_path / "auto-check.json"
    config_file.write_text(json.dumps(config), encoding="utf-8")
    config_file.chmod(0o600)

    with pytest.raises(ValueError, match=expected_error):
        load_openai_auto_check_api_config(config_file)


def test_auto_check_json_config_requires_private_permissions(
    tmp_path: Path,
) -> None:
    config_file = tmp_path / "auto-check.json"
    config_file.write_text(
        '{"version":1,"profiles":{}}',
        encoding="utf-8",
    )
    config_file.chmod(0o644)

    with pytest.raises(ValueError, match="chmod 600"):
        load_openai_auto_check_api_config(config_file)


def test_auto_check_json_config_requires_explicit_final_judge(
    tmp_path: Path,
) -> None:
    config_file = tmp_path / "auto-check.json"
    config_file.write_text(
        json.dumps(
            {
                "version": 1,
                "profiles": {
                    "api_a": {
                        "initial": {
                            "api_key": "fake-key",
                            "base_url": "https://api.example.test/v1",
                            "model": "gpt-5.6-luna",
                        }
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    config_file.chmod(0o600)

    with pytest.raises(ValueError, match="final_judge"):
        load_openai_auto_check_api_config(config_file)


def test_auto_check_api_profile_supports_separate_stage_connections(
    tmp_path: Path,
) -> None:
    environment_file = tmp_path / "profiles.env"
    environment_file.write_text(
        "MMDD_AUTO_CHECK_API_PROFILES=gateway_a\n"
        "MMDD_AUTO_CHECK_API_GATEWAY_A_API_KEY=legacy-key\n"
        "MMDD_AUTO_CHECK_API_GATEWAY_A_BASE_URL=https://legacy.example.test/v1\n"
        "MMDD_AUTO_CHECK_API_GATEWAY_A_INITIAL_API_KEY=initial-key\n"
        "MMDD_AUTO_CHECK_API_GATEWAY_A_INITIAL_BASE_URL=https://initial.example.test/v1\n"
        "MMDD_AUTO_CHECK_API_GATEWAY_A_MODEL=gpt-5.6-luna\n"
        "MMDD_AUTO_CHECK_API_GATEWAY_A_FINAL_JUDGE_API_KEY=final-key\n"
        "MMDD_AUTO_CHECK_API_GATEWAY_A_FINAL_JUDGE_BASE_URL=https://final.example.test/v1\n"
        "MMDD_AUTO_CHECK_API_GATEWAY_A_FINAL_JUDGE_MODEL=grok-4.5\n",
        encoding="utf-8",
    )
    environment_file.chmod(0o600)
    environ: dict[str, str] = {}

    loaded = load_openai_environment_file(environment_file, environ=environ)
    profile = load_openai_compatible_api_profiles(environ=environ)[0]

    assert len(loaded) == 9
    assert profile.initial_api_key == "initial-key"
    assert profile.initial_api_base_url == "https://initial.example.test/v1"
    assert profile.final_judge_api_key == "final-key"
    assert profile.final_judge_api_base_url == "https://final.example.test/v1"


def test_auto_check_none_profile_needs_only_initial_connection() -> None:
    profile = load_openai_compatible_api_profiles(
        environ={
            "MMDD_AUTO_CHECK_API_PROFILES": "gateway_a",
            "MMDD_AUTO_CHECK_API_GATEWAY_A_INITIAL_API_KEY": "initial-key",
            "MMDD_AUTO_CHECK_API_GATEWAY_A_INITIAL_BASE_URL": (
                "https://initial.example.test/v1"
            ),
            "MMDD_AUTO_CHECK_API_GATEWAY_A_MODEL": "gpt-5.6-luna",
            "MMDD_AUTO_CHECK_API_GATEWAY_A_FINAL_JUDGE_MODEL": "none",
        }
    )[0]

    assert profile.final_judge_model is None
    assert profile.final_judge_api_key is None
    assert profile.final_judge_api_base_url is None


@pytest.mark.parametrize(
    ("extra_environment", "expected_error"),
    [
        ({}, "FINAL_JUDGE_API_KEY"),
        (
            {
                "MMDD_AUTO_CHECK_API_GATEWAY_A_FINAL_JUDGE_API_KEY": (
                    "final-key"
                )
            },
            "FINAL_JUDGE_BASE_URL",
        ),
    ],
)
def test_auto_check_final_profile_requires_final_connection(
    extra_environment: dict[str, str],
    expected_error: str,
) -> None:
    environ = {
        "MMDD_AUTO_CHECK_API_PROFILES": "gateway_a",
        "MMDD_AUTO_CHECK_API_GATEWAY_A_INITIAL_API_KEY": "initial-key",
        "MMDD_AUTO_CHECK_API_GATEWAY_A_INITIAL_BASE_URL": (
            "https://initial.example.test/v1"
        ),
        "MMDD_AUTO_CHECK_API_GATEWAY_A_MODEL": "gpt-5.6-luna",
        "MMDD_AUTO_CHECK_API_GATEWAY_A_FINAL_JUDGE_MODEL": "grok-4.5",
        **extra_environment,
    }

    with pytest.raises(ValueError, match=expected_error):
        load_openai_compatible_api_profiles(environ=environ)


def test_auto_check_env_rejects_undeclared_profile_fields(tmp_path: Path) -> None:
    environment_file = tmp_path / "profiles.env"
    environment_file.write_text(
        "MMDD_AUTO_CHECK_API_PROFILES=gateway_a\n"
        "MMDD_AUTO_CHECK_API_GATEWAY_A_API_KEY=fake-a\n"
        "MMDD_AUTO_CHECK_API_GATEWAY_A_BASE_URL=https://a.example.test/v1\n"
        "MMDD_AUTO_CHECK_API_GATEWAY_B_API_KEY=fake-b\n",
        encoding="utf-8",
    )
    environment_file.chmod(0o600)

    with pytest.raises(ValueError, match="does not allow"):
        load_openai_environment_file(environment_file, environ={})


def test_auto_check_profile_requires_explicit_final_judge_model() -> None:
    environ = {
        "MMDD_AUTO_CHECK_API_PROFILES": "gateway_a",
        "MMDD_AUTO_CHECK_API_GATEWAY_A_API_KEY": "fake-a",
        "MMDD_AUTO_CHECK_API_GATEWAY_A_BASE_URL": (
            "https://a.example.test/v1"
        ),
        "MMDD_AUTO_CHECK_API_GATEWAY_A_MODEL": "gpt-5.6-luna",
    }

    with pytest.raises(ValueError, match="full model name or none"):
        load_openai_compatible_api_profiles(environ=environ)


def test_adaptive_openai_concurrency_starts_at_five_and_adjusts() -> None:
    controller = OpenAIRequestController(
        max_inflight=9,
        adaptive=True,
        initial_inflight=5,
        successes_per_increase=2,
    )

    assert controller.summary()["current_inflight_limit"] == 5
    controller.record_success()
    controller.record_success()
    assert controller.summary()["current_inflight_limit"] == 6
    controller.record_failure()
    assert controller.summary()["current_inflight_limit"] == 3
    assert controller.summary()["concurrency_increases"] == 1
    assert controller.summary()["concurrency_decreases"] == 1


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
