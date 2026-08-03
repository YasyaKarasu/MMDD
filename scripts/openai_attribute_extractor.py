"""OpenAI Chat Completions adapter for the existing attribute extractor."""

from __future__ import annotations

import json
import logging
import math
import os
import random
import re
import stat
import threading
import time
from contextlib import contextmanager
from collections.abc import MutableMapping
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator
from urllib.parse import urlsplit

import requests

import build_mm_joinability_dataset as join_builder


OPENAI_OUTPUT_SCHEMA_VERSION = "openai_attribute_extraction_schema_v2_values_only"
OPENAI_TRANSPORT_VERSION = "openai_chat_completions_transport_v1"
OPENAI_ENVIRONMENT_KEYS = frozenset(
    {"OPENAI_API_KEY", "OPENAI_BASE_URL"}
)
_ENVIRONMENT_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")

ATTRIBUTE_EXTRACTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "attributes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "minLength": 1},
                    "value": {"type": "string", "minLength": 1},
                },
                "required": ["name", "value"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["attributes"],
    "additionalProperties": False,
}


def _estimated_request_tokens(
    messages: Iterable[dict[str, Any]],
    *,
    max_output_tokens: int,
    image_detail: str,
) -> int:
    text_characters = 0
    image_count = 0
    message_count = 0
    for message in messages:
        message_count += 1
        content = message.get("content")
        if isinstance(content, str):
            text_characters += len(content)
            continue
        if not isinstance(content, list):
            continue
        for item in content:
            if not isinstance(item, dict):
                continue
            item_type = str(item.get("type") or "")
            if item_type in {"text", "input_text"}:
                text_characters += len(str(item.get("text") or ""))
            elif item_type in {"image_url", "input_image"}:
                image_count += 1
    image_tokens = image_count * (85 if image_detail == "low" else 1024)
    input_tokens = math.ceil(text_characters / 4) + message_count * 8
    return max(1, input_tokens + image_tokens + int(max_output_tokens))


def retry_after_seconds(response: Any, *, now: float | None = None) -> float | None:
    headers = getattr(response, "headers", None)
    if not isinstance(headers, MutableMapping) and not hasattr(headers, "items"):
        return None
    retry_after = ""
    for name, value in headers.items():
        if str(name).casefold() == "retry-after":
            retry_after = str(value).strip()
            break
    if not retry_after:
        return None
    try:
        seconds = float(retry_after)
    except ValueError:
        try:
            retry_at = parsedate_to_datetime(retry_after)
            if retry_at.tzinfo is None:
                return None
            seconds = retry_at.timestamp() - (time.time() if now is None else now)
        except (TypeError, ValueError, OverflowError):
            return None
    if not math.isfinite(seconds):
        return None
    return max(0.0, seconds)


class OpenAITransientModelError(join_builder.TransientModelEndpointError):
    def __init__(
        self,
        message: str,
        *,
        retry_after: float | None = None,
    ) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class OpenAIRequestController:
    """Share concurrency and rate budgets across text and image requests."""

    def __init__(
        self,
        *,
        max_inflight: int,
        requests_per_minute: int = 0,
        tokens_per_minute: int = 0,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if max_inflight <= 0:
            raise ValueError("OpenAI max inflight requests must be positive")
        if requests_per_minute < 0 or tokens_per_minute < 0:
            raise ValueError("OpenAI rate limits must be non-negative")
        self.max_inflight = int(max_inflight)
        self.requests_per_minute = int(requests_per_minute)
        self.tokens_per_minute = int(tokens_per_minute)
        self._monotonic = monotonic
        self._sleep = sleep
        self._inflight = threading.BoundedSemaphore(self.max_inflight)
        self._schedule_lock = threading.Lock()
        self._request_ready_at = 0.0
        self._token_ready_at = 0.0
        self._cooldown_until = 0.0

    def wait_for_budget(self, estimated_tokens: int) -> None:
        estimated_tokens = max(1, int(estimated_tokens))
        with self._schedule_lock:
            now = self._monotonic()
            scheduled_at = max(now, self._cooldown_until)
            if self.requests_per_minute:
                scheduled_at = max(scheduled_at, self._request_ready_at)
            if self.tokens_per_minute:
                scheduled_at = max(scheduled_at, self._token_ready_at)
            if self.requests_per_minute:
                self._request_ready_at = (
                    scheduled_at + 60.0 / self.requests_per_minute
                )
            if self.tokens_per_minute:
                self._token_ready_at = (
                    scheduled_at
                    + 60.0 * estimated_tokens / self.tokens_per_minute
                )
        while True:
            with self._schedule_lock:
                wait_seconds = max(
                    0.0,
                    scheduled_at,
                    self._cooldown_until,
                ) - self._monotonic()
            if wait_seconds <= 0:
                return
            self._sleep(wait_seconds)

    def reconcile_tokens(self, estimated_tokens: int, actual_tokens: int) -> None:
        if not self.tokens_per_minute:
            return
        token_difference = max(0, int(actual_tokens)) - max(
            1,
            int(estimated_tokens),
        )
        if token_difference <= 0:
            return
        adjustment = 60.0 * token_difference / self.tokens_per_minute
        with self._schedule_lock:
            now = self._monotonic()
            self._token_ready_at += adjustment
            self._cooldown_until = max(
                self._cooldown_until,
                now + adjustment,
            )

    def defer(self, seconds: float) -> None:
        seconds = max(0.0, float(seconds))
        if not seconds:
            return
        with self._schedule_lock:
            self._cooldown_until = max(
                self._cooldown_until,
                self._monotonic() + seconds,
            )

    @contextmanager
    def request_slot(self, estimated_tokens: int) -> Iterator[None]:
        self._inflight.acquire()
        try:
            self.wait_for_budget(estimated_tokens)
            yield
        finally:
            self._inflight.release()

    def summary(self) -> dict[str, int]:
        return {
            "max_inflight": self.max_inflight,
            "requests_per_minute": self.requests_per_minute,
            "tokens_per_minute": self.tokens_per_minute,
        }


def _environment_value(raw_value: str, *, line_number: int) -> str:
    value = raw_value.strip()
    if not value:
        return ""
    if value[0] not in {"'", '"'}:
        if "#" in value:
            raise ValueError(
                "OpenAI environment file does not support inline comments "
                f"(line {line_number})"
            )
        return value
    quote = value[0]
    if len(value) < 2 or value[-1] != quote:
        raise ValueError(
            f"OpenAI environment file has an unterminated quote on line {line_number}"
        )
    return value[1:-1]


def load_openai_environment_file(
    path: Path,
    *,
    environ: MutableMapping[str, str] | None = None,
) -> frozenset[str]:
    """Load a strict, non-executable OpenAI dotenv file into an environment."""

    environment_path = Path(path).expanduser()
    try:
        path_stat = environment_path.stat()
    except OSError as error:
        raise ValueError(
            f"OpenAI environment file is not readable: {environment_path}"
        ) from error
    if not stat.S_ISREG(path_stat.st_mode):
        raise ValueError(
            f"OpenAI environment file is not a regular file: {environment_path}"
        )
    if path_stat.st_mode & (stat.S_IRWXG | stat.S_IRWXO):
        raise ValueError(
            "OpenAI environment file permissions are too broad; run "
            f"chmod 600 {environment_path}"
        )
    try:
        lines = environment_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        raise ValueError(
            f"OpenAI environment file is not valid UTF-8: {environment_path}"
        ) from error

    values: dict[str, str] = {}
    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        name, separator, raw_value = line.partition("=")
        name = name.strip()
        if not separator or _ENVIRONMENT_NAME.fullmatch(name) is None:
            raise ValueError(
                f"OpenAI environment file has an invalid assignment on line {line_number}"
            )
        if name not in OPENAI_ENVIRONMENT_KEYS:
            raise ValueError(
                f"OpenAI environment file does not allow {name!r} on line {line_number}"
            )
        if name in values:
            raise ValueError(
                f"OpenAI environment file repeats {name!r} on line {line_number}"
            )
        value = _environment_value(raw_value, line_number=line_number)
        if not value:
            raise ValueError(
                f"OpenAI environment file has an empty {name!r} on line {line_number}"
            )
        if "\x00" in value or "\r" in value or "\n" in value:
            raise ValueError(
                f"OpenAI environment file has an invalid {name!r} on line {line_number}"
            )
        values[name] = value

    target = os.environ if environ is None else environ
    target.update(values)
    return frozenset(values)


def openai_inference_identity(
    *,
    model: str,
    api_base_url: str,
    reasoning_effort: str,
    verbosity: str,
    max_output_tokens: int,
    image_detail: str,
    image_max_pixels: int,
    context_retry_image_max_pixels: int,
) -> dict[str, Any]:
    """Return every setting that can change a cached extraction result."""

    return {
        "provider": "openai",
        "transport_version": OPENAI_TRANSPORT_VERSION,
        "api": "chat_completions",
        "api_base_url": api_base_url.rstrip("/"),
        "model": model,
        "prompt_version": join_builder.PROMPT_VERSION,
        "output_schema_version": OPENAI_OUTPUT_SCHEMA_VERSION,
        "output_schema": ATTRIBUTE_EXTRACTION_SCHEMA,
        "reasoning_effort": reasoning_effort,
        "verbosity": verbosity,
        "max_output_tokens": int(max_output_tokens),
        "image_detail": image_detail,
        "image_max_pixels": int(image_max_pixels),
        "context_retry_image_max_pixels": int(
            context_retry_image_max_pixels
        ),
    }


def validate_api_base_url(value: str) -> str:
    api_base_url = str(value).strip().rstrip("/")
    try:
        parsed = urlsplit(api_base_url)
        port = parsed.port
    except ValueError as error:
        raise ValueError("OpenAI API base URL is invalid") from error
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or port is not None and not 1 <= port <= 65535
    ):
        raise ValueError(
            "OpenAI API base URL must be an HTTPS origin/path without "
            "credentials, query parameters, or a fragment"
        )
    return api_base_url


def _chat_message_content(
    content: Any,
    *,
    image_detail: str,
) -> str | list[dict[str, Any]]:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        raise TypeError("OpenAI message content must be text or a content list")
    converted: list[dict[str, Any]] = []
    for item in content:
        if not isinstance(item, dict):
            raise TypeError("OpenAI message content items must be objects")
        item_type = str(item.get("type") or "")
        if item_type in {"text", "input_text"}:
            converted.append(
                {
                    "type": "text",
                    "text": str(item.get("text") or ""),
                }
            )
            continue
        if item_type in {"image_url", "input_image"}:
            image_value = item.get("image_url")
            if isinstance(image_value, dict):
                image_url = str(image_value.get("url") or "")
            else:
                image_url = str(image_value or "")
            if not image_url:
                raise ValueError("OpenAI image input is missing image_url")
            converted.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_url,
                        "detail": image_detail,
                    },
                }
            )
            continue
        raise ValueError(f"unsupported OpenAI message content type: {item_type!r}")
    return converted


def chat_messages(
    messages: Iterable[dict[str, Any]],
    *,
    image_detail: str,
) -> list[dict[str, Any]]:
    converted = []
    for message in messages:
        role = str(message.get("role") or "")
        if role not in {"system", "developer", "user", "assistant"}:
            raise ValueError(f"unsupported OpenAI message role: {role!r}")
        converted.append(
            {
                "role": role,
                "content": _chat_message_content(
                    message.get("content"),
                    image_detail=image_detail,
                ),
            }
        )
    return converted


def chat_output_text(payload: dict[str, Any]) -> str:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError("OpenAI chat completion did not contain choices")
    choice = choices[0]
    if not isinstance(choice, dict):
        raise RuntimeError("OpenAI chat completion choice was not an object")
    message = choice.get("message")
    if not isinstance(message, dict):
        raise RuntimeError("OpenAI chat completion did not contain a message")

    refusal = message.get("refusal")
    if refusal:
        raise RuntimeError(f"OpenAI model refused extraction: {str(refusal)[:300]}")
    content = message.get("content")
    if isinstance(content, str) and content.strip():
        return content.strip()
    if isinstance(content, list):
        text_parts = [
            str(item.get("text") or "")
            for item in content
            if isinstance(item, dict)
            and item.get("type") in {"text", "output_text"}
        ]
        rendered = "\n".join(part for part in text_parts if part).strip()
        if rendered:
            return rendered

    finish_reason = str(choice.get("finish_reason") or "")
    if finish_reason and finish_reason != "stop":
        raise RuntimeError(
            "OpenAI chat completion did not finish normally: "
            f"{finish_reason}"
        )
    raise RuntimeError("OpenAI chat completion did not contain message content")


def _token_count(usage: dict[str, Any], *keys: str) -> int:
    for key in keys:
        if key not in usage:
            continue
        try:
            return max(0, int(usage.get(key) or 0))
        except (TypeError, ValueError):
            return 0
    return 0


def normalized_usage(usage: Any) -> dict[str, int]:
    value = usage if isinstance(usage, dict) else {}
    input_details = value.get("input_tokens_details") or value.get(
        "prompt_tokens_details"
    )
    output_details = value.get("output_tokens_details") or value.get(
        "completion_tokens_details"
    )
    input_details = input_details if isinstance(input_details, dict) else {}
    output_details = output_details if isinstance(output_details, dict) else {}
    input_tokens = _token_count(value, "input_tokens", "prompt_tokens")
    output_tokens = _token_count(
        value,
        "output_tokens",
        "completion_tokens",
    )
    total_tokens = _token_count(value, "total_tokens")
    if total_tokens <= 0:
        total_tokens = input_tokens + output_tokens
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "cached_input_tokens": _token_count(input_details, "cached_tokens"),
        "reasoning_output_tokens": _token_count(
            output_details,
            "reasoning_tokens",
        ),
    }


class OpenAIAttributeExtractor(join_builder.LocalAttributeExtractor):
    """Use the existing extraction prompt/parser with Chat Completions."""

    def __init__(
        self,
        *,
        api_key: str | None,
        model: str,
        api_base_url: str = "https://api.openai.com/v1",
        timeout_seconds: float = 120.0,
        max_retries: int = 2,
        retry_sleep_seconds: float = 2.0,
        max_output_tokens: int = 1024,
        reasoning_effort: str = "none",
        verbosity: str = "low",
        image_detail: str = "auto",
        image_max_pixels: int = join_builder.DEFAULT_IMAGE_REQUEST_MAX_PIXELS,
        context_retry_image_max_pixels: int = (
            join_builder.DEFAULT_CONTEXT_RETRY_IMAGE_MAX_PIXELS
        ),
        max_inflight: int = 4,
        requests_per_minute: int = 0,
        tokens_per_minute: int = 0,
        retry_max_seconds: float = 60.0,
        usage_journal_path: Path | None = None,
    ) -> None:
        model = str(model).strip()
        api_base_url = validate_api_base_url(api_base_url)
        if not model:
            raise ValueError("OpenAI model must not be empty")
        if not api_base_url:
            raise ValueError("OpenAI API base URL must not be empty")
        if timeout_seconds <= 0:
            raise ValueError("OpenAI timeout must be positive")
        if max_retries < 0 or retry_sleep_seconds < 0:
            raise ValueError("OpenAI retry settings must be non-negative")
        if not math.isfinite(retry_max_seconds) or retry_max_seconds < 0:
            raise ValueError("OpenAI retry max seconds must be non-negative")
        if max_output_tokens <= 0:
            raise ValueError("OpenAI max output tokens must be positive")
        if reasoning_effort not in {
            "omit",
            "none",
            "minimal",
            "low",
            "medium",
            "high",
            "xhigh",
        }:
            raise ValueError("unsupported OpenAI reasoning effort")
        if verbosity not in {"low", "medium", "high"}:
            raise ValueError("unsupported OpenAI text verbosity")
        if image_detail not in {"auto", "low", "high"}:
            raise ValueError("unsupported OpenAI image detail")

        self.api_key = str(api_key or "").strip()
        self.api_base_url = api_base_url
        self.model = model
        self.text_model_name = model
        self.image_model_name = model
        self.text_model_api_key = self.api_key
        self.image_model_api_key = self.api_key
        self.timeout = float(timeout_seconds)
        self.max_retries = int(max_retries)
        self.retry_sleep = float(retry_sleep_seconds)
        self.retry_max_seconds = float(retry_max_seconds)
        self.max_output_tokens = int(max_output_tokens)
        self.reasoning_effort = reasoning_effort
        self.verbosity = verbosity
        self.image_detail = image_detail
        self.image_request_max_pixels = max(0, int(image_max_pixels))
        self.context_retry_image_max_pixels = max(
            1,
            int(context_retry_image_max_pixels),
        )
        self.usage_journal_path = (
            Path(usage_journal_path) if usage_journal_path is not None else None
        )
        self.model_call_stats = join_builder.ModelCallStats()
        self._usage_lock = threading.Lock()
        self.request_controller = OpenAIRequestController(
            max_inflight=max_inflight,
            requests_per_minute=requests_per_minute,
            tokens_per_minute=tokens_per_minute,
        )
        self.abort_on_transient_error = True

    @contextmanager
    def lease_model_base_url(self, _model_kind: str) -> Iterator[str]:
        yield self.api_base_url

    def ensure_endpoints_ready(
        self,
        modalities: set[str],
        timeout_seconds: float,
        poll_seconds: float = 2.0,
    ) -> None:
        del timeout_seconds, poll_seconds
        unsupported = set(modalities) - {"text", "image"}
        if unsupported:
            raise ValueError(f"unsupported OpenAI modalities: {sorted(unsupported)}")
        if modalities and not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is required for OpenAI model calls")

    def request_payload(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": model,
            "messages": chat_messages(
                messages,
                image_detail=self.image_detail,
            ),
            "stream": False,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "attribute_extraction",
                    "strict": True,
                    "schema": ATTRIBUTE_EXTRACTION_SCHEMA,
                },
            },
            "max_completion_tokens": self.max_output_tokens,
            "store": False,
            "verbosity": self.verbosity,
        }
        if self.reasoning_effort != "omit":
            payload["reasoning_effort"] = self.reasoning_effort
        return payload

    @staticmethod
    def _http_error(response: Any, status_code: int) -> RuntimeError:
        message = ""
        try:
            payload = response.json()
            error = payload.get("error") if isinstance(payload, dict) else None
            if isinstance(error, dict):
                message = str(error.get("message") or "")
        except (TypeError, ValueError):
            message = ""
        if not message:
            message = str(getattr(response, "text", "") or "")
        message = join_builder.clean_text(message)[:500]
        rendered = f"OpenAI Chat Completions API returned HTTP {status_code}"
        if message:
            rendered = f"{rendered}: {message}"
        if status_code in {408, 409, 429} or status_code >= 500:
            return OpenAITransientModelError(
                rendered,
                retry_after=retry_after_seconds(response),
            )
        return RuntimeError(rendered)

    def _retry_delay(self, attempt: int, error: Exception) -> float:
        exponential = self.retry_sleep * (2 ** max(0, int(attempt)))
        jitter = random.uniform(0.0, min(1.0, exponential * 0.25))
        retry_after = getattr(error, "retry_after", None)
        server_delay = (
            max(0.0, float(retry_after))
            if isinstance(retry_after, (int, float))
            and math.isfinite(float(retry_after))
            else 0.0
        )
        return min(
            self.retry_max_seconds,
            max(server_delay, exponential + jitter),
        )

    def _append_usage(
        self,
        *,
        model_kind: str,
        response_payload: dict[str, Any],
    ) -> None:
        if self.usage_journal_path is None:
            return
        record = {
            "record_type": "openai_api_usage",
            "transport_version": OPENAI_TRANSPORT_VERSION,
            "recorded_at": time.time(),
            "response_id": str(response_payload.get("id") or ""),
            "model": self.model,
            "model_kind": model_kind,
            **normalized_usage(response_payload.get("usage")),
        }
        try:
            with self._usage_lock:
                self.usage_journal_path.parent.mkdir(parents=True, exist_ok=True)
                with self.usage_journal_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, sort_keys=True) + "\n")
                    handle.flush()
        except OSError as error:
            logging.warning("Could not append OpenAI usage journal: %s", error)

    def chat(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str | None,
        messages: list[dict[str, Any]],
        model_kind: str = "text",
    ) -> str:
        del api_key
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is required for OpenAI model calls")
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = self.request_payload(model=model, messages=messages)
        estimated_tokens = _estimated_request_tokens(
            messages,
            max_output_tokens=self.max_output_tokens,
            image_detail=self.image_detail,
        )
        last_error: Exception | None = None
        last_error_transient = False
        for attempt in range(self.max_retries + 1):
            started = time.perf_counter()
            usage: dict[str, Any] | None = None
            response: Any | None = None
            retry_deferred = False
            try:
                with self.request_controller.request_slot(estimated_tokens):
                    response = requests.post(
                        f"{base_url.rstrip('/')}/chat/completions",
                        headers=headers,
                        json=payload,
                        timeout=self.timeout,
                        allow_redirects=False,
                        stream=False,
                    )
                    status_code = int(
                        getattr(response, "status_code", 200) or 200
                    )
                    if status_code < 200 or status_code >= 300:
                        request_error = self._http_error(
                            response,
                            status_code,
                        )
                        if (
                            isinstance(
                                request_error,
                                join_builder.TransientModelEndpointError,
                            )
                            and attempt < self.max_retries
                        ):
                            self.request_controller.defer(
                                self._retry_delay(attempt, request_error)
                            )
                            retry_deferred = True
                        raise request_error
                    data = response.json()
                    if not isinstance(data, dict):
                        raise RuntimeError(
                            "OpenAI Chat Completions API returned non-object JSON"
                        )
                    usage_value = data.get("usage")
                    usage = (
                        usage_value
                        if isinstance(usage_value, dict)
                        else None
                    )
                    self._append_usage(
                        model_kind=model_kind,
                        response_payload=data,
                    )
                    content = chat_output_text(data)
            except Exception as error:  # pragma: no cover - branches unit tested.
                self.model_call_stats.record(
                    model_kind,
                    elapsed_seconds=time.perf_counter() - started,
                    failed=True,
                    usage=usage,
                )
                last_error = error
                last_error_transient = isinstance(
                    error,
                    join_builder.TransientModelEndpointError,
                ) or join_builder.is_transient_request_exception(error)
                if last_error_transient and attempt < self.max_retries:
                    if not retry_deferred:
                        self.request_controller.defer(
                            self._retry_delay(attempt, error)
                        )
                    continue
                break
            finally:
                if usage is not None:
                    self.request_controller.reconcile_tokens(
                        estimated_tokens,
                        normalized_usage(usage)["total_tokens"],
                    )
                close_response = getattr(response, "close", None)
                if callable(close_response):
                    close_response()
            self.model_call_stats.record(
                model_kind,
                elapsed_seconds=time.perf_counter() - started,
                usage=usage,
            )
            return content
        message = f"OpenAI {model_kind} model call failed: {last_error}"
        if last_error_transient:
            raise join_builder.TransientModelEndpointError(message) from last_error
        raise RuntimeError(message) from last_error


def _empty_usage_summary() -> dict[str, int]:
    return {
        "responses": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "cached_input_tokens": 0,
        "reasoning_output_tokens": 0,
    }


def _with_averages(summary: dict[str, int]) -> dict[str, int | float]:
    responses = int(summary["responses"])
    result: dict[str, int | float] = dict(summary)
    result["average_input_tokens"] = round(
        summary["input_tokens"] / responses,
        3,
    ) if responses else 0.0
    result["average_output_tokens"] = round(
        summary["output_tokens"] / responses,
        3,
    ) if responses else 0.0
    result["average_total_tokens"] = round(
        summary["total_tokens"] / responses,
        3,
    ) if responses else 0.0
    return result


def summarize_usage_journal(path: Path) -> dict[str, dict[str, int | float]]:
    buckets = {
        "text": _empty_usage_summary(),
        "image": _empty_usage_summary(),
        "all": _empty_usage_summary(),
    }
    path = Path(path)
    if path.is_file():
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(record, dict):
                    continue
                model_kind = str(record.get("model_kind") or "")
                if model_kind not in {"text", "image"}:
                    continue
                for bucket_name in (model_kind, "all"):
                    bucket = buckets[bucket_name]
                    bucket["responses"] += 1
                    for key in (
                        "input_tokens",
                        "output_tokens",
                        "total_tokens",
                        "cached_input_tokens",
                        "reasoning_output_tokens",
                    ):
                        try:
                            bucket[key] += max(0, int(record.get(key) or 0))
                        except (TypeError, ValueError):
                            continue
    return {
        name: _with_averages(bucket)
        for name, bucket in buckets.items()
    }


def current_process_usage(
    extractor: OpenAIAttributeExtractor,
) -> dict[str, dict[str, int | float]]:
    raw = join_builder.model_call_stats_summary(extractor)
    result: dict[str, dict[str, int | float]] = {}
    for model_kind in ("text", "image"):
        bucket = raw.get(model_kind) or {}
        responses = int(bucket.get("responses_with_usage") or 0)
        input_tokens = int(bucket.get("prompt_tokens") or 0)
        output_tokens = int(bucket.get("completion_tokens") or 0)
        total_tokens = int(bucket.get("total_tokens") or 0)
        result[model_kind] = {
            "requests": int(bucket.get("requests") or 0),
            "failed_requests": int(bucket.get("failed_requests") or 0),
            "responses_with_usage": responses,
            "elapsed_seconds": float(bucket.get("elapsed_seconds") or 0.0),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "average_input_tokens": round(input_tokens / responses, 3)
            if responses
            else 0.0,
            "average_output_tokens": round(output_tokens / responses, 3)
            if responses
            else 0.0,
            "average_total_tokens": round(total_tokens / responses, 3)
            if responses
            else 0.0,
        }
    return result


__all__ = [
    "ATTRIBUTE_EXTRACTION_SCHEMA",
    "OPENAI_OUTPUT_SCHEMA_VERSION",
    "OPENAI_TRANSPORT_VERSION",
    "OPENAI_ENVIRONMENT_KEYS",
    "OpenAIAttributeExtractor",
    "OpenAIRequestController",
    "OpenAITransientModelError",
    "chat_messages",
    "chat_output_text",
    "current_process_usage",
    "load_openai_environment_file",
    "normalized_usage",
    "openai_inference_identity",
    "retry_after_seconds",
    "summarize_usage_journal",
    "validate_api_base_url",
]
