"""Shared endpoint-level concurrency control for multimodal model calls."""

from __future__ import annotations

import json
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Iterator
from urllib.parse import urlsplit


ENDPOINT_CONFIG_SCHEMA_VERSION = "mmdd-model-endpoints-v1"
MODEL_KINDS = ("text", "image")
ENDPOINT_POOLS = ("local", "remote")


class EndpointPoolUnavailableError(RuntimeError):
    """Raised when no configured endpoint can serve a requested role."""


@dataclass(frozen=True)
class ModelEndpointSpec:
    endpoint_id: str
    base_url: str
    pool: str
    text_max_inflight: int
    image_max_inflight: int
    total_max_inflight: int

    def modality_limit(self, model_kind: str) -> int:
        if model_kind == "text":
            return self.text_max_inflight
        if model_kind == "image":
            return self.image_max_inflight
        raise ValueError(f"unsupported model kind: {model_kind}")

    def supports(self, model_kind: str) -> bool:
        return self.modality_limit(model_kind) > 0

    def payload(self) -> dict[str, object]:
        return {
            "endpoint_id": self.endpoint_id,
            "base_url": self.base_url,
            "pool": self.pool,
            "max_inflight": {
                "text": self.text_max_inflight,
                "image": self.image_max_inflight,
                "total": self.total_max_inflight,
            },
        }


@dataclass(frozen=True)
class ModelEndpointConfig:
    served_model_name: str
    endpoints: tuple[ModelEndpointSpec, ...]

    def payload(self) -> dict[str, object]:
        return {
            "schema_version": ENDPOINT_CONFIG_SCHEMA_VERSION,
            "served_model_name": self.served_model_name,
            "endpoints": [endpoint.payload() for endpoint in self.endpoints],
        }


def _positive_int(value: object, field_name: str, *, allow_zero: bool) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer")
    parsed = value
    if parsed < 0 or (parsed == 0 and not allow_zero):
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{field_name} must be {qualifier}")
    return parsed


def _base_url(value: object) -> str:
    base_url = str(value or "").strip().rstrip("/")
    parsed = urlsplit(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("endpoint base_url must be an absolute HTTP(S) URL")
    return base_url


def _parse_endpoint(raw: object, index: int) -> ModelEndpointSpec:
    if not isinstance(raw, dict):
        raise ValueError(f"endpoints[{index}] must be an object")
    required = {"endpoint_id", "base_url", "pool", "max_inflight"}
    if set(raw) != required:
        raise ValueError(
            f"endpoints[{index}] must contain exactly {sorted(required)}"
        )
    endpoint_id = str(raw["endpoint_id"] or "").strip()
    if not endpoint_id:
        raise ValueError(f"endpoints[{index}].endpoint_id must be non-empty")
    pool = str(raw["pool"] or "").strip()
    if pool not in ENDPOINT_POOLS:
        raise ValueError(
            f"endpoints[{index}].pool must be one of {ENDPOINT_POOLS}"
        )
    limits = raw["max_inflight"]
    if not isinstance(limits, dict) or set(limits) != {"text", "image", "total"}:
        raise ValueError(
            f"endpoints[{index}].max_inflight must contain exactly text, image, total"
        )
    text_limit = _positive_int(
        limits["text"],
        f"endpoints[{index}].max_inflight.text",
        allow_zero=True,
    )
    image_limit = _positive_int(
        limits["image"],
        f"endpoints[{index}].max_inflight.image",
        allow_zero=True,
    )
    total_limit = _positive_int(
        limits["total"],
        f"endpoints[{index}].max_inflight.total",
        allow_zero=False,
    )
    if text_limit == image_limit == 0:
        raise ValueError(f"endpoints[{index}] must enable text or image")
    if max(text_limit, image_limit) > total_limit:
        raise ValueError(
            f"endpoints[{index}] modality limits cannot exceed its total limit"
        )
    return ModelEndpointSpec(
        endpoint_id=endpoint_id,
        base_url=_base_url(raw["base_url"]),
        pool=pool,
        text_max_inflight=text_limit,
        image_max_inflight=image_limit,
        total_max_inflight=total_limit,
    )


def load_model_endpoint_config(path: Path) -> ModelEndpointConfig:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"model endpoint config is unreadable: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError("model endpoint config must be an object")
    required = {"schema_version", "served_model_name", "endpoints"}
    if set(payload) != required:
        raise ValueError(
            f"model endpoint config must contain exactly {sorted(required)}"
        )
    if payload["schema_version"] != ENDPOINT_CONFIG_SCHEMA_VERSION:
        raise ValueError("unsupported model endpoint config schema_version")
    served_model_name = str(payload["served_model_name"] or "").strip()
    if not served_model_name:
        raise ValueError("served_model_name must be non-empty")
    raw_endpoints = payload["endpoints"]
    if not isinstance(raw_endpoints, list) or not raw_endpoints:
        raise ValueError("model endpoint config endpoints must be a non-empty list")
    endpoints = tuple(
        _parse_endpoint(raw_endpoint, index)
        for index, raw_endpoint in enumerate(raw_endpoints)
    )
    endpoint_ids = [endpoint.endpoint_id for endpoint in endpoints]
    if len(set(endpoint_ids)) != len(endpoint_ids):
        raise ValueError("model endpoint IDs must be unique")
    base_urls = [endpoint.base_url for endpoint in endpoints]
    if len(set(base_urls)) != len(base_urls):
        raise ValueError(
            "each physical model endpoint URL must appear exactly once; enable both modalities in max_inflight"
        )
    return ModelEndpointConfig(
        served_model_name=served_model_name,
        endpoints=endpoints,
    )


class ModelEndpointScheduler:
    """Lease endpoints under per-modality and shared total concurrency caps."""

    def __init__(self, config: ModelEndpointConfig) -> None:
        self.config = config
        self._condition = threading.Condition(threading.RLock())
        self._total_inflight: dict[str, int] = {}
        self._modality_inflight: dict[tuple[str, str], int] = {}
        self._tie_break: dict[tuple[str, str], int] = {}

    def endpoints(
        self,
        pool: str,
        model_kind: str,
    ) -> tuple[ModelEndpointSpec, ...]:
        if pool not in ENDPOINT_POOLS:
            raise ValueError(f"unsupported endpoint pool: {pool}")
        if model_kind not in MODEL_KINDS:
            raise ValueError(f"unsupported model kind: {model_kind}")
        return tuple(
            endpoint
            for endpoint in self.config.endpoints
            if endpoint.pool == pool and endpoint.supports(model_kind)
        )

    def urls(self, pool: str, model_kind: str) -> list[str]:
        return [endpoint.base_url for endpoint in self.endpoints(pool, model_kind)]

    def capacity(self, pool: str, model_kind: str) -> int:
        return sum(
            endpoint.modality_limit(model_kind)
            for endpoint in self.endpoints(pool, model_kind)
        )

    def total_inflight(self, endpoint_id: str) -> int:
        with self._condition:
            return self._total_inflight.get(endpoint_id, 0)

    def modality_inflight(self, endpoint_id: str, model_kind: str) -> int:
        with self._condition:
            return self._modality_inflight.get((endpoint_id, model_kind), 0)

    @staticmethod
    def _available_urls(
        available_urls_getter: Callable[[], Iterable[str]] | None,
    ) -> set[str] | None:
        if available_urls_getter is None:
            return None
        return {
            str(url).strip().rstrip("/")
            for url in available_urls_getter()
            if str(url).strip()
        }

    def _eligible_endpoints(
        self,
        pool: str,
        model_kind: str,
        available_urls_getter: Callable[[], Iterable[str]] | None,
    ) -> tuple[ModelEndpointSpec, ...]:
        available_urls = self._available_urls(available_urls_getter)
        return tuple(
            endpoint
            for endpoint in self.endpoints(pool, model_kind)
            if available_urls is None or endpoint.base_url in available_urls
        )

    def _has_capacity(
        self,
        endpoint: ModelEndpointSpec,
        model_kind: str,
    ) -> bool:
        return (
            self._total_inflight.get(endpoint.endpoint_id, 0)
            < endpoint.total_max_inflight
            and self._modality_inflight.get(
                (endpoint.endpoint_id, model_kind), 0
            )
            < endpoint.modality_limit(model_kind)
        )

    def _select(
        self,
        endpoints: tuple[ModelEndpointSpec, ...],
        pool: str,
        model_kind: str,
    ) -> ModelEndpointSpec:
        scored = []
        for index, endpoint in enumerate(endpoints):
            total = self._total_inflight.get(endpoint.endpoint_id, 0)
            modality = self._modality_inflight.get(
                (endpoint.endpoint_id, model_kind), 0
            )
            scored.append(
                (
                    total / endpoint.total_max_inflight,
                    modality / endpoint.modality_limit(model_kind),
                    index,
                    endpoint,
                )
            )
        minimum_score = min((item[0], item[1]) for item in scored)
        candidates = [
            item for item in scored if (item[0], item[1]) == minimum_score
        ]
        tie_key = (pool, model_kind)
        cursor = self._tie_break.get(tie_key, 0)
        selected = candidates[cursor % len(candidates)]
        self._tie_break[tie_key] = cursor + 1
        return selected[3]

    def acquire(
        self,
        pool: str,
        model_kind: str,
        *,
        available_urls_getter: Callable[[], Iterable[str]] | None = None,
    ) -> ModelEndpointSpec:
        with self._condition:
            while True:
                eligible = self._eligible_endpoints(
                    pool,
                    model_kind,
                    available_urls_getter,
                )
                if not eligible:
                    raise EndpointPoolUnavailableError(
                        f"no {pool} {model_kind} endpoint is currently routable"
                    )
                available = tuple(
                    endpoint
                    for endpoint in eligible
                    if self._has_capacity(endpoint, model_kind)
                )
                if available:
                    endpoint = self._select(available, pool, model_kind)
                    self._total_inflight[endpoint.endpoint_id] = (
                        self._total_inflight.get(endpoint.endpoint_id, 0) + 1
                    )
                    modality_key = (endpoint.endpoint_id, model_kind)
                    self._modality_inflight[modality_key] = (
                        self._modality_inflight.get(modality_key, 0) + 1
                    )
                    return endpoint
                # Endpoint files can change without sharing this condition.
                self._condition.wait(0.2)

    def release(self, endpoint: ModelEndpointSpec, model_kind: str) -> None:
        with self._condition:
            total = self._total_inflight.get(endpoint.endpoint_id, 0)
            modality_key = (endpoint.endpoint_id, model_kind)
            modality = self._modality_inflight.get(modality_key, 0)
            if total <= 0 or modality <= 0:
                raise RuntimeError("model endpoint lease was released twice")
            if total == 1:
                self._total_inflight.pop(endpoint.endpoint_id, None)
            else:
                self._total_inflight[endpoint.endpoint_id] = total - 1
            if modality == 1:
                self._modality_inflight.pop(modality_key, None)
            else:
                self._modality_inflight[modality_key] = modality - 1
            self._condition.notify_all()

    @contextmanager
    def lease(
        self,
        pool: str,
        model_kind: str,
        *,
        available_urls_getter: Callable[[], Iterable[str]] | None = None,
    ) -> Iterator[ModelEndpointSpec]:
        endpoint = self.acquire(
            pool,
            model_kind,
            available_urls_getter=available_urls_getter,
        )
        try:
            yield endpoint
        finally:
            self.release(endpoint, model_kind)

    def wait_for_capacity(
        self,
        pool: str,
        model_kind: str,
        timeout_seconds: float,
        *,
        available_urls_getter: Callable[[], Iterable[str]] | None = None,
    ) -> bool:
        import time

        deadline = time.monotonic() + timeout_seconds
        with self._condition:
            while True:
                eligible = self._eligible_endpoints(
                    pool,
                    model_kind,
                    available_urls_getter,
                )
                if any(
                    self._has_capacity(endpoint, model_kind)
                    for endpoint in eligible
                ):
                    return True
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._condition.wait(min(remaining, 0.2))
