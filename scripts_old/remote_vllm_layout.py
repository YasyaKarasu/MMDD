"""Local controller for the MMDD remote vLLM layout protocol.

The remote host is reached exclusively through pre-existing local SSH tunnel
URLs.  This module never starts SSH or executes a remote command.
"""

from __future__ import annotations

import fcntl
import json
import logging
import math
import os
import sqlite3
import stat
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping

try:
    import requests
except ImportError:  # pragma: no cover - integration environment issue.
    requests = None  # type: ignore[assignment]


SCHEMA_VERSION = "mmdd-remote-vllm-layout-v1"
ROUTING_SCHEMA_VERSION = "mmdd-model-routing-v1"
LAYOUTS = {"balanced", "image_burst"}
AGENT_STATES = {"starting", "ready", "transitioning", "degraded", "stopping"}
SERVICE_STATES = {
    "starting",
    "ready",
    "stopping",
    "stopped",
    "failed",
    "unavailable",
}
OPERATION_STATES = {
    "accepted",
    "remote_draining",
    "stopping_switchable",
    "starting_switchable",
    "verifying_switchable",
    "ready",
    "rollback_starting",
    "rolled_back",
    "failed",
}
TERMINAL_OPERATION_STATES = {"ready", "rolled_back", "failed"}
ENDPOINT_IDS = {"primary_image", "switchable"}


class LayoutProtocolError(RuntimeError):
    """Base class for safe, sanitized local protocol failures."""


class InvalidRemoteResponse(LayoutProtocolError):
    """The remote response cannot authorize a routing change."""


class RemoteTransportError(LayoutProtocolError):
    """The local control tunnel is unavailable or timed out."""


class AgentBootChanged(LayoutProtocolError):
    """The remote agent restarted while local state was authoritative."""


class RoutingUnavailableError(LayoutProtocolError):
    """No endpoint is currently published for a requested modality."""


class EndpointVerificationError(LayoutProtocolError):
    """An inference tunnel did not serve its configured exact model."""

    def __init__(self, endpoint_id: str, message: str) -> None:
        super().__init__(message)
        self.endpoint_id = endpoint_id


class RemoteAPIError(LayoutProtocolError):
    """A validated remote protocol error response."""

    def __init__(
        self,
        *,
        status_code: int,
        code: str,
        retryable: bool,
        agent_boot_id: str | None,
    ) -> None:
        super().__init__(f"remote layout API returned {code} (HTTP {status_code})")
        self.status_code = status_code
        self.code = code
        self.retryable = retryable
        self.agent_boot_id = agent_boot_id


def _uuid(value: Any, field: str) -> str:
    if not isinstance(value, str):
        raise InvalidRemoteResponse(f"remote {field} is not a UUID")
    try:
        parsed = uuid.UUID(value)
    except (ValueError, AttributeError) as error:
        raise InvalidRemoteResponse(f"remote {field} is not a UUID") from error
    if str(parsed) != value.lower():
        raise InvalidRemoteResponse(f"remote {field} is not a canonical UUID")
    return value


def _uint(value: Any, field: str, *, positive: bool = False) -> int:
    minimum = 1 if positive else 0
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise InvalidRemoteResponse(f"remote {field} is invalid")
    return value


def _string(value: Any, field: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value):
        raise InvalidRemoteResponse(f"remote {field} is invalid")
    return value


def _object(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise InvalidRemoteResponse(f"remote {field} is not an object")
    return value


def _require_schema(payload: Mapping[str, Any]) -> None:
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise InvalidRemoteResponse("remote schema_version is unsupported")


def _atomic_json(path: Path, payload: Mapping[str, Any], *, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ) + "\n"
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            mode,
        )
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, mode)
        temporary.replace(path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


@dataclass(frozen=True)
class WorkloadSnapshot:
    pair_identity: str
    text_kind: str
    image_kind: str
    text_unfinished: int
    image_unfinished: int


def read_durable_workload(database_path: Path) -> WorkloadSnapshot:
    """Read demand from the newest authoritative job-set pair and jobs."""
    uri = f"file:{database_path.resolve()}?mode=ro"
    try:
        connection = sqlite3.connect(uri, uri=True, timeout=5.0)
        connection.row_factory = sqlite3.Row
        try:
            pair = connection.execute(
                """
                SELECT identity, text_kind, image_kind
                FROM model_jobset_pairs
                ORDER BY updated_at DESC, identity DESC
                LIMIT 1
                """
            ).fetchone()
            if pair is None:
                raise LayoutProtocolError("no authoritative model job-set pair exists")
            rows = connection.execute(
                """
                SELECT kind,
                       SUM(CASE WHEN status NOT IN ('success', 'terminal')
                                THEN 1 ELSE 0 END) AS unfinished
                FROM jobs
                WHERE kind IN (?, ?)
                GROUP BY kind
                """,
                (str(pair["text_kind"]), str(pair["image_kind"])),
            ).fetchall()
        finally:
            connection.close()
    except sqlite3.Error as error:
        raise LayoutProtocolError("durable model workload is unavailable") from error
    unfinished = {str(row["kind"]): int(row["unfinished"] or 0) for row in rows}
    text_kind = str(pair["text_kind"])
    image_kind = str(pair["image_kind"])
    return WorkloadSnapshot(
        pair_identity=str(pair["identity"]),
        text_kind=text_kind,
        image_kind=image_kind,
        text_unfinished=unfinished.get(text_kind, 0),
        image_unfinished=unfinished.get(image_kind, 0),
    )


def desired_layout_for_workload(
    workload: WorkloadSnapshot,
    *,
    current_healthy_layout: str | None,
) -> str | None:
    if workload.text_unfinished > 0:
        return "balanced"
    if workload.image_unfinished > 0:
        return "image_burst"
    return current_healthy_layout


@dataclass(frozen=True)
class EndpointConfig:
    endpoint_id: str
    base_url: str

    def __post_init__(self) -> None:
        if self.endpoint_id not in ENDPOINT_IDS:
            raise ValueError(f"unsupported endpoint ID: {self.endpoint_id}")
        if not self.base_url.startswith(("http://127.0.0.1:", "http://localhost:")):
            raise ValueError(
                f"endpoint {self.endpoint_id} must use a local HTTP tunnel URL"
            )


@dataclass(frozen=True)
class VerifiedEndpointState:
    """One independently verified service generation from a remote snapshot."""

    endpoint_id: str
    base_url: str
    role: str
    served_model_id: str
    instance_generation: int
    max_inflight: int

    def __post_init__(self) -> None:
        if self.endpoint_id not in ENDPOINT_IDS:
            raise ValueError("verified endpoint ID is invalid")
        if not self.base_url.startswith(("http://127.0.0.1:", "http://localhost:")):
            raise ValueError("verified endpoint must use a local tunnel URL")
        if self.role not in {"text", "image"}:
            raise ValueError("verified endpoint role is invalid")
        if not self.served_model_id:
            raise ValueError("verified endpoint model ID is empty")
        if (
            isinstance(self.instance_generation, bool)
            or self.instance_generation < 0
            or isinstance(self.max_inflight, bool)
            or self.max_inflight <= 0
        ):
            raise ValueError("verified endpoint bounds are invalid")

    def routed(self) -> "RoutedEndpoint":
        return RoutedEndpoint(
            endpoint_id=self.endpoint_id,
            base_url=self.base_url,
            served_model_id=self.served_model_id,
            instance_generation=self.instance_generation,
            max_inflight=self.max_inflight,
        )


@dataclass(frozen=True)
class RoutedEndpoint:
    endpoint_id: str
    base_url: str
    served_model_id: str
    instance_generation: int
    max_inflight: int

    def __post_init__(self) -> None:
        if self.endpoint_id not in ENDPOINT_IDS:
            raise ValueError("routing endpoint ID is invalid")
        if not self.base_url.startswith(("http://127.0.0.1:", "http://localhost:")):
            raise ValueError("routing endpoint must use a local tunnel URL")
        if not self.served_model_id:
            raise ValueError("routing endpoint model ID is empty")
        if (
            isinstance(self.instance_generation, bool)
            or self.instance_generation < 0
            or isinstance(self.max_inflight, bool)
            or self.max_inflight <= 0
        ):
            raise ValueError("routing endpoint bounds are invalid")

    def payload(self) -> dict[str, Any]:
        return {
            "endpoint_id": self.endpoint_id,
            "base_url": self.base_url,
            "served_model_id": self.served_model_id,
            "instance_generation": self.instance_generation,
            "max_inflight": self.max_inflight,
        }


@dataclass(frozen=True)
class RoutingManifest:
    controller_id: str
    session_id: str
    routing_revision: int
    agent_boot_id: str
    remote_operation_id: str | None
    layout: str
    text: tuple[RoutedEndpoint, ...]
    image: tuple[RoutedEndpoint, ...]

    def payload(self) -> dict[str, Any]:
        return {
            "schema_version": ROUTING_SCHEMA_VERSION,
            "controller_id": self.controller_id,
            "session_id": self.session_id,
            "routing_revision": self.routing_revision,
            "agent_boot_id": self.agent_boot_id,
            "remote_operation_id": self.remote_operation_id,
            "layout": self.layout,
            "text": [endpoint.payload() for endpoint in self.text],
            "image": [endpoint.payload() for endpoint in self.image],
        }


def _routing_endpoint(value: Any) -> RoutedEndpoint:
    raw = _object(value, "routing endpoint")
    required = {
        "endpoint_id",
        "base_url",
        "served_model_id",
        "instance_generation",
        "max_inflight",
    }
    if set(raw) != required:
        raise ValueError("routing endpoint fields are invalid")
    endpoint_id = str(raw["endpoint_id"])
    if endpoint_id not in ENDPOINT_IDS:
        raise ValueError("routing endpoint ID is invalid")
    return RoutedEndpoint(
        endpoint_id=endpoint_id,
        base_url=str(raw["base_url"]),
        served_model_id=str(raw["served_model_id"]),
        instance_generation=int(raw["instance_generation"]),
        max_inflight=int(raw["max_inflight"]),
    )


def load_routing_manifest(path: Path) -> RoutingManifest:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("routing manifest is unreadable") from error
    if not isinstance(payload, dict):
        raise ValueError("routing manifest must be an object")
    required = {
        "schema_version",
        "controller_id",
        "session_id",
        "routing_revision",
        "agent_boot_id",
        "remote_operation_id",
        "layout",
        "text",
        "image",
    }
    if set(payload) != required or payload["schema_version"] != ROUTING_SCHEMA_VERSION:
        raise ValueError("routing manifest schema is invalid")
    if payload["layout"] not in LAYOUTS:
        raise ValueError("routing manifest layout is invalid")
    text = tuple(_routing_endpoint(item) for item in payload["text"])
    image = tuple(_routing_endpoint(item) for item in payload["image"])
    endpoints = [*text, *image]
    if len({item.endpoint_id for item in endpoints}) != len(endpoints):
        raise ValueError("routing manifest endpoint IDs are duplicated")
    if any(item.max_inflight <= 0 or item.instance_generation < 0 for item in endpoints):
        raise ValueError("routing manifest endpoint bounds are invalid")
    remote_operation_id = payload["remote_operation_id"]
    if remote_operation_id is not None:
        remote_operation_id = _uuid(remote_operation_id, "remote_operation_id")
    return RoutingManifest(
        controller_id=_uuid(payload["controller_id"], "controller_id"),
        session_id=_uuid(payload["session_id"], "session_id"),
        routing_revision=_uint(payload["routing_revision"], "routing_revision"),
        agent_boot_id=_uuid(payload["agent_boot_id"], "agent_boot_id"),
        remote_operation_id=remote_operation_id,
        layout=str(payload["layout"]),
        text=text,
        image=image,
    )


class RoutingScheduler:
    """Capacity-limited endpoint selection sharing one withdrawal lock."""

    def __init__(
        self,
        manifest_path: Path,
        *,
        load_existing: bool = True,
    ) -> None:
        self.manifest_path = manifest_path
        self._condition = threading.Condition(threading.RLock())
        self._manifest: RoutingManifest | None = None
        self._inflight: dict[str, int] = {}
        self._tie_break = {"text": 0, "image": 0}
        if load_existing and manifest_path.is_file():
            self._manifest = load_routing_manifest(manifest_path)
        elif not load_existing:
            self.clear()

    def snapshot(self) -> RoutingManifest | None:
        with self._condition:
            return self._manifest

    def publish(self, manifest: RoutingManifest) -> None:
        endpoints = [*manifest.text, *manifest.image]
        if len({item.endpoint_id for item in endpoints}) != len(endpoints):
            raise ValueError("one endpoint cannot be routed to two modalities")
        if len({item.base_url for item in endpoints}) != len(endpoints):
            raise ValueError("routing endpoint tunnel URLs must be unique")
        with self._condition:
            current_revision = (
                -1 if self._manifest is None else self._manifest.routing_revision
            )
            if manifest.routing_revision <= current_revision:
                raise ValueError("routing revision must increase")
            _atomic_json(self.manifest_path, manifest.payload())
            self._manifest = manifest
            self._condition.notify_all()

    def clear(self) -> None:
        """Withdraw every route and remove any persisted stale snapshot."""
        with self._condition:
            try:
                self.manifest_path.unlink(missing_ok=True)
                if self.manifest_path.parent.exists():
                    directory = os.open(self.manifest_path.parent, os.O_RDONLY)
                    try:
                        os.fsync(directory)
                    finally:
                        os.close(directory)
            finally:
                self._manifest = None
                self._condition.notify_all()

    def endpoints(self, modality: str) -> tuple[RoutedEndpoint, ...]:
        if modality not in {"text", "image"}:
            raise ValueError(f"unsupported modality: {modality}")
        with self._condition:
            if self._manifest is None:
                return ()
            return getattr(self._manifest, modality)

    def capacity(self, modality: str) -> int:
        return sum(endpoint.max_inflight for endpoint in self.endpoints(modality))

    def inflight(self, endpoint_id: str) -> int:
        with self._condition:
            return self._inflight.get(endpoint_id, 0)

    def wait_drained(self, endpoint_id: str, timeout_seconds: float) -> bool:
        deadline = time.monotonic() + timeout_seconds
        with self._condition:
            while self._inflight.get(endpoint_id, 0) > 0:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._condition.wait(remaining)
            return True

    def wait_all_drained(
        self,
        timeout_seconds: float,
        *,
        on_wait: Callable[[], None] | None = None,
    ) -> bool:
        deadline = time.monotonic() + timeout_seconds
        with self._condition:
            while any(count > 0 for count in self._inflight.values()):
                if on_wait is not None:
                    on_wait()
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._condition.wait(
                    min(remaining, 0.5) if on_wait is not None else remaining
                )
            return True

    def wait_for_capacity(self, modality: str, timeout_seconds: float) -> bool:
        if modality not in {"text", "image"}:
            raise ValueError(f"unsupported modality: {modality}")
        deadline = time.monotonic() + timeout_seconds
        with self._condition:
            while True:
                endpoints = (
                    ()
                    if self._manifest is None
                    else getattr(self._manifest, modality)
                )
                if any(
                    self._inflight.get(endpoint.endpoint_id, 0)
                    < endpoint.max_inflight
                    for endpoint in endpoints
                ):
                    return True
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._condition.wait(remaining)

    def acquire(self, modality: str) -> RoutedEndpoint:
        if modality not in {"text", "image"}:
            raise ValueError(f"unsupported modality: {modality}")
        with self._condition:
            while True:
                endpoints = () if self._manifest is None else getattr(self._manifest, modality)
                if not endpoints:
                    raise RoutingUnavailableError(
                        f"no {modality} endpoint is currently routable"
                    )
                available = [
                    endpoint
                    for endpoint in endpoints
                    if self._inflight.get(endpoint.endpoint_id, 0)
                    < endpoint.max_inflight
                ]
                if not available:
                    self._condition.wait()
                    continue
                minimum = min(
                    self._inflight.get(endpoint.endpoint_id, 0)
                    for endpoint in available
                )
                candidates = [
                    endpoint
                    for endpoint in available
                    if self._inflight.get(endpoint.endpoint_id, 0) == minimum
                ]
                index = self._tie_break[modality]
                endpoint = candidates[index % len(candidates)]
                self._tie_break[modality] = index + 1
                self._inflight[endpoint.endpoint_id] = (
                    self._inflight.get(endpoint.endpoint_id, 0) + 1
                )
                return endpoint

    def release(self, endpoint: RoutedEndpoint) -> None:
        with self._condition:
            count = self._inflight.get(endpoint.endpoint_id, 0)
            if count <= 0:
                raise RuntimeError("routing endpoint lease was released twice")
            if count == 1:
                self._inflight.pop(endpoint.endpoint_id, None)
            else:
                self._inflight[endpoint.endpoint_id] = count - 1
            self._condition.notify_all()

    @contextmanager
    def lease(self, modality: str) -> Iterator[str]:
        endpoint = self.acquire(modality)
        try:
            yield endpoint.base_url
        finally:
            self.release(endpoint)

    def is_current(self, modality: str, base_url: str) -> bool:
        return any(endpoint.base_url == base_url for endpoint in self.endpoints(modality))


class ExclusiveControllerLock:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._handle: Any | None = None

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.path.open("a+", encoding="utf-8")
        os.chmod(self.path, 0o600)
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            handle.close()
            raise LayoutProtocolError("another local layout controller is active") from error
        self._handle = handle

    def release(self) -> None:
        if self._handle is None:
            return
        fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
        self._handle.close()
        self._handle = None


def load_or_create_controller_id(path: Path) -> str:
    if path.exists():
        try:
            value = path.read_text(encoding="ascii").strip()
        except OSError as error:
            raise LayoutProtocolError("controller ID file is unreadable") from error
        return _uuid(value, "controller_id")
    value = str(uuid.uuid4())
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w", encoding="ascii") as handle:
            handle.write(value + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            return load_or_create_controller_id(path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)
    return value


@dataclass(frozen=True)
class Lease:
    agent_boot_id: str
    lease_id: str
    ttl_seconds: int


class LayoutHTTPClient:
    """Strict version-1 client for a loopback-forwarded control API."""

    def __init__(
        self,
        *,
        control_url: str,
        token_file: Path,
        request_timeout_seconds: float,
    ) -> None:
        if requests is None:
            raise RuntimeError("requests is required for remote layout control")
        base = control_url.rstrip("/")
        if not base.startswith(("http://127.0.0.1:", "http://localhost:")):
            raise ValueError("layout control URL must use a local HTTP tunnel")
        if token_file.name == ".env.openai" or token_file.resolve().name == ".env.openai":
            raise ValueError("the protected OpenAI environment file cannot be a token file")
        mode = stat.S_IMODE(token_file.stat().st_mode)
        if mode != 0o600:
            raise ValueError("layout control token file must have mode 0600")
        token = token_file.read_text(encoding="utf-8").strip()
        if not token or "\n" in token or "\r" in token:
            raise ValueError("layout control token file is invalid")
        self.control_url = base
        self._headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        self.request_timeout_seconds = request_timeout_seconds

    def _request(
        self,
        method: str,
        path: str,
        *,
        payload: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        try:
            response = requests.request(
                method,
                f"{self.control_url}{path}",
                headers=self._headers,
                json=payload,
                timeout=self.request_timeout_seconds,
            )
        except Exception as error:
            raise RemoteTransportError("remote layout control tunnel is unavailable") from error
        if len(response.content) > 1024 * 1024:
            raise InvalidRemoteResponse("remote layout response is too large")
        try:
            body = response.json()
        except ValueError as error:
            raise InvalidRemoteResponse("remote layout response is not JSON") from error
        body = _object(body, "response")
        _require_schema(body)
        if 200 <= response.status_code < 300:
            return body
        error = _object(body.get("error"), "error")
        code = _string(error.get("code"), "error.code")
        retryable = error.get("retryable")
        if not isinstance(retryable, bool):
            raise InvalidRemoteResponse("remote error.retryable is invalid")
        boot_id = body.get("agent_boot_id")
        if boot_id is not None:
            boot_id = _uuid(boot_id, "agent_boot_id")
        raise RemoteAPIError(
            status_code=int(response.status_code),
            code=code,
            retryable=retryable,
            agent_boot_id=boot_id,
        )

    @staticmethod
    def _boot(payload: Mapping[str, Any], expected_boot_id: str | None) -> str:
        boot_id = _uuid(payload.get("agent_boot_id"), "agent_boot_id")
        if expected_boot_id is not None and boot_id != expected_boot_id:
            raise AgentBootChanged("remote agent boot ID changed")
        return boot_id

    def acquire_lease(
        self,
        *,
        controller_id: str,
        session_id: str,
        requested_ttl_seconds: int,
        expected_boot_id: str | None = None,
    ) -> Lease:
        body = self._request(
            "PUT",
            "/v1/lease",
            payload={
                "schema_version": SCHEMA_VERSION,
                "controller_id": controller_id,
                "session_id": session_id,
                "requested_ttl_seconds": requested_ttl_seconds,
            },
        )
        boot_id = self._boot(body, expected_boot_id)
        if body.get("controller_id") != controller_id or body.get("session_id") != session_id:
            raise InvalidRemoteResponse("remote lease ownership is invalid")
        lease_id = _uuid(body.get("lease_id"), "lease_id")
        ttl = _uint(body.get("ttl_seconds"), "ttl_seconds", positive=True)
        if not 5 <= ttl <= 60:
            raise InvalidRemoteResponse("remote lease TTL is outside protocol bounds")
        _uint(body.get("server_time_unix_ms"), "server_time_unix_ms")
        return Lease(boot_id, lease_id, ttl)

    def release_lease(
        self,
        *,
        controller_id: str,
        session_id: str,
        lease_id: str,
        expected_boot_id: str,
    ) -> None:
        body = self._request(
            "POST",
            "/v1/lease/release",
            payload={
                "schema_version": SCHEMA_VERSION,
                "controller_id": controller_id,
                "session_id": session_id,
                "lease_id": lease_id,
            },
        )
        self._boot(body, expected_boot_id)

    def status(self, *, expected_boot_id: str | None = None) -> dict[str, Any]:
        body = self._request("GET", "/v1/status")
        self._boot(body, expected_boot_id)
        state = body.get("agent_state")
        if state not in AGENT_STATES:
            raise InvalidRemoteResponse("remote agent_state is invalid")
        ready_layout = body.get("ready_layout")
        target_layout = body.get("target_layout")
        if ready_layout is not None and ready_layout not in LAYOUTS:
            raise InvalidRemoteResponse("remote ready_layout is invalid")
        if target_layout is not None and target_layout not in LAYOUTS:
            raise InvalidRemoteResponse("remote target_layout is invalid")
        operation_id = body.get("active_operation_id")
        if operation_id is not None:
            _uuid(operation_id, "active_operation_id")
        _uint(body.get("status_revision"), "status_revision")
        services = _object(body.get("services"), "services")
        if set(services) != ENDPOINT_IDS:
            raise InvalidRemoteResponse("remote services topology is invalid")
        for endpoint_id, value in services.items():
            self._validate_service(value, expected_endpoint_id=endpoint_id)
        _uint(body.get("server_time_unix_ms"), "server_time_unix_ms")
        return body

    @staticmethod
    def _validate_service(value: Any, *, expected_endpoint_id: str) -> dict[str, Any]:
        service = _object(value, f"service {expected_endpoint_id}")
        if service.get("endpoint_id") != expected_endpoint_id:
            raise InvalidRemoteResponse("remote service endpoint identity is invalid")
        if service.get("role") not in {"text", "image"}:
            raise InvalidRemoteResponse("remote service role is invalid")
        if service.get("state") not in SERVICE_STATES:
            raise InvalidRemoteResponse("remote service state is invalid")
        _uint(service.get("remote_port"), "service.remote_port", positive=True)
        _string(service.get("served_model_id"), "service.served_model_id")
        _uint(service.get("instance_generation"), "service.instance_generation")
        _uint(service.get("max_inflight"), "service.max_inflight", positive=True)
        return service

    def request_layout(
        self,
        *,
        payload: Mapping[str, Any],
        expected_boot_id: str,
    ) -> dict[str, Any]:
        body = self._request("PUT", "/v1/layout", payload=payload)
        self._boot(body, expected_boot_id)
        return self._validate_operation(body.get("operation"))

    def operation(
        self,
        operation_id: str,
        *,
        expected_boot_id: str,
    ) -> dict[str, Any]:
        body = self._request("GET", f"/v1/operations/{operation_id}")
        self._boot(body, expected_boot_id)
        operation = self._validate_operation(body.get("operation"))
        if operation["operation_id"] != operation_id:
            raise InvalidRemoteResponse("remote operation identity changed")
        return operation

    @staticmethod
    def _validate_operation(value: Any) -> dict[str, Any]:
        operation = _object(value, "operation")
        _uuid(operation.get("operation_id"), "operation_id")
        _uuid(operation.get("controller_id"), "operation.controller_id")
        _uuid(operation.get("session_id"), "operation.session_id")
        _uint(operation.get("sequence"), "operation.sequence", positive=True)
        if operation.get("from_layout") is not None and operation.get("from_layout") not in LAYOUTS:
            raise InvalidRemoteResponse("remote operation from_layout is invalid")
        if operation.get("desired_layout") not in LAYOUTS:
            raise InvalidRemoteResponse("remote operation desired_layout is invalid")
        state = operation.get("state")
        if state not in OPERATION_STATES:
            raise InvalidRemoteResponse("remote operation state is invalid")
        if state in TERMINAL_OPERATION_STATES:
            resulting = operation.get("resulting_layout")
            if resulting is not None and resulting not in LAYOUTS:
                raise InvalidRemoteResponse("remote operation resulting_layout is invalid")
        if state in {"ready", "rolled_back"}:
            expected_result = (
                operation.get("desired_layout")
                if state == "ready"
                else operation.get("from_layout")
            )
            if (
                expected_result not in LAYOUTS
                or operation.get("resulting_layout") != expected_result
            ):
                raise InvalidRemoteResponse(
                    f"{state} operation resulting_layout is invalid"
                )
            services = _object(operation.get("services"), "operation.services")
            if set(services) != ENDPOINT_IDS:
                raise InvalidRemoteResponse(
                    f"{state} operation services are invalid"
                )
            for endpoint_id, service in services.items():
                LayoutHTTPClient._validate_service(
                    service,
                    expected_endpoint_id=endpoint_id,
                )
        elif state == "failed":
            _string(operation.get("error_code"), "operation.error_code")
            message = _string(operation.get("error_message"), "operation.error_message")
            if len(message.encode("utf-8")) > 512:
                raise InvalidRemoteResponse("operation error_message is too long")
        return operation


@dataclass(frozen=True)
class ControllerConfig:
    control_url: str
    token_file: Path
    database_path: Path | None
    controller_id_file: Path
    lock_file: Path
    endpoints: Mapping[str, EndpointConfig]
    text_model_id: str
    image_model_id: str
    text_api_key: str | None = None
    image_api_key: str | None = None
    lease_ttl_seconds: int = 15
    lease_renew_seconds: float = 5.0
    request_timeout_seconds: float = 3.0
    reconnect_timeout_seconds: float = 120.0
    operation_timeout_seconds: float = 1200.0
    drain_timeout_seconds: float = 300.0
    poll_seconds: float = 1.0
    workload_poll_seconds: float = 2.0
    image_burst_stability_seconds: float = 10.0
    endpoint_health_timeout_seconds: float = 3.0
    endpoint_health_stable_polls: int = 2
    workload_reader: Callable[[], WorkloadSnapshot] | None = None

    def __post_init__(self) -> None:
        if set(self.endpoints) != ENDPOINT_IDS:
            raise ValueError("remote layout requires both fixed endpoint mappings")
        if not 5 <= self.lease_ttl_seconds <= 60:
            raise ValueError("lease TTL must be between 5 and 60 seconds")
        if not 0 < self.lease_renew_seconds < self.lease_ttl_seconds:
            raise ValueError("lease renewal interval must be below the TTL")
        finite_positive = (
            self.request_timeout_seconds,
            self.reconnect_timeout_seconds,
            self.operation_timeout_seconds,
            self.drain_timeout_seconds,
            self.poll_seconds,
            self.workload_poll_seconds,
            self.endpoint_health_timeout_seconds,
        )
        if any(not math.isfinite(value) or value <= 0 for value in finite_positive):
            raise ValueError("remote layout timeouts must be finite and positive")
        if (
            not math.isfinite(self.image_burst_stability_seconds)
            or self.image_burst_stability_seconds < 0
        ):
            raise ValueError("image burst stability interval must be non-negative")
        if self.endpoint_health_stable_polls <= 0:
            raise ValueError("endpoint health stable polls must be positive")
        if self.database_path is None and self.workload_reader is None:
            raise ValueError("a durable database or workload reader is required")


class RemoteLayoutController:
    """Own the local lease, reconciliation loop, drain, and route publication."""

    def __init__(self, config: ControllerConfig, scheduler: RoutingScheduler) -> None:
        self.config = config
        self.scheduler = scheduler
        self.client = LayoutHTTPClient(
            control_url=config.control_url,
            token_file=config.token_file,
            request_timeout_seconds=config.request_timeout_seconds,
        )
        self.controller_id = load_or_create_controller_id(config.controller_id_file)
        self.session_id = str(uuid.uuid4())
        self._file_lock = ExclusiveControllerLock(config.lock_file)
        self._lease_lock = threading.Lock()
        self._recovery_lock = threading.Lock()
        self._routing_mutation_lock = threading.RLock()
        self._lease: Lease | None = None
        self._sequence = 0
        self._stop = threading.Event()
        self._renew_stop = threading.Event()
        self._renew_thread: threading.Thread | None = None
        self._monitor_thread: threading.Thread | None = None
        self._image_only_since: float | None = None

    @property
    def agent_boot_id(self) -> str:
        with self._lease_lock:
            if self._lease is None:
                raise LayoutProtocolError("remote layout lease is not acquired")
            return self._lease.agent_boot_id

    def _set_lease(self, lease: Lease) -> None:
        with self._lease_lock:
            current = self._lease
            if current is not None and lease.agent_boot_id != current.agent_boot_id:
                raise AgentBootChanged("remote agent boot ID changed")
            self._lease = lease

    def _lease_snapshot(self) -> Lease:
        with self._lease_lock:
            if self._lease is None:
                raise LayoutProtocolError("remote layout lease is unavailable")
            return self._lease

    def _acquire_until(self, *, expected_boot_id: str | None = None) -> Lease:
        deadline = time.monotonic() + self.config.reconnect_timeout_seconds
        while not self._stop.is_set():
            try:
                lease = self.client.acquire_lease(
                    controller_id=self.controller_id,
                    session_id=self.session_id,
                    requested_ttl_seconds=self.config.lease_ttl_seconds,
                    expected_boot_id=expected_boot_id,
                )
                self._set_lease(lease)
                return lease
            except AgentBootChanged:
                raise
            except RemoteAPIError as error:
                if error.code not in {"lease_conflict", "lease_expired"} and not error.retryable:
                    raise
            except RemoteTransportError:
                pass
            if time.monotonic() >= deadline:
                raise RemoteTransportError("timed out acquiring remote layout lease")
            self._stop.wait(min(self.config.poll_seconds, max(0.0, deadline - time.monotonic())))
        raise RemoteTransportError("remote layout controller is stopping")

    def start(self) -> None:
        self._file_lock.acquire()
        try:
            self._acquire_until()
            if self._stop.is_set():
                raise RemoteTransportError(
                    "remote layout controller stopped during lease acquisition"
                )
            # A persisted route belongs to an earlier controller session.
            # Keep primary usable, but re-authorize switchable only after the
            # new session has checked status and the inference tunnel.
            self._fail_closed_switchable()
            self._renew_thread = threading.Thread(
                target=self._renew_loop,
                name="remote-layout-lease",
                daemon=True,
            )
            self._renew_thread.start()
            self.reconcile_once(wait_for_stability=True)
            if self._stop.is_set():
                raise RemoteTransportError(
                    "remote layout controller stopped during reconciliation"
                )
            self._monitor_thread = threading.Thread(
                target=self._monitor_loop,
                name="remote-layout-monitor",
                daemon=True,
            )
            self._monitor_thread.start()
        except EndpointVerificationError as error:
            self._withdraw_endpoint(error.endpoint_id)
            self._fail_closed_switchable()
            self.close()
            raise
        except BaseException:
            self._fail_closed_switchable()
            self.close()
            raise

    def close(
        self,
        *,
        withdraw_routes: bool = False,
        drain_timeout_seconds: float | None = None,
        drain_progress: Callable[[], None] | None = None,
        require_lease_release: bool = False,
    ) -> None:
        self.request_stop()
        current = threading.current_thread()
        join_timeout = max(1.0, self.config.request_timeout_seconds * 2)

        def join_thread(thread: threading.Thread | None) -> None:
            if thread is None or thread is current:
                return
            deadline = time.monotonic() + join_timeout
            while thread.is_alive():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return
                thread.join(min(remaining, 0.5))
                if drain_progress is not None:
                    drain_progress()

        if self._monitor_thread is not None and self._monitor_thread is not current:
            join_thread(self._monitor_thread)
        if withdraw_routes:
            live_threads = [
                thread.name
                for thread in (self._monitor_thread,)
                if thread is not None
                and thread is not current
                and thread.is_alive()
            ]
            if live_threads:
                raise LayoutProtocolError(
                    "remote layout threads did not stop before route withdrawal"
                )
            timeout = (
                self.config.drain_timeout_seconds
                if drain_timeout_seconds is None
                else drain_timeout_seconds
            )
            with self._routing_mutation_lock:
                self.scheduler.clear()
            if not self.scheduler.wait_all_drained(
                timeout,
                on_wait=drain_progress,
            ):
                raise LayoutProtocolError(
                    "timed out draining remote routes during controller close"
                )
        self._renew_stop.set()
        if self._renew_thread is not None and self._renew_thread is not current:
            join_thread(self._renew_thread)
            if require_lease_release and self._renew_thread.is_alive():
                raise LayoutProtocolError(
                    "remote lease renewal did not stop before release"
                )
        try:
            lease = self._lease_snapshot()
        except LayoutProtocolError:
            lease = None
        if lease is not None:
            try:
                self.client.release_lease(
                    controller_id=self.controller_id,
                    session_id=self.session_id,
                    lease_id=lease.lease_id,
                    expected_boot_id=lease.agent_boot_id,
                )
            except LayoutProtocolError:
                if require_lease_release:
                    raise
        self._file_lock.release()

    def request_stop(self) -> None:
        """Interrupt lease acquisition or reconciliation before a safe close."""
        self._stop.set()

    def __enter__(self) -> "RemoteLayoutController":
        self.start()
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()

    def _renew_loop(self) -> None:
        while not self._renew_stop.wait(self.config.lease_renew_seconds):
            try:
                expected = self.agent_boot_id
                lease = self.client.acquire_lease(
                    controller_id=self.controller_id,
                    session_id=self.session_id,
                    requested_ttl_seconds=self.config.lease_ttl_seconds,
                    expected_boot_id=expected,
                )
                self._set_lease(lease)
            except AgentBootChanged:
                self._fail_closed_switchable()
                try:
                    self._recover_agent_restart(expected)
                except LayoutProtocolError:
                    continue
            except LayoutProtocolError:
                # The foreground path reacquires before its next mutation.
                continue

    def _monitor_loop(self) -> None:
        while not self._stop.wait(self.config.workload_poll_seconds):
            try:
                self.reconcile_once(wait_for_stability=False)
            except AgentBootChanged:
                stale_boot_id = self.agent_boot_id
                self._fail_closed_switchable()
                try:
                    self._recover_agent_restart(stale_boot_id)
                except LayoutProtocolError as recovery_error:
                    logging.warning(
                        "Remote layout agent restart recovery paused: %s",
                        recovery_error,
                    )
            except EndpointVerificationError as error:
                self._withdraw_endpoint(error.endpoint_id)
            except InvalidRemoteResponse as error:
                self._fail_closed_switchable()
                logging.warning("Remote layout response rejected: %s", error)
            except RemoteTransportError as error:
                # Section 16.1 keeps the prior route when no transition was
                # accepted. A transition withdraws switchable before I/O.
                logging.warning("Remote layout control unavailable: %s", error)
            except LayoutProtocolError as error:
                logging.warning("Remote layout reconciliation paused: %s", error)

    def _recover_agent_restart(self, stale_boot_id: str) -> None:
        """Invalidate the old lease, reacquire under the new boot, then reconcile."""
        with self._recovery_lock:
            with self._lease_lock:
                current = self._lease
                if (
                    current is not None
                    and current.agent_boot_id != stale_boot_id
                ):
                    return
            self._fail_closed_switchable()
            with self._lease_lock:
                self._lease = None
            self._acquire_until(expected_boot_id=None)

    def _next_revision(self) -> int:
        current = self.scheduler.snapshot()
        return 1 if current is None else current.routing_revision + 1

    def _withdraw_endpoint(self, endpoint_id: str) -> None:
        with self._routing_mutation_lock:
            current = self.scheduler.snapshot()
            if current is None:
                return
            text = tuple(
                item for item in current.text if item.endpoint_id != endpoint_id
            )
            image = tuple(
                item for item in current.image if item.endpoint_id != endpoint_id
            )
            if text == current.text and image == current.image:
                return
            manifest = RoutingManifest(
                controller_id=self.controller_id,
                session_id=self.session_id,
                routing_revision=self._next_revision(),
                agent_boot_id=self.agent_boot_id,
                remote_operation_id=current.remote_operation_id,
                layout=current.layout,
                text=text,
                image=image,
            )
            self.scheduler.publish(manifest)

    def _fail_closed_switchable(self) -> None:
        try:
            self._withdraw_endpoint("switchable")
        except LayoutProtocolError:
            pass

    def _workload_desired(
        self,
        workload: WorkloadSnapshot,
        *,
        current_layout: str | None,
    ) -> str | None:
        desired = desired_layout_for_workload(
            workload,
            current_healthy_layout=current_layout,
        )
        if desired != "image_burst":
            self._image_only_since = None
            return desired
        now = time.monotonic()
        if self._image_only_since is None:
            self._image_only_since = now
        if now - self._image_only_since >= self.config.image_burst_stability_seconds:
            return desired
        return current_layout

    def _read_workload(self) -> WorkloadSnapshot:
        if self.config.workload_reader is not None:
            workload = self.config.workload_reader()
            if not isinstance(workload, WorkloadSnapshot):
                raise LayoutProtocolError(
                    "remote layout workload reader returned an invalid snapshot"
                )
            return workload
        if self.config.database_path is None:
            raise LayoutProtocolError("remote layout workload is unavailable")
        return read_durable_workload(self.config.database_path)

    def _wait_for_image_stability(
        self,
        workload: WorkloadSnapshot,
        current_layout: str | None,
    ) -> str | None:
        desired = self._workload_desired(workload, current_layout=current_layout)
        while (
            desired is None
            and workload.text_unfinished == 0
            and workload.image_unfinished > 0
            and not self._stop.is_set()
        ):
            self._stop.wait(self.config.workload_poll_seconds)
            workload = self._read_workload()
            desired = self._workload_desired(workload, current_layout=current_layout)
        return desired

    def reconcile_once(self, *, wait_for_stability: bool = False) -> None:
        stale_boot_id = self.agent_boot_id
        try:
            self._reconcile_current_boot(wait_for_stability=wait_for_stability)
        except AgentBootChanged:
            self._fail_closed_switchable()
            self._recover_agent_restart(stale_boot_id)
            self._reconcile_current_boot(wait_for_stability=wait_for_stability)

    def _reconcile_current_boot(self, *, wait_for_stability: bool = False) -> None:
        boot_id = self.agent_boot_id
        observed_rollback = False
        try:
            status = self.client.status(expected_boot_id=boot_id)
        except RemoteAPIError as error:
            if error.code in {"lease_expired", "boot_id_mismatch"}:
                self._acquire_until(expected_boot_id=boot_id)
                status = self.client.status(expected_boot_id=boot_id)
            else:
                raise
        if status["agent_state"] == "transitioning":
            operation_id = status.get("active_operation_id")
            if operation_id is None:
                raise InvalidRemoteResponse("transitioning status lacks an operation ID")
            operation = self._wait_operation_terminal(
                operation_id,
                expected_identity=None,
            )
            self._publish_terminal_operation(operation)
            observed_rollback = operation["state"] == "rolled_back"
            status = self.client.status(expected_boot_id=boot_id)
        current_layout = status.get("ready_layout")
        if current_layout in LAYOUTS:
            current_endpoints = self._verified_routes(status, current_layout)
            self._publish_routes(
                layout=current_layout,
                endpoints=current_endpoints,
                operation_id=None,
            )
        if observed_rollback:
            return
        workload = self._read_workload()
        desired = self._workload_desired(workload, current_layout=current_layout)
        if wait_for_stability and desired is None:
            desired = self._wait_for_image_stability(workload, current_layout)
        if desired is None:
            return
        if current_layout != desired:
            self._transition(status, desired, workload)

    def _transition(
        self,
        status: Mapping[str, Any],
        desired_layout: str,
        workload: WorkloadSnapshot,
    ) -> None:
        self._withdraw_endpoint("switchable")
        if not self.scheduler.wait_drained(
            "switchable", self.config.drain_timeout_seconds
        ):
            raise LayoutProtocolError("timed out draining local switchable endpoint")
        self._sequence += 1
        sequence = self._sequence
        reason = (
            "text jobs unfinished; balanced layout required"
            if desired_layout == "balanced"
            else "text queue complete; image jobs remain"
        )
        semantic = {
            "schema_version": SCHEMA_VERSION,
            "controller_id": self.controller_id,
            "session_id": self.session_id,
            "sequence": sequence,
            "expected_agent_boot_id": self.agent_boot_id,
            "expected_current_layout": status.get("ready_layout"),
            "desired_layout": desired_layout,
            "drained_endpoint_ids": ["switchable"],
            "local_routing_revision": self._next_revision(),
            "reason": reason,
        }
        operation = self._submit_layout(semantic)
        expected = (self.controller_id, self.session_id, sequence, desired_layout)
        operation = self._wait_operation_terminal(
            operation["operation_id"],
            expected_identity=expected,
            initial=operation,
        )
        if operation["state"] not in {"ready", "rolled_back"}:
            raise LayoutProtocolError(
                f"remote layout operation ended in {operation['state']}"
            )
        resulting_layout = self._publish_terminal_operation(operation)
        status = self.client.status(expected_boot_id=self.agent_boot_id)
        endpoints = self._verified_routes(status, resulting_layout)
        self._publish_routes(
            layout=resulting_layout,
            endpoints=endpoints,
            operation_id=operation["operation_id"],
        )
        if operation["state"] == "ready" and resulting_layout != desired_layout:
            raise InvalidRemoteResponse(
                "ready operation did not produce the requested layout"
            )

    def _submit_layout(self, semantic: Mapping[str, Any]) -> dict[str, Any]:
        deadline = time.monotonic() + self.config.reconnect_timeout_seconds
        while not self._stop.is_set():
            lease = self._lease_snapshot()
            payload = dict(semantic)
            payload["lease_id"] = lease.lease_id
            try:
                return self.client.request_layout(
                    payload=payload,
                    expected_boot_id=lease.agent_boot_id,
                )
            except RemoteAPIError as error:
                if error.agent_boot_id is not None and error.agent_boot_id != lease.agent_boot_id:
                    raise AgentBootChanged("remote agent boot ID changed")
                if error.code not in {"lease_expired", "transition_in_progress"} and not error.retryable:
                    raise
            except RemoteTransportError:
                pass
            if time.monotonic() >= deadline:
                raise RemoteTransportError("layout request acceptance is unknown")
            self._acquire_until(expected_boot_id=lease.agent_boot_id)
            self._stop.wait(self.config.poll_seconds)
        raise RemoteTransportError("remote layout controller is stopping")

    def _wait_operation_terminal(
        self,
        operation_id: str,
        *,
        expected_identity: tuple[str, str, int, str] | None,
        initial: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        deadline = time.monotonic() + self.config.operation_timeout_seconds
        operation = dict(initial) if initial is not None else None
        while not self._stop.is_set():
            if operation is None:
                try:
                    operation = self.client.operation(
                        operation_id,
                        expected_boot_id=self.agent_boot_id,
                    )
                except RemoteTransportError:
                    if time.monotonic() >= deadline:
                        raise RemoteTransportError("timed out recovering remote operation")
                    self._stop.wait(self.config.poll_seconds)
                    continue
            if expected_identity is not None:
                actual = (
                    operation["controller_id"],
                    operation["session_id"],
                    operation["sequence"],
                    operation["desired_layout"],
                )
                if actual != expected_identity:
                    raise InvalidRemoteResponse("remote operation ownership is invalid")
            if operation["state"] in TERMINAL_OPERATION_STATES:
                return operation
            if time.monotonic() >= deadline:
                raise LayoutProtocolError("timed out waiting for remote layout operation")
            self._stop.wait(self.config.poll_seconds)
            operation = None
        raise LayoutProtocolError("remote layout controller is stopping")

    def _publish_terminal_operation(
        self,
        operation: Mapping[str, Any],
    ) -> str:
        state = operation.get("state")
        if state not in {"ready", "rolled_back"}:
            raise LayoutProtocolError(
                f"remote layout operation ended in {state}"
            )
        resulting_layout = operation.get("resulting_layout")
        if resulting_layout not in LAYOUTS:
            raise InvalidRemoteResponse(
                "terminal operation resulting layout is invalid"
            )
        endpoints = self._verified_endpoint_states(
            _object(operation.get("services"), "operation.services"),
            resulting_layout,
        )
        self._publish_routes(
            layout=resulting_layout,
            endpoints=endpoints,
            operation_id=str(operation["operation_id"]),
        )
        return resulting_layout

    def _verified_routes(
        self,
        status: Mapping[str, Any],
        layout: str,
    ) -> dict[str, VerifiedEndpointState]:
        if status.get("agent_boot_id") != self.agent_boot_id:
            raise AgentBootChanged("remote agent boot ID changed")
        if status.get("agent_state") != "ready" or status.get("ready_layout") != layout:
            raise InvalidRemoteResponse("remote layout is not ready")
        return self._verified_endpoint_states(
            _object(status.get("services"), "status.services"),
            layout,
        )

    def _verified_endpoint_states(
        self,
        services: Mapping[str, Any],
        layout: str,
    ) -> dict[str, VerifiedEndpointState]:
        expected_roles = {
            "primary_image": "image",
            "switchable": "text" if layout == "balanced" else "image",
        }
        expected_models = {
            "primary_image": self.config.image_model_id,
            "switchable": (
                self.config.text_model_id
                if layout == "balanced"
                else self.config.image_model_id
            ),
        }
        result: dict[str, VerifiedEndpointState] = {}
        for endpoint_id in ("primary_image", "switchable"):
            service = _object(services.get(endpoint_id), "service")
            if (
                service.get("endpoint_id") != endpoint_id
                or service.get("state") != "ready"
                or service.get("role") != expected_roles[endpoint_id]
                or service.get("served_model_id") != expected_models[endpoint_id]
            ):
                raise InvalidRemoteResponse(
                    f"remote {endpoint_id} service does not match the ready layout"
                )
            endpoint_config = self.config.endpoints[endpoint_id]
            max_inflight = _uint(
                service.get("max_inflight"),
                f"{endpoint_id}.max_inflight",
                positive=True,
            )
            self._verify_model_endpoint(
                endpoint_id=endpoint_id,
                base_url=endpoint_config.base_url,
                model_id=expected_models[endpoint_id],
                api_key=(
                    self.config.text_api_key
                    if expected_roles[endpoint_id] == "text"
                    else self.config.image_api_key
                ),
            )
            result[endpoint_id] = VerifiedEndpointState(
                endpoint_id=endpoint_id,
                base_url=endpoint_config.base_url,
                role=expected_roles[endpoint_id],
                served_model_id=expected_models[endpoint_id],
                instance_generation=_uint(
                    service.get("instance_generation"),
                    f"{endpoint_id}.instance_generation",
                ),
                max_inflight=max_inflight,
            )
        return result

    def _verify_model_endpoint(
        self,
        *,
        endpoint_id: str,
        base_url: str,
        model_id: str,
        api_key: str | None,
    ) -> None:
        headers = {"Accept": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        for poll in range(self.config.endpoint_health_stable_polls):
            try:
                response = requests.get(
                    f"{base_url.rstrip('/')}/models",
                    headers=headers,
                    timeout=self.config.endpoint_health_timeout_seconds,
                )
                if response.status_code != 200:
                    raise ValueError("non-200 response")
                body = response.json()
                data = body["data"]
                models = {
                    item.get("id")
                    for item in data
                    if isinstance(item, dict) and isinstance(item.get("id"), str)
                }
                if model_id not in models:
                    raise ValueError("wrong model ID")
            except Exception as error:
                raise EndpointVerificationError(
                    endpoint_id,
                    f"inference tunnel verification failed for {endpoint_id}",
                ) from error
            if poll + 1 < self.config.endpoint_health_stable_polls:
                self._stop.wait(self.config.poll_seconds)

    def _publish_routes(
        self,
        *,
        layout: str,
        endpoints: Mapping[str, VerifiedEndpointState],
        operation_id: str | None,
    ) -> None:
        with self._routing_mutation_lock:
            text = tuple(
                endpoints[endpoint_id].routed()
                for endpoint_id in ("primary_image", "switchable")
                if endpoints[endpoint_id].role == "text"
            )
            image = tuple(
                endpoints[endpoint_id].routed()
                for endpoint_id in ("primary_image", "switchable")
                if endpoints[endpoint_id].role == "image"
            )
            expected_endpoint_ids = (
                ({"switchable"}, {"primary_image"})
                if layout == "balanced"
                else (set(), {"primary_image", "switchable"})
            )
            if (
                {item.endpoint_id for item in text} != expected_endpoint_ids[0]
                or {item.endpoint_id for item in image}
                != expected_endpoint_ids[1]
            ):
                raise InvalidRemoteResponse(
                    "verified endpoint roles do not match the layout"
                )
            current = self.scheduler.snapshot()
            if current is not None and (
                current.agent_boot_id == self.agent_boot_id
                and current.layout == layout
                and current.text == text
                and current.image == image
                and current.controller_id == self.controller_id
                and current.session_id == self.session_id
            ):
                return
            self.scheduler.publish(
                RoutingManifest(
                    controller_id=self.controller_id,
                    session_id=self.session_id,
                    routing_revision=self._next_revision(),
                    agent_boot_id=self.agent_boot_id,
                    remote_operation_id=operation_id,
                    layout=layout,
                    text=text,
                    image=image,
                )
            )
