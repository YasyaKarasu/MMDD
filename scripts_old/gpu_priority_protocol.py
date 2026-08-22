"""Filesystem protocol for lending GPUs to a preemptible local worker.

The priority owner publishes generation-scoped requests.  A borrower that has
registered a live heartbeat must acknowledge a matching reclaim request only
after its endpoint files are empty and all of its local GPU process groups have
stopped.  The owner never treats an old acknowledgement as permission to start.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable


PROTOCOL_SCHEMA_VERSION = "mmdd-priority-gpu-v1"
BORROWABLE_STATE = "borrowable"
PRIORITY_REQUESTED_STATE = "priority_requested"
RELEASED_STATUS = "released"
SERVING_STATUS = "serving"


@dataclass(frozen=True)
class ProtocolPaths:
    root: Path

    @property
    def request(self) -> Path:
        return self.root / "priority_request.json"

    @property
    def borrower(self) -> Path:
        return self.root / "borrower_status.json"

    @property
    def acknowledgement(self) -> Path:
        return self.root / "borrower_ack.json"

    @property
    def borrower_lock(self) -> Path:
        return self.root / "borrower.lock"


@dataclass(frozen=True)
class PriorityRequest:
    generation: str
    sequence: int
    state: str
    owner: str
    gpu_ids: tuple[str, ...]
    reason: str
    timestamp: float

    @property
    def token(self) -> tuple[str, int]:
        return self.generation, self.sequence

    def payload(self) -> dict[str, Any]:
        return {
            "schema_version": PROTOCOL_SCHEMA_VERSION,
            "generation": self.generation,
            "sequence": self.sequence,
            "state": self.state,
            "owner": self.owner,
            "gpu_ids": list(self.gpu_ids),
            "reason": self.reason,
            "timestamp": self.timestamp,
        }


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(
                payload,
                handle,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
        _fsync_directory(path.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def atomic_write_endpoints(path: Path, urls: Iterable[str]) -> None:
    normalized: list[str] = []
    seen: set[str] = set()
    for url in urls:
        value = str(url).strip().rstrip("/")
        if value and value not in seen:
            seen.add(value)
            normalized.append(value)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            if normalized:
                handle.write("\n".join(normalized))
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
        _fsync_directory(path.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def read_priority_request(path: Path) -> PriorityRequest | None:
    payload = read_json(path)
    if payload is None:
        return None
    try:
        generation = payload["generation"]
        sequence = payload["sequence"]
        state = payload["state"]
        owner = payload["owner"]
        gpu_ids = payload["gpu_ids"]
        reason = payload.get("reason", "")
        timestamp = payload["timestamp"]
    except KeyError:
        return None
    if (
        payload.get("schema_version") != PROTOCOL_SCHEMA_VERSION
        or not isinstance(generation, str)
        or not generation
        or not isinstance(sequence, int)
        or isinstance(sequence, bool)
        or sequence < 0
        or state not in {BORROWABLE_STATE, PRIORITY_REQUESTED_STATE}
        or not isinstance(owner, str)
        or not owner
        or not isinstance(gpu_ids, list)
        or not gpu_ids
        or not all(isinstance(value, str) and value for value in gpu_ids)
        or not isinstance(reason, str)
        or not isinstance(timestamp, (int, float))
        or isinstance(timestamp, bool)
    ):
        return None
    return PriorityRequest(
        generation=generation,
        sequence=sequence,
        state=state,
        owner=owner,
        gpu_ids=tuple(gpu_ids),
        reason=reason,
        timestamp=float(timestamp),
    )


def request_is_current(path: Path, expected: PriorityRequest) -> bool:
    current = read_priority_request(path)
    return current is not None and current.token == expected.token and (
        current.state == expected.state
    )


def acknowledgement_matches(
    path: Path,
    request: PriorityRequest,
    *,
    status: str,
) -> bool:
    payload = read_json(path)
    return bool(
        payload
        and payload.get("schema_version") == PROTOCOL_SCHEMA_VERSION
        and payload.get("generation") == request.generation
        and payload.get("sequence") == request.sequence
        and payload.get("request_state") == request.state
        and payload.get("status") == status
        and isinstance(payload.get("borrower_id"), str)
        and payload.get("borrower_id")
        and isinstance(payload.get("timestamp"), (int, float))
        and not isinstance(payload.get("timestamp"), bool)
    )


def write_acknowledgement(
    path: Path,
    request: PriorityRequest,
    *,
    borrower_id: str,
    status: str,
    timestamp: float | None = None,
) -> None:
    atomic_write_json(
        path,
        {
            "schema_version": PROTOCOL_SCHEMA_VERSION,
            "generation": request.generation,
            "sequence": request.sequence,
            "request_state": request.state,
            "status": status,
            "borrower_id": borrower_id,
            "timestamp": time.time() if timestamp is None else timestamp,
        },
    )


def write_borrower_status(
    path: Path,
    *,
    borrower_id: str,
    state: str,
    request: PriorityRequest | None,
    timestamp: float | None = None,
) -> None:
    atomic_write_json(
        path,
        {
            "schema_version": PROTOCOL_SCHEMA_VERSION,
            "borrower_id": borrower_id,
            "pid": os.getpid(),
            "state": state,
            "request_generation": (
                request.generation if request is not None else ""
            ),
            "request_sequence": (
                request.sequence if request is not None else -1
            ),
            "heartbeat_timestamp": (
                time.time() if timestamp is None else timestamp
            ),
        },
    )


def remove_borrower_status(path: Path, *, borrower_id: str) -> None:
    payload = read_json(path)
    if payload is not None and payload.get("borrower_id") != borrower_id:
        return
    try:
        Path(path).unlink(missing_ok=True)
    finally:
        if Path(path).parent.exists():
            _fsync_directory(Path(path).parent)


class PriorityGpuOwner:
    """Publish priority state and wait for a live borrower to return GPUs."""

    def __init__(
        self,
        coordination_dir: Path,
        *,
        gpu_ids: Iterable[str],
        owner: str = "entitables",
        reclaim_timeout_seconds: float = 90.0,
        borrower_stale_seconds: float = 10.0,
        unregistered_grace_seconds: float = 1.0,
        poll_seconds: float = 0.2,
        borrower_label: str = "WDC GPU borrower",
        owner_action_label: str = "EntiTables vLLM",
        wall_time: Callable[[], float] = time.time,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        normalized_gpu_ids = tuple(
            dict.fromkeys(str(value).strip() for value in gpu_ids)
        )
        if not normalized_gpu_ids or any(not value for value in normalized_gpu_ids):
            raise ValueError("gpu_ids must contain at least one non-empty value")
        if (
            reclaim_timeout_seconds <= 0
            or borrower_stale_seconds <= 0
            or unregistered_grace_seconds < 0
            or poll_seconds <= 0
        ):
            raise ValueError("GPU coordination timeouts must be positive")
        self.paths = ProtocolPaths(Path(coordination_dir).resolve())
        self.gpu_ids = normalized_gpu_ids
        self.owner = owner
        self.reclaim_timeout_seconds = float(reclaim_timeout_seconds)
        self.borrower_stale_seconds = float(borrower_stale_seconds)
        self.unregistered_grace_seconds = float(
            unregistered_grace_seconds
        )
        self.poll_seconds = float(poll_seconds)
        self.borrower_label = str(borrower_label).strip() or "GPU borrower"
        self.owner_action_label = (
            str(owner_action_label).strip() or "priority workload"
        )
        self._wall_time = wall_time
        self._monotonic = monotonic
        self._sleep = sleep
        self._generation = uuid.uuid4().hex
        self._sequence = 0
        self.paths.root.mkdir(parents=True, exist_ok=True)

    def _publish(self, state: str, *, reason: str) -> PriorityRequest:
        self._sequence += 1
        request = PriorityRequest(
            generation=self._generation,
            sequence=self._sequence,
            state=state,
            owner=self.owner,
            gpu_ids=self.gpu_ids,
            reason=reason,
            timestamp=self._wall_time(),
        )
        atomic_write_json(self.paths.request, request.payload())
        return request

    def release_gpus(self, *, reason: str) -> PriorityRequest:
        return self._publish(BORROWABLE_STATE, reason=reason)

    def request_gpus(self, *, reason: str) -> PriorityRequest:
        request = self._publish(PRIORITY_REQUESTED_STATE, reason=reason)
        self._wait_for_borrower_release(request)
        return request

    def _wait_for_borrower_release(
        self,
        request: PriorityRequest,
    ) -> None:
        started = self._monotonic()
        deadline = started + self.reclaim_timeout_seconds
        no_borrower_since: float | None = None
        while True:
            if acknowledgement_matches(
                self.paths.acknowledgement,
                request,
                status=RELEASED_STATUS,
            ):
                return
            now = self._monotonic()
            borrower = read_json(self.paths.borrower)
            if borrower is None:
                if no_borrower_since is None:
                    no_borrower_since = now
                if now - no_borrower_since >= self.unregistered_grace_seconds:
                    return
            else:
                no_borrower_since = None
                heartbeat = borrower.get("heartbeat_timestamp")
                if (
                    borrower.get("schema_version")
                    != PROTOCOL_SCHEMA_VERSION
                    or not isinstance(borrower.get("borrower_id"), str)
                    or not borrower.get("borrower_id")
                    or not isinstance(heartbeat, (int, float))
                    or isinstance(heartbeat, bool)
                ):
                    raise RuntimeError(
                        f"{self.borrower_label} status is invalid; refusing "
                        "an unsafe reclaim"
                    )
                heartbeat_age = max(0.0, self._wall_time() - float(heartbeat))
                if heartbeat_age > self.borrower_stale_seconds:
                    raise RuntimeError(
                        f"{self.borrower_label} heartbeat is stale; refusing "
                        f"to start {self.owner_action_label} until the borrower "
                        "is cleaned up"
                    )
            if now >= deadline:
                raise RuntimeError(
                    f"Timed out waiting for {self.borrower_label} to acknowledge "
                    f"reclaim generation={request.generation} "
                    f"sequence={request.sequence}"
                )
            self._sleep(min(self.poll_seconds, max(0.0, deadline - now)))
