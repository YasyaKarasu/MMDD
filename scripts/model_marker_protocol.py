"""Shared strict marker protocol for dynamic model orchestration."""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from wdc200k_io import GuardedWriteTracker, PreWriteGuard


MODEL_MARKER_SCHEMA_VERSION = "wdc200k-model-markers-v1"
MODEL_START_STAGE = "wdc200k_model_start"
MODEL_READY_STAGE = "wdc200k_model_ready"
MODEL_DONE_STAGE = "wdc200k_model_done"
READY_MODEL_KIND = "text+image"


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def is_sha256(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def is_timestamp(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def atomic_write_json(
    path: Path,
    payload: Mapping[str, Any],
    *,
    pre_write_guard: PreWriteGuard | None = None,
) -> None:
    path = Path(path)
    tracker = GuardedWriteTracker(path, pre_write_guard)
    encoded = (
        json.dumps(
            dict(payload),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    tracker.before_write(len(encoded.encode("utf-8")))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        tracker.before_commit(0)
        temporary.replace(path)
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


@dataclass(frozen=True)
class ModelMarkerContext:
    run_fingerprint: str
    text_jobset_fingerprint: str
    image_jobset_fingerprint: str
    text_task_count: int
    image_task_count: int
    upstream_identities: tuple[dict[str, Any], ...]
    start_fingerprint: str

    def fingerprint_for(self, model_kind: str) -> str:
        if model_kind == "text":
            return self.text_jobset_fingerprint
        if model_kind == "image":
            return self.image_jobset_fingerprint
        raise ValueError(f"unsupported model kind: {model_kind}")

    def task_count_for(self, model_kind: str) -> int:
        if model_kind == "text":
            return self.text_task_count
        if model_kind == "image":
            return self.image_task_count
        raise ValueError(f"unsupported model kind: {model_kind}")


def build_marker_context(
    *,
    run_fingerprint: str,
    text_jobset_fingerprint: str,
    image_jobset_fingerprint: str,
    text_task_count: int,
    image_task_count: int,
    upstream_identities: Iterable[Mapping[str, Any]],
) -> ModelMarkerContext:
    upstream = tuple(
        json.loads(value)
        for value in sorted(
            canonical_json(dict(identity))
            for identity in upstream_identities
        )
    )
    base = {
        "stage": MODEL_START_STAGE,
        "schema_version": MODEL_MARKER_SCHEMA_VERSION,
        "status": "model_cache_ready_to_start",
        "run_fingerprint": str(run_fingerprint),
        "text_jobset_fingerprint": str(text_jobset_fingerprint),
        "image_jobset_fingerprint": str(image_jobset_fingerprint),
        "text_task_count": int(text_task_count),
        "image_task_count": int(image_task_count),
        "upstream_manifests": list(upstream),
    }
    if (
        not base["text_jobset_fingerprint"]
        or not base["image_jobset_fingerprint"]
        or base["text_task_count"] < 0
        or base["image_task_count"] < 0
    ):
        raise ValueError("model marker context identity is invalid")
    return ModelMarkerContext(
        run_fingerprint=str(run_fingerprint),
        text_jobset_fingerprint=str(text_jobset_fingerprint),
        image_jobset_fingerprint=str(image_jobset_fingerprint),
        text_task_count=int(text_task_count),
        image_task_count=int(image_task_count),
        upstream_identities=upstream,
        start_fingerprint=sha256_json(base),
    )


def task_jobset_fingerprint(
    tasks: Iterable[Any],
    *,
    model_kind: str,
    model_identity: str,
    prompt_version: str,
    policy_identity: Mapping[str, Any],
) -> str:
    if model_kind not in {"text", "image"}:
        raise ValueError(f"unsupported model kind: {model_kind}")
    members: list[str] = []
    for task in tasks:
        member = {
            "cache_key": str(task.cache_key),
            "source_table_id": str(task.source_table_id),
            "source_row_id": int(task.source_row_id),
            "entity_column_index": int(task.entity_column_index),
            "entity_column_name": str(task.entity_column_name),
            "entity": dict(task.entity),
            "asset": dict(task.asset),
            "candidate_attribute_names": list(
                task.candidate_attribute_names
            ),
        }
        members.append(canonical_json(member))
    members.sort()
    membership = hashlib.sha256()
    for member in members:
        membership.update(member.encode("utf-8"))
        membership.update(b"\n")
    return sha256_json(
        {
            "marker_schema_version": MODEL_MARKER_SCHEMA_VERSION,
            "model_kind": model_kind,
            "model_identity": str(model_identity),
            "prompt_version": str(prompt_version),
            "policy_identity": dict(policy_identity),
            "task_count": len(members),
            "membership_sha256": membership.hexdigest(),
        }
    )


def start_marker_payload(
    context: ModelMarkerContext,
    *,
    timestamp: float,
) -> dict[str, Any]:
    return {
        "stage": MODEL_START_STAGE,
        "schema_version": MODEL_MARKER_SCHEMA_VERSION,
        "status": "model_cache_ready_to_start",
        "run_fingerprint": context.run_fingerprint,
        "text_jobset_fingerprint": context.text_jobset_fingerprint,
        "image_jobset_fingerprint": context.image_jobset_fingerprint,
        "text_task_count": context.text_task_count,
        "image_task_count": context.image_task_count,
        "upstream_manifests": list(context.upstream_identities),
        "start_fingerprint": context.start_fingerprint,
        "timestamp": timestamp,
    }


def ready_marker_payload(
    context: ModelMarkerContext,
    *,
    timestamp: float,
) -> dict[str, Any]:
    return {
        "stage": MODEL_READY_STAGE,
        "schema_version": MODEL_MARKER_SCHEMA_VERSION,
        "status": "vllm_servers_ready",
        "model_kind": READY_MODEL_KIND,
        "run_fingerprint": context.run_fingerprint,
        "text_jobset_fingerprint": context.text_jobset_fingerprint,
        "image_jobset_fingerprint": context.image_jobset_fingerprint,
        "text_task_count": context.text_task_count,
        "image_task_count": context.image_task_count,
        "start_fingerprint": context.start_fingerprint,
        "timestamp": timestamp,
    }


def done_marker_payload(
    context: ModelMarkerContext,
    *,
    model_kind: str,
    task_count: int,
    timestamp: float,
) -> dict[str, Any]:
    expected_count = context.task_count_for(model_kind)
    if task_count != expected_count:
        raise ValueError(
            f"{model_kind} done count {task_count} != {expected_count}"
        )
    return {
        "stage": MODEL_DONE_STAGE,
        "schema_version": MODEL_MARKER_SCHEMA_VERSION,
        "status": f"{model_kind}_model_cache_precomputed",
        "model_kind": model_kind,
        "task_count": task_count,
        f"{model_kind}_task_count": task_count,
        "jobset_fingerprint": context.fingerprint_for(model_kind),
        "run_fingerprint": context.run_fingerprint,
        "text_jobset_fingerprint": context.text_jobset_fingerprint,
        "image_jobset_fingerprint": context.image_jobset_fingerprint,
        "text_task_count": context.text_task_count,
        "image_task_count": context.image_task_count,
        "start_fingerprint": context.start_fingerprint,
        "timestamp": timestamp,
    }


def read_marker(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def marker_matches(
    path: Path,
    *,
    expected_stage: str,
    expected_status: str,
    context: ModelMarkerContext | None = None,
    model_kind: str | None = None,
    task_count: int | None = None,
    run_fingerprint: str | None = None,
    jobset_fingerprint: str | None = None,
    text_jobset_fingerprint: str | None = None,
    image_jobset_fingerprint: str | None = None,
    text_task_count: int | None = None,
    image_task_count: int | None = None,
    start_fingerprint: str | None = None,
) -> bool:
    payload = read_marker(path)
    if (
        payload.get("stage") != expected_stage
        or payload.get("schema_version") != MODEL_MARKER_SCHEMA_VERSION
        or payload.get("status") != expected_status
        or not isinstance(payload.get("run_fingerprint"), str)
        or not all(
            isinstance(payload.get(field), str) and bool(payload[field])
            for field in (
                "text_jobset_fingerprint",
                "image_jobset_fingerprint",
            )
        )
        or not all(
            isinstance(payload.get(field), int)
            and not isinstance(payload[field], bool)
            and payload[field] >= 0
            for field in ("text_task_count", "image_task_count")
        )
        or not isinstance(payload.get("start_fingerprint"), str)
        or not is_sha256(payload["start_fingerprint"])
        or not is_timestamp(payload.get("timestamp"))
    ):
        return False
    if expected_stage == MODEL_START_STAGE:
        if not isinstance(payload.get("upstream_manifests"), list):
            return False
        identity = {
            key: value
            for key, value in payload.items()
            if key not in {"start_fingerprint", "timestamp"}
        }
        if payload["start_fingerprint"] != sha256_json(identity):
            return False
    elif expected_stage == MODEL_READY_STAGE:
        if payload.get("model_kind") != READY_MODEL_KIND:
            return False
    elif expected_stage == MODEL_DONE_STAGE:
        current_kind = payload.get("model_kind")
        if (
            current_kind not in {"text", "image"}
            or payload.get("status")
            != f"{current_kind}_model_cache_precomputed"
            or payload.get("jobset_fingerprint")
            != payload.get(f"{current_kind}_jobset_fingerprint")
            or payload.get("task_count")
            != payload.get(f"{current_kind}_task_count")
        ):
            return False
    if context is not None:
        run_fingerprint = context.run_fingerprint
        text_jobset_fingerprint = context.text_jobset_fingerprint
        image_jobset_fingerprint = context.image_jobset_fingerprint
        text_task_count = context.text_task_count
        image_task_count = context.image_task_count
        start_fingerprint = context.start_fingerprint
    expected = {
        "model_kind": model_kind,
        "task_count": task_count,
        "run_fingerprint": run_fingerprint,
        "jobset_fingerprint": jobset_fingerprint,
        "text_jobset_fingerprint": text_jobset_fingerprint,
        "image_jobset_fingerprint": image_jobset_fingerprint,
        "text_task_count": text_task_count,
        "image_task_count": image_task_count,
        "start_fingerprint": start_fingerprint,
    }
    return all(
        value is None or payload.get(field) == value
        for field, value in expected.items()
    )


def context_from_start_marker(path: Path) -> ModelMarkerContext | None:
    payload = read_marker(path)
    if not marker_matches(
        path,
        expected_stage=MODEL_START_STAGE,
        expected_status="model_cache_ready_to_start",
    ):
        return None
    return ModelMarkerContext(
        run_fingerprint=payload["run_fingerprint"],
        text_jobset_fingerprint=payload["text_jobset_fingerprint"],
        image_jobset_fingerprint=payload["image_jobset_fingerprint"],
        text_task_count=payload["text_task_count"],
        image_task_count=payload["image_task_count"],
        upstream_identities=tuple(payload["upstream_manifests"]),
        start_fingerprint=payload["start_fingerprint"],
    )
