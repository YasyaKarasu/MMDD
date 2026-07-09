#!/usr/bin/env python
"""Build a multimodal joinability discovery dataset from EntiTables JSON.

The builder first fixes a source-table pool that acts as the data-lake base.
It then asks local text/image models which row-level entity attributes can be
extracted from each entity's Wikipedia assets. A source column becomes a query
join column only when enough row values can be recovered from multimodal
evidence.
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import mimetypes
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable
from urllib.parse import quote

try:
    import requests
except ImportError:  # pragma: no cover - exercised only in minimal envs.
    requests = None  # type: ignore[assignment]

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - exercised only in minimal envs.
    tqdm = None  # type: ignore[assignment]

from build_mm_table_dataset import (
    ShardedJsonlWriter,
    WikipediaClient,
    build_bridge_assets,
    finalize_entities,
    iter_jsonl_records,
    iter_with_progress,
    is_useful_image,
    normalize_title,
    parse_source_table,
    read_entitables_json,
    select_relevant_text_chunks,
    split_text_asset_content,
    update_entities_from_table,
    write_jsonl_record,
    write_sharded_jsonl,
    write_table_asset_links_from_jsonl,
)
from stage1_io import (
    clean_text,
    column_profiles,
    get_cell,
    get_cell_text,
    get_column_name,
    make_columns,
    setup_logging,
    stable_hash,
    write_json,
    write_jsonl,
)


PROMPT_VERSION = "entity_attribute_extraction_v2_entity_connection"
DEFAULT_SHARED_CACHE_DIR = Path("cache") / "mm_joinability"


def resolve_shared_cache_paths(args: argparse.Namespace) -> dict[str, Path]:
    root_dir = Path(args.cache_dir).expanduser().resolve()
    wikipedia_cache_dir = (
        Path(args.wikipedia_cache_dir).expanduser().resolve()
        if args.wikipedia_cache_dir
        else root_dir / "wikipedia"
    )
    wikipedia_image_dir = (
        Path(args.wikipedia_image_dir).expanduser().resolve()
        if args.wikipedia_image_dir
        else root_dir / "images"
    )
    return {
        "root_dir": root_dir,
        "wikipedia_cache_dir": wikipedia_cache_dir,
        "wikipedia_image_dir": wikipedia_image_dir,
        "model_attribute_extractions": root_dir / "model_attribute_extractions.jsonl",
    }


def normalize(value: Any) -> str:
    return clean_text(value).casefold()


def values_match(predicted: Any, expected: Any) -> bool:
    pred = normalize(predicted)
    exp = normalize(expected)
    if not pred or not exp:
        return False
    if pred == exp:
        return True
    return len(exp) >= 4 and (exp in pred or pred in exp)


def iter_json_objects(text: str) -> Iterable[dict[str, Any]]:
    decoder = json.JSONDecoder()
    start = 0
    while True:
        start = text.find("{", start)
        if start < 0:
            break
        try:
            payload, end = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            start += 1
            continue
        if isinstance(payload, dict):
            yield payload
        start += max(1, end)


def safe_json_object(text: str) -> dict[str, Any]:
    text = clean_text(text)
    if not text:
        return {}
    try:
        payload = json.loads(text)
        return payload if isinstance(payload, dict) else {}
    except json.JSONDecodeError:
        pass
    objects = list(iter_json_objects(text))
    for payload in reversed(objects):
        if isinstance(payload.get("attributes"), list):
            return payload
    return objects[-1] if objects else {}


def is_placeholder_text(value: str) -> bool:
    value = clean_text(value)
    return len(value) >= 2 and value.startswith("<") and value.endswith(">")


def normalize_extracted_attributes(
    payload: dict[str, Any],
    candidate_attributes: list[str] | None = None,
    require_connection_evidence: bool = False,
) -> list[dict[str, Any]]:
    items = payload.get("attributes")
    if not isinstance(items, list):
        return []
    candidate_names = {normalize(name) for name in candidate_attributes or [] if clean_text(name)}
    attrs: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        name = clean_text(item.get("name") or item.get("attribute") or item.get("column_name"))
        value = clean_text(item.get("value"))
        if not name or not value:
            continue
        if is_placeholder_text(name) or is_placeholder_text(value):
            continue
        if candidate_names and normalize(name) not in candidate_names:
            continue
        connection_evidence = clean_text(
            item.get("connection_evidence")
            or item.get("entity_connection_evidence")
            or item.get("link_evidence")
        )[:500]
        if require_connection_evidence and (not connection_evidence or is_placeholder_text(connection_evidence)):
            continue
        attr = {
            "name": name,
            "value": value,
            "evidence": clean_text(item.get("evidence") or item.get("rationale"))[:500],
        }
        if connection_evidence:
            attr["connection_evidence"] = connection_evidence
        attrs.append(attr)
    return attrs


def image_data_url(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{data}"


def usage_token_count(usage: dict[str, Any], *keys: str) -> int:
    for key in keys:
        value = usage.get(key)
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return 0


def model_call_stats_summary(extractor: Any) -> dict[str, dict[str, int | float]]:
    stats = getattr(extractor, "model_call_stats", None)
    summary = getattr(stats, "summary", None)
    if callable(summary):
        return summary()
    return ModelCallStats().summary()


class ModelCallStats:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._stats: dict[str, dict[str, int | float]] = {
            "text": self._empty_bucket(),
            "image": self._empty_bucket(),
        }

    @staticmethod
    def _empty_bucket() -> dict[str, int | float]:
        return {
            "requests": 0,
            "failed_requests": 0,
            "elapsed_seconds": 0.0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "responses_with_usage": 0,
        }

    def record(
        self,
        model_kind: str,
        *,
        elapsed_seconds: float,
        usage: dict[str, Any] | None = None,
        failed: bool = False,
    ) -> None:
        with self._lock:
            bucket = self._stats.setdefault(model_kind, self._empty_bucket())
            bucket["requests"] = int(bucket["requests"]) + 1
            bucket["elapsed_seconds"] = float(bucket["elapsed_seconds"]) + max(0.0, elapsed_seconds)
            if failed:
                bucket["failed_requests"] = int(bucket["failed_requests"]) + 1
            if isinstance(usage, dict):
                prompt_tokens = usage_token_count(usage, "prompt_tokens", "input_tokens")
                completion_tokens = usage_token_count(usage, "completion_tokens", "output_tokens")
                total_tokens = usage_token_count(usage, "total_tokens")
                if total_tokens <= 0:
                    total_tokens = prompt_tokens + completion_tokens
                bucket["prompt_tokens"] = int(bucket["prompt_tokens"]) + prompt_tokens
                bucket["completion_tokens"] = int(bucket["completion_tokens"]) + completion_tokens
                bucket["total_tokens"] = int(bucket["total_tokens"]) + total_tokens
                bucket["responses_with_usage"] = int(bucket["responses_with_usage"]) + 1

    def summary(self) -> dict[str, dict[str, int | float]]:
        summary: dict[str, dict[str, int | float]] = {}
        with self._lock:
            items = [(model_kind, dict(bucket)) for model_kind, bucket in self._stats.items()]
        for model_kind, bucket in sorted(items):
            summary[model_kind] = {
                "requests": int(bucket["requests"]),
                "failed_requests": int(bucket["failed_requests"]),
                "elapsed_seconds": round(float(bucket["elapsed_seconds"]), 6),
                "prompt_tokens": int(bucket["prompt_tokens"]),
                "completion_tokens": int(bucket["completion_tokens"]),
                "total_tokens": int(bucket["total_tokens"]),
                "responses_with_usage": int(bucket["responses_with_usage"]),
            }
        return summary


class LocalAttributeExtractor:
    """OpenAI-compatible client for local text and image extraction models."""

    def __init__(self, args: argparse.Namespace) -> None:
        if requests is None:
            raise RuntimeError("requests is required for local model calls")
        self.text_model_base_url = args.text_model_base_url.rstrip("/")
        self.text_model_name = args.text_model_name
        self.text_model_api_key = args.text_model_api_key
        self.image_model_base_url = args.image_model_base_url.rstrip("/")
        self.image_model_name = args.image_model_name
        self.image_model_api_key = args.image_model_api_key
        self.timeout = args.model_timeout_seconds
        self.temperature = args.model_temperature
        self.max_tokens = args.model_max_tokens
        self.disable_thinking = args.disable_thinking
        self.max_retries = args.model_max_retries
        self.retry_sleep = args.model_retry_sleep_seconds
        self.model_call_stats = ModelCallStats()

    def chat(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str | None,
        messages: list[dict[str, Any]],
        model_kind: str = "text",
    ) -> str:
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        payload = {
            "model": model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if self.disable_thinking:
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            started = time.perf_counter()
            try:
                response = requests.post(
                    f"{base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=self.timeout,
                )
                status_code = int(getattr(response, "status_code", 200) or 200)
                if status_code >= 400:
                    body = clean_text(getattr(response, "text", ""))[:500]
                    raise RuntimeError(f"HTTP {status_code}: {body}")
                response.raise_for_status()
                data = response.json()
                content = clean_text(data["choices"][0]["message"]["content"])
            except Exception as exc:  # pragma: no cover - integration only.
                self.model_call_stats.record(
                    model_kind,
                    elapsed_seconds=time.perf_counter() - started,
                    failed=True,
                )
                last_error = exc
                if attempt < self.max_retries:
                    time.sleep(self.retry_sleep)
            else:
                usage = data.get("usage") if isinstance(data, dict) else None
                self.model_call_stats.record(
                    model_kind,
                    elapsed_seconds=time.perf_counter() - started,
                    usage=usage if isinstance(usage, dict) else None,
                )
                return content
        raise RuntimeError(f"Local model call failed: {last_error}")

    def extraction_prompt(
        self,
        *,
        entity_text: str,
        entity_wiki_title: str,
        candidate_attributes: list[str],
    ) -> str:
        return (
            "You extract factual attributes about one entity from one evidence item.\n"
            f"Entity display text: {entity_text}\n"
            f"Entity Wikipedia title: {entity_wiki_title}\n"
            "Candidate attribute names from the table:\n"
            + "\n".join(f"- {name}" for name in candidate_attributes)
            + "\nReturn strict JSON only in this shape:\n"
            '{"attributes":[{"name":"<one candidate attribute name>","value":"<extracted value>","evidence":"<short quote or visual evidence>","connection_evidence":"<why this evidence item itself can be linked to the entity>"}]}\n'
            "Only include attributes directly supported by the evidence item. "
            "Only include an attribute when the evidence item itself lets a reader connect the evidence to this entity, "
            "for example through the entity name, an alias, a visible/quoted identifier, or an entity-specific attribute. "
            "Do not rely on Wikipedia page provenance, source URL, or the fact that the asset was collected from the entity page. "
            "Do not guess. Do not include attributes outside the candidate list."
        )

    def extract(self, asset: dict[str, Any], entity: dict[str, Any], candidate_attributes: list[str]) -> dict[str, Any]:
        prompt = self.extraction_prompt(
            entity_text=clean_text(entity.get("cell_text")),
            entity_wiki_title=clean_text(entity.get("wiki_title")),
            candidate_attributes=candidate_attributes,
        )
        if asset.get("asset_type") == "text":
            content = clean_text(asset.get("content"))[:6000]
            messages = [
                {"role": "system", "content": "You are a precise information extraction engine."},
                {"role": "user", "content": f"{prompt}\n\nText evidence:\n{content}"},
            ]
            raw = self.chat(
                base_url=self.text_model_base_url,
                model=self.text_model_name,
                api_key=self.text_model_api_key,
                messages=messages,
                model_kind="text",
            )
        elif asset.get("asset_type") == "image":
            image_url = clean_text(asset.get("image_url"))
            local_path = clean_text(asset.get("local_path"))
            if local_path and Path(local_path).exists():
                image_url = image_data_url(Path(local_path))
            if not image_url:
                raise ValueError(f"Image asset {asset.get('asset_id')} has no usable image URL or local path")
            messages = [
                {"role": "system", "content": "You are a precise visual information extraction engine."},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                },
            ]
            raw = self.chat(
                base_url=self.image_model_base_url,
                model=self.image_model_name,
                api_key=self.image_model_api_key,
                messages=messages,
                model_kind="image",
            )
        else:
            return {"attributes": [], "raw_response": "", "error": f"unsupported_asset_type:{asset.get('asset_type')}"}
        payload = safe_json_object(raw)
        return {
            "attributes": normalize_extracted_attributes(
                payload,
                candidate_attributes,
                require_connection_evidence=True,
            ),
            "raw_response": raw,
            "error": "",
        }


class ExtractionCache:
    def __init__(self, path: Path, reuse: bool = True) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.items: dict[str, dict[str, Any]] = {}
        if reuse and path.exists():
            for record in iter_jsonl_records([path]):
                key = clean_text(record.get("cache_key"))
                if key:
                    self.items[key] = record

    def get(self, key: str) -> dict[str, Any] | None:
        return self.items.get(key)

    def put(self, key: str, record: dict[str, Any]) -> None:
        self.items[key] = record
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


class ModelAnalysisProgress:
    def __init__(self, *, total: int, cached_keys: set[str], enabled: bool) -> None:
        self.enabled = enabled and tqdm is not None
        self.completed_keys: set[str] = set(cached_keys)
        self.cached = len(cached_keys)
        self.model = 0
        self.errors = 0
        self.bar = None
        if self.enabled:
            self.bar = tqdm(
                total=total,
                initial=len(cached_keys),
                desc="Local model analysis",
                unit="asset",
                dynamic_ncols=True,
            )
            self._postfix()

    def _postfix(self) -> None:
        if self.bar is not None:
            self.bar.set_postfix(cached=self.cached, model=self.model, errors=self.errors)

    def mark(self, cache_key: str, status: str) -> None:
        if cache_key in self.completed_keys:
            return
        self.completed_keys.add(cache_key)
        if status == "cached":
            self.cached += 1
        elif status == "error":
            self.model += 1
            self.errors += 1
        else:
            self.model += 1
        if self.bar is not None:
            self.bar.update(1)
            self._postfix()

    def close(self) -> None:
        if self.bar is not None:
            self.bar.close()


@dataclass(frozen=True)
class ExtractionTask:
    order: int
    cache_key: str
    source_table_id: str
    source_row_id: int
    entity_column_index: int
    entity_column_name: str
    entity: dict[str, Any]
    asset: dict[str, Any]
    candidate_attribute_names: list[str]


@dataclass
class ModelConcurrencyState:
    text_workers: int
    image_workers: int
    text_oom_downgrades: int = 0
    image_oom_downgrades: int = 0

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "ModelConcurrencyState":
        return cls(
            text_workers=max(1, int(getattr(args, "text_model_workers", 1) or 1)),
            image_workers=max(1, int(getattr(args, "image_model_workers", 1) or 1)),
        )

    def workers_for(self, model_kind: str) -> int:
        return self.image_workers if model_kind == "image" else self.text_workers

    def downgrade_after_oom(self, model_kind: str) -> None:
        if model_kind == "image":
            if self.image_workers > 1:
                self.image_oom_downgrades += 1
            self.image_workers = 1
        else:
            if self.text_workers > 1:
                self.text_oom_downgrades += 1
            self.text_workers = 1

    def summary(self) -> dict[str, int]:
        return {
            "text_workers": self.text_workers,
            "image_workers": self.image_workers,
            "text_oom_downgrades": self.text_oom_downgrades,
            "image_oom_downgrades": self.image_oom_downgrades,
        }


def model_kind_for_asset(asset: dict[str, Any]) -> str:
    return "image" if asset.get("asset_type") == "image" else "text"


def is_oom_error(message: Any) -> bool:
    text = clean_text(message).casefold()
    if not text:
        return False
    return any(
        marker in text
        for marker in (
            "out of memory",
            "cuda oom",
            "cuda error: out of memory",
            "torch.cuda.outofmemoryerror",
            "cudnn_status_alloc_failed",
            "memory allocation",
            "kv cache",
        )
    )


def extraction_record_from_result(task: ExtractionTask, result: dict[str, Any]) -> dict[str, Any]:
    return {
        "cache_key": task.cache_key,
        "prompt_version": PROMPT_VERSION,
        "entity_id": task.entity["entity_id"],
        "entity_text": task.entity["cell_text"],
        "entity_wiki_title": task.entity["wiki_title"],
        "asset_id": task.asset["asset_id"],
        "asset_type": task.asset.get("asset_type"),
        "candidate_attribute_names": task.candidate_attribute_names,
        "attributes": result.get("attributes", []),
        "raw_response": result.get("raw_response", ""),
        "error": clean_text(result.get("error")),
    }


def run_extraction_task(extractor: LocalAttributeExtractor, task: ExtractionTask) -> dict[str, Any]:
    try:
        result = extractor.extract(task.asset, task.entity, task.candidate_attribute_names)
    except Exception as exc:
        result = {"attributes": [], "raw_response": "", "error": str(exc)}
    return extraction_record_from_result(task, result)


def run_extraction_task_group(
    *,
    extractor: LocalAttributeExtractor,
    tasks: list[ExtractionTask],
    workers: int,
) -> dict[str, dict[str, Any]]:
    if not tasks:
        return {}
    workers = max(1, min(workers, len(tasks)))
    if workers == 1:
        return {task.cache_key: run_extraction_task(extractor, task) for task in tasks}

    records: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        future_to_task = {pool.submit(run_extraction_task, extractor, task): task for task in tasks}
        for future in as_completed(future_to_task):
            task = future_to_task[future]
            try:
                records[task.cache_key] = future.result()
            except Exception as exc:  # pragma: no cover - run_extraction_task catches normal failures.
                records[task.cache_key] = extraction_record_from_result(
                    task,
                    {"attributes": [], "raw_response": "", "error": str(exc)},
                )
    return records


def run_extraction_kind_adaptive(
    *,
    extractor: LocalAttributeExtractor,
    tasks: list[ExtractionTask],
    model_kind: str,
    state: ModelConcurrencyState,
) -> dict[str, dict[str, Any]]:
    workers = state.workers_for(model_kind)
    records = run_extraction_task_group(extractor=extractor, tasks=tasks, workers=workers)
    oom_tasks = [
        task
        for task in tasks
        if is_oom_error(records.get(task.cache_key, {}).get("error"))
    ]
    if oom_tasks and workers > 1:
        logging.warning(
            "%s model hit OOM-like errors with %d workers; retrying %d failed tasks serially and downgrading future %s workers to 1",
            model_kind,
            workers,
            len(oom_tasks),
            model_kind,
        )
        state.downgrade_after_oom(model_kind)
        records.update(run_extraction_task_group(extractor=extractor, tasks=oom_tasks, workers=1))
    return records


def run_uncached_extraction_tasks(
    *,
    extractor: LocalAttributeExtractor,
    tasks: list[ExtractionTask],
    state: ModelConcurrencyState,
) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[ExtractionTask]] = {"text": [], "image": []}
    for task in tasks:
        groups[model_kind_for_asset(task.asset)].append(task)

    records: dict[str, dict[str, Any]] = {}
    active_groups = [(kind, group_tasks) for kind, group_tasks in groups.items() if group_tasks]
    if len(active_groups) <= 1:
        for kind, group_tasks in active_groups:
            records.update(
                run_extraction_kind_adaptive(
                    extractor=extractor,
                    tasks=group_tasks,
                    model_kind=kind,
                    state=state,
                )
            )
        return records

    with ThreadPoolExecutor(max_workers=len(active_groups)) as pool:
        futures = {
            pool.submit(
                run_extraction_kind_adaptive,
                extractor=extractor,
                tasks=group_tasks,
                model_kind=kind,
                state=state,
            ): kind
            for kind, group_tasks in active_groups
        }
        for future in as_completed(futures):
            records.update(future.result())
    return records


def cached_extraction_is_reusable(record: dict[str, Any], args: argparse.Namespace) -> bool:
    if clean_text(record.get("error")) and not getattr(args, "cache_failed_model_outputs", False):
        return False
    if getattr(args, "refresh_invalid_model_cache", False) and should_refresh_cached_extraction(record):
        return False
    return True


def resolve_extraction_tasks(
    *,
    extractor: LocalAttributeExtractor,
    cache: ExtractionCache,
    tasks: list[ExtractionTask],
    args: argparse.Namespace,
    state: ModelConcurrencyState,
    progress: ModelAnalysisProgress | None = None,
) -> list[tuple[ExtractionTask, dict[str, Any]]]:
    resolved_by_key: dict[str, dict[str, Any]] = {}
    uncached_by_key: dict[str, ExtractionTask] = {}
    for task in tasks:
        if task.cache_key in resolved_by_key or task.cache_key in uncached_by_key:
            continue
        cached = cache.get(task.cache_key)
        if cached:
            cached_record = cached
            if getattr(args, "reparse_cached_model_outputs", True):
                cached_record, changed = reparse_extraction_record(
                    cached_record,
                    task.candidate_attribute_names,
                    require_connection_evidence=True,
                )
                if changed:
                    cache.put(task.cache_key, cached_record)
            if cached_extraction_is_reusable(cached_record, args):
                resolved_by_key[task.cache_key] = cached_record
                if progress is not None:
                    progress.mark(task.cache_key, "cached")
                continue
        uncached_by_key[task.cache_key] = task

    model_records = run_uncached_extraction_tasks(
        extractor=extractor,
        tasks=list(uncached_by_key.values()),
        state=state,
    )
    for cache_key, record in model_records.items():
        resolved_by_key[cache_key] = record
        has_error = bool(clean_text(record.get("error")))
        if not has_error or getattr(args, "cache_failed_model_outputs", False):
            cache.put(cache_key, record)
        if progress is not None:
            progress.mark(cache_key, "error" if has_error else "model")

    return [(task, resolved_by_key[task.cache_key]) for task in tasks if task.cache_key in resolved_by_key]


def source_splits(source_records: list[dict[str, str]], args: argparse.Namespace) -> dict[str, Any]:
    ratios = [args.train_ratio, args.dev_ratio, args.test_ratio]
    total = sum(ratios)
    if total <= 0:
        raise ValueError("train/dev/test ratios must sum to a positive value")
    ratios = [value / total for value in ratios]
    groups: dict[str, list[str]] = defaultdict(list)
    for record in source_records:
        key = clean_text(record.get("page_title")) if args.split_by == "page_title" else ""
        groups[key or record["source_table_id"]].append(record["source_table_id"])
    keys = sorted(groups, key=lambda key: stable_hash("split", args.seed, key))
    train_n = int(len(keys) * ratios[0])
    dev_n = int(len(keys) * ratios[1])
    if len(keys) >= 3:
        train_n = max(1, train_n) if ratios[0] > 0 else 0
        dev_n = max(1, dev_n) if ratios[1] > 0 else 0
        if train_n + dev_n >= len(keys):
            dev_n = max(0, len(keys) - train_n - 1)
    split_keys = {
        "train": keys[:train_n],
        "dev": keys[train_n : train_n + dev_n],
        "test": keys[train_n + dev_n :],
    }
    splits: dict[str, Any] = {}
    for split, selected in split_keys.items():
        splits[split] = {
            "source_table_ids": sorted({source_id for key in selected for source_id in groups[key]}),
            "query_table_ids": [],
            "data_lake_table_ids": [],
        }
    splits["split_key"] = "page_title_or_source_table_id" if args.split_by == "page_title" else "source_table_id"
    splits["note"] = "source-level split; data_lake contains generated targets for queryable tables and raw tables for rejected tables"
    return splits


def split_map(splits: dict[str, Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    for split, payload in splits.items():
        if isinstance(payload, dict):
            for source_id in payload.get("source_table_ids", []):
                result[source_id] = split
    return result


def choose_entity_column(table: dict[str, Any]) -> int | None:
    candidates = list(table.get("metadata", {}).get("candidate_entity_columns", []) or [])
    if not candidates:
        return None
    profiles = column_profiles(table)
    candidates.sort(
        key=lambda idx: (
            -float(profiles.get(int(idx), {}).get("wiki_link_ratio", 0.0)),
            int(idx),
        )
    )
    return int(candidates[0])


def candidate_attribute_columns(table: dict[str, Any], entity_col: int, min_non_empty_ratio: float) -> list[int]:
    profiles = column_profiles(table)
    cols: list[int] = []
    for column in table.get("columns", []):
        try:
            idx = int(column.get("column_index"))
        except (TypeError, ValueError):
            continue
        if idx == entity_col:
            continue
        if float(profiles.get(idx, {}).get("non_empty_ratio", 0.0)) < min_non_empty_ratio:
            continue
        cols.append(idx)
    return cols


def row_id(row: dict[str, Any], fallback: int) -> int:
    try:
        return int(row.get("row_id", fallback))
    except (TypeError, ValueError):
        return fallback


def project_selected_rows(
    table: dict[str, Any],
    column_indices: list[int],
    source_row_ids: set[int],
    *,
    min_required_cols: int,
) -> tuple[list[dict[str, Any]], list[int]]:
    rows: list[dict[str, Any]] = []
    source_rows: list[int] = []
    required = set(column_indices[:min_required_cols])
    for fallback, source_row in enumerate(table.get("rows", [])):
        source_row_id = row_id(source_row, fallback)
        if source_row_id not in source_row_ids:
            continue
        values = {idx: get_cell_text(source_row, idx) for idx in column_indices}
        if any(not values.get(idx) for idx in required):
            continue
        cells = []
        for out_idx, source_idx in enumerate(column_indices):
            cell = dict(get_cell(source_row, source_idx))
            cell["column_index"] = out_idx
            cell["source_column_index"] = source_idx
            cell["column_name"] = get_column_name(table, source_idx)
            cell["text"] = values.get(source_idx, "")
            cells.append(cell)
        rows.append({"row_id": len(rows), "source_row_id": source_row_id, "cells": cells})
        source_rows.append(source_row_id)
    return rows, source_rows


def table_record(
    *,
    table_id: str,
    role: str,
    split: str,
    source_table: dict[str, Any],
    column_indices: list[int],
    rows: list[dict[str, Any]],
    source_row_indices: list[int],
    extra: dict[str, Any],
) -> dict[str, Any]:
    return {
        "table_id": table_id,
        "object_id": table_id,
        "object_type": "table",
        "role": role,
        "split": split,
        "source_table_id": source_table["source_table_id"],
        "page_title": clean_text(source_table.get("page_title")),
        "caption": clean_text(source_table.get("caption")),
        "section_title": clean_text(source_table.get("section_title")),
        "columns": make_columns(source_table, column_indices),
        "rows": rows,
        "source_column_indices": column_indices,
        "source_row_indices": source_row_indices,
        "provenance": {
            "builder": "build_mm_joinability_dataset.py",
            "source_file": source_table.get("source_file"),
        },
        **extra,
    }


def raw_data_lake_record(source_table: dict[str, Any], split: str) -> dict[str, Any]:
    cols = [int(col["column_index"]) for col in source_table.get("columns", [])]
    source_rows = [row_id(row, fallback) for fallback, row in enumerate(source_table.get("rows", []))]
    rows, projected_source_rows = project_selected_rows(
        source_table,
        cols,
        set(source_rows),
        min_required_cols=0,
    )
    return table_record(
        table_id=f"dl_raw_{source_table['source_table_id']}",
        role="raw_data_lake_table",
        split=split,
        source_table=source_table,
        column_indices=cols,
        rows=rows,
        source_row_indices=projected_source_rows,
        extra={"queryable": False, "reason": "no_column_met_recovered_value_ratio"},
    )


def context_columns(table: dict[str, Any], excluded: set[int], limit: int) -> list[int]:
    profiles = column_profiles(table)
    candidates: list[tuple[float, float, int]] = []
    for column in table.get("columns", []):
        idx = int(column.get("column_index"))
        if idx in excluded:
            continue
        profile = profiles.get(idx, {})
        candidates.append(
            (
                float(profile.get("non_empty_ratio", 0.0)),
                -float(profile.get("unique_ratio", 1.0)),
                idx,
            )
        )
    candidates.sort(reverse=True)
    if limit <= 0:
        return [idx for _non_empty, _unique, idx in candidates]
    return [idx for _non_empty, _unique, idx in candidates[:limit]]


def extraction_cache_key(
    *,
    asset_id: str,
    entity_id: str,
    candidate_attribute_names: list[str],
    asset_type: str,
    args: argparse.Namespace,
) -> str:
    model = args.image_model_name if asset_type == "image" else args.text_model_name
    return stable_hash(PROMPT_VERSION, model, asset_id, entity_id, "|".join(candidate_attribute_names), length=24)


def reparse_extraction_record(
    record: dict[str, Any],
    candidate_attribute_names: list[str],
    *,
    require_connection_evidence: bool = False,
) -> tuple[dict[str, Any], bool]:
    raw_response = clean_text(record.get("raw_response"))
    if not raw_response:
        return record, False
    attributes = normalize_extracted_attributes(
        safe_json_object(raw_response),
        candidate_attribute_names,
        require_connection_evidence=require_connection_evidence,
    )
    if attributes == record.get("attributes"):
        return record, False
    updated = dict(record)
    updated["attributes"] = attributes
    updated["reparsed_raw_response"] = True
    return updated, True


def should_refresh_cached_extraction(record: dict[str, Any]) -> bool:
    return not clean_text(record.get("error")) and not record.get("attributes")


def estimate_model_analysis_keys(
    *,
    source_paths: Iterable[Path],
    assets: dict[str, dict[str, Any]],
    entity_to_assets: dict[str, list[str]],
    wiki_to_entity_id: dict[str, str],
    args: argparse.Namespace,
) -> set[str]:
    keys: set[str] = set()
    for source_table in iter_jsonl_records(source_paths):
        entity_col = choose_entity_column(source_table)
        if entity_col is None:
            continue
        attribute_cols = candidate_attribute_columns(source_table, entity_col, args.min_column_non_empty_ratio)
        if not attribute_cols:
            continue
        candidate_attribute_names = [get_column_name(source_table, col) for col in attribute_cols]
        for source_row in source_table.get("rows", []):
            entity_cell = get_cell(source_row, entity_col)
            wiki_title = clean_text(entity_cell.get("wiki_title"))
            entity_id = wiki_to_entity_id.get(wiki_title)
            if not wiki_title or not entity_id:
                continue
            for asset_id in entity_to_assets.get(entity_id, []):
                asset = assets.get(asset_id)
                if not asset:
                    continue
                keys.add(
                    extraction_cache_key(
                        asset_id=asset_id,
                        entity_id=entity_id,
                        candidate_attribute_names=candidate_attribute_names,
                        asset_type=str(asset.get("asset_type")),
                        args=args,
                    )
                )
    return keys


def extract_asset_attributes(
    *,
    extractor: LocalAttributeExtractor,
    cache: ExtractionCache,
    asset: dict[str, Any],
    entity: dict[str, Any],
    candidate_attribute_names: list[str],
    args: argparse.Namespace,
    progress: ModelAnalysisProgress | None = None,
) -> dict[str, Any]:
    cache_key = extraction_cache_key(
        asset_id=asset["asset_id"],
        entity_id=entity["entity_id"],
        candidate_attribute_names=candidate_attribute_names,
        asset_type=str(asset.get("asset_type")),
        args=args,
    )
    cached = cache.get(cache_key)
    if cached:
        cached_record = cached
        if getattr(args, "reparse_cached_model_outputs", True):
            cached_record, changed = reparse_extraction_record(
                cached_record,
                candidate_attribute_names,
                require_connection_evidence=True,
            )
            if changed:
                cache.put(cache_key, cached_record)
        if cached_extraction_is_reusable(cached_record, args):
            if progress is not None:
                progress.mark(cache_key, "cached")
            return cached_record
    try:
        result = extractor.extract(asset, entity, candidate_attribute_names)
    except Exception as exc:
        result = {"attributes": [], "raw_response": "", "error": str(exc)}
    record = {
        "cache_key": cache_key,
        "prompt_version": PROMPT_VERSION,
        "entity_id": entity["entity_id"],
        "entity_text": entity["cell_text"],
        "entity_wiki_title": entity["wiki_title"],
        "asset_id": asset["asset_id"],
        "asset_type": asset.get("asset_type"),
        "candidate_attribute_names": candidate_attribute_names,
        "attributes": result.get("attributes", []),
        "raw_response": result.get("raw_response", ""),
        "error": result.get("error", ""),
    }
    if not record["error"] or getattr(args, "cache_failed_model_outputs", False):
        cache.put(cache_key, record)
    if progress is not None:
        progress.mark(cache_key, "error" if record["error"] else "model")
    return record


def asset_preview(asset: dict[str, Any]) -> dict[str, Any]:
    if asset.get("asset_type") == "text":
        return {
            "title": clean_text(asset.get("entity_wiki_title")),
            "content_snippet": clean_text(asset.get("content"))[:1600],
            "url": clean_text(asset.get("url")),
            "source": clean_text(asset.get("source")),
        }
    return {
        "title": clean_text(asset.get("entity_wiki_title")),
        "file_name": clean_text(asset.get("file_name")),
        "relative_path": clean_text(asset.get("relative_path")),
        "local_path": clean_text(asset.get("local_path")),
        "image_url": clean_text(asset.get("image_url")),
        "description_url": clean_text(asset.get("description_url")),
        "source": clean_text(asset.get("source")),
    }


def build_table_join_records(
    *,
    source_table: dict[str, Any],
    split: str,
    assets: dict[str, dict[str, Any]],
    entity_to_assets: dict[str, list[str]],
    wiki_to_entity_id: dict[str, str],
    extractor: LocalAttributeExtractor,
    cache: ExtractionCache,
    progress: ModelAnalysisProgress | None,
    concurrency_state: ModelConcurrencyState,
    extraction_writer: ShardedJsonlWriter,
    recovery_writer: ShardedJsonlWriter,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    entity_col = choose_entity_column(source_table)
    if entity_col is None:
        raw = raw_data_lake_record(source_table, split)
        return [], [raw], [], {"reason": "no_entity_column", "qualified_columns": []}

    attribute_cols = candidate_attribute_columns(source_table, entity_col, args.min_column_non_empty_ratio)
    if not attribute_cols:
        raw = raw_data_lake_record(source_table, split)
        return [], [raw], [], {"reason": "no_candidate_attribute_columns", "qualified_columns": []}

    candidate_attribute_names = [get_column_name(source_table, col) for col in attribute_cols]
    eligible_rows_by_col: dict[int, set[int]] = defaultdict(set)
    recovered_rows_by_col: dict[int, set[int]] = defaultdict(set)
    recoveries_by_col: dict[int, list[dict[str, Any]]] = defaultdict(list)
    extraction_count = 0
    extraction_tasks: list[ExtractionTask] = []
    source_rows_by_id = {
        row_id(source_row, fallback): source_row
        for fallback, source_row in enumerate(source_table.get("rows", []))
    }

    for fallback, source_row in enumerate(source_table.get("rows", [])):
        source_row_id = row_id(source_row, fallback)
        entity_cell = get_cell(source_row, entity_col)
        wiki_title = clean_text(entity_cell.get("wiki_title"))
        entity_text = clean_text(entity_cell.get("text"))
        entity_id = wiki_to_entity_id.get(wiki_title)
        if not wiki_title or not entity_id:
            continue
        for attr_col in attribute_cols:
            if clean_text(get_cell_text(source_row, attr_col)):
                eligible_rows_by_col[attr_col].add(source_row_id)
        asset_ids = entity_to_assets.get(entity_id, [])
        if not asset_ids:
            continue
        entity = {
            "entity_id": entity_id,
            "wiki_title": wiki_title,
            "cell_text": entity_text,
            "entity_column_index": entity_col,
            "entity_column_name": get_column_name(source_table, entity_col),
        }
        for asset_id in asset_ids:
            asset = assets.get(asset_id)
            if not asset:
                continue
            cache_key = extraction_cache_key(
                asset_id=asset["asset_id"],
                entity_id=entity["entity_id"],
                candidate_attribute_names=candidate_attribute_names,
                asset_type=str(asset.get("asset_type")),
                args=args,
            )
            extraction_tasks.append(
                ExtractionTask(
                    order=len(extraction_tasks),
                    cache_key=cache_key,
                    source_table_id=source_table["source_table_id"],
                    source_row_id=source_row_id,
                    entity_column_index=entity_col,
                    entity_column_name=get_column_name(source_table, entity_col),
                    entity=entity,
                    asset=asset,
                    candidate_attribute_names=candidate_attribute_names,
                )
            )

    for task, extraction in resolve_extraction_tasks(
        extractor=extractor,
        cache=cache,
        tasks=extraction_tasks,
        args=args,
        state=concurrency_state,
        progress=progress,
    ):
        source_row_id = task.source_row_id
        entity = task.entity
        asset = task.asset
        write_jsonl_record(
            extraction_writer,
            {
                **extraction,
                "source_table_id": task.source_table_id,
                "source_row_id": source_row_id,
                "entity_column_index": task.entity_column_index,
                "entity_column_name": task.entity_column_name,
            },
        )
        extraction_count += 1
        attr_by_name: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for item in extraction.get("attributes", []):
            attr_by_name[normalize(item.get("name"))].append(item)
        source_row = source_rows_by_id.get(source_row_id)
        if source_row is None:
            continue
        for attr_col in attribute_cols:
            attr_name = get_column_name(source_table, attr_col)
            expected = clean_text(get_cell_text(source_row, attr_col))
            if not expected:
                continue
            for predicted in attr_by_name.get(normalize(attr_name), []):
                if not values_match(predicted.get("value"), expected):
                    continue
                recovered_rows_by_col[attr_col].add(source_row_id)
                recoveries_by_col[attr_col].append(
                    {
                        "source_table_id": source_table["source_table_id"],
                        "source_row_id": source_row_id,
                        "split": split,
                        "query_entity": entity,
                        "recovered_attribute": {
                            "column_index": attr_col,
                            "column_name": attr_name,
                            "value": expected,
                            "model_value": clean_text(predicted.get("value")),
                            "hidden_in_query": True,
                        },
                        "evidence": {
                            "asset_id": asset["asset_id"],
                            "asset_type": asset.get("asset_type"),
                            **asset_preview(asset),
                            "model_evidence": clean_text(predicted.get("evidence")),
                            "model_connection_evidence": clean_text(predicted.get("connection_evidence")),
                            "extraction_cache_key": extraction.get("cache_key"),
                        },
                    }
                )
                break

    qualified_cols: list[dict[str, Any]] = []
    for attr_col in attribute_cols:
        eligible = eligible_rows_by_col.get(attr_col, set())
        recovered = recovered_rows_by_col.get(attr_col, set())
        ratio = len(recovered) / max(1, len(eligible))
        if len(eligible) >= args.min_recovery_denominator and ratio >= args.min_recovered_value_ratio:
            qualified_cols.append(
                {
                    "column_index": attr_col,
                    "column_name": get_column_name(source_table, attr_col),
                    "eligible_rows": len(eligible),
                    "recovered_rows": len(recovered),
                    "recovered_value_ratio": round(ratio, 6),
                }
            )

    if not qualified_cols:
        raw = raw_data_lake_record(source_table, split)
        return [], [raw], [], {
            "reason": "no_column_met_recovered_value_ratio",
            "entity_column_index": entity_col,
            "candidate_attribute_columns": candidate_attribute_names,
            "attribute_extractions": extraction_count,
            "qualified_columns": [],
        }

    query_tables: list[dict[str, Any]] = []
    data_lake_tables: list[dict[str, Any]] = []
    qrels: list[dict[str, Any]] = []
    if args.max_query_tables_per_source_table > 0:
        qualified_cols = qualified_cols[: args.max_query_tables_per_source_table]

    for qualified in qualified_cols:
        join_col = int(qualified["column_index"])
        eligible_source_rows = eligible_rows_by_col[join_col]
        excluded = {entity_col, join_col}
        other_cols = context_columns(source_table, excluded, 0)
        if not other_cols:
            continue
        query_context = other_cols[: max(1, args.max_query_context_attrs)]
        target_context_pool = [col for col in other_cols if col not in query_context]
        target_context = target_context_pool[: args.max_target_context_attrs]
        if not target_context:
            target_context = other_cols[:1]
        query_cols = [entity_col] + query_context
        target_cols = [join_col] + target_context
        if len(query_cols) < 2 or len(target_cols) < 2:
            continue
        query_rows, query_source_rows = project_selected_rows(
            source_table,
            query_cols,
            eligible_source_rows,
            min_required_cols=1,
        )
        target_rows, target_source_rows = project_selected_rows(
            source_table,
            target_cols,
            eligible_source_rows,
            min_required_cols=1,
        )
        if min(len(query_rows), len(target_rows)) < args.min_rows_per_output_table:
            continue
        chain_id = f"chain_{stable_hash(source_table['source_table_id'], entity_col, join_col)}"
        query_table_id = f"query_{stable_hash(chain_id, 'query')}"
        target_table_id = f"target_{stable_hash(chain_id, 'target')}"
        hidden_attribute = {
            "source_column_index": join_col,
            "column_name": qualified["column_name"],
            "role": "model_recoverable_join_column",
            "eligible_rows": qualified["eligible_rows"],
            "recovered_rows": qualified["recovered_rows"],
            "recovered_value_ratio": qualified["recovered_value_ratio"],
        }
        query_tables.append(
            table_record(
                table_id=query_table_id,
                role="query",
                split=split,
                source_table=source_table,
                column_indices=query_cols,
                rows=query_rows,
                source_row_indices=query_source_rows,
                extra={
                    "chain_id": chain_id,
                    "query_entity_col": entity_col,
                    "query_entity_col_name": get_column_name(source_table, entity_col),
                    "hidden_attributes": [hidden_attribute],
                    "query_context_col_names": [get_column_name(source_table, col) for col in query_context],
                },
            )
        )
        data_lake_tables.append(
            table_record(
                table_id=target_table_id,
                role="target_data_lake_table",
                split=split,
                source_table=source_table,
                column_indices=target_cols,
                rows=target_rows,
                source_row_indices=target_source_rows,
                extra={
                    "chain_id": chain_id,
                    "queryable_source_table": True,
                    "join_col": join_col,
                    "join_col_name": qualified["column_name"],
                    "target_context_col_names": [get_column_name(source_table, col) for col in target_context],
                },
            )
        )
        qrels.append(
            {
                "query_table_id": query_table_id,
                "target_table_id": target_table_id,
                "data_lake_table_id": target_table_id,
                "rel": 3,
                "split": split,
                "chain_id": chain_id,
                "source_table_id": source_table["source_table_id"],
                "join_attribute": hidden_attribute,
                "reason": "model_recoverable_join_column",
            }
        )
        source_to_query_row = {row["source_row_id"]: row["row_id"] for row in query_rows}
        source_to_target_rows: dict[int, list[int]] = defaultdict(list)
        for row in target_rows:
            source_to_target_rows[int(row["source_row_id"])].append(int(row["row_id"]))
        seen_recoveries: set[str] = set()
        for recovery in recoveries_by_col.get(join_col, []):
            source_row_id = int(recovery["source_row_id"])
            if source_row_id not in source_to_query_row:
                continue
            recovery_id = f"evrec_{stable_hash(query_table_id, target_table_id, source_row_id, recovery['evidence']['asset_id'], recovery['recovered_attribute']['value'])}"
            if recovery_id in seen_recoveries:
                continue
            seen_recoveries.add(recovery_id)
            path_id = f"path_{stable_hash(query_table_id, recovery['evidence']['asset_id'], target_table_id, source_row_id)}"
            write_jsonl_record(
                recovery_writer,
                {
                    "recovery_id": recovery_id,
                    "path_id": path_id,
                    "query_table_id": query_table_id,
                    "target_table_id": target_table_id,
                    "data_lake_table_id": target_table_id,
                    "query_row_id": source_to_query_row[source_row_id],
                    "target_row_ids": source_to_target_rows.get(source_row_id, []),
                    "path_nodes": [
                        {"node_id": query_table_id, "node_type": "query_table"},
                        {"node_id": recovery["evidence"]["asset_id"], "node_type": f"{recovery['evidence']['asset_type']}_asset"},
                        {"node_id": target_table_id, "node_type": "target_table"},
                    ],
                    **recovery,
                },
            )

    if not query_tables:
        raw = raw_data_lake_record(source_table, split)
        return [], [raw], [], {
            "reason": "qualified_columns_failed_query_target_split",
            "entity_column_index": entity_col,
            "qualified_columns": qualified_cols,
        }
    return query_tables, data_lake_tables, qrels, {
        "reason": "queryable",
        "entity_column_index": entity_col,
        "attribute_extractions": extraction_count,
        "qualified_columns": qualified_cols,
    }


def load_assets(paths: Iterable[Path]) -> dict[str, dict[str, Any]]:
    assets: dict[str, dict[str, Any]] = {}
    for asset in iter_jsonl_records(paths):
        if asset.get("asset_id"):
            assets[asset["asset_id"]] = asset
    return assets


def build_bridge_assets_for_entity(
    *,
    entity: dict[str, Any],
    max_images_per_entity: int,
    text_asset_chunk_chars: int,
    min_text_asset_chunk_chars: int,
    max_text_asset_chunks_per_entity: int,
    wikipedia_client: WikipediaClient,
) -> list[dict[str, Any]]:
    page = wikipedia_client.get_page(entity["wiki_title"])
    if not page or page.get("missing"):
        return []

    records: list[dict[str, Any]] = []
    text_chunks = split_text_asset_content(
        page.get("extract"),
        max_chars=text_asset_chunk_chars,
        min_chars=min_text_asset_chunk_chars,
        max_chunks=0,
    )
    selected_text_chunks = select_relevant_text_chunks(
        text_chunks,
        entity,
        max_text_asset_chunks_per_entity,
    )
    source_asset_id = f"asset_text_{stable_hash(entity['entity_id'], 'extract')}"
    for chunk_index, chunk, chunk_score in selected_text_chunks:
        asset_id = f"{source_asset_id}_{chunk_index:03d}"
        records.append(
            {
                "asset_id": asset_id,
                "source_asset_id": source_asset_id,
                "entity_id": entity["entity_id"],
                "entity_wiki_title": entity["wiki_title"],
                "asset_type": "text",
                "content": chunk,
                "text_chunk_index": chunk_index,
                "text_chunk_count": len(text_chunks),
                "selected_text_chunk_count": len(selected_text_chunks),
                "text_chunk_relevance_score": round(chunk_score, 6),
                "source": "wikipedia_extract_chunk",
                "url": page.get("canonicalurl")
                or f"https://en.wikipedia.org/wiki/{quote(entity['wiki_title'].replace(' ', '_'))}",
            }
        )

    image_titles: list[str] = []
    if page.get("pageimage"):
        image_titles.append(f"File:{page['pageimage']}")
    for image in page.get("images") or []:
        title = image.get("title") if isinstance(image, dict) else None
        if title:
            image_titles.append(title)

    seen_images: set[str] = set()
    kept = 0
    for image_title in image_titles:
        normalized_title = normalize_title(image_title)
        if normalized_title in seen_images or not is_useful_image(normalized_title):
            continue
        seen_images.add(normalized_title)
        imageinfo = wikipedia_client.get_imageinfo(normalized_title)
        if not imageinfo or not imageinfo.get("url"):
            continue
        if not is_useful_image(normalized_title, imageinfo):
            continue
        asset_id = f"asset_img_{stable_hash(entity['entity_id'], normalized_title)}"
        downloaded_image = wikipedia_client.download_image(imageinfo, asset_id)
        if downloaded_image is None:
            continue
        records.append(
            {
                "asset_id": asset_id,
                "entity_id": entity["entity_id"],
                "entity_wiki_title": entity["wiki_title"],
                "asset_type": "image",
                "image_url": imageinfo.get("url"),
                "description_url": imageinfo.get("descriptionurl"),
                "local_path": downloaded_image["local_path"],
                "relative_path": downloaded_image["relative_path"],
                "file_name": downloaded_image["file_name"],
                "bytes": downloaded_image["bytes"],
                "sha256": downloaded_image["sha256"],
                "metadata": {
                    "file_title": imageinfo.get("file_title"),
                    "mime": imageinfo.get("mime"),
                    "mediatype": imageinfo.get("mediatype"),
                    "width": imageinfo.get("width"),
                    "height": imageinfo.get("height"),
                    "size": imageinfo.get("size"),
                    "extmetadata": imageinfo.get("extmetadata") or {},
                    "downloaded": downloaded_image["downloaded"],
                },
                "source": "wikipedia_image_download",
            }
        )
        kept += 1
        if kept >= max_images_per_entity:
            break
    return records


def build_bridge_assets_parallel(
    *,
    entities: list[dict[str, Any]],
    max_entities: int | None,
    max_images_per_entity: int,
    text_asset_chunk_chars: int,
    min_text_asset_chunk_chars: int,
    max_text_asset_chunks_per_entity: int,
    wikipedia_client_factory: Callable[[], WikipediaClient] | None,
    asset_writer: ShardedJsonlWriter,
    flush_every_records: int,
    workers: int,
) -> tuple[dict[str, list[str]], int, int, int]:
    if wikipedia_client_factory is None:
        return defaultdict(list), 0, 0, 0
    selected_entities = entities[:max_entities] if max_entities else entities
    workers = max(1, int(workers or 1))
    if workers <= 1:
        return build_bridge_assets(
            entities,
            max_entities,
            max_images_per_entity,
            text_asset_chunk_chars,
            min_text_asset_chunk_chars,
            max_text_asset_chunks_per_entity,
            wikipedia_client_factory(),
            asset_writer,
            flush_every_records,
        )

    entity_to_assets: dict[str, list[str]] = defaultdict(list)
    text_asset_count = 0
    image_asset_count = 0
    written_assets = 0
    worker_state = threading.local()
    clients: list[WikipediaClient] = []
    clients_lock = threading.Lock()
    pending_results: dict[int, list[dict[str, Any]]] = {}
    next_write_index = 0

    def worker_client() -> WikipediaClient:
        client = getattr(worker_state, "client", None)
        if client is None:
            client = wikipedia_client_factory()
            worker_state.client = client
            with clients_lock:
                clients.append(client)
        return client

    def fetch_entity(entity: dict[str, Any]) -> list[dict[str, Any]]:
        return build_bridge_assets_for_entity(
            entity=entity,
            max_images_per_entity=max_images_per_entity,
            text_asset_chunk_chars=text_asset_chunk_chars,
            min_text_asset_chunk_chars=min_text_asset_chunk_chars,
            max_text_asset_chunks_per_entity=max_text_asset_chunks_per_entity,
            wikipedia_client=worker_client(),
        )

    def write_ready_results() -> None:
        nonlocal next_write_index, written_assets, text_asset_count, image_asset_count
        while next_write_index in pending_results:
            records = pending_results.pop(next_write_index)
            for record in records:
                write_jsonl_record(asset_writer, record)
                entity_to_assets[record["entity_id"]].append(record["asset_id"])
                if record.get("asset_type") == "image":
                    image_asset_count += 1
                else:
                    text_asset_count += 1
                written_assets += 1
                if flush_every_records > 0 and written_assets % flush_every_records == 0:
                    asset_writer.flush()
            next_write_index += 1

    futures = {}
    with ThreadPoolExecutor(max_workers=min(workers, max(1, len(selected_entities)))) as pool:
        for idx, entity in enumerate(selected_entities):
            futures[pool.submit(fetch_entity, entity)] = idx
        iterator: Iterable[Any] = as_completed(futures)
        if tqdm is not None:
            iterator = tqdm(iterator, total=len(futures), desc="Fetching Wikipedia assets", unit="entity", dynamic_ncols=True)
        for future in iterator:
            idx = futures[future]
            try:
                pending_results[idx] = future.result()
            except Exception as exc:  # pragma: no cover - integration only.
                logging.warning("Wikipedia asset fetch failed for %s: %s", selected_entities[idx].get("wiki_title"), exc)
                pending_results[idx] = []
            write_ready_results()
    write_ready_results()
    asset_writer.flush()

    return entity_to_assets, sum(int(getattr(client, "api_failures", 0)) for client in clients), text_asset_count, image_asset_count


def build_dataset(args: argparse.Namespace) -> dict[str, Any]:
    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_paths = resolve_shared_cache_paths(args)

    records_per_shard = max(1, args.records_per_shard)
    flush_every = max(1, args.flush_every_records)
    source_tables_dir = output_dir / "source_tables"
    query_tables_dir = output_dir / "query_tables"
    data_lake_tables_dir = output_dir / "data_lake_tables"
    entities_dir = output_dir / "entities"
    bridge_assets_dir = output_dir / "bridge_assets"
    table_asset_links_dir = output_dir / "table_asset_links"
    extraction_dir = output_dir / "attribute_extractions"
    recovery_dir = output_dir / "evidence_recoveries"

    entity_records: dict[str, dict[str, Any]] = {}
    wiki_to_entity_id: dict[str, str] = {}
    source_split_records: list[dict[str, str]] = []
    skip_reasons: Counter[str] = Counter()
    processed_tables = 0
    skipped_tables = 0
    source_table_count = 0

    json_files = sorted(input_dir.rglob("*.json"))
    logging.info("Found %d JSON files under %s", len(json_files), input_dir)
    source_writer = ShardedJsonlWriter(source_tables_dir, records_per_shard)
    stop = False
    with source_writer as source_handle:
        for json_file in iter_with_progress(json_files, "Reading EntiTables JSON"):
            if stop:
                break
            payload = read_entitables_json(json_file)
            if payload is None:
                skipped_tables += 1
                skip_reasons["malformed_json_file"] += 1
                continue
            for table_id, table_obj in payload.items():
                if args.max_source_tables is not None and source_table_count >= args.max_source_tables:
                    stop = True
                    break
                processed_tables += 1
                result = parse_source_table(
                    str(table_id),
                    table_obj,
                    json_file,
                    input_dir,
                    args.min_rows,
                    args.min_cols,
                    args.wiki_link_threshold,
                )
                if result.source_table is None:
                    skipped_tables += 1
                    skip_reasons[result.skip_reason or "unknown"] += 1
                    continue
                source_table = result.source_table
                write_jsonl_record(source_handle, source_table)
                source_table_count += 1
                source_split_records.append(
                    {
                        "source_table_id": source_table["source_table_id"],
                        "page_title": source_table.get("page_title") or "",
                    }
                )
                update_entities_from_table(entity_records, wiki_to_entity_id, source_table)
                if source_table_count % flush_every == 0:
                    source_handle.flush()
        source_handle.flush()

    entities = finalize_entities(entity_records)
    entities_writer = write_sharded_jsonl(entities_dir, entities, records_per_shard)

    wikipedia_client: WikipediaClient | None = None
    wikipedia_client_factory: Callable[[], WikipediaClient] | None = None
    if not args.no_wikipedia:
        def make_wikipedia_client() -> WikipediaClient:
            return WikipediaClient(
                cache_dir=cache_paths["wikipedia_cache_dir"],
                image_output_dir=cache_paths["wikipedia_image_dir"],
                output_dir=output_dir,
                sleep=args.sleep,
                user_agent=(
                    "MMJoinabilityDatasetBuilder/0.2 "
                    "(https://example.invalid; research dataset construction)"
                ),
            )

        wikipedia_client_factory = make_wikipedia_client
        wikipedia_client = make_wikipedia_client()
    bridge_assets_writer = ShardedJsonlWriter(bridge_assets_dir, records_per_shard)
    with bridge_assets_writer:
        if max(1, int(getattr(args, "wikipedia_workers", 1) or 1)) <= 1:
            entity_to_assets, api_failures, text_asset_count, image_asset_count = build_bridge_assets(
                entities,
                args.max_entities,
                args.max_images_per_entity,
                args.text_asset_chunk_chars,
                args.min_text_asset_chunk_chars,
                args.max_text_asset_chunks_per_entity,
                wikipedia_client,
                bridge_assets_writer,
                flush_every,
            )
        else:
            entity_to_assets, api_failures, text_asset_count, image_asset_count = build_bridge_assets_parallel(
                entities=entities,
                max_entities=args.max_entities,
                max_images_per_entity=args.max_images_per_entity,
                text_asset_chunk_chars=args.text_asset_chunk_chars,
                min_text_asset_chunk_chars=args.min_text_asset_chunk_chars,
                max_text_asset_chunks_per_entity=args.max_text_asset_chunks_per_entity,
                wikipedia_client_factory=wikipedia_client_factory,
                asset_writer=bridge_assets_writer,
                flush_every_records=flush_every,
                workers=args.wikipedia_workers,
            )

    table_asset_links_writer = ShardedJsonlWriter(table_asset_links_dir, records_per_shard)
    with table_asset_links_writer:
        table_asset_link_count = write_table_asset_links_from_jsonl(
            source_writer.paths(),
            [],
            table_asset_links_writer,
            wiki_to_entity_id,
            entity_to_assets,
            flush_every,
        )

    splits = source_splits(source_split_records, args)
    source_to_split = split_map(splits)
    assets = load_assets(bridge_assets_writer.paths())
    extractor = LocalAttributeExtractor(args)
    cache = ExtractionCache(cache_paths["model_attribute_extractions"], reuse=not args.no_reuse_model_cache)
    concurrency_state = ModelConcurrencyState.from_args(args)
    progress: ModelAnalysisProgress | None = None
    if getattr(args, "model_progress", True):
        planned_keys = estimate_model_analysis_keys(
            source_paths=source_writer.paths(),
            assets=assets,
            entity_to_assets=entity_to_assets,
            wiki_to_entity_id=wiki_to_entity_id,
            args=args,
        )
        cached_keys = {
            key
            for key in planned_keys
            if key in cache.items and cached_extraction_is_reusable(cache.items[key], args)
        }
        progress = ModelAnalysisProgress(total=len(planned_keys), cached_keys=cached_keys, enabled=True)

    query_writer = ShardedJsonlWriter(query_tables_dir, records_per_shard)
    data_lake_writer = ShardedJsonlWriter(data_lake_tables_dir, records_per_shard)
    extraction_writer = ShardedJsonlWriter(extraction_dir, records_per_shard)
    recovery_writer = ShardedJsonlWriter(recovery_dir, records_per_shard)
    qrels: list[dict[str, Any]] = []
    table_decisions: list[dict[str, Any]] = []
    query_table_count = 0
    data_lake_table_count = 0
    queryable_source_tables = 0
    rejected_source_tables = 0
    try:
        with query_writer as query_handle, data_lake_writer as data_lake_handle, extraction_writer as extraction_handle, recovery_writer as recovery_handle:
            for source_table in iter_jsonl_records(source_writer.paths()):
                split = source_to_split.get(source_table["source_table_id"], "test")
                query_tables, data_lake_tables, table_qrels, decision = build_table_join_records(
                    source_table=source_table,
                    split=split,
                    assets=assets,
                    entity_to_assets=entity_to_assets,
                    wiki_to_entity_id=wiki_to_entity_id,
                    extractor=extractor,
                    cache=cache,
                    progress=progress,
                    concurrency_state=concurrency_state,
                    extraction_writer=extraction_handle,
                    recovery_writer=recovery_handle,
                    args=args,
                )
                if query_tables:
                    queryable_source_tables += 1
                else:
                    rejected_source_tables += 1
                decision["source_table_id"] = source_table["source_table_id"]
                decision["split"] = split
                table_decisions.append(decision)
                for record in query_tables:
                    write_jsonl_record(query_handle, record)
                    query_table_count += 1
                    splits[split]["query_table_ids"].append(record["table_id"])
                for record in data_lake_tables:
                    write_jsonl_record(data_lake_handle, record)
                    data_lake_table_count += 1
                    splits[split]["data_lake_table_ids"].append(record["table_id"])
                qrels.extend(table_qrels)
                if (query_table_count + data_lake_table_count) % flush_every == 0:
                    query_handle.flush()
                    data_lake_handle.flush()
                    extraction_handle.flush()
                    recovery_handle.flush()
    finally:
        if progress is not None:
            progress.close()

    for split in ("train", "dev", "test"):
        splits[split]["query_table_ids"] = sorted(splits[split]["query_table_ids"])
        splits[split]["data_lake_table_ids"] = sorted(splits[split]["data_lake_table_ids"])

    qrels_count = write_jsonl(output_dir / "qrels.jsonl", qrels)
    write_jsonl(output_dir / "table_queryability_decisions.jsonl", table_decisions)
    write_json(output_dir / "splits.json", splits)

    stats = {
        "processed_tables": processed_tables,
        "skipped_tables": skipped_tables,
        "source_tables": source_table_count,
        "queryable_source_tables": queryable_source_tables,
        "rejected_source_tables": rejected_source_tables,
        "query_tables": query_table_count,
        "data_lake_tables": data_lake_table_count,
        "qrels": qrels_count,
        "unique_wiki_entities": len(entities),
        "text_assets": text_asset_count,
        "image_assets": image_asset_count,
        "table_asset_links": table_asset_link_count,
        "api_failures": api_failures,
        "attribute_extractions": extraction_writer.total_records,
        "evidence_recoveries": recovery_writer.total_records,
        "model_inference": model_call_stats_summary(extractor),
        "model_concurrency": concurrency_state.summary(),
        "wikipedia_workers": max(1, int(getattr(args, "wikipedia_workers", 1) or 1)),
        "min_recovered_value_ratio": args.min_recovered_value_ratio,
        "min_recovery_denominator": args.min_recovery_denominator,
        "skipped_reasons": dict(skip_reasons),
        "notes": [
            "source_tables are the fixed data-lake base pool",
            "query_tables are created only for source columns whose row values are model-recoverable from entity assets above the threshold",
            "data_lake_tables contain generated targets for queryable source tables and raw source tables for rejected source tables",
            "evidence_recoveries record query_table -> multimodal evidence -> target_table paths at entity/row/attribute granularity",
        ],
    }
    write_json(output_dir / "stats.json", stats)

    manifest = {
        "format": "sharded_jsonl",
        "records_per_shard": records_per_shard,
        "artifacts": {
            "source_tables": source_writer.manifest(output_dir),
            "query_tables": query_writer.manifest(output_dir),
            "data_lake_tables": data_lake_writer.manifest(output_dir),
            "entities": entities_writer.manifest(output_dir),
            "bridge_assets": bridge_assets_writer.manifest(output_dir),
            "table_asset_links": table_asset_links_writer.manifest(output_dir),
            "attribute_extractions": extraction_writer.manifest(output_dir),
            "evidence_recoveries": recovery_writer.manifest(output_dir),
        },
        "single_files": {
            "qrels": "qrels.jsonl",
            "splits": "splits.json",
            "stats": "stats.json",
            "table_queryability_decisions": "table_queryability_decisions.jsonl",
        },
        "model_endpoints": {
            "text_model_base_url": args.text_model_base_url,
            "text_model_name": args.text_model_name,
            "image_model_base_url": args.image_model_base_url,
            "image_model_name": args.image_model_name,
            "prompt_version": PROMPT_VERSION,
            "disable_thinking": args.disable_thinking,
            "reparse_cached_model_outputs": args.reparse_cached_model_outputs,
            "refresh_invalid_model_cache": args.refresh_invalid_model_cache,
            "cache_failed_model_outputs": args.cache_failed_model_outputs,
            "configured_text_model_workers": args.text_model_workers,
            "configured_image_model_workers": args.image_model_workers,
            "final_model_concurrency": stats["model_concurrency"],
            "inference_stats": stats["model_inference"],
        },
        "cache": {
            "root_dir": str(cache_paths["root_dir"]),
            "wikipedia_cache_dir": str(cache_paths["wikipedia_cache_dir"]),
            "wikipedia_image_dir": str(cache_paths["wikipedia_image_dir"]),
            "model_attribute_extractions": str(cache_paths["model_attribute_extractions"]),
        },
        "wikipedia_cache": {
            "cache_dir": str(cache_paths["wikipedia_cache_dir"]),
            "image_dir": str(cache_paths["wikipedia_image_dir"]),
            "workers": max(1, int(getattr(args, "wikipedia_workers", 1) or 1)),
        },
        "note": "Read shards listed in this manifest; stale files from older runs may exist if an output directory is reused.",
    }
    write_json(output_dir / "dataset_manifest.json", manifest)
    return stats


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a multimodal joinability discovery dataset by asking local "
            "text/image models which entity attributes can be extracted from Wikipedia assets."
        )
    )
    parser.add_argument("--input_dir", required=True, help="Directory containing EntiTables .json files.")
    parser.add_argument("--output_dir", required=True, help="Directory where dataset artifacts are written.")
    parser.add_argument("--max_source_tables", type=int, default=20000, help="Number of filtered EntiTables source tables to use as the data-lake base.")
    parser.add_argument("--max_tables", type=int, default=None, dest="max_source_tables", help="(Deprecated) Use --max_source_tables instead.")
    parser.add_argument("--max_entities", type=int, default=None, help="Maximum number of entities for Wikipedia asset fetching.")
    parser.add_argument("--max_images_per_entity", type=int, default=3)
    parser.add_argument("--text_asset_chunk_chars", type=int, default=800)
    parser.add_argument("--min_text_asset_chunk_chars", type=int, default=120)
    parser.add_argument("--max_text_asset_chunks_per_entity", type=int, default=3)
    parser.add_argument("--wiki_link_threshold", type=float, default=0.3)
    parser.add_argument("--min_rows", type=int, default=2)
    parser.add_argument("--min_cols", type=int, default=2)
    parser.add_argument("--min_rows_per_output_table", type=int, default=2)
    parser.add_argument("--sleep", type=float, default=0.2)
    parser.add_argument("--wikipedia_workers", type=int, default=1, help="Parallel Wikipedia entity/material fetch workers. Default 1 preserves the old serial fetch path.")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--flush_every_records", type=int, default=500)
    parser.add_argument("--records_per_shard", type=int, default=50000)
    parser.add_argument("--no_wikipedia", action="store_true", help="Skip MediaWiki API calls. Queryable tables will normally be zero.")
    parser.add_argument("--cache_dir", default=str(DEFAULT_SHARED_CACHE_DIR), help="Shared cache root for Wikipedia metadata, downloaded images, and model attribute extraction cache.")
    parser.add_argument("--wikipedia_cache_dir", default=None, help="Override MediaWiki page and imageinfo JSONL cache directory. Defaults to <cache_dir>/wikipedia.")
    parser.add_argument("--wikipedia_image_dir", default=None, help="Override downloaded Wikipedia image directory. Defaults to <cache_dir>/images.")
    parser.add_argument("--split_by", choices=["source_table_id", "page_title"], default="page_title")
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--dev_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--min_column_non_empty_ratio", type=float, default=0.5)
    parser.add_argument("--min_recovered_value_ratio", type=float, default=0.6, help="Minimum fraction of eligible row values recovered from entity assets for a column to become a join column.")
    parser.add_argument("--min_recovery_denominator", type=int, default=2, help="Minimum eligible row count for a candidate join column.")
    parser.add_argument("--max_query_tables_per_source_table", type=int, default=0, help="0 means emit all qualifying join columns.")
    parser.add_argument("--max_query_context_attrs", type=int, default=1)
    parser.add_argument("--max_target_context_attrs", type=int, default=2)
    parser.add_argument("--text_model_base_url", default="http://localhost:8001/v1")
    parser.add_argument("--text_model_name", default="Qwen3.5-9B")
    parser.add_argument("--text_model_api_key", default=None)
    parser.add_argument("--image_model_base_url", default="http://localhost:8000/v1")
    parser.add_argument("--image_model_name", default="Qwen3-VL-8B-Thinking")
    parser.add_argument("--image_model_api_key", default=None)
    parser.add_argument("--model_timeout_seconds", type=float, default=120.0)
    parser.add_argument("--model_temperature", type=float, default=0.0)
    parser.add_argument("--model_max_tokens", type=int, default=1024)
    parser.add_argument("--text_model_workers", type=int, default=1, help="Concurrent text-model requests. Default 1 is conservative for 24GB GPUs.")
    parser.add_argument("--image_model_workers", type=int, default=1, help="Concurrent image-model requests. Default 1 is conservative for 24GB GPUs.")
    parser.add_argument("--enable_thinking", dest="disable_thinking", action="store_false", help="Allow Qwen thinking mode. By default, chat_template_kwargs disables thinking for extraction calls.")
    parser.add_argument("--no_reparse_cached_model_outputs", dest="reparse_cached_model_outputs", action="store_false", help="Use cached parsed attributes as-is instead of reparsing cached raw_response with the current JSON parser.")
    parser.add_argument("--refresh_invalid_model_cache", action="store_true", help="When a cached raw_response still reparses to no valid candidate attributes, call the model again with the current request settings.")
    parser.add_argument("--model_max_retries", type=int, default=2)
    parser.add_argument("--model_retry_sleep_seconds", type=float, default=2.0)
    parser.add_argument("--no_reuse_model_cache", action="store_true")
    parser.add_argument("--cache_failed_model_outputs", action="store_true", help="Persist failed model calls in the extraction cache. By default failures are written only to this run's output so the next run retries them.")
    parser.add_argument("--no_model_progress", dest="model_progress", action="store_false", help="Disable the local model analysis progress bar.")
    parser.set_defaults(disable_thinking=True, reparse_cached_model_outputs=True)
    parser.set_defaults(model_progress=True)
    return parser.parse_args(argv)


def main() -> None:
    setup_logging()
    stats = build_dataset(parse_args())
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
