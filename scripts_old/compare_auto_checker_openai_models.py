#!/usr/bin/env python
"""Replay recent Terra auto-check inputs with Luna and compare decisions."""

from __future__ import annotations

import argparse
import json
import logging
import mmap
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from build_mm_joinability_dataset import (
    DEFAULT_IMAGE_REQUEST_MAX_PIXELS,
    ExtractionTask,
    LocalAttributeExtractor,
    clean_text,
    is_useful_image,
    normalize,
    normalize_title,
    split_text_asset_content,
    stable_hash,
    values_match,
)
from build_mm_table_dataset import WikipediaClient, default_wikipedia_user_agent
from mm_joinability_dataset_auto_checker import (
    AutoReviewCache,
    _run_extraction_stage,
    _single_extraction,
    attribute_cache_key,
    prepare_reviewer,
)
from openai_attribute_extractor import (
    load_openai_environment_file,
    summarize_usage_journal,
)
from stage1_io import write_json, write_jsonl


DEFAULT_MODEL_CACHE = Path("cache/mm_joinability/model_attribute_extractions.jsonl")
DEFAULT_WIKIPEDIA_CACHE = Path("cache/mm_joinability/wikipedia")
DEFAULT_IMAGE_CACHE = Path("cache/mm_joinability/images")
DEFAULT_ENV_FILE = Path(".env.openai")


def iter_jsonl_reverse(path: Path) -> Iterable[dict[str, Any]]:
    """Yield JSONL objects from newest to oldest without loading the file."""
    with path.open("rb") as handle:
        if handle.seek(0, os.SEEK_END) == 0:
            return
        with mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as data:
            end = len(data)
            while end > 0:
                while end > 0 and data[end - 1 : end] in {b"\n", b"\r"}:
                    end -= 1
                if end <= 0:
                    break
                newline = data.rfind(b"\n", 0, end)
                start = newline + 1
                raw = data[start:end].strip()
                end = max(0, newline)
                if not raw:
                    continue
                try:
                    record = json.loads(raw)
                except (UnicodeDecodeError, json.JSONDecodeError):
                    continue
                if isinstance(record, dict):
                    yield record


def usage_record_count(path: Path) -> int:
    count = 0
    if not path.is_file():
        return count
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict) and record.get("record_type") == "openai_api_usage":
                count += 1
    return count


def collect_recent_terra_reviews(
    cache_path: Path,
    expected_reviews: int,
) -> list[dict[str, Any]]:
    """Collect the latest unique completed Terra-secondary review inputs."""
    if expected_reviews <= 0:
        raise ValueError("expected Terra review count must be positive")
    selected: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for record in iter_jsonl_reverse(cache_path):
        cache_key = clean_text(record.get("cache_key"))
        auto_check = record.get("auto_check")
        if not cache_key or not isinstance(auto_check, dict):
            continue
        reviews = auto_check.get("reviews")
        if not isinstance(reviews, list):
            continue
        for review in reversed(reviews):
            if not isinstance(review, dict):
                continue
            if clean_text(review.get("decision_source")) not in {
                "secondary_openai",
                "terra_adjudication",
            }:
                continue
            if not bool(review.get("review_complete")) or clean_text(
                review.get("error_code")
            ):
                continue
            identity = (
                cache_key,
                normalize(review.get("attribute_name")),
                clean_text(review.get("claimed_value")),
            )
            if identity in seen:
                continue
            seen.add(identity)
            selected.append({"record": record, "review": review})
            if len(selected) >= expected_reviews:
                return list(reversed(selected))
    return list(reversed(selected))


class CachedWikipediaAssetResolver:
    """Rebuild only assets referenced by cached extraction records."""

    def __init__(
        self,
        *,
        wikipedia_cache_dir: Path,
        image_cache_dir: Path,
        output_dir: Path,
        text_chunk_chars: int,
        min_text_chunk_chars: int,
    ) -> None:
        self.client = WikipediaClient(
            cache_dir=wikipedia_cache_dir,
            image_output_dir=image_cache_dir,
            output_dir=output_dir,
            sleep=0.0,
            user_agent=default_wikipedia_user_agent(),
        )
        self.text_chunk_chars = text_chunk_chars
        self.min_text_chunk_chars = min_text_chunk_chars
        self._assets: dict[str, dict[str, Any]] = {}

    def resolve(self, record: dict[str, Any]) -> dict[str, Any]:
        asset_id = clean_text(record.get("asset_id"))
        if asset_id in self._assets:
            return self._assets[asset_id]
        asset_type = clean_text(record.get("asset_type"))
        page_title = normalize_title(record.get("entity_wiki_title"))
        page = self.client.page_cache.get(page_title)
        if not isinstance(page, dict) or page.get("missing"):
            raise ValueError("cached Wikipedia page is unavailable")
        if asset_type == "text":
            asset = self._text_asset(record, page)
        elif asset_type == "image":
            asset = self._image_asset(record, page)
        else:
            raise ValueError("unsupported cached asset type")
        self._assets[asset_id] = asset
        return asset

    def _text_asset(
        self,
        record: dict[str, Any],
        page: dict[str, Any],
    ) -> dict[str, Any]:
        asset_id = clean_text(record.get("asset_id"))
        try:
            chunk_index = int(asset_id.rsplit("_", 1)[1])
        except (IndexError, ValueError) as error:
            raise ValueError("text asset ID has no chunk index") from error
        chunks = split_text_asset_content(
            page.get("extract"),
            max_chars=self.text_chunk_chars,
            min_chars=self.min_text_chunk_chars,
            max_chunks=0,
        )
        if not 0 <= chunk_index < len(chunks):
            raise ValueError("cached text chunk index is unavailable")
        expected_id = "asset_text_" + stable_hash(
            clean_text(record.get("entity_id")), "extract"
        )
        expected_id = f"{expected_id}_{chunk_index:03d}"
        if expected_id != asset_id:
            raise ValueError("cached text asset identity does not match")
        return {
            "asset_id": asset_id,
            "entity_id": clean_text(record.get("entity_id")),
            "entity_wiki_title": clean_text(record.get("entity_wiki_title")),
            "asset_type": "text",
            "content": chunks[chunk_index],
            "source": "wikipedia_extract_chunk",
        }

    def _image_asset(
        self,
        record: dict[str, Any],
        page: dict[str, Any],
    ) -> dict[str, Any]:
        asset_id = clean_text(record.get("asset_id"))
        entity_id = clean_text(record.get("entity_id"))
        image_titles: list[str] = []
        if page.get("pageimage"):
            image_titles.append(f"File:{page['pageimage']}")
        for image in page.get("images") or []:
            title = image.get("title") if isinstance(image, dict) else None
            if title:
                image_titles.append(str(title))
        seen: set[str] = set()
        for image_title in image_titles:
            normalized_title = normalize_title(image_title)
            if normalized_title in seen:
                continue
            seen.add(normalized_title)
            if "asset_img_" + stable_hash(entity_id, normalized_title) != asset_id:
                continue
            imageinfo = self.client.image_cache.get(normalized_title)
            if not isinstance(imageinfo, dict) or not imageinfo.get("url"):
                raise ValueError("cached image metadata is unavailable")
            if not is_useful_image(normalized_title, imageinfo):
                raise ValueError("cached image no longer passes the media policy")
            downloaded = self.client.download_image(imageinfo, asset_id)
            if downloaded is None:
                raise ValueError("cached image file is unavailable")
            return {
                "asset_id": asset_id,
                "entity_id": entity_id,
                "entity_wiki_title": clean_text(record.get("entity_wiki_title")),
                "asset_type": "image",
                "local_path": downloaded["local_path"],
                "sha256": downloaded["sha256"],
                "source": "wikipedia_image_download",
            }
        raise ValueError("cached image asset identity is unavailable")


def replay_task(record: dict[str, Any], asset: dict[str, Any]) -> ExtractionTask:
    row_attributes = list(record.get("row_attributes") or [])
    entity_index = next(
        (
            index
            for index, item in enumerate(row_attributes)
            if isinstance(item, dict) and bool(item.get("is_entity"))
        ),
        0,
    )
    entity_name = ""
    if row_attributes and isinstance(row_attributes[entity_index], dict):
        entity_name = clean_text(row_attributes[entity_index].get("name"))
    cache_key = clean_text(record.get("cache_key"))
    return ExtractionTask(
        order=0,
        cache_key=cache_key,
        source_table_id=f"semantic_replay:{cache_key}",
        source_row_id=0,
        entity_column_index=entity_index,
        entity_column_name=entity_name,
        entity={
            "entity_id": clean_text(record.get("entity_id")),
            "wiki_title": clean_text(record.get("entity_wiki_title")),
            "cell_text": clean_text(record.get("entity_text")),
            "entity_column_index": entity_index,
            "entity_column_name": entity_name,
            "row_attributes": row_attributes,
        },
        asset=asset,
        candidate_attribute_names=list(record.get("candidate_attribute_names") or []),
    )


def build_replay_batches(
    selected: list[dict[str, Any]],
    resolver: CachedWikipediaAssetResolver,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    batches: list[dict[str, Any]] = []
    metadata: list[dict[str, Any]] = []
    batch_builder = object.__new__(LocalAttributeExtractor)
    for index, selected_item in enumerate(selected):
        record = selected_item["record"]
        terra_review = selected_item["review"]
        asset = resolver.resolve(record)
        task = replay_task(record, asset)
        batch = batch_builder._auto_check_review_batch(
            task=task,
            attribute_name=clean_text(terra_review.get("attribute_name")),
            claimed_value=clean_text(terra_review.get("claimed_value")),
        )
        batch["query_table_id"] = f"terra_luna_replay:{index:06d}"
        batch["items"][0]["query_row_id"] = f"replay:{index:06d}"
        batch["query_row_ids"] = [f"replay:{index:06d}"]
        batches.append(batch)
        metadata.append(
            {
                "cache_key": clean_text(record.get("cache_key")),
                "asset_id": clean_text(record.get("asset_id")),
                "asset_type": clean_text(record.get("asset_type")),
                "entity_column_name": task.entity_column_name,
                "attribute_name": clean_text(terra_review.get("attribute_name")),
                "claimed_value": clean_text(terra_review.get("claimed_value")),
                "terra_extracted_value": clean_text(
                    terra_review.get("terra_extracted_value")
                    or terra_review.get("secondary_extracted_value")
                ),
                "terra_verdict": clean_text(
                    terra_review.get("terra_verdict")
                    or terra_review.get("secondary_verdict")
                )
                or clean_text(terra_review.get("verdict")),
            }
        )
    return batches, metadata


def comparison_row(
    metadata: dict[str, Any],
    luna_extracted_value: str,
) -> dict[str, Any]:
    attribute_name = metadata["attribute_name"]
    entity_column_name = clean_text(metadata.get("entity_column_name"))
    claimed_value = metadata["claimed_value"]

    def derived_verdict(value: str) -> str:
        if not value:
            return "insufficient"
        if values_match(
            value,
            claimed_value,
            attribute_name=attribute_name,
            entity_column_name=entity_column_name,
        ):
            return "supported"
        return "contradicted"

    luna_value = clean_text(luna_extracted_value)
    luna_verdict = derived_verdict(luna_value)
    terra_value = metadata["terra_extracted_value"]
    terra_recorded_verdict = clean_text(metadata.get("terra_verdict"))
    terra_verdict = derived_verdict(terra_value)
    return {
        **metadata,
        "terra_recorded_verdict": terra_recorded_verdict,
        "terra_verdict": terra_verdict,
        "luna_extracted_value": luna_value,
        "luna_verdict": luna_verdict,
        "value_agreement": (
            not terra_value and not luna_value
        )
        or (
            bool(terra_value)
            and bool(luna_value)
            and (
                values_match(
                    terra_value,
                    luna_value,
                    attribute_name=attribute_name,
                    entity_column_name=entity_column_name,
                )
                or values_match(
                    luna_value,
                    terra_value,
                    attribute_name=attribute_name,
                    entity_column_name=entity_column_name,
                )
            )
        ),
        "verdict_agreement": terra_verdict == luna_verdict,
    }


def summarize_comparisons(
    rows: list[dict[str, Any]],
    *,
    planned: int,
    errors: int,
) -> dict[str, Any]:
    def summarize_group(group: list[dict[str, Any]]) -> dict[str, Any]:
        terra = Counter(row["terra_verdict"] for row in group)
        luna = Counter(row["luna_verdict"] for row in group)
        verdict_agreement = sum(bool(row["verdict_agreement"]) for row in group)
        value_agreement = sum(bool(row["value_agreement"]) for row in group)
        return {
            "compared": len(group),
            "verdict_agreement": verdict_agreement,
            "verdict_agreement_rate": (
                verdict_agreement / len(group) if group else 0.0
            ),
            "value_agreement": value_agreement,
            "value_agreement_rate": value_agreement / len(group) if group else 0.0,
            "terra_verdicts": dict(sorted(terra.items())),
            "luna_verdicts": dict(sorted(luna.items())),
            "terra_supported_luna_not": sum(
                row["terra_verdict"] == "supported"
                and row["luna_verdict"] != "supported"
                for row in group
            ),
            "luna_supported_terra_not": sum(
                row["luna_verdict"] == "supported"
                and row["terra_verdict"] != "supported"
                for row in group
            ),
        }

    by_modality: dict[str, Any] = {}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[clean_text(row.get("asset_type")) or "unknown"].append(row)
    for modality, group in sorted(grouped.items()):
        by_modality[modality] = summarize_group(group)
    return {
        "schema_version": "mmdd-terra-luna-auto-check-comparison-v2",
        "replay_mode": "semantic_equivalent_identifiers_replaced",
        "planned": planned,
        "compared": len(rows),
        "errors": errors,
        "complete": len(rows) == planned and errors == 0,
        "overall": summarize_group(rows),
        "by_modality": by_modality,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay recent completed Terra secondary checks with Luna and "
            "compare extracted values and derived verdicts."
        )
    )
    parser.add_argument("--model_cache_path", default=str(DEFAULT_MODEL_CACHE))
    parser.add_argument("--terra_usage_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--expected_reviews", type=int, default=None)
    parser.add_argument("--wikipedia_cache_dir", default=str(DEFAULT_WIKIPEDIA_CACHE))
    parser.add_argument("--image_cache_dir", default=str(DEFAULT_IMAGE_CACHE))
    parser.add_argument("--text_chunk_chars", type=int, default=800)
    parser.add_argument("--min_text_chunk_chars", type=int, default=120)
    parser.add_argument("--openai_env_file", default=str(DEFAULT_ENV_FILE))
    parser.add_argument("--openai_base_url", default="")
    parser.add_argument("--openai_api_key_env", default="OPENAI_API_KEY")
    parser.add_argument("--luna_model", default="gpt-5.6-luna")
    parser.add_argument(
        "--reasoning_effort",
        choices=("omit", "none", "minimal", "low", "medium", "high", "xhigh"),
        default="medium",
    )
    parser.add_argument("--verbosity", choices=("low", "medium", "high"), default="low")
    parser.add_argument("--max_output_tokens", type=int, default=2048)
    parser.add_argument("--max_inflight", type=int, default=5)
    parser.add_argument("--timeout_seconds", type=float, default=180.0)
    parser.add_argument("--max_retries", type=int, default=2)
    parser.add_argument("--retry_sleep_seconds", type=float, default=2.0)
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    terra_usage_path = Path(args.terra_usage_path).resolve()
    expected_reviews = (
        int(args.expected_reviews)
        if args.expected_reviews is not None
        else usage_record_count(terra_usage_path)
    )
    selected = collect_recent_terra_reviews(
        Path(args.model_cache_path).resolve(), expected_reviews
    )
    if len(selected) != expected_reviews:
        raise ValueError(
            "could not recover every Terra review from the extraction cache"
        )
    resolver = CachedWikipediaAssetResolver(
        wikipedia_cache_dir=Path(args.wikipedia_cache_dir).resolve(),
        image_cache_dir=Path(args.image_cache_dir).resolve(),
        output_dir=output_dir,
        text_chunk_chars=args.text_chunk_chars,
        min_text_chunk_chars=args.min_text_chunk_chars,
    )
    batches, metadata = build_replay_batches(selected, resolver)
    plan = {
        "schema_version": "mmdd-terra-luna-auto-check-replay-plan-v1",
        "replay_mode": "semantic_equivalent_identifiers_replaced",
        "terra_reviews": expected_reviews,
        "reconstructed_batches": len(batches),
        "modalities": dict(sorted(Counter(row["asset_type"] for row in metadata).items())),
        "luna_model": args.luna_model,
        "reasoning_effort": args.reasoning_effort,
    }
    write_json(output_dir / "replay_plan.json", plan)
    if args.dry_run:
        return {"plan": plan, "dry_run": True}

    env_path = Path(args.openai_env_file)
    if env_path.exists():
        load_openai_environment_file(env_path)
    reviewer_args = argparse.Namespace(
        provider="openai",
        openai_model=args.luna_model,
        openai_base_url=(
            clean_text(args.openai_base_url)
            or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        ),
        openai_api_key_env=args.openai_api_key_env,
        openai_reasoning_effort=args.reasoning_effort,
        openai_verbosity=args.verbosity,
        openai_max_output_tokens=args.max_output_tokens,
        openai_image_detail="auto",
        openai_image_max_pixels=DEFAULT_IMAGE_REQUEST_MAX_PIXELS,
        openai_max_inflight=args.max_inflight,
        openai_requests_per_minute=0,
        openai_tokens_per_minute=0,
        model_timeout_seconds=args.timeout_seconds,
        model_max_retries=args.max_retries,
        model_retry_sleep_seconds=args.retry_sleep_seconds,
        openai_retry_max_seconds=60.0,
    )
    usage_path = output_dir / "luna_usage.jsonl"
    reviewer = prepare_reviewer(reviewer_args, usage_journal_path=usage_path)
    cache = AutoReviewCache(output_dir / "luna_replay_cache.sqlite3")
    cache.initialize()
    stage = _run_extraction_stage(
        requests_to_extract=batches,
        extractor=reviewer,
        cache=cache,
        workers=args.max_inflight,
        stage="luna_replay",
        progress_every=25,
    )
    rows: list[dict[str, Any]] = []
    for batch, row_metadata in zip(batches, metadata):
        key = attribute_cache_key(batch, reviewer.identity)
        extraction = _single_extraction(stage.results.get(key))
        if extraction is None:
            continue
        rows.append(
            comparison_row(
                row_metadata,
                clean_text(extraction.get("extracted_value")),
            )
        )
    summary = summarize_comparisons(
        rows,
        planned=len(batches),
        errors=len(stage.errors),
    )
    summary["luna_usage"] = summarize_usage_journal(usage_path)
    write_jsonl(output_dir / "comparison_rows.jsonl", rows)
    write_jsonl(output_dir / "errors.jsonl", stage.errors)
    write_json(output_dir / "summary.json", summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    try:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(message)s",
        )
        result = run(parse_args(argv))
        print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
        return 0 if result.get("dry_run") or result.get("complete") else 1
    except Exception as error:
        print(
            f"ERROR: Terra/Luna comparison stopped ({type(error).__name__})",
            file=sys.stderr,
            flush=True,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
