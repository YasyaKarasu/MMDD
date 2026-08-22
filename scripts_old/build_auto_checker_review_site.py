#!/usr/bin/env python
"""Build a self-contained review site for Terra/Luna auto-check candidates."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from build_mm_joinability_dataset import clean_text, stable_hash, values_match
from compare_auto_checker_openai_models import (
    DEFAULT_IMAGE_CACHE,
    DEFAULT_MODEL_CACHE,
    DEFAULT_WIKIPEDIA_CACHE,
    CachedWikipediaAssetResolver,
    iter_jsonl_reverse,
)
from stage1_io import write_json


DEFAULT_TEMPLATE = Path(__file__).with_name("templates") / "auto_checker_review.html"
HUMAN_VERDICTS = {"supported", "contradicted", "insufficient", "schema_bad"}


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                record = json.loads(raw)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSONL at line {line_number}") from error
            if isinstance(record, dict):
                yield record


def select_review_rows(
    rows: Iterable[dict[str, Any]],
    *,
    include_mode: str,
) -> list[dict[str, Any]]:
    """Select candidate rows while preserving one stable review grain."""
    selected: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for row in rows:
        terra_supported = clean_text(row.get("terra_verdict")) == "supported"
        luna_supported = clean_text(row.get("luna_verdict")) == "supported"
        if include_mode == "union_supported" and not (
            terra_supported or luna_supported
        ):
            continue
        if include_mode == "support_disagreements" and (
            terra_supported == luna_supported
        ):
            continue
        if include_mode not in {"all", "union_supported", "support_disagreements"}:
            raise ValueError("unsupported include mode")
        identity = (
            clean_text(row.get("cache_key")),
            clean_text(row.get("attribute_name")),
            clean_text(row.get("claimed_value")),
        )
        if not all(identity) or identity in seen:
            continue
        seen.add(identity)
        selected.append(dict(row))

    # Avoid revealing the disagreement group through the review order.
    selected.sort(
        key=lambda row: stable_hash(
            clean_text(row.get("cache_key")),
            clean_text(row.get("attribute_name")),
            clean_text(row.get("claimed_value")),
            "blind-review-order-v1",
        )
    )
    return selected


def collect_extraction_records(
    cache_path: Path,
    rows: Iterable[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    targets = {
        clean_text(row.get("cache_key")): clean_text(row.get("asset_id"))
        for row in rows
    }
    if not targets or "" in targets:
        raise ValueError("review rows are missing cache identities")
    found: dict[str, dict[str, Any]] = {}
    for record in iter_jsonl_reverse(cache_path):
        cache_key = clean_text(record.get("cache_key"))
        if cache_key not in targets or cache_key in found:
            continue
        if clean_text(record.get("asset_id")) != targets[cache_key]:
            continue
        found[cache_key] = record
        if len(found) == len(targets):
            break
    missing = sorted(set(targets) - set(found))
    if missing:
        raise ValueError(f"missing {len(missing)} extraction cache records")
    return found


def _clean_row_attributes(value: Any) -> list[dict[str, Any]]:
    cleaned: list[dict[str, Any]] = []
    for item in value if isinstance(value, list) else []:
        if not isinstance(item, dict):
            continue
        cleaned.append(
            {
                "name": clean_text(item.get("name")),
                "value": clean_text(item.get("value")),
                "is_entity": bool(item.get("is_entity")),
            }
        )
    return cleaned


def reclassify_review_rows(
    rows: Iterable[dict[str, Any]],
    records: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Derive verdicts with the current contextual value-equivalence policy."""
    output: list[dict[str, Any]] = []
    for source_row in rows:
        row = dict(source_row)
        cache_key = clean_text(row.get("cache_key"))
        record = records.get(cache_key)
        if not isinstance(record, dict):
            raise ValueError("review row is missing extraction context")
        entity_column_name = next(
            (
                clean_text(item.get("name"))
                for item in record.get("row_attributes") or []
                if isinstance(item, dict) and bool(item.get("is_entity"))
            ),
            "",
        )
        attribute_name = clean_text(row.get("attribute_name"))
        claimed_value = clean_text(row.get("claimed_value"))

        def verdict(value: Any) -> str:
            extracted_value = clean_text(value)
            if not extracted_value:
                return "insufficient"
            if values_match(
                extracted_value,
                claimed_value,
                attribute_name=attribute_name,
                entity_column_name=entity_column_name,
            ):
                return "supported"
            return "contradicted"

        row["entity_column_name"] = entity_column_name
        row["terra_recorded_verdict"] = clean_text(row.get("terra_verdict"))
        row["luna_recorded_verdict"] = clean_text(row.get("luna_verdict"))
        row["terra_verdict"] = verdict(row.get("terra_extracted_value"))
        row["luna_verdict"] = verdict(row.get("luna_extracted_value"))
        output.append(row)
    return output


def _copy_image(source_path: Path, assets_dir: Path, asset_id: str) -> str:
    if not source_path.is_file():
        raise ValueError("review image file is unavailable")
    suffix = source_path.suffix.lower()
    if suffix not in {".jpg", ".jpeg", ".png", ".webp", ".gif"}:
        suffix = ".img"
    destination = assets_dir / f"{asset_id}{suffix}"
    if not destination.is_file() or destination.stat().st_size != source_path.stat().st_size:
        shutil.copy2(source_path, destination)
    return f"assets/{destination.name}"


def build_review_items(
    rows: list[dict[str, Any]],
    records: dict[str, dict[str, Any]],
    *,
    resolver: CachedWikipediaAssetResolver,
    assets_dir: Path,
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for row in rows:
        cache_key = clean_text(row.get("cache_key"))
        record = records[cache_key]
        asset = resolver.resolve(record)
        asset_type = clean_text(row.get("asset_type"))
        if asset_type == "text":
            evidence = {
                "kind": "text",
                "text": clean_text(asset.get("content")),
            }
            if not evidence["text"]:
                raise ValueError("empty text evidence in review item")
        elif asset_type == "image":
            local_path = Path(clean_text(asset.get("local_path"))).resolve()
            evidence = {
                "kind": "image",
                "url": _copy_image(
                    local_path,
                    assets_dir,
                    clean_text(row.get("asset_id")),
                ),
            }
        else:
            raise ValueError("unsupported review asset type")

        terra_verdict = clean_text(row.get("terra_verdict"))
        luna_verdict = clean_text(row.get("luna_verdict"))
        review_id = "review_" + stable_hash(
            cache_key,
            clean_text(row.get("attribute_name")),
            clean_text(row.get("claimed_value")),
        )
        items.append(
            {
                "review_id": review_id,
                "cache_key": cache_key,
                "asset_id": clean_text(row.get("asset_id")),
                "asset_type": asset_type,
                "entity_text": clean_text(record.get("entity_text")),
                "entity_wiki_title": clean_text(record.get("entity_wiki_title")),
                "attribute_name": clean_text(row.get("attribute_name")),
                "claimed_value": clean_text(row.get("claimed_value")),
                "row_attributes": _clean_row_attributes(record.get("row_attributes")),
                "evidence": evidence,
                "comparison_group": (
                    "both_supported"
                    if terra_verdict == luna_verdict == "supported"
                    else "support_disagreement"
                ),
                "terra": {
                    "extracted_value": clean_text(row.get("terra_extracted_value")),
                    "verdict": terra_verdict,
                },
                "luna": {
                    "extracted_value": clean_text(row.get("luna_extracted_value")),
                    "verdict": luna_verdict,
                },
            }
        )
    return items


def _script_safe_json(value: Any) -> str:
    return (
        json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )


def render_review_html(template: str, manifest: dict[str, Any]) -> str:
    required = {"__REVIEW_TITLE__", "__REVIEW_DATA__"}
    missing = sorted(token for token in required if token not in template)
    if missing:
        raise ValueError("review template is missing required placeholders")
    return template.replace(
        "__REVIEW_TITLE__", html.escape(clean_text(manifest.get("title")))
    ).replace("__REVIEW_DATA__", _script_safe_json(manifest))


def build_site(args: argparse.Namespace) -> dict[str, Any]:
    comparison_path = Path(args.comparison_path).resolve()
    output_dir = Path(args.output_dir).resolve()
    assets_dir = output_dir / "assets"
    output_dir.mkdir(parents=True, exist_ok=True)
    assets_dir.mkdir(parents=True, exist_ok=True)

    raw_rows = list(read_jsonl(comparison_path))
    all_records = collect_extraction_records(
        Path(args.model_cache_path).resolve(),
        raw_rows,
    )
    rows = select_review_rows(
        reclassify_review_rows(raw_rows, all_records),
        include_mode=args.include_mode,
    )
    if args.expected_items is not None and len(rows) != args.expected_items:
        raise ValueError(
            f"expected {args.expected_items} review items but selected {len(rows)}"
        )
    records = {
        cache_key: all_records[cache_key]
        for cache_key in {clean_text(row.get("cache_key")) for row in rows}
    }
    resolver = CachedWikipediaAssetResolver(
        wikipedia_cache_dir=Path(args.wikipedia_cache_dir).resolve(),
        image_cache_dir=Path(args.image_cache_dir).resolve(),
        output_dir=output_dir,
        text_chunk_chars=args.text_chunk_chars,
        min_text_chunk_chars=args.min_text_chunk_chars,
    )
    items = build_review_items(
        rows,
        records,
        resolver=resolver,
        assets_dir=assets_dir,
    )
    item_ids = "\n".join(item["review_id"] for item in items)
    dataset_id = "terra-luna-" + hashlib.sha256(item_ids.encode("utf-8")).hexdigest()[:16]
    modality_counts = Counter(item["asset_type"] for item in items)
    group_counts = Counter(item["comparison_group"] for item in items)
    manifest = {
        "schema_version": "mmdd-auto-check-human-review-v1",
        "dataset_id": dataset_id,
        "title": args.title,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scope": {
            "include_mode": args.include_mode,
            "item_count": len(items),
            "modalities": dict(sorted(modality_counts.items())),
            "comparison_groups": dict(sorted(group_counts.items())),
        },
        "review_protocol": {
            "blind_until_locked": True,
            "human_verdicts": sorted(HUMAN_VERDICTS),
            "supported_definition": (
                "The evidence directly supports extracting the claimed value "
                "for the target entity and attribute."
            ),
            "contradicted_definition": (
                "The same attribute is explicit in the evidence but has a "
                "different value."
            ),
            "insufficient_definition": (
                "The evidence does not reliably establish the target attribute."
            ),
            "schema_bad_definition": (
                "The attribute label or row semantics are too ambiguous to audit."
            ),
        },
        "sources": [
            {"name": comparison_path.name, "role": "Terra/Luna comparison"},
            {"name": "model_attribute_extractions.jsonl", "role": "row context"},
            {"name": "Wikipedia evidence cache", "role": "text and image evidence"},
        ],
        "items": items,
    }

    template = Path(args.template_path).read_text(encoding="utf-8")
    (output_dir / "review.html").write_text(
        render_review_html(template, manifest),
        encoding="utf-8",
    )
    write_json(output_dir / "review_manifest.json", manifest)
    build_receipt = {
        "schema_version": "mmdd-auto-check-review-build-v1",
        "dataset_id": dataset_id,
        "item_count": len(items),
        "text_items": modality_counts.get("text", 0),
        "image_items": modality_counts.get("image", 0),
        "support_disagreements": group_counts.get("support_disagreement", 0),
        "both_supported": group_counts.get("both_supported", 0),
        "review_html": "review.html",
        "manifest": "review_manifest.json",
    }
    write_json(output_dir / "build_receipt.json", build_receipt)
    return build_receipt


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a blind human-review site from Terra/Luna comparisons."
    )
    parser.add_argument("--comparison_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_cache_path", default=str(DEFAULT_MODEL_CACHE))
    parser.add_argument("--wikipedia_cache_dir", default=str(DEFAULT_WIKIPEDIA_CACHE))
    parser.add_argument("--image_cache_dir", default=str(DEFAULT_IMAGE_CACHE))
    parser.add_argument("--template_path", default=str(DEFAULT_TEMPLATE))
    parser.add_argument(
        "--include_mode",
        choices=("union_supported", "support_disagreements", "all"),
        default="union_supported",
    )
    parser.add_argument("--expected_items", type=int, default=None)
    parser.add_argument("--text_chunk_chars", type=int, default=800)
    parser.add_argument("--min_text_chunk_chars", type=int, default=120)
    parser.add_argument("--title", default="Terra / Luna Auto-check 人工复核")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    try:
        receipt = build_site(parse_args(argv))
        print(json.dumps(receipt, ensure_ascii=False, sort_keys=True), flush=True)
        return 0
    except Exception as error:
        print(
            f"ERROR: review site build stopped ({type(error).__name__})",
            file=sys.stderr,
            flush=True,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
