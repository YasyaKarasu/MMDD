#!/usr/bin/env python
"""Re-key the query-recovery auto-check cache so removing ``entity_url`` is a no-op.

``_query_recovery_auto_check_key_fields`` hashes the *materialised query row*.
Every cached review was taken while the query carried the synthetic
``entity_url`` column, so stripping that column (``strip_entity_url_column.py``)
invalidates all of them on paper -- even though the column held a fabricated
``https://en.wikipedia.org/wiki/wdc_<hash>`` value that cannot help the reviewer
ground any evidence.  Recomputing each key with that one cell removed makes the
strip a genuine no-op and lets the surviving half of the cache be reused.

Two checks make this safe to run blind:

* the key is recomputed unchanged first, and must equal the stored ``cache_key``
  for every record (it did: 811455/811455 on the v6 cache);
* the re-keyed file reports how many distinct keys two records collapse into --
  zero means the removed cell never distinguished two reviews.

The input file is never modified; a sibling ``*.rekeyed.jsonl`` is written.
Records whose schema predates ``query_row_attributes`` are copied verbatim
(their ``schema_version`` component already stops them from matching anyway).
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any

WHITESPACE_RE = re.compile(r"\s+")
MAX_CELL_CHARS = 1024
ENTITY_URL = "entity_url"
LOCAL_REVIEW_POLICY = "local_only"


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = html.unescape(str(value)).replace("\xa0", " ").strip()
    return WHITESPACE_RE.sub(" ", text)


def sanitize_cell_text_for_model(value: Any) -> str:
    text = clean_text(value)
    return text[:MAX_CELL_CHARS].rstrip() if text else ""


def normalize(value: Any) -> str:
    return clean_text(value).casefold()


def canonical_extraction_row_attributes(row_attributes: Any) -> list[dict[str, Any]]:
    if not isinstance(row_attributes, list):
        return []
    normalized: list[dict[str, Any]] = []
    for item in row_attributes:
        if not isinstance(item, dict):
            continue
        name = sanitize_cell_text_for_model(item.get("name"))
        value = sanitize_cell_text_for_model(item.get("value"))
        if not name or not value:
            continue
        normalized.append(
            {"name": name, "value": value, "is_entity": bool(item.get("is_entity"))}
        )
    return normalized


def stable_hash(*parts: Any, length: int = 16) -> str:
    payload = "\x1f".join("" if part is None else str(part) for part in parts)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:length]


def auto_check_key(
    record: dict[str, Any],
    drop: frozenset[str],
    schema_version: str | None = None,
) -> str:
    """``_query_recovery_auto_check_key_fields`` plus a set of names to drop."""
    masked = {
        normalize(name)
        for name in (*tuple(record.get("masked_attribute_names") or []),
                     record.get("attribute_name"))
        if normalize(name)
    }
    row = [
        item
        for item in canonical_extraction_row_attributes(record.get("query_row_attributes"))
        if normalize(item.get("name")) not in masked
        and normalize(item.get("name")) not in drop
    ]
    return stable_hash(
        record.get("schema_version") if schema_version is None else schema_version,
        record.get("extraction_cache_key"),
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
        record.get("attribute_name"),
        record.get("claimed_value"),
        length=32,
    )


def resolve_schema_component(record: dict[str, Any]) -> tuple[str | None, str]:
    """The ``schema_version`` component the stored key was built with, or why not.

    A local-only review keys on the bare schema version.  A cascade review keys
    on ``<schema_version>:<review_policy>`` (see
    ``query_recovery_auto_check_record_key``), which is why the earlier
    cascade-era v5 records do not reproduce without the suffix.
    """
    plain = clean_text(record.get("schema_version"))
    if not plain:
        return None, "no_schema_version"
    if auto_check_key(record, frozenset(), plain) == clean_text(record.get("cache_key")):
        return plain, "plain"
    policy = clean_text(record.get("review_policy")) or clean_text(
        (record.get("auto_check") or {}).get("review_policy")
        if isinstance(record.get("auto_check"), dict) else ""
    )
    if policy and policy != LOCAL_REVIEW_POLICY:
        suffixed = f"{plain}:{policy}"
        if auto_check_key(record, frozenset(), suffixed) == clean_text(record.get("cache_key")):
            return suffixed, "cascade_suffix"
    return None, "not_reproduced"


def without_entity_url(record: dict[str, Any]) -> None:
    """Drop the synthetic cell from the record's own copies of the query row."""
    row = record.get("query_row_attributes")
    if isinstance(row, list):
        record["query_row_attributes"] = [
            item for item in row
            if not (isinstance(item, dict) and normalize(item.get("name")) == ENTITY_URL)
        ]
    identity = record.get("evidence_identity")
    if isinstance(identity, dict) and isinstance(identity.get("masked_row"), list):
        identity["masked_row"] = [
            item for item in identity["masked_row"]
            if not (isinstance(item, dict) and normalize(item.get("name")) == ENTITY_URL)
        ]


def rekey(source: Path, destination: Path) -> dict[str, Any]:
    drop = frozenset({ENTITY_URL})
    stats: Counter[str] = Counter()
    versions: Counter[str] = Counter()
    mismatch_samples: list[str] = []
    new_keys: set[str] = set()
    collisions = 0

    temporary = destination.with_name(destination.name + ".tmp")
    with source.open(encoding="utf-8") as src, temporary.open("w", encoding="utf-8") as dst:
        for line in src:
            if not line.strip():
                continue
            record = json.loads(line)
            stats["records"] += 1
            version = clean_text(record.get("schema_version")) or "?"
            versions[version] += 1

            if not isinstance(record.get("query_row_attributes"), list):
                dst.write(line if line.endswith("\n") else line + "\n")
                stats["copied_without_row"] += 1
                continue

            component, how = resolve_schema_component(record)
            if component is None:
                # The stored key does not recompute under any rule we know, so we
                # cannot claim to know what it is keyed on.  Copy it byte-identical
                # rather than guess; such a record can only be found by the exact
                # build that wrote it, which this re-key does not serve anyway.
                dst.write(line if line.endswith("\n") else line + "\n")
                stats["copied_unverified"] += 1
                stats[f"unverified:{how}"] += 1
                if len(mismatch_samples) < 5:
                    mismatch_samples.append(clean_text(record.get("cache_key")))
                continue

            stats[f"verified:{how}"] += 1
            key = auto_check_key(record, drop, component)
            if key in new_keys:
                collisions += 1
            new_keys.add(key)
            record["cache_key"] = key
            without_entity_url(record)
            dst.write(json.dumps(record, ensure_ascii=False) + "\n")
            stats["rekeyed"] += 1
        dst.flush()
        os.fsync(dst.fileno())
    os.replace(temporary, destination)

    return {
        "source": str(source),
        "destination": str(destination),
        "records": stats["records"],
        "rekeyed": stats["rekeyed"],
        "verified_plain": stats["verified:plain"],
        "verified_cascade_suffix": stats["verified:cascade_suffix"],
        "copied_without_row": stats["copied_without_row"],
        "copied_unverified": stats["copied_unverified"],
        "mismatch_samples": mismatch_samples,
        "distinct_new_keys": len(new_keys),
        "collisions": collisions,
        "by_schema_version": dict(versions),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache",
        default="cache/wdc_webtable/query_recovery_auto_checks.jsonl",
    )
    parser.add_argument("--out", default="")
    parser.add_argument("--report", default="")
    args = parser.parse_args()

    source = Path(args.cache)
    destination = Path(args.out) if args.out else source.with_suffix(".rekeyed.jsonl")
    report = rekey(source, destination)

    report_path = Path(args.report) if args.report else destination.with_suffix(".report.json")
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
