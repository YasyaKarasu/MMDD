"""Raw EntiTables-20K wiring: artifacts, GT combination, populations.

Spec sections 2 and 7.  Objects are enumerated from the original artifacts only;
no historical feature manifest, target list or cohort is consulted.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from .config import ConfigError, assert_path_allowed
from .util import (
    byte_order,
    digest_jsonl,
    file_metadata,
    read_json,
    sha256_path,
    stable_digest,
    stable_order,
    write_jsonl,
)

IMAGE_KINDS = tuple(range(3, 11))
KIND_GLOBAL = 0
KIND_SCHEMA = 1
KIND_ROW_GROUP = 2

DEFAULT_MAX_CELL_CHARS = 1024
TABLE_ROW_FORMATS = ("values", "named_cells")


# --------------------------------------------------------------------------
# artifact reading
# --------------------------------------------------------------------------


def artifact_paths(output_dir: Path, artifact: str) -> list[Path]:
    """Paths backing one raw artifact, taken only from the dataset's own manifest."""
    manifest_path = output_dir / "dataset_manifest.json"
    if not manifest_path.is_file():
        single = output_dir / f"{artifact}.jsonl"
        if single.is_file():
            return [single]
        return []
    payload = read_json(manifest_path)
    if "complete" in payload and payload["complete"] is not True:
        raise ConfigError(f"dataset is incomplete: {output_dir}")
    record = payload.get("artifacts", {}).get(artifact)
    if record is not None:
        entries = (
            list(record.get("shards", []))
            if "shards" in record
            else [record]
        )
    else:
        single = payload.get("single_files", {}).get(artifact)
        if single is None:
            return []
        entries = [{"path": single}]
    paths: list[Path] = []
    for entry in entries:
        path = output_dir / str(entry["path"])
        if not path.is_file():
            raise ConfigError(f"dataset artifact is missing: {path}")
        if "sha256" in entry and sha256_path(path) != entry["sha256"]:
            raise ConfigError(f"dataset artifact hash mismatch: {path}")
        if "bytes" in entry and path.stat().st_size != int(entry["bytes"]):
            raise ConfigError(f"dataset artifact size mismatch: {path}")
        paths.append(path)
    return paths


def iter_artifact(output_dir: Path, artifact: str) -> Iterable[dict[str, Any]]:
    for path in artifact_paths(output_dir, artifact):
        with path.open("r", encoding="utf-8") as handle:
            for number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ConfigError(f"record is not an object: {path}:{number}")
                yield value


# --------------------------------------------------------------------------
# pure content serialization (spec section 4.2)
# --------------------------------------------------------------------------


def clean_text(value: Any) -> str:
    return "" if value is None else str(value)


def get_cell(row: dict[str, Any], column_index: int) -> dict[str, Any]:
    for cell in row.get("cells", []):
        if int(cell.get("column_index", -1)) == int(column_index):
            return cell
    for cell in row.get("cells", []):
        if int(cell.get("column_name", -1) or -1) == int(column_index):
            return cell
    raise KeyError(f"row {row.get('row_id')} has no column_index {column_index}")


def serialize_table_parts(
    table: dict[str, Any],
    max_rows: int,
    max_cell_chars: int = DEFAULT_MAX_CELL_CHARS,
    *,
    row_format: str = "values",
) -> list[str]:
    """Visible content only.  Column names and cell text; no annotation fields."""
    if row_format not in TABLE_ROW_FORMATS:
        raise ValueError(f"row_format must be one of: {', '.join(TABLE_ROW_FORMATS)}")
    headers = [clean_text(column.get("column_name")) for column in table["columns"]]
    parts = ["Columns: " + " | ".join(headers)]
    for row in table["rows"][:max_rows]:
        values = []
        for column in table["columns"]:
            value = clean_text(get_cell(row, int(column["column_index"])).get("text"))
            values.append(value[:max_cell_chars].rstrip())
        if row_format == "named_cells":
            values = [
                f"{header or f'column_{index}'}: {value}"
                for index, (header, value) in enumerate(zip(headers, values))
            ]
        parts.append("Row: " + " | ".join(values))
    return parts


def resolve_table_references(
    records: list[dict[str, Any]],
    sources: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Inline every ``source_table_ref`` from the referenced source table.

    A missing referenced source table is a hard error (spec section 2 item 3); it
    is never degraded to an empty-column negative table.
    """
    needed = {
        str(record["source_table_ref"]["source_table_id"])
        for record in records
        if "source_table_ref" in record
    }
    missing = sorted(needed - set(sources))
    if missing:
        raise ConfigError(
            "missing referenced source tables: " + ", ".join(missing[:20])
            + (f" (+{len(missing) - 20} more)" if len(missing) > 20 else "")
        )
    resolved = []
    for record in records:
        if "source_table_ref" not in record:
            resolved.append(record)
            continue
        source = sources[str(record["source_table_ref"]["source_table_id"])]
        merged = {**source}
        merged.update({k: v for k, v in record.items() if k != "source_table_ref"})
        resolved.append(merged)
    return resolved


# --------------------------------------------------------------------------
# the loaded lake
# --------------------------------------------------------------------------


class RawLake:
    """Everything CLEAN-R1 reads from the original dataset directory."""

    def __init__(self, dataset_root: Path) -> None:
        self.root = Path(dataset_root)
        self.query_tables: dict[str, dict[str, Any]] = {}
        self.lake_tables: dict[str, dict[str, Any]] = {}
        self.assets: dict[str, dict[str, Any]] = {}
        self.qrels: list[dict[str, Any]] = []
        self.recoveries: list[dict[str, Any]] = []
        self.source_tables: dict[str, dict[str, Any]] = {}
        self.counts: Counter = Counter()
        self.failures: list[dict[str, Any]] = []
        self.split_of_query: dict[str, str] = {}

    # -- loading -----------------------------------------------------------

    def load(self) -> None:
        self.source_tables = {
            str(record["source_table_id"]): record
            for record in iter_artifact(self.root, "source_tables")
        }
        self.counts["source_tables_total"] = len(self.source_tables)
        self.counts["source_tables_referenced"] = 0

        for record in iter_artifact(self.root, "query_tables"):
            table_id = str(record["table_id"])
            if table_id in self.query_tables or table_id in self.lake_tables:
                raise ConfigError(f"duplicate query table id: {table_id}")
            self.query_tables[table_id] = record
            split = record.get("split")
            if split is None:
                self.failures.append(
                    {"kind": "query_without_split", "object_id": table_id}
                )
            else:
                self.split_of_query[table_id] = str(split)
        self.counts["query_tables"] = len(self.query_tables)

        raw_lake = list(iter_artifact(self.root, "data_lake_tables"))
        referenced = {
            str(r["source_table_ref"]["source_table_id"])
            for r in raw_lake
            if "source_table_ref" in r
        }
        self.counts["source_tables_referenced"] = len(referenced)
        resolved_lake = resolve_table_references(raw_lake, self.source_tables)
        for record in resolved_lake:
            table_id = str(record["table_id"])
            if table_id in self.lake_tables or table_id in self.query_tables:
                raise ConfigError(f"duplicate lake table id: {table_id}")
            self.lake_tables[table_id] = record
        self.counts["lake_tables"] = len(self.lake_tables)

        for record in iter_artifact(self.root, "bridge_assets"):
            asset_id = str(record["asset_id"])
            if asset_id in self.assets:
                raise ConfigError(f"duplicate asset id: {asset_id}")
            self.assets[asset_id] = record
        self.counts["bridge_assets"] = len(self.assets)
        self.counts["bridge_assets_text"] = sum(
            1 for a in self.assets.values() if str(a.get("asset_type")) == "text"
        )
        self.counts["bridge_assets_image"] = sum(
            1 for a in self.assets.values() if str(a.get("asset_type")) == "image"
        )

        self.qrels = list(iter_artifact(self.root, "qrels"))
        self.counts["qrels"] = len(self.qrels)
        self.recoveries = list(iter_artifact(self.root, "evidence_recoveries"))
        self.counts["evidence_recoveries"] = len(self.recoveries)

    # -- validation --------------------------------------------------------

    def validate(self, data_config: dict[str, Any]) -> dict[str, Any]:
        """Hard-stop on referential, split or reason violations."""
        known = set(self.query_tables) | set(self.lake_tables)
        explicit_reason = str(data_config["explicit_reason"])
        implicit_reason = str(data_config["implicit_reason"])
        reasons: Counter = Counter()
        missing_targets: list[str] = []
        split_conflicts: list[str] = []

        for record in self.qrels:
            query_id = str(record["query_table_id"])
            target_id = str(record["target_table_id"])
            reason = str(record.get("reason"))
            reasons[reason] += 1
            if query_id not in self.query_tables:
                missing_targets.append(f"qrel query missing: {query_id}")
            if target_id not in self.lake_tables:
                missing_targets.append(f"qrel target missing: {target_id}")
            declared = self.split_of_query.get(query_id)
            if declared is not None and str(record.get("split")) != declared:
                split_conflicts.append(
                    f"{query_id}: qrel split {record.get('split')} != table split {declared}"
                )
        unknown_reasons = sorted(set(reasons) - {explicit_reason, implicit_reason})
        if unknown_reasons:
            raise ConfigError(
                "unexpected qrel reason(s): "
                + ", ".join(f"{r} ({reasons[r]})" for r in unknown_reasons)
                + f". Only {explicit_reason!r} and {implicit_reason!r} are defined; "
                "an unknown reason is never silently treated as explicit."
            )
        if missing_targets:
            raise ConfigError(
                "dangling GT references (hard error, spec section 2.2 item 3): "
                + "; ".join(missing_targets[:20])
            )
        if split_conflicts:
            raise ConfigError("qrel/table split conflicts: " + "; ".join(split_conflicts[:20]))

        asset_ids = set(self.assets)
        bad_recovery = 0
        for record in self.recoveries:
            asset_id = str((record.get("evidence") or {}).get("asset_id", ""))
            if asset_id not in asset_ids:
                bad_recovery += 1

        groups: dict[str, set[str]] = defaultdict(set)
        for record in self.qrels:
            query_id = str(record["query_table_id"])
            split = self.split_of_query.get(query_id)
            if split is None:
                continue
            groups[split].add(str(record.get("source_table_id")))
        overlaps = {}
        ordered = sorted(groups)
        for i, left in enumerate(ordered):
            for right in ordered[i + 1 :]:
                shared = groups[left] & groups[right]
                if shared:
                    overlaps[f"{left}/{right}"] = len(shared)
        if overlaps:
            raise ConfigError(
                "train/dev/test query source groups are not disjoint: "
                + ", ".join(f"{k}={v}" for k, v in sorted(overlaps.items()))
            )
        return {
            "qrel_reason_counts": {k: reasons[k] for k in sorted(reasons)},
            "unknown_reasons": [],
            "missing_target_references": 0,
            "split_conflicts": 0,
            "recoveries_with_unknown_asset": bad_recovery,
            "source_groups_per_split": {
                k: len(v) for k, v in sorted(groups.items())
            },
            "source_group_overlap": {},
        }

    # -- content accessors -------------------------------------------------

    def visible_content(self, object_id: str, max_rows: int, max_cell_chars: int,
                        row_format: str) -> dict[str, Any]:
        if object_id in self.query_tables:
            table = self.query_tables[object_id]
            split = str(table.get("split"))
            role = "query"
        elif object_id in self.lake_tables:
            table = self.lake_tables[object_id]
            # Targets are shared across splits and carry no split of their own.
            split = None
            role = "target"
        else:
            raise KeyError(object_id)
        parts = serialize_table_parts(
            table, max_rows, max_cell_chars, row_format=row_format
        )
        return {
            "object_id": object_id,
            "object_type": "table",
            "embedding_role": role,
            "table_parts": parts,
            "num_rows_visible": min(len(table["rows"]), max_rows),
            "num_rows_total": len(table["rows"]),
            "split": split,
        }


# --------------------------------------------------------------------------
# GT combination (spec sections 1.2, 2.3, 7)
# --------------------------------------------------------------------------


def witness_filter(
    recovery: dict[str, Any], policy_name: str
) -> tuple[bool, str]:
    """Decide whether one recovery record is a witness positive.

    Records without ``auto_check`` are dataset GT without extra review.  Records
    with ``auto_check`` need a non-empty review list whose verdicts are all
    ``supported`` and the original policy name preserved verbatim.
    """
    auto_check = recovery.get("auto_check")
    if not auto_check:
        return True, "dataset_gt_without_extra_review"
    reviews = auto_check.get("reviews") or []
    policy = str(auto_check.get("policy") or "")
    if not reviews:
        return False, "auto_check_missing_reviews"
    if not all(str(item.get("verdict")) == "supported" for item in reviews):
        return False, "auto_check_not_all_supported"
    if policy != policy_name:
        return False, f"auto_check_policy_mismatch:{policy}"
    return True, f"auto_check_supported:{policy}"


def content_key(asset_type: str, asset: dict[str, Any]) -> tuple[str, str] | None:
    """Deterministic content identity (spec section 3).

    Text: NFC, CRLF->LF, strip outer whitespace, then SHA256 of UTF-8.
    Image: SHA256 of the raw file bytes.
    """
    import unicodedata

    if asset_type == "text":
        text = clean_text(asset.get("content"))
        text = unicodedata.normalize("NFC", text).replace("\r\n", "\n").replace("\r", "\n")
        text = text.strip()
        if not text:
            return None
        return "text", sha256_text_key(text)
    if asset_type == "image":
        # The dataset already records the content hash of the media blob; using it
        # avoids re-reading ~36 GiB of images on every command while still keying
        # on the original file bytes (spec 3: SHA256 of the raw file).
        recorded = asset.get("sha256")
        if recorded:
            return "image", str(recorded)
        path = asset.get("local_path") or asset.get("relative_path")
        if not path:
            return None
        path = Path(str(path))
        if not path.is_file():
            return None
        return "image", sha256_path(path)
    return None


def sha256_text_key(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_gt(
    lake: RawLake, data_config: dict[str, Any], policy_name: str
) -> dict[str, Any]:
    """Raw GT tables: G_Q, D_Q, I_Q and W(Q,T) per split."""
    explicit_reason = str(data_config["explicit_reason"])
    implicit_reason = str(data_config["implicit_reason"])

    # canonical asset id == smallest content-equivalent asset_id (spec section 3)
    content_groups: dict[tuple[str, str], list[str]] = defaultdict(list)
    unusable: dict[str, str] = {}
    for asset_id in sorted(lake.assets):
        asset = lake.assets[asset_id]
        asset_type = str(asset.get("asset_type"))
        if asset_type not in ("text", "image"):
            unusable[asset_id] = f"unsupported_asset_type:{asset_type}"
            continue
        key = content_key(asset_type, asset)
        if key is None:
            reason = "empty_text" if asset_type == "text" else "undecodable_image"
            unusable[asset_id] = reason
            continue
        content_groups[key].append(asset_id)

    alias: dict[str, str] = {}
    canonical: dict[str, dict[str, Any]] = {}
    for key, members in content_groups.items():
        members = byte_order(members)
        canon = members[0]
        for member in members:
            alias[member] = canon
        canonical[canon] = {
            "asset_id": canon,
            "asset_type": key[0],
            "content_sha256": key[1],
            "aliases": members[1:],
            "alias_count": len(members) - 1,
        }

    stats: Counter = Counter()
    witnesses: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    witness_provenance: Counter = Counter()
    for record in lake.recoveries:
        query_id = str(record.get("query_table_id", ""))
        target_id = str(record.get("target_table_id", ""))
        split = str(record.get("split"))
        asset_id = str((record.get("evidence") or {}).get("asset_id", ""))
        ok, provenance = witness_filter(record, policy_name)
        witness_provenance[provenance] += 1
        if not ok:
            stats["recovery_rejected"] += 1
            continue
        query_split = lake.split_of_query.get(query_id)
        if query_id not in lake.query_tables or target_id not in lake.lake_tables:
            stats["recovery_dropped_missing_object"] += 1
            continue
        if query_split is None or query_split != split:
            stats["recovery_dropped_split_mismatch"] += 1
            continue
        canon = alias.get(asset_id)
        if canon is None:
            stats["recovery_dropped_unusable_asset"] += 1
            continue
        if canonical[canon]["asset_type"] not in ("text", "image"):
            stats["recovery_dropped_modality"] += 1
            continue
        witnesses[query_id][target_id].add(canon)
        stats["witness_pairs"] += 1

    qrels_by_query: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in lake.qrels:
        query_id = str(record["query_table_id"])
        if int(record.get("rel", 0)) <= 0:
            continue
        qrels_by_query[query_id].append(record)

    per_split: dict[str, dict[str, Any]] = {}
    for split in ("train", "dev", "test"):
        population: list[dict[str, Any]] = []
        direct: dict[str, list[str]] = {}
        implicit: dict[str, list[str]] = {}
        all_positive: dict[str, list[str]] = {}
        kind_of_query: dict[str, str] = {}
        no_positive = 0
        for query_id in byte_order(lake.split_of_query):
            if lake.split_of_query[query_id] != split:
                continue
            labels = qrels_by_query.get(query_id, [])
            if not labels:
                no_positive += 1
                continue
            d_targets = {
                str(r["target_table_id"])
                for r in labels
                if str(r.get("reason")) == explicit_reason
            }
            i_targets = {
                str(r["target_table_id"])
                for r in labels
                if str(r.get("reason")) == implicit_reason
            }
            g_targets = d_targets | i_targets
            direct[query_id] = byte_order(d_targets)
            implicit[query_id] = byte_order(i_targets - d_targets)
            all_positive[query_id] = byte_order(g_targets)
            if d_targets and i_targets - d_targets:
                kind_of_query[query_id] = "mixed"
            elif d_targets:
                kind_of_query[query_id] = "explicit"
            else:
                kind_of_query[query_id] = "implicit"
            query_table = lake.query_tables[query_id]
            source_ids = {str(r.get("source_table_id")) for r in labels}
            source_ids.add(str(query_table.get("source_table_id")))
            source_ids.discard("None")
            if len(source_ids) != 1:
                raise ConfigError(
                    f"source group is not unique for {query_id}: {sorted(source_ids)}"
                )
            population.append(
                {
                    "query_id": query_id,
                    "split": split,
                    "query_kind": kind_of_query[query_id],
                    "source_table_id": source_ids.pop(),
                    "positive_target_ids": all_positive[query_id],
                    "direct_target_ids": direct[query_id],
                    "implicit_target_ids": implicit[query_id],
                    "witnesses": {
                        t: byte_order(v) for t, v in sorted(witnesses.get(query_id, {}).items())
                    },
                }
            )
        population.sort(key=lambda r: r["query_id"].encode("utf-8"))
        per_split[split] = {
            "population": population,
            "direct": direct,
            "implicit": implicit,
            "positives": all_positive,
            "kind": kind_of_query,
            "no_positive_qrel": no_positive,
        }
        stats[f"{split}_population"] = len(population)
        stats[f"{split}_no_positive_qrel"] = no_positive
        stats[f"{split}_implicit_targets"] = sum(len(v) for v in implicit.values())
        stats[f"{split}_direct_targets"] = sum(len(v) for v in direct.values())

    return {
        "per_split": per_split,
        "alias": alias,
        "canonical": canonical,
        "unusable_assets": unusable,
        "stats": dict(stats),
        "witness_provenance": dict(witness_provenance),
    }


def semantic_hash(population: list[dict[str, Any]], extra: dict[str, Any]) -> str:
    """Hash over supervision and the unlabelled corpus identity, not file bytes."""
    payload = {"population": population, "config": extra}
    return digest_jsonl([payload])


_GT_MEMO: dict[tuple[str, str], dict[str, Any]] = {}
_LAKE_MEMO: dict[str, RawLake] = {}


def load_gt_cached(dataset_root: Path, data_config: dict[str, Any], policy_name: str) -> dict[str, Any]:
    """Memoised GT combination so a multi-shard cache build reads the raw lake once."""
    key = (str(Path(dataset_root).resolve()), stable_digest(data_config, policy_name))
    if key not in _GT_MEMO:
        _GT_MEMO[key] = build_gt(load_lake_cached(dataset_root), data_config, policy_name)
    return _GT_MEMO[key]


def load_lake_cached(dataset_root: Path) -> RawLake:
    """One RawLake per process; the raw artifacts are read-only inputs."""
    key = str(Path(dataset_root).resolve())
    cached = _LAKE_MEMO.get(key)
    if cached is None:
        cached = RawLake(dataset_root)
        cached.load()
        _LAKE_MEMO[key] = cached
    return cached


_LAKE_MEMO: dict[str, RawLake] = {}
