"""Subcommand implementations for ``python -m mmdd_stage1_clean``."""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch

from . import cache as cache_module
from . import config as config_module
from . import data
from . import evaluate
from . import models
from . import retrieve
from . import sampling
from . import train
from .config import (
    ConfigError,
    assert_path_allowed,
    freeze_resolved,
    load_resolved,
    resolve_config,
)
from .data import RawLake, artifact_paths, build_gt
from .models import build_student, build_teacher, set_seed
from .timing import Timing
from .util import (
    byte_order,
    digest_jsonl,
    file_metadata,
    git_commit,
    log_line,
    read_json,
    read_jsonl,
    sha256_path,
    stable_digest,
    utcnow,
    write_json,
    write_jsonl,
)


AUTO_CHECK_POLICY = "keep_source_canonical_supported_only_fail_closed"


def _resolved_or_raise(output_root: Path) -> dict[str, Any]:
    return config_module.load_resolved(output_root)


def _artifact_inventory(dataset_root: Path) -> dict[str, Any]:
    """SHA256 of every raw input file, plus a hash over the ordered listing."""
    listed: list[dict[str, Any]] = []
    for path in sorted(dataset_root.rglob("*")):
        if not path.is_file():
            continue
        listed.append(file_metadata(path, dataset_root))
    listed.sort(key=lambda r: r["path"].encode("utf-8"))
    return {
        "root": str(dataset_root),
        "file_count": len(listed),
        "total_bytes": sum(r["bytes"] for r in listed),
        "files": listed,
        "ordered_listing_sha256": digest_jsonl(listed),
    }


def _backbone_inventory(backbone: Path) -> dict[str, Any]:
    listed = []
    for path in sorted(backbone.rglob("*")):
        if path.is_file() and ".hfd" not in path.parts:
            listed.append(file_metadata(path, backbone))
    listed.sort(key=lambda r: r["path"].encode("utf-8"))
    return {
        "root": str(backbone),
        "file_count": len(listed),
        "total_bytes": sum(r["bytes"] for r in listed),
        "files": listed,
        "ordered_listing_sha256": digest_jsonl(listed),
    }


def cmd_audit_input(args, spec, spec_dir, cwd, output_root, receipt) -> dict[str, Any]:
    """Spec step A: resolve paths, hash inputs, derive the raw GT tables."""
    resolved = resolve_config(spec, cwd=cwd)
    output_root.mkdir(parents=True, exist_ok=True)
    dataset_root = Path(resolved["paths"]["dataset_root"])
    backbone = Path(resolved["paths"]["backbone_dir"])
    assert_path_allowed(dataset_root, purpose="raw dataset root")

    inventory = {
        "protocol_id": resolved["protocol_id"],
        "recorded_utc": utcnow(),
        "source_spec_dir": str(spec_dir),
        "spec_config_path": resolved["spec_config_path"],
        "spec_config_sha256": resolved["spec_config_sha256"],
        "spec_doc_sha256": resolved["spec_doc_sha256"],
        "dataset": _artifact_inventory(dataset_root),
        "backbone": _backbone_inventory(backbone),
        "git_commit": git_commit(Path(resolved["paths"]["repo_root"])),
    }

    lake = RawLake(dataset_root)
    lake.load()
    validation = lake.validate(resolved["data"])
    gt = build_gt(lake, resolved["data"], AUTO_CHECK_POLICY)

    write_jsonl(
        output_root / "objects" / "content_aliases.jsonl",
        (
            {
                "canonical_id": canon,
                "asset_type": entry["asset_type"],
                "content_sha256": entry["content_sha256"],
                "alias_ids": entry["aliases"],
            }
            for canon, entry in sorted(gt["canonical"].items())
        ),
    )
    write_jsonl(
        output_root / "objects" / "failures.jsonl",
        (
            {"asset_id": asset_id, "reason": reason}
            for asset_id, reason in sorted(gt["unusable_assets"].items())
        ),
    )

    population_summary: dict[str, Any] = {}
    semantic_payload: dict[str, Any] = {}
    for split in ("train", "dev", "test"):
        data = gt["per_split"][split]
        write_jsonl(
            output_root / "raw_gt" / f"{split}.population.jsonl",
            data["population"],
        )
        write_jsonl(
            output_root / "raw_gt" / f"{split}.qrels.jsonl",
            (
                record
                for record in lake.qrels
                if lake.split_of_query.get(str(record["query_table_id"])) == split
            ),
        )
        keep = {row["query_id"] for row in data["population"]}
        allowed_targets = {
            target
            for row in data["population"]
            for target in row["positive_target_ids"]
        }
        write_jsonl(
            output_root / "raw_gt" / f"{split}.recoveries.jsonl",
            (
                record
                for record in lake.recoveries
                if str(record.get("split")) == split
                and str(record.get("query_table_id")) in keep
            ),
        )
        population_summary[split] = {
            "queries_with_positive_qrel": len(data["population"]),
            "queries_without_positive_qrel": data["no_positive_qrel"],
            "positive_targets": sum(len(v) for v in data["positives"].values()),
            "direct_targets": sum(len(v) for v in data["direct"].values()),
            "implicit_targets": sum(len(v) for v in data["implicit"].values()),
            "kinds": {
                kind: sum(1 for v in data["kind"].values() if v == kind)
                for kind in sorted(set(data["kind"].values()))
            },
        }
        semantic_payload[split] = data["population"]

    semantic_hashes = {
        split: digest_jsonl([rows]) for split, rows in semantic_payload.items()
    }

    audit = {
        "counts": dict(lake.counts),
        "validation": validation,
        "population": population_summary,
        "witness_stats": gt["stats"],
        "witness_provenance": gt["witness_provenance"],
        "canonical_evidence": len(gt["canonical"]),
        "unusable_assets": len(gt["unusable_assets"]),
        "semantic_hashes": semantic_hashes,
        "evidence_recoveries_present": lake.counts["evidence_recoveries"] > 0,
    }
    if lake.counts["evidence_recoveries"] == 0:
        raise ConfigError(
            "no evidence_recoveries exist anywhere in the lake; the pre-registered "
            "bridging training cannot be formed. Stopping at input audit rather than "
            "silently degrading to a QT-only run."
        )
    write_json(output_root / "INPUT_INVENTORY.json", inventory)
    write_json(output_root / "SOURCE_SHA256.json", {
        "dataset_files": {
            r["path"]: r["sha256"] for r in inventory["dataset"]["files"]
        },
        "backbone_files": {
            r["path"]: r["sha256"] for r in inventory["backbone"]["files"]
        },
    })
    write_json(output_root / "audit" / "INPUT_AUDIT.json", audit)
    write_json(output_root / "audit" / "LAKE_COUNTS.json", dict(lake.counts))
    resolved["semantic_hashes"] = semantic_hashes
    resolved["input_audit_sha256"] = stable_digest(audit)
    freeze_resolved(output_root, resolved)
    log_line(
        f"audit-input: {lake.counts['query_tables']} queries, "
        f"{lake.counts['lake_tables']} lake tables, {lake.counts['bridge_assets']} assets, "
        f"{len(gt['canonical'])} canonical evidence objects"
    )
    return {
        "counts": dict(lake.counts),
        "population": population_summary,
        "semantic_hashes": semantic_hashes,
    }


def _load_lake(resolved: dict[str, Any]) -> RawLake:
    return data.load_lake_cached(Path(resolved["paths"]["dataset_root"]))


def _load_gt(resolved: dict[str, Any]) -> dict[str, Any]:
    return data.load_gt_cached(
        Path(resolved["paths"]["dataset_root"]), resolved["data"], AUTO_CHECK_POLICY
    )


def _merge_cache_shards(output_root: Path, resolved: dict[str, Any]) -> dict[str, Any]:
    """Merge per-GPU shard manifests into one canonical, object-ordered manifest."""
    feature_dir = output_root / "cache"
    # A shard set can legitimately contain the same object twice: repartitioning
    # the encoding work (say, to spread a remainder over two devices) re-encodes
    # objects another shard already holds.  Both encodings come from the same
    # model and prompt, so they are interchangeable; the newer file wins, but only
    # after the overlapping rows are checked to describe the same object.
    merged: dict[str, dict[str, Any]] = {}
    found: list[str] = []
    overlaps: list[dict[str, Any]] = []
    conflicts: list[str] = []
    for path in sorted(feature_dir.glob("gpu-*/manifest.jsonl")):
        found.append(path.parent.name)
        for line in path.open("r", encoding="utf-8"):
            if not line.strip():
                continue
            row = json.loads(line)
            row["_source_shard"] = path.parent.name
            candidate = {**row, "shard": f"{path.parent.name}/{row['shard']}"}
            existing = merged.get(row["object_id"])
            if existing is not None:
                if not _same_encoded_object(existing, candidate):
                    conflicts.append(
                        f"{row['object_id']} differs between "
                        f"{existing['_source_shard']} and {candidate['_source_shard']}"
                    )
                    continue
                overlaps.append(
                    {
                        "object_id": row["object_id"],
                        "kept": candidate,
                        "superseded": existing,
                    }
                )
            merged[row["object_id"]] = candidate
    # The merged cache is the authority on which objects are retrievable, so an
    # incomplete merge must stop the run rather than silently shrink every corpus.
    failures: list[dict[str, Any]] = []
    # Failure records land beside the manifest they belong to, which for a
    # partitioned run is the shard directory but for a single-process run is the
    # cache root.  Both are consulted so an object is never counted as missing
    # merely because its record sits in the other place.
    for path in sorted(feature_dir.glob("gpu-*/encode_failures.jsonl")) + [
        feature_dir / "encode_failures.jsonl"
    ]:
        if path.is_file():
            failures.extend(read_jsonl(path))
    corpus_path = output_root / "objects" / "corpus.jsonl"
    if corpus_path.is_file():
        seen_ids: set[str] = set()
        for row in read_jsonl(corpus_path):
            for object_id in row["object_ids"]:
                seen_ids.add(str(object_id))
        missing = sorted(seen_ids - set(merged))
        recorded_failures = {str(f.get("object_id")) for f in failures}
        unexplained = [o for o in missing if o not in recorded_failures]
        if unexplained:
            raise ConfigError(
                f"the merged cache covers {len(merged)} objects but {len(unexplained)} "
                f"enumerated objects are neither encoded nor recorded as failures, "
                f"e.g. {unexplained[:8]}. All shards must finish before merging; a "
                "short cache would silently shrink every corpus and every denominator."
            )
    if conflicts:
        raise ConfigError(
            "cache shards hold conflicting encodings of the same object: "
            + "; ".join(conflicts[:10])
            + f" ({len(conflicts)} total). Identical re-encodings may overlap; "
            "disagreeing ones never may."
        )
    overlap_report = {
        "overlapping_objects": len({o["object_id"] for o in overlaps}),
        "by_source_pair": {},
        "note": (
            "Duplicates arise when the encoding work is repartitioned. The rows "
            "agreed on modality, kind layout and mask structure, so either copy is "
            "usable; the file processed later is kept."
        ),
    }
    for entry in overlaps:
        pair = f"{entry['superseded']} -> {entry['kept']}"
        overlap_report["by_source_pair"][pair] = (
            overlap_report["by_source_pair"].get(pair, 0) + 1
        )
    ordered = [merged[k] for k in byte_order(merged)]
    # The merged manifest is the run's canonical object index: every later stage
    # resolves object ids through it, so it is written before the receipt that
    # describes it.
    write_jsonl(feature_dir / "manifest.jsonl", ordered)
    identity = verify_cache_identity(ordered)
    if overlaps:
        identity["overlap_byte_check"] = _verify_overlap_bytes(
            overlaps,
            feature_dir,
            int(resolved["cache"]["global_dimension"]),
            int(resolved["cache"]["summary_slots"]),
            limit=512,
        )
        if not identity["overlap_byte_check"]["identical"]:
            raise ConfigError(
                "overlapping cache shards disagree at the byte level: "
                + json.dumps(identity["overlap_byte_check"])[:600]
            )
    if not identity["consistent"]:
        raise ConfigError(
            "cache shards disagree on the encoding identity: "
            + json.dumps(identity)
            + ". Features from different models or prompt versions are never mixed; "
            "the affected shards must be re-encoded."
        )
    files = []
    for path in sorted(feature_dir.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(feature_dir).as_posix()
        if relative in ("manifest.jsonl", "CACHE_RECEIPT.json"):
            continue
        files.append(
            {"path": relative, "bytes": path.stat().st_size, "sha256": sha256_path(path)}
        )
    timing = _cache_timing_from_shards(feature_dir, found, ordered)
    write_json(
        feature_dir / "CACHE_RECEIPT.json",
        {
            "gpu_shards": found,
            "objects": len(ordered),
            "cache_fingerprint": ordered[0]["cache_fingerprint"],
            "cache_identity": identity,
            "overlaps": overlap_report,
            "encode_failures": len(failures),
            "files": files,
            "total_bytes": sum(item["bytes"] for item in files),
            "timing": timing,
        },
    )
    log_line(f"cache merge: {len(ordered)} objects from {len(found)} shards")
    log_line(f"cache merge timing: {json.dumps(timing)[:400]}")
    return {"objects": len(ordered), "shards": found, "timing": timing}


def _cache_timing_from_shards(
    feature_dir: Path, shards: list[str], ordered: list[dict[str, Any]]
) -> dict[str, Any]:
    """Combine shard timing receipts, falling back to the recorded command span.

    A shard that ran before the per-phase instrumentation existed has no
    TIMING.json, but every shard did write a CLI receipt with its own UTC start
    and end, so its wall clock is still a recorded measurement rather than an
    inference from file timestamps.
    """
    per_modality: dict[str, int] = {}
    for row in ordered:
        modality = row["modality"]
        per_modality[modality] = per_modality.get(modality, 0) + 1
    reports: dict[str, Any] = {}
    for shard in shards:
        path = feature_dir / shard / "TIMING.json"
        if path.is_file():
            reports[shard] = read_json(path)

    receipts: dict[str, Any] = {}
    receipt_path = feature_dir.parent / "COMMANDS.jsonl"
    if receipt_path.is_file():
        for record in read_jsonl(receipt_path):
            if record.get("command") != "cache":
                continue
            argv = " ".join(str(a) for a in record.get("argv", []))
            duration = record.get("duration_seconds")
            if duration is None:
                continue
            tag = (
                f"gpu-{argv.split('--shard-id')[1].split()[0]}"
                if "--shard-id" in argv
                else "shards"
            )
            receipts[tag] = {
                "start_utc": record.get("start_utc"),
                "end_utc": record.get("end_utc"),
                "duration_seconds": float(duration),
            }

    reconstructed: dict[str, Any] = {}
    for shard in shards:
        if shard in reports:
            continue
        entry: dict[str, Any] = {"source": "CLI receipt wall clock", "shard": shard}
        if shard in receipts:
            entry.update(receipts[shard])
        manifest = feature_dir / shard / "manifest.jsonl"
        if manifest.is_file():
            rows = list(read_jsonl(manifest))
            counts: dict[str, int] = {}
            for row in rows:
                counts[row["modality"]] = counts.get(row["modality"], 0) + 1
            entry["objects"] = len(rows)
            entry["objects_per_modality"] = counts
            if "duration_seconds" in entry and rows:
                entry["seconds_per_object"] = entry["duration_seconds"] / len(rows)
        reconstructed[shard] = entry
    return {
        "objects_per_modality": per_modality,
        "instrumented_shards": sorted(reports),
        "uninstrumented_shards": sorted(set(shards) - set(reports)),
        "shard_reports": reports,
        "reconstructed": reconstructed,
        "note": (
            "Shards under uninstrumented_shards were encoded before the per-phase "
            "instrumentation existed; their total cost comes from the CLI receipt's "
            "recorded start/end, so it is measured wall clock, not an estimate."
        ),
    }


def cmd_build_objects(args, spec, spec_dir, cwd, output_root, receipt) -> dict[str, Any]:
    """Enumerate the object set and the per-modality corpora from raw artifacts."""
    resolved = _resolved_or_raise(output_root)
    lake = _load_lake(resolved)
    gt = _load_gt(resolved)
    cache_config = resolved["cache"]
    objects, kinds = cache_module.enumerate_objects(
        lake,
        gt,
        table_max_rows=int(cache_config["table_max_rows"]),
        table_max_cell_chars=int(cache_config["table_max_cell_chars"]),
        table_row_format=str(cache_config["table_row_format"]),
    )
    records = cache_module.build_object_records(lake, gt, objects, cache_config)
    write_jsonl(output_root / "objects" / "objects.jsonl", records)

    corpus_map: dict[str, list[str]] = {
        "target": [],
        "query": [],
        "evidence": [],
        "evidence_text": [],
        "evidence_image": [],
        "all": [],
    }
    for entry in objects:
        object_id = entry["object_id"]
        kind = kinds[object_id]
        corpus_map["all"].append(object_id)
        if kind == "target":
            corpus_map["target"].append(object_id)
        elif kind == "query":
            corpus_map["query"].append(object_id)
        elif kind == "evidence_text":
            corpus_map["evidence"].append(object_id)
            corpus_map["evidence_text"].append(object_id)
        elif kind == "evidence_image":
            corpus_map["evidence"].append(object_id)
            corpus_map["evidence_image"].append(object_id)
    corpus_lines = []
    for name in ("target", "query", "evidence", "evidence_text", "evidence_image", "all"):
        ids = byte_order(corpus_map[name])
        corpus_lines.append({"corpus": name, "count": len(ids), "object_ids": ids})
    write_jsonl(output_root / "objects" / "corpus.jsonl", corpus_lines)

    summary = {
        "objects": len(objects),
        "corpora": {name: len(byte_order(corpus_map[name])) for name in corpus_map},
        "train_queries": len(gt["per_split"]["train"]["population"]),
        "dev_queries": len(gt["per_split"]["dev"]["population"]),
        "test_queries": len(gt["per_split"]["test"]["population"]),
    }
    write_json(output_root / "objects" / "SUMMARY.json", summary)
    log_line(
        f"build-objects: {summary['objects']} objects; corpora "
        + ", ".join(f"{k}={v}" for k, v in summary["corpora"].items())
    )
    return summary


def cmd_cache(args, spec, spec_dir, cwd, output_root, receipt) -> dict[str, Any]:
    """One-pass frozen encoding, optionally split across GPU shards."""
    if getattr(args, "merge_shards", False):
        return _merge_cache_shards(output_root, _resolved_or_raise(output_root))
    resolved = _resolved_or_raise(output_root)
    lake = _load_lake(resolved)
    gt = _load_gt(resolved)
    objects, kinds = cache_module.enumerate_objects(
        lake, gt,
        table_max_rows=int(resolved["cache"]["table_max_rows"]),
        table_max_cell_chars=int(resolved["cache"]["table_max_cell_chars"]),
        table_row_format=str(resolved["cache"]["table_row_format"]),
    )
    objects.sort(key=lambda r: r["object_id"].encode("utf-8"))
    object_total = len(objects)   # kept before any filtering below
    range_count = int(getattr(args, "range_count", 1) or 1)
    range_id = int(getattr(args, "range_id", 0) or 0)
    if args.shard_count > 1 or range_count > 1:
        objects = cache_module.shard_objects(
            objects,
            shard_id=args.shard_id,
            shard_count=args.shard_count,
            range_id=range_id,
            range_count=range_count,
            rank_offset=int(getattr(args, "rank_offset", 0) or 0),
            rank_limit=getattr(args, "rank_limit", None),
        )
        log_line(
            f"cache shard {args.shard_id}/{args.shard_count} "
            f"range {range_id}/{range_count}: {len(objects)} objects"
        )
    if getattr(args, "only_missing", False):
        # Encode exactly the objects no existing shard manifest covers.  This is
        # the safe way to resume after a repartition: without it a slice is
        # re-encoded from its start even when only a tail is absent, which is both
        # wasteful and the reason slice boundaries used to matter.
        covered: set[str] = set()
        for path in sorted((output_root / "cache").glob("gpu-*/manifest.jsonl")):
            if path.parent.name == _shard_tag(args, range_count):
                continue
            for row in read_jsonl(path):
                covered.add(str(row["object_id"]))
        before = len(objects)
        objects = [o for o in objects if o["object_id"] not in covered]
        log_line(
            f"cache only-missing: {before} in slice, {len(covered)} covered "
            f"elsewhere, {len(objects)} left to encode"
        )
    feature_dir = output_root / "cache"
    feature_dir.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(output_root).free
    reserve = int(resolved["cache"]["min_free_disk_gib"]) * (1024 ** 3)
    planned = len(objects) * int(resolved["cache"]["global_dimension"]) * (
        4 + int(resolved["cache"]["summary_slots"]) * 2
    )
    if free < planned + reserve:
        raise ConfigError(
            f"insufficient disk: free={free / 1024**3:.1f}GiB, planned features="
            f"{planned / 1024**3:.1f}GiB, required reserve="
            f"{reserve / 1024**3:.0f}GiB. Source of the byte count: "
            f"{len(objects)} objects x 4096 x (4 + 8*2) bytes (spec Eq. 9). "
            "Nothing is deleted automatically and no lake is shrunk."
        )
    limit = 16 if args.dry_run else args.limit_per_modality
    result = cache_module.build_cache(
        output_root=output_root,
        resolved=resolved,
        lake=lake,
        gt=gt,
        objects=objects,
        limit_per_modality=limit,
        limit_tables=args.limit_tables,
        limit_evidence=args.limit_evidence,
        device=args.device,
        timing=Timing(),
        tag=_shard_tag(args, range_count),
        object_total=object_total,
    )
    write_json(output_root / "cache" / f"CACHE_RECEIPT_{args.shard_id}.json", result)
    log_line(f"cache: {result}")
    return result



def _load_corpus(output_root: Path) -> dict[str, list[str]]:
    """The retrievable corpora, restricted to objects that actually have features.

    An object whose content could not be encoded -- empty text, an undecodable
    image, a decompression-bomb rejection -- is not retrievable and therefore
    cannot be in a corpus that is scored against (spec section 3).  Every dropped
    object is written to a registry and counted, never silently removed: its
    queries and targets stay in the evaluation denominator elsewhere.
    """
    path = output_root / "objects" / "corpus.jsonl"
    if not path.is_file():
        raise ConfigError(f"corpus is missing: {path}; run build-objects first")
    corpora: dict[str, list[str]] = {}
    for row in read_jsonl(path):
        corpora[str(row["corpus"])] = [str(v) for v in row["object_ids"]]

    manifest = output_root / "cache" / "manifest.jsonl"
    if not manifest.is_file():
        return corpora
    cached = set(cache_module.read_cache_index(output_root / "cache"))
    dropped: list[dict[str, Any]] = []
    filtered: dict[str, list[str]] = {}
    for name, ids in corpora.items():
        kept = []
        for object_id in ids:
            if object_id in cached:
                kept.append(object_id)
            else:
                dropped.append({"object_id": object_id, "corpus": name})
        filtered[name] = kept
    if dropped:
        write_jsonl(
            output_root / "objects" / "non_retrievable.jsonl",
            sorted(dropped, key=lambda r: r["object_id"].encode("utf-8")),
        )
        write_json(
            output_root / "objects" / "NON_RETRIEVABLE_SUMMARY.json",
            {
                # "evidence" and its per-modality views overlap, so the entry count
                # and the distinct-object count are reported separately.
                "dropped_entries": len(dropped),
                "dropped_objects": len({d["object_id"] for d in dropped}),
                "by_corpus": {
                    name: sum(1 for d in dropped if d["corpus"] == name)
                    for name in sorted(corpora)
                },
                "reason": (
                    "no cached features: the object's content could not be encoded "
                    "(empty text, undecodable image, or a rejected oversized image)"
                ),
                "effect": (
                    "excluded from the retrievable corpus only. Queries and targets "
                    "that reference them remain in the evaluation denominator, and no "
                    "target is removed from a Recall denominator."
                ),
            },
        )
    # The filtered corpora must now agree with the cache exactly.
    for name in ("target", "query", "evidence_text", "evidence_image"):
        unknown = [o for o in filtered.get(name, []) if o not in cached]
        if unknown:
            raise ConfigError(
                f"{name} still references {len(unknown)} uncached objects, e.g. "
                f"{unknown[:5]}; the filter above must remove every one"
            )
    filtered["all"] = (
        filtered.get("query", [])
        + filtered.get("target", [])
        + filtered.get("evidence_text", [])
        + filtered.get("evidence_image", [])
    )
    return filtered


def _load_bank(output_root: Path, resolved: dict[str, Any], corpora: dict[str, list[str]]):
    manifest_path = output_root / "cache" / "manifest.jsonl"
    if not manifest_path.is_file():
        raise ConfigError(f"cache manifest is missing: {manifest_path}; run cache first")
    index = cache_module.read_cache_index(output_root / "cache")
    expected = set(corpora["all"])
    if set(index) != expected:
        extra = sorted(set(index) - expected)[:5]
        missing = sorted(expected - set(index))[:5]
        raise ConfigError(
            "cache manifest does not match the enumerated object set "
            f"(cached={len(index)}, expected={len(expected)}, extra={extra}, "
            f"missing={missing}). Differences are reported rather than zero-filled; "
            "a partial dry-run cache cannot be used for retrieval or training."
        )
    # Rows may carry more than one fingerprint value when the run repartitioned
    # its encoding work: an older revision of the formula folded in the slice size,
    # so identical encodings can hash differently.  The encoding itself is verified
    # per object instead -- every row must agree on modality, kind layout and mask
    # structure, which is what actually makes the features interchangeable.
    identity = verify_cache_identity(list(index.values()))
    if not identity["consistent"]:
        raise ConfigError(
            "cache manifest rows do not describe one encoding: "
            + json.dumps(
                {
                    "problem_count": identity["problem_count"],
                    "problems": identity["problems"][:5],
                }
            )
        )
    fingerprints = set(identity["distinct_fingerprints"])
    object_ids = list(corpora["all"])
    rows = [index[o] for o in object_ids]
    dim = int(resolved["cache"]["global_dimension"])
    slots = int(resolved["cache"]["summary_slots"])
    # Returns ids, not manifest rows: every caller indexes by object id, and the
    # bank's row order is exactly ``object_ids``.
    return cache_module.load_shard_vectors(
        output_root / "cache", rows, dim, slots
    ), object_ids, next(iter(fingerprints))


def cmd_raw_retrieve(args, spec, spec_dir, cwd, output_root, receipt) -> dict[str, Any]:
    """Spec 7.1: label-free exact inner-product rankings from the frozen z vectors."""
    resolved = _resolved_or_raise(output_root)
    corpora = _load_corpus(output_root)
    (z, _summary, _mask), object_ids, _fingerprint = _load_bank(output_root, resolved, corpora)
    position = {object_id: i for i, object_id in enumerate(object_ids)}

    target_ids = corpora["target"]
    text_ids = corpora["evidence_text"]
    image_ids = corpora["evidence_image"]
    target_z = z[[position[o] for o in target_ids]]
    text_z = z[[position[o] for o in text_ids]]
    image_z = z[[position[o] for o in image_ids]]

    split = args.split
    population = list(read_jsonl(output_root / "raw_gt" / f"{split}.population.jsonl"))
    query_ids = [str(row["query_id"]) for row in population]
    if not query_ids:
        raise ConfigError(f"split {split} has no queries in the population")
    query_z = z[[position[q] for q in query_ids]]

    retrieval = resolved["retrieval"]
    pool = int(retrieval.get("train_target_pool", 128))
    evidence_pool = int(retrieval.get("train_evidence_pool", 128))
    second_hop_pool = int(retrieval.get("train_second_hop_pool", 128))
    qb = int(retrieval["exact_query_batch"])
    cc = int(retrieval["exact_corpus_chunk"])

    started = time.time()
    d_order, d_score = retrieve.exact_topk(
        query_z, target_z, pool, query_batch=qb, corpus_chunk=cc
    )
    log_line(f"raw-retrieve {split}: direct ranking done ({time.time() - started:.0f}s)")
    t_order, t_score = retrieve.exact_topk(
        query_z, text_z, evidence_pool, query_batch=qb, corpus_chunk=cc
    )
    i_order, i_score = retrieve.exact_topk(
        query_z, image_z, evidence_pool, query_batch=qb, corpus_chunk=cc
    )
    log_line(f"raw-retrieve {split}: evidence ranking done ({time.time() - started:.0f}s)")

    anchor_ids: list[str] = []
    if split == "train":
        # Every canonical witness is a possible C-packet anchor (spec 7.1), so its
        # own z_E -> T ranking is precomputed here rather than during training.
        seen: set[str] = set()
        for row in population:
            for values in row["witnesses"].values():
                for value in values:
                    if value not in seen and value in position:
                        seen.add(value)
                        anchor_ids.append(value)
        anchor_ids.sort(key=lambda v: v.encode("utf-8"))
        anchor_z = z[[position[a] for a in anchor_ids]]
        a_order, a_score = retrieve.exact_topk(
            anchor_z, target_z, second_hop_pool, query_batch=qb, corpus_chunk=cc
        )
        log_line(f"raw-retrieve {split}: witness second hop done ({time.time() - started:.0f}s)")
    else:
        a_order = np.zeros((0, 0), dtype=np.int64)
        a_score = np.zeros((0, 0), dtype=np.float32)

    out_dir = output_root / "raw_retrieval" / split
    out_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for row_index, query_id in enumerate(query_ids):
        records.append(
            {
                "query_id": query_id,
                "split": split,
                "target_ids": [target_ids[i] for i in d_order[row_index]],
                "target_scores": [float(v) for v in d_score[row_index]],
                "text_ids": [text_ids[i] for i in t_order[row_index]],
                "text_scores": [float(v) for v in t_score[row_index]],
                "image_ids": [image_ids[i] for i in i_order[row_index]],
                "image_scores": [float(v) for v in i_score[row_index]],
            }
        )
    write_jsonl(out_dir / "query_rankings.jsonl", records)
    write_jsonl(
        out_dir / "anchor_rankings.jsonl",
        (
            {
                "anchor_id": anchor_id,
                "target_ids": [target_ids[i] for i in a_order[i]],
                "target_scores": [float(v) for v in a_score[i]],
            }
            for i, anchor_id in enumerate(anchor_ids)
        ),
    )
    meta = {
        "split": split,
        "queries": len(query_ids),
        "target_pool": pool,
        "evidence_pool": evidence_pool,
        "second_hop_pool": second_hop_pool,
        "anchor_count": len(anchor_ids),
        "corpus_sizes": {
            "target": len(target_ids),
            "text": len(text_ids),
            "image": len(image_ids),
        },
        "elapsed_seconds": time.time() - started,
        "z_source": "frozen Qwen global vectors, float32 exact inner product",
    }
    write_json(out_dir / "meta.json", meta)
    log_line(f"raw-retrieve {split}: {meta}")
    return meta


def cmd_build_supervision(args, spec, spec_dir, cwd, output_root, receipt) -> dict[str, Any]:
    """Spec 7: materialise the GT-derived supervision tables and witness coverage.

    Only labels come from here: the E/P/D/C/B positive sets, the C-packet masks,
    and the per-epoch witness cycle.  Candidate competitors are drawn during
    training from the run's own mining ranks, never stored as a fixed list.
    """
    resolved = _resolved_or_raise(output_root)
    corpus = _load_corpus(output_root)
    gt = {split: {} for split in ("train", "dev", "test")}
    for split in ("train", "dev", "test"):
        rows = list(read_jsonl(output_root / "raw_gt" / f"{split}.population.jsonl"))
        gt[split]["population"] = rows
    builder = train.PacketBuilder(gt=gt, corpora=corpus)
    epochs = int(resolved["teacher"]["epochs"])
    bundle_epochs = {int(e) for e in resolved["teacher"]["bundle_epochs"]}
    per_modality = int(resolved["retrieval"]["evidence_per_modality"])

    summary: dict[str, Any] = {}
    for split in ("train", "dev", "test"):
        rows = builder.population(split)
        records = []
        witness_hits: dict[str, set[int]] = {}
        for row in rows:
            query_id = row["query_id"]
            direct, implicit, all_positive = builder.positive_sets(row)
            witnesses = builder.witnesses(row)
            witness_union = builder.witness_union(row)
            epochs_record = []
            for epoch in range(1, epochs + 1):
                anchor = sampling.select_witness_anchor(witness_union, query_id, epoch)
                if anchor is not None:
                    witness_hits.setdefault(anchor, set()).add(epoch)
                c_positives = sorted(
                    sampling.support_positive_set(
                        direct=direct, implicit=implicit, witnesses=witnesses,
                        context={anchor} if anchor else set(),
                    )
                )
                entry = {
                    "epoch": epoch,
                    "witness_anchor": anchor,
                    "C_positive_ids": c_positives,
                    "C_excluded_ids": sorted(all_positive - set(c_positives)),
                }
                if epoch in bundle_epochs:
                    entry["B_view"] = (
                        "natural"
                        if sampling.use_natural_bundle(epoch, query_id)
                        else "witness_augmented"
                    )
                epochs_record.append(entry)
            records.append(
                {
                    "query_id": query_id,
                    "split": split,
                    "source_table_id": row["source_table_id"],
                    "query_kind": row["query_kind"],
                    "G_Q": sorted(all_positive),
                    "D_Q": sorted(direct),
                    "I_Q": sorted(implicit),
                    "W_Q": witness_union,
                    "W_by_target": {k: sorted(v) for k, v in sorted(witnesses.items())},
                    "epochs": epochs_record,
                }
            )
        write_jsonl(output_root / "raw_gt" / f"{split}.supervision.jsonl", records)

        sampled = {w: sorted(v) for w, v in sorted(witness_hits.items())}
        total_pairs = sum(len(witnesses) for row in rows for witnesses in [builder.witnesses(row)])
        ever_sampled = len(sampled)
        summary[split] = {
            "queries": len(rows),
            "witness_targets_total": total_pairs,
            "distinct_witness_evidence": len({w for row in rows for w in builder.witness_union(row)}),
            "witness_evidence_sampled_in_six_epochs": ever_sampled,
            "witness_sampling_coverage": (
                ever_sampled / max(1, len({w for row in rows for w in builder.witness_union(row)}))
            ),
            "epochs_per_query": epochs,
            "bundle_epochs": sorted(bundle_epochs),
        }
        if split == "train":
            write_json(
                output_root / "raw_gt" / "witness_sampling_coverage.json",
                {
                    "note": (
                        "The C packet samples exactly one witness anchor per query per "
                        "epoch (spec 7.3). A single six-epoch run therefore never "
                        "enumerates every (Q,T,E) triple; this file reports the realised "
                        "coverage rather than claiming a full pass."
                    ),
                    "coverage": sampled,
                    "summary": summary[split],
                },
            )
    write_json(output_root / "raw_gt" / "SUPERVISION_SUMMARY.json", summary)
    log_line(f"build-supervision: {summary}")
    return summary


# --------------------------------------------------------------------------
# shared training-side helpers
# --------------------------------------------------------------------------


def _load_rankings(output_root: Path, split: str) -> dict[str, dict[str, Any]]:
    path = output_root / "raw_retrieval" / split / "query_rankings.jsonl"
    if not path.is_file():
        raise ConfigError(f"raw retrieval is missing for {split}: {path}")
    return {str(row["query_id"]): row for row in read_jsonl(path)}


def _load_rank_tables(
    output_root: Path, split: str, retrieval: dict[str, Any]
) -> tuple[dict[str, train.RankTable], dict[str, list[tuple[str, float]]]]:
    rankings = _load_rankings(output_root, split)
    pool = int(retrieval.get("train_target_pool", 128))
    evidence_pool = int(retrieval.get("train_evidence_pool", 128))
    tables = {
        "D": train.RankTable(
            {q: list(zip(r["target_ids"][:pool], r["target_scores"][:pool])) for q, r in rankings.items()}
        ),
        "E_text": train.RankTable(
            {q: list(zip(r["text_ids"][:evidence_pool], r["text_scores"][:evidence_pool])) for q, r in rankings.items()}
        ),
        "E_image": train.RankTable(
            {q: list(zip(r["image_ids"][:evidence_pool], r["image_scores"][:evidence_pool])) for q, r in rankings.items()}
        ),
    }
    anchor_path = output_root / "raw_retrieval" / split / "anchor_rankings.jsonl"
    anchor_rank: dict[str, list[tuple[str, float]]] = {}
    if anchor_path.is_file():
        second_hop = int(retrieval.get("train_second_hop_pool", 128))
        for row in read_jsonl(anchor_path):
            anchor_rank[str(row["anchor_id"])] = list(
                zip(row["target_ids"][:second_hop], row["target_scores"][:second_hop])
            )
    return tables, anchor_rank


def _load_bank_objects(output_root: Path, resolved: dict[str, Any]):
    corpora = _load_corpus(output_root)
    (z, summary, mask), object_ids, _fingerprint = _load_bank(output_root, resolved, corpora)
    manifest = cache_module.read_cache_index(output_root / "cache")
    modality_id = np.array([int(manifest[o]["modality_id"]) for o in object_ids], dtype=np.int64)
    kind_ids = np.array([list(manifest[o]["kind_ids"]) for o in object_ids], dtype=np.int64)
    bank = models.ObjectBank(object_ids, modality_id, kind_ids, z, summary, mask)
    return bank, corpora


def _make_builder(output_root: Path, corpora: dict[str, list[str]]) -> train.PacketBuilder:
    gt: dict[str, Any] = {}
    for split in ("train", "dev", "test"):
        path = output_root / "raw_gt" / f"{split}.population.jsonl"
        population = list(read_jsonl(path)) if path.is_file() else []
        gt[split] = {"population": population}
    return train.PacketBuilder(gt=gt, corpora=corpora)


def _freeze(module: torch.nn.Module) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    module.eval()


def _model_hash(module: torch.nn.Module) -> str:
    return stable_digest(
        {name: [float(v) for v in tensor.detach().float().flatten()[:8]] for name, tensor in module.state_dict().items()},
        len(list(module.state_dict())),
    )


def _save_checkpoint(path: Path, **payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def _load_teacher(path: Path, resolved: dict[str, Any], device: str):
    teacher = build_teacher(resolved["teacher"]).to(device)
    state = torch.load(path, map_location=device, weights_only=False)
    teacher.load_state_dict(state["model"])
    _freeze(teacher)
    return teacher, state


def cmd_train_teacher(args, spec, spec_dir, cwd, output_root, receipt) -> dict[str, Any]:
    """Spec 8.3: six epochs, one optimizer, one in-run refresh after epoch 2."""
    resolved = _resolved_or_raise(output_root)
    bank, corpora = _load_bank_objects(output_root, resolved)
    builder = _make_builder(output_root, corpora)
    rank_tables, anchor_rank = _load_rank_tables(output_root, "train", resolved["retrieval"])
    teacher_dir = output_root / "teacher"
    teacher_dir.mkdir(parents=True, exist_ok=True)
    set_seed(int(resolved["seed"]))
    trainer = train.TeacherTrainer(
        resolved=resolved,
        bank=bank,
        builder=builder,
        rank_tables=rank_tables,
        anchor_rank=anchor_rank,
        output_dir=teacher_dir,
        device=args.device,
        limit_queries=args.limit_queries,
        max_epochs=args.max_epochs,
        prefetch_workers=getattr(args, "prefetch_workers", 1),
    )
    _save_checkpoint(teacher_dir / "init.pt", model=trainer.teacher.state_dict(), epoch=0)
    result = trainer.train()
    _save_checkpoint(
        teacher_dir / "last.pt",
        model=trainer.teacher.state_dict(),
        epoch=trainer.epochs,
        steps=result["steps"],
        history=result["epochs"],
        protocol=resolved["protocol_id"],
    )
    write_json(teacher_dir / "training.jsonl", {"history": result["epochs"]})
    write_jsonl(teacher_dir / "epoch_summaries.jsonl", result["epochs"])
    return result


def _teacher_dev_ranker(
    *,
    output_root: Path,
    resolved: dict[str, Any],
    bank: models.ObjectBank,
    teacher_hash: str,
    logit_cache: evaluate.TeacherLogitCache,
    device: str,
    tag: str,
):
    """Teacher-only dev scoring on the fixed raw C100 / raw B20 candidate set.

    The candidate set is the *raw Qwen* retrieval, fixed before training, so the
    epoch comparison isolates the Teacher instead of drifting with a retriever.
    """

    def ranker(teacher, batch, records) -> dict[str, Any]:
        chunk = int(resolved["retrieval"]["teacher_target_chunk"])
        ks = [int(k) for k in resolved["evaluation"]["ks"]]
        per_query = []
        detail = []
        for record in records:
            query_id = record["query_id"]
            c100 = record["C100"]
            bundle = record["B_Q"]
            scores = score_selection(
                teacher=teacher, batch=batch, cache=logit_cache,
                teacher_hash=teacher_hash, mode="J", query_id=query_id,
                candidates=c100, context=bundle, chunk=chunk,
            )
            ranked = rank_by_scores(c100, scores)
            metrics = evaluate.query_metrics(
                ranked, record["direct_target_ids"], record["implicit_target_ids"], ks
            )
            per_query.append(metrics)
            detail.append({"query_id": query_id, "Top50": ranked[:50], "metrics": metrics})
        macro = evaluate.macro_metrics(per_query, ks)
        return {
            "overall_R10": macro["overall_R10"],
            "implicit_R10": macro["implicit_R10"],
            "overall_R20": macro["overall_R20"],
            "explicit_R10": macro["explicit_R10"],
            "overall_R50": macro["overall_R50"],
            "metrics": macro,
            "per_query": detail,
            "queries": len(per_query),
            "tag": tag,
        }

    return ranker


def cmd_freeze_teacher(args, spec, spec_dir, cwd, output_root, receipt) -> dict[str, Any]:
    """Spec 8.3: choose the best epoch on dev by the pre-registered key."""
    resolved = _resolved_or_raise(output_root)
    teacher_dir = output_root / "teacher"
    epochs = int(resolved["teacher"]["epochs"])
    population = _population_index(output_root, "dev")
    dev_records = list(
        read_jsonl(output_root / "raw_retrieval" / "dev" / "query_rankings.jsonl")
    )
    if not dev_records:
        raise ConfigError("dev raw retrieval is missing; run raw-retrieve --split dev")
    # Dev candidates are the RAW Qwen C100/B20, fixed before any training.
    retrieval = resolved["retrieval"]
    bank, _corpora = _load_bank_objects(output_root, resolved)
    per_modality = int(retrieval["evidence_per_modality"])
    fixed: list[dict[str, Any]] = []
    for row in dev_records:
        query_id = str(row["query_id"])
        meta = population.get(query_id)
        if meta is None:
            continue
        direct = row["target_ids"][: int(retrieval["direct_k"])]
        bundle = retrieve.interleave_text_image(
            row["text_ids"][:per_modality], row["image_ids"][:per_modality],
            per_modality, per_modality * 2,
        )
        fixed.append(
            {
                "query_id": query_id,
                "C100": direct,
                "B_Q": bundle,
                "direct_target_ids": meta["direct_target_ids"],
                "implicit_target_ids": meta["implicit_target_ids"],
                "positive_target_ids": meta["positive_target_ids"],
            }
        )
    cache = evaluate.TeacherLogitCache(output_root / "teacher_logits.sqlite")
    candidates: list[dict[str, Any]] = []
    for epoch in range(1, epochs + 1):
        epoch_path = teacher_dir / f"epoch_{epoch:02d}.pt"
        if not epoch_path.is_file():
            raise ConfigError(
                f"{epoch_path} is missing; every trained epoch must be selectable"
            )
        teacher = build_teacher(resolved["teacher"]).to(args.device)
        teacher.load_state_dict(
            torch.load(epoch_path, map_location=args.device, weights_only=False)["model"]
        )
        _freeze(teacher)
        teacher_hash = _model_hash(teacher)
        batch = train.TeacherBatch(bank, device=args.device)
        ranker = _teacher_dev_ranker(
            output_root=output_root, resolved=resolved, bank=bank,
            teacher_hash=teacher_hash, logit_cache=cache, device=args.device,
            tag=f"T_epoch{epoch:02d}",
        )
        evaluation = ranker(teacher, batch, fixed)
        summary = {
            "epoch": epoch,
            "overall_R10": evaluation["overall_R10"],
            "implicit_R10": evaluation["implicit_R10"],
            "overall_R20": evaluation["overall_R20"],
            "explicit_R10": evaluation["explicit_R10"],
            "overall_R50": evaluation["overall_R50"],
            "teacher_hash": teacher_hash,
            "queries": evaluation["queries"],
        }
        candidates.append(summary)
        write_json(teacher_dir / f"dev_epoch_{epoch:02d}.json", {**summary, "metrics": evaluation["metrics"]})
        write_jsonl(
            teacher_dir / f"dev_epoch_{epoch:02d}.rankings.jsonl", evaluation["per_query"]
        )
        log_line(
            f"freeze-teacher: epoch {epoch} dev R@10={summary['overall_R10']} "
            f"implicit={summary['implicit_R10']} R@20={summary['overall_R20']}"
        )
        del teacher, batch
        torch.cuda.empty_cache()
    best = max(
        candidates,
        key=lambda c: (
            _value(c["overall_R10"]), _value(c["implicit_R10"]),
            _value(c["overall_R20"]), -c["epoch"],
        ),
    )
    state = torch.load(
        teacher_dir / f"epoch_{best['epoch']:02d}.pt", map_location="cpu", weights_only=False
    )
    _save_checkpoint(
        teacher_dir / "best.pt",
        model=state["model"],
        epoch=best["epoch"],
        selection={
            "key": "overall_R10 -> implicit_R10 -> overall_R20 -> earlier epoch",
            "candidates": candidates,
        },
    )
    selection = {
        "selected_epoch": best["epoch"],
        "selected": best,
        "candidates": candidates,
        "key": "overall_R10 -> implicit_R10 -> overall_R20 -> earlier epoch",
        "dev_candidates": "raw Qwen C100 with raw Qwen B20, fixed before training",
        "teacher_hash": best["teacher_hash"],
        "note": (
            "the frozen deployment checkpoint is best.pt; last.pt is retained for "
            "audit only and must never be substituted for best"
        ),
    }
    write_json(teacher_dir / "selection.json", selection)
    cache.close()
    log_line(f"freeze-teacher: selected epoch {best['epoch']}")
    return selection


def _value(raw: Any) -> float:
    """A missing class is N/A, not a zero: it sorts below every real value."""
    return float("-inf") if raw is None else float(raw)


def _select_epoch(student_dir: Path, epochs: int, arm: str) -> dict[str, Any]:
    """Spec 9.3: apply the pre-registered lexicographic dev key over the epochs."""
    candidates: list[dict[str, Any]] = []
    for epoch in range(1, epochs + 1):
        path = student_dir / f"epoch_{epoch:02d}.dev.json"
        if not path.is_file():
            continue
        record = json.loads(path.read_text(encoding="utf-8"))
        candidates.append(
            {
                "epoch": epoch,
                "overall_R10": record.get("overall_R10"),
                "implicit_R10": record.get("implicit_R10"),
                "overall_R20": record.get("overall_R20"),
            }
        )
    if not candidates:
        raise ConfigError(
            f"{student_dir}: no per-epoch dev records exist, so the pre-registered "
            "selection cannot be applied"
        )
    best = max(
        candidates,
        key=lambda c: (
            _value(c["overall_R10"]),
            _value(c["implicit_R10"]),
            _value(c["overall_R20"]),
            -c["epoch"],
        ),
    )
    return {
        "arm": arm,
        "selected_epoch": best["epoch"],
        "candidates": candidates,
        "key": "overall_R10 -> implicit_R10 -> overall_R20 -> earlier epoch",
        "note": (
            "next-epoch mining is driven by the last parameters; deployment uses the "
            "epoch selected here. The two must not be conflated."
        ),
    }


def _dev_ranker_factory(
    *,
    output_root: Path,
    resolved: dict[str, Any],
    bank: models.ObjectBank,
    corpora: dict[str, list[str]],
    teacher,
    teacher_hash: str,
    teacher_batch: train.TeacherBatch,
    logit_cache: evaluate.TeacherLogitCache,
    device: str,
    max_queries: int,
    tag: str,
):
    """Per-epoch dev scoring: the Student's own natural retrieval, the frozen T's J.

    The dev population is hashed down to a fixed subset (spec 12.5) and the same
    subset is reused for every epoch, so an epoch can never be compared on a
    more favourable slice than another.
    """

    def ranker(student, keys) -> dict[str, Any]:
        engine = evaluate.RetrievalEngine(
            student=student, bank=bank, corpora=corpora, retrieval=resolved["retrieval"],
            ann=resolved["ann"], seed=int(resolved["seed"]), device=device,
            build_ann=True, timing=Timing(),
        )
        summary = run_split(
            output_root=output_root, resolved=resolved, split="dev", teacher=teacher,
            teacher_hash=teacher_hash, teacher_batch=teacher_batch, engine=engine,
            method=tag, logit_cache=logit_cache, max_queries=max_queries,
            write_artifacts=False, run_exact=False,
        )
        view = summary["views"]["ann"]["teacher_reranked"]
        del engine
        torch.cuda.empty_cache()
        return {
            "overall_R10": view["overall_R10"],
            "implicit_R10": view["implicit_R10"],
            "overall_R20": view["overall_R20"],
            "explicit_R10": view["explicit_R10"],
            "overall_R50": view["overall_R50"],
            "implicit_queries": view.get("implicit_R10_queries"),
            "candidate_recall": summary["views"]["ann"]["candidate_recall"],
            "nn_fidelity": summary["nn_fidelity"],
            "queries": summary["queries"],
        }

    return ranker


def cmd_train_student(args, spec, spec_dir, cwd, output_root, receipt) -> dict[str, Any]:
    """Spec 9: one Student arm, fresh init, per-epoch own-model mining refresh."""
    resolved = _resolved_or_raise(output_root)
    arm = args.arm
    student_dir = output_root / f"student_{arm}"
    student_dir.mkdir(parents=True, exist_ok=True)
    bank, corpora = _load_bank_objects(output_root, resolved)
    builder = _make_builder(output_root, corpora)
    rank_tables, anchor_rank = _load_rank_tables(output_root, "train", resolved["retrieval"])

    init_path = output_root / "student_init.pt"
    if not init_path.is_file():
        set_seed(int(resolved["seed"]))
        _save_checkpoint(
            init_path,
            model=build_student(resolved["student"]).state_dict(),
            seed=int(resolved["seed"]),
            note="one shared fresh initialisation; both arms load separate copies",
        )
    init_state = torch.load(init_path, map_location="cpu", weights_only=False)["model"]

    # Both arms rank their dev candidates through the same frozen Teacher, because
    # the pre-registered best-epoch key is a Teacher-scored recall and must mean the
    # same thing for the control arm as for the distilled one.  Only S-KD also uses
    # the Teacher as a training signal.
    teacher_for_selection = None
    teacher_for_training = None
    teacher_batch = None
    teacher_hash = "none"
    best_path = output_root / "teacher" / "best.pt"
    if not best_path.is_file():
        raise ConfigError(
            f"{best_path} is missing; the frozen Teacher must exist before any "
            "Student arm, since the dev selection is scored by it. A historical "
            "checkpoint is never an acceptable substitute."
        )
    teacher_for_selection, _state = _load_teacher(best_path, resolved, args.device)
    teacher_hash = _model_hash(teacher_for_selection)
    teacher_batch = train.TeacherBatch(bank, device=args.device)
    if arm == "KD":
        teacher_for_training = teacher_for_selection

    # A fresh trajectory: seed reset here, never inherited from the Teacher's RNG.
    set_seed(int(resolved["seed"]))
    trainer = train.StudentTrainer(
        resolved=resolved, bank=bank, builder=builder, rank_tables=rank_tables,
        anchor_rank=anchor_rank, output_dir=student_dir, arm=arm, init_state=init_state,
        device=args.device, limit_queries=args.limit_queries, max_epochs=args.max_epochs,
    )
    _save_checkpoint(student_dir / "init.pt", model=trainer.student.state_dict(), arm=arm)

    logit_cache = evaluate.TeacherLogitCache(output_root / "teacher_logits.sqlite")
    ranker = _dev_ranker_factory(
        output_root=output_root, resolved=resolved, bank=bank, corpora=corpora,
        teacher=teacher_for_selection, teacher_hash=teacher_hash,
        teacher_batch=teacher_batch, logit_cache=logit_cache, device=args.device,
        max_queries=int(resolved["evaluation"]["dev_health_queries"]),
        tag=f"{arm}-epoch",
    )
    result = trainer.train(
        teacher=teacher_for_training, teacher_batch=teacher_batch,
        query_rankings={}, dev_ranker=ranker,
    )
    _save_checkpoint(
        student_dir / "last.pt", model=trainer.student.state_dict(), arm=arm,
        epochs=trainer.epochs, history=trainer.history, protocol=resolved["protocol_id"],
    )
    write_jsonl(student_dir / "training.jsonl", trainer.history)
    selection = _select_epoch(student_dir, trainer.epochs, arm)
    best_epoch_path = student_dir / f"epoch_{selection['selected_epoch']:02d}.pt"
    if not best_epoch_path.is_file():
        raise ConfigError(f"selected epoch checkpoint is missing: {best_epoch_path}")
    state = torch.load(best_epoch_path, map_location="cpu", weights_only=False)
    _save_checkpoint(
        student_dir / "best.pt", model=state["model"], arm=arm,
        epoch=selection["selected_epoch"], selection=selection,
    )
    write_json(student_dir / "selection.json", selection)
    logit_cache.close()
    return {"arm": arm, "selection": selection, "epochs": trainer.history, "steps": result["steps"]}


def cmd_freeze_selection(args, spec, spec_dir, cwd, output_root, receipt) -> dict[str, Any]:
    """Spec 11: lock every pre-registered choice before any test GT is read."""
    resolved = _resolved_or_raise(output_root)
    lock: dict[str, Any] = {
        "protocol_id": resolved["protocol_id"],
        "frozen_utc": utcnow(),
        "seed": int(resolved["seed"]),
        "selection_key": "overall_R10 -> implicit_R10 -> overall_R20 -> earlier epoch",
        "main_method": str(resolved["evaluation"]["main_method"]),
        "primary_comparisons": list(resolved["evaluation"]["primary_comparisons"]),
        "methods": list(resolved["evaluation"]["formal_methods"]),
        "spec_config_sha256": resolved["spec_config_sha256"],
        "resolved_config_sha256": resolved["resolved_config_sha256"],
        "semantic_hashes": resolved.get("semantic_hashes"),
        "models": {},
    }
    teacher_path = output_root / "teacher" / "best.pt"
    if not teacher_path.is_file():
        raise ConfigError(f"frozen Teacher is missing: {teacher_path}")
    teacher_selection = json.loads(
        (output_root / "teacher" / "selection.json").read_text(encoding="utf-8")
    )
    teacher = build_teacher(resolved["teacher"])
    teacher.load_state_dict(
        torch.load(teacher_path, map_location="cpu", weights_only=False)["model"]
    )
    lock["models"]["teacher"] = {
        "path": teacher_path.relative_to(output_root).as_posix(),
        "sha256": sha256_path(teacher_path),
        "parameter_hash": _model_hash(teacher),
        "parameters": int(sum(p.numel() for p in teacher.parameters())),
        "selected_epoch": teacher_selection.get("selected_epoch"),
        "dev_candidates": teacher_selection.get("candidates"),
    }
    arms: dict[str, Any] = {}
    init_states: dict[str, dict[str, Any]] = {}
    for arm in ("SUP", "KD"):
        path = output_root / f"student_{arm}" / "best.pt"
        init = output_root / f"student_{arm}" / "init.pt"
        if not path.is_file():
            raise ConfigError(f"frozen Student arm is missing: {path}")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        student = build_student(resolved["student"])
        student.load_state_dict(payload["model"])
        arms[arm] = {
            "path": path.relative_to(output_root).as_posix(),
            "sha256": sha256_path(path),
            "parameter_hash": _model_hash(student),
            "epoch": payload.get("epoch"),
            "parameters": int(sum(p.numel() for p in student.parameters())),
        }
        if init.is_file():
            init_states[arm] = torch.load(init, map_location="cpu", weights_only=False)["model"]
    lock["models"]["students"] = arms
    init_path = output_root / "student_init.pt"
    if init_path.is_file():
        lock["models"]["student_init"] = {
            "path": init_path.relative_to(output_root).as_posix(),
            "sha256": sha256_path(init_path),
        }
    if len(init_states) == 2:
        sup, kd = init_states["SUP"], init_states["KD"]
        same = sorted(sup) == sorted(kd) and all(
            torch.equal(sup[key], kd[key]) for key in sup
        )
        lock["shared_student_initialisation"] = bool(same)
        if not same:
            raise ConfigError(
                "S-SUP and S-KD did not start from identical parameters; the two arms "
                "are not the pre-registered contrast and the comparison is void"
            )
    write_json(output_root / "FINAL_SELECTION_LOCK.json", lock)
    log_line(f"freeze-selection: locked teacher + {sorted(arms)}")
    return lock


def cmd_evaluate(args, spec, spec_dir, cwd, output_root, receipt) -> dict[str, Any]:
    """Spec 12: the formal comparison table, ANN and exact, on a frozen split."""
    resolved = _resolved_or_raise(output_root)
    split = args.split
    if split == "test" and not (output_root / "FINAL_SELECTION_LOCK.json").is_file():
        raise ConfigError(
            "FINAL_SELECTION_LOCK.json is missing; models and budgets must be frozen "
            "before test GT is read"
        )
    requested = None
    if getattr(args, "methods", None):
        requested = {m.strip() for m in str(args.methods).split(",") if m.strip()}
        known = set(resolved["evaluation"]["formal_methods"])
        unknown = requested - known
        if unknown:
            raise ConfigError(
                f"unknown method(s) {sorted(unknown)}; the formal table is {sorted(known)}"
            )

    def wanted(method: str) -> bool:
        return requested is None or method in requested

    bank, corpora = _load_bank_objects(output_root, resolved)
    # The Teacher is only loaded when a requested method actually scores with it.
    # RAW-D and RAW-2H re-rank nothing, so a run of just those methods needs no
    # Teacher at all -- which is what lets them proceed independently of it.
    teacher_hash = "not_loaded"
    teacher = teacher_batch = cache = None
    if wanted("RAW+T") or wanted("SUP+T") or wanted("KD+T"):
        teacher, _state = _load_teacher(
            output_root / "teacher" / "best.pt", resolved, args.device
        )
        teacher_hash = _model_hash(teacher)
        teacher_batch = train.TeacherBatch(bank, device=args.device)
        cache = evaluate.TeacherLogitCache(output_root / "teacher_logits.sqlite")
    run_exact = split == "test"
    summary: dict[str, Any] = {
        "split": split,
        "started_utc": utcnow(),
        "teacher_hash": teacher_hash,
        "methods": {},
        "run_exact": run_exact,
        "requested_methods": sorted(requested) if requested else "all",
    }

    # The raw index is 172 s of HNSW construction, so it is only built when a raw
    # method is actually part of this run.
    need_raw = wanted("RAW-D") or wanted("RAW-2H") or wanted("RAW+T")
    raw_engine = (
        evaluate.RawEngine(
            bank=bank, corpora=corpora, retrieval=resolved["retrieval"], ann=resolved["ann"],
            seed=int(resolved["seed"]), device=args.device, build_ann=True,
        )
        if need_raw else None
    )
    for method in ("RAW-D", "RAW-2H"):
        if not wanted(method):
            continue
        summary["methods"][method] = _run_raw_method(
            output_root=output_root, resolved=resolved, split=split, method=method,
            raw_engine=raw_engine, bank=bank, cache=cache, run_exact=run_exact,
        )
        log_line(f"evaluate {split} {method}: {_headline(summary['methods'][method])}")
    if wanted("RAW+T"):
        summary["methods"]["RAW+T"] = _run_teacher_on_raw(
            output_root=output_root, resolved=resolved, split=split, bank=bank,
            raw_engine=raw_engine, teacher=teacher, teacher_hash=teacher_hash,
            teacher_batch=teacher_batch, cache=cache, run_exact=run_exact,
        )
        log_line(f"evaluate {split} RAW+T: {_headline(summary['methods']['RAW+T'])}")
    if raw_engine is not None:
        del raw_engine
        torch.cuda.empty_cache()

    lock = json.loads((output_root / "FINAL_SELECTION_LOCK.json").read_text(encoding="utf-8"))
    for arm in ("SUP", "KD"):
        method = f"{arm}+T"
        if not wanted(method):
            continue
        path = output_root / f"student_{arm}" / "best.pt"
        payload = torch.load(path, map_location="cpu", weights_only=False)
        student = build_student(resolved["student"])
        student.load_state_dict(payload["model"])
        student.to(args.device)
        _freeze(student)
        engine = evaluate.RetrievalEngine(
            student=student, bank=bank, corpora=corpora, retrieval=resolved["retrieval"],
            ann=resolved["ann"], seed=int(resolved["seed"]), device=args.device,
            build_ann=True, timing=Timing(),
        )
        summary["methods"][method] = run_split(
            output_root=output_root, resolved=resolved, split=split, teacher=teacher,
            teacher_hash=teacher_hash, teacher_batch=teacher_batch, engine=engine,
            method=method, logit_cache=cache, run_exact=run_exact,
        )
        summary["methods"][method]["student_epoch"] = payload.get("epoch")
        summary["methods"][method]["student_hash"] = lock["models"]["students"][arm]["parameter_hash"]
        log_line(f"evaluate {split} {method}: {_headline(summary['methods'][method])}")
        del engine, student
        torch.cuda.empty_cache()
    summary["finished_utc"] = utcnow()
    summary["teacher_logit_cache"] = cache.stats() if cache is not None else None
    suffix = "." + "-".join(sorted(requested)) if requested else ""
    write_json(output_root / "diagnostics" / f"EVAL_{split}{suffix}.json", summary)
    if cache is not None:
        cache.close()
    return summary


def merge_method_summaries(output_root: Path, split: str) -> dict[str, Any]:
    """Fold the per-group evaluation files for a split into one summary.

    Running the formal table as several processes leaves one file per group; the
    report and the timing collector both read a single ``EVAL_{split}.json``, so
    the groups are recombined here.  A group's file is matched by the method names
    it declares, and any method produced by two groups is reported as a conflict
    rather than silently overwritten.
    """
    directory = output_root / "diagnostics"
    partials = sorted(directory.glob(f"EVAL_{split}.*.json"))
    if not partials:
        raise ConfigError(
            f"no per-group evaluation files found for {split}; expected "
            f"EVAL_{split}.<methods>.json under {directory}"
        )
    merged: dict[str, Any] = {}
    origins: dict[str, str] = {}
    conflicts: list[str] = []
    for path in partials:
        payload = json.loads(path.read_text(encoding="utf-8"))
        for method, summary in payload.get("methods", {}).items():
            if method in merged:
                conflicts.append(f"{method} produced by {origins[method]} and {path.name}")
                continue
            merged[method] = summary
            origins[method] = path.name
    if conflicts:
        raise ConfigError(
            "two evaluation groups produced the same method: " + "; ".join(conflicts)
        )
    combined: dict[str, Any] = {
        "split": split,
        "merged_from": [p.name for p in partials],
        "methods": merged,
        "method_origins": origins,
        "finished_utc": utcnow(),
    }
    # Settings that are properties of the split rather than of one group.
    for path in partials:
        payload = json.loads(path.read_text(encoding="utf-8"))
        for key in ("run_exact", "teacher_hash", "started_utc"):
            if key in payload and key not in combined:
                combined[key] = payload[key]
    write_json(directory / f"EVAL_{split}.json", combined)
    log_line(
        f"merge-methods {split}: {sorted(merged)} from {len(partials)} group file(s)"
    )
    return combined


def cmd_merge_methods(args, spec, spec_dir, cwd, output_root, receipt) -> dict[str, Any]:
    """Recombine per-group evaluation summaries written by parallel processes."""
    _resolved_or_raise(output_root)
    result = {
        split: merge_method_summaries(output_root, split)
        for split in ("dev", "test")
        if list((output_root / "diagnostics").glob(f"EVAL_{split}.*.json"))
    }
    if not result:
        raise ConfigError("no per-group evaluation files exist to merge")
    return {split: sorted(payload["methods"]) for split, payload in result.items()}


def _headline(summary: dict[str, Any]) -> str:
    view = summary.get("views", {}).get("ann", {})
    reranked = view.get("teacher_reranked") or view.get("final_order") or {}
    return (
        f"R@10={reranked.get('overall_R10')} implicit={reranked.get('implicit_R10')} "
        f"explicit={reranked.get('explicit_R10')}"
    )


def _run_raw_method(
    *, output_root, resolved, split, method, raw_engine, bank, cache, run_exact,
) -> dict[str, Any]:
    """RAW-D (no re-ranking at all) and RAW-2H (C100 admission order)."""
    population = list(read_jsonl(output_root / "raw_gt" / f"{split}.population.jsonl"))
    ks = [int(k) for k in resolved["evaluation"]["ks"]]
    records = []
    for row in population:
        query_id = row["query_id"]
        outputs = raw_engine.pipeline_both(bank, query_id)
        record: dict[str, Any] = {
            "query_id": query_id, "split": split, "method": method,
            "direct_target_ids": row["direct_target_ids"],
            "implicit_target_ids": row["implicit_target_ids"],
            "positive_target_ids": row["positive_target_ids"],
        }
        for label, output in outputs.items():
            if method == "RAW-D":
                ranked = output["D100"]
                scores = raw_engine.direct_scores(bank, query_id, ranked)
            else:
                ranked = output["C100"]
                scores = {}
            record[label] = {
                "C100": output["C100"], "D100": output["D100"], "B_Q": output["B_Q"],
                "U": output["U"], "R_E": output["R_E"], "L_E": output["L_E"],
                "Top50": ranked[:50], "scores": scores,
                "metrics": evaluate.query_metrics(
                    ranked, row["direct_target_ids"], row["implicit_target_ids"], ks
                ),
            }
        records.append(record)
    return _summarize_records(records, ks, method, split, output_root, write=True)


def _run_teacher_on_raw(
    *, output_root, resolved, split, bank, raw_engine, teacher, teacher_hash,
    teacher_batch, cache, run_exact,
) -> dict[str, Any]:
    """RAW+T: the same raw C100 / raw B20 candidates, re-ranked by this run's T."""
    population = list(read_jsonl(output_root / "raw_gt" / f"{split}.population.jsonl"))
    ks = [int(k) for k in resolved["evaluation"]["ks"]]
    chunk = int(resolved["retrieval"]["teacher_target_chunk"])
    records = []
    for row in population:
        query_id = row["query_id"]
        outputs = raw_engine.pipeline_both(bank, query_id)
        record: dict[str, Any] = {
            "query_id": query_id, "split": split, "method": "RAW+T",
            "direct_target_ids": row["direct_target_ids"],
            "implicit_target_ids": row["implicit_target_ids"],
            "positive_target_ids": row["positive_target_ids"],
            "witnesses": row["witnesses"],
        }
        for label, output in outputs.items():
            c100 = output["C100"]
            scores = score_selection(
                teacher=teacher, batch=teacher_batch, cache=cache, teacher_hash=teacher_hash,
                mode="J", query_id=query_id, candidates=c100, context=output["B_Q"],
                chunk=chunk,
            )
            ranked = rank_by_scores(c100, scores)
            record[label] = {
                "C100": c100, "D100": output["D100"], "B_Q": output["B_Q"],
                "U": output["U"], "R_E": output["R_E"], "L_E": output["L_E"],
                "arrival_evidence_by_target": output["arrival_evidence_by_target"],
                "Top50": ranked[:50], "J_scores": {t: scores[t] for t in c100},
                "metrics": evaluate.query_metrics(
                    ranked, row["direct_target_ids"], row["implicit_target_ids"], ks
                ),
                "raw_admission_metrics": evaluate.query_metrics(
                    c100, row["direct_target_ids"], row["implicit_target_ids"], ks
                ),
                "direct100_metrics": evaluate.query_metrics(
                    output["D100"], row["direct_target_ids"], row["implicit_target_ids"], ks
                ),
            }
        records.append(record)
    return _summarize_records(records, ks, "RAW+T", split, output_root, write=True)


def _summarize_records(records, ks, method, split, output_root, write: bool) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "split": split, "method": method, "queries": len(records), "views": {},
    }
    directory = (
        (output_root / "test" / method)
        if split == "test"
        else (output_root / "diagnostics" / "dev" / method)
    )
    directory.mkdir(parents=True, exist_ok=True)
    for label in ("ann", "exact"):
        present = [r for r in records if label in r]
        if not present:
            continue
        views: dict[str, Any] = {
            "final_order": evaluate.macro_metrics([r[label]["metrics"] for r in present], ks),
            "candidate_recall": {
                "C100": float(np.mean([
                    evaluate.recall_at_k(r[label]["C100"], r["positive_target_ids"], 100)
                    for r in present
                ])),
                "U": float(np.mean([
                    evaluate.recall_at_k(r[label]["U"], r["positive_target_ids"], len(r[label]["U"]))
                    for r in present
                ])),
                "D100": float(np.mean([
                    evaluate.recall_at_k(r[label]["D100"], r["positive_target_ids"], 100)
                    for r in present
                ])),
            },
        }
        if "raw_admission_metrics" in present[0][label]:
            views["raw_admission_order"] = evaluate.macro_metrics(
                [r[label]["raw_admission_metrics"] for r in present], ks
            )
            views["raw_direct100"] = evaluate.macro_metrics(
                [r[label]["direct100_metrics"] for r in present], ks
            )
        summary["views"][label] = views
        if write:
            write_jsonl(
                directory / f"{label}.jsonl",
                (
                    {
                        "query_id": r["query_id"],
                        "C100": r[label]["C100"],
                        "Top50": r[label].get("Top50"),
                        "J_scores": r[label].get("J_scores"),
                        "scores": r[label].get("scores"),
                        "B_Q": r[label]["B_Q"],
                        "U": r[label]["U"],
                        "L_E": r[label].get("L_E"),
                        "metrics": r[label]["metrics"],
                    }
                    for r in present
                ),
            )
    if write:
        write_json(output_root / "diagnostics" / f"split_{split}_{method}.json", summary)
    return summary


def score_selection(
    *,
    teacher,
    batch: train.TeacherBatch,
    cache: evaluate.TeacherLogitCache,
    teacher_hash: str,
    mode: str,
    query_id: str,
    candidates: Sequence[str],
    context: Sequence[str],
    chunk: int,
) -> dict[str, float]:
    """Teacher scores for a fixed (query, candidate list, context), with reuse.

    The cache key carries the Teacher identity, the mode, the query, the full
    context id list and the candidate, so a score computed under another model or
    another bundle can never be served here (spec 14.14).
    """
    mode_id = models.MODE_P if mode == "P" else models.MODE_J
    keys = [
        evaluate.TeacherLogitCache.key(teacher_hash, mode_id, query_id, context, candidate)
        for candidate in candidates
    ]
    found = cache.get_many(keys)
    missing = [
        (position, candidate)
        for position, (key, candidate) in enumerate(zip(keys, candidates))
        if key not in found
    ]
    if missing:
        positions = [position for position, _ in missing]
        todo = [candidate for _, candidate in missing]
        values = batch.score(
            teacher,
            [query_id] * len(todo),
            [[candidate] for candidate in todo],
            [list(context) for _ in todo],
            mode_id,
            chunk=chunk,
        )
        flat = torch.cat(values).detach().float().cpu().tolist()
        rows = []
        for position, candidate, value in zip(positions, todo, flat):
            found[keys[position]] = float(value)
            rows.append(
                (
                    keys[position], teacher_hash, mode_id, query_id,
                    json.dumps(list(context), separators=(",", ":")), candidate, float(value),
                )
            )
        cache.put_many(rows)
    return {candidate: found[key] for candidate, key in zip(candidates, keys)}


def rank_by_scores(candidates: Sequence[str], scores: dict[str, float]) -> list[str]:
    """Descending score, then ascending canonical id (spec 3)."""
    return sorted(candidates, key=lambda c: (-scores[c], c))


def cmd_diagnose(args, spec, spec_dir, cwd, output_root, receipt) -> dict[str, Any]:
    """Spec 12.3-12.5: fixed-pool perturbations, strict EO, ANN/exact, hubness."""
    resolved = _resolved_or_raise(output_root)
    bank, corpora = _load_bank_objects(output_root, resolved)
    teacher, _state = _load_teacher(output_root / "teacher" / "best.pt", resolved, args.device)
    teacher_hash = _model_hash(teacher)
    teacher_batch = train.TeacherBatch(bank, device=args.device)
    dev_path = output_root / "diagnostics" / "per_query" / "dev_KD+T_ann.jsonl"
    if not dev_path.is_file():
        raise ConfigError(f"dev KD+T per-query records are missing: {dev_path}")
    dev_records = list(read_jsonl(dev_path))
    population = _population_index(output_root, "dev")
    student_path = output_root / "student_KD" / "best.pt"
    if not student_path.is_file():
        raise ConfigError(f"S-KD is missing: {student_path}")
    student = build_student(resolved["student"])
    student.load_state_dict(
        torch.load(student_path, map_location="cpu", weights_only=False)["model"]
    )
    student.to(args.device)
    _freeze(student)
    engine = evaluate.RetrievalEngine(
        student=student, bank=bank, corpora=corpora, retrieval=resolved["retrieval"],
        ann=resolved["ann"], seed=int(resolved["seed"]), device=args.device,
        build_ann=True, timing=Timing(),
    )
    cache = evaluate.TeacherLogitCache(output_root / "teacher_logits.sqlite")
    chunk = int(resolved["retrieval"]["teacher_target_chunk"])
    limit = args.dev_queries or int(resolved["evaluation"]["dev_health_queries"])
    records = dev_records[:limit]

    # J-shuffle replaces the whole B_Q with another dev query's B_Q, cycled by one
    # position in canonical query order.  No GT participates in that choice.
    ordered = sorted(records, key=lambda r: r["query_id"].encode("utf-8"))
    shuffled = {
        ordered[i]["query_id"]: ordered[(i + 1) % len(ordered)]["ann"]["B_Q"]
        for i in range(len(ordered))
    }

    fixed: list[dict[str, Any]] = []
    for record in records:
        query_id = record["query_id"]
        row = population[query_id]
        block = record["ann"]
        candidates = block["C100"]
        natural = block["B_Q"]
        p_scores = score_selection(
            teacher=teacher, batch=teacher_batch, cache=cache, teacher_hash=teacher_hash,
            mode="P", query_id=query_id, candidates=candidates, context=[], chunk=chunk,
        )
        j_empty = score_selection(
            teacher=teacher, batch=teacher_batch, cache=cache, teacher_hash=teacher_hash,
            mode="J", query_id=query_id, candidates=candidates, context=[], chunk=chunk,
        )
        j_natural = block["J_scores"]
        j_shuffle = score_selection(
            teacher=teacher, batch=teacher_batch, cache=cache, teacher_hash=teacher_hash,
            mode="J", query_id=query_id, candidates=candidates,
            context=shuffled[query_id], chunk=chunk,
        )
        witnesses = {k: set(v) for k, v in row["witnesses"].items()}
        fixed.append(
            {
                "query_id": query_id,
                "candidates": candidates,
                "natural_B_Q": natural,
                "shuffled_B_Q": shuffled[query_id],
                "P_QT": p_scores,
                "J_empty": j_empty,
                "J_natural": j_natural,
                "J_shuffle": j_shuffle,
                "P_rank": rank_by_scores(candidates, p_scores),
                "J_empty_rank": rank_by_scores(candidates, j_empty),
                "J_natural_rank": rank_by_scores(candidates, j_natural),
                "J_shuffle_rank": rank_by_scores(candidates, j_shuffle),
                "implicit_witness_in_bundle": {
                    t: sorted(witnesses.get(t, set()) & set(natural))
                    for t in row["implicit_target_ids"]
                },
                "witness_removed_from_bundle": {
                    t: sorted(witnesses[t] - set(natural))
                    for t in row["implicit_target_ids"]
                    if witnesses.get(t)
                },
                "direct_target_ids": row["direct_target_ids"],
                "implicit_target_ids": row["implicit_target_ids"],
                "positive_target_ids": row["positive_target_ids"],
                "strict_eo": _strict_eo(record, row),
            }
        )
        if len(fixed) % 25 == 0:
            log_line(f"diagnose: fixed pool {len(fixed)}/{len(records)}")

    ks = [int(k) for k in resolved["evaluation"]["ks"]]
    deltas = [
        entry["J_natural"][target] - entry["J_empty"][target]
        for entry in fixed
        for target, hits in entry["implicit_witness_in_bundle"].items()
        if hits
    ]
    necessity = {
        "implicit_pairs_with_witness_in_bundle": len(deltas),
        "mean_j_natural_minus_j_empty": float(np.mean(deltas)) if deltas else None,
        "median_j_natural_minus_j_empty": float(np.median(deltas)) if deltas else None,
        "share_positive_delta": float(np.mean([d > 0 for d in deltas])) if deltas else None,
        "note": (
            "Only implicit pairs whose annotated witness actually appears in the "
            "natural bundle enter this comparison. An implicit pair with no annotated "
            "witness is NOT evidence that the bundle is unsupported."
        ),
    }
    perturbation = {
        name: evaluate.macro_metrics(
            [
                evaluate.query_metrics(
                    entry[f"{name}_rank"], entry["direct_target_ids"],
                    entry["implicit_target_ids"], ks,
                )
                for entry in fixed
            ],
            ks,
        )
        for name in ("P_QT", "J_empty", "J_natural", "J_shuffle")
    }
    perturbation["note"] = (
        "P and J are different tasks, so their absolute scores are not comparable, "
        "only their rankings. J-natural and J-empty are the same task, so their "
        "difference and its ranking change are meaningful."
    )
    strict_eo = _aggregate_strict_eo([e["strict_eo"] for e in fixed])
    admission = _admission_diagnostic(records, population)
    hubness = _hubness(records, resolved)
    engine_summary = run_split(
        output_root=output_root, resolved=resolved, split="dev", teacher=teacher,
        teacher_hash=teacher_hash, teacher_batch=teacher_batch, engine=engine,
        method="KD+T_health", logit_cache=cache, max_queries=limit,
        write_artifacts=False, run_exact=True,
    )
    diagnostics_dir = output_root / "diagnostics"
    write_jsonl(diagnostics_dir / "fixed_pool" / "per_query.jsonl", fixed)
    write_json(diagnostics_dir / "fixed_pool" / "summary.json", {
        "queries": len(fixed), "perturbation": perturbation, "necessity": necessity,
    })
    write_json(diagnostics_dir / "strict_eo" / "summary.json", strict_eo)
    write_jsonl(
        diagnostics_dir / "strict_eo" / "per_query.jsonl",
        ({"query_id": e["query_id"], **e["strict_eo"]} for e in fixed),
    )
    write_json(diagnostics_dir / "ann_exact" / "summary.json", engine_summary)
    write_json(diagnostics_dir / "necessity" / "summary.json", necessity)
    write_json(diagnostics_dir / "resource" / "index_and_online.json", {
        "engine": engine_summary,
        "online_latency": engine_summary.get("timing"),
        "hubness": hubness,
        "admission": admission,
        "teacher_logit_cache": cache.stats(),
        "new_query_end_to_end_seconds": "not_measured",
        "note": (
            "Online latency for a brand-new query needs a live frozen-Qwen forward, "
            "which this diagnostic does not perform, so the new-query end-to-end time "
            "is reported as not_measured rather than estimated."
        ),
    })
    cache.close()
    summary = {
        "fixed_pool_queries": len(fixed),
        "perturbation": perturbation,
        "necessity": necessity,
        "strict_eo": strict_eo,
        "hubness": hubness,
        "admission": admission,
    }
    write_json(diagnostics_dir / "DIAGNOSE_SUMMARY.json", summary)
    log_line(f"diagnose: {json.dumps({k: v for k, v in summary.items() if k != 'strict_eo'})[:400]}")
    return summary


def _strict_eo(record: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    """Spec 12.4 strict evidence-only targets.

    T is strict-EO when it is a GT positive, lies in the natural union U, is absent
    from the same model's exact Direct100, and at least one natural L_E contains it.
    """
    block = record["ann"]
    exact_block = record.get("exact", block)
    witnesses = {t: set(v) for t, v in row["witnesses"].items()}
    positives = set(row["positive_target_ids"])
    in_union = set(block["U"])
    direct100 = set(exact_block["D100"])
    arrival = block["arrival_evidence_by_target"]
    strict = sorted(
        t for t in positives if t in in_union and t not in direct100 and arrival.get(t)
    )
    within = set(block["Top50"])
    return {
        "strict_eo_targets": len(strict),
        "entering_c100": len([t for t in strict if t in set(block["C100"])]),
        "entering_top10": len([t for t in strict if t in set(block["Top50"][:10])]),
        "entering_top50": len([t for t in strict if t in within]),
        "arrival_evidence_matches_witness": len(
            [t for t in strict if witnesses.get(t, set()) & set(arrival.get(t, []))]
        ),
        "strict_eo_target_ids": strict[:50],
        "has_strict_eo": bool(strict),
    }


def _aggregate_strict_eo(entries: Sequence[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {"queries": len(entries)}
    for key in (
        "strict_eo_targets", "entering_c100", "entering_top10",
        "entering_top50", "arrival_evidence_matches_witness",
    ):
        values = [int(e[key]) for e in entries]
        out[f"{key}_total"] = int(sum(values))
        out[f"{key}_query_macro"] = float(np.mean(values)) if values else None
    out["queries_with_any_strict_eo"] = int(sum(1 for e in entries if e["has_strict_eo"]))
    out["note"] = (
        "Target totals and query-macro are reported separately; neither substitutes "
        "for the other."
    )
    return out


def _admission_diagnostic(
    records: Sequence[dict[str, Any]], population: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    """Spec 12.4: U against a fair direct budget M_q of exactly |U_q| targets."""
    union_cover: list[float] = []
    matched_cover: list[float] = []
    for record in records:
        block = record["ann"]
        exact_block = record.get("exact", block)
        row = population[record["query_id"]]
        positives = row["positive_target_ids"]
        union = block["U"]
        size = len(union)
        matched = exact_block["D100"][:size]
        union_cover.append(evaluate.recall_at_k(union, positives, size))
        matched_cover.append(evaluate.recall_at_k(matched, positives, size))
    return {
        "queries": len(records),
        "U_coverage_mean": float(np.mean(union_cover)) if union_cover else None,
        "matched_direct_coverage_mean": float(np.mean(matched_cover)) if matched_cover else None,
        "difference": (
            float(np.mean(union_cover) - np.mean(matched_cover)) if union_cover else None
        ),
        "note": (
            "M_q is the exact direct top-|U_q|, so both sides are scored at the same "
            "target budget and only target admission differs."
        ),
    }


def _hubness(records: Sequence[dict[str, Any]], resolved: dict[str, Any]) -> dict[str, Any]:
    """Spec 12.5: exact Top-1 occupancy and Top-10 distinct counts."""
    out: dict[str, Any] = {}
    for label, key in (("D100", "D100"), ("C100_j_ranked", None)):
        counts: dict[str, int] = {}
        distinct: list[int] = []
        for record in records:
            block = record["ann"]
            ranked = block["D100"][:1] if key else block["Top50"][:10]
            for value in ranked:
                counts[value] = counts.get(value, 0) + 1
            distinct.append(len(set(block["C100"][:10])))
        total = sum(counts.values())
        top = max(counts.items(), key=lambda kv: kv[1]) if counts else (None, 0)
        out[label] = {
            "top1_occupancy_target": top[0],
            "top1_occupancy_share": (top[1] / total) if total else None,
            "top10_distinct_targets_query_macro": (
                float(np.mean(distinct)) if distinct else None
            ),
        }
    out["note"] = (
        "Occupancy is computed per relation over the saved exact rankings of this "
        "run; it is a concentration diagnostic, not a recall result."
    )
    return out


def cmd_package(args, spec, spec_dir, cwd, output_root, receipt) -> dict[str, Any]:
    """Spec 15: assemble the deliverable set, the report and the manifest."""
    resolved = _resolved_or_raise(output_root)
    manifest: list[dict[str, Any]] = []
    for path in sorted(output_root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(output_root).as_posix()
        if relative in ("MANIFEST.sha256", "REPORT.md"):
            continue
        manifest.append(
            {"path": relative, "bytes": path.stat().st_size, "sha256": sha256_path(path)}
        )
    manifest.sort(key=lambda r: r["path"].encode("utf-8"))
    (output_root / "MANIFEST.sha256").write_text(
        "\n".join(f"{row['sha256']}  {row['path']}" for row in manifest) + "\n",
        encoding="utf-8",
    )
    (output_root / "REPORT.md").write_text(
        render_report(output_root, resolved, manifest), encoding="utf-8"
    )
    timing = collect_timing_report(output_root)
    write_json(output_root / "TIMING_REPORT.json", timing)
    summary = {
        "files": len(manifest),
        "bytes": sum(row["bytes"] for row in manifest),
        "manifest_sha256": sha256_path(output_root / "MANIFEST.sha256"),
        "timing_report": "TIMING_REPORT.json",
        "stage_totals_seconds": timing["stage_totals_seconds"],
    }
    write_json(output_root / "diagnostics" / "package_summary.json", summary)
    log_line(f"package: {summary['files']} files, {summary['bytes'] / 1e6:.1f} MB")
    return summary


def render_report(output_root: Path, resolved: dict[str, Any], manifest: list[dict[str, Any]]) -> str:
    """A report that keeps trained models, replayed diagnostics and gaps separate."""
    audit_path = output_root / "audit" / "INPUT_AUDIT.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8")) if audit_path.is_file() else {}
    lines: list[str] = []
    add = lines.append
    add("# MMDD Stage1 CLEAN-R1 — run report")
    add("")
    add(f"- protocol: `{resolved['protocol_id']}`")
    add(f"- spec config sha256: `{resolved['spec_config_sha256']}`")
    add(f"- spec document sha256: `{resolved.get('spec_doc_sha256')}`")
    add(f"- resolved config sha256: `{resolved.get('resolved_config_sha256')}`")
    add(f"- seed: {resolved['seed']}")
    add(f"- dataset root: `{resolved['paths']['dataset_root']}`")
    add(f"- backbone: `{resolved['paths']['backbone_dir']}`")
    add("")
    add("## 1. What was actually trained")
    add("")
    add("| trajectory | architecture | initialisation | epochs | status |")
    add("|---|---|---|---|---|")
    teacher_status = "trained" if (output_root / "teacher" / "last.pt").is_file() else "not run"
    sup_status = "trained" if (output_root / "student_SUP" / "last.pt").is_file() else "not run"
    kd_status = "trained" if (output_root / "student_KD" / "last.pt").is_file() else "not run"
    add(f"| T | shared 3-layer Relation Transformer + one readout | fresh random (seed 13) | 6 | {teacher_status} |")
    add(f"| S-SUP | compact object pooler + low-rank conditional query | fresh random (seed 13) | 6 | {sup_status} |")
    add(f"| S-KD | same architecture as S-SUP | identical `student_init.pt` | 6 | {kd_status} |")
    add("")
    add("There is exactly one Teacher architecture and one Student architecture. P and J")
    add("are task tokens on the same weights and the same checkpoint, not two encoders and")
    add("not two training lineages.")
    add("")
    add("Not trained in this round: Stage2 (any part), any column selector, any generator,")
    add("any evidence selector or resampler, and the 200K efficiency run. No historical")
    add("checkpoint, optimizer state, PCA, Teacher logit, candidate list or ranking was read;")
    add("`parent_checkpoint` is `null` for every trajectory.")
    add("")
    add("## 2. Input audit")
    add("")
    if audit:
        counts = audit.get("counts", {})
        add(
            f"- objects: {counts.get('query_tables')} query tables, "
            f"{counts.get('lake_tables')} lake tables, {counts.get('bridge_assets')} bridge "
            f"assets ({counts.get('bridge_assets_text')} text, {counts.get('bridge_assets_image')} image)"
        )
        add(f"- canonical evidence after content dedup: {audit.get('canonical_evidence')}")
        add(f"- unusable evidence (empty text / undecodable image): {audit.get('unusable_assets')}")
        validation = audit.get("validation", {})
        add(f"- qrel reasons: {validation.get('qrel_reason_counts')}")
        add(f"- source groups per split: {validation.get('source_groups_per_split')}")
        add(f"- source-group overlap across splits: {validation.get('source_group_overlap')}")
        for split in ("train", "dev", "test"):
            entry = audit.get("population", {}).get(split, {})
            add(
                f"- {split}: {entry.get('queries_with_positive_qrel')} queries, "
                f"{entry.get('positive_targets')} positive targets, kinds {entry.get('kinds')}, "
                f"queries without any positive qrel {entry.get('queries_without_positive_qrel')}"
            )
        add(f"- semantic hashes: `{audit.get('semantic_hashes')}`")
    else:
        add("- input audit not found")
    add("")
    add("## 3. Results")
    add("")
    for label in ("dev", "test"):
        path = output_root / "diagnostics" / f"EVAL_{label}.json"
        add(f"### {label}")
        add("")
        if not path.is_file():
            add(f"not executed: `{path.relative_to(output_root)}` is absent.")
            add("")
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        add("| method | ranker | R@10 | implicit R@10 | explicit R@10 | R@20 | R@50 | C100 recall | U recall |")
        add("|---|---|---|---|---|---|---|---|---|")
        for method, summary in payload.get("methods", {}).items():
            for ranker in ("ann", "exact"):
                view = summary.get("views", {}).get(ranker)
                if not view:
                    continue
                metrics = view.get("teacher_reranked") or view.get("final_order") or {}
                recall = view.get("candidate_recall", {})
                add(
                    f"| {method} | {ranker} | {_fmt(metrics.get('overall_R10'))} | "
                    f"{_fmt(metrics.get('implicit_R10'))} | {_fmt(metrics.get('explicit_R10'))} | "
                    f"{_fmt(metrics.get('overall_R20'))} | {_fmt(metrics.get('overall_R50'))} | "
                    f"{_fmt(recall.get('C100'))} | {_fmt(recall.get('U'))} |"
                )
        add("")
        add(f"- ANN fidelity (not a GT recall): {payload.get('nn_fidelity', {})}")
        add("")
    add("## 4. Diagnostics (replay and perturbation only)")
    add("")
    add("These numbers replay the frozen models with perturbed or replaced contexts. They")
    add("are not additional training runs and add no architecture.")
    add("")
    path = output_root / "diagnostics" / "DIAGNOSE_SUMMARY.json"
    if path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
        add(f"- fixed-pool perturbation: {json.dumps(payload.get('perturbation', {}))}")
        add(f"- witness necessity: {json.dumps(payload.get('necessity', {}))}")
        add(
            "- strict EO: "
            + json.dumps({
                k: v for k, v in payload.get("strict_eo", {}).items()
                if not k.startswith("strict_eo_target_ids")
            })
        )
        add(f"- admission: {json.dumps(payload.get('admission', {}))}")
        add(f"- hubness: {json.dumps(payload.get('hubness', {}))}")
    else:
        add("diagnostics not executed.")
    add("")
    add("## 5. Cost and timing")
    add("")
    timing_path = output_root / "TIMING_REPORT.json"
    if timing_path.is_file():
        timing = json.loads(timing_path.read_text(encoding="utf-8"))
        add("Wall clock per command, summed over every invocation in this run")
        add("(including failed attempts, which are counted separately):")
        add("")
        add("| command | runs | failed | total seconds |")
        add("|---|---|---|---|")
        for name, entry in timing.get("stages", {}).get(
            "command_wall_clock", {}
        ).get("commands", {}).items():
            add(
                f"| {name} | {entry['runs']} | {entry.get('failed_runs', 0)} | "
                f"{entry['total_seconds']:.1f} |"
            )
        add("")
        offline = timing.get("stages", {}).get("offline_feature_build")
        if isinstance(offline, dict):
            add(f"- offline feature build: {offline.get('objects')} objects; "
                f"per-shard timing in `cache/CACHE_RECEIPT.json`.")
        training = timing.get("stages", {}).get("training")
        if isinstance(training, dict):
            for arm, payload in training.items():
                epochs = payload.get("epochs", [])
                add(f"- {arm} training: " + ", ".join(
                    f"epoch {e['epoch']} {_fmt_seconds(e.get('elapsed_seconds'))}"
                    for e in epochs
                ))
        online = timing.get("stages", {}).get("online_retrieval")
        if isinstance(online, dict):
            for name, payload in list(online.items())[:6]:
                if not isinstance(payload, dict):
                    continue
                add(f"- online retrieval `{name}`: queries={payload.get('queries')} "
                    f"elapsed={_fmt_seconds(payload.get('elapsed_seconds'))}")
        add("")
        add("Per-phase and per-modality breakdowns, including the frozen-Qwen forward")
        add("against CPU preprocessing, are in `TIMING_REPORT.json` and")
        add("`cache/CACHE_RECEIPT.json`.")
        if timing.get("missing"):
            add("")
            add("Not measured in this run:")
            for item in timing["missing"]:
                add(f"- {item}")
    else:
        add("`TIMING_REPORT.json` is absent, so no consolidated timing is reported.")
    add("")
    add("## 6. Not executed")
    add("")
    add("- The 200K-scale efficiency and budget run: only resource arithmetic over the")
    add("  actual object counts of this lake is reported (spec Eq. 9).")
    add("- Stage2 in any form: no training, no execution, no refactor. The interface it")
    add("  needs (`r_T`, full bundles, arrival provenance) is exported instead.")
    add("- No structural, temperature or budget grid search; one training seed only.")
    add("")
    add("## 7. Deliverables")
    add("")
    add(f"{len(manifest)} files are listed in `MANIFEST.sha256`.")
    add("")
    return "\n".join(lines) + "\n"


def _fmt_seconds(value: Any) -> str:
    if value is None:
        return "N/A"
    value = float(value)
    if value < 90:
        return f"{value:.1f}s"
    if value < 5400:
        return f"{value / 60:.1f}min"
    return f"{value / 3600:.2f}h"


def _fmt(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _population_index(output_root: Path, split: str) -> dict[str, dict[str, Any]]:
    path = output_root / "raw_gt" / f"{split}.population.jsonl"
    if not path.is_file():
        raise ConfigError(f"population is missing: {path}")
    return {str(row["query_id"]): row for row in read_jsonl(path)}


def run_split(
    *,
    output_root: Path,
    resolved: dict[str, Any],
    split: str,
    teacher,
    teacher_hash: str,
    teacher_batch: train.TeacherBatch,
    engine: evaluate.RetrievalEngine,
    method: str,
    logit_cache: evaluate.TeacherLogitCache,
    max_queries: int | None = None,
    write_artifacts: bool = True,
    run_exact: bool = True,
) -> dict[str, Any]:
    """Spec 10.2 / 12.1: full natural retrieval plus Teacher J re-ranking.

    The ANN and the exact index run on the *same* natural B_Q, because step 4 of
    the pipeline reads no target and no GT, so the bundle is a property of the
    query alone.  That keeps ANN error and candidate-set differences separable.
    """
    population = list(read_jsonl(output_root / "raw_gt" / f"{split}.population.jsonl"))
    if max_queries is not None:
        population = population[: int(max_queries)]
    chunk = int(resolved["retrieval"]["teacher_target_chunk"])
    ks = [int(k) for k in resolved["evaluation"]["ks"]]
    started = time.time()
    records: list[dict[str, Any]] = []
    retrieval_timing = Timing()
    teacher_timing = Timing()
    buckets: dict[str, int] = {
        "teacher_logit_hits": 0, "teacher_logit_misses": 0, "candidates_scored": 0
    }
    fidelity: dict[str, list[float]] = {"D": [], "E_text": [], "E_image": [], "C": []}
    for row in population:
        query_id = row["query_id"]
        with retrieval_timing.stage("retrieve_query"):
            outputs = (
                engine.pipeline_both(query_id)
                if run_exact
                else {"ann": engine.pipeline(query_id, use_ann=True)}
            )
        result = outputs.get("ann") or outputs["exact"]
        exact_result = outputs.get("exact")
        if exact_result is not None and "ann" in outputs:
            fidelity["D"].append(
                retrieve.nn_fidelity(
                    result["D100"], exact_result["D100"],
                    int(resolved["retrieval"]["direct_k"]),
                )
            )
            for key in ("E_text", "E_image"):
                fidelity[key].append(
                    retrieve.nn_fidelity(
                        result[key], exact_result[key],
                        int(resolved["retrieval"]["evidence_per_modality"]),
                    )
                )
            for evidence in [e for e in result["B_Q"] if e in exact_result["L_E"]]:
                fidelity["C"].append(
                    retrieve.nn_fidelity(
                        result["L_E"][evidence], exact_result["L_E"][evidence],
                        int(resolved["retrieval"]["second_hop_k"]),
                    )
                )
        record: dict[str, Any] = {
            "query_id": query_id,
            "split": split,
            "method": method,
            "query_kind": row["query_kind"],
            "positive_target_ids": row["positive_target_ids"],
            "direct_target_ids": row["direct_target_ids"],
            "implicit_target_ids": row["implicit_target_ids"],
            "witnesses": row["witnesses"],
        }
        for label, output in outputs.items():
            c100 = output["C100"]
            with teacher_timing.stage("teacher_rerank", path=label):
                before = (logit_cache.hits, logit_cache.misses)
                scores = score_selection(
                    teacher=teacher, batch=teacher_batch, cache=logit_cache,
                    teacher_hash=teacher_hash, mode="J", query_id=query_id,
                    candidates=c100, context=output["B_Q"], chunk=chunk,
                )
            buckets["teacher_logit_hits"] += logit_cache.hits - before[0]
            buckets["teacher_logit_misses"] += logit_cache.misses - before[1]
            buckets["candidates_scored"] += len(c100)
            ranked = rank_by_scores(c100, scores)
            student_scores = dict(
                zip(c100, engine.student_target_scores(query_id, c100).tolist())
            )
            record[label] = {
                "D100": output["D100"],
                "B_Q": output["B_Q"],
                "E_text": output["E_text"],
                "E_image": output["E_image"],
                "R_E": output["R_E"],
                "U": output["U"],
                "C100": c100,
                "Top50": ranked[:50],
                "J_scores": {t: scores[t] for t in c100},
                "student_scores": student_scores,
                "L_E": output["L_E"],
                "arrival_evidence_by_target": output["arrival_evidence_by_target"],
                "metrics": evaluate.query_metrics(
                    ranked, row["direct_target_ids"], row["implicit_target_ids"], ks
                ),
                "raw_admission_metrics": evaluate.query_metrics(
                    c100, row["direct_target_ids"], row["implicit_target_ids"], ks
                ),
                "direct100_metrics": evaluate.query_metrics(
                    output["D100"], row["direct_target_ids"], row["implicit_target_ids"], ks
                ),
                "student_direct_metrics": evaluate.query_metrics(
                    rank_by_scores(c100, student_scores),
                    row["direct_target_ids"], row["implicit_target_ids"], ks,
                ),
            }
        records.append(record)
    if write_artifacts:
        per_query_dir = output_root / "diagnostics" / "per_query"
        method_dir = (
            (output_root / "test" / method)
            if split == "test"
            else (output_root / "diagnostics" / "dev" / method)
        )
        per_query_dir.mkdir(parents=True, exist_ok=True)
        method_dir.mkdir(parents=True, exist_ok=True)
        for label in ("ann", "exact"):
            present = [r for r in records if label in r]
            if not present:
                continue
            write_jsonl(per_query_dir / f"{split}_{method}_{label}.jsonl", records)
            write_jsonl(
                method_dir / f"{label}.jsonl",
                (
                    {
                        "query_id": r["query_id"],
                        "C100": r[label]["C100"],
                        "Top50": r[label]["Top50"],
                        "J_scores": r[label]["J_scores"],
                        "student_scores": r[label]["student_scores"],
                        "B_Q": r[label]["B_Q"],
                        "D100": r[label]["D100"],
                        "U": r[label]["U"],
                        "R_E": r[label]["R_E"],
                        "L_E": r[label]["L_E"],
                        "arrival_evidence_by_target": r[label]["arrival_evidence_by_target"],
                        "metrics": r[label]["metrics"],
                    }
                    for r in present
                ),
            )
    elapsed = time.time() - started
    summary: dict[str, Any] = {
        "split": split,
        "method": method,
        "queries": len(records),
        "elapsed_seconds": elapsed,
        "timing": {
            # Retrieval is measured with the Student keys already built; the frozen
            # Qwen encode for a brand-new query is deliberately excluded here and
            # reported separately as new_query_end_to_end.
            "retrieval": retrieval_timing.report(name="online_retrieval"),
            "teacher": teacher_timing.report(name="online_teacher_rerank"),
            "index_build": engine.timing.report(name="index_build"),
            "seconds_per_query_end_to_end": elapsed / len(records) if records else None,
            "teacher_logit_cache": dict(buckets),
            "new_query_end_to_end_seconds": (
                "not_measured: this pass reuses frozen cache entries and performs no "
                "frozen-Qwen forward for a new query"
            ),
        },
        "nn_fidelity": {
            key: (float(np.mean(values)) if values else None)
            for key, values in fidelity.items()
        },
        "views": {},
        "teacher_logit_cache": logit_cache.stats(),
    }
    summary["ann_error_below_0_95"] = sorted(
        key
        for key, value in summary["nn_fidelity"].items()
        if value is not None and value < 0.95
    )
    for label in ("ann", "exact"):
        present = [r for r in records if label in r]
        if not present:
            continue
        summary["views"][label] = {
            "teacher_reranked": evaluate.macro_metrics(
                [r[label]["metrics"] for r in present], ks
            ),
            "raw_admission_order": evaluate.macro_metrics(
                [r[label]["raw_admission_metrics"] for r in present], ks
            ),
            "raw_direct100": evaluate.macro_metrics(
                [r[label]["direct100_metrics"] for r in present], ks
            ),
            "student_direct_scores": evaluate.macro_metrics(
                [r[label]["student_direct_metrics"] for r in present], ks
            ),
            "candidate_recall": {
                "C100": float(np.mean([
                    evaluate.recall_at_k(r[label]["C100"], r["positive_target_ids"], 100)
                    for r in present
                ])),
                "U": float(np.mean([
                    evaluate.recall_at_k(
                        r[label]["U"], r["positive_target_ids"], len(r[label]["U"])
                    )
                    for r in present
                ])),
                "D100": float(np.mean([
                    evaluate.recall_at_k(r[label]["D100"], r["positive_target_ids"], 100)
                    for r in present
                ])),
            },
            "candidate_identity": {
                "unique_targets_mean": float(np.mean([
                    len(set(r[label]["C100"])) for r in present
                ])),
                "duplicates_total": int(sum(
                    len(r[label]["C100"]) - len(set(r[label]["C100"])) for r in present
                )),
            },
        }
    if write_artifacts:
        (output_root / "diagnostics").mkdir(parents=True, exist_ok=True)
        write_json(output_root / "diagnostics" / f"split_{split}_{method}.json", summary)
    return summary


def collect_timing_report(output_root: Path) -> dict[str, Any]:
    """One consolidated timing view over every stage of the run.

    Each entry names the artifact it was read from, so a number in the report can
    always be traced back to the file that measured it, and a stage that never
    ran is listed as missing rather than reported as zero.
    """
    report: dict[str, Any] = {
        "protocol_id": None,
        "generated_utc": utcnow(),
        "stages": {},
        "missing": [],
        "note": (
            "Every duration here is measured wall clock from the artifact named in "
            "its `source` field. A stage that never ran is listed under `missing`, "
            "never reported as zero."
        ),
    }
    resolved_path = output_root / "resolved_config.json"
    if resolved_path.is_file():
        report["protocol_id"] = read_json(resolved_path).get("protocol_id")

    # 1. per-command wall clock, from the receipts every command writes
    receipts_path = output_root / "COMMANDS.jsonl"
    commands: dict[str, Any] = {}
    if receipts_path.is_file():
        for record in read_jsonl(receipts_path):
            name = str(record.get("command"))
            entry = commands.setdefault(
                name, {"runs": 0, "failed_runs": 0, "total_seconds": 0.0}
            )
            entry["runs"] += 1
            entry["total_seconds"] += float(record.get("duration_seconds") or 0.0)
            if int(record.get("exit_code") or 0) != 0:
                entry["failed_runs"] += 1
            entry["last_exit_code"] = record.get("exit_code")
            entry["start_utc"] = record.get("start_utc")
            entry["end_utc"] = record.get("end_utc")
        for entry in commands.values():
            extra = {
                key: entry[key] for key in ("start_utc", "end_utc", "last_exit_code")
                if key in entry
            }
            entry.update(extra)
    report["stages"]["command_wall_clock"] = {
        "source": "COMMANDS.jsonl",
        "commands": commands,
    }

    # 2. offline preprocessing: per-phase and per-modality encoding cost
    cache_receipt = output_root / "cache" / "CACHE_RECEIPT.json"
    if cache_receipt.is_file():
        receipt = read_json(cache_receipt)
        report["stages"]["offline_feature_build"] = {
            "source": "cache/CACHE_RECEIPT.json",
            "objects": receipt.get("objects"),
            "timing": receipt.get("timing"),
        }
    else:
        report["missing"].append("cache/CACHE_RECEIPT.json (offline feature build)")

    # 3. training: per-epoch attribution for the Teacher and each Student arm
    training: dict[str, Any] = {}
    teacher_dir = output_root / "teacher"
    teacher_epochs = []
    for path in sorted(teacher_dir.glob("epoch_*.json")) if teacher_dir.is_dir() else []:
        record = read_json(path)
        teacher_epochs.append(
            {
                "epoch": record.get("epoch"),
                "elapsed_seconds": record.get("elapsed_seconds"),
                "mean_loss": record.get("mean_loss"),
                "step": record.get("step"),
            }
        )
    if teacher_epochs:
        training["teacher"] = {"source": "teacher/epoch_*.json", "epochs": teacher_epochs}
    for arm in ("SUP", "KD"):
        directory = output_root / f"student_{arm}"
        epochs = []
        for path in sorted(directory.glob("epoch_*.json")) if directory.is_dir() else []:
            record = read_json(path)
            epochs.append(
                {
                    "epoch": record.get("epoch"),
                    "elapsed_seconds": record.get("elapsed_seconds"),
                    "mean_loss": record.get("mean_loss"),
                    "step": record.get("step"),
                    "packet_terms": record.get("packet_terms"),
                }
            )
        if epochs:
            training[arm] = {"source": f"student_{arm}/epoch_*.json", "epochs": epochs}
    report["stages"]["training"] = training or "missing"
    if not training:
        report["missing"].append("teacher/ and student_*/ epoch records (training)")

    # 4. index build and online retrieval latency
    online: dict[str, Any] = {}
    for path in sorted((output_root / "diagnostics").glob("split_*.json")) if (output_root / "diagnostics").is_dir() else []:
        summary = read_json(path)
        online[path.name] = {
            "queries": summary.get("queries"),
            "elapsed_seconds": summary.get("elapsed_seconds"),
            "timing": summary.get("timing"),
            "nn_fidelity": summary.get("nn_fidelity"),
        }
    for label in ("dev", "test"):
        path = output_root / "diagnostics" / f"EVAL_{label}.json"
        if path.is_file():
            payload = read_json(path)
            online[f"EVAL_{label}"] = {
                method: {
                    "queries": summary.get("queries"),
                    "elapsed_seconds": summary.get("elapsed_seconds"),
                    "timing": summary.get("timing"),
                }
                for method, summary in payload.get("methods", {}).items()
            }
    report["stages"]["online_retrieval"] = online or "missing"
    if not online:
        report["missing"].append("diagnostics/ eval summaries (online retrieval)")

    # 5. rolled-up totals per command, for the headline table
    report["stage_totals_seconds"] = {
        name: entry["total_seconds"] for name, entry in sorted(commands.items())
    }
    return report


def verify_cache_identity(ordered: list[dict[str, Any]]) -> dict[str, Any]:
    """Check every cached row belongs to one encoding, and describe how.

    The stored ``cache_fingerprint`` covers the model, prompt and cache
    configuration.  An earlier revision of that formula also folded in the number
    of objects a slice happened to contain, which made two slices of the same lake
    look like different encodings.  Both formulas are therefore accepted, but only
    when the per-object encoding metadata proves the rows are interchangeable.
    """
    expected_kinds = {
        cache_module.MODALITY_TABLE: list(cache_module.TABLE_KINDS),
        cache_module.MODALITY_TEXT: list(cache_module.TEXT_KINDS),
        cache_module.MODALITY_IMAGE: list(cache_module.IMAGE_KINDS_FULL),
    }
    fingerprints: dict[str, int] = {}
    problems: list[str] = []
    for row in ordered:
        fingerprints[row["cache_fingerprint"]] = fingerprints.get(row["cache_fingerprint"], 0) + 1
        modality_id = int(row["modality_id"])
        kinds = list(row["kind_ids"])
        if expected_kinds.get(modality_id) != kinds:
            problems.append(
                f"{row['object_id']}: kind/modality mismatch ({modality_id} -> {kinds})"
            )
        if int(row.get("mask_popcount") or 0) < 1:
            problems.append(f"{row['object_id']}: no valid slots")
    return {
        "consistent": not problems,
        "rows": len(ordered),
        "distinct_fingerprints": {
            key: count for key, count in sorted(fingerprints.items())
        },
        "fingerprint_formula_note": (
            "Two hashes for the same encoding appear only when shards were encoded "
            "by different revisions of the fingerprint formula, which at one point "
            "included the slice size. Per-object kind and modality metadata is "
            "checked independently and must be uniform."
        ),
        "problems": problems[:20],
        "problem_count": len(problems),
    }


def _same_encoded_object(left: dict[str, Any], right: dict[str, Any]) -> bool:
    """Do two manifest rows describe the same encoded object?

    A repartitioned re-encode reproduces the same modality, the same kind layout,
    the same validity pattern and a normalised vector of norm 1, all computed from
    an identical prompt and input. Any disagreement means the rows came from
    different inputs or different encoders, so they may never be merged.
    """
    if int(left.get("modality_id", -1)) != int(right.get("modality_id", -1)):
        return False
    if list(left.get("kind_ids") or []) != list(right.get("kind_ids") or []):
        return False
    if int(left.get("mask_popcount") or 0) != int(right.get("mask_popcount") or 0):
        return False
    if left.get("object_type") != right.get("object_type"):
        return False
    return True


def _verify_overlap_bytes(
    overlap_pairs: list[dict[str, Any]],
    feature_dir: Path,
    dim: int,
    slots: int,
    limit: int = 512,
) -> dict[str, Any]:
    """Byte-compare a sample of overlapping rows across the shard files holding them.

    Two shards that re-encoded the same objects must agree bit for bit, because
    both read the same prompt from the same frozen model. Metadata agreement alone
    would not catch a silent encoder change, so the real z/C/mask bytes of a sample
    are compared. Any disagreement means the shards are not interchangeable and the
    merge must stop rather than pick a winner.
    """
    import numpy as np

    z_dtype, c_dtype, m_dtype = np.float32, np.float16, np.uint8
    if not overlap_pairs:
        return {"sampled": 0, "identical": True, "reason": "no overlapping objects"}

    # Deterministic sample: byte order over the object id, so a rerun checks the
    # same objects rather than a fresh random draw.
    ordered_pairs = sorted(
        overlap_pairs, key=lambda p: p["object_id"].encode("utf-8")
    )[:limit]
    handles: dict[str, Any] = {}
    mismatches: list[str] = []
    checked = 0
    try:
        for pair in ordered_pairs:
            values = []
            for side in ("kept", "superseded"):
                row = pair[side]
                key = f"{row['_source_shard']}/{row['shard']}"
                if key not in handles:
                    root = feature_dir / row["_source_shard"] / "shards" / row["shard"].split("/", 1)[1]
                    handles[key] = {
                        "z": np.memmap(root / "z.f32", dtype=z_dtype, mode="r").reshape(-1, dim),
                        "C": np.memmap(root / "C.f16", dtype=c_dtype, mode="r").reshape(-1, slots, dim),
                        "mask": np.memmap(root / "mask.u8", dtype=m_dtype, mode="r").reshape(-1, slots),
                    }
                offset = int(row["offset"])
                handle = handles[key]
                values.append(
                    (
                        handle["z"][offset].copy(),
                        handle["C"][offset].copy(),
                        handle["mask"][offset].copy(),
                    )
                )
            left, right = values
            if not (
                np.array_equal(left[0], right[0])
                and np.array_equal(left[1], right[1])
                and np.array_equal(left[2], right[2])
            ):
                mismatches.append(pair["object_id"])
            checked += 1
    finally:
        for handle in handles.values():
            for view in handle.values():
                del view
    return {
        "sampled": checked,
        "identical": not mismatches,
        "mismatched_objects": mismatches[:20],
        "mismatch_count": len(mismatches),
        "reason": (
            "z, C and mask bytes compared for a deterministic sample of the "
            "overlapping objects across the two shards that encoded them"
        ),
    }













# --------------------------------------------------------------------------
# shared training-side helpers
# --------------------------------------------------------------------------














































































def _shard_tag(args: Any, range_count: int) -> str:
    """Directory tag for one encoding process's own manifest and shards."""
    if args.shard_count <= 1 and range_count <= 1:
        return "shards"
    if range_count <= 1:
        return f"gpu-{args.shard_id}"
    return f"gpu-{args.shard_id}r{args.range_id}"
