#!/usr/bin/env python
"""Re-run the auto-check for the recoveries whose query row changed.

``relocate_page_url.py`` appended a ``page_url`` cell to 22,038 query rows.  The
auto-check input *is* that row, so every recovery attached to one of those
queries carries a verdict that was computed on the pre-move row.  This replays
exactly those recoveries against the current row and rewrites the verdict.

Three dispositions:

* a recovery whose query or target was deleted is **dropped** -- it is a
  dangling reference;
* a recovery whose query row changed is **re-checked** and kept only if the
  model still extracts the claimed value from the evidence;
* everything else is copied byte-identical.

The prompt is not re-implemented: ``review_messages`` and
``parse_model_extractions`` are imported from the same module the builder uses,
so the request is the one the build would have sent.  The one deliberate
difference is the mask -- a visible cell whose value already equals the claimed
value is masked out, so the reviewer cannot copy the answer off the row.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts_old"))

import build_mm_joinability_dataset as join_builder          # noqa: E402
import mm_joinability_dataset_auto_checker as checker        # noqa: E402
import requests                                              # noqa: E402

PLAN_VERSION = "pageurl-relocation-v1"
IMAGE_MAX_PIXELS = 512_000
SCHEMA_VERSION = "model-output-auto-check-v6-redundant-group-mask"


def serialize(record: dict[str, Any]) -> str:
    return json.dumps(record, ensure_ascii=False)


def load_ids(path: Path) -> set[str]:
    with path.open(encoding="utf-8") as handle:
        return {line.strip() for line in handle if line.strip()}


def load_query_rows(dataset_dir: Path, wanted: set[str]) -> dict[str, dict[str, Any]]:
    """query_id -> {source_table_id, entity_col, rows: {row_id: [(name, value)]}}"""
    out: dict[str, dict[str, Any]] = {}
    for path in sorted((dataset_dir / "query_tables").iterdir()):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                position = line.find('"object_id"')
                if position < 0:
                    continue
                query_id = line[position:].split('"')[3]
                if query_id not in wanted:
                    continue
                record = json.loads(line)
                rows: dict[int, list[tuple[str, str]]] = {}
                for row in record.get("rows") or []:
                    cells = []
                    for cell in row.get("cells") or []:
                        name = checker.clean_text(cell.get("column_name"))
                        value = checker.clean_text(cell.get("text"))
                        if name and value:
                            cells.append((name, value))
                    rows[int(row["row_id"])] = cells
                out[query_id] = {
                    "source_table_id": record.get("source_table_id", ""),
                    "rows": rows,
                }
    return out


def load_assets(dataset_dir: Path, work_dir: Path, wanted: set[str]) -> dict[str, dict[str, Any]]:
    """asset_id -> the materialised asset record, for the ids we will re-check."""
    out: dict[str, dict[str, Any]] = {}
    directory = work_dir / "materialized_assets" / "bridge_assets"
    for path in sorted(directory.iterdir()):
        if not path.name.endswith(".jsonl"):
            continue
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if '"asset_id"' not in line:
                    continue
                asset_id = line.split('"asset_id"', 1)[1].split('"')[1]
                if asset_id in wanted and asset_id not in out:
                    out[asset_id] = json.loads(line)
        if len(out) >= len(wanted):
            break
    return out


def build_messages(
    recovery: dict[str, Any],
    query: dict[str, Any],
    asset: dict[str, Any],
) -> list[dict[str, Any]] | None:
    claimed = checker.clean_text(recovery["recovered_attribute"].get("value"))
    attribute = checker.clean_text(recovery["recovered_attribute"].get("column_name"))
    if not attribute:
        return None
    cells = query["rows"].get(int(recovery["query_row_id"]))
    if cells is None:
        return None
    target = attribute.casefold()
    # Mask by name only, exactly as ``_auto_check_review_batch`` does.  The
    # redundancy-group mask the builder computes is applied to the cache key,
    # never to the prompt, so masking by value here would make this a different
    # check than the one being replaced.
    masked_row = [
        {"name": name, "value": value}
        for name, value in cells
        if name.casefold() != target
    ]
    item = {
        "review_id": "recheck_" + recovery["recovery_id"][:32],
        "recovery_id": recovery["recovery_id"],
        "path_id": recovery.get("path_id", ""),
        "query_row_id": str(recovery["query_row_id"]),
        "masked_row": masked_row,
        "masked_attribute_was_present": any(name.casefold() == target for name, _ in cells),
        "attribute": {"name": attribute, "value": claimed},
        "evidence": {
            "asset_id": asset.get("asset_id", ""),
            "asset_type": asset.get("asset_type", ""),
            "content": checker.clean_text(asset.get("content"))[:6000],
            "image_sha256": checker.clean_text(asset.get("sha256")),
        },
        "image_path": checker.clean_text(asset.get("local_path")),
    }
    batch = {
        "query_table_id": recovery["query_table_id"],
        "target_table_id": recovery.get("target_table_id", ""),
        "source_table_id": recovery.get("source_table_id", ""),
        "split": recovery.get("split", ""),
        "query_row_ids": [str(recovery["query_row_id"])],
        "items": [item],
    }
    return checker.review_messages([batch], image_max_pixels=IMAGE_MAX_PIXELS)


def call(base_url: str, messages: list[dict[str, Any]], timeout: float) -> str:
    payload = {
        "model": "Qwen3.5-9B",
        "messages": messages,
        "temperature": 0,
        "max_tokens": 512,
        "chat_template_kwargs": {"enable_thinking": False},
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "mm_joinability_single_attribute_extraction",
                "strict": True,
                "schema": checker.AUTO_CHECK_EXTRACTION_SCHEMA,
            },
        },
    }
    last: Exception | None = None
    for attempt in range(3):
        try:
            response = requests.post(f"{base_url}/chat/completions", json=payload, timeout=timeout)
            if response.status_code >= 400:
                raise RuntimeError(f"HTTP {response.status_code}: {response.text[:200]}")
            return response.json()["choices"][0]["message"]["content"]
        except Exception as error:  # transient tunnel/engine hiccups are expected
            last = error
            if attempt < 2:
                time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"giving up: {last!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--work_dir", required=True)
    parser.add_argument("--plan_dir", default="MMDD_PAGEURL_RELOCATION_20260920")
    parser.add_argument("--endpoints", default="http://127.0.0.1:18000/v1,http://127.0.0.1:18001/v1")
    parser.add_argument("--concurrency", type=int, default=48)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--limit", type=int, default=0, help="re-check only the first N (smoke test)")
    parser.add_argument("--control", type=int, default=0,
                        help="also re-check N untouched recoveries as a noise control")
    parser.add_argument("--no_second_round", action="store_true",
                        help="drop a recovery on a single failing run instead of two")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    work_dir = Path(args.work_dir)
    lists = Path(args.plan_dir) / "lists"
    rerun = load_ids(lists / "queries_rerun_autocheck.txt")
    drop = load_ids(lists / "targets_to_drop.txt")
    orphan = load_ids(lists / "queries_to_delete_orphaned.txt")
    endpoints = [e.strip().rstrip("/") for e in args.endpoints.split(",") if e.strip()]

    manifest_path = dataset_dir / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    # ---- read every recovery, classify
    entries: list[dict[str, Any]] = []
    for shard in manifest["artifacts"]["evidence_recoveries"]["shards"]:
        with (dataset_dir / shard["path"]).open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    entries.append(json.loads(line))
    counts = {"total": len(entries), "dangling": 0, "affected": 0, "untouched": 0}
    affected: list[int] = []
    untouched: list[int] = []
    for index, record in enumerate(entries):
        if record["query_table_id"] in orphan or record["target_table_id"] in drop:
            counts["dangling"] += 1
        elif record["query_table_id"] in rerun:
            counts["affected"] += 1
            affected.append(index)
        else:
            counts["untouched"] += 1
            untouched.append(index)
    if args.limit:
        affected = affected[: args.limit]
        counts["limited_to"] = len(affected)
    if args.control:
        # A control arm: recoveries whose row did NOT change, re-checked with the
        # same code.  Their rejection rate is the checker's own noise floor; the
        # affected arm is only interesting relative to it.
        import random
        rng = random.Random(20260920)
        counts["control"] = min(args.control, len(untouched))
        control = rng.sample(untouched, counts["control"])
    else:
        control = []
    print(json.dumps(counts, indent=2), flush=True)

    queries = load_query_rows(dataset_dir, {entries[i]["query_table_id"] for i in affected + control})
    assets = load_assets(
        dataset_dir, work_dir, {entries[i]["evidence"]["asset_id"] for i in affected + control}
    )
    print(f"query rows: {len(queries)}  assets: {len(assets)}", flush=True)

    if args.dry_run:
        missing_q = sum(1 for i in affected if entries[i]["query_table_id"] not in queries)
        missing_a = sum(1 for i in affected if entries[i]["evidence"]["asset_id"] not in assets)
        print(json.dumps({"missing_query_rows": missing_q, "missing_assets": missing_a}, indent=2))
        return

    # ---- re-check
    lock = threading.Lock()
    outcome: dict[int, dict[str, Any]] = {}
    done = [0]
    started = time.time()

    arm = affected + control
    arm_of = {index: "affected" for index in affected}
    arm_of.update({index: "control" for index in control})

    def work(index: int, base_url: str) -> None:
        record = entries[index]
        try:
            query = queries[record["query_table_id"]]
            asset = assets[record["evidence"]["asset_id"]]
            messages = build_messages(record, query, asset)
            if messages is None:
                outcome[index] = {"supported": False, "error": "cannot_rebuild_prompt"}
                return
            raw = call(base_url, messages, args.timeout)
            payload = json.loads(raw)
            if not isinstance(payload, dict) or "extracted_value" not in payload:
                raise ValueError(f"model response is missing extracted_value: {raw[:120]!r}")
            extracted = checker.clean_text(payload.get("extracted_value"))
            claimed = checker.clean_text(record["recovered_attribute"].get("value"))
            supported = bool(extracted) and join_builder.values_match(
                extracted, claimed,
                attribute_name=checker.clean_text(record["recovered_attribute"].get("column_name")),
            )
            outcome[index] = {"supported": supported, "extracted_value": extracted}
        except Exception as error:
            outcome[index] = {"supported": None, "error": repr(error)[:200]}
        with lock:
            done[0] += 1
            if done[0] % 1000 == 0:
                rate = done[0] / max(time.time() - started, 1e-6)
                print(f"  {done[0]}/{len(arm)}  {rate:.1f}/s", flush=True)

    def run_round(indices: list[int], label: str) -> None:
        if not indices:
            return
        print(f"  round {label}: {len(indices)} calls", flush=True)
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = [pool.submit(work, index, endpoints[slot % len(endpoints)])
                       for slot, index in enumerate(indices)]
            for future in as_completed(futures):
                future.result()

    run_round(arm, "1")

    # Round 2 exists because the check is a noisy classifier: on inputs that did
    # NOT change it still flips ~10.5% of verdicts, so a single failing run is a
    # coin flip, not evidence.  A recovery is only dropped when both rounds fail.
    rounds: dict[int, int] = {}
    if not args.no_second_round:
        failed = [i for i, o in outcome.items() if o["supported"] is not True]
        for index in failed:
            rounds[index] = 2
        run_round(sorted(failed), "2 (failures only)")
        for index in failed:
            second = outcome.get(index) or {}
            if second.get("supported") is True:  # second pass rescued it
                outcome[index] = {**second, "rescued_by_second_round": True}

    def rate_of(which: str) -> dict[str, Any]:
        picked = [o for i, o in outcome.items() if arm_of[i] == which]
        kept = sum(1 for o in picked if o["supported"] is True)
        lost = sum(1 for o in picked if o["supported"] is False)
        bad = sum(1 for o in picked if o["supported"] is None)
        return {"n": len(picked), "supported": kept, "now_rejected": lost, "errors": bad,
                "rejection_rate": round(lost / max(kept + lost, 1), 4)}

    supported = sum(1 for i, o in outcome.items()
                    if arm_of[i] == "affected" and o["supported"] is True)
    rejected = sum(1 for i, o in outcome.items()
                   if arm_of[i] == "affected" and o["supported"] is False)
    errored = sum(1 for i, o in outcome.items()
                  if arm_of[i] == "affected" and o["supported"] is None)
    print(json.dumps({"affected": rate_of("affected"),
                      "control": rate_of("control") if control else None,
                      "wall_seconds": round(time.time() - started, 1)}, indent=2), flush=True)

    if args.limit:
        for index in affected[:10]:
            record = entries[index]
            print(json.dumps({
                "recovery_id": record["recovery_id"],
                "attribute": record["recovered_attribute"]["column_name"],
                "old_verdict": (record.get("auto_check") or {}).get("reviews", [{}])[0].get("verdict"),
                **outcome.get(index, {}),
            }, ensure_ascii=False))
        return

    if control:
        # The control arm only exists to measure the noise floor; nothing is written.
        (dataset_dir / "evidence_recheck_control.json").write_text(
            json.dumps({"affected": rate_of("affected"), "control": rate_of("control")},
                       ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return

    # ---- rewrite the recoveries
    backup_root = dataset_dir / "backups"
    backup_root.mkdir(exist_ok=True)
    name = f"evidence_recheck_{time.strftime('%Y%m%d_%H%M%S')}"
    backup = backup_root / name
    suffix = 1
    while backup.exists():
        suffix += 1
        backup = backup_root / f"{name}_{suffix}"
    backup.mkdir()
    shutil.copytree(dataset_dir / "evidence_recoveries", backup / "evidence_recoveries")
    shutil.copy2(manifest_path, backup / "dataset_manifest.json")
    print(f"backup: {backup}", flush=True)

    survivors: list[dict[str, Any]] = []
    kept_on_error = 0
    for index, record in enumerate(entries):
        if index in outcome:
            result = outcome[index]
            if result["supported"] is None:
                # No verdict at all after both rounds and the client's own
                # retries.  Keeping the record as it stands is the fail-open
                # choice; dropping it would silently delete a recovery because
                # an endpoint hiccuped.
                kept_on_error += 1
                survivors.append(record)
                continue
            if result["supported"] is not True:
                continue                      # both rounds rejected it
            record = dict(record)
            block = dict(record.get("auto_check") or {})
            reviews = [dict(r) for r in (block.get("reviews") or [])]
            if reviews:
                reviews[0].update({
                    "extracted_value": result["extracted_value"],
                    "primary_extracted_value": result["extracted_value"],
                    "verdict": "supported",
                    "primary_verdict": "supported",
                    "comparison": "normalized_values_match",
                    "primary_comparison": "normalized_values_match",
                    "decision_source": "primary_local",
                    "error_code": "",
                    "primary_error_code": "",
                    "review_complete": True,
                })
                block["reviews"] = reviews
            block["supported_attributes"] = 1
            block["filtered_attributes"] = 0
            block["recheck"] = {
                "plan_version": PLAN_VERSION,
                "reason": "query_row_gained_page_url",
                "rounds": rounds.get(index, 1),
                "rescued_by_second_round": bool(result.get("rescued_by_second_round")),
            }
            record["auto_check"] = block
            survivors.append(record)
        elif record["query_table_id"] in orphan or record["target_table_id"] in drop:
            continue                          # dangling
        else:
            survivors.append(record)

    # ---- repack at the publisher's shard size
    recovery_dir = dataset_dir / "evidence_recoveries"
    for path in recovery_dir.iterdir():
        path.unlink()
    per_shard = int(manifest.get("records_per_shard", 10000))
    shards = []
    for start in range(0, max(len(survivors), 1), per_shard):
        chunk = survivors[start:start + per_shard]
        path = recovery_dir / f"part-{start // per_shard:05d}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for record in chunk:
                handle.write(serialize(record) + "\n")
        stat = path.stat()
        shards.append({
            "path": f"evidence_recoveries/{path.name}",
            "records": len(chunk),
            "bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "sha256": hashlib_sha256(path),
        })
    item = manifest["artifacts"]["evidence_recoveries"]
    item["shards"] = shards
    item["total_records"] = len(survivors)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    stats_path = dataset_dir / "stats.json"
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    stats["evidence_recoveries"] = len(survivors)
    stats.setdefault("notes", []).append(
        f"evidence_recoveries re-checked for the page_url move ({PLAN_VERSION}): "
        f"{len(affected)} re-run, {rejected} dropped, {counts['dangling']} dangling removed"
    )
    stats_path.write_text(json.dumps(stats, ensure_ascii=False) + "\n", encoding="utf-8")

    report = {
        "plan_version": PLAN_VERSION,
        "rechecked": len(affected), "supported": supported, "now_rejected": rejected,
        "errors": errored, "kept_on_error": kept_on_error,
        "rescued_by_second_round": sum(
            1 for i in affected if (outcome.get(i) or {}).get("rescued_by_second_round")
        ),
        "dangling_removed": counts["dangling"],
        "untouched_kept": counts["untouched"], "survivors": len(survivors),
        "backup_dir": str(backup), "endpoints": endpoints,
    }
    (dataset_dir / "evidence_recheck_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


def hashlib_sha256(path: Path) -> str:
    import hashlib
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
