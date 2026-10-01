"""Encode changed tables and compose feature stores within one fresh experiment."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import torch

from run_abebooks_fresh import ROOT, finish, launch, read_rows, write_json, write_rows
import run_stage1
from mmdd_stage1 import content as features

# Physical GPU index (PCI bus order) for encoding jobs; set from --gpu in main().
GPU = 0
from mmdd_dataset.abebooks_ablation import (commercial_text_assets, duplicate_book_image_assets, make_view, nongold_hubs, nongold_text_hubs, nonpositive_source_assets,
                                           unanchored_text_assets, unlabelled_text_assets, unsupported_source_assets)


def curate(root: Path) -> None:
    """Run after baseline training has published train-only popularity counts."""
    baseline = root / "baseline"
    dataset = baseline / "dataset_view"
    with gzip.open(baseline / "CONTENT_ALIASES.jsonl.gz", "rt") as handle:
        aliases = [json.loads(line) for line in handle]
    canonical = {r["asset_id"]: r["canonical_evidence_id"] for r in aliases}
    gold = {r["evidence"]["asset_id"] for r in read_rows(dataset / "evidence_recoveries/part-00000.jsonl")}
    popularity = read_rows(baseline / "train_evidence_popularity.jsonl")
    threshold = json.loads((baseline / "TRAIN_POPULARITY_READY.json").read_text())["threshold"]
    removed = nongold_hubs(popularity, canonical, gold, threshold)
    assets = {a["asset_id"]: a for a in read_rows(dataset / "bridge_assets/part-00000.jsonl")}
    write_json(root / "HUB_FILTER.json", {"threshold": threshold, "removed_raw": len(removed),
        "removed_canonical": len({canonical[e] for e in removed}), "removed_assets": sorted(removed),
        "gold_protected_raw": len(gold), "rule": "train Raw QE20 per modality; protect all gold aliases",
        "frequent_removed": [{**r, "asset_type": assets[r["evidence_id"]]["asset_type"],
            "content": (assets[r["evidence_id"]].get("content") or "")[:350],
            "local_path": assets[r["evidence_id"]].get("local_path")}
            for r in popularity if r["evidence_id"] in removed]})
    for arm, prune, filter_hubs in (("columns", True, False), ("hubs", False, True), ("both", True, True)):
        run = root / arm
        (run / "isolated_cwd").mkdir(parents=True)
        report = make_view(dataset, run / "dataset_view", prune_columns=prune,
                           removed_assets=removed if filter_hubs else set())
        write_json(run / "CURATION.json", report)
        protocol = json.loads((baseline / "protocol.json").read_text())
        protocol["paths"] = {**{k: v.replace(str(baseline), str(run)) for k, v in protocol["paths"].items()},
                             **run_stage1.feature_paths(run)}  # this arm recomposes its own features
        write_json(run / "protocol.json", protocol)


def build_inputs(run: Path) -> None:
    finish("build_data", launch(run, "build_data", [str(ROOT / "src/build_stage1_training_data.py"),
        "--dataset-root", str(run / "dataset_view"), "--dataset-name", "abebooks_joinability_bal04_clean",
        "--output-dir", str(run / "data"), "--max-rows", "20", "--seed", "13"], gpu=GPU))


def encode_tables(run: Path) -> None:
    build_inputs(run)
    tables = [r for r in read_rows(run / "data/stage1_objects.jsonl") if r["object_type"] == "table"]
    write_rows(run / "data/tables.jsonl", tables)
    finish("encode_tables", launch(run, "encode_tables", [str(ROOT / "src/cache_stage1_features.py"),
        "--input-jsonl", str(run / "data/tables.jsonl"), "--output-dir", str(run / "table_encoder"),
        "--model-dir", str(ROOT / "hf_models/Qwen3-VL-Embedding-8B"), "--device", "cuda:0",
        "--object-batch-size", "4", "--teacher-data", str(run / "data/tables.jsonl"),
        "--teacher-split", "all"], gpu=GPU))


def encode_changed_text(run: Path, reference: Path) -> None:
    """Re-encode every changed text input in both frozen feature tiers."""
    current = read_rows(run / "data/stage1_objects.jsonl")
    original = {r["object_id"]: r for r in read_rows(reference / "data/stage1_objects.jsonl")}
    changed = [r for r in current if r["object_type"] == "text" and r != original[r["object_id"]]]
    write_rows(run / "data/changed_text.jsonl", changed)
    finish("encode_changed_text", launch(run, "encode_changed_text", [str(ROOT / "src/cache_stage1_features.py"),
        "--input-jsonl", str(run / "data/changed_text.jsonl"), "--output-dir", str(run / "text_encoder"),
        "--model-dir", str(ROOT / "hf_models/Qwen3-VL-Embedding-8B"), "--device", "cuda:0",
        "--teacher-data", str(run / "data/changed_text.jsonl"), "--teacher-split", "all"], gpu=GPU))


def compose(run: Path) -> None:
    baseline = run.parent / "baseline"
    if not (run / "data/stage1_objects.jsonl").exists():
        build_inputs(run)
    objects = read_rows(run / "data/stage1_objects.jsonl")
    expected = {r["object_id"] for r in objects}
    changed = run.name in {"columns", "both"}
    table_encoder = run.parent / "columns/table_encoder" if changed else baseline / "encoder"
    original_tables = {r["object_id"]: r["table_parts"] for r in
                       read_rows(baseline / "data/stage1_objects.jsonl") if r["object_type"] == "table"}
    encoded_tables = {r["object_id"]: r["table_parts"] for r in
        read_rows(run.parent / "columns/data/tables.jsonl") if r["object_type"] == "table"} if changed else original_tables
    for obj in objects:
        if obj["object_type"] == "table":
            assert obj["table_parts"] == encoded_tables[obj["object_id"]]
    encoder = run / "encoder"
    encoder.mkdir()
    for filename, key in (("manifest.jsonl", "feature_path"), ("teacher_manifest.jsonl", "teacher_feature_path")):
        records = []
        for source, is_table in ((baseline / "encoder", False), (table_encoder, True)):
            for row in read_rows(source / filename):
                if row["object_id"] not in expected or (row["object_type"] == "table") != is_table:
                    continue
                path = encoder / row[key]
                path.parent.mkdir(parents=True, exist_ok=True)
                os.link(source / row[key], path)
                records.append(row)
        records.sort(key=lambda r: r["object_id"].encode())
        write_rows(encoder / filename, records)
    features.build_z_memmap(encoder, run / "features/z")
    table_content = []
    for row in read_rows(encoder / "teacher_manifest.jsonl"):
        if row["object_type"] == "table":
            payload = torch.load(encoder / row["teacher_feature_path"], weights_only=True)
            table_content.append((row["object_id"], "table", features.table_tokens(payload["hidden_states"]).numpy()))
    features.write_chunk(run / "table_content/chunks", 0, table_content)
    index = features.merge_chunk_dirs([run / "table_content/chunks", baseline / "content_shard0/chunks",
                                     baseline / "content_shard1/chunks"], run / "features/content/chunks")
    positions = [i for i, oid in enumerate(index["ids"]) if oid in expected]
    index = {k: [v[i] for i in positions] for k, v in index.items()}
    assert set(index["ids"]) == expected
    write_json(run / "features/content/index.json", index)
    write_json(run / "features/content/coverage.json", {"objects": len(expected), "filtered": True})
    zindex = json.loads((run / "features/z/z_index.json").read_text())
    assert set(zindex["ids"]) == expected
    assert np.isfinite(np.load(run / "features/z/z.f32.npy", mmap_mode="r")).all()
    write_json(run / "FEATURE_COMPOSITION.json", {"objects": len(expected),
        "tables_freshly_encoded_from": str(table_encoder), "evidence_freshly_encoded_from": str(baseline),
        "old_experiment_features_reused": False, "all_table_serializations_match_encoding_input": True,
        "shared_chunks": "immutable current-experiment evidence; removed IDs absent from every object index"})
    print(f"COMPOSE COMPLETE {run.name}: {len(expected)} objects", flush=True)


def compose_reference_evidence(run: Path, reference: Path) -> None:
    """Combine fresh tables/changed text with identical frozen evidence inputs."""
    objects = read_rows(run / "data/stage1_objects.jsonl")
    original = {r["object_id"]: r for r in read_rows(reference / "data/stage1_objects.jsonl")}
    expected = {r["object_id"] for r in objects}
    encoded = {r["object_id"]: r for r in read_rows(run / "data/tables.jsonl")}
    text_input = run / "data/changed_text.jsonl"
    changed_text = {r["object_id"]: r for r in read_rows(text_input)} if text_input.exists() else {}
    for obj in objects:
        source = encoded if obj["object_type"] == "table" else (
            changed_text if obj["object_id"] in changed_text else original)
        if obj != source[obj["object_id"]]:
            raise ValueError(f"Feature input changed: {obj['object_id']}")
    encoder = run / "encoder"
    encoder.mkdir(exist_ok=False)
    for filename, key in (("manifest.jsonl", "feature_path"), ("teacher_manifest.jsonl", "teacher_feature_path")):
        records = []
        sources = [(reference / "encoder", expected - set(encoded) - set(changed_text)),
                   (run / "table_encoder", set(encoded))]
        if changed_text:
            sources.append((run / "text_encoder", set(changed_text)))
        for source, selected in sources:
            for row in read_rows(source / filename):
                if row["object_id"] not in selected:
                    continue
                target = encoder / row[key]
                target.parent.mkdir(parents=True, exist_ok=True)
                os.link(source / row[key], target)
                records.append(row)
        write_rows(encoder / filename, sorted(records, key=lambda r: r["object_id"].encode()))
    features.build_z_memmap(encoder, run / "features/z")
    rows, chunk = [], 0
    for record in read_rows(encoder / "teacher_manifest.jsonl"):
        if record["object_type"] == "table" or record["object_id"] in changed_text:
            payload = torch.load(encoder / record["teacher_feature_path"], weights_only=True)
            compress = features.table_tokens if record["object_type"] == "table" else features.compress_bins
            rows.append((record["object_id"], record["object_type"], compress(payload["hidden_states"]).numpy()))
            if len(rows) == 64:
                features.write_chunk(run / "table_content/chunks", chunk, rows)
                rows, chunk = [], chunk + 1
    if rows:
        features.write_chunk(run / "table_content/chunks", chunk, rows)
    index = features.merge_chunk_dirs([run / "table_content/chunks", reference / "features/content/chunks"],
                                      run / "features/content/chunks")
    positions = [i for i, oid in enumerate(index["ids"]) if oid in expected]
    index = {k: [v[i] for i in positions] for k, v in index.items()}
    assert set(index["ids"]) == expected and len(index["ids"]) == len(expected)
    assert set(json.loads((run / "features/z/z_index.json").read_text())["ids"]) == expected
    write_json(run / "features/content/index.json", index)
    write_json(run / "features/content/coverage.json", {"objects": len(expected)})
    write_json(run / "FEATURE_COMPOSITION.json", {"reference": str(reference),
        "table_encoder": str((run / "table_encoder").resolve()),
        "all_evidence_inputs_equal": not changed_text, "changed_text_freshly_encoded": len(changed_text),
        "all_reused_evidence_inputs_equal": True, "all_table_inputs_equal_to_encoder_inputs": True,
        "reused_frozen_evidence_features": True, "reused_trainable_weights": False,
        "objects": len(expected), "tables": len(encoded)})
    previous = json.loads((run / "FRESH_INPUTS.json").read_text()) if (run / "FRESH_INPUTS.json").exists() else {}
    write_json(run / "FRESH_INPUTS.json", {"dataset": str(run / "dataset_view"), **previous,
        "old_features_reused": True, "reuse_scope": "identical frozen evidence inputs only",
        "reference": str(reference), "old_checkpoints_reused": False})


def reuse_identical_tables(run: Path, reference: Path) -> None:
    """Reuse frozen table tensors after an ID/split-only dataset reconstruction."""
    from cache_stage1_features import _source_fingerprint

    current = [r for r in read_rows(run / "data/stage1_objects.jsonl") if r["object_type"] == "table"]
    original = {r["object_id"]: r for r in read_rows(reference / "data/tables.jsonl")}
    old_ids = {oid.rsplit("_source_", 1)[0]: oid for oid in original}
    mapping = {}
    for obj in current:
        old_id = old_ids[obj["object_id"].rsplit("_source_", 1)[0]]
        assert {k: v for k, v in obj.items() if k != "object_id"} == {
            k: v for k, v in original[old_id].items() if k != "object_id"}
        mapping[old_id] = obj
    assert len(mapping) == len(current) == len(original)
    destination = run / "table_encoder"
    destination.mkdir(exist_ok=False)
    for filename, key in (("manifest.jsonl", "feature_path"), ("teacher_manifest.jsonl", "teacher_feature_path")):
        records = []
        for row in read_rows(reference / "table_encoder" / filename):
            assert row["source_fingerprint"] == _source_fingerprint(original[row["object_id"]])
            obj = mapping[row["object_id"]]
            relative = str(Path(row[key]).parent / (hashlib.sha256(obj["object_id"].encode()).hexdigest() + ".pt"))
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            os.link(reference / "table_encoder" / row[key], target)
            records.append({**row, "object_id": obj["object_id"], key: relative,
                            "source_fingerprint": _source_fingerprint(obj)})
        assert len(records) == len(current)
        write_rows(destination / filename, records)
    write_rows(run / "data/tables.jsonl", current)
    write_json(run / "TABLE_INPUT_PARITY.json", {"reference": str(reference), "tables": len(current),
        "all_encoder_inputs_except_ids_equal": True, "frozen_tensors_reused": True,
        "learned_weights_reused": False})
    compose_reference_evidence(run, reference / "main")
    for filename in ("FEATURE_COMPOSITION.json", "FRESH_INPUTS.json"):
        record = json.loads((run / filename).read_text())
        record.update(reused_frozen_table_inputs=True, reuse_scope="identical frozen table and evidence inputs",
                      table_input_parity=str(run / "TABLE_INPUT_PARITY.json"))
        write_json(run / filename, record)


def curate_reference_evidence(run: Path, reference: Path, *, filter_kind: str) -> None:
    """Freeze an evidence filter; keep every query, target and gold fact."""
    if run.exists():
        raise FileExistsError(run)
    (run / "isolated_cwd").mkdir(parents=True)
    with gzip.open(reference / "CONTENT_ALIASES.jsonl.gz", "rt") as handle:
        canonical = {r["asset_id"]: r["canonical_evidence_id"] for r in map(json.loads, handle)}
    dataset = reference / "dataset_view"
    recoveries = read_rows(dataset / "evidence_recoveries/part-00000.jsonl")
    gold = {r["evidence"]["asset_id"] for r in recoveries}
    queries = read_rows(dataset / "query_tables/part-00000.jsonl")
    train_count = sum(q["split"] == "train" for q in queries)
    threshold = math.ceil(train_count * 0.1)
    popularity = read_rows(reference / "train_evidence_popularity.jsonl")
    if filter_kind == "unlabelled_text":
        removed = unlabelled_text_assets(read_rows(dataset / "bridge_assets/part-00000.jsonl"),
                                         recoveries, canonical)
        rule = "Evidence catalog ablation: remove text without an existing recovery label or labelled content alias; retain all images and all existing facts"
    elif filter_kind == "duplicate_book_images":
        removed = duplicate_book_image_assets(read_rows(dataset / "source_tables/part-00000.jsonl"),
            read_rows(dataset / "bridge_assets/part-00000.jsonl"), recoveries, canonical)
        rule = "Within identical title/author/publisher/year records, remove image content classes with mean pixel difference <= 3/255 after complete-image RGB 128x128 resize; protect every labelled content class"
    elif filter_kind == "text_hubs":
        removed = nongold_text_hubs(read_rows(dataset / "bridge_assets/part-00000.jsonl"),
                                    popularity, canonical, gold, threshold)
        rule = "Remove text content classes retrieved by at least 10% of TRAIN queries at Raw QE20; retain all images and all labelled content aliases"
    elif filter_kind == "positive_sources":
        removed = nonpositive_source_assets(read_rows(dataset / "bridge_assets/part-00000.jsonl"),
            read_rows(dataset / "qrels.jsonl"), recoveries, canonical)
        rule = "Retain all evidence of sources with any current explicit or implicit positive qrel, plus all labelled aliases; queries and all lake tables remain unchanged"
    elif filter_kind == "title_anchored_text":
        removed = unanchored_text_assets(read_rows(dataset / "source_tables/part-00000.jsonl"),
            read_rows(dataset / "bridge_assets/part-00000.jsonl"), recoveries, canonical)
        rule = "Text must mention two of the three rarest source-title words occurring in at most 1% of books (one for a one-word anchor); retain images and all gold content aliases"
    elif filter_kind == "commercial_text":
        removed = commercial_text_assets(read_rows(dataset / "bridge_assets/part-00000.jsonl"),
                                         recoveries, canonical)
        rule = "Remove seller descriptions, sales policies and shipping policies by source provenance; retain all labelled content aliases"
    elif filter_kind == "supported_sources":
        removed = unsupported_source_assets(read_rows(dataset / "bridge_assets/part-00000.jsonl"),
                                            recoveries, canonical)
        rule = "Retain all assets of source tables with any currently approved evidence/value fact, plus gold aliases"
    else:
        removed = nongold_hubs(popularity, canonical, gold, threshold)
        rule = "At least 10% distinct TRAIN queries at Raw QE20; exclude all-split gold aliases"
    report = make_view(dataset, run / "dataset_view", prune_columns=False, removed_assets=removed)
    filename = "HUB_FILTER.json" if filter_kind == "hubs" else "EVIDENCE_FILTER.json"
    write_json(run / filename, {"reference": str(reference), "train_queries": train_count,
        "filter_kind": filter_kind,
        "threshold": threshold if filter_kind in {"hubs", "text_hubs"} else None, "rule": rule,
        "label_scope": "Dataset curation uses existing all-split annotations (qrels for positive_sources); not an unlabeled deployment filter",
        "removed_assets": sorted(removed), "curation": report})
    protocol = json.loads((reference / "protocol.json").read_text())
    protocol["paths"] = {**{k: v.replace(str(reference), str(run)) for k, v in protocol["paths"].items()},
                         **run_stage1.feature_paths(run)}  # this run recomposes its own features
    write_json(run / "protocol.json", protocol)
    build_inputs(run)
    tables = [r for r in read_rows(run / "data/stage1_objects.jsonl") if r["object_type"] == "table"]
    write_rows(run / "data/tables.jsonl", tables)
    encoder = json.loads((reference / "FEATURE_COMPOSITION.json").read_text()).get("table_encoder")
    if encoder is None:
        encoder = str(reference.parent / "table_encoder")
    (run / "table_encoder").symlink_to(encoder, target_is_directory=True)
    compose_reference_evidence(run, reference)
    for name in ("query_tables", "data_lake_tables", "source_tables", "evidence_recoveries"):
        assert read_rows(run / f"dataset_view/{name}/part-00000.jsonl") == read_rows(dataset / name / "part-00000.jsonl")
    assert read_rows(run / "dataset_view/qrels.jsonl") == read_rows(dataset / "qrels.jsonl")


def main() -> None:
    filters = {"curate-reference-hubs": "hubs",
               "curate-reference-unlabelled-text": "unlabelled_text",
               "curate-reference-duplicate-book-images": "duplicate_book_images",
               "curate-reference-text-hubs": "text_hubs",
               "curate-reference-supported-sources": "supported_sources",
               "curate-reference-bibliographic-evidence": "commercial_text",
               "curate-reference-title-anchored-text": "title_anchored_text",
               "curate-reference-positive-sources": "positive_sources"}
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["curate", "encode-tables", "encode-changed-text", "compose", "compose-reference", "reuse-identical-tables", *filters])
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--gpu", type=int, default=0, help="physical GPU index (PCI bus order) for encoding jobs")
    args = parser.parse_args()
    global GPU
    GPU = args.gpu
    if args.command in {"compose-reference", "reuse-identical-tables", "encode-changed-text"} or args.command in filters:
        if args.reference is None:
            parser.error(f"{args.command} requires --reference")
        if args.command == "encode-changed-text":
            encode_changed_text(args.run_root.resolve(), args.reference.resolve())
        elif args.command == "reuse-identical-tables":
            reuse_identical_tables(args.run_root.resolve(), args.reference.resolve())
        elif args.command == "compose-reference":
            compose_reference_evidence(args.run_root.resolve(), args.reference.resolve())
        else:
            curate_reference_evidence(args.run_root.resolve(), args.reference.resolve(),
                                      filter_kind=filters[args.command])
        return
    {"curate": curate, "encode-tables": encode_tables, "compose": compose}[args.command](args.run_root.resolve())


if __name__ == "__main__":
    main()
