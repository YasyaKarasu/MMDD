"""Compare source-level evidence catalogs on dev with one fixed Student.

This is a retrieval diagnostic, not a trained result for a new dataset. It
keeps the checkpoint, query/target objects and retrieval algorithm fixed.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from collections import defaultdict
from pathlib import Path

import torch

from mmdd_stage1 import pipeline
from mmdd_stage1.evaluate import evaluate_student_retrieval
from mmdd_stage1.labels import export_eval_labels
from mmdd_dataset.abebooks_ablation import duplicate_book_image_assets, nongold_text_hubs, read_rows, unanchored_text_assets
from run_abebooks_data_ablation import runtime
from run_abebooks_fresh import write_json


def audit(run: Path, output: Path) -> list[dict]:
    rt = runtime(run)
    freeze = json.loads((run / "SELECTION_FREEZE.json").read_text())
    model = pipeline._load_native(Path(freeze["selected"]["KD"]), rt, device="cpu")
    dataset = run / "dataset_view"
    assets = read_rows(dataset / "bridge_assets/part-00000.jsonl")
    recoveries = read_rows(dataset / "evidence_recoveries/part-00000.jsonl")
    qrels = read_rows(dataset / "qrels.jsonl")
    bridge_sources = {r["source_table_id"] for r in recoveries}
    positive_sources = {r["source_table_id"] for r in qrels if r["rel"] > 0}
    canonical = rt.labels.canonical_map
    protected = {canonical[r["evidence"]["asset_id"]] for r in recoveries}
    all_evidence = set(canonical.values())
    unanchored = unanchored_text_assets(read_rows(dataset / "source_tables/part-00000.jsonl"),
                                        assets, recoveries, canonical)
    train_count = sum(q["split"] == "train" for q in read_rows(dataset / "query_tables/part-00000.jsonl"))
    text_hubs = nongold_text_hubs(assets, read_rows(run / "train_evidence_popularity.jsonl"), canonical,
                                 {r["evidence"]["asset_id"] for r in recoveries}, math.ceil(train_count * 0.1))
    copies = duplicate_book_image_assets(read_rows(dataset / "source_tables/part-00000.jsonl"),
                                         assets, recoveries, canonical)
    catalogs = {"all": all_evidence,
        "bridge_sources": {canonical[a["asset_id"]] for a in assets if a["source_table_id"] in bridge_sources} | protected,
        "positive_target_sources": {canonical[a["asset_id"]] for a in assets if a["source_table_id"] in positive_sources} | protected,
        "bibliographic_evidence": {canonical[a["asset_id"]] for a in assets
            if a["source"] in {"abebooks_synopsis", "abebooks_about_author",
                               "abebooks_catalogue_cover", "abebooks_seller_cover"}} | protected,
        "image_evidence": {canonical[a["asset_id"]] for a in assets if a["asset_type"] == "image"} | protected,
        "title_anchored_text": {canonical[a["asset_id"]] for a in assets if a["asset_id"] not in unanchored},
        "text_hubs_removed": {canonical[a["asset_id"]] for a in assets if a["asset_id"] not in text_hubs},
        "duplicate_book_images_removed": {canonical[a["asset_id"]] for a in assets if a["asset_id"] not in copies}}
    gt = export_eval_labels(rt.paths, canonical, "dev")
    units = defaultdict(set)
    for record in recoveries:
        if record["split"] == "dev":
            key = (record["query_table_id"], record["target_table_id"], record["query_row_id"],
                   record["recovered_attribute"]["column_name"], record["recovered_attribute"]["value"])
            units[key].add(canonical[record["evidence"]["asset_id"]])
    results = []
    for name, kept in catalogs.items():
        labels = copy.copy(rt.labels)
        labels.canonical_text = [e for e in labels.canonical_text if e in kept]
        labels.canonical_image = [e for e in labels.canonical_image if e in kept]
        pools = evaluate_student_retrieval(model, rt.z_store, rt.row_store, sorted(gt), labels, "dev",
                                           device="cpu", generator_id=f"fixed_model_{name}")
        recall = {q: len(set(p.C150[:10]) & set(gt[q]["G"])) / len(gt[q]["G"]) for q, p in pools.items()}
        first = {q: {e for hits in p.first_hop.values() for e, _ in hits} for q, p in pools.items()}
        result = {"catalog": name, "canonical_assets": len(kept),
            "R@10": {kind: sum(recall[q] for q in gt if kind == "overall" or gt[q]["kind"] == kind) /
                     sum(kind == "overall" or gt[q]["kind"] == kind for q in gt)
                     for kind in ("overall", "implicit", "explicit")},
            "first_hop_units": sum(bool(eids & first[key[0]]) for key, eids in units.items()),
            "retained_units": sum(bool(eids & set(pools[key[0]].retained_paths.get(key[1], [])))
                                   for key, eids in units.items())}
        results.append(result)
        write_json(output, {"split": "dev", "fixed_checkpoint": freeze["selected"]["KD"],
            "scope": "CPU fixed-model catalog diagnostic; no retraining or test selection", "results": results})
        print(json.dumps(result), flush=True)
    return results


def annotated_path_ranks(run: Path, output: Path) -> dict:
    """Locate first/second-hop failures using existing train/dev facts only."""
    rt = runtime(run)
    freeze = json.loads((run / "SELECTION_FREEZE.json").read_text())
    model = pipeline._load_native(Path(freeze["selected"]["KD"]), rt, device="cpu")
    model.eval()
    recoveries = [r for r in read_rows(run / "dataset_view/evidence_recoveries/part-00000.jsonl")
                  if r["split"] in {"train", "dev"}]
    targets = sorted(rt.labels.legal_targets, key=lambda x: x.encode())
    evidence = {m: sorted(getattr(rt.labels, f"canonical_{m}"), key=lambda x: x.encode())
                for m in ("text", "image")}
    first, second, records = {}, {}, []
    with torch.no_grad():
        target_vectors = model.index_vectors("QT", rt.z_store.rows(targets))
        evidence_vectors = {m: model.index_vectors(f"Q_{m}", rt.z_store.rows(ids))
                            for m, ids in evidence.items()}
        for record in recoveries:
            qid, tid = record["query_table_id"], record["target_table_id"]
            eid = rt.labels.canonical_map[record["evidence"]["asset_id"]]
            modality = record["evidence"]["asset_type"]
            if eid not in second:
                scores = model.ann_query(f"{modality}_T", rt.z_store.vector(eid)) @ target_vectors.T
                second[eid] = {targets[i]: rank for rank, i in enumerate(
                    torch.argsort(scores, descending=True, stable=True).tolist(), 1)}
            if (qid, modality) not in first:
                scores = model.ann_query(f"Q_{modality}", rt.z_store.vector(qid)) @ evidence_vectors[modality].T
                first[qid, modality] = {evidence[modality][i]: rank for rank, i in enumerate(
                    torch.argsort(scores, descending=True, stable=True).tolist(), 1)}
            records.append({"query_id": qid, "target_id": tid, "evidence_id": eid,
                "split": record["split"], "attribute": record["recovered_attribute"]["column_name"],
                "source_row": record["source_row_id"], "qe_exact_rank": first[qid, modality][eid],
                "et_exact_rank": second[eid][tid]})
    summary = {}
    for split in ("train", "dev"):
        selected = [r for r in records if r["split"] == split]
        summary[split] = {"recovery_records": len(selected),
            "QE_top20": sum(r["qe_exact_rank"] <= 20 for r in selected),
            "ET_top50": sum(r["et_exact_rank"] <= 50 for r in selected),
            "both": sum(r["qe_exact_rank"] <= 20 and r["et_exact_rank"] <= 50 for r in selected)}
    result = {"scope": "Exact-score diagnosis on existing train/dev facts; no new labels or oracle retrieval result",
              "checkpoint": freeze["selected"]["KD"], "summary": summary, "records": records}
    write_json(output, result)
    print(json.dumps(summary), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--path-ranks-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    (annotated_path_ranks if args.path_ranks_only else audit)(args.run_root.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
