#!/usr/bin/env python
"""Rebuild historical indices and measure full-lake funnels and exact neighbors."""

from __future__ import annotations

import argparse
import gc
import gzip
import json
import shutil
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from diagnose_stage1_r11_exact_search import _positive_sets, _sample_by_source, _source_groups
from mmdd_stage1.artifacts import checkpoint_fingerprint, write_json
from mmdd_stage1.checkpoints import load_student
from mmdd_stage1.data import load_edge_examples, load_target_examples
from mmdd_stage1.evaluation import evaluate_student_retrieval
from mmdd_stage1.features import FeatureStore
from mmdd_stage1.pca import load_pca_projection
from mmdd_stage1.retrieval import StudentANNIndices, build_indices, load_corpus_ids
from mmdd_stage1.scoring import score_edge_batch
from mmdd_stage1.edge_metrics import summarize_edge_quality


def _requests(targets, edges, dataset_root: Path) -> list[tuple]:
    groups = _source_groups(dataset_root)
    selected_queries = []
    for kind in ("implicit", "explicit"):
        selected_queries.extend(_sample_by_source(
            [row for row in targets if row.query_kind == kind],
            lambda row: groups[row.query_id], count=64, seed=f"13:query:{kind}",
        ))
    evidence_groups = {}
    for row in targets:
        for ids in (row.positive_evidence_by_target or {}).values():
            for evidence_id in ids:
                evidence_groups.setdefault(evidence_id, groups[row.query_id])
    selected_evidence = []
    for kind in ("text", "image"):
        unique = {row.query_id: row for row in reversed(edges)
                  if row.source_type == kind and row.destination_type == "table"
                  and row.query_id in evidence_groups}
        selected_evidence.extend(_sample_by_source(
            list(unique.values()), lambda row: evidence_groups[row.query_id],
            count=64, seed=f"13:evidence:{kind}",
        ))
    query_positives, edge_positives = _positive_sets(targets, edges)
    return [
        (f"table_to_{kind}", [row.query_id for row in selected_queries], "table", kind,
         100 if kind == "table" else 20, query_positives)
        for kind in ("table", "text", "image")
    ] + [
        (f"{kind}_to_table", [row.query_id for row in selected_evidence if row.source_type == kind],
         kind, "table", 20, edge_positives) for kind in ("text", "image")
    ]


def _quantiles(values: torch.Tensor) -> dict[str, float]:
    return dict(zip(("p10", "p50", "p90"),
                    torch.quantile(values.float(), torch.tensor([0.1, 0.5, 0.9], device=values.device)).cpu().tolist()))


@torch.inference_mode()
def exact_relation(model, indices, store, ids_by_type, request, basis, device):
    name, source_ids, source_type, destination_type, k, positives = request
    destination_ids = ids_by_type[destination_type]
    columns = {value: index for index, value in enumerate(destination_ids)}
    embeddings = torch.stack([store.embedding_features(value).embedding for value in destination_ids]).to(device)
    vectors = model.index_vector(
        embeddings,
        destination_type,
        destination_role="target" if destination_type == "table" else None,
    )
    reference_vectors = F.linear(embeddings, basis)
    geometry = {"destination_projected_norm": _quantiles(vectors.norm(dim=1)),
                "destination_pca_direction_cosine": _quantiles(F.cosine_similarity(vectors, reference_vectors, dim=1))}
    del embeddings, reference_vectors
    ann = indices.search_many(source_ids, destination_type, k)
    rows = []
    for start in range(0, len(source_ids), 16):
        batch_ids = source_ids[start:start + 16]
        embeddings = torch.stack([store.embedding_features(value).embedding for value in batch_ids]).to(device)
        projected = model.project(
            embeddings,
            source_type,
            role="query" if source_type == "table" else None,
        )
        source_vectors = model.relation_query(
            embeddings,
            source_type,
            destination_type,
            source_role="query" if source_type == "table" else None,
        )
        pca_vectors = F.linear(embeddings, basis)
        scores = source_vectors @ vectors.T
        if not bool(torch.isfinite(scores).all()):
            raise ValueError(f"{name}: nonfinite exact scores")
        values, positions = torch.topk(scores, min(k + 1, scores.shape[1]), dim=1)
        for index, source_id in enumerate(batch_ids):
            source_ann = ann[start + index]
            ann_ids = [value for value, score in source_ann]
            exact = sorted(zip(positions[index, :k].cpu().tolist(), values[index, :k].cpu().tolist()),
                           key=lambda pair: (-pair[1], destination_ids[pair[0]]))
            exact_ids = [destination_ids[column] for column, score in exact]
            relevant = positives.get((source_id, destination_type), set())
            present = sorted(relevant & columns.keys())
            gt = []
            for value in present:
                gt_score = scores[index, columns[value]]
                greater = int((scores[index] > gt_score).sum())
                equal = int((scores[index] == gt_score).sum())
                gt.append({"target_id": value, "score": float(gt_score), "rank_min": greater + 1,
                           "rank_max": greater + equal,
                           "ann_rank": ann_ids.index(value) + 1 if value in ann_ids else None})
            cutoff = values[index, min(k, scores.shape[1]) - 1]
            rows.append({
                "source_id": source_id, "positive_ids": sorted(relevant),
                "positive_neighbors": len(relevant), "missing_positive_ids": sorted(relevant - columns.keys()),
                "ann_positive_count": len(set(ann_ids) & relevant),
                "exact_positive_count": len(set(exact_ids) & relevant),
                "ann_any_positive_hit": int(bool(set(ann_ids) & relevant)),
                "exact_any_positive_hit": int(bool(set(exact_ids) & relevant)),
                "ann_ids": ann_ids, "ann_scores": [float(score) for value, score in source_ann],
                "exact_ids": exact_ids, "exact_scores": [float(score) for column, score in exact],
                "topk_overlap_count": len(set(ann_ids) & set(exact_ids)),
                "tied_at_exact_cutoff": int((scores[index] == cutoff).sum()),
                "cutoff_score": float(cutoff),
                "ann_below_exact_cutoff_count": sum(float(scores[index, columns[value]]) < float(cutoff) - 1e-6 for value in ann_ids),
                "source_projected_norm": float(projected[index].norm()),
                "source_relation_query_norm": float(source_vectors[index].norm()),
                "source_pca_direction_cosine": float(F.cosine_similarity(projected[index], pca_vectors[index], dim=0)),
                "relation_query_pca_direction_cosine": float(F.cosine_similarity(source_vectors[index], pca_vectors[index], dim=0)),
                "gt_edges": gt,
            })
    eligible = [row for row in rows if row["positive_neighbors"]]
    denominator = sum(row["positive_neighbors"] for row in rows)
    hubs = Counter(value for row in rows for value in row["ann_ids"])
    return {
        "relation": name, "k": k, "sources": len(rows), "eligible_sources": len(eligible),
        "positive_neighbors": denominator,
        "ann_positive_count": sum(row["ann_positive_count"] for row in rows),
        "exact_positive_count": sum(row["exact_positive_count"] for row in rows),
        "ann_positive_recall_micro": sum(row["ann_positive_count"] for row in rows) / denominator if denominator else None,
        "exact_positive_recall_micro": sum(row["exact_positive_count"] for row in rows) / denominator if denominator else None,
        "ann_positive_recall_macro": sum(row["ann_positive_count"] / row["positive_neighbors"] for row in eligible) / len(eligible) if eligible else None,
        "exact_positive_recall_macro": sum(row["exact_positive_count"] / row["positive_neighbors"] for row in eligible) / len(eligible) if eligible else None,
        "ann_any_positive_hits": sum(row["ann_any_positive_hit"] for row in eligible),
        "exact_any_positive_hits": sum(row["exact_any_positive_hit"] for row in eligible),
        "ann_any_positive_hit_rate": sum(row["ann_any_positive_hit"] for row in eligible) / len(eligible) if eligible else None,
        "exact_any_positive_hit_rate": sum(row["exact_any_positive_hit"] for row in eligible) / len(eligible) if eligible else None,
        "topk_overlap": sum(row["topk_overlap_count"] for row in rows) / (k * len(rows)),
        "hub_top20": hubs.most_common(20), "geometry": geometry, "per_source": rows,
        "tie_policy": "torch.topk boundary ties are reported, not assumed to disprove ANN error",
    }


@torch.inference_mode()
def list_metrics(model, edges, store, device):
    ranking = []
    records = []
    for start in range(0, len(edges), 64):
        batch = edges[start:start + 64]
        scores = score_edge_batch(model, batch, store, device).logits.cpu()
        for example, row in zip(batch, scores):
            values = row[:len(example.candidate_ids)].tolist()
            ranking.append(values)
            records.append({"source_id": example.query_id, "source_type": example.source_type,
                            "destination_type": example.destination_type, "candidate_ids": example.candidate_ids,
                            "positive_ids": example.positive_ids, "raw_scores": values})
    confidence = [torch.sigmoid(torch.tensor(row)).tolist() for row in ranking]
    return summarize_edge_quality(edges, ranking, confidence, recall_ks=(1, 5, 10, 20)), records


def _explicit_checkpoints(specifications: list[str]) -> list[tuple[str, Path]]:
    checkpoints = []
    seen = set()
    for specification in specifications:
        if "=" not in specification:
            raise ValueError("--checkpoint must use ID=PATH")
        checkpoint_id, path_text = specification.split("=", 1)
        if not checkpoint_id or checkpoint_id in seen or Path(checkpoint_id).name != checkpoint_id:
            raise ValueError(f"Invalid or duplicate checkpoint ID: {checkpoint_id!r}")
        seen.add(checkpoint_id)
        checkpoints.append((checkpoint_id, Path(path_text)))
    return checkpoints


def _r12_checkpoints(output_root: Path, arm: str) -> list[tuple[str, Path]]:
    short_arm = arm.removeprefix("r12_")
    extension = short_arm.endswith("_extension")
    if extension:
        short_arm = short_arm.removesuffix("_extension")
        folder = (
            output_root
            / "taskC_training"
            / f"c_{short_arm}_extension_seed13/checkpoints"
        )
        steps = (659, 1318)
    else:
        folder = output_root / "taskC_training" / f"c_{short_arm}_seed13/checkpoints"
        steps = (0, 178, 356)
    return [(f"step{step}", folder / f"step_{step:06d}.pt") for step in steps]


def _optimizer_updates(checkpoint_id: str, path: Path) -> int:
    metrics_path = path.with_suffix(".json")
    if metrics_path.is_file():
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        if "optimizer_updates" in payload:
            return int(payload["optimizer_updates"])
    if checkpoint_id.startswith("step") and checkpoint_id[4:].isdigit():
        return int(checkpoint_id[4:])
    return 0


def run(args: argparse.Namespace) -> None:
    torch.set_num_threads(args.cpu_threads)
    root, output_root = args.root, args.output_root
    r11 = root / "work/stage1_optimization_r11_20260908"
    r10 = root / "work/stage1_optimization_r10_20260907"
    supervision = output_root / "taskA_correctness/supervision"
    targets = load_target_examples(supervision / "target_lists.dev.jsonl", split="dev")
    edges = load_edge_examples(supervision / "edge_lists.dev.jsonl", split="dev")
    dataset_root = Path(json.loads((supervision / "manifest.json").read_text())["dataset_root"])
    requests = _requests(targets, edges, dataset_root)
    sample_ids = {name: ids for name, ids, *_ in requests}
    r12_arm = args.arm.startswith("r12_")
    output = (
        output_root / "taskC_training/full_lake_evaluations" / args.arm
        if r12_arm or args.checkpoint
        else output_root / "taskA_correctness/checkpoints" / args.arm
    )
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "exact_samples.json", sample_ids)
    store = FeatureStore.from_path(r10 / "features_qwen3_vl_embedding_8b", cache_size=260000)
    corpus = r10 / "stage1_data/stage1_corpus.jsonl"
    corpus_hash = checkpoint_fingerprint(corpus)
    ids_by_type = load_corpus_ids(corpus, store)
    preload_started = time.monotonic()
    store.preload_embeddings([*(value for ids in ids_by_type.values() for value in ids), *(row.query_id for row in targets)])
    preload_seconds = time.monotonic() - preload_started
    device = torch.device(args.device)
    basis = load_pca_projection(
        r10 / "baselines/pca_entitables_v9_1024.pt",
        input_dim=store.embedding_dimension(), student_dim=1024,
    )
    basis = basis.to(device)
    if args.checkpoint:
        checkpoints = _explicit_checkpoints(args.checkpoint)
    elif r12_arm:
        checkpoints = _r12_checkpoints(output_root, args.arm)
    elif args.arm == "pca":
        checkpoints = [("initial", r11 / "taskA_protocol/baselines/pca_init.pt")]
    else:
        folder = r11 / ("taskD_controls/c2_d1_long" if args.arm == "d1_long" else f"taskC_clean/{args.arm}")
        checkpoints = [(f"edge_epoch{epoch}", folder / f"student_edge.epochs/epoch_{epoch:03d}.pt") for epoch in (1, 2)]
        checkpoints += [(f"path_epoch{epoch}", folder / f"student_path.epochs/epoch_{epoch:03d}.pt") for epoch in (0, 1, 2)]
        resolved = []
        for name, path in checkpoints:
            if not path.exists():
                stage, epoch_text = name.split("_epoch")
                epoch = int(epoch_text)
                history = json.loads((folder / f"student_{stage}.pt.history.json").read_text())["epochs"]
                entry = next(row for row in history if row["epoch"] == epoch)
                if epoch == history[-1]["epoch"]:
                    path = folder / f"student_{stage}.last.pt"
                elif stage == "path":
                    path = output_root / "taskA_correctness/reconstructed" / args.arm / f"epoch_{epoch:03d}.pt"
                if path.exists() and checkpoint_fingerprint(path) != entry["candidate_checkpoint_sha256"]:
                    raise ValueError(f"Historical checkpoint identity differs: {path}")
            resolved.append((name, path))
        checkpoints = resolved
    for name, path in checkpoints:
        if args.checkpoints and name not in args.checkpoints:
            continue
        if not path.is_file():
            raise FileNotFoundError(f"Missing checkpoint: {path}")
        target = output / name
        if (target / "metrics.json").is_file():
            print(f"Completed checkpoint retained: {target}", flush=True)
            continue
        target.mkdir(parents=True, exist_ok=True)
        started = time.monotonic()
        model = load_student(path, device).eval()
        weight_hash = checkpoint_fingerprint(path)
        index_dir = target / "index"
        if index_dir.exists():
            raise FileExistsError(f"Inspect existing unfinished index before resuming: {index_dir}")
        index_started = time.monotonic()
        index_manifest = build_indices(model, store, ids_by_type, index_dir, device=device,
                                       checkpoint_sha256=weight_hash, corpus_sha256=corpus_hash,
                                       batch_size=4096, num_threads=args.index_threads)
        index_seconds = time.monotonic() - index_started
        index_bytes = sum(value.stat().st_size for value in index_dir.iterdir() if value.is_file())
        indices = StudentANNIndices(model, store, index_dir, device=device, checkpoint_sha256=weight_hash,
                                    corpus_sha256=corpus_hash, score_space="raw_logit")
        before = time.monotonic()
        retrieval = evaluate_student_retrieval(
            targets, indices, recall_ks=(10, 20, 50), direct_k=100, evidence_k=20, targets_per_evidence=20,
            evidence_types=("text", "image"), evidence_aggregation="logsumexp", evidence_top_k=4,
            evidence_temperature=1.0, path_combination="sum", return_per_query=True,
        )
        retrieval_seconds = time.monotonic() - before
        before = time.monotonic()
        lists, list_predictions = list_metrics(model, edges, store, device)
        with gzip.open(target / "list_predictions.jsonl.gz", "wt") as handle:
            for row in list_predictions:
                handle.write(json.dumps(row) + "\n")
        del list_predictions
        exact = {request[0]: exact_relation(model, indices, store, ids_by_type, request, basis, device)
                 for request in requests}
        payload = {
            "format_version": 1, "arm": args.arm, "checkpoint_id": name,
            "checkpoint": str(path), "checkpoint_sha256": weight_hash,
            "identity": (
                "R12 checkpoint evaluation with exact checkpoint/index fingerprint binding"
                if r12_arm or args.checkpoint
                else "historical inference only, no retraining or anchor repair to old weights"
            ),
            "exact_sample_sha256": checkpoint_fingerprint(output / "exact_samples.json"),
            "corpus_sha256": corpus_hash, "index_manifest": index_manifest,
            "retrieval": retrieval, "constructed_list_metrics": lists, "exact_diagnostics": exact,
            "cost": {"index_build_seconds": index_seconds, "index_bytes": index_bytes,
                     "shared_arm_feature_preload_seconds": preload_seconds,
                     "full_dev_retrieval_seconds": retrieval_seconds,
                     "list_and_exact_seconds": time.monotonic() - before,
                     "elapsed_seconds_excluding_shared_feature_preload": time.monotonic() - started,
                     "device": args.device, "cpu_threads": args.cpu_threads, "index_threads": args.index_threads,
                     "reader_calls": 0, "optimizer_updates": _optimizer_updates(name, path)},
            "index_retained": args.keep_index,
            "code_sha256": checkpoint_fingerprint(Path(__file__)),
        }
        write_json(target / "metrics.json", payload)
        with (output_root / "runs.jsonl").open("a") as handle:
            handle.write(json.dumps({"task": "A4.2, A4.3, C0 checkpoint diagnostics",
                                     "ended_at_utc": datetime.now(timezone.utc).isoformat(),
                                     "command": [sys.executable, *sys.argv], "output": str(target / "metrics.json"),
                                     "checkpoint": str(path), "status": "pass", "cost": payload["cost"]}) + "\n")
        print(json.dumps({"arm": args.arm, "checkpoint": name,
                          "valid_pool": retrieval["evidence_funnel"]["valid_pool_count"],
                          "elapsed_seconds": payload["cost"]["elapsed_seconds_excluding_shared_feature_preload"]}), flush=True)
        del indices, model, retrieval, exact, payload
        gc.collect()
        torch.cuda.empty_cache()
        if not args.keep_index:
            shutil.rmtree(index_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--arm",
        choices=(
            "pca", "c1_long", "c2_long", "d1_long",
            "r12_base", "r12_candidates", "r12_function", "r12_kd_off",
            "r12_base_extension", "r12_candidates_extension",
            "r12_c2_path_only", "r12_c2_path_edge",
        ),
        required=True,
    )
    parser.add_argument(
        "--checkpoint",
        action="append",
        default=[],
        help="Evaluate an arbitrary checkpoint as ID=PATH; repeat as needed",
    )
    parser.add_argument("--checkpoints", nargs="+", default=[])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--index-threads", type=int, default=12)
    parser.add_argument("--keep-index", action="store_true")
    run(parser.parse_args())
