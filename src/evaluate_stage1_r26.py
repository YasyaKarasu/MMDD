"""Measure each R26 generator's actual whole-lake D/E/U/M retrieval."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from mmdd_stage1.checkpoints import load_student
from mmdd_stage1.features import FeatureStore
from mmdd_stage1.r26_metrics import fuse_channels, fused_lse, population_metrics
from mmdd_stage1.retrieval import (RawEmbeddingANNIndices, StudentANNIndices, build_indices,
                                 load_corpus_ids, retrieve_zero_one_hop_detailed_many)
from mmdd_stage1.row_support import load_evidence_content_keys
from prepare_stage1_r26 import OUT, ROOT, file_record, parameter_sha, stable_sha
from run_stage1_r11_task_e import select_evidence
from run_stage1_r21 import paths, read_rows, write_rows
from run_stage1_r25 import _json, sha256


# Large R29 runs use the FeatureStore's bounded lazy cache instead of retaining
# every corpus embedding at once.  Keep the historical default unchanged.
PRELOAD_ALL = True


def retain_evidence(query_id: str, retrieved: dict, store: FeatureStore, content_keys: dict) -> list[dict]:
    """Historical D1 selection, with both D1 coverage and actual path-LSE scores."""
    result = []
    cache = {}
    for row in retrieved["evidence"]:
        selected, coverage = select_evidence("e2_row_coverage", row["paths"], query_id=query_id,
                                             store=store, content_keys=content_keys, top_l=20,
                                             budget=4, support_cache=cache)
        if selected:
            selected_paths = [p for p in row["paths"] if p.get("evidence_id") in selected]
            path_scores = torch.tensor([p["path_score"] for p in selected_paths], dtype=torch.float64)
            result.append({**row, "selected_evidence_ids": selected, "retained_paths": selected_paths,
                           "evidence_score": coverage, "retained_path_lse": float(torch.logsumexp(path_scores, 0)),
                           "routed_rows": {e: max(range(len(cache[e])), key=lambda r: (cache[e][r], -r)) for e in selected}})
    return sorted(result, key=lambda row: (-row["evidence_score"], row["target_id"]))


def summarize(rows: list[dict]) -> dict:
    result = {}
    for kind in ("overall", "implicit", "explicit"):
        subset = [r for r in rows if kind == "overall" or r["query_kind"] == kind]
        if not subset:
            continue
        qrels = {r["query_id"]: r["positive_target_ids"] for r in subset}
        methods = sorted({key for r in subset for key in r["rankings"]})
        result[kind] = {method: population_metrics({r["query_id"]: r["rankings"].get(method, []) for r in subset}, qrels, (10, 20, 50))
                        for method in methods}
        result[kind]["admission"] = {name: sum(len(r[name]) / len(r["positive_target_ids"]) for r in subset) / len(subset)
                                    for name in ("EO_ANN", "EO_EXACT", "U_ONLY_VS_M")}
        result[kind]["queries"] = len(subset)
    return result


@torch.inference_mode()
def evaluate(generator_id: str, device_name: str, index_threads: int = 4, *, legacy_diagnostics: bool = True) -> dict:
    torch.set_num_threads(4)
    device = torch.device(device_name)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable for full-lake evaluation")
        torch.cuda.set_device(device)
        torch.cuda.init()
        torch.cuda.reset_peak_memory_stats(device)

    def synchronize() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    inventory = json.loads((OUT / "MODEL_INVENTORY.json").read_text())
    spec = next(row for row in inventory if row["generator_id"] == generator_id)
    checkpoint = Path(spec["checkpoint"]) if spec["checkpoint"] else None
    model = load_student(checkpoint, device).eval() if checkpoint else None
    ps = paths(ROOT)
    protocol = json.loads((OUT / "PROTOCOL.json").read_text())["stage1"]
    signature = {"generator_id": generator_id, "checkpoint_sha256": sha256(checkpoint) if checkpoint else None,
                 "parameter_sha256": parameter_sha(model) if model is not None else "raw_no_parameters",
                 "protocol": protocol, "corpus_sha256": sha256(ps["corpus"]),
                 "feature_manifest_sha256": sha256(ps["features"] / "manifest.jsonl"),
                 "query_sha256": sha256(OUT / "common/dev_queries.jsonl"),
                 "code_sha256": {name: sha256(ROOT / "src" / name) for name in (
                     "evaluate_stage1_r26.py", "mmdd_stage1/r26_metrics.py", "mmdd_stage1/retrieval.py",
                     "run_stage1_r11_task_e.py", "mmdd_stage1/row_support.py", "mmdd_stage1/models.py")}}
    if not legacy_diagnostics:
        signature["legacy_diagnostics"] = False
    destination = OUT / "rankings" / generator_id
    destination.mkdir(parents=True, exist_ok=True)
    receipt_path = destination / "RETRIEVAL_RECEIPT.json"
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        if receipt["signature"] != signature:
            raise ValueError("Existing evaluation identity changed; inspect before rerunning")
        if sha256(destination / "rankings.jsonl.gz") != receipt["rankings"]["sha256"]:
            raise ValueError("Ranking artifact changed")
        return {"generator_id": generator_id, "status": "verified_cached", "metrics": receipt["metrics"]}
    _json(destination / "RUNNING.json", {"signature": signature, "pid": __import__("os").getpid()})
    print(json.dumps({"event": "load_features", "generator": generator_id}), flush=True)
    started = time.monotonic()
    store = FeatureStore.from_path(ps["features"], cache_size=300000)
    ids = load_corpus_ids(ps["corpus"], store)
    population = list(read_rows(OUT / "common/dev_queries.jsonl"))
    if PRELOAD_ALL:
        store.preload_embeddings([*(t for values in ids.values() for t in values), *(r["query_id"] for r in population)])
    feature_seconds = time.monotonic() - started
    index_dir = OUT / "indexes" / generator_id
    # Reuse only existing raw/baseline index bytes that the loader verifies against the real corpus/checkpoint.
    if model is None:
        index_dir = ROOT / "work/stage1_optimization_r25_final_20260914/common/raw_qwen_index"
    elif generator_id == "B13":
        index_dir = ps["b13_index"]
    index_started = time.monotonic()
    built = False
    if model is not None and not (index_dir / "manifest.json").is_file():
        print(json.dumps({"event": "build_index", "generator": generator_id, "objects": {k: len(v) for k, v in ids.items()}}), flush=True)
        build_indices(model, store, ids, index_dir, device=device,
                      checkpoint_sha256=signature["checkpoint_sha256"], corpus_sha256=signature["corpus_sha256"],
                      batch_size=4096, num_threads=index_threads)
        built = True
    if model is None:
        indices = RawEmbeddingANNIndices(store, index_dir, corpus_sha256=signature["corpus_sha256"])
    else:
        indices = StudentANNIndices(model, store, index_dir, device=device,
                                    checkpoint_sha256=signature["checkpoint_sha256"], corpus_sha256=signature["corpus_sha256"])
    for index in indices.indices.values():
        index.set_num_threads(index_threads)
    for kind, object_ids in ids.items():
        if indices.object_ids[kind] != object_ids:
            raise ValueError(f"Index target list differs from frozen corpus: {kind}")
    index_seconds = time.monotonic() - index_started
    _json(destination / "INDEX_RECEIPT.json", {"signature": signature, "built_this_run": built, "load_and_build_seconds": index_seconds,
        "index_dir": str(index_dir), "files": [file_record(p) for p in sorted(index_dir.iterdir()) if p.suffix in (".json", ".hnsw")],
        "objects": {k: len(v) for k, v in ids.items()}, "normalization": "cached frozen embeddings; Student linear P/R without extra normalization"})
    print(json.dumps({"event": "retrieval", "generator": generator_id, "index_seconds": index_seconds}), flush=True)
    content_keys, _ = load_evidence_content_keys(ROOT / "work/stage1_optimization_r10_20260907/taskB_g5/evidence_content_keys.jsonl")
    table_ids = ids["table"]
    table_position = {t: i for i, t in enumerate(table_ids)}
    target_blocks = []
    for start in range(0, len(table_ids), 4096):
        embeddings = torch.stack([store.embedding_features(t).embedding for t in table_ids[start:start+4096]]).to(device)
        target_blocks.append(model.index_vector(embeddings, "table") if model is not None else embeddings)
    target_vectors = torch.cat(target_blocks)
    del target_blocks
    output_rows = []
    ann_latencies, exact_latencies, retention_latencies = [], [], []
    for start in range(0, len(population), 16):
        batch = population[start:start+16]
        qids = [row["query_id"] for row in batch]
        if model is not None:
            indices.clear_query_cache()
        ann_start = time.monotonic()
        detailed = retrieve_zero_one_hop_detailed_many(qids, indices, direct_k=100, evidence_k=20,
                                                     targets_per_evidence=20, evidence_aggregation="logsumexp", query_batch_size=16)
        synchronize()
        ann_time = (time.monotonic() - ann_start) / len(batch)
        ann_latencies.extend([ann_time] * len(batch))
        exact_start = time.monotonic()
        query_embeddings = torch.stack([store.embedding_features(q).embedding for q in qids]).to(device)
        queries = model.relation_query(query_embeddings, "table", "table") if model is not None else query_embeddings
        matrix = queries @ target_vectors.T
        # At most 100 direct + 2*20*20 E targets. M is always the actual U cardinality.
        values, positions = matrix.topk(min(900, len(table_ids)), dim=1)
        values, positions = values.cpu().tolist(), positions.cpu().tolist()
        synchronize()
        exact_latencies.extend([(time.monotonic() - exact_start) / len(batch)] * len(batch))
        for i, (meta, retrieved) in enumerate(zip(batch, detailed)):
            retention_start = time.monotonic()
            evidence = retain_evidence(meta["query_id"], retrieved, store, content_keys)
            retention_latencies.append(time.monotonic() - retention_start)
            direct = retrieved["direct"]
            d = [row["target_id"] for row in direct]
            e = [row["target_id"] for row in evidence]
            union = sorted(set(d) | set(e))
            score_values = matrix[i, [table_position[t] for t in union]].cpu().tolist()
            qt = dict(zip(union, score_values))
            qt_rank = sorted(union, key=lambda t: (-qt[t], t))
            exact = [table_ids[j] for j in positions[i]]
            m = exact[:len(union)]
            if legacy_diagnostics:
                fusion = fuse_channels(direct, evidence)
                natural_scores = fused_lse(qt, retrieved["evidence"])
            else:
                from run_stage1_r27_scores import equal_rank
                equal, equal_scores = equal_rank(d, e)
                fusion = {"rankings": {"Equal": equal}, "scores": {"Equal": equal_scores}}
                natural_scores = None
            truth = set(meta["positive_target_ids"])
            rankings = {"D100_ANN": d, "D100_EXACT": exact[:100], "U": qt_rank, "M_EXACT": m,
                        "QT_OVER_U": qt_rank, "E_ONLY": e,
                        **fusion["rankings"]}
            if legacy_diagnostics:
                rankings["PATH_FUSED_LSE"] = sorted(union, key=lambda t: (-natural_scores[t], t))
            output_rows.append({**meta, "generator_id": generator_id, "parameter_sha": signature["parameter_sha256"],
                "index_id": stable_sha(signature), "retrieval_protocol_id": stable_sha(protocol),
                "candidate_pool_id": stable_sha({"q": meta["query_id"], "D": d, "E": e}), "rankings": rankings,
                "D100_ANN": direct, "D100_EXACT": exact[:100], "exact_scores": values[i][:len(m)], "M_exact": m,
                "E_pre_retention": retrieved["evidence"], "E_paths": evidence, "E_target_ids": e,
                "U_pre_retention": sorted(set(d) | {r["target_id"] for r in retrieved["evidence"]}), "U": union,
                "QT_OVER_U_scores": qt, "PATH_FUSED_LSE_scores": natural_scores, "fusion": fusion,
                "EO_ANN": sorted(truth & (set(e) - set(d))), "EO_EXACT": sorted(truth & (set(e) - set(exact[:100]))),
                "U_ONLY_VS_M": sorted(truth & (set(union) - set(m))),
                "execution_status": "ran", "scientific_validity": "valid"})
        if start % 160 == 0:
            print(json.dumps({"event": "queries", "generator": generator_id, "completed": len(output_rows), "total": len(population)}), flush=True)
    write_rows(destination / "rankings.jsonl.gz", output_rows)
    metrics = summarize(output_rows)
    _json(destination / "metrics.json", metrics)
    receipt = {"signature": signature, "execution_status": "ran", "scientific_validity": "valid",
        "rankings": file_record(destination / "rankings.jsonl.gz"), "metrics": metrics,
        "cost": {"device": torch.cuda.get_device_name(device) if device.type == "cuda" else str(device), "feature_seconds": feature_seconds, "index_seconds": index_seconds,
                 "index_built": built, "total_seconds": time.monotonic() - started,
                 "peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
                 "per_query_batch_amortized_seconds": {name: {"mean": float(np.mean(times)), "p50": float(np.quantile(times, .5)), "p95": float(np.quantile(times, .95))}
                    for name, times in (("ANN", ann_latencies), ("exact", exact_latencies), ("retention", retention_latencies))},
                 "online_latency_status": "batch amortized; single-query cold/warm benchmarks pending"}}
    _json(receipt_path, receipt)
    return {"generator_id": generator_id, "status": "ran", "metrics": metrics}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generator", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--index-threads", type=int, default=4)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.generator, args.device, args.index_threads)))
