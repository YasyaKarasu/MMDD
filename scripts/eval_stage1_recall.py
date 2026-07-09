#!/usr/bin/env python
"""Evaluate stage-1 table-to-table and path-aware recall."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from stage1_io import clean_text, iter_jsonl, iter_manifest_records, load_json, update_stage1_manifest, write_json, write_jsonl
from train_student import Student, TYPES, load_embeddings

DATA_LAKE_SPLIT_MODE_CHOICES = ("auto", "strict", "query_corpus")

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - exercised only in minimal envs.
    tqdm = None  # type: ignore[assignment]


def progress_enabled(args: argparse.Namespace) -> bool:
    return bool(getattr(args, "progress", True))


def infer_table_only(args: argparse.Namespace, stage1_dir: Path) -> bool:
    if getattr(args, "table_only", False):
        return True
    hnsw_dir = Path(getattr(args, "hnsw_dir", stage1_dir / "hnsw_indices"))
    stats_path = hnsw_dir / "hnsw_stats.json"
    if stats_path.exists():
        try:
            return bool(json.loads(stats_path.read_text(encoding="utf-8")).get("table_only"))
        except (json.JSONDecodeError, OSError):
            return False
    return False


def infer_raw_embedding_hnsw(args: argparse.Namespace, stage1_dir: Path) -> bool:
    if getattr(args, "raw_embedding_hnsw", False):
        return True
    hnsw_dir = Path(getattr(args, "hnsw_dir", stage1_dir / "hnsw_indices"))
    stats_path = hnsw_dir / "hnsw_stats.json"
    if stats_path.exists():
        try:
            return json.loads(stats_path.read_text(encoding="utf-8")).get("embedding_backend") == "raw"
        except (json.JSONDecodeError, OSError):
            return False
    return False


def infer_data_lake_split_mode(args: argparse.Namespace, stage1_dir: Path) -> str:
    requested = getattr(args, "data_lake_split_mode", "auto") or "auto"
    if requested != "auto":
        return requested
    hnsw_dir = Path(getattr(args, "hnsw_dir", stage1_dir / "hnsw_indices"))
    for path in (hnsw_dir / "hnsw_stats.json", stage1_dir / "manifest.json"):
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        mode = payload.get("data_lake_split_mode")
        if path.name == "manifest.json":
            mode = payload.get("logic_connectivity", {}).get("data_lake_split_mode")
        if mode in {"strict", "query_corpus"}:
            return mode
    target_splits = {
        rec.get("split")
        for rec in iter_jsonl(stage1_dir / "logic_fragments.jsonl")
        if rec.get("role") == "right_target"
    }
    return "query_corpus" if "corpus" in target_splits else "strict"


def target_visible_for_query(query: dict[str, Any], target: dict[str, Any], data_lake_split_mode: str) -> bool:
    if data_lake_split_mode != "strict":
        return True
    return target.get("split") == query.get("split")


def load_student(student_dir: Path, device: torch.device) -> Student:
    ckpt = torch.load(student_dir / "student.pt", map_location=device)
    model = Student(int(ckpt["in_dim"]), int(ckpt["student_dim"])).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def score_pair(model: Student, vectors: dict[str, np.ndarray], a: str, ta: str, b: str, tb: str, device: torch.device) -> float:
    with torch.no_grad():
        za = torch.tensor(vectors[a], dtype=torch.float32, device=device).unsqueeze(0)
        zb = torch.tensor(vectors[b], dtype=torch.float32, device=device).unsqueeze(0)
        return float(torch.sigmoid(model.score(za, ta, zb, tb))[0].cpu())


def raw_cosine_score(vectors: dict[str, np.ndarray], a: str, b: str) -> float:
    va = vectors[a].astype("float32")
    vb = vectors[b].astype("float32")
    denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
    if denom <= 1e-12:
        return 0.0
    cosine = float(np.dot(va, vb) / denom)
    return max(0.0, min(1.0, (cosine + 1.0) / 2.0))


def edge_score(
    args: argparse.Namespace,
    model: Student | None,
    vectors: dict[str, np.ndarray],
    a: str,
    ta: str,
    b: str,
    tb: str,
    device: torch.device,
) -> float:
    if getattr(args, "raw_embedding_hnsw", False):
        return raw_cosine_score(vectors, a, b)
    if model is None:
        raise SystemExit("Student model is required unless --raw_embedding_hnsw is set.")
    return score_pair(model, vectors, a, ta, b, tb, device)


def relation_query_from_projected(model: Student, projected_vec: np.ndarray, source_type: str, target_type: str, device: torch.device) -> np.ndarray:
    with torch.no_grad():
        ua = torch.tensor(projected_vec, dtype=torch.float32, device=device).unsqueeze(0)
        query = ua @ model.rel[f"{source_type}__{target_type}"]
        query = torch.nn.functional.normalize(query, p=2, dim=-1)
        return query.cpu().numpy().astype("float32")


def dcg(rels: list[int]) -> float:
    return sum((2**rel - 1) / math.log2(idx + 2) for idx, rel in enumerate(rels))


def metrics_for(qrels: list[dict[str, Any]], rankings: dict[str, list[str]], topks: list[int]) -> dict[str, float]:
    rel_by_q: dict[str, dict[str, int]] = defaultdict(dict)
    for qrel in qrels:
        rel_by_q[qrel["query_id"]][qrel["target_id"]] = int(qrel["rel"])
    out: dict[str, float] = {}
    n = len(rel_by_q)
    for k in topks:
        recall = 0.0
        ndcg = 0.0
        for q, rels in rel_by_q.items():
            ranked = rankings.get(q, [])[:k]
            hits = sum(1 for t in ranked if t in rels)
            recall += hits / max(1, len(rels))
            ranked_rels = [rels.get(t, 0) for t in ranked]
            ideal = sorted(rels.values(), reverse=True)[:k]
            ndcg += dcg(ranked_rels) / max(1e-12, dcg(ideal))
        out[f"Recall@{k}"] = recall / max(1, n)
        out[f"nDCG@{k}"] = ndcg / max(1, n)
    mrr = 0.0
    for q, rels in rel_by_q.items():
        rank = 0
        for idx, target in enumerate(rankings.get(q, []), 1):
            if target in rels:
                rank = idx
                break
        mrr += 1.0 / rank if rank else 0.0
    out["MRR"] = mrr / max(1, n)
    out["queries"] = n
    return out


def qrels_by_query(qrels: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for qrel in qrels:
        grouped[qrel["query_id"]].append(qrel)
    return grouped


def relevant_by_target(qrels: list[dict[str, Any]]) -> dict[str, int]:
    return {qrel["target_id"]: int(qrel.get("rel", 1)) for qrel in qrels}


def recall_fraction(ranked: list[str], correct: set[str], k: int) -> float:
    if not correct:
        return 0.0
    hits = sum(1 for target_id in ranked[:k] if target_id in correct)
    return hits / len(correct)


def recall_record_header(query_id: str, qrels: list[dict[str, Any]], retrieval_mode: str, topk: int) -> dict[str, Any]:
    first = qrels[0] if qrels else {}
    return {
        "sample_kind": "recall_ranking",
        "retrieval_mode": retrieval_mode,
        "query_id": query_id,
        "query_role": first.get("query_role"),
        "split": first.get("split"),
        "chain_id": first.get("chain_id"),
        "topk": topk,
        "relevant_targets": [
            {
                "target_id": qrel["target_id"],
                "rel": int(qrel.get("rel", 1)),
                "chain_id": qrel.get("chain_id"),
                "target_role": qrel.get("target_role"),
            }
            for qrel in qrels
        ],
    }


def direct_recall_records(qrels: list[dict[str, Any]], rankings: dict[str, list[str]], topk: int) -> list[dict[str, Any]]:
    records = []
    for query_id, query_qrels in sorted(qrels_by_query(qrels).items()):
        relevant = relevant_by_target(query_qrels)
        record = recall_record_header(query_id, query_qrels, "direct_table", topk)
        record["targets"] = [
            {
                "rank": rank,
                "target_id": target_id,
                "is_relevant": target_id in relevant,
                "rel": relevant.get(target_id),
            }
            for rank, target_id in enumerate(rankings.get(query_id, [])[:topk], 1)
        ]
        records.append(record)
    return records


def direct_eval_qrels(qrels: list[dict[str, Any]], table_only: bool) -> list[dict[str, Any]]:
    if not table_only:
        return qrels
    return [qrel for qrel in qrels if qrel.get("query_role") == "left_visible"]


def serialize_path_nodes(path: list[tuple[str, str]]) -> list[dict[str, str]]:
    return [{"node_id": node_id, "node_type": node_type} for node_id, node_type in path]


def stage1_input_dir(stage1_dir: Path) -> Path:
    manifest_path = stage1_dir / "manifest.json"
    if manifest_path.exists():
        manifest = load_json(manifest_path)
        input_dir = (
            manifest.get("evidence_paths", {}).get("input_dir")
            or manifest.get("logic_connectivity", {}).get("input_dir")
            or "output_medium"
        )
        return Path(input_dir)
    return Path("output_medium")


def load_bridge_asset_previews(stage1_dir: Path) -> dict[str, dict[str, Any]]:
    input_dir = stage1_input_dir(stage1_dir)
    if not (input_dir / "dataset_manifest.json").exists():
        return {}
    previews = {}
    for asset in iter_manifest_records(input_dir, "bridge_assets", log_every=50000):
        asset_id = clean_text(asset.get("asset_id"))
        if asset_id:
            previews[asset_id] = {
                "asset_id": asset_id,
                "asset_type": clean_text(asset.get("asset_type")),
                "title": clean_text(asset.get("entity_wiki_title")) or clean_text(asset.get("title")),
                "source": clean_text(asset.get("source")),
                "url": clean_text(asset.get("url") or asset.get("description_url") or asset.get("image_url")),
                "content_snippet": clean_text(asset.get("content"))[:1600],
                "file_name": clean_text(asset.get("file_name")),
                "local_path": clean_text(asset.get("local_path")),
                "relative_path": clean_text(asset.get("relative_path")),
                "text_chunk_index": asset.get("text_chunk_index"),
                "text_chunk_count": asset.get("text_chunk_count"),
            }
    return previews


def bridge_path_metadata(
    query_id: str,
    target_id: str,
    path: list[tuple[str, str]],
    paths_by_query_asset_target: dict[tuple[str, str, str], dict[str, Any]],
    asset_previews: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    for node_id, node_type in path:
        if node_type not in {"text_asset", "image_asset"}:
            continue
        metadata: dict[str, Any] = {
            "asset_id": node_id,
            "asset_object_type": node_type,
            "asset_preview": (asset_previews or {}).get(node_id, {}),
        }
        record = paths_by_query_asset_target.get((query_id, node_id, target_id))
        if record:
            metadata.update(
                {
                    "path_id": record.get("path_id"),
                    "asset_id": record.get("asset_id") or node_id,
                    "asset_type": record.get("asset_type"),
                    "entity_text": record.get("entity_text"),
                    "bridge_col_name": record.get("bridge_col_name"),
                    "bridge_value": record.get("bridge_value"),
                    "target_bridge_col_name": record.get("target_bridge_col_name"),
                    "claim_text": record.get("claim_text"),
                    "weak_label": record.get("weak_label"),
                    "weak_score": record.get("weak_score"),
                    "human_label": record.get("human_label"),
                    "teacher_score": record.get("teacher_score"),
                }
            )
        return metadata
    return {}


def bridge_recall_record(
    query_id: str,
    query_qrels: list[dict[str, Any]],
    ranked: list[str],
    best_paths: dict[str, dict[str, Any]],
    paths_by_query_asset_target: dict[tuple[str, str, str], dict[str, Any]],
    asset_previews: dict[str, dict[str, Any]] | None,
    topk: int,
) -> dict[str, Any]:
    relevant = relevant_by_target(query_qrels)
    record = recall_record_header(query_id, query_qrels, "bridge_aware", topk)
    targets = []
    for rank, target_id in enumerate(ranked[:topk], 1):
        payload = best_paths.get(target_id, {})
        path = payload.get("path", [])
        path_metadata = bridge_path_metadata(query_id, target_id, path, paths_by_query_asset_target, asset_previews)
        targets.append(
            {
                "rank": rank,
                "target_id": target_id,
                "score": payload.get("score"),
                "is_relevant": target_id in relevant,
                "rel": relevant.get(target_id),
                "path": serialize_path_nodes(path),
                "has_bridge": any(node_type in {"text_asset", "image_asset"} for _, node_type in path),
                "path_metadata": path_metadata,
            }
        )
    record["targets"] = targets
    return record


def table_rankings(
    args: argparse.Namespace,
    stage1_dir: Path,
    model: Student | None,
    vectors: dict[str, np.ndarray],
    device: torch.device,
    topks: list[int],
) -> dict[str, list[str]]:
    if getattr(args, "data_lake_split_mode", "auto") == "auto":
        args.data_lake_split_mode = infer_data_lake_split_mode(args, stage1_dir)
    fragments = {rec["fragment_id"]: rec for rec in iter_jsonl(stage1_dir / "logic_fragments.jsonl")}
    query_roles = {"left_visible"} if getattr(args, "table_only", False) else {"left_visible", "left_hidden"}
    queries = [f for f in fragments.values() if f.get("role") in query_roles and f["fragment_id"] in vectors]
    targets = [f for f in fragments.values() if f.get("role") == "right_target" and f["fragment_id"] in vectors]
    if getattr(args, "table_only", False) or getattr(args, "raw_embedding_hnsw", False):
        return hnsw_table_rankings(args, model, queries, targets, device, topks)
    if model is None:
        raise SystemExit("Student model is required unless --raw_embedding_hnsw is set.")
    rankings: dict[str, list[str]] = {}
    total_pairs = len(queries) * len(targets)
    progress = None
    if tqdm is not None and progress_enabled(args):
        progress = tqdm(total=total_pairs, desc="Scoring table recall", unit="pair")
    try:
        for q_idx, q in enumerate(queries, 1):
            should_log = q_idx == 1 or q_idx == len(queries) or q_idx % max(1, len(queries) // 20) == 0
            if progress is None and progress_enabled(args) and should_log:
                print(f"Scoring table recall: {q_idx}/{len(queries)} queries", flush=True)
            scored = []
            for t in targets:
                if not target_visible_for_query(q, t, getattr(args, "data_lake_split_mode", "strict")):
                    continue
                scored.append(
                    (
                        score_pair(
                            model,
                            vectors,
                            q["fragment_id"],
                            "table_fragment",
                            t["fragment_id"],
                            "table_fragment",
                            device,
                        ),
                        t["fragment_id"],
                    )
                )
                if progress is not None:
                    progress.update(1)
            scored.sort(reverse=True)
            rankings[q["fragment_id"]] = [tid for _, tid in scored]
    finally:
        if progress is not None:
            progress.close()
    return rankings


def split_index_name(object_type: str, split: str) -> str:
    safe_split = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(split))
    return f"{object_type}_{safe_split or 'unknown'}"


def hnsw_current_count(index: Any, ids: list[str]) -> int:
    if hasattr(index, "get_current_count"):
        return min(len(ids), int(index.get_current_count()))
    return len(ids)


def requested_hnsw_k(args: argparse.Namespace, indexed_count: int) -> int:
    requested_k = int(getattr(args, "table_hnsw_k", 0) or 0)
    return min(requested_k if requested_k > 0 else indexed_count, indexed_count)


def hnsw_table_rankings(
    args: argparse.Namespace,
    model: Student | None,
    queries: list[dict[str, Any]],
    targets: list[dict[str, Any]],
    device: torch.device,
    topks: list[int],
) -> dict[str, list[str]]:
    raw_embedding_hnsw = bool(getattr(args, "raw_embedding_hnsw", False))
    projected = load_index_vectors(Path(args.embedding_dir)) if raw_embedding_hnsw else load_projected(Path(args.student_dir))
    table_projected = {oid: vec for oid, (object_type, vec) in projected.items() if object_type == "table_fragment"}
    if not table_projected:
        location = "embeddings/table_fragment.npy" if raw_embedding_hnsw else "student/index_embeddings/table_fragment.npy"
        raise SystemExit(f"HNSW table evaluation requires {location}")
    dim = len(next(iter(table_projected.values())))
    target_ids = {target["fragment_id"] for target in targets}
    targets_by_id = {target["fragment_id"]: target for target in targets}
    data_lake_split_mode = getattr(args, "data_lake_split_mode", "strict")
    global_index = None
    global_table_ids: list[str] = []
    if data_lake_split_mode != "strict":
        global_index, global_table_ids = load_hnsw(Path(args.hnsw_dir), "table_fragment", dim)
        if global_index is None:
            raise SystemExit("table-only HNSW evaluation requires hnsw_indices/table_fragment.bin")
        unexpected_ids = [table_id for table_id in global_table_ids if table_id not in target_ids]
        if unexpected_ids:
            raise SystemExit(
                "table-only HNSW evaluation expects hnsw_indices/table_fragment_ids.json to contain only right_target fragments. "
                "Rebuild the index with scripts/stage1_index_eval.py --table_only."
            )
    target_ids_by_split: dict[str, set[str]] = defaultdict(set)
    for target in targets:
        target_ids_by_split[str(target.get("split", "unknown"))].add(target["fragment_id"])
    split_indexes: dict[str, tuple[Any, list[str]]] = {}
    rankings: dict[str, list[str]] = {}
    query_iter = queries
    if tqdm is not None and progress_enabled(args):
        query_iter = tqdm(queries, desc="HNSW table recall", unit="query")  # type: ignore[assignment]
    for q_idx, q in enumerate(query_iter, 1):
        qid = q["fragment_id"]
        should_log = q_idx == 1 or q_idx == len(queries) or q_idx % max(1, len(queries) // 20) == 0
        if tqdm is None and progress_enabled(args) and should_log:
            print(f"HNSW table recall: {q_idx}/{len(queries)} queries", flush=True)
        if qid not in table_projected:
            rankings[qid] = []
            continue
        if raw_embedding_hnsw:
            query = table_projected[qid].reshape(1, -1).astype("float32")
        else:
            if model is None:
                raise SystemExit("Student model is required unless --raw_embedding_hnsw is set.")
            query = relation_query_from_projected(model, table_projected[qid], "table_fragment", "table_fragment", device)
        if data_lake_split_mode == "strict":
            split = str(q.get("split", "unknown"))
            if split not in split_indexes:
                split_object_type = split_index_name("table_fragment", split)
                split_index, split_table_ids = load_hnsw(Path(args.hnsw_dir), split_object_type, dim)
                if split_index is None:
                    if target_ids_by_split.get(split):
                        raise SystemExit(
                            f"strict HNSW evaluation requires hnsw_indices/{split_object_type}.bin for query split {split!r}. "
                            "Rebuild the index with scripts/stage1_index_eval.py --data_lake_split_mode strict."
                        )
                    rankings[qid] = []
                    continue
                unexpected_ids = [table_id for table_id in split_table_ids if table_id not in target_ids_by_split.get(split, set())]
                if unexpected_ids:
                    raise SystemExit(
                        f"strict HNSW index {split_object_type} must contain only right_target fragments from split {split!r}. "
                        "Rebuild the index with scripts/stage1_index_eval.py --data_lake_split_mode strict."
                    )
                split_indexes[split] = (split_index, split_table_ids)
            index, table_ids = split_indexes[split]
        else:
            index, table_ids = global_index, global_table_ids
        indexed_count = hnsw_current_count(index, table_ids)
        hnsw_k = requested_hnsw_k(args, indexed_count)
        if hnsw_k <= 0:
            rankings[qid] = []
            continue
        index.set_ef(max(100, hnsw_k))
        labels, _ = index.knn_query(query, k=hnsw_k)
        ranked_ids = [table_ids[int(label)] for label in labels[0]]
        rankings[qid] = [
            target_id
            for target_id in ranked_ids
            if target_visible_for_query(q, targets_by_id.get(target_id, {}), data_lake_split_mode)
        ]
    return rankings


def load_hnsw(hnsw_dir: Path, object_type: str, dim: int):
    try:
        import hnswlib
    except ImportError:
        return None, []
    bin_path = hnsw_dir / f"{object_type}.bin"
    ids_path = hnsw_dir / f"{object_type}_ids.json"
    if not bin_path.exists() or not ids_path.exists():
        return None, []
    index = hnswlib.Index(space="cosine", dim=dim)
    ids = json.loads(ids_path.read_text(encoding="utf-8"))
    index.load_index(str(bin_path), max_elements=len(ids))
    index.set_ef(100)
    return index, ids


def load_projected(student_dir: Path) -> dict[str, tuple[str, np.ndarray]]:
    return load_index_vectors(student_dir / "index_embeddings")


def load_index_vectors(emb_dir: Path) -> dict[str, tuple[str, np.ndarray]]:
    projected = {}
    for object_type in TYPES:
        npy = emb_dir / f"{object_type}.npy"
        ids_path = emb_dir / f"{object_type}_ids.json"
        if npy.exists() and ids_path.exists():
            arr = np.load(npy).astype("float32")
            ids = json.loads(ids_path.read_text(encoding="utf-8"))
            projected.update({oid: (object_type, arr[i]) for i, oid in enumerate(ids)})
    return projected


def compose_path_score(prev_score: float, edge_score: float, composition: str) -> float:
    if composition == "product":
        return prev_score * edge_score
    return min(prev_score, edge_score)


def relation_aware_neighbors(
    args: argparse.Namespace,
    model: Student | None,
    projected: dict[str, tuple[str, np.ndarray]],
    indexes: dict[str, Any],
    ids_by_type: dict[str, list[str]],
    node_id: str,
    source_type: str,
    target_type: str,
    k: int,
    device: torch.device,
) -> list[str]:
    if target_type not in indexes or not ids_by_type.get(target_type) or node_id not in projected:
        return []
    if getattr(args, "raw_embedding_hnsw", False):
        query = projected[node_id][1].reshape(1, -1).astype("float32")
    else:
        query = relation_query_from_projected(model, projected[node_id][1], source_type, target_type, device)
    labels, _ = indexes[target_type].knn_query(query, k=min(k, len(ids_by_type[target_type])))
    return [ids_by_type[target_type][int(label)] for label in labels[0]]


def beam_search_tables(
    args: argparse.Namespace,
    model: Student | None,
    vectors: dict[str, np.ndarray],
    projected: dict[str, tuple[str, np.ndarray]],
    indexes: dict[str, Any],
    ids_by_type: dict[str, list[str]],
    query_id: str,
    device: torch.device,
) -> tuple[list[str], dict[str, dict[str, Any]]]:
    if query_id not in projected:
        return [], {}
    start_type = projected[query_id][0]
    frontier = [
        {
            "node_id": query_id,
            "node_type": start_type,
            "score": 1.0,
            "path": [(query_id, start_type)],
        }
    ]
    best_tables: dict[str, dict[str, Any]] = {}
    neighbor_k = max(1, int(args.beam_neighbors))
    for _ in range(max(1, int(args.max_hops))):
        candidates = []
        for state in frontier:
            for target_type in TYPES:
                neighbors = relation_aware_neighbors(
                    args,
                    model,
                    projected,
                    indexes,
                    ids_by_type,
                    state["node_id"],
                    state["node_type"],
                    target_type,
                    neighbor_k,
                    device,
                )
                for next_id in neighbors:
                    if next_id == query_id or any(node_id == next_id for node_id, _ in state["path"]):
                        continue
                    if state["node_id"] not in vectors or next_id not in vectors:
                        continue
                    next_edge_score = edge_score(args, model, vectors, state["node_id"], state["node_type"], next_id, target_type, device)
                    path_score = compose_path_score(float(state["score"]), next_edge_score, args.path_composition)
                    path = [*state["path"], (next_id, target_type)]
                    next_state = {"node_id": next_id, "node_type": target_type, "score": path_score, "path": path}
                    candidates.append(next_state)
                    if target_type == "table_fragment":
                        previous = best_tables.get(next_id)
                        if previous is None or path_score > previous["score"]:
                            best_tables[next_id] = {"score": path_score, "path": path}
        candidates.sort(key=lambda item: item["score"], reverse=True)
        frontier = candidates[: max(1, int(args.beam_width))]
        if not frontier:
            break
    ranked = [tid for tid, _ in sorted(best_tables.items(), key=lambda item: item[1]["score"], reverse=True)]
    return ranked, best_tables


def path_aware_metrics(
    args: argparse.Namespace,
    stage1_dir: Path,
    model: Student | None,
    vectors: dict[str, np.ndarray],
    device: torch.device,
    topks: list[int],
    recall_records: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    student_dir = Path(args.student_dir)
    emb_dir = Path(args.embedding_dir) if getattr(args, "raw_embedding_hnsw", False) else student_dir / "index_embeddings"
    table_arr = np.load(emb_dir / "table_fragment.npy")
    dim = int(table_arr.shape[1])
    indexes: dict[str, Any] = {}
    ids_by_type: dict[str, list[str]] = {}
    for object_type in TYPES:
        index, ids = load_hnsw(Path(args.hnsw_dir), object_type, dim)
        if index is not None:
            indexes[object_type] = index
            ids_by_type[object_type] = ids
    if "table_fragment" not in indexes:
        return {"available": False}
    projected = load_index_vectors(emb_dir) if getattr(args, "raw_embedding_hnsw", False) else load_projected(student_dir)
    human = {rec["path_id"]: rec.get("human_label") for rec in iter_jsonl(stage1_dir / "human_labeled_paths.jsonl")} if (stage1_dir / "human_labeled_paths.jsonl").exists() else {}
    paths_by_query_asset_target = {}
    pool_iter = iter_jsonl(stage1_dir / "hitl_pool.jsonl") if (stage1_dir / "hitl_pool.jsonl").exists() else []
    for path in pool_iter:
        paths_by_query_asset_target[(path["query_fragment_id"], path["asset_id"], path["target_fragment_id"])] = path
    asset_previews = load_bridge_asset_previews(stage1_dir) if recall_records is not None else {}
    qrels = [rec for rec in iter_jsonl(Path(args.qrels))]
    qrels_by_qid = qrels_by_query(qrels)
    hidden_q = sorted({q["query_id"] for q in qrels if q.get("query_role") == "left_hidden"})
    correct = defaultdict(set)
    for q in qrels:
        if q.get("query_role") == "left_hidden":
            correct[q["query_id"]].add(q["target_id"])
    recalls = {f"Bridge-aware Table Recall@{k}": 0.0 for k in topks}
    direct_indirect = 0
    related_only = 0
    evidence_seen = 0
    total_retrieved_tables = 0
    query_iter = hidden_q
    if tqdm is not None and progress_enabled(args):
        query_iter = tqdm(hidden_q, desc="Path-aware recall", unit="query")  # type: ignore[assignment]
    for q_idx, qid in enumerate(query_iter, 1):
        should_log = q_idx == 1 or q_idx == len(hidden_q) or q_idx % max(1, len(hidden_q) // 20) == 0
        if tqdm is None and progress_enabled(args) and should_log:
            print(f"Path-aware recall: {q_idx}/{len(hidden_q)} hidden queries", flush=True)
        if qid not in projected:
            continue
        ranked, best_paths = beam_search_tables(args, model, vectors, projected, indexes, ids_by_type, qid, device)
        if recall_records is not None:
            recall_records.append(
                bridge_recall_record(
                    qid,
                    qrels_by_qid.get(qid, []),
                    ranked,
                    best_paths,
                    paths_by_query_asset_target,
                    asset_previews,
                    max(topks),
                )
            )
        total_retrieved_tables += len(ranked)
        for tid, payload in best_paths.items():
            path_nodes = payload["path"]
            if len(path_nodes) == 3 and path_nodes[1][1] in {"text_asset", "image_asset"}:
                aid = path_nodes[1][0]
                path = paths_by_query_asset_target.get((qid, aid, tid))
                if path and path.get("path_id") in human:
                    evidence_seen += 1
                    if human[path["path_id"]] in {1, 2}:
                        direct_indirect += 1
                    elif human[path["path_id"]] == 0:
                        related_only += 1
        for k in topks:
            recalls[f"Bridge-aware Table Recall@{k}"] += recall_fraction(ranked, correct[qid], k)
    denom = max(1, len(hidden_q))
    return {
        "available": True,
        **{k: v / denom for k, v in recalls.items()},
        "Bridge Evidence Precision": direct_indirect / max(1, evidence_seen),
        "Related-only False Positive Rate": related_only / max(1, evidence_seen),
        "labeled_evidence_seen": evidence_seen,
        "retrieved_table_endpoints": total_retrieved_tables,
        "max_hops": int(args.max_hops),
        "beam_width": int(args.beam_width),
        "beam_neighbors": int(args.beam_neighbors),
    }


def run(args: argparse.Namespace) -> None:
    topks = [int(k) for k in args.topk]
    stage1_dir = Path(args.stage1_dir)
    args.table_only = infer_table_only(args, stage1_dir)
    if getattr(args, "embedding_dir", None) is None:
        args.embedding_dir = str(stage1_dir / "embeddings")
    args.raw_embedding_hnsw = infer_raw_embedding_hnsw(args, stage1_dir)
    args.data_lake_split_mode = infer_data_lake_split_mode(args, stage1_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vectors, _, _, _ = load_embeddings(Path(args.embedding_dir))
    model = None if getattr(args, "raw_embedding_hnsw", False) else load_student(Path(args.student_dir), device)
    qrels = list(iter_jsonl(Path(args.qrels)))
    rankings = table_rankings(args, stage1_dir, model, vectors, device, topks)
    direct_qrels = direct_eval_qrels(qrels, getattr(args, "table_only", False))
    recall_records = direct_recall_records(direct_qrels, rankings, max(topks)) if getattr(args, "write_recall_records", True) else []
    results: dict[str, Any] = {}
    roles = ("left_visible",) if getattr(args, "table_only", False) else ("left_visible", "left_hidden")
    for split in ("dev", "test"):
        for role in roles:
            subset = [q for q in direct_qrels if q.get("split") == split and q.get("query_role") == role]
            results[f"{split}_{role}"] = metrics_for(subset, rankings, topks)
    if not getattr(args, "table_only", False):
        results["path_aware"] = path_aware_metrics(
            args,
            stage1_dir,
            model,
            vectors,
            device,
            topks,
            recall_records if getattr(args, "write_recall_records", True) else None,
        )
    if getattr(args, "write_recall_records", True):
        recall_records_path = Path(getattr(args, "recall_records", "") or stage1_dir / "recall_rankings.jsonl")
        write_jsonl(recall_records_path, recall_records)
        results["recall_records"] = {"path": str(recall_records_path), "records": len(recall_records), "topk": max(topks)}
    write_json(stage1_dir / "eval_results.json", results)
    update_stage1_manifest(stage1_dir, "eval", {"results": results, "args": vars(args)})
    print(json.dumps(results, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1_dir", default="output_stage1_logic")
    parser.add_argument("--student_dir", default="output_stage1_logic/student")
    parser.add_argument("--embedding_dir", default=None)
    parser.add_argument("--hnsw_dir", default="output_stage1_logic/hnsw_indices")
    parser.add_argument("--qrels", default="output_stage1_logic/qrels.jsonl")
    parser.add_argument("--topk", nargs="+", default=["10", "50", "100"])
    parser.add_argument("--max_hops", type=int, default=3)
    parser.add_argument("--beam_width", type=int, default=64)
    parser.add_argument("--beam_neighbors", type=int, default=50)
    parser.add_argument("--path_composition", choices=["min", "product"], default="min")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--table_only", action="store_true", help="Only compute direct table-to-table recall; skip path-aware metrics.")
    parser.add_argument("--raw_embedding_hnsw", action="store_true", help="Evaluate HNSW recall directly over frozen raw embeddings without loading a teacher/student model.")
    parser.add_argument(
        "--data_lake_split_mode",
        choices=DATA_LAKE_SPLIT_MODE_CHOICES,
        default="auto",
        help="Candidate data lake split mode for direct recall; auto reads stage metadata.",
    )
    parser.add_argument("--recall_records", default=None, help="Output JSONL path for per-query recalled targets and bridge paths.")
    parser.add_argument("--no_recall_records", dest="write_recall_records", action="store_false", help="Skip writing per-query recall_rankings.jsonl.")
    parser.add_argument(
        "--table_hnsw_k",
        type=int,
        default=0,
        help="right_target table neighbors to retrieve for table-only HNSW eval; 0 means all indexed target tables.",
    )
    parser.add_argument("--no_progress", dest="progress", action="store_false")
    parser.set_defaults(progress=True, write_recall_records=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
