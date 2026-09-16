"""Run the zero-training C1 evidence-collapse mechanism audit.

The runner consumes the actual Bridge checkpoints at C1 steps 356, 500, and
659.  It keeps relation identity explicit: Q->text/image and evidence->table
are scored with their own directed Student relations, while the fixed-path
view is recomputed from one frozen evaluation graph.  No labels are used to
construct a retrieval pool; dev witness labels are only used after scoring.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import torch

from mmdd_stage1.checkpoints import load_student
from mmdd_stage1.features import FeatureStore
from mmdd_stage1.retrieval import StudentANNIndices
from run_stage1_bridge import OUT as BRIDGE

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ROOT / "work/stage1_optimization_r10_20260907/features_qwen3_vl_embedding_8b"
DEV_TARGET_LISTS = ROOT / "work/stage1_optimization_r11_20260908/taskA_protocol/supervision/target_lists.dev.jsonl"
PATH_GRAPH = ROOT / (
    "work/stage1_diagnostics_r27_20260915_rev2_b13_exact/score_handoff/"
    "R25-C1/seed13/step659/raw_candidates_paths.jsonl.gz"
)
INDEX_ROOT = BRIDGE / "evaluation/indexes"
DIAG_ROOT = BRIDGE / "evidence_collapse_diagnostic"
SEEDS = (13, 29)
STEPS = (356, 500, 659)
QE_CUTOFFS = (1, 5, 10, 20, 50, 100)
ET_CUTOFFS = (1, 5, 10, 20, 50, 100)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_dev_rows(path: Path = DEV_TARGET_LISTS) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                if row.get("split") != "dev":
                    raise ValueError(f"{path}: expected only dev rows")
                rows.append(row)
    if len({row["query_id"] for row in rows}) != len(rows):
        raise ValueError("dev target lists contain duplicate query IDs")
    return rows


def read_ids(index_dir: Path) -> dict[str, list[str]]:
    manifest = json.loads((index_dir / "manifest.json").read_text(encoding="utf-8"))
    return {
        object_type: [str(value) for value in json.loads((index_dir / record["ids_path"]).read_text())]
        for object_type, record in manifest["types"].items()
    }


def checkpoint_path(seed: int, step: int) -> Path:
    return BRIDGE / f"training/B5/seed{seed}/C1/checkpoints/step_{step:06d}.pt"


def checkpoint_identity(seed: int, step: int) -> dict[str, Any]:
    path = checkpoint_path(seed, step)
    if not path.is_file():
        raise FileNotFoundError(path)
    execution = json.loads((path.parents[1] / "EXECUTION.json").read_text(encoding="utf-8"))
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return {
        "seed": seed,
        "step": step,
        "path": str(path.resolve()),
        "sha256": sha256(path),
        "parent": execution.get("parent"),
        "optimizer_initial_state": execution.get("optimizer_initial_state"),
        "start_step": execution.get("start_step"),
        "completed_stage": payload.get("completed_stage"),
        "bridge": payload.get("bridge"),
        "config": payload.get("config"),
    }


def positive_witnesses(row: dict[str, Any]) -> dict[str, list[str]]:
    target = str(row["positive_target_ids"][0])
    values = row.get("positive_evidence_by_target", {}).get(target, [])
    return {"text": [], "image": [], "all": [str(value) for value in values]}


def object_types(store: FeatureStore, ids: Iterable[str]) -> dict[str, str]:
    return {str(object_id): store.object_type(str(object_id)) for object_id in ids}


def rank_of_positive(scores: torch.Tensor, ids: list[str], positive_ids: Iterable[str]) -> dict[str, int]:
    """Return exact one-based ranks with deterministic ID tie-breaking."""

    values = scores.detach().float().cpu()
    positions = {object_id: index for index, object_id in enumerate(ids)}
    result = {}
    for positive_id in positive_ids:
        position = positions.get(positive_id)
        if position is None:
            continue
        score = values[position]
        greater = int((values > score).sum())
        # Float ties are uncommon; only scan the equality bucket when one is
        # actually present, keeping the ordinary exact-rank path vectorized.
        equal_positions = torch.nonzero(values == score, as_tuple=False).flatten().tolist()
        ties_before = sum(1 for index in equal_positions if index != position and ids[index] < positive_id)
        result[positive_id] = 1 + greater + ties_before
    return result


def rank_summary(ranks: list[int | None], cutoffs: tuple[int, ...]) -> dict[str, Any]:
    observed = [int(value) for value in ranks if value is not None]
    denominator = len(ranks)
    return {
        "queries": denominator,
        "recall_at_rank": {
            str(cutoff): (sum(value is not None and value <= cutoff for value in ranks) / denominator if denominator else 0.0)
            for cutoff in cutoffs
        },
        "mrr": (sum(1.0 / value for value in observed) / denominator if denominator else 0.0),
        "median_positive_rank": statistics.median(observed) if observed else None,
        "p90_positive_rank": (float(torch.tensor(observed, dtype=torch.float32).quantile(0.9)) if observed else None),
        "censored_above": denominator - len(observed),
    }


def concentration(rows: list[list[str]], *, top_fraction: float = 0.01) -> dict[str, Any]:
    counts = Counter(value for row in rows for value in row)
    slots = sum(len(row) for row in rows)
    values = sorted(counts.values(), reverse=True)
    if not values:
        return {"queries": len(rows), "slots": 0, "unique": 0, "gini": 0.0, "top_fraction_slot_share": 0.0}
    n_top = max(1, math.ceil(len(values) * top_fraction))
    gini = sum((2 * index - len(values) - 1) * value for index, value in enumerate(sorted(values), 1))
    gini = gini / (len(values) * sum(values)) if sum(values) else 0.0
    return {
        "queries": len(rows),
        "slots": slots,
        "unique": len(counts),
        "gini": float(gini),
        "top_fraction_slot_share": sum(values[:n_top]) / slots if slots else 0.0,
        "top_items": counts.most_common(20),
    }


def modality_witnesses(rows: list[dict[str, Any]], types: dict[str, str]) -> dict[str, dict[str, list[str]]]:
    result = {}
    for row in rows:
        values = [str(value) for value in row.get("positive_evidence_by_target", {}).get(str(row["positive_target_ids"][0]), [])]
        result[str(row["query_id"])] = {
            modality: [value for value in values if types.get(value) == modality]
            for modality in ("text", "image")
        }
        result[str(row["query_id"])] ["all"] = values
    return result


@torch.inference_mode()
def exact_relation_ranks(
    model: torch.nn.Module,
    store: FeatureStore,
    source_ids: list[str],
    source_type: str,
    destination_ids: list[str],
    destination_type: str,
    positive_by_source: dict[str, set[str]],
    device: torch.device,
    destination_matrix: torch.Tensor,
    batch_size: int = 16,
) -> tuple[dict[str, dict[str, int]], dict[str, list[str]]]:
    ranks: dict[str, dict[str, int]] = {}
    top_rows: dict[str, list[str]] = {}
    id_to_row = {object_id: index for index, object_id in enumerate(destination_ids)}
    for start in range(0, len(source_ids), batch_size):
        batch_ids = source_ids[start : start + batch_size]
        source = torch.stack([store.embedding_features(value).embedding for value in batch_ids]).to(device)
        queries = model.relation_query(source, source_type, destination_type)
        scores = queries @ destination_matrix.T
        k = min(100, len(destination_ids))
        top = scores.topk(k, dim=1).indices.cpu().tolist()
        for row_index, source_id in enumerate(batch_ids):
            ranks[source_id] = rank_of_positive(scores[row_index], destination_ids, positive_by_source.get(source_id, set()))
            top_rows[source_id] = [destination_ids[index] for index in top[row_index]]
    return ranks, top_rows


def ann_relation_ranks(
    model: torch.nn.Module,
    store: FeatureStore,
    index_dir: Path,
    checkpoint_sha: str,
    source_ids: list[str],
    destination_type: str,
    positive_by_source: dict[str, set[str]],
    device: torch.device,
    k: int = 100,
) -> tuple[dict[str, dict[str, int]], dict[str, list[str]]]:
    indices = StudentANNIndices(model, store, index_dir, device=device, checkpoint_sha256=checkpoint_sha)
    results = indices.search_many(source_ids, destination_type, k)
    ranks = {}
    top_rows = {}
    for source_id, found in zip(source_ids, results):
        found_ids = [object_id for object_id, _score in found]
        ranks[source_id] = {object_id: index + 1 for index, object_id in enumerate(found_ids) if object_id in positive_by_source.get(source_id, set())}
        top_rows[source_id] = found_ids
    return ranks, top_rows


def aggregate_query_ranks(
    rows: list[dict[str, Any]],
    rank_maps: dict[str, dict[str, int]],
    witness_map: dict[str, dict[str, list[str]]],
    modality: str,
) -> dict[str, Any]:
    per_query: dict[str, int | None] = {}
    for row in rows:
        query_id = str(row["query_id"])
        if not witness_map[query_id][modality]:
            continue
        values = [rank_maps.get(evidence_id, {}).get(str(row["positive_target_ids"][0])) for evidence_id in witness_map[query_id][modality]]
        values = [value for value in values if value is not None]
        per_query[query_id] = min(values) if values else None
    return {"per_query": per_query, "summary": rank_summary(list(per_query.values()), ET_CUTOFFS)}


def load_fixed_graph(rows: list[dict[str, Any]]) -> dict[str, Any]:
    wanted = {str(row["query_id"]) for row in rows}
    graph_rows = {}
    with gzip.open(PATH_GRAPH, "rt", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            query_id = str(record["query_id"])
            if query_id in wanted:
                graph_rows[query_id] = record
    if set(graph_rows) != wanted:
        raise ValueError(f"fixed path graph is missing {len(wanted - set(graph_rows))} dev queries")
    triples = []
    for query_id, record in sorted(graph_rows.items()):
        for candidate in record.get("E_pre_retention", []):
            target_id = str(candidate["target_id"])
            for path in candidate.get("paths", []):
                evidence_id = path.get("evidence_id")
                if evidence_id:
                    triples.append((query_id, str(evidence_id), target_id))
    graph_hash = hashlib.sha256(json.dumps(sorted(triples), separators=(",", ":")).encode()).hexdigest()
    return {"rows": graph_rows, "triples": triples, "sha256": graph_hash, "queries": len(graph_rows)}


@torch.inference_mode()
def score_pairs(
    model: torch.nn.Module,
    store: FeatureStore,
    pairs: list[tuple[str, str, str, str]],
    device: torch.device,
    batch_size: int = 4096,
) -> list[float]:
    grouped: dict[tuple[str, str], list[tuple[int, tuple[str, str, str, str]]]] = defaultdict(list)
    for index, pair in enumerate(pairs):
        grouped[(pair[1], pair[3])].append((index, pair))
    scores = [0.0] * len(pairs)
    for (source_type, dest_type), indexed_pairs in grouped.items():
        for start in range(0, len(indexed_pairs), batch_size):
            batch = indexed_pairs[start : start + batch_size]
            source = torch.stack([store.embedding_features(pair[0]).embedding for _index, pair in batch]).to(device)
            dest = torch.stack([store.embedding_features(pair[2]).embedding for _index, pair in batch]).to(device)
            values = model.score_embeddings(source, source_type, dest, dest_type)
            for (index, _pair), value in zip(batch, values.detach().float().cpu().tolist()):
                scores[index] = value
    return scores


def fixed_path_scores(model: torch.nn.Module, store: FeatureStore, graph: dict[str, Any], rows: list[dict[str, Any]], device: torch.device) -> dict[str, Any]:
    triples = graph["triples"]
    edge_qe = list(dict.fromkeys((q, e, "text" if e.startswith("asset_text_") else "image") for q, e, _t in triples))
    edge_et = list(dict.fromkeys((e, t, "table") for _q, e, t in triples))
    qe_pairs = [(q, "table", e, typ) for q, e, typ in edge_qe]
    et_pairs = [(e, "text" if e.startswith("asset_text_") else "image", t, "table") for e, t, _typ in edge_et]
    qe_values = score_pairs(model, store, qe_pairs, device)
    et_values = score_pairs(model, store, et_pairs, device)
    qe_map = {(q, e): value for (q, _e, e, _typ), value in zip(qe_pairs, qe_values)}
    et_map = {(e, t): value for (e, t, _typ), value in zip(edge_et, et_values)}
    by_query_target: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for q, e, t in triples:
        by_query_target[q][t].append(qe_map[(q, e)] + et_map[(e, t)])
    positive = {str(row["query_id"]): str(row["positive_target_ids"][0]) for row in rows}
    per_query = {}
    qe_positive = []
    et_positive = []
    best_positive = []
    lse_positive = []
    margins = []
    hits10 = []
    for query_id, target_scores in by_query_target.items():
        ranking_scores = {target: float(torch.logsumexp(torch.tensor(values), 0)) for target, values in target_scores.items()}
        ranking = sorted(ranking_scores, key=lambda target: (-ranking_scores[target], target))
        target = positive.get(query_id)
        if target is None or target not in target_scores:
            hits10.append(0)
            continue
        target_paths = target_scores[target]
        positive_evidence = set()
        row = graph["rows"][query_id]
        for candidate in row.get("E_pre_retention", []):
            if str(candidate["target_id"]) == target:
                positive_evidence.update(str(path["evidence_id"]) for path in candidate.get("paths", []) if path.get("evidence_id"))
        qe_positive.extend(qe_map[(query_id, evidence)] for evidence in positive_evidence if (query_id, evidence) in qe_map)
        et_positive.extend(et_map[(evidence, target)] for evidence in positive_evidence if (evidence, target) in et_map)
        best_positive.append(max(target_paths))
        lse_positive.append(float(torch.logsumexp(torch.tensor(target_paths), 0)))
        negatives = [ranking_scores[value] for value in ranking if value != target]
        margins.append(ranking_scores[target] - (max(negatives) if negatives else ranking_scores[target]))
        hits10.append(int(ranking.index(target) < 10))
        per_query[query_id] = {"positive_best_path": max(target_paths), "positive_lse": lse_positive[-1], "margin_vs_top_negative": margins[-1], "rank": ranking.index(target) + 1}
    return {
        "graph_sha256": graph["sha256"],
        "triples": len(triples),
        "positive_qe_score": statistics.fmean(qe_positive) if qe_positive else None,
        "positive_et_score": statistics.fmean(et_positive) if et_positive else None,
        "positive_best_path": statistics.fmean(best_positive) if best_positive else None,
        "positive_lse": statistics.fmean(lse_positive) if lse_positive else None,
        "positive_negative_margin": statistics.fmean(margins) if margins else None,
        "fixed_path_e_r10": statistics.fmean(hits10) if hits10 else 0.0,
        "queries_with_positive_path": len(per_query),
        "per_query": per_query,
    }


def parameter_drift(seed: int, device: torch.device) -> dict[str, Any]:
    states = {}
    for step in STEPS:
        payload = torch.load(checkpoint_path(seed, step), map_location="cpu", weights_only=False)
        states[step] = payload["state_dict"]
    groups = {
        "P": ("projections.", "projection_residual_inputs.", "projection_residual_outputs."),
        "R": ("relations.", "relation_as.", "relation_bs.", "confidence_alphas.", "confidence_biases."),
    }
    output = {}
    for left, right in ((356, 500), (500, 659), (356, 659)):
        output[f"{left}->{right}"] = {}
        for group, prefixes in groups.items():
            numerator = 0.0
            denominator = 0.0
            per_key = {}
            keys = [key for key in states[left] if key.startswith(prefixes)]
            for key in keys:
                if key not in states[right] or not torch.is_floating_point(states[left][key]):
                    continue
                delta = (states[right][key].float() - states[left][key].float()).norm().item()
                base = states[left][key].float().norm().item()
                numerator += delta * delta
                denominator += base * base
                per_key[key] = delta / base if base else None
            output[f"{left}->{right}"][group] = {"relative_frobenius": math.sqrt(numerator / denominator) if denominator else None, "per_key": per_key}
    return output


def _is_projection_key(key: str) -> bool:
    return key.startswith(("projections.", "projection_residual_inputs.", "projection_residual_outputs."))


def _is_relation_key(key: str) -> bool:
    return key.startswith(("relations.", "relation_as.", "relation_bs.", "confidence_alphas.", "confidence_biases."))


def hybrid_model(seed: int, p_step: int, r_step: int, device: torch.device) -> torch.nn.Module:
    """Build an inference-only P/R cross-swap with exact state-key checks."""

    base_path = checkpoint_path(seed, 356)
    base = load_student(base_path, device).eval()
    base_state = {key: value.detach().clone() for key, value in base.state_dict().items()}
    p_state = torch.load(checkpoint_path(seed, p_step), map_location="cpu", weights_only=False)["state_dict"]
    r_state = torch.load(checkpoint_path(seed, r_step), map_location="cpu", weights_only=False)["state_dict"]
    for key in base_state:
        if _is_projection_key(key):
            base_state[key] = p_state[key]
        elif _is_relation_key(key):
            base_state[key] = r_state[key]
    incompatible = base.load_state_dict(base_state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"hybrid state mismatch: {incompatible}")
    base.eval()
    return base


@torch.inference_mode()
def run_restore_seed(seed: int, device: torch.device, *, output_root: Path = DIAG_ROOT) -> dict[str, Any]:
    started = time.monotonic()
    rows = read_dev_rows()
    index_dir = INDEX_ROOT / f"B5_356{'' if seed == 13 else '_seed29'}"
    ids = read_ids(index_dir)
    query_ids = [str(row["query_id"]) for row in rows]
    witness_ids = [str(value) for row in rows for value in row.get("positive_evidence_by_target", {}).get(str(row["positive_target_ids"][0]), [])]
    store = FeatureStore.from_path(FEATURES, cache_size=0)
    print(json.dumps({"seed": seed, "event": "preload_embeddings_restore", "objects": len(set([*ids["table"], *ids["text"], *ids["image"], *query_ids, *witness_ids]))}), flush=True)
    store.preload_embedding_matrix([*ids["table"], *ids["text"], *ids["image"], *query_ids, *witness_ids])
    types = object_types(store, witness_ids)
    witness_map = modality_witnesses(rows, types)
    positive_by_query_modality = {modality: {query_id: set(witness_map[query_id][modality]) for query_id in query_ids} for modality in ("text", "image")}
    positive_by_witness = {evidence_id: {str(row["positive_target_ids"][0])} for row in rows for evidence_id in witness_map[str(row["query_id"])] ["all"]}
    graph = load_fixed_graph(rows)
    states = {"H356": (356, 356), "H659": (659, 659), "HP": (356, 659), "HR": (659, 356)}
    results = {"seed": seed, "states": {}, "fixed_graph": {"triple_sha256": graph["sha256"], "triples": len(graph["triples"])}}
    for state_name, (p_step, r_step) in states.items():
        model = hybrid_model(seed, p_step, r_step, device)
        destination_matrices = {}
        for destination_type in ("table", "text", "image"):
            raw = torch.stack([store.embedding_features(object_id).embedding for object_id in ids[destination_type]]).to(device)
            destination_matrices[destination_type] = model.index_vector(raw, destination_type).detach()
        state_result = {"P_source_step": p_step, "R_source_step": r_step, "relations": {}}
        for modality in ("text", "image"):
            qe_exact, _ = exact_relation_ranks(model, store, query_ids, "table", ids[modality], modality, positive_by_query_modality[modality], device, destination_matrices[modality])
            qe_best = [min(qe_exact.get(query_id, {}).values(), default=None) for query_id in query_ids if positive_by_query_modality[modality][query_id]]
            et_sources = list(dict.fromkeys(evidence_id for query_id in query_ids for evidence_id in witness_map[query_id][modality]))
            et_positive = {evidence_id: positive_by_witness[evidence_id] for evidence_id in et_sources}
            et_exact, _ = exact_relation_ranks(model, store, et_sources, modality, ids["table"], "table", et_positive, device, destination_matrices["table"])
            state_result["relations"][f"QE_{modality}"] = rank_summary(qe_best, QE_CUTOFFS)
            state_result["relations"][f"ET_{modality}"] = aggregate_query_ranks(rows, et_exact, witness_map, modality)["summary"]
        state_result["fixed_path"] = fixed_path_scores(model, store, graph, rows, device)
        results["states"][state_name] = state_result
        del destination_matrices, model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(json.dumps({"seed": seed, "state": state_name, "event": "restore_complete", "elapsed_seconds": time.monotonic() - started}), flush=True)
    results["runtime_seconds"] = time.monotonic() - started
    return results


def run_seed(seed: int, device: torch.device, *, output_root: Path = DIAG_ROOT) -> dict[str, Any]:
    started = time.monotonic()
    rows = read_dev_rows()
    index_dir = INDEX_ROOT / f"B5_356{'' if seed == 13 else '_seed29'}"
    ids = read_ids(index_dir)
    query_ids = [str(row["query_id"]) for row in rows]
    witness_ids = [str(value) for row in rows for value in row.get("positive_evidence_by_target", {}).get(str(row["positive_target_ids"][0]), [])]
    all_ids = [*ids["table"], *ids["text"], *ids["image"], *query_ids, *witness_ids]
    store = FeatureStore.from_path(FEATURES, cache_size=0)
    print(json.dumps({"seed": seed, "event": "preload_embeddings", "objects": len(set(all_ids))}), flush=True)
    store.preload_embedding_matrix(all_ids)
    types = object_types(store, witness_ids)
    witness_map = modality_witnesses(rows, types)
    graph = load_fixed_graph(rows)
    positive_by_query_modality = {
        modality: {query_id: set(witness_map[query_id][modality]) for query_id in query_ids}
        for modality in ("text", "image")
    }
    positive_by_witness = {
        evidence_id: {str(row["positive_target_ids"][0])}
        for row in rows
        for evidence_id in witness_map[str(row["query_id"])] ["all"]
    }
    results: dict[str, Any] = {"seed": seed, "inputs": {"features": {"path": str(FEATURES.resolve()), "sha256": sha256(FEATURES / "manifest.jsonl")}, "dev_target_lists": {"path": str(DEV_TARGET_LISTS.resolve()), "sha256": sha256(DEV_TARGET_LISTS)}, "fixed_graph": {"path": str(PATH_GRAPH.resolve()), "sha256": sha256(PATH_GRAPH), "triple_sha256": graph["sha256"], "queries": graph["queries"], "triples": len(graph["triples"])}}, "checkpoints": {}, "parameter_drift": parameter_drift(seed, device), "relation_identity": {"QE": ["table_to_text", "table_to_image"], "ET": ["text_to_table", "image_to_table"], "labels_used_for": "offline ranks only"}}
    for step in STEPS:
        checkpoint = checkpoint_path(seed, step)
        identity = checkpoint_identity(seed, step)
        model = load_student(checkpoint, device).eval()
        checkpoint_result: dict[str, Any] = {"identity": identity, "relations": {}, "fixed_path": None}
        funnel_inputs: dict[str, dict[str, Any]] = {}
        destination_matrices = {}
        for destination_type in ("table", "text", "image"):
            raw = torch.stack([store.embedding_features(object_id).embedding for object_id in ids[destination_type]]).to(device)
            destination_matrices[destination_type] = model.index_vector(raw, destination_type).detach()
        for modality in ("text", "image"):
            exact_ranks, exact_top = exact_relation_ranks(model, store, query_ids, "table", ids[modality], modality, positive_by_query_modality[modality], device, destination_matrices[modality])
            ann_dir = INDEX_ROOT / f"B5_{step}{'' if seed == 13 else '_seed29'}"
            ann_ranks, ann_top = ann_relation_ranks(model, store, ann_dir, identity["sha256"], query_ids, modality, positive_by_query_modality[modality], device)
            exact_best = [min(exact_ranks.get(query_id, {}).values(), default=None) for query_id in query_ids if positive_by_query_modality[modality][query_id]]
            ann_best = [min(ann_ranks.get(query_id, {}).values(), default=None) for query_id in query_ids if positive_by_query_modality[modality][query_id]]
            checkpoint_result["relations"][f"QE_{modality}"] = {"exact": rank_summary(exact_best, QE_CUTOFFS), "ann": rank_summary(ann_best, QE_CUTOFFS), "per_query_best_exact": {query_id: min(exact_ranks.get(query_id, {}).values(), default=None) for query_id in query_ids if positive_by_query_modality[modality][query_id]}, "per_query_best_ann": {query_id: min(ann_ranks.get(query_id, {}).values(), default=None) for query_id in query_ids if positive_by_query_modality[modality][query_id]}, "exact_top_k": concentration([exact_top[query_id][:50] for query_id in query_ids if positive_by_query_modality[modality][query_id]]), "ann_top_k": concentration([ann_top[query_id][:50] for query_id in query_ids if positive_by_query_modality[modality][query_id]])}
            et_sources = [evidence_id for query_id in query_ids for evidence_id in witness_map[query_id][modality]]
            et_sources = list(dict.fromkeys(et_sources))
            et_positive = {evidence_id: positive_by_witness[evidence_id] for evidence_id in et_sources}
            et_exact, et_exact_top = exact_relation_ranks(model, store, et_sources, modality, ids["table"], "table", et_positive, device, destination_matrices["table"])
            et_ann, et_ann_top = ann_relation_ranks(model, store, INDEX_ROOT / f"B5_{step}{'' if seed == 13 else '_seed29'}", identity["sha256"], et_sources, "table", et_positive, device)
            checkpoint_result["relations"][f"ET_{modality}"] = {"exact": aggregate_query_ranks(rows, et_exact, witness_map, modality), "ann": aggregate_query_ranks(rows, et_ann, witness_map, modality), "exact_top_k": concentration([et_exact_top[evidence_id][:50] for evidence_id in et_sources]), "ann_top_k": concentration([et_ann_top[evidence_id][:50] for evidence_id in et_sources])}
            funnel_inputs[modality] = {"qe_ann": ann_ranks, "et_ann": et_ann}
        funnel_rows = []
        for row in rows:
            query_id = str(row["query_id"])
            target_id = str(row["positive_target_ids"][0])
            witness_rows = []
            for modality in ("text", "image"):
                for evidence_id in witness_map[query_id][modality]:
                    qe_rank = funnel_inputs[modality]["qe_ann"].get(query_id, {}).get(evidence_id)
                    et_rank = funnel_inputs[modality]["et_ann"].get(evidence_id, {}).get(target_id)
                    witness_rows.append((qe_rank, et_rank))
            qe_admitted = any(qe is not None and qe <= 20 for qe, _et in witness_rows)
            et_conditional = any(qe is not None and qe <= 20 and et is not None and et <= 20 for qe, et in witness_rows)
            funnel_rows.append({"query_id": query_id, "qe_witness_admitted": qe_admitted, "et_conditional_success": et_conditional, "e_target_admitted": et_conditional})
        checkpoint_result["funnel"] = {"budget": {"qe": 20, "et": 20}, "query_macro": {key: sum(row[key] for row in funnel_rows) / len(funnel_rows) if funnel_rows else 0.0 for key in ("qe_witness_admitted", "et_conditional_success", "e_target_admitted")}, "queries": len(funnel_rows), "per_query": funnel_rows}
        checkpoint_result["fixed_path"] = fixed_path_scores(model, store, graph, rows, device)
        results["checkpoints"][str(step)] = checkpoint_result
        del destination_matrices, model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(json.dumps({"seed": seed, "step": step, "event": "checkpoint_complete", "elapsed_seconds": time.monotonic() - started}), flush=True)
    results["runtime_seconds"] = time.monotonic() - started
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=SEEDS, action="append", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, default=DIAG_ROOT / "RESULTS.json")
    parser.add_argument("--restore-only", action="store_true", help="Run only the 2x2 P/R inference-only restore audit.")
    args = parser.parse_args()
    seeds = tuple(args.seed or SEEDS)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use --device cpu only for a smoke test")
    runner = run_restore_seed if args.restore_only else run_seed
    payload = {"format_version": 1, "plan": "MMDD_C1_Evidence_Collapse_Next_Experiment_Plan.md", "status": "completed", "mode": "restore" if args.restore_only else "A_B_D", "device": str(device), "seeds": {str(seed): runner(seed, device) for seed in seeds}}
    write_json(args.output, payload)
    print(json.dumps({"output": str(args.output.resolve()), "seeds": list(seeds), "status": "completed"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
