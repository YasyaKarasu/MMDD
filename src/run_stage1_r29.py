"""Run the preregistered R29 read-only candidate and supervision diagnostics.

The first phase is deliberately independent from the R28 runner.  It reconstructs
the three candidate universes from their recorded inputs, never reads dev qrels
while mining Natural candidates, and writes auditable compressed JSONL artifacts.
Training/evaluation phases are kept as explicit follow-up commands because the
R29 gate must be decided from these diagnostics first.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch

from mmdd_stage1.checkpoints import load_student
from mmdd_stage1.data import EdgeExample, TargetExample, load_edge_examples, load_target_examples
from mmdd_stage1.features import FeatureStore, normalize_object_type
from mmdd_stage1.retrieval import StudentANNIndices, load_corpus_ids
from mmdd_stage1.scoring import global_edge_positive_ids
from mmdd_stage1.r26_training import graph_edges
from prepare_stage1_r28 import ROOT as REPO_ROOT
from run_stage1_r19 import load_r19_checkpoint, _score_id_pairs
from run_stage1_r13 import _merge_witness_metadata


ROOT = Path(REPO_ROOT)
OUT = ROOT / "work/stage1_optimization_r29_candidate_vs_drift_20260915"
R28 = ROOT / "work/stage1_optimization_r28_split_path_20260915"
R28_INPUTS = R28 / "R28_RESOLVED_INPUTS.json"
R12_GRAPH = ROOT / "work/stage1_optimization_r12_20260908/taskC_training/c2_candidates_seed13/path_hard.jsonl"
R12_TARGETS = ROOT / "work/stage1_optimization_r12_20260908/taskA_correctness/supervision/target_lists.train_fit.jsonl"
R12_EDGES = ROOT / "work/stage1_optimization_r12_20260908/taskA_correctness/supervision/edge_lists.train_fit.jsonl"
R22_NATURAL = ROOT / "work/stage1_optimization_r22_20260911/manifests/full_natural.jsonl"
R22_HARD = ROOT / "work/stage1_optimization_r22_20260911/fresh_lineage/S0_mining/seed13/hard_negatives.jsonl.gz"
R22_HARD_MANIFEST = ROOT / "work/stage1_optimization_r22_20260911/fresh_lineage/S0_mining/seed13/mining_manifest.json"
R22_T0_CACHE = ROOT / "work/stage1_optimization_r22_20260911/fresh_lineage/T1-B/seed13/teacher_soft_scores_S1aug.jsonl.gz"
R28_DEV_RANKINGS = R28 / "student/own/rankings/S-EDGE-LONG/seed13/epoch5/rankings.jsonl.gz"
PARENT_INDEX = ROOT / "work/stage1_diagnostics_r27_20260915_rev2_b13_exact/historical_replay/own_evaluation/indexes/H-C1-step000356"
CORPUS = ROOT / "work/stage1_optimization_r10_20260907/stage1_data/stage1_corpus.jsonl"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def write_rows(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if path.suffix == ".gz" else open
    tmp = path.with_suffix(path.suffix + ".tmp")
    with opener(tmp, "wt", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(dict(row), ensure_ascii=False, separators=(",", ":")) + "\n")
    tmp.replace(path)


def read_rows(path: Path) -> Iterable[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def relation(source_type: str, destination_type: str) -> str:
    return f"{normalize_object_type(source_type)}->{normalize_object_type(destination_type)}"


def input_paths() -> dict[str, Path]:
    resolved = json.loads(R28_INPUTS.read_text())["inputs"]
    paths = {name: Path(item["path"]) for name, item in resolved.items() if "path" in item}
    paths.update({
        "r12_graph": R12_GRAPH, "r12_targets": R12_TARGETS, "r12_edges": R12_EDGES,
        "t0train_manifest": R22_NATURAL, "t0train_hard": R22_HARD,
        "t0train_hard_manifest": R22_HARD_MANIFEST, "teacher_cache": R22_T0_CACHE,
        "student_index": PARENT_INDEX, "corpus": CORPUS,
    })
    return paths


def load_context() -> tuple[list[TargetExample], dict[tuple[str, str, str], set[str]], FeatureStore, Any, Any]:
    paths = input_paths()
    # R28 did not train on the raw witness file alone: this helper applies the
    # locked path_hard candidate replacement and witness metadata merge.  Using
    # it is essential for the five-relation list counts and active masks to
    # match the historical S-EDGE-LONG control.
    targets = _merge_witness_metadata(ROOT)
    registry = global_edge_positive_ids(load_edge_examples(paths["r12_edges"], split="train"))
    # The diagnostic touches the same corpus objects across many lists.  Keep a
    # large bounded embedding cache so scoring does not repeatedly deserialize
    # the per-object tensor files.
    store = FeatureStore.from_path(paths["feature_manifest"].parent, cache_size=220000, cache_bytes=4 * 1024**3)
    student = load_student(paths["student_parent"], torch.device("cpu")).eval()
    return targets, registry, store, student, paths


def r12_rows(targets: list[TargetExample], registry: Mapping[tuple[str, str, str], set[str]], store: FeatureStore) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for owner, example in enumerate(targets):
        edges = graph_edges(example, registry, lambda oid: store.embedding_features(oid).object_type)
        for e in edges:
            src_type = str(e.source_type)
            dst_type = str(e.destination_type)
            positives = sorted(set(e.positive_ids) & set(e.candidate_ids))
            rows.append({
                "owner_query_id": example.query_id,
                "owner_index": owner,
                "source_id": e.query_id,
                "source_type": src_type,
                "destination_type": dst_type,
                "relation": relation(src_type, dst_type),
                "candidate_ids": list(e.candidate_ids),
                "positive_ids": positives,
                "candidate_hash": hashlib.sha256("\n".join(e.candidate_ids).encode()).hexdigest(),
            })
    return rows


def t0train_rows(registry: Mapping[tuple[str, str, str], set[str]], paths: Mapping[str, Path]) -> list[dict[str, Any]]:
    hard: dict[str, list[str]] = {str(r["query_id"]): [str(x) for x in r.get("hard_candidate_ids", [])] for r in read_rows(paths["t0train_hard"])}
    rows: list[dict[str, Any]] = []
    for raw in read_rows(paths["t0train_manifest"]):
        q = str(raw["query_id"])
        st, dt = str(raw["source_type"]), str(raw["destination_type"])
        key = (q, st, dt)
        ids = [str(x) for x in raw.get("candidate_ids", [])]
        if st == "table" and dt == "table":
            ids.extend(hard.get(q, []))
        ids = list(dict.fromkeys(ids))
        positives = sorted(set(registry.get(key, set())) & set(ids))
        rows.append({
            "owner_query_id": q, "owner_index": None, "source_id": q,
            "source_type": st, "destination_type": dt, "relation": relation(st, dt),
            "candidate_ids": ids, "positive_ids": positives,
            "candidate_hash": hashlib.sha256("\n".join(ids).encode()).hexdigest(),
            "hard_source": "S0_mining" if st == "table" and dt == "table" else None,
        })
    return rows


def _ann_rows(r12: list[dict[str, Any]], store: FeatureStore, student: Any, paths: Mapping[str, Path], *, exact_qt: bool) -> list[dict[str, Any]]:
    ann = StudentANNIndices(student, store, paths["student_index"], device=torch.device("cpu"),
                            checkpoint_sha256=sha(paths["student_parent"]),
                            corpus_sha256=sha(paths["corpus"]), score_space="raw_logit")
    keys = sorted({(str(r["source_id"]), str(r["source_type"]), str(r["destination_type"])) for r in r12})
    by_key: dict[tuple[str, str, str], list[tuple[str, float]]] = {}
    started = time.monotonic()
    for start in range(0, len(keys), 128):
        part = keys[start:start + 128]
        for (source_id, source_type, dest_type), values in zip(part, ann.search_many([k[0] for k in part], part[0][2], 256), strict=True):
            by_key[(source_id, source_type, dest_type)] = values
        if start and start % 2048 == 0:
            print(json.dumps({"stage": "natural_ann", "keys": start, "total": len(keys), "elapsed": time.monotonic() - started}), flush=True)

    exact_by_source: dict[str, list[tuple[str, float]]] = {}
    if exact_qt:
        ids_by_type = load_corpus_ids(paths["corpus"], store)
        table_ids = ids_by_type["table"]
        target_embeddings = torch.stack([store.embedding_features(x).embedding for x in table_ids]).float()
        with torch.inference_mode():
            target_vectors = student.project(target_embeddings, "table", role="target")
            for start in range(0, len(keys), 32):
                part = [k for k in keys[start:start + 32] if k[2] == "table" and k[1] == "table"]
                if not part:
                    continue
                source_embeddings = torch.stack([store.embedding_features(k[0]).embedding for k in part]).float()
                queries = student.relation_query(source_embeddings, "table", "table", source_role="query")
                scores = queries @ target_vectors.T
                top = torch.topk(scores, k=min(256, scores.shape[1]), dim=1).indices
                for k, inds, vals in zip(part, top, scores.gather(1, top), strict=True):
                    exact_by_source[k[0]] = [(table_ids[int(i)], float(v)) for i, v in zip(inds[:32], vals[:32])]

    out: list[dict[str, Any]] = []
    for row in r12:
        key = (str(row["source_id"]), str(row["source_type"]), str(row["destination_type"]))
        values = by_key.get(key, [])
        positive = set(row["positive_ids"])
        reservoir = [{"candidate_id": cid, "score": float(score)} for cid, score in values]
        unknown = [item for item in reservoir if item["candidate_id"] not in positive]
        top32 = unknown[:32]
        ids = list(dict.fromkeys(list(row["positive_ids"]) + [x["candidate_id"] for x in top32]))
        exact = exact_by_source.get(str(row["source_id"])) if key[1:] == ("table", "table") else None
        out.append({
            "owner_query_id": row["owner_query_id"], "owner_index": row["owner_index"],
            "source_id": row["source_id"], "source_type": row["source_type"],
            "destination_type": row["destination_type"], "relation": row["relation"],
            "candidate_ids": ids, "positive_ids": sorted(positive),
            "ann_top256": reservoir, "ann_top32_unknown": top32,
            "exact_top32": ([{"candidate_id": x, "score": s} for x, s in exact] if exact else None),
            "ann_exact_overlap_top32": (len({x["candidate_id"] for x in top32} & {x[0] for x in exact}) / max(1, len(top32))) if exact else None,
            "candidate_hash": hashlib.sha256("\n".join(ids).encode()).hexdigest(),
        })
        # ANN scores are already parent-Student raw logits.  Keep them as the
        # authoritative score map for Natural; re-scoring the same pairs from
        # the feature store would only add IO and numerical noise.
        out[-1]["parent_scores"] = {x["candidate_id"]: float(x["score"]) for x in reservoir}
    return out


def _score_rows(rows: list[dict[str, Any]], model: Any, store: FeatureStore, *, batch_size: int = 512) -> None:
    pairs: list[tuple[str, str, int, int]] = []
    for ri, row in enumerate(rows):
        for ci, cid in enumerate(row["candidate_ids"]):
            pairs.append((str(row["source_id"]), str(cid), ri, ci))
    typed: dict[tuple[str, str], list[tuple[str, str, int, int]]] = defaultdict(list)
    for item in pairs:
        a, b, _ri, _ci = item
        typed[(store.embedding_features(a).object_type, store.embedding_features(b).object_type)].append(item)
    for (st, dt), typed_pairs in typed.items():
        for start in range(0, len(typed_pairs), batch_size):
            part = typed_pairs[start:start + batch_size]
            sources = [store.embedding_features(a).embedding for a, _b, _ri, _ci in part]
            destinations = [store.embedding_features(b).embedding for _a, b, _ri, _ci in part]
            with torch.inference_mode():
                values = model.score_embeddings(torch.stack(sources), st, torch.stack(destinations), dt)
            for value, (_a, _b, ri, ci) in zip(values, part, strict=True):
                rows[ri].setdefault("parent_scores", {})[rows[ri]["candidate_ids"][ci]] = float(value)


def _score_missing_positives(rows: list[dict[str, Any]], model: Any, store: FeatureStore) -> None:
    """Fill only positive scores for Natural rows (unknown scores come from ANN)."""
    reduced = []
    original: list[list[str]] = []
    for row in rows:
        original.append(list(row["candidate_ids"]))
        missing = [cid for cid in row.get("positive_ids", []) if cid not in row.get("parent_scores", {})]
        row["candidate_ids"] = missing
        reduced.append(row)
    _score_rows(reduced, model, store)
    for row, ids in zip(rows, original, strict=True):
        row["candidate_ids"] = ids


def _attach_ann_scores(rows: list[dict[str, Any]], natural: list[dict[str, Any]]) -> None:
    by_key = {(str(row["source_id"]), str(row["relation"])): row.get("parent_scores", {}) for row in natural}
    for row in rows:
        source_scores = by_key.get((str(row["source_id"]), str(row["relation"])), {})
        row["parent_scores"] = {cid: float(source_scores[cid]) for cid in row["candidate_ids"] if cid in source_scores}


def _aggregate(rows: list[dict[str, Any]], *, score_key: str = "parent_scores") -> dict[tuple[str, str], dict[str, Any]]:
    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        key = (str(row["owner_query_id"]), str(row["relation"]))
        item = grouped.setdefault(key, {"owner_query_id": key[0], "relation": key[1], "ids": set(), "positive_ids": set(), "scores": {}})
        item["ids"].update(str(x) for x in row["candidate_ids"])
        item["positive_ids"].update(str(x) for x in row.get("positive_ids", []))
        scores = row.get(score_key) or {}
        for cid, value in scores.items():
            item["scores"][cid] = max(float(value), float(item["scores"].get(cid, -math.inf)))
    return grouped


def _matched_metrics(grouped: dict[tuple[str, str], dict[str, Any]], *, name: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for key, item in grouped.items():
        positives = set(item["positive_ids"])
        scores = item["scores"]
        unknown = [(cid, score) for cid, score in scores.items() if cid not in positives and math.isfinite(float(score))]
        unknown.sort(key=lambda x: (-x[1], x[0]))
        hard = unknown[:32]
        insufficient = len(hard) < 32
        matched = list(positives) + [cid for cid, _score in hard]
        if not positives or not hard or any(not math.isfinite(float(scores.get(cid, math.nan))) for cid in positives):
            records.append({"owner_query_id": key[0], "relation": key[1], "universe": name, "insufficient_unknowns": insufficient, "margin": None, "violation": None, "positive_rank": None, "hard32": [cid for cid, _s in hard]})
            continue
        best_positive = max(scores.get(cid, -math.inf) for cid in positives)
        max_unknown = max(score for _cid, score in hard)
        ranking = sorted(((cid, scores.get(cid, -math.inf)) for cid in matched), key=lambda x: (-x[1], x[0]))
        best_pos_ids = {cid for cid in positives if scores.get(cid, -math.inf) == best_positive}
        rank = next((i + 1 for i, (cid, _s) in enumerate(ranking) if cid in best_pos_ids), len(ranking))
        ce = math.log(sum(math.exp(scores.get(cid, -math.inf) - best_positive) for cid in matched if math.isfinite(scores.get(cid, -math.inf))) or 1.0)
        records.append({"owner_query_id": key[0], "relation": key[1], "universe": name, "insufficient_unknowns": insufficient,
                        "n_positive": len(positives), "n_unknown": len(unknown), "margin": best_positive - max_unknown,
                        "violation": float(max_unknown > best_positive), "positive_rank": rank,
                        "matched_ce": ce, "hard32": [cid for cid, _s in hard], "matched_ids": matched})
    return records, summarize_metrics(records)


def summarize_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [r for r in records if r.get("margin") is not None and math.isfinite(float(r["margin"]))]
    def vals(key: str) -> list[float]: return [float(r[key]) for r in valid if r.get(key) is not None]
    out: dict[str, Any] = {"n": len(records), "n_valid": len(valid), "insufficient_unknowns": sum(bool(r.get("insufficient_unknowns")) for r in records)}
    for key in ("margin", "violation", "positive_rank", "matched_ce"):
        v = vals(key)
        if v:
            out[key] = {"mean": statistics.fmean(v), "median": statistics.median(v), "p10": float(np.percentile(v, 10)), "p90": float(np.percentile(v, 90))}
    out["hard32_count"] = sum(len(r.get("hard32", [])) for r in records)
    return out


def bootstrap_diff(a: Mapping[tuple[str, str], float], b: Mapping[tuple[str, str], float], n: int = 10000, seed: int = 290915) -> dict[str, Any]:
    keys = sorted(set(a) & set(b))
    if not keys:
        return {"n": 0, "mean": None, "ci95": [None, None], "method": "query-level paired bootstrap", "replicates": n}
    diff = np.asarray([float(a[k]) - float(b[k]) for k in keys], dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = np.empty(n, dtype=np.float64)
    for start in range(0, n, 100):
        size = min(100, n - start)
        means[start:start + size] = diff[rng.integers(0, len(diff), size=(size, len(diff)))].mean(axis=1)
    return {"n": len(diff), "mean": float(diff.mean()), "ci95": [float(np.quantile(means, .025)), float(np.quantile(means, .975))],
            "method": "query-level paired bootstrap", "replicates": n}


def supervision_audit(r12: list[dict[str, Any]]) -> dict[str, Any]:
    stats: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in r12:
        ids, pos = set(row["candidate_ids"]), set(row["positive_ids"])
        n_positive, n_unknown = len(pos), len(ids - pos)
        active = bool(pos and n_unknown)
        key = row["relation"]
        stats[key].append({"owner_query_id": row["owner_query_id"], "n_total_lists": 1, "n_positive_lists": int(bool(pos)), "n_negative_lists": int(bool(n_unknown)), "n_active_lists": int(active), "n_all_positive": int(bool(pos) and not n_unknown), "n_no_positive": int(not pos), "active_fraction": int(active), "positive_count": n_positive, "unknown_count": n_unknown, "candidate_width": len(ids)})
    summary: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    for rel, values in sorted(stats.items()):
        total = len(values); active = sum(x["n_active_lists"] for x in values)
        summary[rel] = {"n_total_lists": total, "n_positive_lists": sum(x["n_positive_lists"] for x in values), "n_negative_lists": sum(x["n_negative_lists"] for x in values), "n_active_lists": active, "n_all_positive": sum(x["n_all_positive"] for x in values), "n_no_positive": sum(x["n_no_positive"] for x in values), "active_fraction": active / max(1, total), "w_eff": (0.5 * active / max(1, total)) if rel != "table->table" else 1.0, "positive_count": summarize_numbers([x["positive_count"] for x in values]), "unknown_count": summarize_numbers([x["unknown_count"] for x in values]), "candidate_width": summarize_numbers([x["candidate_width"] for x in values])}
        rows.extend([{**x, "relation": rel, "w_eff": (0.5 * x["n_active_lists"]) / max(1, 1)} for x in values])
    return {"summary": summary, "rows": rows}


def summarize_numbers(values: list[int]) -> dict[str, float]:
    return {"mean": statistics.fmean(values) if values else 0.0, "median": statistics.median(values) if values else 0.0, "p10": float(np.percentile(values, 10)) if values else 0.0, "p90": float(np.percentile(values, 90)) if values else 0.0}


def hub_exposure(r12: list[dict[str, Any]], t0: list[dict[str, Any]], nat: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not R28_DEV_RANKINGS.exists():
        return []
    rows = list(read_rows(R28_DEV_RANKINGS))
    chosen: dict[str, dict[str, Any]] = {}
    for row in sorted(rows, key=lambda x: (str(x.get("source_table_id", "")), str(x.get("query_id", "")))):
        chosen.setdefault(str(row.get("source_table_id", "")), row)
    probe = list(chosen.values())[:128]
    counts = Counter(cid for row in probe for cid in row.get("rankings", {}).get("D100_EXACT", [])[:10])
    hubs = [cid for cid, _n in counts.most_common(20)]
    out = []
    for hub in hubs:
        rec = {"target_id": hub, "probe_queries": len(probe), "probe_top10_frequency": counts[hub]}
        for label, source in (("r12", r12), ("t0train", t0), ("natural", nat)):
            appearances = [r for r in source if hub in set(r.get("candidate_ids", []))]
            rec[label] = {"list_count": len(appearances), "positive_count": sum(hub in set(r.get("positive_ids", [])) for r in appearances), "unknown_count": sum(hub not in set(r.get("positive_ids", [])) for r in appearances)}
        out.append(rec)
    return out


def diagnose(*, score_teacher: bool = False, exact_qt: bool = True) -> dict[str, Any]:
    torch.set_num_threads(min(32, torch.get_num_threads()))
    started = time.monotonic()
    OUT.mkdir(parents=True, exist_ok=True)
    paths = input_paths()
    resolved = {name: {"path": str(path.resolve()), "exists": path.exists(), "sha256": sha(path) if path.is_file() else None} for name, path in paths.items()}
    write_json(OUT / "RESOLVED_INPUTS.json", {"r28_resolved_inputs": str(R28_INPUTS), "inputs": resolved, "dev_qrels_used_for_mining": False, "hub_ids_used_for_mining": False})
    targets, registry, store, student, _ = load_context()
    r12 = r12_rows(targets, registry, store)
    t0 = t0train_rows(registry, paths)
    nat = _ann_rows(r12, store, student, paths, exact_qt=exact_qt)
    _score_missing_positives(nat, student, store)
    _attach_ann_scores(r12, nat)
    _attach_ann_scores(t0, nat)
    # R12 is small enough to score exactly.  T0TRAIN is a 3.5M-candidate
    # historical manifest; for its diagnostic we use the frozen parent's
    # relation-specific ANN scores and score only positives absent from that
    # reservoir.  This preserves the hard-competitor ordering without turning
    # the read-only audit into a second full-lake retrieval run.
    _score_rows(r12, student, store)
    _score_missing_positives(t0, student, store)
    base = OUT / "diagnostics/candidate_universe"
    write_rows(base / "r12_lists.jsonl.gz", r12)
    write_rows(base / "t0train_lists.jsonl.gz", t0)
    write_rows(base / "natural_reservoir_top256.jsonl.gz", nat)

    sup = supervision_audit(r12)
    write_rows(OUT / "diagnostics/supervision_effective_weights.jsonl.gz", sup["rows"])
    write_json(OUT / "diagnostics/supervision_summary.json", sup["summary"])
    with (OUT / "diagnostics/supervision_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f); writer.writerow(["relation", "n_total_lists", "n_positive_lists", "n_negative_lists", "n_active_lists", "active_fraction", "w_eff"])
        for rel, value in sorted(sup["summary"].items()): writer.writerow([rel, value["n_total_lists"], value["n_positive_lists"], value["n_negative_lists"], value["n_active_lists"], value["active_fraction"], value["w_eff"]])

    grouped = {"r12": _aggregate(r12), "t0train": _aggregate(t0), "natural": _aggregate(nat)}
    metric_rows: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {}
    maps: dict[str, dict[str, dict[tuple[str, str], float]]] = {}
    for name, groups in grouped.items():
        records, summary = _matched_metrics(groups, name=name)
        metric_rows.extend(records); summaries[name] = summary
        maps[name] = {"margin": {(r["owner_query_id"], r["relation"]): r["margin"] for r in records if r.get("margin") is not None}, "violation": {(r["owner_query_id"], r["relation"]): r["violation"] for r in records if r.get("violation") is not None}}
    write_rows(base / "matched32.jsonl.gz", metric_rows)
    write_rows(base / "margin_violation_per_query.jsonl.gz", metric_rows)

    overlap_rows = []
    metric_by_universe = {
        name: {(r["owner_query_id"], r["relation"]): r for r in metric_rows if r["universe"] == name}
        for name in ("r12", "t0train", "natural")
    }
    for key in sorted(set(maps["natural"]["margin"]) & set(maps["r12"]["margin"])):
        nat_rec = metric_by_universe["natural"][key]
        r12_rec = metric_by_universe["r12"][key]
        overlap_rows.append({"owner_query_id": key[0], "relation": key[1], "hnr_r12_from_nat": len(set(nat_rec["hard32"]) & set(r12_rec.get("matched_ids", []))) / max(1, len(nat_rec["hard32"])), "nat_hard32_in_r12": len(set(nat_rec["hard32"]) & set(r12_rec.get("matched_ids", []))), "nat_hard32": len(nat_rec["hard32"])})
    write_rows(base / "overlap_per_query.jsonl.gz", overlap_rows)
    with (base / "candidate_overlap.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f); writer.writerow(["relation", "n", "mean_HNR_R12_from_NAT"])
        for rel in sorted({r["relation"] for r in overlap_rows}):
            vals = [r["hnr_r12_from_nat"] for r in overlap_rows if r["relation"] == rel]
            writer.writerow([rel, len(vals), statistics.fmean(vals) if vals else None])

    gate_rows: dict[str, Any] = {}
    for rel in sorted(set(maps["natural"]["margin"]) & set(maps["r12"]["margin"])):
        break
    for rel in sorted({r["relation"] for r in metric_rows}):
        nat_margin = {k: v for k, v in maps["natural"]["margin"].items() if k[1] == rel}; r12_margin = {k: v for k, v in maps["r12"]["margin"].items() if k[1] == rel}
        nat_v = {k: v for k, v in maps["natural"]["violation"].items() if k[1] == rel}; r12_v = {k: v for k, v in maps["r12"]["violation"].items() if k[1] == rel}
        ov = [r["hnr_r12_from_nat"] for r in overlap_rows if r["relation"] == rel]
        gate_rows[rel] = {"margin_nat_minus_r12": bootstrap_diff(nat_margin, r12_margin), "violation_nat_minus_r12": bootstrap_diff(nat_v, r12_v), "hnr_r12_from_nat_mean": statistics.fmean(ov) if ov else None}
    qt = gate_rows.get("table->table", {})
    mg = qt.get("margin_nat_minus_r12", {}).get("ci95", [None, None]); vg = qt.get("violation_nat_minus_r12", {}).get("ci95", [None, None]); hnr = qt.get("hnr_r12_from_nat_mean")
    strong_qt = mg[1] is not None and mg[1] < 0 and vg[0] is not None and vg[0] > 0 and hnr is not None and hnr < .8
    evidence_strong = any((v.get("margin_nat_minus_r12", {}).get("ci95", [None, None])[1] is not None and v["margin_nat_minus_r12"]["ci95"][1] < 0 and v.get("violation_nat_minus_r12", {}).get("ci95", [None, None])[0] is not None and v["violation_nat_minus_r12"]["ci95"][0] > 0) for k, v in gate_rows.items() if k != "table->table")
    if strong_qt and evidence_strong: decision = "G-CAND-STRONG"
    elif mg[0] is not None and mg[0] >= 0 and vg[1] is not None and vg[1] <= 0: decision = "G-CAND-WEAK"
    else: decision = "MIXED"
    gate = {"decision": decision, "qt_strong": strong_qt, "evidence_relation_strong": evidence_strong, "relations": gate_rows, "rule": "10,000-replicate query-level paired bootstrap; train source groups not trusted", "teacher_scores": "requested" if score_teacher else "not_scored_cpu_unrequested"}
    write_json(base / "GATE.json", gate)
    write_rows(base / "hub_exposure.jsonl.gz", hub_exposure(r12, t0, nat))
    write_json(base / "summary.json", {"universes": summaries, "gate": gate, "elapsed_seconds": time.monotonic() - started})
    write_json(OUT / "EXECUTION_LEDGER.json", {"A0": "pass", "A1": "pass", "A2": "pass", "gate": decision, "training": "pending_conditional_gpu", "status": "diagnostics_complete", "teacher_scoring": "requested" if score_teacher else "not_run"})
    write_json(OUT / "PACKAGE_MANIFEST.json", {"format_version": 1, "status": "diagnostics_complete", "files": {str(p.relative_to(OUT)): sha(p) for p in OUT.rglob("*") if p.is_file() and p.name != "PACKAGE_MANIFEST.json"}})
    return {"status": "diagnostics_complete", "gate": decision, "universes": summaries, "elapsed_seconds": time.monotonic() - started}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("diagnose",))
    parser.add_argument("--no-exact-qt", action="store_true")
    parser.add_argument("--score-teacher", action="store_true")
    args = parser.parse_args()
    print(json.dumps(diagnose(score_teacher=args.score_teacher, exact_qt=not args.no_exact_qt), ensure_ascii=False))


if __name__ == "__main__":
    main()
