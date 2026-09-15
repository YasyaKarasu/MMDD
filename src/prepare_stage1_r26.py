"""Freeze R26 inputs and audit the existing C1 tensors before reusing them."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random

import torch

from mmdd_stage1.checkpoints import load_student
from run_stage1_r21 import paths, read_rows, write_rows
from run_stage1_r25 import ARMS, _json, _r25_path_pool, _schedule_examples, c1_schedule_path, out as r25_out, sha256

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "work/stage1_optimization_r26_20260914"


def parameter_sha(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in model.named_parameters():
        if value.requires_grad:
            digest.update(name.encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def stable_sha(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def file_record(path: Path) -> dict:
    return {"path": str(path), "exists": path.is_file(),
            **({"bytes": path.stat().st_size, "sha256": sha256(path)} if path.is_file() else {})}


def model_inventory() -> list[dict]:
    rows = [{"generator_id": "Qwen-Raw", "checkpoint": None, "seed": None, "step": None}]
    def add(name: str, path: Path, seed: int | None, step: int):
        rows.append({"generator_id": name, "checkpoint": str(path), "seed": seed, "step": step})
    add("PCA", r25_out(ROOT) / "training/C1/seed13/checkpoints/step_000000.pt", None, 0)
    add("B13", paths(ROOT)["b13"], None, 178)
    for seed in (13, 29):
        add(f"N-U/seed{seed}", ROOT / f"work/stage1_optimization_r24_20260913/N-U/seed{seed}/checkpoints/step_001318.pt", seed, 1318)
        for step in (0, 330, 659):
            add(f"R25-C1/seed{seed}/step{step}", r25_out(ROOT) / f"training/C1/seed{seed}/checkpoints/step_{step:06d}.pt", seed, step)
        for arm in ARMS:
            for step in ((0, 89, 178) if arm in ("B13-FULL", "SPLIT-SUP") else (178,)):
                add(f"R25-{arm}/seed{seed}/step{step}", r25_out(ROOT) / f"training/C2/{arm}/seed{seed}/checkpoints/step_{step:06d}.pt", seed, step)
    add("pre-B13-C1", ROOT / "work/stage1_optimization_r12_20260908/taskC_training/c_candidates_seed13/checkpoints/step_000356.pt", None, 356)
    return rows


def prepare() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    ps = paths(ROOT)
    graph = _r25_path_pool(ROOT)
    graph_sha = sha256(graph)
    if graph_sha != "8a07fdc39f2caf32c63860a33a1b8ab588f43a9025e1edad89a4c28fae9351d5":
        raise ValueError("R26 graph does not match the preregistered R25 graph")
    graph_rows = list(read_rows(graph))
    order = list(range(len(graph_rows)))
    random.Random(13).shuffle(order)
    ordered = [{"position": i, "batch": i // 64, "source_row": j,
                "query_id": graph_rows[j]["query_id"], "list_sha256": stable_sha(graph_rows[j])}
               for i, j in enumerate(order)]
    write_rows(OUT / "common/c2_order.jsonl", ordered)
    protocol = {"stage1": {"direct_k": 100, "evidence_k_per_modality": 20, "targets_per_evidence": 20,
                            "evidence_aggregation": "logsumexp", "path_combination": "sum",
                            "retention": "e2_row_coverage", "retention_top_l": 20, "retention_budget": 4,
                            "hnsw_m": 32, "ef_construction": 200, "ef_search": 100,
                            "rrf_k": 60, "self_exclusion": "frozen corpus excludes query objects; verify intersection"},
                "stage2": {"candidate_budget": 18, "teacher_prerank_budget": 18, "recall_ks": [1, 3, 5, 7, 9],
                           "fill_tables": 10, "evidence_per_table": 4, "rows_per_table": 4,
                           "generators": ["Qwen-Raw", "B13"],
                           "conditions": ["Real-crop", "Real-crop+original", "NoE-fill"],
                           "queries": 64, "engineering_records": 32, "max_new_tokens": 256,
                           "length_retry_tokens": 512},
                "training": {"arms": ["O-NATIVE", "O-SUP", "E-GRAPH"], "seeds": [13, 29], "data_order_seed": 13,
                             "batch_size": 64, "graph_sha256": graph_sha, "order_sha256": sha256(OUT / "common/c2_order.jsonl"),
                             "projection_lr": 1e-6, "relation_lr": 1e-5, "weight_decay": .01, "coverage": 1},
                "statistics": {"bootstrap_replicates": 10000, "seed": 260914, "cluster": "source_table_id"},
                "hardware": "local 2x RTX4090; bounded concurrent processes per GPU"}
    protocol_path = OUT / "PROTOCOL.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise ValueError("Existing R26 protocol differs; do not silently replace frozen protocol")
    _json(protocol_path, protocol)
    population = [{key: row[key] for key in ("query_id", "query_kind", "positive_target_ids", "source_table_id") if key in row}
                  for row in read_rows(ps["candidate_pools"])]
    if any(not row["positive_target_ids"] for row in population):
        raise ValueError("Frozen population includes a query without qrels")
    corpus_ids = {str(row["object_id"]) for row in read_rows(ps["corpus"])}
    overlap = sorted(corpus_ids.intersection(row["query_id"] for row in population))
    if overlap:
        raise ValueError("Query objects in target corpus: explicit common self-exclusion required")
    write_rows(OUT / "common/dev_queries.jsonl", population)
    roles = {"graph": graph, "features": ps["features"] / "manifest.jsonl", "corpus": ps["corpus"],
             "qrels_source": ps["candidate_pools"], "edge_registry": ps["train"],
             "actual_pca": ROOT / "work/stage1_pca_dimension_ceiling_20260828/pca_spectrum.pt",
             "historical_pca": ROOT / "work/stage1_optimization_r10_20260907/baselines/pca_entitables_v9_1024.pt",
             "historical_pca_init": ROOT / "work/stage1_optimization_r11_20260908/taskA_protocol/baselines/pca_init.pt",
             "teacher_T0": ROOT / "work/stage1_optimization_r22_20260911/fresh_lineage/T1-B/seed13/checkpoints/step_010536.pt",
             "teacher_native_cache": r25_out(ROOT) / "common/teacher_native_path_cache.jsonl.gz",
             "content_keys": ROOT / "work/stage1_optimization_r10_20260907/taskB_g5/evidence_content_keys.jsonl"}
    inputs = {name: file_record(path) for name, path in roles.items()}
    inventory = model_inventory()
    for row in inventory:
        if row["checkpoint"]:
            row.update(file_record(Path(row["checkpoint"])))
    _json(OUT / "MODEL_INVENTORY.json", inventory)
    _json(OUT / "RESOLVED_INPUTS.json", inputs)
    registry = defaultdict(set)
    for row in read_rows(ps["train"]):
        positive = set(row.get("positive_ids", []))
        if row.get("positive_id"):
            positive.add(row["positive_id"])
        registry[(row["query_id"], row["source_type"], row["destination_type"])].update(positive)
    pca = torch.load(roles["actual_pca"], map_location="cpu", weights_only=True)["projection"][:1024]
    audit = {"graph_sha256": graph_sha, "query_count": len(population), "self_overlap": overlap, "c1": {}}
    for seed in (13, 29):
        job = r25_out(ROOT) / f"training/C1/seed{seed}"
        receipt = json.loads((job / "C1_COMPLETION_RECEIPT.json").read_text())
        start = load_student(job / "checkpoints/step_000000.pt", torch.device("cpu"))
        final = load_student(job / "checkpoints/step_000659.pt", torch.device("cpu"))
        counts = Counter()
        closure_errors = []
        batches = _schedule_examples(c1_schedule_path(ROOT, seed))
        for batch in batches:
            for row in batch:
                key = (row["query_id"], row["source_type"], row["destination_type"])
                counts[f"{key[1]}->{key[2]}"] += 1
                labels = set(row.get("positive_ids", [])) | ({row["positive_id"]} if row.get("positive_id") else set())
                missing = registry[key].intersection(row["candidate_ids"]) - labels
                if missing:
                    closure_errors.append({"key": key, "missing_positive_labels": sorted(missing)})
        checks = {"all_initial_P_equal_actual_pca": all(torch.equal(p.weight, pca) for p in start.projections.values()),
                  "all_initial_R_identity": all(torch.equal(r, torch.eye(r.shape[0])) for r in start.relations.values()),
                  "initial_hash_matches_receipt": parameter_sha(start) == receipt["initial_parameter_sha256"],
                  "final_hash_matches_receipt": parameter_sha(final) == receipt["final_parameter_sha256"],
                  "anchor_unchanged": torch.equal(start.initial_projection_weights, final.initial_projection_weights),
                  "anchor_matches_actual_pca": all(torch.equal(x, pca) for x in start.initial_projection_weights),
                  "full_coverage": sum(counts.values()) == 42143 and len(batches) == 659,
                  "five_relations_match_receipt": dict(counts) == receipt["relation_counts"],
                  "no_known_positives_mislabeled": not closure_errors,
                  "final_checkpoint_sha_matches_receipt": sha256(job / "checkpoints/step_000659.pt") == receipt["checkpoint_sha256"]}
        audit["c1"][str(seed)] = {"checks": checks, "reuse_valid": all(checks.values()), "relation_counts": dict(counts),
                                  "closure_errors": closure_errors, "registry_keys": len(registry)}
    audit["c1_reuse_valid"] = all(row["reuse_valid"] for row in audit["c1"].values())
    _json(OUT / "acceptance/C1_TENSOR_AUDIT.json", audit)
    _json(OUT / "RECIPE_DIFF.json", {"user_overrides": {"hardware": protocol["hardware"], "stage2": protocol["stage2"]},
                                   "C1_reuse_valid": audit["c1_reuse_valid"], "PCA_provenance": "actual pca_spectrum tensor audited; historical basis comparison pending",
                                   "C2_order": "Random(13).shuffle full list indices, same across model seeds",
                                   "retrieval": "actual own ANN/exact with historical D1 retention and genuine E ranks"})
    _json(OUT / "EXECUTION_MATRIX.json", {"execution_status": "in_progress", "scientific_validity": "unassessable",
        "modules": [{"id": name, "execution_status": "pending", "scientific_validity": "unassessable"}
                    for name in ("A0", "E25", "TRAJ", "O-NATIVE", "O-SUP", "E-GRAPH", "F", "R", "S2", "FB-DIAG", "STATISTICS")],
        "conditional_modules": ["C2_EXPANSION", "TEACHER_REFINEMENT", "REDISTILLATION"]})
    return {"c1_reuse_valid": audit["c1_reuse_valid"], "queries": len(population), "models": len(inventory),
            "missing_models": [row["generator_id"] for row in inventory if row.get("exists") is False]}


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    torch.set_num_threads(4)
    print(json.dumps(prepare(), ensure_ascii=False))
