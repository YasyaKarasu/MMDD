#!/usr/bin/env python
"""Complete R15 exact/ANN diagnostics for the two non-residual references."""

from __future__ import annotations

import argparse
import gzip
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from audit_stage1_r15_g import _quantiles, checkpoint_fingerprint_from_ids, full_dev_direct_exact_ann
from audit_stage1_r15_training_scores import endpoints
from mmdd_stage1.artifacts import checkpoint_fingerprint, write_json
from mmdd_stage1.checkpoints import load_student
from mmdd_stage1.data import load_edge_examples, load_target_examples
from mmdd_stage1.features import FeatureStore
from mmdd_stage1.pca import load_pca_projection
from mmdd_stage1.retrieval import StudentANNIndices, load_corpus_ids
from reevaluate_stage1_r12_checkpoints import _requests, exact_relation
from run_stage1_r13 import _paths as r13_paths


@torch.inference_mode()
def base_geometry(model: Any, s0: Any, store: FeatureStore,
                  ids_by_type: dict[str, list[str]], device: torch.device) -> dict[str, Any]:
    """Use the same first 1,024 corpus objects per modality as residual audits."""

    result = {}
    for kind, object_ids in ids_by_type.items():
        selected = object_ids[:1024]
        z = torch.stack([store.embedding_features(value).embedding for value in selected]).to(device)
        full = model.project(z, kind)
        reference = s0.project(z, kind)
        full_mean = full.mean(dim=0)
        reference_mean = reference.mean(dim=0)
        singular = torch.linalg.svdvals(full - full.mean(dim=0, keepdim=True))
        energy = singular.square()
        probabilities = energy / energy.sum().clamp_min(1e-30)
        rms = float(full.square().mean().sqrt())
        result[kind] = {
            "samples": len(selected), "sample_ids_sha256": checkpoint_fingerprint_from_ids(selected),
            "residual_enabled": False, "projection_scale_c_tau": None,
            "raw_Az": None, "scaled_Az": None, "gelu_scaled_Az": None,
            "base_output": {
                "rms": rms,
                "norm_quantiles": _quantiles(full.norm(dim=1)),
                "mean_vector": full_mean.cpu().tolist(),
                "mean_vector_norm": float(full_mean.norm()),
            },
            "residual_output": {"rms": 0.0}, "residual_to_base_rms": 0.0,
            "full_output": {
                "rms": rms, "norm_quantiles": _quantiles(full.norm(dim=1)),
                "mean_vector": full_mean.cpu().tolist(),
                "mean_vector_norm": float(full_mean.norm()),
                "random_pair_cosine": _quantiles(F.cosine_similarity(full[0::2], full[1::2], dim=1)),
                "effective_rank": float(torch.exp(-(probabilities * probabilities.clamp_min(1e-30).log()).sum())),
                "stable_rank": float(energy.sum() / energy.max()),
                "top_singular_values": singular[:20].cpu().tolist(),
            },
            "s0_output": {
                "mean_vector": reference_mean.cpu().tolist(),
                "mean_vector_norm": float(reference_mean.norm()),
            },
            "full_vs_s0_direction_cosine": _quantiles(F.cosine_similarity(full, reference, dim=1)),
            "parameters": {
                "P_update_norm_from_step0": float((model.projections[kind].weight - s0.projections[kind].weight).norm()),
                "A_weight_norm": None, "A_update_norm_from_step0": None,
                "B_weight_norm": None, "B_update_norm_from_step0": None,
            },
        }
    return {"sample_size_per_type": 1024, "by_type": result}


def b13_direct_from_provenance(path: Path, target_count: int) -> dict[str, Any]:
    """Reuse C's full-lake exact results without another B13 model evaluation."""

    rows = []
    exact_hubs, ann_hubs = Counter(), Counter()
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            original = json.loads(line)
            exact_ids, ann_ids = original["D100_exact"], original["D100_ANN"]
            positives = set(original["positive_target_ids"])
            exact_hubs.update(exact_ids[:1])
            ann_hubs.update(ann_ids[:1])
            row = {
                "query_id": original["query_id"], "query_kind": original["query_kind"],
                "source_table_id": original["source_table_id"], "positive_denominator": len(positives),
                "exact_ids": exact_ids, "ann_ids": ann_ids,
                "ann_exact_overlap@100": len(set(exact_ids) & set(ann_ids)) / 100,
                "positive_targets": [{
                    "target_id": value["target_id"], "present": True,
                    "score": value["exact_direct_score"], "exact_rank": value["exact_direct_rank"],
                    "exact_rank_min": value["exact_direct_rank_min"], "exact_rank_max": value["exact_direct_rank_max"],
                    "ann_rank": value["natural_direct_rank"],
                } for value in original["positive_targets"]],
            }
            for name, ids in (("exact", exact_ids), ("ann", ann_ids)):
                for k in (10, 20, 50, 100):
                    row[f"{name}_recall@{k}"] = len(positives & set(ids[:k])) / len(positives)
            rows.append(row)
    aggregate = {}
    for kind in ("all", "implicit", "explicit"):
        selected = [row for row in rows if kind == "all" or row["query_kind"] == kind]
        aggregate[kind] = {f"{name}_recall@{k}": float(np.mean([row[f"{name}_recall@{k}"] for row in selected]))
                           for name in ("exact", "ann") for k in (10, 20, 50, 100)}
        aggregate[kind]["ann_exact_overlap@100"] = float(np.mean([row["ann_exact_overlap@100"] for row in selected]))
    return {
        "queries": len(rows), "legal_targets": target_count, "aggregate": aggregate, "per_query": rows,
        "exact_top1_hubs": exact_hubs.most_common(20), "ann_top1_hubs": ann_hubs.most_common(20),
        "exact_distinct_top1": len(exact_hubs), "ann_distinct_top1": len(ann_hubs),
        "reused_source": str(path.resolve()), "reused_source_sha256": checkpoint_fingerprint(path),
    }


@torch.inference_mode()
def run(args: argparse.Namespace) -> None:
    torch.set_num_threads(args.cpu_threads)
    device = torch.device(args.device)
    paths = r13_paths(args.root)
    targets = load_target_examples(paths["dev_targets"], split="dev")
    edges = load_edge_examples(paths["dev_edges"], split="dev")
    supervision = json.loads((paths["r12"] / "taskA_correctness/supervision/manifest.json").read_text())
    requests = _requests(targets, edges, Path(supervision["dataset_root"]))
    historical_path = args.root / "work/stage1_optimization_r13_20260909/statistics/mechanism_and_reproducibility_audit.json"
    historical = json.loads(historical_path.read_text())["mechanism_panel"]["exact_and_ann_by_arm"]["p_s_target_only"]
    for name, source_ids, *_rest in requests:
        if [row["source_id"] for row in historical[name]["per_source"]] != source_ids:
            raise ValueError(f"Historical B13 panel {name} uses different source identities")
    store = FeatureStore.from_path(paths["features"], cache_size=260000)
    ids_by_type = load_corpus_ids(paths["corpus"], store)
    store.preload_embeddings([*(value for ids in ids_by_type.values() for value in ids), *(row.query_id for row in targets)])
    basis = load_pca_projection(args.root / "work/stage1_optimization_r10_20260907/baselines/pca_entitables_v9_1024.pt",
                                input_dim=store.embedding_dimension(), student_dim=1024).to(device)
    s0 = load_student(paths["s0"], device).eval()
    output = args.root / "work/stage1_optimization_r15_20260909/stageG_correctness/references"
    output.mkdir(parents=True, exist_ok=True)
    for arm in ("s_full", "s_eoff"):
        started = time.monotonic()
        checkpoint_path = endpoints(args.root)[arm]
        checkpoint_hash = checkpoint_fingerprint(checkpoint_path)
        model = load_student(checkpoint_path, device).eval()
        index_dir = checkpoint_path.parent.parent / "evaluation_step178/index"
        index_manifest = json.loads((index_dir / "manifest.json").read_text())
        if index_manifest["student_checkpoint_sha256"] != checkpoint_hash:
            raise ValueError(f"{arm} index checkpoint mismatch")
        if index_manifest["corpus_sha256"] != checkpoint_fingerprint(paths["corpus"]):
            raise ValueError(f"{arm} index corpus mismatch")
        if arm == "s_full":
            panels = historical
            direct = b13_direct_from_provenance(
                args.root / "work/stage1_optimization_r15_20260909/stageC_candidate_delivery/candidate_provenance.jsonl.gz",
                len(ids_by_type["table"]),
            )
        else:
            indices = StudentANNIndices(model, store, index_dir, device=device,
                                        checkpoint_sha256=checkpoint_hash,
                                        corpus_sha256=index_manifest["corpus_sha256"], score_space="raw_logit")
            panels = {request[0]: exact_relation(model, indices, store, ids_by_type, request, basis, device) for request in requests}
            direct = full_dev_direct_exact_ann(model, indices, store, targets, sorted(ids_by_type["table"]), device, batch_size=16)
            del indices
        for panel in panels.values():
            for row in panel["per_source"]:
                for edge in row["gt_edges"]:
                    edge["margin_over_exact_cutoff"] = edge["score"] - row["cutoff_score"]
            for name in ("exact", "ann"):
                hubs = Counter(value for row in panel["per_source"] for value in row[f"{name}_ids"])
                panel[f"{name}_hub_top20"] = hubs.most_common(20)
                panel[f"{name}_distinct_returned"] = len(hubs)
        payload = {
            "format_version": 1, "status": "complete", "arm": arm, "checkpoint_id": "step_000178",
            "checkpoint": str(checkpoint_path.resolve()), "checkpoint_sha256": checkpoint_hash,
            "index": {"reused": True, "manifest": index_manifest, "path": str(index_dir.resolve()), "rebuilt": False},
            "fixed_sources": {request[0]: request[1] for request in requests},
            "exact_relation_panels": panels, "full_dev_direct": direct,
            "projection_and_parameter_diagnostics": base_geometry(model, s0, store, ids_by_type, device),
            "reference_panel_provenance": ({"reused": True, "path": str(historical_path.resolve()),
                                              "sha256": checkpoint_fingerprint(historical_path), "source_id_order_matches_g": True}
                                             if arm == "s_full" else {"reused": False, "source_id_order_matches_g": True}),
            "cost": {"seconds": time.monotonic() - started, "device": args.device,
                     "cpu_threads": args.cpu_threads, "optimizer_updates": 0, "new_teacher_inference": 0},
            "code_sha256": checkpoint_fingerprint(Path(__file__)),
        }
        write_json(output / f"{arm}.json", payload)
        print(json.dumps({"arm": arm, "exact_recall@10": direct["aggregate"]["all"]["exact_recall@10"],
                          "ann_recall@10": direct["aggregate"]["all"]["ann_recall@10"], "status": "complete"}), flush=True)
        del model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cpu-threads", type=int, default=2)
    run(parser.parse_args())
