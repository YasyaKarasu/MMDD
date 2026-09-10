#!/usr/bin/env python
"""Complete trained-checkpoint cache, save/load, and projection identity audits."""

from __future__ import annotations

import argparse
import gc
import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch

from audit_stage1_r15_g import (
    FP32_ATOL,
    FP32_RTOL,
    adapter_off_model,
    merged_linear_model,
    projection_diagnostics,
)
from mmdd_stage1.artifacts import checkpoint_fingerprint, write_json
from mmdd_stage1.checkpoints import load_checkpoint, load_path_aggregator, load_student
from mmdd_stage1.data import load_target_examples
from mmdd_stage1.features import FeatureStore
from mmdd_stage1.retrieval import StudentANNIndices, load_corpus_ids
from mmdd_stage1.training import checkpoint
from run_stage1_r13 import _paths
from run_stage1_r15 import ARM_SPECS, arm_directory, freeze_plan, output_root


def close_summary(left: torch.Tensor, right: torch.Tensor) -> dict[str, Any]:
    """Report magnitude and fixed FP32 tolerance without tuning from retrieval."""

    return {
        "max_abs_difference": float((left - right).abs().max()),
        "max_abs_reference": float(left.abs().max()),
        "passed": torch.allclose(left, right, atol=FP32_ATOL, rtol=FP32_RTOL),
    }


def fixed_pairs(examples: list[Any], store: FeatureStore) -> dict[str, list[tuple[str, str]]]:
    """Choose deterministic train/dev pairs, including both sides of every relation."""

    pairs: dict[str, list[tuple[str, str]]] = {
        name: [] for name in (
            "table_to_table", "table_to_text", "table_to_image",
            "text_to_table", "image_to_table",
        )
    }
    for example in sorted(examples, key=lambda row: row.query_id):
        for candidate in example.candidates:
            pairs["table_to_table"].append((example.query_id, candidate.target_id))
            for evidence_id in candidate.evidence_ids:
                kind = store.embedding_features(evidence_id).object_type
                pairs[f"table_to_{kind}"].append((example.query_id, evidence_id))
                pairs[f"{kind}_to_table"].append((evidence_id, candidate.target_id))
        if all(len(set(value)) >= 32 for value in pairs.values()):
            break
    return {name: list(dict.fromkeys(values))[:32] for name, values in pairs.items()}


@torch.inference_mode()
def audit_pairs(
    model: Any,
    reloaded: Any,
    indices: StudentANNIndices,
    store: FeatureStore,
    pairs: dict[str, list[tuple[str, str]]],
    device: torch.device,
) -> dict[str, Any]:
    result = {}
    positions = {
        kind: {object_id: i for i, object_id in enumerate(ids)}
        for kind, ids in indices.object_ids.items()
    }
    for name, requested_pairs in pairs.items():
        source_type, destination_type = name.split("_to_")
        selected = [pair for pair in requested_pairs if pair[1] in positions[destination_type]]
        if not selected:
            raise ValueError(f"No legal corpus destinations in {name}")
        sources = [left for left, _ in selected]
        destinations = [right for _, right in selected]
        source = torch.stack([store.embedding_features(value).embedding for value in sources]).to(device)
        destination = torch.stack([store.embedding_features(value).embedding for value in destinations]).to(device)
        forward = model.score_embeddings(source, source_type, destination, destination_type)
        query = model.relation_query(source, source_type, destination_type)
        fresh = model.index_vector(destination, destination_type)
        cached = torch.from_numpy(np.asarray(indices.indices[destination_type].get_items(
            [positions[destination_type][value] for value in destinations]
        ))).to(device)
        fresh_ip = (query * fresh).sum(-1)
        cache_ip = (query * cached).sum(-1)
        after_reload = reloaded.score_embeddings(source, source_type, destination, destination_type)
        ann = indices.search_many(sources, destination_type, 5)
        returned = []
        for source_id, row, qvector in zip(sources, ann, query):
            ids = [value for value, _ in row]
            vectors = torch.from_numpy(np.asarray(indices.indices[destination_type].get_items(
                [positions[destination_type][value] for value in ids]
            ))).to(device)
            cached_scores = qvector @ vectors.T
            embedding = store.embedding_features(source_id).embedding.to(device).unsqueeze(0)
            destination_embeddings = torch.stack([
                store.embedding_features(value).embedding for value in ids
            ]).to(device)
            exact = model.score_embedding_matrix(
                embedding, source_type, destination_embeddings, destination_type
            )[0]
            saved_scores = torch.tensor([score for _, score in row], device=device)
            returned.append({
                "source_id": source_id, "returned_ids": ids,
                "cached_ip_vs_exact": close_summary(exact, cached_scores),
                "ann_score_vs_exact": close_summary(exact, saved_scores),
            })
        checks = {
            "forward_vs_fresh_vector_ip": close_summary(forward, fresh_ip),
            "forward_vs_cached_vector_ip": close_summary(forward, cache_ip),
            "fresh_vs_cached_destination_vector": close_summary(fresh, cached),
            "forward_vs_actual_save_load": close_summary(forward, after_reload),
        }
        result[name] = {
            "pairs": selected,
            "excluded_noncorpus_pairs": [pair for pair in requested_pairs if pair not in selected],
            "forward_scores": forward.cpu().tolist(),
            "cached_ip_scores": cache_ip.cpu().tolist(),
            "checks": checks,
            "ann_returned_checks": returned,
            "passed": all(value["passed"] for value in checks.values())
            and all(row[key]["passed"] for row in returned for key in (
                "cached_ip_vs_exact", "ann_score_vs_exact"
            )),
        }
    return result


def run(args: argparse.Namespace) -> None:
    torch.set_num_threads(2)
    plan = freeze_plan(args.root)
    paths = _paths(args.root)
    output = output_root(args.root) / "stageG_correctness/deployment_supplement"
    output.mkdir(parents=True, exist_ok=True)
    store = FeatureStore.from_path(paths["features"], cache_size=8000)
    panels = {
        split: fixed_pairs(load_target_examples(paths[path_key], split=None), store)
        for split, path_key in (("train_fit", "train_targets"), ("dev", "dev_targets"))
    }
    write_json(output / "fixed_panels.json", panels)
    device = torch.device(args.device)
    summary = {}
    scale_checks = {}
    for arm in ARM_SPECS:
        expected_scales = load_checkpoint(Path(plan["r14_checkpoints"][arm]["0"]["path"]))[
            "config"
        ]["projection_scales"]
        for family in ("full", "eoff"):
            for step in (0, 45, 89, 178):
                path = (
                    Path(plan["r14_checkpoints"][arm][str(step)]["path"])
                    if family == "full"
                    else arm_directory(args.root, arm) / "checkpoints" / f"step_{step:06d}.pt"
                )
                payload = load_checkpoint(path)
                scale_checks[f"{arm}_{family}_{step}"] = {
                    "checkpoint": str(path), "sha256": checkpoint_fingerprint(path),
                    "saved_scales": payload["config"]["projection_scales"],
                    "equal_to_step0": payload["config"]["projection_scales"] == expected_scales,
                }
            name = f"{arm}_{family}"
            metrics_path = output / f"{name}.json"
            if metrics_path.is_file():
                summary[name] = json.loads(metrics_path.read_text(encoding="utf-8"))
                continue
            trained = path
            model = load_student(trained, device).eval()
            index_dir = (
                trained.parent.parent / "evaluation_step178/index"
            )
            indices = StudentANNIndices(
                model, store, index_dir, device=device,
                checkpoint_sha256=checkpoint_fingerprint(trained),
                corpus_sha256=checkpoint_fingerprint(paths["corpus"]),
                score_space="raw_logit",
            )
            with tempfile.TemporaryDirectory(prefix="r15-roundtrip-") as temporary:
                saved = Path(temporary) / "checkpoint.pt"
                torch.save(checkpoint(model, "student-path", load_path_aggregator(trained)), saved)
                reloaded = load_student(saved, device).eval()
                panel_results = {
                    split: audit_pairs(model, reloaded, indices, store, pairs, device)
                    for split, pairs in panels.items()
                }
                state_equal = all(torch.equal(value, reloaded.state_dict()[key])
                                  for key, value in model.state_dict().items())
                scales_equal = model.projection_scales == reloaded.projection_scales == expected_scales
            merge = None
            if model.projection_adapter == "linear":
                merged = merged_linear_model(model)
                merge = {}
                for kind in ("table", "text", "image"):
                    selected = indices.object_ids[kind][:1024]
                    embeddings = torch.stack([
                        store.embedding_features(value).embedding for value in selected
                    ]).to(device)
                    merge[kind] = close_summary(model.project(embeddings, kind), merged.project(embeddings, kind))
                    merge[kind]["object_ids"] = selected
                del merged
            result = {
                "status": "complete", "arm": name,
                "checkpoint": str(trained), "sha256": checkpoint_fingerprint(trained),
                "fixed_panel_sha256": checkpoint_fingerprint(output / "fixed_panels.json"),
                "tolerance": {"dtype": "float32", "atol": FP32_ATOL, "rtol": FP32_RTOL},
                "actual_save_load_state_identical": state_equal,
                "scales_identical_to_saved_and_step0": scales_equal,
                "panels": panel_results, "linear_merge_projection_vectors": merge,
                "passed": state_equal and scales_equal
                and all(row["passed"] for panel in panel_results.values() for row in panel.values())
                and (merge is None or all(row["passed"] for row in merge.values())),
                "code_sha256": checkpoint_fingerprint(Path(__file__)),
            }
            if family == "full":
                # The first G diagnostic computed latent residuals even when disabled.
                # Export the actual adapter-off F=P output as an explicit correction.
                ids_by_type = load_corpus_ids(paths["corpus"], store)
                step0 = load_student(
                    Path(plan["r14_checkpoints"][arm]["0"]["path"]), device
                ).eval()
                s0 = load_student(paths["s0"], device).eval()
                result["adapter_off_geometry_correction"] = projection_diagnostics(
                    adapter_off_model(model), step0, s0, store, ids_by_type, device
                )
                result["adapter_off_geometry_correction_reason"] = (
                    "Supersedes initial G adapter-off geometry fields only; full_output "
                    "now uses disabled residual=0. Original exact/ANN scores were correct."
                )
                del step0, s0
            write_json(metrics_path, result)
            summary[name] = result
            print(json.dumps({"arm": name, "passed": result["passed"]}), flush=True)
            del model, reloaded, indices
            gc.collect()
            torch.cuda.empty_cache()
    write_json(output / "SUMMARY.json", {
        "status": "complete",
        "passed": all(row["passed"] for row in summary.values())
        and all(row["equal_to_step0"] for row in scale_checks.values()),
        "arms": {name: {"passed": row["passed"], "path": str(output / f"{name}.json")}
                 for name, row in summary.items()},
        "all_checkpoint_scale_checks": scale_checks,
        "timing_note": "Supplement completed after I training; original G gate retained unchanged.",
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    run(parser.parse_args())
