#!/usr/bin/env python
"""Export frozen MMDD-CQET V4.1 retrieval for the generic Stage-2 pipeline."""

from __future__ import annotations

import argparse
import gc
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable

import torch

from mmdd_cqet_v4_1.artifacts import load_pool_bundle, save_pool_bundle
from mmdd_cqet_v4_1.evaluate import evaluate_student_retrieval, evaluate_teacher_matrix
from mmdd_cqet_v4_1.pipeline import (
    _load_native,
    _load_teacher,
    load_runtime,
)
from mmdd_dataset.wdc_runtime import iter_dataset_artifact
from mmdd_stage1.selection import validate_stage2_gate
from mmdd_stage2.data import validate_retrieval_path_budget


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl_gz(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _materialize_train(args: argparse.Namespace, selection: dict[str, Any]) -> Path:
    output_dir = args.output_dir / "v4_1_train_retrieval"
    ranking_path = output_dir / "rankings.TB_CQET.Real.jsonl.gz"
    logits_path = output_dir / "logits.TB_CQET.Real.jsonl.gz"
    pool_manifest = output_dir / "POOL_MANIFEST.json"
    if pool_manifest.is_file() and ranking_path.is_file() and logits_path.is_file():
        return output_dir

    runtime = load_runtime(args.protocol, args.stage1_run)
    pools = None
    student = None
    teacher = None
    try:
        if pool_manifest.is_file():
            pools = load_pool_bundle(output_dir)
        else:
            student = _load_native(Path(selection["SUP_checkpoint"]), runtime, device=args.device)
            pools = evaluate_student_retrieval(
                student,
                runtime.z_store,
                runtime.row_store,
                runtime.labels.query_ids,
                runtime.labels,
                "train",
                hnsw_seed=args.seed,
                generator_id="native_sup_stage2_train",
                index_dir=str(output_dir / "indices"),
            )
            save_pool_bundle(
                output_dir,
                pools,
                runtime.labels,
                seed=args.seed,
                generator="native_sup_stage2_train",
            )
            del student
            student = None
            torch.cuda.empty_cache()

        teacher = _load_teacher(
            args.stage1_run / f"seed{args.seed}" / "TB_CQET" / "checkpoints" / "end.pt",
            device=args.device,
        )
        evaluate_teacher_matrix(
            {"TB_CQET": teacher},
            runtime.bank,
            pools,
            runtime.labels,
            seed=args.seed,
            generator="native_sup_stage2_train",
            split="train",
            output_dir=output_dir,
            device=args.device,
        )
    finally:
        del pools, student, teacher, runtime
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return output_dir


def _build_record(
    ranking: dict[str, Any],
    pool: dict[str, Any],
    logits: dict[str, dict[str, Any]],
    *,
    checkpoint_sha256: str,
    top_k: int,
    evidence_path_k: int,
) -> dict[str, Any]:
    direct_ids = {str(value) for value in pool["D100_ANN"]}
    target_ids = [str(value) for value in ranking["target_ids"][:top_k]]
    scores = [float(value) for value in ranking["scores"][:top_k]]
    if len(target_ids) != len(scores):
        raise ValueError(f"{ranking['query_id']}: ranking target/score length mismatch")

    results = []
    for target_id, score in zip(target_ids, scores, strict=True):
        try:
            item = logits[target_id]
        except KeyError as exc:
            raise KeyError(f"{ranking['query_id']}: no Teacher logits for {target_id}") from exc
        aggregated = float(item["aggregated_score"])
        if not math.isclose(score, aggregated, rel_tol=0.0, abs_tol=1e-6):
            raise ValueError(f"{ranking['query_id']} -> {target_id}: ranking/logit score mismatch")

        paths: list[dict[str, Any]] = []
        direct_score = None
        if target_id in direct_ids:
            direct_score = float(item["f0"])
            paths.append({"kind": "direct", "path_score": direct_score})

        seen_evidence: set[str] = set()
        evidence_paths = []
        for source in item.get("paths", []):
            evidence_id = str(source["evidence_id"])
            if evidence_id in seen_evidence:
                continue
            seen_evidence.add(evidence_id)
            evidence_paths.append(
                {
                    "kind": "evidence",
                    "evidence_id": evidence_id,
                    "modality": str(source["modality"]),
                    "path_score": float(source["raw_QET"]),
                    "v4_1_slot": int(source["slot"]),
                }
            )
            if len(evidence_paths) == evidence_path_k:
                break
        paths.extend(evidence_paths)
        if not paths:
            raise ValueError(
                f"{ranking['query_id']} -> {target_id}: V4.1 Top-{top_k} target has no retained path"
            )
        result = {
            "target_id": target_id,
            "score": aggregated,
            "direct_score": direct_score,
            "evidence_score": aggregated if evidence_paths else None,
            "stage2_table_score": aggregated,
            "paths": paths,
        }
        results.append(result)

    return {
        "query_id": str(ranking["query_id"]),
        "split": str(ranking["split"]),
        "student_checkpoint_sha256": checkpoint_sha256,
        "v4_1": {
            "schema_version": str(ranking["schema_version"]),
            "generator": str(ranking["generator"]),
            "pool_id": str(ranking["pool_id"]),
            "teacher": str(ranking["teacher"]),
            "teacher_state_hash": str(ranking["teacher_state_hash"]),
            "view": str(ranking["view"]),
        },
        "path_aggregation": {
            "path_result_k": top_k,
            "evidence_path_k": evidence_path_k,
            "table_score": "TB_CQET_end.Real.nonempty_path_logmeanexp_else_f0",
        },
        "results": results,
    }


def _export_split(
    source_dir: Path,
    output_path: Path,
    *,
    checkpoint_sha256: str,
    top_k: int,
    evidence_path_k: int,
) -> list[dict[str, Any]]:
    pools = {str(row["query_id"]): row for row in _read_jsonl_gz(source_dir / "pools.jsonl.gz")}
    rankings = _read_jsonl_gz(source_dir / "rankings.TB_CQET.Real.jsonl.gz")
    logits_by_query: dict[str, dict[str, dict[str, Any]]] = {}
    for row in _read_jsonl_gz(source_dir / "logits.TB_CQET.Real.jsonl.gz"):
        logits_by_query.setdefault(str(row["query_id"]), {})[str(row["target_id"])] = row

    records = []
    for ranking in rankings:
        query_id = str(ranking["query_id"])
        record = _build_record(
            ranking,
            pools[query_id],
            logits_by_query[query_id],
            checkpoint_sha256=checkpoint_sha256,
            top_k=top_k,
            evidence_path_k=evidence_path_k,
        )
        validate_retrieval_path_budget(
            record,
            max_targets=top_k,
            top_k_evidence=evidence_path_k,
        )
        records.append(record)
    records.sort(key=lambda row: str(row["query_id"]).encode("utf-8"))
    _write_jsonl(output_path, records)
    return records


def _dev_evidence_coverage(dataset_root: Path, records: list[dict[str, Any]]) -> dict[str, Any]:
    positives = {
        (str(row["query_table_id"]), str(row["target_table_id"]))
        for row in iter_dataset_artifact(dataset_root, "qrels")
        if row.get("split", "train") == "dev"
        and row.get("reason") == "model_recoverable_join_column"
    }
    by_query = {str(row["query_id"]): row for row in records}
    covered = 0
    for query_id, target_id in positives:
        record = by_query.get(query_id)
        if record is None:
            continue
        for result in record["results"][:10]:
            if str(result["target_id"]) == target_id and any(
                path["kind"] == "evidence" for path in result["paths"]
            ):
                covered += 1
                break
    return {
        "positive_pairs": len(positives),
        "covered_pairs": covered,
        "positive_evidence_path_coverage@10": covered / len(positives) if positives else 0.0,
    }


def run(args: argparse.Namespace) -> None:
    selection_path = args.stage1_run / f"seed{args.seed}" / "selections" / "NATIVE_C2_COMMON.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    checkpoint = Path(selection["SUP_checkpoint"])
    checkpoint_sha256 = _sha256(checkpoint)
    if checkpoint_sha256 != selection["SUP_checkpoint_sha256"]:
        raise ValueError("Selected V4.1 NATIVE_C2_SUP checkpoint hash mismatch")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_dir = _materialize_train(args, selection)
    sources = {
        "train": train_dir,
        "dev": args.stage1_run / f"seed{args.seed}" / "eval" / "dev" / "native_sup",
        "test": args.stage1_run / f"seed{args.seed}" / "eval" / "test" / "native_sup",
    }
    exports = {}
    records_by_split = {}
    for split, source in sources.items():
        destination = args.output_dir / f"retrieval.{split}.jsonl"
        records = _export_split(
            source,
            destination,
            checkpoint_sha256=checkpoint_sha256,
            top_k=args.top_k,
            evidence_path_k=args.evidence_path_k,
        )
        records_by_split[split] = records
        exports[split] = {
            "path": str(destination.resolve()),
            "sha256": _sha256(destination),
            "records": len(records),
            "source_dir": str(source.resolve()),
        }

    dev_coverage = _dev_evidence_coverage(args.dataset_root, records_by_split["dev"])
    gate = {
        "format_version": 1,
        "completed_stage": "student-path",
        "selection_split": "dev",
        "stage2_allowed": True,
        "best_checkpoint": str(checkpoint.resolve()),
        "best_checkpoint_sha256": checkpoint_sha256,
        "best_metrics": dev_coverage,
        "v4_1_selection": {
            "path": str(selection_path.resolve()),
            "sha256": _sha256(selection_path),
            "selection_owner": selection["selection_owner"],
            "selected_fraction": selection["selected_fraction"],
            "state_sha256": selection["SUP_state_sha256"],
        },
        "performance_gate_note": (
            "This manifest authenticates the frozen V4.1 handoff requested for Stage-2. "
            "It does not change the Stage-1 performance-gate decision."
        ),
        "retrievals": exports,
    }
    gate_path = args.output_dir / "stage1_gate.json"
    _write_json(gate_path, gate)
    validate_stage2_gate(
        gate_path,
        [Path(exports[split]["path"]) for split in ("train", "dev", "test")],
    )
    _write_json(
        args.output_dir / "EXPORT_MANIFEST.json",
        {
            "status": "COMPLETE",
            "stage1_run": str(args.stage1_run.resolve()),
            "dataset_root": str(args.dataset_root.resolve()),
            "gate": {"path": str(gate_path.resolve()), "sha256": _sha256(gate_path)},
            "retrievals": exports,
            "dev_evidence_coverage": dev_coverage,
        },
    )
    print(json.dumps({"gate": str(gate_path), "retrievals": exports}, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--stage1-run", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--evidence-path-k", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.top_k <= 0 or args.evidence_path_k <= 0:
        parser.error("--top-k and --evidence-path-k must be positive")
    return args


if __name__ == "__main__":
    run(parse_args())
