"""Hand the frozen Stage-1 selection to Stage 2.

Writes one ``retrieval.<split>.jsonl`` per split in the format consumed by
``train_stage2.py`` / ``run_stage2.py`` plus a ``stage1_gate.json`` that binds every
record to the selected Native C2 Student checkpoint. Target order and table
scores are the frozen ``TB_CQET`` end-point reranking of that Student's C150 pool.
"""

from __future__ import annotations

import gc
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable

from mmdd_dataset.wdc_runtime import iter_dataset_artifact
from mmdd_stage2.data import validate_retrieval_path_budget


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_stage1_selection(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("format_version") != 1:
        raise ValueError(f"{path}: unsupported Stage-1 selection manifest")
    return payload


def validate_stage2_gate(
    selection_path: Path,
    retrieval_paths: list[Path] | None = None,
) -> dict[str, Any]:
    """Check that Stage-2 inputs come from the dev-gated Stage-1 checkpoint."""
    selection = load_stage1_selection(selection_path)
    if selection.get("completed_stage") != "student-path":
        raise ValueError("Stage 2 requires a dev-gated student-path checkpoint")
    if selection.get("selection_split") != "dev":
        raise ValueError("Stage 2 requires checkpoint selection on the dev split")
    if not selection.get("stage2_allowed"):
        coverage = selection.get("best_metrics", {}).get(
            "positive_evidence_path_coverage@10"
        )
        raise ValueError(
            "Stage 2 is blocked: the best Student checkpoint has insufficient "
            f"positive evidence-path dev coverage ({coverage})"
        )
    checkpoint_path = Path(str(selection["best_checkpoint"]))
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Stage-1 best checkpoint is missing: {checkpoint_path}"
        )
    expected_sha256 = str(selection["best_checkpoint_sha256"])
    if _sha256(checkpoint_path) != expected_sha256:
        raise ValueError("Stage-1 best checkpoint fingerprint no longer matches its gate")
    for retrieval_path in retrieval_paths or []:
        text = retrieval_path.read_text(encoding="utf-8")
        try:
            payload = json.loads(text)
            records = payload if isinstance(payload, list) else [payload]
        except json.JSONDecodeError:
            records = [json.loads(line) for line in text.splitlines() if line.strip()]
        for record_number, record in enumerate(records, 1):
            if record.get("student_checkpoint_sha256") != expected_sha256:
                raise ValueError(
                    f"{retrieval_path}:record {record_number}: retrieval was not produced "
                    "by the dev-gated best Student checkpoint"
                )
    return selection


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


def _materialize_train(
    protocol_path: Path,
    run_root: Path,
    output_dir: Path,
    checkpoint: Path,
    *,
    seed: int,
    device: str,
) -> Path:
    """Retrieve and rerank the train queries once; dev/test reuse the frozen evaluation."""
    import torch

    from .artifacts import load_pool_bundle, save_pool_bundle
    from .evaluate import evaluate_student_retrieval, evaluate_teacher_matrix
    from .pipeline import _load_native, _load_teacher, load_runtime

    ranking_path = output_dir / "rankings.TB_CQET.Real.jsonl.gz"
    logits_path = output_dir / "logits.TB_CQET.Real.jsonl.gz"
    pool_manifest = output_dir / "POOL_MANIFEST.json"
    if pool_manifest.is_file() and ranking_path.is_file() and logits_path.is_file():
        return output_dir

    runtime = load_runtime(protocol_path, run_root)
    generator = f"{output_dir.name}"
    pools = None
    student = None
    teacher = None
    try:
        if pool_manifest.is_file():
            pools = load_pool_bundle(output_dir)
        else:
            student = _load_native(checkpoint, runtime, device=device)
            pools = evaluate_student_retrieval(
                student,
                runtime.z_store,
                runtime.row_store,
                runtime.labels.query_ids,
                runtime.labels,
                "train",
                hnsw_seed=seed,
                generator_id=generator,
                index_dir=str(output_dir / "indices"),
            )
            save_pool_bundle(output_dir, pools, runtime.labels, seed=seed, generator=generator)
            del student
            student = None
            torch.cuda.empty_cache()

        teacher = _load_teacher(
            run_root / f"seed{seed}" / "TB_CQET" / "checkpoints" / "end.pt",
            device=device,
        )
        evaluate_teacher_matrix(
            {"TB_CQET": teacher},
            runtime.bank,
            pools,
            runtime.labels,
            seed=seed,
            generator=generator,
            split="train",
            output_dir=output_dir,
            device=device,
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
                    "slot": int(source["slot"]),
                }
            )
            if len(evidence_paths) == evidence_path_k:
                break
        paths.extend(evidence_paths)
        if not paths:
            raise ValueError(
                f"{ranking['query_id']} -> {target_id}: Top-{top_k} target has no retained path"
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
        "stage1": {
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


def export_stage2(
    protocol_path: Path,
    run_root: Path,
    output_dir: Path,
    *,
    seed: int,
    arm: str = "KD",
    top_k: int = 50,
    evidence_path_k: int = 4,
    device: str = "cuda:0",
) -> dict[str, Any]:
    """Export train/dev/test retrieval of the dev-selected ``NATIVE_C2_<arm>`` Student."""
    if arm not in ("SUP", "KD"):
        raise ValueError(f"unknown Native C2 arm: {arm}")
    if top_k <= 0 or evidence_path_k <= 0:
        raise ValueError("top_k and evidence_path_k must be positive")
    protocol = json.loads(Path(protocol_path).read_text(encoding="utf-8"))
    dataset_root = Path(protocol["paths"]["dataset_root"])
    if not dataset_root.is_absolute():
        dataset_root = Path(__file__).resolve().parents[2] / dataset_root
    seed_dir = run_root / f"seed{seed}"
    if not (run_root / "GLOBAL_SELECTION_FREEZE.json").is_file():
        raise RuntimeError("Stage-2 export requires a completed, frozen Stage-1 run")
    selection_path = seed_dir / "selections" / "NATIVE_C2_COMMON.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    checkpoint = Path(selection[f"{arm}_checkpoint"])
    checkpoint_sha256 = _sha256(checkpoint)
    if checkpoint_sha256 != selection[f"{arm}_checkpoint_sha256"]:
        raise ValueError(f"Selected NATIVE_C2_{arm} checkpoint hash mismatch")

    output_dir.mkdir(parents=True, exist_ok=True)
    generator = f"native_{arm.lower()}"
    train_dir = _materialize_train(
        protocol_path, run_root, output_dir / f"{generator}_train_retrieval", checkpoint,
        seed=seed, device=device,
    )
    sources = {
        "train": train_dir,
        "dev": seed_dir / "eval" / "dev" / generator,
        "test": seed_dir / "eval" / "test" / generator,
    }
    exports = {}
    records_by_split = {}
    for split, source in sources.items():
        destination = output_dir / f"retrieval.{split}.jsonl"
        records = _export_split(
            source,
            destination,
            checkpoint_sha256=checkpoint_sha256,
            top_k=top_k,
            evidence_path_k=evidence_path_k,
        )
        records_by_split[split] = records
        exports[split] = {
            "path": str(destination.resolve()),
            "sha256": _sha256(destination),
            "records": len(records),
            "source_dir": str(source.resolve()),
        }

    dev_coverage = _dev_evidence_coverage(dataset_root, records_by_split["dev"])
    gate = {
        "format_version": 1,
        "completed_stage": "student-path",
        "selection_split": "dev",
        "stage2_allowed": True,
        "best_checkpoint": str(checkpoint.resolve()),
        "best_checkpoint_sha256": checkpoint_sha256,
        "best_metrics": dev_coverage,
        "stage1_selection": {
            "path": str(selection_path.resolve()),
            "sha256": _sha256(selection_path),
            "arm": f"NATIVE_C2_{arm}",
            "selection_owner": selection["selection_owner"],
            "selected_fraction": selection["selected_fraction"],
            "state_sha256": selection[f"{arm}_state_sha256"],
        },
        "performance_gate_note": (
            "This manifest authenticates the frozen Stage-1 handoff for Stage-2. "
            "It does not change the Stage-1 performance-gate decision."
        ),
        "retrievals": exports,
    }
    gate_path = output_dir / "stage1_gate.json"
    _write_json(gate_path, gate)
    validate_stage2_gate(
        gate_path,
        [Path(exports[split]["path"]) for split in ("train", "dev", "test")],
    )
    manifest = {
        "status": "COMPLETE",
        "stage1_run": str(run_root.resolve()),
        "dataset_root": str(dataset_root.resolve()),
        "gate": {"path": str(gate_path.resolve()), "sha256": _sha256(gate_path)},
        "retrievals": exports,
        "dev_evidence_coverage": dev_coverage,
    }
    _write_json(output_dir / "EXPORT_MANIFEST.json", manifest)
    return manifest
