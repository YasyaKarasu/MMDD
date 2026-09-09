"""Corrected candidate-column protocol for the R12 end-to-end experiment."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import random
import time
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any

import torch
from mmdd_dataset.wdc_runtime import iter_dataset_artifact
from torch.nn import functional as F

from .checkpoints import save_candidate_scorer
from .data import (
    Stage2ObjectIndex,
    load_stage2_index,
    local_column_index,
    permute_table_columns,
)
from .oracle import select_oracle_evidence
from .pipeline import Stage2Backend
from .verifier import CandidateColumnScorer

EXAMPLE_TYPES = ("positive", "wrong_target", "no_available_column")


@dataclass(frozen=True)
class R12ColumnExample:
    dataset: str
    dataset_root: str
    protocol_split: str
    query_id: str
    target_id: str
    source_table_id: str
    gold_source_column: int
    gold_local_column: int
    evidence_ids: tuple[str, ...]
    evidence_modalities: tuple[str, ...]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_fingerprint(payload: Any) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def load_r12_column_examples(
    dataset_root: Path,
    target_lists: dict[str, Path],
    *,
    top_k_evidence: int = 4,
) -> tuple[list[R12ColumnExample], Stage2ObjectIndex, dict[str, Any]]:
    """Load R12 train/cal examples without exposing the frozen Task F dev sample."""

    if top_k_evidence <= 0:
        raise ValueError("top_k_evidence must be positive")
    rows: list[tuple[str, dict[str, Any]]] = []
    selected_pairs: set[tuple[str, str]] = set()
    query_ids: set[str] = set()
    target_ids: set[str] = set()
    evidence_ids: set[str] = set()
    for protocol_split, path in target_lists.items():
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                if record.get("query_kind") != "implicit":
                    continue
                query_id = str(record["query_id"])
                for target_id, ids in record.get("positive_evidence_by_target", {}).items():
                    if not ids:
                        continue
                    pair = (query_id, str(target_id))
                    if pair in selected_pairs:
                        raise ValueError(f"Duplicate R12 column pair: {query_id}->{target_id}")
                    selected_pairs.add(pair)
                    rows.append(
                        (
                            protocol_split,
                            {
                                "query_id": query_id,
                                "target_id": str(target_id),
                                "evidence_ids": tuple(str(value) for value in ids),
                            },
                        )
                    )
                    query_ids.add(query_id)
                    target_ids.add(str(target_id))
                    evidence_ids.update(str(value) for value in ids)

    qrels = {
        (str(record["query_table_id"]), str(record["target_table_id"])): record
        for record in iter_dataset_artifact(dataset_root, "qrels")
        if (str(record["query_table_id"]), str(record["target_table_id"])) in selected_pairs
        and record.get("reason") == "model_recoverable_join_column"
    }
    missing_qrels = selected_pairs - qrels.keys()
    if missing_qrels:
        query_id, target_id = min(missing_qrels)
        raise KeyError(f"Missing recoverable qrel for {query_id}->{target_id}")
    objects = load_stage2_index(
        dataset_root,
        query_ids=query_ids,
        target_ids=target_ids,
        evidence_ids=evidence_ids,
    )
    dataset = dataset_root.name
    examples = []
    for protocol_split, row in rows:
        qrel = qrels[(row["query_id"], row["target_id"])]
        source_column = int(qrel["join_attribute"]["source_column_index"])
        local_column = local_column_index(objects.targets[row["target_id"]], source_column)
        selected = select_oracle_evidence(
            row["evidence_ids"], objects.evidence, top_k=top_k_evidence
        )
        examples.append(
            R12ColumnExample(
                dataset=dataset,
                dataset_root=str(dataset_root.resolve()),
                protocol_split=protocol_split,
                query_id=row["query_id"],
                target_id=row["target_id"],
                source_table_id=str(qrel["source_table_id"]),
                gold_source_column=source_column,
                gold_local_column=local_column,
                evidence_ids=selected,
                evidence_modalities=tuple(
                    str(objects.evidence[asset_id]["asset_type"])
                    for asset_id in selected
                ),
            )
        )
    examples.sort(
        key=lambda item: (item.protocol_split, item.query_id, item.target_id)
    )
    audit = {
        "format_version": 1,
        "dataset_root": str(dataset_root.resolve()),
        "target_lists": {
            split: {"path": str(path.resolve()), "sha256": _sha256(path)}
            for split, path in sorted(target_lists.items())
        },
        "examples": len(examples),
        "by_split": dict(sorted(Counter(item.protocol_split for item in examples).items())),
        "source_groups_by_split": {
            split: len(
                {item.source_table_id for item in examples if item.protocol_split == split}
            )
            for split in sorted(target_lists)
        },
        "task_f_dev_queries_loaded": False,
    }
    return examples, objects, audit


def _wrong_target_examples(
    examples: Sequence[R12ColumnExample], *, seed: int
) -> dict[tuple[str, str, str], R12ColumnExample]:
    result = {}
    by_split: dict[str, list[R12ColumnExample]] = defaultdict(list)
    for example in examples:
        by_split[example.protocol_split].append(example)
    for split, values in by_split.items():
        if len({item.target_id for item in values}) < 2:
            raise ValueError(f"{split}: wrong-target controls require two target IDs")
        ordered = sorted(
            values,
            key=lambda item: hashlib.sha256(
                f"{seed}\0{split}\0{item.query_id}\0{item.target_id}".encode()
            ).digest(),
        )
        for position, example in enumerate(ordered):
            donors = [
                candidate
                for offset in range(1, len(ordered))
                if (
                    (candidate := ordered[(position + offset) % len(ordered)]).source_table_id
                    != example.source_table_id
                    and candidate.target_id != example.target_id
                )
            ]
            if not donors:
                raise ValueError(
                    f"{split}: no cross-source wrong target for {example.query_id}"
                )
            donor = donors[0]
            result[(split, example.query_id, example.target_id)] = donor
    return result


def _cache_record(
    example: R12ColumnExample,
    *,
    example_type: str,
    presented_target_id: str,
    presented_target: dict[str, Any],
    open_states: torch.Tensor,
    close_states: torch.Tensor,
    reader_image_policy: str,
) -> dict[str, Any]:
    if open_states.shape != close_states.shape or open_states.ndim != 2:
        raise ValueError("Invalid R12 reader-state shapes")
    candidate_indices = tuple(
        int(column["column_index"]) for column in presented_target["columns"]
    )
    if open_states.shape[0] != len(candidate_indices):
        raise ValueError("R12 reader marker count does not match presented columns")
    gold_position = None
    if example_type == "positive":
        gold_position = candidate_indices.index(example.gold_local_column)
    return {
        "dataset": example.dataset,
        "protocol_split": example.protocol_split,
        "query_id": example.query_id,
        "gold_target_id": example.target_id,
        "presented_target_id": presented_target_id,
        "source_table_id": example.source_table_id,
        "example_type": example_type,
        "gold_column_position": gold_position,
        "gold_column_index": example.gold_local_column if gold_position is not None else None,
        "candidate_column_indices": candidate_indices,
        "candidate_column_count": len(candidate_indices),
        "evidence_ids": example.evidence_ids,
        "evidence_modalities": example.evidence_modalities,
        "reader_image_policy": reader_image_policy,
        "open_states": open_states.detach().half().cpu(),
        "close_states": close_states.detach().half().cpu(),
    }


def build_r12_reader_cache(
    backend: Stage2Backend,
    examples: Sequence[R12ColumnExample],
    objects: Stage2ObjectIndex,
    cache_dir: Path,
    *,
    model_path: Path,
    model_dtype: str,
    top_k_evidence: int,
    column_permutation_seed: int,
    shard_size: int = 32,
    reverse_shards: bool = False,
) -> dict[str, Any]:
    """Cache positive, wrong-target, and missing-column reader states."""

    if shard_size <= 0 or not examples:
        raise ValueError("R12 reader cache requires examples and a positive shard size")
    model = getattr(backend, "model", None)
    if model is None or model.training or any(
        parameter.requires_grad for parameter in model.parameters()
    ):
        raise ValueError("R12 reader cache requires a frozen backbone in eval mode")
    wrong_targets = _wrong_target_examples(examples, seed=column_permutation_seed)
    metadata = {
        "format_version": 1,
        "protocol": "r12_candidate_column_with_rejection",
        "dataset_roots": sorted({item.dataset_root for item in examples}),
        "protocol_splits": sorted({item.protocol_split for item in examples}),
        "model_path": str(model_path.resolve()),
        "model_dtype": model_dtype,
        "hidden_dim": int(backend.hidden_dim),
        "top_k_evidence": top_k_evidence,
        "column_permutation_seed": column_permutation_seed,
        "column_permutation_key": "sha256(seed,target_id,column_index)",
        "control_types": list(EXAMPLE_TYPES),
        "source_examples": len(examples),
        "source_fingerprint": _json_fingerprint([asdict(item) for item in examples]),
        "shard_size": shard_size,
    }
    fingerprint = _json_fingerprint(metadata)
    manifest_path = cache_dir / "manifest.json"
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("metadata_fingerprint") != fingerprint:
            raise ValueError(f"{cache_dir}: R12 reader cache metadata mismatch")
    cache_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    parameter_versions = tuple(parameter._version for parameter in model.parameters())
    shard_starts = list(range(0, len(examples), shard_size))
    shard_paths = [
        f"shard_{start // shard_size:05d}.pt" for start in shard_starts
    ]
    skipped_no_available = 0
    ordered_starts = list(reversed(shard_starts)) if reverse_shards else shard_starts
    for completed_index, start in enumerate(ordered_starts, 1):
        shard_index = start // shard_size
        shard_examples = examples[start : start + shard_size]
        shard_path = cache_dir / f"shard_{shard_index:05d}.pt"
        expected_source_ids = [
            (item.protocol_split, item.query_id, item.target_id) for item in shard_examples
        ]
        if shard_path.is_file():
            payload = torch.load(shard_path, map_location="cpu", weights_only=True)
            if payload.get("metadata_fingerprint") != fingerprint:
                raise ValueError(f"{shard_path}: R12 reader cache metadata mismatch")
            if payload.get("source_ids") != expected_source_ids:
                raise ValueError(f"{shard_path}: R12 reader cache sample mismatch")
            skipped_no_available += int(payload.get("skipped_no_available", 0))
            continue
        records = []
        shard_skipped = 0
        with torch.inference_mode():
            for example in shard_examples:
                query = objects.queries[example.query_id]
                evidence = [objects.evidence[value] for value in example.evidence_ids]
                target = objects.targets[example.target_id]
                positive_target = permute_table_columns(
                    target, seed=column_permutation_seed
                )
                open_states, close_states = backend.reader_states(
                    query, positive_target, evidence
                )
                records.append(
                    _cache_record(
                        example,
                        example_type="positive",
                        presented_target_id=example.target_id,
                        presented_target=positive_target,
                        open_states=open_states,
                        close_states=close_states,
                        reader_image_policy=str(
                            getattr(backend, "last_reader_image_policy", "processor_default")
                        ),
                    )
                )

                donor = wrong_targets[
                    (example.protocol_split, example.query_id, example.target_id)
                ]
                wrong_target = permute_table_columns(
                    objects.targets[donor.target_id], seed=column_permutation_seed
                )
                open_states, close_states = backend.reader_states(
                    query, wrong_target, evidence
                )
                records.append(
                    _cache_record(
                        example,
                        example_type="wrong_target",
                        presented_target_id=donor.target_id,
                        presented_target=wrong_target,
                        open_states=open_states,
                        close_states=close_states,
                        reader_image_policy=str(
                            getattr(backend, "last_reader_image_policy", "processor_default")
                        ),
                    )
                )

                no_available = permute_table_columns(
                    target,
                    seed=column_permutation_seed,
                    excluded_column_indices=(example.gold_local_column,),
                )
                if no_available["columns"]:
                    open_states, close_states = backend.reader_states(
                        query, no_available, evidence
                    )
                    records.append(
                        _cache_record(
                            example,
                            example_type="no_available_column",
                            presented_target_id=example.target_id,
                            presented_target=no_available,
                            open_states=open_states,
                            close_states=close_states,
                            reader_image_policy=str(
                                getattr(
                                    backend,
                                    "last_reader_image_policy",
                                    "processor_default",
                                )
                            ),
                        )
                    )
                else:
                    shard_skipped += 1
        temporary = shard_path.with_suffix(f".pt.{os.getpid()}.tmp")
        torch.save(
            {
                "format_version": 1,
                "metadata_fingerprint": fingerprint,
                "source_ids": expected_source_ids,
                "skipped_no_available": shard_skipped,
                "records": records,
            },
            temporary,
        )
        temporary.replace(shard_path)
        skipped_no_available += shard_skipped
        print(
            f"r12-reader-cache {cache_dir}: worker item {completed_index}/"
            f"{math.ceil(len(examples) / shard_size)} (shard {shard_index + 1})",
            flush=True,
        )
    if parameter_versions != tuple(parameter._version for parameter in model.parameters()):
        raise RuntimeError("Frozen Qwen parameters changed while building the R12 cache")
    record_count = len(examples) * 3 - skipped_no_available
    manifest = {
        **metadata,
        "metadata_fingerprint": fingerprint,
        "shards": shard_paths,
        "record_count": record_count,
        "skipped_no_available_column": skipped_no_available,
        "complete": True,
        "elapsed_seconds_this_run": time.monotonic() - started,
        "device": str(getattr(backend, "device", "cpu")),
        "backbone_parameter_versions_unchanged": True,
    }
    _write_json(manifest_path, manifest)
    return manifest


def load_r12_reader_cache(cache_dirs: Sequence[Path]) -> tuple[list[dict[str, Any]], str]:
    records = []
    fingerprints = []
    seen = set()
    hidden_dims = set()
    permutation_seeds = set()
    for cache_dir in cache_dirs:
        manifest = json.loads((cache_dir / "manifest.json").read_text(encoding="utf-8"))
        if not manifest.get("complete"):
            raise ValueError(f"{cache_dir}: incomplete R12 reader cache")
        fingerprints.append(str(manifest["metadata_fingerprint"]))
        hidden_dims.add(int(manifest["hidden_dim"]))
        permutation_seeds.add(int(manifest["column_permutation_seed"]))
        for shard_name in manifest["shards"]:
            payload = torch.load(
                cache_dir / shard_name, map_location="cpu", weights_only=True
            )
            if payload.get("metadata_fingerprint") != manifest["metadata_fingerprint"]:
                raise ValueError(f"{cache_dir / shard_name}: fingerprint mismatch")
            for record in payload["records"]:
                key = (
                    record["protocol_split"],
                    record["query_id"],
                    record["gold_target_id"],
                    record["example_type"],
                )
                if key in seen:
                    raise ValueError("Duplicate R12 reader-cache record: " + ":".join(key))
                seen.add(key)
                records.append(record)
    if len(hidden_dims) != 1 or len(permutation_seeds) != 1:
        raise ValueError("R12 reader caches disagree on model or permutation seed")
    records.sort(
        key=lambda item: (
            item["protocol_split"],
            item["query_id"],
            item["gold_target_id"],
            item["example_type"],
        )
    )
    return records, _json_fingerprint(fingerprints)


def _record_loss(scorer: CandidateColumnScorer, record: dict[str, Any]) -> torch.Tensor:
    logits = scorer(record["open_states"].float(), record["close_states"].float())
    if record["example_type"] != "positive":
        return F.binary_cross_entropy_with_logits(logits, torch.zeros_like(logits))
    gold = int(record["gold_column_position"])
    positive_loss = F.binary_cross_entropy_with_logits(
        logits[gold], torch.ones_like(logits[gold])
    )
    negative_mask = torch.ones(len(logits), dtype=torch.bool)
    negative_mask[gold] = False
    if not negative_mask.any():
        return positive_loss
    negative_loss = F.binary_cross_entropy_with_logits(
        logits[negative_mask], torch.zeros_like(logits[negative_mask])
    )
    return (positive_loss + negative_loss) / 2


@torch.inference_mode()
def score_r12_records(
    scorer: CandidateColumnScorer, records: Sequence[dict[str, Any]]
) -> list[dict[str, Any]]:
    scorer.eval()
    scored = []
    for record in records:
        logits = scorer(record["open_states"].float(), record["close_states"].float())
        probabilities = torch.sigmoid(logits)
        predicted = int(logits.argmax())
        scored.append(
            {
                "dataset": record["dataset"],
                "protocol_split": record["protocol_split"],
                "query_id": record["query_id"],
                "gold_target_id": record["gold_target_id"],
                "presented_target_id": record["presented_target_id"],
                "source_table_id": record["source_table_id"],
                "example_type": record["example_type"],
                "gold_column_position": record["gold_column_position"],
                "predicted_column_position": predicted,
                "gold_column_index": record["gold_column_index"],
                "predicted_column_index": int(record["candidate_column_indices"][predicted]),
                "candidate_column_count": int(record["candidate_column_count"]),
                "max_probability": float(probabilities.max()),
                "column_correct_without_rejection": (
                    record["example_type"] == "positive"
                    and predicted == int(record["gold_column_position"])
                ),
                "logits": [float(value) for value in logits],
                "probabilities": [float(value) for value in probabilities],
            }
        )
    return scored


def select_rejection_threshold(scored: Sequence[dict[str, Any]]) -> float:
    values = sorted({float(record["max_probability"]) for record in scored})
    if not values:
        raise ValueError("Cannot fit a rejection threshold without records")
    candidates = [0.0, *values, math.nextafter(values[-1], math.inf)]
    best_key = None
    best_threshold = None
    for threshold in candidates:
        metrics = evaluate_r12_scores(scored, threshold=threshold)
        key = (
            float(metrics["macro_decision_accuracy"]),
            float(metrics["by_example_type"]["positive"]["decision_accuracy"]),
            -float(metrics["control_false_accept_rate"]),
            threshold,
        )
        if best_key is None or key > best_key:
            best_key = key
            best_threshold = threshold
    assert best_threshold is not None
    return best_threshold


def evaluate_r12_scores(
    scored: Sequence[dict[str, Any]], *, threshold: float
) -> dict[str, Any]:
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    predictions = []
    for record in scored:
        accepted = float(record["max_probability"]) >= threshold
        correct = (
            accepted and bool(record["column_correct_without_rejection"])
            if record["example_type"] == "positive"
            else not accepted
        )
        result = {**record, "accepted": accepted, "decision_correct": correct}
        by_type[str(record["example_type"])].append(result)
        predictions.append(result)
    metrics_by_type = {}
    for example_type in EXAMPLE_TYPES:
        items = by_type.get(example_type, [])
        metrics_by_type[example_type] = {
            "examples": len(items),
            "decision_accuracy": mean(float(item["decision_correct"]) for item in items)
            if items
            else math.nan,
            "acceptance_rate": mean(float(item["accepted"]) for item in items)
            if items
            else math.nan,
            "column_accuracy_without_rejection": mean(
                float(item["column_correct_without_rejection"]) for item in items
            )
            if items and example_type == "positive"
            else None,
        }
    controls = [
        item for item in predictions if item["example_type"] != "positive"
    ]
    return {
        "examples": len(predictions),
        "threshold": threshold,
        "macro_decision_accuracy": mean(
            float(metrics_by_type[value]["decision_accuracy"])
            for value in EXAMPLE_TYPES
            if metrics_by_type[value]["examples"]
        ),
        "control_false_accept_rate": mean(float(item["accepted"]) for item in controls),
        "by_example_type": metrics_by_type,
        "gold_column_position": dict(
            sorted(
                Counter(
                    str(item["gold_column_position"])
                    for item in predictions
                    if item["example_type"] == "positive"
                ).items()
            )
        ),
        "predictions": predictions,
    }


def train_r12_candidate_scorer(
    train_records: Sequence[dict[str, Any]],
    cal_fit_records: Sequence[dict[str, Any]],
    *,
    hidden_dim: int,
    seed: int,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    output_dir: Path,
    checkpoint_metadata: dict[str, Any],
) -> tuple[CandidateColumnScorer, dict[str, Any]]:
    if epochs <= 0 or not train_records or not cal_fit_records:
        raise ValueError("R12 scorer training requires records and positive epochs")
    torch.manual_seed(seed)
    scorer = CandidateColumnScorer(hidden_dim)
    initial_state = copy.deepcopy(scorer.state_dict())
    optimizer = torch.optim.AdamW(
        scorer.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    generator = random.Random(seed)
    history = []
    best_key = None
    best_state = None
    best_threshold = None
    best_epoch = None
    started = time.monotonic()
    order = list(range(len(train_records)))
    for epoch in range(1, epochs + 1):
        generator.shuffle(order)
        scorer.train()
        total_loss = 0.0
        for index in order:
            loss = _record_loss(scorer, train_records[index])
            if not torch.isfinite(loss):
                raise ValueError(f"Non-finite R12 column loss at epoch {epoch}")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach())
        scored = score_r12_records(scorer, cal_fit_records)
        threshold = select_rejection_threshold(scored)
        metrics = evaluate_r12_scores(scored, threshold=threshold)
        metrics.pop("predictions")
        key = (
            float(metrics["macro_decision_accuracy"]),
            float(metrics["by_example_type"]["positive"]["decision_accuracy"]),
            -float(metrics["control_false_accept_rate"]),
            -epoch,
        )
        history.append(
            {
                "epoch": epoch,
                "mean_train_loss": total_loss / len(order),
                "cal_fit": metrics,
            }
        )
        if best_key is None or key > best_key:
            best_key = key
            best_state = copy.deepcopy(scorer.state_dict())
            best_threshold = threshold
            best_epoch = epoch
    assert best_state is not None and best_threshold is not None and best_epoch is not None
    scorer.load_state_dict(best_state)
    metadata = {
        **checkpoint_metadata,
        "protocol": "r12_candidate_column_with_rejection",
        "seed": seed,
        "epochs": epochs,
        "selected_epoch": best_epoch,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "rejection_threshold": best_threshold,
        "dev_selection": "cal_fit_macro_three_class_decision_then_positive_then_false_accept_then_earlier",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    save_candidate_scorer(output_dir / "candidate.pt", scorer, metadata=metadata)
    summary = {
        "format_version": 1,
        "selected_epoch": best_epoch,
        "rejection_threshold": best_threshold,
        "head_parameters_updated": any(
            not torch.equal(initial_state[name], scorer.state_dict()[name])
            for name in initial_state
        ),
        "elapsed_seconds": time.monotonic() - started,
        "history": history,
    }
    _write_json(output_dir / "history.json", summary)
    return scorer, summary


def write_r12_evaluation(
    output_path: Path,
    scorer: CandidateColumnScorer,
    records: Sequence[dict[str, Any]],
    *,
    threshold: float,
) -> dict[str, Any]:
    payload = evaluate_r12_scores(
        score_r12_records(scorer, records), threshold=threshold
    )
    predictions = payload.pop("predictions")
    _write_json(output_path, payload)
    prediction_path = output_path.with_name(output_path.stem + "_predictions.jsonl")
    temporary = prediction_path.with_suffix(".jsonl.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in predictions:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(prediction_path)
    return payload
