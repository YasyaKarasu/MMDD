"""Frozen-reader cache and linear-head training for Stage-2 round 1."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import random
import time
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import torch
from torch.nn import functional as F

from .checkpoints import save_candidate_scorer
from .data import Stage2ObjectIndex
from .metrics import evidence_modality_bucket, macro_dataset_accuracy
from .oracle import OracleColumnExample, oracle_examples_fingerprint
from .pipeline import Stage2Backend
from .verifier import CandidateColumnScorer


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def write_jsonl_atomic(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(path)


def _json_fingerprint(payload: Any) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _record_from_states(
    example: OracleColumnExample,
    open_states: torch.Tensor,
    close_states: torch.Tensor,
    *,
    reader_image_policy: str,
) -> dict[str, Any]:
    if open_states.shape != close_states.shape:
        raise ValueError("Opening and closing reader states differ in shape")
    if open_states.ndim != 2 or open_states.shape[0] != len(
        example.candidate_column_indices
    ):
        raise ValueError(
            f"{example.query_id}: reader marker count does not match target columns"
        )
    return {
        "dataset": example.dataset,
        "split": example.split,
        "query_id": example.query_id,
        "target_id": example.target_id,
        "gold_column_position": example.gold_column_position,
        "gold_column_index": example.gold_local_column,
        "gold_source_column_index": example.gold_source_column,
        "candidate_column_indices": example.candidate_column_indices,
        "candidate_column_count": len(example.candidate_column_indices),
        "evidence_ids": example.positive_bundle.evidence_ids,
        "evidence_modalities": example.evidence_modalities,
        "reader_image_policy": reader_image_policy,
        "open_states": open_states.detach().float().cpu(),
        "close_states": close_states.detach().float().cpu(),
    }


def build_reader_cache(
    backend: Stage2Backend,
    examples: Sequence[OracleColumnExample],
    objects: Stage2ObjectIndex,
    cache_dir: Path,
    *,
    model_path: Path,
    model_dtype: str,
    top_k_evidence: int,
    evidence_policy: str,
    shard_size: int = 64,
) -> dict[str, Any]:
    """Materialize resumable, atomically-written frozen reader-state shards."""

    if shard_size <= 0:
        raise ValueError("shard_size must be positive")
    if not examples:
        raise ValueError("Reader cache requires at least one example")
    model = getattr(backend, "model", None)
    if model is None:
        raise ValueError("Reader cache backend must expose its frozen model")
    if model.training or any(parameter.requires_grad for parameter in model.parameters()):
        raise ValueError("Reader cache requires a frozen backbone in eval mode")

    metadata = {
        "format_version": 1,
        "training_source": "oracle_positive",
        "dataset": sorted({example.dataset for example in examples}),
        "split": sorted({example.split for example in examples}),
        "dataset_roots": sorted({example.dataset_root for example in examples}),
        "model_path": str(model_path.resolve()),
        "hidden_dim": int(backend.hidden_dim),
        "model_dtype": model_dtype,
        "top_k_evidence": top_k_evidence,
        "evidence_policy": evidence_policy,
        "sample_count": len(examples),
        "sample_fingerprint": oracle_examples_fingerprint(examples),
        "shard_size": shard_size,
    }
    fingerprint = _json_fingerprint(metadata)
    manifest_path = cache_dir / "manifest.json"
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("metadata_fingerprint") != fingerprint:
            raise ValueError(f"{cache_dir}: reader cache metadata mismatch")
    cache_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    parameter_versions = tuple(parameter._version for parameter in model.parameters())
    backend_device = getattr(backend, "device", torch.device("cpu"))
    if torch.device(backend_device).type == "cuda":
        torch.cuda.reset_peak_memory_stats(torch.device(backend_device))
    shard_paths = []
    shard_starts = list(range(0, len(examples), shard_size))
    for shard_index, start in enumerate(shard_starts):
        shard_examples = examples[start : start + shard_size]
        shard_path = cache_dir / f"shard_{shard_index:05d}.pt"
        shard_paths.append(shard_path.name)
        if shard_path.is_file():
            payload = torch.load(shard_path, map_location="cpu", weights_only=True)
            if payload.get("metadata_fingerprint") != fingerprint:
                raise ValueError(f"{shard_path}: reader cache shard metadata mismatch")
            expected_ids = [example.query_id for example in shard_examples]
            if [record["query_id"] for record in payload["records"]] != expected_ids:
                raise ValueError(f"{shard_path}: reader cache shard sample mismatch")
            continue
        records = []
        with torch.inference_mode():
            for example in shard_examples:
                try:
                    open_states, close_states = backend.reader_states(
                        objects.queries[example.query_id],
                        objects.targets[example.target_id],
                        [
                            objects.evidence[asset_id]
                            for asset_id in example.positive_bundle.evidence_ids
                        ],
                    )
                except Exception as error:
                    evidence_ids = ",".join(example.positive_bundle.evidence_ids)
                    raise RuntimeError(
                        f"reader cache failed for {example.dataset}/{example.split}/"
                        f"{example.query_id}->{example.target_id}; evidence={evidence_ids}"
                    ) from error
                records.append(
                    _record_from_states(
                        example,
                        open_states,
                        close_states,
                        reader_image_policy=str(
                            getattr(backend, "last_reader_image_policy", "processor_default")
                        ),
                    )
                )
        temporary = shard_path.with_suffix(shard_path.suffix + ".tmp")
        torch.save(
            {
                "format_version": 1,
                "metadata_fingerprint": fingerprint,
                "records": records,
            },
            temporary,
        )
        temporary.replace(shard_path)
        print(
            f"reader-cache {cache_dir}: shard {shard_index + 1}/{len(shard_starts)} "
            f"({min(start + shard_size, len(examples))}/{len(examples)} examples)",
            flush=True,
        )
    if parameter_versions != tuple(parameter._version for parameter in model.parameters()):
        raise RuntimeError("Frozen Qwen parameters changed while building reader cache")
    manifest = {
        **metadata,
        "metadata_fingerprint": fingerprint,
        "shards": shard_paths,
        "complete": True,
        "elapsed_seconds_this_run": time.monotonic() - started,
        "device": str(backend_device),
        "peak_gpu_memory_bytes": (
            int(torch.cuda.max_memory_allocated(torch.device(backend_device)))
            if torch.device(backend_device).type == "cuda"
            else None
        ),
        "reader_oom_image_max_pixels": getattr(
            backend, "reader_oom_image_max_pixels", None
        ),
        "reader_oom_retry_examples": sum(
            1
            for shard_name in shard_paths
            for record in torch.load(
                cache_dir / shard_name, map_location="cpu", weights_only=True
            )["records"]
            if record.get("reader_image_policy", "processor_default")
            != "processor_default"
        ),
        "backbone_requires_grad": False,
        "backbone_parameter_versions_unchanged": True,
    }
    write_json_atomic(manifest_path, manifest)
    return manifest


def load_reader_cache(cache_dirs: Sequence[Path]) -> tuple[list[dict[str, Any]], str]:
    records: list[dict[str, Any]] = []
    manifests = []
    seen_samples: set[tuple[str, str, str]] = set()
    hidden_dims = set()
    for cache_dir in cache_dirs:
        manifest_path = cache_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not manifest.get("complete"):
            raise ValueError(f"{cache_dir}: reader cache is incomplete")
        metadata = {
            key: value
            for key, value in manifest.items()
            if key
            not in {
                "metadata_fingerprint",
                "shards",
                "complete",
                "elapsed_seconds_this_run",
                "device",
                "peak_gpu_memory_bytes",
                "reader_oom_image_max_pixels",
                "reader_oom_retry_examples",
                "backbone_requires_grad",
                "backbone_parameter_versions_unchanged",
            }
        }
        if _json_fingerprint(metadata) != manifest.get("metadata_fingerprint"):
            raise ValueError(f"{cache_dir}: reader cache manifest fingerprint mismatch")
        hidden_dims.add(int(manifest["hidden_dim"]))
        manifests.append(manifest)
        for shard_name in manifest["shards"]:
            shard_path = cache_dir / shard_name
            payload = torch.load(shard_path, map_location="cpu", weights_only=True)
            if payload.get("metadata_fingerprint") != manifest["metadata_fingerprint"]:
                raise ValueError(f"{shard_path}: reader cache shard metadata mismatch")
            for record in payload["records"]:
                key = (
                    str(record["dataset"]),
                    str(record["query_id"]),
                    str(record["target_id"]),
                )
                if key in seen_samples:
                    raise ValueError(
                        "Duplicate reader-cache sample: " + ":".join(key)
                    )
                seen_samples.add(key)
                records.append(record)
    if len(hidden_dims) != 1:
        raise ValueError("Reader cache hidden dimensions do not match")
    records.sort(
        key=lambda item: (
            item["dataset"],
            item["split"],
            item["query_id"],
            item["target_id"],
        )
    )
    cache_fingerprint = _json_fingerprint(
        [manifest["metadata_fingerprint"] for manifest in manifests]
    )
    return records, cache_fingerprint


def _balanced_epoch_order(
    records: Sequence[dict[str, Any]], generator: random.Random
) -> tuple[list[int], dict[str, int]]:
    by_dataset: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        by_dataset[str(record["dataset"])].append(index)
    if len(by_dataset) < 2:
        indices = list(range(len(records)))
        generator.shuffle(indices)
        return indices, {dataset: len(indices) for dataset in by_dataset}
    for indices in by_dataset.values():
        generator.shuffle(indices)
    width = max(len(indices) for indices in by_dataset.values())
    order = []
    seen = Counter()
    datasets = sorted(by_dataset)
    for position in range(width):
        for dataset in datasets:
            indices = by_dataset[dataset]
            order.append(indices[position % len(indices)])
            seen[dataset] += 1
    return order, dict(seen)


def _column_count_bucket(count: int) -> str:
    if count <= 2:
        return str(count)
    if count <= 4:
        return "3-4"
    if count <= 8:
        return "5-8"
    return "9+"


@torch.inference_mode()
def evaluate_scorer(
    scorer: CandidateColumnScorer,
    records: Sequence[dict[str, Any]],
    *,
    include_predictions: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    scorer.eval()
    device = next(scorer.parameters()).device
    results = []
    for record in records:
        logits = scorer(
            record["open_states"].to(device), record["close_states"].to(device)
        ).float().cpu()
        probabilities = torch.softmax(logits, dim=0)
        gold = int(record["gold_column_position"])
        order = sorted(range(len(logits)), key=lambda index: (-float(logits[index]), index))
        predicted = order[0]
        rank = order.index(gold) + 1
        candidate_indices = [int(value) for value in record["candidate_column_indices"]]
        results.append(
            {
                "dataset": str(record["dataset"]),
                "split": str(record["split"]),
                "query_id": str(record["query_id"]),
                "target_id": str(record["target_id"]),
                "gold_column_position": gold,
                "predicted_column_position": predicted,
                "gold_column_index": int(record["gold_column_index"]),
                "predicted_column_index": candidate_indices[predicted],
                "correct": predicted == gold,
                "candidate_column_count": int(record["candidate_column_count"]),
                "evidence_ids": list(record["evidence_ids"]),
                "evidence_modalities": list(record["evidence_modalities"]),
                "modality_bucket": evidence_modality_bucket(
                    record["evidence_modalities"]
                ),
                "column_count_bucket": _column_count_bucket(
                    int(record["candidate_column_count"])
                ),
                "rank": rank,
                "nll": float(-torch.log(probabilities[gold].clamp_min(1e-30))),
                "logits": [float(value) for value in logits],
                "probabilities": [float(value) for value in probabilities],
            }
        )

    def aggregate(items: Sequence[dict[str, Any]]) -> dict[str, float | int]:
        if not items:
            return {"examples": 0, "column_accuracy@1": math.nan, "column_mrr": math.nan, "column_nll": math.nan}
        return {
            "examples": len(items),
            "column_accuracy@1": mean(float(item["correct"]) for item in items),
            "column_mrr": mean(1.0 / int(item["rank"]) for item in items),
            "column_nll": mean(float(item["nll"]) for item in items),
        }

    metrics: dict[str, Any] = aggregate(results)
    for field, output_name in (
        ("dataset", "by_dataset"),
        ("modality_bucket", "by_evidence_modality"),
        ("column_count_bucket", "by_candidate_column_count"),
    ):
        values: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for result in results:
            values[str(result[field])].append(result)
        metrics[output_name] = {
            key: aggregate(items) for key, items in sorted(values.items())
        }
    return metrics, results if include_predictions else []


def train_cached_scorer(
    train_records: Sequence[dict[str, Any]],
    dev_records: Sequence[dict[str, Any]],
    *,
    hidden_dim: int,
    seed: int,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    output_dir: Path,
    checkpoint_metadata: dict[str, Any],
) -> tuple[CandidateColumnScorer, dict[str, Any], CandidateColumnScorer]:
    if epochs <= 0 or not train_records or not dev_records:
        raise ValueError("Cached training requires positive epochs and non-empty train/dev data")
    torch.manual_seed(seed)
    scorer = CandidateColumnScorer(hidden_dim)
    epoch_zero = copy.deepcopy(scorer)
    initial_state = copy.deepcopy(scorer.state_dict())
    optimizer = torch.optim.AdamW(
        scorer.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    generator = random.Random(seed)
    history = []
    best_key: tuple[float, float, int] | None = None
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch: int | None = None
    started = time.monotonic()
    for epoch in range(1, epochs + 1):
        scorer.train()
        order, seen = _balanced_epoch_order(train_records, generator)
        total_loss = 0.0
        for index in order:
            record = train_records[index]
            logits = scorer(record["open_states"], record["close_states"])
            loss = F.cross_entropy(
                logits.unsqueeze(0),
                torch.tensor([int(record["gold_column_position"])], dtype=torch.long),
            )
            if not torch.isfinite(loss):
                raise ValueError(f"Non-finite Stage-2 column loss at epoch {epoch}")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach())
        dev_metrics, _ = evaluate_scorer(scorer, dev_records)
        macro_accuracy = macro_dataset_accuracy(dev_metrics)
        key = (macro_accuracy, -float(dev_metrics["column_nll"]), -epoch)
        history.append(
            {
                "epoch": epoch,
                "mean_train_loss": total_loss / len(order),
                "examples_seen_by_dataset": seen,
                "dev": dev_metrics,
                "dev_macro_column_accuracy@1": macro_accuracy,
            }
        )
        if best_key is None or key > best_key:
            best_key = key
            best_state = copy.deepcopy(scorer.state_dict())
            best_epoch = epoch
    if best_state is None or best_epoch is None:
        raise RuntimeError("No Stage-2 checkpoint was selected")
    scorer.load_state_dict(best_state)
    head_updated = any(
        not torch.equal(initial_state[name], scorer.state_dict()[name])
        for name in initial_state
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        **checkpoint_metadata,
        "training_source": "oracle_positive",
        "seed": seed,
        "epochs": epochs,
        "selected_epoch": best_epoch,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "dev_selection_metric": "macro_mean_per_lake_column_accuracy@1_then_nll_then_earlier_epoch",
    }
    save_candidate_scorer(output_dir / "candidate.pt", scorer, metadata=metadata)
    run_summary = {
        "seed": seed,
        "selected_epoch": best_epoch,
        "head_parameters_updated": head_updated,
        "elapsed_seconds": time.monotonic() - started,
        "history": history,
    }
    write_json_atomic(output_dir / "history.json", run_summary)
    return scorer, run_summary, epoch_zero


def majority_position(train_records: Sequence[dict[str, Any]]) -> int:
    counts = Counter(int(record["gold_column_position"]) for record in train_records)
    return min(counts, key=lambda position: (-counts[position], position))


def deterministic_baselines(
    train_records: Sequence[dict[str, Any]], records: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    majority = majority_position(train_records)
    results = []
    for record in records:
        gold = int(record["gold_column_position"])
        count = int(record["candidate_column_count"])
        results.append(
            {
                "dataset": str(record["dataset"]),
                "uniform_expectation": 1.0 / count,
                "majority_correct": gold == majority and majority < count,
            }
        )

    def aggregate(items: Sequence[dict[str, Any]]) -> dict[str, Any]:
        return {
            "examples": len(items),
            "uniform_random_expectation": mean(
                item["uniform_expectation"] for item in items
            ),
            "majority_column_position": majority,
            "majority_column_position_accuracy@1": mean(
                float(item["majority_correct"]) for item in items
            ),
        }

    by_dataset = {
        dataset: aggregate([item for item in results if item["dataset"] == dataset])
        for dataset in sorted({item["dataset"] for item in results})
    }
    return {**aggregate(results), "by_dataset": by_dataset, "per_sample": results}


def paired_bootstrap_ci(
    trained_predictions: Sequence[dict[str, Any]],
    baseline_values: Sequence[float],
    *,
    samples: int = 10_000,
    seed: int = 13,
) -> dict[str, float | int]:
    if len(trained_predictions) != len(baseline_values) or not baseline_values:
        raise ValueError("Paired bootstrap inputs must have equal non-zero lengths")
    differences = torch.tensor(
        [
            float(prediction["correct"]) - float(baseline)
            for prediction, baseline in zip(
                trained_predictions, baseline_values, strict=True
            )
        ],
        dtype=torch.float64,
    )
    generator = torch.Generator().manual_seed(seed)
    estimates = torch.empty(samples, dtype=torch.float64)
    batch_size = 128
    for start in range(0, samples, batch_size):
        width = min(batch_size, samples - start)
        indices = torch.randint(
            len(differences), (width, len(differences)), generator=generator
        )
        estimates[start : start + width] = differences[indices].mean(dim=1)
    return {
        "samples": samples,
        "seed": seed,
        "mean_difference": float(differences.mean()),
        "ci95_low": float(torch.quantile(estimates, 0.025)),
        "ci95_high": float(torch.quantile(estimates, 0.975)),
    }


def mean_std(values: Sequence[float]) -> dict[str, float]:
    return {"mean": mean(values), "std": pstdev(values)}
