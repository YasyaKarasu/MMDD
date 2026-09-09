#!/usr/bin/env python
"""Profile frozen R13 online phases for S0 or the selected recipe."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from mmdd_stage1.artifacts import checkpoint_fingerprint, write_json
from mmdd_stage1.checkpoints import load_student
from mmdd_stage1.data import load_target_examples
from mmdd_stage1.features import FeatureStore
from mmdd_stage1.retrieval import (
    StudentANNIndices,
    fuse_ranked_channels,
    load_corpus_ids,
    retrieve_zero_one_hop_detailed_many,
)
from mmdd_stage1.row_support import load_evidence_content_keys
from run_stage1_r11_task_e import empty_intervention_stats
from run_stage1_r11_task_f import _target_channels
from run_stage1_r13 import (
    _DirectScorer,
    _output_root,
    _paths,
    _paths_by_target,
    freeze_plan,
)


ARM_PATHS = {
    "s0": (
        None,
        "taskA_stage1_protocol/s0/evaluation_step0/index",
    ),
    "p_s_target_only": (
        "taskD_witness_supervision/p_s_target_only/checkpoints/step_000178.pt",
        "taskD_witness_supervision/p_s_target_only/evaluation_step178/index",
    ),
}


def _rss_bytes() -> int:
    for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("/proc/self/status has no VmRSS")


def _percentiles(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    return {
        "mean": statistics.fmean(ordered),
        "p50": statistics.median(ordered),
        "p95": ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))],
    }


class _TimedIndices:
    def __init__(self, base: StudentANNIndices) -> None:
        self.base = base
        self.store = base.store
        self.seconds: dict[str, float] = defaultdict(float)
        self.vectors: dict[str, int] = defaultdict(int)

    def search_many(
        self, source_ids: list[str], destination_type: str, k: int
    ) -> list[list[tuple[str, float]]]:
        if destination_type != "table":
            phase = "QE"
        elif source_ids and self.store.embedding_features(source_ids[0]).object_type == "table":
            phase = "QT"
        else:
            phase = "ET"
        started = time.monotonic()
        result = self.base.search_many(source_ids, destination_type, k)
        self.seconds[phase] += time.monotonic() - started
        self.vectors[phase] += len(source_ids)
        return result


class _TimedDirectScorer(_DirectScorer):
    def __init__(self, *args: Any) -> None:
        super().__init__(*args)
        self.seconds = 0.0

    def score(self, *args: Any, **kwargs: Any) -> dict[str, float]:
        started = time.monotonic()
        result = super().score(*args, **kwargs)
        self.seconds += time.monotonic() - started
        return result


def _repeat(
    examples: list[Any],
    indices: StudentANNIndices,
    model: Any,
    store: FeatureStore,
    content_keys: dict[str, str],
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    timed = _TimedIndices(indices)
    scorer = _TimedDirectScorer(model, store, device)
    intervention_stats = empty_intervention_stats()
    phase_per_query = {name: [] for name in ("QT", "QE", "ET", "union_direct_supplement", "retention", "fusion", "total")}
    unique_union_targets = []
    repeat_started = time.monotonic()
    for start in range(0, len(examples), batch_size):
        batch = examples[start : start + batch_size]
        before_ann = dict(timed.seconds)
        detailed = retrieve_zero_one_hop_detailed_many(
            [example.query_id for example in batch],
            timed,
            k=50,
            direct_k=100,
            evidence_k=20,
            targets_per_evidence=20,
            evidence_types=("text", "image"),
            evidence_aggregation="logsumexp",
            evidence_top_k=4,
            evidence_temperature=1.0,
            path_combination="sum",
            fusion_mode="rrf",
            query_batch_size=batch_size,
        )
        ann_delta = {
            phase: timed.seconds[phase] - before_ann.get(phase, 0.0)
            for phase in ("QT", "QE", "ET")
        }
        for phase, seconds in ann_delta.items():
            phase_per_query[phase].extend([seconds / len(batch)] * len(batch))
        for example, retrieved in zip(batch, detailed):
            record = {
                "query_id": example.query_id,
                "positive_target_ids": list(example.positive_target_ids),
                "positive_evidence_by_target": {
                    key: list(value)
                    for key, value in (example.positive_evidence_by_target or {}).items()
                },
                "positive_evidence_rows_by_target": {
                    target_id: {
                        evidence_id: list(rows)
                        for evidence_id, rows in evidence_rows.items()
                    }
                    for target_id, evidence_rows in (
                        example.positive_evidence_rows_by_target or {}
                    ).items()
                },
                "query_row_count": example.query_row_count,
                "query_kind": example.query_kind,
                "paths_by_target": _paths_by_target(retrieved),
            }
            unique_union_targets.append(len(record["paths_by_target"]))
            before_direct = scorer.seconds
            retention_started = time.monotonic()
            direct, evidence = _target_channels(
                record,
                retention="e2_row_coverage",
                scorer=scorer,
                store=store,
                content_keys=content_keys,
                top_l=20,
                evidence_budget=4,
                pair_batch_size=256,
                intervention="original_mixed",
                intervention_stats=intervention_stats,
            )
            direct_seconds = scorer.seconds - before_direct
            retention_seconds = time.monotonic() - retention_started - direct_seconds
            fusion_started = time.monotonic()
            fuse_ranked_channels(direct, evidence, rrf_k=60, fusion_mode="rrf")
            fusion_seconds = time.monotonic() - fusion_started
            phase_per_query["union_direct_supplement"].append(direct_seconds)
            phase_per_query["retention"].append(max(retention_seconds, 0.0))
            phase_per_query["fusion"].append(fusion_seconds)
    for index in range(len(examples)):
        phase_per_query["total"].append(
            sum(
                phase_per_query[phase][index]
                for phase in ("QT", "QE", "ET", "union_direct_supplement", "retention", "fusion")
            )
        )
    return {
        "elapsed_seconds": time.monotonic() - repeat_started,
        "phase_seconds_per_query": {
            phase: _percentiles(values) for phase, values in phase_per_query.items()
        },
        "search_vectors": {
            **dict(timed.vectors),
            "total": sum(timed.vectors.values()),
            "per_query_mean": sum(timed.vectors.values()) / len(examples),
            "per_query_theoretical_max": 43,
        },
        "direct_supplement_pairs": scorer.scored_pairs,
        "mean_unique_union_targets": statistics.fmean(unique_union_targets),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    torch.set_num_threads(args.cpu_threads)
    device = torch.device(args.device)
    plan = freeze_plan(args.root)
    output = _output_root(args.root)
    checkpoint_relative, index_relative = ARM_PATHS[args.arm]
    checkpoint = (
        Path(plan["s0"]["path"])
        if checkpoint_relative is None
        else output / checkpoint_relative
    )
    paths = _paths(args.root)
    store = FeatureStore.from_path(paths["features"], cache_size=260_000)
    examples = load_target_examples(paths["dev_targets"], split="dev")
    ids_by_type = load_corpus_ids(paths["corpus"], store)
    preload_started = time.monotonic()
    store.preload_embeddings(
        [
            *(value for ids in ids_by_type.values() for value in ids),
            *(example.query_id for example in examples),
        ]
    )
    preload_seconds = time.monotonic() - preload_started
    model = load_student(checkpoint, device).eval()
    rss_before = _rss_bytes()
    indices = StudentANNIndices(
        model,
        store,
        output / index_relative,
        device=device,
        checkpoint_sha256=checkpoint_fingerprint(checkpoint),
        corpus_sha256=plan["inputs"]["corpus"]["sha256"],
        score_space="raw_logit",
    )
    rss_after = _rss_bytes()
    content_keys, content_keys_sha256 = load_evidence_content_keys(
        paths["evidence_content_keys"]
    )
    repeats = [
        _repeat(
            examples,
            indices,
            model,
            store,
            content_keys,
            device,
            args.query_batch_size,
        )
        for _repeat_index in range(args.repeats)
    ]
    payload = {
        "format_version": 1,
        "status": "complete",
        "arm": args.arm,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_fingerprint(checkpoint),
        "plan_sha256": plan["plan_sha256"],
        "query_stream": {
            "split": "dev",
            "queries": len(examples),
            "order": "materialized target-list order",
            "batch_size": args.query_batch_size,
        },
        "cache_protocol": {
            "repeat_0": (
                "fresh StudentANNIndices object after feature preload; OS page cache "
                "was not forcibly dropped"
            ),
            "repeat_1_plus": "same index object with hot relation-query and process caches",
            "repeats": args.repeats,
        },
        "index_resident_memory": {
            "process_rss_before_index_load_bytes": rss_before,
            "process_rss_after_index_load_bytes": rss_after,
            "rss_delta_bytes": max(rss_after - rss_before, 0),
            "measurement": "/proc/self/status VmRSS",
        },
        "feature_preload_seconds": preload_seconds,
        "role_or_H_vector_construction_seconds": None,
        "role_or_H_note": (
            "Shared P has no separate H phase; relation-vector construction is included "
            "in QT/QE/ET search_many timings."
        ),
        "content_keys_sha256": content_keys_sha256,
        "repeats": repeats,
        "hardware": {
            "device": args.device,
            "gpu": torch.cuda.get_device_name(device),
            "cpu_threads": args.cpu_threads,
        },
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv],
        "code_sha256": checkpoint_fingerprint(Path(__file__)),
    }
    target = output / f"statistics/latency_profile_{args.arm}.json"
    write_json(target, payload)
    with (output / "runs.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "task": "R13 online phase latency profile",
                    "arm": args.arm,
                    "status": "complete",
                    "output": str(target.resolve()),
                    "command": payload["command"],
                }
            )
            + "\n"
        )
    print(json.dumps({"status": "complete", "output": str(target)}, indent=2))
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--arm", choices=tuple(ARM_PATHS), required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--query-batch-size", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=3)
    run(parser.parse_args())
