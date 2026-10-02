"""Auditable materialization for pools, training records, and model artifacts."""
from __future__ import annotations

import gzip
import json
import os
from collections import defaultdict
from contextlib import ExitStack
from pathlib import Path
from typing import Mapping, Sequence

import torch

from . import SCHEMA_VERSION
from .data import json_identity, sha256_file, utf8_sorted, write_json, write_jsonl_gz
from .labels import Labels
from .retrieval import PoolRecord


def pool_identity(pool: PoolRecord) -> str:
    return json_identity(
        {
            "query_id": pool.query_id,
            "generator": pool.generator_id,
            "C150": pool.C150,
            "retained_bags": {t: pool.retained_paths.get(t, []) for t in pool.C150},
            "object_vector_hash": pool.object_vector_hash,
            "index_hash": pool.index_hash,
        }
    )


def save_pool_bundle(
    directory: Path,
    pools: Mapping[str, PoolRecord],
    labels: Labels,
    *,
    seed: int,
    generator: str,
) -> dict[str, str]:
    directory.mkdir(parents=True, exist_ok=True)
    files = {
        "pools": directory / "pools.jsonl.gz",
        "prepaths": directory / "prepaths.jsonl.gz",
        "first_hop": directory / "first_hop.jsonl.gz",
        "second_hop": directory / "second_hop.jsonl.gz",
        "direct_exact": directory / "direct_exact.jsonl.gz",
        "matched_direct": directory / "matched_direct.jsonl.gz",
    }
    temporary = {
        name: path.with_name(f".{path.name}.tmp.{os.getpid()}")
        for name, path in files.items()
    }
    with ExitStack() as stack:
        handles = {
            name: stack.enter_context(gzip.open(path, "wt", encoding="utf-8"))
            for name, path in temporary.items()
        }

        def emit(name: str, row: dict) -> None:
            handles[name].write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")

        for query_id in utf8_sorted(pools):
            pool = pools[query_id]
            d100 = [target for target, _ in pool.direct]
            evidence_targets = utf8_sorted(pool.pre_paths)
            emit("pools", {
                "schema_version": SCHEMA_VERSION,
                "query_id": query_id,
                "split": pool.split,
                "seed": seed,
                "generator": generator,
                "pool_id": pool_identity(pool),
                "D100_ANN": d100,
                "D100_exact": pool.direct_exact[:100],
                "E": evidence_targets,
                "U": pool.U,
                "C150": pool.C150,
                "D150": [target for target, _ in pool.D150],
                "MatchedDirectC": [target for target, _ in pool.MatchedDirectC],
                "MatchedDirectU": [target for target, _ in pool.MatchedDirectU],
                "retained_bags": pool.retained_paths,
                "all_U_QT_scores": pool.qt_scores_all_U,
                "D1_scores": pool.d1_scores,
                "QT_ranks": pool.qt_ranks,
                "D1_ranks": pool.d1_ranks,
                "admission_scores": pool.admission_scores,
                "object_vector_hash": pool.object_vector_hash,
                "index_hash": pool.index_hash,
                "score_space": pool.score_space,
                "ann_exact_overlap": pool.ann_exact_overlap,
            })
            first_rank = {
                evidence: rank
                for modality in ("text", "image")
                for rank, (evidence, _score) in enumerate(pool.first_hop.get(modality, ()), 1)
            }
            second_by_evidence: dict[str, list[tuple[str, float]]] = defaultdict(list)
            for target_id, paths in pool.pre_paths.items():
                for path in paths:
                    second_by_evidence[path.evidence_id].append((target_id, path.second_score))
            second_rank = {
                (evidence_id, target_id): rank
                for evidence_id, values in second_by_evidence.items()
                for rank, (target_id, _score) in enumerate(
                    sorted(values, key=lambda item: (-item[1], item[0].encode("utf-8"))), 1
                )
            }
            for modality in ("text", "image"):
                for rank, (evidence_id, score) in enumerate(pool.first_hop.get(modality, ()), 1):
                    emit("first_hop", {
                        "schema_version": SCHEMA_VERSION,
                        "query_id": query_id,
                        "split": pool.split,
                        "seed": seed,
                        "generator": generator,
                        "modality": modality,
                        "evidence_id": evidence_id,
                        "rank": rank,
                        "score": score,
                    })
            for target_id in utf8_sorted(pool.pre_paths):
                retained = set(pool.retained_paths.get(target_id, ()))
                for path in pool.pre_paths[target_id]:
                    path_score = path.raw_path_score
                    trace = next(
                        (item for item in pool.d1_trace.get(target_id, ())
                         if item["evidence_id"] == path.evidence_id),
                        None,
                    )
                    emit("prepaths", {
                        "schema_version": SCHEMA_VERSION,
                        "query_id": query_id,
                        "split": pool.split,
                        "seed": seed,
                        "generator": generator,
                        "target_id": target_id,
                        "evidence_id": path.evidence_id,
                        "modality": path.modality,
                        "first_rank": first_rank[path.evidence_id],
                        "second_rank": second_rank[(path.evidence_id, target_id)],
                        "first_raw_score": path.first_score,
                        "second_raw_score": path.second_score,
                        "path_raw_score": path_score,
                        "content_hash": labels.content_hash[path.evidence_id],
                        "retained": path.evidence_id in retained,
                        "drop_reason": None if path.evidence_id in retained else "not_selected_by_D1",
                        "D1_target_score": pool.d1_scores.get(target_id, 0.0),
                        "row_support": trace["row_support_mean"] if trace is not None else None,
                        "D1_marginal_gain": trace["marginal_gain"] if trace is not None else None,
                        "D1_selected_step": trace["selected_step"] if trace is not None else None,
                    })
                    emit("second_hop", {
                        "schema_version": SCHEMA_VERSION,
                        "query_id": query_id,
                        "split": pool.split,
                        "seed": seed,
                        "generator": generator,
                        "evidence_id": path.evidence_id,
                        "target_id": target_id,
                        "rank": second_rank[(path.evidence_id, target_id)],
                        "score": path.second_score,
                    })
            emit("direct_exact", {
                "schema_version": SCHEMA_VERSION,
                "query_id": query_id,
                "split": pool.split,
                "seed": seed,
                "generator": generator,
                "target_ids": pool.direct_exact,
            })
            emit("matched_direct", {
                "schema_version": SCHEMA_VERSION,
                "query_id": query_id,
                "split": pool.split,
                "seed": seed,
                "generator": generator,
                "D150": pool.D150,
                "MatchedDirectC": pool.MatchedDirectC,
                "MatchedDirectU": pool.MatchedDirectU,
            })
    for name, path in files.items():
        temporary[name].replace(path)
    internal = directory / "pool_records.pt"
    torch.save(dict(pools), internal)
    identities = {name: sha256_file(path) for name, path in files.items()}
    identities["internal"] = sha256_file(internal)
    write_json(directory / "POOL_MANIFEST.json", identities)
    return identities


def load_pool_bundle(directory: Path) -> dict[str, PoolRecord]:
    manifest = json.loads((directory / "POOL_MANIFEST.json").read_text(encoding="utf-8"))
    internal = directory / "pool_records.pt"
    if sha256_file(internal) != manifest["internal"]:
        raise ValueError("pool internal artifact hash mismatch")
    return torch.load(internal, map_location="cpu", weights_only=False)


def save_training_records(path: Path, rows: Sequence[dict]) -> str:
    write_jsonl_gz(path, ({"schema_version": SCHEMA_VERSION, **row} for row in rows))
    return sha256_file(path)
