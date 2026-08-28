from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import probe_stage1_identity
from mmdd_stage1.data import load_target_examples
from mmdd_stage1.evaluation import (
    evaluate_direct_retrieval,
    evaluate_student_retrieval,
)
from mmdd_stage1.features import FeatureStore
from mmdd_stage1.retrieval import (
    RawEmbeddingANNIndices,
    build_raw_embedding_indices,
    checkpoint_fingerprint,
    load_corpus_ids,
)


def test_identity_probe_matches_raw_retrieval_through_student_index(tmp_path):
    features_path = tmp_path / "features.pt"
    torch.save(
        {
            "objects": {
                "q": {"object_type": "table", "embedding": torch.tensor([1.0, 0.0])},
                "positive": {
                    "object_type": "table",
                    "embedding": torch.tensor([2.0, 0.0]),
                },
                "negative": {
                    "object_type": "table",
                    "embedding": torch.tensor([0.0, 1.0]),
                },
            }
        },
        features_path,
    )
    corpus_path = tmp_path / "corpus.jsonl"
    corpus_path.write_text(
        "".join(
            json.dumps({"object_id": object_id}) + "\n"
            for object_id in ("positive", "negative")
        ),
        encoding="utf-8",
    )
    dev_path = tmp_path / "targets.jsonl"
    dev_path.write_text(
        json.dumps(
            {
                "query_id": "q",
                "direct_positive_target_id": "positive",
                "evidence_positive_target_id": "positive",
                "positive_target_ids": ["positive"],
                "candidates": [
                    {"target_id": "positive", "evidence_ids": []},
                    {"target_id": "negative", "evidence_ids": []},
                ],
                "split": "dev",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    store = FeatureStore.from_path(features_path)
    ids_by_type = load_corpus_ids(corpus_path, store)
    raw_index = tmp_path / "raw"
    build_raw_embedding_indices(
        store,
        ids_by_type,
        raw_index,
        corpus_sha256=checkpoint_fingerprint(corpus_path),
    )
    result = probe_stage1_identity.run(
        argparse.Namespace(
            features=str(features_path),
            corpus=str(corpus_path),
            dev_data=[str(dev_path)],
            output_dir=str(tmp_path / "identity"),
            raw_index=str(raw_index),
            device="cpu",
            batch_size=2,
            feature_cache_size=8,
            hnsw_m=16,
            ef_construction=100,
            ef_search=100,
            direct_k=100,
            evidence_k=50,
            targets_per_evidence=50,
            evidence_aggregation="logsumexp",
            evidence_top_k=4,
            rrf_k=60,
        )
    )

    assert result["training_steps"] == 0
    assert result["identity"]["direct"]["recall@10"] == 1.0
    assert result["raw_embedding"]["direct"]["recall@10"] == 1.0
    assert result["direct_recall@10_delta"] == pytest.approx(0.0)

    raw_indices = RawEmbeddingANNIndices(
        store,
        raw_index,
        corpus_sha256=checkpoint_fingerprint(corpus_path),
        destination_types=("table",),
    )
    examples = load_target_examples(dev_path, split="dev")
    direct_metrics = evaluate_direct_retrieval(examples, raw_indices)
    direct_metrics.pop("queries")
    assert direct_metrics == evaluate_student_retrieval(examples, raw_indices)["direct"]
