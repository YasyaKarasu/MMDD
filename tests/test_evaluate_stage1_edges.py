from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pytest
import torch

from evaluate_stage1_edges import run


def test_raw_edge_evaluator_writes_ranking_and_confidence_metrics(
    tmp_path: Path,
) -> None:
    features = tmp_path / "features.pt"
    torch.save(
        {
            "objects": {
                "q": {
                    "object_type": "table",
                    "embedding": torch.tensor([2.0, 0.0]),
                },
                "positive": {
                    "object_type": "text",
                    "embedding": torch.tensor([3.0, 0.0]),
                },
                "negative": {
                    "object_type": "text",
                    "embedding": torch.tensor([-1.0, 0.0]),
                },
            }
        },
        features,
    )
    edges = tmp_path / "edges.jsonl"
    edges.write_text(
        json.dumps(
            {
                "query_id": "q",
                "positive_id": "positive",
                "positive_ids": ["positive"],
                "candidate_ids": ["negative", "positive"],
                "confirmed_labels": [0, 1],
                "source_type": "table",
                "destination_type": "text",
                "split": "dev",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "metrics.json"

    payload = run(
        argparse.Namespace(
            model_kind="raw",
            checkpoint=None,
            features=str(features),
            teacher_features=[],
            edge_data=str(edges),
            split="dev",
            output=str(output),
            device="cpu",
            batch_size=2,
            feature_cache_size=8,
            student_score_space="raw_logit",
            teacher_amp="off",
            recall_ks=(1, 2),
            reliability_bins=2,
        )
    )

    metrics = payload["metrics"]["overall"]
    assert metrics["ranking"] == {
        "recall@1": pytest.approx(1.0),
        "recall@2": pytest.approx(1.0),
    }
    assert metrics["confirmed_quality"]["auroc"] == pytest.approx(1.0)
    assert metrics["confirmed_quality"]["auprc"] == pytest.approx(1.0)
    assert metrics["confirmed_quality"]["brier"] == pytest.approx(
        (1.0 - 1.0 / (1.0 + math.exp(-1.0))) ** 2
    )
    assert payload["model"]["ranking_score_space"] == (
        "raw_embedding_cosine_similarity"
    )
    assert json.loads(output.read_text(encoding="utf-8")) == payload


def test_edge_evaluator_requires_checkpoint_for_learned_models(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="requires --checkpoint"):
        run(
            argparse.Namespace(
                model_kind="student",
                checkpoint=None,
                features=str(tmp_path / "missing-features"),
                teacher_features=[],
                edge_data=str(tmp_path / "missing-edges"),
                split="all",
                output=str(tmp_path / "metrics.json"),
                device="cpu",
                batch_size=2,
                feature_cache_size=8,
                student_score_space="raw_logit",
                teacher_amp="off",
                recall_ks=(1,),
                reliability_bins=2,
            )
        )
