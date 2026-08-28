from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import probe_stage1_pca_dimensions
from mmdd_stage1.models import ProjectedIdentityStudentJoinabilityModel
from mmdd_stage1.pca import compute_pca_spectrum


def test_pca_spectrum_and_projected_identity_model_preserve_the_prescribed_geometry():
    embeddings = torch.tensor(
        [
            [-4.0, -0.2, 0.0],
            [-2.0, 0.1, 0.0],
            [2.0, -0.1, 0.0],
            [4.0, 0.2, 0.0],
        ]
    )
    projection, _mean, eigenvalues, explained = compute_pca_spectrum(
        embeddings,
        2,
        device=torch.device("cpu"),
        batch_size=2,
    )

    assert eigenvalues.tolist() == sorted(eigenvalues.tolist(), reverse=True)
    assert explained[0] > 0.99
    assert explained[-1] == pytest.approx(1.0)
    model = ProjectedIdentityStudentJoinabilityModel(projection)
    source = torch.tensor([1.0, 2.0, 3.0])
    destination = torch.tensor([-1.0, 0.5, 4.0])
    expected = torch.dot(projection @ source, projection @ destination)
    actual = torch.dot(
        model.relation_query(source, "table", "table"),
        model.index_vector(destination, "table"),
    )
    torch.testing.assert_close(actual, expected)


def test_zero_training_pca_dimension_probe_writes_resumable_results(tmp_path):
    features_path = tmp_path / "features.pt"
    objects = {
        "q": {"object_type": "table", "embedding": torch.tensor([0.0, 1.0, 0.0])},
        "positive": {
            "object_type": "table",
            "embedding": torch.tensor([0.0, 1.0, 0.0]),
        },
        "opposite": {
            "object_type": "table",
            "embedding": torch.tensor([0.0, -1.0, 0.0]),
        },
        "x1": {"object_type": "table", "embedding": torch.tensor([4.0, 0.0, 0.0])},
        "x2": {"object_type": "table", "embedding": torch.tensor([-4.0, 0.0, 0.0])},
        "z1": {"object_type": "table", "embedding": torch.tensor([0.0, 0.0, 0.1])},
        "z2": {"object_type": "table", "embedding": torch.tensor([0.0, 0.0, -0.1])},
    }
    torch.save({"objects": objects}, features_path)
    corpus_path = tmp_path / "corpus.jsonl"
    corpus_ids = ("x1", "x2", "opposite", "positive", "z1", "z2")
    corpus_path.write_text(
        "".join(
            json.dumps({"object_id": object_id}) + "\n" for object_id in corpus_ids
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
                    {"target_id": "opposite", "evidence_ids": []},
                ],
                "split": "dev",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "probe"
    args = argparse.Namespace(
        features=str(features_path),
        corpus=str(corpus_path),
        dev_data=[str(dev_path)],
        output_dir=str(output_dir),
        raw_index=None,
        dimensions=[1, 2],
        raw_fraction_threshold=0.9,
        device="cpu",
        batch_size=2,
        covariance_batch_size=2,
        feature_cache_size=8,
        hnsw_m=8,
        ef_construction=50,
        ef_search=50,
        direct_k=10,
    )

    first = probe_stage1_pca_dimensions.run(args)
    second = probe_stage1_pca_dimensions.run(args)

    assert first["training_steps"] == 0
    assert first["projection"].startswith("P = U_d^T")
    assert first["raw_direct"]["recall@10"] == 1.0
    assert first["dimensions"][1]["direct"]["recall@10"] == 1.0
    assert (
        first["dimensions"][0]["explained_variance_ratio"]
        < first["dimensions"][1]["explained_variance_ratio"]
    )
    assert second["dimensions"] == first["dimensions"]
    assert (output_dir / "pca_spectrum.pt").is_file()
    assert (output_dir / "variance_curve.csv").is_file()
    assert (output_dir / "dimension_probe.csv").is_file()
    assert (output_dir / "pca_dimension_ceiling.png").is_file()
    assert (output_dir / "summary.json").is_file()
