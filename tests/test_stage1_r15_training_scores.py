from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from audit_stage1_r15_training_scores import audit_batch, lse_responsibilities
from mmdd_stage1.data import TargetCandidate, TargetExample
from mmdd_stage1.features import FeatureStore, ObjectFeatures
from mmdd_stage1.models import StudentJoinabilityModel
from mmdd_stage1.objectives import PathAggregator


def test_responsibility_uses_actual_temperature_and_all_paths():
    values = [0.0, 0.0, 0.0, 0.0, 2.0]
    masses = lse_responsibilities(values, temperature=2.0)
    assert len(masses) == 5
    assert sum(masses) == pytest.approx(1.0)
    assert masses[-1] / masses[0] == pytest.approx(torch.e)


def test_training_score_audit_preserves_masks_and_does_not_truncate_lse():
    model = StudentJoinabilityModel(4, 4, initialization="identity")
    objects = {name: ObjectFeatures(name, "table", torch.ones(4)) for name in ("q", "t1", "t2", "t3")}
    for index in range(5):
        objects[f"e{index}"] = ObjectFeatures(f"e{index}", "text" if index % 2 else "image", torch.ones(4) * index / 10)
    store = FeatureStore(objects)
    batch = [TargetExample(
        "q", (TargetCandidate("t1", tuple(f"e{i}" for i in range(5))),
              TargetCandidate("t2", ("e1",)), TargetCandidate("t3", ())),
        direct_positive_index=0, evidence_positive_index=0,
        positive_target_ids=("t1", "t3"), positive_evidence_by_target={"t1": ("e4",)},
    )]
    rows, summary = audit_batch(model, batch, store, torch.device("cpu"), PathAggregator("logsumexp", 4, temperature=2.0))
    assert summary["passed"]
    assert rows[0]["training_paths_used_by_lse"] == 5
    assert len(rows[0]["paths"]) == 5
    assert rows[0]["known_witness_responsibility_mass"] > 0
    assert rows[0]["evidence_positive_mask"] is True
    assert rows[1]["evidence_positive_mask"] is False
    assert rows[2]["direct_positive_mask"] is True
    assert rows[2]["evidence_candidate_mask"] is False
    assert rows[2]["evidence_positive_mask"] is False
    assert rows[2]["known_witness_responsibility_mass"] is None
