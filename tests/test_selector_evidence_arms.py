from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mmdd_stage2.column_data import write_json, write_jsonl
from mmdd_stage2.column_r2_models import mix_condition
from mmdd_stage2.column_r2_training import (
    NATURAL_EVIDENCE_ARM,
    NO_EVIDENCE_ARM,
    PLAIN_HEAD_ARMS,
    evidence_subset_ids,
    load_head,
    metadata_for,
    prediction_evidence,
    train_arm,
    training_condition,
)
from mmdd_stage2.natural_evidence import NATURAL_EVIDENCE_KEY
from mmdd_stage2.verifier import CandidateColumnScorer


def item(**evidence: list[str]) -> dict:
    return {"dataset": "mm", "query_id": "q1", "target_id": "t1",
            "evidence_ids": {"O-O": ["oo1", "oo2"], "O-R": ["or1", "or2"],
                             NATURAL_EVIDENCE_KEY: ["nat1", "nat2"], **evidence}}


def test_prediction_evidence_keeps_the_historical_witness_defaults():
    sample = item()
    assert prediction_evidence(NO_EVIDENCE_ARM, sample) == []
    assert prediction_evidence("OO_CONTROL", sample) == ["or1", "or2"]
    assert prediction_evidence("FLAT_MIX", sample) == ["or1", "or2"]
    assert prediction_evidence(NATURAL_EVIDENCE_ARM, sample) == ["nat1", "nat2"]


def test_prediction_evidence_returns_a_copy_of_the_stored_list():
    sample = item()
    ids = prediction_evidence(NATURAL_EVIDENCE_ARM, sample)
    ids.append("mutated")
    assert sample["evidence_ids"][NATURAL_EVIDENCE_KEY] == ["nat1", "nat2"]


def test_training_condition_matches_the_previous_arm_schedule_exactly():
    sample = item()
    assert training_condition(NO_EVIDENCE_ARM, sample, 13, 1) == ("No-E", [])
    assert training_condition("OO_CONTROL", sample, 13, 1) == ("O-O", ["oo1", "oo2"])
    assert training_condition(NATURAL_EVIDENCE_ARM, sample, 13, 4) == ("NATURAL_E", ["nat1", "nat2"])
    for epoch in range(1, 5):
        condition, ids = training_condition("PVR_BUNDLE", sample, 13, epoch)
        assert condition == mix_condition(13, epoch, "q1", "t1")
        assert condition in {"O-O", "O-R"}
        assert ids == sample["evidence_ids"][condition]


def test_natural_evidence_arm_uses_the_flat_candidate_head():
    assert NATURAL_EVIDENCE_ARM in PLAIN_HEAD_ARMS
    assert {"OO_CONTROL", "FLAT_MIX", "PRIOR"} <= PLAIN_HEAD_ARMS


def test_load_head_round_trips_a_natural_evidence_checkpoint(tmp_path: Path):
    model = CandidateColumnScorer(8, head_type="mlp")
    path = tmp_path / "selected.pt"
    torch.save({"metadata": {"arm": NATURAL_EVIDENCE_ARM, "hidden_dim": 8}, "state_dict": model.state_dict()}, path)
    loaded, meta = load_head(path)
    assert meta["arm"] == NATURAL_EVIDENCE_ARM
    assert isinstance(loaded, CandidateColumnScorer)
    assert loaded.input_dim == 16


def test_fixed_epoch_requires_a_compatible_schedule():
    r1 = output = Path("/nonexistent")
    with pytest.raises(ValueError, match="fixed_epoch requires"):
        train_arm(r1, output, NO_EVIDENCE_ARM, 13, fixed_epoch=20)  # PRIOR keeps its default patience
    with pytest.raises(ValueError, match="fixed_epoch requires"):
        train_arm(r1, output, NATURAL_EVIDENCE_ARM, 13, fixed_epoch=0, early_stopping_patience=0)
    with pytest.raises(ValueError, match="fixed_epoch requires"):
        train_arm(r1, output, NATURAL_EVIDENCE_ARM, 13, epochs=10, fixed_epoch=20, early_stopping_patience=0)


def test_fixed_epoch_schedule_passes_validation_and_then_needs_real_inputs():
    # Validation must accept the controlled-comparison schedule; anything past it is an IO error.
    with pytest.raises(Exception) as failure:
        train_arm(Path("/nonexistent"), Path("/nonexistent"), NATURAL_EVIDENCE_ARM, 13,
                  fixed_epoch=20, early_stopping_patience=0)
    assert "fixed_epoch requires" not in str(failure.value)


def test_dev_subsets_follow_the_arm_evidence_source(tmp_path: Path):
    r1 = tmp_path / "r1"
    r1.mkdir()
    write_jsonl(r1 / "COLUMN_POPULATION.dev.jsonl",
                [{"dataset": "mm", "query_id": "q1", "target_id": "t1", "candidate_column_indices": [0],
                  "gold_column_indices": [0], "source_table_id": "st"}])
    objects = {"evidence": {name: {"asset_type": "text"} for name in ("or1", "or2", "nat1", "nat2")}}
    natural = metadata_for(r1, [item()], objects, "dev", arm=NATURAL_EVIDENCE_ARM)[0]
    witness = metadata_for(r1, [item()], objects, "dev")[0]
    assert natural["evidence_count"] == 2 and natural["evidence_source"] == NATURAL_EVIDENCE_KEY
    assert witness["evidence_count"] == 2 and witness["evidence_source"] == "O-R"
    # the historical default is retained for every pre-existing arm, including the no-evidence arm
    assert evidence_subset_ids(NATURAL_EVIDENCE_ARM, item()) == ["nat1", "nat2"]
    assert evidence_subset_ids(NO_EVIDENCE_ARM, item()) == ["or1", "or2"]
    assert evidence_subset_ids("FLAT_MIX", item()) == ["or1", "or2"]


def test_natural_evidence_arm_trains_without_any_checkpoint_or_prior_shortlist(tmp_path: Path, monkeypatch):
    import mmdd_stage2.column_r2_training as training

    r1, output = tmp_path / "r1", tmp_path / "r2"
    r1.mkdir()
    inputs = [{"dataset": "d", "query_id": f"q{i}", "target_id": "t",
               "evidence_ids": {"O-O": ["oracle"], "O-R": ["r1_natural"],
                                NATURAL_EVIDENCE_KEY: ["nat1", "nat2"] if i == 0 else []}}
              for i in range(2)]
    population = [{**row, "candidate_column_indices": [0, 1, 2, 3], "gold_column_indices": [i],
                   "source_table_id": row["query_id"]} for i, row in enumerate(inputs)]
    for split in ("train", "dev"):
        write_jsonl(r1 / f"COLUMN_POPULATION.{split}.jsonl", population)
    write_json(r1 / "READER_IDENTITY.json", {})
    write_json(output / "NATURAL_TRAIN/MANIFEST.json", {})

    class FakeData:
        def __init__(self, *args):
            self.objects = {"evidence": {name: {"asset_type": "text"} for name in ("nat1", "nat2")}}
            self.loaded = {}
            self.index = SimpleNamespace(entries={})

        def get(self, item, ids, view):
            return {**item, "candidate_column_indices": [0, 1, 2, 3],
                    "open_states": torch.zeros(4, 4), "close_states": torch.zeros(4, 4)}

        def freeze_prior_inputs(self, seed):
            raise AssertionError("the natural-evidence arm must not freeze prior shortlists")

    real_load_head = training.load_head

    def guarded_load_head(path, *args, **kwargs):
        # re-loading this run's own selected checkpoint is legitimate; a historical one is not
        if "PRIOR" in str(path):
            raise AssertionError("the natural-evidence arm must not load the PRIOR checkpoint")
        return real_load_head(path, *args, **kwargs)

    monkeypatch.setattr(training, "TrainingData", FakeData)
    monkeypatch.setattr(training, "inputs_for", lambda *args, **kwargs: inputs)
    monkeypatch.setattr(training, "load_head", guarded_load_head)
    receipt = training.train_arm(r1, output, NATURAL_EVIDENCE_ARM, 13, epochs=1, fixed_epoch=1,
                                 early_stopping_patience=0)
    assert receipt["prior_sha256"] is None and receipt["prior_frozen"] is False
    assert receipt["fixed_epoch_selection"] == 1 and receipt["selected_epoch"] == 1
    assert receipt["stop_reason"] == "fixed_epoch" and receipt["actual_epochs"] == 1
    assert receipt["condition_schedule"] == "current Stage-1 retained paths, ordered, <=4"
    assert receipt["selection"] == "fixed epoch 1, dev monitoring only"
