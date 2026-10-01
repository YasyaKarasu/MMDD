from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pytest
import torch

import mmdd_stage2.reader_cache as reader_cache_module
import train_stage2
from mmdd_stage2.data import Stage2ObjectIndex
from mmdd_stage2.oracle import (
    ORACLE_EVIDENCE_POLICY,
    OracleColumnExample,
    OracleDataError,
    load_oracle_column_data,
    select_oracle_evidence,
)
from mmdd_stage2.reader_cache import (
    build_reader_cache,
    load_reader_cache,
    train_cached_scorer,
)
from mmdd_stage2.r12_column import (
    R12ColumnExample,
    build_r12_reader_cache,
    evaluate_r12_scores,
    load_r12_reader_cache,
    score_r12_records,
    train_r12_candidate_scorer,
)
from mmdd_stage2.verifier import EvidenceBundle


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )


def _dataset(
    root: Path,
    *,
    split: str = "train",
    query_id: str = "q1",
    target_id: str = "t1",
    evidence: list[dict] | None = None,
) -> Path:
    evidence = evidence or [
        {"asset_id": "text_b", "asset_type": "text", "content": "b"},
        {"asset_id": "text_a", "asset_type": "text", "content": "a"},
    ]
    artifact_paths = {
        "qrels": "qrels.jsonl",
        "query_tables": "query_tables/part.jsonl",
        "data_lake_tables": "data_lake_tables/part.jsonl",
        "bridge_assets": "bridge_assets/part.jsonl",
        "evidence_recoveries": "evidence_recoveries/part.jsonl",
    }
    root.mkdir(parents=True)
    (root / "dataset_manifest.json").write_text(
        json.dumps(
            {
                "single_files": {"qrels": artifact_paths["qrels"]},
                "artifacts": {
                    name: {"path": path}
                    for name, path in artifact_paths.items()
                    if name != "qrels"
                },
            }
        ),
        encoding="utf-8",
    )
    _write_jsonl(
        root / artifact_paths["qrels"],
        [
            {
                "query_table_id": query_id,
                "target_table_id": target_id,
                "split": split,
                "reason": "model_recoverable_join_column",
                "source_table_id": f"source_{split}_{query_id}",
                "chain_id": f"chain_{split}_{query_id}",
                "join_attribute": {"source_column_index": 7},
            }
        ],
    )
    _write_jsonl(
        root / artifact_paths["query_tables"],
        [
            {
                "table_id": query_id,
                "split": split,
                "columns": [{"column_index": 0, "column_name": "entity"}],
                "rows": [],
            }
        ],
    )
    _write_jsonl(
        root / artifact_paths["data_lake_tables"],
        [
            {
                "table_id": target_id,
                "split": split,
                "columns": [
                    {
                        "column_index": 2,
                        "source_column_index": 9,
                        "column_name": "other",
                    },
                    {
                        "column_index": 5,
                        "source_column_index": 7,
                        "column_name": "gold",
                    },
                ],
                "rows": [],
            }
        ],
    )
    _write_jsonl(root / artifact_paths["bridge_assets"], evidence)
    _write_jsonl(
        root / artifact_paths["evidence_recoveries"],
        [
            {
                "query_table_id": query_id,
                "target_table_id": target_id,
                "split": split,
                "evidence": {
                    "asset_id": item["asset_id"],
                    "asset_type": item["asset_type"],
                },
            }
            for item in evidence
        ],
    )
    return root


def test_select_oracle_evidence_is_deduplicated_stable_and_multimodal():
    evidence = {
        "text_z": {"asset_type": "text"},
        "text_a": {"asset_type": "text"},
        "image_z": {"asset_type": "image"},
        "image_a": {"asset_type": "image"},
        "text_b": {"asset_type": "text"},
    }

    selected = select_oracle_evidence(
        ["text_z", "image_z", "text_a", "image_a", "text_b", "text_z"],
        evidence,
        top_k=4,
    )

    assert selected == ("text_a", "image_a", "image_z", "text_b")
    assert {evidence[item]["asset_type"] for item in selected} == {"text", "image"}


def test_oracle_loader_filters_split_matches_recovery_and_maps_local_column(tmp_path):
    train = _dataset(tmp_path / "lake_train", split="train", query_id="q_train")
    examples, objects, audit = load_oracle_column_data(
        [train], splits=("train",), strict=True
    )

    assert len(examples) == 1
    assert examples[0].split == "train"
    assert examples[0].gold_source_column == 7
    assert examples[0].gold_local_column == 5
    assert examples[0].gold_column_position == 1
    assert examples[0].positive_bundle.evidence_ids == ("text_a", "text_b")
    assert set(objects.queries) == {"q_train"}
    assert audit["datasets"]["lake_train"]["splits"]["train"]["usable_examples"] == 1


def test_oracle_loader_accepts_multiple_targets_for_one_query(tmp_path):
    root = _dataset(tmp_path / "lake", query_id="q", target_id="t1")
    qrels_path = root / "qrels.jsonl"
    qrels = [json.loads(line) for line in qrels_path.read_text().splitlines()]
    qrels.append(
        {
            **qrels[0],
            "target_table_id": "t2",
            "chain_id": "chain_t2",
            "join_attribute": {"source_column_index": 8},
        }
    )
    _write_jsonl(qrels_path, qrels)

    targets_path = root / "data_lake_tables/part.jsonl"
    targets = [json.loads(line) for line in targets_path.read_text().splitlines()]
    targets.append(
        {
            **targets[0],
            "table_id": "t2",
            "columns": [
                {
                    "column_index": 4,
                    "source_column_index": 8,
                    "column_name": "second_gold",
                }
            ],
        }
    )
    _write_jsonl(targets_path, targets)

    recoveries_path = root / "evidence_recoveries/part.jsonl"
    recoveries = [json.loads(line) for line in recoveries_path.read_text().splitlines()]
    recoveries.append({**recoveries[0], "target_table_id": "t2"})
    _write_jsonl(recoveries_path, recoveries)

    examples, objects, audit = load_oracle_column_data(
        [root], splits=("train",), strict=True
    )

    assert [(example.query_id, example.target_id) for example in examples] == [
        ("q", "t1"),
        ("q", "t2"),
    ]
    assert [example.gold_source_column for example in examples] == [7, 8]
    assert set(objects.targets) == {"t1", "t2"}
    assert audit["datasets"]["lake"]["duplicate_qrel_pairs"] == []


def test_oracle_loader_rejects_recovery_target_mismatch(tmp_path):
    root = _dataset(tmp_path / "lake")
    recovery_path = root / "evidence_recoveries/part.jsonl"
    records = [json.loads(line) for line in recovery_path.read_text().splitlines()]
    for record in records:
        record["target_table_id"] = "wrong_target"
    _write_jsonl(recovery_path, records)

    with pytest.raises(OracleDataError, match="recovery") as error:
        load_oracle_column_data([root], strict=True)

    assert error.value.audit["datasets"]["lake"]["missing"]["recovery"] == [
        "q1->t1"
    ]


def test_oracle_loader_audits_missing_image_file(tmp_path):
    root = _dataset(
        tmp_path / "lake",
        evidence=[
            {
                "asset_id": "image_a",
                "asset_type": "image",
                "local_path": str(tmp_path / "missing.jpg"),
            }
        ],
    )

    with pytest.raises(OracleDataError, match="image_file") as error:
        load_oracle_column_data([root], strict=True)

    assert error.value.audit["datasets"]["lake"]["missing"]["image_file"] == [
        "image_a"
    ]


def test_oracle_loader_rejects_cross_dataset_id_conflicts(tmp_path):
    left = _dataset(tmp_path / "left", query_id="shared", target_id="shared_target")
    right = _dataset(tmp_path / "right", query_id="shared", target_id="shared_target")

    with pytest.raises(OracleDataError, match="ID conflicts") as error:
        load_oracle_column_data([left, right], splits=("train",), strict=True)

    kinds = {item["kind"] for item in error.value.audit["id_conflicts"]}
    assert kinds == {"query", "target", "evidence"}


class _FrozenBackend:
    hidden_dim = 3

    def __init__(self) -> None:
        self.model = torch.nn.Linear(1, 1).eval()
        self.model.requires_grad_(False)

    def reader_states(self, _query, target, _evidence):
        count = len(target["columns"])
        return torch.ones(count, 3), torch.full((count, 3), 2.0)


def test_reader_cache_round_trip_and_metadata_mismatch(tmp_path):
    example = OracleColumnExample(
        dataset="lake",
        dataset_root=str(tmp_path / "dataset"),
        split="train",
        query_id="q1",
        target_id="t1",
        source_table_id="source",
        chain_id="chain",
        gold_source_column=7,
        gold_local_column=5,
        gold_column_position=1,
        candidate_column_indices=(2, 5),
        positive_bundle=EvidenceBundle("t1", 0.0, ("e1",)),
        evidence_modalities=("text",),
        evidence_count_before_truncation=1,
    )
    objects = Stage2ObjectIndex(
        {"q1": {"table_id": "q1"}},
        {"t1": {"table_id": "t1", "columns": [{}, {}]}},
        {"e1": {"asset_id": "e1", "asset_type": "text"}},
    )
    cache_dir = tmp_path / "cache"
    manifest = build_reader_cache(
        _FrozenBackend(),
        [example],
        objects,
        cache_dir,
        model_path=tmp_path / "model",
        model_dtype="bf16",
        top_k_evidence=4,
        evidence_policy=ORACLE_EVIDENCE_POLICY,
        shard_size=1,
    )
    records, fingerprint = load_reader_cache([cache_dir])

    assert manifest["complete"] is True
    assert fingerprint
    assert records[0]["open_states"].shape == (2, 3)
    assert records[0]["gold_column_position"] == 1

    with pytest.raises(ValueError, match="metadata mismatch"):
        build_reader_cache(
            _FrozenBackend(),
            [example],
            objects,
            cache_dir,
            model_path=tmp_path / "different_model",
            model_dtype="bf16",
            top_k_evidence=4,
            evidence_policy=ORACLE_EVIDENCE_POLICY,
            shard_size=1,
        )


def test_reader_cache_allows_multiple_gold_targets_for_one_query(tmp_path):
    examples = [
        OracleColumnExample(
            dataset="lake",
            dataset_root=str(tmp_path / "dataset"),
            split="train",
            query_id="q1",
            target_id=target_id,
            source_table_id="source",
            chain_id=f"chain-{target_id}",
            gold_source_column=source_column,
            gold_local_column=source_column,
            gold_column_position=0,
            candidate_column_indices=(source_column,),
            positive_bundle=EvidenceBundle(target_id, 0.0, (evidence_id,)),
            evidence_modalities=("text",),
            evidence_count_before_truncation=1,
        )
        for target_id, evidence_id, source_column in (
            ("t1", "e1", 1),
            ("t2", "e2", 2),
        )
    ]
    objects = Stage2ObjectIndex(
        {"q1": {"table_id": "q1"}},
        {
            "t1": {"table_id": "t1", "columns": [{}]},
            "t2": {"table_id": "t2", "columns": [{}]},
        },
        {
            "e1": {"asset_id": "e1", "asset_type": "text"},
            "e2": {"asset_id": "e2", "asset_type": "text"},
        },
    )
    cache_dir = tmp_path / "cache"
    build_reader_cache(
        _FrozenBackend(),
        examples,
        objects,
        cache_dir,
        model_path=tmp_path / "model",
        model_dtype="bf16",
        top_k_evidence=4,
        evidence_policy=ORACLE_EVIDENCE_POLICY,
        shard_size=1,
    )

    records, _fingerprint = load_reader_cache([cache_dir])

    assert [(record["query_id"], record["target_id"]) for record in records] == [
        ("q1", "t1"),
        ("q1", "t2"),
    ]


def test_r12_reader_cache_physically_permutes_columns_and_adds_rejection_controls(
    tmp_path,
):
    examples = [
        R12ColumnExample(
            dataset="lake",
            dataset_root=str(tmp_path / "dataset"),
            protocol_split="train_fit",
            query_id=query_id,
            target_id=target_id,
            source_table_id=source_id,
            gold_source_column=gold,
            gold_local_column=gold,
            evidence_ids=(evidence_id,),
            evidence_modalities=("text",),
        )
        for query_id, target_id, source_id, evidence_id, gold in (
            ("q1", "t1", "s1", "e1", 1),
            ("q2", "t2", "s2", "e2", 0),
        )
    ]
    objects = Stage2ObjectIndex(
        {"q1": {"table_id": "q1"}, "q2": {"table_id": "q2"}},
        {
            target_id: {
                "table_id": target_id,
                "columns": [
                    {"column_index": 0, "column_name": "left"},
                    {"column_index": 1, "column_name": "right"},
                ],
            }
            for target_id in ("t1", "t2")
        },
        {
            "e1": {"asset_id": "e1", "asset_type": "text"},
            "e2": {"asset_id": "e2", "asset_type": "text"},
        },
    )

    class Backend(_FrozenBackend):
        def __init__(self):
            super().__init__()
            self.presented = []

        def reader_states(self, query, target, _evidence):
            indices = [int(column["column_index"]) for column in target["columns"]]
            self.presented.append((query["table_id"], target["table_id"], indices))
            values = torch.tensor(indices, dtype=torch.float32).unsqueeze(1).repeat(1, 3)
            return values, values + 1

    backend = Backend()
    cache_dir = tmp_path / "r12_cache"
    manifest = build_r12_reader_cache(
        backend,
        examples,
        objects,
        cache_dir,
        model_path=tmp_path / "model",
        model_dtype="fp32",
        top_k_evidence=4,
        column_permutation_seed=13,
        shard_size=2,
    )
    records, _fingerprint = load_r12_reader_cache([cache_dir])

    assert manifest["record_count"] == 6
    assert Counter(record["example_type"] for record in records) == {
        "positive": 2,
        "wrong_target": 2,
        "no_available_column": 2,
    }
    for record in records:
        if record["example_type"] == "positive":
            assert record["candidate_column_indices"][record["gold_column_position"]] == record[
                "gold_column_index"
            ]
        elif record["example_type"] == "no_available_column":
            assert record["gold_column_index"] not in record["candidate_column_indices"]
    assert len(backend.presented) == 6
    assert sum(len(indices) == 1 for _, _, indices in backend.presented) == 2


def test_r12_cached_scorer_learns_column_selection_and_control_rejection(tmp_path):
    def record(example_type, values, gold=None):
        return {
            "dataset": "lake",
            "protocol_split": "train_fit",
            "query_id": f"q-{example_type}-{values}",
            "gold_target_id": "gold",
            "presented_target_id": "shown",
            "source_table_id": "source",
            "example_type": example_type,
            "gold_column_position": gold,
            "gold_column_index": gold,
            "candidate_column_indices": tuple(range(len(values))),
            "candidate_column_count": len(values),
            "evidence_ids": ("e",),
            "evidence_modalities": ("text",),
            "open_states": torch.tensor(values, dtype=torch.float32).unsqueeze(1),
            "close_states": torch.zeros(len(values), 1),
        }

    base = [
        record("positive", [-2.0, 2.0], 1),
        record("positive", [2.0, -2.0], 0),
        record("wrong_target", [-2.0, -2.0]),
        record("no_available_column", [-2.0]),
    ]
    train = base * 8
    scorer, summary = train_r12_candidate_scorer(
        train,
        base,
        hidden_dim=1,
        seed=13,
        epochs=10,
        learning_rate=0.1,
        weight_decay=0.0,
        output_dir=tmp_path / "scorer",
        checkpoint_metadata={"model_dir": str(tmp_path / "model")},
    )
    metrics = evaluate_r12_scores(
        score_r12_records(scorer, base),
        threshold=summary["rejection_threshold"],
    )

    assert summary["head_parameters_updated"] is True
    assert metrics["macro_decision_accuracy"] == pytest.approx(1.0)
    assert metrics["by_example_type"]["positive"]["column_accuracy_without_rejection"] == 1.0
    assert metrics["control_false_accept_rate"] == 0.0


def test_cached_scorer_selects_earlier_tied_nonfinal_epoch(tmp_path, monkeypatch):
    record = {
        "dataset": "lake",
        "open_states": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        "close_states": torch.tensor([[0.5, 0.0], [0.0, 0.5]]),
        "gold_column_position": 0,
    }
    dev_values = [(0.8, 0.5), (0.9, 0.4), (0.9, 0.4)]
    observed_states = []

    def fake_evaluate(scorer, _records, **_kwargs):
        observed_states.append(
            {name: value.detach().clone() for name, value in scorer.state_dict().items()}
        )
        accuracy, nll = dev_values[len(observed_states) - 1]
        return {
            "column_nll": nll,
            "by_dataset": {"lake": {"column_accuracy@1": accuracy}},
        }, []

    monkeypatch.setattr(reader_cache_module, "evaluate_scorer", fake_evaluate)
    monkeypatch.setattr(
        reader_cache_module,
        "save_candidate_scorer",
        lambda *_args, **_kwargs: None,
    )

    scorer, summary, _epoch_zero = train_cached_scorer(
        [record],
        [record],
        hidden_dim=2,
        seed=13,
        epochs=3,
        learning_rate=1e-2,
        weight_decay=0.0,
        output_dir=tmp_path,
        checkpoint_metadata={},
    )

    assert summary["selected_epoch"] == 2
    for name, value in scorer.state_dict().items():
        torch.testing.assert_close(value, observed_states[1][name])


def test_train_stage2_oracle_mode_does_not_require_gate(tmp_path, monkeypatch):
    events = []
    example = OracleColumnExample(
        dataset="lake",
        dataset_root="dataset",
        split="train",
        query_id="q1",
        target_id="t1",
        source_table_id="source",
        chain_id="chain",
        gold_source_column=1,
        gold_local_column=1,
        gold_column_position=0,
        candidate_column_indices=(1,),
        positive_bundle=EvidenceBundle("t1", 0.0, ("e1",)),
        evidence_modalities=("text",),
        evidence_count_before_truncation=1,
    )
    objects = Stage2ObjectIndex({}, {}, {})

    monkeypatch.setattr(
        train_stage2,
        "validate_stage2_gate",
        lambda *_args: pytest.fail("Oracle mode must not validate a Stage-1 gate"),
    )
    monkeypatch.setattr(
        train_stage2,
        "load_oracle_column_data",
        lambda *_args, **_kwargs: ([example], objects, {}),
    )

    class Backend:
        hidden_dim = 2
        device = torch.device("cpu")

        def __init__(self, *_args, **_kwargs):
            events.append("backend")

    monkeypatch.setattr(train_stage2, "QwenStage2Backend", Backend)
    monkeypatch.setattr(
        train_stage2,
        "train_candidate_scorer",
        lambda *_args, **_kwargs: events.append("train") or [],
    )
    monkeypatch.setattr(
        train_stage2,
        "save_candidate_scorer",
        lambda *_args, **_kwargs: events.append("save"),
    )
    args = argparse.Namespace(
        training_source="oracle-positive",
        dataset_root=["dataset"],
        retrieval_results=None,
        stage1_gate=None,
        output=str(tmp_path / "candidate.pt"),
        model_dir="model",
        device="cpu",
        dtype="fp32",
        top_k_evidence=4,
        max_targets=10,
        epochs=1,
        learning_rate=1e-3,
        weight_decay=1e-4,
        seed=13,
    )

    train_stage2.run(args)

    assert events == ["backend", "train", "save"]
