"""``continue_teacher_on_student_pool``: T_B records mined from a Student's pools and the acceptance summary."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch

from continue_teacher_on_student_pool import acceptance_summary, mine_records
from mmdd_stage1.labels import Labels
from mmdd_stage1.metrics import evaluate_matrix
from mmdd_stage1.models import NativeStudent
from mmdd_stage1.retrieval import build_pools


class TinyZStore:
    def __init__(self, vectors: dict[str, torch.Tensor]):
        self.ids = list(vectors)
        self.index = {object_id: i for i, object_id in enumerate(self.ids)}
        self.z = torch.stack([vectors[object_id] for object_id in self.ids]).float()
        self.dim = self.z.shape[1]

    def vector(self, object_id: str) -> torch.Tensor:
        return self.z[self.index[object_id]]

    def rows(self, object_ids) -> torch.Tensor:
        return self.z[[self.index[object_id] for object_id in object_ids]]


class TinyRowStore:
    def __init__(self, rows: dict[str, np.ndarray]):
        self.rows = rows

    def get(self, object_id: str) -> np.ndarray:
        return self.rows[object_id]


def tiny_world():
    generator = torch.Generator().manual_seed(20261002)
    targets = [f"t{i:03d}" for i in range(160)]
    text = [f"et{i:03d}" for i in range(140)]
    image = [f"ei{i:03d}" for i in range(140)]
    ids = ["q0", "q1", *targets, *text, *image]
    matrix = torch.nn.functional.normalize(torch.randn(len(ids), 4, generator=generator), dim=1)
    z_store = TinyZStore(dict(zip(ids, matrix)))
    row_store = TinyRowStore({q: np.stack([z_store.vector("et000").numpy(), z_store.vector("ei000").numpy()])
                              for q in ("q0", "q1")})
    entry = {"G": ["t000", "t001"], "W": {"t000": ["et000", "ei000"]}, "Qpos": {"text": ["et000"], "image": ["ei000"]}}
    labels = Labels(
        queries={"q0": entry, "q1": entry}, epos={"et000": ["t000"], "ei000": ["t000"]}, legal_targets=targets,
        edge_anchors=[], canonical_map={e: e for e in [*text, *image]},
        modality={e: "text" for e in text} | {e: "image" for e in image},
        content_hash={e: hashlib.sha256(e.encode()).hexdigest() for e in [*text, *image]},
        canonical_text=text, canonical_image=image,
    )
    return z_store, row_store, labels


def test_mined_records_follow_the_tb_schema_on_the_student_pool(tmp_path: Path):
    z_store, row_store, labels = tiny_world()
    student = NativeStudent(torch.eye(4), torch.zeros(4), dim=4)
    with torch.no_grad():  # a Student that is not the identity, so the pool differs from Raw
        student.R["QT"].add_(0.3 * torch.randn(4, 4, generator=torch.Generator().manual_seed(1)))
    records, summary = mine_records(z_store, row_store, labels, student, ["q0", "q1"], 13, tmp_path / "idx", device="cpu")
    pools = build_pools(z_store, row_store, ["q0", "q1"], labels, "train", student=student, generator_id="student_c2",
                        hnsw_seed=13, device="cpu", training_exact=True)

    assert [row["query_id"] for row in records] == ["q0", "q1"]
    assert (tmp_path / "idx" / "targets.hnsw").exists()
    for row in records:
        pool = pools[row["query_id"]]
        assert set(row) == {"query_id", "targets", "positives", "natural_bags", "support_records", "qet_lists"}
        assert row["positives"] == ["t000", "t001"]
        assert row["targets"] == sorted(set(row["targets"]), key=lambda t: t.encode())
        # C^B = U | D150 | G | U32 on the Student's pool, with the Student's retained bags.
        assert set(row["targets"]) >= set(pool.U) | {t for t, _ in pool.D150} | {"t000", "t001"}
        assert set(row["natural_bags"]) == set(row["targets"])
        assert all(row["natural_bags"][t] == pool.retained_paths.get(t, []) for t in row["targets"])
        assert row["support_records"] and all(r["target_id"] == "t000" for r in row["support_records"])
    # q0 and q1 have different frozen vectors, so their Student pools and records differ.
    assert records[0]["targets"] != records[1]["targets"]
    assert summary["queries"] == 2 and 0.0 <= summary["C150_target_coverage"] <= 1.0
    assert summary["mean_targets"] >= 150 and summary["mean_support_records"] > 0


def test_acceptance_summary_reports_each_point_against_the_student(tmp_path: Path):
    z_store, row_store, labels = tiny_world()
    student = NativeStudent(torch.eye(4), torch.zeros(4), dim=4)
    records, _ = mine_records(z_store, row_store, labels, student, ["q0", "q1"], 13, None, device="cpu")
    pools = build_pools(z_store, row_store, ["q0", "q1"], labels, "dev", student=student, generator_id="s",
                        hnsw_seed=13, device="cpu")
    gt = {q: {"G": ["t000", "t001"], "kind": "implicit", "source_group": "g0", "W": {}} for q in ("q0", "q1")}
    gold_first = {q: {"target_ids": ["t000", "t001", *[t for t in pools[q].C150 if t not in ("t000", "t001")]]}
                  for q in pools}
    gold_last = {q: {"target_ids": [t for t in pools[q].C150 if t not in ("t000", "t001")] + ["t000", "t001"]}
                 for q in pools}
    matrix = {"init": {v: gold_last for v in ("f0", "Real", "Swap")},
              "half": {v: gold_last for v in ("f0", "Real", "Swap")},
              "end": {v: gold_first for v in ("f0", "Real", "Swap")}}
    metrics = evaluate_matrix(pools, matrix, gt, tmp_path / "m")
    summary = acceptance_summary(metrics, gt, pools, ("init", "half", "end"))

    assert summary["teacher"]["end"]["Real"]["overall"] == 1.0
    assert summary["teacher"]["init"]["Real"]["overall"] == 0.0
    assert summary["teacher"]["end"]["Real"]["explicit"] is None  # no explicit queries in this split
    assert summary["candidate"]["overall"]["queries"] == 2
    direct = summary["candidate"]["overall"]["Direct_ANN_R10"]
    contrasts = summary["contrasts"]
    assert abs(contrasts["end.Real_minus_student_Direct_ANN_R10"]["mean_delta_pp"] - 100 * (1.0 - direct)) < 1e-6
    assert contrasts["end.Real_minus_init.Real"]["mean_delta_pp"] == 100.0
    assert contrasts["half.f0_minus_init.f0"]["mean_delta_pp"] == 0.0
    assert records  # the mined records and the evaluated pools come from the same Student
