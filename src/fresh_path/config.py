"""Configuration and protocol loading for FRESH-PATH v2.1.

This module deliberately refuses legacy profiles: the v2.1 package supersedes
FRESH-PATH v2.0 / QCPATH-R1 v1.0 / v1.1 / CLEAN-R1 (EXPERIMENT_SPEC 0.1).
No CLI flag may reintroduce a historical task parent (SPEC 2.4).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROTOCOL_ID = "MMDD-S1-FRESH-PATH"
PROTOCOL_VERSION = "2.1"
SCOPE = "complete_fresh_stage1"
NAMESPACE_PREFIX = "FRESH-PATH-v2|20260920"

# Roles that must never appear as training inputs (SPEC 2.2).
FORBIDDEN_ROOT_ROLES = (
    "student_base",
    "teacher_parent",
    "historical_PCA",
    "old_train_fit",
    "old_hard32",
    "teacher_feature_store",
    "cal_fit",
    "cal_check",
)

REQUIRED_ROOT_ROLES = (
    "dataset",
    "raw_splits",
    "raw_train_GT",
    "raw_dev_GT",
    "raw_test_GT_location",
    "public_backbone",
    "encoder_contract",
    "code",
    "protocol",
)

STAGES = (
    "T_EDGE",
    "T_PATH",
    "T_QT",
    "S_SUP_C1",
    "S_KD_C1",
    "S_SUP_QE_C2",
    "S_KD_QE_C2",
    "S_KD_EONLY_C2",
    "S_KD_NATIVE_C2",
    "S_QT_SUP_C1",
    "S_QT_KD_C1",
    "S_QT_SUP_C2",
    "S_QT_KD_C2",
)

TEACHER_STAGES = ("T_EDGE", "T_PATH", "T_QT")
STUDENT_STAGES = tuple(s for s in STAGES if s.startswith("S_"))


def load_protocol(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    validate_protocol(data)
    return data


def validate_protocol(p: dict[str, Any]) -> None:
    if p.get("protocol_id") != PROTOCOL_ID:
        raise ValueError(f"wrong protocol_id: {p.get('protocol_id')!r}")
    if p.get("version") != PROTOCOL_VERSION:
        raise ValueError(f"wrong protocol version: {p.get('version')!r} (need {PROTOCOL_VERSION})")
    if p.get("scope") != SCOPE:
        raise ValueError(f"wrong scope: {p.get('scope')!r}")
    lin = p["lineage"]
    for key in (
        "external_task_checkpoint",
        "external_training_list",
        "external_teacher_logits",
        "external_learned_pooler_cache",
        "external_PCA",
    ):
        if lin[key] is not False:
            raise ValueError(f"lineage.{key} must be false for a fresh run")
    if p["training"]["max_stage_jobs_total"] != 26 or p["training"]["max_stage_jobs_per_seed"] != 13:
        raise ValueError("v2.1 budget must be 13 stages/seed and 26 total")
    if p["training"]["model_seeds"] != [13, 29]:
        raise ValueError("v2.1 fixes model seeds [13, 29]")
    if tuple(p["training"]["paths"]) != STAGES:
        raise ValueError("protocol paths list does not match the 13 v2.1 stages")
    if p["teacher"]["input_dim"] != 4096 or p["encoder"]["embedding_dim"] != 4096:
        raise ValueError("encoder/teacher dimension contract changed")
    if p["resources"]["allowed_gpu_models"] != ["RTX 4090"] or p["resources"]["gpu_count"] != 2:
        raise ValueError("v2.1 hardware contract is 2x RTX 4090")
    if p["encoder"]["query_max_rows"] is not None or p["encoder"]["query_merge_rows"]:
        raise ValueError("query rows must never be truncated or merged (F06)")
    if p["teacher"]["hard_refreshes"] != 1:
        raise ValueError("exactly one hard-example refresh is authorised")


@dataclass(frozen=True)
class Paths:
    dataset_root: Path
    backbone_dir: Path
    protocol: Path
    work_dir: Path

    @property
    def seed_dir(self) -> Path:
        return self.work_dir

    def stage_dir(self, seed: int, stage: str) -> Path:
        if stage.startswith("T_"):
            return self.work_dir / f"seed{seed}" / "teacher" / stage[2:].lower()
        return self.work_dir / f"seed{seed}" / "student" / stage[2:].lower()

    def eval_dir(self, seed: int, kind: str) -> Path:
        return self.work_dir / f"seed{seed}" / "eval" / kind


def resolve_paths(
    dataset_root: str | Path,
    backbone_dir: str | Path,
    protocol: str | Path,
    work_dir: str | Path,
) -> Paths:
    ds = Path(dataset_root).resolve()
    bb = Path(backbone_dir).resolve()
    pr = Path(protocol).resolve()
    wk = Path(work_dir).resolve()
    for label, path in (("dataset-root", ds), ("backbone-dir", bb), ("protocol", pr)):
        if not path.exists():
            raise FileNotFoundError(f"{label} does not exist: {path}")
    wk.mkdir(parents=True, exist_ok=True)
    return Paths(dataset_root=ds, backbone_dir=bb, protocol=pr, work_dir=wk)


def namespace(*parts: object) -> str:
    """SPEC 10.3: a namespace string hashed once into a local RNG seed."""
    return "|".join([NAMESPACE_PREFIX, *(str(x) for x in parts)])
