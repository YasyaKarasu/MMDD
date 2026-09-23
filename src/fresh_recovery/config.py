from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import PROTOCOL_ID, PROTOCOL_VERSION

SCOPE = "complete_fresh_stage1_budget_aligned_native_student_and_supported_QET_path"
NAMESPACE_PREFIX = "MMDD-FRESH-RECOVERY-v3.1"
SEEDS = (13, 29)
RELATIONS = ("QT", "Q_text", "Q_image", "text_T", "image_T")


@dataclass(frozen=True)
class Paths:
    dataset_root: Path
    backbone_dir: Path
    package_dir: Path
    protocol_path: Path
    work_dir: Path

    @property
    def labels_dir(self) -> Path:
        return self.work_dir / "labels"


def load_protocol(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_protocol(payload)
    return payload


def validate_protocol(payload: dict[str, Any]) -> None:
    if payload.get("protocol_id") != PROTOCOL_ID:
        raise ValueError(f"wrong protocol_id: {payload.get('protocol_id')!r}")
    if payload.get("version") != PROTOCOL_VERSION:
        raise ValueError(f"wrong protocol version: {payload.get('version')!r}")
    if payload.get("scope") != SCOPE:
        raise ValueError(f"wrong protocol scope: {payload.get('scope')!r}")
    if payload["fresh"] != {
        "historical_task_weights": False,
        "historical_lists": False,
        "historical_PCA": False,
        "historical_teacher_logits": False,
        "learned_feature_caches": False,
        "compatible_pure_backbone_cache": True,
        "current_run_finite_DAG_parents": True,
    }:
        raise ValueError("fresh lineage contract changed")
    if payload["student"]["conditional_adapter"] is not False:
        raise ValueError("v3 must not train a Student Q-adapter")
    if payload["student"]["P_lr"] != 1e-6 or payload["student"]["R_lr"] != 1e-5:
        raise ValueError("v3 P/R learning rates changed")
    if payload["student"]["KD_weight"] != 0.3 or payload["student"]["KD_temperature"] != 1.0:
        raise ValueError("v3 KD recipe changed")
    if payload["data"]["target_rows"] != 20 or payload["data"]["query_rows"] != "all_independent":
        raise ValueError("v3 table row contract changed")
    if payload["budget"]["seeds"] != list(SEEDS) or payload["budget"]["phases_per_seed"] != 12:
        raise ValueError("v3 seed/stage budget changed")
    if payload["hardware"]["model"] != "RTX 4090" or payload["hardware"]["count"] != 2:
        raise ValueError("v3 hardware contract is exactly 2x RTX 4090")
    if payload["retrieval"]["ET_k"] != 50 or payload["retrieval"]["candidate_budget"] != 150:
        raise ValueError("v3.1 retrieval budget changed")
    if payload["retrieval"]["admission"] != "QTALL_equal_RRF_D1" or not payload["retrieval"]["QT_all_U"]:
        raise ValueError("v3.1 QTALL admission changed")
    if payload["student"]["C2"]["snapshots_epoch"] != [0.5, 1.0, 1.5, 2.0]:
        raise ValueError("v3.1 C2 snapshot contract changed")


def resolve_paths(
    dataset_root: Path,
    backbone_dir: Path,
    package_dir: Path,
    work_dir: Path,
) -> Paths:
    paths = Paths(
        dataset_root=Path(dataset_root).resolve(),
        backbone_dir=Path(backbone_dir).resolve(),
        package_dir=Path(package_dir).resolve(),
        protocol_path=(Path(package_dir) / "protocol.json").resolve(),
        work_dir=Path(work_dir).resolve(),
    )
    for role, path in (
        ("dataset", paths.dataset_root),
        ("public backbone", paths.backbone_dir),
        ("audit package", paths.package_dir),
        ("protocol", paths.protocol_path),
    ):
        if not path.exists():
            raise FileNotFoundError(f"missing {role}: {path}")
    load_protocol(paths.protocol_path)
    paths.work_dir.mkdir(parents=True, exist_ok=True)
    return paths
