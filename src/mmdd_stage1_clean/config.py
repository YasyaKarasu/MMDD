"""Configuration resolution and the fresh-run allowlist (spec sections 2, 11, 14.1)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .util import read_json, sha256_path, stable_digest, write_json

SPEC_CONFIG_NAME = "clean_r1.json"
RESOLVED_CONFIG_NAME = "resolved_config.json"

# Substrings that mark a path as a historical training artifact.  Any such path
# is rejected before it can reach a model or loss (spec section 14.1).
FORBIDDEN_PATH_MARKERS = (
    "checkpoint",
    "optimizer",
    "teacher_logits",
    "hard_negative",
    "hard_negatives",
    "target_lists",
    "fixed_cohort",
    "pca",
    "projection",
    "row_support",
    "rankings",
    "b13",
    "c3_",
    "r22_",
    "r26_",
    "r30_",
)

# Paths that are explicitly allowed to be opened even though they live under a
# historical-looking root.  Populated from config["fresh"] and used sparingly.
FORBIDDEN_STATE_FIELDS = (
    "parent_checkpoint",
    "resume_checkpoint",
    "historical_train_lists",
    "historical_rankings",
    "historical_teacher_logits",
    "historical_pca",
)


class ConfigError(RuntimeError):
    """Raised when configuration or input wiring violates the contract."""


def missing_backbone_path(searched: list[str]) -> ConfigError:
    return ConfigError(
        "missing_backbone_path: no usable Qwen3-VL-Embedding-8B directory found. "
        f"Searched {searched}. Set MMDD_QWEN_MODEL_DIR to an existing local copy; "
        "this run does not download a second copy of the weights."
    )


def assert_path_allowed(path: Path, *, purpose: str) -> Path:
    """Reject historical training artifacts from reaching a loader."""
    text = str(Path(path)).lower()
    for marker in FORBIDDEN_PATH_MARKERS:
        if marker in text:
            raise ConfigError(
                f"read-allowlist violation: refusing to open {path} for {purpose}; "
                f"it matches historical-artifact marker {marker!r}. "
                "CLEAN-R1 may only read raw data, raw GT and verified frozen Qwen caches."
            )
    return Path(path)


def _backbone_looks_usable(directory: Path) -> bool:
    if not (directory / "config.json").is_file():
        return False
    if not (directory / "model.safetensors.index.json").is_file():
        return False
    return (directory / "scripts" / "qwen3_vl_embedding.py").is_file()


def resolve_backbone(repo_root: Path, configured: str | None) -> tuple[Path, list[str]]:
    """Resolve the frozen Qwen directory using only the spec's deterministic rule.

    Order: $MMDD_QWEN_MODEL_DIR, then <repo>/models/Qwen3-VL-Embedding-8B, then
    <repo>/Qwen3-VL-Embedding-8B.  The first usable directory wins.  Nothing is
    ever downloaded.
    """
    searched: list[str] = []
    candidates: list[Path] = []
    if configured:
        candidates.append(Path(configured).expanduser())
    searched.append(f"$MMDD_QWEN_MODEL_DIR={configured or '<unset>'}")
    candidates.append(repo_root / "models" / "Qwen3-VL-Embedding-8B")
    searched.append(str(repo_root / "models" / "Qwen3-VL-Embedding-8B"))
    candidates.append(repo_root / "Qwen3-VL-Embedding-8B")
    searched.append(str(repo_root / "Qwen3-VL-Embedding-8B"))
    for candidate in candidates:
        if candidate and _backbone_looks_usable(candidate):
            return candidate.resolve(), searched
    raise missing_backbone_path(searched)


def load_spec_config(spec_dir: Path) -> dict[str, Any]:
    path = Path(spec_dir) / SPEC_CONFIG_NAME
    if not path.is_file():
        raise ConfigError(f"spec config not found: {path}")
    config = read_json(path)
    config["_spec_config_path"] = str(path.resolve())
    config["_spec_config_sha256"] = sha256_path(path)
    return config


def resolve_config(config: dict[str, Any], *, cwd: Path) -> dict[str, Any]:
    """Resolve every environment path exactly once and freeze the result."""
    repo_root = (cwd / config["paths"]["repo_root"]).resolve()
    dataset_root = (repo_root / config["paths"]["dataset_root"]).resolve()
    if not dataset_root.is_dir():
        raise ConfigError(
            f"dataset_root does not exist: {dataset_root}. "
            "Only paths.dataset_root may be changed when relocating the lake; "
            "do not substitute a similarly named historical list directory."
        )
    output_root = (repo_root / config["paths"]["output_root"]).resolve()
    import os

    backbone, searched = resolve_backbone(
        repo_root, os.environ.get(config["paths"]["qwen_model_env"])
    )
    resolved = {
        "protocol_id": config["protocol_id"],
        "source_spec_dir": str(Path(config["_spec_config_path"]).parent),
        "spec_config_path": config["_spec_config_path"],
        "spec_config_sha256": config["_spec_config_sha256"],
        "spec_doc_sha256": None,
        "seed": int(config["seed"]),
        "stage2_enabled": bool(config["stage2_enabled"]),
        "paths": {
            "repo_root": str(repo_root),
            "dataset_root": str(dataset_root),
            "output_root": str(output_root),
            "backbone_dir": str(backbone),
            "backbone_search": searched,
        },
        "fresh": config["fresh"],
        "data": config["data"],
        "cache": config["cache"],
        "student": config["student"],
        "teacher": config["teacher"],
        "optimizer": config["optimizer"],
        "sampling": config["sampling"],
        "retrieval": config["retrieval"],
        "ann": config["ann"],
        "evaluation": config["evaluation"],
    }
    spec_doc = Path(config["_spec_config_path"]).parent / "EXPERIMENT_SPEC.zh-CN.md"
    if spec_doc.is_file():
        resolved["spec_doc_sha256"] = sha256_path(spec_doc)
    resolved["resolved_config_sha256"] = stable_digest(
        {k: v for k, v in resolved.items() if k != "resolved_config_sha256"}
    )
    return resolved


def load_resolved(output_root: Path) -> dict[str, Any]:
    path = Path(output_root) / RESOLVED_CONFIG_NAME
    if not path.is_file():
        raise ConfigError(
            f"resolved config missing: {path}. Run `audit-input` first; every later "
            "command reads only the resolved version."
        )
    return read_json(path)


def freeze_resolved(output_root: Path, resolved: dict[str, Any]) -> Path:
    path = Path(output_root) / RESOLVED_CONFIG_NAME
    write_json(path, resolved)
    return path
