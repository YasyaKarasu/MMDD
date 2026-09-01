"""Small resumable primitives for multi-round Stage-1 hard-negative mining."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .artifacts import checkpoint_fingerprint, write_json


def workflow_fingerprint(
    config: dict[str, Any], input_paths: list[Path]
) -> str:
    inputs = []
    for path in input_paths:
        if not path.is_file():
            raise FileNotFoundError(f"Workflow input is missing: {path}")
        inputs.append(
            {
                "path": str(path.resolve()),
                "sha256": checkpoint_fingerprint(path),
            }
        )
    encoded = json.dumps(
        {"config": config, "inputs": inputs},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class RoundStep:
    """Resume a completed step only when its inputs and outputs are unchanged."""

    def __init__(self, marker: Path, fingerprint: str) -> None:
        self.marker = marker
        self.fingerprint = fingerprint

    def completed(self) -> bool:
        if not self.marker.is_file():
            return False
        payload = json.loads(self.marker.read_text(encoding="utf-8"))
        if payload.get("fingerprint") != self.fingerprint:
            raise ValueError(
                f"{self.marker}: completed round step has a different fingerprint; "
                "use a new round directory"
            )
        for record in payload.get("outputs", []):
            path = Path(str(record["path"]))
            if not path.is_file():
                raise FileNotFoundError(
                    f"{self.marker}: completed step output is missing: {path}"
                )
            if checkpoint_fingerprint(path) != record["sha256"]:
                raise ValueError(
                    f"{self.marker}: completed step output changed: {path}"
                )
        return payload.get("status") == "complete"

    def complete(self, outputs: list[Path]) -> None:
        write_json(
            self.marker,
            {
                "format_version": 1,
                "status": "complete",
                "fingerprint": self.fingerprint,
                "outputs": [
                    {
                        "path": str(path.resolve()),
                        "sha256": checkpoint_fingerprint(path),
                    }
                    for path in outputs
                ],
            },
        )


def validate_round_index(
    index_dir: Path,
    *,
    student_checkpoint_sha256: str,
    corpus_sha256: str,
) -> dict[str, Any]:
    manifest_path = index_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Round ANN manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("student_checkpoint_sha256") != student_checkpoint_sha256:
        raise ValueError(
            f"{manifest_path}: round index belongs to a different Student checkpoint"
        )
    if manifest.get("corpus_sha256") != corpus_sha256:
        raise ValueError(f"{manifest_path}: round index belongs to a different corpus")
    for record in manifest.get("types", {}).values():
        for key in ("index_path", "ids_path"):
            artifact = index_dir / str(record[key])
            if not artifact.is_file():
                raise FileNotFoundError(
                    f"{manifest_path}: index artifact is missing: {artifact}"
                )
    return manifest
