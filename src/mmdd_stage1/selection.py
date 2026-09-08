"""Checkpoint gating, history paths, and the Stage-1-to-Stage-2 contract."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .artifacts import checkpoint_fingerprint, write_json


def _metric_raw_value(metrics: dict[str, Any], name: str) -> int | float:
    value: Any = metrics
    for part in name.split("."):
        if not isinstance(value, dict) or part not in value:
            raise ValueError(f"Primary metric {name!r} is absent from dev metrics")
        value = value[part]
    if not isinstance(value, (int, float)):
        raise ValueError(f"Primary metric {name!r} is not numeric")
    return value


def metric_value(metrics: dict[str, Any], name: str) -> float:
    value = _metric_raw_value(metrics, name)
    return float(value)


@dataclass(frozen=True)
class MetricCriterion:
    name: str
    maximize: bool = True


def compare_metric_vectors(
    left: dict[str, Any],
    right: dict[str, Any],
    criteria: list[MetricCriterion],
    *,
    tolerance: float = 1e-12,
) -> tuple[int, str]:
    """Compare metric dictionaries in declared lexicographic order."""

    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")
    for criterion in criteria:
        left_value = _metric_raw_value(left, criterion.name)
        right_value = _metric_raw_value(right, criterion.name)
        tied = (
            left_value == right_value
            if isinstance(left_value, int) and isinstance(right_value, int)
            else abs(left_value - right_value) <= tolerance
        )
        if tied:
            continue
        left_better = left_value > right_value
        if not criterion.maximize:
            left_better = not left_better
        direction = "higher" if criterion.maximize else "lower"
        reason = (
            f"{criterion.name}: {left_value!r} vs {right_value!r}; "
            f"{direction} is preferred"
        )
        return (1 if left_better else -1), reason
    return 0, "all declared metrics tied"


def select_lexicographic(
    candidates: list[dict[str, Any]],
    criteria: list[MetricCriterion],
    *,
    metrics_key: str = "metrics",
    step_key: str = "step",
    id_key: str | None = None,
    tolerance: float = 1e-12,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Select by metrics, then earlier step, then an optional stable ID."""

    if not candidates:
        raise ValueError("Cannot select from an empty candidate list")
    if not criteria:
        raise ValueError("At least one metric criterion is required")
    best = candidates[0]
    decisions = []
    for candidate in candidates[1:]:
        candidate_step = int(candidate[step_key])
        comparison, reason = compare_metric_vectors(
            candidate[metrics_key],
            best[metrics_key],
            criteria,
            tolerance=tolerance,
        )
        if comparison == 0:
            best_step = int(best[step_key])
            if candidate_step != best_step:
                comparison = 1 if candidate_step < best_step else -1
                reason = (
                    f"{step_key}: {candidate_step} vs {best_step}; earlier is preferred"
                )
            elif id_key is not None:
                candidate_id = str(candidate[id_key])
                best_id = str(best[id_key])
                if candidate_id != best_id:
                    comparison = 1 if candidate_id < best_id else -1
                    reason = (
                        f"{id_key}: {candidate_id!r} vs {best_id!r}; "
                        "lexically smaller is preferred"
                    )
        decisions.append(
            {
                "candidate": candidate.get(id_key) if id_key else candidate_step,
                "incumbent": best.get(id_key) if id_key else int(best[step_key]),
                "comparison": comparison,
                "reason": reason,
            }
        )
        if comparison > 0:
            best = candidate
    return best, {
        "criteria": [
            {"name": value.name, "maximize": value.maximize}
            for value in criteria
        ],
        "tolerance": tolerance,
        "decisions": decisions,
    }


@dataclass(frozen=True)
class GateDecision:
    value: float
    improved: bool
    best_epoch: int
    best_value: float
    bad_epochs: int
    should_stop: bool


class MetricGate:
    def __init__(
        self,
        primary_metric: str,
        *,
        min_delta: float = 0.0,
        patience: int = 3,
        maximize: bool | None = None,
        criteria: list[MetricCriterion] | None = None,
        tolerance: float = 1e-12,
    ) -> None:
        if min_delta < 0 or patience < 0:
            raise ValueError("min_delta and patience must be non-negative")
        self.primary_metric = primary_metric
        self.min_delta = min_delta
        self.patience = patience
        self.maximize = (
            not primary_metric.endswith("loss") if maximize is None else maximize
        )
        self.criteria = criteria
        self.tolerance = tolerance
        self.best_epoch = 0
        self.best_value: float | None = None
        self.best_metrics: dict[str, Any] | None = None
        self.bad_epochs = 0
        self.last_reason = "first candidate"

    def observe(self, epoch: int, metrics: dict[str, Any]) -> GateDecision:
        value = metric_value(metrics, self.primary_metric)
        improved = self.best_value is None
        if self.best_value is not None:
            if self.criteria is not None:
                assert self.best_metrics is not None
                comparison, self.last_reason = compare_metric_vectors(
                    metrics,
                    self.best_metrics,
                    self.criteria,
                    tolerance=self.tolerance,
                )
                improved = comparison > 0
            else:
                change = (
                    value - self.best_value
                    if self.maximize
                    else self.best_value - value
                )
                improved = change > 0 and change >= self.min_delta
                self.last_reason = (
                    f"{self.primary_metric}: change={change!r}, "
                    f"min_delta={self.min_delta!r}"
                )
        if improved:
            self.best_value = value
            self.best_metrics = metrics
            self.best_epoch = epoch
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
        should_stop = self.patience > 0 and self.bad_epochs >= self.patience
        return GateDecision(
            value=value,
            improved=improved,
            best_epoch=self.best_epoch,
            best_value=float(self.best_value),
            bad_epochs=self.bad_epochs,
            should_stop=should_stop,
        )


def checkpoint_artifact_paths(output: Path) -> dict[str, Path]:
    suffix = output.suffix
    stem = output.name[: -len(suffix)] if suffix else output.name
    return {
        "best": output,
        "last": output.with_name(f"{stem}.last{suffix}"),
        "epochs": output.parent / f"{stem}.epochs",
        "history": output.with_suffix(output.suffix + ".history.json"),
        "selection": output.with_suffix(output.suffix + ".selection.json"),
    }


class CheckpointManager:
    def __init__(self, output: Path) -> None:
        self.paths = checkpoint_artifact_paths(output)
        self.paths["best"].parent.mkdir(parents=True, exist_ok=True)
        self.paths["epochs"].mkdir(parents=True, exist_ok=True)

    def save_candidate(self, epoch: int, payload: dict[str, Any]) -> Path:
        candidate = self.paths["epochs"] / f"epoch_{epoch:03d}.pt"
        temporary = candidate.with_suffix(candidate.suffix + ".tmp")
        torch.save(payload, temporary)
        temporary.replace(candidate)
        self._copy(candidate, self.paths["last"])
        return candidate

    def update_best(self, candidate: Path) -> None:
        self._copy(candidate, self.paths["best"])

    def prune_candidates(self, retained_epochs: set[int]) -> None:
        """Keep only explicitly retained per-epoch checkpoints."""

        retained_names = {
            f"epoch_{epoch:03d}.pt" for epoch in retained_epochs
        }
        for candidate in self.paths["epochs"].glob("epoch_*.pt"):
            if candidate.name not in retained_names:
                candidate.unlink()

    @staticmethod
    def _copy(source: Path, destination: Path) -> None:
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        shutil.copyfile(source, temporary)
        temporary.replace(destination)


def load_stage1_selection(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("format_version") != 1:
        raise ValueError(f"{path}: unsupported Stage-1 selection manifest")
    return payload


def validate_stage2_gate(
    selection_path: Path,
    retrieval_paths: list[Path] | None = None,
) -> dict[str, Any]:
    selection = load_stage1_selection(selection_path)
    if selection.get("completed_stage") != "student-path":
        raise ValueError("Stage 2 requires a dev-gated student-path checkpoint")
    if selection.get("selection_split") != "dev":
        raise ValueError("Stage 2 requires checkpoint selection on the dev split")
    if not selection.get("stage2_allowed"):
        coverage = selection.get("best_metrics", {}).get(
            "positive_evidence_path_coverage@10"
        )
        raise ValueError(
            "Stage 2 is blocked: the best Student checkpoint has insufficient "
            f"positive evidence-path dev coverage ({coverage})"
        )
    checkpoint_path = Path(str(selection["best_checkpoint"]))
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Stage-1 best checkpoint is missing: {checkpoint_path}"
        )
    expected_sha256 = str(selection["best_checkpoint_sha256"])
    if checkpoint_fingerprint(checkpoint_path) != expected_sha256:
        raise ValueError("Stage-1 best checkpoint fingerprint no longer matches its gate")
    for retrieval_path in retrieval_paths or []:
        text = retrieval_path.read_text(encoding="utf-8")
        try:
            payload = json.loads(text)
            records = payload if isinstance(payload, list) else [payload]
        except json.JSONDecodeError:
            records = [json.loads(line) for line in text.splitlines() if line.strip()]
        for record_number, record in enumerate(records, 1):
            if record.get("student_checkpoint_sha256") != expected_sha256:
                raise ValueError(
                    f"{retrieval_path}:record {record_number}: retrieval was not produced "
                    "by the dev-gated best Student checkpoint"
                )
    return selection
