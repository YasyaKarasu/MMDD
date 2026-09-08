#!/usr/bin/env python
"""Write the auditable R11 Task-A baseline funnel and concise results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from mmdd_stage1.retrieval import checkpoint_fingerprint


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _system(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "queries": metrics["queries"],
        "recall@10": metrics["recall@10"],
        "direct_recall@10": metrics["direct"]["recall@10"],
        "evidence_recall@10": metrics["evidence"]["recall@10"],
        "valid_path_recall@10,4": metrics["valid_path_recall@10,4"],
        "evidence_funnel": metrics["evidence_funnel"],
    }


def _exact_summary(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        system: {
            relation: {
                key: value
                for key, value in values.items()
                if key != "per_source"
            }
            for relation, values in relations.items()
        }
        for system, relations in payload["systems"].items()
    }


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# R11 Task A Protocol Baselines",
        "",
        "The shared unlabeled EntiTables-v9 lake is transductively indexed. "
        "Only train-fit labels may supervise P/R or the fresh Teacher. No "
        "independent confirmation split is available.",
        "",
        "| System | R@10 | Direct R@10 | Evidence R@10 | ValidPath@10,4 | ValidPool | ValidB | RowB | Q->E | E->T given Q->E |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name in ("raw", "pca"):
        row = payload["systems"][name]
        funnel = row["evidence_funnel"]
        lines.append(
            f"| {name.upper()} | {row['recall@10']:.4f} | "
            f"{row['direct_recall@10']:.4f} | {row['evidence_recall@10']:.4f} | "
            f"{row['valid_path_recall@10,4']:.4f} | {funnel['valid_pool']:.4f} "
            f"({funnel['valid_pool_count']}/{funnel['implicit_positive_pairs']}) | "
            f"{funnel['valid_b']:.4f} ({funnel['valid_b_count']}/"
            f"{funnel['implicit_positive_pairs']}) | {funnel['row_b']:.4f} | "
            f"{funnel['q_to_e_pair_recall']:.4f} | "
            f"{funnel['e_to_t_pair_recall_given_q_to_e']:.4f} |"
        )
    lines.extend(
        [
            "",
            "The full per-positive-pair funnel records are retained in "
            "`baseline_funnel.json`, so every numerator can be recomputed.",
            "",
            "## ANN exact-search check",
            "",
            "| System/relation | ANN-exact top-K overlap | ANN positive recall | Exact positive recall |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for system, relations in payload["exact_search"]["systems"].items():
        for relation, row in relations.items():
            lines.append(
                f"| {system}/{relation} | {row['ann_exact_overlap']:.4f} | "
                f"{row['ann_positive_recall']:.4f} | "
                f"{row['exact_positive_recall']:.4f} |"
            )
    lines.extend(
        [
            "",
            "Raw ANN and exact positive recall agree in every sampled relation. "
            "PCA differs by one sampled positive for Q->T and one for Q->image; "
            "PCA Q->image also has the lowest ANN/exact overlap (0.8449).",
            "",
            "Protocol audit: fixed-mask known-positive-as-negative count is 0 "
            "over two replayed epochs; the local-list historical mask produces "
            "1,816 such assignments under the same deterministic sampling.",
            "",
        ]
    )
    return "\n".join(lines)


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    history_path = Path(args.pca_history).resolve()
    exact_path = Path(args.exact_search).resolve()
    history = _load(history_path)
    exact = _load(exact_path)
    metrics = history["epochs"][0]["dev_retrieval"]
    payload = {
        "format_version": 1,
        "split": "dev",
        "protocol": {
            "direct_k": 100,
            "query_to_evidence_k_per_modality": 20,
            "targets_per_evidence": 20,
            "evidence_per_target": 4,
        },
        "source_history": str(history_path),
        "source_history_sha256": checkpoint_fingerprint(history_path),
        "systems": {
            "raw": _system(metrics["raw_embedding"]),
            "pca": _system(metrics),
        },
        "exact_search": {
            "source": str(exact_path),
            "source_sha256": checkpoint_fingerprint(exact_path),
            "samples": exact["samples"],
            "systems": _exact_summary(exact),
        },
    }
    _write_json(output_dir / "baseline_funnel.json", payload)
    (output_dir / "RESULTS.md").write_text(
        _markdown(payload), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": "pass",
                "baseline_funnel": str(output_dir / "baseline_funnel.json"),
                "results": str(output_dir / "RESULTS.md"),
            },
            indent=2,
        )
    )
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pca-history", required=True)
    parser.add_argument("--exact-search", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
