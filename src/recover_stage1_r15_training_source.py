#!/usr/bin/env python
"""Recover the R14/R15 training entrypoints by their recorded SHA256 hashes."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def fingerprint(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def reconstruct(current: str) -> str:
    """Undo only the later evaluation addition; the final hash is authoritative."""
    start = current.index("@torch.inference_mode()\ndef evaluate_arm")
    end = current.index("def parse_args() -> argparse.Namespace:", start)
    original = current[:start] + current[end:]
    for block in (
        "import gzip\n",
        "import statistics\n",
        re.search(r"from mmdd_stage1.retrieval import \([\s\S]*?\)\n", current).group(),
        "from mmdd_stage1.row_support import load_evidence_content_keys\n",
        "from run_stage1_r11_task_e import empty_intervention_stats\n",
        "from run_stage1_r11_task_f import _accumulate, _empty, _finalize, _target_channels\n",
        '    parser.add_argument("--index-threads", type=int, default=12)\n',
        '    parser.add_argument("--query-batch-size", type=int, default=16)\n',
    ):
        original = original.replace(block, "")
    r13_imports = re.search(r"from run_stage1_r13 import \([\s\S]*?\)\n", original).group()
    original = original.replace(r13_imports, "from run_stage1_r13 import _paths as r13_paths\n")
    original = original.replace('choices=("freeze", "train", "evaluate")', 'choices=("freeze", "train")')
    original = original.replace(
        '    if args.task == "train":\n        train_arm(args)\n    else:\n        evaluate_arm(args)\n',
        "    train_arm(args)\n",
    )
    return original


def reconstruct_r14(current: str) -> str:
    """Undo the analogous evaluation-only addition to the R14 entrypoint."""
    start = current.index("def _evaluation_checkpoint(")
    end = current.index("def parse_args() -> argparse.Namespace:", start)
    original = current[:start] + current[end:]
    for block in (
        "import gzip\n",
        "import statistics\n",
        re.search(r"from mmdd_stage1.retrieval import \([\s\S]*?\)\n", current).group(),
        "from mmdd_stage1.row_support import load_evidence_content_keys\n",
        "from run_stage1_r11_task_e import empty_intervention_stats\n",
        "from run_stage1_r11_task_f import _accumulate, _empty, _finalize, _target_channels\n",
        '    parser.add_argument("--evaluate", action="store_true")\n',
        '    parser.add_argument("--index-threads", type=int, default=12)\n',
        '    parser.add_argument("--query-batch-size", type=int, default=16)\n',
        "    _DirectScorer,\n",
        "    _paths_by_target,\n",
        "    _recall_record,\n",
    ):
        original = original.replace(block, "")
    return original.replace("    elif args.evaluate:\n        evaluate_arm(args)\n", "")


def recover_r14(root: Path, output: Path) -> dict[str, Any]:
    source = root / "src/run_stage1_r14.py"
    current = source.read_text(encoding="utf-8")
    recovered = reconstruct_r14(current)
    digest = hashlib.sha256(recovered.encode()).hexdigest()
    current_digest = hashlib.sha256(current.encode()).hexdigest()
    manifests = sorted((root / "work/stage1_optimization_r14_20260909").glob("stage*/*/manifest.json"))
    matched_reconstructed, matched_current = [], []
    for path in manifests:
        recorded = json.loads(path.read_text(encoding="utf-8"))["code_sha256"]
        if recorded == digest:
            matched_reconstructed.append(path)
        elif recorded == current_digest:
            matched_current.append(path)
        else:
            raise ValueError("An R14 training manifest matches neither available entrypoint")
    if len(matched_reconstructed) != 3 or len(matched_current) != 4:
        raise ValueError("The frozen seven-arm R14 source population differs")
    trees = [{node.name: ast.dump(node, include_attributes=False)
              for node in ast.parse(text).body if isinstance(node, ast.FunctionDef)}
             for text in (current, recovered)]
    unchanged = {name: trees[0][name] == trees[1][name]
                 for name in sorted(set(trees[1]) - {"main", "parse_args"})}
    if not all(unchanged.values()):
        raise ValueError("R14 training/shared function bodies differ after reconstruction")
    destination = output / "source_snapshot/historical/run_stage1_r14.py"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(recovered, encoding="utf-8")
    return {
        "status": "hash_matched_training_entrypoint_reconstructed",
        "current_source": fingerprint(source),
        "recovered_source": fingerprint(destination),
        "reconstructed_source_training_manifests": [fingerprint(path) for path in matched_reconstructed],
        "current_source_training_manifests": [fingerprint(path) for path in matched_current],
        "all_seven_training_manifest_entrypoints_match": True,
        "current_training_and_shared_function_AST_unchanged": unchanged,
        "optimizer_definition_AST_unchanged": unchanged["_optimizer"],
        "optimizer_tensors_recovered": False,
        "limits": "Entrypoint and optimizer-definition source identity is established against recorded hashes; imported-module runtime history and actual Adam tensors are not recovered.",
    }


def recover(root: Path) -> dict[str, Any]:
    root = root.resolve()
    output = root / "work/stage1_optimization_r15_20260909"
    source = root / "src/run_stage1_r15.py"
    current = source.read_text(encoding="utf-8")
    recovered = reconstruct(current)
    recovered_bytes = recovered.encode("utf-8")
    digest = hashlib.sha256(recovered_bytes).hexdigest()
    manifests = [output / f"stageI_interaction/{arm}_seed13/manifest.json"
                 for arm in ("l_eoff", "n_eoff")]
    expected = {json.loads(path.read_text(encoding="utf-8"))["code_sha256"] for path in manifests}
    if expected != {digest}:
        raise ValueError("Reconstruction does not match both training manifest hashes")
    trees = [{node.name: ast.dump(node, include_attributes=False)
              for node in ast.parse(text).body if isinstance(node, ast.FunctionDef)}
             for text in (current, recovered)]
    shared = sorted(set(trees[1]) - {"main", "parse_args"})
    unchanged = {name: trees[0][name] == trees[1][name] for name in shared}
    if not all(unchanged.values()):
        raise ValueError("Training/shared function bodies differ after reconstruction")
    r14 = recover_r14(root, output)
    destination = output / "source_snapshot/historical/run_stage1_r15.py"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(recovered_bytes)
    receipt = {
        "status": "hash_matched_training_entrypoint_reconstructed",
        "recovered_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": "Remove only later evaluate_arm, evaluation-only imports/CLI options and dispatch branch",
        "current_source": fingerprint(source),
        "recovered_source": fingerprint(destination),
        "training_manifest_dependencies": [fingerprint(path) for path in manifests],
        "both_training_manifest_hashes_match": True,
        "current_training_and_shared_function_AST_unchanged": unchanged,
        "historical_code_executed_during_recovery": False,
        "optimizer_tensors_recovered": False,
        "r14_entrypoint": r14,
        "limits": [
            "Hash identity is to the entrypoint file recorded by the training manifests; imported module history is not thereby established.",
            "This is a post hoc hash-matched reconstruction, not a contemporaneously saved source archive.",
            "No optimizer tensors, original case CSV or protocol chronology are recovered.",
        ],
    }
    (output / "RECOVERED_TRAINING_SOURCE.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": receipt["status"], "recovered_source": receipt["recovered_source"]}))
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    recover(parser.parse_args().root)
