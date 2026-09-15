"""Reconstruct saved COV logits from frozen embeddings using independent NumPy math."""
import argparse
import json

import numpy as np
import torch

from prepare_stage1_r27 import record, rows, write_json
from prepare_stage1_r28 import OUT, inputs, feature_store
from pathlib import Path


def main(require_complete: bool) -> None:
    torch.set_num_threads(2)
    historical = list(rows(Path(inputs()["historical_b13_rankings"]["path"])))
    # Fixed positions in the locked query order, chosen independently of scores.
    selected = {historical[i]["query_id"]: historical[i] for i in (0, 599, 1197)}
    store = feature_store(False)
    normalized_rows = {}
    real_evidence = {}
    for q, meta in selected.items():
        matrix = store.embedding_features(q).row_embeddings.numpy().astype(np.float64)
        normalized_rows[q] = matrix / np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)
        real_evidence[q] = {c["target_id"]: c["selected_evidence_ids"] for c in meta["E_paths"]}
    checks = []
    receipts = sorted((OUT / "teacher/evaluation").glob("**/EVALUATION_RECEIPT.json"))
    if require_complete:
        assert len(receipts) == 31
    for receipt_path in receipts:
        receipt = json.loads(receipt_path.read_text())
        gid = receipt["model_id"]
        evidence = {(q, "Real"): bags for q, bags in real_evidence.items()}
        if receipt["shuffle"]:
            for q in selected:
                evidence[q, "Shuffled"] = {}
            for donor in rows(OUT / "teacher/evidence_shuffle" / gid / "donors.jsonl.gz"):
                if donor["query_id"] in selected:
                    evidence[donor["query_id"], "Shuffled"][donor["target_id"]] = donor["evidence_ids"]
        n, max_error, conditions = 0, 0., set()
        for saved in rows(receipt_path.parent / "scores.jsonl.gz"):
            q = saved["query_id"]
            if q not in selected:
                continue
            conditions.add((q, saved["condition"]))
            bags = evidence[q, saved["condition"]]
            for target, logits in saved["path_logits"].items():
                vectors = np.stack([store.embedding_features(e).embedding.numpy().astype(np.float64) for e in sorted(bags[target])])
                vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
                support = np.clip((normalized_rows[q] @ vectors.T + 1) / 2, 0, 1)
                probability = 1 / (1 + np.exp(-np.asarray(logits)))
                coverage = np.clip(np.max(support * probability[None, :], axis=1).mean(), 1e-6, 1 - 1e-6)
                expected = float(np.log(coverage / (1 - coverage)))
                error = abs(expected - saved["E-COV"][target])
                assert error < 1e-4, (gid, q, target, error)
                max_error = max(max_error, error)
                n += 1
        assert conditions == set(evidence)
        checks.append({"model_id": gid, "target_bags": n, "max_float64_vs_saved_error": max_error, "receipt": record(receipt_path)})
        print(json.dumps({k: v for k, v in checks[-1].items() if k != "receipt"}), flush=True)
    write_json(OUT / "INDEPENDENT_COVERAGE_AUDIT.json", {"status": "pass" if len(receipts) == 31 else "partial_pass", "code": record(Path(__file__)),
        "selection": "locked query positions 0,599,1197; every retained target bag and available Real/Shuffled condition",
        "queries": list(selected), "checks": checks})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-complete", action="store_true")
    main(parser.parse_args().require_complete)
