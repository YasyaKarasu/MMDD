"""Check actual Student ANN vectors and exact scores against checkpoint bilinear scores."""
import json
from pathlib import Path

import hnswlib
import numpy as np
import torch

from mmdd_stage1.checkpoints import load_student
from prepare_stage1_r27 import rows, record, write_json
from prepare_stage1_r28 import OUT, feature_store
from evaluate_stage1_r28_student import OWN


@torch.inference_mode()
def main() -> None:
    torch.set_num_threads(2)
    specs = [s for s in json.loads((OWN / "MODEL_INVENTORY.json").read_text()) if s["epoch"] in (1, 5)]
    population = list(rows(OWN / "common/dev_queries.jsonl"))
    qids = [population[i]["query_id"] for i in (0, 599, 1197)]
    store = feature_store(False)
    first = OWN / "indexes" / specs[0]["generator_id"]
    manifest = json.loads((first / "manifest.json").read_text())
    ids = json.loads((first / manifest["types"]["table"]["ids_path"]).read_text())
    position = {t: i for i, t in enumerate(ids)}
    print(json.dumps({"stage": "load_frozen_table_embeddings", "tables": len(ids)}), flush=True)
    targets = torch.stack([store.embedding_features(t).embedding for t in ids])
    queries = torch.stack([store.embedding_features(q).embedding for q in qids])
    results = []
    for spec in specs:
        gid = spec["generator_id"]
        model = load_student(Path(spec["checkpoint"]), torch.device("cpu")).eval()
        directory = OWN / "indexes" / gid
        manifest = json.loads((directory / "manifest.json").read_text())
        item = manifest["types"]["table"]
        assert json.loads((directory / item["ids_path"]).read_text()) == ids
        index = hnswlib.Index(space="ip", dim=manifest["ann_dim"])
        index.load_index(str(directory / item["index_path"]), max_elements=len(ids))
        saved_vectors = torch.from_numpy(index.get_items(np.arange(len(ids))))
        exported = model.index_vector(targets, "table")
        vector_error = float((saved_vectors - exported).abs().max())
        assert torch.allclose(saved_vectors, exported, atol=2e-5, rtol=2e-5)
        actual = model.raw_score_embedding_matrix(queries, "table", targets, "table")
        via_index = model.relation_query(queries, "table", "table") @ saved_vectors.T
        assert torch.allclose(actual, via_index, atol=2e-5, rtol=2e-5)
        selected = {r["query_id"]: r for r in rows(OWN / "rankings" / gid / "rankings.jsonl.gz") if r["query_id"] in qids}
        score_error, membership_changes = 0., []
        for i, q in enumerate(qids):
            row = selected[q]
            union = row["U"]
            recomputed = actual[i, [position[t] for t in union]]
            historical = torch.tensor([row["QT_OVER_U_scores"][t] for t in union])
            score_error = max(score_error, float((recomputed - historical).abs().max()))
            assert torch.allclose(recomputed, historical, atol=2e-5, rtol=2e-5), (gid, q, score_error)
            top = {ids[j] for j in actual[i].topk(100).indices.tolist()}
            old = set(row["D100_EXACT"])
            truth = set(row["positive_target_ids"])
            assert len(top & truth) == len(old & truth), (gid, q, "Direct exact Recall differs")
            membership_changes.append(len(top ^ old))
        result = {"model_id": gid, "checkpoint": record(Path(spec["checkpoint"])), "tables": len(ids),
                  "queries": qids, "max_saved_index_vector_error": vector_error,
                  "max_saved_score_error": score_error, "exact100_symmetric_differences": membership_changes,
                  "actual_model_and_index_score_agree": True, "querywise_exact100_recall_agrees": True}
        results.append(result)
        print(json.dumps({k: v for k, v in result.items() if k not in ("checkpoint", "queries")}), flush=True)
        del model, index, saved_vectors, exported, actual, via_index
    write_json(OUT / "STUDENT_EXPORT_NUMERICAL_AUDIT.json", {"status": "pass", "code": record(Path(__file__)),
        "scope": "All table index vectors at six epoch1 and six epoch5 checkpoints; full-corpus raw bilinear QT scores for three fixed query positions, compared with saved exact/U scores. No inference or training formula changed.", "checks": results})


if __name__ == "__main__":
    main()
