"""R26 frozen-input A/B/C verifier experiment; never generates recovered values."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from mmdd_dataset.utils import values_match
from mmdd_dataset.wdc_runtime import iter_dataset_artifact
from mmdd_stage2.data import column_values, load_stage2_index, local_column_index
from mmdd_stage2.join_diagnostic import (
    ColumnProjection, column_text, column_metrics, digest, multiple_positive_loss,
    read_rows, write_json,
)

ROOT = Path(__file__).resolve().parents[1]
R26 = ROOT / "work/stage1_optimization_r26_20260914"
DATASET = ROOT / "output_mm_joinability_entitables_20000_retry100_rounds5_qwen35_final_survivor_context_gaussian_v9"
OUT = ROOT / "work/r26_joinability_diagnostic_20260915"
INSTRUCTION = "Represent this visible table column for joinability comparison. Preserve the column name and visible cell values; do not infer hidden attributes."


def prepare(out: Path) -> None:
    splits = json.loads((ROOT / "work/stage1_optimization_r10_20260907/taskA_protocol/splits.json").read_text())["source_groups"]
    qrels = read_rows(DATASET / "qrels.jsonl")
    populations = {}
    for split, count, source_key in (("train", 512, "train_fit"), ("calibration", 128, "train_calibration")):
        candidates = defaultdict(list)
        for row in qrels:
            if row["source_table_id"] in set(splits[source_key]):
                candidates[row["reason"]].append(row)
        selected = []
        for reason, rows in sorted(candidates.items()):
            # One query per source group per kind before the fixed hash budget.
            unique = {}
            for row in sorted(rows, key=lambda r: digest(r["query_table_id"])):
                unique.setdefault(row["source_table_id"], row["query_table_id"])
            selected.extend(list(unique.values())[:count // 2])
        populations[split] = sorted(set(selected))
    pilots = {g: read_rows(R26 / f"stage2/pilot/{g}/results.jsonl") for g in ("B13", "Qwen-Raw")}
    populations["eval"] = sorted({r["query_id"] for rows in pilots.values() for r in rows})
    inputs = {g: read_rows(R26 / f"stage2/inputs/{g}/retrieval.jsonl") for g in pilots}
    wanted_q = set(q for values in populations.values() for q in values)
    labels = [r for r in qrels if r["query_table_id"] in wanted_q]
    wanted_t = {r["target_table_id"] for r in labels}
    wanted_t.update(c["target_id"] for rows in pilots.values() for r in rows for c in r["diagnostic_candidates"])
    objects = load_stage2_index(DATASET, query_ids=wanted_q, target_ids=wanted_t, evidence_ids=set())
    source_sets = {split: {objects.queries[q]["source_table_id"] for q in ids} for split, ids in populations.items()}
    assert all(not (source_sets[a] & source_sets[b]) for a, b in (("train", "calibration"), ("train", "eval"), ("calibration", "eval")))
    wanted_sources = set.union(*source_sets.values())
    sources = {r["source_table_id"]: r for r in iter_dataset_artifact(DATASET, "source_tables") if r["source_table_id"] in wanted_sources}
    columns, tables = {}, {"query": {}, "target": {}}
    for role, data in (("query", objects.queries), ("target", objects.targets)):
        for tid, table in data.items():
            keys = []
            for col in table["columns"]:
                key = f"{role}:{tid}:{col['column_index']}"
                full = column_values(table, col["column_index"], include_empty=True)
                columns[key] = {"name": col["column_name"], "values": full, "text": column_text(col["column_name"], full),
                                "source_table_id": table.get("source_table_id"), "column_index": col["column_index"]}
                keys.append(key)
            tables[role][tid] = keys
    split_of = {q: split for split, ids in populations.items() for q in ids}
    examples = []
    for label in labels:
        qid, tid = label["query_table_id"], label["target_table_id"]
        query, target = objects.queries[qid], objects.targets[tid]
        source_col = label["join_attribute"]["source_column_index"]
        gold = f"target:{tid}:{local_column_index(target, source_col)}"
        visible = [c for c in query["columns"] if c.get("source_column_index") == source_col]
        if visible:
            anchor = f"query:{qid}:{visible[0]['column_index']}"
        else:
            anchor = f"oracle:{qid}:{source_col}"
            source_rows = {r["row_id"]: r for r in sources[query["source_table_id"]]["rows"]}
            values = [next(c["text"] for c in source_rows[r["source_row_id"]]["cells"] if c["column_index"] == source_col) for r in query["rows"]]
            name = label["join_attribute"]["column_name"]
            columns[anchor] = {"name": name, "values": values, "text": column_text(name, values), "source_table_id": query["source_table_id"]}
        examples.append({"query_id": qid, "target_id": tid, "split": split_of[qid], "anchor": anchor, "gold": gold,
                         "source_table_id": query["source_table_id"], "source_column_index": source_col,
                         "kind": "explicit" if visible else "implicit", "candidates": tables["target"][tid]})
    for generator, rows in pilots.items():
        for row in rows:
            if row["status"] != "ran":
                continue
            for candidate in row["diagnostic_candidates"]:
                evidence = candidate["branches"].get("evidence", {})
                if evidence.get("verification") is None:
                    continue
                values = [r["value"] for r in evidence["rows"]]
                name = candidate["selection"]["column_name"]
                key = "recovered:" + digest([name, values])
                columns[key] = {"name": name, "values": values, "text": column_text(name, values), "source_table_id": None}
                candidate["recovered_key"] = key
    cache = torch.load(R26 / "fusion/columns/columns.pt", map_location="cpu", weights_only=False)
    assert cache["instruction"] == INSTRUCTION and cache["max_values"] == 5
    old_inputs = {r["key"]: r["text"] for r in read_rows(R26 / "fusion/columns/visible_column_inputs.jsonl.gz")}
    vectors = {}
    for key, col in columns.items():
        if key in cache["vectors"] and old_inputs.get(key) == col["text"]:
            vectors[key] = cache["vectors"][key]
    out.mkdir(parents=True, exist_ok=True)
    torch.save(vectors, out / "column_reused.pt")
    # Cell embeddings only needed for A evaluation/calibration and fixed replay.
    cell_keys = {k for e in examples if e["split"] != "train" for k in [e["anchor"], *e["candidates"], *tables["query"][e["query_id"]]]}
    for rows in pilots.values():
        for row in rows:
            for c in row["diagnostic_candidates"]:
                cell_keys.update(tables["target"][c["target_id"]])
                if c.get("recovered_key"):
                    cell_keys.add(c["recovered_key"])
    cell_values = sorted({v for key in cell_keys for v in columns[key]["values"]})
    plan = {"populations": populations, "columns": columns, "tables": tables, "examples": examples,
            "pilots": pilots, "inputs": inputs, "cell_values": cell_values,
            "population": {r["query_id"]: r for r in read_rows(R26 / "common/dev_queries.jsonl")}}
    write_json(out / "prepared.json", plan)
    protocol = {"base": str(R26), "dataset": str(DATASET), "hypothesis": "Generic cell/column similarity plus maxima creates false direct support and harms fixed C18 ranking.",
        "populations": {s: len(v) for s, v in populations.items()}, "source_groups": {s: sorted(v) for s, v in source_sets.items()},
        "split_overlap": 0, "oracle_policy": "Gold source cells allowed for train/calibration and offline known-column diagnostic only; never in replay inputs.",
        "A": "Qwen3.5-9B isolated cell embeddings; max cosine; normalized exact match; threshold .8; coverage .6; original sorting",
        "B": "Frozen Qwen3-VL-Embedding-8B column name + first five cells; cosine",
        "C": {"projection": "shared bias-free 4096x256 linear", "seeds": [13, 29], "temperature": .07, "lr": .001, "epochs": 30,
              "batch_size": 64, "selection": "maximum source-balanced calibration column MRR among epochs 1..30; earliest trained epoch on tie; epoch 0 is a separate random-projection control",
              "loss": "multi-positive MNRL, in-batch + up to 4 frozen-cosine hard wrong columns; mask same-source in-batch negatives and exact-overlap>=.6 wrong columns"},
        "threshold": "B/C calibrated to >=95% true-column acceptance on explicit calibration positives; never tune on pilot",
        "replay": "fixed C18, generation values, selected column, branch eligibility, Stage1 scores and ranks; swap verifier score only",
        "unlabeled_negative_caveat": "qrels nonpositives/wrong attributes are not exhaustive proof of nonjoinability",
        "cell_values": len(cell_values), "columns": len(columns), "reused_columns": len(vectors),
        "missing_columns": len(columns)-len(vectors), "prepared_sha": digest(plan), "created_unix": time.time()}
    write_json(out / "PROTOCOL.json", protocol)
    print(json.dumps({k: v for k,v in protocol.items() if k != "source_groups"}), flush=True)


def encode_columns(out: Path, device: str, batch_size: int) -> None:
    from cache_stage1_features import _load_embedder_class, encode_inputs
    plan = json.loads((out / "prepared.json").read_text())
    path = out / "column_new.pt"
    vectors = torch.load(path, weights_only=False) if path.exists() else {}
    reused = torch.load(out / "column_reused.pt", weights_only=False)
    texts = defaultdict(list)
    for key, col in plan["columns"].items():
        if key not in vectors and key not in reused:
            texts[col["text"]].append(key)
    if not texts:
        return
    model_dir = ROOT / "hf_models/Qwen3-VL-Embedding-8B"
    model = _load_embedder_class(model_dir)(model_name_or_path=str(model_dir), torch_dtype=torch.bfloat16)
    model.model.to(device).eval()
    items = list(texts)
    started = time.monotonic()
    for start in range(0, len(items), batch_size):
        batch = items[start:start+batch_size]
        with torch.inference_mode():
            encoded = encode_inputs(model, [{"text": t, "instruction": INSTRUCTION} for t in batch], include_hidden=False)
        for txt, (vector, _, _) in zip(batch, encoded, strict=True):
            for key in texts[txt]:
                vectors[key] = F.normalize(vector.float(), dim=0).half().cpu()
        if start // batch_size % 20 == 0:
            print(json.dumps({"phase": "columns", "encoded_texts": start+len(batch), "total": len(items), "seconds": time.monotonic()-started}), flush=True)
            torch.save(vectors, path)
    torch.save(vectors, path)
    write_json(out / "column_encoding_receipt.json", {"texts": len(items), "vectors": len(vectors), "seconds": time.monotonic()-started, "device": device})


def encode_cells(out: Path, device: str, batch_size: int) -> None:
    from mmdd_stage2.qwen import QwenStage2Backend
    plan = json.loads((out / "prepared.json").read_text())
    values = plan["cell_values"]
    path = out / "cells.pt"
    payload = torch.load(path, weights_only=False) if path.exists() else {"values": [], "vectors": torch.empty(0, 4096)}
    assert values[:len(payload["values"])] == payload["values"]
    if len(payload["values"]) == len(values):
        return
    backend = QwenStage2Backend(ROOT / "hf_models/Qwen3.5-9B", device=device, dtype="bf16", embedding_batch_size=batch_size)
    chunks = [payload["vectors"]] if len(payload["values"]) else []
    started = time.monotonic()
    for start in range(len(payload["values"]), len(values), 2048):
        chunks.append(backend.embed_texts(values[start:start+2048]))
        n = min(start+2048, len(values))
        torch.save({"values": values[:n], "vectors": torch.cat(chunks)}, path)
        print(json.dumps({"phase": "cells", "encoded": n, "total": len(values), "seconds": time.monotonic()-started}), flush=True)
    write_json(out / "cell_encoding_receipt.json", {"values": len(values), "seconds": time.monotonic()-started, "device": device, "dtype": "float32 normalized outputs; bf16 backbone"})


def load_columns(out: Path) -> dict[str, torch.Tensor]:
    vectors = torch.load(out / "column_reused.pt", map_location="cpu", weights_only=False)
    vectors.update(torch.load(out / "column_new.pt", map_location="cpu", weights_only=False))
    return vectors


def fit(out: Path, device: str, seed: int) -> None:
    plan = json.loads((out / "prepared.json").read_text())
    vectors = load_columns(out)
    keys = sorted(vectors)
    indices = {k: i for i,k in enumerate(keys)}
    matrix = F.normalize(torch.stack([vectors[k] for k in keys]).float(), dim=1).to(device)
    train = [e for e in plan["examples"] if e["split"] == "train"]
    cal = [e for e in plan["examples"] if e["split"] == "calibration"]
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = ColumnProjection().to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
    gold_by_anchor = defaultdict(set)
    for e in train:
        gold_by_anchor[e["anchor"]].add(e["gold"])
    hard = {}
    for e in train:
        vals = plan["columns"][e["anchor"]]["values"]
        wrong = [k for k in e["candidates"] if k not in gold_by_anchor[e["anchor"]]
                 and sum(bool(v.strip()) and any(values_match(v,t) for t in plan["columns"][k]["values"]) for v in vals)/len(vals) < .6]
        hard[e["anchor"]] = sorted(wrong, key=lambda k: -float(matrix[indices[e["anchor"]]] @ matrix[indices[k]]))[:4]
    history, best = [], -1.
    for epoch in range(31):
        losses = []
        if epoch:
            model.train()
            order = rng.permutation(len(train))
            for start in range(0, len(order), 64):
                batch = [train[i] for i in order[start:start+64]]
                targets = list(dict.fromkeys([e["gold"] for e in batch] + [k for e in batch for k in hard[e["anchor"]]]))
                positive = torch.tensor([[k in gold_by_anchor[e["anchor"]] for k in targets] for e in batch], device=device)
                allowed = torch.tensor([[k in gold_by_anchor[e["anchor"]] or k in hard[e["anchor"]] or plan["columns"][k]["source_table_id"] != e["source_table_id"] for k in targets] for e in batch], device=device)
                scores = model(matrix[[indices[e["anchor"]] for e in batch]]) @ model(matrix[[indices[k] for k in targets]]).T / .07
                loss = multiple_positive_loss(scores, positive, allowed)
                optim.zero_grad()
                loss.backward()
                optim.step()
                losses.append(float(loss.detach()))
        model.eval()
        with torch.inference_mode():
            projected = model(matrix)
            by_source = defaultdict(list)
            for e in cal:
                scores = (projected[indices[e["anchor"]]] @ projected[[indices[k] for k in e["candidates"]]].T).cpu().tolist()
                by_source[e["source_table_id"]].append(column_metrics(scores, [k == e["gold"] for k in e["candidates"]])["mrr"])
        metric = float(np.mean([np.mean(v) for v in by_source.values()]))
        history.append({"epoch": epoch, "loss": float(np.mean(losses)) if losses else None, "calibration_mrr": metric})
        if epoch == 0:
            torch.save({"state_dict": model.cpu().state_dict(), "seed": seed, "epoch": epoch, "calibration_mrr": metric}, out / f"projection_random_seed{seed}.pt")
            model.to(device)
        if epoch > 0 and metric > best:
            best = metric
            torch.save({"state_dict": model.cpu().state_dict(), "seed": seed, "epoch": epoch, "calibration_mrr": metric}, out / f"projection_seed{seed}.pt")
            model.to(device)
        print(json.dumps({"phase": "fit", "seed": seed, **history[-1]}), flush=True)
    write_json(out / f"fit_seed{seed}.json", {"history": history, "train_pairs": len(train), "calibration_pairs": len(cal), "hard_negative_count": sum(map(len, hard.values()))})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "columns", "cells", "fit"))
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.phase == "prepare":
        prepare(args.output)
    elif args.phase == "columns":
        encode_columns(args.output, args.device, args.batch_size)
    elif args.phase == "cells":
        encode_cells(args.output, args.device, args.batch_size)
    else:
        fit(args.output, args.device, args.seed)


if __name__ == "__main__":
    main()
