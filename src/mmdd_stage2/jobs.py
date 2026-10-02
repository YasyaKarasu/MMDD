"""Selector jobs: one (query, target, natural evidence) reader pair each.

Training labels come from qrels and only from ``model_recoverable_join_column`` reasons (the
hidden attribute a selector must find); explicit-join qrels are not latent supervision. A labeled
target outside the query's C30 is dropped, because Stage 2 never sees it. Evaluation jobs cover
every C30 target of every dev/test query and never look at qrels.
"""
from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from mmdd_dataset.wdc_runtime import iter_dataset_artifact

from .catalog import Catalog
from .common import digest, read_json, write_json, write_jsonl
from .stage1 import load_stage1


def holdout_partition(source_group: str, head: dict[str, Any]) -> str:
    """Source-group hash split, so no table family is on both sides of the head's holdout monitor."""
    value = int(hashlib.sha256((head["holdout_hash_salt"] + "|" + source_group).encode()).hexdigest(), 16)
    return "holdout" if value % head["holdout_modulus"] == head["holdout_residue"] else "fit"


def _job(query_id: str, target_id: str, split: str, partition: str, source_group: str,
         evidence: list[str], catalog: Catalog) -> dict[str, Any]:
    return {"pair_id": digest([query_id, target_id]), "query_id": query_id, "target_id": target_id, "split": split,
            "partition": partition, "source_group": source_group, "evidence_ids": evidence,
            "candidate_column_ids": [c["column_index"] for c in catalog.get("target", target_id)["columns"]]}


def build_jobs(config: dict[str, Any], run: Path) -> None:
    paths, scope, depth = config["paths"], config["candidate_scope"], config["output_depth"]
    catalog = Catalog(run)
    population = {row["query_id"]: row for row in read_json(run / "population" / "train.json")}
    stage1 = load_stage1(Path(paths["stage1_handoff"]), "train", scope, depth)

    gold: dict[tuple[str, str], set[int]] = defaultdict(set)
    funnel: Counter = Counter()
    for qrel in iter_dataset_artifact(Path(paths["dataset_root"]), "qrels"):
        if qrel.get("split") != "train" or float(qrel.get("rel", 0)) <= 0:
            continue
        if qrel["reason"] != "model_recoverable_join_column":
            funnel["explicit_qrels_skipped"] += 1
            continue
        query_id, target_id = str(qrel["query_table_id"]), str(qrel["target_table_id"])
        source = str(int(qrel["join_attribute"]["source_column_index"]))
        if source in catalog.get("query_sources", query_id):
            raise ValueError(f"{query_id}: recoverable attribute {source} is visible in the query")
        gold[query_id, target_id].add(catalog.get("target_sources", target_id)[source])

    jobs, labels = [], {}
    for (query_id, target_id), columns in sorted(gold.items()):
        funnel["labeled_pairs"] += 1
        if target_id not in stage1[query_id]["candidates"]:
            funnel["labeled_target_outside_C30"] += 1
            continue
        evidence = stage1[query_id]["evidence"][target_id]
        if len(evidence) > config["max_evidence"]:
            raise ValueError(f"{query_id}->{target_id}: {len(evidence)} evidence > max_evidence")
        group = population[query_id]["source_group"]
        job = _job(query_id, target_id, "train", holdout_partition(group, config["head"]), group, evidence, catalog)
        jobs.append(job)
        labels[job["pair_id"]] = sorted(columns)
        funnel[job["partition"] + "_pairs"] += 1
        funnel["pairs_with_evidence"] += bool(evidence)
    write_jsonl(run / "jobs" / "train.jsonl", jobs)
    write_json(run / "jobs" / "train_labels.json", labels)

    for split in ("dev", "test"):
        stage1 = load_stage1(Path(paths["stage1_handoff"]), split, scope, depth)
        rows = []
        for query in read_json(run / "population" / f"{split}.json"):
            record = stage1[query["query_id"]]
            for target_id in record["candidates"]:
                rows.append(_job(query["query_id"], target_id, split, "evaluation", query["source_group"],
                                 record["evidence"][target_id], catalog))
        write_jsonl(run / "jobs" / f"{split}.jsonl", rows)
        funnel[f"{split}_pairs"] = len(rows)
    write_json(run / "jobs" / "FUNNEL.json", dict(funnel))
    print(dict(funnel), flush=True)
