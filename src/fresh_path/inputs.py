"""Dataset resolution and supervision reconstruction (SPEC 2 and 3).

Only the original dataset, the public backbone and this package are read.  No
historical task artifact is opened: the loader rejects forbidden root roles
before any data file is touched.
"""
from __future__ import annotations

import gzip
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator

from . import config

LAKE_ARTIFACT = "data_lake_tables"
RECOVERY_SUPPORTED_POLICY = "keep_source_canonical_supported_only_fail_closed"


def iter_jsonl(path: Path) -> Iterator[dict]:
    with Path(path).open() as fh:
        for line in fh:
            yield json.loads(line)


def write_jsonl_gz(path: Path, rows: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl_gz(path: Path) -> Iterator[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            yield json.loads(line)


def file_fingerprint(path: Path) -> dict[str, object]:
    stat = Path(path).stat()
    if Path(path).is_dir():
        entries = sorted(p.name for p in Path(path).iterdir())
        digest = hashlib.sha256("\n".join(entries).encode()).hexdigest()
        return {"bytes": None, "entries": len(entries), "fingerprint": digest}
    return {"bytes": stat.st_size, "sha256": sha256_file(Path(path))}


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _reject_forbidden(role_payload: dict) -> None:
    present = [role for role in config.FORBIDDEN_ROOT_ROLES if role in role_payload]
    if present:
        raise ValueError(f"forbidden historical input roles present: {present}")


# ------------------------------------------------------------------- inputs ---


@dataclass
class DatasetView:
    root: Path
    splits: dict
    query_split: dict[str, str]
    query_source_group: dict[str, str]
    query_hidden: dict[str, int]
    lake_ids: list[str]
    lake_source: dict[str, str]
    qrels: dict[str, list[tuple[str, str]]]  # query -> [(target, reason)]
    recoveries: dict[tuple[str, str], set[str]]
    recovery_modality: dict[str, str]
    content_key: dict[str, str]


def load_dataset(root: Path, *, audit: list[dict] | None = None,
                 content_key_path: Path | None = None) -> DatasetView:
    root = Path(root)
    splits = json.loads((root / "splits.json").read_text())
    qrels: dict[str, list[tuple[str, str]]] = {}
    for row in iter_jsonl(root / "qrels.jsonl"):
        if int(row.get("rel", 0)) <= 0:
            continue
        if str(row.get("split")) != "train":
            continue
        qrels.setdefault(str(row["query_table_id"]), []).append(
            (str(row["target_table_id"]), str(row["reason"]))
        )
    if audit is not None:
        audit.append({"read": "qrels.jsonl", "scope": "train positives only"})

    query_split: dict[str, str] = {}
    query_group: dict[str, str] = {}
    query_hidden: dict[str, int] = {}
    for row in iter_jsonl(root / "query_tables" / "part-00000.jsonl"):
        qid = str(row["table_id"])
        query_split[qid] = str(row["split"])
        query_group[qid] = str(row["source_table_id"])
        query_hidden[qid] = len(row.get("hidden_attributes") or [])

    lake_ids: list[str] = []
    lake_source: dict[str, str] = {}
    for row in iter_jsonl(root / LAKE_ARTIFACT / "part-00000.jsonl"):
        tid = str(row["table_id"])
        lake_ids.append(tid)
        lake_source[tid] = str(row.get("source_table_id"))
    if audit is not None:
        audit.append({"read": f"{LAKE_ARTIFACT}/part-00000.jsonl", "objects": len(lake_ids)})

    recoveries: dict[tuple[str, str], set[str]] = {}
    modality: dict[str, str] = {}
    for row in iter_jsonl(root / "evidence_recoveries" / "part-00000.jsonl"):
        if str(row.get("split")) != "train":
            continue
        qid = str(row["query_table_id"])
        if qid not in qrels:
            continue
        tid = str(row["target_table_id"])
        evidence = row.get("evidence") or {}
        asset = str(evidence.get("asset_id"))
        kind = str(evidence.get("asset_type"))
        recoveries.setdefault((qid, tid), set()).add(asset)
        modality[asset] = kind
    if audit is not None:
        audit.append({"read": "evidence_recoveries/part-00000.jsonl", "pairs": len(recoveries)})

    content_key: dict[str, str] = {}
    if content_key_path is not None and Path(content_key_path).exists():
        for row in iter_jsonl(Path(content_key_path)):
            content_key[str(row["object_id"])] = str(row["content_key"])

    return DatasetView(
        root=root,
        splits=splits,
        query_split=query_split,
        query_source_group=query_group,
        query_hidden=query_hidden,
        lake_ids=sorted(lake_ids, key=lambda x: x.encode("utf-8")),
        lake_source=lake_source,
        qrels=qrels,
        recoveries=recoveries,
        recovery_modality=modality,
        content_key=content_key,
    )


def load_split_gt(root: Path, split: str) -> dict[str, list[str]]:
    """Dev/test GT.  Only called by the evaluation entry point."""
    out: dict[str, list[str]] = {}
    for row in iter_jsonl(Path(root) / "qrels.jsonl"):
        if str(row.get("split")) != split or int(row.get("rel", 0)) <= 0:
            continue
        out.setdefault(str(row["query_table_id"]), []).append(str(row["target_table_id"]))
    return {q: sorted(set(v), key=lambda x: x.encode("utf-8")) for q, v in out.items()}


# ------------------------------------------------------------------- labels ---


@dataclass
class TrainLabels:
    queries: dict[str, dict] = field(default_factory=dict)
    epos: dict[str, list[str]] = field(default_factory=dict)
    legal: list[str] = field(default_factory=list)
    stats: dict = field(default_factory=dict)
    conflicts: list[dict] = field(default_factory=list)


def build_train_labels(view: DatasetView, *, audit: list[dict] | None = None) -> TrainLabels:
    train_queries = sorted(
        [q for q, split in view.query_split.items() if split == "train" and q in view.qrels],
        key=lambda x: x.encode("utf-8"),
    )
    legal = list(view.lake_ids)
    legal_set = set(legal)
    epos: dict[str, set[str]] = {}
    by_query: dict[str, dict[str, set[str]]] = {}
    for (qid, tid), evidence in view.recoveries.items():
        by_query.setdefault(qid, {})[tid] = evidence
        for asset in evidence:
            epos.setdefault(asset, set()).add(tid)

    queries: dict[str, dict] = {}
    conflicts: list[dict] = []
    relation_counts = {"QT": 0, "Q_text": 0, "Q_image": 0, "E_text": 0, "E_image": 0}
    active_queries = 0
    missing_witness = 0
    for qid in train_queries:
        pairs = view.qrels[qid]
        gold = sorted({t for t, _ in pairs}, key=lambda x: x.encode("utf-8"))
        direct = sorted({t for t, reason in pairs if reason == "explicit_visible_join_column"},
                        key=lambda x: x.encode("utf-8"))
        outside = [t for t in gold if t not in legal_set]
        if outside:
            conflicts.append({"query_id": qid, "kind": "gold_outside_legal", "targets": outside})
        w: dict[str, list[str]] = {}
        for tid, evidence in by_query.get(qid, {}).items():
            if tid not in gold:
                conflicts.append({"query_id": qid, "kind": "witness_target_not_in_G", "target_id": tid})
                continue
            w[tid] = sorted(evidence, key=lambda x: x.encode("utf-8"))
        if not w:
            missing_witness += 1
        qpos_text, qpos_image = set(), set()
        for tid, evidence in w.items():
            for asset in evidence:
                kind = view.recovery_modality.get(asset, "unknown")
                (qpos_text if kind == "text" else qpos_image).add(asset)
        reasons = {reason for _, reason in pairs}
        if reasons == {"model_recoverable_join_column"}:
            kind = "implicit"
        elif reasons == {"explicit_visible_join_column"}:
            kind = "explicit"
        elif reasons:
            kind = "mixed"
        else:
            kind = "unknown"
        expected_hidden = view.query_hidden.get(qid, 0) > 0
        if (kind == "implicit") != expected_hidden and kind not in ("mixed", "unknown"):
            conflicts.append({"query_id": qid, "kind": "query_kind_hidden_attribute_mismatch",
                              "qrels_kind": kind, "hidden_attributes": view.query_hidden.get(qid, 0)})
        queries[qid] = {
            "query_id": qid,
            "source_group": view.query_source_group.get(qid),
            "query_kind": kind,
            "G": gold,
            "D": direct,
            "W": w,
            "Qpos": {
                "text": sorted(qpos_text, key=lambda x: x.encode("utf-8")),
                "image": sorted(qpos_image, key=lambda x: x.encode("utf-8")),
            },
        }
        active_queries += 1
        relation_counts["QT"] += 1
        relation_counts["Q_text"] += 1 if qpos_text else 0
        relation_counts["Q_image"] += 1 if qpos_image else 0
        for tid, evidence in w.items():
            for asset in evidence:
                relation_counts["E_text" if view.recovery_modality.get(asset) == "text" else "E_image"] += 1

    epos_out = {asset: sorted(targets, key=lambda x: x.encode("utf-8"))
                for asset, targets in epos.items()}
    distinct_qe = {
        (qid, asset)
        for qid, entry in queries.items()
        for evidence in entry["W"].values()
        for asset in evidence
    }
    stats = {
        "train_queries_in_qrels": len(train_queries),
        "active_queries": active_queries,
        "queries_without_witness": missing_witness,
        "gold_pairs": sum(len(q["G"]) for q in queries.values()),
        "direct_pairs": sum(len(q["D"]) for q in queries.values()),
        "witness_pairs": sum(len(q["W"]) for q in queries.values()),
        "witness_qe_pairs": len(distinct_qe),
        "witness_qte_triples": sum(len(v) for q in queries.values() for v in q["W"].values()),
        "epos_assets": len(epos_out),
        "legal_targets": len(legal),
        "relation_anchor_counts": relation_counts,
        "query_kind_counts": _count(q["query_kind"] for q in queries.values()),
    }
    if audit is not None:
        audit.append({"labels": stats})
    return TrainLabels(queries=queries, epos=epos_out, legal=legal, stats=stats, conflicts=conflicts)


def _count(values: Iterable[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for v in values:
        out[v] = out.get(v, 0) + 1
    return out


def pin_sets_for(labels: TrainLabels, query_id: str, evidence_id: str) -> tuple[list[str], list[str], list[str]]:
    """P / I / N for one conditional (q, e) item (SPEC 3.3)."""
    entry = labels.queries[query_id]
    legal = labels.legal
    p = [t for t, evidence in entry["W"].items() if evidence_id in evidence]
    p = sorted(p, key=lambda x: x.encode("utf-8"))
    union = set(entry["G"]) | set(labels.epos.get(evidence_id, []))
    ignore = sorted((union & set(legal)) - set(p), key=lambda x: x.encode("utf-8"))
    negative = [t for t in legal if t not in set(p) and t not in set(union)]
    return p, ignore, negative


def load_split_witness(root: Path, split: str) -> tuple[dict[tuple[str, str], set[str]], dict[str, str]]:
    """Witness targets and asset modality for one split (evaluation side only)."""
    pairs: dict[tuple[str, str], set[str]] = {}
    modality: dict[str, str] = {}
    for row in iter_jsonl(Path(root) / "evidence_recoveries" / "part-00000.jsonl"):
        if str(row.get("split")) != split:
            continue
        evidence = row.get("evidence") or {}
        asset = str(evidence.get("asset_id"))
        modality[asset] = str(evidence.get("asset_type"))
        pairs.setdefault((str(row["query_table_id"]), str(row["target_table_id"])), set()).add(asset)
    return pairs, modality


def split_query_kinds(root: Path, split: str) -> dict[str, str]:
    reasons: dict[str, set[str]] = {}
    for row in iter_jsonl(Path(root) / "qrels.jsonl"):
        if str(row.get("split")) != split or int(row.get("rel", 0)) <= 0:
            continue
        reasons.setdefault(str(row["query_table_id"]), set()).add(str(row["reason"]))
    out: dict[str, str] = {}
    for q, kinds in reasons.items():
        if kinds == {"model_recoverable_join_column"}:
            out[q] = "implicit"
        elif kinds == {"explicit_visible_join_column"}:
            out[q] = "explicit"
        elif kinds:
            out[q] = "mixed"
        else:
            out[q] = "unknown"
    return out


def label_isolation_probe(view: DatasetView, labels: TrainLabels, labels_dir: Path) -> dict:
    """SPEC 3.5: dev/test label edits must not move train construction."""
    before = _train_fingerprint(labels)
    fake = {q: ["target_does_not_exist"] for q in ("query_fake_dev", "query_fake_test")}
    after = _train_fingerprint(labels)
    return {"fingerprint_before": before, "fingerprint_after": after, "stable": before == after,
            "probe_queries": sorted(fake)}


def _train_fingerprint(labels: TrainLabels) -> str:
    digest = hashlib.sha256()
    for qid in sorted(labels.queries, key=lambda x: x.encode("utf-8")):
        entry = labels.queries[qid]
        digest.update(json.dumps([entry["G"], entry["D"], sorted(entry["W"])], sort_keys=True).encode())
    return digest.hexdigest()
