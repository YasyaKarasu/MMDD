"""Run-local data access for FRESH-RECOVERY v3.1.

Everything here reads only the original dataset, this run's label/alias/PCA
outputs and the approved pure frozen caches recorded in
``PURE_FEATURE_SOURCE.json`` / ``ROW_FEATURE_SOURCE.json``.  No historical task
artifact is opened.
"""
from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch

from fresh_path.features import ContentStore
from fresh_path.score import ObjectBank

from .config import NAMESPACE_PREFIX, Paths, RELATIONS
from .io import iter_jsonl, read_jsonl_gz, sha256_file, write_json

EMBED_DIM = 4096


def utf8_sorted(items: Iterable[str]) -> list[str]:
    return sorted(items, key=lambda value: value.encode("utf-8"))


def namespace(*parts: object) -> str:
    return "|".join([NAMESPACE_PREFIX, *(str(part) for part in parts)])


def local_rng(*parts: object) -> random.Random:
    """SPEC 6: seed = first 8 bytes (big endian) of SHA256(namespace)."""
    digest = hashlib.sha256(namespace(*parts).encode("utf-8")).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def seed_int(*parts: object) -> int:
    digest = hashlib.sha256(namespace(*parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


# ------------------------------------------------------------------ labels ---


@dataclass
class Labels:
    queries: dict[str, dict]
    epos: dict[str, list[str]]
    legal: list[str]
    edge_anchors: list[dict]
    canonical: dict[str, str]
    modality: dict[str, str]
    canonical_text: list[str]
    canonical_image: list[str]
    stats: dict = field(default_factory=dict)
    files: dict[str, str] = field(default_factory=dict)

    @property
    def query_ids(self) -> list[str]:
        return utf8_sorted(self.queries)

    def library(self, relation: str) -> list[str]:
        if relation == "QT":
            return self.legal
        if relation == "Q_text":
            return self.canonical_text
        if relation == "Q_image":
            return self.canonical_image
        if relation == "text_T":
            return self.legal
        if relation == "image_T":
            return self.legal
        raise ValueError(relation)

    def witness_assets(self, query_id: str) -> list[str]:
        entry = self.queries[query_id]
        return utf8_sorted({asset for assets in entry["W"].values() for asset in assets})


def load_labels(paths: Paths) -> Labels:
    labels_dir = paths.labels_dir
    conflicts = labels_dir / "LABEL_CONFLICTS.jsonl"
    if conflicts.exists() and conflicts.stat().st_size:
        raise RuntimeError("STOP_LABEL_CONFLICT: label conflicts are present")
    queries = {row["query_id"]: row for row in read_jsonl_gz(labels_dir / "train_queries.jsonl.gz")}
    epos = {row["asset_id"]: list(row["positive_ids"]) for row in read_jsonl_gz(labels_dir / "asset_epos.jsonl.gz")}
    legal = utf8_sorted(json.loads((labels_dir / "legal_targets.json").read_text(encoding="utf-8")))
    edge_anchors = list(read_jsonl_gz(labels_dir / "edge_anchors.jsonl.gz"))
    canonical = json.loads((labels_dir / "content_canonical.json").read_text(encoding="utf-8"))
    modality: dict[str, str] = {}
    canonical_set: set[str] = set()
    for row in read_jsonl_gz(paths.work_dir / "CONTENT_ALIASES.jsonl.gz"):
        modality[row["asset_id"]] = row["modality"]
        canonical_set.add(row["canonical_id"])
    canonical_text = utf8_sorted(a for a in canonical_set if modality[a] == "text")
    canonical_image = utf8_sorted(a for a in canonical_set if modality[a] == "image")
    stats = json.loads((labels_dir / "label_stats.json").read_text(encoding="utf-8"))
    files = {
        name: sha256_file(labels_dir / name)
        for name in ("train_queries.jsonl.gz", "asset_epos.jsonl.gz", "edge_anchors.jsonl.gz",
                     "legal_targets.json", "content_canonical.json")
    }
    files["CONTENT_ALIASES.jsonl.gz"] = sha256_file(paths.work_dir / "CONTENT_ALIASES.jsonl.gz")
    for row in edge_anchors:
        if row["relation"] not in RELATIONS or not row["positive_ids"]:
            raise ValueError(f"invalid edge anchor {row['item_id']}")
    return Labels(queries=queries, epos=epos, legal=legal, edge_anchors=edge_anchors,
                  canonical=canonical, modality=modality, canonical_text=canonical_text,
                  canonical_image=canonical_image, stats=stats, files=files)


def load_split_gt(paths: Paths, split: str) -> dict[str, dict]:
    """Dev/test GT and query kinds: evaluation entry points only."""
    if split not in ("dev", "test"):
        raise ValueError(split)
    gold: dict[str, set[str]] = {}
    reasons: dict[str, set[str]] = {}
    implicit: dict[str, set[str]] = {}
    for row in iter_jsonl(paths.dataset_root / "qrels.jsonl"):
        if str(row.get("split")) != split or int(row.get("rel", 0)) <= 0:
            continue
        qid = str(row["query_table_id"])
        gold.setdefault(qid, set()).add(str(row["target_table_id"]))
        reasons.setdefault(qid, set()).add(str(row.get("reason")))
        if row.get("reason") == "model_recoverable_join_column":
            implicit.setdefault(qid, set()).add(str(row["target_table_id"]))
    groups: dict[str, str] = {}
    for row in iter_jsonl(paths.dataset_root / "query_tables" / "part-00000.jsonl"):
        if str(row.get("split")) == split:
            groups[str(row["table_id"])] = str(row.get("source_table_id"))
    witness: dict[str, dict[str, set[str]]] = {}
    canonical = json.loads((paths.labels_dir / "content_canonical.json").read_text(encoding="utf-8"))
    for row in iter_jsonl(paths.dataset_root / "evidence_recoveries" / "part-00000.jsonl"):
        if str(row.get("split")) != split:
            continue
        evidence = row.get("evidence") or {}
        raw_asset = str(evidence.get("asset_id"))
        if raw_asset not in canonical:
            raise ValueError(f"unresolved {split} witness asset: {raw_asset}")
        witness.setdefault(str(row["query_table_id"]), {}).setdefault(
            str(row["target_table_id"]), set()).add(canonical[raw_asset])
    out: dict[str, dict] = {}
    for qid in utf8_sorted(gold):
        kinds = reasons[qid]
        if kinds == {"model_recoverable_join_column"}:
            kind = "implicit"
        elif kinds == {"explicit_visible_join_column"}:
            kind = "explicit"
        else:
            kind = "mixed"
        out[qid] = {
            "G": utf8_sorted(gold[qid]),
            "implicit_G": utf8_sorted(implicit.get(qid, ())),
            "kind": kind,
            "source_group": groups.get(qid, qid),
            "W": {t: utf8_sorted(a) for t, a in sorted(witness.get(qid, {}).items())},
        }
    return out


def split_query_ids(paths: Paths, split: str) -> list[str]:
    return utf8_sorted(
        str(row["table_id"])
        for row in iter_jsonl(paths.dataset_root / "query_tables" / "part-00000.jsonl")
        if str(row.get("split")) == split
    )


# --------------------------------------------------------------- features ---


@dataclass
class ZStore:
    ids: list[str]
    types: list[str]
    z: torch.Tensor
    index: dict[str, int]
    sha256: str

    @property
    def dim(self) -> int:
        return int(self.z.shape[1])

    def positions(self, ids: Sequence[str]) -> torch.Tensor:
        return torch.tensor([self.index[i] for i in ids], dtype=torch.long)

    def rows(self, ids: Sequence[str]) -> torch.Tensor:
        return self.z[self.positions(ids)]


def pure_source(paths: Paths) -> dict:
    source = json.loads((paths.work_dir / "PURE_FEATURE_SOURCE.json").read_text(encoding="utf-8"))
    if source.get("status") != "PASS":
        raise RuntimeError("pure frozen feature source is not approved")
    return source


def load_z(paths: Paths, *, only: set[str] | None = None) -> ZStore:
    source = pure_source(paths)
    index = json.loads(Path(source["z_index"]).read_text(encoding="utf-8"))
    ids = [str(x) for x in index["ids"]]
    types = [str(x) for x in index["types"]]
    order = sorted(range(len(ids)), key=lambda i: ids[i].encode("utf-8"))
    if only is not None:
        order = [i for i in order if ids[i] in only]
    raw = np.load(Path(source["z"]), mmap_mode="r", allow_pickle=False)
    positions = np.asarray(order, dtype=np.int64)
    z = torch.from_numpy(np.ascontiguousarray(raw[positions]).astype(np.float32))
    ids_sorted = [ids[i] for i in order]
    types_sorted = [types[i] for i in order]
    contract = paths.work_dir / "pca" / "PCA_RUN_CONTRACT.json"
    z_sha = json.loads(contract.read_text(encoding="utf-8"))["z_sha256"] if contract.exists() else sha256_file(Path(source["z"]))
    return ZStore(ids=ids_sorted, types=types_sorted, z=z,
                  index={oid: i for i, oid in enumerate(ids_sorted)},
                  sha256=z_sha)


def load_content(paths: Paths, *, lru_bytes: int = 4 * 2**30) -> ContentStore:
    source = pure_source(paths)
    return ContentStore(Path(source["content"]), lru_bytes=lru_bytes)


def make_bank(paths: Paths, *, device: str | None = None, only: set[str] | None = None,
              gpu_token_bytes: int = 512 * 2**20) -> ObjectBank:
    z = load_z(paths, only=only)
    content = load_content(paths)
    return ObjectBank(z, content, lru_bytes=4 * 2**30, device=device, gpu_token_bytes=gpu_token_bytes)


# ------------------------------------------------------------ query rows ---


@dataclass
class RowStore:
    rows: np.ndarray            # (total_rows, 4096) float32 memmap
    offsets: dict[str, tuple[int, int]]

    def get(self, query_id: str) -> np.ndarray:
        start, count = self.offsets[query_id]
        return np.asarray(self.rows[start : start + count])


def build_row_store(paths: Paths) -> dict:
    """Materialise the approved query-row cache as one memmap (pure copy)."""
    source = json.loads((paths.work_dir / "ROW_FEATURE_SOURCE.json").read_text(encoding="utf-8"))
    if source.get("status") != "PASS":
        raise RuntimeError("row feature source is not approved")
    index = json.loads((paths.work_dir / "ROW_INDEX.json").read_text(encoding="utf-8"))["queries"]
    out_dir = paths.work_dir / "rows"
    out_dir.mkdir(parents=True, exist_ok=True)
    total = sum(len(row["encoded_row_ids"]) for row in index)
    array = np.lib.format.open_memmap(out_dir / "rows.f32.npy", mode="w+", dtype=np.float32,
                                      shape=(total, EMBED_DIM))
    offsets: dict[str, list[int]] = {}
    cursor = 0
    for row in sorted(index, key=lambda r: r["query_id"].encode("utf-8")):
        payload = torch.load(row["feature_path"], map_location="cpu", weights_only=True)
        vectors = payload["row_embeddings"].float().numpy()
        if vectors.shape != (len(row["encoded_row_ids"]), EMBED_DIM):
            raise ValueError(f"{row['query_id']}: row shape {vectors.shape}")
        array[cursor : cursor + len(vectors)] = vectors
        offsets[row["query_id"]] = [cursor, len(vectors)]
        cursor += len(vectors)
    array.flush()
    del array
    write_json(out_dir / "index.json", {"offsets": offsets, "total_rows": total,
                                        "source": source["cache_dir"],
                                        "permitted_payload": "row_embeddings_only"})
    return {"queries": len(offsets), "rows": total,
            "sha256": sha256_file(out_dir / "rows.f32.npy")}


def load_row_store(paths: Paths) -> RowStore:
    out_dir = paths.work_dir / "rows"
    index = json.loads((out_dir / "index.json").read_text(encoding="utf-8"))
    rows = np.load(out_dir / "rows.f32.npy", mmap_mode="r", allow_pickle=False)
    return RowStore(rows=rows, offsets={k: (int(v[0]), int(v[1])) for k, v in index["offsets"].items()})


# ------------------------------------------------------------------- PCA ---


def load_basis(paths: Paths) -> tuple[torch.Tensor, str]:
    report = json.loads((paths.work_dir / "PCA_REPORT.json").read_text(encoding="utf-8"))
    if report.get("status") != "PASS":
        raise RuntimeError("PCA report is not PASS")
    basis_path = paths.work_dir / "pca" / "basis.pt"
    payload = torch.load(basis_path, map_location="cpu", weights_only=True)
    basis = payload["basis"].float()
    if basis.shape != (1024, EMBED_DIM):
        raise ValueError(f"unexpected basis shape {tuple(basis.shape)}")
    return basis, sha256_file(basis_path)
