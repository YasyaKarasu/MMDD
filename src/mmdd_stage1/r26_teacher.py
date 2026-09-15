"""Fixed T0 pair scores shared across independently retrieved R26 pools."""
from __future__ import annotations

import hashlib
from pathlib import Path
import sqlite3
import time

import torch

from .teacher_rerank import _teacher_scores


def feature_digest(features) -> str:
    """Bind the tensors T0 sees, including grouping; no Student identity enters."""
    digest = hashlib.sha256(features.object_type.encode())
    for name in ("embedding", "hidden_states", "token_groups"):
        value = getattr(features, name)
        digest.update(name.encode())
        if value is None:
            digest.update(b"None")
        else:
            value = value.detach().cpu().contiguous()
            digest.update(str((value.dtype, tuple(value.shape))).encode())
            digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


class TeacherPairCache:
    """One sequential writer; keyed by T0 semantics and actual Q/T features."""

    def __init__(self, path: Path, namespace: str, teacher, store, device: torch.device):
        self.db = sqlite3.connect(path)
        self.db.execute("CREATE TABLE IF NOT EXISTS scores (namespace TEXT, q TEXT, t TEXT, qsha TEXT, tsha TEXT, score REAL, PRIMARY KEY(namespace,q,t,qsha,tsha))")
        self.namespace, self.teacher, self.store, self.device = namespace, teacher, store, device
        self.digests: dict[str, str] = {}
        self.compression = teacher.new_compression_cache()

    def digest(self, object_id: str) -> str:
        if object_id not in self.digests:
            self.digests[object_id] = feature_digest(self.store.get(object_id, include_hidden=True))
        return self.digests[object_id]

    def score(self, query_id: str, targets: list[str]) -> tuple[dict[str, float], dict]:
        started = time.monotonic()
        qsha = self.digest(query_id)
        cached = {t: (tsha, score) for t, tsha, score in self.db.execute(
            "SELECT t,tsha,score FROM scores WHERE namespace=? AND q=? AND qsha=?",
            (self.namespace, query_id, qsha))}
        values, missing = {}, []
        for target in dict.fromkeys(targets):
            tsha = self.digest(target)
            if target in cached and cached[target][0] == tsha:
                values[target] = cached[target][1]
            else:
                missing.append(target)
        lookup_seconds = time.monotonic() - started
        started = time.monotonic()
        if missing:
            scores = _teacher_scores(self.teacher, query_id, missing, self.store, self.device,
                                     batch_size=64, compression_cache=self.compression)
            values.update(zip(missing, scores))
            self.db.executemany("INSERT OR REPLACE INTO scores VALUES (?,?,?,?,?,?)",
                [(self.namespace, query_id, t, qsha, self.digest(t), values[t]) for t in missing])
            self.db.commit()
        return values, {"requested_pairs": len(values), "new_pairs": len(missing),
                        "cached_pairs": len(values)-len(missing), "lookup_and_feature_hash_seconds": lookup_seconds,
                        "new_pair_score_and_write_seconds": time.monotonic()-started}


def rerank_pools(row: dict, scores: dict[str, float]) -> dict[str, list[str]]:
    pools = {"BT100": row["rankings"]["Equal"][:100],
             "D100": row["rankings"]["D100_ANN"],
             "U_OFFLINE": row["U"], "M_OFFLINE": row["M_exact"]}
    result = {"BT100_NO_T0": pools["BT100"], "D100_NO_T0": pools["D100"]}
    for name, targets in pools.items():
        result[name + "_T0"] = sorted(targets, key=lambda t: (-scores[t], t))
    return result
