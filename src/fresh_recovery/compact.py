"""Compact, lossless dictionary-coded storage for rankings and Teacher logits (.cjson.xz).

Every object ID is replaced by its index in one ``ids`` table; float logits are
kept verbatim (JSON round-trips Python floats exactly).  ``load`` returns the
same nested Python structure as the original ``.json.gz``.
"""
from __future__ import annotations

import gzip
import json
import lzma
from pathlib import Path
from typing import Any

from .io import sha256_file


class _Coder:
    def __init__(self) -> None:
        self.ids: dict[str, int] = {}

    def code(self, value: str) -> int:
        idx = self.ids.get(value)
        if idx is None:
            idx = self.ids[value] = len(self.ids)
        return idx


def _encode_rankings(rankings: dict, coder: _Coder) -> dict:
    return {system: {coder.code(q): [coder.code(t) for t in ranking] for q, ranking in rows.items()}
            for system, rows in rankings.items()}


def _decode_rankings(payload: dict, ids: list[str]) -> dict:
    return {system: {ids[int(q)]: [ids[i] for i in ranking] for q, ranking in rows.items()}
            for system, rows in payload.items()}


def _encode_logits(logits: dict, coder: _Coder) -> dict:
    out = {}
    for key, queries in logits.items():
        enc_q = {}
        for q, rec in queries.items():
            enc = {}
            for name in ("f0", "qt_scores_all_U", "qt_scores_k20"):
                if name in rec:
                    enc[name] = {coder.code(t): v for t, v in rec[name].items()}
            if "origin" in rec:
                enc["origin"] = {coder.code(t): v for t, v in rec["origin"].items()}
            for name in ("paths", "swap_paths"):
                if name in rec:
                    enc[name] = {coder.code(t): [[coder.code(e), v] for e, v in slots] for t, slots in rec[name].items()}
            enc_q[coder.code(q)] = enc
        out[key] = enc_q
    return out


def _decode_logits(payload: dict, ids: list[str]) -> dict:
    out = {}
    for key, queries in payload.items():
        dec_q = {}
        for q, rec in queries.items():
            dec = {}
            for name in ("f0", "qt_scores_all_U", "qt_scores_k20"):
                if name in rec:
                    dec[name] = {ids[int(t)]: v for t, v in rec[name].items()}
            if "origin" in rec:
                dec["origin"] = {ids[int(t)]: v for t, v in rec["origin"].items()}
            for name in ("paths", "swap_paths"):
                if name in rec:
                    dec[name] = {ids[int(t)]: [[ids[e], v] for e, v in slots] for t, slots in rec[name].items()}
            dec_q[ids[int(q)]] = dec
        out[key] = dec_q
    return out


def save(path: Path, kind: str, payload: dict) -> str:
    coder = _Coder()
    body = _encode_rankings(payload, coder) if kind == "rankings" else _encode_logits(payload, coder)
    raw = json.dumps({"format": "cjson.v1", "kind": kind, "ids": list(coder.ids), "data": body},
                     separators=(",", ":")).encode("utf-8")
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with lzma.open(tmp, "wb", preset=9 | lzma.PRESET_EXTREME) as handle:
        handle.write(raw)
    tmp.replace(path)
    return sha256_file(path)


def load(path: Path) -> dict:
    with lzma.open(Path(path), "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("format") != "cjson.v1":
        raise ValueError(f"{path}: unsupported format")
    ids = payload["ids"]
    return _decode_rankings(payload["data"], ids) if payload["kind"] == "rankings" else _decode_logits(payload["data"], ids)


def load_any(path_stem: Path, name: str) -> dict:
    """Load ``<name>.cjson.xz`` if present, else ``<name>.json.gz``."""
    compact = Path(path_stem) / f"{name}.cjson.xz"
    if compact.exists():
        return load(compact)
    with gzip.open(Path(path_stem) / f"{name}.json.gz", "rt", encoding="utf-8") as handle:
        return json.load(handle)


def convert_eval_dir(eval_dir: Path) -> dict:
    """Convert both heavy files of one EVAL_* dir; verify round trip; drop the .json.gz."""
    report = {}
    for name, kind in (("rankings", "rankings"), ("teacher_logits", "logits")):
        source = Path(eval_dir) / f"{name}.json.gz"
        target = Path(eval_dir) / f"{name}.cjson.xz"
        if not source.exists():
            if target.exists():
                report[name] = {"source_bytes": 0, "compact_bytes": target.stat().st_size,
                                "source_sha256": "already_converted", "compact_sha256": sha256_file(target),
                                "round_trip": "exact"}
                continue
            raise FileNotFoundError(source)
        with gzip.open(source, "rt", encoding="utf-8") as handle:
            original = json.load(handle)
        sha = save(target, kind, original)
        if load(target) != original:
            target.unlink()
            raise RuntimeError(f"{target}: round trip mismatch")
        report[name] = {"source_bytes": source.stat().st_size, "compact_bytes": target.stat().st_size,
                        "source_sha256": sha256_file(source), "compact_sha256": sha, "round_trip": "exact"}
        source.unlink()
    return report
