"""Typed cell values, recovered-value parsing and value-only bridges.

Every comparison in Stage 2 goes through ``value_info``: typed keys (NUMBER, PERCENT, DATE, URL,
SYMBOL, ID) compare exactly; only TEXT values may later be softened by the MiniLM matcher.
"""
from __future__ import annotations

import datetime
import json
import re
import unicodedata
from collections import defaultdict
from decimal import Decimal
from typing import Any

from .common import digest

ROWS = 5  # every query table holds exactly five rows; bridges and scores divide by it
NULLS = frozenset(["", "-", "--", "–", "—", "null", "none", "nan", "n/a"])
DASHES = str.maketrans({c: "-" for c in "‐‑‒–—−"})
NUMBER = re.compile(r"^[+-]?(?:0|[1-9]\d*|[1-9]\d{0,2}(?:,\d{3})+)(?:\.\d+)?$")


def norm(text: str) -> str:
    """NFKC, unified dashes, collapsed whitespace, casefold. Also the column-attribute key."""
    return " ".join(unicodedata.normalize("NFKC", text).translate(DASHES).split()).casefold()


def _decimal(text: str) -> str:
    value = Decimal(text.replace(",", ""))
    return "0" if value == 0 else format(value.normalize(), "f")


def value_info(raw: str) -> dict[str, Any]:
    """``key`` is None exactly for empty cells; TEXT values carry their ordered numeric tokens."""
    n = norm(raw)
    if n in NULLS:
        return {"key": None, "text": n, "kind": "EMPTY", "digits": []}
    if NUMBER.fullmatch(n):
        k = _decimal(n)
        return {"key": "NUMBER:" + k, "text": k, "kind": "NUMBER", "digits": [k]}
    if n.endswith("%") and NUMBER.fullmatch(n[:-1]):
        k = _decimal(n[:-1])
        return {"key": "PERCENT:" + k, "text": k + "%", "kind": "PERCENT", "digits": [k]}
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", n):
        try:
            k = datetime.date.fromisoformat(n).isoformat()
            return {"key": "DATE:" + k, "text": k, "kind": "DATE", "digits": []}
        except ValueError:
            pass
    if n.startswith(("http://", "https://")):
        kind = "URL"
    elif re.fullmatch(r"\\[a-z]", n) or re.fullmatch(r"u\+[0-9a-f]{4,6}", n) or not any(c.isalnum() for c in n):
        kind = "SYMBOL"
    elif re.fullmatch(r"[+-]?\d+(?:\.\d+)?", n):
        kind = "ID"  # numeric identifiers keep leading zeros and stay exact-only
    else:
        kind = "TEXT"
    digits = [_decimal(x) for x in re.findall(r"\d+(?:\.\d+)?", n)] if kind == "TEXT" else []
    return {"key": kind + ":" + n, "text": n, "kind": kind, "digits": digits}


def parse_completion(raw: str) -> tuple[str, str | None]:
    """ROW1 answer: one JSON string or null, optionally inside one markdown fence."""
    text = raw.strip().removeprefix("﻿").strip()
    fenced = re.fullmatch(r"```(?:json)?\s*\n?([\s\S]*?)\n?```", text, re.I)
    if fenced:
        text = fenced[1].strip()
    try:
        value = json.loads(text)
    except (ValueError, TypeError):
        return "PARSE_ERROR", None
    if value is None:
        return "INSUFFICIENT_EVIDENCE", None
    if not isinstance(value, str) or not value.strip():
        return "PARSE_ERROR", None
    return ("VALUE", value) if value_info(value)["key"] else ("NORMALIZED_EMPTY", None)


def query_rows(table: dict[str, Any]) -> list[dict[str, Any]]:
    """Five query rows as ``{query_row_id, cells: [{column_id, column_name, text}]}``."""
    names = {int(c["column_index"]): c["column_name"] for c in table["columns"]}
    return [{"query_row_id": i,
             "cells": [{"column_id": int(c["column_index"]), "column_name": names[int(c["column_index"])],
                        "text": "" if c.get("text") is None else str(c["text"])} for c in row["cells"]]}
            for i, row in enumerate(table["rows"])]


def table_domain(table: dict[str, Any]) -> dict[str, Any]:
    """Every column of a lake table as its distinct typed values (the matching domain)."""
    columns = []
    for column in table["columns"]:
        cid = int(column["column_index"])
        values: dict[str, dict[str, Any]] = {}
        for rid, row in enumerate(table["rows"]):
            cells = [c for c in row["cells"] if int(c["column_index"]) == cid]
            if not cells:
                continue
            info = value_info("" if cells[0].get("text") is None else str(cells[0]["text"]))
            if info["key"] is not None:
                values.setdefault(info["key"], {**info, "row_ids": []})["row_ids"].append(rid)
        columns.append({"column_id": cid, "column_name": column["column_name"], "attribute": norm(column["column_name"]),
                        "values": [values[k] for k in sorted(values)]})
    return {"target_id": table["table_id"], "columns": columns}


def build_bridges(query_id: str, predictions: list[dict[str, Any]], tables: dict[str, Any]) -> list[dict[str, Any]]:
    """Group recovered VALUE claims by attribute into five row slots.

    A row whose claims disagree on the typed key becomes CONFLICT and never votes; the attribute is
    the normalized header of the target column the claim was requested for.
    """
    claims: dict[str, list] = defaultdict(list)
    for p in predictions:
        if p["status"] != "VALUE":
            continue
        column = next(c for c in tables[p["target_id"]]["columns"] if c["column_id"] == p["column_id"])
        claims[norm(column["column_name"])].append((p, value_info(p["value"])))
    bridges = []
    for attribute, items in sorted(claims.items()):
        slots, domain, origins = [], {}, {}
        for rid in range(ROWS):
            row = [(p, v) for p, v in items if p["query_row_id"] == rid]
            keys = sorted({v["key"] for _, v in row})
            ids = sorted(p["unit_id"] for p, _ in row)
            if len(keys) == 1:
                domain[keys[0]] = row[0][1]
                slots.append({"row_id": rid, "status": "VALUE", "value_key": keys[0], "claim_ids": ids})
                origins[str(rid)] = sorted({p["target_id"] for p, _ in row})
            else:
                slots.append({"row_id": rid, "status": "CONFLICT" if keys else "MISSING",
                              "conflict_values": keys, "claim_ids": ids})
        recovered = sum(s["status"] == "VALUE" for s in slots)
        bridges.append({"bridge_id": digest([query_id, attribute, "VALUE_ONLY"]), "query_id": query_id,
                        "attribute": attribute, "policy": "VALUE_ONLY", "m_rows": ROWS, "r_recovered": recovered,
                        "recovery_fraction": recovered / ROWS, "domain": [domain[k] for k in sorted(domain)],
                        "slots": slots, "origin_targets_by_row": origins,
                        "claim_ids": sorted(p["unit_id"] for p, _ in items)})
    return bridges
