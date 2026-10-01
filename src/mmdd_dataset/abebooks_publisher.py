"""Publisher-brand equality and content-bound cover qualification for AbeBooks."""
from __future__ import annotations

import re
import unicodedata
from collections import defaultdict

from .abebooks_curation import cell_values


PUBLISHER_POLICY = "publisher_brand_aliases_v1_no_parent_imprint_equivalence"


def publisher_key(value: str) -> str:
    """Unify spelling/location suffixes within brands, never parent and imprint."""
    text = unicodedata.normalize("NFKC", str(value)).casefold()
    text = re.sub(r"\(\s*edition\b[^)]*\)", "", text)
    key = re.sub(r"[^a-z0-9]", "", text)
    if key in {"", "0", "na", "none", "null", "unknown"}:
        return ""
    # Specific imprints precede parent companies. These are dataset matching
    # rules, not claims that an imprint's parent logo is evidence of the imprint.
    brands = (
        (r"morgankaufman+n?", "morgankaufmann"),
        (r"addisonwesley", "addisonwesley"),
        (r"butterworthheinemann", "butterworthheinemann"),
        (r"academicpr(?:ess)?(?:inc)?", "academicpress"),
        (r"digitalpr(?:ess)?", "digitalpress"),
        (r"prenticehall", "prenticehall"),
        (r"osborne", "osborne"),
        (r"irwin", "irwin"),
        (r"newnes", "newnes"),
        (r"routledge", "routledge"),
        (r"crcpress", "crcpress"),
        (r"taylor(?:and)?francis", "taylorfrancis"),
        (r"pearson", "pearson"),
        (r"elsevier", "elsevier"),
        (r"mcgrawhill", "mcgrawhill"),
        (r"emerald", "emerald"),
    )
    for pattern, brand in brands:
        if re.search(pattern, key):
            return brand
    return key


def visible_publisher_hint(title: str, value: str) -> bool:
    """Detect brand text already visible in the query, including common aliases."""
    key = publisher_key(value)
    visible = re.sub(r"[^a-z0-9]", "", unicodedata.normalize("NFKC", title).casefold())
    return bool(key) and (key in visible or publisher_key(title) == key)


def collect_publisher_facts(data: dict, proposals: list[dict], reviews: list[dict]) -> tuple[list[dict], list[dict]]:
    """Use content-bound pixel reviews; keep blind reader errors in the audit."""
    from .abebooks_standalone import cover_title_matches

    sources = {(s["source_table_id"], r["row_id"]): cell_values(r)
               for s in data["source_tables"] for r in s["rows"]}
    reviewed = {r["asset_id"]: r for r in reviews}
    audit, by_row = [], defaultdict(list)
    for p in proposals:
        loc = p["source_table_id"], p["source_row_id"]
        source = sources[loc]
        response = p.get("response") or {}
        review = reviewed.get(p["asset_id"])
        reason = "awaiting_pixel_review"
        observed = review.get("observed_publisher", "") if review else ""
        if p.get("source_answer_provided") or p.get("attribute") != "publisher":
            reason = "not_a_blind_publisher_reading"
        elif not cover_title_matches((review or {}).get("observed_title", response.get("title", "")), source.get("title", "")):
            reason = "cover_title_not_verified"
        elif review:
            if review["content_sha256"] != p["content_sha256"]:
                raise ValueError("Publisher review does not match the evidence bytes")
            if review["status"] != "supported":
                reason = review["status"]
            elif not publisher_key(observed) or publisher_key(observed) != publisher_key(source.get("publisher", "")):
                reason = "visible_publisher_differs_from_source_brand"
            else:
                reason = "pixel_review_supports_source_brand"
        record = {**p, "column_name": "publisher", "source_publisher": source.get("publisher", ""),
                  "observed_publisher": observed, "review": review, "status": reason,
                  "qualified": reason == "pixel_review_supports_source_brand"}
        audit.append(record)
        if record["qualified"]:
            by_row[loc].append(record)
    facts = [{"source_table_id": sid, "source_row_id": rid, "column_name": "publisher",
              "original_value": sources[sid, rid]["publisher"],
              "evidence_ids": [r["asset_id"] for r in records],
              "observed_values": {r["asset_id"]: r["observed_publisher"] for r in records},
              "strength": "codex_pixel_review_after_blind_local_proposal",
              "annotation_status": "model_assisted_publisher_brand_evidence", "human_reviewed": False}
             for (sid, rid), records in sorted(by_row.items())]
    return facts, audit
