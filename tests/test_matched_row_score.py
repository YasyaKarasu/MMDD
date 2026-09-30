from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mmdd_stage2.matched_row_score import (
    DictVectorStore,
    UnitCosineMatcher,
    bridge_row_scores,
    header_key,
    reference_bridge_row_scores,
    semantic_allowed,
    value_info,
)


def bridge(attribute: str, rows: list[str | None], statuses: list[str] | None = None) -> dict:
    slots, domain = [], {}
    statuses = statuses or ["VALUE"] * len(rows)
    for row_id, (value, status) in enumerate(zip(rows, statuses, strict=True)):
        if status == "VALUE":
            info = value_info(value)
            domain[info["key"]] = info
            slots.append({"row_id": row_id, "status": "VALUE", "value_key": info["key"]})
        else:
            slots.append({"row_id": row_id, "status": status})
    return {"attribute": attribute, "slots": slots, "domain": list(domain.values())}


def table(header: str, values: list[str]) -> dict:
    return {"columns": [{"column_id": 0, "column_name": header, "values": [value_info(v) for v in values]}]}


def store(**vectors: list[float]) -> DictVectorStore:
    return DictVectorStore({name.replace("_", " "): np.asarray(vector, dtype=np.float32)
                            for name, vector in vectors.items()})


def test_typed_keys_separate_numbers_dates_urls_and_identifiers():
    assert value_info("1,234.50")["key"] == "NUMBER:1234.5"
    assert value_info("42")["key"] == value_info("42.0")["key"]
    assert value_info("7%")["key"] == "PERCENT:7"
    assert value_info("2026-01-02")["key"] == "DATE:2026-01-02"
    assert value_info("https://example.org/a")["kind"] == "URL"
    assert value_info("007")["key"] == "ID:007"  # leading zeros stay opaque, never collapse to 7
    assert value_info("--")["kind"] == "EMPTY"
    assert value_info("Acme Ltd")["kind"] == "TEXT"


def test_only_text_pairs_with_equal_numeric_signature_may_soften():
    assert semantic_allowed(value_info("Acme Ltd"), value_info("Acme Limited"))
    assert not semantic_allowed(value_info("Acme 1"), value_info("Acme 2"))
    assert not semantic_allowed(value_info("42"), value_info("42"))
    assert not semantic_allowed(value_info("https://a/b"), value_info("https://a/b"))


def test_exact_typed_hits_never_touch_the_embedding_store():
    class ExplodingStore(DictVectorStore):
        def vector(self, text: str) -> np.ndarray:
            raise AssertionError("embedding store must not be consulted for typed exact hits")

    matcher = UnitCosineMatcher(ExplodingStore({}))
    values = [value_info("42")]
    assert matcher.best(value_info("42.0"), values)["matched"]
    assert matcher.best(value_info("42.0"), values)["match_kind"] == "TYPED_EXACT"


def test_guarded_cosine_requires_the_threshold_and_a_matching_signature():
    matcher = UnitCosineMatcher(store(alpha=[1.0, 0.0], alpha_ltd=[1.0, 0.0], beta=[0.0, 1.0],
                                      alpha_7=[1.0, 0.0], beta_7=[0.0, 1.0]))
    assert matcher.best(value_info("alpha"), [value_info("alpha ltd")])["matched"]
    assert not matcher.best(value_info("alpha"), [value_info("beta")])["matched"]
    # equal digit signatures but far apart in embedding space: still no match
    assert not matcher.best(value_info("alpha 7"), [value_info("beta 7")])["matched"]
    # different digit signatures are never softened, even at cosine 1
    assert not matcher.best(value_info("alpha 7"), [value_info("alpha ltd 8")])["matched"]


def test_repeated_recovered_value_keeps_its_row_multiplicity():
    # 4 alpha rows + 1 beta row; only alpha matches -> 4/5, not one distinct match out of five
    matcher = UnitCosineMatcher(store(alpha=[1.0, 0.0], beta=[0.0, 1.0]))
    recovery = bridge("name", ["alpha", "alpha", "alpha", "alpha", "beta"])
    scored = bridge_row_scores(["t1"], [recovery], {"t1": table("name", ["alpha"])}, matcher)
    assert scored["table_scores"]["t1"] == pytest.approx(0.8)
    assert scored["details"][0]["denominator_rows"] == 5
    assert scored["details"][0]["matched_rows"] == 4


def test_null_and_conflict_rows_score_zero_but_keep_the_denominator():
    matcher = UnitCosineMatcher(store(alpha=[1.0, 0.0]))
    recovery = bridge("name", ["alpha", None, None, None, None], ["VALUE", "MISSING", "CONFLICT", "MISSING", "CONFLICT"])
    scored = bridge_row_scores(["t1"], [recovery], {"t1": table("name", ["alpha"])}, matcher)
    assert scored["table_scores"]["t1"] == pytest.approx(0.2)


def test_one_target_value_can_answer_several_query_rows():
    matcher = UnitCosineMatcher(store(alpha=[1.0, 0.0]))
    recovery = bridge("name", ["alpha"] * 5)
    scored = bridge_row_scores(["t1"], [recovery], {"t1": table("name", ["alpha"])}, matcher)
    assert scored["table_scores"]["t1"] == pytest.approx(1.0)


def test_bridge_tier_requires_matching_attribute_and_five_unique_rows():
    matcher = UnitCosineMatcher(store(alpha=[1.0, 0.0]))
    recovery = bridge("name", ["alpha"] * 5)
    other = {"columns": [{"column_id": 0, "column_name": "other", "values": [value_info("alpha")]}]}
    assert bridge_row_scores(["t1"], [recovery], {"t1": other}, matcher)["table_scores"]["t1"] == 0.0
    broken = {"attribute": "name", "slots": recovery["slots"][:4], "domain": recovery["domain"]}
    with pytest.raises(ValueError, match="five unique row slots"):
        bridge_row_scores(["t1"], [broken], {"t1": table("name", ["alpha"])}, matcher)


def test_independent_row_loop_agrees_with_the_scoring_path():
    vectors = store(alpha=[1.0, 0.0], beta=[0.0, 1.0], alpha_ltd=[1.0, 0.0])
    matcher = UnitCosineMatcher(vectors)
    recovery = bridge("name", ["alpha", "alpha", "beta", "alpha 7", None],
                      ["VALUE", "VALUE", "VALUE", "VALUE", "MISSING"])
    tables = {"t1": table("name", ["alpha ltd", "beta", "alpha 8"]),
              "t2": table("name", ["beta"])}
    scored = bridge_row_scores(["t1", "t2"], [recovery], tables, matcher)["table_scores"]
    reference = reference_bridge_row_scores(["t1", "t2"], [recovery], tables, vectors)
    assert scored == reference
    # 'alpha 7' must not match 'alpha 8' (different numeric token) and must not match 'alpha ltd'
    assert scored["t1"] == pytest.approx(0.6)
    assert scored["t2"] == pytest.approx(0.2)


def test_header_key_is_the_attribute_key_used_by_the_bridge_tier():
    assert header_key("  Name ") == "name"
    assert header_key("Entity_URL") == "entity_url"
