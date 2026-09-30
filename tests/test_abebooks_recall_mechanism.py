from audit_abebooks_recall_mechanism import query_mechanism


def test_evidence_hit_requires_nonvisible_value_and_annotated_retained_witness():
    query = {"table_id": "q", "split": "test", "rows": [{"cells": [{"text": "A Book"}]}]}
    fact = {"query_row_id": 0, "target_table_id": "gold",
            "recovered_attribute": {"column_name": "authors", "value": "Tom Clark"},
            "evidence": {"asset_id": "original"}}
    pool = {"D100_ANN": [f"n{i}" for i in range(10)] + ["gold"], "C150": ["gold"],
            "retained_bags": {"gold": ["canonical"]}}
    result = query_mechanism(query, {"gold"}, [fact], pool, {"original": "canonical"})
    assert result["new_witnessed_nonvisible_recall10"] == 1
    assert result["RRF_recall10"] == 1 and result["Direct_recall10"] == 0
    query["rows"][0]["cells"][0]["text"] = "A Book by TOM-CLARK"
    result = query_mechanism(query, {"gold"}, [fact], pool, {"original": "canonical"})
    assert result["any_known_value_visible"]
    assert result["new_witnessed_nonvisible_recall10"] == 0
    assert result["RRF_recall10"] == 1


def test_unlabelled_path_does_not_count_as_verified_but_recall_is_unchanged():
    query = {"table_id": "q", "split": "test", "rows": [{"cells": [{"text": "A Book"}]}]}
    fact = {"query_row_id": 0, "target_table_id": "gold",
            "recovered_attribute": {"column_name": "authors", "value": "Tom Clark"},
            "evidence": {"asset_id": "known"}}
    pool = {"D100_ANN": [], "C150": ["gold"], "retained_bags": {"gold": ["unlabelled"]}}
    result = query_mechanism(query, {"gold", "other"}, [fact], pool, {"known": "known"})
    assert result["RRF_recall10"] == 0.5
    assert result["witnessed_nonvisible_recall10"] == 0
