"""Reconstruct R28 metrics and pool membership from saved scores and rankings."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path

from prepare_stage1_r27 import record, rows, sha, write_json
from prepare_stage1_r28 import OUT, FAMILIES, inputs
from evaluate_stage1_r28_student import OWN


def keyed(path: Path, keys: tuple[str, ...]) -> dict:
    result = {}
    for row in rows(path):
        key = tuple(row[k] for k in keys)
        assert key not in result, (path, key)
        result[key] = row
    return result


def check_record(rec: dict) -> None:
    assert sha(Path(rec["path"])) == rec["sha256"], rec["path"]


def audit_t0_parity() -> dict:
    historical = keyed(Path(inputs()["historical_b13_teacher_rankings"]["path"]), ("query_id",))
    current = {r["query_id"]: r for r in rows(OUT / "teacher/evaluation/T0/per_query.jsonl.gz")
               if r["condition"] == "Real" and r["view"] == "D" and r["budget"] == "Full-U"}
    assert len(current) == len(historical) == 1198
    differences, max_score_delta = [], 0.
    for q, r in current.items():
        old = historical[q,]
        truth = set(old["positive_target_ids"])
        assert set(r["ranking"]) == set(old["rankings"]["U_OFFLINE_T0"])
        for k in (10, 20, 50):
            recall = len(truth & set(old["rankings"]["U_OFFLINE_T0"][:k])) / len(truth)
            assert abs(recall - r["recall"][str(k)]) < 1e-12, (q, k, recall, r["recall"][str(k)])
        if r["ranking"] != old["rankings"]["U_OFFLINE_T0"]:
            differences.append(q)
    for r in rows(OUT / "teacher/evaluation/T0/scores.jsonl.gz"):
        if r["condition"] == "Real":
            old = historical[r["query_id"],]["teacher_scores"]
            max_score_delta = max(max_score_delta, max(abs(score - old[t]) for t, score in r["D"].items()))
    # The registered protocol requires matching population, pools and Recall,
    # not bitwise historical scalars. Preserve the observed cache discrepancy;
    # the separate probes check unchanged inputs and both current scorer paths.
    diagnostic = json.loads((OUT / "T0_SCORE_DIFFERENCE_DIAGNOSTIC.json").read_text())
    assert max_score_delta == diagnostic["absolute_delta_quantiles"]["max"]
    probe = json.loads((OUT / "T0_SCORE_NUMERICAL_PROBE.json").read_text())
    features = json.loads((OUT / "T0_FEATURE_IDENTITY_PROBE.json").read_text())
    assert len(probe["checks"]) == len(features["checks"]) == 3
    for r in probe["checks"]:
        assert abs(r["recomputed_R28_chunk32"] - r["new_score"]) <= 1e-5
        assert abs(r["recomputed_legacy_batch64"] - r["recomputed_R28_chunk32"]) <= 1e-5
    for r in features["checks"]:
        assert any(c["query_matches"] and c["target_matches"] and c["historical_ranking_score_matches"] for c in r["cached"])
    result = {"status": "pass", "queries_with_identical_Recall_at_10_20_50": len(current),
              "queries_with_full_rank_order_differences": differences, "max_QT_score_delta": max_score_delta,
              "historical_scalar_bitwise_parity": False,
              "scalar_difference_interpretation": "Historical cache discrepancy remains at scalar level; current legacy and R28 scorer paths agree in three worst cases, and actual feature hashes match. Exact historical runtime cause is unresolved. Registered querywise Recall parity passes without replacing any scores.",
              "numerical_probe": record(OUT / "T0_SCORE_NUMERICAL_PROBE.json"),
              "feature_probe": record(OUT / "T0_FEATURE_IDENTITY_PROBE.json"),
              "historical": inputs()["historical_b13_teacher_rankings"],
              "current_receipt": record(OUT / "teacher/evaluation/T0/EVALUATION_RECEIPT.json")}
    write_json(OUT / "T0_HISTORICAL_PARITY.json", result)
    return result


def audit_teacher(gid: str, historical: dict, object_types: dict) -> dict:
    directory = OUT / "teacher/evaluation" / gid
    receipt = json.loads((directory / "EVALUATION_RECEIPT.json").read_text())
    assert receipt["status"] == "completed"
    assert receipt["candidate_membership"]["sha256"] == inputs()["historical_b13_rankings"]["sha256"]
    for name in ("checkpoint", "per_query", "scores"):
        check_record(receipt[name])
    scores = keyed(directory / "scores.jsonl.gz", ("query_id", "condition"))
    metrics = keyed(directory / "per_query.jsonl.gz", ("query_id", "condition", "budget", "view"))
    conditions = ("Real", "Shuffled") if receipt["shuffle"] else ("Real",)
    assert receipt["shuffle"] == (gid == "T0" or gid.endswith("epoch5"))
    assert set(scores) == {(q, c) for q, in historical for c in conditions}
    assert set(metrics) == {(q, c, b, v) for q, in historical for c in conditions
                            for b in (100, 150, 200, "Full-U") for v in ("D", "E-LSE", "E-COV")}
    max_delta = 0.
    for (q,), meta in historical.items():
        truth, universe = set(meta["positive_target_ids"]), set(meta["U"])
        evidence = {c["target_id"]: c["selected_evidence_ids"] for c in meta["E_paths"]}
        strict = truth & (set(meta["E_target_ids"]) - (set(meta["rankings"]["D100_ANN"]) | set(meta["D100_EXACT"])))
        for condition in conditions:
            scalar = scores[q, condition]
            assert set(scalar["D"]) == universe
            assert set(scalar["E-LSE"]) == set(scalar["E-COV"]) == set(scalar["path_logits"]) == set(evidence)
            for t, values in scalar["path_logits"].items():
                assert len(values) == len(evidence[t])
                peak = max(values)
                lse = peak + math.log(sum(math.exp(x - peak) for x in values))
                assert abs(lse - scalar["E-LSE"][t]) < 1e-4
            for view in ("D", "E-LSE", "E-COV"):
                assert all(math.isfinite(v) for v in scalar[view].values())
                for budget in (100, 150, 200, "Full-U"):
                    pool = universe if budget == "Full-U" else set(meta["rankings"]["Equal"][:budget])
                    rank = sorted(pool & scalar[view].keys(), key=lambda t: (-scalar[view][t], t))
                    row = metrics[q, condition, budget, view]
                    assert row["ranking"] == rank and row["model_id"] == gid
                    assert row["source_table_id"] == meta["source_table_id"] and row["query_kind"] == meta["query_kind"]
                    assert row["candidate_count"] == len(pool) and row["rankable_count"] == len(rank)
                    assert row["raw_recall"] == len(truth & pool) / len(truth)
                    assert row["strict_EO_total"] == len(strict) and row["strict_EO_admitted"] == len(strict & pool)
                    for k in (10, 20, 50):
                        hits = len(strict & set(rank[:k]))
                        assert row["recall"][str(k)] == len(truth & set(rank[:k])) / len(truth)
                        assert row["strict_EO_hits"][str(k)] == hits
                        assert row["strict_EO_recall"][str(k)] == (hits / len(strict) if strict else None)
        if receipt["shuffle"]:
            real, shuffled = scores[q, "Real"]["D"], scores[q, "Shuffled"]["D"]
            max_delta = max(max_delta, max(abs(real[t] - shuffled[t]) for t in universe))
            assert sorted(universe, key=lambda t: (-real[t], t)) == sorted(universe, key=lambda t: (-shuffled[t], t))
    assert max_delta <= 1e-6 and max_delta == receipt["shuffle_max_QT_delta"]
    donor_count = 0
    if receipt["shuffle"]:
        donors = keyed(OUT / "teacher/evidence_shuffle" / gid / "donors.jsonl.gz", ("query_id", "target_id"))
        expected = {(q, c["target_id"]) for (q,), meta in historical.items() for c in meta["E_paths"]}
        assert set(donors) == expected
        for (q, t), donor in donors.items():
            original = next(c["selected_evidence_ids"] for c in historical[q,]["E_paths"] if c["target_id"] == t)
            source = historical[donor["donor_query_id"],]
            other = next(c["selected_evidence_ids"] for c in source["E_paths"] if c["target_id"] == donor["donor_target_id"])
            assert donor["evidence_ids"] == other
            assert donor["source_group"] == historical[q,]["source_table_id"]
            assert donor["donor_source_group"] == source["source_table_id"] != donor["source_group"]
            composition = sorted(object_types[e] for e in original)
            assert composition == donor["modality_composition"] == sorted(object_types[e] for e in other)
        donor_count = len(donors)
    return {"model_id": gid, "status": "pass", "queries": len(historical), "metric_rows": len(metrics),
            "donor_bundles_verified": donor_count, "max_QT_shuffle_delta": max_delta,
            "receipt": record(directory / "EVALUATION_RECEIPT.json")}


def audit_student(spec: dict, historical: dict, corpus: dict[str, set[str]]) -> dict:
    gid = spec["generator_id"]
    directory = OWN / "rankings" / gid
    receipt = json.loads((directory / "R28_EVALUATION_RECEIPT.json").read_text())
    assert receipt["status"] == "completed" and receipt["spec"] == spec
    for name in ("own_rankings", "teacher"):
        check_record(receipt[name])
    index = receipt["own_index"]
    for name in ("checkpoint", "feature_manifest", "index_manifest"):
        check_record(index[name])
    checkpoint = json.loads(Path(spec["checkpoint"]).with_suffix(".json").read_text())
    assert index["checkpoint"]["sha256"] == checkpoint["sha256"]
    assert index["parameter_sha256"] == checkpoint["parameter_sha256"]
    assert index["feature_manifest"]["sha256"] == inputs()["feature_manifest"]["sha256"]
    manifest = json.loads(Path(index["index_manifest"]["path"]).read_text())
    assert manifest["student_checkpoint_sha256"] == checkpoint["sha256"]
    frozen = json.loads((OUT / "INPUT_HASHES.json").read_text())
    assert manifest["corpus_sha256"] == frozen["corpus"]["sha256"]
    for modality, item in manifest["types"].items():
        index_dir = Path(index["index_manifest"]["path"]).parent
        ids = json.loads((index_dir / item["ids_path"]).read_text())
        assert len(ids) == len(set(ids)) == index["object_counts"][modality]
        assert set(ids) == corpus[modality]
        assert (index_dir / item["index_path"]).stat().st_size > 1000
    ranking = keyed(directory / "rankings.jsonl.gz", ("query_id",))
    teacher = keyed(OWN / "teacher" / gid / "rankings.jsonl.gz", ("query_id",))
    funnels = keyed(directory / "eo_strict_funnel.jsonl.gz", ("query_id",))
    assert set(ranking) == set(teacher) == set(funnels) == set(historical)
    cache_identity = json.loads((OWN / "teacher/CACHE_IDENTITY.json").read_text())
    assert cache_identity["teacher"]["sha256"] == inputs()["teacher_parent"]["sha256"]
    totals = Counter()
    for q, row in ranking.items():
        old, reranked, funnel = historical[q], teacher[q], funnels[q]
        for field in ("source_table_id", "query_kind", "positive_target_ids"):
            assert row[field] == reranked[field] == old[field]
        assert row["candidate_pool_id"] == reranked["candidate_pool_id"]
        assert row["parameter_sha"] == index["parameter_sha256"]
        assert reranked["teacher_namespace"] == cache_identity["namespace"]
        truth = set(row["positive_target_ids"])
        d, exact, e, u = set(row["rankings"]["D100_ANN"]), set(row["D100_EXACT"]), set(row["E_target_ids"]), set(row["U"])
        assert u == d | e
        assert u <= corpus["table"] and len(row["M_exact"]) == len(u)
        assert set(row["rankings"]["U"]) == u and set(row["rankings"]["M_EXACT"]) == set(row["M_exact"])
        assert set(row["rankings"]["E_ONLY"]) == e and set(row["rankings"]["Equal"]) == u
        assert row["rankings"]["D100_EXACT"] == row["D100_EXACT"]
        assert row["rankings"]["U"] == row["rankings"]["QT_OVER_U"] == sorted(u, key=lambda t: (-row["QT_OVER_U_scores"][t], t))
        assert row["exact_scores"] == sorted(row["exact_scores"], reverse=True)
        assert all(len(r) == len(set(r)) for r in row["rankings"].values())
        fixed = truth & (set(old["E_target_ids"]) - (set(old["rankings"]["D100_ANN"]) | set(old["D100_EXACT"])))
        own = truth & (e - (d | exact))
        assert funnel["fixed_EO_STRICT"] == sorted(fixed) and funnel["own_EO_STRICT"] == sorted(own)
        pools = {"D_ANN100": d, "D_exact100": exact, "E": e, "U": u, "M": set(row["M_exact"]),
                 **{f"C{k}": set(row["rankings"]["Equal"][:k]) for k in (100, 150, 200)}}
        for name, pool in pools.items():
            assert funnel["fixed_tracking"][name] == sorted(fixed & pool)
            assert funnel["own_tracking"][name] == sorted(own & pool)
        scores = reranked["teacher_scores"]
        assert set(scores) == u | set(row["M_exact"])
        for name, pool in {"BT100": row["rankings"]["Equal"][:100], "D100": row["rankings"]["D100_ANN"],
                           "U_OFFLINE": row["U"], "M_OFFLINE": row["M_exact"]}.items():
            assert reranked["rankings"][name + "_T0"] == sorted(pool, key=lambda t: (-scores[t], t))
        assert reranked["U_raw_recall"] == len(truth & u) / len(truth)
        totals.update(fixed_EO=len(fixed), own_EO=len(own))
    assert totals["fixed_EO"] == 207
    return {"model_id": gid, "status": "pass", "queries": len(ranking), **totals,
            "receipt": record(directory / "R28_EVALUATION_RECEIPT.json")}


def audit_available(require_complete: bool = False) -> dict:
    historical = keyed(Path(inputs()["historical_b13_rankings"]["path"]), ("query_id",))
    assert len(historical) == 1198
    frozen = json.loads((OUT / "INPUT_HASHES.json").read_text())
    object_types = {r["object_id"]: r["object_type"] for r in rows(Path(frozen["objects"]["path"]))}
    corpus = {modality: set() for modality in ("table", "text", "image")}
    for r in rows(Path(frozen["corpus"]["path"])):
        corpus[object_types[r["object_id"]]].add(r["object_id"])
    teacher_ids = ["T0"] + [f"{arm}/seed{seed}/epoch{epoch:g}" for arm in FAMILIES if arm.startswith("T-")
                            for seed in (13, 29) for epoch in (.5, 1, 2, 3, 5)]
    completed, missing = [], []
    for gid in teacher_ids:
        if (OUT / "teacher/evaluation" / gid / "EVALUATION_RECEIPT.json").exists():
            completed.append(audit_teacher(gid, historical, object_types))
            print(json.dumps({"audited": gid}), flush=True)
        else:
            missing.append(gid)
    for spec in json.loads((OWN / "MODEL_INVENTORY.json").read_text()):
        if (OWN / "rankings" / spec["generator_id"] / "R28_EVALUATION_RECEIPT.json").exists():
            completed.append(audit_student(spec, historical, corpus))
            print(json.dumps({"audited": spec["generator_id"]}), flush=True)
        else:
            missing.append(spec["generator_id"])
    if require_complete:
        assert not missing
    result = {"status": "pass" if not missing else "partial_pass", "completed": completed, "missing": missing}
    write_json(OUT / "EVALUATION_CONTENT_AUDIT.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-complete", action="store_true")
    audit_available(parser.parse_args().require_complete)
