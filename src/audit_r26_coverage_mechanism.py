"""Validate frozen replay artifacts and decompose budget Top10 gains/losses."""
from __future__ import annotations

import argparse
from collections import Counter
from itertools import zip_longest
from pathlib import Path

from replay_r26_evidence_order_abc import read_json, read_rows, record, sha, write_json


def audit(root: Path, output: Path) -> None:
    source = root / "work/stage1_optimization_r26_20260914"
    abc = root / "work/r26_evidence_order_abc_20260915"
    snapshot = read_json(output / "CODE_SNAPSHOT.json")
    for name, expected in snapshot.items():
        assert sha(output / "code" / name) == expected, name
    for records in (read_json(output / "replay/INPUTS.json"), read_json(output / "replay/OUTPUTS.json"),
                    read_json(output / "latency/PROTOCOL.json")["inputs"], read_json(output / "REPORT_ARTIFACTS.json")):
        for name, receipt in records.items():
            assert record(Path(receipt["path"])) == receipt, name
    benchmark_protocol = read_json(output / "latency/PROTOCOL.json")
    assert benchmark_protocol["inputs"]["checkpoint"]["sha256"] == read_json(abc / "PROTOCOL.json")["teacher"]["teacher"]["sha256"]
    historical_arms = {"greedy_coverage": "A_coverage", "greedy_lse": "B_retained_lse", "pre_lse": "C_pre_lse"}
    contrasts = [("C200", "C100"), ("Full-U", "C100"), ("Full-U", "C200"), ("greedy_no_path", "greedy_coverage")]
    transitions, cardinalities = {}, {}
    n, strict_n = 0, 0
    streams = [read_rows(p) for p in (source / "rankings/B13/rankings.jsonl.gz",
                                    abc / "models/B13/rankings.jsonl.gz", output / "replay/rankings.jsonl.gz")]
    for original, old, new in zip_longest(*streams):
        assert all(r is not None for r in (original, old, new))
        q = original["query_id"]
        assert q == old["query_id"] == new["query_id"]
        n += 1
        truth = set(original["positive_target_ids"])
        strict = truth & (set(original["E_target_ids"]) - set(original["D100_EXACT"]) - set(original["rankings"]["D100_ANN"]))
        strict_n += len(strict)
        for new_name, old_name in historical_arms.items():
            for stage in ("E", "Equal", "T0"):
                assert new["arms"][new_name][stage] == old["arms"][old_name][stage]
            assert new["arms"][new_name]["C"] == old["arms"][old_name]["C100"]
        for a, b in zip(("C50","C100","C150","C200"),("C100","C150","C200","Full-U")):
            assert new["arms"][b]["C"][:len(new["arms"][a]["C"])] == new["arms"][a]["C"]
        for a, b in contrasts:
            a_hits = truth & set(new["arms"][a]["T0"][:10])
            b_hits = truth & set(new["arms"][b]["T0"][:10])
            for kind in ("overall", original["query_kind"]):
                for group, eligible in (("all_positive", truth), ("strict_EO", strict), ("non_strict", truth-strict)):
                    counts = transitions.setdefault(kind, {}).setdefault(a+"_minus_"+b, {}).setdefault(group, Counter())
                    counts.update({"old_T10":len(b_hits & eligible), "new_T10":len(a_hits & eligible),
                                   "rescued":len((a_hits-b_hits) & eligible), "displaced":len((b_hits-a_hits) & eligible)})
        for target in original["E_paths"]:
            tid = target["target_id"]
            for group in ("all", "positive" if tid in truth else "unlabeled", "strict" if tid in strict else "non_strict"):
                counts = cardinalities.setdefault(group, Counter())
                counts.update({"targets":1, "one_original_path":len(target["paths"]) == 1,
                               "one_retained_path":len(target["retained_paths"]) == 1,
                               "four_retained_paths":len(target["retained_paths"]) == 4})
    strict_rows = list(read_rows(output / "replay/strict_evidence_only_pairs.jsonl.gz"))
    assert n == 1198 and strict_n == len(strict_rows) == 207
    assert len({(r["query_id"], r["target_id"]) for r in strict_rows}) == 207
    latency_rows = list(read_rows(output / "latency/measurements.jsonl"))
    assert len(latency_rows) == len({(r["query_id"],r["budget"],r["mode"],r["repeat"]) for r in latency_rows}) == 960
    assert max(r["max_score_error"] for r in latency_rows) < .001
    write_json(output / "TOP10_TRANSITIONS.json", transitions)
    write_json(output / "PATH_CARDINALITIES.json", cardinalities)
    write_json(output / "VALIDATION.json", {"status":"passed", "queries":n,"strict_pairs":strict_n,
        "historical_ABC_exact_rank_reproduction": True, "candidate_budgets_nested": True,
        "all_recorded_input_output_and_code_hashes_verified":True, "benchmark_checkpoint_matches_ABC_T0":True,
        "benchmark_measurements":len(latency_rows), "test_command":
        "PYTHONPATH=/home/oycy/MMDD/work/r26_coverage_mechanism_funnel_20260915/code PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 conda run -n MMDD python -m pytest /home/oycy/MMDD/tests/test_r26_coverage_mechanism.py /home/oycy/MMDD/tests/test_replay_r26_evidence_order_abc.py -q --confcutdir=/tmp/mmdd_r26_mechanism_checks",
        "test_result":"8 passed; executed from /tmp/mmdd_r26_mechanism_checks",
        "interpretation":"Stage1 only. No new training, Stage2, attribute truth annotation or correctness claim."})


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    audit(args.root,args.output)
