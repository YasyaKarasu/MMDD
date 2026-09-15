"""Describe exact Direct/Evidence contributions at Student epochs 1 and 5."""
from collections import defaultdict
import csv
import json
from pathlib import Path

from prepare_stage1_r27 import rows, record, write_json
from prepare_stage1_r28 import OUT
from evaluate_stage1_r28_student import OWN


def main() -> None:
    values = defaultdict(list)
    provenance = []
    for spec in json.loads((OWN / "MODEL_INVENTORY.json").read_text()):
        if spec["epoch"] not in (1, 5):
            continue
        gid = spec["generator_id"]
        receipt_path = OWN / "rankings" / gid / "R28_EVALUATION_RECEIPT.json"
        assert json.loads(receipt_path.read_text())["status"] == "completed"
        provenance.append(record(receipt_path))
        teacher = {r["query_id"]: r for r in rows(OWN / "teacher" / gid / "rankings.jsonl.gz")}
        for row in rows(OWN / "rankings" / gid / "rankings.jsonl.gz"):
            truth = set(row["positive_target_ids"])
            ann = set(row["rankings"]["D100_ANN"])
            exact = set(row["D100_EXACT"])
            evidence = set(row["E_target_ids"])
            union = set(row["U"])
            assert union == ann | evidence
            added = evidence - ann
            strict = evidence - (ann | exact)
            top = set(teacher[row["query_id"]]["rankings"]["U_OFFLINE_T0"][:10])
            n = len(truth)
            metrics = {"U_RawRecall": len(truth & union) / n,
                       "Direct_ANN_RawRecall": len(truth & ann) / n,
                       "Direct_exact100_RawRecall": len(truth & exact) / n,
                       "Evidence_added_outside_ANN_RawRecall": len(truth & added) / n,
                       "Evidence_added_outside_ANN_and_exact_RawRecall": len(truth & strict) / n,
                       "T0_U_R10": len(truth & top) / n,
                       "T0_U_R10_from_Direct_ANN_targets": len(truth & ann & top) / n,
                       "T0_U_R10_from_Evidence_added_targets": len(truth & added & top) / n,
                       "own_strict_EO_pairs": len(truth & strict)}
            assert abs(metrics["U_RawRecall"] - metrics["Direct_ANN_RawRecall"] - metrics["Evidence_added_outside_ANN_RawRecall"]) < 1e-12
            assert abs(metrics["T0_U_R10"] - metrics["T0_U_R10_from_Direct_ANN_targets"] - metrics["T0_U_R10_from_Evidence_added_targets"]) < 1e-12
            for kind in ("overall", row["query_kind"]):
                values[spec["arm"], spec["seed"], spec["epoch"], kind].append((row["query_id"], metrics))
    for (arm, seed, epoch, kind), items in list(values.items()):
        if seed != 13:
            continue
        other = dict(values[arm, 29, epoch, kind])
        assert {q for q, _ in items} == set(other)
        values[arm, "13+29", epoch, kind] = [(q, {m: (v + other[q][m]) / 2 for m, v in metrics.items()}) for q, metrics in items]
    result = []
    for (arm, seed, epoch, kind), items in values.items():
        result.append({"arm": arm, "seed": seed, "epoch": epoch, "kind": kind, "queries": len(items),
                       **{m: sum(v[m] for _, v in items) / len(items) for m in items[0][1]}})
    destination = OUT / "statistics/student_admission_decomposition.csv"
    with destination.open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result[0]))
        writer.writeheader()
        writer.writerows(result)
    write_json(OUT / "STUDENT_ADMISSION_DECOMPOSITION_AUDIT.json", {
        "status": "pass", "code": record(Path(__file__)), "source_receipts": provenance, "table": record(destination),
        "interpretation": "Descriptive accounting, not additional hypothesis testing: Direct means actual ANN100 membership; Evidence-added means outside that membership. Both components exactly sum to U RawRecall and final T0 R@10. The strict outside-ANN-and-exact subset is separately reported; changing own EO membership is not a fixed-population comparison."})
    print(json.dumps([r for r in result if r["kind"] == "overall" and r["seed"] == "13+29"]))


if __name__ == "__main__":
    main()
