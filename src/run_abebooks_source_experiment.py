"""Train the unchanged CQET method and evaluate dev before opening test."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from run_abebooks_fresh import ROOT, finish, launch, prepare_data, write_json


def reference_inputs(root: Path, reference: Path) -> None:
    """Reuse unchanged frozen inputs in a new training experiment."""
    root.mkdir(parents=True, exist_ok=False)
    # Features live in the reference's feature directory (kept by the inherited protocol
    # paths); only the filtered dataset view and the composition record are linked.
    for name in ("dataset_view", "FEATURE_COMPOSITION.json"):
        if (reference / name).exists():
            (root / name).symlink_to(reference / name)
    # The recipe comes from the current template; only the frozen-input locations are inherited.
    protocol = json.loads((ROOT / "configs/mmdd_stage1_cqet_protocol.json").read_text())
    reference_protocol = json.loads((reference / "protocol.json").read_text())
    protocol.update(seeds=[13], max_registered_stages=9)
    protocol["evaluation"]["k"] = reference_protocol["evaluation"]["k"]
    protocol["paths"] = {k: v.replace(str(reference), str(root)) for k, v in reference_protocol["paths"].items()}
    protocol["paths"].setdefault("package_dir", str(root / "protocol_package"))
    write_json(root / "protocol.json", protocol)
    write_json(root / "FRESH_INPUTS.json", {"dataset": str(root / "dataset_view"),
        "reference": str(reference), "frozen_inputs_reused": True, "old_checkpoints_reused": False})


def run_experiment(root: Path, command: str, gpu: int,
                   student_lr_p: float = 1e-4, student_lr_r: float = 1e-3) -> None:
    run = root / "main"
    if command == "train":
        run.mkdir(exist_ok=False)
        (run / "isolated_cwd").mkdir()
        for shared in ("dataset_view", "FEATURE_COMPOSITION.json"):
            if (root / shared).exists():
                (run / shared).symlink_to(root / shared)
        protocol = json.loads((root / "protocol.json").read_text())
        protocol["paths"] = {k: v.replace(str(root), str(run)) for k, v in protocol["paths"].items()}
        protocol["student"].update(P_lr=student_lr_p, R_lr=student_lr_r)
        for stage, batch in (("C1", "logical_batch_edge_lists"), ("C2", "logical_batch_queries")):
            protocol["student"][stage].update(epochs=10, **{batch: 16},
                                               checkpoints=[i / 10 for i in range(11)])
        write_json(run / "protocol.json", protocol)
        hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                  for package in ("mmdd_stage1",)
                  for p in (ROOT / "src" / package).glob("*.py")}
        hypothesis = ("Removing train-frequent nongold evidence improves evidence recall and target ranking on fixed queries/targets"
                      if (root / "HUB_FILTER.json").exists() else
                      "Source-level removal of commerce/rating/sparse columns improves book evidence and target discrimination")
        if (root / "EVIDENCE_FILTER.json").exists():
            hypothesis = "Evidence attached to source tables without any approved bridge fact creates ungrounded paths; curate at the source level"
            evidence_filter = json.loads((root / "EVIDENCE_FILTER.json").read_text())
            if evidence_filter.get("filter_kind") == "unlabelled_text":
                hypothesis = "Catalog ablation: unlabelled text creates competing topical paths; retain every image and all labelled text aliases with unchanged queries, targets and supervision"
            elif evidence_filter.get("filter_kind") == "duplicate_book_images":
                hypothesis = "Near-identical covers of the same four-field book record occupy first-hop slots repeatedly; remove unlabelled pixel-near copies while protecting every existing witness"
            elif evidence_filter.get("filter_kind") == "text_hubs":
                hypothesis = "Train-frequent unlabelled text creates generic topical paths; remove those text classes while retaining every image and labelled content alias"
            elif evidence_filter.get("filter_kind") == "commercial_text":
                hypothesis = "Seller descriptions and sale/shipping policies dilute bibliographic evidence paths; retain synopsis, author text and covers by source provenance"
            elif evidence_filter.get("filter_kind") == "title_anchored_text":
                hypothesis = "Text without distinctive source-title words creates ungrounded book paths; retain title-linked text, all covers and existing labelled aliases"
            elif evidence_filter.get("filter_kind") == "positive_sources":
                hypothesis = "Evidence from sources with no currently labelled join task creates irrelevant paths; retain whole source evidence groups for every explicit and implicit positive"
        if (root / "HEADER_PARITY.json").exists():
            hypothesis = "Readable book-specific source headers improve representation with identical cells, rows, task facts and splits"
        if (root / "SOURCE_SCHEMA_PLAN.json").exists():
            hypothesis = json.loads((root / "SOURCE_SCHEMA_PLAN.json").read_text())["hypothesis"]
        if (student_lr_p, student_lr_r) != (1e-4, 1e-3):
            hypothesis = "Existing P/R learning-rate parameters control underfitting on the complete small training set; all method components stay unchanged"
        write_json(root / "EXPERIMENT_PLAN.json", {"seed": 13, "method_source_hashes": hashes,
            "hypothesis": hypothesis,
            "training": "All generated train queries; existing constructors, losses and selection; P/R learning-rate parameters recorded below",
            "student_epochs": 10, "student_batch": 16,
            "student_lr_p": student_lr_p, "student_lr_r": student_lr_r,
            "primary_metric": "selected_kd Multimodal_RRF query_macro_Recall@10",
            "diagnostics": "Direct, teacher Real/f0/Swap and evidence row/attribute coverage",
            "test_policy": "Dev inspected first; test is historically exposed regression data"})
        finish("train", launch(run, "train", [str(ROOT / "src/run_abebooks_data_ablation.py"),
            "train", "--run-root", str(run), "--student-epochs", "10", "--student-batch", "16",
            "--student-lr-p", str(student_lr_p), "--student-lr-r", str(student_lr_r)], gpu))
        freeze = json.loads((run / "SELECTION_FREEZE.json").read_text())
        write_json(root / "ALL_SELECTIONS_FROZEN.json", {"status": "FROZEN_BEFORE_TEST", "main": freeze})
    else:
        splits = ["dev"] if command == "dev" else ["dev", "test"]
        finish(command, launch(run, command, [str(ROOT / "src/run_abebooks_data_ablation.py"),
            "evaluate", "--run-root", str(run), "--splits", *splits], gpu))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "train", "dev", "test"))
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0, help="physical GPU index (PCI bus order)")
    parser.add_argument("--reference-inputs", type=Path)
    parser.add_argument("--dataset", type=Path, help="Versioned dataset for the prepare command")
    parser.add_argument("--student-lr-p", type=float, default=1e-4)
    parser.add_argument("--student-lr-r", type=float, default=1e-3)
    args = parser.parse_args()
    if args.command == "prepare":
        if args.dataset is None:
            parser.error("prepare requires --dataset")
        root, dataset = args.run_root.resolve(), args.dataset.resolve()
        prepare_data(root, dataset, args.gpu)
        report = json.loads((dataset / "REBUILD.json").read_text())
        write_json(root / "DATA_REBUILD.json", report)
        write_json(root / "SOURCE_SCHEMA_PLAN.json", {
            "hypothesis": "Source-level content curation improves evidence paths while the retrieval and training method remain fixed",
            "kept_columns": report["kept_columns"], "title_grouping": report.get("title_grouping", {}),
            "required_test_recall": {"overall": 0.4, "implicit": 0.1},
            "existing_facts_retained": report["retained_approved_facts"],
            "original_facts": report["approved_input_facts"], "new_annotations": 0})
        return
    if args.reference_inputs is not None:
        if args.command != "train":
            parser.error("--reference-inputs is only used when starting new training")
        reference_inputs(args.run_root.resolve(), args.reference_inputs.resolve())
    run_experiment(args.run_root.resolve(), args.command, args.gpu, args.student_lr_p, args.student_lr_r)


if __name__ == "__main__":
    main()
