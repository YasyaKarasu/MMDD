"""Complete SUP/KD recall evaluation after all fresh training stages finish.

Retrospective per-checkpoint gradient probes are not a selection input. This
entrypoint records their incomplete status and evaluates the completed selected
and full-epoch models using the unmodified CQET retrieval/evaluation functions.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from run_abebooks_fresh import bind, summarize, write_json
from mmdd_stage1.artifacts import json_identity, load_pool_bundle, save_pool_bundle
from mmdd_stage1.data import sha256_file
from mmdd_stage1.evaluate import evaluate_student_retrieval, evaluate_teacher_matrix
from mmdd_stage1.labels import export_eval_labels
from mmdd_stage1.lists import build_raw_pools_split
from mmdd_stage1.metrics import evaluate_matrix, export_funnels
from mmdd_stage1.provenance import source_identity


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    run = args.run_root.resolve()
    pipeline, _ = bind(run)
    rt = pipeline.load_runtime(run / "protocol.json", run)
    pipeline._gpu_guard(rt.paths)
    source = source_identity(rt.paths)
    stages = {}
    for stage in pipeline.STAGES:
        directory = run / "seed13" / stage
        receipt = json.loads((directory / "POST_RUN.attempt_001.json").read_text())
        pre = json.loads((directory / "PRE_RUN.attempt_001.json").read_text())
        if receipt["status"] != "SUCCESS" or pre["source_identity_sha256"] != source:
            raise ValueError(f"Incomplete training or changed training source: {stage}")
        stages[stage] = {"counters": receipt["counters"], "timing": receipt["timing"]}
    selections = {name: json.loads((run / f"seed13/selections/{name}.json").read_text())
                  for name in ("NATIVE_C1", "QT_C1", "NATIVE_C2_COMMON", "QT_C2")}
    for arm in ("SUP", "KD"):
        selection = selections["NATIVE_C2_COMMON"]
        if sha256_file(Path(selection[f"{arm}_checkpoint"])) != selection[f"{arm}_checkpoint_sha256"]:
            raise ValueError(f"Selected {arm} checkpoint changed")
    freeze = {"schema_version": "4.1.0", "seed": 13, "status": "FROZEN_BEFORE_TEST", **selections,
              "source_identity_sha256": source, "test_qrels_read": False,
              "retrospective_teacher_gradient_diagnostics": "INCOMPLETE_NOT_USED_FOR_SELECTION"}
    freeze["freeze_sha256"] = json_identity(freeze)
    write_json(run / "seed13/SELECTION_FREEZE.json", freeze)
    global_freeze = {"schema_version": "4.1.0", "status": "ALL_SELECTIONS_FROZEN_BEFORE_TEST",
                     "seeds": {"13": freeze["freeze_sha256"]}, "test_qrels_read": False}
    global_freeze["freeze_sha256"] = json_identity(global_freeze)
    write_json(run / "GLOBAL_SELECTION_FREEZE.json", global_freeze)
    write_json(run / "PHASE_STATUS.json", {"status": "FROZEN_RECALL_EVALUATION", "full_training_completed": True,
               "retrospective_teacher_gradient_diagnostics": "INCOMPLETE_NOT_USED_FOR_SELECTION"})
    teacher = pipeline._load_teacher(run / "seed13/TB_CQET/checkpoints/end.pt")
    checkpoints = {"raw": None,
                   "native_sup": Path(selections["NATIVE_C2_COMMON"]["SUP_checkpoint"]),
                   "native_kd": Path(selections["NATIVE_C2_COMMON"]["KD_checkpoint"]),
                   "endpoint_sup": run / "seed13/NATIVE_C2_SUP/checkpoints/snapshot_frac100.pt",
                   "endpoint_kd": run / "seed13/NATIVE_C2_KD/checkpoints/snapshot_frac100.pt"}
    started = time.time()
    for split in ("dev", "test"):
        gt = export_eval_labels(rt.paths, rt.labels.canonical_map, split)
        for generator, checkpoint in checkpoints.items():
            print(f"EVALUATE {split} {generator}", flush=True)
            directory = run / f"seed13/eval/{split}/{generator}"
            if generator == "raw":
                if (directory / "POOL_MANIFEST.json").exists():
                    pools = load_pool_bundle(directory)
                else:
                    pools = build_raw_pools_split(rt.z_store, rt.row_store, sorted(gt), rt.labels, split, hnsw_seed=13)
            else:
                student = pipeline._load_native(checkpoint, rt)
                pools = evaluate_student_retrieval(student, rt.z_store, rt.row_store, sorted(gt), rt.labels,
                                                  split, hnsw_seed=13, generator_id=generator)
                del student
            save_pool_bundle(directory, pools, rt.labels, seed=13, generator=generator)
            matrix = evaluate_teacher_matrix({"TB_CQET": teacher}, rt.bank, pools, rt.labels, seed=13,
                                              generator=generator, split=split, output_dir=directory, split_gt=gt)
            metrics = evaluate_matrix(pools, matrix, gt, directory)
            pipeline._independent_verify(rt, split, directory, metrics)
            export_funnels(pools, matrix["TB_CQET"]["Real"], gt, rt.labels,
                           run / f"seed13/eval/{split}/funnels/{generator}", seed=13, generator=generator)
            print(f"DONE {split} {generator} elapsed={time.time()-started:.1f}s", flush=True)
            del pools, matrix
            torch.cuda.empty_cache()
    summarize(run)
    write_json(run / "TRAINING_AND_EVALUATION_RECEIPT.json", {
        "status": "COMPLETE", "seed": 13, "training_stages": stages,
        "training_source_identity_sha256": source,
        "evaluation_entrypoint_sha256": sha256_file(Path(__file__).resolve()),
        "evaluation_seconds": time.time() - started,
        "retrospective_teacher_gradient_diagnostics": "INCOMPLETE_NOT_USED_FOR_SELECTION",
        "retrieval_algorithm_changed": False, "old_run_features_or_checkpoints_used": False,
        "global_freeze_sha256": global_freeze["freeze_sha256"],
        "recall_summary_sha256": sha256_file(run / "recall_summary.json"),
    })
    write_json(run / "PHASE_STATUS.json", {"status": "RECALL_EVALUATION_COMPLETE", "full_training_completed": True,
               "retrospective_teacher_gradient_diagnostics": "INCOMPLETE_NOT_USED_FOR_SELECTION"})


if __name__ == "__main__":
    main()
