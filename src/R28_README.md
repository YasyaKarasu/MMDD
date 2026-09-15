# R28 split-path continuation

The experiment is fixed by `mmdd_r27_review/R28_CODEX_PROMPT.md`,
`R28_EXPERIMENT_PLAN.md`, `R28_INPUT_LOCK.json`, and
`R28_OBJECTIVE_CONTRACT.json`. Artifacts go to
`work/stage1_optimization_r28_split_path_20260915`.

The six arms use seeds 13 and 29 and save epochs 0, 0.5, 1, 2, 3, and 5.
Teacher runs have 7,120 updates; Student runs have 890. Path losses add separate
Direct and optional Evidence listwise losses. LSE operates within an evidence
bag; COV uses detached frozen row support. Teacher training continues at the
verified historical learning rate 5e-5. Student keeps the historical projection
and relation learning rates, resets the continuation projection anchor, and
disables KD.

## Entry points

- `prepare_stage1_r28.py` resolves and hashes locked inputs, checks positive
  closure, and freezes schedules and optimizer settings.
- `audit_stage1_r28_objectives.py` records tensor tests and six real-parent
  backward smokes without optimizer updates.
- `train_stage1_r28.py --arm ARM --seed 13 --device cuda:0` runs one fixed job.
  `run_stage1_r28_queue.py` launches the original matrix. Existing training
  outputs are deliberately not overwritten.
- `prepare_stage1_r28_hidden.py` audits the required frozen hidden cache and
  encodes its verified missing entries. `resume_stage1_r28_teacher.py` preserves
  the six failed initial attempts and restarts those same logical jobs from
  their locked parent. Recovery receipts disclose 249 discarded updates.
- `evaluate_stage1_r28_student.py` builds each checkpoint's own ANN/exact/E/U/M
  results, tracks fixed and own strict EO, and applies frozen T0.
- `evaluate_stage1_r28_teacher_fast.py` evaluates frozen B13 candidates using
  D, E-LSE, and E-COV separately. T0 and epoch5 endpoints also use shuffled
  evidence. Its batching audit compares the original chunked scorer.
- `run_stage1_r28_evaluations.py` consumes available checkpoints, with a worker
  per GPU initially. `--kind student` and `--kind teacher` provide disjoint
  queues for one Student and one Teacher evaluation per GPU after training
  releases capacity. `--preview` lists the exact jobs without launching them.
  During a supervised handoff, pause the original supervisor and give the
  corresponding new queue `--wait-pid` and `--wait-receipt` for its active child;
  that child finishes before the new queue starts. The old supervisor must
  remain paused until it is retired, so it cannot dispatch overlapping jobs.
  `split_stage1_r28_evaluations.py` previews this handoff; `--execute` performs
  it only after at least one Teacher has completed on each GPU. It records the
  old and new process identities and retires the old supervisors after their
  children finish.
  `run_stage1_r28_teacher_side_batch.py` performs the recorded final scheduling
  change after all training finishes: it pauses Teacher dispatch, lets active
  Edge evaluations finish, and runs the four remaining Path arm/seed
  trajectories concurrently. It preserves the exact registered commands and
  resumes the original dispatchers after completion. See
  `TEACHER_PARALLEL_BATCH.json` for live progress during that batch.
  `run_stage1_r28_finish.py` monitors actual process identities and
  starts final analysis only after all 67 evaluations are present.
- `analyze_stage1_r28.py --bootstrap` writes query-macro metrics, separate
  pair-pooled EO admission/retention, costs, and 10,000 source-group paired
  bootstrap replicates. Family comparisons average seeds within each query.
- `plot_stage1_r28.py` exports PNG/PDF trajectories. Plotting dependencies are
  installed locally under the experiment output's `plot_dependencies` folder.
- `finalize_stage1_r28.py` verifies training tensors and evaluation contents,
  checks T0 historical parity, and writes the four final deliverables.
- `verify_stage1_r28_statistics.py` independently reconstructs every summary
  cell and the primary family bootstrap intervals from the final per-query
  artifact. Its receipt is `INDEPENDENT_STATISTICS_AUDIT.json`.
- `verify_stage1_r28_coverage.py --require-complete` reconstructs all retained
  bags for three fixed query positions at all Teacher nodes, including
  shuffled evidence, using independent double-precision NumPy calculations.
  It writes `INDEPENDENT_COVERAGE_AUDIT.json`.
- `review_stage1_r28_admission.py` provides descriptive epoch1/epoch5 accounting
  of actual Direct ANN targets versus Evidence-added targets. These components
  exactly sum to own U raw recall and final T0 Recall@10; this is not an extra
  experiment or hypothesis test.
- `verify_stage1_r28_student_export.py` checks all saved table vectors at the
  twelve Student epoch1/epoch5 checkpoints against fresh checkpoint exports.
  It also compares full-corpus bilinear scoring and saved exact/U scores for
  three fixed queries. This investigates the observed large retrieval decline
  without changing the trained models or evaluation protocol.

## Independent checks

Run checks with the `MMDD` environment from an isolated working directory such
as `/tmp/mmdd-r28-checks`. These commands require local experiment artifacts;
the unit tests use synthetic fixtures and do not need a GPU or model server.

```bash
conda run -n MMDD python -m pytest /home/oycy/MMDD/tests/test_stage1_r28.py /home/oycy/MMDD/tests/test_stage1_r26.py -q
conda run -n MMDD python /home/oycy/MMDD/src/audit_stage1_r28_training.py --kind student
conda run -n MMDD python /home/oycy/MMDD/src/audit_stage1_r28_evaluations.py --require-complete
```

The training audit hashes actual checkpoint tensors, validates optimizer groups
and anchors, verifies every batch against frozen orders, and reconstructs every
loss. The evaluation audit reconstructs rankings and metrics from saved scores,
validates own corpus/index/checkpoint identities, and checks shuffle donors and
both EO definitions. Without `--require-complete`, it audits available results
and explicitly lists missing nodes.

`RESULTS.md`, `LIMITATIONS.md`, `NEXT_DECISION.md`, and `EXECUTION_LEDGER.json`
are final only after the complete audit passes. Partial tables or live status
files are not endpoint conclusions. No new fusion, extra seeds, KD, Uniform,
remining, redistillation, or Stage2 is part of this experiment.
