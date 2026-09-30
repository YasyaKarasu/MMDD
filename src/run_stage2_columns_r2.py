#!/usr/bin/env python
"""Run the contracted R2 column-only experiments without modifying R1 artifacts."""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--r1', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--model-dir', type=Path)
    parser.add_argument('--jobs', type=Path)
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--shards', type=int, default=2)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--view-seeds', type=int, nargs='+', default=[13001, 29001, 47001],
                        help='Column permutation seeds, one per reader view')
    parser.add_argument('--kind', choices=['flat', 'separate', 'test', 'prior-test', 'nat_e', 'nat_e_test'],
                        default='flat')
    parser.add_argument('--arm', choices=['OO_CONTROL', 'FLAT_MIX', 'PRIOR', 'NAT_E', 'PVR_BUNDLE',
                                          'PVR_SEPARATE', 'PVR_SUPPORT'])
    parser.add_argument('--stage1-dir', type=Path,
                        help='Stage-1 export directory holding <split>/<query_id>.json for NAT_E inputs')
    parser.add_argument('--seed', type=int, choices=[13,29], default=13)
    parser.add_argument('--head-execution', choices=['scalar', 'batched'], default='batched',
                        help='Batch selector MLPs, retaining per-pair dropout and the full training population')
    parser.add_argument('--training-output', type=Path,
                        help='Write a new training run here while reading existing features from --output')
    parser.add_argument('--epochs', type=int, default=20, help='Maximum selector training epochs')
    parser.add_argument('--early-stopping-patience', type=int,
                        help='Dev-MRR plateau checks; default 5 for PRIOR, 0 (disabled) for matched ablations')
    parser.add_argument('--min-epochs', type=int, default=10,
                        help='Start counting early-stop plateaus at this epoch')
    parser.add_argument('--min-delta', type=float, default=.0005,
                        help='Minimum dev-MRR improvement that resets patience')
    parser.add_argument('--log-every-pairs', type=int, default=1024,
                        help='Print recent loss after this many Q-T pairs; 0 disables progress logs')
    parser.add_argument('--fixed-epoch', type=int,
                        help='Keep this epoch and monitor dev instead of selecting on it; requires '
                             '--early-stopping-patience 0 and is the controlled-comparison schedule')
    parser.add_argument('phase', choices=['audit', 'prepare-a', 'reader', 'evaluate-a', 'natural-train',
                                         'build-natural-evidence', 'prepare-jobs', 'train', 'freeze-shortlists',
                                         'audit-support', 'lock-selection', 'formal-test', 'analyze', 'cost',
                                         'report', 'audit-reader-inputs', 'freeze-runtime'])
    args = parser.parse_args()
    root = args.root.resolve()
    r1 = args.r1 or root / 'work/S2-COL-R1'
    output = args.output or root / 'work/S2_COL_R2'
    model_dir = args.model_dir or root / 'hf_models/Qwen3.5-9B'
    if args.phase == 'audit':
        from mmdd_stage2.column_r2_audit import audit_and_replay
        audit_and_replay(root, r1, output, model_dir)
    elif args.phase == 'prepare-a':
        from mmdd_stage2.column_r2_cache import prepare_phase_a
        prepare_phase_a(r1, output)
    elif args.phase == 'reader':
        from mmdd_stage2.column_r2_cache import reader_worker
        reader_worker(r1, output, model_dir, args.jobs, args.shard, args.shards, args.device,
                      view_seeds=tuple(args.view_seeds))
    elif args.phase == 'evaluate-a':
        from mmdd_stage2.column_r2_phase_a import evaluate_phase_a
        evaluate_phase_a(r1, output)
    elif args.phase == 'natural-train':
        from mmdd_stage2.column_r2_natural import build_natural_train
        build_natural_train(root, r1, output)
    elif args.phase == 'build-natural-evidence':
        from mmdd_stage2.natural_evidence import build_inputs, stage1_directory_loader
        if args.stage1_dir is None:
            parser.error('build-natural-evidence requires --stage1-dir')
        build_inputs(r1, output, stage1_directory_loader(args.stage1_dir))
    elif args.phase == 'prepare-jobs':
        from mmdd_stage2.column_r2_training import prepare_jobs
        prepare_jobs(r1, output, args.kind)
    elif args.phase == 'train':
        from mmdd_stage2.column_r2_training import train_arm
        if args.arm is None:
            parser.error('train requires --arm')
        train_arm(r1, output, args.arm, args.seed, execution=args.head_execution,
                  training_output=args.training_output, epochs=args.epochs,
                  early_stopping_patience=args.early_stopping_patience, min_epochs=args.min_epochs,
                  min_delta=args.min_delta, log_every_pairs=args.log_every_pairs,
                  fixed_epoch=args.fixed_epoch)
    elif args.phase == 'freeze-shortlists':
        from mmdd_stage2.column_r2_training import freeze_shortlists
        freeze_shortlists(r1, output)
    elif args.phase == 'audit-support':
        from mmdd_stage2.column_r2_support import audit_support
        audit_support(r1, output)
    elif args.phase in {'lock-selection', 'formal-test', 'analyze'}:
        from mmdd_stage2.column_r2_evaluation import lock_selection, formal_test, analyze
        {'lock-selection': lock_selection, 'formal-test': formal_test, 'analyze': analyze}[args.phase](r1, output)
    elif args.phase == 'cost':
        from mmdd_stage2.column_r2_cost import audit_cost
        audit_cost(r1, output)
    elif args.phase == 'report':
        from mmdd_stage2.column_r2_report import build_report
        build_report(r1, output)
    elif args.phase == 'audit-reader-inputs':
        from mmdd_stage2.column_r2_audit import audit_reader_inputs
        audit_reader_inputs(r1, output)
    elif args.phase == 'freeze-runtime':
        from mmdd_stage2.column_r2_runtime import freeze_runtime
        freeze_runtime(root, r1, output)


if __name__ == '__main__':
    main()
