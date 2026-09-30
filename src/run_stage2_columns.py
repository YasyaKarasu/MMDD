#!/usr/bin/env python
"""Independent S2-COL-R1 audit, frozen-cache, train, eval, and report entrypoint."""
from __future__ import annotations
import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output', type=Path, required=True)
    sub = parser.add_subparsers(dest='command', required=True)
    audit = sub.add_parser('audit')
    audit.add_argument('--dataset-root', type=Path, action='append', required=True)
    audit.add_argument('--retrieval', type=Path, action='append', default=[])
    audit.add_argument('--candidate-scope-file', type=Path,
                       help='Frozen JSON/JSONL C30 scope: target ids per query or columns per pair')
    cache = sub.add_parser('cache')
    cache.add_argument('--model-dir', type=Path, required=True)
    cache.add_argument('--layout', choices=['header_markers_v0', 'tail_candidates_v1'], required=True)
    cache.add_argument('--split', choices=['train', 'dev', 'test'], required=True)
    cache.add_argument('--condition', choices=['O-O', 'O-R', 'No-E', 'Shuffled-E', 'ValueShuffle'], default='O-O')
    cache.add_argument('--view', type=int, choices=[0, 1, 2], default=0)
    cache.add_argument('--device', default='cuda:0')
    cache.add_argument('--image-pixels', type=int, default=262144)
    cache.add_argument('--limit', type=int)
    cache.add_argument('--view-seeds', type=int, nargs='+', default=[13001, 29001, 47001],
                       help='Column permutation seeds, one per view')
    train = sub.add_parser('train')
    train.add_argument('--arm', choices=['C0', 'C1', 'C2'], required=True)
    train.add_argument('--seed', type=int, choices=[13, 29], required=True)
    train.add_argument('--train-cache', type=Path, action='append', required=True)
    train.add_argument('--dev-cache', type=Path, required=True)
    train.add_argument('--tiny', action='store_true')
    train.add_argument('--epochs', type=int, default=20)
    evaluation = sub.add_parser('eval')
    evaluation.add_argument('--arm', choices=['C0', 'C1', 'C2'], required=True)
    evaluation.add_argument('--seed', type=int, choices=[13, 29], required=True)
    evaluation.add_argument('--checkpoint', type=Path, required=True)
    evaluation.add_argument('--cache', type=Path, required=True)
    replay = sub.add_parser('replay')
    replay.add_argument('--checkpoint', type=Path, required=True)
    replay.add_argument('--legacy-cache-root', type=Path, required=True)
    sub.add_parser('report')
    run = sub.add_parser('run')
    run.add_argument('--model-dir', type=Path, required=True)
    run.add_argument('--device', default='cuda:0')
    run.add_argument('--tiny-only', action='store_true')
    args = parser.parse_args()
    if args.command == 'audit':
        from mmdd_stage2.column_data import audit_data
        result = audit_data(args.dataset_root, args.output, args.retrieval, args.candidate_scope_file)
        print(json.dumps({'locked': result['population_locked'], 'counts': result['split_counts']}))
    elif args.command == 'cache':
        from mmdd_stage2.column_cache import build_features
        build_features(args.output, args.model_dir, layout=args.layout, split=args.split,
                       condition=args.condition, view=args.view, device=args.device,
                       image_pixels=args.image_pixels, limit=args.limit,
                       view_seeds=tuple(args.view_seeds))
    elif args.command == 'train':
        from mmdd_stage2.column_training import train_head
        train_head(args.output, args.arm, args.seed, args.train_cache, args.dev_cache, tiny=args.tiny, epochs=args.epochs)
    elif args.command == 'eval':
        from mmdd_stage2.column_training import evaluate_head
        evaluate_head(args.output, args.checkpoint, args.cache, args.arm, args.seed)
    elif args.command == 'replay':
        from mmdd_stage2.column_reporting import replay_legacy
        replay_legacy(args.output, args.checkpoint, args.legacy_cache_root)
    elif args.command == 'run':
        from mmdd_stage2.column_experiment import run_experiment
        run_experiment(args.output, args.model_dir, device=args.device, tiny_only=args.tiny_only)
    else:
        from mmdd_stage2.column_reporting import report
        report(args.output)


if __name__ == '__main__':
    main()
