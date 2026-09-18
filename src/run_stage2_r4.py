#!/usr/bin/env python
"""S2-R4 driver: real-C50 joint table-column selection, recovery and integration."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, '/home/oycy/MMDD/src')

from mmdd_stage2.r4_common import R4  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('phase', choices=[
        'source-lock', 'population', 'labels', 'p0', 'features', 'joint', 'metrics',
        'schedule', 'recovery', 'funnel', 'integration', 'report',
    ])
    parser.add_argument('--out', type=Path, default=R4)
    parser.add_argument('--scope', default='pilot', choices=['pilot', 'dev'])
    parser.add_argument('--split', default='dev', choices=['dev', 'test'])
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--shards', type=int, default=1)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--queue', default='R-deploy', choices=['R-deploy', 'R-component'])
    parser.add_argument('--groups', type=int, default=0,
                        help='R-deploy: number of source groups to include (0 = all)')
    args = parser.parse_args()

    if args.phase == 'source-lock':
        from mmdd_stage2.r4_population import source_lock
        from mmdd_stage2.r4_common import write_json
        write_json(args.out / 'SOURCE_LOCK.json', source_lock())
        print('source lock written')
    elif args.phase == 'population':
        from mmdd_stage2.r4_population import build_population
        print(json.dumps(build_population(args.scope, args.out), indent=2))
    elif args.phase == 'labels':
        from mmdd_stage2.r4_labels import build_labels
        print(json.dumps(build_labels(args.out, args.scope, split=args.split), indent=2))
    elif args.phase == 'p0':
        from mmdd_stage2.r4_p0_replay import run as run_p0
        print(json.dumps(run_p0(args.out), indent=2))
    elif args.phase == 'features':
        from mmdd_stage2.r4_features import build_features
        print(json.dumps(build_features(args.out, args.scope, device=args.device,
                                        shard=args.shard, shards=args.shards,
                                        limit=args.limit or None), indent=2))
    elif args.phase == 'joint':
        from mmdd_stage2.r4_joint_build import build_joint
        print(json.dumps(build_joint(args.out, args.scope), indent=2))
    elif args.phase == 'recovery':
        from mmdd_stage2.r4_recovery_run import run as run_recovery
        print(json.dumps(run_recovery(args.out, args.scope, queue=args.queue,
                                      device=args.device, shard=args.shard,
                                      shards=args.shards, limit=args.limit or None,
                                      groups=args.groups or None), indent=2))
    elif args.phase == 'funnel':
        from mmdd_stage2.r4_evidence_funnel import build_funnel
        print(json.dumps(build_funnel(args.out, args.scope, split=args.split), indent=2))
    elif args.phase == 'schedule':
        from mmdd_stage2.r4_schedule import build_schedules
        print(json.dumps(build_schedules(args.out, args.scope), indent=2))
    elif args.phase == 'metrics':
        from mmdd_stage2.r4_phase_metrics import build_metrics
        print(json.dumps(build_metrics(args.out, args.scope), indent=2))
    elif args.phase == 'report':
        from mmdd_stage2.r4_report import build_report
        from mmdd_stage2.r4_delivery import collect_costs, write_manifest
        summary = build_report(args.out, args.scope)
        costs = collect_costs(args.out, args.scope)
        from mmdd_stage2.r4_common import write_json
        write_json(args.out / 'COSTS' / f'COSTS.{args.scope}.json', costs)
        summary['costs_written'] = str(args.out / 'COSTS' / f'COSTS.{args.scope}.json')
        from mmdd_stage2.r4_report_md import write_report
        summary.update(write_report(args.out, args.scope))
        summary['manifest'] = write_manifest(args.out, summary.get('module_status', {}))
        print(json.dumps(summary, indent=2))
    else:
        raise SystemExit(f'{args.phase} is not implemented yet')


if __name__ == '__main__':
    main()
