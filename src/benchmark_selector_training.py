"""Compare selector head execution on existing features; never run generation.

The bounded training probe is an engineering check, not a reduced scientific
training protocol. Existing reader features and selected checkpoints are read-only.
"""
from __future__ import annotations

import argparse
import copy
import random
import time
from collections import Counter
from pathlib import Path

import torch

from mmdd_stage2.column_batching import score_records
from mmdd_stage2.column_data import digest, file_hash, read_jsonl, write_json
from mmdd_stage2.column_metrics import evaluate, pair_key, prediction
from mmdd_stage2.column_r2_cache import FeatureIndex, job, job_key
from mmdd_stage2.column_r2_training import inputs_for, load_head
from mmdd_stage2.column_training import column_loss
from mmdd_stage2.verifier import CandidateColumnScorer


def step(model: CandidateColumnScorer, optimizer: torch.optim.Optimizer, records: list[dict],
         metadata: dict, weights: dict, execution: str, *, inspect: bool = False) -> tuple[float, dict]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    logits = score_records(model, records, execution=execution)
    losses = [weights[pair_key(r)]*column_loss(scores, r['candidate_column_indices'],
                                             metadata[pair_key(r)]['gold_column_indices'])
              for r, scores in zip(records, logits)]
    loss = torch.stack(losses).sum()/len(records) + next(model.parameters()).sum()*0
    loss.backward()
    observed = ({'logits': torch.cat(logits).detach(),
                 'gradient': torch.cat([p.grad.flatten() for p in model.parameters()]).clone(),
                 'rng': torch.get_rng_state().clone()} if inspect else {})
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
    optimizer.step()
    return float(loss.detach()), observed


@torch.inference_mode()
def predict(model: CandidateColumnScorer, records: list[dict], execution: str) -> list[dict]:
    model.eval()
    result = []
    for start in range(0, len(records), 32):
        batch = records[start:start+32]
        result.extend(prediction(r, scores.tolist()) for r, scores in
                      zip(batch, score_records(model, batch, execution=execution)))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument('--r1', type=Path, default=root/'work/S2-COL-R1')
    parser.add_argument('--features', type=Path, default=root/'work/S2_COL_R2')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--train-pairs', type=int, default=256,
                        help='Engineering probe only; production train_arm always uses the full population')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--seed', type=int, default=13)
    args = parser.parse_args()
    if args.out.exists():
        parser.error('Use a new report path; existing audit results are immutable')
    if args.train_pairs < 32 or args.epochs < 1:
        parser.error('Use at least 32 probe pairs and one epoch')
    sources = {name: file_hash(root/'src'/name) for name in
               ('benchmark_selector_training.py', 'mmdd_stage2/column_batching.py',
                'mmdd_stage2/column_r2_training.py', 'mmdd_stage2/verifier.py')}
    torch.set_num_threads(4)
    started = time.perf_counter()
    train = inputs_for(args.r1, args.features, 'train')
    dev = inputs_for(args.r1, args.features, 'dev')
    metadata = {pair_key(r): r for r in read_jsonl(args.r1/'COLUMN_POPULATION.train.jsonl')}
    dev_metadata = read_jsonl(args.r1/'COLUMN_POPULATION.dev.jsonl')
    selected_keys = {pair_key(r) for r in sorted(train, key=lambda r: digest([args.seed, pair_key(r)]))[:args.train_pairs]}
    selected = [r for r in train if pair_key(r) in selected_keys]
    index = FeatureIndex(args.r1, args.features)
    feature_hashes = {}

    def cached(items, view):
        records = []
        for item in items:
            key = job_key(job(item, [], view))
            entry = index.entries[key]
            if file_hash(Path(entry['path'])) != entry['sha256']:
                raise ValueError('Cached selector feature changed')
            feature_hashes[key] = entry['sha256']
            records.append(index.get(item, [], view))
        return records

    views = [cached(selected, view) for view in (0, 1)]
    dev_records = cached(dev, 0)
    load_seconds = time.perf_counter() - started
    counts = Counter((r['dataset'], r['query_id']) for r in train)
    lakes = Counter(k[0] for k in counts)
    weights = {pair_key(r): len(train)/(len(lakes)*lakes[r['dataset']]*counts[r['dataset'], r['query_id']])
               for r in train}
    checkpoint = args.features/f'PRIOR/checkpoints/{args.seed}/selected.pt'
    checkpoint_hash = file_hash(checkpoint)
    frozen, _ = load_head(checkpoint)
    torch.manual_seed(args.seed)
    initial = CandidateColumnScorer(frozen.hidden_dim, head_type='mlp')
    initial_rng = torch.get_rng_state().clone()
    comparisons, probes, trained = {}, {}, {}
    for execution in ('scalar', 'batched'):
        model = copy.deepcopy(initial)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
        torch.manual_seed(71)
        loss, observed = step(model, optimizer, views[0][:32], metadata, weights, execution, inspect=True)
        probes[execution] = {**observed, 'loss': loss, 'state': copy.deepcopy(model.state_dict())}
        model = copy.deepcopy(initial)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
        torch.set_rng_state(initial_rng)
        history, train_seconds, dev_seconds, best = [], 0., 0., None
        for epoch in range(1, args.epochs+1):
            order = list(range(len(selected)))
            random.Random(args.seed*1000+epoch).shuffle(order)
            records = views[(epoch-1)%2]
            tick = time.perf_counter()
            losses = [step(model, optimizer, [records[i] for i in order[start:start+32]],
                           metadata, weights, execution)[0] for start in range(0, len(order), 32)]
            train_seconds += time.perf_counter()-tick
            tick = time.perf_counter()
            predictions = predict(model, dev_records, execution)
            metrics = evaluate(dev_metadata, predictions)[0]['query_macro']
            dev_seconds += time.perf_counter()-tick
            score = (metrics['MRR'], metrics['ColHit@1'])
            if best is None or score > best:
                best, best_epoch, best_metrics = score, epoch, metrics
            history.append({'epoch': epoch, 'loss': sum(losses)/len(losses), 'dev': metrics,
                            'order_sha256': digest(order)})
        # Replay the already-selected production checkpoint as a separate check.
        tick = time.perf_counter()
        frozen_predictions = predict(frozen, dev_records, execution)
        frozen_seconds = time.perf_counter()-tick
        trained[execution] = predictions
        comparisons[execution] = {'train_seconds': train_seconds, 'dev_seconds': dev_seconds,
                                  'selected_epoch': best_epoch, 'selected_dev': best_metrics,
                                  'frozen_dev_seconds': frozen_seconds, 'history': history,
                                  'frozen_predictions': frozen_predictions}
        print(f'{execution}: {len(selected)} pairs x {args.epochs} epochs, '
              f'train={train_seconds:.3f}s, dev={dev_seconds:.3f}s', flush=True)
    a, b = probes['scalar'], probes['batched']
    raw_update = {name: float((a['state'][name]-b['state'][name]).abs().max()) for name in a['state']}
    frozen_a = comparisons['scalar'].pop('frozen_predictions')
    frozen_b = comparisons['batched'].pop('frozen_predictions')
    max_frozen_diff = max(abs(x-y) for ra, rb in zip(frozen_a, frozen_b) for x, y in zip(ra['logits'], rb['logits']))
    report = {'scope': 'engineering_probe_not_formal_training', 'arm': 'PRIOR',
              'train_pairs': len(selected), 'formal_train_pairs': len(train), 'dev_pairs': len(dev),
              'epochs': args.epochs, 'seed': args.seed, 'threads': torch.get_num_threads(),
              'torch_version': torch.__version__, 'reader_forwards': 0, 'generation_calls': 0,
              'data_load_seconds': load_seconds, 'cached_inputs_sha256': digest(feature_hashes),
              'train_keys_sha256': digest([pair_key(r) for r in selected]),
              'checkpoint': str(checkpoint), 'checkpoint_sha256': checkpoint_hash,
              'checkpoint_unchanged': file_hash(checkpoint) == checkpoint_hash,
              'sources': sources,
              'sources_unchanged_during_run': all(file_hash(root/'src'/name) == sha for name, sha in sources.items()),
              'single_step': {'loss_abs_diff': abs(a['loss']-b['loss']),
                              'max_logit_abs_diff': float((a['logits']-b['logits']).abs().max()),
                              'max_gradient_abs_diff': float((a['gradient']-b['gradient']).abs().max()),
                              'rng_equal': torch.equal(a['rng'], b['rng']), 'parameter_abs_diff': raw_update},
              'frozen_full_dev': {'ranking_equal': sum(a['ranking']==b['ranking'] for a, b in zip(frozen_a, frozen_b)),
                                  'pairs': len(dev), 'max_logit_abs_diff': max_frozen_diff},
              'trained_probe_final_dev_ranking_equal': sum(a['ranking']==b['ranking'] for a, b in
                                                         zip(trained['scalar'], trained['batched'])),
              'comparisons': comparisons,
              'train_speedup': comparisons['scalar']['train_seconds']/comparisons['batched']['train_seconds'],
              'train_and_dev_speedup': sum(comparisons['scalar'][k] for k in ('train_seconds','dev_seconds')) /
                                      sum(comparisons['batched'][k] for k in ('train_seconds','dev_seconds'))}
    write_json(args.out, report)
    print(f'Report: {args.out}; train speedup {report["train_speedup"]:.2f}x', flush=True)


if __name__ == '__main__':
    main()
