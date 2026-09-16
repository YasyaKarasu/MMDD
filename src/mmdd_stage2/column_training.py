"""Train heads on frozen features with complete base-sample epochs."""
from __future__ import annotations

import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import torch

from .checkpoints import load_candidate_scorer, save_candidate_scorer
from .column_cache import LAYOUTS, load_features, source_fingerprints
from .column_data import digest, file_hash, read_jsonl, write_json, write_jsonl
from .column_metrics import evaluate, pair_key, prediction
from .verifier import CandidateColumnScorer


def parameter_hash(scorer: CandidateColumnScorer) -> str:
    h = hashlib.sha256()
    for name, value in scorer.state_dict().items():
        h.update(name.encode())
        h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


@torch.inference_mode()
def predict(scorer: CandidateColumnScorer, records: list[dict[str, Any]], model_hash: str) -> list[dict[str, Any]]:
    scorer.eval()
    outputs = []
    for r in records:
        if r['status'] != 'ok':
            outputs.append({**{k: r[k] for k in ('dataset', 'query_id', 'target_id', 'candidate_column_indices', 'status')},
                            'reason': r.get('reason', r['status']), 'model_hash': model_hash, 'input_hash': r.get('input_hash')})
            continue
        try:
            logits = scorer(r['open_states'], r['close_states']).tolist()
            outputs.append(prediction(r, logits, model_hash=model_hash, input_hash=r['input_hash']))
        except (ValueError, RuntimeError) as error:
            outputs.append({**{k: r[k] for k in ('dataset', 'query_id', 'target_id', 'candidate_column_indices')},
                            'status': type(error).__name__, 'reason': str(error), 'model_hash': model_hash,
                            'input_hash': r['input_hash']})
    return outputs


def column_loss(logits: torch.Tensor, columns: list[int], gold: list[int]) -> torch.Tensor:
    positions = [i for i, c in enumerate(columns) if c in gold]
    if not positions:
        raise ValueError('Supervision has no mapped candidate')
    return torch.logsumexp(logits, 0) - torch.logsumexp(logits[positions], 0)


def train_head(output: Path, arm: str, seed: int, train_paths: list[Path], dev_path: Path,
               *, tiny: bool = False, epochs: int = 20) -> dict[str, Any]:
    torch.set_num_threads(4)
    torch.manual_seed(seed)
    layout = LAYOUTS[arm]
    loaded = [load_features(p, expected_layout=layout) for p in train_paths]
    dev_manifest, dev = load_features(dev_path, expected_layout=layout)
    train_meta = read_jsonl(output / 'COLUMN_POPULATION.train.jsonl')
    dev_meta = read_jsonl(output / 'COLUMN_POPULATION.dev.jsonl')
    if not tiny and len(loaded) != 2:
        raise ValueError('Formal training requires exactly two alternating training views')
    train = [records for _, records in loaded]
    lookup = {pair_key(r): r for r in train_meta}
    keys = [pair_key(r) for r in train[0]]
    if not tiny and set(keys) != set(lookup):
        raise ValueError('Training cache does not cover frozen train population')
    if any([pair_key(r) for r in rs] != keys for rs in train):
        raise ValueError('Training views have different populations/order')
    for m, _ in loaded:
        if m['contract']['split'] != 'train' or m['contract']['condition'] != 'O-O':
            raise ValueError('C0/C1/C2 must train on O-O train only')
        if m['contract']['input_manifest_sha256'] != file_hash(output / 'INPUT_MANIFEST.json'):
            raise ValueError('Training manifest identity changed')
    if not tiny and {m['contract']['view'] for m, _ in loaded} != {0, 1}:
        raise ValueError('Formal training views must be 0 and 1')
    if not tiny and (dev_manifest['contract']['split'] != 'dev' or dev_manifest['contract']['condition'] != 'O-O'):
        raise ValueError('Selection must use O-O dev')
    if tiny:
        if len(train[0]) < 32 or any(r['status'] != 'ok' for r in train[0][:32]):
            raise ValueError('Tiny-overfit requires 32 successfully cached train pairs')
        train = [rs[:32] for rs in train]
        keys = keys[:32]
        dev, dev_meta = train[0], [lookup[k] for k in keys]
        epochs = 300
    else:
        gate = output / 'RUN_RECEIPTS' / arm / f'tiny_{seed}.json'
        if not gate.is_file() or not json.loads(gate.read_text()).get('passed'):
            raise ValueError('Run matching-seed 32-sample tiny-overfit before formal training')
    scorer = CandidateColumnScorer(loaded[0][0]['hidden_dim'], head_type='mlp' if arm == 'C2' else 'linear')
    optimizer = torch.optim.AdamW(scorer.parameters(), lr=3e-4 if arm == 'C2' else 1e-3, weight_decay=1e-4)
    folder = output / 'CHECKPOINTS' / arm / (f'tiny_{seed}' if tiny else str(seed))
    meta = {'head_type': scorer.head_type, 'reader_layout_version': layout, 'loss_type': 'set_mass_CE',
            'model_dir': loaded[0][0]['contract'].get('model_dir'),
            'input_dim': scorer.input_dim, 'hidden_dim': scorer.hidden_dim,
            'optimizer': 'AdamW', 'learning_rate': 3e-4 if arm == 'C2' else 1e-3,
            'weight_decay': 1e-4, 'effective_batch_pairs': 32, 'gradient_clip': 1., 'training_device': 'cpu',
            'seed': seed, 'input_manifest_hash': file_hash(output / 'INPUT_MANIFEST.json'),
            'cache_hashes': [file_hash(p) for p in train_paths], 'cache_contract': loaded[0][0]['contract'],
            'selection': 'dev query-macro MRR, ColHit@3, ColHit@1, earlier epoch',
            'sources': source_fingerprints(), 'epochs_planned': epochs}
    initial = parameter_hash(scorer)
    save_candidate_scorer(folder / 'epoch0.pt', scorer, metadata={**meta, 'selected_epoch': 0})
    before = predict(scorer, dev, initial)
    query_counts = Counter((lookup[k]['dataset'], lookup[k]['query_id']) for k in keys)
    lake_queries = Counter(k[0] for k in query_counts)
    weights = {k: len(keys) / (len(lake_queries) * lake_queries[k[0]] * query_counts[(k[0], k[1])]) for k in keys}
    history, best, best_epoch, optimizer_steps = [], None, 0, 0
    for epoch in range(1, epochs + 1):
        records = train[(epoch - 1) % len(train)]
        order = list(range(len(records)))
        random.Random(seed * 1000 + epoch).shuffle(order)
        losses, grads, visits, failures = [], [], [], []
        for offset in range(0, len(order), 32):
            scorer.train()
            optimizer.zero_grad()
            batch = order[offset:offset + 32]
            batch_losses = []
            for i in batch:
                r = records[i]
                key = pair_key(r)
                visits.append(key)
                if r['status'] != 'ok':
                    failures.append({'key': key, 'status': r['status']})
                    continue
                logits = scorer(r['open_states'], r['close_states'])
                loss = column_loss(logits, r['candidate_column_indices'], lookup[key]['gold_column_indices'])
                batch_losses.append(loss * weights[key])
                losses.append(float(loss.detach()))
            if batch_losses:
                (torch.stack(batch_losses).sum() / len(batch)).backward()
                grad = torch.nn.utils.clip_grad_norm_(scorer.parameters(), 1.)
                if not torch.isfinite(grad):
                    raise ValueError('Non-finite head gradient')
                grads.append(float(grad))
                optimizer.step()
                optimizer_steps += 1
        model_hash = parameter_hash(scorer)
        predictions = predict(scorer, dev, model_hash)
        metrics, _ = evaluate(dev_meta, predictions)
        macro = metrics['query_macro']
        score = (macro['MRR'], macro['ColHit@3'], macro['ColHit@1'])
        if best is None or score > best:
            best, best_epoch = score, epoch
            save_candidate_scorer(folder / 'selected.pt', scorer, metadata={**meta, 'selected_epoch': epoch})
        if epoch in {1, 2, 5, 10, 15, 20} or epoch == epochs:
            save_candidate_scorer(folder / f'epoch{epoch}.pt', scorer, metadata={**meta, 'selected_epoch': epoch})
        history.append({'epoch': epoch, 'view': (epoch - 1) % len(train), 'base_visits': len(visits),
                        'unique_base_visits': len(set(visits)), 'visits_sha256': digest(visits),
                        'optimizer_steps_total': optimizer_steps, 'loss': sum(losses)/len(losses) if losses else None,
                        'grad_norm_mean': sum(grads)/len(grads) if grads else None, 'failures': failures,
                        'dev_query_macro': macro, 'parameter_sha256': model_hash})
        write_json(folder / 'history.json', history)
        if epoch % (25 if tiny else 1) == 0:
            print(f'{arm} seed{seed} {"tiny" if tiny else "formal"} epoch {epoch}: loss={history[-1]["loss"]} dev MRR={score[0]:.4f}', flush=True)
        if tiny and macro['ColHit@1'] >= .999 and history[-1]['loss'] < .03:
            break
    save_candidate_scorer(folder / 'end.pt', scorer, metadata={**meta, 'selected_epoch': len(history)})
    after = predict(scorer, dev, parameter_hash(scorer))
    reload_head = load_candidate_scorer(folder / 'end.pt', torch.device('cpu'), expected_reader_layout=layout)
    reloaded = predict(reload_head, dev, parameter_hash(reload_head))
    if after != reloaded:
        raise ValueError('Saved/reloaded head outputs changed')
    receipt = {'planned': True, 'implemented': True, 'executed': True, 'evaluated': True,
               'command': sys.argv, 'scope': 'tiny-overfit, not scientific accuracy' if tiny else 'formal O-O training and dev selection',
               'arm': arm, 'seed': seed, 'actual_epochs': len(history), 'optimizer_steps': optimizer_steps,
               'head_parameters': sum(p.numel() for p in scorer.parameters()),
               'initial_parameter_hash': initial, 'end_parameter_hash': parameter_hash(scorer),
               'selected_epoch': best_epoch, 'checkpoint_hashes': {p.name: file_hash(p) for p in folder.glob('*.pt')},
               'reload_equal': after == reloaded, 'head_updated': initial != parameter_hash(scorer),
               'backbone_frozen': all(m['backbone_frozen'] and m['parameter_versions_unchanged'] for m, _ in loaded),
               'logits_changed': [p.get('logits') for p in before] != [p.get('logits') for p in after],
               'ranking_changed': [p.get('ranking') for p in before] != [p.get('ranking') for p in after],
               'passed': bool(best[2] >= .95 and initial != parameter_hash(scorer)) if tiny else True,
               'history_path': str(folder / 'history.json'), **meta}
    write_json(output / 'RUN_RECEIPTS' / arm / (f'tiny_{seed}.json' if tiny else f'{seed}.json'), receipt)
    if tiny:
        write_json(folder / 'before_after_logits.json', {'before': before, 'after': after})
    return receipt


def evaluate_head(output: Path, checkpoint: Path, cache_path: Path, arm: str, seed: int) -> dict[str, Any]:
    manifest, records = load_features(cache_path, expected_layout=LAYOUTS[arm])
    scorer = load_candidate_scorer(checkpoint, torch.device('cpu'), expected_reader_layout=LAYOUTS[arm])
    c = manifest['contract']
    metadata = torch.load(checkpoint, map_location='cpu', weights_only=True)['metadata']
    train_contract = metadata['cache_contract']
    for field in ('model_files', 'sources', 'versions', 'dtype', 'image_policy', 'anonymize_evidence', 'input_manifest_sha256'):
        if train_contract[field] != c[field]:
            raise ValueError(f'Checkpoint and inference cache disagree: {field}')
    population = read_jsonl(output / f'COLUMN_POPULATION.{c["split"]}.jsonl')
    predictions = predict(scorer, records, file_hash(checkpoint))
    metrics, _ = evaluate(population, predictions)
    metrics['natural_retrieval_available_pairs'] = sum(p.get('natural_retrieval_available', False) for p in population)
    metrics['scientific_status'] = ('missing_retrieval_artifact_for_part_of_population' if c['condition'] == 'O-R'
                                    and not all(p.get('natural_retrieval_available', False) for p in population)
                                    else 'evaluated')
    name = f'{c["split"]}.{c["condition"]}.view{c["view"]}'
    write_jsonl(output / 'PREDICTIONS' / arm / str(seed) / f'{name}.jsonl.gz', predictions)
    write_json(output / 'METRICS' / arm / str(seed) / f'{name}.json', metrics)
    return metrics
