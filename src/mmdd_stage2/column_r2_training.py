"""Matched full-population schedules and dev-only model selection for R2."""
from __future__ import annotations

import random
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import torch

from .column_batching import score_records
from .column_data import digest, file_hash, read_jsonl, write_json, write_jsonl
from .column_metrics import pair_key, prediction
from .column_r2_audit import read_json
from .column_r2_cache import FeatureIndex, job, job_key
from .column_r2_metrics import report_metrics
from .column_r2_models import EvidenceCorrection, complete_scores, mix_condition, shortlist
from .column_training import column_loss, parameter_hash
from .verifier import CandidateColumnScorer


@dataclass
class DevPlateauStopper:
    """Count dev-MRR plateaus only after warmup; checkpoint selection is separate."""
    patience: int = 5
    min_epochs: int = 10
    min_delta: float = .0005
    best_mrr: float | None = None
    bad_checks: int = 0

    def update(self, epoch: int, mrr: float) -> bool:
        if self.best_mrr is None or mrr > self.best_mrr + self.min_delta:
            self.best_mrr, self.bad_checks = mrr, 0
        elif epoch >= self.min_epochs:
            self.bad_checks += 1
        if epoch < self.min_epochs:
            self.bad_checks = 0
        return self.patience > 0 and epoch >= self.min_epochs and self.bad_checks >= self.patience


def inputs_for(r1: Path, output: Path, split: str) -> list[dict]:
    path = output / 'NATURAL_TRAIN/COLUMN_INPUTS.train.jsonl' if split == 'train' else r1 / f'COLUMN_INPUTS.{split}.jsonl'
    if split == 'train' and file_hash(path) != read_json(output/'NATURAL_TRAIN/MANIFEST.json')['inputs_sha256']:
        raise ValueError('Frozen natural train input changed')
    return read_jsonl(path)


def metadata_for(r1: Path, items: list[dict], objects: dict, split: str) -> list[dict]:
    lookup = {pair_key(p): p for p in read_jsonl(r1 / f'COLUMN_POPULATION.{split}.jsonl')}
    result = []
    for item in items:
        ids = item['evidence_ids']['O-R']
        result.append({**lookup[pair_key(item)], 'natural_evidence_empty': not ids,
            'natural_retrieval_available': True, 'evidence_count': len(ids),
            'modality': '+'.join(sorted({objects['evidence'][e]['asset_type'] for e in ids})) or 'none'})
    return result


def prepare_jobs(r1: Path, output: Path, kind: str) -> Path:
    jobs = {}
    splits = ('test',) if kind in {'test', 'prior-test'} else ('train', 'dev')
    for split in splits:
        for item in inputs_for(r1, output, split):
            for view in (0, 1) if split == 'train' else (0,):
                bundles = [[]]
                conditions = () if kind == 'prior-test' else ('O-O', 'O-R') if split == 'train' else ('O-R',)
                for condition in conditions:
                    ids = item['evidence_ids'][condition]
                    bundles.append(ids)
                    if kind in {'separate', 'test'}:
                        bundles.extend([[eid] for eid in ids])
                for ids in bundles:
                    record = job(item, ids, view)
                    jobs[job_key(record)] = record
    path = output / f'JOBS/{kind}.jsonl.gz'
    write_jsonl(path, list(jobs.values()))
    print(f'{kind}: {len(jobs)} unique feature inputs', flush=True)
    return path


class TrainingData:
    def __init__(self, r1: Path, output: Path) -> None:
        self.r1, self.output = r1, output
        self.index = FeatureIndex(r1, output)
        self.objects = read_jsonl(r1 / 'OBJECTS.jsonl.gz')[0]
        extra = output / 'NATURAL_TRAIN/EVIDENCE_OBJECTS.jsonl.gz'
        if file_hash(extra) != read_json(output/'NATURAL_TRAIN/MANIFEST.json')['objects_sha256']:
            raise ValueError('Frozen natural evidence objects changed')
        self.objects['evidence'].update({e['asset_id']: e for e in read_jsonl(extra)})
        self.loaded = {}
        self.fixed_prior = {}
        self.prior_inputs_frozen = False

    def freeze_prior_inputs(self, seed: int) -> None:
        manifest = read_json(self.output / 'PRIOR/SHORTLIST_MANIFEST.json')
        prior_hash = file_hash(self.output/f'PRIOR/checkpoints/{seed}/selected.pt')
        self.fixed_prior = {}
        for split, entry in manifest['files'].items():
            path = Path(entry['path'])
            if file_hash(path) != entry['sha256']:
                raise ValueError('Frozen prior shortlist changed')
            for row in read_jsonl(path):
                if row['seed'] == seed:
                    if row['prior_sha256'] != prior_hash:
                        raise ValueError('Frozen shortlist belongs to a different prior checkpoint')
                    self.fixed_prior[pair_key(row), row['view']] = row
        self.prior_inputs_frozen = True

    def get(self, item: dict, ids: list[str], view: int) -> dict:
        key = job_key(job(item, ids, view))
        if key not in self.loaded:
            entry = self.index.entries[key]
            if file_hash(Path(entry['path'])) != entry['sha256']:
                raise ValueError('Training feature hash changed')
            self.loaded[key] = self.index.get(item, ids, view)
        return self.loaded[key]

    def state(self, item: dict, ids: list[str], view: int) -> torch.Tensor:
        r = self.get(item, ids, view)
        return torch.cat([r['open_states'], r['close_states']], dim=-1)


def load_head(path: Path) -> tuple[torch.nn.Module, dict]:
    cp = torch.load(path, map_location='cpu', weights_only=True)
    meta = cp['metadata']
    if meta['arm'] in {'OO_CONTROL', 'FLAT_MIX', 'PRIOR'}:
        model = CandidateColumnScorer(meta['hidden_dim'], head_type='mlp')
    else:
        model = EvidenceCorrection(meta['hidden_dim'], meta['arm'] != 'PVR_BUNDLE')
    model.load_state_dict(cp['state_dict'])
    return model.eval(), meta


def prior_record(data: TrainingData, prior: torch.nn.Module, item: dict, view: int) -> tuple[dict, torch.Tensor, list[int]]:
    record = data.get(item, [], view)
    fixed = data.fixed_prior.get((pair_key(item), view))
    if data.prior_inputs_frozen and fixed is None:
        raise ValueError('Missing frozen shortlist; do not silently regenerate')
    if fixed is not None:
        columns = record['candidate_column_indices']
        if fixed['input_hash'] != record['input_hash'] or fixed['candidate_column_indices'] != columns:
            raise ValueError('Prior shortlist and feature identity differ')
        logits = torch.tensor(fixed['prior_logits'])
        positions = [columns.index(c) for c in fixed['shortlist_columns']]
        if positions != shortlist(logits, columns):
            raise ValueError('Saved shortlist is not label-blind prior Top3')
        return record, logits, positions
    with torch.no_grad():
        logits = prior(record['open_states'], record['close_states'])
    return record, logits, shortlist(logits, record['candidate_column_indices'])


def pvr_forward(data: TrainingData, model: EvidenceCorrection, prior: torch.nn.Module,
                item: dict, ids: list[str], view: int) -> tuple[dict, torch.Tensor, torch.Tensor, list[int]]:
    record, base, positions = prior_record(data, prior, item, view)
    h0 = data.state(item, [], view)[positions]
    if not ids:
        evidence = h0.new_empty((0, len(positions), h0.shape[-1]))
    elif model.separate:
        evidence = torch.stack([data.state(item, [eid], view)[positions] for eid in ids])
    else:
        evidence = data.state(item, ids, view)[positions].unsqueeze(0)
    mods = torch.tensor([int(data.objects['evidence'][e]['asset_type'] == 'image') for e in ids], dtype=torch.long)
    corrected, _ = model(base[positions], h0, evidence, mods)
    return record, base, corrected, positions


@torch.inference_mode()
def predict_model(data: TrainingData, model: torch.nn.Module, items: list[dict], arm: str,
                  prior: torch.nn.Module | None = None, view: int = 0, *,
                  execution: str = 'scalar') -> tuple[list[dict], list[dict]]:
    model.eval()
    predictions, priors = [], []
    if prior is None:
        for start in range(0, len(items), 32):
            batch = items[start:start+32]
            evidence = [[] if arm == 'PRIOR' else item['evidence_ids']['O-R'] for item in batch]
            records = [data.get(item, ids, view) for item, ids in zip(batch, evidence)]
            logits = score_records(model, records, execution=execution)
            predictions.extend(prediction(record, scores.tolist(), evidence_ids=ids)
                               for record, scores, ids in zip(records, logits, evidence))
        return predictions, priors
    for item in items:
        ids = [] if arm == 'PRIOR' else item['evidence_ids']['O-R']
        record, base, corrected, positions = pvr_forward(data, model, prior, item, ids, view)
        columns = record['candidate_column_indices']
        logits = complete_scores(base, corrected, positions, columns) if ids else base
        p = prediction(record, logits.tolist(), evidence_ids=ids,
            prior_logits=base.tolist(), shortlist_columns=[columns[i] for i in positions],
            shortlist_correction_logits=corrected.tolist(), outside_scores='ranking_only_sentinels' if ids else 'prior')
        priors.append(prediction(record, base.tolist()))
        predictions.append(p)
    return predictions, priors


def train_arm(r1: Path, output: Path, arm: str, seed: int, *, execution: str = 'batched',
              training_output: Path | None = None, epochs: int = 20,
              early_stopping_patience: int | None = None, min_epochs: int = 10,
              min_delta: float = .0005, log_every_pairs: int = 1024) -> dict:
    if execution not in {'scalar', 'batched'}:
        raise ValueError(f'Unknown head execution: {execution}')
    # Only the deployed selector changes its default stopping policy. Matched
    # evidence ablations retain complete schedules unless explicitly requested.
    patience = (5 if arm == 'PRIOR' else 0) if early_stopping_patience is None else early_stopping_patience
    if epochs < 1 or patience < 0 or min_epochs < 1 or min_delta < 0 or log_every_pairs < 0:
        raise ValueError('Invalid training or early-stopping schedule')
    stopper = DevPlateauStopper(patience, min_epochs, min_delta)
    training_output = training_output or output
    folder = training_output / f'{arm}/checkpoints/{seed}'
    if (folder/'MANIFEST.json').exists():
        raise ValueError('Completed training arm is immutable; use its selected checkpoint')
    torch.set_num_threads(4)
    started = time.perf_counter()
    data = TrainingData(r1, output)
    train, dev = inputs_for(r1, output, 'train'), inputs_for(r1, output, 'dev')
    train_meta = {pair_key(p): p for p in read_jsonl(r1 / 'COLUMN_POPULATION.train.jsonl')}
    dev_meta = metadata_for(r1, dev, data.objects, 'dev')
    pvr = arm.startswith('PVR_')
    dimension_ids = [] if arm == 'PRIOR' or pvr else train[0]['evidence_ids']['O-O']
    hidden_dim = data.get(train[0], dimension_ids, 0)['open_states'].shape[1]
    if pvr and not (output / 'PRIOR/SHORTLIST_MANIFEST.json').is_file():
        raise ValueError('Freeze label-blind prior shortlists before verifier training')
    prior_path = output / f'PRIOR/checkpoints/{seed}/selected.pt'
    prior = load_head(prior_path)[0].requires_grad_(False) if pvr else None
    if pvr:
        data.freeze_prior_inputs(seed)
    support = arm == 'PVR_SUPPORT'
    witnesses, donors = {}, {}
    if support:
        if not read_json(output/'SUPPORT_AUDIT/MANIFEST.json')['execute']:
            raise ValueError('Support supervision threshold not met')
        for w in read_jsonl(output/'SUPPORT_AUDIT/witnesses.jsonl.gz'):
            witnesses.setdefault(pair_key(w), []).append(w)
        donors = {pair_key(r):r['synthetic_negative_ids'] for r in read_jsonl(output/'SUPPORT_AUDIT/donors.jsonl.gz')}
    torch.manual_seed(seed)
    model = EvidenceCorrection(hidden_dim, arm != 'PVR_BUNDLE') if pvr else CandidateColumnScorer(hidden_dim, head_type='mlp')
    initial = parameter_hash(model)
    if arm in {'OO_CONTROL', 'FLAT_MIX', 'PRIOR'}:
        expected = read_json(r1 / f'RUN_RECEIPTS/C2/{seed}.json')['initial_parameter_hash']
        if initial != expected:
            raise ValueError('Flat/Prior initialization differs from R1 C2')
    lr = 1e-4 if pvr else 3e-4
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    folder.mkdir(parents=True, exist_ok=True)
    meta = {'arm': arm, 'seed': seed, 'hidden_dim': hidden_dim, 'initial_parameter_sha256': initial,
        'reader_layout': 'tail_candidates_v1', 'reader_identity_sha256': file_hash(r1/'READER_IDENTITY.json'),
        'data_manifest_sha256': file_hash(output/'NATURAL_TRAIN/MANIFEST.json'),
        'prior_sha256': file_hash(prior_path) if pvr else None,
        'sources': {p.name: file_hash(p) for p in sorted(Path(__file__).parent.glob('column_r2_*.py'))
                    + [Path(__file__).with_name(name) for name in ('column_batching.py', 'column_training.py', 'verifier.py')]},
        'optimizer': 'AdamW', 'lr': lr, 'weight_decay': 1e-4, 'gradient_clip': 1., 'effective_batch': 32,
        'epochs': epochs, 'head_parameters': sum(p.numel() for p in model.parameters()),
        'early_stopping': {'enabled': patience > 0, 'metric': 'dev query_macro MRR',
                           'patience': patience, 'min_epochs': min_epochs, 'min_delta': min_delta,
                           'check_every': 'complete_epoch', 'restore': 'best_selected_checkpoint'},
        'log_every_pairs': log_every_pairs,
        'selection': 'dev No-E MRR, H1, earlier epoch' if arm == 'PRIOR' else
            'dev O-R full MRR, non-empty H1, lower damage (PVR only), earlier epoch',
        'condition_schedule': 'No-E' if arm == 'PRIOR' else 'O-O' if arm == 'OO_CONTROL' else
            'sha256(seed,epoch,query_id,target_id) parity: even O-O, odd O-R',
        'view_schedule': '(epoch-1)%2', 'prior_frozen': pvr, 'backbone_frozen': True,
        'head_execution': 'scalar' if pvr else execution,
        'dropout_schedule': 'per_pair_original_order_and_shape',
        'training_feature_root': str(output.resolve())}
    runtime_manifest = output/'RUNTIME_SOURCE/MANIFEST.json'
    if runtime_manifest.exists():
        meta['runtime_source_manifest_sha256'] = file_hash(runtime_manifest)
    if pvr:
        meta['shortlist_manifest_sha256'] = file_hash(output/'PRIOR/SHORTLIST_MANIFEST.json')
    if support:
        meta.update(support_lambda=.2, support_audit_sha256=file_hash(output/'SUPPORT_AUDIT/MANIFEST.json'),
                    support_negative='Shuffled-E synthetic_negative; unknown natural evidence excluded')
    torch.save({'metadata': meta, 'state_dict': model.state_dict()}, folder/'epoch0.pt')
    counts = Counter((p['dataset'], p['query_id']) for p in train)
    lakes = Counter(k[0] for k in counts)
    weights = {pair_key(p): len(train)/(len(lakes)*lakes[p['dataset']]*counts[p['dataset'],p['query_id']]) for p in train}
    history, best, steps, stopped_early = [], None, 0, False
    for epoch in range(1, epochs+1):
        epoch_started = time.perf_counter()
        model.train()
        view = (epoch-1)%2
        order = list(range(len(train)))
        random.Random(seed*1000+epoch).shuffle(order)
        visits, selected_conditions, losses = [], [], []
        progress, recent_losses, seen_queries = [], [], set()
        next_log = log_every_pairs
        admitted = nonempty = support_pairs = 0
        for start in range(0, len(order), 32):
            batch = order[start:start+32]
            optimizer.zero_grad(set_to_none=True)
            batch_losses = []
            prepared = []
            for i in batch:
                item = train[i]
                key = pair_key(item)
                visits.append(key)
                seen_queries.add((item['dataset'], item['query_id']))
                condition = 'No-E' if arm == 'PRIOR' else 'O-O' if arm == 'OO_CONTROL' else mix_condition(seed,epoch,item['query_id'],item['target_id'])
                ids = [] if condition == 'No-E' else item['evidence_ids'][condition]
                selected_conditions.append([key, condition, ids])
                nonempty += int(bool(ids))
                prepared.append((item, key, ids))
            if not pvr:
                records = [data.get(item, ids, view) for item, _, ids in prepared]
                scores = score_records(model, records, execution=execution)
            for j, (item, key, ids) in enumerate(prepared):
                gold = train_meta[key]['gold_column_indices']
                if pvr:
                    record, base, logits, positions = pvr_forward(data, model, prior, item, ids, view)
                    columns = [record['candidate_column_indices'][i] for i in positions]
                    if not set(gold).intersection(columns):
                        continue
                else:
                    record = records[j]
                    columns = record['candidate_column_indices']
                    logits = scores[j]
                admitted += 1
                loss = column_loss(logits, columns, gold)
                if support and ids and key in witnesses:
                    eligible = [w for w in witnesses[key] if w['evidence_id'] in ids and w['column_index'] in columns]
                    auxiliary = []
                    for w in eligible:
                        canonical_position = record['candidate_column_indices'].index(w['column_index'])
                        positive_id = w['evidence_id']
                        all_ids = [positive_id] + donors[key]
                        h0 = data.state(item, [], view)[[canonical_position]]
                        he = torch.stack([data.state(item, [eid], view)[[canonical_position]] for eid in all_ids])
                        modalities = torch.tensor([int(data.objects['evidence'][e]['asset_type']=='image') for e in all_ids])
                        _, _, scores = model.item_states(h0, he, modalities)
                        auxiliary.append(torch.nn.functional.softplus(-(scores[0]-scores[1:])).mean())
                    if auxiliary:
                        loss = loss + .2*torch.stack(auxiliary).mean()
                        support_pairs += 1
                batch_losses.append(weights[key]*loss)
                losses.append(float(loss.detach()))
                recent_losses.append(losses[-1])
            # Keep the same base-batch schedule even for all-empty/all-missed verifier batches.
            loss = torch.stack(batch_losses).sum()/len(batch) if batch_losses else torch.tensor(0.)
            loss = loss + next(model.parameters()).sum()*0
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            if not torch.isfinite(norm):
                raise ValueError('Non-finite R2 gradient')
            optimizer.step()
            steps += 1
            if log_every_pairs and (len(visits) >= next_log or len(visits) == len(train)):
                point = {'pairs_seen': len(visits), 'unique_queries_seen': len(seen_queries),
                         'optimizer_steps': steps, 'view': view,
                         'mean_loss': sum(recent_losses)/len(recent_losses) if recent_losses else None}
                progress.append(point)
                print(f'{arm} seed{seed} epoch{epoch}: {len(visits)}/{len(train)} Q-T pairs, '
                      f'{len(seen_queries)} unique queries, recent loss={point["mean_loss"]}', flush=True)
                recent_losses = []
                next_log = (len(visits)//log_every_pairs+1)*log_every_pairs
        train_seconds = time.perf_counter() - epoch_started
        dev_started = time.perf_counter()
        predictions, prior_predictions = predict_model(data, model, dev, arm, prior, execution=execution)
        metrics, _ = report_metrics(dev_meta, predictions, prior_predictions if pvr else None)
        macro = metrics['query_macro']
        nonempty_h1 = metrics['subsets']['non_empty']['query_macro']['ColHit@1']
        damage = metrics['subsets']['full'].get('damage_rate') or 0.
        score = (macro['MRR'], macro['ColHit@1']) if arm == 'PRIOR' else (macro['MRR'], nonempty_h1, -damage if pvr else 0.)
        stopped_early = stopper.update(epoch, macro['MRR'])
        history.append({'epoch': epoch, 'base_samples_seen': len(visits), 'unique_base_visits': len(set(visits)),
            'visits_sha256': digest(visits), 'condition_and_evidence_schedule_sha256': digest(selected_conditions),
            'conditions': dict(Counter(c[1] for c in selected_conditions)), 'nonempty_samples': nonempty,
            'admitted_samples': admitted, 'optimizer_steps': steps, 'train_loss': sum(losses)/len(losses),
            'support_pairs': support_pairs,
            'loss_progress': progress, 'early_stopping_bad_checks': stopper.bad_checks,
            'early_stopping_reference_mrr': stopper.best_mrr,
            'early_stop_triggered': stopped_early,
            'train_seconds': train_seconds, 'dev_seconds': time.perf_counter() - dev_started,
            'dev_metrics': metrics, 'parameter_sha256': parameter_hash(model)})
        if best is None or score > best:
            best = score
            selected_epoch = epoch
            selected_meta = {**meta, 'selected_epoch': epoch, 'parameter_sha256': parameter_hash(model)}
            torch.save({'metadata': selected_meta, 'state_dict': model.state_dict()}, folder/'selected.pt')
        write_json(folder/'history.json', history)
        print(f'{arm} seed{seed} epoch{epoch}: loss={history[-1]["train_loss"]:.5f} dev H1={macro["ColHit@1"]:.5f} MRR={macro["MRR"]:.5f}', flush=True)
        if stopped_early:
            print(f'Early stop at epoch {epoch}; restore best checkpoint from epoch {selected_epoch}.', flush=True)
            break
    loaded, _ = load_head(folder/'selected.pt')
    predictions, prior_predictions = predict_model(data, loaded, dev, arm, prior, execution=execution)
    metrics, _ = report_metrics(dev_meta, predictions, prior_predictions if pvr else None)
    write_jsonl(training_output/f'{arm}/predictions/{seed}/dev.jsonl.gz', predictions)
    write_json(training_output/f'{arm}/metrics/{seed}/dev.json', metrics)
    feature_hashes = {k: data.index.entries[k]['sha256'] for k in sorted(data.loaded)}
    write_json(folder/'FEATURES.json', feature_hashes)
    receipt = {**meta, 'selected_epoch': selected_epoch, 'optimizer_steps': steps,
        'actual_epochs': len(history), 'stopped_early': stopped_early,
        'stop_reason': 'dev_mrr_plateau' if stopped_early else 'epoch_budget',
        'wall_seconds': time.perf_counter() - started,
        'selected_sha256': file_hash(folder/'selected.pt'), 'feature_hashes_sha256': file_hash(folder/'FEATURES.json'),
        'planned': True, 'implemented': True, 'actually_executed': True, 'actually_evaluated': ['dev']}
    write_json(folder/'MANIFEST.json', receipt)
    return receipt


@torch.inference_mode()
def freeze_shortlists(r1: Path, output: Path) -> None:
    if (output/'PRIOR/SHORTLIST_MANIFEST.json').exists():
        raise ValueError('Prior shortlists already frozen')
    torch.set_num_threads(4)
    data = TrainingData(r1, output)
    manifests = {}
    for split in ('train', 'dev', 'test'):
        rows = []
        items = inputs_for(r1, output, split)
        for seed in (13, 29):
            path = output / f'PRIOR/checkpoints/{seed}/selected.pt'
            prior_hash = file_hash(path)
            prior = load_head(path)[0].requires_grad_(False)
            for item in items:
                for view in (0, 1) if split == 'train' else (0,):
                    record, logits, positions = prior_record(data, prior, item, view)
                    rows.append({**{k: item[k] for k in ('dataset', 'query_id', 'target_id')},
                        'seed': seed, 'view': view, 'prior_sha256': prior_hash,
                        'candidate_column_indices': record['candidate_column_indices'], 'prior_logits': logits.tolist(),
                        'shortlist_columns': [record['candidate_column_indices'][i] for i in positions],
                        'input_hash': record['input_hash']})
        path = output / f'PRIOR/shortlist.{split}.jsonl.gz'
        write_jsonl(path, rows)
        manifests[split] = {'path': str(path), 'sha256': file_hash(path), 'rows': len(rows)}
    write_json(output / 'PRIOR/SHORTLIST_MANIFEST.json', {'label_blind': True, 'shortlist_size': 3,
        'no_gold_insertion': True, 'files': manifests, 'test_labels_read': False})
