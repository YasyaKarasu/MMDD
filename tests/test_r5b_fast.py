"""CPU integration with synthetic tasks and the actual archived R5b functions."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "audit/MMDD_S2_R5b_EvidenceCalibration_FrozenQET_S13_GPU1_v1_PACKAGE"


@pytest.mark.skipif(not PACKAGE.is_dir(), reason="Requires the local R5b source package")
def test_fast_generation_resume_and_native_feature_extraction(tmp_path):
    # A separate interpreter prevents the archive's unqualified module names
    # (common, features, engine) leaking into other repository tests.
    script = r'''
import sys, types, os
from pathlib import Path
import numpy as np
import run_r5b_fast as fast
package, out = map(Path, sys.argv[1:])
snap, cache, run = out/'snapshot', out/'cache', out/'run'
ids = ['q_fit', 'q_val']
base = [f't{i:02}' for i in range(50)]
config = fast.read(package/'train_runtime/config/LOCKED.json')
config['cohorts'] = {'train': 2}
pop = [{'query_id': q, 'source_group': q, 'split': 'train'} for q in ids]
rows = [{'query_row_id': i, 'cells': [{'column_id': 0, 'column_name': 'name', 'text': name}]} for i, name in enumerate(['Alpha','Bravo','Charlie','Delta','Echo'])]
tables = {t: {'columns': [{'column_id': 0, 'column_name': 'nation', 'attribute': 'nation', 'values': []}]} for t in base}
d = {'cohort': 'train', 'baselines': {q: base for q in ids}, 'query_rows': {q: rows for q in ids}, 'source_groups': {q:q for q in ids},
     'tables': tables, 'assets': {'e': {'asset_id':'e', 'asset_type':'text', 'content':'Alpha Bravo Charlie Delta Echo'}}}
plans = [{'query_id':q, 'A_P0': {'views':[{'column_name':'nation','attribute':'nation','evidence_ids':['e'], 'priority_pair_rank':1,
           'donor_links':[{'target_id':base[0], 'column_id':0,'column_name':'nation','stage1_rank':1,'pair_rank':1}]}]}} for q in ids]
fast.write(snap/'prepared/train/INPUTS.json', d)
fast.write(snap/'prepared/train/PLANS.json', plans)
fast.write(snap/'PREPARATION.json', {'query_signatures':dict.fromkeys(ids,'synthetic')})
fast.write(snap/'RUNTIME_CONFIG.json', config)
fast.write(snap/'population/POPULATION.json', pop)
for q in ids:
    fast.write(snap/'stage1/train'/f'{q}.json', {'C50':base,'path_ranking':base,'path_logits':{t:float(-i) for i,t in enumerate(base)}})
state = dict(package=package, snap=snap, cache=cache, run=run, config=config, identity='synthetic-v1', cohort={'query_ids':ids,'full_train_queries':2})
rt = fast.bind_runtime(state)
rt.seal(snap,'FAST_PREPARE_SEAL.json',[snap/'prepared/train/INPUTS.json',snap/'prepared/train/PLANS.json',snap/'PREPARATION.json'])
counter = {'loads':0,'calls':0}
class FakeEngine:
    def __init__(self,*args): counter['loads']+=1
    def run_tasks(self,tasks,batch_size,phase):
        counter['calls']+=len(tasks)
        return [{'task_id':t['task_id'],'raw_completion':'null','raw_batch_ref':None} for t in tasks], {}
fake = types.ModuleType('engine');fake.Engine = FakeEngine;sys.modules['engine'] = fake
fast.generate(state)
assert counter == {'loads':1, 'calls':10}, counter
fast.generate(state)
assert counter == {'loads':1, 'calls':10}, counter
assert fast.read(snap/'GENERATION_RECEIPT.json')['resumed'] == 2
fast.link_file(cache/'query_runs',snap/'query_runs')
sys.path.insert(0,str(package/'code'))
import r5b_common
r5b_common.ROOT = run/'experiment'
fast.write(run/'experiment/reference/train_ids.json',ids)
fast.link_file(package/'code',run/'experiment/code')
from build_features import run as build
os.environ['CUDA_VISIBLE_DEVICES']=''
build(snap,'train',run/'features')
data = np.load(run/'features/train.npz',allow_pickle=False)
assert data['x'].shape == (2,50,32)
assert np.all(data['rec_score'] == 0)
receipt = fast.read(run/'features/train_receipt.json')
assert receipt['A_full_rankings_equal'] == receipt['PURE_full_rankings_equal'] == 2
dest = cache/'query_runs/train/q_fit/N_MISSING_ALL/rankings.json'
fast.write(dest,{'tampered':True})
try: fast.generate(state)
except RuntimeError as e: assert 'SEALED_FILE_CHANGED' in str(e)
else: raise AssertionError('Corrupt query was silently resumed')
print('Synthetic generation, exact resume, feature extraction, and corruption rejection passed')
'''
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "CUDA_VISIBLE_DEVICES": ""}
    result = subprocess.run([sys.executable, "-c", script, str(PACKAGE), str(tmp_path)], cwd=tmp_path,
                            env=env, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr


def test_cancelled_calibration_cli_never_initializes(monkeypatch):
    import run_r5b_fast as fast
    def forbidden(*args):
        raise AssertionError('Cancelled calibration must not initialize a train workload')
    monkeypatch.setattr(fast, 'initialize', forbidden)
    monkeypatch.setattr(sys, 'argv', ['run_r5b_fast.py', 'all'])
    with pytest.raises(SystemExit) as stopped:
        fast.main()
    assert stopped.value.code == 2
