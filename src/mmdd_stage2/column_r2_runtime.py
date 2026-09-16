"""Freeze experiment imports without reverting concurrent workspace changes."""
from __future__ import annotations

import ast
import shutil
from pathlib import Path

from .column_data import file_hash, write_json
from .column_r2_audit import read_json


def freeze_runtime(root: Path, r1: Path, output: Path) -> None:
    destination = output/'RUNTIME_SOURCE'
    if destination.exists():
        raise ValueError('Runtime snapshot already exists')
    original = {r['path']:r['sha256'] for r in read_json(r1/'SOURCE_SNAPSHOT/MANIFEST.json')['files']}
    permitted_unused = {'src/mmdd_stage2/column_data.py':{'audit_data'},
                        'src/mmdd_stage2/oracle.py':{'load_oracle_column_data'}}
    drift = []
    for name, expected in original.items():
        current, snapshot = root/name, r1/'SOURCE_SNAPSHOT'/name
        if file_hash(snapshot) != expected:
            raise ValueError('R1 source snapshot changed')
        if file_hash(current) != expected:
            ignored = permitted_unused.get(name,set())
            def active_tree(path: Path) -> str:
                tree = ast.parse(path.read_text())
                tree.body = [n for n in tree.body if not isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) or n.name not in ignored]
                return ast.dump(tree,include_attributes=False)
            if not ignored or active_tree(current) != active_tree(snapshot):
                raise ValueError(f'Concurrent changes affect active experiment code: {name}')
            drift.append({'path':name,'original_sha256':expected,'workspace_sha256':file_hash(current),
                'changed_unused_definitions':sorted(ignored),'all_other_AST_identical':True})
    files = []
    for current in sorted((root/'src').rglob('*.py')):
        relative = current.relative_to(root)
        source = r1/'SOURCE_SNAPSHOT'/relative if str(relative) in original else current
        target = destination/relative
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,target)
        files.append({'path':str(relative),'sha256':file_hash(target),
            'source':'audited R1 snapshot' if str(relative) in original else 'R2/current dependency source'})
    write_json(destination/'MANIFEST.json',{'files':files,'workspace_drift':drift,
        'reason':'Preserve user-owned concurrent edits while keeping the experimental runtime frozen.',
        'active_R1_code_unchanged_before_snapshot':True,'secret_files_copied':False,
        'launch_root':str(root),'entrypoint':str(destination/'src/run_stage2_columns_r2.py')})
    print(f'Frozen runtime: {destination}; {len(files)} Python sources; {len(drift)} unused-function drifts',flush=True)
