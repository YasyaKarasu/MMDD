from __future__ import annotations

import py_compile
import marshal
from datetime import datetime, timedelta, timezone

from audit_stage1_r15_runtime_sources import (
    cache_record,
    code_equivalent,
    local_import_closure,
)


def test_local_import_closure_follows_only_local_modules(tmp_path):
    source = tmp_path / "src"
    source.mkdir()
    (source / "entry.py").write_text("import helper\nimport json\n", encoding="utf-8")
    (source / "helper.py").write_text("VALUE = 1\n", encoding="utf-8")
    closure = local_import_closure(tmp_path, source / "entry.py")
    assert closure == {"helper": source / "helper.py"}


def test_cache_record_matches_exact_compiled_source(tmp_path):
    source = tmp_path / "example.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    pyc = tmp_path / "__pycache__/example.cpython-310.pyc"
    pyc.parent.mkdir()
    py_compile.compile(str(source), cfile=str(pyc), doraise=True)
    cached = marshal.loads(pyc.read_bytes()[16:])
    compiled = compile(
        source.read_text(), cached.co_filename, "exec", dont_inherit=True
    )
    names = [name for name in dir(cached) if name.startswith("co_")
             and not callable(getattr(cached, name))]
    differences = [name for name in names if getattr(cached, name) != getattr(compiled, name)]
    assert code_equivalent(cached, compiled), differences
    record = cache_record(
        source, datetime.now(timezone.utc) + timedelta(seconds=1)
    )
    assert record["passed"] is True, record
    assert record["cached_bytecode_matches_current_source"] is True
