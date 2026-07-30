# Fast WDC Scale-Gate Sampling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in deterministic `global_lowest` mode that produces exact low-row 100/1,000-table WDC system gates while preserving round-robin defaults and formal 200K behavior.

**Architecture:** Keep the helper's current two-pass validation and single safe publication path. Select candidates through either the unchanged per-bucket round-robin pool or one `N`-bounded global heap ranked by `(rows, stable_hash(seed, relative_path), relative_path)`, then pass both modes through identical ZIP, symlink, checksum, staging-validation, and no-clobber publication logic.

**Tech Stack:** Python 3.10+, `argparse`, `heapq`, SQLite, `pathlib`, ZIP/JSON, pytest, existing WDC scale-gate helper.

## Global Constraints

- `--selection_mode` accepts exactly `round_robin` and `global_lowest`.
- Python API and CLI default to `round_robin`; existing calls and output order remain unchanged.
- `global_lowest` returns the exact globally smallest `N` existing tables by `(rows, stable_hash(seed, relative_path), relative_path)`.
- Global selection retains at most `N` ranked candidates in memory.
- Both modes perform both strict catalog/path-validation passes and use the same SQLite dedupe index.
- Both modes use the existing filtered ZIP, absolute symlink, source checksum, staging validation, fsync, and `RENAME_NOREPLACE` publication path.
- Only the documented 100/1,000 quick gates opt into `global_lowest`; the formal 200K selector is unchanged.
- Quick-gate samples are documented as non-representative of category balance, table quality, and row-count distribution.
- Use the `MMDD` conda environment for every Python and pytest command.

---

## File and Module Map

- Modify `scripts/create_wdc200k_scale_gate_input.py`: add mode validation,
  the bounded global candidate pool, selector dispatch, metadata, and CLI
  plumbing.
- Modify `tests/test_wdc200k_scale_gate_input.py`: cover backward
  compatibility, exact global selection, determinism, bounded retention,
  shared validation, shared artifacts, and shared atomic publication.
- Modify `README.md`: make only the 100/1,000 quick-gate commands request
  `global_lowest` and state its purpose and sampling limitation.
- Modify
  `docs/superpowers/reports/2026-07-17-wdc-200k-scale-validation.md`: use the
  same explicit mode in the measurement commands and interpret results as
  systems evidence, not representative data-quality evidence.

---

### Task 1: Bounded Global Candidate Pool

**Files:**

- Modify: `scripts/create_wdc200k_scale_gate_input.py:41-154`
- Modify: `tests/test_wdc200k_scale_gate_input.py:488-659`

**Interfaces:**

- Consumes: `_RankedCandidate.key:
  tuple[int, str, str]`.
- Produces: `_GlobalLowestCandidatePool(table_count: int)`,
  `.retained_count`, `.add(ranked)`, and
  `.selected() -> list[_RankedCandidate]`.

- [ ] **Step 1: Write failing unit tests for exact global order and retention**

```python
def test_global_lowest_pool_returns_exact_global_lowest_order() -> None:
    pool = gate_module._GlobalLowestCandidatePool(table_count=3)
    candidates = [
        _pool_candidate("A", 100),
        _pool_candidate("B", 2),
        _pool_candidate("C", 3),
        _pool_candidate("D", 1),
        _pool_candidate("E", 200),
    ]
    for ranked in candidates:
        pool.add(ranked)

    assert [item.key for item in pool.selected()] == [
        item.key for item in sorted(candidates, key=lambda item: item.key)[:3]
    ]


def test_global_lowest_pool_retains_at_most_table_count() -> None:
    pool = gate_module._GlobalLowestCandidatePool(table_count=7)
    for class_index in range(80):
        for row_count in range(100):
            pool.add(
                _pool_candidate(
                    f"Class{class_index:03d}",
                    row_count,
                )
            )

    assert pool.retained_count <= 7
    assert len(pool.selected()) == 7
```

- [ ] **Step 2: Run the new tests and confirm RED**

Run:

```bash
conda run -n MMDD python -m pytest \
  tests/test_wdc200k_scale_gate_input.py \
  -q -k 'global_lowest_pool'
```

Expected: both tests fail because `_GlobalLowestCandidatePool` is absent.

- [ ] **Step 3: Implement the minimal bounded max-oriented heap**

```python
class _GlobalLowestCandidatePool:
    """Keep the globally lowest ranked candidates in bounded memory."""

    def __init__(self, table_count: int) -> None:
        if table_count <= 0:
            raise ValueError("table_count must be positive")
        self.table_count = table_count
        self._heap: list[_RankedCandidate] = []

    @property
    def retained_count(self) -> int:
        return len(self._heap)

    def add(self, ranked: _RankedCandidate) -> None:
        if len(self._heap) < self.table_count:
            heapq.heappush(self._heap, ranked)
        elif ranked.key < self._heap[0].key:
            heapq.heapreplace(self._heap, ranked)

    def selected(self) -> list[_RankedCandidate]:
        return sorted(self._heap, key=lambda ranked: ranked.key)
```

`_RankedCandidate.__lt__` already reverses key comparison, so heap index zero
is the worst retained candidate and replacement keeps the global lowest `N`.

- [ ] **Step 4: Run focused pool tests and existing round-robin pool tests**

Run:

```bash
conda run -n MMDD python -m pytest \
  tests/test_wdc200k_scale_gate_input.py \
  -q -k 'global_lowest_pool or round_robin_quota or candidate_pool'
```

Expected: all selected tests pass; existing round-robin tests are unchanged.

- [ ] **Step 5: Commit the independently tested pool**

```bash
git add scripts/create_wdc200k_scale_gate_input.py \
  tests/test_wdc200k_scale_gate_input.py
git commit -m "Add bounded global WDC gate pool"
```

---

### Task 2: Mode Dispatch, End-to-End Safety, and CLI

**Files:**

- Modify: `scripts/create_wdc200k_scale_gate_input.py:247-324`
- Modify: `scripts/create_wdc200k_scale_gate_input.py:582-744`
- Modify: `tests/test_wdc200k_scale_gate_input.py:99-336`
- Modify: `tests/test_wdc200k_scale_gate_input.py:339-487`

**Interfaces:**

- Consumes: `_RoundRobinCandidatePool`,
  `_GlobalLowestCandidatePool`, `_validated_source_path`,
  `_validate_staging`, and `_publish_staging`.
- Produces:
  `SELECTION_MODES = ("round_robin", "global_lowest")`,
  `_select_candidates(..., selection_mode: str)`,
  and `create_scale_gate_input(..., selection_mode:
  str = "round_robin")`.

- [ ] **Step 1: Write failing default-regression and global reference tests**

Add a helper that enumerates existing candidates from the synthetic source:

```python
def _expected_global_paths(
    source: Path,
    *,
    table_count: int,
    seed: int,
) -> list[str]:
    candidates = []
    for archive in gate_module._statistics_archives(source):
        for candidate in read_statistics_catalog(archive):
            if (source / candidate.relative_path).is_file():
                candidates.append(candidate)
    return [
        candidate.relative_path
        for candidate in sorted(
            candidates,
            key=lambda candidate: (
                candidate.rows,
                gate_module.stable_hash(seed, candidate.relative_path),
                candidate.relative_path,
            ),
        )[:table_count]
    ]
```

Add the behavioral tests:

```python
def test_default_selection_mode_preserves_round_robin_result(
    tmp_path: Path,
) -> None:
    source = tmp_path / "real-wdc"
    _make_real_input(source)
    implicit = create_scale_gate_input(
        source_dir=source,
        target_dir=tmp_path / "implicit",
        table_count=6,
        seed=13,
    )
    explicit = create_scale_gate_input(
        source_dir=source,
        target_dir=tmp_path / "explicit",
        table_count=6,
        seed=13,
        selection_mode="round_robin",
    )
    assert [
        record["target"] for record in _read_manifest(implicit.manifest_path)
    ] == [
        record["target"] for record in _read_manifest(explicit.manifest_path)
    ]


def test_global_lowest_is_exact_deterministic_and_matches_reference(
    tmp_path: Path,
) -> None:
    source = tmp_path / "real-wdc"
    _make_real_input(source)
    first = create_scale_gate_input(
        source_dir=source,
        target_dir=tmp_path / "first",
        table_count=5,
        seed=23,
        selection_mode="global_lowest",
    )
    second = create_scale_gate_input(
        source_dir=source,
        target_dir=tmp_path / "second",
        table_count=5,
        seed=23,
        selection_mode="global_lowest",
    )
    first_paths = [
        str(record["target"])
        for record in _read_manifest(first.manifest_path)
    ]
    second_paths = [
        str(record["target"])
        for record in _read_manifest(second.manifest_path)
    ]
    assert len(first_paths) == 5
    assert first_paths == second_paths
    assert first_paths == _expected_global_paths(
        source,
        table_count=5,
        seed=23,
    )
```

- [ ] **Step 2: Extend shared safety tests to both modes**

Parameterize the malformed-path, duplicate-path, injected staging failure, and
publish-race tests:

```python
@pytest.mark.parametrize(
    "selection_mode",
    ("round_robin", "global_lowest"),
)
def test_scale_gate_input_rejects_duplicate_candidate_path_atomically(
    tmp_path: Path,
    selection_mode: str,
) -> None:
    source = tmp_path / "real-wdc"
    target = tmp_path / "gate"
    archive = _write_statistics_rows(
        source,
        schema_class="Product",
        subset="top100",
        rows=[("duplicate.test", 1, 3), ("duplicate.test", 1, 3)],
    )
    candidate = next(read_statistics_catalog(archive))
    (source / candidate.relative_path).write_bytes(b"source")
    source_before = _tree_snapshot(source)

    with pytest.raises(ValueError, match="duplicate"):
        create_scale_gate_input(
            source_dir=source,
            target_dir=target,
            table_count=1,
            selection_mode=selection_mode,
        )

    assert not target.exists()
    assert _tree_snapshot(source) == source_before
```

Apply the same parameter and call argument to
`test_scale_gate_input_rejects_traversal_statistics_before_publish`,
`test_scale_gate_input_link_failure_leaves_no_partial_target`, and
`test_publish_noreplace_preserves_empty_directory_created_in_window`. Keep
their current assertions that the source snapshot is unchanged, no temporary
staging directory remains, and no raced target is overwritten.

Add artifact and metadata assertions to the global end-to-end test:

```python
records = _read_manifest(first.manifest_path)
assert all((first.target_dir / str(record["target"])).is_symlink()
           for record in records)
filtered_paths = {
    candidate.relative_path
    for class_dir in first.target_dir.iterdir()
    if class_dir.is_dir()
    for candidate in read_statistics_catalog(
        class_dir / f"{class_dir.name}_statistics.zip"
    )
}
assert filtered_paths == set(first_paths)
checksums = json.loads(first.checksums_path.read_text(encoding="utf-8"))
assert checksums["selection_mode"] == "global_lowest"
assert all(
    _sha256(first.target_dir / relative_path) == digest
    for relative_path, digest in checksums["files"].items()
)
```

Reject unknown Python API modes before creating a target or sibling staging
directory:

```python
def test_scale_gate_input_rejects_unknown_selection_mode_before_writing(
    tmp_path: Path,
) -> None:
    source = tmp_path / "real-wdc"
    target = tmp_path / "gate"
    _make_real_input(source)

    with pytest.raises(ValueError, match="selection_mode"):
        create_scale_gate_input(
            source_dir=source,
            target_dir=target,
            table_count=1,
            selection_mode="unknown",
        )

    assert not target.exists()
    assert not list(tmp_path.glob(".gate.*.tmp"))
```

- [ ] **Step 3: Run the behavioral and safety tests and confirm RED**

Run:

```bash
conda run -n MMDD python -m pytest \
  tests/test_wdc200k_scale_gate_input.py \
  -q -k 'selection_mode or global_lowest or traversal or duplicate or link_failure or publish_noreplace'
```

Expected: global-mode calls fail because the API/selector/CLI do not accept the
new argument.

- [ ] **Step 4: Wire mode validation and pool dispatch without branching publication**

Add:

```python
SELECTION_MODES = ("round_robin", "global_lowest")
```

At the start of `create_scale_gate_input`, before directory creation:

```python
if selection_mode not in SELECTION_MODES:
    raise ValueError(
        "selection_mode must be one of: "
        + ", ".join(SELECTION_MODES)
    )
```

Pass `selection_mode` into `_select_candidates`. Keep the first catalog pass
and SQLite insertion unchanged, then dispatch only candidate-pool creation:

```python
existing_count = sum(capacities.values())
if existing_count < table_count:
    raise ValueError(
        f"requested {table_count} existing tables but found "
        f"only {existing_count}"
    )
if selection_mode == "round_robin":
    quotas = _allocate_round_robin_quotas(
        dict(capacities),
        table_count=table_count,
    )
    pool = _RoundRobinCandidatePool(quotas)
else:
    pool = _GlobalLowestCandidatePool(table_count)
```

Do not change either catalog loop. The second loop continues to call
`_validated_source_path` for every row and offers the same `_RankedCandidate`
key to the chosen pool.

Record `"selection_mode": selection_mode` next to `seed` in
`scale_gate_checksums.json`. Do not create a second materialization or
publication path.

- [ ] **Step 5: Add the CLI choice and prove its default**

Add:

```python
parser.add_argument(
    "--selection_mode",
    choices=SELECTION_MODES,
    default="round_robin",
)
```

Pass `args.selection_mode` to `create_scale_gate_input` and include it in the
printed JSON result. Extend the CLI test:

```python
assert "--selection_mode" in completed.stdout
assert "{round_robin,global_lowest}" in completed.stdout
assert gate_module.parse_args(
    [
        "--source_dir", "source",
        "--target_dir", "target",
        "--table_count", "1",
    ]
).selection_mode == "round_robin"
```

- [ ] **Step 6: Run the complete helper test file**

Run:

```bash
conda run -n MMDD python -m pytest \
  tests/test_wdc200k_scale_gate_input.py -q
```

Expected: all tests pass, including unchanged round-robin, both safety
parameterizations, global exactness, and global retention.

- [ ] **Step 7: Commit the integrated mode**

```bash
git add scripts/create_wdc200k_scale_gate_input.py \
  tests/test_wdc200k_scale_gate_input.py
git commit -m "Add global-lowest WDC gate mode"
```

---

### Task 3: Quick-Gate Documentation and Final Verification

**Files:**

- Modify: `README.md:274-308`
- Modify:
  `docs/superpowers/reports/2026-07-17-wdc-200k-scale-validation.md:57-84`

**Interfaces:**

- Consumes: CLI option
  `--selection_mode {round_robin,global_lowest}`.
- Produces: explicit 100/1,000 quick-gate commands and interpretation limits.

- [ ] **Step 1: Update only the 100/1,000 quick-gate commands**

Add this argument to both helper invocations in each document:

```bash
  --table_count 100 \
  --selection_mode global_lowest \
  --seed 13
```

and equivalently for `--table_count 1000`. Do not change the 10K structural
gate or any formal 200K launch command.

- [ ] **Step 2: State the sampling limitation precisely**

Replace the round-robin description for those commands with:

```markdown
These quick gates globally minimize source row counts by
`(rows, stable_hash(seed, relative_path), relative_path)`. They validate
pipeline plumbing, interruption/resume, request deduplication, checksums,
resource bounds, and canonical output. They are not representative samples of
WDC category balance, table quality, or row-count distribution, and they do
not change the formal stratified 200K selection.
```

Keep the existing filtered-ZIP, absolute-symlink, checksum, staging, and
no-clobber safety explanation.

- [ ] **Step 3: Run focused and repository verification**

Run:

```bash
conda run -n MMDD python -m pytest \
  tests/test_wdc200k_scale_gate_input.py -q
conda run -n MMDD python -m pytest tests -q
conda run -n MMDD python -m py_compile \
  scripts/create_wdc200k_scale_gate_input.py
git diff --check
```

Expected: focused and full test suites pass, compilation exits zero, and
`git diff --check` prints nothing.

- [ ] **Step 4: Verify scope and documentation mechanically**

Run:

```bash
rg -n 'selection_mode|global_lowest|round_robin' \
  scripts/create_wdc200k_scale_gate_input.py \
  tests/test_wdc200k_scale_gate_input.py \
  README.md \
  docs/superpowers/reports/2026-07-17-wdc-200k-scale-validation.md
git diff --stat
```

Expected: `global_lowest` occurs only in the helper, its tests, and the two
quick-gate documentation sections; formal 200K commands remain unchanged.

- [ ] **Step 5: Commit documentation and verified integration**

```bash
git add README.md \
  docs/superpowers/reports/2026-07-17-wdc-200k-scale-validation.md
git commit -m "Document fast WDC scale gates"
```
