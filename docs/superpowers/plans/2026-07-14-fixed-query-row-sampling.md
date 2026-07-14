# Fixed Query Row Sampling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate aligned query/target pairs with exactly a configurable number of valid-entity rows while treating empty join attributes as recovery failures.

**Architecture:** Add small pure helpers for capped recovery thresholds, deterministic quota-based row selection, and argument validation. Integrate those helpers into the existing `build_table_join_records` flow so valid entity rows replace non-empty attribute rows as the denominator, while preserving raw data-lake fallback and the user's existing uncommitted model-cache changes.

**Tech Stack:** Python 3.10+, argparse, pytest, JSON/JSONL fixtures.

## Global Constraints

- `--query_rows_per_table` is a positive integer and defaults to `5`.
- Empty candidate join-attribute values count as recovery failures, not invalid entity rows.
- Every emitted query and target contains the same exact `Q` source rows.
- Source tables with fewer than `Q` valid entity rows remain raw data-lake tables.
- `--min_recovery_denominator` and `--min_rows_per_output_table` remain compatible.
- Existing uncommitted changes in `scripts/build_mm_joinability_dataset.py`, `scripts/run_mm_joinability_dynamic_vllm.py`, and `tests/test_mm_joinability_extraction.py` belong to the user and must be preserved.
- Do not commit overlapping implementation files automatically; report the working-tree diff for user review.

---

### Task 1: Recovery Threshold And Deterministic Row Selection

**Files:**
- Modify: `tests/test_mm_joinability_extraction.py:10-60`
- Modify: `scripts/build_mm_joinability_dataset.py:1-30,1123-1245`

**Interfaces:**
- Produces: `required_recovered_row_count(valid_entity_rows: int, query_rows_per_table: int, min_ratio: float) -> int`.
- Produces: `recovery_column_profile(valid_source_rows: set[int], recovered_source_rows: set[int], query_rows_per_table: int, min_recovery_denominator: int, min_ratio: float) -> dict[str, Any] | None`.
- Produces: `select_query_source_rows(source_row_order: list[int], recovered_source_rows: set[int], query_rows_per_table: int, required_recovered_rows: int) -> list[int]`.

- [ ] **Step 1: Write failing helper tests**

Add imports and tests:

```python
from build_mm_joinability_dataset import (
    recovery_column_profile,
    required_recovered_row_count,
    select_query_source_rows,
)


def test_required_recovered_rows_caps_denominator_at_query_size():
    assert required_recovered_row_count(4, 5, 0.6) == 3
    assert required_recovered_row_count(5, 5, 0.6) == 3
    assert required_recovered_row_count(20, 5, 0.6) == 3


def test_recovery_profile_counts_empty_attribute_rows_as_failures():
    profile = recovery_column_profile(
        valid_source_rows={0, 1, 2, 3, 4, 5},
        recovered_source_rows={0, 1, 2},
        query_rows_per_table=5,
        min_recovery_denominator=2,
        min_ratio=0.6,
    )

    assert profile == {
        "eligible_rows": 6,
        "valid_entity_rows": 6,
        "recovered_rows": 3,
        "required_recovered_rows": 3,
        "recovered_value_ratio": 0.5,
    }


def test_select_query_rows_uses_recovery_quota_then_failures():
    assert select_query_source_rows(
        source_row_order=[0, 1, 2, 3, 4, 5],
        recovered_source_rows={0, 2, 4},
        query_rows_per_table=5,
        required_recovered_rows=3,
    ) == [0, 2, 4, 1, 3]


def test_select_query_rows_uses_extra_recoveries_when_failures_are_exhausted():
    assert select_query_source_rows(
        source_row_order=[0, 1, 2, 3, 4, 5],
        recovered_source_rows={0, 1, 2, 3, 4},
        query_rows_per_table=5,
        required_recovered_rows=3,
    ) == [0, 1, 2, 5, 3]
```

- [ ] **Step 2: Run the helper tests and verify RED**

Run:

```bash
conda run -n MMDD python -m pytest \
  tests/test_mm_joinability_extraction.py::test_required_recovered_rows_caps_denominator_at_query_size \
  tests/test_mm_joinability_extraction.py::test_recovery_profile_counts_empty_attribute_rows_as_failures \
  tests/test_mm_joinability_extraction.py::test_select_query_rows_uses_recovery_quota_then_failures \
  tests/test_mm_joinability_extraction.py::test_select_query_rows_uses_extra_recoveries_when_failures_are_exhausted -q
```

Expected: collection fails because the three helpers do not exist.

- [ ] **Step 3: Implement the pure helpers**

Import `ceil` and add:

```python
from math import ceil


def required_recovered_row_count(
    valid_entity_rows: int,
    query_rows_per_table: int,
    min_ratio: float,
) -> int:
    denominator = min(valid_entity_rows, query_rows_per_table)
    return ceil(denominator * min_ratio)


def recovery_column_profile(
    *,
    valid_source_rows: set[int],
    recovered_source_rows: set[int],
    query_rows_per_table: int,
    min_recovery_denominator: int,
    min_ratio: float,
) -> dict[str, Any] | None:
    valid_count = len(valid_source_rows)
    recovered_count = len(recovered_source_rows & valid_source_rows)
    required_count = required_recovered_row_count(valid_count, query_rows_per_table, min_ratio)
    if valid_count < min_recovery_denominator or recovered_count < required_count:
        return None
    return {
        "eligible_rows": valid_count,
        "valid_entity_rows": valid_count,
        "recovered_rows": recovered_count,
        "required_recovered_rows": required_count,
        "recovered_value_ratio": round(recovered_count / max(1, valid_count), 6),
    }


def select_query_source_rows(
    *,
    source_row_order: list[int],
    recovered_source_rows: set[int],
    query_rows_per_table: int,
    required_recovered_rows: int,
) -> list[int]:
    recovered = [row for row in source_row_order if row in recovered_source_rows]
    unrecovered = [row for row in source_row_order if row not in recovered_source_rows]
    if len(source_row_order) < query_rows_per_table or len(recovered) < required_recovered_rows:
        return []
    selected = recovered[:required_recovered_rows]
    selected.extend(unrecovered[: query_rows_per_table - len(selected)])
    if len(selected) < query_rows_per_table:
        selected.extend(
            recovered[
                required_recovered_rows : required_recovered_rows + query_rows_per_table - len(selected)
            ]
        )
    return selected
```

- [ ] **Step 4: Run the helper tests and verify GREEN**

Run the command from Step 2.

Expected: `4 passed`.

---

### Task 2: CLI Contract And Early Validation

**Files:**
- Modify: `tests/test_mm_joinability_extraction.py`
- Modify: `scripts/build_mm_joinability_dataset.py:2130-2145,2401-2498,2501-2590`

**Interfaces:**
- Produces: `configured_query_rows_per_table(args: argparse.Namespace) -> int`.
- Adds CLI: `--query_rows_per_table`, default `5`.

- [ ] **Step 1: Write failing CLI and validation tests**

```python
def test_query_rows_per_table_defaults_to_five(tmp_path):
    args = joinability_dataset.parse_args(
        ["--input_dir", str(tmp_path), "--output_dir", str(tmp_path / "out")]
    )
    assert args.query_rows_per_table == 5


def test_query_rows_per_table_rejects_contradictory_output_minimum(tmp_path):
    args = joinability_dataset.parse_args(
        [
            "--input_dir", str(tmp_path),
            "--output_dir", str(tmp_path / "out"),
            "--query_rows_per_table", "4",
            "--min_rows_per_output_table", "5",
        ]
    )
    with pytest.raises(ValueError, match="min_rows_per_output_table"):
        joinability_dataset.configured_query_rows_per_table(args)
```

Also add these module-level imports:

```python
import pytest
import build_mm_joinability_dataset as joinability_dataset
```

- [ ] **Step 2: Run the CLI tests and verify RED**

Run:

```bash
conda run -n MMDD python -m pytest \
  tests/test_mm_joinability_extraction.py::test_query_rows_per_table_defaults_to_five \
  tests/test_mm_joinability_extraction.py::test_query_rows_per_table_rejects_contradictory_output_minimum -q
```

Expected: failures because the CLI argument and validator do not exist.

- [ ] **Step 3: Implement configuration validation and provenance**

Add:

```python
def configured_query_rows_per_table(args: argparse.Namespace) -> int:
    query_rows = int(getattr(args, "query_rows_per_table", 5))
    min_output_rows = int(getattr(args, "min_rows_per_output_table", 2))
    if query_rows <= 0:
        raise ValueError("query_rows_per_table must be positive")
    if min_output_rows > query_rows:
        raise ValueError(
            "min_rows_per_output_table cannot exceed query_rows_per_table"
        )
    return query_rows
```

Call it at the start of `build_dataset`, store the normalized value back on
`args`, add the parser argument below `--min_rows_per_output_table`, add
`query_rows_per_table` to `stats`, and add this manifest block:

```python
"query_construction": {
    "query_rows_per_table": args.query_rows_per_table,
    "min_rows_per_output_table": args.min_rows_per_output_table,
    "min_recovered_value_ratio": args.min_recovered_value_ratio,
    "min_recovery_denominator": args.min_recovery_denominator,
},
```

- [ ] **Step 4: Run the CLI tests and verify GREEN**

Run the command from Step 2.

Expected: `2 passed`.

---

### Task 3: Integrate Valid-Entity Qualification And Fixed Row Projection

**Files:**
- Modify: `tests/test_stage1_pipeline.py:2270-2425`
- Modify: `scripts/build_mm_joinability_dataset.py:1598-1905`

**Interfaces:**
- Consumes: all helpers from Tasks 1 and 2.
- Produces: exact-size aligned query/target rows and raw fallback for tables with too few valid entities.

- [ ] **Step 1: Convert the pipeline fixture into a failing fixed-row contract**

Extend the queryable fixture to six linked entities. Use three recoverable
`City` values and three empty/unrecoverable values, pass
`--query_rows_per_table 5`, `--min_recovered_value_ratio 0.6`, and assert:

```python
assert len(query["rows"]) == 5
assert len(target["rows"]) == 5
assert query["source_row_indices"] == target["source_row_indices"]
assert query["source_row_indices"] == [0, 1, 2, 3, 4]
assert target["rows"][3]["cells"][0]["text"] == ""
assert query["hidden_attributes"][0]["valid_entity_rows"] == 6
assert query["hidden_attributes"][0]["required_recovered_rows"] == 3
assert query["hidden_attributes"][0]["selected_rows"] == 5
assert stats["query_rows_per_table"] == 5
assert manifest["query_construction"]["query_rows_per_table"] == 5
```

Add a second four-row linked source table with fully recoverable attributes and
assert it is emitted as `raw_data_lake_table` without a query or qrel.

- [ ] **Step 2: Run the pipeline test and verify RED**

Run:

```bash
conda run -n MMDD python -m pytest \
  tests/test_stage1_pipeline.py::test_joinability_dataset_maps_evidence_to_query_entity_attribute -q
```

Expected: failure because empty join values are excluded from the denominator
and target projection, and output rows are not capped at five.

- [ ] **Step 3: Integrate the new qualification and selection rules**

In `build_table_join_records`:

```python
query_rows_per_table = configured_query_rows_per_table(args)
valid_entity_source_rows: set[int] = set()
source_row_order: list[int] = []
```

After validating `wiki_title` and `entity_id`, add every row to both valid-row
collections without inspecting candidate attribute values. Replace the old
ratio qualification with `recovery_column_profile`, merging the returned
profile into each qualified-column record.

Before projection, call:

```python
selected_source_rows = select_query_source_rows(
    source_row_order=source_row_order,
    recovered_source_rows=recovered_rows_by_col.get(join_col, set()),
    query_rows_per_table=query_rows_per_table,
    required_recovered_rows=int(qualified["required_recovered_rows"]),
)
if len(selected_source_rows) != query_rows_per_table:
    continue
selected_source_row_set = set(selected_source_rows)
```

Project the query with `min_required_cols=1`, project the target from the same
set with `min_required_cols=0`, and require both projected source-row lists to
match and contain exactly `query_rows_per_table` rows. Add the new metadata
fields and `selected_rows` to `hidden_attribute`.

- [ ] **Step 4: Run the pipeline test and verify GREEN**

Run the command from Step 2.

Expected: `1 passed`.

---

### Task 4: Regression Verification

**Files:**
- Verify only: `scripts/build_mm_joinability_dataset.py`
- Verify only: `scripts/run_mm_joinability_dynamic_vllm.py`
- Verify only: `tests/test_mm_joinability_extraction.py`
- Verify only: `tests/test_stage1_pipeline.py`

**Interfaces:**
- Consumes: completed Tasks 1-3.
- Produces: verification evidence and a scoped working-tree diff.

- [ ] **Step 1: Run focused joinability tests**

```bash
conda run -n MMDD python -m pytest \
  tests/test_mm_joinability_extraction.py \
  tests/test_mm_joinability_repair.py \
  tests/test_mm_joinability_viewer.py -q
```

Expected: all tests pass.

- [ ] **Step 2: Run the Stage-1 pipeline suite**

```bash
conda run -n MMDD python -m pytest tests/test_stage1_pipeline.py -q
```

Expected: all tests pass.

- [ ] **Step 3: Run syntax and CLI checks**

```bash
conda run -n MMDD python -m py_compile \
  scripts/build_mm_joinability_dataset.py \
  scripts/run_mm_joinability_dynamic_vllm.py
conda run -n MMDD python scripts/build_mm_joinability_dataset.py --help
```

Expected: exit code `0`; help lists `--query_rows_per_table`.

- [ ] **Step 4: Inspect the final diff without committing overlapping files**

```bash
git diff --check
git status --short
git diff -- scripts/build_mm_joinability_dataset.py tests/test_mm_joinability_extraction.py tests/test_stage1_pipeline.py
```

Expected: no whitespace errors; the three pre-existing modified files remain
uncommitted, and only scoped query-row changes appear alongside preserved user
changes.
