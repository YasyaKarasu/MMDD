# Image Attribute Dataset Table-Grouped Resplit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reassign the existing 2,000 samples to exact label-balanced train, validation, and test splits without splitting any source table.

**Architecture:** Add a standalone resplit utility that loads the existing sample/table JSONL files, groups records by `source_table_id`, and uses deterministic two-dimensional subset selection to allocate complete table groups. It atomically rewrites only split JSONL files and statistics; image files and record contents remain unchanged.

**Tech Stack:** Python 3.10+, standard library, pytest.

---

### Task 1: Exact Complete-Table Subset Selection

**Files:**
- Create: `scripts/resplit_mm_image_attribute_dataset.py`
- Create: `tests/test_resplit_mm_image_attribute_dataset.py`

- [ ] **Step 1: Write the failing selector tests**

```python
def test_select_table_groups_hits_exact_label_target_without_splitting():
    groups = {
        "a": [{"extractable": True}, {"extractable": True}],
        "b": [{"extractable": True}],
        "c": [{"extractable": False}],
        "d": [{"extractable": True}, {"extractable": False}],
    }
    selected = select_table_groups(groups, positive_target=2, negative_target=1)
    assert sum(row["extractable"] for key in selected for row in groups[key]) == 2
    assert sum(not row["extractable"] for key in selected for row in groups[key]) == 1


def test_select_table_groups_rejects_impossible_target():
    with pytest.raises(ValueError, match="cannot satisfy"):
        select_table_groups(
            {"a": [{"extractable": True}, {"extractable": True}]},
            positive_target=1,
            negative_target=0,
        )
```

- [ ] **Step 2: Run tests and confirm RED**

Run: `conda run -n MMDD python -m pytest tests/test_resplit_mm_image_attribute_dataset.py -q`

Expected: FAIL because `resplit_mm_image_attribute_dataset` does not exist.

- [ ] **Step 3: Implement deterministic two-dimensional dynamic programming**

Implement:

```python
def select_table_groups(
    groups: dict[str, list[dict[str, Any]]],
    positive_target: int,
    negative_target: int,
) -> set[str]:
```

Sort table IDs, update reachable `(positive, negative)` states without exceeding
the target, retain predecessor state and table ID, and backtrack from the exact
target. Raise `ValueError` if the target is unreachable.

- [ ] **Step 4: Run tests and confirm GREEN**

Run: `conda run -n MMDD python -m pytest tests/test_resplit_mm_image_attribute_dataset.py -q`

Expected: PASS.

### Task 2: Atomic Existing-Dataset Resplit

**Files:**
- Modify: `scripts/resplit_mm_image_attribute_dataset.py`
- Modify: `tests/test_resplit_mm_image_attribute_dataset.py`

- [ ] **Step 1: Write the failing integration test**

Create a synthetic dataset with complete-table sample groups, table JSONL
records, and image paths. Run:

```python
resplit_dataset(
    dataset_dir,
    targets={
        "train": (3, 2),
        "val": (1, 1),
        "test": (1, 1),
    },
)
```

Assert exact per-split label counts, no table overlap, identical pre/post
sample IDs and table IDs, and unchanged image files.

- [ ] **Step 2: Run integration test and confirm RED**

Run: `conda run -n MMDD python -m pytest tests/test_resplit_mm_image_attribute_dataset.py -q`

Expected: FAIL because `resplit_dataset` is missing.

- [ ] **Step 3: Implement the resplit command**

Load all current sample and table split files. Validate that each table occurs
in only one input split and every sample has a matching table. Select val
groups for `(137, 63)`, select test groups from the remaining groups for
`(137, 63)`, and assign all remaining groups to train. Verify train is
`(1095, 505)`.

Write new sample/table files under a temporary sibling directory, validate
exact counts and invariant ID sets, then replace the six JSONL files. Update
`stats.json`. Expose CLI arguments for the dataset directory and the six label
targets, with these values as defaults.

- [ ] **Step 4: Run integration and regression tests**

Run: `conda run -n MMDD python -m pytest tests/test_resplit_mm_image_attribute_dataset.py tests/test_mm_image_attribute_dataset.py -q`

Expected: PASS.

### Task 3: Resplit Production Data and Update Documentation

**Files:**
- Modify: `output_mm_image_attribute_2k/samples/*.jsonl`
- Modify: `output_mm_image_attribute_2k/tables/*.jsonl`
- Modify: `output_mm_image_attribute_2k/stats.json`
- Modify: `output_mm_image_attribute_2k/README.md`

- [ ] **Step 1: Run the production resplit**

Run: `conda run -n MMDD python scripts/resplit_mm_image_attribute_dataset.py --dataset_dir output_mm_image_attribute_2k`

Expected: train `1095/505`, val `137/63`, test `137/63`.

- [ ] **Step 2: Update README statistics**

Replace the split label counts and table/column-bucket counts with values
computed from the rewritten files. State that all splits preserve the global
label ratio and remain table-isolated.

- [ ] **Step 3: Run full verification**

Run:

```bash
conda run -n MMDD python -m pytest \
  tests/test_resplit_mm_image_attribute_dataset.py \
  tests/test_mm_image_attribute_dataset.py \
  tests/test_mm_joinability_extraction.py -q
```

Expected: all tests PASS.

Verify exact counts, 2,000 unique entities/images, 1,366 tables, zero table
overlap, existing image paths, and unchanged ID sets using the resplit
utility's validation report.

- [ ] **Step 4: Commit implementation**

Run:

```bash
git add scripts/resplit_mm_image_attribute_dataset.py tests/test_resplit_mm_image_attribute_dataset.py
git commit -m "Add table-grouped image dataset resplit"
```

Generated dataset artifacts remain untracked.
