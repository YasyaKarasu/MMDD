# Image Attribute 5K Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a standalone 5,000-sample entity-image-attribute dataset from an existing multimodal joinability dataset, with balanced labels and table-isolated splits.

**Architecture:** A new CLI joins the source-table, image-asset, asset-link, and VLM-extraction shards; it selects source-grounded positive and omitted-attribute negative samples under exact quotas; it writes complete original tables and copies all selected images into a self-contained output directory.

**Tech Stack:** Python 3.10+, `argparse`, `json`, `hashlib`, `shutil`, `pathlib`, pytest, and Pillow when available for raster validation.

---

### Task 1: Define the command contract

**Files:**
- Create: `scripts/build_mm_image_attribute_dataset.py`
- Create: `tests/test_mm_image_attribute_dataset.py`

- [ ] **Step 1: Write the failing defaults test**

```python
def test_parse_args_uses_5k_defaults(tmp_path):
    from build_mm_image_attribute_dataset import parse_args
    args = parse_args(["--input_dir", str(tmp_path / "input")])
    assert args.output_dir == "output_mm_image_attribute_5k"
    assert (args.total_samples, args.positive_samples, args.negative_samples) == (5000, 3500, 1500)
    assert (args.train_samples, args.val_samples, args.test_samples) == (4000, 500, 500)
```

- [ ] **Step 2: Confirm the test fails**

Run: `conda run -n MMDD python -m pytest tests/test_mm_image_attribute_dataset.py::test_parse_args_uses_5k_defaults -q`

Expected: FAIL because the module does not exist.

- [ ] **Step 3: Implement the minimal CLI**

Add `parse_args`, count validation, `clean_text`, `normalize`, `values_match`, `stable_hash`, and `column_bucket`. Expose input/output, sample and split quotas, seed, entity/table caps, and `--overwrite`; reject inconsistent totals and a nonempty output directory without overwrite.

- [ ] **Step 4: Confirm the test passes**

Run: `conda run -n MMDD python -m pytest tests/test_mm_image_attribute_dataset.py::test_parse_args_uses_5k_defaults -q`

Expected: PASS.

- [ ] **Step 5: Commit**

Run: `git add scripts/build_mm_image_attribute_dataset.py tests/test_mm_image_attribute_dataset.py && git commit -m "Add image attribute dataset CLI"`

### Task 2: Derive validated positives and negatives

**Files:**
- Modify: `scripts/build_mm_image_attribute_dataset.py`
- Modify: `tests/test_mm_image_attribute_dataset.py`

- [ ] **Step 1: Write the failing candidate test**

```python
def test_build_candidates_matches_positive_values_and_uses_omissions_as_negatives(tmp_path):
    from build_mm_image_attribute_dataset import build_candidates
    input_dir = write_synthetic_input(tmp_path, attributes=[{"name": "State", "value": "Nevada", "evidence": "NEVADA"}])
    positives, negatives, exclusions = build_candidates(input_dir)
    assert [(x["attribute_name"], x["extractable"], x["ground_truth_value"]) for x in positives] == [("State", True, "Nevada")]
    assert [(x["attribute_name"], x["extractable"], x["ground_truth_value"]) for x in negatives] == [("Party", False, "Independent")]
    assert exclusions["positive_value_mismatch"] == 0
```

- [ ] **Step 2: Confirm the test fails**

Run: `conda run -n MMDD python -m pytest tests/test_mm_image_attribute_dataset.py::test_build_candidates_matches_positive_values_and_uses_omissions_as_negatives -q`

Expected: FAIL because `build_candidates` is missing.

- [ ] **Step 3: Implement manifest-backed joins**

Read `source_tables`, `bridge_assets`, `attribute_extractions`, and `table_asset_links` through `stage1_io.iter_manifest_records`. Index full tables, image assets, and valid entity/table/row/column/image links. For every image extraction, map candidate names to nonempty referenced-row cells. A positive must have a nonempty VLM value matching the source value. A negative must be a nonempty candidate column omitted from VLM output for the same entity-image-row. Put stable IDs, source references, source value, bucket, and VLM evidence (positive only) in every candidate. Count missing and mismatched inputs.

- [ ] **Step 4: Confirm all candidate tests pass**

Run: `conda run -n MMDD python -m pytest tests/test_mm_image_attribute_dataset.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

Run: `git add scripts/build_mm_image_attribute_dataset.py tests/test_mm_image_attribute_dataset.py && git commit -m "Build grounded image attribute candidates"`

### Task 3: Select exact balanced split groups

**Files:**
- Modify: `scripts/build_mm_image_attribute_dataset.py`
- Modify: `tests/test_mm_image_attribute_dataset.py`

- [ ] **Step 1: Write the failing selection test**

```python
def test_select_samples_balances_labels_and_buckets_without_table_leakage():
    from build_mm_image_attribute_dataset import select_samples
    positives, negatives = make_balanced_candidates(tables_per_bucket=8)
    selected = select_samples(positives, negatives, seed=7, split_sizes={"train": 8, "val": 2, "test": 2}, split_positive_sizes={"train": 6, "val": 1, "test": 1}, max_samples_per_table=2, max_samples_per_entity=1)
    assert {key: len(value) for key, value in selected.items()} == {"train": 8, "val": 2, "test": 2}
    assert {key: sum(row["extractable"] for row in value) for key, value in selected.items()} == {"train": 6, "val": 1, "test": 1}
    assert not ({x["source_table_id"] for x in selected["train"]} & {x["source_table_id"] for x in selected["val"]})
```

- [ ] **Step 2: Confirm the test fails**

Run: `conda run -n MMDD python -m pytest tests/test_mm_image_attribute_dataset.py::test_select_samples_balances_labels_and_buckets_without_table_leakage -q`

Expected: FAIL because `select_samples` is missing.

- [ ] **Step 3: Implement deterministic quota selection**

Allocate each label quota as evenly as possible over `2-4`, `5-7`, `8-12`, and `13+`. Assign each source table to exactly one split using seed-stable ordering and remaining label/bucket feasibility. Select candidates with per-table and per-entity caps; do not select duplicate sample IDs. Raise `ValueError` showing unfilled split, label, and bucket quotas if exact allocation cannot be met.

- [ ] **Step 4: Confirm selection behavior**

Run: `conda run -n MMDD python -m pytest tests/test_mm_image_attribute_dataset.py -q`

Expected: PASS, including deterministic repeated selection and insufficient-candidate failure tests.

- [ ] **Step 5: Commit**

Run: `git add scripts/build_mm_image_attribute_dataset.py tests/test_mm_image_attribute_dataset.py && git commit -m "Balance image attribute dataset splits"`

### Task 4: Write self-contained output and verify it

**Files:**
- Modify: `scripts/build_mm_image_attribute_dataset.py`
- Modify: `tests/test_mm_image_attribute_dataset.py`

- [ ] **Step 1: Write the failing materialization test**

```python
def test_run_copies_images_and_preserves_complete_original_tables(tmp_path):
    from build_mm_image_attribute_dataset import parse_args, run
    input_dir, original_rows = write_selection_ready_input(tmp_path)
    output_dir = tmp_path / "standalone"
    run(parse_args(["--input_dir", str(input_dir), "--output_dir", str(output_dir), "--total_samples", "4", "--train_samples", "2", "--val_samples", "1", "--test_samples", "1", "--positive_samples", "3", "--negative_samples", "1"]))
    sample = read_jsonl(output_dir / "samples" / "train.jsonl")[0]
    assert (output_dir / sample["image_path"]).is_file()
    assert not Path(sample["image_path"]).is_absolute()
    assert read_jsonl(output_dir / "tables" / "train.jsonl")[0]["rows"] == original_rows
```

- [ ] **Step 2: Confirm the test fails**

Run: `conda run -n MMDD python -m pytest tests/test_mm_image_attribute_dataset.py::test_run_copies_images_and_preserves_complete_original_tables -q`

Expected: FAIL because `run` does not materialize artifacts.

- [ ] **Step 3: Implement atomic materialization**

Write in a temporary sibling directory: `samples`, deduplicated complete original `tables`, and deduplicated `assets` JSONL for each split. Copy each selected image once to `images/<asset_id><suffix>`, validate it as a raster with Pillow when available, calculate SHA-256 and byte count, then replace source paths with POSIX relative paths. Write a manifest with source-manifest digest and arguments, and stats with exclusions, split/label/bucket counts, unique tables/entities/assets, copy results, and integrity checks. Validate all sample references before atomically publishing the output directory.

- [ ] **Step 4: Run full verification**

Run: `conda run -n MMDD python -m pytest tests/test_mm_image_attribute_dataset.py -q && conda run -n MMDD python -m py_compile scripts/build_mm_image_attribute_dataset.py && conda run -n MMDD python scripts/build_mm_image_attribute_dataset.py --help`

Expected: all tests PASS, compilation exits 0, and help lists input/output and quotas.

- [ ] **Step 5: Commit**

Run: `git add scripts/build_mm_image_attribute_dataset.py tests/test_mm_image_attribute_dataset.py && git commit -m "Write standalone image attribute dataset"`

### Task 5: Build and audit the requested dataset

**Files:**
- Verify only: `output_mm_image_attribute_5k/`

- [ ] **Step 1: Build from the requested input**

Run: `conda run -n MMDD python scripts/build_mm_image_attribute_dataset.py --input_dir output_mm_joinability_v5 --output_dir output_mm_image_attribute_5k`

Expected: exit 0 and materialize 5,000 self-contained samples.

- [ ] **Step 2: Audit counts and local image references**

Run: `conda run -n MMDD python -c "import json; from pathlib import Path; r=Path('output_mm_image_attribute_5k'); d={s:[json.loads(x) for x in (r/'samples'/f'{s}.jsonl').read_text().splitlines()] for s in ('train','val','test')}; assert {s:len(v) for s,v in d.items()}=={'train':4000,'val':500,'test':500}; assert sum(x['extractable'] for v in d.values() for x in v)==3500; assert all((r/x['image_path']).is_file() for v in d.values() for x in v); print('verified')"`

Expected: prints `verified`.

- [ ] **Step 3: Audit table isolation**

Run: `conda run -n MMDD python -c "import json; from pathlib import Path; r=Path('output_mm_image_attribute_5k'); d={s:{json.loads(x)['source_table_id'] for x in (r/'samples'/f'{s}.jsonl').read_text().splitlines()} for s in ('train','val','test')}; assert not d['train']&d['val'] and not d['train']&d['test'] and not d['val']&d['test']; print('table-isolated')"`

Expected: prints `table-isolated`.

- [ ] **Step 4: Commit implementation only**

Run: `git add scripts/build_mm_image_attribute_dataset.py tests/test_mm_image_attribute_dataset.py && git commit -m "Build standalone image attribute subset"`

Generated output remains untracked.
