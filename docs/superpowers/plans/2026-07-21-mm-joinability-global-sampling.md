# MM Joinability Global Table Sampling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Select initial and replacement candidates uniformly from valid tables across all EntiTables JSON files instead of stopping after a few shuffled files.

**Architecture:** Perform a deterministic priority-sampling pass over every valid table, retain lightweight selected references, then materialize selected full tables in bounded chunks. Existing replacement and evaluation code continues consuming the same iterator interface.

**Tech Stack:** Python 3.10+, pathlib, heapq or equivalent bounded priority selection, pytest, tqdm.

## Global Constraints

- Every discovered JSON file is scanned before the first candidate table is yielded in capped runs.
- Priority is stable from `--seed`, relative POSIX file path, and table ID.
- Candidate capacity equals `max_source_tables * (unrecoverable_replacement_rounds + 1)` for capped runs.
- Full parsed tables are materialized in chunks no larger than `max_source_tables`.
- Replacement RNG draws, parsing filters, cache cleanup, model-start gating, and final artifacts remain unchanged.

---

### Task 1: Implement deterministic global candidate sampling

**Files:**
- Modify: `scripts/build_mm_joinability_dataset.py`
- Modify: `tests/test_mm_joinability_sampling.py`

**Interfaces:**
- Consumes: `iter_random_source_tables(input_dir, args, counters)` and `parse_source_table(...)`.
- Produces: the same source-table iterator, now globally priority sampled.

- [ ] **Step 1: Add failing global-sampling tests**

Create small multi-file fixtures and spy on reads. Assert a capped sample scans all files before yielding, includes tables according to stable global priority rather than the first shuffled file block, supports initial plus configured replacement capacity, is reproducible for the same seed, differs for an alternate seed, and is invariant to reversed filesystem enumeration.

- [ ] **Step 2: Verify RED**

Run `conda run -n MMDD python -m pytest tests/test_mm_joinability_sampling.py -k global -q` and confirm the old lazy shuffled-file sampler fails the full-scan/global-selection assertions.

- [ ] **Step 3: Implement bounded reference selection and chunked materialization**

Add a small immutable selected-reference type, compute stable integer priorities, keep only the configured candidate capacity in capped runs, sort selected references, and materialize each capacity chunk by grouping references per source file. Update progress descriptions without changing table validation or counters.

- [ ] **Step 4: Verify GREEN and regressions**

Run:

```bash
conda run -n MMDD python -m pytest tests/test_mm_joinability_sampling.py tests/test_mm_joinability_extraction.py -q
conda run -n MMDD python -m pytest tests -q
git diff --check
```

Expected: all tests pass and diff check emits no output.

- [ ] **Step 5: Commit**

```bash
git add scripts/build_mm_joinability_dataset.py tests/test_mm_joinability_sampling.py docs/superpowers/specs/2026-07-21-mm-joinability-global-sampling-design.md docs/superpowers/plans/2026-07-21-mm-joinability-global-sampling.md
git commit -m "Sample joinability tables globally"
```
