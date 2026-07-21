# MM Joinability Progress Display Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restore tqdm progress feedback while randomized tables and initial bridge materials are prepared before vLLM startup.

**Architecture:** Wrap the two existing iterators without changing their work. Reuse `--no_model_progress` as the display opt-out and keep all candidate/material state transitions untouched.

**Tech Stack:** Python 3.10+, tqdm, pytest.

## Global Constraints

- Progress instrumentation must not consume RNG or change iteration/fetch order.
- `--no_model_progress` disables both new bars.
- Model startup remains gated until initial material preparation completes.
- Seeded sampling, replacement policy, cache cleanup, and final artifacts remain unchanged.

---

### Task 1: Restore randomized-read and material-preparation progress

**Files:**
- Modify: `scripts/build_mm_joinability_dataset.py`
- Modify: `tests/test_mm_joinability_sampling.py`

**Interfaces:**
- Consumes: module-level `tqdm`, `iter_random_source_tables(...)`, and `prepare_candidate_batch(...)`.
- Produces: progress bars named `Reading randomized EntiTables JSON` and `Preparing initial candidate materials`.

- [ ] **Step 1: Add failing tqdm-spy tests**

Add tests that exhaust the randomized table iterator and run material preparation while a tqdm spy records `desc`, `total`, `unit`, and `disable`. Assert correct totals and that `--no_model_progress` disables both paths.

- [ ] **Step 2: Verify RED**

Run `conda run -n MMDD python -m pytest tests/test_mm_joinability_sampling.py -k progress -q` and confirm failure because neither new path creates a progress wrapper.

- [ ] **Step 3: Add minimal progress wrappers**

Wrap the shuffled JSON-file iterable and initial source-table iterable with tqdm when available, use the exact labels from the global design, and update the material postfix with cumulative eligible-entity count. Do not add new concurrency or reorder work.

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
git add scripts/build_mm_joinability_dataset.py tests/test_mm_joinability_sampling.py docs/superpowers/specs/2026-07-21-mm-joinability-progress-design.md docs/superpowers/plans/2026-07-21-mm-joinability-progress.md
git commit -m "Restore joinability preparation progress"
```
