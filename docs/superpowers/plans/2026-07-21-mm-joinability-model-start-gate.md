# MM Joinability Model Start Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Delay dynamic vLLM startup until every table in the initial random candidate batch has completed material preparation.

**Architecture:** Add an initial-batch lifecycle hook to the replacement state machine. The builder uses it to prepare the whole initial batch, emit the model-start marker, wait for service readiness, and install the extractor; later replacement batches continue through the existing evaluator with services kept alive.

**Tech Stack:** Python 3.10+, argparse, pytest, dynamic vLLM marker protocol.

## Global Constraints

- The initial candidate batch must complete Wikipedia/material preparation before `model_start.json` is written.
- vLLM services start at most once and remain available through replacement rounds.
- An empty initial batch releases the runner without starting vLLM services.
- Seeded sampling, 50% drop probability, two replacement rounds, and cache cleanup behavior remain unchanged.

---

### Task 1: Gate model startup on initial material preparation

**Files:**
- Modify: `scripts/build_mm_joinability_dataset.py`
- Modify: `tests/test_mm_joinability_sampling.py`

**Interfaces:**
- Consumes: `run_replacement_rounds(...)`, `_ensure_candidate_assets(...)`, `write_model_start_marker(...)`, and `wait_for_model_ready_marker(...)`.
- Produces: a one-time initial-batch callback on `run_replacement_rounds` and a material-preparation helper used by `build_dataset` before model startup.

- [ ] **Step 1: Write the failing lifecycle test**

Add a test that supplies two initial candidates and one replacement. Record `prepare:<ids>`, `start`, and `evaluate:<ids>` events. Assert initial preparation precedes `start`, `start` precedes initial evaluation, replacement preparation/evaluation follows, and `start` occurs exactly once.

- [ ] **Step 2: Run the focused test and verify RED**

Run: `conda run -n MMDD python -m pytest tests/test_mm_joinability_sampling.py -k model_start -q`

Expected: FAIL because the replacement state machine has no initial-preparation/start lifecycle callback.

- [ ] **Step 3: Implement the minimum lifecycle change**

Add a preparation-only batch helper that updates entity mappings and calls `_ensure_candidate_assets` without model extraction. Add a one-time callback parameter to `run_replacement_rounds`, invoke it after initial slot filling (including the empty case), and have `build_dataset` use the callback to prepare the initial batch, write the marker using `bool(initial_batch)`, wait for readiness, and assign `LocalAttributeExtractor(args)` to the evaluation context.

- [ ] **Step 4: Verify GREEN and regressions**

Run:

```bash
conda run -n MMDD python -m pytest tests/test_mm_joinability_sampling.py tests/test_mm_joinability_extraction.py -q
conda run -n MMDD python -m pytest tests -q
git diff --check
```

Expected: all tests pass and the diff check emits no output.

- [ ] **Step 5: Commit**

```bash
git add scripts/build_mm_joinability_dataset.py tests/test_mm_joinability_sampling.py docs/superpowers/specs/2026-07-21-mm-joinability-model-start-gate-design.md docs/superpowers/plans/2026-07-21-mm-joinability-model-start-gate.md
git commit -m "Delay joinability model startup until materials are ready"
```
