# MM Joinability Batched Material Preparation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restore the original batched Wikipedia and media-fetch pipeline without changing global random table selection or replacement semantics.

**Architecture:** Separate per-round material preparation from candidate evaluation. Reuse `build_bridge_assets` with an in-memory writer for each round's new entities, then let evaluation consume only already-prepared assets.

**Tech Stack:** Python 3.10+, MediaWiki Action API batching, concurrent media downloader, pytest.

## Global Constraints

- Initial and replacement material preparation use the original `build_bridge_assets` batch path.
- Preparation makes zero calls to `build_bridge_assets_for_entity` for already batch-prepared entities.
- Initial vLLM startup occurs only after initial batch material preparation finishes.
- Each replacement round prepares its whole pending batch before evaluation.
- Global sampling, seed behavior, 50% drop probability, two-round limit, reference-aware cleanup, and final artifacts remain unchanged.

---

### Task 1: Restore batch material preparation for every candidate round

**Files:**
- Modify: `scripts/build_mm_joinability_dataset.py`
- Modify: `tests/test_mm_joinability_sampling.py`

**Interfaces:**
- Consumes: `build_bridge_assets(...)`, `run_replacement_rounds(...)`, `CandidateEvaluationContext`.
- Produces: batch-only `prepare_candidate_batch(...)` and a per-round preparation callback in `run_replacement_rounds(...)`.

- [ ] **Step 1: Add failing batch-path and lifecycle tests**

Spy on `build_bridge_assets` and make `build_bridge_assets_for_entity` fail if called. Assert a multi-table initial batch invokes the batch helper once with all new entities, shared entities are not fetched twice, empty batches make no call, and each replacement batch is prepared before its evaluation. Assert model startup remains between initial preparation and initial evaluation.

- [ ] **Step 2: Verify RED**

Run `conda run -n MMDD python -m pytest tests/test_mm_joinability_sampling.py -k 'batch_material or round_preparation' -q` and confirm failures show the current per-entity path and missing replacement preparation callback.

- [ ] **Step 3: Implement original-pipeline reuse**

Add `flush()` to the in-memory writer, collect ordered new eligible entities for a whole batch, invoke `build_bridge_assets` once, merge returned records and mappings, mark empty-result entities prepared, and populate dependency-key tracking. Extend the replacement state machine with a preparation callback invoked before every batch evaluation; use a separate initial-prepared callback to start models exactly once.

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
git add scripts/build_mm_joinability_dataset.py tests/test_mm_joinability_sampling.py docs/superpowers/specs/2026-07-21-mm-joinability-batched-materials-design.md docs/superpowers/plans/2026-07-21-mm-joinability-batched-materials.md
git commit -m "Restore batched joinability material fetching"
```
