# WDC Scale-Gate Telemetry Implementation Plan

## Task 1: Durable Transport Attempts

Use TDD in `tests/test_wdc200k_fetch.py` and
`tests/test_wdc200k_assets.py`.

1. Add failing tests for one attempt per unique URL, cache/resume reuse,
   duplicate aggregation, terminal replay suppression, unfinished crash rows,
   and disk-guard failure.
2. Add guarded append-only attempt storage to the page and image outcome
   stores.
3. Instrument only the final cache-miss boundary around the physical transport
   call.
4. Expose validated summaries on fetch results/manifests.
5. Run focused fetch and asset tests and commit.

## Task 2: URL-Unit Progress and ETA

Use TDD in `tests/test_wdc200k_fetch.py`,
`tests/test_wdc200k_assets.py`, and `tests/test_wdc200k_pipeline.py`.

1. Add failing callback tests for initial durable baselines and live page/image
   completion updates.
2. Add failing fake-clock tests for URL-unit rates, resume monotonicity,
   bounded per-stage sample retention, stage completion, final-half ETA
   factors, and guarded atomic publication.
3. Extend fetchers with optional progress callbacks and bounded update
   intervals.
4. Extend `ProgressReporter` with URL units and bounded completed-stage
   telemetry.
5. Run focused tests and commit.

## Task 3: Pipeline Authority and Counters

Use TDD in `tests/test_wdc200k_pipeline.py`.

1. Add failing end-to-end tests proving attempt counts and ETA summaries reach
   page/image stage registries and remain stable after resume.
2. Add tamper/stale-policy tests for attempt authority.
3. Propagate additive counters and manifest identities through the pipeline.
4. Confirm dry-run and structural stops create no telemetry.
5. Run all WDC pipeline tests and commit.

## Task 4: Verification and Review

1. Update the operational report definitions and commands if field names
   changed.
2. Run WDC-focused tests, the full repository suite, `py_compile`, CLI help
   smoke tests, and `git diff --check`.
3. Request independent specification and quality reviews; fix every Critical
   or Important finding with a failing test first.
4. Run the 100-table interruption/resume gate only after the review passes.
