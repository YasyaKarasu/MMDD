# WDC Censored-Tail ETA Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement bounded v2 URL progress telemetry whose deterministic censored-tail ETA passes the recorded page gate without degrading the recorded image gate.

**Architecture:** A new pure `wdc200k_eta` module owns the fixed histogram schema, immutable snapshots, estimator, and bounded tracker. Page and image schedulers feed the same thread-safe tracker and perform one indexed scalar durable refresh before each bounded callback. `ProgressReporter` maps monotonic execution epochs onto its rollback-safe logical clock, persists fixed histograms as base64, and binds the resulting completion authority into network manifests and stage registries.

**Tech Stack:** Python 3.10+, dataclasses, `threading.Lock`, SQLite indexed aggregate queries, `struct`, `base64`, JSON, pytest, existing `GuardedWriteTracker`/atomic JSON helpers.

## Global Constraints

- Keep the formal acceptance rule unchanged: retain every eligible final-half sample and require maximum symmetric ETA factor `<= 2.0`.
- Do not change retries, deadlines, concurrency, durable outcomes, source/query construction, multimodal budgets, or canonical dataset contents.
- Use `wdc200k-url-telemetry-v2`; completed v1 stages remain readable, while incomplete v1 stages start a fresh v2 epoch from exact durable counts.
- Use exactly 64 deadline-capped transport/event bins and 32 commit bins; maturity is exactly `H/3`; overflow correction is exactly `2 / durable_rate`.
- Persist the 160 `uint64` counters as exactly 1,280 little-endian bytes encoded into one base64 `histogram_blob` per sample.
- Keep at most 256 samples per stage and keep worst-width two-stage `progress.json` below 3 MiB and the existing 4 MiB restore limit.
- Fetch callbacks never scan SQLite or retain URLs. Before a callback, the scheduler may execute one indexed read-only grouped scalar refresh outside the callback.
- All new state remains bounded, atomic, guarded by the configured disk reserve, deterministic across resume, and free of wall/monotonic clock mixing.
- `ProgressReporter` atomically persists start, peak, and current bytes for output/work/cache plus the minimum observed free bytes for each root; resume preserves every prior extremum. This is pipeline state, not a log.
- No stage-specific multiplier, threshold relaxation, sample exclusion, future timestamp, final completion timestamp, or gate-specific branch is permitted.
- Use the `MMDD` conda environment for every Python/test command.

## File Map

- Create `scripts/wdc200k_eta.py`: v2 dataclasses, fixed binning, base64 codec, KM/RMST estimator, epoch validation, and bounded thread-safe tracker.
- Create `tests/test_wdc200k_eta.py`: pure estimator, codec, tracker, topology, clock, and size tests.
- Create `tests/fixtures/wdc_gate100_page_eta_events.json`: aggregate page starts/finishes/durable callbacks; no URLs or payloads.
- Create `tests/fixtures/wdc_gate100_image_eta_events.json`: aggregate image starts/finishes/durable callbacks; no URLs or payloads.
- Modify `scripts/wdc200k_fetch.py`: page tracker transitions, indexed durable refresh, and snapshot callback.
- Modify `scripts/wdc200k_assets.py`: image tracker transitions, claim/no-physical transitions, indexed durable refresh, and snapshot callback.
- Modify `scripts/build_wdc200k_mm_joinability_dataset.py`: v2 reporter epochs, persistence/restore, manifest authority, and page/image callback integration.
- Modify `tests/test_wdc200k_fetch.py`: page callback/topology/concurrency/resume tests.
- Modify `tests/test_wdc200k_assets.py`: image callback/claim/no-physical/concurrency/resume tests.
- Modify `tests/test_wdc200k_pipeline.py`: reporter v2 persistence, tamper, clock rollback, v1 compatibility, manifest, and structural/dry-run tests.
- Modify `docs/superpowers/reports/2026-07-17-wdc-200k-scale-validation.md`: fresh gate command, v2 field definitions, interruption evidence, and observed results.

---

### Task 1: Aggregate Gate Replay Fixtures and Pure Estimator

**Files:**
- Create: `scripts/wdc200k_eta.py`
- Create: `tests/test_wdc200k_eta.py`
- Create: `tests/fixtures/wdc_gate100_page_eta_events.json`
- Create: `tests/fixtures/wdc_gate100_image_eta_events.json`

**Interfaces:**
- Produces: `UrlProgressSnapshot`, `UrlEtaEstimate`, `encode_histogram_blob`, `decode_histogram_blob`, `fixed_bin`, and `estimate_url_eta`.
- `UrlProgressSnapshot` is a frozen dataclass with exact fields
  `execution_epoch: str`, `baseline_completed: int`,
  `completed_durable: int`, `total: int`,
  `local_buffered_not_started: int`, `in_flight_jobs: int`,
  `physical_in_flight: int`, `finished_not_durable: int`,
  `unobserved_nonlocal: int`, `deadline_seconds: float`,
  `effective_concurrency: int`, `epoch_elapsed_seconds: float`,
  `transport_event_histogram: tuple[int, ...]`,
  `active_censor_histogram: tuple[int, ...]`,
  `commit_event_histogram: tuple[int, ...]`,
  `transport_overflow_events: int`, `active_overflow_censors: int`, and
  `commit_overflow_events: int`. Histogram lengths are 64, 64, and 32.
- `estimate_url_eta(snapshot: UrlProgressSnapshot) -> UrlEtaEstimate` returns `durable_rate`, `rate_eta`, `queue_eta`, `inflight_eta`, `commit_eta`, `overflow_eta`, `predicted_remaining_seconds`, and `fallback`.
- Each gate fixture is an object with `schema_version`, `stage`, `total`, `deadline_seconds`, `effective_concurrency`, `baseline_completed`, and `events`. Each event has `kind`, `attempt_ordinal`, and `elapsed_seconds`; ordinals are dense integers with no URL-derived value. `start`, `finish`, and `durable` events for an ordinal allow durations/topology to be reconstructed without payload data.

- [ ] **Step 1: Export and verify aggregate fixtures**

Create the fixtures once from read-only gate state, not from test runtime and
not through a committed production exporter. Assign attempt ordinals by
sorting the `(started_at, finished_at)` rows by `started_at`, preserving each
row's paired finish. Pair the `N` sorted finish events with durable callback
samples `completed_units=1..N` in completion order. Subtract the stage's first
callback timestamp from every event and omit policy, URL keys, payloads, and
hosts. The page fixture must contain exactly 100 starts, 100 finishes, and 100
durable completions; the image fixture must contain exactly 135 of each.

```python
def test_gate_event_fixtures_are_aggregate_and_frozen() -> None:
    for path, expected_count in ((PAGE_FIXTURE, 100), (IMAGE_FIXTURE, 135)):
        payload = path.read_bytes()
        assert b"url" not in payload.casefold()
        assert b"payload" not in payload.casefold()
        fixture = json.loads(payload)
        events = fixture["events"]
        assert sorted({item["attempt_ordinal"] for item in events}) == list(
            range(expected_count)
        )
        assert sum(item["kind"] == "start" for item in events) == expected_count
        assert sum(item["kind"] == "finish" for item in events) == expected_count
        assert sum(item["kind"] == "durable" for item in events) == expected_count
```

- [ ] **Step 2: Write estimator replay and invariant tests**

Add tests that replay only each fixture prefix and assert:

```python
assert page.max_symmetric_factor == pytest.approx(1.8978047370910645)
assert page.worst_completed_units == 82
assert image.max_symmetric_factor <= 1.8537476708755196
assert all(sample.used_future_state is False for sample in page.samples + image.samples)
```

Also add exact-edge bin tests (`0`, `H/64`, `H`, overflow), KM risk-set/RMST hand calculations, `H/3` maturity boundaries, `2/rate` overflow, young-censor fallback, commit midpoint mean, topology equality, `uint64` range, and non-finite rejection.

- [ ] **Step 3: Run RED tests**

Run:

```bash
conda run -n MMDD python -m pytest tests/test_wdc200k_eta.py -q
```

Expected: collection fails because `wdc200k_eta` and its public interfaces do
not exist. A test-local legacy `remaining / cumulative_rate` replay over the
same frozen fixtures must separately reproduce page factor
`2.879517510143001` and image factor `1.8537476708755196`, proving the fixture
captures the failure without implementing production behavior first.

- [ ] **Step 4: Implement the pure schema, codec, and formula**

Implement the approved formulas literally. The codec must use:

```python
_HISTOGRAM_STRUCT = struct.Struct("<" + "Q" * 160)

def encode_histogram_blob(snapshot: UrlProgressSnapshot) -> str:
    values = (*snapshot.transport_event_histogram,
              *snapshot.active_censor_histogram,
              *snapshot.commit_event_histogram)
    return base64.b64encode(_HISTOGRAM_STRUCT.pack(*values)).decode("ascii")
```

`decode_histogram_blob` must require canonical base64 and exactly 1,280 decoded bytes. Keep estimation pure: it receives one prefix snapshot and cannot receive `completed_at`.

- [ ] **Step 5: Run GREEN and regression tests**

Run:

```bash
conda run -n MMDD python -m pytest tests/test_wdc200k_eta.py -q
conda run -n MMDD python -m pytest tests/test_wdc200k_pipeline.py -q
```

Expected: all estimator tests pass with page factor exactly `1.8978047370910645`; pipeline v1 tests remain green.

- [ ] **Step 6: Commit Task 1**

```bash
git add scripts/wdc200k_eta.py tests/test_wdc200k_eta.py tests/fixtures/wdc_gate100_page_eta_events.json tests/fixtures/wdc_gate100_image_eta_events.json
git commit -m "Add censored-tail URL ETA estimator"
```

### Task 2: Thread-Safe Tracker and Page/Image Fetcher Integration

**Files:**
- Modify: `scripts/wdc200k_eta.py`
- Modify: `scripts/wdc200k_fetch.py`
- Modify: `scripts/wdc200k_assets.py`
- Modify: `tests/test_wdc200k_eta.py`
- Modify: `tests/test_wdc200k_fetch.py`
- Modify: `tests/test_wdc200k_assets.py`

**Interfaces:**
- Produces `UrlProgressTracker(total: int, deadline_seconds: float, effective_concurrency: int, execution_epoch: str, baseline_completed: int, monotonic: Callable[[], float])`.
- Tracker methods are `buffered(delta: int) -> None`,
  `buffered_durable(delta: int = 1) -> None`,
  `job_submitted(job_id: str) -> None`,
  `physical_started(job_id: str) -> None`,
  `physical_finished(job_id: str) -> None`,
  `future_finished(job_id: str) -> None`,
  `durable_completed(job_id: str) -> None`, and
  `snapshot(refresh: DurableUrlCounts) -> UrlProgressSnapshot`.
- `DurableUrlCounts(completed: int, pending: int, leased: int, total: int)` contains only scalars from the indexed external refresh.
- Page/image `progress_callback` becomes `Callable[[UrlProgressSnapshot], None] | None`.
- Page and image modules each expose a private `_refresh_*_url_counts(...) -> DurableUrlCounts` using indexed `jobs.kind` and policy/outcome joins.
- Page and image schedulers share a production publication bound of 224 completion snapshots. They use `ceil(total / 224)` and stage-global absolute completion multiples across resume; `progress_callback_every` cannot reduce that interval. Up to 32 mandatory execution-epoch baselines reserve the remaining sample capacity.

- [ ] **Step 1: Write failing tracker race/topology tests**

Use a fake monotonic clock and barriers to interleave 128 starts/finishes with snapshot calls. Assert histogram totals, `physical_in_flight <= in_flight_jobs`, exact topology equality, non-decreasing event histograms, mutable active censors, and one immutable capture under the tracker lock. Add overflow and unknown/duplicate transition failures.

- [ ] **Step 2: Write failing fetcher transition tests**

Page tests must cover physical success, exception, synchronous cache reconciliation, final-check suppression, batch `FIRST_COMPLETED`, and another execution's durable completion. Image tests must additionally cover image-claim wait, another claimant's outcome, and claim loss. Assert no-physical paths create no latency event and still move jobs to durable completion.

Add symmetric page/image guard and SQLite finish-fence failures. Assert the
original exception type and message, unfinished attempt ledger, no false
durable completion, physical-active tracker state, and successful retry after
resume. In a completed batch, mark only successful futures finished, release
scheduler resources, and propagate failed `future.result()` unchanged.

- [ ] **Step 3: Run RED tests**

```bash
conda run -n MMDD python -m pytest tests/test_wdc200k_eta.py tests/test_wdc200k_fetch.py tests/test_wdc200k_assets.py -q
```

Expected: failures show the old two-integer callback and missing tracker methods/transition snapshots.

- [ ] **Step 4: Implement the bounded tracker**

Store active start times only for currently local submitted/physical jobs, bounded by the existing claim buffer/concurrency. Store histograms, not completed job identities. Capture `epoch_elapsed_seconds`, topology, and all histograms under one `threading.Lock`; call the user callback after releasing the lock.

- [ ] **Step 5: Implement indexed durable refreshes**

Add/verify indexes supporting `jobs(kind, status)` and outcome `(policy_fingerprint, url_key)` lookups. Each refresh executes grouped `COUNT(*)` queries and returns at most one scalar row per status. Add `EXPLAIN QUERY PLAN` tests that reject `SCAN jobs` without the kind index, and a million-row synthetic test whose Python peak memory remains below 16 MiB.

- [ ] **Step 6: Integrate page and image schedulers**

Place tracker hooks immediately after the existing durable attempt start/finish fences and around future/durable commits. Before each bounded absolute-completion milestone, run one external refresh, pass it into `tracker.snapshot`, then invoke the callback. Preserve one mandatory initial durable-baseline callback per epoch. Coalesce the forced final callback with an already published completion and count it within the 224 completion budget. Cover multiple resumes, external jumps, and completed batches.

- [ ] **Step 7: Run GREEN, focused regressions, and disk checks**

```bash
conda run -n MMDD python -m pytest tests/test_wdc200k_eta.py tests/test_wdc200k_fetch.py tests/test_wdc200k_assets.py -q
conda run -n MMDD python -m pytest tests/test_wdc200k_pipeline.py -q
```

Expected: all tests pass; each stage retains every publication and has at most 224 completion samples plus 32 epoch baselines, physical attempt summaries are unchanged, and disk-guard tests remain green.

- [ ] **Step 8: Commit Task 2**

```bash
git add scripts/wdc200k_eta.py scripts/wdc200k_fetch.py scripts/wdc200k_assets.py tests/test_wdc200k_eta.py tests/test_wdc200k_fetch.py tests/test_wdc200k_assets.py
git commit -m "Track bounded WDC URL scheduler telemetry"
```

### Task 3: ProgressReporter v2 Epochs, Persistence, Restore, and Authority

**Files:**
- Modify: `scripts/build_wdc200k_mm_joinability_dataset.py`
- Modify: `scripts/wdc200k_eta.py`
- Modify: `tests/test_wdc200k_eta.py`
- Modify: `tests/test_wdc200k_pipeline.py`

**Interfaces:**
- `ProgressReporter.update(..., url_snapshot: UrlProgressSnapshot | None = None)` replaces direct URL-unit scalar updates.
- `ProgressReporter` persists `telemetry_schema_version`, `execution_epoch`, `baseline_completed`, `baseline_timestamp`, `epoch_elapsed_seconds`, scalar topology/components, and `histogram_blob` per v2 sample.
- `ProgressReporter.stage_completion_summary(stage)` remains the manifest-facing authority and adds `telemetry_schema_version` plus v2 estimator metadata without changing existing counter names.
- `_validate_network_telemetry` must restore/recompute the same v2 completion summary and compare it exactly with the manifest.
- Disk state contains, for each `output`, `work`, and `cache` root, `start_bytes`, `peak_bytes`, `current_bytes`, `start_free_bytes`, and `min_free_bytes`; all are non-negative integers and extrema survive resume.

- [ ] **Step 1: Write failing v2 reporter tests**

Add fake-clock tests for two epochs, resume from a nonzero durable baseline, wall-clock rollback during the second epoch, exact `baseline_timestamp + epoch_elapsed_seconds` mapping, unique contiguous epoch IDs, and unchanged historical samples.

Add a fake disk-usage sequence for all three roots. Assert start is captured once, peak is the maximum current bytes, minimum free is the minimum observation, current follows the latest sample, and a resumed reporter cannot reset peak or minimum extrema.

- [ ] **Step 2: Write failing codec/size/tamper tests**

Construct two stages with 256 maximum-width samples. Assert decoded blob length 1,280, canonical base64 round-trip, output `< 3 * 1024 * 1024`, and restore rejects truncated/extra/noncanonical blobs, counter overflow, decreasing event histograms, topology mismatch, duplicate/noncontiguous epochs, changed deadline/total, and forged ETA components. The 257th sample and 33rd unique epoch must fail before replacing the previously published progress bytes.

- [ ] **Step 3: Write failing pipeline authority/compatibility tests**

Cover v2 page/image manifest summaries and registry counters, resume stability, tampered progress versus updated manifest checksum, completed v1 read compatibility, incomplete v1-to-v2 epoch upgrade, dry-run/structural no telemetry, and disk-guard failure preserving the previous `progress.json` byte-for-byte.

- [ ] **Step 4: Run RED tests**

```bash
conda run -n MMDD python -m pytest tests/test_wdc200k_eta.py tests/test_wdc200k_pipeline.py -q
```

Expected: v2 persistence/restore tests fail because the reporter still accepts scalar URL updates and writes v1 samples.

- [ ] **Step 5: Implement v2 reporter epochs**

On the first snapshot of an epoch, allocate a rollback-safe logical baseline and require elapsed zero. For later snapshots, derive timestamp only from the persisted baseline plus monotonic elapsed. Call `estimate_url_eta`, serialize the exact histograms, append at most 256 samples, and recompute final-half summaries from restored sample predictions. Never trim an accepted sample or epoch to recover capacity; fail closed before append instead.

During the same atomic snapshot, update output/work/cache current bytes, peak bytes, and per-root minimum free bytes. Initialize start bytes/free bytes only when absent. Restore and merge extrema before the first resume publication so a restarted process cannot erase a prior peak or raise a prior minimum-free value.

- [ ] **Step 6: Implement strict restore and v1 compatibility**

Validate every epoch and estimator component by decoding and recomputing, never trusting persisted ETA scalars. Preserve completed v1 summaries. For incomplete v1, retain historical v1 samples and append a new v2 epoch whose baseline is the exact durable refresh; never reinterpret v1 histograms.

- [ ] **Step 7: Bind v2 completion authority into manifests**

Publish `progress.json` before page/image network manifests as today, add the v2 schema metadata to `url_completion`, and require exact equality with a separately restored reporter during registry validation. Do not alter transport-attempt authority, artifact records, or existing integer counter names.

- [ ] **Step 8: Run GREEN and full WDC regressions**

```bash
conda run -n MMDD python -m pytest tests/test_wdc200k_eta.py tests/test_wdc200k_pipeline.py -q
conda run -n MMDD python -m pytest tests/test_wdc200k_fetch.py tests/test_wdc200k_assets.py tests/test_wdc200k_pipeline.py -q
```

Expected: all tests pass; page fixture factor remains exactly `1.8978047370910645`, image does not exceed `1.8537476708755196`, and v1 tests remain unchanged.

- [ ] **Step 9: Commit Task 3**

```bash
git add scripts/wdc200k_eta.py scripts/build_wdc200k_mm_joinability_dataset.py tests/test_wdc200k_eta.py tests/test_wdc200k_pipeline.py
git commit -m "Persist authoritative WDC ETA epochs"
```

### Task 4: Verification, Independent Review, and Fresh Gate Rerun

**Files:**
- Modify: `docs/superpowers/reports/2026-07-17-wdc-200k-scale-validation.md`
- Modify only if review finds defects: files from Tasks 1-3, with a failing test first.

**Interfaces:**
- Consumes all Task 1-3 public interfaces and the existing dynamic vLLM runner.
- Produces an independently reviewed commit and a fresh, non-reused 100-table interruption/resume acceptance record.

- [ ] **Step 1: Run focused and full verification**

```bash
conda run -n MMDD python -m pytest tests/test_wdc200k_eta.py tests/test_wdc200k_fetch.py tests/test_wdc200k_assets.py tests/test_wdc200k_pipeline.py -q
conda run -n MMDD python -m pytest -q
conda run -n MMDD python -m py_compile scripts/wdc200k_eta.py scripts/wdc200k_fetch.py scripts/wdc200k_assets.py scripts/build_wdc200k_mm_joinability_dataset.py scripts/run_mm_joinability_dynamic_vllm.py
conda run -n MMDD python scripts/build_wdc200k_mm_joinability_dataset.py --help
conda run -n MMDD python scripts/run_mm_joinability_dynamic_vllm.py --help
git diff --check
```

Expected: every command exits 0; no test is skipped because of the ETA change; both help commands describe the staged WDC builder/runner.

- [ ] **Step 2: Request independent specification and quality reviews**

Provide reviewers the approved spec, this plan, Task 1 base SHA, and current HEAD. Require explicit review of fixture prefix purity, formula constants, topology transitions, indexed refresh, thread safety, epoch/clock authority, tamper rejection, bounded persistence, and v1 compatibility. Fix every Critical/Important finding with a RED test before proceeding.

- [ ] **Step 3: Prepare fresh gate roots and dry-run**

Use new absent roots; do not delete or reuse prior gate state:

```text
/home/oycy/MMDD/output_wdc_gate100_eta_v2
/home/oycy/MMDD/work_wdc_gate100_eta_v2
/home/oycy/MMDD/cache/wdc_gate100_eta_v2
```

Run the builder `--dry_run` with the exact global-lowest 100-table input and 100 GiB reserve. Assert all three roots remain absent after dry-run.

```bash
conda run -n MMDD python scripts/build_wdc200k_mm_joinability_dataset.py --input_dir /home/oycy/MMDD/gate_inputs/wdc_100_lowest --output_dir /home/oycy/MMDD/output_wdc_gate100_eta_v2 --work_dir /home/oycy/MMDD/work_wdc_gate100_eta_v2 --cache_dir /home/oycy/MMDD/cache/wdc_gate100_eta_v2 --runtime_dir /home/oycy/MMDD/work_wdc_gate100_eta_v2/runtime --max_source_tables 100 --selection_seed 13 --web_max_retries 0 --web_max_response_seconds 8 --web_global_concurrency 128 --web_per_host_concurrency 2 --max_image_attempts_per_entity 3 --max_images_per_entity 3 --min_free_disk_bytes 107374182400 --progress_interval_seconds 5 --dry_run
```

- [ ] **Step 4: Launch fresh initial run in tmux with direct stdout**

Create `wdc_gate100_eta_v2:run`, set `remain-on-exit on`, and run `/usr/bin/time -v conda run -n MMDD --no-capture-output python scripts/run_mm_joinability_dynamic_vllm.py` with the same model paths/policy as the prior gate, the fresh roots above, `--run_fingerprint gate100-eta-v2-20260718`, and no shell redirection. Add a separate `progress` window that only watches `work_wdc_gate100_eta_v2/progress.json`.

```bash
tmux new-session -d -s wdc_gate100_eta_v2 -n run -c /home/oycy/MMDD/.worktrees/wdc-200k "sleep 3600"
tmux set-option -t wdc_gate100_eta_v2 remain-on-exit on
tmux respawn-pane -k -t wdc_gate100_eta_v2:run "/usr/bin/time -v conda run -n MMDD --no-capture-output python scripts/run_mm_joinability_dynamic_vllm.py --input_dir /home/oycy/MMDD/gate_inputs/wdc_100_lowest --output_dir /home/oycy/MMDD/output_wdc_gate100_eta_v2 --work_dir /home/oycy/MMDD/work_wdc_gate100_eta_v2 --cache_dir /home/oycy/MMDD/cache/wdc_gate100_eta_v2 --text_model_path /home/oycy/MMDD/hf_models/Qwen3.5-9B --image_model_path /home/oycy/MMDD/hf_models/Qwen3-VL-8B-Thinking --run_fingerprint gate100-eta-v2-20260718 --runtime_dir /home/oycy/MMDD/work_wdc_gate100_eta_v2/runtime --max_source_tables 100 --selection_seed 13 --web_max_retries 0 --web_max_response_seconds 8 --web_global_concurrency 128 --web_per_host_concurrency 2 --max_image_attempts_per_entity 3 --max_images_per_entity 3 --min_free_disk_bytes 107374182400 --progress_interval_seconds 5 --no-resume"
tmux new-window -d -t wdc_gate100_eta_v2 -n progress -c /home/oycy/MMDD/.worktrees/wdc-200k "watch -n 5 cat /home/oycy/MMDD/work_wdc_gate100_eta_v2/progress.json"
```

- [ ] **Step 5: Interrupt once and record the durable cutoff**

After both page and image network manifests are complete and the model stage
has begun, send one normal `Ctrl-C` to the run pane. Record both cutoff
timestamps, outcome counts, full-row SHA-256 values, attempt summaries,
progress epoch/sample counts, pane status, elapsed time, and peak RSS. Do not
remove or rewrite state. Automated tests separately cover interruption inside
an incomplete URL epoch.

- [ ] **Step 6: Resume identically and monitor to completion**

Respawn the same pane with the identical command plus `--resume`, still without
redirection. Verify both cutoff row digests are unchanged, every prior URL
sample object is unchanged, completed v2 summaries are reused without
appending URL samples, and no successful/terminal URL is physically replayed.

Use the Step 4 `tmux respawn-pane` command verbatim except replace the final
`--no-resume` with `--resume`; do not alter `run_fingerprint`, paths, policy,
or concurrency.

- [ ] **Step 7: Validate final acceptance evidence**

Require pane status 0; selected/validated/source tables `100`; source
rows/entities `100`; no source `image` column; page/image duplicate, terminal
replay, unfinished, and blocked counters all zero; every
manifest/registry/artifact checksum and byte count valid;
`progress.json < 3 MiB`; page and image maximum factors both `<= 2.0`; and
free disk never below `107374182400` bytes. The frozen image replay test, not
the stochastic fresh run, enforces no degradation from the historical
`1.8537476708755196` factor.

- [ ] **Step 8: Update the scale report**

Record exact initial/resume commands, SHAs, gate-input hashes, start/interruption/resume/end times, cutoff digest, epoch baselines, transport counts, ETA factors/worst samples, RSS, output/work/cache start/peak/current bytes, each root's start/minimum free bytes, manifest validation, and decision. Mark PASS only if every Step 7 condition holds.

- [ ] **Step 9: Commit verification documentation**

```bash
git add docs/superpowers/reports/2026-07-17-wdc-200k-scale-validation.md
git commit -m "Validate censored-tail WDC ETA gate"
```
