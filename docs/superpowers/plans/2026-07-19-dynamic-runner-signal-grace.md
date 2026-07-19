# Dynamic Runner Signal Grace Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let a signalled WDC builder complete bounded scheduler cleanup before the dynamic runner applies its existing TERM/KILL fallback.

**Architecture:** Add one validated runner option and one small helper that waits only for the already-signalled builder. Call it in the `ForwardedSignal` handler before entering unchanged best-effort cleanup; cover the process boundary with both deterministic fakes and a real child process group.

**Tech Stack:** Python 3.10, `signal`, `subprocess`, pytest, conda environment `MMDD`.

---

### Task 1: Graceful Forwarded-Signal Builder Exit

**Files:**
- Modify: `scripts/run_mm_joinability_dynamic_vllm.py`
- Modify: `tests/test_mm_joinability_extraction.py`

- [ ] **Step 1: Write RED real-process and main-ordering tests**

Add a real child process started with `start_new_session=True`. Its `SIGINT`
handler records signal receipt, waits on a short deterministic synchronization
point, writes a cleanup-complete marker, and exits. Forward `SIGINT`, invoke the
new wait helper, and assert the marker exists and the child exited without a
fallback `SIGTERM`. Add a main-level fake process test whose event sequence must
be `forward -> builder_wait -> best_effort_stop`, plus timeout and already-exited
cases. Add parser validation for a negative grace value.

- [ ] **Step 2: Run RED tests and capture the intended failures**

Run:

```bash
conda run -n MMDD python -m pytest tests/test_mm_joinability_extraction.py -k 'forwarded_signal_grace or waits_for_signalled_builder' -q
```

Expected: failures because the option/helper and pre-cleanup wait do not exist.

- [ ] **Step 3: Implement the minimal bounded wait**

Add `--forwarded_signal_grace_seconds` with default `30.0`; reject negative
values before any write or process start. Add a typed helper equivalent to:

```python
def wait_for_forwarded_process_exit(
    process: subprocess.Popen[str] | None,
    *,
    timeout_seconds: float,
) -> bool:
    if process is None or process.poll() is not None:
        return True
    try:
        process.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        return False
    return True
```

In `except ForwardedSignal`, call the helper for `builder_proc` before returning
`128 + signum`. Do not signal any process from the helper and do not change
`stop_process()` or final cleanup ordering.

- [ ] **Step 4: Run GREEN focused and regression tests**

Run:

```bash
conda run -n MMDD python -m pytest tests/test_mm_joinability_extraction.py -q
conda run -n MMDD python -m pytest tests/test_wdc200k_io.py tests/test_wdc200k_assets.py tests/test_wdc200k_pipeline.py -q
conda run -n MMDD python -m py_compile scripts/run_mm_joinability_dynamic_vllm.py
git diff --check
```

Expected: every command exits zero.

- [ ] **Step 5: Commit and obtain two-stage review**

```bash
git add scripts/run_mm_joinability_dynamic_vllm.py tests/test_mm_joinability_extraction.py
git commit -m "Gracefully stop forwarded WDC builders"
```

Require independent spec-compliance and code-quality approval. Fix every
Critical or Important finding with a new RED test and rerun Step 4.

- [ ] **Step 6: Rerun the fresh 100-table interruption gate**

Use entirely new absent output/work/cache/evidence roots. Interrupt during an
incomplete native-v2 image stage, preserve a mode-0444 progress copy, and prove
the builder exits with current-owner job leases released to `retryable`, current
URL claims removed, and foreign fencing unchanged. Immediate `--resume` must
complete with zero duplicate, replay, unfinished, and blocked transport anomaly
counters.
