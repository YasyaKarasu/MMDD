# Task 9A: 100-table page ETA investigation

## Status

`BLOCKED`: the observed factor is reproducible and its cause is established,
but no honest, reliable correction can be implemented from the current
progress callback's information without expanding the telemetry contract.
No production code or gate state was changed.

## Evidence inspected

- `/home/oycy/MMDD/work_wdc_gate100_lowest/progress.json`
- `/home/oycy/MMDD/work_wdc_gate100_lowest/page_jobs/network/network-manifest.json`
- `/home/oycy/MMDD/cache/wdc_gate100_lowest/page_cache/outcomes.sqlite3`
- `.superpowers/sdd/gate100-validation-report.md`
- `.superpowers/sdd/gate100-review.md`
- `scripts/build_wdc200k_mm_joinability_dataset.py`
- `scripts/wdc200k_fetch.py`

All inspection of gate work/cache/output was read-only.

## Reproduction and trace

The durable page summary contains 101 samples for 100 URLs. Recomputing the
specified symmetric factor reproduces the persisted maximum exactly:

```text
completed=73/100
sample timestamp=1784374719.0986507
completion timestamp=1784374723.949654
cumulative rate=16.02698786785459 URLs/s
predicted remaining=1.6846584163299978 s
actual remaining=4.851003408432007 s
symmetric factor=2.879517510143001
```

The attempt ledger contains exactly 100 page attempts. Their starts span only
`0.1127057076` seconds (`1784374714.569131` through
`1784374714.6818368`), so host scheduling or late enqueue did not cause the
tail. At the failing sample all 100 calls had started, 73 had finished, and 27
were still in flight. Those 27 had already been in flight for approximately
4.4--4.5 seconds; the last one required another 4.845 seconds.

The code path is:

1. `fetch_unique_pages` invokes the callback with only `(completed, total)`.
2. `ProgressReporter` timestamps that callback.
3. `_unit_rates_locked` computes a cumulative completion rate from the stage
   baseline. The 60-second rolling rate is identical to the cumulative rate
   because the complete page stage lasted about 9.4 seconds.
4. `_record_unit_sample_locked` computes `remaining / rate`.

Thus the estimator treats the early parallel completion burst as a sustainable
serial throughput. It has no in-flight start/age/deadline information with
which to represent the right-censored slow URLs.

## Root cause

The accuracy failure is caused by model/input mismatch, not an arithmetic or
resume defect: a count-only cumulative-throughput ETA cannot infer the residual
time of a fully concurrent, right-censored network tail.

This is not repairable reliably by changing only the current rate formula. Two
runs can have exactly the same `(completed, total, timestamp)` history through
73/100 and then have disjoint valid futures (for example, the remaining URLs
complete in 0.5 seconds versus 10 seconds). An online estimator using the
current contract must emit the same prediction for both histories, while no
single prediction is within a factor of two of both futures. The observed gate
is one concrete instance of that information loss.

## Hypotheses rejected

- **Resume or wall-clock rollback:** the samples are monotonic and belong to
  the completed initial page stage; the failure is reproduced directly from
  persisted timestamps.
- **Late scheduling / per-host queueing:** every host is unique and all 100
  attempts started within 0.113 seconds.
- **Use the existing rolling rate:** the 60-second window covers the entire
  9.4-second stage, so rolling and cumulative rates are equal at the failing
  sample.
- **Tune a shorter window, percentile, safety multiplier, or threshold:** these
  choices either still fail other eligible samples or merely fit this one run.
  They are not justified by the information available and would violate the
  prohibition on gate-specific relaxation/discarding.
- **Use completion time while generating historical samples:** this would use
  future information and is expressly forbidden.

## TDD decision

The brief requires a RED test only after confirming a production implementation
defect with a valid minimal semantic fix. No such fix exists within the current
callback/state contract, so no test or production change was written. Creating
a test whose expected value is selected to make this historical gate pass would
encode an overfit constant rather than required behavior.

## Required design decision

A principled fix requires expanding the fetch-to-progress contract with bounded
online state such as per-attempt start/finish ages, active/queued counts, and
the applicable deadline, followed by choosing and specifying a censored-latency
ETA model. That is an interface and estimator design change, not the minimal
local correction authorized by this brief. It must define behavior for queued
work, concurrent cohorts, resume, timeout tails, and zero/very small stages
before a deterministic RED test can state the correct prediction.

## Commands and results

Read-only reproduction and attempt inspection used the `MMDD` environment.
The key results were:

```text
persisted maximum factor: 2.879517510143001 at 73/100
attempts: 100
start range: 1784374714.569131 .. 1784374714.6818368
at failing timestamp: started=100 finished=73 inflight=27 notstarted=0
```

No RED/GREEN commands were run because production behavior was not modified.
The baseline full repository result supplied for this investigation was
`669 passed` at `e5444120da32369b592b7df88c882c0a4627abaf`.

## Files changed

- `.superpowers/sdd/gate100-eta-fix-report.md` (this diagnostic report only)

## Concerns

- The 100-table acceptance gate remains failed on page ETA.
- Re-running it unchanged may produce a different stochastic factor but would
  not resolve the estimator's missing right-censoring information.
- No gate output/work/cache root was modified or deleted.
