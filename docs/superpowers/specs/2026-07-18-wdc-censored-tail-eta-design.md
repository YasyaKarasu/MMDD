# WDC Censored-Tail ETA Design

## Goal

Replace the URL-stage ETA's completion-count-only estimate with a bounded,
deterministic drain estimate that accounts for queued work, right-censored
in-flight requests, and transport-finished outcomes awaiting durable commit.
The formal acceptance rule remains unchanged: every eligible final-half sample
is retained, and the maximum symmetric ETA factor must be at most `2.0`.

This change affects telemetry only. Fetch policy, retries, deadlines,
concurrency, durable outcomes, dataset contents, and stage completion semantics
do not change.

## Observed failure

At page `73/100`, all transports had started in one wave and 27 old requests
remained in flight. `remaining / cumulative_rate` predicted `1.685 s`, while
the drain took `4.851 s`. Those active requests were censored observations,
not ordinary unseen work. The image stage instead had queued and newly started
work, and its existing ETA passed. A global multiplier or a deadline floor is
therefore rejected: either is gate-specific and can degrade images.

## Callback contract

The fetchers replace the `(completed, total)` callback payload with an
immutable `UrlProgressSnapshot`. The reporter owns the logical-clock
`timestamp` and derives it from the epoch's reporter baseline plus the
snapshot's monotonic elapsed time; the fetcher does not manufacture a wall or
logical timestamp. One snapshot contains:

- `completed_durable`, `total`;
- `local_buffered_not_started`, `in_flight_jobs`,
  `finished_not_durable`;
- `physical_in_flight`, a diagnostic subset of `in_flight_jobs` used by the
  censor histogram and not added separately to the topology equality;
- `unobserved_nonlocal`, the residual jobs that are pending, unclaimed, or
  leased by another execution;
- `deadline_seconds`, `effective_concurrency`;
- `execution_epoch`, `baseline_completed`, and `epoch_elapsed_seconds`;
- `transport_event_histogram[64]`: completed physical-transport durations;
- `active_censor_histogram[64]`: current ages of local active transports;
- `commit_event_histogram[32]`: finish-to-durable-callback durations;
- `transport_overflow_events`, `active_overflow_censors`, and
  `commit_overflow_events`.

All counts are integers in `[0, 2**64-1]`. The following equality is mandatory
at every callback:

```text
completed_durable + local_buffered_not_started + in_flight_jobs
    + finished_not_durable + unobserved_nonlocal == total
```

`physical_in_flight <= in_flight_jobs`, and the active-censor histogram sums
exactly to `physical_in_flight`. Overflow counters cannot exceed their
corresponding histogram totals.

`completed_durable` means both the outcome and current job completion are
durable. A scheduled worker job belongs to `in_flight_jobs` whether it makes a
physical request, waits for another image URL claimant, or finds a final-check
cache/suppressed outcome. A real physical attempt also increments
`physical_in_flight` after its durable start fence and leaves it after its
finish fence. When any worker future returns, its job moves from
`in_flight_jobs` to `finished_not_durable`; only the serial outcome/job commit
moves it to `completed_durable`. A cache hit reconciled synchronously during
claiming commits the current job before the next snapshot and moves directly
from local buffered state to completed. Thus no-transport paths obey the same
job topology without fabricating a latency event.

The scheduler maintains local aggregates in a thread-safe, bounded tracker.
Each stage permits at most 224 completion publications plus one mandatory
baseline publication for each of at most 32 execution epochs, for an absolute
maximum of 256 persisted samples. The production completion interval is
`ceil(total / 224)`. Completion milestones are absolute multiples of that
interval, not counts relative to the current process baseline. Resume starts
at the first milestone strictly above the indexed durable baseline, so it
neither resets nor repeats old milestones. A user `progress_callback_every`
can only increase the effective interval. A refresh that crosses several
milestones, including an external durable-completion jump or a completed
future batch, emits one current snapshot. The forced final publication counts
toward the 224 budget and is coalesced when the same durable completion was
already published.

Immediately before each bounded callback, the scheduler performs one indexed,
read-only aggregate refresh of exact durable job/outcome counts;
this occurs outside the callback and returns only scalar counts. It then
obtains `unobserved_nonlocal` by subtraction, so completions made by another
live execution are visible and concurrent leases cannot break the topology
equality. Small start/finish hooks sit immediately
inside the existing durable physical-attempt fence and use `time.monotonic()`
for ages and durations. The callback never scans SQLite and never retains a
URL. A batch returned by `wait(FIRST_COMPLETED)` enters
`finished_not_durable` before its serial outcome/job commits, then leaves that
state one result at a time.

The durable finish fence is also the tracker transition fence. If the attempt
finish guard or SQLite transaction fails, the worker future fails and the
tracker correctly remains physical-active. For a completed future batch, the
scheduler distinguishes failed futures first and calls `future_finished` only
for futures that crossed the finish fence successfully. It releases scheduler
resources before calling `future.result()`, which propagates the original
guard or SQLite exception without replacement, retry, or conversion.

At the start of an epoch, the tracker records `epoch_started_monotonic`; every
snapshot captures topology, histograms, and `captured_monotonic` under the same
tracker lock and carries
`epoch_elapsed_seconds = captured_monotonic - epoch_started_monotonic`.
The reporter assigns the epoch's first logical timestamp as
`baseline_timestamp`, then maps every sample in that epoch to
`baseline_timestamp + epoch_elapsed_seconds`. This binds the histogram capture
instant to the logical timeline without mixing wall and monotonic clocks.

Each process execution has a new `execution_epoch`. On resume, the indexed
durable refresh establishes `baseline_completed`; current transport, active,
and commit histograms start empty because a dead prior process has no live
transport. Previously published samples remain immutable under their prior
epoch. Work held by another live execution remains in
`unobserved_nonlocal`. This avoids treating stale unfinished-attempt rows as
active work or loading per-attempt history.

## Fixed histograms

Transport horizon `H` equals the configured request deadline. Transport bin
edges are linear and left-closed/right-open, except that the last bin also
contains the capped horizon:

```text
T[j] = j * H / 64, j = 0..64
bin(x) = bisect_right(T, max(0, x)) - 1, for x < H
```

Durations and active ages `>= H` increment the corresponding overflow counter
and bin 63. Overflow values are not assigned a fabricated duration beyond the
policy deadline.

The `x >= H` overflow check occurs before the bisection. Exact-edge tests use
the same generated edges and require `T[j]` for `0 <= j < 64` to enter bin
`j`, and `H` to enter bin 63 with overflow set. This definition also applies
to non-binary horizons such as `0.1` and `1.3`.

Commit horizon `C` is `min(2.0, H)` seconds, with 32 linear bins
`C[j] = j*C/32`. Commit durations `>= C` increment `commit_overflow_events`
and bin 31. These constants are schema fields and may change only with a new
telemetry schema version.

## Deterministic Kaplan-Meier estimate

For transport bin `j`, let `d[j]` be completed transport events in the current
execution epoch and `c[j]` be its currently active right-censors. Compute the
risk set from high to low:

```text
n[j] = sum(d[k] + c[k] for k in j..63)
S[0] = 1
S[j+1] = S[j] * (1 - d[j] / n[j]) if n[j] > 0 else S[j]
```

All arithmetic uses Python `float`; persisted values must be finite. Within a
bin, survival is piecewise constant. The deadline-capped restricted mean
remaining time for an active request of age-bin `a` is:

```text
R(a) = sum((T[j+1]-T[j]) * S[j] / max(S[a], 1e-12)
           for j in a..63)
```

Clamp `R(a)` to `[0, H-T[a]]`. Active overflow censors have `R=0`; their
uncertain beyond-deadline residual is handled by the fallback below rather
than inventing an uncapped tail.

Raw RMST for a newly started final request overpredicted the recorded image
tail by more than 30 times. Censor correction therefore activates only after
the oldest local active request has survived one third of the configured
deadline. This maturity fraction is a schema constant, applies identically to
pages and images, and changes only with a telemetry schema version. Let
`a_oldest` be the oldest occupied active bin:

```text
if H/3 <= T[a_oldest] < H:
    inflight_eta = R(a_oldest)
else:
    inflight_eta = unavailable
```

Using the oldest occupied bin, not individual attempts, corrects a mature
right-censored drain without allowing a young request's unconditional RMST to
override an already accurate observed completion rate.

## Queue and commit drain

The existing durable completion rate remains:

```text
durable_rate = (completed_durable - baseline_completed)
               / epoch_elapsed_seconds
```

When the denominator or numerator is zero, the rate is unavailable.
`baseline_timestamp` remains the persisted logical-time authority for sample
ordering and completion-factor evaluation; monotonic epoch elapsed time is the
rate denominator.

The queue diagnostic is:

```text
queue_eta = (local_buffered_not_started + unobserved_nonlocal)
            / durable_rate
```

It is persisted for observability. It is not added to `inflight_eta` because
`rate_eta` below already covers every non-durable job; adding both double-
counts the queued portion.

Commit latency uses the deterministic midpoint mean of the 32-bin completed
commit histogram, capped at `C`. For bin `j`, its representative duration is
`(C[j] + C[j+1]) / 2`; `commit_mean` is the count-weighted mean of those
representatives. With no commit events it falls back to `C/32`.

```text
commit_mean = histogram_midpoint_mean(commit_event_histogram, horizon=C)
commit_eta = finished_not_durable * commit_mean
```

Durable commits happen serially in the scheduler thread, so this component is
not divided by transport concurrency.

The prediction is:

```text
rate_eta = (total - completed_durable) / durable_rate
eta = max(rate_eta, inflight_eta, commit_eta, overflow_eta)
```

Unavailable components are omitted, not replaced with infinity.

## Fallbacks

Fallback order is deterministic:

1. If no durable rate and no completed transport events exist, ETA is null.
2. If the KM risk set is empty, or the oldest active request is younger than
   `H/3`, `inflight_eta` is unavailable and `rate_eta` remains authoritative.
3. If any active overflow censor exists and durable rate is available, use a
   two-completion continuity correction: `overflow_eta = 2 / durable_rate`.
   It represents two observed durable inter-arrival intervals and does not
   invent an uncapped request duration. Without a durable rate, overflow ETA
   is unavailable.
4. When `completed_durable == total`, emit the normal zero-remaining sample and
   complete the stage exactly as today.

Every fallback choice and overflow count is persisted in the sample so resume
recomputes the same summary and tampering fails closed.

## Progress schema and bounds

Each sample contains scalar topology fields, the chosen ETA components,
overflow counters, the final prediction, and fixed 64-bin transport/current-
censor plus 32-bin commit histograms. For JSON persistence, the 160 counters
are concatenated in that order as unsigned little-endian `uint64` and base64
encoded in one `histogram_blob`; values outside `[0, 2**64-1]` fail closed.
The decoder requires exactly 1,280 bytes. This preserves exact counters while
avoiding pretty-printed per-element overhead. A serializer test constructs
both stages at 256 samples with every counter and scalar at its maximum width;
the resulting actual schema must be below 3 MiB, leaving at least 1 MiB below
the existing 4 MiB restore limit.

Snapshots must be monotonic in timestamp and durable completion. `total` and
the deadline cannot change within a stage. Within one `execution_epoch`, event
histograms cannot decrease; `active_censor_histogram` is current state and may
change. Resume validates prior epochs, then starts a new epoch with empty
histograms and a fresh durable baseline.

Epoch and sample authority is append-only and fail-closed:

- an epoch identifier is non-empty, unique within the stage, and occupies one
  contiguous sample range; a later sample cannot return to an older epoch;
- the first sample has `epoch_elapsed_seconds=0`, empty histograms,
  `completed_durable=baseline_completed`, and `baseline_completed` equal to
  the exact indexed durable refresh used to start that epoch;
- `baseline_timestamp` is finite, not earlier than the prior sample, and is
  identical on every sample of the epoch;
- elapsed time and durable completion are monotonic inside the epoch;
- a new epoch's baseline completion is at least the prior sample's completion,
  while the stage total/deadline/rate basis remain identical.
- the 33rd unique execution epoch or 257th sample is rejected before telemetry
  append or `progress.json` replacement; previously published bytes and sample
  objects remain unchanged;
- no accepted sample or epoch is trimmed, compacted, or excluded to recover
  capacity. Every generated eligible final-half sample remains authoritative.

## Validation and compatibility

- Manifest `url_completion` remains derived from bounded progress telemetry.
- Additive fields receive a new `wdc200k-url-telemetry-v2` schema version.
- A v1 completed stage remains readable and retains its recorded ETA; it is
  not retroactively re-estimated.
- A resumed incomplete v1 URL stage starts a v2 execution epoch from its exact
  durable completion baseline; it does not reinterpret historical v1 samples.
- Dry-run and structural-only runs create no transport telemetry.
- All new persistence uses the existing atomic, guarded progress write; no
  estimator callback performs a database query.

## Tests

### Approved formula replay

Read-only prefix replay was performed during design against the recorded
gate-100 page and image attempt/durable timestamps. The rejected raw design
produced factors `9.156` for pages and `33.507` for images. Applying the
one-third maturity rule, oldest-bin RMST, no queue double-counting, and the
two-interval overflow correction produced `1.8978047370910645` for pages and
left images at `1.8537476708755196`. These are design evidence only; the first
implementation step must encode the same aggregate event streams as RED
regression fixtures before production code changes.

Unit tests use a fake logical clock and deterministic snapshots:

1. KM risk sets, piecewise survival, RMST, overflow, and empty-risk fallback;
2. queue-only, in-flight-only, commit-only, and mixed drain equations;
3. topology equality, monotonicity, finite-number, bin-length, and deadline
   validation failures;
4. all exact edges for binary and non-binary horizons, `uint64` blob
   length/range/tamper validation, 224 absolute completion publications plus
   at most 32 epoch baselines, and the worst-width file below 3 MiB;
5. resume preserves prior samples, starts an empty-histogram epoch from the
   durable baseline, and does not replay attempts;
6. epoch identifiers are unique/contiguous, baselines bind to the durable
   refresh, and rollback-safe logical timestamps map exact monotonic elapsed;
7. cache hits, suppressed final checks, and other-claim outcomes follow the
   no-physical job transitions without violating topology;
8. a concurrent execution's durable completion is observed by the next
   indexed refresh without loading rows;
9. v1 completed telemetry remains readable; incomplete v1 upgrades to v2;
10. disk-guard failure leaves the prior atomic progress snapshot intact;
11. dry-run/structural runs create no v2 telemetry.
12. finish-fence guard and SQLite failures retain physical-active tracker
    authority, leave the attempt unfinished, and propagate the root error;
13. frozen fixtures have fixed SHA-256 digests, global chronology, and strict
    per-attempt start/finish/durable ordering, with no host, URL, or payload.

Before any production implementation, aggregate regression fixtures are
created from the exact gate-100 page and image attempt start/finish timestamps
and durable callback timestamps. They contain no URL or payload. Tests replay
the online event streams using only each event prefix. Both must retain every
eligible final-half sample. Required results are:

- page maximum symmetric factor `1.8978047370910645` (`<= 2.0`);
- image maximum symmetric factor `<= 1.8537476708755196` (no degradation from
  the accepted observed image gate);
- predicted values at every sample are reproducible after interruption and
  resume;
- no future timestamp or final completion time is visible to the estimator.

If the fixed estimator fails either replay constraint, implementation stops;
the formula or constants require a new approved design revision rather than a
threshold change, sample exclusion, or gate-specific stage multiplier.
