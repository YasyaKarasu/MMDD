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
immutable `UrlProgressSnapshot`. One snapshot contains:

- `timestamp`: reporter logical-clock seconds;
- `completed_durable`, `total`;
- `pending_not_started`, `in_flight`, `finished_not_durable`;
- `deadline_seconds`, `effective_concurrency`;
- `transport_events[64]`: completed physical-transport durations;
- `active_censors[64]`: current ages of active transports;
- `commit_events[32]`: finish-to-durable-callback durations;
- `transport_overflow_events`, `active_overflow_censors`, and
  `commit_overflow_events`.

All counts are non-negative integers. The following equality is mandatory at
every callback:

```text
completed_durable + pending_not_started + in_flight
    + finished_not_durable == total
```

Suppressed durable replays are already durable and belong only in
`completed_durable`. A physical attempt becomes `in_flight` after its durable
start fence, moves to `finished_not_durable` after its finish fence, and moves
to `completed_durable` after outcome and job completion commit. Exceptions use
the same transitions.

The scheduler maintains these aggregates in memory under its existing state
lock. The callback must not scan SQLite. Resume initializes the aggregates
from one bounded grouped query per store before scheduling begins.

## Fixed histograms

Transport horizon `H` equals the configured request deadline. Transport bin
edges are linear and closed on the right:

```text
T[j] = j * H / 64, j = 0..64
bin(x) = min(63, floor(64 * max(0, x) / H))
```

Durations and active ages `>= H` increment the corresponding overflow counter
and bin 63. Overflow values are not assigned a fabricated duration beyond the
policy deadline.

Commit horizon `C` is `min(2.0, H)` seconds, with 32 linear bins
`C[j] = j*C/32`. Commit durations `>= C` increment `commit_overflow_events`
and bin 31. These constants are schema fields and may change only with a new
telemetry schema version.

## Deterministic Kaplan-Meier estimate

For transport bin `j`, let `d[j]` be completed transport events and `c[j]` be
active right-censors. Compute the risk set from high to low:

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

The in-flight component is the expected drain of the current active cohort:

```text
inflight_eta = max(R(j) for every occupied active-censor bin j)
```

Using occupied bins, not individual attempts, keeps the operation bounded.

## Queue and commit drain

The existing durable completion rate remains:

```text
durable_rate = (completed_durable - baseline_completed)
               / (timestamp - baseline_timestamp)
```

When the denominator or numerator is zero, the rate is unavailable.

Queue drain is:

```text
queue_eta = pending_not_started / durable_rate
```

Commit latency uses the deterministic midpoint mean of the 32-bin completed
commit histogram, capped at `C`. For bin `j`, its representative duration is
`(C[j] + C[j+1]) / 2`; `commit_mean` is the count-weighted mean of those
representatives. With no commit events it falls back to `C/32`.

```text
commit_mean = histogram_RMST(commit_events, horizon=C)
commit_eta = finished_not_durable * commit_mean
             / max(1, effective_concurrency)
```

The prediction is:

```text
rate_eta = (total - completed_durable) / durable_rate
eta = max(rate_eta, queue_eta + inflight_eta, commit_eta)
```

Unavailable components are omitted, not replaced with infinity.

## Fallbacks

Fallback order is deterministic:

1. If no durable rate and no completed transport events exist, ETA is null.
2. If the KM risk set is empty but active requests exist, use
   `inflight_eta = max(0, H - oldest_active_age_capped_to_H)`.
3. If any active overflow censor exists, add one commit horizon to the current
   `rate_eta`: `overflow_eta = rate_eta + C`; include it in the final maximum.
   This covers the observed beyond-deadline final-request/commit window while
   remaining bounded.
4. When `completed_durable == total`, emit the normal zero-remaining sample and
   complete the stage exactly as today.

Every fallback choice and overflow count is persisted in the sample so resume
recomputes the same summary and tampering fails closed.

## Progress schema and bounds

Each sample adds only scalar topology fields, the chosen ETA components,
overflow counters, and the final prediction. Histogram arrays belong to the
reporter's current-stage accumulator and are not copied into every sample.
Completed-stage telemetry persists one final 64-bin transport histogram and
one 32-bin commit histogram. With two URL stages and at most 256 samples per
stage, `progress.json` remains below its existing 4 MiB limit.

Snapshots must be monotonic in timestamp and durable completion. `total` and
the deadline cannot change within a stage. Histogram counts cannot decrease
except `active_censors`, which is a current-state histogram. Resume restores
completed-event histograms and reconstructs the current active histogram from
durable unfinished attempt rows.

## Validation and compatibility

- Manifest `url_completion` remains derived from bounded progress telemetry.
- Additive fields receive a new `wdc200k-url-telemetry-v2` schema version.
- A v1 completed stage remains readable and retains its recorded ETA; it is
  not retroactively re-estimated.
- A resumed incomplete v1 URL stage upgrades only after reconstructing v2
  scheduler state from durable jobs and attempts.
- Dry-run and structural-only runs create no transport telemetry.
- All grouped initialization queries are read-only and disk bounded; all new
  persistence uses the existing atomic, guarded progress write.

## Tests

Unit tests use a fake logical clock and deterministic snapshots:

1. KM risk sets, piecewise survival, RMST, overflow, and empty-risk fallback;
2. queue-only, in-flight-only, commit-only, and mixed drain equations;
3. topology equality, monotonicity, finite-number, bin-length, and deadline
   validation failures;
4. at most 64/32 histogram bins and 256 samples regardless of URL count;
5. resume reconstructs identical aggregates and does not replay attempts;
6. v1 completed telemetry remains readable; incomplete v1 upgrades to v2;
7. disk-guard failure leaves the prior atomic progress snapshot intact;
8. dry-run/structural runs create no v2 telemetry.

Regression fixtures replay the exact gate-100 page and image online event
streams using only each event prefix. Both must retain every eligible
final-half sample. Required results are:

- page maximum symmetric factor `<= 2.0`;
- image maximum symmetric factor `<= 1.8537476708755196` (no degradation from
  the accepted observed image gate);
- predicted values at every sample are reproducible after interruption and
  resume;
- no future timestamp or final completion time is visible to the estimator.

If the fixed estimator fails either replay constraint, implementation stops;
the formula or constants require a new approved design revision rather than a
threshold change, sample exclusion, or gate-specific stage multiplier.
