# WDC Scale-Gate Telemetry Design

## Goal

Make the 100- and 1,000-table gates directly prove three properties that the
durable outcome sets alone cannot prove:

1. how many physical transport invocations were started;
2. whether a normalized URL was invoked more than once or a durable terminal
   outcome was replayed after resume; and
3. whether the pages and images ETA stayed within the documented final-half
   accuracy bound.

The telemetry is pipeline state, not redirected stdout or a log file. It must
remain bounded, disk guarded, atomic, resumable, and covered by the same
authority checks as other stage state.

## Transport Attempts

Page and image outcome stores gain an append-only `transport_attempts` table.
Each row contains:

- a generated attempt id and process execution id;
- stage, policy fingerprint, normalized URL key, and normalized URL;
- start and finish timestamps plus final status;
- the durable outcome status visible at the execution baseline; and
- whether the call was suppressed because a durable outcome already existed.

The scheduler writes and commits an attempt immediately after the final cache
and durable-outcome check and immediately before calling
`fetch_page`/`download_image`. Completion or exception fences the matching row
with a finish timestamp. A durable success or terminal outcome suppresses the
physical call. The suppressed anomaly is counted separately and is never
reported as a physical invocation.

Gate counters are calculated directly from the attempt rows:

- `transport_attempts`: non-suppressed starts;
- `duplicate_physical_requests`: sum of attempts beyond the first per
  `(stage, policy_fingerprint, url_key)`;
- `terminal_replays`: non-suppressed attempts whose execution baseline already
  contained a terminal outcome;
- `unfinished_transport_attempts`: starts without a fenced completion; and
- `blocked_durable_replays`: calls suppressed by the final durable check.

There is an unavoidable external-call crash window between the committed
attempt start and the fenced completion. Such rows are conservatively counted
as started and also as unfinished. A gate may claim exact, zero-duplicate
physical execution only when unfinished attempts are zero. Resume never
silently erases or rewrites attempt rows.

All schema changes, inserts, and updates use the existing guarded SQLite write
path. Attempt summaries and their database identity are included in page/image
manifests and stage registries so tampering or stale-policy reuse fails closed.

## URL-Unit Progress and ETA

`ProgressReporter` accepts `completed_units`, `total_units`, and a
`rate_basis`. Pages use terminal page outcomes over unique page jobs; images
use terminal image outcomes over unique image jobs. Resume starts from the
durable baseline rather than zero.

Fetchers expose a progress callback and publish bounded updates. The builder
chooses an interval that yields at most roughly 256 updates per stage while
still producing intermediate samples for small gates. The reporter maintains
at most 256 samples per stage and atomically publishes them with bounded
progress state. Completed-stage summaries remain available after the pipeline
moves to later stages.

Each sample records time, completed/total units, rate, rolling rate, and
predicted remaining seconds. At stage completion, the reporter records the
completion timestamp and computes the symmetric ETA factor for samples in the
final half:

`max(predicted / actual, actual / predicted)`

Zero-duration and null-ETA samples are excluded and their count is reported.
A formal gate passes the ETA requirement only when it has at least one
eligible final-half sample and the maximum factor is at most 2.0.

## Acceptance and Compatibility

The existing outcome schemas, public artifacts, source rows, query
construction, and multimodal budgets do not change. New telemetry fields are
additive. Dry-run remains read-only. Structural-only runs create no transport
telemetry. All new state observes the configured minimum free-disk reserve.

