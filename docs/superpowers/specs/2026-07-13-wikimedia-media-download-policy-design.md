# Wikimedia Media Download Policy Design

## Context

The joinability builders use the MediaWiki Action API to fetch page and
`imageinfo` metadata, then download image bodies from
`upload.wikimedia.org`. These are different traffic classes and must not share
one fixed-delay policy.

The current Action API client is serial and rate limited, but media downloads
use two worker threads without an aggregate bandwidth limiter. Media responses
are attempted once, so HTTP 429 responses create cache holes that are retried
only when the entire pipeline is run again. Media files are also keyed by
entity-specific asset IDs, which permits repeated downloads of the same source
URL when multiple entities reference it.

The current cache audit found 75,162 image assets selected by the active
configuration, of which 60,291 final files exist and 14,871 are missing. This
design must fill those gaps without discarding the 60,291 usable files.

## Goals

- Keep Action API and media-download controls independent.
- Follow Wikimedia's media rules: aggregate concurrency at most two and
  aggregate download speed below 25 Mbps over ten-second intervals.
- Respect `Retry-After` for client-wide rate-limit responses.
- Retry transient media failures during the current run, then record and skip
  exhausted failures so one URL cannot block the whole build.
- Retry skipped failures on the next run; do not create a permanent negative
  cache.
- Avoid downloading the same media URL more than once in a run and reuse new
  URL-keyed media files across runs.
- Preserve all valid legacy asset-ID-keyed image files.
- Make cache hits, retries, throttling, and terminal failures observable.

## Non-goals

- Coordinating multiple builder processes or multiple machines. The deployment
  runs one builder process at a time behind one public IP.
- Authenticating to Wikimedia or requesting a higher quota.
- Replacing the live fetch pipeline with Wikimedia dumps or Wikimedia
  Enterprise.
- Rewriting existing model-extraction cache keys or entity asset IDs.

## External Constraints

The implementation follows these Wikimedia requirements:

- Unauthenticated Action API traffic remains serial and below five requests
  per second, using batch requests where supported.
- Media traffic to `upload.wikimedia.org` has total concurrency at most two and
  total download speed at most 25 Mbps, measured over ten-second intervals.
- Media consumers should prefer standard thumbnail sizes and prefer thumbnails
  to originals when practical.
- A meaningful bot User-Agent with operator contact information is required.
- HTTP 429 responses must honor `Retry-After`. When 429 or 503 lacks that
  header, the client waits at least five seconds and uses exponential backoff.

The default media limit will be 24 Mbps, leaving headroom below the 25 Mbps
policy ceiling. The requested thumbnail width will be the standard 960 pixels.

## Architecture

### Action API path

`WikipediaClient._get()` remains responsible only for Action API requests.
It keeps the existing serial lock, batching, and five-requests-per-second
ceiling. Non-interactive queries add `maxlag=5`. The existing `--sleep` option
continues to affect only the Action API path.

### Media path

A new `WikimediaMediaDownloader` component owns all media-body behavior:

1. `MediaConcurrencyGate` enforces one or two active media responses, never
   more than two.
2. `MediaBandwidthLimiter` meters bytes across both workers at a default
   aggregate rate of 24 Mbps.
3. `MediaCooldown` stores a shared `blocked_until` timestamp for server-issued
   client-wide cooldowns.
4. `MediaRetryPolicy` classifies failures and computes retry delays.
5. `MediaCache` checks legacy asset paths, URL-keyed shared paths, and
   in-process single-flight results.
6. `MediaFailureRecorder` writes terminal failures and aggregate counters.

`WikipediaClient.download_image()` keeps its public return contract of a
download-record dictionary or `None`, but delegates network and shared-cache
work to the downloader. This limits changes to existing callers.

The component should live in a focused module such as
`scripts/wikimedia_media.py`; `build_mm_table_dataset.py` wires it into the
existing client and build flow.

## Configuration

The builders expose the following options and pass them through the dynamic
vLLM wrapper unchanged:

- `--media_download_workers`, default `2`, accepted range `1..2`.
- `--media_max_mbps`, default `24.0`, accepted range `(0, 25]`.
- `--media_max_retries`, default `5`, meaning five retries after the initial
  attempt.
- `--media_retry_base_seconds`, default `5.0`.
- `--media_retry_max_seconds`, default `60.0`.
- `--media_chunk_bytes`, default `131072`.

Invalid values fail during argument validation instead of being silently
clamped. Tests may set zero retry delays through direct component construction;
the production CLI requires non-negative delays.

`--wikipedia_workers` remains a deprecated compatibility option and does not
control media concurrency. Help text will explicitly distinguish the two
traffic classes.

## Bandwidth Algorithm

Both media workers share a monotonic-clock token bucket:

- Refill rate is `media_max_mbps * 1_000_000 / 8` bytes per second.
- Bucket capacity is one configured media chunk, keeping bursts small enough
  that any ten-second interval remains under the 25 Mbps ceiling at the
  default 24 Mbps setting.
- A worker reserves one chunk before reading the next streamed response chunk.
  Unused capacity from a short final chunk is returned.
- Waiting uses a condition variable and monotonic deadlines; it does not hold
  the cache or failure-recorder locks.
- Cache hits consume no tokens.

The limiter controls response-body bytes, not request count. This is why it is
separate from the Action API interval limiter.

## Retry and Cooldown State Machine

Before opening a new media HTTP request, a worker:

1. waits for the shared cooldown to expire;
2. acquires the media concurrency gate;
3. opens the streaming response;
4. reads through the shared bandwidth limiter;
5. atomically promotes the temporary file only after a complete successful
   response and any required SVG conversion.

Failure classification is:

- HTTP 429 caused by request volume: retryable and client-wide. Parse
  `Retry-After` as either seconds or an HTTP date and extend shared
  `blocked_until` with the maximum observed deadline.
- HTTP 503 with `Retry-After`: retryable and client-wide, using the same shared
  cooldown behavior.
- HTTP 408, 500, 502, 503 without `Retry-After`, 504, connection errors, and
  timeouts: retryable for that media item only.
- HTTP 429 whose body says a non-standard thumbnail size was requested:
  configuration failure, not rate limiting; do not retry or set a cooldown.
- Other HTTP 4xx responses, non-image content, and deterministic SVG
  conversion errors: terminal for the current run.

When no usable `Retry-After` exists, retry delay is exponential backoff starting
at at least five seconds and capped at 60 seconds, plus small random jitter.
A server-provided delay is never shortened by local caps.

Shared cooldown blocks only new requests. A different worker that already has
a successful HTTP 200 response may finish streaming it. Ordinary per-item
network failures do not stop the other worker.

After the configured retry budget is exhausted, the method removes temporary
files, records the failure, returns `None`, and allows the build to continue.
Because terminal failures do not create a cache entry, the next pipeline run
tries them again.

## Cache and Deduplication

Cache lookup order is:

1. Existing non-empty legacy file named from the entity-specific `asset_id`.
2. Existing non-empty shared media file named from a stable hash of the exact
   selected download URL and its output format.
3. An in-process single-flight entry for the same media key.
4. A new network download.

New media bodies are stored under deterministic URL-keyed names in the existing
shared image cache directory. Multiple asset records may reference one shared
local path while retaining their distinct asset IDs and entity relationships.
The model-extraction cache therefore remains keyed exactly as it is today.

Single-flight stores both success and terminal failure outcomes for the life of
one process. If multiple entity assets request one URL concurrently, only the
leader performs retries; followers receive the same outcome. A failure is not
persisted as a negative cache and is eligible again on the next run.

Legacy files are not renamed, deleted, hashed, or rewritten. This preserves the
60,291 currently usable selected files and avoids a migration download storm.

The imageinfo request explicitly asks for the standard 960-pixel thumbnail.
The downloader prefers that thumbnail when it does not upscale a raster image,
and otherwise uses the smaller original. SVG thumbnails remain rasterized
server-side when available.

## Failure Records and Metrics

Each run overwrites `<output_dir>/media_download_failures.jsonl` at startup and
appends one record per terminal media key. Records contain:

- media key, selected URL, file title, and affected asset ID;
- final HTTP status or exception class;
- total attempts and retry delays;
- whether a `Retry-After` header was present;
- timestamp and a short sanitized error message.

The file must not contain response bodies beyond a short diagnostic excerpt or
any credentials.

The run manifest reports:

- legacy cache hits;
- shared URL-cache hits;
- single-flight followers;
- successful network downloads;
- downloaded bytes;
- retry attempts, 429 count, and 503 count;
- bandwidth-throttle wait seconds and shared-cooldown wait seconds;
- terminal failures.

Progress output separates entity/metadata traversal from media completion so
the entity progress bar is not mistaken for a count of network downloads.

## User-Agent Handling

The same descriptive User-Agent is sent on Action API and media requests. When
Wikipedia access is enabled and the built-in placeholder User-Agent is still
in use, the builder emits a prominent warning that operator contact information
is required. A CLI-supplied value or `WIKIPEDIA_USER_AGENT` environment value
continues to take precedence. This change does not print the configured contact
value in logs.

## Testing

Unit tests use fake sessions, fake clocks, and temporary directories; they make
no Wikimedia network calls. Required coverage includes:

1. Action API requests remain serial, batched, and independent of media
   throttling.
2. Media concurrency never exceeds two.
3. Two workers together remain within the configured aggregate token-bucket
   rate.
4. A rate-limit 429 with `Retry-After` blocks both workers from starting new
   requests, while an already streaming HTTP 200 response may finish.
5. A timeout or connection error retries only its media item.
6. A thumbnail-configuration 429 and non-retryable 4xx response fail without a
   cooldown loop.
7. Retry exhaustion writes one failure record, creates no final cache file, and
   permits a subsequent process to retry.
8. Legacy cache files are returned without network access.
9. URL-keyed cache hits and same-process duplicate requests perform one network
   download.
10. Temporary files are removed after every terminal failure.
11. Imageinfo requests use the standard 960-pixel thumbnail width.
12. Manifest counters distinguish cache hits, downloads, retries, and failures.

The existing Stage-1 command remains the regression suite:

```bash
conda run -n MMDD python -m pytest tests/test_stage1_pipeline.py -q
```

Focused media tests should also be runnable independently while developing.

## Rollout and Verification

The first real run after deployment reuses legacy files, fills only missing
media, and writes the new failure ledger. Verification compares:

- selected image count;
- pre-run and post-run successful cache coverage;
- terminal failures by status;
- observed aggregate throughput and maximum concurrency;
- absence of repeated request-volume 429 bursts.

Success means the builder stays at or below two media connections and 24 Mbps,
honors every server cooldown, continues after exhausted failures, and increases
cache coverage without redownloading legacy successes.
