# MediaWiki External-Link Filtering and HTTP 414 Recovery

## Goal

Prevent EntiTables-style external Wikimedia links from being treated as Wikipedia entity titles, and recover valid titles when a MediaWiki API batch is rejected with HTTP 414 `URI Too Long`.

## Parsing behavior

`parse_wiki_cell` will continue to recognize bracketed link markup. When the parsed target starts with `//`, `http://`, or `https://` (case-insensitive), it will return `wiki_title=None` and `has_wiki_link=True`. The display text remains available through the existing `text` field. Normal internal Wikipedia titles and the existing non-entity namespace filtering remain unchanged.

## Request behavior

MediaWiki page-title and image-title batches will keep their current maximum of 50 titles. If a request returns HTTP 414, the client will not apply the normal exponential retry loop to the same URL. Instead, the caller will split that batch into two non-empty halves and process each half independently.

If a one-title batch still returns HTTP 414, the client will count one API failure and skip that title. This is the recursion termination condition. Existing retry behavior for HTTP 429, HTTP 503, and transient request failures remains unchanged.

## Implementation boundary

The low-level request helper will expose HTTP 414 distinctly to the batching layer, because only the batching layer knows how to split a title list safely. A shared batch-processing helper may be introduced if needed to keep page and image-info behavior consistent, but unrelated MediaWiki behavior will not be refactored.

## Tests

Regression tests will verify:

- protocol-relative, HTTP, and HTTPS external links are not entity titles;
- normal internal links still parse as entity titles;
- a 414 page batch is split and valid sub-batches are returned;
- a single-title 414 terminates without repeated identical requests;
- existing non-414 retry behavior remains covered by the existing suite.
