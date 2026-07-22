# Remote vLLM Endpoints Design

## Goal

Allow the WDC 200K model stage to use text and vision vLLM servers on a remote
host without turning temporary endpoint outages into permanent model-task
failures.

## Existing behavior

`LocalAttributeExtractor` already accepts independent OpenAI-compatible text
and image base URLs, optional endpoint pools, model names, and bearer tokens.
The WDC builder forwards those settings unchanged. The missing production
guardrails are endpoint readiness validation and transient transport failure
handling: an unavailable endpoint currently becomes a committed `terminal`
model result.

## Client and configuration

The existing `--text_model_base_url` and `--image_model_base_url` options remain
the canonical remote configuration. The plural URL options remain additive;
therefore a single remote endpoint must replace the singular default rather
than only being added through the plural option.

API keys may be supplied through the existing CLI options or through
`MMDD_TEXT_MODEL_API_KEY`, `MMDD_IMAGE_MODEL_API_KEY`, with `VLLM_API_KEY` as a
shared fallback. An explicit CLI value wins. Secret values must not be added to
run fingerprints, manifests, or logs.

The consolidated WDC CLI will expose the model request timeout, retry count,
retry delay, and endpoint preflight timeout needed for a remote network path.

## Endpoint preflight

Before the first job claim, the model stage probes every configured endpoint
for each modality that still has unfinished work. It sends an authenticated
`GET <base-url>/models`, requires a 2xx response, parses the OpenAI model list,
and verifies that the configured served model name is present.

The probe waits up to a configurable timeout using bounded polling. If any
endpoint is unavailable, unauthorized, malformed, or serving the wrong model,
the model stage raises a clear error before claiming work. Completed modalities
are not probed during resume. Extractor test doubles without a readiness method
remain supported.

## Transient failure handling

Connection failures, request timeouts, HTTP 429, and HTTP 5xx responses are
classified as transient endpoint errors. If one occurs after preflight, the
affected claimed job is finished as `retryable` without a durable
`model_results` row or model cache entry. The current bounded group may finish,
then the model stage fails fast so an outage cannot sweep the remaining queue.

Non-transient request errors, invalid model output, and evidence-specific
errors retain the existing terminal behavior. A normal `--resume` run reclaims
retryable and expired leased jobs while preserving successful jobs and the
existing model-task adapter.

## Remote deployment

For one A100 80GB, run two vLLM processes on GPU 0: text on port 8001 and vision
on port 8000. Each process receives a bounded GPU-memory fraction so their sum
stays below one. Both expose explicit served model names matching the local
builder configuration and share an API key.

The preferred network path is an SSH tunnel with both vLLM servers bound to
`127.0.0.1`. Direct trusted-LAN access may bind to `0.0.0.0`, but firewall rules
must restrict both ports to the builder host. The service must not be exposed
unrestricted to the public Internet.

## Tests

Focused tests use fake HTTP responses only and cover:

- exact remote text/image URL, model, and bearer-token routing;
- authenticated `/models` validation and wrong-model/HTTP failure reporting;
- preflight before any job claim, with untouched pending jobs on failure;
- probing only modalities with unfinished work;
- transient connection/timeout/429/5xx results becoming retryable without a
  committed result;
- consolidated CLI forwarding and environment-key precedence;
- resume preserving completed model results.

No unit test starts a server, uses a GPU, or accesses the network.
