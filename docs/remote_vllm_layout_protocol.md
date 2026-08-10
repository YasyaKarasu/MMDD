# MMDD Remote vLLM Layout Control Protocol

Status: implementation contract for protocol version 1

Schema version: `mmdd-remote-vllm-layout-v1`

This document defines the control protocol between an MMDD dataset builder on
the local machine and a vLLM layout agent on a remote two-GPU machine. The two
implementations may be developed independently, but they must follow the wire
format, state transitions, safety ordering, and failure behavior defined here.

## 1. Goal and fixed topology

The remote machine has two independently usable GPUs:

- `primary_image`: always serves the image model.
- `switchable`: serves either the text model or a second image-model instance.

Version 1 supports exactly two layouts:

| Layout | `primary_image` | `switchable` |
| --- | --- | --- |
| `balanced` | image | text |
| `image_burst` | image | image |

The local machine decides the desired layout from durable workload state. The
remote machine owns all vLLM process lifecycle operations. The remote API never
accepts shell commands, model paths, arbitrary ports, environment variables, or
vLLM arguments from the local machine.

Normal layout changes must never stop or restart `primary_image`. Only the
`switchable` service changes role.

All inference traffic to `switchable` must pass through the local endpoint
scheduler governed by this protocol. Direct or third-party clients of that
endpoint are unsupported because they cannot participate in the local drain
barrier.

## 2. Non-goals

Protocol version 1 does not provide:

- arbitrary GPU placement or arbitrary model selection;
- more than one active local controller;
- scheduling across multiple remote hosts;
- automatic model downloading;
- result transport or job leasing between machines;
- cancellation of an in-flight inference request;
- remote execution of user-supplied commands.

## 3. Components and ownership

### 3.1 Local layout controller

The local controller:

1. reads the durable text and image demand;
2. holds the single active control lease;
3. withdraws the `switchable` endpoint from local routing;
4. waits until its local endpoint in-flight count reaches zero;
5. requests a remote layout;
6. waits for the remote operation and exact model health checks;
7. publishes the new local routing snapshot and concurrency budget.

There must be exactly one active local controller. It must hold an exclusive
local file lock for its entire active lifetime.

The recommended implementation runs the controller in the builder process, or
otherwise exposes a builder acknowledgement that proves the endpoint has been
withdrawn under the same lock used to lease endpoints. Merely editing an
endpoint file and sleeping is not a valid drain proof because a request can
already have leased the old endpoint without reaching the remote server yet.

### 3.2 Remote layout agent

The remote agent:

1. exposes the versioned control API;
2. holds an exclusive remote agent lock;
3. validates the controller lease and request ordering;
4. manages complete vLLM process groups;
5. checks remote request drain metrics when available;
6. verifies the exact served model ID before reporting readiness;
7. persists operation state atomically;
8. rolls back a failed switch when possible.

All model paths, GPU IDs, served model IDs, ports, API keys, and vLLM arguments
come from static remote configuration.

### 3.3 SSH transport

The remote agent must bind only to a remote loopback address. The local machine
reaches it through a persistent SSH local-forward tunnel. A typical mapping is:

| Purpose | Local address | Remote address |
| --- | --- | --- |
| control API | `127.0.0.1:18999` | `127.0.0.1:18999` |
| primary inference | `127.0.0.1:18000` | `127.0.0.1:8000` |
| switchable inference | `127.0.0.1:18001` | `127.0.0.1:8001` |

Ports are deployment configuration. Endpoint identity, not a hard-coded port,
is authoritative in the protocol.

### 3.4 WDC-priority handoff to EntiTables

WDC and EntiTables may be separate candidate controller processes, but they
must never hold active remote leases concurrently. A shared filesystem priority
directory provides the handoff outside the remote wire protocol:

1. WDC publishes `priority_requested` before acquiring its remote lease.
2. EntiTables atomically withdraws all remote routes under the same scheduler
   used to lease inference endpoints.
3. EntiTables stops layout reconciliation, waits for every local endpoint lease
   to drain, releases the remote controller lease, and only then acknowledges
   the matching WDC request generation and sequence.
4. WDC acquires the remote lease and runs its model stage.
5. WDC releases the lease and publishes `borrowable` after the model stage.

While WDC holds priority, EntiTables continues on its local GPU pool and leaves
remote worker threads idle. The remote agent lease remains the final mutual
exclusion authority. A missing, malformed, stale, or unacknowledged handoff
state fails closed; it never authorizes WDC to start while an EntiTables route
may still be in flight.

## 4. Security requirements

All control requests require both SSH transport authentication and a dedicated
control bearer token:

```text
Authorization: Bearer <layout-control-token>
Content-Type: application/json
Accept: application/json
```

The layout-control token must be distinct from the vLLM inference API key. Both
machines must read it from mode-`0600` files. It must never appear in command
arguments, URLs, JSON state files, logs, error messages, or process metadata.

The remote server must:

- bind to loopback only;
- reject request bodies larger than 16 KiB;
- reject unknown request fields;
- reject unknown layout names and endpoint identities;
- use constant-time bearer-token comparison;
- return `Cache-Control: no-store`;
- redact authorization headers and raw exceptions;
- reject any request field that attempts to supply a command, path, port, GPU,
  model ID, environment variable, or vLLM option.

TLS is not required inside the authenticated SSH tunnel. Deployments that expose
the control port beyond loopback must add TLS and network access controls, but
such exposure is outside this protocol.

## 5. Identifiers and ordering

The following identifiers are mandatory:

- `agent_boot_id`: random UUID generated on every remote agent start.
- `controller_id`: stable random UUID persisted by the local installation.
- `session_id`: random UUID generated on every local controller start.
- `lease_id`: random UUID returned by the remote agent after lease acquisition.
- `sequence`: unsigned 64-bit integer starting at 1 for each `session_id` and
  increasing for every layout request.
- `operation_id`: random UUID assigned by the remote agent to an accepted
  layout request.
- `instance_generation`: unsigned integer incremented whenever one remote vLLM
  service is restarted.

Wall-clock timestamps are informational only. Lease validity and timeouts must
use monotonic clocks. Ordering is determined by identifiers and `sequence`, not
timestamps.

The idempotency key for a layout request is:

```text
(controller_id, session_id, sequence)
```

For idempotency comparison, the semantic payload consists of
`schema_version`, `controller_id`, `session_id`, `sequence`,
`expected_agent_boot_id`, `expected_current_layout`, `desired_layout`, the
sorted `drained_endpoint_ids`, `local_routing_revision`, and `reason`.
`lease_id` is deliberately excluded because an expired lease can be reacquired
while an already accepted operation remains authoritative.

For an already observed key under a currently valid lease:

- the same semantic payload returns the existing operation;
- a different payload returns HTTP `409` with `idempotency_conflict`.

A lower sequence for the current session returns HTTP `409` with
`stale_sequence`.

## 6. Controller lease

Only a live lease holder may request a layout. The recommended lease TTL is 15
seconds, renewed every 5 seconds. The server must accept TTL values only in the
inclusive range 5 through 60 seconds.

### 6.1 Acquire or renew lease

`PUT /v1/lease`

Request:

```json
{
  "schema_version": "mmdd-remote-vllm-layout-v1",
  "controller_id": "203ea253-9163-4d86-b09d-d9c94664e61f",
  "session_id": "8a73594c-ab4c-4af4-b76c-34312cd616a4",
  "requested_ttl_seconds": 15
}
```

Successful response, HTTP `200`:

```json
{
  "schema_version": "mmdd-remote-vllm-layout-v1",
  "agent_boot_id": "e7e3c8de-fca4-4518-a329-37d08843a792",
  "lease_id": "35058535-c69f-48c2-9d58-949947ad55c8",
  "controller_id": "203ea253-9163-4d86-b09d-d9c94664e61f",
  "session_id": "8a73594c-ab4c-4af4-b76c-34312cd616a4",
  "ttl_seconds": 15,
  "server_time_unix_ms": 1785810000000
}
```

Acquisition succeeds when no lease exists, the prior lease has expired, or the
request exactly matches the current controller and session. Renewing an
unexpired matching lease preserves its `lease_id`. Acquiring after expiry
always returns a new `lease_id`, including for the same controller and session.
A different live controller or session receives HTTP `409` with
`lease_conflict`.

Lease expiry must not stop healthy inference services and must not initiate a
new layout. If expiry occurs during an accepted transition, the agent completes
that transition or its rollback, then refuses further transitions until a lease
is acquired.

### 6.2 Release lease

`POST /v1/lease/release`

Request:

```json
{
  "schema_version": "mmdd-remote-vllm-layout-v1",
  "controller_id": "203ea253-9163-4d86-b09d-d9c94664e61f",
  "session_id": "8a73594c-ab4c-4af4-b76c-34312cd616a4",
  "lease_id": "35058535-c69f-48c2-9d58-949947ad55c8"
}
```

The operation is idempotent. It returns HTTP `200` for a matching lease or an
already released lease. It must not change the current model layout.

## 7. Status API

`GET /v1/status`

This endpoint requires bearer authentication but not a controller lease.

Successful response, HTTP `200`:

```json
{
  "schema_version": "mmdd-remote-vllm-layout-v1",
  "agent_boot_id": "e7e3c8de-fca4-4518-a329-37d08843a792",
  "agent_state": "ready",
  "ready_layout": "balanced",
  "target_layout": "balanced",
  "active_operation_id": null,
  "status_revision": 17,
  "services": {
    "primary_image": {
      "endpoint_id": "primary_image",
      "role": "image",
      "state": "ready",
      "remote_port": 8000,
      "served_model_id": "Qwen3-VL-8B-Thinking",
      "instance_generation": 3,
      "max_inflight": 32
    },
    "switchable": {
      "endpoint_id": "switchable",
      "role": "text",
      "state": "ready",
      "remote_port": 8001,
      "served_model_id": "Qwen3.5-9B",
      "instance_generation": 8,
      "max_inflight": 32
    }
  },
  "server_time_unix_ms": 1785810000000
}
```

`agent_state` is one of:

- `starting`: the agent is reconciling its initial process state;
- `ready`: both services required by `ready_layout` are healthy;
- `transitioning`: a layout operation is active;
- `degraded`: at least one required service is unavailable;
- `stopping`: the agent itself is shutting down.

`ready_layout` is `balanced`, `image_burst`, or `null`. It names only a fully
verified layout. During a transition it remains the prior verified layout until
the target is ready. After an unrecoverable failure it is `null` if neither
complete layout is healthy.

`status_revision` increases after every externally visible state change.

The local controller must map `endpoint_id` to its configured local tunnel URL.
It must not construct a routing URL from an untrusted response field.

## 8. Layout request API

`PUT /v1/layout`

Request:

```json
{
  "schema_version": "mmdd-remote-vllm-layout-v1",
  "controller_id": "203ea253-9163-4d86-b09d-d9c94664e61f",
  "session_id": "8a73594c-ab4c-4af4-b76c-34312cd616a4",
  "lease_id": "35058535-c69f-48c2-9d58-949947ad55c8",
  "sequence": 4,
  "expected_agent_boot_id": "e7e3c8de-fca4-4518-a329-37d08843a792",
  "expected_current_layout": "balanced",
  "desired_layout": "image_burst",
  "drained_endpoint_ids": ["switchable"],
  "local_routing_revision": 29,
  "reason": "text queue complete; image jobs remain"
}
```

Validation rules:

- the lease must be live and match all three lease identifiers;
- `expected_agent_boot_id` must equal the current boot ID;
- `desired_layout` must be `balanced` or `image_burst`;
- `sequence` must satisfy the ordering rules in section 5;
- `reason` must be at most 256 UTF-8 bytes;
- changing layouts requires `drained_endpoint_ids` to contain exactly
  `switchable`;
- `expected_current_layout` must match `ready_layout` unless it is `null` on the
  first reconciliation request after lease acquisition;
- only one non-idempotent operation may run at a time.

Validation order is significant. After authentication, schema, body, live
lease, and boot-ID validation, the server first looks up the idempotency key. A
matching known semantic payload returns its existing operation without
re-evaluating sequence freshness, layout preconditions, or drain proof. A new
key is then checked for sequence freshness, layout preconditions, drain proof,
and transition conflicts. This ordering makes a retry valid after the original
operation has already changed `ready_layout`.

If the desired layout is already fully healthy, the server records the request
as an idempotent completed operation and returns HTTP `200`. A newly accepted
transition returns HTTP `202`. A different request during an active transition
returns HTTP `409` with `transition_in_progress`.

Layout execution must be asynchronous. The HTTP request handler returns after
durably accepting the operation and must remain able to process lease renewals,
status reads, and operation reads while model loading is in progress.

Response:

```json
{
  "schema_version": "mmdd-remote-vllm-layout-v1",
  "agent_boot_id": "e7e3c8de-fca4-4518-a329-37d08843a792",
  "operation": {
    "operation_id": "9fa003a6-8cc4-4b29-9b3c-f25127d68fd7",
    "controller_id": "203ea253-9163-4d86-b09d-d9c94664e61f",
    "session_id": "8a73594c-ab4c-4af4-b76c-34312cd616a4",
    "sequence": 4,
    "from_layout": "balanced",
    "desired_layout": "image_burst",
    "state": "accepted",
    "error_code": null,
    "error_message": null
  }
}
```

## 9. Operation API

`GET /v1/operations/{operation_id}`

Successful response shape:

```json
{
  "schema_version": "mmdd-remote-vllm-layout-v1",
  "agent_boot_id": "e7e3c8de-fca4-4518-a329-37d08843a792",
  "operation": {
    "operation_id": "9fa003a6-8cc4-4b29-9b3c-f25127d68fd7",
    "controller_id": "203ea253-9163-4d86-b09d-d9c94664e61f",
    "session_id": "8a73594c-ab4c-4af4-b76c-34312cd616a4",
    "sequence": 4,
    "from_layout": "balanced",
    "desired_layout": "image_burst",
    "resulting_layout": "image_burst",
    "state": "ready",
    "created_at_unix_ms": 1785810000000,
    "updated_at_unix_ms": 1785810120000,
    "error_code": null,
    "error_message": null,
    "services": {
      "primary_image": {
        "endpoint_id": "primary_image",
        "role": "image",
        "state": "ready",
        "remote_port": 8000,
        "served_model_id": "Qwen3-VL-8B-Thinking",
        "instance_generation": 3,
        "max_inflight": 32
      },
      "switchable": {
        "endpoint_id": "switchable",
        "role": "image",
        "state": "ready",
        "remote_port": 8001,
        "served_model_id": "Qwen3-VL-8B-Thinking",
        "instance_generation": 9,
        "max_inflight": 32
      }
    }
  }
}
```

`resulting_layout` and `services` are `null` until a terminal state. For
`rolled_back`, `resulting_layout` is the restored prior layout. For `failed`, it
is `null` unless a complete layout remains verified.

Operation `state` is one of:

- `accepted`;
- `remote_draining`;
- `stopping_switchable`;
- `starting_switchable`;
- `verifying_switchable`;
- `ready`;
- `rollback_starting`;
- `rolled_back`;
- `failed`.

Terminal states are `ready`, `rolled_back`, and `failed`.

An operation in `ready` must include the same service objects that would appear
in `GET /v1/status`. A `rolled_back` operation reports the restored layout. A
`failed` operation reports `error_code` and a sanitized `error_message` no
longer than 512 UTF-8 bytes.

Unknown operation IDs return HTTP `404` with `operation_not_found`.

Operation records and their semantic payload hashes must remain queryable across
lease expiry and local controller reconnects. Because layout changes are rare,
version 1 agents should retain them for the lifetime of the agent state store.

## 10. Error response

Every non-2xx response uses this shape:

```json
{
  "schema_version": "mmdd-remote-vllm-layout-v1",
  "agent_boot_id": "e7e3c8de-fca4-4518-a329-37d08843a792",
  "error": {
    "code": "stale_sequence",
    "message": "layout request sequence is older than the current session sequence",
    "retryable": false
  }
}
```

Required HTTP mappings:

| HTTP | Error code examples |
| --- | --- |
| `400` | `invalid_json`, `invalid_request` |
| `401` | `unauthorized` |
| `404` | `operation_not_found` |
| `409` | `lease_conflict`, `lease_expired`, `stale_sequence`, `idempotency_conflict`, `boot_id_mismatch`, `layout_precondition_failed`, `transition_in_progress` |
| `413` | `request_too_large` |
| `422` | `unsupported_layout`, `invalid_drain_proof` |
| `426` | `unsupported_schema_version` |
| `500` | `agent_internal_error` |
| `503` | `agent_not_ready` |

Raw exceptions and response bodies from vLLM must never be returned.

## 11. Required transition ordering

### 11.1 `balanced` to `image_burst`

Local side:

1. Confirm durable text unfinished count is zero and image unfinished count is
   greater than zero.
2. Under the endpoint-selection lock, withdraw `switchable` from text routing.
3. Wait until the local `switchable` in-flight count is zero.
4. Send the layout request and continue renewing the lease.
5. Poll the operation until terminal.
6. On `ready`, fetch status and verify:
   - the boot ID is unchanged;
   - `ready_layout` is `image_burst`;
   - both endpoint states are `ready`;
   - both endpoint roles are `image`;
   - both exact served model IDs match the configured image model.
7. Atomically add `switchable` to image routing.
8. Set effective image capacity to the sum of both endpoint capacities.

Remote side:

1. Recheck the lease, boot ID, layout precondition, and drain proof.
2. If vLLM request metrics are available, require zero running and zero waiting
   requests on `switchable` for two consecutive polls.
3. Stop the complete switchable text process group, using bounded `SIGTERM` then
   `SIGKILL` cleanup.
4. Start the statically configured switchable image service.
5. Require its `/v1/models` response to contain the exact image model ID.
6. Increment `instance_generation` and publish the ready operation atomically.

### 11.2 `image_burst` to `balanced`

Local side:

1. As soon as durable text unfinished count becomes greater than zero, withdraw
   `switchable` from image routing and reduce image capacity to the primary
   endpoint capacity.
2. Wait until the local `switchable` in-flight count is zero.
3. Request `balanced` and continue serving image work through `primary_image`.
4. On `ready`, verify the exact text model ID on `switchable`.
5. Atomically add `switchable` to text routing and enable its text capacity.

Remote side follows the same sequence as section 11.1, with the image and text
roles reversed.

The local side must never publish the new role before the remote ready response
and independent `/v1/models` verification through the inference tunnel.

## 12. Remote rollback policy

If starting or verifying the target switchable role fails, the remote agent
must attempt to restore the prior role:

- `balanced` to `image_burst` failure: restart switchable text.
- `image_burst` to `balanced` failure: restart switchable image.

If rollback succeeds, the operation terminates as `rolled_back` and status
returns to the prior ready layout. The local side keeps `switchable` withdrawn
until it observes and verifies the rollback, then may republish the old role.

If rollback fails, the operation terminates as `failed`, `agent_state` becomes
`degraded`, and `ready_layout` becomes `null`. `primary_image` must remain
available whenever possible. The local side may route only independently
verified healthy endpoints.

## 13. Workload-to-layout policy

The protocol carries desired state; it does not infer workload. The local
implementation must use durable queue state rather than progress display text or
ETA estimates.

For the WDC model job database, unfinished means any status other than
`success` or `terminal`, including `pending`, `retryable`, and `leased`.

Required policy:

```text
if text_unfinished > 0:
    desired_layout = balanced
elif text_unfinished == 0 and image_unfinished > 0:
    desired_layout = image_burst
else:
    keep the current healthy layout
```

The newest authoritative model job-set pair is the demand source. A new job-set
generation with text work must cause a return to `balanced`.

Entering `image_burst` should require the condition to remain true for a
configurable stability interval, recommended 10 seconds. Returning to
`balanced` has text priority and must not wait for that interval. Implementations
may skip `image_burst` when the remaining image backlog is too small to recover
the model switch cost, but this optimization must not affect protocol behavior.

## 14. Local routing and concurrency contract

Adding a second endpoint does not by itself double builder concurrency. The
local implementation must enforce capacity per endpoint and derive the total
worker budget from the current healthy routing snapshot.

For each endpoint:

- `max_inflight` is an upper bound, not a target;
- no scheduler may lease more concurrent calls than `max_inflight`;
- selection among endpoints with available capacity uses least-inflight with a
  round-robin tie break;
- withdrawing an endpoint prevents new leases immediately;
- existing leases decrement the in-flight count in a `finally` block.

Effective modality capacity is:

```text
sum(endpoint.max_inflight for each healthy routed endpoint of the modality)
```

With two equal image endpoints of capacity 32, `image_burst` therefore exposes
64 image workers and `balanced` exposes 32. If capacities differ, the total is
their sum rather than a literal factor of two.

The local routing snapshot should be persisted atomically using this separate
local-only schema:

```json
{
  "schema_version": "mmdd-model-routing-v1",
  "controller_id": "203ea253-9163-4d86-b09d-d9c94664e61f",
  "session_id": "8a73594c-ab4c-4af4-b76c-34312cd616a4",
  "routing_revision": 30,
  "agent_boot_id": "e7e3c8de-fca4-4518-a329-37d08843a792",
  "remote_operation_id": "9fa003a6-8cc4-4b29-9b3c-f25127d68fd7",
  "layout": "image_burst",
  "text": [],
  "image": [
    {
      "endpoint_id": "primary_image",
      "base_url": "http://127.0.0.1:18000/v1",
      "served_model_id": "Qwen3-VL-8B-Thinking",
      "instance_generation": 3,
      "max_inflight": 32
    },
    {
      "endpoint_id": "switchable",
      "base_url": "http://127.0.0.1:18001/v1",
      "served_model_id": "Qwen3-VL-8B-Thinking",
      "instance_generation": 9,
      "max_inflight": 32
    }
  ]
}
```

This file is not sent to the remote agent. It is the local handoff between the
controller and inference scheduler. When dynamic routing is configured, it is
authoritative: an empty modality list means no endpoint is currently routable.
The builder must not silently fall back to a static URL.

## 15. Health and readiness

Remote readiness requires all of the following:

- the managed process is alive;
- TCP connection to the configured loopback port succeeds;
- `GET /v1/models` returns HTTP `200`;
- the response contains the exact configured served model ID;
- the endpoint remains healthy for two consecutive polls;
- the process belongs to the expected agent-managed process group and service
  instance generation.

The local controller must independently repeat the model-ID check through the
local SSH inference tunnel before publishing a route. A successful control-plane
status alone is insufficient because the inference tunnel may be broken or
misrouted.

The remote agent should supervise the active layout after readiness. If a
service dies unexpectedly, it marks the service unavailable, increments status
revision, and attempts a bounded restart of the same role. It must not change
roles without a valid layout request.

## 16. Failure and recovery behavior

### 16.1 Control connection lost before request acceptance

The local side keeps the prior routing. No remote state is assumed.

### 16.2 Control connection lost after request submission

The local side has already withdrawn `switchable`, so it keeps that endpoint
withdrawn. After reconnecting it fetches status. If it did not receive an
`operation_id`, it repeats `PUT /v1/layout` with the same idempotency key and
semantic payload, using the current valid `lease_id`; the remote agent returns
the existing operation. It then reconciles from that operation and status. It
must not submit the opposite layout until the first operation is terminal.

### 16.3 SSH inference tunnel failure

The local health monitor withdraws unreachable endpoints. It does not request a
role change solely because a tunnel is down. After tunnel recovery it verifies
model identity before republishing.

### 16.4 Remote agent restart

A restart changes `agent_boot_id`, invalidates the old lease, and makes all old
nonterminal operations non-authoritative. The local side withdraws
`switchable`, acquires a new lease, fetches current status, verifies services,
and submits a reconciliation request with the new boot ID.

### 16.5 Local controller restart

The controller preserves `controller_id`, generates a new `session_id`, and
waits for the old lease to expire before acquiring a new lease. It reads the
durable workload, remote status, and local routing snapshot, then reconciles.

### 16.6 Stale or malformed state

Both implementations fail closed. Stale acknowledgements, lower sequences,
unknown boot IDs, invalid JSON, and model-ID mismatches must never make an
endpoint routable.

## 17. Remote persistence

The remote agent must atomically persist, without secrets:

- `agent_boot_id` for the current process lifetime;
- current and target layouts;
- status revision;
- active lease identity and monotonic expiry metadata for the current boot;
- highest sequence and canonical payload hash per active session;
- operation records;
- service role, PID/process-group identity, and instance generation.

State files must be mode `0600`. The agent must use an exclusive lock to prevent
two agents from managing the same GPUs. On startup it must validate managed
process identity before signaling a recorded PID or process group; a recycled
PID must never be killed. Persisted monotonic lease values are diagnostic only
after an agent restart; every old lease is invalid when `agent_boot_id` changes.

## 18. Compatibility

Every request and response contains `schema_version`. Version 1 clients must
reject a different response version. Version 1 servers return HTTP `426` for an
unsupported request version.

New optional response fields may be added without changing the schema version.
Removing a field, changing field meaning, adding a required request field, or
adding a layout requires a new schema version.

## 19. Required acceptance tests

Each side must include unit tests, and the combined deployment must pass the
following end-to-end cases:

1. Start from no services and reconcile to `balanced`.
2. Switch `balanced` to `image_burst` with no request loss.
3. Switch `image_burst` to `balanced` while primary image inference continues.
4. Verify image capacity changes from one endpoint capacity to the sum of two.
5. Repeat an identical layout request and receive the same operation.
6. Reject the same idempotency key with a different payload.
7. Reject a stale sequence.
8. Reject a second controller while the first lease is live.
9. Expire a lease without stopping healthy inference services.
10. Lose the control connection after submission and reconcile correctly.
11. Restart the remote agent and reject the old boot ID and lease.
12. Fail target model startup and successfully roll back the prior role.
13. Fail both target startup and rollback and enter `degraded` without stopping
    a healthy primary image service.
14. Return a wrong served model ID and prove the local route is not published.
15. Withdraw `switchable` under concurrent load and prove no new request leases
    it after the drain starts.
16. Reintroduce text jobs in a new job-set generation and return to `balanced`.
17. Send invalid JSON, an oversized body, an invalid bearer token, an unknown
    layout, and command-like fields; verify safe rejection and redacted logs.

## 20. Implementation split

The remote-machine implementation owns:

- the authenticated loopback HTTP agent;
- lease, status, operation, idempotency, and persistence logic;
- static service configuration;
- vLLM process-group lifecycle, metrics drain check, health verification, and
  rollback;
- remote unit tests for the API and state machine.

The local-machine implementation owns:

- the persistent SSH tunnel and its health monitoring;
- durable workload-to-layout policy;
- lease renewal and layout API client;
- endpoint withdrawal under the endpoint-selection lock and local drain proof;
- exact model verification through inference tunnels;
- atomic local routing snapshots;
- per-endpoint capacity enforcement and dynamic modality worker budgets;
- local recovery logic and unit/integration tests.

Neither implementation may infer that a transition succeeded from elapsed time.
Only a matching terminal operation, matching `agent_boot_id`, exact service
roles, and independent model health checks authorize the new route.
