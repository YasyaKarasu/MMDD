# Dynamic Runner Signal Grace Design

## Problem

`run_mm_joinability_dynamic_vllm.py` starts the dataset builder in a separate
process group. Its signal handler forwards `SIGINT`, `SIGTERM`, or `SIGHUP` to
the builder and raises `ForwardedSignal`. The `finally` block then immediately
calls `stop_process()` for the same builder, which sends `SIGTERM`. A builder
handling the first signal can therefore be terminated while the image
scheduler is draining completed futures and releasing execution-owned job and
URL leases. The retry6 gate reproduced this as 64 live job leases and 16 live
URL claims after a normal forwarded `SIGINT`.

## Decision

After catching `ForwardedSignal`, the runner will give the already-signalled
builder a bounded grace period to exit on its own. The wait is builder-only:
all live process groups have already received the original signal, and the
existing `finally` cleanup remains responsible for any process still alive.
If the builder exits during the grace period, `stop_process()` observes it as
dead and sends no second signal. If it does not exit, the existing
`SIGTERM`/timeout/`SIGKILL` fallback remains unchanged.

The grace period is exposed as `--forwarded_signal_grace_seconds`, defaults to
30 seconds, and must be non-negative. Zero preserves immediate fallback for
operators that explicitly need it. A second signal may interrupt the grace
wait and proceed to the existing best-effort cleanup.

## Alternatives Rejected

1. Keeping the runner signal handler active without raising would require a
   larger state-machine rewrite across marker and server waits.
2. Reclaiming apparently dead owners during resume cannot safely distinguish a
   dead execution from a concurrent live execution and would weaken fencing.
3. Increasing lease time does not address the immediate-resume requirement.

## Error And Exit Semantics

- The runner still returns `128 + signum` for a forwarded signal.
- A grace timeout is not a new error; it hands control to existing cleanup.
- Cleanup warnings and process-stop behavior remain unchanged.
- The runner never edits job leases, URL claims, outcomes, or transport
  attempts itself.

## Verification

Tests must prove with a real child process group that a forwarded `SIGINT`
allows the child signal handler to finish delayed cleanup and exit before any
fallback signal. Main-level tests must prove the builder wait occurs before
best-effort stop, an already exited builder receives no second signal, timeout
falls back to existing cleanup, zero/negative CLI behavior is explicit, and
the first forwarded signal still produces the same exit code. Existing dynamic
runner, image scheduler, pipeline, and full WDC regressions must remain green.
