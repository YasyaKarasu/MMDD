# WDC ETA Advisory Gate Design

## Decision

URL-stage ETA point-prediction accuracy is advisory evidence. It is always
measured and reported, but it does not determine whether a dataset correctness
gate passes. URL telemetry integrity remains a hard gate.

This is a gate-policy change only. It does not modify the
`wdc200k-url-telemetry-v2` schema, estimator, reporter, page or image
scheduler, network policy, dataset pipeline, or any persisted artifact.
Where older design, plan, or report text requires an ETA factor `<= 2.0` for
acceptance, this design supersedes only that decision rule. All associated
measurement, retention, integrity, and reporting requirements remain in force.

The authoritative scale report,
`docs/superpowers/reports/2026-07-17-wdc-200k-scale-validation.md`, must be
updated to implement this policy before any new gate decision is made. That
update must replace, rather than supplement, its ETA `<= 2.0` acceptance rules
for the 100-table gate, the 1,000-table gate, and formal-launch authorization.
Until the report has been updated, it cannot issue a new `PASS`, `FAIL`, or
launch decision under this policy.

The read-only deterministic validator and its focused tests specified below
must also be implemented and verified before that authoritative report makes
any new gate decision. If the validator or its tests are absent, incomplete,
or not verified, the required evidence is `NOT_RUN_OR_INCOMPLETE` and the gate
decision is `BLOCKED`.

## Background

The fresh gate produced these maximum symmetric final-half ETA factors:

- page: `3.9951699758`;
- image: `7.5741343298`.

The values are valid observations, not telemetry corruption. An online point
estimate can use only the event prefix available at publication time. It
cannot causally predict which future request will become a network straggler
or how a host-specific tail will resolve after the sample. A hard per-sample
accuracy bound therefore conflates unpredictable future network timing with
dataset correctness.

The ETA remains operationally useful. It exposes drain behavior, queue and
in-flight state, commit delay, overflow, fallback selection, and large tail
errors. Those observations must stay exact and visible even when the point
estimate is inaccurate.

## Hard Gates

The following remain mandatory pass conditions. Any failure is a hard failure
regardless of the reported ETA factors.

### Dataset correctness

- Selected, validated, emitted, and source-row counts match their declared
  scale and exact source authorities.
- Dataset and shard schemas validate.
- The source `image` column is removed as required by the pipeline contract.
- Canonical dataset contents and stage completion semantics are unchanged.

### Artifact authority

- Every artifact that the pipeline must have published through the specific
  gate's declared stage boundary, including its manifests, registries, and
  shards, has the expected path, record count, byte count, SHA-256, input
  identity, and upstream fingerprint.
- Registry, manifest, and artifact references agree exactly.
- Missing, extra, truncated, or checksum-invalid artifacts fail the gate.
- Artifacts belonging only to stages after the declared boundary are
  `NOT APPLICABLE`, not missing. In particular, downstream network, model, and
  final-dataset artifacts are `NOT APPLICABLE` at the 10,000-table
  structural-only gate.

### Resume and transport authority

- For every planned interruption, allow the process to stop and, before
  launching the resume, copy the current `progress.json` to a gate-specific
  evidence directory as `progress-before-resume.json`. The evidence copy is
  made read-only, and the report records the SHA-256 of both files at copy time
  and their exact equality. This file is preserved pipeline-state evidence,
  not redirected or reconstructed log output.
- Evidence collection only reads `progress.json`: it must never rename, edit,
  replace, or otherwise rewrite that pipeline state file. The resumed pipeline
  may continue its normal guarded atomic publications to the original path.
- A resume validator restores both files canonically and compares each
  pre-resume stage's sample list with the equal-length prefix of the final
  stage sample list, object by object. For each stage, the report records the
  prefix length, SHA-256 of the canonically serialized prefix, and comparison
  result. A missing object, changed value or type, reordering, or inserted
  object inside the prefix fails the gate.
- Duplicate physical requests, terminal replays, unfinished transport
  attempts, and blocked durable replays are all zero.
- Durable outcomes are not replayed or reclassified.
- No ETA sample is deleted, compacted, rewritten, or discretionarily excluded
  to improve the advisory result. Only the existing canonical v2 eligibility
  and exclusion rules apply.

### Disk safety

- The configured reserve remains `107374182400` bytes (100 GiB).
- All output, work, and cache guard checks succeed, and the minimum observed
  free bytes recorded for each configured progress root is at least the
  reserve.
- Guard failures remain fail-closed and preserve the prior durable state.

This evidence establishes the result at every guard and progress observation.
It does not claim that no unobserved instantaneous dip occurred between those
observations.

### Telemetry integrity

- Progress and URL telemetry are canonical, bounded, append-only, and
  restorable under their existing limits.
- Histograms, topology, epochs, baselines, clocks, overflow counters, and ETA
  components validate and are recomputed rather than trusted from persisted
  scalar predictions.
- Manifest `url_completion`, registry counters, and independently restored
  progress telemetry agree exactly.
- Every eligible and excluded sample is counted under the existing v2 rules.
- Any malformed, forged, noncanonical, oversized, nonmonotonic, or
  nonrecomputable telemetry fails the gate.
- Fresh URL-stage hard evidence for the 100-table and 1,000-table gates must be
  native `wdc200k-url-telemetry-v2` from the first retained sample of each
  stage. A completed v1 stage remains readable for compatibility, but it
  cannot satisfy fresh telemetry-integrity evidence for either gate.

These checks preserve the distinction between an honest but inaccurate
forecast and invalid evidence. Only the former is advisory.

## ETA Advisory

For every completed page and image URL stage, the report always records:

- `eligible_final_half_samples`;
- `excluded_final_half_samples`;
- `max_symmetric_eta_factor`;
- the worst eligible sample, including stage, execution epoch, completed and
  total units, logical timestamp, predicted remaining seconds, actual
  remaining seconds, symmetric factor, fallback, topology fields, and
  overflow counters. If canonical recomputation yields zero eligible samples,
  the worst-sample field is explicitly `NONE` with reason
  `ZERO_ELIGIBLE_SAMPLES`; it is never omitted.

There is no advisory pass threshold. In particular, `2.0` is no longer a gate
boundary. The observed value is reported exactly as recomputed.

The operator must not improve the presentation by rewriting a prediction,
hiding the worst sample, dropping an eligible sample, changing eligibility,
substituting a different completion time, or labeling a computable value
`unknown`. A null maximum is valid only when the canonical recomputation has
zero eligible samples; the eligible and excluded counts must make that reason
explicit. A large factor is reported as a large factor.

### Deterministic advisory validator

The required future implementation adds the read-only command-line validator
`scripts/validate_wdc_eta_gate.py`, invoked as
`--progress <final-progress.json>` and, for an interruption gate,
`--before-resume <evidence/progress-before-resume.json>`. It must not introduce
a v3 schema, change the v2 estimator, or alter pipeline publication or restore
behavior. The validator hashes the exact input bytes with SHA-256, restores
the canonical samples and `completed_at` using the existing v2 validation
rules, and fails closed if restoration or recomputation fails. With
`--before-resume`, it performs the per-stage object-prefix comparison defined
under Resume and transport authority without changing either input.

For both `pages` and `images`, the validator enumerates every retained
final-half sample, where `completed_units * 2 >= total_units`. Each sample is
classified under the existing v2 eligibility rules and includes its original
zero-based position in the stage sample list. Eligible records include the
recomputed actual remaining seconds and symmetric factor. They are sorted by
factor descending and then by original zero-based sample position ascending.
The worst eligible record is the first record in that sorted eligible list.
Excluded records retain original sample order and include the canonical
exclusion reason. Every final-half sample occurs exactly once across those two
lists.

The validator writes one deterministic JSON object to stdout containing the
final input `progress.json` SHA-256 and, for each URL stage, the eligible and
excluded counts, the complete eligible and excluded record lists, the maximum
factor, and the worst eligible record. With zero eligible samples, the maximum
and `worst_record` are `null`, and `worst_reason` is
`ZERO_ELIGIBLE_SAMPLES`. When `--before-resume` is present, the object also
contains that input's SHA-256 and each stage's prefix length, canonical-prefix
SHA-256, and comparison result.
Successful validation exits zero; malformed or noncanonical input produces a
machine-testable JSON error on stdout and exits nonzero. The implementation
must have focused tests for restoration failures, enumeration, arithmetic,
ordering and tie-breaking, zero-eligible output, stable JSON, input SHA,
read-only operation, and object-prefix comparison.

Each gate report records the exact validator command and its complete JSON
result. The report's advisory values must come from that result, not from a
hand-selected or separately calculated sample.

## Scale Decisions

### 100-table full gate

Run the complete pipeline on fresh roots with one normal interruption and an
identical resume. The gate passes only when every hard dataset, artifact,
resume, transport, disk, and telemetry-integrity condition above passes.
Page and image telemetry must be native v2 from each stage's first retained
sample, and the required pre-resume evidence copy and prefix comparison must
be present. Record the deterministic validator command and JSON result,
including both ETA advisories and worst samples, without applying an accuracy
threshold. A hard pass is required before the 1,000-table gate proceeds.

### 1,000-table full gate

Run only after the 100-table hard gate passes. Require the same hard
correctness, resume, transport, reserve, artifact, and telemetry-integrity
conditions at 1,000 tables, including native-v2 telemetry from the first
retained page and image sample and the pre-resume evidence copy and prefix
comparison. Record peak RSS, its comparison with the 100-table run, and
separate output/work/cache growth as existing scale evidence. Record the exact
validator command and JSON result, and report the ETA advisory without using
it to change the hard decision. This hard gate must pass before the
10,000-table structural/preflight gate proceeds.

### 10,000-table structural/preflight gate

Run only after the 1,000-table hard gate passes. Run the full corpus with
`--max_source_tables 10000 --stop_after structural` on fresh roots. It
performs no network or GPU work. Require exactly 10,000 selected and validated
tables, exact source-row preservation, valid structural artifacts and
checksums, measured peak RSS, the 100 GiB reserve, and an explicit
determination that projected network work fits the available disk budget. URL
ETA is not produced because no URL stage runs; this is a declared stage
boundary, not suppression of an observed advisory. Network, model, and
final-dataset artifacts beyond the structural boundary are explicitly
`NOT APPLICABLE`.

### Formal 200K launch

Formal launch is authorized only when:

1. the 100-table full gate has a hard `PASS`;
2. the 1,000-table full gate has a hard `PASS`;
3. the 10,000-table structural/preflight gate has a hard `PASS` and confirms
   the live reserve can support projected network work;
4. fresh verification passes at the exact launch commit; and
5. the command, commit, roots, tmux session, timestamps, and monitoring record
   are captured as required by the scale report.

ETA advisory values neither authorize a launch nor veto one. They accompany
the decision as operational evidence. They cannot compensate for a failed
hard gate, and a high factor cannot be described as a dataset correctness
failure when all telemetry-integrity checks pass.

During the formal run, any hard integrity, correctness, replay, checksum, or
reserve failure stops the run and blocks acceptance. ETA outliers remain
visible advisories in the final report.

### Existing fresh-v2 evidence limitation

The existing fresh-v2 run that produced the page and image factors in this
design did not preserve a complete `progress-before-resume.json`. It did
record canonical page- and image-sample digests at the interruption cutoff and
confirmed those recorded digests after resume. The current report can prove
only that recorded digest equality; it cannot retroactively prove the full
object-by-object prefix comparison defined here. Reports must state that
limitation without upgrading it to stronger evidence. Every future planned
interruption gate must preserve the full read-only evidence copy before
resume.

## Failure Classification

Every gate report assigns one of these classifications:

- `HARD_DATASET_FAILURE`: counts, rows, schema, image-column removal, or
  canonical dataset contents are wrong.
- `HARD_ARTIFACT_FAILURE`: a manifest, registry, artifact hash, byte count,
  record count, identity, or reference is invalid.
- `HARD_RESUME_TRANSPORT_FAILURE`: prior samples changed, or duplicate,
  replay, unfinished, or blocked attempt counters are nonzero.
- `HARD_DISK_FAILURE`: an observed guard or per-root minimum violates the
  100 GiB reserve, or a guarded write does not fail closed.
- `HARD_TELEMETRY_INTEGRITY_FAILURE`: telemetry is noncanonical, unbounded,
  missing, forged, inconsistent, nonmonotonic, or cannot be recomputed.
- `ETA_ADVISORY_RECORDED`: telemetry is valid and its exact ETA accuracy
  statistics are reported. Notable tail error may be described in operational
  notes, without a numeric pass threshold or a hard-failure effect.
- `NOT_RUN_OR_INCOMPLETE`: required hard evidence was not produced. The gate
  is `BLOCKED`, never inferred to pass from partial output.

Multiple hard classifications may apply. `ETA_ADVISORY_RECORDED` accompanies
every completed URL-stage report, whether the hard decision is `PASS` or
`FAIL`, but it never replaces a hard classification.

## Report Format

Each scale section uses this structure:

```text
Hard gate decision: PASS | FAIL | BLOCKED
Hard failure classifications: [exact classifications, or NONE]
Run identity: commit, commands, roots, input hashes, timestamps
Dataset evidence: counts, rows, schema, image-column removal
Artifact evidence: manifest/registry/artifact hashes, bytes, records
Resume/transport evidence: prior-sample identity and anomaly counters
  pre-resume copy path and SHA, per-stage prefix length/digest/comparison
Disk evidence: guard results; per-root start/peak/end and minimum observed free
  bytes; 100 GiB reserve
Telemetry integrity: schema, bounds, canonical restore/recompute result
ETA advisory:
  validator: exact command, input progress SHA, complete JSON result
  page: eligible, excluded, max factor, worst sample or NONE+reason
  image: eligible, excluded, max factor, worst sample or NONE+reason
Operational notes: stragglers, host-tail observations, RSS, duration
```

All numeric fields use observed or independently recomputed values. A field is
`NOT RUN` only when its required stage did not run. A field is `NOT APPLICABLE`
only when the approved gate definition omits that stage, such as URL ETA in
the 10,000-table structural-only gate.

## Non-Goals and Rejected Alternatives

This decision does not relax dataset quality, artifact validation, transport
deduplication, replay rules, disk reserve, concurrency, retries, deadlines,
host limits, URL policy, or any other network behavior.

The following alternatives are rejected:

- **Hard per-sample symmetric factor `<= 2.0`.** An online point estimator
  cannot causally know a future network straggler or host-tail outcome. Keeping
  this as a correctness condition would reject valid datasets based on
  irreducible future timing.
- **A heavyweight probabilistic tail model.** Such a model would require a
  calibrated distribution, additional state and validation, and new operating
  assumptions. It still could not guarantee a deterministic per-sample bound
  and is unnecessary for establishing dataset correctness.
- **Sample suppression or cosmetic relabeling.** Dropping, hiding, rewriting,
  or marking computable outliers unknown would corrupt the evidence rather
  than improve the estimator.

## Compatibility

This design itself changes no production code or tests. Its required future
implementation is limited to the read-only deterministic validator and its
focused tests, which must be implemented and verified before the authoritative
scale report makes any new gate decision. Their absence or incomplete
verification is `NOT_RUN_OR_INCOMPLETE` and leaves the gate `BLOCKED`. The
implementation must not change the v2 schema, estimator formula, histogram
codec, bounds, epoch rules, completion authority, manifest fields, or pipeline
behavior.

Completed-v1 telemetry remains readable under the existing compatibility
rules, but it is not fresh native-v2 gate evidence. Persisted telemetry is
never rewritten to adopt this policy. The authoritative scale report must be
updated before any new decision and must replace its old 100-table,
1,000-table, and formal-launch ETA `<= 2.0` rules with the hard/advisory split
defined here.
