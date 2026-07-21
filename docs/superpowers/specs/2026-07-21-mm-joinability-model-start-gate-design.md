# MM Joinability Model Start Gate Design

## Goal

Restore the dynamic-vLLM phase boundary: the initial randomly selected source-table batch must finish Wikipedia/material preparation before the builder signals the runner to load vLLM models.

## Data flow

1. Fill the initial source-table slots from the seeded random candidate iterator.
2. Prepare entity records and fetch/cache bridge materials for the entire initial batch without issuing model requests.
3. Write `model_start.json`, wait for `model_ready.json`, and create the local model extractor.
4. Evaluate the prepared initial batch and apply the existing 50%/two-round replacement policy.
5. Prepare and evaluate later replacement batches while the already-started model services remain alive.

If the initial batch is empty, write a zero-work start marker so the dynamic runner exits without loading models.

## Compatibility and scope

- Keep seeded random ordering, replacement probability, replacement-round limits, cache cleanup, and final materialization unchanged.
- Keep `evaluate_candidate_batch` usable by existing callers; initial preparation may be repeated logically, but cached entity/assets prevent duplicate downloads.
- Add one lifecycle callback to `run_replacement_rounds` that runs exactly once after initial slots are filled and before their evaluation.

## Verification

- A focused unit test records lifecycle events and proves all initial material preparation completes before the model-start callback and evaluation.
- The test also proves replacement preparation occurs after the one-time start callback and does not start models again.
- Run the joinability test files and then the complete test suite.
