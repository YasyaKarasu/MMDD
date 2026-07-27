# MM Joinability Progress Display Design

## Goal

Restore visible progress during the pre-vLLM phase introduced by seeded random sampling and initial-batch material preparation.

## Design

- Wrap the seeded randomized JSON-file iterator with tqdm and label it `Reading randomized EntiTables JSON`; its total is the discovered JSON-file count.
- Wrap initial candidate material preparation with tqdm and label it `Preparing initial candidate materials`; its total is the initial batch table count.
- Update the material bar postfix with the cumulative number of eligible entities seen so a slow table still exposes useful state.
- Respect `--no_model_progress` for these bars as the pipeline's existing opt-out switch.
- Do not change iteration order, RNG use, fetching order, cache behavior, model-start gating, or replacement decisions.

## Verification

Unit tests replace tqdm with a spy and assert the descriptions, totals, and disabled behavior while continuing to exercise real iterators. Existing joinability tests and the full suite must remain green.
