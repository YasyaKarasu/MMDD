# MM Joinability Batched Material Preparation Design

## Goal

Keep global random table selection while restoring the original batched Wikipedia/material pipeline for the initial pool and every replacement round.

## Architecture

Candidate selection remains independent and globally priority sampled. Before evaluating any round, the replacement state machine invokes a batch-preparation callback with all tables in that round. The callback updates entity mappings in first-seen order, applies the existing global `max_entities` budget, finds only entities whose assets have not already been prepared, and sends that complete entity list through the original `build_bridge_assets` implementation.

The original implementation provides batched `get_pages`, batched `get_imageinfos`, and concurrent media downloads. Returned in-memory records are merged into the candidate evaluation context. Entities with no returned assets are still marked prepared so evaluation never falls back to per-entity network requests.

## Lifecycle

1. Globally sample and materialize the initial table batch.
2. Batch-prepare all new initial entities using the original asset pipeline.
3. Write the model-start marker and start vLLM.
4. Evaluate the initial batch using prepared assets only.
5. For each replacement round, batch-prepare that round's new entities, then evaluate while reusing the running models.

## Cleanup compatibility

Populate per-entity page and imageinfo dependency keys from the batched page/image data so discarded tables retain the same reference-aware cleanup behavior. Shared entities/assets remain protected. Sampling, 50% replacement probability, two-round limit, and final materialization remain unchanged.

## Verification

Tests assert one original batch-pipeline call per non-empty round, zero per-entity fetch calls during preparation/evaluation, no refetch for shared/prepared entities, model startup after initial batch preparation, batch preparation before every replacement evaluation, and preserved cleanup dependency keys.
