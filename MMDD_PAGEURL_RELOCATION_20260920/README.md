# page_url relocation — wdc200k (2026-09-20)

The WDC 200k joinability dataset put its `page_url` column on whichever side of
the query/target split the seeded shuffle happened to drop it. It was designed
to sit in the **query**, as the entity's own URL — the role the fabricated
`entity_url` column played before `scripts_old/strip_entity_url_column.py`
removed it. This directory holds the plan that moves it there.

## Scope

**Only `page_url`.** The schema.org `url`, `logo`, `link` and `image_url`
columns are left exactly where they are: those properties usually hold a nested
object (`{"datemodified": ..., "name": [...]}`) rather than a URL string, so
matching them by column name would misfire.

## The three edits

1. **Drop** every target whose join column is `page_url` (the join dies with the
   column, and the query/target context pools are disjoint, so no other shared
   column can replace it), and every target that would be left with at most one
   column once `page_url` is removed.
2. **Strip** `page_url` from the remaining targets and **append** it to the query
   tables that do not already carry it. The new column goes last and is
   deliberately *not* listed in `query_context_col_names` — the exact slot
   `entity_url` occupied before the strip — so a query row differs from its
   pre-strip state by one cell's name and value only.
3. **Delete** queries whose every positive target was dropped. A query with no
   positive scores as a retrieval failure for every model, so it is removed
   rather than kept as a negative.

## Numbers

| | before | after |
|---|---|---|
| data-lake targets | 70,967 | 54,819 |
| data lake (incl. raw stubs) | 209,602 | 193,454 |
| query tables | 70,403 | 54,277 |
| qrels pairs | 79,277 | 60,967 |

Drops break down as 7,559 (join column is `page_url`) + 8,589 (would be left
with ≤1 column). Of the 70,967 targets, 22,396 lose only the column and 32,423
are untouched.

Orphans by split — note the damage is **not** evenly spread, because the split is
assigned per `page_title` (whole schema.org type per split) and URL density
varies sharply by type:

| split | orphaned / total | lost |
|---|---|---|
| train | 12,741 / 53,151 | 24.0% |
| dev | 1,256 / 10,717 | 11.7% |
| test | 2,129 / 6,535 | 32.6% |

dev is mostly `Event`/`JobPosting`; test is mostly `CreativeWork`/`Restaurant`.
The dev:test ratio moves from 1.64:1 to 2.15:1.

## auto_check: reuse the half that did not change

`_query_recovery_auto_check_key_fields` hashes the *materialised query row*, so
a query row that gains a `page_url` cell needs a new call. Three things follow.

**The existing cache can be re-keyed, not discarded.** Every cached review was
taken while the query carried `entity_url`, so the stored key no longer matches.
`rekey_query_recovery_cache.py` recomputes each key with that one cell removed.
The key is reproduced exactly for 1,118,620 of 1,138,946 re-keyable records
(811,455/811,455 on the current v6 schema), and the re-keyed keys have zero
collisions on v6 — evidence that the fabricated `https://en.wikipedia.org/wiki/
wdc_<hash>` value never distinguished two reviews. Records whose key does not
recompute are copied byte-identical rather than guessed at.

**Only the moving half re-runs.** On the v6 cache (811,455 checks):

| bucket | checks | disposition |
|---|---|---|
| `page_url` already visible in the query row | 148,904 | reuse |
| `page_url` moves into the query row | 539,867 | **re-run** |
| the reviewed attribute *is* `page_url` | 122,684 | dropped with the target |

At query level: 22,038 re-run, 31,816 reuse. Measured throughput is ~21 rps with
both endpoints (18000 and 18001 both serve `Qwen3.5-9B`; `max_model_len` 8192, so
~1.9% of image-evidence calls overflow — image evidence is only 2.9% of
recoveries). Re-run volume lands around 40–54万 calls: **7–8 hours**.

**42 queries** end up with an all-empty `page_url` column (their row views only
cover source rows the target never materialised — rows whose join column is
empty). Their visible row and key are unchanged, so they hit the cache and cost
nothing.

## Files

| file | purpose |
|---|---|
| `generate_plan.py` | read-only; writes `lists/` and `plan.json` |
| `relocate_page_url.py` | the in-place transform; `--dry_run` validates without writing |
| `rekey_query_recovery_cache.py` | re-keys the auto-check cache; input untouched |
| `lists/` | the five id lists the two scripts consume |
| `plan.json` | the numbers above, machine-readable |

## Reproduce

```bash
python3 MMDD_PAGEURL_RELOCATION_20260920/generate_plan.py \
  --dataset_dir output_wdc_webtable_200000_qwen35_local_autocheck_v9

python3 MMDD_PAGEURL_RELOCATION_20260920/rekey_query_recovery_cache.py \
  --cache cache/wdc_webtable/query_recovery_auto_checks.jsonl \
  --out   cache/wdc_webtable/query_recovery_auto_checks.rekeyed.jsonl

python3 MMDD_PAGEURL_RELOCATION_20260920/relocate_page_url.py \
  --dataset_dir output_wdc_webtable_200000_qwen35_local_autocheck_v9 --dry_run
```

The transform copies `data_lake_tables/`, `query_tables/` and the manifest,
qrels, splits and stats into `backups/pageurl_relocation_<timestamp>/` before it
writes, and records what it did in `pageurl_relocation_report.json`.

## The upstream fix

This is a projection repair, not a cure. `_balanced_context_partition`
(`src/mmdd_dataset/joinability.py:147`) still treats `page_url` as an ordinary
column when it splits each source table's columns between the query and target
pools, and `explicit_join_column_policy: "seeded_random_non_entity_visible_column"`
still draws the shared join column from that same pool at random. A rebuild will
grow the same defect unless `page_url` is excluded from both.
