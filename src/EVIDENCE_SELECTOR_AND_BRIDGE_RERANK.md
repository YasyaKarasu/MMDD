# Natural-evidence selector and bridge-first reranking

Two Stage-2 components were merged here from the R7 fresh-AB experiment package
(`audit/MMDD_R7_FRESH_AB_C30_D098_PACKAGE`). Both are now part of the tracked Stage-2 implementation
and reuse the existing reader, feature cache and candidate head instead of duplicating them.

| Component | Module | Replaces |
|---|---|---|
| Natural-evidence selector arm (`NAT_E`) | `natural_evidence.py` + `column_r2_training.py` | the R7 `reader.py` / `features.py` / `heads.py` / `plans.py` glue |
| Bridge-first reranking | `bridge_first_rerank.py` | the `LEXICO_IDF_ABLATION_V1` scorer |
| Matched-original-row bridge score | `matched_row_score.py` | the R7 `d_matching.py` and its typed key rules |

## 1. Natural retrieved evidence as a reader condition

Evidence keys in the frozen column inputs are not interchangeable:

| Key | Meaning | Source |
|---|---|---|
| `O-O` | oracle evidence | `oracle.select_oracle_evidence` over the qrels positives, balanced across text/image |
| `O-R` | natural retrieval evidence | the R1 stage's frozen `retrieval_paths`, top 4 evidence paths per target |
| `NAT_E` | natural retrieval evidence | a Stage-1 export supplied at build time (this round's re-inference) |

So `NAT_E` is not the first natural-evidence condition: `O-R` already is one, and it is what the
non-witness arms default to. What `NAT_E` adds is a first-class way to point the selector at a
*chosen* Stage-1 export instead of R1's frozen retrieval, with its own provenance receipt and its
own dev reporting subsets. An empty retrieval stays empty; nothing is synthesised or oracle-derived.

Augment the frozen column inputs once:

```python
from mmdd_stage2.natural_evidence import build_inputs

manifest = build_inputs(r1, output, stage1_records, splits=("train", "dev", "test"))
```

This writes `NAT_E/COLUMN_INPUTS.<split>.jsonl` (the frozen inputs plus an `NAT_E` evidence key and
a `natural_partition` label) and `NAT_E/MANIFEST.json` with per-split SHA-256 and the coverage
funnel: pairs, pairs with non-empty evidence, evidence items, fit/holdout pairs and groups. The
`NAT_E` key is added next to the existing `O-O`/`O-R` keys, so witness arms keep resolving.

Cache reader features for the new condition and train both arms from one random initialisation:

```bash
# 1. write NAT_E/COLUMN_INPUTS.<split>.jsonl + MANIFEST.json from the current Stage-1 export
python src/run_stage2_columns_r2.py build-natural-evidence --stage1-dir <stage1-export>

# 2. enumerate reader jobs: [] plus the natural evidence, train views 0/1 and dev/test view 0
python src/run_stage2_columns_r2.py prepare-jobs --kind nat_e
python src/run_stage2_columns_r2.py prepare-jobs --kind nat_e_test

# 3. cache the frozen reader states for those jobs (one shard per device)
python src/run_stage2_columns_r2.py reader \
  --jobs <output>/JOBS/nat_e.jsonl.gz --shard 0 --shards 2 --device cuda:0
```

```bash
# 4. the controlled comparison: same seed, same fixed epoch, dev monitored not selected
python src/run_stage2_columns_r2.py train --arm PRIOR --seed 13 \
  --fixed-epoch 20 --early-stopping-patience 0 --training-output <out>
python src/run_stage2_columns_r2.py train --arm NAT_E --seed 13 \
  --fixed-epoch 20 --early-stopping-patience 0 --training-output <out>
```

`fixed_epoch=20` with `early_stopping_patience=0` is the controlled-comparison schedule: the epoch-20
checkpoint is kept and dev is monitored rather than used for selection. Compare the two
`initial_parameter_sha256` receipts before reading any result; they must be equal. The default
schedule is unchanged, so `PRIOR` keeps its patience-based early stopping and dev checkpoint
selection.

Holdout split: `int(sha256('<salt>|<source_group>'), 16) % 10 == 0` is holdout, everything else is
fit. The partition is a function of the source table only, so no source group can appear on both
sides and both arms see the same split. Holdout is a monitor; it must never choose an epoch.

## 2. Bridge-first reranking

The reranking scores the frozen candidate scope in three tiers:

1. candidates with a recovered-bridge score, ordered by `matched original rows / 5`;
2. remaining candidates ordered by the IDF-weighted visible score;
3. everything else, left in the frozen Stage-1 order.

```python
from mmdd_stage2.bridge_first_rerank import rerank

result = rerank(base, bridges, query_rows, tables, matcher, bridge_scores)
result["orders"]["PURE"]    # tiered order
result["orders"]["RRF60"]   # reciprocal-rank fusion with the Stage-1 order
```

The visible tier ignores column headers on purpose: a physical (query column, target column) value
pair defines the match, exactly as in the Direct scorer. A query column is eligible only when it
holds at least two distinct typed values, and `entity_url` is excluded.

IDF weight over a scope of `K` candidates:

```
df(x) = number of distinct candidates whose compatible column holds a matching value
w(x)  = log((K + 1) / (df(x) + 1)) / log(K + 1)
S_visible_IDF(qcol, tcol) = 1/5 * sum_i match(x_i, tcol) * w(x_i)
S_visible_IDF(target)     = max over (qcol, tcol)
```

The denominator stays at five rows. It is never `sum(w)` and never renormalized: dividing by the
weight sum would let a generic column recover a full score and lose the discriminativeness penalty.
Missing, conflicting and invalid cells contribute zero. Tier 2 breaks ties on the unweighted row
score and then on the Stage-1 rank. A bridge score is never allowed to lose to a visible-only score,
because the tiers are ordered, not blended.

RRF uses `1/(60 + stage1_rank) + 1/(60 + reranked_rank)`, rounded to 12 digits, tie-broken by
Stage-1 rank and then target id. The constant is deliberately large, so a single-rank promotion only
ties with Stage-1 rank 1 and Stage-1 order wins that tie; the tiering matters most for the `PURE`
order and for multi-rank moves.

## 3. Matched-original-row bridge score

`matched_row_score.bridge_row_scores` produces the tier-1 input. Typed keys compare exactly, and
only TEXT-to-TEXT pairs with an identical ordered numeric-token signature may fall back to a
unit-normalized cosine at `TAU_COSINE = 0.98`. Repeated recovered values keep their row
multiplicity, one target value may answer several query rows, and NULL/MISSING/CONFLICT rows score
zero against a fixed denominator of five. `reference_bridge_row_scores` recomputes the same table
scores with an independent row-by-row loop and shares no scoring helper; a production run should
require the two to agree.

Key rules worth remembering when reading results: `1,234.50` and `1234.5` are the same NUMBER;
`007` stays an opaque ID rather than becoming `7`; percent and ISO date cells keep their own keys;
URL and SYMBOL cells are exact-only and never softened by embeddings.

## 4. Provenance and immutability

- The reader and the candidate head are already implemented in `qwen.py` and `verifier.py`; this
  merge adds an evidence *source*, not a new architecture. Do not add a second reader path.
- Training an arm is immutable once `MANIFEST.json` exists. Re-running `NAT_E` on a new
  `--training-output` is the supported way to repeat a comparison.
- The receipts now hash `natural_evidence.py`, `matched_row_score.py` and `bridge_first_rerank.py`
  alongside the previous sources.
- Results from this merged code are not interchangeable with the frozen historical PRIOR
  checkpoints. A new run must state which evidence source and which schedule it used.
- Merged runs inherit one measured property of the underlying data: on the frozen entitables
  population the MiniLM cosine path supplies roughly 0.2% of all matches (about 21 of 10 500) and no
  decision lies within `1e-5` of the `0.98` threshold. The threshold is therefore not a knife-edge
  for this population, but that must be re-measured, not assumed, on any new population.

## 5. Tests

```bash
conda run -n MMDD python -m pytest \
  tests/test_matched_row_score.py tests/test_bridge_first_rerank.py \
  tests/test_natural_evidence.py tests/test_selector_evidence_arms.py -q
```

`tests/test_stage2_columns_r2.py` additionally covers the matched PRIOR/FLAT_MIX schedules and the
`fixed_epoch` guard.
