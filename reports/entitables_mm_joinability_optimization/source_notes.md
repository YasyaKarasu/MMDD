# EntiTables MM Joinability optimization report — source notes

Audience: technical. Delivery mode: portable HTML.

## Report structure mapping

- Technical summary: `technical_summary`
- Key findings with visual evidence: `entity_rows_finding`, `version_finding`, `modality_finding`
- Scope, data, and definitions: `metric_definition`, `scope_methods`
- Methodology: source metadata plus this file
- Limitations and robustness: `quality_finding`, `scope_methods`
- Recommended next steps: `recommended_next_steps`
- Further questions: `further_questions`

## Chart map

1. `entity_rows_rate_chart`
   - Question: Which cheap structural signal best separates queryable from non-queryable sources?
   - Takeaway: Queryability rises sharply with active entity rows; fewer than five is a hard failure under the current contract.
   - Family/type: comparison / vertical bar.
   - Fields: `entity_rows_bin`, `queryable_rate`, plus table and queryable-source counts in tooltip.
   - Palette: single blue root; labels and ordering provide non-color distinction.
   - Source: v9 source tables, entities, and queryability decisions.

2. `modality_recovery_chart`
   - Question: Which modality supplies emitted positive evidence?
   - Takeaway: Text dominates recovery evidence; image is the multimodal bottleneck.
   - Family/type: comparison / vertical bar.
   - Fields: `modality`, `recovery_records`, with query, extraction volume, and non-empty rate in tooltip.
   - Palette: single blue root; modality is already encoded on the x axis, so no redundant color legend.
   - Source: v9 evidence recoveries and attribute extractions.

## Reproducibility notes

- All calculations were read-only and use local artifacts under `output_mm_joinability_v7`, `v8`, and `v9`.
- The `sql/` files reproduce the reviewed snapshot rows in SQLite-compatible SQL so every report visual and table has an executable source affordance; raw-file lineage remains in `artifact.json` source metadata.
- Hard-failure counts join the recorded chosen entity column to source rows and treat a `wiki_title` present in the v9 entities artifact as active.
- The disjoint-row-view calculation uses `min(floor(eligible_rows / 5), floor(recovered_rows / 2))`, capped at 2 per qualified attribute.
- The matcher sensitivity audit removes only short non-exact substring matches for the reported 15-qrel boundary check; it is not a proposed final matching policy.
- Omitted visual: optimization priorities remain a table because several impacts are estimates or qualitative risks rather than commensurate numeric measures.
