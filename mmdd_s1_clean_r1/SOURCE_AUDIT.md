# 本轮源码接线核对

以下只核对上传源码，不代表当前对话拥有原始数据湖、Qwen权重或任何真实新训练结果。历史run脚本仅用于字段定位，严禁作为训练依赖。

## `mmdd_stage1/construction.py:22–110`

原始数据入口和纯内容序列化

SHA256: `bcf0c87c75789d998ef0a364c60f0bde0d0cb33fda529ab8af8b01eb71de13f3`

```python
22: def _artifact_records(root: Path, name: str, *, required: bool = True) -> list[dict[str, Any]]:
23:     manifest = root / "dataset_manifest.json"
24:     path = root / f"{name}.jsonl"
25:     if manifest.is_file():
26:         records = list(iter_dataset_artifact(root, name))
27:     elif path.is_file():
28:         records = list(read_jsonl(path))
29:     else:
30:         records = []
31:     if required and not records:
32:         raise ValueError(f"Dataset artifact is empty or missing: {name}")
33:     return records
34:
35:
36: def serialize_table_parts(
37:     table: dict[str, Any],
38:     max_rows: int,
39:     max_cell_chars: int = DEFAULT_MAX_CELL_CHARS,
40:     *,
41:     row_format: str = "values",
42: ) -> list[str]:
43:     if row_format not in TABLE_ROW_FORMATS:
44:         raise ValueError(
45:             f"row_format must be one of: {', '.join(TABLE_ROW_FORMATS)}"
46:         )
47:     headers = [clean_text(column.get("column_name")) for column in table["columns"]]
48:     parts = ["Columns: " + " | ".join(headers)]
49:     for row in table["rows"][:max_rows]:
50:         values = []
51:         for column in table["columns"]:
52:             value = clean_text(
53:                 get_cell(row, int(column["column_index"])).get("text")
54:             )
55:             values.append(value[:max_cell_chars].rstrip())
56:         if row_format == "named_cells":
57:             values = [
58:                 f"{header or f'column_{index}'}: {value}"
59:                 for index, (header, value) in enumerate(zip(headers, values))
60:             ]
61:         parts.append("Row: " + " | ".join(values))
62:     return parts
63:
64:
65: def _table_object(
66:     table: dict[str, Any],
67:     max_rows: int,
68:     max_cell_chars: int = DEFAULT_MAX_CELL_CHARS,
69:     *,
70:     embedding_role: str,
71:     row_format: str = "values",
72: ) -> dict[str, Any]:
73:     parts = serialize_table_parts(
74:         table,
75:         max_rows,
76:         max_cell_chars,
77:         row_format=row_format,
78:     )
79:     record = {
80:         "object_id": str(table["table_id"]),
81:         "object_type": "table",
82:         "embedding_role": embedding_role,
83:         "table_parts": parts,
84:     }
85:     return record
86:
87:
88: def _asset_object(asset: dict[str, Any], dataset_root: Path) -> dict[str, Any]:
89:     asset_id = str(asset["asset_id"])
90:     asset_type = str(asset["asset_type"])
91:     record = {
92:         "object_id": asset_id,
93:         "object_type": asset_type,
94:     }
95:     if asset_type == "text":
96:         record["text"] = clean_text(asset.get("content"))
97:     elif asset_type == "image":
98:         local_path = Path(str(asset.get("local_path", "")))
99:         if not local_path.is_absolute():
100:             local_path = dataset_root / local_path
101:         relative_path = dataset_root / str(asset.get("relative_path", ""))
102:         image_path = local_path if local_path.is_file() else relative_path
103:         if not image_path.is_file():
104:             raise FileNotFoundError(f"{asset_id}: image artifact has no local file")
105:         record["image"] = str(image_path.resolve())
106:     else:
107:         raise ValueError(f"{asset_id}: unsupported asset_type {asset_type!r}")
108:     return record
109:
110:
```

## `mmdd_stage1/construction.py:184–259`

不能继承的target/source evidence回退，以及可复用的target引用解析

SHA256: `bcf0c87c75789d998ef0a364c60f0bde0d0cb33fda529ab8af8b01eb71de13f3`

```python
184: def _evidence_by_target(
185:     targets: dict[str, dict[str, Any]],
186:     assets: list[dict[str, Any]],
187:     recoveries: Iterable[dict[str, Any]],
188: ) -> dict[str, list[str]]:
189:     asset_ids = {str(asset["asset_id"]) for asset in assets}
190:     result: dict[str, list[str]] = defaultdict(list)
191:     for recovery in recoveries:
192:         target_id = str(recovery["target_table_id"])
193:         evidence_id = str(recovery.get("evidence", {}).get("asset_id", ""))
194:         if evidence_id in asset_ids and evidence_id not in result[target_id]:
195:             result[target_id].append(evidence_id)
196:
197:     assets_by_source: dict[str, list[str]] = defaultdict(list)
198:     for asset in assets:
199:         source_id = asset.get("source_table_id")
200:         if source_id is not None:
201:             assets_by_source[str(source_id)].append(str(asset["asset_id"]))
202:     for target_id, target in targets.items():
203:         if not result[target_id]:
204:             result[target_id].extend(assets_by_source.get(str(target.get("source_table_id")), ()))
205:     return result
206:
207:
208: def _recovery_evidence(
209:     queries: dict[str, dict[str, Any]],
210:     targets: dict[str, dict[str, Any]],
211:     assets: list[dict[str, Any]],
212:     recoveries: Iterable[dict[str, Any]],
213: ) -> dict[tuple[str, str], list[str]]:
214:     asset_ids = {str(asset["asset_id"]) for asset in assets}
215:     result: dict[tuple[str, str], list[str]] = defaultdict(list)
216:     for recovery in recoveries:
217:         query_id = str(recovery.get("query_table_id", ""))
218:         target_id = str(recovery.get("target_table_id", ""))
219:         evidence_id = str(recovery.get("evidence", {}).get("asset_id", ""))
220:         key = (query_id, target_id)
221:         if (
222:             query_id in queries
223:             and target_id in targets
224:             and evidence_id in asset_ids
225:             and evidence_id not in result[key]
226:         ):
227:             result[key].append(evidence_id)
228:     return result
229:
230:
231: def _resolve_target_references(
232:     dataset_root: Path, records: list[dict[str, Any]]
233: ) -> list[dict[str, Any]]:
234:     source_ids = {
235:         str(record["source_table_ref"]["source_table_id"])
236:         for record in records
237:         if "source_table_ref" in record
238:     }
239:     if not source_ids:
240:         return records
241:     sources = {
242:         str(record["source_table_id"]): record
243:         for record in _artifact_records(dataset_root, "source_tables")
244:         if str(record["source_table_id"]) in source_ids
245:     }
246:     missing = source_ids - sources.keys()
247:     if missing:
248:         raise KeyError(f"source_tables has no records for: {', '.join(sorted(missing))}")
249:     return [
250:         {
251:             **sources[str(record["source_table_ref"]["source_table_id"])],
252:             **{key: value for key, value in record.items() if key != "source_table_ref"},
253:         }
254:         if "source_table_ref" in record
255:         else record
256:         for record in records
257:     ]
258:
259:
```

## `cache_stage1_features.py:120–158`

官方last_hidden_state与last-token pooling封装

SHA256: `1e2a8fa2b1280a873dd72e7ad579d3e3a16cebc66803a5c0335f700722231ba2`

```python
120:     embedder: Any,
121:     inputs: dict[str, torch.Tensor],
122:     *,
123:     include_hidden: bool = True,
124: ) -> list[tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]]:
125:     """Run already-preprocessed inputs and return their requested CPU payload."""
126:
127:     inputs = {name: tensor.to(embedder.model.device) for name, tensor in inputs.items()}
128:     outputs = embedder.forward(inputs)
129:     hidden_states = outputs["last_hidden_state"]
130:     attention_mask = outputs["attention_mask"].bool()
131:     pooled = embedder._pooling_last(
132:         hidden_states,
133:         attention_mask.to(dtype=torch.long),
134:     )
135:     embeddings = F.normalize(pooled.float(), p=2, dim=-1)
136:     if not include_hidden:
137:         return [
138:             (embeddings[index].cpu(), None, None)
139:             for index in range(hidden_states.shape[0])
140:         ]
141:     return [
142:         (
143:             embeddings[index].cpu(),
144:             hidden_states[index][attention_mask[index]].cpu(),
145:             inputs["input_ids"][index][attention_mask[index]].cpu(),
146:         )
147:         for index in range(hidden_states.shape[0])
148:     ]
149:
150:
151: def preprocess_input_items(
152:     embedder: Any, items: list[dict[str, Any]]
153: ) -> dict[str, torch.Tensor]:
154:     """Format and preprocess model items on CPU."""
155:
156:     conversations = [
157:         embedder.format_model_input(
158:             text=item.get("text"),
```

## `cache_stage1_features.py:181–242`

可复用的表内容token定位方法

SHA256: `1e2a8fa2b1280a873dd72e7ad579d3e3a16cebc66803a5c0335f700722231ba2`

```python
181:
182: def _table_token_groups(
183:     embedder: Any,
184:     item: dict[str, Any],
185:     parts: list[str],
186:     input_ids: torch.Tensor,
187: ) -> tuple[torch.Tensor, torch.Tensor]:
188:     text = item.get("text")
189:     if not isinstance(text, str) or not text.strip():
190:         raise ValueError("A table requires non-empty text containing all table_parts")
191:
192:     part_spans = []
193:     cursor = 0
194:     for part in parts:
195:         start = text.find(part, cursor)
196:         if start < 0:
197:             raise ValueError("table_parts must occur in order within the table text")
198:         part_spans.append((start, start + len(part)))
199:         cursor = start + len(part)
200:
201:     conversation = embedder.format_model_input(
202:         text=text,
203:         image=item.get("image"),
204:         instruction=item.get("instruction"),
205:     )
206:     rendered = embedder.processor.apply_chat_template(
207:         conversation, add_generation_prompt=True, tokenize=False
208:     )
209:     text_start = rendered.rfind(text)
210:     if text_start < 0:
211:         raise ValueError("Qwen chat template did not preserve the serialized table text")
212:     rendered_spans = [(text_start + start, text_start + end) for start, end in part_spans]
213:
214:     tokenized = None
215:     for add_special_tokens in (False, True):
216:         candidate = embedder.processor.tokenizer(
217:             rendered,
218:             add_special_tokens=add_special_tokens,
219:             truncation=True,
220:             max_length=embedder.max_length,
221:             return_offsets_mapping=True,
222:         )
223:         if list(candidate["input_ids"]) == input_ids.tolist():
224:             tokenized = candidate
225:             break
226:     if tokenized is None:
227:         raise ValueError("Tokenizer offsets do not align with Qwen preprocessing")
228:
229:     selected_indices = []
230:     groups = []
231:     for token_index, (start, end) in enumerate(tokenized["offset_mapping"]):
232:         if end <= start:
233:             continue
234:         for group, (part_start, part_end) in enumerate(rendered_spans):
235:             if end > part_start and start < part_end:
236:                 selected_indices.append(token_index)
237:                 groups.append(group)
238:                 break
239:     if set(groups) != set(range(len(parts))):
240:         raise ValueError("Table truncation removed all tokens from at least one schema/row group")
241:     return torch.tensor(selected_indices, dtype=torch.long), torch.tensor(groups, dtype=torch.long)
242:
```

## `cache_stage1_features.py:290–375`

旧完整hidden缓存与表截断重试，不能直接作为新缓存流程

SHA256: `1e2a8fa2b1280a873dd72e7ad579d3e3a16cebc66803a5c0335f700722231ba2`

```python
290:     }
291:     embedding, hidden_states, input_ids = encode_inputs(
292:         embedder,
293:         [item],
294:         include_hidden=include_hidden,
295:     )[0]
296:     payload = {"embedding": embedding.float()}
297:
298:     if object_type != "table":
299:         if include_hidden:
300:             payload["hidden_states"] = hidden_states.to(dtype=storage_dtype)
301:         return payload
302:
303:     assert isinstance(parts, list)
304:     if include_hidden:
305:         assert hidden_states is not None
306:         assert input_ids is not None
307:         try:
308:             indices, groups = _table_token_groups(embedder, item, parts, input_ids)
309:         except ValueError as error:
310:             if (
311:                 str(error)
312:                 != "Table truncation removed all tokens from at least one schema/row group"
313:             ):
314:                 raise
315:             for max_chars in (4096, 2048, 1024, 512, 256, 128):
316:                 truncated_parts = [part[:max_chars].rstrip() for part in parts]
317:                 if truncated_parts == parts:
318:                     continue
319:                 teacher_item = {**item, "text": "\n".join(truncated_parts)}
320:                 _, hidden_states, input_ids = encode_inputs(embedder, [teacher_item])[0]
321:                 assert hidden_states is not None
322:                 assert input_ids is not None
323:                 try:
324:                     indices, groups = _table_token_groups(
325:                         embedder, teacher_item, truncated_parts, input_ids
326:                     )
327:                 except ValueError as retry_error:
328:                     if str(retry_error) == str(error):
329:                         continue
330:                     raise
331:                 print(
332:                     json.dumps(
333:                         {
334:                             "object_id": str(record["object_id"]),
335:                             "event": "table_parts_truncated",
336:                             "max_chars_per_part": max_chars,
337:                         }
338:                     )
339:                 )
340:                 break
341:             else:
342:                 raise
343:         # Legacy caches loaded storage-dtype tokens as float32 before pooling.
344:         # Preserve that numerical order, then keep the much smaller pooled table
345:         # representation in float32 so no second quantization is introduced.
346:         selected_hidden = hidden_states.index_select(0, indices).to(
347:             dtype=storage_dtype
348:         ).float()
349:         pooled_hidden, pooled_groups = structural_table_pool_with_groups(
350:             selected_hidden,
351:             groups,
352:             table_tokens_per_group,
353:         )
354:         payload["hidden_states"] = pooled_hidden
355:         if table_tokens_per_group > 1:
356:             assert pooled_groups is not None
357:             payload["token_groups"] = pooled_groups
358:
359:     if embedding_role == "query" and include_row_embeddings:
360:         routing_items = [
361:             {
362:                 "text": f"{parts[0]}\n{row}",
363:                 "instruction": row_instruction,
364:             }
365:             for row in parts[1:]
366:         ]
367:         routing_outputs = [
368:             output
369:             for start in range(0, len(routing_items), table_row_batch_size)
370:             for output in encode_inputs(
371:                 embedder,
372:                 routing_items[start : start + table_row_batch_size],
373:                 include_hidden=False,
374:             )
375:         ]
```

## `mmdd_dataset/joinability.py:498–550`

原始qrel与recovery字段

SHA256: `eb0270532ea8074cc6bc27d7826c1ec278fc8517cc5f907f80c6dc28d89d91b5`

```python
498:         qrels.append(
499:             {
500:                 "query_table_id": query_id,
501:                 "target_table_id": target_id,
502:                 "rel": 3,
503:                 "split": split,
504:                 "chain_id": chain_id,
505:                 "source_table_id": source_table_id,
506:                 "join_attribute": hidden_attribute,
507:                 "reason": "model_recoverable_join_column",
508:             }
509:         )
510:         target_row_by_source = {row["source_row_id"]: row["row_id"] for row in target_rows}
511:         seen_recovery_ids: set[str] = set()
512:         for source_row_id in retained_query_rows:
513:             for extraction in candidate["recoveries"].get(source_row_id, []):
514:                 member_value = sanitize_cell_text(
515:                     get_cell(source_rows_by_id[source_row_id], member).get("text")
516:                 )
517:                 recovery_id = "rec_" + stable_hash(
518:                     query_id, target_id, source_row_id, extraction["asset_id"], member_value
519:                 )
520:                 if recovery_id in seen_recovery_ids:
521:                     continue
522:                 seen_recovery_ids.add(recovery_id)
523:                 asset = assets_by_id.get(extraction["asset_id"], {})
524:                 recoveries.append(
525:                     {
526:                         "recovery_id": recovery_id,
527:                         "query_table_id": query_id,
528:                         "target_table_id": target_id,
529:                         "source_table_id": source_table_id,
530:                         "source_row_id": source_row_id,
531:                         "query_row_id": query_row_by_source[source_row_id],
532:                         "target_row_ids": [target_row_by_source[source_row_id]],
533:                         "split": split,
534:                         "query_entity": {
535:                             "text": clean_text(
536:                                 get_cell(source_rows_by_id[source_row_id], entity_col).get("text")
537:                             ),
538:                             "wiki_title": clean_text(
539:                                 get_cell(source_rows_by_id[source_row_id], entity_col).get("wiki_title")
540:                             ),
541:                         },
542:                         "recovered_attribute": {
543:                             "column_index": member,
544:                             "column_name": member_names[member],
545:                             "value": member_value,
546:                             "model_value": clean_text(extraction.get("value")),
547:                             "hidden_in_query": True,
548:                         },
549:                         "evidence": {
550:                             "asset_id": extraction["asset_id"],
```

## `diagnose_final_rerank_witness.py:136–164`

原始recovery中的auto_check字段；不读取此脚本历史结果

SHA256: `d3cd4390e1b00c112c036b665cf86e8b57309667e99771c6f74464cce55211e6`

```python
136: def load_witness_labels() -> tuple[dict[tuple[str, str], dict[str, dict]], dict[str, Any]]:
137:     """(query_id, target_id) -> {asset_id: label record}. Dev split only."""
138:
139:     labels: dict[tuple[str, str], dict[str, dict]] = defaultdict(dict)
140:     stats: Counter = Counter()
141:     paths = sorted(DATASET.glob("evidence_recoveries/part-*.jsonl"))
142:     if not paths:
143:         raise SystemExit(f"no evidence_recoveries under {DATASET}")
144:     for path in paths:
145:         for record in rows(path):
146:             stats[f"split:{record.get('split')}"] += 1
147:             if record.get("split") != WITNESS_ANNOTATION_SCOPE:
148:                 continue
149:             evidence = record.get("evidence") or {}
150:             asset_id = str(evidence.get("asset_id") or "")
151:             if not asset_id:
152:                 stats["skipped_missing_asset"] += 1
153:                 continue
154:             review = record.get("auto_check") or {}
155:             reviews = review.get("reviews") or []
156:             policy = str(review.get("policy") or "")
157:             supported = bool(reviews) and all(
158:                 str(item.get("verdict")) == "supported" for item in reviews
159:             )
160:             if not supported or policy != "keep_source_canonical_supported_only_fail_closed":
161:                 stats["skipped_not_fail_closed_supported"] += 1
162:                 continue
163:             key = (str(record["query_table_id"]), str(record["target_table_id"]))
164:             entry = labels[key].setdefault(
```

## `evaluate_stage1_b4_test.py:62–91`

原始GT reason枚举与source group字段；不继承固定query数

SHA256: `b442e599ba5b90a41e31631b2caa4d346d6d116dcb06e26a328a9fe20c7b8741`

```python
62:
63:     expected_ids = set(target_by_query)
64:     if set(query_meta) != expected_ids or set(qrels) != expected_ids:
65:         raise ValueError("Frozen target lists, query tables, and qrels have different test populations")
66:
67:     population = []
68:     for target_row in targets:
69:         query_id = str(target_row["query_id"])
70:         labels = qrels[query_id]
71:         reasons = {str(row["reason"]) for row in labels}
72:         if reasons == {"model_recoverable_join_column"}:
73:             query_kind = "implicit"
74:         elif reasons == {"explicit_visible_join_column"}:
75:             query_kind = "explicit"
76:         else:
77:             raise ValueError(f"Unexpected test qrel reasons for {query_id}: {sorted(reasons)}")
78:         positive_ids = sorted(str(row["target_table_id"]) for row in labels)
79:         if positive_ids != sorted(map(str, target_row["positive_target_ids"])):
80:             raise ValueError(f"Frozen qrels changed for {query_id}")
81:         source_ids = {str(row["source_table_id"]) for row in labels}
82:         source_ids.add(str(query_meta[query_id]["source_table_id"]))
83:         if len(source_ids) != 1:
84:             raise ValueError(f"Source group changed for {query_id}")
85:         population.append(
86:             {
87:                 "query_id": query_id,
88:                 "query_kind": query_kind,
89:                 "positive_target_ids": positive_ids,
90:                 "source_table_id": source_ids.pop(),
91:             }
```

## `mmdd_stage1/query_conditioned_et.py:23–70`

旧非受限残差结构，仅用于说明本轮不复刻的结构

SHA256: `2e72dd433b98fa26199b7b6d4dae01bee343b780bab0343ee72c635ab02ee0a0`

```python
23:
24:     def __init__(self, dimension: int, hidden_dimension: int = 256) -> None:
25:         super().__init__()
26:         if dimension <= 0 or hidden_dimension <= 0:
27:             raise ValueError("Adapter dimensions must be positive")
28:         self.dimension = int(dimension)
29:         self.hidden_dimension = int(hidden_dimension)
30:         self.input = nn.Linear(4 * dimension, hidden_dimension)
31:         self.output = nn.Linear(hidden_dimension, dimension)
32:         nn.init.zeros_(self.output.weight)
33:         nn.init.zeros_(self.output.bias)
34:
35:     def features(
36:         self,
37:         query: torch.Tensor,
38:         evidence: torch.Tensor,
39:         arm: str,
40:     ) -> torch.Tensor:
41:         if arm not in ARMS:
42:             raise ValueError(f"Unknown adapter arm: {arm}")
43:         if query.shape != evidence.shape or query.shape[-1] != self.dimension:
44:             raise ValueError("Query and evidence vectors must have equal adapter dimensions")
45:         if arm == "e_only":
46:             zero = torch.zeros_like(query)
47:             return torch.cat((zero, evidence, zero, zero), dim=-1)
48:         return torch.cat(
49:             (query, evidence, query * evidence, (query - evidence).abs()), dim=-1
50:         )
51:
52:     def forward(
53:         self,
54:         query: torch.Tensor,
55:         evidence: torch.Tensor,
56:         arm: str,
57:     ) -> torch.Tensor:
58:         return self.output(F.gelu(self.input(self.features(query, evidence, arm))))
59:
60:     def conditioned_query(
61:         self,
62:         base_query: torch.Tensor,
63:         query: torch.Tensor,
64:         evidence: torch.Tensor,
65:         arm: str,
66:     ) -> torch.Tensor:
67:         return base_query + self(query, evidence, arm)
68:
69:     def config(self) -> dict[str, int]:
70:         return {
```
