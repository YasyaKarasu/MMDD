# 源码定位与复核边界

所有行号均指上传包的源文件，未修改生产源码。合成探针不是实际训练重跑。

## T1：训练没有自然B

`src/mmdd_stage1_clean/train.py:271–296`

SHA256: `0dab311d84c65bfd0a64e58a00376a72fd5df908959e4589dca0818c0da17b63`

```python
0271:         # --- B packet: same target list ids as D, context varies by epoch
0272:         if include_bundle:
0273:             natural = self.natural_bundle(q_rank or {}, per_modality) if q_rank else []
0274:             if q_rank is None:
0275:                 natural = []
0276:             if use_natural_bundle(epoch, query_id) or anchor is None:
0277:                 bundle = list(natural)
0278:                 view = "natural"
0279:             else:
0280:                 bundle = augmented_bundle(natural, anchor, modality=self.modality)
0281:                 view = "witness_augmented"
0282:             b_positives = support_positive_set(
0283:                 direct=direct, implicit=implicit, witnesses=witnesses, context=set(bundle)
0284:             )
0285:             b_excluded = all_positive - b_positives
0286:             add(B_PACKET, {
0287:                 "list": d_list,
0288:                 "mode": "J",
0289:                 "candidates": d_list.ordered_ids,
0290:                 "context": bundle,
0291:                 "view": view,
0292:                 "positive_ids": sorted(b_positives),
0293:                 "excluded_ids": sorted(b_excluded),
0294:             })
0295:             diagnostics["packets"][B_PACKET]["positive_ids"] = sorted(b_positives)
0296:         return {"packets": packets, "diagnostics": diagnostics}
```

## T1：同步入口传None

`src/mmdd_stage1_clean/train.py:529–551`

SHA256: `0dab311d84c65bfd0a64e58a00376a72fd5df908959e4589dca0818c0da17b63`

```python
0529:     def query_loss(
0530:         self, query_id: str, epoch: int, prefetched: dict[str, Any] | None = None
0531:     ) -> dict[str, Any]:
0532:         row = self.builder.state("train", query_id)
0533:         assert row is not None
0534:         direct, implicit, all_positive = self.builder.positive_sets(row)
0535:         witnesses = self.builder.witnesses(row)
0536:         if prefetched is not None:
0537:             built = prefetched
0538:         else:
0539:             with self.timing.stage("packet_build"):
0540:                 built = self.builder.build(
0541:                     split="train",
0542:                     row=row,
0543:                     epoch=epoch,
0544:                     phase="teacher",
0545:                     sampling_arm="T",
0546:                     q_rank=None,
0547:                     anchor_rank=self.anchor_rank,
0548:                     rank_tables=self.rank_tables,
0549:                     per_modality=self.per_modality,
0550:                     include_bundle=epoch in self.bundle_epochs,
0551:                 )
```

## T1：正式预取入口同样传None

`src/mmdd_stage1_clean/train.py:619–646`

SHA256: `0dab311d84c65bfd0a64e58a00376a72fd5df908959e4589dca0818c0da17b63`

```python
0619:         for epoch in range(1, self.epochs + 1):
0620:             order = stable_order(self.queries, query_order_namespace("T", epoch))
0621:             pending: list[torch.Tensor] = []
0622:             pending_diag: list[dict[str, Any]] = []
0623:             self.teacher.train()
0624:             include_bundle = epoch in self.bundle_epochs
0625:             requests = (
0626:                 PacketRequest(
0627:                     split="train",
0628:                     row=self.builder.state("train", query_id),
0629:                     epoch=epoch,
0630:                     phase="teacher",
0631:                     sampling_arm="T",
0632:                     q_rank=None,
0633:                     per_modality=self.per_modality,
0634:                     include_bundle=include_bundle,
0635:                 )
0636:                 for query_id in order
0637:             )
0638:             prefetcher = PacketPrefetcher(builder=self.builder, workers=self.prefetch_workers)
0639:             with self.timing.stage("packet_wait"):
0640:                 for query_id, prefetched in zip(
0641:                     order,
0642:                     prefetcher.stream(
0643:                         requests, anchor_rank=self.anchor_rank, rank_tables=self.rank_tables
0644:                     ),
0645:                 ):
0646:                     result = self.query_loss(query_id, epoch, prefetched=prefetched)
```

## T2：B的正确空正集检查

`src/mmdd_stage1_clean/train.py:403–418`

SHA256: `0dab311d84c65bfd0a64e58a00376a72fd5df908959e4589dca0818c0da17b63`

```python
0403: def teacher_rank_term(
0404:     scores: torch.Tensor,
0405:     packet: dict[str, Any],
0406: ) -> torch.Tensor | None:
0407:     negative_list: NegativeList = packet["list"]
0408:     positive_ids = packet.get("positive_ids", negative_list.positive_ids)
0409:     if not positive_ids or not negative_list.negative_ids:
0410:         return None
0411:     index = {value: i for i, value in enumerate(packet["candidates"])}
0412:     positive_mask = torch.zeros_like(scores, dtype=torch.bool)
0413:     negative_mask = torch.zeros_like(scores, dtype=torch.bool)
0414:     positive_mask[[index[v] for v in positive_ids]] = True
0415:     negative_mask[[index[v] for v in negative_list.negative_ids]] = True
0416:     if (positive_mask & negative_mask).any():
0417:         raise ConfigError("a known positive target entered the negative competition")
0418:     return reference.rank_loss(scores, positive_mask, negative_mask)
```

## T2：预取序列化丢失空override

`src/mmdd_stage1_clean/train.py:1495–1531`

SHA256: `0dab311d84c65bfd0a64e58a00376a72fd5df908959e4589dca0818c0da17b63`

```python
1495: def _compact_packets(built: dict[str, Any]) -> dict[str, Any]:
1496:     """Reduce a build result to plain data, so the pool never pickles a module."""
1497:     packets = {}
1498:     for name, packet in built["packets"].items():
1499:         negative_list = packet["list"]
1500:         packets[name] = {
1501:             "positive_ids": list(negative_list.positive_ids),
1502:             "negative_ids": list(negative_list.negative_ids),
1503:             "ordered_ids": list(negative_list.ordered_ids),
1504:             "provenance": dict(negative_list.provenance),
1505:             "mode": packet["mode"],
1506:             "candidates": list(packet["candidates"]),
1507:             "context": list(packet["context"]),
1508:             "view": packet.get("view"),
1509:             "positive_ids_override": packet.get("positive_ids"),
1510:         }
1511:     return {"packets": packets, "diagnostics": built["diagnostics"]}
1512:
1513:
1514: def _expand_packets(compact: dict[str, Any]) -> dict[str, Any]:
1515:     packets = {}
1516:     for name, data in compact["packets"].items():
1517:         packets[name] = {
1518:             "list": NegativeList(
1519:                 positive_ids=data["positive_ids"],
1520:                 negative_ids=data["negative_ids"],
1521:                 ordered_ids=data["ordered_ids"],
1522:                 provenance=data["provenance"],
1523:             ),
1524:             "mode": data["mode"],
1525:             "candidates": data["candidates"],
1526:             "context": data["context"],
1527:             **({"view": data["view"]} if data.get("view") else {}),
1528:             **({"positive_ids": data["positive_ids_override"]}
1529:                if data.get("positive_ids_override") else {}),
1530:         }
1531:     return {"packets": packets, "diagnostics": compact["diagnostics"]}
```

## T2：单进程也执行相同错误转换

`src/mmdd_stage1_clean/train.py:1586–1606`

SHA256: `0dab311d84c65bfd0a64e58a00376a72fd5df908959e4589dca0818c0da17b63`

```python
1586:         if self.workers == 1:
1587:             # Inline: no pool, no worker globals, identical code path to a direct
1588:             # build, so a machine with one usable core still trains correctly.
1589:             for request in requests:
1590:                 yield _expand_packets(
1591:                     _compact_packets(
1592:                         self.builder.build(
1593:                             split=request.split,
1594:                             row=request.row,
1595:                             epoch=request.epoch,
1596:                             phase=request.phase,
1597:                             sampling_arm=request.sampling_arm,
1598:                             q_rank=request.q_rank,
1599:                             anchor_rank=anchor_rank,
1600:                             rank_tables=rank_tables,
1601:                             per_modality=request.per_modality,
1602:                             include_bundle=request.include_bundle,
1603:                         )
1604:                     )
1605:                 )
1606:             return
```

## T3：Teacher刷新实际仅P调用

`src/mmdd_stage1_clean/train.py:714–772`

SHA256: `0dab311d84c65bfd0a64e58a00376a72fd5df908959e4589dca0818c0da17b63`

```python
0714:     def refresh_hard_ranks(self) -> dict[str, Any]:
0715:         """Spec 8.3: one in-run refresh after epoch 2, using the current Teacher."""
0716:         started = time.time()
0717:         self.teacher.eval()
0718:         refreshed: dict[str, dict[str, list[tuple[str, float]]]] = {
0719:             "D": {}, "E_text": {}, "E_image": {}, "anchor": {}
0720:         }
0721:         with torch.no_grad():
0722:             for row in self.builder.population("train"):
0723:                 query_id = row["query_id"]
0724:                 pool_targets = _dedupe(
0725:                     self.rank_tables["D"].ids(query_id)
0726:                     + [v for v, _ in self.anchor_hard_pool(query_id)]
0727:                 )
0728:                 if pool_targets:
0729:                     scores = torch.cat(
0730:                         self.batch.score(
0731:                             self.teacher, [query_id], [pool_targets], None, MODE_P,
0732:                             chunk=self.chunk,
0733:                         )
0734:                     )
0735:                     refreshed["D"][query_id] = _ranked(pool_targets, scores)
0736:                 pool_text = self.rank_tables["E_text"].ids(query_id)
0737:                 pool_image = self.rank_tables["E_image"].ids(query_id)
0738:                 if pool_text:
0739:                     scores = torch.cat(
0740:                         self.batch.score(self.teacher, [query_id], [pool_text], None, MODE_P,
0741:                                          chunk=self.chunk)
0742:                     )
0743:                     refreshed["E_text"][query_id] = _ranked(pool_text, scores)
0744:                 if pool_image:
0745:                     scores = torch.cat(
0746:                         self.batch.score(self.teacher, [query_id], [pool_image], None, MODE_P,
0747:                                          chunk=self.chunk)
0748:                     )
0749:                     refreshed["E_image"][query_id] = _ranked(pool_image, scores)
0750:         payload = {
0751:             key: RankTable(value).to_rows() for key, value in refreshed.items()
0752:         }
0753:         write_jsonl(self.output_dir / "mining_epoch2" / "refreshed_ranks.jsonl", (
0754:             {"table": table, **row} for table, rows in payload.items() for row in rows
0755:         ))
0756:         self.rank_tables["D"] = RankTable(refreshed["D"])
0757:         self.rank_tables["E_text"] = RankTable(refreshed["E_text"])
0758:         self.rank_tables["E_image"] = RankTable(refreshed["E_image"])
0759:         write_json(
0760:             self.output_dir / "mining_epoch2" / "meta.json",
0761:             {
0762:                 "refreshed_queries": {k: len(v) for k, v in refreshed.items()},
0763:                 "elapsed_seconds": time.time() - started,
0764:                 "source": "current in-run Teacher, mode=P",
0765:             },
0766:         )
0767:         log_line(f"teacher: refreshed hard ranks after epoch 2 ({time.time() - started:.0f}s)")
0768:         return {"refreshed": {k: len(v) for k, v in refreshed.items()}}
0769:
0770:     def anchor_hard_pool(self, query_id: str) -> list[tuple[str, float]]:
0771:         return self.rank_tables["D"].get(query_id)
0772:
```

## T4：先交替

`src/mmdd_stage1_clean/train.py:307–327`

SHA256: `0dab311d84c65bfd0a64e58a00376a72fd5df908959e4589dca0818c0da17b63`

```python
0307: def merge_modality_ranks(
0308:     left: Sequence[tuple[str, float]] | None,
0309:     right: Sequence[tuple[str, float]] | None,
0310:     limit: int,
0311: ) -> list[tuple[str, float]]:
0312:     """Alternate two already ranked lists by position, never comparing raw scores
0313:     across modalities (spec 7.3 E packet)."""
0314:     out: list[tuple[str, float]] = []
0315:     seen: set[str] = set()
0316:     left = list(left or [])
0317:     right = list(right or [])
0318:     for position in range(max(len(left), len(right))):
0319:         for stream in (left, right):
0320:             if position < len(stream):
0321:                 value, score = stream[position]
0322:                 if value not in seen:
0323:                     seen.add(value)
0324:                     out.append((value, score))
0325:         if len(out) >= limit:
0326:             break
0327:     return out[:limit]
```

## T4：后按不同来源score重排

`src/mmdd_stage1_clean/sampling.py:95–114`

SHA256: `39079df18ec5f8191e68812836f3803707dff8aa9c45feed839adc01d5a90941`

```python
0095:     def hard_pool(
0096:         self,
0097:         hard_rank: Sequence[tuple[str, float]] | None,
0098:         *,
0099:         top_n: int = HARD_POOL_TOP_N,
0100:     ) -> list[str]:
0101:         """HardRank is真实score降序/ID升序; no random perturbation is added."""
0102:         if not hard_rank:
0103:             return []
0104:         ordered = sorted(hard_rank, key=lambda item: (-float(item[1]), item[0]))
0105:         out: list[str] = []
0106:         seen: set[str] = set()
0107:         for value, _ in ordered:
0108:             if value in seen:
0109:                 continue
0110:             seen.add(value)
0111:             out.append(value)
0112:             if len(out) >= top_n:
0113:                 break
0114:         return out
```

## T4：C混合raw ET与QT/P分数

`src/mmdd_stage1_clean/train.py:241–263`

SHA256: `0dab311d84c65bfd0a64e58a00376a72fd5df908959e4589dca0818c0da17b63`

```python
0241:         # --- C packet: one witness anchor per epoch
0242:         anchor = select_witness_anchor(self.witness_union(row), query_id, epoch)
0243:         diagnostics["witness_anchor"] = anchor
0244:         if anchor is not None:
0245:             c_positives = support_positive_set(
0246:                 direct=direct, implicit=implicit, witnesses=witnesses, context={anchor}
0247:             )
0248:             c_excluded = all_positive - c_positives
0249:             c_hard = merge_modality_ranks(
0250:                 anchor_rank.get(anchor, []),
0251:                 rank_tables["D"].get(query_id),
0252:                 self.hard_pool_top_n,
0253:             )
0254:             c_list = make.build(
0255:                 packet=C_PACKET,
0256:                 epoch=epoch,
0257:                 query_id=query_id,
0258:                 anchor=anchor,
0259:                 destination="target",
0260:                 positives=sorted(c_positives),
0261:                 excluded=c_excluded,
0262:                 hard_rank=c_hard,
0263:             )
```

## E1：评测实际使用D100

`src/mmdd_stage1_clean/commands.py:1090–1127`

SHA256: `86bb8aa28f0cf495eb7dfed327343e3aaf521f59e82299f2fef6a0837b6dc77a`

```python
1090: def cmd_freeze_teacher(args, spec, spec_dir, cwd, output_root, receipt) -> dict[str, Any]:
1091:     """Spec 8.3: choose the best epoch on dev by the pre-registered key."""
1092:     resolved = _resolved_or_raise(output_root)
1093:     teacher_dir = output_root / "teacher"
1094:     epochs = int(resolved["teacher"]["epochs"])
1095:     population = _population_index(output_root, "dev")
1096:     dev_records = list(
1097:         read_jsonl(output_root / "raw_retrieval" / "dev" / "query_rankings.jsonl")
1098:     )
1099:     if not dev_records:
1100:         raise ConfigError("dev raw retrieval is missing; run raw-retrieve --split dev")
1101:     # Dev candidates are the RAW Qwen C100/B20, fixed before any training.
1102:     retrieval = resolved["retrieval"]
1103:     bank, _corpora = _load_bank_objects(output_root, resolved)
1104:     per_modality = int(retrieval["evidence_per_modality"])
1105:     fixed: list[dict[str, Any]] = []
1106:     for row in dev_records:
1107:         query_id = str(row["query_id"])
1108:         meta = population.get(query_id)
1109:         if meta is None:
1110:             continue
1111:         direct = row["target_ids"][: int(retrieval["direct_k"])]
1112:         bundle = retrieve.interleave_text_image(
1113:             row["text_ids"][:per_modality], row["image_ids"][:per_modality],
1114:             per_modality, per_modality * 2,
1115:         )
1116:         fixed.append(
1117:             {
1118:                 "query_id": query_id,
1119:                 "C100": direct,
1120:                 "B_Q": bundle,
1121:                 "direct_target_ids": meta["direct_target_ids"],
1122:                 "implicit_target_ids": meta["implicit_target_ids"],
1123:                 "positive_target_ids": meta["positive_target_ids"],
1124:             }
1125:         )
1126:     cache = evaluate.TeacherLogitCache(output_root / "teacher_logits.sqlite")
1127:     candidates: list[dict[str, Any]] = []
```

## Q1：Query/T共用截12行

`src/mmdd_stage1_clean/cache.py:681–723`

SHA256: `aeaec8f30cfb490746751980c143c991098b0b8fba79a680712ec31455fac290`

```python
0681: def encode_one(
0682:     encoder: FrozenEncoder,
0683:     lake: Any,
0684:     entry: dict[str, Any],
0685:     cache_config: dict[str, Any],
0686:     gt: dict[str, Any],
0687: ) -> dict[str, Any]:
0688:     """One frozen forward for one object, with per-phase and per-modality timing.
0689:
0690:     Exactly one Qwen call is made per object; OOM is handled by the caller's batch
0691:     splitting, never by silently substituting a shorter input.
0692:     """
0693:     object_id = entry["object_id"]
0694:     modality = "table" if entry["object_type"] == "table" else entry["modality"]
0695:     phase = Timing()
0696:     previous, encoder._active = encoder._active, phase
0697:     try:
0698:         with phase.stage("resolve_content"):
0699:             if entry["object_type"] == "table":
0700:                 max_rows = int(cache_config["table_max_rows"])
0701:                 max_cell = int(cache_config["table_max_cell_chars"])
0702:                 row_format = str(cache_config["table_row_format"])
0703:                 content = lake.visible_content(object_id, max_rows, max_cell, row_format)
0704:                 parts = content["table_parts"]
0705:                 clipped, original, kept = clip_table_parts(
0706:                     parts, encoder.tokenizer, encoder.part_token_limit
0707:                 )
0708:             else:
0709:                 canonical = gt["canonical"][object_id]
0710:                 asset = lake.assets[object_id]
0711:                 content = parts = clipped = original = kept = None
0712:
0713:         if entry["object_type"] == "table":
0714:             payload = encoder.encode_table(
0715:                 object_id=object_id,
0716:                 parts=parts,
0717:                 instruction=EMBEDDING_INSTRUCTIONS[(content["embedding_role"], "table")],
0718:                 clipped_parts=clipped,
0719:                 token_counts=kept,
0720:             )
0721:             payload["kind_ids"] = list(TABLE_KINDS)
0722:             payload["modality_id"] = MODALITY_TABLE
0723:             payload["part_token_original"] = original
```

## Q1：仍将行压到七槽

`src/mmdd_stage1_clean/cache.py:394–419`

SHA256: `aeaec8f30cfb490746751980c143c991098b0b8fba79a680712ec31455fac290`

```python
0394:
0395:         z = self._pool(hidden, mask)
0396:         schema = pooled[0]
0397:         rows = pooled[1:]
0398:         slots = self._group_rows(rows)
0399:         return {"z": z, "summary": torch.stack([schema] + slots, dim=0),
0400:                 "token_counts": token_counts}
0401:
0402:     @staticmethod
0403:     def _group_rows(rows: torch.Tensor) -> list[torch.Tensor]:
0404:         """C[1..7]: one slot per row when m<=7, else seven equal contiguous groups."""
0405:         m = rows.shape[0]
0406:         if m == 0:
0407:             return [torch.zeros_like(rows[0]) for _ in range(7)]
0408:         if m <= 7:
0409:             slots = [rows[i] for i in range(m)]
0410:             slots += [torch.zeros_like(rows[0]) for _ in range(7 - m)]
0411:             return slots
0412:         slots = []
0413:         for j in range(7):
0414:             start = (j * m) // 7
0415:             stop = ((j + 1) * m) // 7
0416:             slots.append(rows[start:stop].mean(dim=0))
0417:         return slots
0418:
0419:     def encode_text(
```

## S1/S2：Student任务路由与candidate keys错误

`src/mmdd_stage1_clean/train.py:1013–1038`

SHA256: `0dab311d84c65bfd0a64e58a00376a72fd5df908959e4589dca0818c0da17b63`

```python
1013:         }
1014:
1015:     def score_packet(self, query_id: str, packet: dict[str, Any]) -> torch.Tensor:
1016:         """Spec Eq. (15) for one packet.
1017:
1018:         Every object in the list is re-encoded with the **current** parameters,
1019:         so the whole path from the raw cache through pooling, projection and the
1020:         conditional query is differentiable.  Eq. (14) conditions only the query
1021:         vector, so the candidate keys go through the shared pooling plus the
1022:         target-side projection and never see the evidence.
1023:         """
1024:         candidates = packet["candidates"]
1025:         objects = [query_id] + list(candidates)
1026:         if packet["context"]:
1027:             objects = objects + [packet["context"][0]]
1028:         cache, valid, modality, kind = self.bank.keys(objects, self.device)
1029:         vectors = self.student.encode(cache, valid, modality, kind)
1030:         if packet["context"]:
1031:             query_vector = self.student.query_next(vectors[0], vectors[-1])
1032:         elif packet["mode"] == "P":
1033:             query_vector = self.student.query_direct(vectors[0])
1034:         else:
1035:             query_vector = self.student.query_evidence(vectors[0])
1036:         candidate_keys = self.student.base_e(vectors[1 : 1 + len(candidates)])
1037:         return self.student.logits(query_vector, candidate_keys)
1038:
```

## S2：挖掘key不是原始nu

`src/mmdd_stage1_clean/train.py:882–918`

SHA256: `0dab311d84c65bfd0a64e58a00376a72fd5df908959e4589dca0818c0da17b63`

```python
0882:     def key_index(self) -> dict[str, Any]:
0883:         """Per-path projected corpus keys for this epoch (spec 9.2, 10.1).
0884:
0885:         Each object's Student key is computed once and reused for all four
0886:         projections: re-encoding the identical object set per path would not
0887:         change a single stored vector.
0888:         """
0889:         corpora = self.builder.corpora
0890:         base = self.object_keys()
0891:         index = {object_id: i for i, object_id in enumerate(self.object_ids)}
0892:         target_ids = list(corpora["target"])
0893:         text_ids = list(corpora["evidence_text"])
0894:         image_ids = list(corpora["evidence_image"])
0895:         base_tensor = torch.from_numpy(base)
0896:
0897:         def project(ids: list[str], linear: torch.nn.Module) -> np.ndarray:
0898:             vectors = base_tensor[[index[o] for o in ids]]
0899:             return reference.unit(vectors @ linear.weight.detach().cpu().T).numpy()
0900:
0901:         return {
0902:             "target_ids": target_ids,
0903:             "text_ids": text_ids,
0904:             "image_ids": image_ids,
0905:             "target_position": index,
0906:             # C path: the target index key is base_e(nu_T); Eq. (14) conditions only
0907:             # the query vector, never the target key or the target index.
0908:             "target_keys_C": torch.from_numpy(
0909:                 reference.unit(
0910:                     base_tensor[[index[o] for o in target_ids]]
0911:                     @ self.student.base_e.weight.detach().cpu().T
0912:                 ).numpy()
0913:             ).to(self.device),
0914:             "D_keys": project(target_ids, self.student.direct),
0915:             "E_text_keys": project(text_ids, self.student.evidence),
0916:             "E_image_keys": project(image_ids, self.student.evidence),
0917:         }
0918:
```

## S2/S3：全湖刷新没有C，查询与key快照不一致风险

`src/mmdd_stage1_clean/train.py:1041–1092`

SHA256: `0dab311d84c65bfd0a64e58a00376a72fd5df908959e4589dca0818c0da17b63`

```python
1041:     def refresh_mining(self, epoch: int, keys: dict[str, Any]) -> dict[str, Any]:
1042:         """Spec 9.2: exact top-128 mining with the previous epoch's *last* model."""
1043:         started = time.time()
1044:         refreshed: dict[str, dict[str, list[tuple[str, float]]]] = {
1045:             "D": {}, "E_text": {}, "E_image": {}, "anchor": {}
1046:         }
1047:         self.student.eval()
1048:         target_ids = keys["target_ids"]
1049:         text_ids = keys["text_ids"]
1050:         image_ids = keys["image_ids"]
1051:         rows = self.builder.population("train")
1052:         query_ids = [row["query_id"] for row in rows]
1053:         base = self._base_vectors(query_ids)
1054:         with self.timing.stage("mining_project_queries"), torch.no_grad():
1055:             d_queries = torch.from_numpy(
1056:                 _unit_np(base @ self.student.direct.weight.detach().cpu().numpy().T)
1057:             ).numpy()
1058:             e_queries = torch.from_numpy(
1059:                 _unit_np(base @ self.student.evidence.weight.detach().cpu().numpy().T)
1060:             ).numpy()
1061:         with self.timing.stage("mining_exact_topk"):
1062:             d_scores = d_queries @ keys["D_keys"].T
1063:             text_scores = e_queries @ keys["E_text_keys"].T
1064:             image_scores = e_queries @ keys["E_image_keys"].T
1065:         for i, row in enumerate(rows):
1066:             query_id = row["query_id"]
1067:             positives = set(row["positive_target_ids"])
1068:             refreshed["D"][query_id] = _top_pairs(target_ids, d_scores[i], 128, positives)
1069:             refreshed["E_text"][query_id] = _top_pairs(text_ids, text_scores[i], 128, set())
1070:             refreshed["E_image"][query_id] = _top_pairs(image_ids, image_scores[i], 128, set())
1071:         payload = {key: RankTable(value).to_rows() for key, value in refreshed.items()}
1072:         out_dir = self.output_dir / f"lists_epoch{epoch:02d}"
1073:         write_jsonl(
1074:             out_dir / "refreshed_ranks.jsonl",
1075:             ({"table": table, **row} for table, rows_ in payload.items() for row in rows_),
1076:         )
1077:         self.rank_tables["D"] = RankTable(refreshed["D"])
1078:         self.rank_tables["E_text"] = RankTable(refreshed["E_text"])
1079:         self.rank_tables["E_image"] = RankTable(refreshed["E_image"])
1080:         meta = {
1081:             "epoch": epoch,
1082:             "generator": f"{self.arm} epoch {epoch - 1} last parameters",
1083:             "elapsed_seconds": time.time() - started,
1084:             "corpus_sizes": {
1085:                 "target": len(target_ids), "text": len(text_ids), "image": len(image_ids)
1086:             },
1087:             "note": "hard-negative ids only; no loss is computed against these keys",
1088:         }
1089:         write_json(out_dir / "meta.json", meta)
1090:         log_line(f"{self.arm}: refreshed mining for epoch {epoch} ({time.time() - started:.0f}s)")
1091:         return meta
1092:
```

## S3：keys在epoch开始创建

`src/mmdd_stage1_clean/train.py:1118–1127`

SHA256: `0dab311d84c65bfd0a64e58a00376a72fd5df908959e4589dca0818c0da17b63`

```python
1118:         started = time.time()
1119:         step = 0
1120:         chunk = int(self.resolved["retrieval"]["teacher_target_chunk"])
1121:         for epoch in range(1, self.epochs + 1):
1122:             keys = self.key_index()
1123:             order = stable_order(self.queries, query_order_namespace("S", epoch))
1124:             self.student.train()
1125:             self.student.requires_grad_(True)
1126:             pending: list[torch.Tensor] = []
1127:             diagnostics: list[dict[str, Any]] = []
```

## S3：epoch结束复用旧keys

`src/mmdd_stage1_clean/train.py:1191–1207`

SHA256: `0dab311d84c65bfd0a64e58a00376a72fd5df908959e4589dca0818c0da17b63`

```python
1191:             # Keep the per-epoch checkpoint: dev selection needs every candidate
1192:             # epoch, and next-epoch mining is driven by *last*, never by best.
1193:             torch.save(
1194:                 {"model": self.student.state_dict(), "epoch": epoch, "arm": self.arm},
1195:                 self.output_dir / f"epoch_{epoch:02d}.pt",
1196:             )
1197:             if dev_ranker is not None:
1198:                 evaluation = dev_ranker(self.student, keys)
1199:                 record["dev"] = evaluation
1200:                 write_json(self.output_dir / f"epoch_{epoch:02d}.dev.json", evaluation)
1201:                 log_line(
1202:                     f"{self.arm} epoch {epoch} dev R@10={evaluation.get('overall_R10')} "
1203:                     f"implicit={evaluation.get('implicit_R10')}"
1204:                 )
1205:             if epoch < self.epochs:
1206:                 self.refresh_mining(epoch + 1, keys)
1207:         return {"epochs": self.history, "steps": step, "elapsed_seconds": time.time() - started}
```

## S4：pipeline返回外层字典

`src/mmdd_stage1_clean/evaluate.py:230–240`

SHA256: `0814121d5e802428c8c3a3e1fd794226fbf6cbd69c6b22e456e8fda7f5f3a521`

```python
0230:     def pipeline(self, query_id: str, *, use_ann: bool) -> dict[str, Any]:
0231:         """Steps 2-8 once for one query, on a single ranker path.
0232:
0233:         ``pipeline_both`` is preferred where both paths are wanted, because it
0234:         computes the conditional second hop once for them.  This entry point exists
0235:         for callers that only need one path -- a per-epoch dev check, for instance,
0236:         scores the Student's own candidates and does not need the exact index.
0237:         """
0238:         with self.timing.stage("pipeline_total"):
0239:             return self._pipeline_both_inner(query_id, paths=(("ann", True),) if use_ann else (("exact", False),))
0240:
```

## S4：调用端再次套ann

`src/mmdd_stage1_clean/commands.py:2398–2413`

SHA256: `86bb8aa28f0cf495eb7dfed327343e3aaf521f59e82299f2fef6a0837b6dc77a`

```python
2398:     fidelity: dict[str, list[float]] = {"D": [], "E_text": [], "E_image": [], "C": []}
2399:     for row in population:
2400:         query_id = row["query_id"]
2401:         with retrieval_timing.stage("retrieve_query"):
2402:             outputs = (
2403:                 engine.pipeline_both(query_id)
2404:                 if run_exact
2405:                 else {"ann": engine.pipeline(query_id, use_ann=True)}
2406:             )
2407:         result = outputs.get("ann") or outputs["exact"]
2408:         exact_result = outputs.get("exact")
2409:         if exact_result is not None and "ann" in outputs:
2410:             fidelity["D"].append(
2411:                 retrieve.nn_fidelity(
2412:                     result["D100"], exact_result["D100"],
2413:                     int(resolved["retrieval"]["direct_k"]),
```

## P1：模型hash仅取每tensor前8值

`src/mmdd_stage1_clean/commands.py:976–988`

SHA256: `86bb8aa28f0cf495eb7dfed327343e3aaf521f59e82299f2fef6a0837b6dc77a`

```python
0976: def _freeze(module: torch.nn.Module) -> None:
0977:     for parameter in module.parameters():
0978:         parameter.requires_grad_(False)
0979:     module.eval()
0980:
0981:
0982: def _model_hash(module: torch.nn.Module) -> str:
0983:     return stable_digest(
0984:         {name: [float(v) for v in tensor.detach().float().flatten()[:8]] for name, tensor in module.state_dict().items()},
0985:         len(list(module.state_dict())),
0986:     )
0987:
0988:
```
