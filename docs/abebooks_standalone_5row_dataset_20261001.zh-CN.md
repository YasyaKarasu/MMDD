# AbeBooks 五行 implicit 数据集

日期：2026-10-01。历史副本：`dataset/abebooks_standalone_5row_balanced_20261001`。

此版仍含两行 explicit，已被[所有查询均为五行的版本](abebooks_standalone_all5_dataset_20261001.zh-CN.md)替代。以下保留历史构造与统计，不代表当前 viewer 加载的数据。

**每个 implicit 查询固定 5 行，只要求至少 2 行有已核验的作者恢复证据。** 其余行可以没有合格恢复标注，不要求五行都能恢复。explicit 沿用两行；train、dev、test 各自保持 implicit / explicit 为 50% / 50%。原数据及两个历史 standalone 版本均保留。

## 实际规模

| 划分 | implicit（5 行） | explicit（2 行） | 查询总数 | 不重复源行 | 独立分组 |
|---|---:|---:|---:|---:|---:|
| train | 70 | 70 | 140 | 490 | 50 |
| dev | 23 | 23 | 46 | 161 | 18 |
| test | 23 | 23 | 46 | 161 | 18 |
| 合计 | 116 | 116 | 232 | 812 | 86 |

116 个 implicit 查询的已核验恢复覆盖如下：

| 每个查询有证据的行数 | 查询数 |
|---|---:|
| 2 / 5 | 29 |
| 3 / 5 | 31 |
| 4 / 5 | 24 |
| 5 / 5 | 32 |

implicit 共 580 行，其中 407 行有合格恢复证据，173 行未核验。**这 173 行不标为恢复成功，也不当作恢复失败或负例。** 29 个只有两行恢复证据的五行查询已实际进入数据集。

新版本保留 174 张候选表、130 张源表、1,289 条原记录和 5,441 个素材，包含 286 条 query–target 正例和 711 条恢复路径。路径数会因同一事实有多份证据、多个合法目标而增加，不等同于独立查询数或恢复行数。

五行查询占用更多源行，所以总数低于历史两行平衡版的 392 个查询。本版没有重复组合相同行凑数量；同一源行只进入一个查询。划分已按新查询重新生成，不能按旧查询 ID 对齐评测结果。

## 构造规则

1. 在原始图书源表内筛选有效书名、完整作者值和可产生正确记录连接的行。每张源表采用固定 seed=13 的行顺序，不重组跨源表查询。
2. implicit 隐藏作者，只显示书名。排除书名直接泄漏作者的情况，包括所有格和跨行提示。先为每个五行视图预留两行已核验证据，再用未使用行补足五行；有剩余合格证据的行优先补入，但不把五行均有证据作为门槛。
3. 从剩余源行构造 explicit 两行视图，显示书名和作者。任何源行都不同时进入显式和隐式查询。
4. 以源表、重复书目及版本、相同已核验证据、近重复封面形成分组，约按 60/20/20 分配。随后仅在各 split 内下采样 explicit，保留全部 implicit。下采样前分别为 70/103、23/36、23/36，最终每个 split 均为 1:1。
5. 保留原来的 174 个候选表成员集合。图书候选暴露作者、移除书名，与查询形成互补字段视图；53 张候选补入源表原有作者列，43 张移除书名，其他保留单元格不变。

连接使用完整作者列表的保守规范化等值匹配，不采用部分作者列表、模糊别名或源行 ID。来源身份只用于审核执行后的行对是否正确。

查询行数、恢复覆盖和候选标签是三个不同口径：五行是模型实际输入；至少两行是恢复证据入选条件；候选表的 oracle 正例仍要求使用源作者真值时，所有查询行均有正确且有用的连接，不产生错误书目扩张。后者是候选标签检查，不表示模型必须恢复全部五行，也不会将未核验行的源作者值注入恢复监督。

## 监督与验证

每条 implicit 元数据分别记录 `selected_rows=5`、实际 `recovered_rows`、`required_recovered_rows=2` 和 `unreviewed_rows`。恢复路径只写入有合格证据的行。

复用已有 2,249 份盲读提案和历史审核记录，共 465 个源行通过作者证据资格检查，本版实际使用其中 407 行。它们是模型辅助标注，不是独立人工金标；本次布局修正没有重新运行模型读取，也不声称新查询经过了新一轮视觉审核。

Stage-1 构造继续使用 `recovery_records_only_no_provenance_fallback`：不把同源但未核验的素材补成正例。`query_row_count` 对 implicit 为 5；`positive_evidence_rows_by_target` 只包含实际有恢复证据的行。完整评测文件包含全部查询，路径监督文件仅含 implicit 查询。

已完成的核验：

- 232 × 174 = 40,368 个 query–target 关系全部判定，保留所有合法目标。
- 无源行重复使用；源组、书名版本组和已核验证据无跨 split 重叠。三个 split 仍共享完整候选库和素材库。
- 原始读取值在已核验行上回放，得到 505 个正确连接行对；按查询循环交换这些值后，505 个行对中正确数为 0。710 个 oracle 行对包含未核验行，不能充当已恢复行对数。
- 149 项相关测试通过，包含“5 行仅 2 行有证据可入选、减少到 1 行则校验失败”以及源行不复用的测试。Stage-1 三个 split 的实际输入和读取器检查见 `audit/DELIVERY_VALIDATION.json`。
- 原始数据、历史 standalone 数据、历史 balanced 数据均按构造前后的文件哈希核验，未被修改。

## 使用与复现

Stage-1 文件位于 `work/abebooks_standalone_5row_balanced_20261001/stage1/`：

- `edge_lists.{train,dev,test}.jsonl`：有向边监督。
- `target_lists.evidence_supervised.{train,dev,test}.jsonl`：70 / 23 / 23 个有恢复标注的查询，供路径监督使用。
- `target_lists.{train,dev,test}.jsonl`：完整的 140 / 46 / 46 个查询，供各 split 的检索评测使用。
- `stage1_objects.jsonl`、`stage1_corpus.jsonl`：全部查询输入、目标表及素材库。

新路径必须不存在。以下离线命令从隔离工作目录执行，数据构造时复用已有提案：

```bash
cd /tmp/abebooks_curation_20261001
conda run -n MMDD python /home/oycy/MMDD/src/build_abebooks_standalone_dataset.py \
  --dataset /home/oycy/MMDD/dataset/abebooks_joinability_no4_disjoint_20260930 \
  --output /home/oycy/MMDD/dataset/abebooks_standalone_5row_rebuild \
  --proposals /home/oycy/MMDD/work/abebooks_standalone_20261001/local_author_proposals.jsonl \
  --historical-reviews /home/oycy/MMDD/cache/abebooks_mm_joinability/query_recovery_auto_checks.jsonl \
  --implicit-rows 5 --explicit-rows 2 --minimum-recovered-rows 2 --balanced

conda run -n MMDD python /home/oycy/MMDD/src/build_stage1_training_data.py \
  --dataset-root /home/oycy/MMDD/dataset/abebooks_standalone_5row_rebuild \
  --output-dir /home/oycy/MMDD/work/abebooks_standalone_5row_rebuild/stage1

conda run -n MMDD python /home/oycy/MMDD/src/validate_abebooks_standalone_dataset.py \
  --dataset-root /home/oycy/MMDD/dataset/abebooks_standalone_5row_rebuild \
  --stage1-dir /home/oycy/MMDD/work/abebooks_standalone_5row_rebuild/stage1
```

各训练阶段仅使用 train，选择 checkpoint 使用 dev，test 用于最终报告。重新编码新表输入，使用新缓存命名空间；分别报告显隐检索结果、恢复行覆盖、完整值正确率和实际连接质量，查询恢复门槛为至少两行。具体路径见副本内 `TRAINING_PROTOCOL.json`。

本次完成数据和训练输入准备，没有启动训练或证明训练收益。232 个查询仍是小规模实验集；原始记录历史上已被使用，重新划分不等于获得未暴露的外部测试集。分组数和测试规模应如实保留在实验报告中。
