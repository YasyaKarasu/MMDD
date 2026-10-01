# AbeBooks 独立训练、验证、测试数据集

日期：2026-10-01。数据副本：`dataset/abebooks_standalone_20261001`。

本文件记录历史两行版。当前版本已改为“所有查询五行、implicit 至少两行有恢复证据、各 split 显隐各半”，见[五行版报告](abebooks_standalone_all5_dataset_20261001.zh-CN.md)。以下统计仅属于本历史版本。

本版按“仅使用 AbeBooks 完成当前 Teacher/Student 的训练、验证和测试”重建，替代此前不合格的 13-query 案例集。输入和监督已准备好，已通过现有训练读取器检查；本次没有执行模型训练，不能将数据构造检查当作训练收益。

## 规模与划分

| 划分 | query 总数 | implicit | explicit | 不重复使用的源行 | 独立分组 | 源表组 |
|---|---:|---:|---:|---:|---:|---:|
| train | 273 | 118 | 155 | 546 | 50 | 68 |
| dev | 92 | 39 | 53 | 184 | 18 | 24 |
| test | 92 | 39 | 53 | 184 | 18 | 25 |
| 合计 | 457 | 196 | 261 | 914 | 86 | 117 |

每个 query 为两个 example row，每条源记录只进入一个 query；没有反复组合相同行、复制 query 或将显隐两个版本重复计算。914 条使用行对应 904 个按书名归并的书目组，不能将 457 个 query 全部解释为相互独立的实验重复。

保留全部 **174 张候选表、130 张源表、1,289 条原记录和 5,441 个素材**。源表、实体、素材目录等保留文件与原版逐字节一致，原数据集全部文件哈希也保持一致。图片继续使用原本的本地路径，只读共享，没有修改图片。

数据包含 577 条 query–target 正例、719 条证据恢复路径、392 个实际使用的不同作者恢复事实。证据路径数不是独立 query 数，也不是独立事实数。

## 相比原报告的必要变化

原报告以少量机制案例为目标，优先固定目标表内容；本次用户明确要求独立训练，因此采用新的任务版本，不能与原版 Recall 直接比较。

- 候选身份一一对应原来的 174 张表，行成员保持不变，没有出版社分组或缩小候选湖。
- 53 张图书目标表补入其原源表已有的作者列；43 张移除书名列。其他已有单元格保持原值，共 96 张表改变投影。投影后仍有 174 种不同表输入。
- implicit query 只提供书名；explicit query 提供书名和作者。只隐藏作者却在书名中暴露作者的样本，归入 explicit；规则覆盖 `Joe Celko's ...` 这样的所有格。
- 使用完整作者列表的保守规范化等值连接：支持明确的姓、名倒序和作者列表顺序，不补全首字母、不做模糊别名匹配、不把部分作者列表当作完整匹配。
- query 的两个作者连接值必须不同。正例要求执行连接后两行都获得非空的新字段，且不扩张到其他书目记录。只有一行连接成功不符合本版的完整两行任务定义。
- 全部 query 和 target 使用新 ID 命名空间。需要重新生成表特征、训练缓存和 ANN，不能沿用旧 checkpoint 作为本版新训练的结果。

这是一项“在保留候选行分组的互补字段视图中，通过作者字段完成书目记录”的受控任务；它不证明作者能在任意真实书库中唯一标识图书。显式样本使用原记录可见作者值，其作者字段没有被自动宣称为外部权威书目真值。

## 证据标注

本地 `Qwen3.5-9B` 对 1,157 张封面和 1,092 份作者简介进行了离线读取，总计 2,249 份。读取输入仅有图片或文本，不提供源作者答案。之后再检查完整作者列表是否与源值一致、封面书名是否匹配、作者与编辑角色是否混淆，以及简介中是否明确出现姓名。

465 个源行通过资格检查：151 个与历史读取结果一致，109 个封面和简介相互支持，205 个由本次封面读取与源值一致支持。这些强度标签是溯源信息，不是独立人工金标，也不保证读取模型没有错误。每个 implicit query 的两行均具有合格封面证据；简介仅在额外通过核验时作为正例。

另外进行了 72 个不同封面的 Codex 视觉抽查，所见完整作者列表与源值一致；其中发现作者所有格书名泄漏并修正了统一规则。抽样和最终 split 去向见 `audit/visual_spotchecks.jsonl`。这仍属于模型辅助复核，不能写成独立人工标注质量为 100%。原始输出、排除原因和标注来源保存在 `audit/evidence_audit.jsonl`。

未通过资格检查的素材继续保留在检索库中，没有按模型得分删掉。它们不自动成为可信负证据。Stage-1 导出器识别本版 manifest 的 `recovery_records_only_no_provenance_fallback` 策略，只从核验恢复记录建立证据正例；同源资产不会再自动冒充 query 的证据正例。候选证据池使用来源信息，不使用留出集恢复标签来挑选训练负例。

## 划分和验证

以源表为最小边界，将相同实体、书名及其版本、相同已核验素材、近重复封面进一步连接成分组，固定 seed=13，按约 60/20/20 分配。只平衡类型和数量，不读取检索表现。最大分组包含 44 个 query；报告区间或显著性时应按分组计算。

已验证源组、源行、书名版本组、已核验正例证据不跨 split；3 对相似封面参与了分组。检索库仍按任务定义由三个 split 共享，未核验的作者简介、介绍、摘要及卖家图片中仍有重复内容，具体数量见 `audit/DELIVERY_VALIDATION.json`。不应将“监督分组隔离”写成“全部素材互不相同”。

对 457 × 174 = **79,518** 个 query–target 关系逐一执行连接判断，补齐所有满足任务定义的正例。无未判定候选进入主任务。来源 ID 只用于核验行对，等值连接和模型输入均不使用它。多正例评测应区分 Recall@k 与 Hit@k。

原始证据读取值回放与循环交换检查见 `VALIDATION.json`：合格值能重现全部正确行对，交换两行值后正确行对为零。这是经过资格筛选后的连接一致性检查，不是无偏属性抽取准确率，也不是端到端检索结果。

174 项相关测试通过；三套 split 的 edge 和 target 文件均通过现有读取器检查。完整命令、导出数量和文件完整性结果见 `audit/DELIVERY_VALIDATION.json`。

## 训练输入与复现

可以直接使用数据副本中的 `dataset_manifest.json`、`splits/*.queries.jsonl`、`splits/*.qrels.jsonl`、`splits/*.recoveries.jsonl`。三个 split 均只来自 AbeBooks，不依赖 EntiTables 或 WDC 的训练样本。

已导出的 Stage-1 文件位于 `work/abebooks_standalone_20261001/stage1/`：

- `stage1_objects.jsonl` 和 `stage1_corpus.jsonl`：新 query、全部目标表及完整证据库。
- `edge_lists.train.jsonl` / `.dev.jsonl` / `.test.jsonl`：各 split 的五种有向关系监督，用于 edge 阶段。
- `target_lists.evidence_supervised.train.jsonl` / `.dev.jsonl` / `.test.jsonl`：仅有合格恢复证据的 118/39/39 个 query，用于 evidence/path 监督。explicit 没有恢复标注，不能伪造路径金标。
- `target_lists.train.jsonl` / `.dev.jsonl` / `.test.jsonl`：全部 273/92/92 个 query，用于完整候选评测，分别报告 implicit、explicit 和总体结果。不要将证据监督子集误当作完整测试分母。

特征需要针对新表输入重新编码。Teacher/Student 各训练阶段仅使用 train，checkpoint 选择仅使用 dev，test 留给最终报告。冻结底层编码器、训练原方案中的 Teacher 与 P/R 等组件；是否得到改善要靠后续实际训练和学习曲线判断。本次没有改变模型、损失或融合方法。

保留的训练规模是当前小数据任务的规模，不足以保证大模型从零训练成功或支持大规模自然数据湖泛化。历史构造时的 train≥200、dev/test≥50 下限不代表统计充分性；当前构造器已改为独立配置查询行数和最低恢复行数，并如实报告实际规模。

复现构造需先在新路径准备、读取素材，然后构造另一份副本（输出目录必须不存在）。以下命令从隔离工作目录执行：

```bash
cd /tmp/abebooks_curation_20261001
conda run -n MMDD python /home/oycy/MMDD/src/build_abebooks_standalone_dataset.py \
  --dataset /home/oycy/MMDD/dataset/abebooks_joinability_no4_disjoint_20260930 \
  --output /home/oycy/MMDD/dataset/abebooks_standalone_rebuild \
  --proposals /home/oycy/MMDD/work/abebooks_standalone_20261001/local_author_proposals.jsonl \
  --historical-reviews /home/oycy/MMDD/cache/abebooks_mm_joinability/query_recovery_auto_checks.jsonl \
  --implicit-rows 2 --explicit-rows 2 --minimum-recovered-rows 2

conda run -n MMDD python /home/oycy/MMDD/src/build_stage1_training_data.py \
  --dataset-root /home/oycy/MMDD/dataset/abebooks_standalone_20261001 \
  --output-dir /home/oycy/MMDD/work/abebooks_standalone_20261001/stage1

PYTHONPATH=/home/oycy/MMDD/src conda run -n MMDD python \
  /home/oycy/MMDD/work/abebooks_standalone_20261001/validate_delivery.py
```

原始输入和旧标签保存在 `provenance/before_standalone/`；逐表修改见 `audit/target_projection_changes.jsonl`，全湖标签依据见 `audit/candidate_judgments.jsonl`。原始书目历史上已参与过实验，重新划分无法把它变成从未接触的外部测试集。新训练必须使用新的初始化和命名空间，并如实披露历史暴露。
