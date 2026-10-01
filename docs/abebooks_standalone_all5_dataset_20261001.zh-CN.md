# AbeBooks：所有查询均为五行

日期：2026-10-01。当前副本：`dataset/abebooks_standalone_all5_balanced_20261001`。

**implicit 和 explicit 全部固定 5 行。implicit 只要求至少 2 行有已核验恢复证据，不要求五行全部恢复。** 各 split 保持 implicit / explicit 为 50% / 50%。

上一版 `abebooks_standalone_5row_balanced_20261001` 只改了 implicit，遗漏了 explicit。用户指出的 `query_94c8b1e80e539656983b` 属于 `st_book_009` 的 explicit 查询，实际确为两行。本次修正构造器与命令行两类查询的默认值，在新目录重新构造；原版及全部旧副本保留。新视图使用新 ID，旧 ID 不再用于当前查询输入。

## 数据规模

| split | implicit（5 行） | explicit（5 行） | 总查询数 | 不重复源行 | 独立分组 |
|---|---:|---:|---:|---:|---:|
| train | 40 | 40 | 80 | 400 | 57 |
| dev | 13 | 13 | 26 | 130 | 18 |
| test | 13 | 13 | 26 | 130 | 20 |
| 合计 | 66 | 66 | 132 | 660 | 95 |

保留全部 174 张候选表的行成员、130 张源表、1,289 条原记录和 5,441 个素材。查询数是 132，query–target 正例组合数是 159，恢复路径数是 405；viewer 按正例组合分页。

66 个 implicit 的恢复覆盖分布为：2 行的 20 个，3 行的 18 个，4 行的 12 个，5 行的 16 个。共 330 个隐式输入行，其中 222 行有核验证据，108 行未核验；未核验行不作为恢复成功、失败或确认负例。

## 构造与平衡

在原始源表内构造互不重用的五行视图。implicit 只显示书名，并排除书名中泄漏作者的情况；explicit 显示书名和完整作者。恢复证据的选择条件、候选表投影和完整作者等值连接规则沿用上一版。

先构造合格的候选视图，再按源表、重复书目和版本、相同已核验证据及近重复封面分组划分。初始共有 135 个有效五行视图：train 为 70 implicit / 11 explicit，dev、test 各为 23 / 4。

对于 implicit 较多的 split，按固定 seed=13、源表轮流选取部分五行视图，公开其源作者列，将其分配为 explicit。共重新分配 50 个视图；重新生成 ID、可见列和元数据，重新执行全候选标签检查，移除这些视图的恢复监督。每个视图只保留一种形式，源行不会同时出现在显式和隐式查询中。

每个 split 的视图总数初始均为奇数，最后各下采样一个 explicit，得到严格 1:1 的 132 个查询。逐项记录见 `audit/query_kind_assignments.jsonl` 和 `table_queryability_decisions.jsonl`。这一步不读取模型检索表现。

由于所有查询都需要五行，而合格行还必须位于同一源表并通过完整连接检查，不能直接沿用两行 explicit 版本的 232 个查询。未进入查询的原记录仍保留在源数据中，没有用重复行或短查询填补数量。

查询恢复门槛与候选标签条件分别记录：implicit 至少两行有核验证据即可；oracle 正例检查使用源真值验证五行均能正确补充目标字段、不扩张到错误书目。这不表示五行均已被模型恢复，也不会把未核验行的源值补入恢复监督。

## 验证和训练输入

- 132 个查询在数据文件、Stage-1 序列化输入及 viewer 中均为五行，显隐分别检查。
- 所有 implicit 至少两行有恢复记录，explicit 没有伪造的恢复记录或路径标签。
- 源行不重用，源组、书名版本组及已核验证据不跨 split；候选库和素材库按任务定义共享。
- 132 × 174 = 22,968 个候选关系全部判定。核验值回放得到 287 个正确行对，循环交换后正确数为 0；这是条件回放检查，不是新模型评测。
- 152 项相关测试通过，覆盖 explicit 默认五行、部分恢复，以及公开作者后不复用源行、不残留恢复标签的情况。
- 原始数据与三个历史 standalone 副本均按文件哈希检查，保持不变。

Stage-1 输入位于 `work/abebooks_standalone_all5_balanced_20261001/stage1/`。`edge_lists.{train,dev,test}.jsonl` 用于边监督；`target_lists.evidence_supervised.{train,dev,test}.jsonl` 含 40/13/13 个 implicit，用于证据路径监督；`target_lists.{train,dev,test}.jsonl` 含完整 80/26/26 个查询，用于完整 split 的评测。详细读取器检查见 `audit/DELIVERY_VALIDATION.json`，训练路径见 `TRAINING_PROTOCOL.json`。

两类查询的源作者使用方式不同：explicit 的作者是可见输入，implicit 的作者需要证据恢复。标注仍是模型辅助标注，不是独立人工金标。当前完成的是数据及训练输入准备，没有执行训练；模型收益和统计能力需要后续实验检验。历史原记录已暴露，不能把此次重划分称为未接触的外部测试集。

## 复现

输出路径必须不存在。从隔离工作目录运行：

```bash
cd /tmp/abebooks_curation_20261001
conda run -n MMDD python /home/oycy/MMDD/src/build_abebooks_standalone_dataset.py \
  --dataset /home/oycy/MMDD/dataset/abebooks_joinability_no4_disjoint_20260930 \
  --output /home/oycy/MMDD/dataset/abebooks_standalone_all5_rebuild \
  --proposals /home/oycy/MMDD/work/abebooks_standalone_20261001/local_author_proposals.jsonl \
  --historical-reviews /home/oycy/MMDD/cache/abebooks_mm_joinability/query_recovery_auto_checks.jsonl \
  --implicit-rows 5 --explicit-rows 5 --minimum-recovered-rows 2 --balanced

conda run -n MMDD python /home/oycy/MMDD/src/build_stage1_training_data.py \
  --dataset-root /home/oycy/MMDD/dataset/abebooks_standalone_all5_rebuild \
  --output-dir /home/oycy/MMDD/work/abebooks_standalone_all5_rebuild/stage1

conda run -n MMDD python /home/oycy/MMDD/src/validate_abebooks_standalone_dataset.py \
  --dataset-root /home/oycy/MMDD/dataset/abebooks_standalone_all5_rebuild \
  --stage1-dir /home/oycy/MMDD/work/abebooks_standalone_all5_rebuild/stage1
```

使用新的表特征和缓存命名空间，训练仅使用 train，checkpoint 选择仅使用 dev，test 用于最终报告。评测应分别报告显隐结果、恢复行覆盖、完整作者值准确率以及实际连接质量，implicit 的恢复门槛保持至少两行。
