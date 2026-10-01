# 新版 AbeBooks 数据实验记录

目标：在原方法的 selected KD Student 全局 RRF 排名上，test 的 query-macro
Recall@10 ≥ 0.40，implicit Recall@10 ≥ 0.10。**本轮固定候选后的 test 结果为
0.410714 / 0.142857，两个门槛均已达到，并从原始 qrels 独立复算一致。**

选中数据副本：`dataset/abebooks_context2_selected_20261001`。
训练与评测：`work/abebooks_context2_columns_lr10_20261001`。
检查点：`main/C2_KD/snapshot_frac000.pt`，SHA-256 为
`f8ce16ecad402d356f7baab38a2c700b1bad8d1ff97de460dd8700e63a62ce1e`。
该检查点继承已训练 C1 的 0.2 进度，C2 原选择规则选中 0.0；不能称为 KD 带来的收益。

最终副本只规范化显式 qrel 名称，并从源表删去 `edition_number`、
`goodreads_rating_count`。所有 5,753 个编码对象及查询、目标、证据、qrels、
恢复事实均与所选训练输入核对相等。源表列数与统计元数据已刷新，原始列索引
保留。最终副本不是根据 test 排名修改的。

## 固定候选后的 test 对照

在最后一个 dev 实验之前登记双指标选择规则：先最大化
`min(dev overall / 0.4, dev implicit / 0.1)`，再比较 overall、implicit。
选中 columns_lr10 后，先写入 `ROUND1_FINAL_CANDIDATE_FREEZE.json`，再首次评测
test。随后评测 compatible 和 columns 作固定对照，没有根据这些 test 结果
更换数据、超参数或检查点。

| 固定设置 | test RRF R@10 | test implicit R@10 | test Direct R@10 |
|---|---:|---:|---:|
| compatible 基线 | 0.357143 | 0.000000 | 0.285714 |
| columns，默认学习率 | 0.321429 | 0.000000 | 0.196429 |
| **columns，十倍 P/R 学习率** | **0.410714** | **0.142857** | **0.160714** |

共评测全部 28 个 test 查询，其中 14 个 implicit。这里使用 query-macro
Recall@10；多正例查询按命中正例比例计分。相对兼容基线，总体提高 0.053571，
implicit 提高 0.142857；不能将提升单独归因于删列，默认学习率的删列对照在
test 上反而退化。P/R 学习率分别为 1e-5/1e-4，其他方法组件不变。

约束审计：全部 82 个 train 查询进入 TA/TB/C2 构造器；每个 split 显隐各半，
所有 query 五行两列；174 个候选表与所有行成员保持不变，全部 24,012 个
pair 判定保持不变。146 个正例目标列组合至少有 5 个不同有效连接值，所有
正例目标至少两列。候选池原有一个单列 seller-region 非正例表，逐字保留，
没有通过制造单值连接列简化任务。没有新增标注。

复评已有检查点（从隔离目录运行，避免隐式加载用户环境文件）：

```bash
cd /tmp/abebooks_opt_20261001
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 conda run -n MMDD python \
  /home/oycy/MMDD/src/run_abebooks_data_ablation.py evaluate \
  --run-root /home/oycy/MMDD/work/abebooks_context2_columns_lr10_20261001/main \
  --splits dev test --device cpu --retrieval-only
```

`--retrieval-only` 只跳过 Teacher 重排诊断，调用相同的 Direct/RRF 召回函数，
使用独立的 `retrieval_recall_summary.json` 保存结果；所有 dev 生成器及四个 k
的召回与完整评测逐项相等。训练仍在 GPU 上完整执行。
最终核对记录在 `work/abebooks_optimization_20261001/ROUND1_FINAL_AUDIT.json`。

原始版本是 `dataset/abebooks_authors_publishers_context2_all5_20261001`。
138 个查询，train/dev/test 为 82/28/28，每个划分显隐各半；查询均为五行、
两列。174 个候选表，5,441 个证据素材，164 条正例关系。前一会话只完成
数据构造和验证。原文件哈希与方法源码哈希记录在
`work/abebooks_optimization_20261001/INITIAL_STATE.json`。

## 先修复数据格式兼容性

新构造器输出的显式标签原因是 `explicit_join_column`，现有 CQET 方法只识别
`explicit_visible_join_column`。前者会被归为 mixed，并丢失 Direct 正例集合。
已修复数据构造器的输出名称，增加实际调用现有 CQET 标签读取器的回归测试。
没有修改模型、损失、采样、融合或选择规则。

`dataset/abebooks_context2_compatible_20261001` 是规范化副本，83 条显式
qrel 只改标签名称，所有查询、单元格、证据和正例关系保持不变。最初使用
不兼容标签启动的 `work/abebooks_context2_baseline_20261001/main` 在训练中止，
不能作为有效对照。它的冻结 Qwen 特征在逐对象输入相等检查后复用；所有
可训练模型重新初始化训练。

## 数据控制与已完成的 dev 结果

共同设置：seed 13，原 TA/TB/Native C1/C2 SUP/KD 流程，全部 82 个训练查询，
Student 10 epochs、batch 16，默认 P/R 学习率 1e-6/1e-5。检查点沿用现有
dev 选择；没有人工筛选训练样本，没有增加标注，也没有使用 test 指标选数据。

| 副本 | 变更 | dev 总体 R@10 | dev implicit R@10 |
|---|---|---:|---:|
| compatible | 仅兼容标签名称 | 0.285714 | 0.071429 |
| columns | 原表删除 edition_number、goodreads_rating_count | 0.339286 | 0.071429 |
| hubs | 删除 59 个训练热门无标注文本素材 | 0.321429 | 0.071429 |
| columns_hubs | 删列与文本过滤组合 | 0.339286 | 0.071429 |
| natural | 660 个源作者字段改为无歧义自然顺序 | 0.339286 | 0.071429 |
| columns_lr10 | columns 数据，现有 P/R 学习率增大十倍 | 0.375000 | 0.071429 |
| columns_lr100 | columns 数据，现有 P/R 学习率增大百倍 | 0.339286 | 0.071429 |
| natural_columns | 自然作者顺序与删列组合 | 0.357143 | 0.071429 |
| columns_titles | columns 数据，文本补回原书目标题 | 0.232143 | 0.000000 |

目录前缀均为 `dataset/abebooks_context2_` / `work/abebooks_context2_`，
后缀均为 `_20261001`。每组都保留 138 个查询与 174 个目标表。

源字段删除采用 `project_abebooks_sources.py`，重新使用 `project_rows` 从原表
生成视图，执行全部 24,012 个 query–target 判定，拒绝改变既有连接判定的
投影。版次缺失率约 72%；同时删除重量会损坏一个原正例，已拒绝，未删查询
来掩盖这个问题。源表的列索引保留，既有事实的属性引用保持有效。

文本清理使用基线 Raw QE20 在训练集上的频率：至少出现在 9/82 个训练
查询中，且不属于任何既有标注证据的内容别名组。删除 59 个素材、47 个
内容组，占原素材约 1.1%。保留全部图像与全部已标注证据，不把“未标注”
解释为已证明错误。频率来自 train；全划分既有标注只用于保护证据。

作者显示规范化只接受完整作者匹配键不变的转换，含糊多人名单不处理。
例如 `Ousterhout, John K.` 变为 `John K. Ousterhout`。source、query 和
target 使用同一原表投影；没有增加作者、改变书目身份或标签。

学习率对照事先登记在 `EXPERIMENT_PLAN.json`，仍使用原始检查点选择规则。
十倍学习率的 C1 选中 0.2、C2 选中 0.0；不能将其提升归因于 KD。
十倍学习率训练到终点时 SUP/KD 的总体召回分别降到 0.267857/0.196429，
implicit 均未改善，保留此退化结果。百倍学习率两阶段均选中初始化点。
natural_columns 的 C1 选中终点、C2 选中初始化；唯一 implicit 命中同时在
Direct top-10 中，新增的有正确证据支持的 implicit 召回为 0。

`columns_titles` 对全部 3,374 个原表关联文本片段统一添加其所属书目的
`Book title` 标题；完整正文、2,067 张图像和 354 条既有恢复事实均保留。
不添加隐藏作者或出版社，不依照 query、划分、标签或排名决定是否添加。
这是内容变换实验，所有变化的文本需重新编码；历史标注并非重新抽取。
查询、目标表及 qrels 与 columns 副本逐字节一致，内容哈希与变换来源见
该副本的 `TEXT_TRANSFORMATION_AUDIT.json` 和 `SOURCE_PROJECTION.json`。
此对照显著退化，正确证据覆盖单元降至 10，未采用。标题使原本相同的
文本正文成为更多独立内容对象（canonical assets 从 3,660 增至 5,383），
这是该控制的结构性变化，不能仅归因于句子语义。

## 机制与负面结果

最终模型在 test 上有两个 implicit 命中，Direct 有一个。已知正确证据保留
覆盖为 15/51 个已标注 query–target–row–attribute–value 单元。两个命中中，
一个保留了已有正确证据，同时也被 Direct 命中；另一个由 RRF 新增，但没有
保留既有标注中的正确证据。因此新增、有已标注正确证据支持且隐藏值不可见的
implicit Recall@10 为 0，不能把它描述为已验证的属性补全收益。未标注证据
也不能直接判错，本轮没有补充新标注来解释这个命中。

源数据资格检查中的条件值回放保留：258 个 observed-value 连接均正确，
对应 swapped-value 控制为 0/258。这是既有标注的条件回放，**不是新做的盲测
属性抽取或端到端 verifier 测试**。Stage-1 召回门槛达标与机制成立是不同结论。

兼容基线 dev 的唯一 implicit top-10 命中是 Direct 未命中、RRF 命中，保留
了既有正确证据；没有可见的已知隐藏值。删列将保留证据中的已标注
row–attribute–value 覆盖单元从 12 增至 16，但 implicit top-10 命中数尚未提升。
这是覆盖诊断，不代表新做了盲测属性恢复。

固定检查点的证据库诊断保存在 `BASELINE_CATALOG_DEV.json`、
`COLUMNS_CATALOG_DEV.json`、`ALL_HUBS_FIXED_DEV.json` 中。更激进的 5% 热门
证据过滤降低总体召回；仅保留图像也未同时达到两个目标。它们不是重训
结果，不用于替代主结果。当前 baseline 的 C1/C2 选择均落在初始化点，
因此不能据此宣称蒸馏已带来收益。

183 项相关测试通过；后续源表元数据与文本变换更新又运行了 44 项相关测试，
均通过。运行目录为隔离的 `/tmp/abebooks_opt_20261001`。
原始数据的 110 个文件和 80 个方法源码文件均已核对哈希不变。详细实验
状态见 `work/abebooks_optimization_20261001/PROGRESS.json`。历史语料暴露、
模型辅助标注和小样本限制仍然适用。test 只有 14 个 implicit 查询，且仅运行
seed 13；不把一次达标解释为稳定的总体性能保证。
