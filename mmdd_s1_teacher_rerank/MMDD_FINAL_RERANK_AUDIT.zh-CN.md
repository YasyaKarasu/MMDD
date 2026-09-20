# MMDD FINAL_RERANK：独立实现审计与机制分析

审计日期：2026-09-16。输入仅为本次上传源码与 FINAL_RERANK 结果包。本报告不以旧对话或 GitHub 当前版本代替执行源码，不使用外部实验数字。本次审计没有训练、retrieval、挖负例、ANN 建库或拟合 fusion 权重。

## 0. 结论摘要

**现有三种 Path-only 方案在这些固定 C100 的 dev 实验上，都不适合替代冻结 QT reranker。这个结论主要来自同支持集内的排序失败，而不仅是 no-path 正例不能返回。**

B13 的 QT-full / QT-on-P / Student-D1 / Student-LSE / Teacher-LSE R@10 分别为 47.33 / 42.01 / 21.13 / 13.08 / 8.89%。Teacher-LSE 相对 QT-full 的 -38.44pp 中，-33.12pp 出现在同一个可返回集合 P 内；只有 -5.32pp 是限制到 P 后的净变化。

对于本轮导出的固定历史207对，B13有101对进入C100，且这101对全部有路径。QT-full Top10留75对，QT-on-P留80对，Teacher-LSE仅留30对；QT-on-P Top50保留全部101对。**本轮证据不支持“QT普遍压掉了已经被Evidence带进小池的strict目标，所以应该直接用Path替代QT”的假设。**

但不能据此推出“Evidence在final ranking永远没有独立决策价值”，也不能将失败唯一归因于“Teacher没有Path训练”。包中没有 verified witness 标注、pre-retention bags 和原始语义内容，无法区分错误路径输入与正确路径打分失败。Teacher-LSE还在固定207对上在所有端点都多于Student-LSE保留正例，因此“Teacher scorer处处更差”也不成立。

## 1. 审计范围、执行与证据等级

| 检查项            | 已证实                                                                                                                     | 边界 / 缺陷                                                                                     |
|:------------------|:---------------------------------------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------------|
| Archive 与执行    | 两个 zstd 压缩完整性检查通过；结果包 29 文件，源码 426 文件；22 项已打包账本 hash/size 匹配。                              | 没有独立 PACKAGE/MANIFEST；原始执行命令、stdout、执行时源码 hash/commit 未随包提供。            |
| C100 与 admission | 5990 个 endpoint-query 的100个候选均唯一；从 Direct100、E 排名按生产 Equal RRF60 重建 C100 全部一致。                      | 原始 own retrieval 文件本体未随包提供，不能从全湖重放 retrieval。                               |
| Student-D1        | 取生产 evidence_score；5990 个 D1 排名等于 E 排名限制到 C100；无重新定义 coverage。                                        | 缺少 Q-row support tensors，不能独立从行特征重算 coverage 数值。                                |
| Student-LSE       | 933210 个 retained path occurrence 中，QE+ET 及逐 target LSE 可复算；不用 pre-retention full bag。                         | 原始 pre-retention bags 未打包。                                                                |
| Teacher-LSE       | 四种真实关系的调用实现存在；全部264728个唯一 QE/ET pair 有分数；跨 endpoint 重复 pair 完全一致；sum/LSE/ranking 均可复算。 | 最终阶段全部为缓存命中；没有权重、原始 sqlite cache 和特征文件，不能独立重做模型 forward。      |
| 支持集与 fallback | 每个 query 的三种 Path ranker 都恰好在同一个 P 上排序；QT-on-P 用同一 QT score；没有 no-path QT fallback。                 | C100哈希和path哈希在原程序各view间来自共享变量，单凭原C1/C2标记并非独立验证；本次另作完整重建。 |
| 模型身份和统计    | B13 seed=null 单独报告；B4和F-P各两个真实 endpoint；query-macro Recall及先同query平均再source-group bootstrap均正确。      | dev-set结果；两个Student seed不能代表Teacher seed不确定性或held-out泛化。                       |
| strict207         | 五个 endpoint 的207对完全一致；是GT且B13中在E、不在ANN100；不是213改名。                                                   | 原始exact100和历史baseline文件缺失；仅导出的布尔标记不能独立证实exact排除。                     |
| 失败机制诊断      | 发现 no_correct_path / correct_path_below_competitor 命名没有验证 witness。                                                | 这是机制标签错误，不改变已保存排名和Recall。最高非GT竞争者的margin也不是Top10错误判据。         |

### 1.1 数值检查规模

独立遍历了5990个endpoint-query、599000个C100 target、933210个retained path occurrence、264728个唯一Teacher QE/ET pair。重算的252个原主表Recall单元与原表最大差小于1.7e-16；原bootstrap估计与区间最大差小于1.2e-16。所有新诊断均由相同raw artifacts生成，不调用上传源码的指标函数。

这意味着**没有发现足以推翻当前主指标的候选池、路径membership、排序、Recall计算或seed平均错误**；不意味着已经验证未打包的所有上游文件和实际GPU forward。

审计记录：[INDEPENDENT_AUDIT_CHECKS.json](INDEPENDENT_AUDIT_CHECKS.json)、[artifact_hashes.csv](artifact_hashes.csv)、[additional_integrity_checks.json](additional_integrity_checks.json)。

### 1.2 Teacher是否真的完成、是否真的100%覆盖

EXECUTION_LEDGER 与所有端点状态均为complete；264728个所需唯一directed pair全部有有限数值，无blocked或proxy字段被实际使用。四种关系计数：table→text 27824，table→image 17608，text→table 177347，image→table 41949。

执行账本的 consumed=264728，computed_this_run=0，hidden-state backfill objects=2904。**本轮最终打分过程全部复用了缓存，不是现场重新forward了264728对。** 源码确实调用真实 `teacher.score_pairs(source,dest)`，而不是QT替身或Student proxy；同一Teacher实例和namespace供四种关系使用。

记录的Teacher为 `fresh_lineage/T1-B/seed13/checkpoints/step_010536.pt`，SHA256：`ab0e3c3f85f006d2fdc4ba5194a0021680ab8fa1341441cb8eb003410ded68cc`。实际权重、sqlite缓存和feature manifest/tensor没有打包。`has_teacher_features` 在有索引时检查manifest成员，不是逐tensor读取。因此可证明的是**retained pair分数覆盖100%**；实际特征文件覆盖和checkpoint数值来源仍需原缓存/特征文件或独立forward补验。

### 1.3 执行版本和缓存的小问题

上传源码已经直接审查并记录hash，但执行账本未记录执行时源码hash/commit/命令/stdout，不能证明历史运行的每个源码字节就是当前压缩包版本。缺失的是provenance，不是已经观察到版本冲突。

不同endpoint的历史QT缓存对同一(q,t)有微小差异：最大0.000310421，重复pair差值中位数约1.43e-5。将同pair统一为首次出现分数重排，只有两个endpoint各1个query的Top10成员发生变化，**所有endpoint的正例R@10/20/50均不变**。其量级与数值抖动相容，但没有运行日志，不能确认具体成因。每个endpoint内QT-full与QT-on-P确实使用同一分数。这不是本轮巨大差距的解释。

### 1.4 必须修正的诊断标签

`evaluate_final_path_rerank.py:900–909`以“正target是否有任意retained path”决定 `no_correct_path` / `correct_path_below_competitor`，没有验证evidence是否为annotated witness。应更名为 `no_retained_path`、`retained_path_exists_witness_unknown` 等，并另设verified字段。不能把没有label当作“没有正确witness”。

此外，该标签使用正例相对最高非GT competitor的margin；margin<0只说明至少有一个竞争者更高，不等同于Top10失败。非GT target也不一定是经过人工确认的false join。

原 `LIMITATIONS.md` 已承认dev-set、D1/LSE聚合不同、按最高路径分模态有选择偏差；本报告不把这些写成原报告隐瞒的问题。但原报告没有完成同支持集对照，也不能从现有标签判断正确witness是否存在。

### 1.5 源码定位

- `evaluate_final_path_rerank.py:149–193, 296–339, 374–500, 576–631, 900–915, 1170–1313`：身份、候选验证、Teacher调用、路径分数、失败标签、统计。
- `evaluate_stage1_r26.py:28–45`：生产D1/row coverage，topL20/budget4及retained bags。
- `run_stage1_r11_task_e.py:207–296`：行支持、sigmoid(pathscore)和greedy coverage；D1含有额外Q-row覆盖信息，不仅是换一个LSE公式。
- `mmdd_stage1/r26_metrics.py:8–47`：query Recall与Equal RRF60。
- `run_stage1_r19.py:282–308,455–472,645–652`：真实pair Teacher、权重读取与edge-listwise loss。
- `run_stage1_r22_f1.py:485–581`：当前可见训练recipe仍为edge lists，TT加入hard candidates，没有target path-bag objective；实际训练receipt未提供。
- `mmdd_stage1/features.py:504–513`：feature manifest membership检查。

含行号的审阅片段：[SOURCE_AUDIT_EXCERPTS.md](SOURCE_AUDIT_EXCERPTS.md)；完整被审阅模块位于 `audited_sources/`。这些是本次上传源码，不是仓库远端版本。

## 2. 独立复算主指标

每个endpoint共有1198个query，implicit/explicit各599；共有1279个GT (q,t) pair。先逐query计算 `|G∩TopK|/|G|`，再macro平均。共有1000个source_table_id group。下表每格依次为R@10/R@20/R@50，单位%。QT-on-P是本次新增控制；后三种Path本来就只返回P中的targets。

### 2.1 五个真实端点，不合并身份

| Endpoint       | Scorer      | Overall R10/20/50 (%)   | Implicit R10/20/50 (%)   | Explicit R10/20/50 (%)   |
|:---------------|:------------|:------------------------|:-------------------------|:-------------------------|
| Historical-B13 | QT-full     | 47.33 / 53.87 / 57.94   | 39.65 / 46.88 / 51.02    | 55.01 / 60.85 / 64.86    |
| Historical-B13 | QT-on-P     | 42.01 / 46.96 / 48.86   | 40.28 / 46.84 / 49.64    | 43.74 / 47.08 / 48.08    |
| Historical-B13 | Student-D1  | 21.13 / 28.14 / 39.51   | 26.41 / 33.40 / 43.80    | 15.86 / 22.87 / 35.23    |
| Historical-B13 | Student-LSE | 13.08 / 21.95 / 37.35   | 19.14 / 27.71 / 41.97    | 7.01 / 16.19 / 32.72     |
| Historical-B13 | Teacher-LSE | 8.89 / 19.32 / 39.68    | 13.61 / 25.63 / 44.31    | 4.17 / 13.02 / 35.06     |
| Healthy-B4-s13 | QT-full     | 47.30 / 53.83 / 57.84   | 40.09 / 47.65 / 51.82    | 54.51 / 60.02 / 63.86    |
| Healthy-B4-s13 | QT-on-P     | 42.15 / 47.26 / 49.22   | 40.72 / 47.44 / 50.36    | 43.57 / 47.08 / 48.08    |
| Healthy-B4-s13 | Student-D1  | 21.04 / 28.04 / 39.39   | 27.05 / 33.88 / 44.71    | 15.03 / 22.20 / 34.06    |
| Healthy-B4-s13 | Student-LSE | 14.13 / 22.05 / 37.68   | 19.42 / 28.24 / 41.97    | 8.85 / 15.86 / 33.39     |
| Healthy-B4-s13 | Teacher-LSE | 9.18 / 18.86 / 39.12    | 14.02 / 25.71 / 44.85    | 4.34 / 12.02 / 33.39     |
| Healthy-B4-s29 | QT-full     | 47.65 / 54.43 / 58.31   | 39.96 / 47.68 / 51.93    | 55.34 / 61.19 / 64.69    |
| Healthy-B4-s29 | QT-on-P     | 42.37 / 47.48 / 49.65   | 40.66 / 47.55 / 50.72    | 44.07 / 47.41 / 48.58    |
| Healthy-B4-s29 | Student-D1  | 21.38 / 28.10 / 39.76   | 27.24 / 34.32 / 45.46    | 15.53 / 21.87 / 34.06    |
| Healthy-B4-s29 | Student-LSE | 13.66 / 21.95 / 38.31   | 18.48 / 27.71 / 42.72    | 8.85 / 16.19 / 33.89     |
| Healthy-B4-s29 | Teacher-LSE | 9.06 / 19.03 / 39.09    | 13.11 / 25.04 / 44.46    | 5.01 / 13.02 / 33.72     |
| R30_F-P659_s13 | QT-full     | 46.40 / 52.64 / 56.87   | 39.29 / 46.77 / 51.89    | 53.51 / 58.51 / 61.85    |
| R30_F-P659_s13 | QT-on-P     | 39.59 / 44.16 / 46.66   | 39.44 / 46.26 / 50.08    | 39.73 / 42.07 / 43.24    |
| R30_F-P659_s13 | Student-D1  | 16.81 / 24.07 / 36.14   | 22.93 / 31.62 / 43.39    | 10.68 / 16.53 / 28.88    |
| R30_F-P659_s13 | Student-LSE | 12.01 / 20.58 / 35.45   | 17.67 / 28.96 / 42.85    | 6.34 / 12.19 / 28.05     |
| R30_F-P659_s13 | Teacher-LSE | 7.84 / 15.10 / 34.79    | 13.34 / 23.86 / 41.53    | 2.34 / 6.34 / 28.05      |
| R30_F-P659_s29 | QT-full     | 46.10 / 52.48 / 56.73   | 39.37 / 46.94 / 52.27    | 52.84 / 58.01 / 61.19    |
| R30_F-P659_s29 | QT-on-P     | 39.71 / 44.41 / 46.85   | 39.69 / 46.76 / 50.46    | 39.73 / 42.07 / 43.24    |
| R30_F-P659_s29 | Student-D1  | 16.76 / 23.82 / 36.64   | 22.68 / 31.62 / 43.89    | 10.85 / 16.03 / 29.38    |
| R30_F-P659_s29 | Student-LSE | 12.05 / 20.95 / 35.34   | 17.75 / 29.55 / 42.64    | 6.34 / 12.35 / 28.05     |
| R30_F-P659_s29 | Teacher-LSE | 8.05 / 15.46 / 34.85    | 13.26 / 23.90 / 41.65    | 2.84 / 7.01 / 28.05      |

### 2.2 两seed family的同query均值

以下family仅是对应两个真实Student endpoint的平均，不是两个Teacher seed；B13未复制成双seed。

| Endpoint                | Scorer      | Overall R10/20/50 (%)   | Implicit R10/20/50 (%)   | Explicit R10/20/50 (%)   |
|:------------------------|:------------|:------------------------|:-------------------------|:-------------------------|
| Healthy-B4-seed-average | QT-full     | 47.47 / 54.13 / 58.08   | 40.03 / 47.66 / 51.88    | 54.92 / 60.60 / 64.27    |
| Healthy-B4-seed-average | QT-on-P     | 42.26 / 47.37 / 49.44   | 40.69 / 47.50 / 50.54    | 43.82 / 47.25 / 48.33    |
| Healthy-B4-seed-average | Student-D1  | 21.21 / 28.07 / 39.57   | 27.14 / 34.10 / 45.09    | 15.28 / 22.04 / 34.06    |
| Healthy-B4-seed-average | Student-LSE | 13.90 / 22.00 / 37.99   | 18.95 / 27.98 / 42.35    | 8.85 / 16.03 / 33.64     |
| Healthy-B4-seed-average | Teacher-LSE | 9.12 / 18.95 / 39.11    | 13.56 / 25.38 / 44.66    | 4.67 / 12.52 / 33.56     |
| R30_F-P659-seed-average | QT-full     | 46.25 / 52.56 / 56.80   | 39.33 / 46.86 / 52.08    | 53.17 / 58.26 / 61.52    |
| R30_F-P659-seed-average | QT-on-P     | 39.65 / 44.29 / 46.76   | 39.57 / 46.51 / 50.27    | 39.73 / 42.07 / 43.24    |
| R30_F-P659-seed-average | Student-D1  | 16.78 / 23.95 / 36.39   | 22.80 / 31.62 / 43.64    | 10.77 / 16.28 / 29.13    |
| R30_F-P659-seed-average | Student-LSE | 12.03 / 20.76 / 35.40   | 17.71 / 29.26 / 42.74    | 6.34 / 12.27 / 28.05     |
| R30_F-P659-seed-average | Teacher-LSE | 7.94 / 15.28 / 34.82    | 13.30 / 23.88 / 41.59    | 2.59 / 6.68 / 28.05      |

全部额外MAX/LME等诊断也见 [main_metrics_recomputed.csv](main_metrics_recomputed.csv)。

### 2.3 同支持集的成对不确定性

先对同query取两个seed的差值平均，再以source_table_id整组重采样；10000次，seed260916，bootstrap估计为抽中query差值总和/抽中query数，并非给不同大小source组同权后平均。下表为overall R@10差值，pp。

| family         | comparison    |   delta_pp | CI95_pp            |
|:---------------|:--------------|-----------:|:-------------------|
| Historical-B13 | S_D1 - QT_P   |    -20.875 | [-23.521, -18.205] |
| Historical-B13 | S_LSE - QT_P  |    -28.93  | [-31.885, -26.043] |
| Historical-B13 | T_LSE - QT_P  |    -33.118 | [-36.055, -30.200] |
| Historical-B13 | T_LSE - S_LSE |     -4.188 | [-6.360, -2.083]   |
| Healthy-B4     | S_D1 - QT_P   |    -21.049 | [-23.666, -18.378] |
| Healthy-B4     | S_LSE - QT_P  |    -28.36  | [-31.239, -25.517] |
| Healthy-B4     | T_LSE - QT_P  |    -33.139 | [-36.141, -30.247] |
| Healthy-B4     | T_LSE - S_LSE |     -4.779 | [-6.854, -2.727]   |
| R30_F-P659     | S_D1 - QT_P   |    -22.864 | [-25.585, -20.172] |
| R30_F-P659     | S_LSE - QT_P  |    -27.622 | [-30.417, -24.838] |
| R30_F-P659     | T_LSE - QT_P  |    -31.706 | [-34.781, -28.755] |
| R30_F-P659     | T_LSE - S_LSE |     -4.083 | [-6.084, -2.117]   |

这些区间是给定dev样本与固定Student/Teacher checkpoint的抽样区间；不提供held-out确认或充分训练随机性评估。

## 3. Path覆盖与Path排序分解

P_q定义为 `C100中具有合法且非空retained path multiset的target`。QT-on-P复用QT分数，只删去非P候选，不重取路径，不改变任何Path view。

### 3.1 支持规模和上界

| endpoint       |   path_targets_mean |   path_targets_min |   path_targets_max |   C100_recall |   P_recall_upper |   positive_in_C100 |   positive_in_P |   path_lt10_fraction |   path_lt20_fraction |   path_lt50_fraction |
|:---------------|--------------------:|-------------------:|-------------------:|--------------:|-----------------:|-------------------:|----------------:|---------------------:|---------------------:|---------------------:|
| Historical-B13 |              78.205 |                 50 |                100 |       58.1038 |          48.8592 |                736 |             622 |                    0 |                    0 |                    0 |
| Healthy-B4-s13 |              78.053 |                 50 |                100 |       58.1733 |          49.2209 |                738 |             628 |                    0 |                    0 |                    0 |
| Healthy-B4-s29 |              78.643 |                 50 |                100 |       58.6464 |          49.6522 |                744 |             633 |                    0 |                    0 |                    0 |
| R30_F-P659_s13 |              79.909 |                 50 |                 99 |       57.0395 |          46.6611 |                723 |             595 |                    0 |                    0 |                    0 |
| R30_F-P659_s29 |              79.934 |                 50 |                 99 |       56.8934 |          46.8489 |                722 |             598 |                    0 |                    0 |                    0 |

所有query的P规模至少50，因此 `<10/<20/<50` 比例全部为0。Path-only不是因为“返回不满10个”才差；不过no-path正例确实永远无法返回。

B13的736个C100正例中有114个无retained path；P中有622个正例。但QT-on-P **Top50已找回全部622个P正例**，其他四个endpoint同样达到各自P上界（628/633/595/598）。Teacher-LSE Top50则只找到506/502/501/446/447个。由此可知，在这个小池规模上，有路径正例大多能被QT识别，而当前Path分数不能。

### 3.2 精确的可加分解

`R(Path)-R(QTfull) = [R(Path)-R(QTonP)] + [R(QTonP)-R(QTfull)]`。

| Endpoint                | Path        |   Path-QTfull (pp) |   Path-QTonP (pp) |   QTonP-QTfull (pp) |
|:------------------------|:------------|-------------------:|------------------:|--------------------:|
| Historical-B13          | Student-D1  |            -26.196 |           -20.875 |              -5.321 |
| Historical-B13          | Student-LSE |            -34.252 |           -28.93  |              -5.321 |
| Historical-B13          | Teacher-LSE |            -38.439 |           -33.118 |              -5.321 |
| Healthy-B4-seed-average | Student-D1  |            -26.266 |           -21.049 |              -5.217 |
| Healthy-B4-seed-average | Student-LSE |            -33.577 |           -28.36  |              -5.217 |
| Healthy-B4-seed-average | Teacher-LSE |            -38.356 |           -33.139 |              -5.217 |
| R30_F-P659-seed-average | Student-D1  |            -29.466 |           -22.864 |              -6.601 |
| R30_F-P659-seed-average | Student-LSE |            -34.224 |           -27.622 |              -6.601 |
| R30_F-P659-seed-average | Teacher-LSE |            -38.307 |           -31.706 |              -6.601 |

右侧第二项是“限制候选支持集的净影响”，不是纯粹的no-path损失：删候选还可能让其他正例上升。B13 Top10删除no-path正例造成-6.907pp，而已有路径正例上升贡献+1.586pp，净-5.321pp。不要将这项当成独立因果占比。

特别地，implicit R@10中QT-on-P相对QT-full反而提高：B13+0.626pp，Healthy-B4均值+0.668pp，F-P均值+0.237pp。**implicit Path退化不能靠no-path覆盖解释。** explicit的支持损失更大（B13 -11.269pp），但即使删掉这部分，Path的同支持排序依然明显落后。

明细：[path_support_summary.csv](path_support_summary.csv)、[support_scoring_decomposition.csv](support_scoring_decomposition.csv)、新增排名 [QT_on_path_supported_rankings.jsonl.gz](QT_on_path_supported_rankings.jsonl.gz)。

## 4. strict historical207的机制

### 4.1 身份检查与可验证边界

导出207个唯一GT pair，五个endpoint完全相同；B13导出的own_strict也恰好是同207对，没有213更名。B13中所有207确实在导出E中、在ANN100外。原始exact100列表及历史baseline本体缺失，所以本报告称其为“本轮导出的fixed historical207”，并保留exact排除尚未从原始列表复核的限制。

own_strict pair数分别为207/210/215/206/207；不能与fixed207互换。fixed207在当前B4s13/B4s29/F-Ps13/F-Ps29的ANN100内已有3/2/17/17对，所以它也不等于这些模型各自的strict预算外集合。

### 4.2 全部TopK funnel

| Endpoint       |   固定对数 |   inC100=inP | QT-full Top10/20/50   | QT-on-P Top10/20/50   | Student-D1 Top10/20/50   | Student-LSE Top10/20/50   | Teacher-LSE Top10/20/50   |
|:---------------|-----------:|-------------:|:----------------------|:----------------------|:-------------------------|:--------------------------|:--------------------------|
| Historical-B13 |        207 |          101 | 75 / 92 / 100         | 80 / 93 / 101         | 29 / 48 / 91             | 22 / 42 / 84              | 30 / 50 / 90              |
| Healthy-B4-s13 |        207 |          108 | 79 / 96 / 107         | 85 / 98 / 108         | 34 / 51 / 96             | 24 / 44 / 86              | 34 / 58 / 97              |
| Healthy-B4-s29 |        207 |          109 | 78 / 97 / 108         | 83 / 98 / 109         | 34 / 53 / 99             | 20 / 41 / 89              | 31 / 55 / 99              |
| R30_F-P659_s13 |        207 |          100 | 73 / 85 / 99          | 77 / 89 / 100         | 32 / 55 / 96             | 25 / 49 / 86              | 32 / 53 / 88              |
| R30_F-P659_s29 |        207 |          102 | 73 / 86 / 101         | 78 / 91 / 102         | 29 / 54 / 97             | 25 / 49 / 87              | 32 / 58 / 90              |

这里是正pair计数，不是主表query-macro Recall。所有endpoint中，fixed207只要进入C100就有retained path，故该cohort内“Path不能返回无路径目标”不是劣势解释。

B13的207→101过程中先有106对未获admission；已admit的101对中，QT-full Top10丢26对，而Teacher-LSE丢71对。QT-full Top50只漏1对，QT-on-P Top50全部保留。**最大的strict损失发生在进入小池之前，至少不能归咎于QT普遍无法识别这些target。**

Direct ANN预算外并不逻辑等价于“任何Q,T模型都无法识别”；ANN Student和interaction Teacher的表达与排序能力不同。

### 4.3 相对QT-full的Top10转移

| endpoint       | view   |   rescued |   dropped |   both_correct |   both_wrong |   net |
|:---------------|:-------|----------:|----------:|---------------:|-------------:|------:|
| Historical-B13 | S_D1   |         6 |        52 |             23 |          126 |   -46 |
| Historical-B13 | S_LSE  |         4 |        57 |             18 |          128 |   -53 |
| Historical-B13 | T_LSE  |         5 |        50 |             25 |          127 |   -45 |
| Healthy-B4-s13 | S_D1   |         9 |        54 |             25 |          119 |   -45 |
| Healthy-B4-s13 | S_LSE  |         6 |        61 |             18 |          122 |   -55 |
| Healthy-B4-s13 | T_LSE  |         6 |        51 |             28 |          122 |   -45 |
| Healthy-B4-s29 | S_D1   |         9 |        53 |             25 |          120 |   -44 |
| Healthy-B4-s29 | S_LSE  |         4 |        62 |             16 |          125 |   -58 |
| Healthy-B4-s29 | T_LSE  |         4 |        51 |             27 |          125 |   -47 |
| R30_F-P659_s13 | S_D1   |         8 |        49 |             24 |          126 |   -41 |
| R30_F-P659_s13 | S_LSE  |         7 |        55 |             18 |          127 |   -48 |
| R30_F-P659_s13 | T_LSE  |        10 |        51 |             22 |          124 |   -41 |
| R30_F-P659_s29 | S_D1   |         7 |        51 |             22 |          127 |   -44 |
| R30_F-P659_s29 | S_LSE  |         8 |        56 |             17 |          126 |   -48 |
| R30_F-P659_s29 | T_LSE  |        10 |        51 |             22 |          124 |   -41 |

both_wrong以全207为分母，包含未admit的pair；不得把它们都解释为reranker失败。Top20/50完整转移见CSV。

### 4.4 相对QT-on-P的Top10转移

| endpoint       | view   |   rescued |   dropped |   both_correct |   both_wrong |   net |
|:---------------|:-------|----------:|----------:|---------------:|-------------:|------:|
| Historical-B13 | S_D1   |         4 |        55 |             25 |          123 |   -51 |
| Historical-B13 | S_LSE  |         3 |        61 |             19 |          124 |   -58 |
| Historical-B13 | T_LSE  |         4 |        54 |             26 |          123 |   -50 |
| Healthy-B4-s13 | S_D1   |         6 |        57 |             28 |          116 |   -51 |
| Healthy-B4-s13 | S_LSE  |         4 |        65 |             20 |          118 |   -61 |
| Healthy-B4-s13 | T_LSE  |         5 |        56 |             29 |          117 |   -51 |
| Healthy-B4-s29 | S_D1   |         6 |        55 |             28 |          118 |   -49 |
| Healthy-B4-s29 | S_LSE  |         2 |        65 |             18 |          122 |   -63 |
| Healthy-B4-s29 | T_LSE  |         3 |        55 |             28 |          121 |   -52 |
| R30_F-P659_s13 | S_D1   |         6 |        51 |             26 |          124 |   -45 |
| R30_F-P659_s13 | S_LSE  |         5 |        57 |             20 |          125 |   -52 |
| R30_F-P659_s13 | T_LSE  |         9 |        54 |             23 |          121 |   -45 |
| R30_F-P659_s29 | S_D1   |         6 |        55 |             23 |          123 |   -49 |
| R30_F-P659_s29 | S_LSE  |         5 |        58 |             20 |          124 |   -53 |
| R30_F-P659_s29 | T_LSE  |         8 |        54 |             24 |          121 |   -46 |

B13 Teacher确有4对在QT-on-P Top10之外而被救回，但同时丢54对，净-50。**少数互补命中是事实，然而它本身不足以证明可泛化、可利用的独立决策价值。** 需要排除随机排序扰动、模型固有target偏好与未确认witness等解释。

### 4.5 不可忽略的异质性

Teacher-LSE在fixed207的Top10比Student-LSE多8/10/11/7/7对，方向在五个endpoint一致；与overall上Teacher-LSE更差并不矛盾。该cohort强调对Evidence预算外target的识别，而overall也包含很多explicit和direct容易识别的目标。

额外按fixed pair计数加权的source-group bootstrap：Teacher-Student在B13为+3.865pp，CI[-1.887,9.524]；B4均值+5.072pp，CI[0,10.337]；F-P均值+3.382pp，CI[-2.174,8.696]。并非跨family稳健显著。这些是pair-weighted retention，不混入query-macro主表。

数据：[strict_eo_summary.csv](strict_eo_summary.csv)、[strict_eo_transition_summary.csv](strict_eo_transition_summary.csv)、[strict207_pair_weighted_bootstrap.csv](strict207_pair_weighted_bootstrap.csv)。

## 5. 有retained path不等于有verified witness

| endpoint       |   GT未进入C100 |   C100内无retained path |   C100内有retained path但witness未知 | C/D/E verified类别   |
|:---------------|---------------:|------------------------:|-------------------------------------:|:---------------------|
| Historical-B13 |            543 |                     114 |                                  622 | 无法判定             |
| Healthy-B4-s13 |            541 |                     110 |                                  628 | 无法判定             |
| Healthy-B4-s29 |            535 |                     111 |                                  633 | 无法判定             |
| R30_F-P659_s13 |            556 |                     128 |                                  595 | 无法判定             |
| R30_F-P659_s29 |            557 |                     124 |                                  598 | 无法判定             |

1279个GT pair是正target标签；包中没有对应的正确evidence标签、未保留路径全集或可供人工语义判定的原始内容。因而目前能可靠区分的只有：未进入C100、C100内无path、C100内有path但正确性未知。

用户要求的C（pre-retention中有verified witness但被丢弃）、D（verified retained但排序低）、E（verified retained且排序正确）都**无法判断，不能记0**。B类必须再拆为“尚未验证”与“已穷尽验证且无正确witness”；本包只能支持前者。

GT annotation只能加入离线diagnostic，不能改变本轮ranking。未进入C100不能被标为“全流程不存在path”，因为导出仅覆盖当前被审计候选。

## 6. Teacher-LSE究竟失败在哪里

### 6.1 MAX / log-mean-exp：简单数量修正没有修好

所有额外scorer仍然使用同一P与同一retained bags。MAX只改聚合为最大路径；LME为LSE减log(path数)，不改任何pair分数。QE-only和ET-only是target bag聚合诊断，不是带真实edge标签的QE/ET检索质量评估。

| Scorer   | Historical-B13        | Healthy-B4-seed-average   | R30_F-P659-seed-average   |
|:---------|:----------------------|:--------------------------|:--------------------------|
| QT_P     | 42.01 / 46.96 / 48.86 | 42.26 / 47.37 / 49.44     | 39.65 / 44.29 / 46.76     |
| S_LSE    | 13.08 / 21.95 / 37.35 | 13.90 / 22.00 / 37.99     | 12.03 / 20.76 / 35.40     |
| S_MAX    | 6.41 / 14.13 / 34.93  | 7.60 / 15.07 / 36.41      | 13.62 / 20.35 / 34.36     |
| S_LME    | 4.99 / 12.47 / 34.27  | 5.83 / 13.51 / 35.75      | 12.90 / 19.83 / 34.70     |
| T_LSE    | 8.89 / 19.32 / 39.68  | 9.12 / 18.95 / 39.11      | 7.94 / 15.28 / 34.82      |
| T_MAX    | 9.06 / 18.99 / 39.73  | 8.93 / 18.72 / 39.44      | 7.57 / 15.19 / 34.96      |
| T_LME    | 8.47 / 17.57 / 39.70  | 8.19 / 17.63 / 39.24      | 7.34 / 14.48 / 34.71      |
| T_QE_LSE | 8.97 / 18.20 / 37.16  | 9.06 / 18.21 / 38.31      | 7.76 / 15.96 / 33.14      |
| T_ET_LSE | 9.28 / 18.66 / 38.49  | 9.45 / 18.46 / 38.59      | 7.84 / 14.90 / 34.26      |

B13：Teacher-LSE8.890%，MAX9.057%，LME8.472%；Student-LSE13.077%，MAX6.413%，LME4.994%。因此“去掉LSE的路径数量奖励就能解决”已经被削弱，且Student简单去数量后更差。

### 6.2 Teacher和Student的有效聚合行为不同

B13所有P bags最大4条，无重复evidence造成的虚假multiplicity。正target平均2.416条，全部P target平均1.979条；路径数与正例有关，所以数量不是纯噪声。

在真正多路径的bags中，Teacher最高路径softmax权重中位数约0.855，42.22%的bag该权重大于0.9；Student中位数约0.500，大于0.9的比例为0。Teacher原始分数尺度使LSE已经常常近似MAX。**相同公式不代表同样的路径证据融合行为；Teacher失败不能简单归因于LSE累加了很多低质path。**

### 6.3 原始logit零点问题确实存在于训练目标，但尚未被证实是唯一根因

可见Teacher训练代码是固定source的edge-listwise loss。对每个e，把全部ET分数变为 `b(e,t)+c(e)`，不会改变该e内部的softmax和排序。但是路径分数变成 `LSE_e[a(q,e)+b(e,t)+c(e)]`，target之间的最终顺序可能改变。

这是由损失与聚合公式直接推出的可识别性问题：edge排序任务不约束不同e之间的绝对logit零点，所以“单边pair能力强”不能自动推出raw logits可相加。但本轮没有正确edge label，事实上也尚未证明QE、ET单边能力强。

不同关系缓存logit的均值/标准差约为：QE-text -0.315/4.389，QE-image -3.497/4.964，ET-text -1.992/4.009，ET-image -3.541/5.061。它们来自被Student选择后的不同样本分布，不能仅凭均值差认定Teacher有错误模态偏置。

### 6.4 又补了一个无训练局部归一化诊断

仅在现有retained子图上，使用 `logsumexp_e(logsoftmax_e(a_qe) + logsoftmax_t(b_et))`。ET分母仅包含当前query下同一个e的retained targets，所有pair值不变。这个操作保持每个e内部的ET排序，但去掉局部source加性零点。

| endpoint       |   T_localConditional_R10 |   T_localConditional_R20 |   T_localConditional_R50 |   mean_singleton_ET_source_fraction |
|:---------------|-------------------------:|-------------------------:|-------------------------:|------------------------------------:|
| Historical-B13 |                   8.4307 |                  16.5693 |                  37.4583 |                            0.18932  |
| Healthy-B4-s13 |                   9.2654 |                  16.7362 |                  37.6183 |                            0.18927  |
| Healthy-B4-s29 |                   8.3472 |                  16.8614 |                  37.6878 |                            0.205181 |
| R30_F-P659_s13 |                   6.963  |                  13.7243 |                  33.9316 |                            0.228517 |
| R30_F-P659_s29 |                   7.4638 |                  13.9538 |                  33.8689 |                            0.229194 |

B13降至8.431%，其余也没有系统修复。注意其中约19%–23%的ET source局部只有1个target，归一化分母存在选择与截断偏差，不能称为全湖校准概率。该负结果削弱的是**这一个局部归一化recipe**，不是排除所有校准/目标级学习方案。

### 6.5 模态、hub和margin

原报告按各scorer最高路径划分模态，组成员会随scorer变化。本次按固定bag组成划分text-only/image-only/mixed，并对同一批正target比较：

| bag_modality   |   positive_targets |   QT_P_Top10 |   S_D1_Top10 |   S_LSE_Top10 |   T_LSE_Top10 |
|:---------------|-------------------:|-------------:|-------------:|--------------:|--------------:|
| image_only     |                 69 |        84.06 |        34.78 |         30.43 |          8.7  |
| mixed          |                 60 |        80    |        40    |         41.67 |         23.33 |
| text_only      |                493 |        85.8  |        46.04 |         25.15 |         19.47 |

Teacher在image-only正例Top10留存低，但text-only也远低于QT-on-P，因此不能只修image就解释整体差距。这是pair留存比例，不是macro Recall。

B13 Top10中单个最常见target出现次数：QT-full25、QT-on-P30、Student-D1 116、Student-LSE109、Teacher-LSE52；Teacher有集中化，但并不比Student有更大的最高target hub，不能等同于此前residual recipe的全面公共hub崩塌。

B13 P正target相对最高非GT competitor的Teacher margin中位数-6.310；97.59%为负。该值只能在同一score space内解读；不能拿它与QT的-0.839直接比较“差多少”，也不能由负margin直接判断Top10命中。

一个数值例子：`query_1238d7986a4de800` / `target_742ff1f23e4f8023`，QT rank8、QT-on-P rank5、Teacher rank77。其单条image path QE=-7.444、ET=-6.302、sum=-13.746，而最高非GT competitor Teacher-LSE=7.579。**没有witness label时，这既可能是正确image路径被低估，也可能是保留了无关image，不能单凭分数选定解释。**

文件：[multipath_concentration.csv](multipath_concentration.csv)、[target_hub_diagnostics.csv](target_hub_diagnostics.csv)、[teacher_unique_pair_score_distribution.csv](teacher_unique_pair_score_distribution.csv)、[numerical_case_examples.json](numerical_case_examples.json)。

## 7. 科学判断：Supported / Weakened / Unknown

### Supported

当前C100、retained bags与score的主排序实现和指标可复算，没有证据表明巨大差距是pool mismatch、QT proxy、错误Recall或seed平均导致。三种Path-only替换QT均差；在同P上依然大幅差。fixed207已admit目标QT通常识别得很好。Teacher总体不如Student-LSE，但在固定历史strict cohort有相反方向；D1与Teacher不能直接解释成纯Student-vs-Teacher对比。

### Weakened

“主要因为Path不能返回无path目标”；“QT普遍把Evidence引入的strict目标压掉”；“LSE数量奖励或单个公共hub解释全部失败”；“只要把QT换成强Teacher edge logits相加就更强”；“一个局部归一化就可修复”。这些解释都被当前数据削弱。

### Unknown

真正witness是否保留、QE/ET在verified edges上的能力、完整pre-retention中的witness上界、原始Teacher pair分数的checkpoint forward复现、target-level path训练是否有效、三元交互(Q,E,T)是否提供额外可利用信息，以及是否存在可泛化的QT之外独立决策信号，都没有被当前实验判定。

**最可能的瓶颈（可定位层级）**：从retained evidence转换为target discrimination的最终打分链路，而不是本轮的集合定义或TopK列表过短。较具体的第一解释是edge训练目标/score尺度与target-bag排序目标不匹配；这还不是已经证明的唯一原因。

**第二竞争解释**：Student保留的path与目标的真实join机制不匹配，verified witness没进来或被retention裁掉；Teacher可能是在给错误输入合理地打低分。缺少标签使这条解释无法排除。

“Path-only最终重排不合理”应限定为：**当前这些实现，在当前端点与预算下，不应替换QT。** “只是Teacher没有Path-level训练”不足以解释全部结果，也不能保证补训能解决。

## 8. 下一轮：至多三个有门槛的实验

详见 [NEXT_EXPERIMENTS.zh-CN.md](NEXT_EXPERIMENTS.zh-CN.md)。顺序是补witness证据→若确有retention丢失才做受控保留测试→若真实witness已保留但打分失败才做目标级Teacher监督。不是三个大矩阵同时开跑。

现阶段在线基线保持Student admission→冻结T0(Q,T)；raw Qwen对照应沿用已有独立baseline流程，本轮没有其原始ranking，不能制造补值。暂不新增Student path训练或weighted fusion搜索；二者会在尚未确认Teacher/路径信号有效前增加循环依赖和解释混淆。

## 9. 交付与复算范围

附带全部每query指标、同支持集QT排名、fixed/own strict funnel与转移、source bootstrap、0-training聚合诊断和独立Python脚本。原始输入没有被覆盖。轻量包不含用户已提供的原始183MB压缩结果，不含生成过程的pickle缓存或92MB逐target特征表；可按README由原包重建。完整逐target特征表另外保留。

本报告所有实验结论都是dev、artifact-replay范围内的结论，未新增训练结果、未虚构witness标签、未把单seed B13变成两seed。
