# 原始方案主线与现有实验审查

审查日期：2026-09-07。范围：`方案.md` 的 Git 历史、`work/` 中初始化实验、Stage-1 r1-r9、pipeline unification、Stage-2 round1 的主要报告及关键 JSON，以及当前训练、检索、指标和 verifier 代码。没有重新训练或运行模型；数值复核使用已有 JSON 中的逐 query 结果。本报告区分已确认事实、由公式推出的限制和需要实验验证的假设。

## 1. 判断与原始主线

目前最主要的问题是实验选择标准逐渐偏向“保持 direct 检索、提高 fused R@10”，而原始方案需要证明的是“多模态证据补出缺失属性，使原本缺少连接条件的表能够 join”。已有结果支持一部分任务相关检索学习，但尚不足以证明完整的 evidence 桥接机制。

Git 中最早可追溯版本是 `0e8dad6`（2026-08-22），当前文档最后一次修改是 `d9f5185`（2026-08-25）。这里的“最初”仅指仓库中可恢复的最早版本。可复核命令：

```bash
git show 0e8dad6:方案.md
git diff 0e8dad6 HEAD -- 方案.md
```

两个版本共同保留了以下承诺：

1. query 是 example rows；从可见内容出发寻找缺失属性的证据。
2. 局部 primitive 是有向连接性 `J(a -> b)`；支持直接路径和单 evidence 中介路径。
3. Teacher 学细粒度跨对象关系，Student 学类型投影 P 与关系矩阵 R；底层 encoder 冻结。
4. 多条 evidence 可以为不同 query rows 提供互补属性信息。
5. 二阶段选择桥接列、定位行级证据、填值、验证最终 joinability。

原始文档本来就允许 direct，因此不必要求每个答案必须走 evidence。论文需要识别并验证 evidence 带来的新增能力，尤其是 direct 无法直接提供连接条件的样本。

后续文档改动中，低维 adapter、可见内容约束、完整 row 作为实体锚点、direct/evidence 分开监督、统一候选预算，都可以服务这条主线。应单独重新审视的是“每条 evidence 只能分配给唯一一行”：这是后来加入的计算约束，不是原始科学问题的要求。

文档还有两处需要最终对齐：开头称 Teacher 为 pairwise MLP，后文实际定义为 Relation Transformer 加 MLP head；开头“缺什么属性 -> 找证据”的描述，应与当前“先召回潜在桥接目标，再确定具体属性”的执行顺序相互解释。这属于表述一致性，优先级低于实验可识别性。

## 2. 已有实验实际上证明了什么

| 实验范围 | 可确认的结果 | 对原始主线的含义 |
| --- | --- | --- |
| 初始化 A/B、r1-r3 | 早期训练严重损伤检索；r2/r3 最终退回 epoch 0；在线 direct Teacher ensemble 有增益 | 初始化、Teacher 和训练列表确实需要修复；不能据此永久否定 P、edge KD 或 mining |
| r4-r5 | 分湖训练后 Student 改善；r5 EntiTables fused 37.74%，WDC 64.36% | 支持任务适配；仍要区分 direct 学习、KD 和 evidence 的贡献 |
| r5 Task X | 做过 text-only、target-bound、balanced 处理 | 模态及绑定问题已经被探索，不能说“完全没做”；但实验定义和统计效力不足以关闭这些方向 |
| r6 | 改归一化和聚合能大幅改变 evidence-only 排名，fused 改善小 | evidence 排名存在可优化空间；当前 fusion 和指标可能限制了这些改进的可见贡献 |
| r7-r8 | 低秩、温度、多种推理组合；WDC 推理改聚合可达 65.35% | 是配置条件下的结果，不能推导出函数类或 evidence 优化空间已经耗尽 |
| unification | 统一训练/推理后 EntiTables 38.06%，WDC 62.87%；保留 r5 为 primary | 补做一致性是有价值的；该结果不是整个方法的“上界” |
| r9 | 已有 P 漂移插桩、显式 R 学习率为 0 的修复、多正例数据适配；已有解冻等脚本 | P、edge、mining 已进入计划；本次未找到 A/B/C 完整训练结论，不能算完成的消融 |
| Stage-2 round1 | 7,829 条 Oracle 样本正确列全部在位置 0；模型和 majority 都是 100% | 这一轮不能证明语义选列；真实检索下的补全、定位和最终验证尚未形成有效结果 |

来源：[r2](../work/stage1_optimization_r2_20260829/FINAL.md)、[r3](../work/stage1_optimization_r3_20260829/taskI_final/FINAL.md)、[r4](../work/stage1_optimization_r4_20260829/FINAL.md)、[r5](../work/stage1_optimization_r5_20260829/FINAL.md)、[r6](../work/stage1_optimization_r6_20260830/FINAL.md)、[r7](../work/stage1_optimization_r7_20260831/taskS_integration/FINAL_TABLE.md)、[r8](../work/stage1_optimization_r8_20260831/FINAL.md)、[unification](../work/stage1_pipeline_unification_20260831/FINAL.md)、[r9 插桩](../work/stage1_optimization_r9_20260901/taskA0_projection_drift/RESULTS.md)、[r9 数据适配](../work/stage1_optimization_r9_20260901/taskF_data_adaptation/RESULTS.md)、[Stage-2](../work/stage2_round1_20260831/RESULTS.md)。各轮语料、候选预算和配置不同，不能把这些数字直接连成一条同口径进步曲线。

## 3. 优先纠正的偏移

### 3.1 小权重 RRF 在当前预算下排除了 evidence 独立发现新表

这比“evidence 的影响较小”更严重。实际实现按分支是否检索到目标计算加权 RRF，没有为 evidence 独有目标补算 direct 分数。r5 的配置是：

```text
score(T) = 1 / (60 + rank_D(T)) + 0.05 / (60 + rank_E(T))
```

不存在的分支不贡献分数。因此，一个只由 evidence 发现的目标，即使排在 evidence 第一名，分数最高也只有 `0.05 / 61 = 0.00081967`。direct 第 100 名仅靠 direct 就有 `1 / 160 = 0.00625`。

top-10 评测召回 100 个 direct 候选；这 100 个全部排在任何 evidence 独有目标前面。top-50 使用 500 个 direct 候选，结论仍成立。更一般地，只要至少有 K 个 direct 候选且 `w_E / 61 < 1 / (60 + K)`，evidence 独有目标就无法进入最终 top-K。

所以当前 primary 的 evidence 通道可以给 direct 池内候选加分、附带证据，但不能实现论文希望展示的“经过 evidence 发现 direct 候选池外的目标”。把 RRF 权重从 0.05 小幅调到 0.1 或 0.5，仍不能让 evidence 独有第一名进入 top-10。

注意：这个结论针对当前加权 RRF，不可直接套到 normalized-score fusion；不同分数尺度下的 0.05 不是同一种影响大小。

代码：[retrieval.py](../src/mmdd_stage1/retrieval.py)，`fuse_ranked_channels`、`rank_detailed_paths`、`retrieve_zero_one_hop_detailed_many`。

### 3.2 r5 的 fused 主结果没有给出净正的 evidence 排名收益

从 r5 final evaluation 的逐 query 数组复算，而非比较不同运行的报告数字：

| 同一 Student、同一轮候选检索 | EntiTables | WDC |
| --- | ---: | ---: |
| query 数 | 938 | 202 |
| direct R@10 | 37.95%（356） | 64.85%（131） |
| fused R@10 | 37.74%（354） | 64.36%（130） |
| fusion 救回的 query | 2 | 0 |
| fusion 丢失的 query | 4 | 1 |
| fused 的已知正 evidence path coverage | 3.41%（32） | 24.75%（50） |
| evidence 权重为 0 时的同一 coverage | 3.30%（31） | 25.25%（51） |

“救回”在这里指 direct 前十未命中而 fused 前十命中，不代表发现了 direct 前 100 以外的目标。上一节已经说明后者在这个配置下不可能。

evidence 权重为 0 时 coverage 仍非零，是因为 direct 候选仍携带已检索到的 evidence 路径。因此，非零 coverage 不能证明 evidence 导致了排名改善。它仍然可能帮助二阶段补值，这部分价值需要单独测量。

来源：[EntiTables 原始 JSON](../work/stage1_optimization_r5_20260829/task4_final/final_evaluation/entitables/metrics.json)、[WDC 原始 JSON](../work/stage1_optimization_r5_20260829/task4_final/final_evaluation/wdc/metrics.json)。

### 3.3 在线 direct Teacher 重排逐渐代替了离线蒸馏的贡献

r5 EntiTables 的 `Student + online reranker` 为 46.16%，Student 本身为 37.74%。在线重排处理的是 direct Q-T 候选，报告中 evidence coverage 为 n/a。它是一项有效的对照或扩展系统，但不能用这 46.16% 来支持“Student 已经学会可索引的 evidence 连接性”。

KD 自身的增量也需要克制陈述：r5 EntiTables KD 相对 supervised 为 +0.32 个百分点，95% CI 为 [-1.07, 1.71]；WDC 为 +2.48，CI 为 [-0.50, 5.45]。有正向点估计，但尚不能称为两湖都已确证的蒸馏收益。Teacher 的细粒度输入还缺少同池的“只给 pooled embedding”对照，尚不能把所有 Teacher 增益归因于细粒度交互。

建议论文分别列出 raw、supervised P+R、KD P+R、可选 online Teacher、完整 Stage-2。以 Stage-1 在线仅运行 Student 的行支撑原始效率主张。

### 3.4 选择标准和旧门槛持续偏向 direct

早期以 WDC direct 保留率为 gate，后期以几乎受 direct 主导的 fused R@10 为 primary。这个选择方式会淘汰“evidence 更可用、最终补全更好、但粗召回略低”的模型。r8 将 evidence 9.81% 因未达到 10% 判为不通过，可以是预先约定的选择规则，却不能作为某类方法被证伪的科学依据。

同样，r2-r4 因旧 Teacher 或旧 Student 问题冻结 mining，并不构成永久冻结的理由。r9 已认识到这一点，但仍继承 `weighted_rrf e0.05` 和 fused-first gate；仅按现有 r9 计划解冻 P，仍可能优先选出改善 direct 的模型。

建议固定研究问题后设置两层验收：实验有效性检查，以及“端到端 bridge 成功率与总体质量”的多指标选择。保留不同取舍的候选，在 dev 上选择；不要用单个整数门槛替代机制分析。CI 下界不低于 -2 个百分点表示容忍范围内未明显退化，不能表述成显著提高。

## 4. 训练层面的遗漏与可尝试方向

### 4.1 P 尚未在修复后的配方上被充分测试

r5 history 明确记录 `freeze_projections=true`、4096 -> 1024 的 PCA 初始化、`hard_examples=0`。代码为 table/text/image 创建三个 P，但冻结时都复制同一 PCA 基。这时模型实际学习的是固定子空间内的关系矩阵。

如果两对象差异位于 PCA 舍弃的方向，固定 P 下任何 R 都无法恢复它。解冻 P 可以旋转到任务相关子空间，并允许各模态分化。这是原始方案中缺失的能力，优先级高于继续约束 R 的秩或漂移。

不过不能说 P 历史上“完全没训练过”：最早初始化 A/B 包含可训练投影；缺少的是修复后的数据、Teacher、负例和学习率配置下的受控比较。r9 也已准备了解冻脚本。

建议从相同初始化做 `P固定/R固定`、`P固定/R训练`、`P训练/R固定`、`P训练/R训练`，再对最后一项比较短 R warm-up 后联合训练。显式固定 P/R 各自学习率，记录每种模态的更新、向量范数和各关系检索质量。联合训练需要重建对应候选向量和 ANN 索引；固定旧索引的解冻评测会失去可比性。

P 的锚定强度可与 R 分开，但只在观察到约束确实妨碍学习时扩展。不要直接依据一次解冻失败断言线性函数类不足；优化、监督和候选分布都可能是原因。先完成线性 P+R，再考虑类型相关的小型残差 MLP，并保留每对象独立编码以支持 ANN。

来源：[初始化 A/B](../work/stage1_student_initialization_ab_20260828/RESULTS.md)、[r9 计划](../stage1_optimization_r9_plan_20260901.md)、[models.py](../src/mmdd_stage1/models.py)、[train_stage1.py](../src/train_stage1.py)。

### 4.2 仅训练路径终点，没有充分约束两条局部边

当前主配方跳过 student-edge，直接从 PCA 开始 student-path。路径分数 `s(Q,E)+s(E,T)` 高，不保证 Q-E 的全库 top-L 或 E-T 的全库 top-M 好。前一跳一旦漏掉 evidence，后面的聚合就无法补救。

从目标函数看，对某条 evidence 的第一跳分数加偏置、第二跳减同一偏置，路径和不变；但第一跳检索排序可能改变。这是路径监督本身不能唯一约束分解的例子，并不声称当前双线性模型能任意实现所有此类偏置。

优先实验：较短的 edge KD warm-up 后做 path fine-tune；对照 path 阶段继续保留小权重的 edge loss。分别测 Q-text、Q-image、text-T、image-T 的召回、有效路径比例和最终 bridge 成功率。检查 Teacher 在这些边上的质量，按关系选择 KD 权重，不能只依据 direct Teacher gate 决定所有关系是否蒸馏。

### 4.3 难负例的增益主要给了 direct，训练路径还不够接近推理

`train_student_paths` 的 in-batch 分支只调用 `score_target_direct_batch_in_batch`；evidence 仍使用原来绑定的 evidence lists。现有 raw-ANN retrieval-aligned lists 是有价值的，因此缺的不是一切难负例，而是健康 Student 当前错误所产生的动态 Q-E、E-T 和完整路径负例。

`align_target_record` 在默认 query-hard 模式下，为新增 ANN target 绑定同一组 query 检索到的 hard evidence；已有正 target 则保留原始 evidence。这种构造可能让“evidence 来自哪种列表”与标签相关，也没有完整模拟在线 E-T 扩展。这是需检查的分布偏差，尚不能仅凭代码断言模型已经使用了这个捷径。

建议保留三种独立负例配额，并构造可解释的错误：实体对但不能支持当前桥接属性、属性对但实体错、第一跳相关而终点列不相容、两跳都高但没有真实可恢复值。对当前 query 能通过其他属性形成合法桥接的 evidence，不能直接当作 Q-E 负例；属性不相容应在相应路径或 E-T 条件下判断。

同一 evidence 跨候选终点的对照有助于减少来源捷径；原有 `corrupted_path` 已做了一部分，应该加强其真实性而非宣称完全缺失。所有已知正 target/有效桥接路径都应从负例中排除；未标注目标仍可能是假负例，需要小样本审核。

动态 listwise distillation 与多样化训练列表在 [RocketQAv2](https://aclanthology.org/2021.emnlp-main.224/) 中有直接依据。这里的三类路径负例与属性一致性设计是针对本项目的实验建议，并非该论文已经验证的结论。

来源：[training.py](../src/mmdd_stage1/training.py)、[scoring.py](../src/mmdd_stage1/scoring.py)、[retrieval_aligned.py](../src/mmdd_stage1/retrieval_aligned.py)。

### 4.4 多正例虽然已适配，但 loss 仍允许忽略困难正例

当前多正例 loss 是 `-log(sum_{T in positives} p(T))`，准确地说是在最大化正例集合的总概率。它与“对每个正例均匀做 soft-label CE”不是同一个目标。例如两个正例概率为 0.89 和 0.01，正例总质量已经达到 0.90，loss 可以很低，但第二个正例仍很难召回。

而当前 Recall@K 按命中的正例数 / 全部正例数计算，要求尽可能发现所有目标。新版数据已包含多 target、多隐藏属性，因此需要比较集合概率 loss、正例平均 log-probability，以及对每个正例单独组织列表的训练方式。后两种也有概率竞争等取舍，应该以多正例召回和属性覆盖判断。

这是当前代码的目标对齐问题，不能追溯解释旧单正例实验的失败。来源：[objectives.py](../src/mmdd_stage1/objectives.py) 的 `listwise_cross_entropy`、[evaluation.py](../src/mmdd_stage1/evaluation.py) 的 `_recall`、[r9 数据审计](../work/stage1_optimization_r9_20260901/taskF_data_adaptation/RESULTS.md)。

## 5. Path aggregation 应围绕互补证据改进

### 5.1 数量不是互补性，两个高分边也不一定支持同一个属性

当前聚合器只接收两条边的 scalar score 和 evidence mask，不知道 evidence 支持哪一行、哪种属性，也不识别重复正文、同图不同 ID 或相同来源的冗余信息。它不能直接实现文档中“多条 evidence 分别覆盖不同 rows”的动机。

对 N 条分数近似相等的路径，LSE 约为 `mean_score + log(N)`；top-k sum 也偏好更多高分路径。对现有检索边分数较集中的情况，路径数量可能比细微的质量差异更影响排序。r5 WDC 的候选列表 evidence 出现次数中，top-1 asset 占 12.35%、top-10 占 36.23%。这是列表复用/集中现象，提示应检查 hub 和重复支持；它本身不证明这些 evidence 错误。

另一个具体细节是：当前 `logsumexp` 分支对所有传入路径聚合，`evidence_top_k=4` 不会截断它；K 对 top-k mean/sum 生效，二阶段又单独只取 top evidence。于是表可以因大量路径被抬高，但 verifier 实际只看到其中四条。所谓“logsumexp/top4”不等于 LSE 始终只用了四条证据。

建议按下面顺序比较：

1. 固定同一检索路径池，比较全路径 LSE、真正的 top-K LSE、log-mean-exp，分解数量奖励与质量奖励。log-mean-exp 只是诊断对照，不宜直接抹去原方案中的互补支持。
2. 在真实重复检测基础上按内容/来源分组，每组保留主要支持；来源 ID 只用于去重审计，不能成为预测 joinability 的捷径特征。
3. 利用已有 query-row embeddings，在已检索的小候选集合内估计 evidence 对各行的支持；按“每行最强支持，再跨行累积”聚合，使重复支持同一行的收益饱和。
4. 构造“相同 evidence 数量，但覆盖一行/多行”的成对实验；只增添重复证据时，分数不应不受控制地提高。
5. 检查两跳是否支持同一属性语义。高 Q-E 相关性加高 E-T 相关性，并不自动意味着存在可抽取的实体-属性-值链。

一个简单的候选聚合形式是 `mean_i max_E [row_support(i,E) * calibrated_path_strength(Q,E,T)]`。它保留跨行证据累积，但抑制单行堆积；作为内部聚合对照即可，仍在线检索单个 object，不需要枚举 evidence bundle。先验证 row support 是否可靠，再决定是否加入训练目标。

### 5.2 两跳校准与瓶颈打分仍有空间

现有 r6 已做 edge z-score、不同均值及最大值等聚合，不能再把它们列为全新方向。不过“平均绝对分数没有超过 2 倍”只否定了那个具体尺度假设，不能说明所有概率校准和模态分布偏差已解决。EntiTables 的同一 Student，LSE 的 evidence R@10 从未归一化 4.05% 到 edge z-score 9.81%，已经说明排序对这一细节敏感。

下一步可比较按有序类型对拟合的温度/偏置，检查置信度是否预测真实边正确性；在明确的校准假设下比较两跳 log-probability 相加、弱边惩罚或 smooth-min，防止很强的一跳掩盖很弱的另一跳。不能直接把任意 bilinear score 当作概率相乘。

所有拟合只使用 train/dev。聚合与校准选定后，再做匹配的 Teacher/Student 训练和完整检索评测；推理侧重组可作为消融，不能单凭它宣称学到了新的路径机制。

来源：[objectives.py](../src/mmdd_stage1/objectives.py)、[retrieval.py](../src/mmdd_stage1/retrieval.py)、[r6 EntiTables 聚合](../work/stage1_optimization_r6_20260830/taskD_path_aggregation/entitables_RESULTS.md)、[r6 WDC 聚合](../work/stage1_optimization_r6_20260830/taskD_path_aggregation/wdc_RESULTS.md)、[r5 Task X JSON](../work/stage1_optimization_r5_20260829/taskX_evidence_modality_ablation/metrics.json)。

## 6. Fusion 的可行方向与验收

目标应是让可靠 evidence 能引入真正有用的候选，同时控制错误证据的代价。大权重本身不是科学贡献，小权重也不能替代证据质量改进。

建议首先保留一个 fixed-budget 对照：direct-only、evidence-only、等权 RRF、当前 0.05 RRF、校准 score fusion。每个系统给二阶段的唯一 target 总数一致。额外做按分支预留部分名额的 union，测量“放开 evidence 候选准入”是否存在端到端收益；这项是瓶颈诊断，不预设它就是最终方法。

主方法可尝试：先取 D/E 候选并集，在并集内补算缺失的双线性 direct score，再按校准后的 direct/evidence 强度统一排序。补算不需要全库打分；这能减少“只因不在 direct 截断列表就完全没有 direct 贡献”的截断效应。evidence 分数本身若缺路径，不应伪造。

进一步的 query/target 条件权重，可依据两跳可靠性、证据一致性、行覆盖和列匹配置信度学习。要验证权重是否在需要 evidence 的样本上提高，且损坏证据后相应降低；否则 adaptive fusion 也可能学成另一个关闭 evidence 的常数门。

至少分别报告三个量：

1. 最终正确目标中，仅通过 evidence 才进入 Stage-1 候选池的比例。
2. D/E 都召回但由 evidence 改善排名的比例，以及同时损失的目标。
3. target 早已被 direct 找到，但必须通过 evidence 才能补出有效 join 属性的成功率。

第三项同样支持原始故事。即使第一项不大，evidence 仍可能是完成 join 的必要条件；现有实验还没有测出这一区别。

## 7. 二阶段与任务定义的关键缺口

### 7.1 缺少能排除捷径的端到端验证

Stage-2 round1 的位置退化已被原报告正确识别；不应把 100% 写成成果。新版构造代码和 r9 计划已经涉及列顺序打乱与多正例，下一步是对实际新数据重跑位置分布审计和 Oracle 实验，而不是再次从头讨论同一个已知 bug。

即使位置打乱后选列正确，也可能只根据列名、类型或表格常识完成。需要相同 Q/T 条件下比较真实 evidence、无 evidence、同实体但不支持当前属性的 evidence、错误实体的 evidence，以及交换列顺序。image 独有样本上还需要图像删除/区域遮挡对照。热图可用于说明定位，但热图漂亮不是 evidence 被有效使用的证据。

原始参考 RATA 的任务包括列/行增补与缺值恢复，评价最终恢复内容是合理的对齐方向，参见 [Retrieval-Based Transformer for Table Augmentation](https://aclanthology.org/2023.findings-acl.348/)。FOCUS 针对图像细节定位后的 VQA，不能自动背书本项目对完整 row anchor 和文本 span 的扩展；这些应单独做定位及答案正确性验证，参见 [FOCUS](https://arxiv.org/abs/2506.21710)。

### 7.2 唯一行分配与单 evidence 生成限制了可恢复性

当前 `SimilarityEvidenceRouter` 将每条 evidence 强制分配给一个最相似行，没有拒绝不相关证据的选项；一段文字即使支持多个实体，也只能帮助一行。若 bundle 只有 K 条 evidence，最多只有 K 行拿到非空输入。K=4 时，query 行数一旦大于 4，就存在可直接算出的行覆盖上限；实际分配到重复行还会更低。

随后每行只选择一条 localized evidence 生成值。这对“一个图给实体线索、一个文档给属性值”或者多条证据互补的样本不够。

优先比较 threshold/top-2 的稀疏多行分配与当前 argmax；允许无支持的 evidence 不分配。保持总定位预算，比较单 evidence 与少量互补 evidence 的填值。用真实行级标注报告 routing recall、可支持行覆盖与错误路由率。

来源：[routing.py](../src/mmdd_stage2/routing.py)、[pipeline.py](../src/mmdd_stage2/pipeline.py) 的 `verify`。

### 7.3 正例内部选列不能代替候选表验证，多正例还只执行一个桥接目标

当前联合概率为 `softmax(r_T) * softmax(g_T,c)`，所以对某个表的所有列求和，结果仍然严格等于 `softmax(r_T)`。列头不能将“整张表都不适合桥接”的概率直接调低，只能在表内分配概率。训练主体也只对 positive bundle 做列 CE。

这不是后来偏离方案才出现的问题，而是原始分解本身需要验证的局限。可以保留 RATA 风格列头，补充 no-valid-column 选项、负目标表训练，或增加轻量表级兼容分数；测量是否能拒绝整张错误表。尤其在表宽变化后，表内 softmax 的置信度也可能受到列数影响。

`verify` 全局 argmax 选一个表-列对后，只对这一条桥接分支填值。对旧单目标设置是有意的成本限制，但新版一个 query 多 target、多属性时需要比较 top-1 与固定预算的 top-B；并明确论文输出的是“任意一个有效 join”还是“尽可能发现全部 joins”。这也影响训练 loss 和 Recall 的选择。

来源：[verifier.py](../src/mmdd_stage2/verifier.py) 的 `joint_candidate_probabilities`、[training.py](../src/mmdd_stage2/training.py)、[pipeline.py](../src/mmdd_stage2/pipeline.py)。

### 7.4 Semantic containment 不是正确补值的充分证据

当前最终检查计算每个生成值与目标列任意值的最大相似度/精确匹配，再判断覆盖率。它不独立验证“这个值是否属于当前 query row 的实体”。例如把不同实体都错误填成目标列中的同一个常见值，可能通过 containment，却没有正确恢复属性。

因此必须分别测 row-level value accuracy、证据是否蕴含该值、最终 join precision/recall，以及空值/拒答。补值正确与最终通过检查不能由同一个 containment 指标循环证明。对“真实 evidence / 无 evidence / 损坏 evidence”的比较尤其重要，它能区分证据利用与模型记忆。

来源：[verifier.py](../src/mmdd_stage2/verifier.py) 的 `semantic_joinability`。

## 8. 数据和评测中还需补足的部分

**关系标签的含义。** 当前 construction 从所有正 qrel 构造 direct 正目标，未在该入口根据 qrel 的显式/隐式 reason 区分监督。对于必须补属性的 target，`Q -> T` 可以解释为“潜在可连接目标”，但不是“可见表已经能直接 join”的标签。应明确区分 target retrieval relevance 与已成立的 direct joinability，分别统计 explicit-visible 和 evidence-required 子集。保留 direct 学潜在目标是可行的，只是不能把它当成 evidence 已被使用。

**GT evidence 与可用 evidence。** 现有 Coverage@10 是“前十某个已知正 target 携带至少一条候选列表中的正 evidence”，不是覆盖全部行或全部属性。分母是全部评测 query，还应单独报告在 implicit、确有可用 evidence 的 query 上的条件覆盖率，不能直接与 Oracle 子集的成功率比较。候选 evidence 还可能来自 source-level fallback，未必都已验证支持特定 row-attribute；另一方面未标注 evidence 也可能有效。因此需要严格 recovery-supported 指标与未判定支持分开，并对典型失败做人工审核。

**多模态样本效力。** r5 Task X 的 WDC balanced 实际设 `min_train_relation_records=500`，将只有 149 条的 image-to-table 和 table-to-image edge 记录删除，path 阶段和检索仍保留图像。这不是完整的 image-balanced 监督。Stage-2 WDC test 仅 1 条 image-only 和 1 条 text+image，不能据此判断图像贡献。已有 text-only 对照有价值，但还缺足够 image 必要样本，以及相同预算下的 train-time/test-time 分别消融。

**召回曲线与预算。** 当前评测每个 K 都重新以 `gamma*K`、`gamma_e*K` 生成路径池。因此它是“随输出预算增加而增加搜索预算”的系统曲线，不是同一个固定 ranking 的截断曲线；个别 R@K 非单调不自动是指标实现 bug。论文应补固定搜索预算下的 K 曲线，以及固定延迟/总候选预算的比较。

**独立测试与选择不确定性。** 关键 r5 final evaluation 脚本明确读 `split="dev"`。现有这些主表首先是开发集选择结果；本次没有找到修复后主配方的完整独立 test 主表。大量配置中挑出的最大值，其 query bootstrap CI 不会自动包含超参选择偏差、训练 seed 和 ANN 重建变异。固定最终配方后测 untouched test；关键比较用多训练 seed，候选评分消融共享池，模型召回比较保留各自真实检索并辅以 exact sanity check。

**数据版本与负例完整性。** r9 新数据的多正例、列分布、候选库都变化，必须在同一版本上重跑 raw、PCA identity 和 r5 配方。已知正例排除不保证未标注表一定是负例；同源其他投影、语义等价属性、实体重叠应单独审计。列顺序修复会使旧 Stage-2 数字失去比较资格，但旧问题仍是重要的历史诊断。

**跨湖和规模。** 分湖 PCA/参数/索引是合法实验设置，但只能支持分湖适配，不能据此宣称单模型跨湖泛化。r8 WDC 只有 3,540 个候选表，报告也明确未证明大规模饱和。跨湖迁移和扩池可以作为后续外部效度实验，优先级低于先把现有 evidence 主线验证清楚。

来源：[evaluation.py](../src/mmdd_stage1/evaluation.py)、[construction.py](../src/mmdd_stage1/construction.py)、[evaluate_stage1_r3_baselines.py](../src/evaluate_stage1_r3_baselines.py)、[WDC balanced preflight](../work/stage1_optimization_r5_20260829/taskX_evidence_modality_ablation/wdc/balanced/data/preflight.json)、[Stage-2 数据审计](../work/stage2_round1_20260831/DATA_AUDIT.md)。

## 9. 建议下一轮实验顺序

| 顺序 | 最小受控实验 | 要回答的问题 | 关键判据 |
| --- | --- | --- | --- |
| 0 | 固定新版数据，重跑 raw/PCA/r5；审计列位置、显隐式标签和多正例 | 基线是否可信且可比 | 同版本、同 split、候选预算清晰 |
| 1 | 当前检索、gold evidence 注入、gold target、gold target+evidence 的小规模对照 | 瓶颈在 Q-E、E-T、聚合、选列还是补值 | 分阶段失败计数；Oracle 行单独报告 |
| 2 | P/R 四格消融，加一条 warm-up 联训 | 原始 student 是否真正具备任务投影能力 | 各关系召回、有效路径、bridge 成功率，而不只 fused |
| 3 | path-only、edge warm-up+path、小权重 edge/path 联合 | 局部可检索关系是否被学好 | Q-E 与 E-T 同时改善，完整路径可用性提高 |
| 4 | 受控的一轮 Student hard mining | 模型是否学会排除真实的伪桥接 | 三类独立负例配额；假负例抽查；同预算 |
| 5 | top-K LSE、去重 LSE、行覆盖聚合 | 多 evidence 是否真的互补 | 相同证据数下跨行覆盖提高；重复证据无虚假收益 |
| 6 | 等权/校准 fusion、并集补打分、预算固定的分支预留对照 | evidence 能否引入并完成新的 join | 新目标准入、排名救回/损失、最终正确补值 |
| 7 | 位置修复后的 Oracle reader，加真实/缺失/损坏 evidence | 二阶段是否依赖语义和证据 | 超过 position/schema-only；值恢复与 grounding 改善 |
| 8 | 小规模完整两阶段，随后冻结配方做独立 test | 原始故事是否最终成立 | implicit 子集增益、总体 precision/recall、延迟 |

不必顺序跑完前六项才能开始第七项。阶段 1 的小规模诊断足以支持并行推进实验设计，但本次审查没有启动任何训练任务。

建议每一行固定附带：target Recall@K、正确 evidence Recall@L、已知有效路径 recall、行支持覆盖率、bridge-column accuracy、row-value accuracy、最终 join precision/recall，以及总延迟。尚未执行下游时留空，避免用前面的 proxy 代替后面的结果。

## 10. 适合保留与暂时降级的结论

值得保留：冻结底层 encoder 的可索引 P+R 设计；显式 0/1 跳；Teacher 细粒度知识向 Student 蒸馏的假设；retrieval-aligned 训练；不同 example rows 需要互补证据；证据定位后补属性再验证的闭环。

适合暂时降为对照：在线 direct Teacher ensemble、低秩 R 效率消融、按湖选择不同聚合/参数、0.05 RRF 工程配置、仅推理替换聚合。这些实验有信息价值，但应由各自测到的行为支撑表述。

需要避免的推断：某几个温度或低秩设置失败就称“结构杠杆用尽”；统一配方未提高就称“方法上界”；未达到点数门槛就说机制被证伪；coverage 非零就说 evidence 有因果贡献；Teacher 重排提高就说 Student 蒸馏成功。

最有价值的论文主张应落在可检验行为上：在缺少桥接属性的 query 中，系统能找到与实体及属性匹配的多模态证据，正确恢复连接值，并在控制候选/计算预算后发现更多真实 joins；移除或损坏证据，这部分收益应相应下降。这个主张允许 direct 保持强基线，也允许部分模块给出负结果。

## 附：关键数值复核

以下命令只读取已有实验 JSON。`fusion_rescued` 和 `fusion_lost` 在当前单正例历史数据上按 query 计数：

```bash
jq '.systems.student.metrics as $m |
    $m.per_query.direct["recall@10"] as $d |
    $m.per_query.fused["recall@10"] as $f |
    {queries: $m.queries,
     direct_hits: ($d|add),
     fused_hits: ($f|add),
     fusion_rescued: ([range(0;$d|length) | select($f[.] > $d[.])]|length),
     fusion_lost: ([range(0;$d|length) | select($f[.] < $d[.])]|length)}' \
    work/stage1_optimization_r5_20260829/task4_final/final_evaluation/entitables/metrics.json \
    work/stage1_optimization_r5_20260829/task4_final/final_evaluation/wdc/metrics.json
```

本报告对研究方向的判断依据现有产物与实现，提出的优化均是待检验假设。没有把缺少结果等同于方法失败，也没有把报告中的测试通过次数当作本次重新执行的验证结果。
