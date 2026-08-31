# Stage-1 第七轮计划:低秩残差关系 + 联合配置选择 + r6 收尾(2026-08-31)

## 背景:r6(work/stage1_optimization_r6_20260830)已确认的结论

1. **尺度错配假设被证伪**(Task A):student 各 type-pair 分数统计与 raw 几乎一致(无 >2× 失配)。μ=0.1 anchor 有效抑制了尺度漂移。**归一化融合的收益来自"利用幅值信息",不是"修尺度"**——论文表述要按此修正。
2. **Fusion 归一化有效**(Task B):WDC `normalized_score_minmax_e0.05` 65.35% vs RRF 64.36%;EntiTables `zscore_e0.1` 同样最优。**RRF → 归一化分数加权融合**成为新默认。
3. **Path 聚合按湖异质**(Task D):WDC 最优 `topk_sum`(无归一化),EntiTables 最优 `logsumexp + 每边 zscore`。配合 relation weight 2,evidence R@10 大幅改善(WDC 23.27→44.06,Enti 4.05→9.70)。
4. **τ 自适应启发式可行**(Task C):自动选 Enti=0.7 / WDC=0.5,均落在 r5 τ 曲线的高性能平台区;作为免调参默认可接受。
5. **表表示增强关闭**(Task F):F1(更多行+命名单元格)在 WDC 伤 student evidence −6.44;F2(每行 4 token+重训 teacher)增益不显著。**表 token 预算不是瓶颈,此方向不再投入。**
6. **r6 的失败教训:组件贪心选择不等于联合最优**。各组件单独为正,合成后 vs r5 是 Enti 0.00、WDC −1.49(不显著)。原因:(a) Task E 用 evidence-first 规则选 relation weight 2,牺牲了 WDC fused(64.36→62.38);(b) fusion/聚合/relation-weight/τ 各自贪心,未做联合验证。
7. r5 基线(对比锚点):Enti fused 37.74 / WDC fused 64.36;WDC teacher 线上重排 no-op 维持不变(诚实负结果)。

## 本轮两个主轴

**主轴 1(架构,主要投入)**:把 Student 关系矩阵改为**低秩残差参数化** `R = I + A·Bᵀ`,把"joinability 是对 frozen similarity 的受控方向性校正"写进模型形式。动机:
- r2/r5 的 relation-drift 证据:伤 WDC 的是 R 偏离 I 的"坏方向";高秩 R 有 1024 个可偏方向,anchor 只能压总幅度,不能限制方向数;
- 低秩把可偏方向限制在 k 个,**有可能让 WDC 在训练中不再回退**(这是唯一剩下的、可能改善 WDC 的结构杠杆);
- step-0 严格等于 raw(A,B 零初始化),与 epoch-0 gate 天然一致;可去掉 μ 调参(或大幅放松);
- 论文叙事升级:"我们不替换底座相似度,而是为每个 type-pair 学习低秩方向性校正"——这是现有全部实验结论(R≈I 最优、epoch-0=raw、按湖漂移不同)的数学化。

**主轴 2(流程,修 r6 教训)**:配置的**联合选择**——fusion/聚合/relation-weight/τ 不再各自贪心,在小笛卡尔积上按"fused R@10 优先、evidence/coverage 为次"联合验证,并以 r5 为硬基线(合成配置必须 ≥ r5,否则回退该组件)。

## 全局约束
- 评测口径与 r6 相同:recall_ks={10..50}、γ=10、γ_e=2、配对 bootstrap(10,000, seed 13)、按湖 CI gate(tolerance 0.02)、同池对比。
- teacher 不动(各湖 Task-M teacher);image 模态保留;WDC teacher 线上重排保持 no-op。
- **对比基线双锚**:r5 手动配置(Enti 37.74 / WDC 64.36)与 r6 各组件单项最优。任何最终配置不得低于 r5(CI 口径)。
- 结果写入 `work/stage1_optimization_r7_20260831/RESULTS.md` + FINAL.md;新增代码配测试。

---

## Task P:低秩残差 Student(本轮核心)

**实现**:
1. `StudentJoinabilityModel` 新增参数化模式 `relation_param ∈ {full(现状), lowrank}`:
   - lowrank:每 type-pair `R = I + A·Bᵀ`,`A,B ∈ R^{d×k}`,**A 零初始化、B 正态小初始化**(或反之;保证 step-0 residual=0);
   - 打分实现避免显式构造 R:`s = (u_a·u_b) + (u_a A)·(B^T u_b)`(两次 d×k 矩阵乘);
   - `index_vector`/`relation_query` 相应改写:query 侧向量 = `[u_q ; u_q A]`,index 侧 = `[u_t ; B^T u_t]` 的拼接形式(维度 d+k),保持 ANN 内积等价;manifest 记录 k。
2. **k 扫描**:k ∈ {16, 64, 256},每湖分别训练(path-only、in-batch、蒸馏 τ 用 r6 自适应值、蒸馏权重 0.3);anchor μ 设 0(低秩本身就是结构约束)与 μ=0.1 两组对照(共 3k × 2μ,每湖 6 组;若算力紧,先跑 μ=0)。
3. 每 epoch 记录:各 type-pair 的 `‖A·Bᵀ‖_F`(残差幅度)、fused/direct/evidence R@10、CI-vs-raw。
4. 与 full-R(r6 最终配置)同池对比。

**验收/判读**:
- **EntiTables**:某 k 的 fused R@10 ≥ full-R(37.74)且不劣于 r5 → 低秩不损失表达力;
- **WDC(关键)**:训练过程中 direct R@10 **不再回退到 epoch-0 以下**、最优 epoch > 0 且 fused ≥ 64.36 → 低秩结构性抑制了坏方向漂移,这是本轮最重要的可能收益;
- 若 WDC 仍回退 → 记录"低秩不足以消除湖间机制差异",低秩仍可因参数量小(9×2dk vs 9d²)和叙事价值保留,按 EntiTables 表现决定去留;
- 产出 k-敏感性表(论文消融:相似度骨架 vs 语义校正 rank 的关系)。

---

## Task Q:配置联合选择(修 r6 教训,零训练)

**实现**:
1. 候选空间(基于 r6 单项最优的邻域,不做全网格):
   - fusion:{r5 weighted_rrf_e0.05, r6 归一化最优(minmax_e0.05 / zscore_e0.1)}
   - 聚合:{logsumexp(r5), r6 每湖最优(topk_sum / lse+zscore)}
   - relation weight:{1, 2}
   - τ:{r5 手动, r6 自适应}
   共 2×2×2×2=16 组合/湖;其中 relation weight 与 τ 涉及训练的,复用 Task P 与 r6 已训 checkpoint,不重训——**只对已有 checkpoint 做检索侧组合评测**。
2. 选择规则:**fused R@10 为主序,≥ r5 为硬约束**;满足硬约束的组合中,再按 evidence R@10 + coverage@10 排序取优。
3. 对选出组合跑完整 recall_ks + CI,确认合成不劣于任何单项。

**验收**:每湖得到一个联合最优配置,fused ≥ r5(CI 口径),evidence/coverage 尽可能保留 r6 的增益;写明各组件的边际贡献(逐项 knock-out 表)。

---

## Task R(低成本附加,与 P 并行):KD 温度/尺度对齐

**理由**:teacher(MLP 无界输出)与 student(内积)logits 尺度天然不同,蒸馏 KL 的温度目前全局 1.0。Task A 证明 student 侧尺度没漂,但 **teacher-student 间的尺度对齐**没检查过。一个可学习的每 type-pair 标量 `α_ij`(乘在 student logits 上,仅蒸馏损失内使用)或按湖校准的全局温度,可能让 KD 的 τ 曲线整体上移。

**实现**:蒸馏时 student logits 乘可学习标量(初始化 1.0,每 type-pair 一个;或先做更简单的版本——网格搜温度 T ∈ {0.5, 1, 2, 4} 重算 KD);在 r6 自适应 τ 配置上对比开/关。

**验收**:任一湖 fused R@10 较不加对齐提升 ≥ 0.5pt(CI 不劣)则保留;否则记录"KD 尺度对齐非瓶颈"关闭该线。

---

## Task S:最终整合与主表

**实现**:
1. Task P 选出的关系参数化(lowrank-k 或保留 full)+ Task Q 联合配置 + Task R 结论,重跑完整评测;
2. FINAL.md 主表(每湖):raw / student epoch-0 / supervised / KD student(final) / student+reranker(Enti)或 no-op(WDC),recall@{10..50} + mrr@50 + coverage + CI;
3. 记录方法层面的最终配方(论文 methods 对应):低秩残差(若采纳)+ 归一化融合 + 按湖聚合 + 自适应 τ + 蒸馏权重 0.3;
4. 更新论文素材清单:τ 曲线(r5)、relation-drift 动机图(r2/r3)、k-敏感性表(Task P)、联合配置 knock-out 表(Task Q)、"尺度假设证伪 + 表 token 预算非瓶颈"两条负结果(r6 Task A/F)。

**验收**:FINAL.md 每湖 fused ≥ r5 锚点(Enti 37.74 / WDC 64.36),理想情况 WDC 借低秩突破 65+;主表可直接支撑论文实验节。

---

## 执行顺序与决策树

```
Task P(低秩残差,k×μ 扫描,每湖)与 Task R(KD 对齐)并行
Task Q(联合配置,零训练)在 P 出 checkpoint 后做(P 未完成时可先用 r6 checkpoint 演练管线)
  ├─ P 在 WDC 上消除回退且 ≥64.36 → 低秩为最终参数化,写入主叙事
  ├─ P 仅 Enti 不劣 → 低秩以"参数量小 + step-0=raw + 结构化锚定"价值保留,WDC 结论如实记录
  └─ P 双湖均劣于 full-R → 保留 full-R,低秩作为消融记录
Task S 整合出 FINAL.md
```

## 产出要求
- 目录 `work/stage1_optimization_r7_20260831/task{P,Q,R,S}_*/`;RESULTS.md 逐任务追加。
- Task P 的 k-敏感性表与残差幅度轨迹(CSV);Task Q 的 knock-out 表;全部关键对比附配对 CI。
- checkpoint/索引只保留 epoch-0/best/final;向后兼容(relation_param 默认 full)。
