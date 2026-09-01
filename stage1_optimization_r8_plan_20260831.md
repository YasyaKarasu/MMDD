# Stage-1 第八轮计划:召回饱和曲线 + 候选池扩规模条件实验 + 组件重组(2026-08-31)

## 背景:r7(work/stage1_optimization_r7_20260831)已确认的结论

### r7 结果(保留 r5 为论文默认)
1. **低秩残差 `R=I+ABᵀ`(Task P)**:step-0 精确=raw、μ 可去掉(μ=0 与 μ=0.1 在 k=16 上几乎相同)、参数量降至 1/32(k=16);但两湖都未达到 r5 锚点——WDC 最优 `k=256` fused 63.86% → 配检索配置后 64.36%(打平 r5);EntiTables 最优 `k=64` 35.18% / `k=256` 35.07%(高于 raw 30.92% 但低于 r5 37.74%)。**"WDC 漂移是坏方向太多"假设被证伪**——低秩没有消除漂移(只有 k=64 回退=0,最优 k=256 仍回退),根因是数据分布/任务本质不是结构容量。
2. **Task Q 的 Enti 组合让 evidence 大幅上涨**:低秩 k=256 + `normalized_score_zscore_e0.1 + logsumexp_edges_zscore` 使 Enti evidence R@10 达到 **10.66%**(r5 full-R 仅 ~4%),coverage 3.41%,但 fused 反而 35.61%(direct 掉得多)。**这说明 Task Q 的检索组合对 evidence 通道的调优是真实有效的,只是混在"低秩"变量里没被发现;应与 r5 的 full-R 重组验证。**
3. **KD 温度(Task R)**:T={0.5,2,4} 均差于 T=1,无 +0.5pt 提升;**KD 温度/尺度对齐不是瓶颈**,该线关闭。
4. **r7 门禁未通过**(Enti 显著低于 r5、WDC 仅打平),`FINAL.md` 未声明;r5 保持论文默认。低秩保留为参数效率/efficiency ablation(9.4M → 0.2M@k=16 检索不变)。
5. r6 已关闭的线:尺度错配假设(证伪)、表 token 预算(非瓶颈)、WDC teacher 线上重排(no-op、诚实负结果)。

### 本轮要回答的核心问题
**扩大 WDC 数据集是否有效?** 初步判断:扩"训练样本"大概率无效(r5 混合负样本反而更差、r7 低秩/全秩 WDC 打平说明任务已近饱和、Task A 证明 student 与 raw 尺度一致);但**扩"候选池/语料"**(WDC 当前仅 3,540 表、~2,700 target,LakeBench 完整 WebTable 有 ~16M 表)是唯一可能真正改变 WDC 召回余量的方向——前提是证明 raw 召回还远未饱和。**该问题由 Task R1 的饱和曲线裁决,未裁决前不做任何扩规模训练。**

## 全局约束
- 评测口径不变:recall_ks={10..50}、γ=10、γ_e=2、配对 bootstrap(10,000, seed 13)、按湖 CI gate(tolerance 0.02)、同池对比。
- **r5 为论文默认配置(full-R, τ Enti=1.0/WDC=0.7 手动的 r5 选择)**;任何新配置必须经 Task R2 验证后,在 fused ≥ r5 锚点(Enti 37.74 / WDC 64.36, CI 口径)才可替换。
- teacher 不动;image 模态保留;WDC teacher 线上重排 no-op 维持。
- 结果写入 `work/stage1_optimization_r8_20260831/RESULTS.md` + FINAL.md;新增代码配测试。

---

## Task R1:WDC 召回-深度饱和曲线(零训练、决定后续方向,最高优先)

**理由**:扩规模有效与否取决于"WDC 的 raw 召回是否还有未饱和余量"。若 raw 在 R@50 已近饱和(如 R@200 仍 ~60% 说明还有余量;若 R@50 已 80% 后横盘说明任务已饱和)。饱和曲线一张图直接裁决"扩训练 vs 扩候选池 vs 都不做"——**不做这个先耗算力扩规模是浪费。**

**实现**:
1. 对 WDC dev(202 查询),用 raw 冻结 embedding 在 **当前 WDC 语料(3,540 表)** 上做**穷举/全量检索**(brute-force 矩阵乘,而非 HNSW,消除近似误差),对每个查询按 raw 内积全排序,输出每查询的正 target 排名。
2. 逐 k ∈ {10, 30, 50, 100, 200, 500, 1000, 全部} 计算 raw `R@k`(direct 通道),并画曲线。
3. 同时给出 WDC 正 target 在 raw 全排序中的**排名分位数分布**(中位、P75、P95),这代表"可召回的下限"。
4. 若 WDC 语料太小,补充说明:当前表只有 3,540,任意 k>3,540 都截断,因此"饱和"是**当前候选池的饱和**,不是 WebTable 全体表空间的饱和。在报告里明确区分这两个概念,避免误导。

**裁决规则**:
- 若 raw `R@50` 已 ≥ 85% → 当前 WDC 可召回空间接近耗尽,**扩规模(s 训练或候选池)都无益**;记录"WDC 已饱和",关闭扩规模方向。
- 若 raw `R@100/200` 还显著低于 80%(如 R@100 仍 ~70%)→ 存在未饱和余量,**扩候选池**(更多表)可能有效,继续 Task R3;扩训练样本仍不推荐(证据见背景)。
- 若发现**表数量本身是瓶颈**(3,540 表导致 R@k 被 capping)——这是最可能的情况,标记"候选池规模受限",为 Task R3 提供依据。

**产出**:`taskR1_saturation_curve/RESULTS.md` + 曲线图(CSV/PNG);结论(饱和/未饱和/池受限)。

---

## Task R2:full-R × Task-Q 检索组合重组(零训练,低成本,最有希望改进 Enti 主结果)

**理由**:r7 的 Task Q 把 `normalized_score_zscore_e0.1 + logsumexp_edges_zscore` 验证为 Enti 的 evidence 通道最优(evidence R@10 4%→10.66%),但该组合是在**低秩** student 上跑的。**关键未测的格子:r5 的 full-R student × Task Q 的 Enti 检索组合**——若它既保持 r5 full-R 的 fused(37.74),又继承 evidence 10.66%,就是"best of both worlds"。这是零成本(复用 r5 full-R checkpoint + r6/r7 检索配置),却是最有希望实质改善 Enti 主结果的一项。

**实现**:
1. 用 r5 full-R EntiTables student checkpoint,在 dev 上跑**全部 4 组合**(2 fusion × 2 path aggregation,与 Task Q 同候选空间),输出 fused/direct/evidence R@10、coverage@10、MRR@50、CI-vs-r5。
2. 对比:同一 4 组合在 r7 低秩 k=256 Enti student 上的结果(已有,从 taskQ 读取),与 r5 full-R 自身的 r6 检索组合。
3. 若存在组合使 **fused ≥ r5(37.74)且 evidence ≥ 10%** → 选定为 Enti 主结果,这是本轮最重要的可能收益。
4. WDC 侧同样跑 full-R × Task Q 组合(WDC 当前用 `weighted_rrf+topk_sum`,fused 64.36=r5),确认 WDC 主结果不劣。

**验收**:Enti 得到 fused ≥ 37.74 且 evidence ≥ 10% 的组合;WDC 保持 ≥ 64.36。若存在,则主表两湖都明显改善。

---

## Task R4(可选,若 R2 Enti 组合成功):平衡 main-tune / WDC 的论文学科定位

**理由**:若 R2 给 Enti 拿到 fused+evidence 双赢,则可在论文里把 Enti 作为"全链路成功"主例、WDC 作为"表面相似型、teacher 退化为 no-op"的负例/边界例,增强"按湖自适应"叙事的完整性。本任务主要是**报告与整合**,不是新训练。

**实现**:用 R2 选定的每湖最终配置重跑完整评测,产出 `FINAL.md`(论文主表),明确:湖级配置(fusion/聚合/τ/关系参数化)、每湖的 teacher 角色(在线重排 enabled/no-op)、raw/student/reranker 三行、索引与延迟、以及低秩作为 efficiency ablation 的段落。

**验收**:FINAL.md 每湖 fused ≥ r5(Enti 37.74 / WDC 64.36),或明确解释未达标的原因并保留 r5 为默认。

---

## 执行顺序与决策树

```
Task R1(饱和曲线,零训练,先行)→ 裁决
  ├─ 已饱和 → 关闭"扩规模",直接 Task R2
  ├─ 未饱和/池受限 → 记录该事实,但不做扩规模训练;仍直接 Task R2
Task R2(full-R × Task Q 组合,零训练)→ 若 Enti 双赢 → 选定为 Enti 主结果
Task R4(整合,FINAL.md)→ 每湖 fused ≥ r5 或明确保留 r5
```

## 产出要求
- 目录 `work/stage1_optimization_r8_20260831/task{R1,R2,R3,R4}_*/`;RESULTS.md 逐任务追加。
- R1 饱和曲线图(CSV+PNG);R2 组合对比表(附 CI);R3 扩池前后 raw/fused 对照。
- 所有关键对比用配对 bootstrap;checkpoint/索引只保留 epoch-0/best/final;向后兼容。
- 若全部确认 r5 为最终选择,R4 FINAL.md 作为定稿论文主表;并负Responsibly 保留：r7 低秩、KD 温度、表 token、尺度假设四条负结果作为论文 ablation/机制章节。
