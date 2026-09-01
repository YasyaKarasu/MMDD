# Stage-1 统一 Pipeline + 聚合系列实验计划(2026-08-31)

## 背景:为什么这轮要做(核心动机)

### 已确认的事实(代码 + r8 实验)
1. **`PathAggregator` 是训练与推理共用的同一个类**(`src/mmdd_stage1/objectives.py`),有 7 种变体:`logsumexp`(当前默认)、`max`、`topk_mean`、`topk_sum`、`softmax_weighted_mean`、`power_mean`、`comb_mnz`。
2. **teacher 训练涉及 path 聚合**:`train_teacher_paths` → `score_target_batch(model, ..., aggregator)` → 聚合后的 evidence 分数进入 `listwise` loss(`_path_supervised_losses`)。所以**改聚合 = 改变 teacher 的 path 训练目标 = teacher 参数会变 = 需要重训 teacher**。
3. **student KD 也涉及聚合**:`train_student_paths` 用 `score_target_batch(student, ..., aggregator)` 算 student 分数;而其蒸馏目标(teacher logits)由 `score_and_cache_teacher_logits`(按聚合方式缓存)提供。所以**改聚合 = teacher logits 缓存失效 = 需要重算 + 重跑 student KD**。
4. **因此:训练期聚合与推理期聚合必须一致**(同一 `PathAggregator`,同一 `retrieval.py` 里的 `PathAggregator` 配置)。r8 的"只改推理、复用 r5 checkpoint"(WDC 65.35)是在**训练(用旧聚合)与推理(用新聚合)不一致**的配置下拿到的——这不符合"统一 pipeline",其 WDC 数字**不应被采用**,除非在"聚合一致"下复现。
5. r8 的另一问题:**它逐湖选了不同的 fusion 与聚合**(Enti: zscore+logsumexp;WDC: RRF+topk_sum)——这也是违背"统一 pipeline"的。**统一 pipeline 要求两湖用同一套"形式",只允许内部权重等按湖调参。**

### 本轮目标
**建立两湖统一的 pipeline(同一 fusion 形式 + 同一聚合形式),并让训练与推理聚合一致,然后在这个统一 pipeline 下重新评测两湖,看能否复现/超越 r5(WDC 64.36,Enti 37.74)。**

## 全局约束
- **统一性硬约束**:两个湖必须用**相同的 fusion 函数** + **相同的聚合函数**;只允许下列"调参项"按湖不同:`evidence_weight`、`top_k`、`temperature`、`power`、归一化的开关。
- **训练=推理一致性硬约束**:训练(teacher/student 的 KD)`PathAggregator` 配置必须与 `retrieve_zero_one_hop_detailed` 中 `PathAggregator` 配置**完全一致**。禁止"只改推理不改训练"。
- 评测口径:recall_ks={10..50}、γ=10、γ_e=2、配对 bootstrap(10,000, seed 13)、按湖 CI gate(tolerance 0.02)。
- 结果写入 `work/stage1_pipeline_unification_20260831/RESULTS.md` + 最终 `FINAL.md`。

---

## Task A:聚合 & 融合形式的再评估(零训练,先在保存的 logits 上做,选定统一形式)

**目的**:不做训练前,先用已有 logits 在"统一的形式"假设下评估哪些聚合/融合候选在两湖上都好——决定"统一用哪一个形式"。

**实现**:
1. 对 r5 的 full-R student checkpoint + 现有 teacher logits,取 dev 上的 4 组合(2 fusion × 2 path aggregation)——但**排除**逐湖不同的组合,只保留"同一形式"下两湖都跑得动(用所有候选,输出两湖并排表)。
2. 新加聚合形式:**在所有 7 种 `PathAggregator` 变体**上做一次"零训练"评估(因为 `PathAggregator` 是训练/推理共用的),对 r5 full-R 学生直接跑推理(仅改聚合形式,不做训练),两湖并排表,选出**两湖都最优的形式**(而不是每湖各自最优)。
3. fusion 形式:re-evaluate RRF / normalized_score(zscore & minmax)/ 各自在统一形式下的表现。

**输出**:
- 一张表:每湖 × 每个候选聚合 × 每个 fusion 的 fused R@10 / evidence / coverage / MRR,并给"统一"选出的形式。
- **结论判断**:是否存在一个"聚合 + fusion 形式"在两湖上都不劣于 r5?若有,选它;若无,标记"需要训练侧配合"。

---

## Task B:Toy 实验——在"统一"聚合下,teacher 从零重训(单一湖)

**目的**:验证"改聚合 = 重训 teacher"是机制上可行的,并且在新聚合下 teacher 还能学到东西(而不是对"统一形式"过度不利)。

**实现**:
1. 用 `PATH_AGGREGATIONS` 候选(先在 `logsumexp` vs `topk_sum` 两档对比;若时间允许,再扩展到 4–5 种)重训 EntiTables 和 WDC 两个 teacher(检索对齐列表,同 task7/taskM 的流程)。
2. 每个 teacher 的 rerank 门禁(与 task7 同口径:同池 top-100 重排,±3pt 判定)。
3. 输出:每湖 teacher 的 rerank R@10,看"统一聚合"对 teacher 是否有系统性 hurt。

**判定**:
- 若 topk_sum 下 teacher 两湖 rerank 仍接近 task7/taskM→ 统一聚合不太伤 teacher,可接受。
- 若 topk_sum 显著伤 teacher → 标记"topk_sum 不适合 teacher 训练",选另一聚合。

---

## Task C:统一 pipeline 的重训——teacher 重训 → logits 重算 → student KD 重跑(两湖)

**目的**:在最终确定的统一形式下,完整重跑训练链。这是"统一 pipeline"的核心,也是任何"训练-推理一致"方案必须做的。

**实现**:
1. 选定统一聚合 + 统一 fusion(A 的结果)。
2. **重训 teacher**(每湖,但**同一聚合形式**):edge→path,检索对齐列表。
3. **重建 teacher logits 缓存**(新 teacher + 新聚合)。—— teacher logits 缓存依赖 e aggregation/top_k;聚合一变,缓存 key 会 miss,需重建。
4. **重跑 student KD**(新 teacher logits + 统一聚合),蒸馏权重 0.3、τ 用 r5 值(或 r6 自适应)。
5. 在**统一的 pipeline** 下评测两湖(recall_ks, CI)。

**判定**:
- 若统一 pipeline 下 WDC fused ≥ 64.36 且 Enti ≥ 37.74 → 成功,采用为最终配置。
- 若 WDC 达不到"topk_sum 的让利"(因为统一牺牲了 topk_sum 而用别的),但仍 ≥ r5 → 采用;若达不到 → 记录"统一形式的收益上界",保留 r5 并做"统一 vs 分湖"的事后分析(为论文写"统一 pipeline 的收益上限")。

---

## Task D:统一 pipeline 下的最新"性能档"确认——WDC 是否还能到 65+

**目的**:前面 r5 是"逐湖最优"的锚;在统一 pipeline 下,WDC 的 65.35 是否能被"统一"复现?(这是最关键的判断点——如果 WDC 能到 65 且 Enti 也能 ≥37.74,你的故事就完整了;如果 WDC 掉回 64,则确认"统一 pipeline 的代价=牺牲 0.99pt,换取真正的统一"。)

**实现**:
1. 用 Task B/C 选定的统一配置,在 WDC 上重跑(用统一的聚合/融合)。
2. 单独对比"统一 pipeline WDC" vs "r8 分湖 WDC 65.35" vs "r5 WDC 64.36"。
3. 若统一 WDC 达到 65,且 Enti 也 ≥37.74,则将统一形式定为论文配置。

**验收**:
- WDC 在统一 pipeline 下 fused ≥ 64.36(优先看是否 ≥65.35);
- Enti 同样在统一 pipeline 下 fused ≥ 37.74。
- 无论是否达到 65.35,输出"统一 pipeline 收益上限"表(统一 vs 分湖 vs r5)。

---

## Task E:统一 pipeline 的论文叙事整合

**目的**:如果统一 pipeline 成立,论文"统一框架、按湖实例化"的主张就有了完整实验支持;若成立不了,也报告"统一 pipeline 的收益/损失"。

**实现**:
1. 产出 FINAL.md 主表:两湖在统一 pipeline 下的 raw / supervised / KD / reranker(Enti enabled, WDC no-op),附 CI。
2. **明确写**:fusion 形式 = X、聚合形式 = Y(两湖一致);调参项(evidence_weight 等)按湖。
3. 报告"统一 vs 分湖"的收益差,作为论文的一段分析(它同时回答"统一 pipeline 的代价"与"为什么我们的统一是成立的")。
4. 若最终 WDC < 64.36 但仍 > r5 的原始,记录为"统一 pipeline 的稳健性"而非失败。

---

## 执行顺序与决策树

```
Task A(零训练,统一形式选择)
  ├─ 存在两湖都不劣于 r5 的统一形式 → 用它
  └─ 无 → 标记"统一形式有收益上限,需要训练配合" → Task B/C 做"最不伤的统一形式"
Task B(Toy 重训,单一聚合 × 两湖 teacher)→ 确定"teacher 是否受 hurt"
Task C(统一 pipeline 完整重跑:teacher→logits→student→统一评测)
  ├─ WDC ≥ 64.36 且 Enti ≥ 37.74 → 成功,定为论文配置,Task E
  └─ 未达 → 记录"统一 pipeline 收益上限",保留 r5,做"统一 vs 分湖"分析,Task E
Task D(统一 pipeline 的 WDC 65+ 确认)→ 若 65 可复现,则完整;若掉回 64,记录为"统一代价"
Task E(整合 FINAL.md → 论文)
```

## 产出要求
- 目录 `work/stage1_pipeline_unification_20260831/task{A,B,C,D,E}_*/`;RESULTS.md 逐任务追加;Task E FINAL.md。
- 每个任务的对比表(两湖并排)都带 CI;明确标注哪些数字是"训练-推理一致"下得到的(可采用的),哪些不是(仅参考)。
- 所有新增开关(聚合形式、fusion 形式、调参项)向后兼容;新增代码配测试。
- 明确把"r8 的 WDC 65.35 是在不一致配置下得到、不作为采用"写进综合结论,统一后的 WDC 数字才是正数。

## 关键提醒(给执行者)
- **`PathAggregator` 是共享类**——改它 = 同时改训练(teacher/student)与推理。不要在训练用 A、推理用 B。
- **teacher logits 缓存**对聚合方式敏感——聚合一变,缓存 key 失效,必须重建;**不要试着"复用 r5 checkpoint + 只改推理"**,那是 r8 的错误路线。
- 统一 pipeline 的"形式"指函数(fusion / 聚合),不是参数(weight / top_k / temperature)——只有后者按湖调,前者两湖必须一致。
