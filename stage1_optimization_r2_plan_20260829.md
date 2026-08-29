# Stage-1 优化第二轮计划(2026-08-29)

## 背景:第一轮(stage1_optimization_20260828)已确认的结论

后续所有任务以此为前提,不要重复验证:

1. **破坏源头在 student-edge 阶段的旧 Teacher KD**。各配置 path epoch-0 direct R@10:fresh PCA 直接 path = 35.35%;edge(KD=0, 无 in-batch)后 = 35.53%;edge(KD=0, in-batch)后 = **28.07%(in-batch edge 伤 7pt)**;edge(KD=1, 旧 teacher)后 = **9.0–9.4%(毁 26pt)**。第一轮 Task 1 的 anchor sweep 全部建立在被 KD 毁掉的 9% 基座上,**并未真正测试 anchor**。
2. **当前最优 student 配方**:fresh PCA-1024 冻结 P → **跳过 student-edge** → student-path only + KD=0 + in-batch negatives(`task3_fresh_pca_path_kd0_inbatch`)。overall direct R@10 = **36.67%(与 raw 打平)**,EntiTables = **32.84%(超过 raw 的 31.24%)**,无任何 epoch 跌破 epoch-0。
3. **贯穿性问题:所有训练都让 WDC 回退**。raw WDC direct = 61.88%;最优 student = 54.46%(−7.4);task2(KD=0 无 in-batch)= 55.45%(−6.4);重训 teacher 重排 = 49.50%(−13.9)。两个不同模型同一症状 → 数据/目标层面的系统性偏差,候选假设:(A)WDC 依赖表面相似、训练学的语义偏移对它是噪声;(B)alpha=0 等量采样导致 WDC(3,564 path 样本)每 epoch 重复 ~1.9 次过拟合;(C)共享 R_table_table 的数据集梯度干扰。
4. **Evidence 通道:训练只有破坏作用**。最优 student 的 evidence R@10 从 epoch-0 的 8.16% 掉到 2.37%;fusion sweep 最优解为 `weighted_rrf_e0`(evidence 权重为零)。但 e∈[0.025, 0.1] 时 coverage@10 可达 7.28–7.46% 且 fused 只损失 0.2–0.6pt → evidence 有信息,当前分数质量配不上权重。
5. **旧 Teacher 确认有害**(重排 −16.2pt,已弃用);**重训 Teacher(task7 cache24k)有真实信号但偏科**:overall 重排 40.18%(+3.25),EntiTables 38.17%(**+6.9**),Spearman(raw, teacher)≈0.01(与 raw 正交的信号);但 WDC 49.50%(−13.9)、E→T 重排 28.04% < raw 33.13%、MRR@100 0.213 < raw 0.274(擅长把 GT 捞进 top-10,不擅长顶到第 1)。
6. 第一轮 Task 5(oversample/image 降权)建立在被 in-batch edge 损坏的 28% 基座上,**结论作废**,不要引用。

## 全局约束

- 语料/特征/dev 固定 `work/stage1_stage2_wdc2k_entitables20k_v4_20260828` 那套;aggregator 固定 `logsumexp, top_k=4`。
- Student 一律 fresh PCA-1024 冻结 P、跳过 student-edge、path-only、in-batch negatives 开启——即在结论 2 的最优配方上做增量,除非任务明确说明改动点。
- KD 保持 0,直到 Task C 的判定条件满足(见 Task C)。
- 每个实验必须报 overall + 分数据集(EntiTables/WDC)的 direct/evidence/fused R@10、R@100、mrr@100、coverage@10,以及 epoch-0 与 raw 对照。
- 每个任务结果(配置、指标表、一句话结论、产物路径)追加到 `work/stage1_optimization_r2_20260829/RESULTS.md`。
- 所有新增开关向后兼容;新增代码配套测试,全套测试保持通过。

## 本轮总验收目标(四个数)

1. Student fused overall R@10 **≥ 37%**(超过 raw 的 36.67%);
2. Student WDC direct R@10 **≥ 60.9%**(raw − 1pt 以内);
3. fused coverage@10 **≥ 7%**(evidence 通道以健康分数参与融合);
4. Teacher ensemble 重排 overall **≥ 42%** 且 WDC ≥ raw。

---

## Task A:几何锚定关系学习——在健康基座上真正测 anchor + 分数据集 gate + alpha

**理由**:第一轮 anchor sweep 在被旧 KD 毁掉的 9% 基座上跑,等于没测。现在在最优配方上重测,anchor 的预期作用恰好是抑制"伤 WDC 的偏移"(结论 3 假设 A)。同时 checkpoint gate 目前只看 overall,938/1140 的 EntiTables 权重把 WDC 掉 7pt 稀释成 overall 1.3pt,gate 看不见回退。**方法论定位(论文视角,写实现时注意保留完整记录):anchor 不是工程补丁,是"R = I + ΔR、ΔR 受几何锚定约束"的方法部件,μ 谱系和各关系矩阵偏离 I 的程度(‖R−I‖_F per type-pair per epoch)都要记录,后者用于论文分析"模型自己选择学了哪些关系"。**

**实现**:
1. 在最优配方(结论 2)上,anchor 正则 `μ‖R−I‖²_F/d²` 扫 μ ∈ {0, 0.01, 0.1, 1.0}(μ=0 即第一轮 task3_fresh 复现,可直接复用其结果)。
2. **关系矩阵分组 μ**:新增 `--anchor-weight-evidence`,允许 evidence 相关的 4 个关系(table↔text、table↔image)使用独立的、更大的 μ(默认取 10×主 μ)。本任务先用统一 μ 跑完主 sweep,然后对最优 μ 补一组 evidence μ=10×主 μ 的对照(这是 Task B 的衔接点)。
3. **分数据集 checkpoint gate**:`--per-dataset-gate "wdc2k_v2:direct_recall@10>=0.609"` 之类的参数化约束——epoch 选择改为"满足所有 per-dataset 约束的 epoch 中 overall fused R@10 最大者";若无 epoch 满足约束,回退 epoch-0 并在 selection.json 标记 `gate_unsatisfied`。
4. **alpha sweep**:在 anchor 最优 μ 下,`--dataset-sampling-alpha` 扫 {0, 0.5, 1.0}(3 组),区分结论 3 的假设 B。
5. 每 epoch 把每个 type-pair 的 `‖R−I‖_F` 写入 history JSON(新字段 `relation_drift`)。

**验收/判读**:
- 存在 (μ, alpha) 组合使 overall ≥ 36.5% 且 WDC ≥ 60.9% → 固定该组合为新默认配方。
- 若所有组合 WDC 仍回退 > 1pt:对比 alpha=1 与 alpha=0 的 WDC 回退幅度判定假设 B;若 alpha 无关,做一组 **WDC-only 训练** 对照(只用 wdc2k_v2 的 path 列表,同配置)——若 WDC-only 也回退,支持假设 A(训练目标本身与 WDC 的表面相似分布冲突),把"WDC 用 μ→大/回退 epoch-0"记录为该数据集结论,主配方按 EntiTables 优化;若 WDC-only 不回退,支持假设 C(共享 R 干扰),上报讨论(候选方案:per-dataset R 残差,本轮不实现)。

---

## Task B:Evidence 关系的锚定保持 + 融合配置落地(依赖 Task A 的 evidence-μ 对照)

**理由**:evidence 关系在当前监督(稀疏路径信用分配)下只会退化(8.16%→2.37%),但 fusion sweep 证明 raw 质量的 evidence 在小权重下无害且 coverage 更高(7.46% vs 0)。**方法论定位:不做硬冻结**——用大 μ 让 evidence 关系停留在 I 附近(μ→∞ 的连续化),同时保留"识别出非平凡偏移"的可能;`R=I` 的 evidence 分数作为 **identity relation baseline** 永久保留在评测输出里,后续任何 evidence 关系学习(Task C 之后的蒸馏)都必须超过它才算数。

**实现**:
1. Task A 第 2 步的 evidence-μ 机制即本任务的训练侧;确认最优配置下 evidence R@10 保持 ≥ 8%(≈ epoch-0 水平)。
2. 融合默认配置改为 `weighted_rrf` 且 `evidence_weight=0.05`(fine sweep 中 fused 36.14%、coverage 7.28% 的档),并在评测里同时输出 e=0 与 e=0.05 两条 fused 线,后续所有实验默认带这两条线。
3. `evaluate_student_retrieval` 输出新增 `evidence_identity_baseline` 字段:用 R=I 重算 evidence 通道指标(实现上等价于用 raw 索引的 evidence 通道,直接复用 raw_embedding 评测结果中的 evidence 分项即可,不需要额外建索引)。

**验收**:最优配置下 fused overall ≥ direct − 0.3pt,coverage@10 ≥ 7%,evidence R@10 ≥ 8%。

---

## Task C:Teacher ensemble 重排——正交信号的组合使用(零训练,可最先跑)

**理由**:重训 teacher 的信号与 raw 几乎正交(Spearman≈0.01)却能重排 +3.25pt,说明它携带独立信息;但 WDC −13.9 和 MRR 0.213 < 0.274 说明单独使用会丢掉 raw 的表面相似信号。正交信号的标准用法是组合:raw 项兜底 WDC 与 MRR,teacher 项贡献 EntiTables 的语义增益。**这是 teacher "在线 rerank 层"角色能否立住的判决实验,也决定 KD 是否重新开启。**

**实现**:
1. 扩展 task2b 的 rerank 诊断脚本:重排分改为 `α·z(s_T) + (1−α)·z(cos(z_q,z_t))`,z(·) 为每查询候选列表内的 z-score 归一;α 扫 {0.3, 0.4, 0.5, 0.6, 0.7}。
2. 输出 overall + 分数据集的 R@10/MRR@100,以及每个 α 的 per-dataset 最优对比表。
3. **E→T 列表对齐检查**:审计 task7 训练数据生成逻辑,确认 E→T(text→table / image→table)边的负样本是否与 Q→T 一样来自 raw ANN top-k;若否(大概率),重新生成 E→T 检索对齐列表,在现有 checkpoint 上继续微调 2–3 epoch(lr 2e-5),复测 E→T 重排。
4. 若 ensemble 达标,把 ensemble 分数导出为可复用的重排接口(供 Stage-2 前置 rerank 与后续蒸馏目标使用)。

**验收/判读**:
- ensemble overall ≥ 42% 且 WDC ≥ 63.37%(raw)→ teacher rerank 层成立;**KD 解锁条件达成**:student 蒸馏目标改为蒸 ensemble 分数分布(新任务,待此结果后规划;或先做按数据集门控的 KD——EntiTables 样本 distillation-weight=0.3、WDC 样本=0,本身是干净 ablation)。
- ensemble 达不到 WDC ≥ raw → teacher 只作为 EntiTables 分析工具,KD 保持 0,重心回到 Task A/B 的 student 配方收尾。
- E→T 微调后重排仍 < raw 33.13% → evidence 关系蒸馏暂缓,Task B 的 identity 附近锚定成为 evidence 通道本轮终态,如实记录。

---

## Task D:最终配方整合 + 完整评测落盘

**理由**:前三个任务的最优组件需要合成一个可复现的最终 run,并产出论文表格需要的完整数字。

**实现**:
1. 用 Task A 最优 (μ, alpha) + Task B 融合配置,完整重跑一次 student(fresh PCA-1024, path-only, KD 按 Task C 判定),作为本轮 final checkpoint。
2. 完整评测:dev 上 overall + 分数据集 + 分通道 + coverage + identity baseline + raw 对照;若 Task C 达标,附 teacher ensemble 重排数字。
3. 产出汇总表(markdown)写入 `work/stage1_optimization_r2_20260829/FINAL.md`,包含:raw / identity-PCA(epoch-0)/ final student / teacher ensemble 四行 × 各指标列——这就是论文主表的雏形。
4. Hard-negative mining(原 Task 6)仍然冻结,除非 Task C 的 KD 解锁条件达成且 final student ≥ 本轮总验收目标——满足时可另行规划 mining 轮(挖掘索引用 final student,打分用重训 teacher)。

**验收**:FINAL.md 四行表齐全;final checkpoint 与全部配置可从 selection.json 复现。

---

## 执行顺序与决策树

```
Task C(零训练 ensemble 扫描 + E→T 审计)先行 ‖ Task A(anchor × alpha sweep)并行
  Task A:
    ├─ 有组合 WDC ≥ 60.9% → 固定配方 → Task B 确认 evidence 保持 → Task D
    └─ 全部回退 → alpha 对照判定假设 B;WDC-only 对照判定假设 A/C → 按判读记录,主配方按 EntiTables 优化 → Task D
  Task C:
    ├─ ensemble ≥ 42% 且 WDC ≥ raw → KD 解锁(蒸 ensemble / 按数据集门控),并入 Task D 的 final run
    └─ 未达标 → KD 保持 0,teacher 记录为 EntiTables rerank 工具
  Task D 汇总,mining 是否重启按 Task C/D 结果另行规划
```

## 产出要求

- 每任务独立目录 `work/stage1_optimization_r2_20260829/task{A,B,C,D}_*/`。
- RESULTS.md 逐任务追加;Task D 额外产出 FINAL.md 四行主表。
- 所有 sweep 的中间 checkpoint 只保留 best 与 epoch-0,索引只保留 best/latest,控制磁盘占用。
