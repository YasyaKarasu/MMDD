# Stage-1 第六轮计划:分数尺度归一化 + 自适应 τ + 保留 image 的模态平衡(2026-08-30)

## 背景:r5 已确认的关键事实(执行前必读)

### r5 主结果(task1/task2/taskX/FINAL)
1. **双通道、按湖自适应叙事成立,不是崩了**:
   - EntiTables:teacher 蒸馏重排全程正向(τ 单调升到 1.0;线上重排 +15.25),KD 链判定 **pass**(+0.11 [-1.07, 1.71])。
   - WDC:teacher 线上重排 no-op,但**蒸馏保住了 teacher 知识**——τ 归因的 "teacher residual"(τ=0.7 vs 纯 cosine)= **+2.48 [0.00, 4.95],`delta<0`≈5%,真实独立贡献**。
   - WDC 的 τ 曲线**非单调**(0→0.3→0.7 升,1.0 回落)→ teacher 有知识但噪声大,0.7 是判别力与噪声的平衡点。
2. **WDC teacher 线上重排 = no-op/负资产,确认为任务类型不匹配**。Task 2:混合负样本反而更差(-29.70% vs -23.27%),排除了"湖内负样本多样性不足"假设;**不要再救 WDC 线上重排**。
3. **Task X:保留 image 的"平衡"策略在 WDC teacher 上 +2.48 点(方向对、力度不够)**,EntiTables student +0.32;但 target-bound / text_only / balanced 三变体均未达 WDC evidence+coverage ≥3pt 的统计线。
4. 最终配置:τ(EntiTables=1.0, WDC=0.7)、蒸馏权重 0.3、PCA-1024 冻结 P、μ=0.1、in-batch 256、γ=10、γ_e=2、weighted_rrf(e=0.05)、logsumexp/top4、配对 bootstrap gate。

### 本轮核心机制假设(点 2 与点 4 共用的根因)
evidence 路径分数 = `s(Q,E) + s(E,T)`,两段用**独立学习的关系矩阵** `R_{table→text}`/`R_{table→image}` 与 `R_{text→table}`/`R_{image→table}`,**各有独立尺度**。训练后 R 漂移(EntiTables `‖R−I‖` 曾达 3),两侧幅值拉开,导致:
- **path 聚合**:LSE/加权和/CE 被幅值大的一侧主导 → 路径分数失真;
- **fusion**:RRF 只按 rank、丢弃幅值,evidence 分数一旦漂移就不可信 → 只能靠 `evidence_weight=0.05` 硬压。

**若成立,归一化(fusion 与 path 聚合共用)是零训练的根治,且无需改训练目标或模型结构。**

## 全局约束
- 沿用 r5 最终配置(τ 除外,见 Task C):recall_ks={10,20,30,40,50}、γ=10、γ_e=2、weighted_rrf(e=0.05)、logsumexp/top4、配对 bootstrap gate(iterations 10,000, seed 13)。
- **每个湖用本湖 Task-M teacher 作为唯一 teacher 源**(不动 teacher 本身);student 用冻结 PCA 冻结 P。
- **不得删除 image 模态**——任何模态改动必须"保留 image",只能重加权/过采样/改表示,不允许 text_only 方案进入最终结果。
- 所有对比必须**同一候选池**、新数字与 r5 基线用同池同 CI 对齐。
- 新增开关向后兼容;新增代码配测试,全套测试保持通过。
- 结果写入 `work/stage1_optimization_r6_20260830/RESULTS.md`;最终 FINAL.md。
- **WDC 教师线上重排 no-op 是诚实负结果,不列为待修复目标**(本计划不做"修 WDC 教师重排"的事)。

## 本轮验收目标
1. 确认/否定"分数尺度错配"假设(Task A),并据此给出归一化方案;
2. fusion(Task B)与 path 聚合(Task D)在归一化后的 fused R@10 / coverage / MRR 上超过 r5 基线(按湖、CI 口径);
3. τ 自适应启发式(Task C)自动复现/逼近 r5 的手动最优(EntiTables→1.0, WDC→0.7),且无需训练额外模型;
4. 保留 image 的模态平衡(Task E)在 WDC evidence 通道上给出正增益,EntiTables 不受损;
5. 主表定稿。

---

## Task A:每关系分数尺度诊断(零成本,先行)

**理由**:在动 fusion/聚合之前,先量化"幅度失配"是否真实存在,避免在两个不同问题上白忙。

**实现**:
1. 对 r5 最终 student 与冻结 embedding,针对每个 type-pair(尤其 `table→text`/`text→table`、`table→image`/`image→table`),在 dev 检索的候选池上统计分数分布:`mean/std/min/max`、`|mean|_ratio`(两侧幅值比)。
2. 同时比较 **raw cosine 的对应统计**(作为"无漂移"基准)——若 raw 两侧幅值接近、而 student 两侧幅值差 >2×,假设成立。
3. 输出 `taskA_scale_diagnostic/metrics.json` + 汇总表。

**判定**:
- 若两侧幅值比 >2×:进入 Task B/D 的归一化方案,并在论文写"关系尺度错配是 evidence 通道弱的根因之一"。
- 若两侧幅值比 ≈1:归一化可能帮助有限,Task B/D 仍跑但作为消融,不要基于错误假设下结论。

---

## Task B:Fusion 加权归一化(零训练)

**理由**:RRF 纯 rank、丢弃幅值,是"evidence 分数一旦漂移就不可信"的直接来源。归一化后再加权,能同时解决幅值错配与 rank 丢弃信息两个问题。

**实现**:
1. 在 `retrieve_zero_one_hop_detailed` 中,将 direct 分数与 evidence 分数**在每 query 候选池内分别归一化**,再加权融合:
   - 变体一:z-score(减均值除标准差);
   - 变体二:min-max;
   - 变体三:softmax / softmax-temperature(建议 T ∈ {1, 0.1, 0.3});
   - 变体四:RRC 在归一化分数上做(而非 rank),即 `wD/(k+norm_score_D) + wE/(k+norm_score_E)`(保留 rank 但用归一化幅值缩放);
   - 每变体下 evidence_weight ∈ {0.05(基线), 0.1, 0.25, 0.5}。
2. 对每个变体,用 r5 最终 student 与 raw embedding 两个检索器,在 dev(overall + 分湖)上输出 fused R@10、coverage@10、MRR@50、Δ vs 基线 + CI。
3. 产生`taskB_fusion_normalization/RESULTS.md`,并选出每个湖的最优变体——若各湖选不同变体,记录为按湖自适应的 fusion 配置。

**验收**:某变体在两个湖上(或至少 EntiTables)fused R@10 及 coverage 同时 ≥ 基线(CI 口径),且不损伤另一湖。若归一化对 coverage≥基线、对 fused 无显著差异,仍选 coverage 更优者(evidence 通道是当前薄弱点)。

---

## Task C:τ 自适应启发式(免模型,复现 r5 观察)

**理由**:r5 手动 τ(Enti=1.0, WDC=0.7)恰好可由"teacher 重排 delta 符号"解释,说明该规律可以形式化;不训模型、用 dev 做一次一维网格选标量超参即可。

**启发式规则(替代"trust teacher iff rerank 好"的错误版本)**:
1. 计算本湖 teacher **只重排** dev 的 delta(用标准化同池,已有);此 delta 反映 teacher 判别力的可信度。
2. **τ 上限 bracket**:
   - `delta ≥ +3pt`(teacher 明显正贡献)→ τ ∈ {0.5, 0.7, 1.0}(可到 1.0);
   - `delta < +3pt` 或为负(teacher 在线重排 no-op/噪声)→ τ ∈ {0, 0.3, 0.5, 0.7}(**上限 0.7**,因为 τ=1.0 时噪声放大已证,见 r5 WDC τ 曲线);
   - 这是"teacher 有知识但噪声大"情形下的合理 bracket,而非"teacher 差就少信"。
3. 在 bracket 内,用 **冻结模型**(student 用 epoch-0 PCA-1024,R=I 近似 cos;teacher 冻结)在 dev 上对每个候选 τ 算 fused R@10(仅改变 τ 重新组合缓存 logits,零训练),选最优 τ。
4. 用选出的 τ 训练 student(蒸馏权重 0.3),并对照 r5 手动 τ 的结果。

**实现/说明**:
- 关键简化:student(epoch-0)与 teacher 均冻结,τ 只改变 KD 目标,不改变 encoder;选 τ 的计算在训练前一次性完成,不需要训练多个 student。
- 若 bracket 规则选出的 τ 与 r5 手动值一致(Enti→1.0, WDC→0.7),即验证启发式正确;若略不同,取偏差小者为修正版规则。
- 加一个**特殊对照组**:τ=0(纯 cos,不开 teacher)——用于识别"该湖 teacher 有没有知识"的边界情形。

**验收**:τ 自动选择复现或逼近 r5 手动值;使用自适应 τ 的 student 在 fused R@10 上不小于对应 r5 手动 τ 的结果(CI 不劣)。

---

## Task D:Path 聚合方法枚举 + 每边归一化(零训练)

**理由**:path 聚合是 evidence 通道最薄弱处(联合可达率是上限瓶颈),LSE 仅其一;LSE 分数式聚合对尺度敏感,每边归一化可根治;多方法枚举才能定位真正的上限。

**实现**:
1. 在 `PathAggregator` 加入变体:max、topk_mean、topk_sum、logsumexp(基线)、softmax_weighted_mean(温度参数)、power_mean(p=2, 3)、comb_mnz,每者前可选"分边归一化"——将 `s(Q,E)` 与 `s(E,T)` 各自在候选 evidence 中 z-score 后再相加。
2. 每个变体 × 每边归一化(开/关)在 r5 最终 student 与 raw 检索器上跑,输出 evidence R@10、coverage@10、fused R@10,overall + 分湖,与 r5 基线对比。
3. 汇总每个变体的最优参数与归一化开关。

**验收**:某种聚合(尤其带有每边归一化的)在 evidence R@10 或 coverage@10 上 ≥ 基线(CI 口径),且 fused 不劣。若 LSE+归一化已足够,选择它并说明"其余方法在 scale 修正后差别不大"——这本身是有效结论。

---

## Task E:保留 image 的模态平衡(融合 Task X 的正方向)

**理由**:Task X 的"平衡(保留 image)"在 WDC teacher 上 +2.48 点,是唯一方向为正的;只是力度不够。本任务系统性强化,并**保持 image 不删**。

**实现**:
1. **每关系 loss 权重**:对 WDC train 中稀疏的 image 关系(`table→image`/`image→table`,约 149 条)在训练 loss 中设受控权重(如 `--relation-weight table_to_image=2, image_to_table=2`),使 image 边在梯度中不被 text 淹没。为"过度放大稀疏边噪声"问题设上限,对比权重 ∈ {1(现状), 2, 4}。
2. **受控过采样**:仅当有真实 image 证据可扩充时,对 WDC image 边做 upsampling;无真实证据时不人工造 image 边(避免合成噪声)。
3. **关系尺度归一化(源头)** :在 student 打分 `s_S(a,b)` 上,按 type-pair 除以各自 `‖R‖` 或整个 type-pair 在校验集上的分数 std,使打分在跨模态间尺度可比(与 Task A/B 同思路,但在打分源头做)。或者用维度缩放 `s/√dim`。
4. **更好的 image 表示(可选)**:teacher 的 `image_latents` 从 24 上调(如 32/48),让 image 证据的 teacher 侧表示更细;并考虑对 image embedding 做 L2 归一化后再进投影(与 text 一致),消除 image/text 范数差异。
5. 所有变体在 dev(overall + 分湖)上输出 evidence R@10、coverage@10、fused R@10,与 r5 基线对比;**EntiTables 作为对照湖,确认改动只对 WDC 预期生效、不损伤 EntiTables**。

**验收**:
- WDC 的 evidence R@10 与 coverage@10 较 r5 基线有正增益(≥ +1pt,CI 口径),且 EntiTables 不劣;
- 达到后,将最优变体选为 WDC 默认,并在论文中记录"保留 image 的模态平衡可提升 WDC 证据通道"。

---

## Task F(可选,时间允许):表表示增强

**理由**:当前表按 `max_rows=12` 序列化,teacher 的细粒度信号和证据通道的 query 表示受限。此改动主要惠及 EntiTables 的 evidence/teacher 细粒度信号,**不救 WDC 教师重排**。

**实现**:
1. F1 增加序列化内容预算:`max_rows` 12→20,并按 `column_name: 值` 而非仅 `value` 拼接。
2. F2 保持原始 `max_rows=12` 与 value 序列化,将 schema/每行从 1 个均值池化 token 改为最多 4 个连续分段池化 token。
3. 两个变体分别重新生成所需表特征并独立评测;F1 影响 Student 整表 embedding 与 Teacher,F2 只改变 Teacher token 表示,Student 指标按构造保持不变。
4. 由于需要重跑特征缓存,本任务排在其余之后;磁盘紧张时复用未改变的 base/text/image 特征。

**验收**:分别报告 F1/F2 的 EntiTables 与 WDC teacher rerank delta;F1 另报告 student evidence R@10。若无收益,区分“序列化内容预算”和“每行 Teacher token 预算”两个机制结论。

---

## 执行顺序与决策树

```
Task A(尺度诊断,零训练,先行)
  ├─ 幅值比 >2× → 确认假设 → Task B(fusion 归一化) + Task D(path 聚合归一化) 按同一套归一化实现
  └─ 幅值比 ≈1 → Task B/D 仍跑,但作为消融
Task C(τ 自适应,零训练) 独立于 A,B,D —— 可最先或并行
Task E(保留 image 模态平衡) 使用 Task A 的尺度归一化成果;顺带测 image_latents 上调
Task F(表表示增强,可选) 若时间允许,作为收尾

最终:Task B/D/E 最优变体 → Task C 自动 τ → 重跑完整评测 → FINAL.md(修正/更新叙事)
```

## 产出要求
- 每任务目录 `work/stage1_optimization_r6_20260830/task{A,B,C,D,E,F}_*/`;RESULTS.md 逐任务追加;FINAL.md。
- Task A 尺度统计表、Task B/D τ/聚合变体对比表、Task C 的 τ 自动选择 vs 手动对照、Task E 的 WDC/Enti 对比,均附配对 CI。
- checkpoint/索引只保留 epoch-0/best/final;image 模态在任何最终配置中保留。
- 把审计 9.3 的只读不变量(尤其"image 模态不删""同池对比""双 teacher provenance")固化为测试。
