# Stage-1 第五轮计划:蒸馏机制归因 + WDC Teacher 可行性 + 主表定稿(2026-08-30)

## 背景:已确认的关键事实(执行前必读)

### 审计结论(task_audit_wdc_teacher/audit_report.md,2026-08-30)
1. **无数据生成 bug**。分湖拆分精确互斥(逐类型 EntiTables+WDC=Mixed、无重复、无跨湖 evidence、无 query 错湖);Task-M WDC 负样本全部来自本湖 raw ANN(224,864/224,864)。"蒸馏源 teacher ≠ 在线重排 teacher"是明确的 gate fallback(有意的 Test-L 控制逻辑),不是漏改。
2. **同候选池复算**:混合 teacher 在 WDC-local top-100 上重排 −18.32% [−25.25,−11.39],WDC-only teacher −23.27%(统计上真实,非噪声);两者差异 −4.95% [−11.39,+1.49] **不显著**。
3. **WDC 训练分布风险**(审计 8.2):(a) 湖内负样本多样性降低;(b) query-hard evidence 高度集中(top-10 占 36.23%);(c) image 关系极度稀疏(WDC train 中 image→table + table→image 仅 149 条 vs text 4,680 条)。
4. 审计建议优先级:mixed-negative ablation > evidence-binding / modality-balance ablation > 扩充数据 > 原配置重训。**不要盲目补数据。**

### r4 结果(FINAL.md)
| 湖 | raw direct | supervised student | KD student(Task-M 本湖 teacher) | student+ensemble |
|---|---|---|---|---|
| EntiTables | 30.92% | 37.42% | **37.53%**(fused) | **45.52%** |
| WDC | 62.38% | 61.88% | **64.36%** | 64.36%(重排 no-op) |

索引:tudent 为 raw 的 1/3.79;student+ensemble 每查询 EntiTables 16ms / WDC 25ms。

### 核心待证命题(本轮要钉死)
用户关键问题:**"为什么 teacher 本身重排很弱(−18%/−23%),蒸馏却能带来增益(EntiTables +0.11, WDC +2.48)?"**
拟验证的机制假设:蒸馏目标含 0.7·z(s_T) + 0.3·z(cos),其中的 **cos 锚 + 低容量 student + 锚定正则** 使 teacher 的错误被结构性过滤;teacher 提供的更多是"让训练变稳定"的锚,而非可迁移知识。WDC 上 supervised 训练恒回退(选中 epoch 0),而带蒸馏目标能训到 epoch 5 且转正——这是"锚定作用"的直接证据。

**本轮验收目标**:
1. 归因消融(Task 1)给出每湖"蒸馏增益来自 teacher 知识 vs 来自锚定"的定量划分;
2. WDC teacher 若可修(Task 2),重排转正或接近 raw;若不可修,明确判定为"no-op/负资产",不影响主线;
3. FINAL.md 产出双通道叙事的主表,并修正此前 mislead 的 "distillation chain: fail" 文字。

## 全局约束
- 沿用 r4 最终配置:recall_ks={10,20,30,40,50}、γ=10、γ_e=2、weighted_rrf(evidence 0.05)、aggregator logsumexp/top4、paired bootstrap gate(iterations 10,000, seed 13)、湖级 CI gate(tolerance 0.02)、PCA-1024 冻结 P、μ=0.1、in-batch max 256。
- **KD 目标参数**:现有 `teacher_logit_mode="ensemble"`, `teacher_ensemble_alpha=0.7` 表示目标 = 0.7·z(s_T)+0.3·z(cos)。**本轮引入可配置的 α`τ`(teacher 系数)**:τ=1 → 纯 teacher,τ=0 → 纯 cosine,τ=0.7 → 现状。
- **同一候选池约束**:任何 teacher 对比必须在同一 corpus、同一 raw ANN pool、同一 γ·k 下进行,并报配对 CI。引用历史数字时必须注明候选池。
- 每任务结果追加 `work/stage1_optimization_r5_20260829/RESULTS.md`;最终产出 FINAL.md。
- 新增开关向后兼容;新增代码配测试,全套测试通过。

---

## Task 1:蒸馏目标 τ 归因消融(本轮最高优先级机制实验)

**理由**:直接回答"重排弱为何蒸馏仍增"。τ 从纯 teacher(1.0)扫到纯 cosine(0.0),观察每湖 student 的增益如何随 τ 变化,区分"teacher 知识"与"cos 锚定"两个来源。

**实现**:
1. **teacher logits 缓存按 τ 重新生成成本可控**:teacher 冻结、特征已缓存,只需按不同 τ 重算目标分数(τ=0 时无需 teacher,直接用冻结 embedding 的 cosine)。加 `--kd-target-teacher-alpha` 参数,支持重新生成/复用缓存(缓存 key 含 τ)。**τ=0.7 的现有缓存直接复用,不重算。**
2. **每湖固定自家 Task-M teacher 为蒸馏源**(EntiTables:`taskM_entitables_teacher/checkpoints/teacher_path.pt`;WDC:`taskM_entitables_teacher/per_lake/wdc/checkpoints/teacher_path.pt`)。蒸馏权重 `distillation-weight` 固定 0.3(现状赢家)。KD 仅用于 path 阶段;frozen PCA + μ=0.1 不变。
3. **τ 扫描矩阵**(每湖):τ ∈ {0.0(纯 cosine), 0.3, 0.7(现状), 1.0(纯 teacher)},加一组 τ=0 且 distillation-weight=0(即纯监督对照,复用 r4 数据)。共 每湖 4 个训练 run(τ=0.7 可复用 r4 现成 checkpoint,实际新增 3 个)。
4. **评分口径**:fusion 用 weighted_rrf(e=0.05);**主指标用 fused R@10**(与部署一致,而不是 direct),并同时报 direct;选中 epoch 仍用湖级 CI gate(本湖 raw 为基准)。
5. 每 epoch 记录 relation_drift(‖R_tt−I‖_F)、direct/fused/evidence、CI-vs-raw、selected epoch/gate。

**验收/判读**:
- **WDC 上 τ 单调性**:若随 τ 增大增益下降、τ→0 增益最大 → 增益来自 cos 锚定,teacher 知识贡献≈0;若 τ=1 与 τ=0 都显著高于纯监督且相近 → 锚定与知识并存;若只有 τ≥0.3 有增益 → teacher 知识是必要成分。
- **EntiTables 上 τ**:预期 τ=0.3 与 τ=0.7 相近(现状),τ=1 略降(teacher 噪声放大),τ=0 略低于 τ=0.3(teacher 有少量正贡献)。
- 产出"每湖 τ-增益曲线"(增益 vs τ,横轴 τ,纵轴 fused R@10 相对 raw 的 delta),作为论文机制章节的图。

---

## Task 2:WDC Teacher 负样本来源消融(审计 9.2.1)

**理由**:审计判定 WDC teacher 重排差的根因是"训练目标/湖内负样本分布与 WDC 表面相似检索不匹配",而非数据 bug。最直接的单变量测试:WDC 的正例与评测保持湖内,仅把 teacher **训练负样本**换成 mixed 来源(即重建检索对齐列表时,负样本从混合 raw ANN 取)。这测试"湖内负样本过于相似(多样性不足)"是否就是打垮 teacher 的原因。

**实现**:
1. 新建 WDC teacher 训练列表:正例=WDC 本湖正例;负样本从 **mixed raw ANN** 的 top-k 取(排除正例与 WDC 检出表);候选宽度 16 不变。
2. 用该列表重训 WDC teacher(edge→path,同参数),**重排门禁**用同候选池重算:本湖 raw top-100 重排 R@10 是否 ≥ 本湖 raw − 3pt(即相对 r4 的 −18% 有明显改善)。
3. 若 teacher 重排显著改善,继续用该 teacher 重跑 WDC 的 τ 消融(Task 1 中 τ 组),看蒸馏增益是否进一步上升。
4. **同时做 audit 9.2.2 的 evidence-binding 变体**:把"query-hard evidence 复制"改成"ANN target 自身的 target-bound evidence",以及把 image 方向设最低样本门槛(如 train image 边 < 500 则从列表剔除或平移至 text)——这两个改动与负样本消融互为正交,可各跑一组。

**验收/判读**:
- 若 mixed-negative 使 teacher 重排转正或接近 raw → WDC teacher 可修,根因是湖内负样本多样性不足;论文可加一行"负样本来源是关键"。
- 若仍未改善(重排仍为负)→ 强化结论 C(WDC 任务类型与 teacher 训练目标不匹配),WDC teacher 定为 no-op/负资产,主线不受影响;把该负结果写进论文,作为"teacher 在表面相似主导湖上的边界"。

---

## Task 3(可选,时间允许):WDC Student 的"锚定 vs 知识"决定性对照

**理由**:Task 1 的 τ 消融已能归因,但若想彻底钉死"WDC +2.48 是锚定而非 teacher 知识",可加一组:distillation 目标 = 纯 cosine(τ=0)且 **不使用 teacher 打分**(即从训练中完全移除 teacher 信号)。若该组也达 ~+2,则 WDC 上 teacher 在蒸馏中的贡献被直接归零。

**实现**:复用 Task 1 的 τ=0 run;训练配置完全一致;输出与 τ=0.3/0.7/1.0 及纯监督(non-KD)放在同一张表,四行对照。

**验收**:表格附 CI;结论归为"锚定主导"或"teacher 贡献≥锚定"。

---

## Task X:Evidence 绑定与模态平衡消融(独立轴,不混入 Task 2)

**理由**:审计 9.2 的三条分布风险中,负样本多样性已由 Task 2 覆盖;剩下两条——query-hard evidence 高度集中(WDC top-10 占 36.23%)与 image 关系极端稀疏(WDC train 仅 149 条)——是**独立于负样本来源**的轴,必须单独成组,否则与 Task 2 混跑无法归因。

**实现**(建立在审计 9.2.2 与 9.2.3,并复用于 teacher 与 student 两侧训练列表):
1. **Evidence-binding 消融**(审计 9.2.2):在重建 WDC 训练列表时,把当前"query-hard evidence 复制"改为"**ANN target 自身的 target-bound evidence**"(即每个新 ANN target 用属于它自己的 evidence 组,而非整组共享同一 query-hard evidence)。若 target-bound 数量不足,退回本湖 evidence 池随机/就近赋给该 target。
2. **Modality-balance 消融**(审计 9.2.3):对 WDC train 中 image→table + table→image 边设置**最低样本门槛**——若某关系在 train 中 < 500 条,将其从训练列表剔除,并把该关系的 logits 在融合/评估中按 `evidence-modality-weights` 降权(建议 image 权重 0.3);或受控过采样(仅当有真实 image 证据可扩充时)。
3. **对照组**:当前配置(现状 r4)、只用 text 通道(禁用 image)、target-bound evidence、balanced(受控过采样)四组,每组在 **WDC 湖** 上同时跑 teacher 与 student(蒸馏 τ=0.3),记录每组的 evidence R@10、coverage@10、fused R@10 与训练分布统计(image 边数、evidence top-10 集中度)。
4. **EntiTables 作为对照湖**:EntiTables 的 evidence 分布健康(image 2,981 条、top-1 3.87%),在其上重复以上变体,用以确认"改动只对 WDC 的 evidence 通道有预期作用,不损害健康湖"。

**验收/判读**:
- 若 target-bound evidence 或禁用 image 使 WDC 的 evidence R@10 与 coverage@10 明显改善(≥ +3pt),且 EntiTables 不受损 → 证据分布修复有效,写入论文作为 evidence 通道的按湖设置;WDC coverage 有望接近 raw。
- 若两者都无法改善 → 确认 WDC 证据通道的瓶颈在证据本身的内容/embedding 质量,而非训练分布;evidence 通道在 WDC 如实标记为低效,主线不受影响。
- 明确报告"image 稀疏(149 条)是否被禁用/降权完全解决"——若解决,则该负 result 有独立论文价值(说明 image 证据在 WebTable 型湖中的数据稀缺性)。

**产出**:`taskX_evidence_modality_ablation/RESULTS.md`(WDC 与 EntiTables 两组 × 四变体 × teacher/student 的 evidence/fused 指标表);训练分布统计(每关系的边数、evidence 集中度)作为补充证据。

---

## Task 4:主表定稿 + 叙事修正(报告任务,零新训练)

**理由**:r4 的 FINAL.md 将 EntiTables 标为 "distillation chain: fail",是用 direct(37.31<37.42)而非 fused(37.53>37.42)口径得出的,且未区分 teacher 版本。此判定文字对论文叙事有误导,需修正;并把双通道叙事与 Task 1/2 结果整合成最终主表。

**实现**:
1. 用最终确定配置(每湖最优 student + 最优 reranker 决策)重跑完整评测,产出 FINAL.md:
   - 每湖主表:raw / supervised student / KD student(最终) / student+ensemble × recall@{10..50} + mrr@50 + coverage@10 + CI;
   - **显式区分双 teacher provenance**(表脚注 + 正文说明):"KD 蒸馏源 = 本湖 Task-M teacher;在线重排器 = gate 决策(EntiTables=本湖 Task-M teacher, WDC=混合 teacher 或 no-op)"——引用审计 5.4;
   - 修正 "distillation chain" 判定:定义为 **fused R@10 上 KD ≥ supervised**(用本湖 Task-M teacher 作蒸馏源),结论改为"EntiTables 上 KD 略优(+0.11 不显著)、WDC 上 KD 显著优于 supervised(+2.48)",并把 Task 1 的 τ 曲线图与机制结论附上;
   - 若 Task 2 成功,追加 WDC teacher 修复后的行;若失败,追加"WDC teacher 判定为 no-op/负资产"的明确结论与依据(同池 CI)。
2. 报告每湖索引大小、每查询耗时(raw vs student vs student+ensemble)、PCA 方差保留率。
3. 记录所有最终默认值(τ、distillation-weight、gate、γ、γ_e、fusion)与 checkpoint/语料 SHA-256。

**验收**:FINAL.md 可作为论文主表直接引用;无"distillation chain: fail"的误导性表述。

---

## 执行顺序与决策树

```
Task 1(τ 归因消融,每湖 3-4 run;τ=0.7 复用 r4)→ Task 4(先出修正版主表,把 Task1 结论写进去)
Task 2(混合负样本重训 WDC teacher)→ 成功后回填 Task 1 的 WDC τ 组;失败则写入 no-op 结论
Task X(evidence-binding + modality-balance,与 Task 2 正交、独立成组)→ 结果补充到 WDC/EntiTables 主表的 evidence 行
Task 3(可选,纯 cosine 对照)→ 依 Task 1 结果决定是否必要
最终 Task 4 整合 → FINAL.md
```

## 产出要求
- 每任务目录 `work/stage1_optimization_r5_20260829/task{N}_*/`;RESULTS.md 逐任务追加;Task 4 出 FINAL.md。
- Task 1 的 τ-增益曲线图(CSV+PNG);Task 2 的同候选池重排对照表(附 CI)。
- checkpoint/索引只保留 epoch-0/best/final。
- 修改只读审计不变量(见审计 9.3)固化为测试,确保后续不回归。
