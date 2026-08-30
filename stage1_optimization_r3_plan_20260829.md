# Stage-1 第三轮计划:k-驱动检索接口重构 + 评测口径切换 + 优化实验(2026-08-29)

## 背景:第二轮(work/stage1_optimization_r2_20260829)已确认的结论

1. **最终 student = Task A μ=0.1/alpha=0 的 epoch-0**(fresh PCA-1024,冻结 P,R=I 附近):overall direct R@10 = 35.53%,WDC 62.38%,evidence 8.95%,coverage 7.54%。训练(无论纯监督还是 KD)沿同一条 trade-off 曲线移动:`relation_drift`(‖R−I‖_F)每增加,EntiTables 涨、WDC 双倍幅度跌,anchor 只能减速不能改方向;alpha 无关(过采样假设已排除)。
2. **Teacher z-score ensemble(α=0.7)是本项目目前最强的排序器**:overall 重排 R@10 = 47.37%(raw+10.7),WDC 64.85% ≥ raw,EntiTables 43.60%,E→T 重排 43.69%(raw 33.13%)。配置在 `taskC_teacher_ensemble/ensemble_config.json`。
3. **KD(蒸 teacher logits)已带完整证据被否定**;蒸 **ensemble 分数** 是 KD 唯一剩余的候选方向(ensemble 在 WDC 上是正增益,监督信号本身不携带伤 WDC 的偏移)。
4. **WDC dev 只有 202 查询、每查询 1 个正例**:1pt = 2 条查询,当前 WDC gate(≥60.9%)工作在噪声量级,历次"gate fail"可能只是 2–5 条查询的波动。
5. 在线架构事实上已是两级:**student ANN 召回 → teacher-ensemble 重排**。ensemble 的天花板受召回池深度限制。

## 本轮两部分:先改代码(Part 1),再做实验(Part 2)。Part 1 全部完成并通过测试后才开始 Part 2。

---

# Part 1:代码修改(三项,全部带测试)

## 修改 1:评测口径切换 —— recall_ks 默认 {10, 20, 30, 40, 50}

**理由**:领域内 joinability discovery 论文的标准口径是 recall@{5,15,25} 或 recall@{10..50},不测 @100。现有 `RECALL_KS=(1,5,10,50,100)` 硬编码,mrr@100 同样超出论文口径。

**实现**:
1. `src/mmdd_stage1/evaluation.py`:`RECALL_KS` 从模块常量改为 `evaluate_student_retrieval(..., recall_ks: tuple[int,...] = (10,20,30,40,50))` 参数;MRR 截断改为 `mrr@{max(recall_ks)}`(默认 mrr@50)。`positive_evidence_path_coverage@10` 保持 @10 不变(它是 Stage-2 gate 口径)。
2. `src/train_stage1.py`:新增 `--recall-ks`(逗号分隔,默认 `10,20,30,40,50`);primary metric 默认仍为 `recall@10`;所有 history/selection JSON 按新口径记录。
3. 所有引用 `recall@100` / `mrr@100` 做判断的代码路径(gate、selection、报告脚本)改为引用参数化的最大 k。
4. 注意:历史报告的 recall@100/mrr@100 数字不迁移、不换算,后续对比一律用本轮 Part 2 重新评测的基线(见 Task E)。

## 修改 2:k-驱动的检索预算 —— gamma·k 派生,每个被测 k 独立跑

**理由**:当前 `direct_k=100, evidence_k=50, targets_per_evidence=50` 是写死的绝对常量:k 小的时候浪费算力,k 大的时候召回不足,论文里还多一个无法辩护的死参数。改为一切从用户请求的 k 派生:`直接通道预算 = γ·k`,`evidence 预算 = γ_e·k`。γ 是可用敏感性曲线辩护的扩展系数(two-stage retrieval 的惯例:rerank depth as multiple of k)。

**实现**:
1. `src/mmdd_stage1/retrieval.py`:
   - `retrieve_zero_one_hop_detailed` / `retrieve_zero_one_hop` 新签名:以 `k`(用户请求的返回数)+ `gamma`(默认 4)+ `gamma_evidence`(默认 2)为主接口,内部派生 `direct_k = gamma*k`、`evidence_k = gamma_evidence*k`、`targets_per_evidence = gamma_evidence*k`。原有绝对参数保留为可覆盖的高级选项(显式传入时优先于派生值),缺省时走派生逻辑——向后兼容,老调用不炸。
   - **ef 联动**:`StudentANNIndices` / `RawEmbeddingANNIndices` 查询前 `set_ef(max(ef_search, requested_k))`,否则 hnswlib 在 k > ef 时静默降质。
2. `src/mmdd_stage1/evaluation.py`:**每个被测 k 用各自的 γ·k 预算独立跑检索**(不是跑一次大池子取前缀——RRF 分数依赖池内排名,k=10 的 top-10 不保证是 k=50 池子的前缀,必须与线上行为一致)。实现上对每个 k∈recall_ks 各调用一次 `retrieve_zero_one_hop_detailed(k=k, gamma=γ)`,`recall@k`/`coverage@10` 从对应池子计算;`mrr@50` 用 k=50 那次的结果。对 dev 1,140 查询 × 5 个 k,检索次数 ×5,注意用 `search_many` 批量化控制耗时。
3. `src/train_stage1.py` 透传 `--gamma`、`--gamma-evidence`;fusion 配置(weighted_rrf, evidence_weight=0.05)不变。
4. 训练中每 epoch 的 dev 评测同样走新口径(成本可控:必要时训练期只测 k∈{10,50} 两档,final 评测测全五档,加开关 `--train-eval-ks`)。

## 修改 3:WDC dev 噪声 —— 配对 bootstrap 置信区间 + 噪声感知 gate

**理由**:WDC dev 202 查询、单正例,1pt=2 查询;第二轮的 gate(硬阈值 60.9%)在 3–5 条查询的波动量级上做判定,ep1 的"fail"(59.90% vs 60.89%)统计上无法与噪声区分。**方案选择:配对 bootstrap(方案 A),不扩 dev 集(方案 B 会破坏与前两轮的可比性且需要新标注),不简单放宽容差(方案 C 无统计依据)。** 配对 bootstrap 直接量化"student 与 raw 在同一批查询上的差值"的不确定性,是 IR 评测的标准做法。

**实现**:
1. 新模块 `src/mmdd_stage1/significance.py`:
   - `paired_bootstrap_delta(per_query_a, per_query_b, iterations=10_000, seed=13)`:输入两个系统在同一查询集上的 per-query recall@k 值,重采样查询(有放回),返回 delta 的均值、95% CI(2.5/97.5 分位)、以及 `P(delta<0)`。
   - `evaluate_student_retrieval` 增加可选输出 per-query 指标(`return_per_query=True`),供 bootstrap 使用。
2. **gate 判定改为噪声感知**:`--per-dataset-gate` 的语义从"点估计 ≥ 阈值"改为"**相对 raw 的配对 delta 的单侧 95% CI 下界 ≥ −0.02**"(即:有统计把握说 WDC 回退不超过 2pt 才算 pass;`--gate-tolerance 0.02` 可调)。raw 的 per-query 结果在评测时顺带产出,无额外检索成本。
3. 所有 RESULTS/FINAL 表格中,WDC 与 EntiTables 的关键对比(vs raw)都附 95% CI,格式如 `62.38% (Δ vs raw +0.50, CI [−2.97, +3.96])`。
4. 测试:构造合成 per-query 数据验证 CI 覆盖性与配对性(同一查询的两个系统值必须一起被重采样)。

**Part 1 完成标准**:全套测试通过;用第二轮 final checkpoint(`taskA_anchor_mu0.1_alpha0/student_path.pt`)在新口径下跑一次评测冒烟,确认新旧代码在 k=10、γ 覆盖为绝对值 100 时结果与旧口径一致(回归校验)。

---

# Part 2:实验(依赖 Part 1)

所有实验产物写入 `work/stage1_optimization_r3_20260829/`,结果逐任务追加到该目录 RESULTS.md。语料/特征/dev/aggregator 约束同前两轮。基线 student = 第二轮 final checkpoint(epoch-0, μ=0.1),不重训,除非任务说明。

## Task E:新口径基线重建(先跑,所有后续对比的地基)

**理由**:口径从 @100 换到 @{10..50}、预算从固定 100 换到 γ·k,历史数字全部失效,必须重建四条基线。

**实现**:γ=4、γ_e=2 下,对 dev 评测四个系统:(1) raw embedding;(2) final student(epoch-0);(3) raw + teacher ensemble α=0.7 重排(重排窗口 = γ·k);(4) student 召回 + ensemble 重排(**新组合,第二轮没测过**:用 student 的 γ·k 召回池喂 ensemble——这是实际部署形态,student 索引只有 raw 的 1/4 大)。
每个系统输出 recall@{10,20,30,40,50}、mrr@50、coverage@10,overall + 分数据集 + CI(vs raw)。

**验收**:四行 × 五 k 的主表落盘 `taskE_baselines/RESULTS.md`。特别记录系统 (4) vs (3) 的差距——若 (4) ≈ (3),论文可以主推"紧凑 student 召回 + ensemble 重排"的完整故事(索引小 4 倍、效果不减)。

## Task F:γ 敏感性扫描(零训练,出论文辩护图)

**理由**:γ 是新接口唯一的自由参数,必须用数据选择而不是拍脑袋,论文里一张饱和曲线即可辩护。

**实现**:γ ∈ {2, 4, 6, 8, 10},γ_e 固定 2,对 Task E 的系统 (2) 和 (4) 各跑 recall@10 与 recall@50 两条线(即每系统 2×5 个点);同时记录每 γ 的平均每查询检索+重排耗时(wall-clock,CPU 检索即可,注明硬件)。输出 CSV + PNG(x=γ,y=recall,两 k 两系统四条线;副图 y=latency)。

**验收**:确定"指标饱和的最小 γ"(预期 γ* 附近曲线平坦),写入 `taskF_gamma_sweep/RESULTS.md`,并把该 γ* 设为后续默认。若 recall@50 在 γ=10 仍未饱和,报告该现象并临时取 γ=10,标记"召回层深度是瓶颈"供下轮处理。

## Task G:γ_e 与 evidence 预算敏感性(小规模,附带)

**理由**:evidence 通道预算是乘性的(`γ_e·k` 条 evidence × 每条 `γ_e·k` 个 target),γ_e 对 coverage@10 与耗时的影响需要一次性摸清,避免它成为下一个"死参数"。

**实现**:γ 固定为 Task F 的 γ*,γ_e ∈ {1, 2, 4},对系统 (2) 测 coverage@10、evidence recall@10、每查询耗时。

**验收**:选 coverage 不再增长的最小 γ_e,更新默认值,记录到 RESULTS.md。

## Task H:Ensemble-KD —— 蒸馏目标换成 ensemble 分数(KD 的最后一次机会)

**理由**:第二轮证明蒸 teacher logits 必伤 WDC(teacher 在 WDC 单独重排 −13.9);但 ensemble 在 WDC 是 +1.5,监督信号本身不携带伤 WDC 的偏移。若 ensemble-KD 能让 student 在 WDC gate(新的 CI 口径)下超过 epoch-0,蒸馏叙事复活;若仍失败,KD 方向带完整证据链关闭。

**实现**:
1. 离线预计算:对现有 path 训练列表的每个 (query, candidate) 对,算 `0.7·z(s_T) + 0.3·z(cos(z_q, z_t))`(z-score 在每查询候选列表内归一;teacher 分数用重训 teacher `checkpoints_cache24k`,cos 用冻结 embedding),存为新的 teacher_logits 字段(direct 与 evidence 通道分别算;evidence 通道的 ensemble 分数按同一公式作用于 E→T 边分数)。复用现有 KD 管线,不改损失代码。
2. 训练:第二轮 final 配方(fresh PCA-1024、path-only、in-batch、μ=0.1、alpha=0)+ `distillation-weight` ∈ {0.3, 1.0} 两组;gate 用修改 3 的 CI 口径(tolerance 0.02)。
3. 对照:同配置 KD=0(即第二轮 final 的复评,Task E 已有)。

**验收/判读**:
- 存在 KD 权重使 overall recall@10 > epoch-0 且 WDC CI-gate pass → ensemble-KD 成立,选该 checkpoint 为新 final student,并用它重跑 Task E 系统 (4)。
- 两组都 gate fail → 在 RESULTS.md 写明"KD 方向关闭:teacher-logits KD 与 ensemble KD 均无法在不伤 WDC 的前提下改进 student",student 终态维持 epoch-0,蒸馏在论文中作为 negative result 章节。

## Task I:最终汇总

**实现**:用最终确定的 (γ, γ_e, student checkpoint, ensemble 配置) 重跑完整评测,产出 `FINAL.md`:
- 主表:四系统 × recall@{10,20,30,40,50} + mrr@50 + coverage@10(overall + 分数据集,关键格附 CI);
- γ 敏感性图与 γ_e 选择记录的引用;
- 明确记录所有默认参数的最终值(recall_ks、γ、γ_e、fusion、gate tolerance)与各 checkpoint SHA-256。
- Hard-negative mining 仍冻结;若 Task H 成功且系统 (4) 在新口径下 overall recall@10 ≥ 45%,在 FINAL.md 里给出 mining 重启建议(挖掘用 student 索引、打分用 ensemble),留待下轮。

## 执行顺序与决策树

```
Part 1(修改 1→2→3,回归冒烟通过)
  → Task E(四基线重建)→ Task F(γ 扫描)→ Task G(γ_e)
      └─ F 中 recall@50 未饱和 → 标记召回深度瓶颈,γ 取 10
  → Task H(ensemble-KD, 2 组)
      ├─ 成功 → 新 final student → 重跑 E 系统(4) → Task I
      └─ 失败 → KD 关闭记录 → Task I(student 维持 epoch-0)
```

## 产出要求

- Part 1 的三项修改分开提交(或分开成三个明确的变更集),每项配测试;不破坏现有测试。
- 每任务独立目录 `work/stage1_optimization_r3_20260829/task{E,F,G,H,I}_*/`;RESULTS.md 逐任务追加;Task I 产出 FINAL.md。
- 敏感性扫描输出 CSV(供论文重绘图)+ PNG(快速查看)。
- 索引与 checkpoint 只保留 best/epoch-0/final,控制磁盘。
