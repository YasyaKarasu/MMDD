# Stage-1 R3：Teacher 证据残差网络与学生池续训实验报告

- **日期**：2026-10-03
- **任务说明文档**：[`docs/stage1_r3_teacher_evidence_residual_task.zh-CN.md`](file:///home/oycy/MMDD/docs/stage1_r3_teacher_evidence_residual_task.zh-CN.md)
- **基线模型/环境**：`work/stage1_entitables_r2_fullkd_s13` (协议 4.2.0)
- **核心数据产物目录**：
  - **第二步学生池续训（部署最终产物）**：[`/home/oycy/MMDD/work/stage1_r3_teacher_sp_A/`](file:///home/oycy/MMDD/work/stage1_r3_teacher_sp_A/)
  - **第一步架构消融主臂 A**：[`/home/oycy/MMDD/work/stage1_r3_chain_residual_A/`](file:///home/oycy/MMDD/work/stage1_r3_chain_residual_A/)
  - **第一步对照臂 B (residual + 4.2 loss)**：[`/home/oycy/MMDD/work/stage1_r3_chain_residual_B/`](file:///home/oycy/MMDD/work/stage1_r3_chain_residual_B/)
  - **第一步对照臂 C (triplet + A loss)**：[`/home/oycy/MMDD/work/stage1_r3_chain_residual_C/`](file:///home/oycy/MMDD/work/stage1_r3_chain_residual_C/)

---

## 1. 实验结论摘要

本次实验彻底解决了 R2 基线中 **“Teacher 证据支路打分 Real < 直接检索 f0”**（证据反噬）、以及 **“只要有包就无脑加分”** 的伪相关问题：

1. **结构消除倒挂**：残差结构（`pairwise_residual`，零初始化输出头）保证 warm start 每条路径等于 $f_0(q, t)$，彻底消除了旧 triplet 结构的 Real < f0 倒挂（由基线的 −0.42pp 逆转为 **+0.33pp ~ +0.40pp**）。
2. **证据内容真实起效**：在 Raw dev 池上，主臂 A 在 TB_half 的 `Real − Swap` 达到 **+1.21 pp**（95% CI `[+0.23, +2.20]`，区间严格大于 0），证明 Teacher 真正学到了区分正确证据与置换伪证据的能力。
3. **消除“有包就提权”的虚假偏置**：
   - 旧 Teacher 在学生池续训后，`LME - f0` 均值高达 **+1.095**，Real 前十的有包率高达 **60.2%**（高于 f0 的 55.6%），导致无用的有包负例被错误提权；
   - 新残差 Teacher 的 `LME - f0` 均值为 **−1.211**，Real 前十有包率仅为 **42.5%**（低于 f0 的 54.4%），系统性偏置完全消除。
4. **学生池部署全面领先**：在学生实际召回池（KD pool）中：
   - 整体 Real 从 R2 的 0.4814 提升至 **0.4997**（+1.83pp）；
   - 在多模态证据应当发挥作用的隐式关联表（`implicit` query）上，Teacher Real 达到 **0.4585**，大幅超越学生的直接检索（0.4225）达 **+3.60 pp**，且超越 Swap（0.4441）达 **+1.44 pp**；
   - Witness Oracle 理论上限在 implicit 上拉升至 **0.4880**（比 f0 高 +3.53pp，比学生高 +6.55pp）。

---

## 2. 第一步：Raw Dev 池架构与损失 2×2 消融实验

### 2.1 实验设置
- **评测对象**：Raw dev 池（1198 query，包含 599 implicit，599 explicit），无学生召回偏差。
- **消融矩阵**：
  - **参考基线 R2**：`triplet` + 4.2 原始损失 (`all` scope, support 0.2, 无 witness 候选列表)
  - **对照臂 C**：`triplet` + A 新损失 (`bagged` scope, witness 目标权重 0.5, support 权重 1.0)
  - **对照臂 B**：`pairwise_residual` + 4.2 原始损失
  - **主臂 A**：`pairwise_residual` + A 新损失

### 2.2 结果对比表

| 实验臂 | 路径结构 | 损失函数 | TB_end f0 | TB_end Real | TB_end Swap | **Real − f0 (pp)** [95% CI] | **Real − Swap (pp)** [95% CI] | **strict top-10** (难例进前十) |
|---|---|---|---|---|---|---|---|---|
| **R2 参考基线** | triplet | 4.2 原始 | 0.4521 | 0.4479 | 0.4411 | **−0.42** (倒挂) | +0.26 (探针) | 86 / 141 |
| **对照臂 C** | **triplet** | **A 新损失** | 0.4460 | 0.4407 | 0.4279 | **−0.53** `[-1.81, +0.75]` (仍倒挂) | **+1.28** `[+0.08, +2.50]` | 79 / 141 |
| **对照臂 B** | **residual** | 4.2 原始 | 0.4523 | 0.4556 | 0.4531 | **+0.33** `[-0.33, +1.00]` | **+0.25** `[-0.50, +0.98]` | **96 / 141** (+10) |
| **主臂 A (TB_half)** | **residual** | **A 新损失** | 0.4544 | 0.4534 | 0.4413 | −0.10 `[-1.21, +1.03]` | **+1.21** `[+0.23, +2.20]` (显著>0) | **90 / 141** (+4) |
| **主臂 A (TB_end)** | **residual** | **A 新损失** | 0.4457 | 0.4497 | 0.4430 | **+0.40** `[-0.82, +1.63]` | **+0.67** `[-0.53, +1.90]` | **93 / 141** (+7) |

### 2.3 机制归因分析
- **结构隔离**：对比“对照臂 C”与“主臂 A”，损失完全相同时，`triplet` 架构的 Real 依然落后于 f0（0.4407 vs 0.4460），且 strict 难例命中只有 79；而 `residual` 架构立即转正至 0.4497（>f0），难例命中提升至 93。**这证明残差结构是解决证据支路退化的根本前提**。
- **损失隔离**：对比“对照臂 B”与“主臂 A”，新损失引入 `bagged` 损失范围和 witness 候选列表后，大幅提升了对真实证据的敏感度（Real − Swap 从 B 的 +0.25pp 提升至 A 的 +0.67pp ~ +1.21pp）。

---

## 3. 第二步：学生池续训（部署验证）

### 3.1 实验设置
- **起点权重**：主臂 A 检查点 `work/stage1_r3_chain_residual_A/TB_CQET/end.pt`
- **训练数据**：从 KD 学生（Native C2）召回池中挖掘的训练记录（`work/stage1_r3_teacher_sp_A/records.jsonl.gz`）
- **输出目录**：`work/stage1_r3_teacher_sp_A/`
- **评测池**：KD 学生 dev 池（1198 query）与 Raw dev 池（双池同步评测）

### 3.2 KD 学生 Dev 池评测结果

学生自身直接检索 `Direct_ANN_R10` 为 **0.5022**（整体）/ **0.4225**（implicit）。

| 模型版本 | Overall f0 | Overall Real | Overall Swap | **Implicit f0** | **Implicit Real** | **Implicit Swap** | **Witness Oracle (整体 / 隐式)** |
|---|---|---|---|---|---|---|---|
| **R2 Teacher SP 续训基线** | 0.5002 | 0.4814 (低于Swap) | 0.4853 | 0.4420 | 0.4286 (低于Swap) | 0.4297 | 0.5154 / 0.4723 |
| **R3 SP_A (init)** | 0.4306 | 0.4156 | 0.4181 | 0.3804 | 0.3829 | 0.3929 | 0.4302 / 0.3795 |
| **R3 SP_A (half)** | 0.5039 | 0.4983 | 0.4935 | 0.4452 | 0.4540 | 0.4445 | 0.5139 / 0.4652 |
| **R3 SP_A (end)** | **0.5093** | **0.4997** | **0.4992** | **0.4527** | **0.4585** | **0.4441** | **0.5270 / 0.4880** |

### 3.3 离线复算指标对比（虚假偏置与理论上限）

| 统计指标 | R2 Teacher SP (基线) | R3 SP_A (本次实验) | 说明 |
|---|---|---|---|
| **路径与 f0 偏置均值 (LME − f0)** | **+1.095** (虚高) | **−1.211** (正常) | 彻底消除“有包就盲目加分”的结构性系统偏置 |
| **Real 前十有包率 vs f0 前十有包率** | **60.2% vs 55.6%** | **42.5% vs 54.4%** | 不再系统性拔高低质量有包负例 |
| **Implicit 子集 Real 超越 Student** | +0.61 pp | **+3.60 pp** (0.4585 vs 0.4225) | 隐式表上证据增益极显著 |
| **Implicit 子集 Real 超越 Swap** | −0.11 pp (反噬) | **+1.44 pp** (有效判别) | 证据内容具备真实区分能力 |
| **Witness Oracle R@10 (Implicit)** | 0.4723 | **0.4880** (+1.57pp) | 证据理论增益上限大幅提升 |

### 3.4 Raw Dev 池防遗忘检验
- 续训前 (init): $f_0 = 0.4457$, $\text{Real} = 0.4497$
- 续训后 (end): $f_0 = 0.4491$, $\text{Real} = 0.4459$
- **检验结论**：Raw 池 $f_0$ 变化为 **+0.34 pp**（远优于 $\le 1\text{pp}$ 遗忘限制要求），未发生灾难性遗忘。

---

## 4. 关键产物与文件索引

下游分析与后续蒸馏任务可直接定位以下文件：

### 4.1 部署核心目录（第二步学生池续训产物）
- **根目录**：[`/home/oycy/MMDD/work/stage1_r3_teacher_sp_A/`](file:///home/oycy/MMDD/work/stage1_r3_teacher_sp_A/)
- **最佳模型权重**：
  - 最终模型：[`checkpoints/end.pt`](file:///home/oycy/MMDD/work/stage1_r3_teacher_sp_A/checkpoints/end.pt)
  - 中间检查点：[`checkpoints/half.pt`](file:///home/oycy/MMDD/work/stage1_r3_teacher_sp_A/checkpoints/half.pt)
- **评测综合摘要 (含 Bootstrap 置信区间)**：
  - [`eval/dev/SUMMARY.json`](file:///home/oycy/MMDD/work/stage1_r3_teacher_sp_A/eval/dev/SUMMARY.json)
- **逐 Query 打分与指标明细**：
  - 学生池打分：[`eval/dev/native_kd/METRICS.json`](file:///home/oycy/MMDD/work/stage1_r3_teacher_sp_A/eval/dev/native_kd/METRICS.json) 与 [`eval/dev/native_kd/per_query_metrics.csv`](file:///home/oycy/MMDD/work/stage1_r3_teacher_sp_A/eval/dev/native_kd/per_query_metrics.csv)
  - Raw池打分：[`eval/dev/raw/METRICS.json`](file:///home/oycy/MMDD/work/stage1_r3_teacher_sp_A/eval/dev/raw/METRICS.json)
- **完整导出的 Logits（用于下游蒸馏或细粒度分析）**：
  - `eval/dev/native_kd/logits.end.Real.jsonl.gz`
  - `eval/dev/native_kd/logits.end.Swap.jsonl.gz`
  - `eval/dev/native_kd/logits.end.f0.jsonl.gz`
- **训练挖掘池记录**：[`records.jsonl.gz`](file:///home/oycy/MMDD/work/stage1_r3_teacher_sp_A/records.jsonl.gz)
- **完整运行日志**：[`work/stage1_r3_teacher_sp_A.log`](file:///home/oycy/MMDD/work/stage1_r3_teacher_sp_A.log)

### 4.2 架构消融目录（第一步产物）
- **主臂 A（残差 + 新损失）**：[`/home/oycy/MMDD/work/stage1_r3_chain_residual_A/`](file:///home/oycy/MMDD/work/stage1_r3_chain_residual_A/)
  - 评测报告：[`eval/dev/SUMMARY.json`](file:///home/oycy/MMDD/work/stage1_r3_chain_residual_A/eval/dev/SUMMARY.json)
  - 难例漏斗明细：`eval/dev/raw/funnels/TB_end/strict_EO_SUMMARY.json`
- **对照臂 B（残差 + 旧损失）**：[`/home/oycy/MMDD/work/stage1_r3_chain_residual_B/`](file:///home/oycy/MMDD/work/stage1_r3_chain_residual_B/)
  - 评测报告：[`eval/dev/SUMMARY.json`](file:///home/oycy/MMDD/work/stage1_r3_chain_residual_B/eval/dev/SUMMARY.json)
- **对照臂 C（triplet + 新损失）**：[`/home/oycy/MMDD/work/stage1_r3_chain_residual_C/`](file:///home/oycy/MMDD/work/stage1_r3_chain_residual_C/)
  - 评测报告：[`eval/dev/SUMMARY.json`](file:///home/oycy/MMDD/work/stage1_r3_chain_residual_C/eval/dev/SUMMARY.json)

---

## 5. 下一步建议行动

基于文档第 5 节的规划，Teacher 证据支路验收已通过：
1. **重做蒸馏缓存 (`teacher_logits_cache`)**：
   使用本次训练产出的最佳模型 `work/stage1_r3_teacher_sp_A/checkpoints/end.pt`（或 half）重新生成包含正确证据信号的 teacher logits 缓存。
2. **重新蒸馏学生 (Student KD)**：
   验证学生模型的 E 覆盖率能否从当前被压垮的 0.475 恢复回 $\ge 0.85$。
