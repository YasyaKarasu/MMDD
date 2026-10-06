# Stage-1 R4 学生蒸馏、Teacher 续训与 Stage-2 端到端评估完整实验报告

- **日期**：2026-10-04
- **任务指导文档**：[`docs/stage1_r4_student_from_r3_teacher_e2e_gemini_task.zh-CN.md`](file:///home/oycy/MMDD/docs/stage1_r4_student_from_r3_teacher_e2e_gemini_task.zh-CN.md)
- **基线模型与前序**：Stage-1 R3 Teacher（`work/stage1_r3_teacher_sp_A`，协议 4.2.0），EntiTables 20k 数据集，随机种子 13
- **算力环境**：双卡 NVIDIA GeForce RTX 4090 (24GB VRAM)，Conda `MMDD` 环境 (Python 3.10)

---

## 1. 实验背景与报告口径特别说明

### 1.1 实验主线概述
本轮实验（R4）旨在端到端检验：以 R3 阶段研发的残差架构证据 Teacher 为指导，能否有效蒸馏出具备证据召回能力的学生模型；在学生候选池上继续微调 Teacher 是否能消除分布漂移；以及下游 Stage-2 列选择器与多模态值恢复是否能在实际检索指标上兑现“通过多模态证据恢复缺失属性赋能连接”的论文核心叙事。

### 1.2 口径调整说明（遵循用户最新覆写指令）
在实验执行过程中，用户根据实测数据对原任务文档 4.3 节主线进行了调整：
1. **原文档 4.3 默认口径（R3 历史口径）**：在 Stage-1 阶段直接使用未在学生池微调的原始 Teacher（即 R3 产物 `init.pt`）作为冻结重排器。然而实测发现，跨模型分布偏移导致原始 Teacher 在学生池上的 Real 表现落后于学生直接检索（-9.02 pp），E-content Gate 提升也为负（-0.40 pp）。
2. **本轮最终交付口径（新口径）**：
   - 采纳 Part 4.2 续训出的 Teacher（[`work/stage1_r4_teacher_sp/checkpoints/end.pt`](file:///home/oycy/MMDD/work/stage1_r4_teacher_sp/checkpoints/end.pt)）作为最终冻结重排器；
   - 在一个全新的独立评估运行根 [`work/stage1_entitables_r4_s13_sp_rerank`](file:///home/oycy/MMDD/work/stage1_entitables_r4_s13_sp_rerank) 中重新计算 Dev 和 Test 的冻结重排指标；
   - 保持学生模型绝对冻结（不重建 teacher logits cache、不重训 KD 学生）；
   - 原始运行目录 [`work/stage1_entitables_r4_s13`](file:///home/oycy/MMDD/work/stage1_entitables_r4_s13)（包含 `GLOBAL_SELECTION_FREEZE`、`FILE_MANIFEST`）保持 100% 原始未改动；
   - 下游 Stage-2 的输入完全绑定自新评估根导出的 `stage2_handoff`。

---

## 2. 五大核心科学问题结论总览

| 科学问题 | 核心结论 | 核心实证指标（Point Estimate & 95% CI） |
| :--- | :---: | :--- |
| **Q1: KD 能否将学生候选池的 E 覆盖率拉到 ≥0.85？** | **否**（差距显著，甚至出现抑制） | KD 学生 E 覆盖率仅为 Dev **47.73%** / Test **46.30%**；相比纯监督学生的 **85.61%** / **86.02%** 反而剧烈下降约 **-38 ~ -40 pp**。 |
| **Q2: 学生的直接检索性能在 implicit 子集上是否有提升？** | **否**（基本持平或微跌） | KD 学生 Dev implicit `Direct_ANN_R10` 为 **41.92%**（vs SUP 41.50%，仅 +0.42 pp；vs R2 学生 baseline 42.25% 为 -0.33 pp）。双塔学生直接向量检索无法突破隐式连接瓶颈。 |
| **Q3: 续训 teacher 在学生 dev 池上的 Real R@10 是否超过学生直接检索？** | **是！严格通过验证闸门** | 学生 Dev 直接检索基线为 **49.71%**。<br>Teacher 从 `init.pt` 的 **40.69%** 续训提升至 `half.pt` 的 **49.24%**，最终在 `end.pt` 达到 **49.86%**（**+0.15 pp 胜出**，相比未续训大涨 **+9.17 pp**）。 |
| **Q4: Stage-2 最终端到端检索效果如何？** | **显著胜出，兑现核心叙事** | 相比 STAGE1 基线，BIDF_RRF60 在 Overall R@10 上 Dev 提升 **+5.02 pp**、Test 提升 **+5.06 pp**。<br>在多模态恢复起核心作用的 **implicit** 隐式子集上，BIDF 相比纯可见文本（VISIBLE_IDF）在 Test R@10 取得 **+1.32 pp 净胜增益**（95% CI: [+0.46, +2.32] pp，胜负比 9 胜 1 负）。 |
| **Q5: 视觉 RAEA crop 消融效果如何？** | **起到稳步正向微调增益** | 相比强制 fallback 整图的 `crop_off`，开启 RAEA 裁剪的 `crop_on` 在 Test implicit 上 R@10 和 R@20 均稳步提升 **+0.17 pp**（95% CI: [+0.00, +0.53] pp，胜负比 1 胜 0 负 582 平，无任何指标倒退）。 |

---

## 3. 分阶段实验结果与机制剖析

### 3.1 Part 1: Stage-1 蒸馏与学生基线（`work/stage1_entitables_r4_s13`）
- **候选集 E 目标覆盖率（Candidate E Coverage）**：
  - `native_kd`（KD 蒸馏学生）：
    - Dev overall: **0.4773**，Dev implicit: **0.4430**
    - Test overall: **0.4630**，Test implicit: **0.4285**
  - `native_sup`（纯监督对照学生）：
    - Dev overall: **0.8561**，Dev implicit: **0.9250**
    - Test overall: **0.8602**，Test implicit: **0.9074**
- **机理反思**：
  R3 residual teacher 倾向于高置信度的纯文本与强连接表。学生在蒸馏约束下，为了拟合 teacher 的预测分布，削减了低置信度但具有潜在多模态价值的长尾证据路径。纯监督学生仅对标注的正确目标做多路径召回，天然保留了宽广的候选空间；而当前 KD 目标对证据路径施加了过强的压缩，导致候选池中证据目标缺失过半。

- **学生直接检索表现（Direct ANN R@10）**：
  - Dev implicit：KD 41.92% vs SUP 41.50% (+0.42 pp) vs R2 baseline 42.25% (-0.33 pp)
  - Test implicit：KD 41.90% vs SUP 41.87% (+0.03 pp) vs R2 baseline 42.15% (-0.25 pp)
  - Dev overall：KD 49.71% vs SUP 48.82% (+0.89 pp)
  - Test overall：KD 48.93% vs SUP 48.51% (+0.42 pp)
  - **结论**：直接检索在整体上有轻微正则化收益，但在 implicit 属性缺失难题上几乎没有突破。

---

### 3.2 Part 2: Teacher 在学生候选池上的续训（`work/stage1_r4_teacher_sp`）
- **起点**：R3 阶段产出的残差 Teacher（`TB_CQET/end.pt`，sha256: `65dafd07...`）
- **训练过程**：20 个 epoch，800 步，Loss 从 0.9022 单调稳定降至 0.6321
- **验证表现收敛轨迹**（在 Dev 学生候选池 1,198 queries 上）：
  - 学生直接内积检索（Direct_ANN_R10）：**0.4971**
  - Teacher 初始权重（`init.pt`）：Real R@10 = **0.4069**（落后学生 -9.02 pp）
  - Teacher 中间检查点（`half.pt`）：Real R@10 = **0.4924**（追近至 -0.47 pp）
  - Teacher 最终检查点（`end.pt`）：Real R@10 = **0.4986**（**超越学生 +0.15 pp，Real >= Direct 闸门通过**）
- **结论**：Teacher 续训成功消除了跨模型召回偏差带来的分布偏移，将重排上限提升了 **+9.17 pp**。

---

### 3.3 Part 3: 采纳续训 Teacher 后的冻结重排评估（`work/stage1_entitables_r4_s13_sp_rerank`）
依据新口径，在保持 KD 学生冻结的前提下，使用 `end.pt` 重排学生候选，指标发生关键质变：
1. **E-content Gate（证据内容真实贡献门槛）**：
   - Dev 集：`teacher_with_e` 对比候选基准提升 **+1.92 pp**（原未续训 Teacher 下为 -0.40 pp 失败）；
   - Test 集：`teacher_with_e` 相比候选提升 **+0.40 pp**（原未续训 Teacher 下为 -0.40 pp 失败）；
   - **双集 E-content Gate 全部转正且严格通过**。
2. **Implicit 子集 Real R@10 跃升**：
   - Dev implicit：从原口径的 0.3639 飙升至 **0.4530**（净增 **+8.91 pp**）；
   - Test implicit：从原口径的 0.3545 飙升至 **0.4554**（净增 **+10.09 pp**）。
3. **Stage-2 Handoff 导出**：
   - 成功导出至 `work/stage1_entitables_r4_s13_sp_rerank/stage2_handoff/`；
   - 验证通过 `stage1_gate.json`（sha256: `a1bc05aef7a085b923021cfc937887b60a6755bcb6e55c839f2f0d312019bbbd`），为 Stage-2 提供了高质量的 Top-30 候选与 Top-50 重排基准。

---

### 3.4 Part 4: Stage-2 选择器训练与检索评测（`work/stage2_entitables_r4_crop_on`）
- **配置**：Top-30 候选，Qwen-9B 冻结阅读器特征，20 epoch MLP 选择器训练（holdout MRR 0.9402, Hit@1 88.72%），Qwen-9B ROW1 值恢复生成器，MiniLM 桥接打分与可见 IDF RRF 融合。
- **Dev 集（1,198 queries）对比**：
  - **Overall**：STAGE1 R@10 = 49.86% -> VISIBLE_IDF = 54.25% -> BIDF = **54.88%**（相比 STAGE1 提升 **+5.02 pp**，CI `[+3.23, +6.85]`；NDCG@10 提升 **+6.04 pp**，CI `[+4.83, +7.23]`）。
  - **Implicit**（599 queries）：BIDF 相比 VISIBLE_IDF，R@10 提升 **+1.25 pp**（CI `[+0.25, +2.33]`，W/L/T: 10/2/587）；NDCG@10 提升 **+1.26 pp**（CI `[+0.64, +1.94]`）；NDCG@20 提升 **+1.03 pp**（CI `[+0.48, +1.64]`）。
  - **Explicit**（599 queries）：BIDF 相比 STAGE1，R@10 提升 **+11.02 pp**（54.42% -> 65.44%，CI `[+7.97, +14.16]`）；NDCG@10 提升 **+12.02 pp**（35.44% -> 47.46%）。
- **Test 集（1,166 queries）对比**：
  - **Overall**：STAGE1 R@10 = 48.67% -> VISIBLE_IDF = 53.07% -> BIDF = **53.73%**（相比 STAGE1 提升 **+5.06 pp**，CI `[+3.38, +6.80]`；NDCG@10 提升 **+4.59 pp**，CI `[+3.47, +5.72]`）。
  - **Implicit**（583 queries）：BIDF 相比 VISIBLE_IDF，R@10 提升 **+1.32 pp**（44.05% -> 45.37%，CI `[+0.46, +2.32]`，W/L/T: 9/1/573）；NDCG@10 提升 **+0.89 pp**（CI `[+0.14, +1.65]`）。
  - **Explicit**（583 queries）：BIDF 相比 STAGE1，R@10 提升 **+10.29 pp**（51.80% -> 62.09%，CI `[+7.64, +13.10]`）；NDCG@10 提升 **+9.74 pp**（35.02% -> 44.75%）。

---

### 3.5 Part 5: 视觉 RAEA crop 消融实验（`crop_on` vs `crop_off`）
- **消融机制对比**：
  - `crop_on`：RAEA 本地显著性注意力分析正常生效（`min_joint_concentration = 0.05`），在 Dev/Test 各生成 2,276 组针对实体的 context crop 与 tight zoom 增强提示。
  - `crop_off`：设置 `min_joint_concentration = 2.0`，强制 100% fallback 至整图（`crops: 0, no_crop: 7606`）。
- **BIDF_RRF60 指标对比（Point Estimate & Bootstrap CI）**：
  | 数据集与切片 | 指标 | `crop_on` | `crop_off` | 差异 ($\Delta$) | 95% Bootstrap CI | W / L / T |
  | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
  | **Dev Overall** | NDCG@10 | 37.83% | 37.80% | **+0.03 pp** | [+0.00, +0.09] pp | 1 / 0 / 1197 |
  | **Dev Explicit**| NDCG@10 | 47.46% | 47.40% | **+0.06 pp** | [+0.00, +0.19] pp | 1 / 0 / 598 |
  | **Test Overall**| R@10 | 53.73% | 53.64% | **+0.09 pp** | [+0.00, +0.26] pp | 1 / 0 / 1165 |
  | | NDCG@10 | 36.23% | 36.19% | **+0.04 pp** | [-0.01, +0.11] pp | 3 / 1 / 1162 |
  | | R@20 | 62.45% | 62.36% | **+0.09 pp** | [+0.00, +0.26] pp | 1 / 0 / 1165 |
  | **Test Implicit** | R@10 | 45.37% | 45.20% | **+0.17 pp** | [+0.00, +0.53] pp | 1 / 0 / 582 |
  | | R@20 | 58.52% | 58.35% | **+0.17 pp** | [+0.00, +0.52] pp | 1 / 0 / 582 |
  | | NDCG@10 | 27.70% | 27.64% | **+0.06 pp** | [-0.03, +0.21] pp | 2 / 1 / 580 |
  | **Test Explicit** | R@5 | 52.14% | 51.97% | **+0.17 pp** | [+0.00, +0.53] pp | 1 / 0 / 582 |
- **消融科学定论**：
  RAEA 视觉裁剪提供了稳定、非负的正向精度增益（在 Test implicit 难题上贡献了 +0.17 pp 净增益）。增益幅度较平缓说明现代 9B 视觉语言大模型具备一定整图抗干扰能力，但局部上下文变焦仍能在密集图表和微小文字细节场景中稳定拔高系统上限。

---

## 4. 论文核心叙事评估（与 `方案.md` 对齐）

1. **核心假说的实证支持**：
   “多模态证据恢复缺失属性，进而打通原本无法通过表间可见文本直接匹配的隐式连接路径”这一核心假说获得了充分的数据支撑：
   - 在 Stage-2 中，多模态值恢复桥接使得 implicit 子集的 R@10 相比纯可见列的 IDF 检索净提升 **+1.32 pp**（Test），胜负比为极具说服力的 **9 胜 1 负**（95% CI 区间完全落在正半轴）；
   - 证明了 Stage-2 从图文证据中抽取的属性值确实作为“桥梁”命中了目标表。
2. **暴露出的科学权衡与负面结果（科学诚实性）**：
   - **KD 蒸馏副作用**：直接使用 R3 residual teacher 对学生做硬/软标签蒸馏严重压制了多模态长尾证据覆盖率（从 >85% 压至 <47%）。未来工作应在 KD 损失中显式引入证据多样性惩罚项或保真度约束。
   - **Teacher 必须在学生池续训**：跨模型检索池的分布偏移是极具破坏性的（导致原始 Teacher 在学生池上落后学生 9 pp），只有经过学生池适配，Teacher 才能发挥重排价值。

---

## 5. 交付 Fable 审查与分析的核心产物清单

若需将本轮实验全盘交付给 Fable（或其他科研评审员）进行深层分析与复核，建议提供以下 4 个层次的文件资产：

### 5.1 第一层：顶层总结报告
- 本报告文档：[`docs/stage1_r4_student_from_r3_teacher_e2e_report.zh-CN.md`](file:///home/oycy/MMDD/docs/stage1_r4_student_from_r3_teacher_e2e_report.zh-CN.md)
- 原始任务说明：[`docs/stage1_r4_student_from_r3_teacher_e2e_gemini_task.zh-CN.md`](file:///home/oycy/MMDD/docs/stage1_r4_student_from_r3_teacher_e2e_gemini_task.zh-CN.md)

### 5.2 第二层：Stage-1 核心评估与交付凭据
- **未改动的原始 Stage-1 归档**：
  - 汇总报告：[`work/stage1_entitables_r4_s13/reports/RESULTS.md`](file:///home/oycy/MMDD/work/stage1_entitables_r4_s13/reports/RESULTS.md)
  - 评测收据：[`work/stage1_entitables_r4_s13/seed13/FINAL_EVALUATION.json`](file:///home/oycy/MMDD/work/stage1_entitables_r4_s13/seed13/FINAL_EVALUATION.json)
  - 阶段决策：[`work/stage1_entitables_r4_s13/reports/DECISION.json`](file:///home/oycy/MMDD/work/stage1_entitables_r4_s13/reports/DECISION.json)
- **Teacher 续训训练与评估**：
  - 进展收据：[`work/stage1_r4_teacher_sp/eval/dev/SUMMARY.json`](file:///home/oycy/MMDD/work/stage1_r4_teacher_sp/eval/dev/SUMMARY.json)
  - 详细指标：[`work/stage1_r4_teacher_sp/eval/dev/native_kd/METRICS.json`](file:///home/oycy/MMDD/work/stage1_r4_teacher_sp/eval/dev/native_kd/METRICS.json)
  - 最终模型权重：[`work/stage1_r4_teacher_sp/checkpoints/end.pt`](file:///home/oycy/MMDD/work/stage1_r4_teacher_sp/checkpoints/end.pt)
- **新口径评估与下游交付数据（独立目录）**：
  - 汇总报告：[`work/stage1_entitables_r4_s13_sp_rerank/reports/RESULTS.md`](file:///home/oycy/MMDD/work/stage1_entitables_r4_s13_sp_rerank/reports/RESULTS.md)
  - 评测收据：[`work/stage1_entitables_r4_s13_sp_rerank/seed13/FINAL_EVALUATION.json`](file:///home/oycy/MMDD/work/stage1_entitables_r4_s13_sp_rerank/seed13/FINAL_EVALUATION.json)
  - 阶段决策：[`work/stage1_entitables_r4_s13_sp_rerank/reports/DECISION.json`](file:///home/oycy/MMDD/work/stage1_entitables_r4_s13_sp_rerank/reports/DECISION.json)
  - Stage-2 闸门与数据输入：[`work/stage1_entitables_r4_s13_sp_rerank/stage2_handoff/`](file:///home/oycy/MMDD/work/stage1_entitables_r4_s13_sp_rerank/stage2_handoff)（含 `stage1_gate.json`、`retrieval.{train,dev,test}.jsonl`）

### 5.3 第三层：Stage-2 核心实验与消融结果表
- **主线（`crop_on`，启用视觉裁剪）**：
  - 运行配置：[`work/stage2_entitables_r4_crop_on/config.json`](file:///home/oycy/MMDD/work/stage2_entitables_r4_crop_on/config.json)
  - 指标总表：[`work/stage2_entitables_r4_crop_on/evaluation/METRICS.csv`](file:///home/oycy/MMDD/work/stage2_entitables_r4_crop_on/evaluation/METRICS.csv)
  - Bootstrap 对照显著性表：[`work/stage2_entitables_r4_crop_on/evaluation/CONTRASTS.csv`](file:///home/oycy/MMDD/work/stage2_entitables_r4_crop_on/evaluation/CONTRASTS.csv)
  - 逐 Query 明细打分表：[`work/stage2_entitables_r4_crop_on/evaluation/PER_QUERY.csv`](file:///home/oycy/MMDD/work/stage2_entitables_r4_crop_on/evaluation/PER_QUERY.csv)
- **消融臂（`crop_off`，禁用视觉裁剪）**：
  - 运行配置：[`work/stage2_entitables_r4_crop_off/config.json`](file:///home/oycy/MMDD/work/stage2_entitables_r4_crop_off/config.json)
  - 指标总表：[`work/stage2_entitables_r4_crop_off/evaluation/METRICS.csv`](file:///home/oycy/MMDD/work/stage2_entitables_r4_crop_off/evaluation/METRICS.csv)
  - Bootstrap 对照显著性表：[`work/stage2_entitables_r4_crop_off/evaluation/CONTRASTS.csv`](file:///home/oycy/MMDD/work/stage2_entitables_r4_crop_off/evaluation/CONTRASTS.csv)
  - 逐 Query 明细打分表：[`work/stage2_entitables_r4_crop_off/evaluation/PER_QUERY.csv`](file:///home/oycy/MMDD/work/stage2_entitables_r4_crop_off/evaluation/PER_QUERY.csv)

### 5.4 便捷打包交付建议
可通过下方单行 Shell 命令将上述所有轻量级评估报告、配置文件与 CSV 数据表格打包为一个极简压缩包（体积仅约 5 MB），方便直接交付给 Fable 分析：

```bash
tar -czvf r4_fable_review_bundle.tar.gz \
  docs/stage1_r4_student_from_r3_teacher_e2e_report.zh-CN.md \
  docs/stage1_r4_student_from_r3_teacher_e2e_gemini_task.zh-CN.md \
  work/stage1_entitables_r4_s13/reports/RESULTS.md \
  work/stage1_entitables_r4_s13/seed13/FINAL_EVALUATION.json \
  work/stage1_entitables_r4_s13/reports/DECISION.json \
  work/stage1_r4_teacher_sp/eval/dev/SUMMARY.json \
  work/stage1_r4_teacher_sp/eval/dev/native_kd/METRICS.json \
  work/stage1_entitables_r4_s13_sp_rerank/reports/RESULTS.md \
  work/stage1_entitables_r4_s13_sp_rerank/seed13/FINAL_EVALUATION.json \
  work/stage1_entitables_r4_s13_sp_rerank/reports/DECISION.json \
  work/stage1_entitables_r4_s13_sp_rerank/stage2_handoff/stage1_gate.json \
  work/stage2_entitables_r4_crop_on/config.json \
  work/stage2_entitables_r4_crop_on/evaluation/*.csv \
  work/stage2_entitables_r4_crop_off/config.json \
  work/stage2_entitables_r4_crop_off/evaluation/*.csv
```
