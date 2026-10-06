# R5 实验数据与方案论证独立评估

- 日期：2026-10-05
- 评估对象：`work/stage1_entitables_r5_s13_sup_rerank`、`work/stage2_entitables_r5_sup_crop_on`，对照 `work/stage2_entitables_r4_crop_on`
- 方法：所有数字由 METRICS.csv / CONTRASTS.csv / PER_QUERY.csv / recovery/*.json / stage2_handoff/retrieval.*.jsonl / 数据集 evidence_recoveries 重新计算；跨轮配对 bootstrap 沿用 `src/mmdd_stage2/evaluate.py` 的 source-group 重采样（10000 次，seed 20260925）

## 结论先行

1. 数据本身可信，但"R5 相对 R4 明显提升"只在 implicit 子集成立：dev 显著，test R@10 置信区间触零、NDCG@10 显著；overall test R@10 不显著。Stage-1 gate FAIL 与 Stage-2 提升不矛盾：gate 评的是 KD 臂、量的是 Stage-1 排序与覆盖，而提升来自 SUP 臂把证据挂到了两倍的候选上。断掉的链条是"Teacher/KD 让 Stage-1 检索更好"，存活的链条是"证据挂载 → 值恢复 → implicit join"。
2. 方案故事线中 4 环被数据支持、4 环被数据否定或削弱、4 环没有实验覆盖。无法自圆其说的环节：KD 蒸馏（部署臂没有用 KD 学生）、证据路径提升召回（两臂的 candidate gain 都为负）、Stage-1 路径聚合残差提升排序（Real−f0 ≤ 0，test 上 α=0 更好）、"多模态"里的图像份额（只占 bridge 的 7–10%，无消融）。
3. 路线：封版 R5 为主结果并立即写作；并行补 3 个 CPU 级实验（合计约 1 个工作日）和最多 1 个后台 GPU run（约 8.5 h）。不做 200K、跨数据集、第二 seed。

## 一、数据与报告的可信度

### 1.1 Stage-1 三个 gate 到底评了什么

`src/mmdd_stage1/pipeline.py:2036-2092` 的 `_reports`：candidate gain = KD 臂 `C150_target_coverage − MatchedDirectC_target_coverage`；KD gate = KD−SUP 同 Teacher 对比与 KD 覆盖差；只有 E-content gate 用到 Teacher。三个 gate 都只读 `native_kd`。

| gate | R4 sp_rerank（KD handoff） | R5 sup_rerank（SUP handoff） | 备注 |
|---|---|---|---|
| candidate gain pp（dev/test） | −9.3907 / −10.3202 | −9.3907 / −10.3202 | 逐位相同：学生 checkpoint 都是 r4_s13 的冻结件 |
| KD coverage pp | −7.4638 / −7.8330 | −7.4638 / −7.8330 | 同上 |
| KD pp（CI） | +1.57 [−0.66, 3.73] / +0.71 [−1.30, 2.68] | +1.04 [−1.00, 3.08] / +0.80 [−1.18, 2.74] | CI 全部跨 0 |
| E-content implicit pp（CI） | +1.92 [0.00, 3.91] / +0.40 [−1.28, 2.11] | +0.53 [−1.29, 2.34] / +0.94 [−0.60, 2.51] | PASS 只靠点估计 ≥ 0，CI 跨 0 |
| Teacher−QT pp（CI） | +6.51 [4.35, 8.72] / +6.06 [4.03, 8.05] | +6.09 [3.95, 8.18] / +6.00 [4.02, 7.96] | 唯一扎实的 PASS |

交接臂是 SUP（`stage2_handoff/stage1_gate.json` arm=NATIVE_C2_SUP，`positive_evidence_path_coverage@10` 0.432 vs R4 KD 的 0.226）。SUP 臂自己的 candidate gain 也是负的：dev 0.8565−0.8853 = −2.88pp，test 0.8539−0.8932 = −3.93pp（`seed13/eval/*/SUMMARY.json` native_sup.candidate）。

Teacher 在 SUP 臂上的证据残差：α=1 时 Real 48.40 / 47.81 低于 f0 49.35 / 49.40（`alpha_scan.txt`）；Real−Swap implicit 点估计 +1.92 / +2.63，无 CI。续训后 Teacher end.Real 对学生 Direct 是 −0.42pp [−3.03, 2.11]（`work/stage1_r5_teacher_sp_sup/eval/dev/SUMMARY.json`），没有越过学生。

α 选择：dev implicit 在 0.5 最高（45.55），但 test 上 α=0 的 overall 49.40、implicit 45.97 都高于 α=0.5 的 48.84 / 45.54；α=0.25 在 test 两个口径都最好。选 α 的规则事先写在任务书里，流程合规，但结论是 Stage-1 层面的证据残差在 test 上没有正收益。

### 1.2 Stage-2 端到端指标与显著性

BIDF_RRF60（方法）R@10 / NDCG@10，来自两轮 `evaluation/METRICS.csv`：

| split / population | R4 | R5 | STAGE1（R4 → R5） |
|---|---|---|---|
| dev overall | 54.88 / 37.83 | 57.20 / 39.08 | 49.86 → 49.99 |
| dev implicit | 44.31 / 28.20 | 49.29 / 30.51 | 45.30 → 45.55 |
| dev explicit | 65.44 / 47.46 | 65.11 / 47.64 | 54.42 → 54.42 |
| test overall | 53.73 / 36.23 | 54.89 / 37.80 | 48.67 → 48.84 |
| test implicit | 45.37 / 27.70 | 48.20 / 30.70 | 45.54 → 45.54 |
| test explicit | 62.09 / 44.75 | 61.58 / 44.91 | 51.80 → 52.14 |

跨轮配对 bootstrap（本次复算，R5 − R4，同 query 配对）：

| split / population | BIDF R@10 | BIDF NDCG@10 | STAGE1 R@10 |
|---|---|---|---|
| dev overall | +2.32 [+0.13, +4.54] | +1.25 [−0.09, +2.62] | +0.13 [−1.93, +2.22] |
| dev implicit | +4.98 [+1.83, +8.11] | +2.31 [+0.52, +4.10] | +0.25 [−2.65, +3.24] |
| dev explicit | −0.33 [−3.40, +2.74] | +0.19 [−1.81, +2.24] | 0.00 |
| test overall | +1.16 [−0.91, +3.17] | +1.58 [+0.33, +2.84] | +0.17 [−1.64, +2.01] |
| test implicit | +2.83 [−0.03, +5.67] | +3.00 [+1.27, +4.78] | 0.00 |
| test explicit | −0.51 [−3.49, +2.41] | +0.15 [−1.68, +1.93] | +0.34 [−2.19, +2.88] |

轮内对比（`CONTRASTS.csv`）：证据净贡献 BIDF−VISIBLE_IDF 在 implicit 上 R5 dev +5.31 [3.26, 7.49]（W/L 41/11）、test +3.60 [1.92, 5.36]（29/4）；R4 只有 +1.25 [0.25, 2.33]（10/2）、+1.32 [0.46, 2.32]（9/1）。两轮之差（本次复算）dev +4.06 [1.97, 6.29]、test +2.29 [0.48, 4.18]，显著。仅靠 bridge 的 BRIDGE−STAGE1 在 implicit 上 R5 dev +3.34 [1.68, 5.09]、test +3.32 [1.77, 5.01]；R4 两个都跨 0。

负面：explicit 上 BIDF−VISIBLE_IDF 的 NDCG@10 dev −1.18 [−1.86, −0.55]、test −1.27 [−2.03, −0.62]，R@10 −0.50 / −0.34 且 W/L 为 0/3、0/2。bridge 分层在可见列已经很强的 query 上只会添乱。

### 1.3 pool 可比性

两轮 Stage-2 `config.json` 除 `stage1_handoff` 外逐字相同；bootstrap seed 相同；selector holdout 规则相同。但两份 handoff 的 C30 只有一半重合：

| 指标（implicit，dev / test） | R4 KD handoff | R5 SUP handoff |
|---|---|---|
| C30 Jaccard（两轮之间） | 0.50 / 0.52 | 同左 |
| gold 落在 C30 的比例 | 0.669 / 0.650 | 0.689 / 0.694 |
| C30 中带证据路径的目标比例 | 0.489 / 0.493 | 0.935 / 0.936 |
| C30 中 gold 带证据的比例 | 0.518 / 0.501 | 0.981 / 0.959 |
| 每 query C30 证据路径数（其中 image） | 23.7 (4.8) / 24.8 (5.0) | 63.1 (3.5) / 65.4 (3.3) |
| recover task 中 NO_EVIDENCE 比例 | 36.9% / 33.8% | 4.0% / 3.5% |
| 恢复出 VALUE 的 slot 数 | 650 / 807 | 1725 / 1801 |
| C30 中 gold 拿到 bridge 的比例 | 0.145 / 0.165 | 0.441 / 0.470 |
| 有 bridge 的 query 数 | 120 / 128 | 363 / 363 |

gold 覆盖几乎没变，变的是"候选身上挂了多少证据"。这正是 Stage-1 gate 不量的东西。

### 1.4 对"矛盾"的正面回答

不矛盾，原因有三。第一，gate 评 KD 臂，交接的是 SUP 臂，R4/R5 的 candidate 与 KD coverage 数字逐位相同就是证据。第二，gate 量 Stage-1 的排序与 gold 覆盖，Stage-2 的提升来自证据挂载面翻倍，两者正交：两份 handoff 的 STAGE1 R@10 几乎相同，但带证据的 C30 目标从 49% 到 93%。第三，gate FAIL 是有信息量的：它否定了"证据路径提升召回"和"KD 有效"两条论断。断掉的是"Teacher/Student 蒸馏 → 更好的 Stage-1 检索"这一层；存活的是"SUP 学生的证据路径 → Stage-2 恢复 → implicit joinability"。

### 1.5 其他扣分项

- test split 是历史暴露的回归集（`reports/RESULTS.md` 自述），不是干净 holdout；全程单 seed 13。
- R4 → R5 同时改了三件事：学生臂 KD→SUP、Teacher 续训池 KD→SUP、α 1.0→0.5。STAGE1 几乎不动说明后两者对排序影响小，但对证据挂载的归因没有被隔离。
- bridge 假阳性高：implicit 有 bridge 的 query 平均带 9–10 个非 gold 的 bridge 目标，top bridge 落在 gold 的比例只有 0.21 / 0.25；explicit 上只有 0.16 / 0.03。
- selector holdout 略降：MRR 0.9402 → 0.9349，Hit@1 0.887 → 0.874（`head/history.json`），训练对数翻倍（`jobs/FUNNEL.json` pairs_with_evidence 3909 → 6559）。
- R4 报告把 crop 消融写成"稳步正向"，实际 BIDF 排序只在 dev 2/1198、test 11/1166 个 query 上有差别，是 null 结果。

## 二、方案故事线的自洽性

| # | 方案主张 | 实测 | 判定 |
|---|---|---|---|
| 1 | 统一有向 J(a→b)，Teacher 做细粒度跨对象交互 | Teacher−QT +6.09 / +6.00pp，CI 不跨 0 | 支持（但未测有向性） |
| 2 | Teacher → Student 蒸馏得到可索引的双线性 Student | KD gate FAIL；KD−SUP +1.04 / +0.80 跨 0；KD 把 E 覆盖从 0.86 压到 0.48；部署臂是 SUP | 否定；部署系统里没有蒸馏 |
| 3 | Q→E→T 作为召回机制扩充候选 | candidate gain KD −9.4 / −10.3，SUP −2.9 / −3.9；C30 里只靠证据到达的 gold 仅 12/452、9/444 | 否定 |
| 4 | Q→E→T 为目标表挂上证据，供二阶段使用 | 带证据候选 0.49 → 0.93 带来 implicit +3–5pp | 强支持，但方案里没把它写成主机制 |
| 5 | 路径聚合（LME 残差）改善 Teacher 排序 | Real−f0 overall −0.74 / −0.70；SUP 臂 α=1 比 f0 低 0.95 / 1.59；test 偏好 α=0 | 否定或至多中性 |
| 6 | Teacher 在学生池续训后作为冻结重排器 | SUP 池 end.Real −0.42pp 对学生，跨 0；但 STAGE1 比学生 Direct 高 1.2 / 2.1pp | 弱支持：价值在 f0 的交叉编码，不在证据 |
| 7 | 二阶段 verifier 做列选择 | holdout MRR 0.935、Hit@1 0.874 | 支持（内部指标，无端到端消融） |
| 8 | 行级证据定位：图像 crop | 排序只变 2/1198、11/1166 | 不支持（null） |
| 9 | 行级证据定位：文本 span | 无消融 | 未覆盖 |
| 10 | 生成桥接属性值 | 隐藏属性值精度 R5 0.763 / 0.707；33% implicit query 至少恢复 1 个正确值（R4 10–12%） | 支持（本次新算，论文里目前没有这个指标） |
| 11 | 补全后提升 joinability | implicit 净 +5.31 / +3.60；overall 净 +2.41 / +1.63；explicit NDCG −1.18 / −1.27 | 支持 implicit，explicit 为代价 |
| 12 | "多模态"证据 | image 来源的 VALUE 占 6.6% / 10.3%，带 image claim 的 bridge 占 7.2% / 9.2%；数据集里 15% implicit query 只能靠图像恢复 | 弱；无隔离消融 |
| 13 | 二阶段用 softmax(r_T)·ρ_{T,c} 打分 | 实现是 RRF(Stage-1, 分层(bridge, vis_idf)) | 方案与代码不一致，论文须按实现写 |
| 14 | 2K/20K/200K、ANN 延迟、索引体积 | 只有 20K EntiTables | 未覆盖 |

无法自圆其说的具体环节：

- 第 2 环。方案把蒸馏写成核心，实际交付链路里 Teacher 只做重排、学生是纯监督的。论文如果照方案写，审稿人对照消融表会直接发现部署臂没有 KD。
- 第 3 与第 5 环。方案把 Q→E→T 写成"召回桥梁"和"路径级 joinability"，数据说证据路径既不扩召回也不改善排序。可以成立的说法是"证据路径决定哪些候选带着什么证据进入 verifier"。
- 第 12 环。标题级的"多模态"只由不到 10% 的 bridge 支撑，且唯一的视觉消融是 null。
- 第 11 环的 explicit 代价。方法在一半 query 上显著变差，目前报告只强调 overall 和 implicit。

## 三、路线决策

决定：封版 R5 作为主结果，写作从今天开始；实验只补"便宜且直接堵审稿人攻击点"的几项，总预算 CPU 约 1 个工作日 + 1 个后台 GPU run，10 月 8 日冻结所有数字。

成本依据（R5 实测墙钟）：Teacher 续训 1.1 h，dev/test 重评 0.8 h（双卡并行），export 1.1 h，Stage-2 7.2 h（features 3.4 h、recover 3.5 h、其余 < 20 min）。完整一轮约 10.5 h。到 10.18 还有 13 天，SIGMOD 正文至少需要 8 天；任何需要重跑 Stage-1 训练或多 seed 的事都不在预算内。

按性价比排序：

**E1. 把机制指标正式落盘（CPU，约 2–3 h，必做）**
- 目的：给第 4、10、11 环提供可引用的表，并把 R5 vs R4 的跨轮配对 CI 写进论文。
- 对应 claim：证据挂载 → 正确值恢复 → implicit join 的机制链。
- 数据与脚本：两轮 `evaluation/PER_QUERY.csv`、`recovery/*/*.json`、`stage2_handoff/retrieval.*.jsonl`、数据集 `evidence_recoveries`；新建 `src/analyze_stage2_mechanism.py`，输出 CSV/Markdown。本文 1.2–1.5 节的数字就是它的预期输出。
- 成功判定：复现本文数字；失败即脚本有 bug，无科学风险。

**E2. 融合规则修 explicit 负收益（CPU，约 1 h，每个变体 score 10 min + evaluate 1 min，必做）**
- 目的：消除 explicit NDCG@10 −1.2pp 的显著倒退，同时保住 implicit 净收益。
- 对应 claim：B+IDF 在 overall 上无损提升。
- 数据与脚本：复制 run root 并软链 `catalog.sqlite`、`population/`、`recovery/`、`matching/`；在 `src/mmdd_stage2/visible.py` 与 `rerank.py` 加 `fusion.mode`：(a) bridge 值做 IDF 加权，与 vis_idf 同式；(b) Stage-1、bridge 序、visible 序三路 RRF 取代分层；(c) 仅当 bridge 分 ≥ vis_idf 时才放进第一层。dev 上选，test 上报。
- 成功判定：test 上 explicit BIDF−VISIBLE NDCG@10 的 CI 覆盖 0，且 implicit BIDF−VISIBLE R@10 ≥ +3.0pp 且 CI > 0，且 overall BIDF−STAGE1 不低于当前 +6.05pp。失败：三者不能同时满足，则保留现规则并在论文里如实写 tradeoff。

**E3. 图像证据的贡献（CPU，约 1 h，必做）**
- 目的：给"多模态"一个数字，当前是空白。
- 对应 claim：第 12 环。
- 数据与脚本：(a) 零成本子集分析：数据集 evidence_recoveries 标出 178 个只能靠图像恢复、264 个图文混合、740 个纯文本的 implicit query，用现有 PER_QUERY.csv 分别算 BIDF−VISIBLE；(b) 打分级消融：复制 recovery/*.json，剔除 `tasks[].image_count>0` 的 claim 并重算 bridge slots，再跑 score + evaluate。
- 成功判定：图像子集或消融差分在 implicit 上 > 0 且 CI > 0。失败：差分落在噪声内，则论文把图像写成"已接入但在当前数据规模下贡献有限"的诚实结论，并引用 7–10% 的占比。注意 (b) 是近似，严格消融要重跑 recover（3.5 h GPU），可选。

**E4. α=0 handoff 的端到端（GPU 后台约 8.5 h，建议今晚启动）**
- 目的：回答 Stage-1 的路径聚合重排是否对端到端有贡献；test 的 α 扫描已经偏向 0。
- 对应 claim：第 5 环与 α 的选择理由。
- 数据与脚本：`run_stage1.py export --arm SUP --residual-scale 0` 到新目录，再跑完整 Stage-2 到 `work/stage2_entitables_r5_sup_alpha0_crop_on`。
- 成功判定：R5(α=0.5) 对 α=0 的 BIDF 在 dev overall 或 implicit 上 CI > 0 → 保留 α=0.5 并写成消融。失败：持平或 α=0 更好 → 论文采用 α=0 作为更简单的系统，删掉路径聚合重排的论断。两种结果都能发表。

**E5. KD handoff 配 R5 Teacher 与 α=0.5（GPU 约 8.5 h，仅当 E4 在 10.7 前完成）**
- 目的：把 R4→R5 的三重混淆拆开，单独量学生臂的效应。
- 对应 claim：第 2 环作为负结果消融。
- 成功判定：KD 臂 BIDF 在 implicit 上显著低于 SUP 臂 → 干净的"KD 压制证据覆盖"消融。失败：差距不显著 → 论文只保留 R4 结果并注明混淆。

不做：200K 规模与延迟、WDC/AbeBooks 泛化、Stage-1 第二 seed、文本 span 定位消融。每项都超过 1 天或需要重跑 Stage-1。

建议时间线：10.5–10.6 做 E1–E3 并启动 E4；10.7 读 E4，视结果启动 E5；10.8 冻结数字；10.8–10.17 写作，10.17 留一天缓冲。

写作上的三点调整：把核心贡献写成"证据挂载的检索 + verifier 驱动的属性恢复使 implicit join 可发现"；Teacher 写成交叉编码重排器（对 QT-only Teacher +6pp，对学生 +1–2pp），KD 作为负结果消融；不写"证据路径提升召回"，explicit 的代价如实列出或用 E2 的结果替代。
