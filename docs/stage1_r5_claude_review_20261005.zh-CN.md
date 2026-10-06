# Stage-1 R5：Claude Code 独立复核与路线建议

- 来源：Claude Code CLI 会话 `b5a7f2a1-36b7-4cca-85a0-1e36460ce51e`（MMDD:stage1-claude 窗口，模型 Fable 5.1 / xhigh effort）
- 提出时间：2026-10-05 12:06 UTC；完成 12:24 UTC（用时 17 分 27 秒）
- 送审问题：①R5 数据与报告可信度（含 stage-1 两个 gate FAIL 与 stage-2 提升是否矛盾）②方案故事与 `方案.md` 的自洽性 ③10.18 SIGMOD 投稿的路线决策
- 说明：以下为 Claude 原始回复的完整存档，未经改写。

---


**结论先行**

1. 数据本身可信，但"R5 相对 R4 明显提升"只在 implicit 子集成立。dev 显著，test R@10 的置信区间触零、NDCG@10 显著，overall test R@10 不显著。Stage-1 gate FAIL 与 Stage-2 提升不矛盾。gate 评的是 KD 臂、量的是 Stage-1 排序与 gold 覆盖，而提升来自 SUP 臂把证据挂到了两倍的候选上。断掉的链条是"Teacher/KD 让 Stage-1 检索更好"，存活的链条是"证据挂载 → 值恢复 → implicit join"。
2. 方案故事线有 4 环被数据支持、4 环被否定或削弱、4 环没有覆盖。无法自圆其说的环节：KD 蒸馏（部署臂没有用 KD 学生）、证据路径提升召回（两臂的 candidate gain 都为负）、Stage-1 路径聚合残差提升排序（Real−f0 ≤ 0，test 上 α=0 更好）、"多模态"里的图像份额（只占 bridge 的 7–10%，无消融）。
3. 路线：封版 R5 为主结果、立即写作；并行补 3 个 CPU 级实验，合计约 1 个工作日，再加最多 1 个后台 GPU run，约 8.5 小时。不做 200K、跨数据集、第二 seed。

## 一、可信度

**Stage-1 三个 gate 评的是 KD 臂，不是交接的 SUP 臂。** `src/mmdd_stage1/pipeline.py:2036` 的 `_reports` 里，candidate gain = KD 臂 `C150_target_coverage − MatchedDirectC_target_coverage`，KD gate = KD−SUP 同 Teacher 对比加 KD 覆盖差，三个 gate 都只读 `native_kd`。

| gate | R4 sp_rerank | R5 sup_rerank | 备注 |
|---|---|---|---|
| candidate gain pp（dev/test） | −9.3907 / −10.3202 | −9.3907 / −10.3202 | 逐位相同：学生 checkpoint 都是 r4_s13 的冻结件 |
| KD coverage pp | −7.4638 / −7.8330 | −7.4638 / −7.8330 | 同上 |
| KD pp | +1.57 [−0.66, 3.73] / +0.71 [−1.30, 2.68] | +1.04 [−1.00, 3.08] / +0.80 [−1.18, 2.74] | CI 全跨 0 |
| E-content implicit pp | +1.92 [0.00, 3.91] / +0.40 [−1.28, 2.11] | +0.53 [−1.29, 2.34] / +0.94 [−0.60, 2.51] | PASS 只靠点估计 ≥ 0 |
| Teacher−QT pp | +6.51 [4.35, 8.72] / +6.06 [4.03, 8.05] | +6.09 [3.95, 8.18] / +6.00 [4.02, 7.96] | 唯一扎实的 PASS |

交接臂是 SUP，`stage2_handoff/stage1_gate.json` 的 `positive_evidence_path_coverage@10` 为 0.432，R4 的 KD 臂是 0.226。SUP 臂自己的 candidate gain 也为负：dev 0.8565−0.8853 = −2.88pp，test −3.93pp。Teacher 的证据残差在 SUP 臂上是负贡献：`alpha_scan.txt` 里 α=1 的 Real 48.40 / 47.81 低于 f0 的 49.35 / 49.40；test 上 α=0 的 overall 49.40、implicit 45.97 都高于被采用的 α=0.5 的 48.84 / 45.54。续训 Teacher 的 end.Real 对学生 Direct 是 −0.42pp [−3.03, 2.11]，没有越过学生。

**Stage-2 端到端。** BIDF_RRF60 的 R@10 / NDCG@10：

| split / population | R4 | R5 | STAGE1 R@10（R4 → R5） |
|---|---|---|---|
| dev overall | 54.88 / 37.83 | 57.20 / 39.08 | 49.86 → 49.99 |
| dev implicit | 44.31 / 28.20 | 49.29 / 30.51 | 45.30 → 45.55 |
| dev explicit | 65.44 / 47.46 | 65.11 / 47.64 | 54.42 → 54.42 |
| test overall | 53.73 / 36.23 | 54.89 / 37.80 | 48.67 → 48.84 |
| test implicit | 45.37 / 27.70 | 48.20 / 30.70 | 45.54 → 45.54 |
| test explicit | 62.09 / 44.75 | 61.58 / 44.91 | 51.80 → 52.14 |

跨轮配对 bootstrap，R5 − R4，同 query 配对（现有 CONTRASTS.csv 里没有这项）：

| split / population | BIDF R@10 | BIDF NDCG@10 |
|---|---|---|
| dev overall | +2.32 [+0.13, +4.54] | +1.25 [−0.09, +2.62] |
| dev implicit | +4.98 [+1.83, +8.11] | +2.31 [+0.52, +4.10] |
| test overall | +1.16 [−0.91, +3.17] | +1.58 [+0.33, +2.84] |
| test implicit | +2.83 [−0.03, +5.67] | +3.00 [+1.27, +4.78] |
| dev / test explicit | −0.33 / −0.51，CI 跨 0 | ≈ 0 |

轮内证据净贡献 BIDF−VISIBLE_IDF 在 implicit 上：R5 dev +5.31 [3.26, 7.49]、test +3.60 [1.92, 5.36]；R4 只有 +1.25 [0.25, 2.33]、+1.32 [0.46, 2.32]。两轮之差 dev +4.06 [1.97, 6.29]、test +2.29 [0.48, 4.18]，显著。负面是 explicit 上 BIDF−VISIBLE_IDF 的 NDCG@10 dev −1.18 [−1.86, −0.55]、test −1.27 [−2.03, −0.62]，W/L 为 3/24、1/25。

**pool 可比性。** 两轮 Stage-2 的 `config.json` 除 handoff 路径外逐字相同，但两份 handoff 的 C30 只有一半重合：

| 指标，implicit，dev / test | R4 KD handoff | R5 SUP handoff |
|---|---|---|
| C30 Jaccard | 0.50 / 0.52 | 同左 |
| gold 落在 C30 的比例 | 0.669 / 0.650 | 0.689 / 0.694 |
| C30 中带证据路径的目标比例 | 0.489 / 0.493 | 0.935 / 0.936 |
| C30 中 gold 带证据的比例 | 0.518 / 0.501 | 0.981 / 0.959 |
| 每 query 证据路径数，括号内为 image | 23.7 (4.8) / 24.8 (5.0) | 63.1 (3.5) / 65.4 (3.3) |
| recover task 中 NO_EVIDENCE 比例 | 36.9% / 33.8% | 4.0% / 3.5% |
| 恢复出 VALUE 的 slot 数 | 650 / 807 | 1725 / 1801 |
| C30 中 gold 拿到 bridge 的比例 | 0.145 / 0.165 | 0.441 / 0.470 |

gold 覆盖几乎没变，变的是候选身上挂了多少证据，而这正是 Stage-1 gate 不量的东西。

**对"矛盾"的正面回答：不矛盾。** 一是 gate 评 KD 臂、交接 SUP 臂，两轮 gate 数字逐位相同就是证据。二是 gate 量排序与覆盖，提升来自证据挂载面翻倍，两者正交：两份 handoff 的 STAGE1 R@10 几乎相同，带证据的 C30 目标却从 49% 到 93%。三是 FAIL 有信息量，它否定了"证据路径提升召回"和"KD 有效"。断掉的是"Teacher/Student 蒸馏 → 更好的 Stage-1 检索"这一层。

**其他扣分项。**
- test split 是历史暴露的回归集，`reports/RESULTS.md` 自述；全程单 seed。
- R4 → R5 同时改了学生臂、Teacher 续训池、α 三件事，归因未隔离。
- bridge 假阳性高：implicit 有 bridge 的 query 平均带 9–10 个非 gold 的 bridge 目标，top bridge 落在 gold 的比例 0.21 / 0.25，explicit 只有 0.16 / 0.03。
- selector holdout 略降，MRR 0.9402 → 0.9349，Hit@1 0.887 → 0.874。
- R4 报告的 crop 消融被写成"稳步正向"，实际 BIDF 排序只在 dev 2/1198、test 11/1166 个 query 上有差别，是 null。

## 二、方案故事线逐条对照

| # | 方案主张 | 实测 | 判定 |
|---|---|---|---|
| 1 | 统一有向 J(a→b)，Teacher 做跨对象交互 | Teacher−QT +6.09 / +6.00，CI 不跨 0 | 支持，有向性未测 |
| 2 | Teacher → Student 蒸馏 | KD gate FAIL；KD 把 E 覆盖从 0.86 压到 0.48；部署臂是 SUP | 否定，部署系统里没有蒸馏 |
| 3 | Q→E→T 扩充召回 | candidate gain KD −9.4 / −10.3，SUP −2.9 / −3.9；C30 里只靠证据到达的 gold 仅 12/452、9/444 | 否定 |
| 4 | Q→E→T 为目标挂证据供二阶段用 | 带证据候选 0.49 → 0.93 带来 implicit +3–5pp | 强支持，方案没把它写成主机制 |
| 5 | 路径聚合残差改善 Teacher 排序 | Real−f0 −0.74 / −0.70；SUP 臂 α=1 比 f0 低 0.95 / 1.59；test 偏好 α=0 | 否定或中性 |
| 6 | 续训 Teacher 作冻结重排器 | 对学生 −0.42pp 跨 0；但 STAGE1 比学生 Direct 高 1.2 / 2.1pp | 弱支持，价值在 f0 的交叉编码 |
| 7 | verifier 列选择 | holdout MRR 0.935、Hit@1 0.874 | 支持，无端到端消融 |
| 8 | 图像 crop 定位 | 排序只变 2/1198、11/1166 | null |
| 9 | 文本 span 定位 | 无消融 | 未覆盖 |
| 10 | 生成桥接属性值 | 隐藏属性值精度 R5 0.763 / 0.707；33% implicit query 至少恢复 1 个正确值，R4 是 10–12% | 支持，本次新算，论文目前没有此指标 |
| 11 | 补全后提升 joinability | implicit 净 +5.31 / +3.60；overall 净 +2.41 / +1.63；explicit NDCG −1.18 / −1.27 | 支持 implicit，以 explicit 为代价 |
| 12 | "多模态"证据 | image 来源的 VALUE 占 6.6% / 10.3%，带 image claim 的 bridge 占 7.2% / 9.2%；数据集里 15% implicit query 只能靠图像 | 弱，无隔离消融 |
| 13 | 二阶段用 softmax(r_T)·ρ 打分 | 实现是 RRF(Stage-1, 分层(bridge, vis_idf)) | 方案与代码不一致 |
| 14 | 2K/20K/200K、ANN 延迟 | 只有 20K EntiTables | 未覆盖 |

无法自圆其说的环节：第 2 环，方案把蒸馏写成核心，交付链路里 Teacher 只做重排、学生是纯监督的。第 3 和第 5 环，证据路径既不扩召回也不改善排序，能成立的说法只有"证据路径决定哪些候选带着什么证据进入 verifier"。第 12 环，标题级的"多模态"只由不到 10% 的 bridge 支撑。第 11 环，方法在一半 query 上显著变差，现有报告只强调 overall 和 implicit。

## 三、路线决策

**决定：封版 R5 为主结果，写作从今天开始；实验只补便宜且直接堵审稿人攻击点的几项，10 月 8 日冻结数字。** 成本依据是 R5 实测墙钟：Teacher 续训 1.1 h，重评 0.8 h，export 1.1 h，Stage-2 7.2 h，其中 features 3.4 h、recover 3.5 h。完整一轮约 10.5 h。到 10.18 还有 13 天，正文至少要 8 天，任何重跑 Stage-1 训练或多 seed 的事都不在预算内。

按性价比排序：

- **E1 机制指标落盘，CPU 2–3 h，必做。** 目的：给第 4、10、11 环提供可引用的表，把 R5 vs R4 的跨轮配对 CI 写进论文。数据：两轮 PER_QUERY.csv、recovery/*.json、handoff retrieval、数据集 evidence_recoveries。脚本：新建 `src/analyze_stage2_mechanism.py`。成功判定：复现本文数字；失败只是脚本 bug。
- **E2 融合规则修 explicit 负收益，CPU 约 1 h，每个变体 score 10 min + evaluate 1 min，必做。** 对应 claim：B+IDF 在 overall 上无损提升。做法：复制 run root 并软链 catalog、population、recovery、matching；在 `src/mmdd_stage2/visible.py` 加 `fusion.mode`，变体为 bridge 值 IDF 加权、三路 RRF 取代分层、仅当 bridge 分 ≥ vis_idf 才进第一层。dev 上选，test 上报。成功：test 上 explicit BIDF−VISIBLE NDCG@10 的 CI 覆盖 0，且 implicit 净收益 ≥ +3.0pp 且 CI > 0，且 overall 不低于当前 +6.05pp。失败：保留现规则，如实写 tradeoff。
- **E3 图像证据贡献，CPU 约 1 h，必做。** 对应第 12 环。做法：零成本子集分析，按数据集标出 178 个只能靠图像、264 个图文混合、740 个纯文本的 implicit query，用现有 PER_QUERY 分别算 BIDF−VISIBLE；再做打分级消融，剔除 `tasks[].image_count>0` 的 claim 重算 bridge 后跑 score + evaluate。成功：图像子集或消融差分在 implicit 上 CI > 0。失败：把图像写成"已接入但当前规模下贡献有限"，引用 7–10% 占比。严格消融要重跑 recover 3.5 h，可选。
- **E4 α=0 handoff 端到端，GPU 后台约 8.5 h，建议今晚启动。** 对应第 5 环和 α 的选择理由。做法：`run_stage1.py export --arm SUP --residual-scale 0` 到新目录，跑完整 Stage-2。成功：α=0.5 对 α=0 的 BIDF 在 dev 上 CI > 0，保留并写成消融。失败：持平或 α=0 更好，论文采用 α=0 作为更简单的系统，删掉路径聚合重排的论断。两种结果都能发表。
- **E5 KD handoff 配 R5 Teacher 与 α=0.5，GPU 约 8.5 h，仅当 E4 在 10.7 前完成。** 目的：拆开 R4→R5 的三重混淆，单独量学生臂效应。成功：KD 臂 implicit 显著低于 SUP 臂，得到干净的"KD 压制证据覆盖"消融。失败：只保留 R4 结果并注明混淆。

不做：200K 规模与延迟、WDC/AbeBooks 泛化、Stage-1 第二 seed、文本 span 消融，每项都超过 1 天或需要重跑 Stage-1。

时间线：10.5–10.6 做 E1–E3 并启动 E4；10.7 读 E4，视结果启动 E5；10.8 冻结数字；10.8–10.17 写作，10.17 缓冲。

写作上的三点调整：核心贡献写成"证据挂载的检索 + verifier 驱动的属性恢复使 implicit join 可发现"；Teacher 写成交叉编码重排器，对 QT-only Teacher +6pp、对学生 +1–2pp，KD 作为负结果消融；不写"证据路径提升召回"，explicit 的代价如实列出或用 E2 的结果替代。
