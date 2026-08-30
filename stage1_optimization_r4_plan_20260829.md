# Stage-1 第四轮计划:按湖实例化 —— 分湖语料、分湖训练、蒸馏回归主线(2026-08-29)

## 背景与动机(执行前必读)

### 第三轮(work/stage1_optimization_r3_20260829)已确认的结论

1. **两级部署形态已立住**:Student ANN 召回(PCA-1024 epoch-0)+ Teacher z-score ensemble(α=0.7)重排,overall R@10 = 46.40%,与 raw 召回 + 重排(47.28%)差距 −0.88pt(95% CI [−1.84, +0.00]),而 student 索引只有 raw 的 1/3.79(1.15 vs 4.34 GiB)。
2. **Task H(ensemble-KD)的真实结果是"蒸馏有效、共享模型不行"**:KD=0.3 下 EntiTables direct 从 29.74% 涨到 ep8 的 35.39%(**+5.65pt,项目至今最大训练增益**),但 WDC 沿既有 trade-off 曲线从 60.4% 掉到 48%,CI gate 全部 fail。失败的不是蒸馏信号,是"单个共享 R_table_table 同时服务两个机制不同的数据湖"这一前提(r2 的 relation-drift 曲线:‖R−I‖ 增加时 EntiTables 单调涨、WDC 单调跌)。
3. γ*=10、γ_e*=2、recall_ks={10,20,30,40,50}、配对 bootstrap gate(tolerance 0.02)已就位,沿用。
4. Teacher ensemble 重排在 E→T 上也有效(43.69% vs raw 33.13%,r2 结论)。

### 领域惯例查证(Snoopy TKDE'25、LakeBench PVLDB'24)

- **两篇工作全部按数据湖分别训练**:Snoopy 的 proxy 矩阵按 repository 训练,其所有微调 baseline(DeepJoin/Starmie/BERT*)也按 repository 各自微调(Sec V-A);LakeBench 对每个 learned 方法的 fine-tune+embed+index 整条离线流水线按湖分别跑,并分湖报离线成本(Table 5/6)。**没有任何一篇跨湖联合训练。**
- **每个湖是独立语料**:评测时查询只在自己湖内检索。我们当前把 EntiTables + WDC 合并成 27.9 万对象的混合语料,EntiTables 查询会检回 WDC 表作干扰项——这不符合领域惯例,也不符合部署现实(数据湖离线各自建索引)。
- 评测口径:Snoopy 用 R/NDCG@{5,15,25},LakeBench 用 P/R@k(k=20 或 50)。我们的 recall@{10..50} 在惯例范围内,保持不变。

### 本轮核心决策:框架统一、实例按湖

同一套方法(冻结 encoder → PCA student → 锚定关系学习 → teacher → ensemble → 蒸馏),**每个数据湖离线训练自己的实例、建自己的索引**。原联合训练的 trade-off 结果(relation-drift 曲线、Task H 轨迹)转为论文的动机分析素材:"为什么必须按湖实例化"。预期叙事增益:蒸馏(ensemble-KD)在 EntiTables 上按 Task H 轨迹应能超过纯监督;WDC 若收敛在 epoch-0 附近(R≈I 即最优),写成框架自适应复杂度——湖由表面相似主导时自动退化为 raw 检索,湖需要语义桥接时蒸馏发力。

## 全局约束

- 沿用 r3 的最终配置:recall_ks={10,20,30,40,50}、γ=10、γ_e=2、weighted_rrf(evidence 0.05)、aggregator logsumexp/top4、配对 bootstrap gate(iterations 10,000, seed 13)。
- **gate 改为按湖独立**:每个湖的训练只受自己湖的 gate 约束(相对本湖 raw 的 delta 下界 ≥ −0.02),不存在跨湖 gate。
- 特征缓存不变(embedding 与湖无关);语料、PCA、索引、训练列表按湖拆分。
- 每任务结果追加 `work/stage1_optimization_r4_20260829/RESULTS.md`;最终产出 FINAL.md。
- 所有新增开关向后兼容;新增代码配套测试,全套测试保持通过。

## 本轮验收目标

1. 分湖口径下,EntiTables:**KD student 召回 direct R@10 > 纯监督 student > raw**(蒸馏 ablation 链成立);
2. 分湖口径下,WDC:student 相对本湖 raw 的 CI gate pass(预期靠近 epoch-0);
3. 两湖的 "student 召回 + teacher ensemble 重排" 均 ≥ 对应 "raw 召回 + 重排" − 1pt(CI 口径);
4. FINAL.md 产出分湖主表(论文主表雏形)。

---

## Task J:分湖语料与基线重建(零训练,一切对比的新地基,必须先做)

**理由**:混合语料让跨湖表互为干扰项,与领域惯例和部署现实都不符;拆开后所有系统的数字预期整体上浮,历史混合口径数字全部作废,四条基线必须在分湖口径下重建。

**实现**:
1. **语料拆分**:从 `mixed_stage1_data/stage1_corpus.jsonl` 按对象所属数据集拆成 `entitables_corpus.jsonl` 与 `wdc_corpus.jsonl`(对象→数据集的归属可从两个源 stage1_data 目录的对象 ID 集合判定;写一个拆分脚本并校验两湖对象数之和等于混合语料数、无交集)。
2. **分湖 PCA**:对每个湖的语料 embedding 各自做 top-1024 主成分(复用 `src/mmdd_stage1/pca.py`),记录各自方差保留率。**不要复用混合 PCA**——每湖的谱不同,这本身是"按湖实例化"的一部分。
3. **分湖索引**:每湖建立自己的 raw 索引与 student(分湖 PCA epoch-0)索引。
4. **分湖评测**:每湖 dev 查询只在本湖语料内检索。四系统 × 两湖:(1) raw;(2) student epoch-0(分湖 PCA,R=I,零训练);(3) raw + teacher ensemble(现有混合训练的 teacher checkpoint,α=0.7);(4) student + teacher ensemble。输出 recall@{10..50}、mrr@50、coverage@10、CI(vs 本湖 raw)、每查询耗时、索引大小。
5. 顺带记录:混合口径 vs 分湖口径的 raw 数字对比(一张小表,量化"干扰项移除"的影响,论文 setup 一句话用)。

**验收**:`taskJ_per_lake_baselines/RESULTS.md` 落盘 4×2 基线表;两湖对象数校验通过。**预期检查点**:若分湖后 raw 数字反而下降,立即停下报告(说明拆分或索引有 bug,不要继续)。

---

## Task K:分湖 student 训练 + 蒸馏 ablation(本轮核心)

**理由**:Task H 已证明 ensemble-KD 信号在 EntiTables 上有效(+5.65pt 轨迹),死因只是共享模型的 WDC 约束。分湖后该约束消失,蒸馏 ablation 链(raw < 纯监督 < KD)有望首次完整成立。WDC 侧同配置跑一次,预期收敛在 epoch-0 附近,作为"框架自适应复杂度"的证据。

**实现**:
1. **训练列表按湖拆分**:现有 path/edge 训练列表本就带 `dataset` 字段,按湖过滤即可;in-batch negatives 自然只来自本湖(确认实现:batch 内候选不跨湖)。
2. **EntiTables 三组**(fresh 分湖 PCA-1024、冻结 P、path-only、in-batch、μ=0.1、本湖 CI gate):
   - K-a:纯监督(KD=0)——分湖口径的训练基线;
   - K-b:ensemble-KD,distillation-weight=0.3(Task H 的赢家配置;蒸馏目标 = 0.7·z(s_T)+0.3·z(cos),用现有混合 teacher 对 EntiTables 列表离线预计算,复用 r3 的 taskH_teacher_logits 流程);
   - K-c:ensemble-KD,distillation-weight=1.0(剂量对照)。
   - epochs 12(Task H 显示 ep8 才到峰值,给足预算),patience 4。
3. **WDC 两组**(同配置):
   - K-d:纯监督;K-e:ensemble-KD 0.3。
   - 预期两组都选中 epoch-0 或极小偏移——**这是预期结果,不是失败**;如实记录 relation_drift 与选中 epoch。
4. 所有组每 epoch 记录本湖 direct/evidence/fused R@10、relation_drift、CI-vs-raw;checkpoint 选择 = 满足本湖 gate 的 epoch 中 overall(本湖)fused R@10 最大者。

**验收/判读**:
- EntiTables:K-b 或 K-c 的 direct R@10 > K-a > raw(本湖)→ **蒸馏 ablation 链成立,这是论文核心表**;若 KD 组 ≤ K-a,记录并分析(KD 目标与分湖 PCA 的适配问题),蒸馏叙事退回"teacher 仅作 reranker"。
- WDC:K-d/K-e 若停在 epoch-0 附近且 gate pass → 记录为自适应复杂度证据;若意外出现正增益,更好,如实记录。

---

## Task L:分湖最优 student + ensemble 重排的端到端合成

**理由**:Task K 选出每湖最优 student 后,需要验证完整部署链(本湖 student 召回 → teacher ensemble 重排)在分湖口径下的最终数字,并与 Task J 的系统 (3)(raw 召回 + 重排)对比,确认"紧凑索引不掉点"在分湖口径下依然成立。

**实现**:每湖用 Task K 最优 checkpoint 建索引,跑 student+ensemble 与 raw+ensemble 的配对对比(CI);同时报每湖索引大小与每查询耗时。若 Task K 的 EntiTables KD student 显著超过 epoch-0,ensemble 重排的增益预期会部分重叠(student 变强后 rerank 边际收益变小)——如实记录两者的边际贡献分解(student-only / +rerank 两行)。

**验收**:两湖 student+ensemble ≥ raw+ensemble − 1pt(CI 下界口径);`taskL_end_to_end/RESULTS.md`。

---

## Task M(可选,时间允许再做):分湖 teacher 重训

**理由**:当前 teacher 是混合训练的。叙事完全干净的版本是 teacher 也按湖实例化;且 EntiTables-only 的 teacher 在本湖重排与蒸馏目标上都可能更强。但 Task K 若已用混合 teacher 拿到达标增益,本任务可降级为 camera-ready 前的完善项。

**实现**:仅当 Task K 的 EntiTables KD 增益 < +2pt(相对 K-a)或 Task L 未达标时执行:用 EntiTables-only 的检索对齐列表(复用 task7 的列表生成逻辑,负样本从本湖 raw ANN 取)重训 teacher(edge→path),重排验收口径同 task7(本湖 raw top-γk 重排 R@10 > 本湖 raw + 3pt),然后用新 teacher 重算 ensemble 与 KD 目标,重跑 K-b。

**验收**:执行与否都在 RESULTS.md 写明决策依据。

---

## Task N:hard-negative mining 解锁评估 + 最终汇总

**理由**:分湖后 mining 首次有了正当基础:挖掘索引用本湖最优 student,打分用 ensemble,负样本不会再被跨湖干扰项污染。但仅在 EntiTables 上尝试(WDC 的 student 若停在 epoch-0,mining 无意义)。

**实现**:
1. 仅当 Task K 的 EntiTables 蒸馏链成立时:对 EntiTables 跑一轮 mining(挖掘用最优 student 索引,候选打分用 ensemble 分数而非旧 teacher logits),hard-fraction 0.5、hard-lr 2e-5,gate 同 Task K;对比 mining 前后。
2. **FINAL.md 主表**(分湖,每湖一张):raw / student epoch-0 / 纯监督 student / KD student / student+ensemble(+mining 若做了)× recall@{10..50} + mrr@50 + coverage@10,关键格附 CI;附索引大小与耗时表、两张 γ 辩护图引用、relation-drift 动机图引用(来自 r2/r3,作为"为什么按湖实例化"的证据链)。
3. 记录所有最终默认值与 checkpoint/语料 SHA-256。

**验收**:FINAL.md 齐全;每湖主表能直接支撑论文实验节的写作。

---

## 执行顺序与决策树

```
Task J(分湖语料/PCA/索引/基线)
  ├─ raw 分湖数字异常下降 → 停,报告拆分 bug
  └─ 正常 → Task K(EntiTables 3 组 + WDC 2 组)
        ├─ EntiTables 蒸馏链成立(KD > 纯监督 > raw)→ Task L → Task N(含 mining)
        ├─ KD ≤ 纯监督 且增益 < +2pt → Task M(分湖 teacher)→ 重跑 K-b → Task L → Task N
        └─ WDC 停在 epoch-0 → 按"自适应复杂度"记录(预期内)
  Task N 汇总 FINAL.md
```

## 产出要求

- 每任务独立目录 `work/stage1_optimization_r4_20260829/task{J,K,L,M,N}_*/`。
- 语料拆分脚本与校验、分湖 PCA、分湖评测的代码改动分开成明确变更集,各配测试。
- checkpoint/索引只保留 epoch-0/best/final;RESULTS.md 逐任务追加。
