# Stage-1 Student 优化计划(2026-08-28)

## 背景与已确认的结论(执行前必读)

以下结论已由实验确认,后续所有改动以此为前提:

1. **Raw 基线**:冻结 Qwen3-VL-Embedding 4096 维原始向量,mixed dev(1,140 查询)上 direct `Recall@10 = 36.84%`,fused(RRF)= 19.65%。
2. **零训练 PCA 天花板**(`work/stage1_pca_dimension_ceiling_20260828/`):PCA-1024 零训练 direct `R@10 = 35.26%`(raw 的 95.7%),**选定 1024 维**作为 Student 维度。
3. **训练在破坏检索几何**:identity-4096 初始化训练后 best 出现在 epoch 1(direct 14.3%),之后逐 epoch 变差;零训练起点本应 ≈36.8%。即当前训练目标(2–5 个候选的窄列表 CE + 蒸馏一个过拟合 Teacher 的 KD)的梯度方向与"保持全局近邻结构"相互对抗。**这是当前最主要的问题。**
4. **RRF 融合在伤害强通道**:raw direct 36.8% → fused 19.6%;identity-4096 组 direct 14.3% → fused 6.1%。弱且噪的 evidence 通道通过 RRF 把 direct 排名拖低。
5. **Evidence 通道有独立的上限问题**(`reports/stage1_evidence_diagnostic_20260828/`):标注干净(recovery 覆盖 100%),text Q→E 良好(中位 rank 5),但 image Q→E 弱(中位 65.5)、E→T 第二跳弱(recall@50 = 59.1%),raw 联合可达率 43.5%,top-10 正确路径仅 13.7%。
6. **Hard-negative 轮无效**:round-1 从已崩坏的 student ANN 索引挖负样本,r10 仅从 1.49% 到 1.67%。在 student 修好之前不要再跑 mining。
7. Teacher 已过拟合(teacher-edge dev loss epoch 2 见底后反弹),暂不重训,冻结复用 `teacher_path.pt`(保持 teacher logits 缓存命中)。

## 全局约束(每个任务都要遵守)

- 语料/特征/dev 集固定使用 `work/stage1_stage2_wdc2k_entitables20k_v4_20260828` 那套(`mixed_stage1_data/stage1_corpus.jsonl` + `features_qwen3_vl_embedding_8b`),保证与 raw 对照口径一致。
- Aggregator 固定 `logsumexp, evidence_top_k=4`,不要改,否则 teacher logits 缓存 key 失配触发重打分。
- 每次实验必须报告:overall 与 **分数据集(EntiTables / WDC)** 的 direct / evidence / fused `R@10`、`R@100`、`mrr@100`,以及 raw 对照。WDC 上 learned=0 是历史问题,必须持续跟踪。
- 不加载任何旧的 128 维 checkpoint;student 一律从新初始化开始。
- 所有训练实验 `hard_examples=0`(base 数据 only)。
- 每完成一个任务,把结果(指标表 + 结论 + 产物路径)追加到 `work/stage1_optimization_20260828/RESULTS.md`。

---

## Task 0:评测插桩——epoch-0 基准 + 分数据集指标(先做,所有后续任务依赖它)

**理由**:当前 `_EpochController` 只在每个 epoch 结束后评测,没有"训练前"基准,无法判断训练是在改善还是破坏。所有后续任务的验收线都是"任何 epoch 不得低于 epoch-0"。

**实现**:
1. `src/train_stage1.py` 的 `_EpochController`:新增 epoch-0 评测——在 `student-path` 阶段进入训练循环之前,对初始(未训练)student 建索引 + 跑 `evaluate_student_retrieval`,结果记为 `epoch: 0` 写入 history,并作为 gate 的初始 best。加开关 `--eval-epoch-zero`(默认开启)。
2. `src/mmdd_stage1/evaluation.py` 的 `evaluate_student_retrieval`:新增按 `example.dataset` 分组的指标输出(`by_dataset` 键,含各数据集的 fused/direct/evidence recall 与 mrr)。`TargetExample` 已有 `dataset` 字段,直接分组即可。
3. history/selection JSON 中保留 raw 对照(现有 `raw_embedding` 字段)以及新的 `by_dataset`。

**验收**:PCA-1024 冻结投影配置下,epoch-0 direct `R@10 ≈ 35.26%`(与零训练探针一致,允许 ±0.5pt 的 HNSW 抖动)。分数据集指标能在 history JSON 里看到。

---

## Task 1:PCA-1024 + 冻结 P + 只训 R + 锚定正则 + 低学习率

**理由**:训练破坏几何的根源是可训空间太大(P 是整个 4096→d 投影)而监督太窄(每列表 2–5 候选)。把可训参数缩小到 9 个 d×d 关系矩阵、并用正则把 R 锚在单位阵附近,训练最坏情况退回零训练天花板(35.26%),只有真实信号足够强才允许偏移。

**实现**:
1. `src/mmdd_stage1/models.py` `StudentJoinabilityModel`:
   - 支持 `init_mode="pca"`:用 `src/mmdd_stage1/pca.py` 现有的共享 top-d 主成分设置 `P = U_d^T`(所有 object type 共享同一个 U),`R = I`(已有实现,确认复用)。
   - 新增 `freeze_projections: bool`:为 True 时对所有 `self.projections` 参数设 `requires_grad=False`,checkpoint 的 config 里记录该标志。
2. `src/train_stage1.py`:新增 `--student-init {random,identity,orthogonal,pca}`(部分已有,补齐 pca-1024 路径)、`--freeze-projection`、`--anchor-weight μ`(默认 0)。
3. `src/mmdd_stage1/training.py` 的 `train_student_edges` / `train_student_paths`:loss 增加锚定项
   `L += μ * Σ_pairs ‖R_pair − I‖²_F / (d²)`(除以 d² 做尺度归一;若未冻结 P,再加 `μ‖P − P₀‖²_F/(4096·d)`,P₀ 为初始投影的常量副本,注册为 buffer)。锚定项也计入 history 的分项 loss。
4. 学习率:student 阶段新增 `--relation-learning-rate`,冻结 P 时默认 `1e-5`(现在的 1e-4 对 identity 量级参数破坏性太强)。

**运行**:PCA-1024、冻结 P、`anchor-weight` 扫 {0, 0.01, 0.1, 1.0}(4 组,每组 student-edge 10 epoch → student-path 到 early stop;若时间紧,先跑 μ=0 与 μ=0.1 两组)。

**验收/判读**:
- 硬性验收线:**每个 epoch 的 direct `R@10` ≥ epoch-0(35.26%)**。做不到即视为该 μ 失败。
- 若某组能稳定不低于 epoch-0 且 fused/evidence 有改善 → Task 1 成功,固定该配置进入 Task 3。
- 若所有 μ 都从 epoch-1 开始跌破 epoch-0 → 坏梯度完全来自目标函数,直接进入 Task 2 归因,并提高 Task 3 优先级。

---

## Task 2:KD 归因消融(一次训练,回答"该不该扔掉 Teacher")

**理由**:破坏性梯度有两个可能来源——窄列表监督 CE、以及蒸馏一个过拟合 Teacher 的 KD。归因结果决定后续是否需要修 Teacher(修 Teacher 成本高,能不修就不修)。

**实现**:在 Task 1 的最优配置上,仅改 `--distillation-weight 0`(纯监督)重跑 student-path,与 `--distillation-weight 1.0` 对照。

**判读**:
- KD=0 明显更好 → Teacher 的 KD 是主要污染源:后续默认 `distillation-weight ≤ 0.1` 或直接 0,Teacher 重训降级为远期任务。
- 两者相近或 KD=0 更差 → 破坏来自窄列表 CE 本身,Task 3(拓宽负样本)成为决定性任务,KD 保留。

---

## Task 2b:Teacher-as-reranker 诊断——直接测量 Teacher 的检索价值(与 Task 2 并行,零训练)

**理由**:Teacher 至今只在 2–5 个手工候选的窄列表上被评估过(dev loss),从未在检索口径上被测量。而 Teacher 的 modality adapter 同样是随机初始化 + 窄列表训练,和 student 得的是同一种病(epoch 2 即过拟合)。Student edge KD 拟合很好(KL 0.198)但检索崩溃,说明"忠实蒸馏一个不含检索几何的分布"完全可能。这个诊断决定 Teacher 的去留,也决定 Task 6 的 mining 数据能否信任(mining 候选由 Teacher 打分标注)。

**实现**(新脚本 `src/diagnose_stage1_teacher_rerank.py`,零训练):
1. 对每个 dev 查询,用 raw 索引(`RawEmbeddingANNIndices`)取 direct top-100 table 候选。
2. 用冻结 `teacher_path.pt` 对 (query, candidate) 逐对打分(`score_pairs`,batch 打分,hidden states 走现有 FeatureStore 缓存;复用 `compression_cache` 避免重复压缩)。
3. 按 teacher 分数重排,计算重排后的 `R@10`/`R@100`/`mrr@100`(overall + 分数据集),与 raw direct(36.84%)对照。GT 不在 top-100 内的查询按 miss 计,口径与 raw 相同。
4. 顺带输出:teacher 分数与 raw 内积分数在候选上的 Spearman 相关(判断 teacher 学到的是不是原始相似度的复读)。

**判读**:
- Teacher 重排 R@10 明显高于 36.84% → Teacher 有真实细粒度信号:student 修好后保留 KD;可考虑把 Teacher 用作在线 rerank 层。
- Teacher 重排 ≈ raw(±2pt)→ Teacher 无增量价值:移除 KD 与 teacher-logits 管线,student 走纯监督 + in-batch(Task 3),Teacher 重训降为远期可选项。
- Teacher 重排 < raw → Teacher 主动有害,且 Task 6 的 mining 标注不可信:冻结 Task 6,Teacher 必须用检索对齐列表(raw top-k 负样本)重训后再进入任何蒸馏/打分环节。

**验收**:诊断报告落盘 `work/stage1_optimization_20260828/task2b_teacher_rerank/`(JSON + RESULTS.md 摘要),明确给出上面三分支中的哪一支。

---

## Task 3:In-batch negatives——把训练分布对齐检索分布(让 Student 有机会超过 raw)

**理由**:Task 1/2 只能做到"不输给零训练天花板"。Student 存在的意义是**超过** raw(否则直接用 raw 索引即可)。目前每个列表只有 2–5 个手工候选,模型学到的是"在小列表里排第一",与"在 27.9 万对象里排前 10"脱钩。In-batch negatives 把有效列表宽度从 ~5 提升到数百,成本几乎为零(student 是双线性双塔,batch 内打分就是一次矩阵乘)。

**实现**(只改 Student 路径,Teacher 不动):
1. `src/mmdd_stage1/scoring.py` 新增 `score_edge_batch_in_batch`(或给 `score_edge_batch` 加开关):对一个 batch 的 edge examples,收集 batch 内所有 destination 候选(按 `destination_type` 分组去重,并排除与当前 query 的正例 ID 相同的对象),对每个 query 将"原列表候选 + batch 内同类型其他候选"一起打分。实现上用 `student.project` 批量投影 + `u_q @ R @ U_cands^T` 一次矩阵乘,不要逐对循环。
2. `src/mmdd_stage1/training.py` `train_student_edges`:监督 CE 在**拓宽后的列表**上计算;KD 仍只在原列表上计算(teacher 对 in-batch 负样本没有 logits,不要为它们请求 teacher 打分)。加开关 `--in-batch-negatives`(默认关,保持兼容)与 `--in-batch-max-negatives`(默认 256,超出时随机下采样)。
3. `train_student_paths` 的 direct 通道同样支持 in-batch 拓宽(target 候选列表加 batch 内其他 target);evidence 通道第一版不改(结构复杂,收益后置)。
4. 正例冲突处理:batch 内其他 query 的正例若与当前 query 的正例集合(`positive_target_ids`)重叠,必须从负样本中排除(用 ID 集合过滤)。

**运行**:Task 1 最优配置 + `--in-batch-negatives`,对比开/关。

**验收**:direct `R@10` **超过 raw 的 36.84%**。这是本计划的核心成功指标。若达成,student 第一次证明了自身价值;若未达成但稳定在 35–36.8%,记录差距并继续 Task 4(融合仍能带来整体收益)。

---

## Task 4:修复 RRF 融合——保证 fused 永不低于 direct

**理由**:raw 口径下 fused(19.6%)远低于 direct(36.8%),说明现行"无条件 RRF 合并两通道"在 evidence 通道弱时是净伤害。修好它对所有配置立即生效,与训练无关。

**实现**(`src/mmdd_stage1/retrieval.py` `retrieve_zero_one_hop_detailed`):
1. 新增融合模式开关 `fusion_mode`:`rrf`(现状)/ `weighted_rrf` / `gated`。
   - `weighted_rrf`:`score = w_D/(k+rank_D) + w_E/(k+rank_E)`,新增参数 `direct_weight`(默认 1.0)、`evidence_weight`(默认扫 {1.0, 0.5, 0.25, 0.1})。
   - `gated`:evidence 通道只有当该 target 的 evidence 路径数 ≥ 2 或 evidence 分数超过分位数阈值(如该查询 evidence 分数分布的 P75)时才参与 RRF,否则只用 direct 排名。
2. `evaluate_student_retrieval` 与 `train_stage1.py` 透传这些参数;评测输出里 direct-only 指标已存在,直接用于对照。
3. 在 dev 上对 raw 索引离线扫参(不训练,复用 `RawEmbeddingANNIndices`):选出 fused ≥ direct − 0.5pt 且 evidence coverage 尽可能高的配置,作为新默认。

**验收**:raw 口径下新融合的 fused `R@10` ≥ 35%(接近 direct),同时 `positive_evidence_path_coverage@10` 不低于现行 RRF 的 6.84%。学习后的 student 沿用同一配置。

---

## Task 5:Evidence 通道修复(独立战线,在 Task 1–4 有结论后启动)

**理由**:诊断确认 evidence 通道上限来自三处——E→T 第二跳弱(raw recall@50 = 59.1%,EntiTables 尤其差)、image Q→E 弱(中位 rank 65.5)、融合损耗(Task 4 解决)。这些与 direct 通道崩塌无关,单独修。

**实现**(按优先级):
1. **E→T 专项训练**:在 Task 3 的 in-batch 框架下,对 `text→table` / `image→table` 的 edge 列表启用 in-batch negatives(第一版 Task 3 已覆盖 edge 阶段,这里确认这两类 type-pair 的样本量与梯度占比,必要时对这两类边过采样,开关 `--edge-type-oversample text_table:2,image_table:2` 之类)。
2. **image 通道降权**:在 Task 4 的 `weighted_rrf`/aggregation 里允许按 evidence 模态给权重(`--evidence-modality-weights text=1.0,image=0.3`),image 检索质量修好之前先止损。
3. **Stage-2 gate 收紧**:`train_stage1.py` 的 gate 从"至少 1 个 evidence path query"改为同时约束 overall 与每个数据集的 `positive_evidence_path_coverage@10`(阈值参数化,建议初值 overall ≥ 3%、每数据集 ≥ 1%),防止再出现"gate 通过但通道实际为 0"的误判。

**验收**:evidence 通道 `R@10` 超过 raw evidence 的 10.2%,coverage@10 明显高于 6.84%,且 WDC 子集不再为 0。

---

## Task 6:Hard-negative mining 重启(只有 Task 3 验收通过后才做)

**理由**:mining 的价值取决于挖掘索引的质量。此前从崩坏 student 索引里挖,等于从错误分布采样。当 student direct ≥ raw 后,它的 ANN 排名里的高分错误才是真正的 hard negatives。

**实现**:复用现有 `refresh_stage1_hard_negatives.py` / `run_stage1_rounds.py` 流程,唯一前置条件是 `--student-checkpoint` 指向 Task 3 的验收 checkpoint;`--hard-learning-rate` 维持 2e-5,轮数先跑 1 轮看增量,验收线同样是"分数据集 direct/fused 不低于 mining 前"。

---

## 执行顺序与决策树

```
Task 0(插桩)
  → Task 1(冻结 P + 锚定,4 组 μ)‖ Task 2b(teacher rerank 诊断,零训练,可并行先跑)
      ├─ 有 μ 不跌破 epoch-0 → Task 2(KD 消融)→ Task 3(in-batch)
      └─ 全部跌破 epoch-0   → Task 2(KD 消融)→ Task 3(in-batch,优先级提升)
  → Task 2b 判读:teacher ≈ raw 或更差 → Task 3 起全程 KD=0;teacher < raw 时冻结 Task 6
  → Task 3 达到 >36.84% → Task 4(融合)→ Task 5(evidence)→ Task 6(mining,需 Task 2b 未冻结)
  → Task 3 未达标但 ≥35% → 仍做 Task 4/5(整体收益),Task 6 暂缓,另行分析差距
```

## 产出要求

- 每个任务一个独立 run 目录:`work/stage1_optimization_20260828/task{N}_*/`。
- `RESULTS.md` 汇总表:配置、epoch-0 与 best 的 direct/fused/evidence R@10(overall + 分数据集)、与 raw 对照、结论一句话。
- 所有新增开关默认值保持向后兼容(不加参数时行为与现状一致)。
- 新增代码配套单元测试(沿用 `tests/test_stage1_*` 的既有风格),全套测试保持通过。
