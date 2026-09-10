# MMDD Stage-1 Optimization R18
## Teacher Representation / Relation Structure under Assumed-Negative Training

**版本：2026-09-10 v3**
**本文件是 R18 的冻结执行方案。**
**本轮只优化 Stage 1；Stage 2 不运行、不作为 gate、不作为模型选择依据。**

---

# 0. 本轮科研问题

R16 已确认当前 Teacher 在自然候选排序上严重弱于 Student B13：

- B13 / S-full13@178：Recall@10 约 29.01%；
- 当前 Teacher-QT：Recall@10 约 8.47%；
- 即使只在原 ANN direct100 内重排，Teacher 也明显弱于 Student。

R17 原计划继续诊断 Teacher，但 tiny-overfit gate 因缺少 verified negative 被阻塞，导致 Global Residual、Relation-Specific Heads 等结构实验没有真正执行。因此：

> **R17 没有得到新 Teacher 结构有效或无效的结论。**

R18 现在采用新的训练语义：

> 对于训练 list，已知 positive 作为正例；其余未被标注为 positive 的有效 candidate（原来为 unknown）暂按 **assumed negative** 使用。

理由不是“unknown 已被证明为负例”，而是：在当前大规模候选空间中，unknown 大概率确实为非相关对象；人工补全 verified negative 成本过高，而本轮目标是先判断 Teacher 的表示/结构假说是否值得继续。

因此 R18 的核心问题是：

1. **Global Representation Bottleneck 是否真实存在？**
2. **Relation-Specific Decision Conflict 是否真实存在？**
3. 在统一 assumed-negative 训练协议下，新 Teacher 是否能明显改善自然候选排序，并缩小与 B13 的差距？

本轮不讨论 Stage2 value recovery / join execution。

---

# 1. R18 标签协议：unknown-as-negative

训练时统一解释：

```text
known positive              -> positive
all other valid candidates  -> assumed_negative
```

其中 `assumed_negative` 不是新的人工 verified negative 标签，而只是本轮训练目标中的负向 competitor。

允许：

- assumed negative 进入 listwise softmax denominator；
- assumed negative 接受降低 score 的梯度；
- 使用同一 assumed-negative 规则训练 A0/A1/A2。

禁止：

- 为 assumed negative 额外增加人工权重；
- 扫 positive/negative weight；
- PU-learning correction；
- hard-negative confidence weighting；
- 用 test qrel 给 train unknown 重标；
- 从测试集信息构造负例；
- 为某个结构单独改变 negative recipe。

若训练数据中已经明确存在的 positive 因某个局部 list/qrel 文件遗漏而落入 unknown，必须在 **train-side 已知正标签** 范围内去重保护；不得使用 test GT 进行这种保护。

本轮不再要求每个 list 必须包含 verified negative。`verified_negative_count=0` 不再触发 stop gate。

---

# 2. Input Contract / Preflight（Codex 必须首先执行）

## 2.1 总原则

在修改源码、创建新 feature、启动 GPU、训练任何模型之前，Codex 必须先执行输入预检，并生成：

```text
work/stage1_optimization_r18_20260910/INPUT_MANIFEST.json
work/stage1_optimization_r18_20260910/PREFLIGHT_AUDIT.md
```

Preflight 必须检查：

- 路径是否存在；
- 文件/目录大小；
- SHA256（对于下面已经冻结 hash 的输入必须核对）；
- checkpoint 能否 strict load；
- checkpoint architecture/config；
- feature dimension；
- object 数量及 ID coverage；
- train/dev list 数量；
- candidate-pool query 数量；
- query ID / target ID 与 feature/object store 的覆盖情况；
- 所有实验臂是否能读取**同一份**候选与监督输入。

### 严格禁止自动补救

如果任何 required input 缺失、hash 不符或 schema 不兼容：

> **停止训练并报告 `blocked_by_missing_or_mismatched_input`。**

禁止 Codex 自动：

- 从 Hugging Face 重新下载 Qwen；
- 重新编码 Qwen3-VL features；
- 重新生成 R16 candidate pools；
- 重新跑 Student ANN；
- 重新构造 edge supervision；
- 用相邻 R10/R11/R12/R13/R16 文件“猜一个替代品”；
- 在 `teacher_edge.pt` 缺失时自行改成 `teacher_path.pt`；
- 因 hash 不一致而悄悄覆盖旧文件。

若存在同名但不同 hash 的输入，只记录并停止，由人工决定后续处理。

---

## 2.2 必需输入：Teacher 初始化

### A0 / A1 / A2 唯一共同父 checkpoint

```text
/home/oycy/MMDD/work/stage1_optimization_r11_20260908/
  taskC_clean/teacher/teacher_edge.pt
```

冻结 SHA256：

```text
fd544cc166f2a24f2c55a040c0a2645bc14b22c3c82b49176823db458fe75bc1
```

文件大小（R17 manifest）：

```text
72,615,817 bytes
```

**A0、A1、A2 必须全部从这个 `teacher_edge.pt` 初始化。**

不要自行使用：

```text
teacher_path.pt
```

即使它存在，也不是本轮结构因果对照的共同初始化。

同时读取：

```text
/home/oycy/MMDD/work/stage1_optimization_r11_20260908/
  taskC_clean/teacher/manifest.json
/home/oycy/MMDD/work/stage1_optimization_r11_20260908/
  taskC_clean/teacher/teacher_edge.pt.history.json
```

对应冻结 SHA256：

```text
manifest.json:
932c20e941e69814659cd44b42d9b344c1c06affbb5a869740dae9546edaacb6

teacher_edge.pt.history.json:
4ba9a3703fbb472d5832392e9ea11091f68b1eecdd9946f4cd5558052af182c1
```

这些文件用于恢复已有 Teacher architecture/config 和历史训练上下文，不用于改变本轮训练预算。

---

## 2.3 必需输入：Teacher train / dev supervision

训练监督固定使用 **R12 corrected supervision**：

```text
/home/oycy/MMDD/work/stage1_optimization_r12_20260908/
  taskA_correctness/supervision/edge_lists.train_fit.jsonl

/home/oycy/MMDD/work/stage1_optimization_r12_20260908/
  taskA_correctness/supervision/edge_lists.dev.jsonl
```

冻结 SHA256：

```text
edge_lists.train_fit.jsonl:
5be5e3aee605397c80b4bb43867e57d02d5e65147b150dc1275375946e543158

edge_lists.dev.jsonl:
d686c35d435631149a62b4f228c7d82c7ed6f5f5d42230149a11524bbb712de6
```

R18 对这些文件只改变**训练时标签解释**：

```text
positive_ids / 已知 positive -> positive
其余有效 candidate          -> assumed_negative
```

不得回退到 R11 withdrawn `label=0` 文件。

不得因为 `confirmed_labels=None` 再次要求人工 verified negative 后才能训练。

Codex 必须在 `INPUT_MANIFEST.json` 中记录：

- train lists 数；
- dev lists 数；
- 五种 relation 的 list 数；
- 每种 relation 的 positive 数；
- assumed-negative candidate 数；
- 因 known-positive protection 被从 assumed-negative 中排除的 candidate 数。

---

## 2.4 必需输入：Frozen Qwen Stage1 features

固定读取：

```text
/home/oycy/MMDD/work/stage1_optimization_r10_20260907/
  features_qwen3_vl_embedding_8b
```

R17 manifest 冻结 SHA256：

```text
c32099430feca4dae5d2f8fbbae60f965e3b0353fdd62c62a7bc929ba24216e1
```

以及 objects：

```text
/home/oycy/MMDD/work/stage1_optimization_r10_20260907/
  stage1_data/stage1_objects.jsonl
```

SHA256：

```text
75f2789bfc62483c85dcecb3213ac6bd3f11b3952e15bba626a3990297e71e00
```

### 非常重要

R18 不应重新运行 Qwen3-VL-Embedding-8B。

Global Residual 直接从已有 feature store 读取 Qwen final object embedding `z_x`；Teacher local token path 继续使用已有 hidden-state / grouped-token feature。

若现有 feature store 中无法取得 A1 所需 `z_x`：

> 标记 `blocked_by_missing_global_feature`，停止 A1；不要自动重新下载模型或重编码全数据湖。

同时检查 `z_x` 的实际维度并写入 manifest。不要把文档中的理论维度写死后不验证实际 tensor shape。

---

## 2.5 必需输入：自然候选与公平评测池

固定使用 R16 冻结 candidate pools：

```text
/home/oycy/MMDD/work/stage1_optimization_r16_20260910/
  candidate_pools.jsonl.gz
```

冻结 SHA256：

```text
4186b5bdd436a14fa61b83c3c6127507c075fd6f16804c5cdb2d4a06e85a01d1
```

同时保留用于历史核对的：

```text
teacher_pair_scores_qt.jsonl.gz
teacher_pair_scores_qe.jsonl.gz
teacher_pair_scores_et.jsonl.gz
teacher_path_aggregation_per_query.jsonl.gz
```

R17 manifest SHA256：

```text
QT:
b6ff7143301dc89cfdeb22c68ac565073c68c982cea32ebd69393cbb9ae78d51

QE:
dbf16c94d4aaad85e48d7c215f4017d77e1fadfb2ad969744a55ebfc0a2a2d71

ET:
448a953ae07efc5ada5ae9a2d17de9b1046a551ac6998c72f92a77f0d0ab9673

Path aggregation:
eebf8f467e14a7adeb50b16fe7998af8d6e7168c0c17c43235e32e3111f60891
```

所有 A0/A1/A2 必须使用完全相同的 target IDs。

不得：

- 为 A1/A2 补 GT target；
- 一个 arm 用 natural union，另一个用 direct pool；
- 因某个新 Teacher 效果差改变 `M_q`；
- 重新运行 ANN 后用一个新的 candidate pool；
- 按 implicit/explicit GT 在线路由不同模型。

---

## 2.6 Teacher extra feature/cache

若 R16 natural union 中的 destination/evidence 对象不全在主 feature store 的现有 Teacher-ready cache 中，优先复用以下已存在目录：

```text
/home/oycy/MMDD/work/stage1_optimization_r12_20260908/
  taskC_training/teacher_extra

/home/oycy/MMDD/work/stage1_optimization_r16_20260910/
  teacher_extra_matched_gpu0
  teacher_extra_matched_gpu1
  teacher_extra_edges_gpu0
  teacher_extra_edges_gpu1
```

不得在没有记录 cache lineage 的情况下把旧 Teacher 参数产生的 **compressed-token cache** 直接复用于新结构训练。

区分：

```text
frozen Qwen feature cache          -> 可以跨 A0/A1/A2 复用
Teacher parameter-dependent cache  -> 必须按 checkpoint / arm/version 隔离或重新计算
```

若某 cache 是参数依赖型但 provenance 不清楚，停止使用它并报告；不要猜。

---

## 2.7 B13 Student reference

Student 固定为：

```text
B13 / S-full13@178
```

本轮：

- 不训练 Student；
- 不做 KD；
- 不做 residual Student；
- 不改 P/R；
- 不改 ANN index；
- 不改变 Student QE/ET retrieval recipe。

若只需要已有 B13 reference metric / frozen ranking，则直接读取已有 R13/R16 产物，不应重新训练 Student。

现有 R13 路径池位置可从源码确认：

```text
/home/oycy/MMDD/work/stage1_optimization_r13_20260909/
  taskD_witness_supervision/p_s_target_only/evaluation_step178/path_pool.jsonl.gz
```

如果某项 R18 分析确实需要重新运行 B13 checkpoint，而路径未能从已有 manifest / R13 产物中可靠解析：

> 先报告 `blocked_by_missing_b13_checkpoint_reference`，不得自行猜 checkpoint 文件。

**不要为了结构实验重新跑 Student。**

---

## 2.8 Preflight 最终 gate

必须输出一张表：

| Input | Required for | Exists | SHA match | Schema/dim check | Status |
|---|---|---:|---:|---:|---|

只有以下关键输入均 PASS 才能启动 A0/A1：

1. `teacher_edge.pt`；
2. Teacher manifest；
3. R12 train/dev supervision；
4. R10 frozen features；
5. stage1 objects；
6. R16 candidate pools；
7. evaluation 所需 qrel/query grouping；
8. A1 所需 final object embedding `z_x`。

若 A2 被触发，还必须确认旧 shared scoring head 能被**逐参数完整复制**到五个 relation heads。

---

# 3. Student 与候选池全部冻结

继续冻结 B13 / S-full13@178。

Stage1 流程仍然是：

```text
Student ANN / evidence retrieval
        ↓
frozen candidate pool U_q
        ↓
Teacher reranking / diagnosis
```

Teacher 不承担全湖 ANN。

本轮 Teacher 的目标不是取代 Student ANN，而是首先证明：

> 在 Student 提供的自然候选分布中，它能成为更可靠的 pairwise / relation-aware reranker。

---

# 4. A0：原 Teacher 同预算 continuation baseline

A0 使用当前原始 Teacher architecture，不增加 global branch，不拆 relation head。

从统一父 checkpoint：

```text
teacher_edge.pt
```

继续训练。

A0 的存在是为了隔离：

> “新结构收益” vs “只是换 assumed-negative 协议并继续训练更多 step 的收益”。

因此 A1/A2 的因果比较都必须相对 A0。

固定：

- checkpoint；
- train/dev lists；
- assumed-negative rule；
- loss；
- optimizer；
- LR；
- batch size；
- update budget；
- seed；
- gradient clipping；
- scheduler；
- checkpoint selection rule；
- evaluation candidates。

不得给 A1/A2 更多 step 或更有利的 checkpoint selection。

---

# 5. A1：Global Representation Residual Bridge（本轮最高优先结构实验）

## 5.1 假说

当前 Teacher：

```math
H_x \rightarrow C_\tau(H_x) \rightarrow RelationTransformer \rightarrow scorer
```

没有显式保留 Student 已经证明对 retrieval 有效的 Qwen final object embedding：

```math
z_x = QwenEmbedding(x).
```

假说 H1：

> compressed token representation 没有充分保留 final object embedding 中有利于自然 retrieval/ranking 的 global semantic signal。

## 5.2 结构

原 Teacher local pair representation：

```math
v_{a,b}^{local}.
```

新增：

```math
g_x = LN(G_{\tau(x)} z_x).
```

构造：

```math
r_g = [g_a, g_b, g_a \odot g_b, |g_a-g_b|, e_{\tau(a),\tau(b)}].
```

残差：

```math
\Delta v = W_2 GELU(W_1 r_g).
```

最终：

```math
v_{a,b}=v_{a,b}^{local}+\Delta v.
```

## 5.3 step0 function equivalence

必须初始化：

```text
W2.weight = 0
W2.bias   = 0
```

从而保证：

```math
s_{A1}^{(0)}(a,b)=s_{A0}^{(0)}(a,b)
```

在启动训练前必须做不少于五类 relation 的 replay，并报告：

- max absolute score difference；
- mean absolute score difference；
- ranking consistency；
- batch/single；
- cached/uncached。

若 step0 不等价：

> 标记 implementation failure，先修实现，不能进入结构效果分析。

## 5.4 方法身份

- Student ANN 完全不变；
- 不改变候选生成；
- 不增加 hop；
- 不解冻 Qwen；
- 仅改变 Teacher reranker；
- Qwen frozen feature 可以离线复用；
- 新增参数量和 online pair scoring latency 必须记录。

---

# 6. A2：Relation-Specific Scoring Heads

## 6.1 假说

Teacher 已有：

- table/text/image adapter；
- modality embedding；
- role embedding；
- type-pair embedding。

所以 H2 不是“Teacher 不知道模态”。

真正假说是：

> shared final decision head 需要同时处理 TT、T→text、T→image、text→T、image→T，不同 relation 的决策边界发生冲突，type-pair conditioning 不足以消除冲突。

## 6.2 最小改动

保留 shared Relation Transformer，只将最终 shared scoring head：

```math
h(v)
```

改为五个有向 head：

```math
h_{T\to T},
 h_{T\to text},
 h_{T\to image},
 h_{text\to T},
 h_{image\to T}.
```

五个 head 必须从旧 shared head **完整复制**，不能随机初始化。

因此 step0：

```math
h_r^{(0)} = h_{shared}
```

所有 relation 的输出应与 A0 相同。

同样执行 step0 replay。

## 6.3 执行顺序

默认先运行：

```text
A0 + A1
```

A2 在以下任一情形触发：

- A1 没有明显改善；
- A1 有改善但仍明显弱于 B13；
- A1 只改善 TT，而 QE/ET relation 明显仍弱；
- A1 结果提示 representation 有价值，但不能解释 relation-specific edge 问题。

A2 仍与同一个 A0 比较，不重新定义 baseline。

本轮不运行：

```text
Global + Relation Heads
```

避免无法区分两种结构贡献。

---

# 7. Training Protocol

A0 / A1 / A2 必须完全一致：

```text
parent checkpoint
train data
train query set
dev query set
positive IDs
assumed-negative candidate set
batch ordering / shuffle seed
list truncation rule
loss
optimizer
LR
scheduler
batch size
gradient accumulation
gradient clipping
update budget
random seed
checkpoint evaluation steps
selection rule
```

默认只有一个固定 seed 用于主结构因果判断；若主结果接近 gate 边界，再做少量预先规定的 seed replication，不得不断增加 seed 直到获得显著结果。

禁止 learning-rate / hidden-size / activation / depth sweep。

---

# 8. Training smoke test：不再以 verified negative 为 gate

训练前对每种 relation 抽小批 train lists。

检查：

- loss 可计算且有限；
- positive score / positive-minus-assumed-negative margin 是否出现学习趋势；
- 预期 module 获得梯度；
- optimizer 覆盖新增参数；
- 参数确实更新；
- train/eval 状态正确；
- parameter-dependent cache 没有 stale reuse；
- A1 global branch 梯度实际非零；
- A2 对应 relation head 实际更新。

但：

> assumed-negative tiny set 上没有很高的 R@1，不再自动阻塞完整结构实验。

只有以下 correctness 问题才阻塞：

- loss NaN/Inf；
- 无梯度；
- optimizer 漏参数；
- checkpoint 加载错误；
- relation mapping 错；
- step0 equivalence 失败；
- candidate/feature ID 对不上；
- cache lineage 错；
- train/test leakage。

---

# 9. 主评测协议

主指标仍是 query-macro target Recall：

```math
Recall@K = \frac{1}{|\mathcal Q|}\sum_q
\frac{|G_q\cap TopK(q)|}{|G_q|}.
```

固定报告：

- Recall@10 —— 主端点；
- Recall@20；
- CandidateRecall@50；
- implicit；
- explicit；
- source-group paired bootstrap；
- per-query W/L/T。

候选池相同前提下，Teacher reranking 不能增加候选池本身的 Recall@N；因此必须区分 candidate admission 与 ranking。

---

# 10. A1/A0 主因果比较

核心：

```math
A1(U)-A0(U).
```

同时报告：

```text
A1 vs B13
A0 vs B13
A1/A0 on direct100
A1/A0 on matched-direct-M
```

## 支持 H1

若：

- A1 显著优于 A0；
- 自然 U 上 R@10/R@20 改善；
- TT direct100 内排序也改善；
- 不是只靠某几个 query；

则支持：

> 显式保留 Qwen final global representation 对 Teacher ranking 有价值。

不能自动写成：

> compression 是唯一根因。

## A1 ≈ A0

说明：

> 简单缺失 final global embedding 不是当前主解释，或者 residual bridge 形式不足以利用它。

不要立刻扫 global hidden width / activation / LR。

## A1 > A0 但仍远弱于 B13

说明 H1 可能是部分瓶颈，但 Teacher 仍未成为可靠上界；此时触发 A2 比较。

---

# 11. A2/A0 主因果比较

核心：

```math
A2(U)-A0(U).
```

同时拆分五种 relation 的诊断。

若 TT 改善但 QE/ET 不改善：

> shared scorer 对 TT 可能存在冲突，但 evidence edge 仍有独立问题。

若跨模态 relation 改善而 TT 基本不变：

> relation specialization 主要帮助 edge scorer，不能解释 R16 QT 主失败。

若全部改善：

> 支持 relation-specific decision boundary 假说。

若 A2≈A0：

> type-pair conditioning + shared head 的决策冲突不是当前主要瓶颈，至少五-head最小干预没有证据。

---

# 12. Evidence / Candidate Source 仍需独立报告

对每个可用 Teacher，用**同一个 Teacher**比较：

```text
Natural Union U
vs
MatchedDirectM
```

必须报告 positive target 来源：

```text
both
U-only
M-only
neither
```

重点追踪：

```text
outside ANN direct100
outside exact direct100
outside matched-direct-M
evidence introduced
rank in U
C50 retained
Top20 retained
Top10 retained
```

不能把 raw union candidate recall 的扩大直接称为 Teacher 或 evidence ranking 成功。

本轮仍优先关注：

> evidence 独有 target 是否在新 Teacher 下更容易从候选池中被保留到 Top10 / Top20 / C50。

---

# 13. Stop Rules

## A1 已经足够强

如果 A1：

- 明显优于 A0；
- natural U 上达到接近 B13 的合理水平；
- direct100 内不再出现巨大退化；

则可以不运行 A2，把 R18 结论聚焦于 global representation path。

## A1 只有部分改善

运行 A2，检验竞争解释。

## A1/A2 均无明显改善

停止结构堆叠。

本轮不继续扫：

- width；
- depth；
- latent count；
- activation；
- LR；
- Global+Heads；
- 五套 Transformer。

下一轮才重新考虑：

- natural candidate training distribution；
- Teacher supervision/objective；
- pairwise compressed-token Teacher 的方法身份；
- evidence-conditioned joint verifier。

---

# 14. 本轮明确不做的内容

R18 不运行：

- Stage2；
- Original / NoEvidence / WrongEvidence；
- value recovery；
- correct join；
- Qwen3.5-9B；
- Stage2 coverage；
- Student KD；
- Student residual；
- weighted RRF；
- fixed direct/evidence weight sweep；
- implicit/explicit GT 在线路由；
- 新 hop；
- backbone finetuning；
- full lake row/column index。

Stage2 相关 R17 问题暂时记录，不作为 R18 gate。

---

# 15. 必须输出的实验产物

至少生成：

```text
work/stage1_optimization_r18_20260910/
  PLAN_FROZEN.json
  INPUT_MANIFEST.json
  PREFLIGHT_AUDIT.md
  COMPLETION_AUDIT.json
  FAILURE_NOTES.md
  CODE_HASH_MANIFEST.json

  training_protocol.json
  label_semantics_audit.json
  assumed_negative_stats.json

  A0_base_continuation/
    config.json
    step0_replay.json
    train_history.jsonl
    checkpoint_*.pt
    selected_checkpoint.json
    natural_union_rankings.jsonl.gz
    direct100_rankings.jsonl.gz
    matched_direct_M_rankings.jsonl.gz
    metrics.json

  A1_global_residual/
    ...same schema...
    function_equivalence.json
    parameter_count.json

  A2_relation_heads/        # only if triggered
    ...same schema...
    function_equivalence.json
    parameter_count.json

  PAIRED_COMPARISON_A1_A0.json
  PAIRED_COMPARISON_A2_A0.json   # if triggered
  CANDIDATE_SOURCE_ANALYSIS.json
  RESULTS.md
```

不得只保存聚合 RESULTS.md。

必须保存 per-query ranking / metric，使后续可以独立复算。

---

# 16. Completion Audit

最终明确写出每个阶段：

```text
complete
failed
blocked
not_triggered
partial
```

例如：

- A2 因 A1 已满足 stop rule 而没跑 -> `not_triggered`；
- feature 缺失导致 A1 不能运行 -> `blocked`；
- step0 equivalence 失败 -> `failed_correctness`；
- 训练完成但指标差 -> `complete_scientific_negative`。

不得把 `not_triggered` 写成负结果。

---

# 17. R18 最终需要回答的问题

按证据顺序回答：

1. assumed-negative 训练协议下，A0 本身是否比旧 frozen Teacher 明显改善？
2. A1 是否在同训练预算下显著优于 A0？
3. Global representation 是否尤其修复了 TT / natural ranking？
4. 若触发 A2，relation-specific heads 是否显著优于 A0？
5. 哪一种结构真正缩小了 Teacher 与 B13 的差距？
6. 新 Teacher 对 evidence-introduced / matched-budget-outside positive target 的 retention 是否更好？
7. 当前第一优先瓶颈应判断为：global representation、relation decision conflict，还是两者都不足以解释？
8. 下一轮是否有资格进入 natural-candidate training / KD / 更高级 Teacher，而不是继续无目的结构扫描？

最后分别列出：

```text
已经可以支持
暂时不能支持
被当前实验削弱
仍然 unknown
```

不要为了论文叙事把部分改善写成根因证明。

---

# 18. Codex 执行前最后检查清单

在真正启动训练前，stdout 和 `PREFLIGHT_AUDIT.md` 必须明确打印：

```text
[ ] MMDD root = /home/oycy/MMDD
[ ] teacher_edge.pt exists + SHA matches
[ ] teacher_edge.pt strict-load succeeds
[ ] R12 train supervision exists + SHA matches
[ ] R12 dev supervision exists + SHA matches
[ ] R10 frozen feature store exists + expected ID coverage
[ ] z_x final object embedding exists for A1
[ ] stage1_objects.jsonl exists + SHA matches
[ ] R16 candidate_pools exists + SHA matches
[ ] evaluation qrels/query groups resolved
[ ] A0/A1 use identical parent checkpoint
[ ] A0/A1 use identical train lists and assumed-negative interpretation
[ ] A0/A1 use identical optimizer/LR/update budget/seed
[ ] A1 step0 score function is equivalent to A0
[ ] no Qwen model download/re-encoding scheduled
[ ] no Student retraining scheduled
[ ] no Stage2 job scheduled
```

只有全部 required 项通过，才开始 A0/A1。

---

## 一句话执行目标

> **在完全冻结 Student、候选池和 Qwen features 的前提下，把 unknown 统一作为 assumed negative，用相同训练预算比较原 Teacher continuation 与 zero-init Global Residual；必要时再单独比较 copied-init Relation-Specific Heads，以判断当前 Teacher 主失败究竟是否来自 global representation 丢失或 relation-specific decision conflict。**
