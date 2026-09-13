# MMDD R21：强 Teacher 的知识迁移与全湖 Candidate Admission

日期：2026-09-11  
版本：**Qwen-Raw baseline revision**  
状态：待实现/执行的冻结方案草案。本文件仅针对 Stage1；不启动 Stage2，不自动执行 T2。

> 本版相对上一版 R21 **只新增 Qwen-Raw 作为正式 read-only baseline 及相应评估**。主训练矩阵仍然只有 `Ssup × 2 seeds` 与 `SKD × 2 seeds`，共 4 个训练任务；不因为新增 baseline 增加训练臂。

---

## 0. 本轮决策：为什么现在进入 Student，而不是继续只训 Teacher

R20 当前固定终点的核心结果：

| 模型 | U R@10 | U R@20 | U CR@50 |
|---|---:|---:|---:|
| B13 Student | 29.01% | 35.08% | 44.91% |
| C3 parent Teacher | 41.04% | 50.28% | 61.97% |
| D0 old-list continuation | 36.72% | 46.78% | 59.73% |
| D1 old-pool resampling | 37.16% | 47.17% | 59.51% |
| **D2 current-Teacher remining** | **46.62%** | **55.72%** | **65.41%** |

D2−D1 在 U R@10 上约 +9.46pp，两个 seed 同向；因此一次 current-Teacher remining 的确有效，不能解释成“仅仅继续训练”或“只在旧池里随便换样”。

与此同时：

```text
D2 U R@10       ≈ 46.62%
D2 U CR@50      ≈ 65.41%
RawUnionRecall  ≈ 69.67%
```

所以固定 U 内仍有明显 Top10 ranking 空间，但：

```text
100% - RawUnionRecall ≈ 30.33pp
```

的 GT 根本没有进入当前 U；而固定 U 内从 R@10 到 RawUnionRecall 的理论空间约 23.05pp。Candidate admission 已经成为最大的单一缺口。

因此本轮主线保持：

```text
T* = 当前可靠 D2 final Teacher
→ 训练新的 Student
→ 检查 Teacher knowledge transfer
→ 检查 full-lake candidate admission
→ 固定 T* 比较不同 candidate generator
→ 再决定是否值得进入 Student→Teacher feedback
```

### 0.1 本版新增的 Qwen-Raw baseline

本轮额外正式加入：

```text
Qwen-Raw
```

回答一个基础但重要的问题：

> 在完全相同的对象构造、同一个 Qwen3-VL-Embedding-8B backbone、同一个数据湖和同一检索预算下，经过训练的 Student 是否真的比 pretrained Qwen embedding 原生相似度更适合 Join Discovery？

因此最终应形成下面这条解释链：

```text
Qwen-Raw
   ↓
B13 Student
   ↓
Ssup
   ↓
SKD
```

分别对应：

```text
预训练多模态语义本身
→ 原 Student 学习带来的增益
→ 更充分的 natural-candidate supervised training
→ 强 Teacher 的额外 knowledge transfer
```

**Qwen-Raw 只增加评估，不增加训练任务。**

---

## 1. 不变边界

保持：

- Query-by-example；
- 冻结 Qwen3-VL-Embedding-8B backbone；
- Student 保持可 ANN 的独立对象编码/打分结构；
- 支持 direct `Q→T`；
- 支持 evidence `Q→E→T`；
- 不增加更多 hop；
- 不输入 GT join column；
- 不按 GT implicit/explicit 在线路由；
- 不增加 value index / Stage2 reranker；
- unknown 继续作为 assumed negative；
- 保护 train 中所有已经知道的 positive；
- 不做 weighted RRF / evidence 超低固定权重退化成 direct-only；
- 不扫描 Global width、relation-specific head、fusion、64/128/256 list length。

本轮允许固定 T* 给训练候选打 QT soft scores，用于 Student KD；但不自动训练 T2，不自动执行多轮 Teacher–Student bootstrapping。

---

## 2. Qwen-Raw 的严格定义

### 2.1 什么叫 Qwen-Raw

设系统当前缓存的、冻结 Qwen3-VL-Embedding-8B 对对象 `x` 产生的 final object embedding 为：

```math
z_x.
```

`Qwen-Raw` 不经过任何 Student 可训练组件：

- 不经过 learned projection `P`；
- 不经过 relation matrix / bilinear scorer `R`；
- 不经过 Student MLP；
- 不使用 Teacher；
- 不做额外训练。

定义：

```math
s_raw(a,b) = cos(z_a, z_b).
```

实现时：

```text
如果缓存 z 已经 L2-normalized：
    inner product == cosine
否则：
    对 raw z 一次性 L2 normalize 后建 IP index
```

必须数值验证 ANN index 内积和直接 cosine 的一致性。

### 2.2 为什么这样定义

Qwen-Raw 必须和 Student 使用：

- 同一个 Qwen checkpoint；
- 同一个 prompt / processor；
- 同一个对象序列化；
- 同一套 cached final object embedding；
- 同一 train/dev/test split；
- 同一个数据湖；
- 同一 candidate budget。

唯一变化是：

```text
Qwen-Raw：直接比较 pretrained z
Student：在相同 z 上学习 task-specific retrieval scoring
```

这样才能公平回答“训练创造了什么”。

### 2.3 Relation 处理

Qwen-Raw **不人为增加 relation-specific 参数**。对：

```text
Q→T
Q→text
Q→image
text→T
image→T
```

统一使用 raw cosine。

如果现有 Qwen feature cache 对不同 object role 使用不同固定 prompt / role serialization，则**原样沿用现有缓存**；不要为了 Qwen-Raw baseline 重新设计 prompt，也不要专门去除共享 prompt bias。Qwen-Raw 的目标就是测量当前 backbone 原生 retrieval baseline。

同时要把 prompt / role / pooling / normalization fingerprint 写入 baseline manifest，避免之后误把另一版 raw embedding 当作同一 baseline。

---

## 3. 服务器输入与模型身份

### 3.1 Teacher T*

默认服务器 root：`/home/oycy/MMDD`（允许显式改挂载前缀，但相对路径与内容 hash 必须一致）。使用 R20 D2 fixed final：

| lineage | T* checkpoint（相对 root） | SHA256 |
|---|---|---|
| 13 | `work/stage1_optimization_r20_20260911/D2/seed13/checkpoints/step_021072.pt` | `792c746b79dc8e61b80be145d20f118fe2dacc580fac829ab093b06aa20e5164` |
| 29 | `work/stage1_optimization_r20_20260911/D2/seed29/checkpoints/step_021072.pt` | `3e8e497129927a9921d5f44f28504483420d076c414675b0a9f4704e8572a0a3` |

不使用中途 best checkpoint；不只取表现更好的 seed；不把两个 Teacher logits 平均成新 ensemble。

### 3.2 Student S0 与冻结输入

```text
S0 = B13
```

优先定位以下既有 artifact；实际执行前全部重新计算 hash 并写入 `INPUT_MANIFEST.json`，不能因为路径存在就默认版本正确：

| 作用 | 相对 root 路径 / 说明 |
|---|---|
| B13 checkpoint | `work/stage1_optimization_r13_20260909/taskD_witness_supervision/p_s_target_only/checkpoints/step_000178.pt` |
| B13 原索引 | 同目录上一级 `evaluation_step178/index` |
| B13 原候选/排名 | `evaluation_step178/path_pool.jsonl.gz`、`rankings.jsonl.gz` |
| frozen Qwen feature | `work/stage1_optimization_r10_20260907/features_qwen3_vl_embedding_8b` |
| objects | `work/stage1_optimization_r10_20260907/stage1_data/stage1_objects.jsonl` |
| split | `work/stage1_optimization_r10_20260907/taskA_protocol/splits.json` |
| train-known positive source | `work/stage1_optimization_r12_20260908/taskA_correctness/supervision/edge_lists.train_fit.jsonl` |
| edge dev | 同目录 `edge_lists.dev.jsonl` |
| 旧固定 U/M/D pool | `work/stage1_optimization_r16_20260910/candidate_pools.jsonl.gz`，已知 SHA256 `4186b5bdd436a14fa61b83c3c6127507c075fd6f16804c5cdb2d4a06e85a01d1` |
| R20 训练骨架 | `work/stage1_optimization_r20_20260911/train_manifest_D0.jsonl`，已知 SHA256 `ed32939668c9b29ba8837e10891f2aa9deec3bdcd3f204c0aa104865beddb025` |
| TT natural membership | R20 `refreshed_reservoir_lineage13.jsonl.gz` / `lineage29`；只复用 candidate membership |

Teacher 需要的 R12/R16 extra features 也必须按真实服务器路径核验覆盖，不能只加载 base embedding 让 Global/Local 分支静默退化。

Ssup / SKD 同一 seed 必须从完全相同的 B13 权重启动。

### 3.3 Qwen-Raw

从 B13 / Stage1 已经使用的 frozen Qwen feature cache 中读取 **Student 输入之前的 raw final object embedding**。

Preflight 必须输出：

```text
raw_feature_root
backbone_id
processor/tokenizer id
prompt / role configuration
pooling definition
raw dimension
raw dtype
normalization status
feature manifest hash
object manifest hash
```

禁止从 Student checkpoint 中倒推出所谓“raw embedding”；必须直接定位真正冻结的 backbone feature artifact。

---

## 4. G0：Correctness / Reproducibility Preflight

在所有新训练或 ANN 评估之前完成。

### 4.1 Student / Teacher 正确性

1. 精确加载 B13 和两个 D2 final；核验 checkpoint hash、state keys/shapes、trainable mask。
2. Teacher 必须使用 Global Residual 版本的完整 `score_pairs`；不能走只使用 compressed tokens 的旧 generic scorer。
3. 对确定性抽样 query/pairs，与 R20 D2 保存的 raw logits 做 score parity。
4. 核验 split、known-positive protection、candidate 去重、score sort direction、cache identity。
5. 验证 Student score 与其 ANN query/target vector construction 数值等价。
6. 冻结完整 R21 源码 hash / commit、环境、CUDA、dtype、ANN 参数。

### 4.2 Qwen-Raw 正确性

额外增加：

1. 随机/确定性抽样 512 个 object pairs，验证：

```text
直接 numpy/torch cosine
≈
Qwen-Raw ANN query vector · index target vector
```

2. 报告 max / mean absolute score difference；
3. 验证 Qwen-Raw 不加载任何 Student trainable weight；
4. 验证 direct、QE、ET 三类 index 都来自同一 raw feature版本；
5. raw index 的 corpus object IDs 与 Student index corpus IDs 完全一致；
6. 不允许因为 Qwen-Raw 不支持某个 relation 就缩小 corpus 或改变 budget。

### 4.3 R20 历史审计缺口

R20 execution source hash、exact Direct100、旧 A1 reservoir 等缺口继续记录为 historical audit partial；它们不应被默默标成“已验证”，也不自动阻断 R21，只要本轮必要 checkpoint / feature / split / score parity 全部通过。

---

## 5. G1：冻结 Student 训练候选与 Teacher soft targets

### 5.1 训练 pool

Student 训练仍复用 R20/B13 natural candidate universe：

```math
C_train(q) = C_B13,natural(q) ∪ G_train,known(q).
```

其中：

- `C_B13,natural(q)`：B13 ANN 生成的较大 natural candidate pool；
- `G_train,known(q)`：train-only known positives；
- unknown 其余项继续作为 assumed negative；
- 不使用 dev/test qrel 修正训练。

TT 使用完整 natural pool 做 Student training/KD，不截成 hard32；其他四个 relation 保持与原训练骨架一致。

### 5.2 Teacher soft target

**不能复用 R20 C3 mining cache 里的 scorer logits。**

必须使用对应 lineage 的 **D2 final T*** 对冻结后的训练 TT candidate list 重新打 QT raw logits。

缓存 identity 至少绑定：

```text
Teacher checkpoint hash
Teacher scorer code hash
feature manifest hash
query id
relation
ordered candidate-list hash
dtype / score-space
```

---

## 6. 主训练矩阵：仍然只有 4 个任务

| 模型 | 初始化 | 是否训练 | Training list | Loss | 角色 |
|---|---|---:|---|---|---|
| **Qwen-Raw** | frozen Qwen raw embedding | 否 | 无 | 无 | pretrained retrieval baseline |
| **B13 / S0** | existing B13 | 否 | 无 | 无 | 已有 Student baseline |
| **Ssup** | B13 | 是 | natural full pool | supervised listwise | 排除额外训练/更大自然池的贡献 |
| **SKD** | B13 | 是 | 与 Ssup 完全相同 | supervised + QT listwise KD | 检验 Teacher knowledge transfer |

训练任务只执行：

```text
Ssup lineage13
Ssup lineage29
SKD lineage13
SKD lineage29
```

Qwen-Raw 和 B13 均为 read-only baseline。

### 6.1 Loss

监督项：

```math
L_sup(q)
= logsumexp_{t∈C(q)} s_S(q,t)
- logsumexp_{t∈G_train,known(q)} s_S(q,t).
```

Teacher KD：

```math
P_T(t|q)=softmax(s_T(q,t)/τ),
P_S(t|q)=softmax(s_S(q,t)/τ),
```

```math
L_KD(q)=τ^2 KL(P_T || P_S).
```

首轮冻结：

```text
τ = 1
λ_KD = 1
```

不做 temperature / λ grid search。

### 6.2 Budget 与 optimizer

Ssup / SKD：

- 同一 Student 初始化；
- 同一 trainable mask；
- 同一 optimizer policy；
- 同一 batch-list顺序；
- 同一 candidate lists；
- 同一 update 数；
- 同一最终 step；
- fixed final checkpoint 为主结果，不使用 dev-best 选点。

保持上一版 R21 的固定训练预算：

```text
42143 个逻辑训练列表
logical batch = 64
2 epochs
每 epoch ceil(42143 / 64) = 659 updates
总计 1318 updates
mid = local step 659
final = local step 1318
```

完整 variable-length TT list 是一个 listwise softmax 单元，不能把一个长 list 切成若干独立局部 softmax。若显存不足，只能用保持数学等价的 microbatch / gradient accumulation，并按实际有效 list 数正确加权。

两臂从 B13 权重开始新的 Student 训练，默认 fresh AdamW；学习率与 param-group 比例优先从 B13 实际 resolved config 恢复。如果服务器 artifact 无法恢复原 LR，必须在任何正式训练前冻结共同 fallback（上一版建议 `lr=1e-5, weight_decay=.01`），不能根据 dev 结果回调。无 scheduler，不新增某一臂专属 clipping / regularizer。

首轮仍优先使用方案指定的 A100；如果实际只能使用别的 GPU，作为 execution deviation 记录，不能静默改变 batch/list/model 协议。

---

## 7. Evaluation Layer 1：固定旧候选池，验证 Student knowledge transfer

这一层仍只评估 Student scorer 本身：

```text
B13
Ssup
SKD
```

在完全相同的旧冻结 B13 U/M/D100 candidate membership 上重新打分，报告：

```text
R@10
R@20
CR@50
implicit / explicit
per-query W/L/T
source-group paired bootstrap
```

主因果比较：

```text
Ssup - B13
SKD - Ssup
SKD - B13
```

解释：

- `Ssup−B13`：natural full-pool supervised continuation 的收益；
- `SKD−Ssup`：Teacher soft ranking supervision 的额外收益；
- 这一层**不测 candidate admission**。

### 7.1 Qwen-Raw 不把固定 B13 U 当主比较

可以额外让 Qwen-Raw 对同一个冻结 U 打分，作为辅助 scorer diagnostic；但不要把它当作 Qwen-Raw 的正式 retrieval baseline，因为候选本身已经由 B13 生成。

Qwen-Raw 的核心比较放在下一层的 full-lake retrieval。

---

## 8. Evaluation Layer 2：Full-lake Retrieval —— 正式加入 Qwen-Raw

这是本版最重要的新增部分。

### 8.1 四个 candidate generator

对每个正式 retriever：

```text
Qwen-Raw
B13
Ssup
SKD
```

全部重新从完整数据湖做检索。

### 8.2 完全一致的 branch budget

每个 retriever 都执行：

```text
Direct:
Q → T Top100

Evidence first hop:
Q → text Top20
Q → image Top20

Evidence expansion:
每个 text/image evidence → T Top20

U = Direct targets ∪ Evidence-path targets
```

如果 source artifact 表明历史真实 budget 与上述数字不同，则所有四个 retriever 统一改成历史实际 budget，并在 frozen config 中写清楚；禁止只给某个 baseline 不同预算。

### 8.3 不使用 RRF 做主排序

对 retriever-alone 指标：

```text
Direct 使用 retriever 自身 QT score
U 也使用 retriever 自身 QT score 排序
```

不得通过 weighted RRF 或证据低权重提高 Qwen/Student 中某一个的成绩。

Candidate admission 主要看 membership / RawRecall，本身不依赖 fusion score。

### 8.4 Qwen-Raw full-lake 指标

正式报告：

```text
Qwen-Raw Direct exact R@10 / 20 / 50 / 100
Qwen-Raw Direct ANN R@10 / 20 / 50 / 100
Qwen-Raw Direct Raw@100
Qwen-Raw U RawRecall
Qwen-Raw U R@10 / R@20 / CR@50
implicit / explicit
```

同样报告 B13 / Ssup / SKD。

核心 baseline table 应包括：

| Retriever | Raw pretrained? | Student-trained? | Teacher KD? | Direct Raw@100 | U RawRecall | U R@10 | U R@20 | U CR@50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen-Raw | ✓ | ✗ | ✗ |  |  |  |  |  |
| B13 | ✗ | ✓ | ✗ |  |  |  |  |  |
| Ssup | ✗ | ✓ | ✗ |  |  |  |  |  |
| SKD | ✗ | ✓ | ✓ |  |  |  |  |  |

核心问题：

```text
B13 − Qwen-Raw
```

回答：

> 原 Student learning 是否已经比 pretrained Qwen retrieval 有实际价值？

```text
Ssup − B13
```

回答：

> 更符合 deployment distribution 的 natural-pool supervised training 是否进一步改善 retrieval？

```text
SKD − Ssup
```

回答：

> 强 Teacher 是否带来超过纯监督 Student training 的额外收益？

### 8.5 Exact vs ANN

Qwen-Raw 也必须有 exact–ANN 对照。

对于 Direct QT：

```text
exact full-lake cosine TopK
vs
Qwen-Raw ANN TopK
```

对 B13 / Ssup / SKD 同理计算各自 exact scorer 与 ANN。

报告两类指标：

1. GT retrieval metric gap；
2. ANN TopK 对 exact TopK 的 neighbor recall。

避免出现：Student 表示实际上更好，但 ANN/index contract 变差，被误判为训练失败。

---

## 9. Evaluation Layer 3：固定 T*，只比较 Candidate Generator

这一层用完全相同的 Teacher T*，把四种 candidate generator 产生的候选交给 Teacher：

```text
Qwen-Raw candidates → T*
B13 candidates      → T*
Ssup candidates     → T*
SKD candidates      → T*
```

这一步尤其重要，因为它可以把：

```text
candidate admission quality
```

和：

```text
retriever自己的排序能力
```

拆开。

### 9.1 必报指标

对每个 generator：

```text
Direct Raw@100
U RawRecall
fixed-T* U R@10
fixed-T* U R@20
fixed-T* U CR@50
implicit / explicit
```

主比较：

```text
T*(C_QwenRaw)
vs
T*(C_B13)
vs
T*(C_Ssup)
vs
T*(C_SKD)
```

如果：

```text
SKD Student-alone R@10 提升
但 fixed-T* RawRecall / R@10 不提升
```

则说明 Student 可能主要学会了已有候选排序，而没有产生更有价值的搜索空间。

反之，如果：

```text
SKD U RawRecall ↑
且 fixed-T* U R@10 ↑
```

才更强地支持：

> Teacher knowledge transfer 最终改善了 scalable candidate generator。

### 9.2 相同预算不等于相同 unique candidate 数

所有四个 generator 使用相同 branch TopK，但 evidence overlap / duplicate 不同，`|U_q|` 可能不同。

因此：

1. 主结果保留真实部署的固定 branch budget；
2. 每个 query 报告 unique U size；
3. 额外做 matched-cardinality diagnostic。

定义：

```math
m_q = min(
|U_q^{raw}|,
|U_q^{B13}|,
|U_q^{Ssup}|,
|U_q^{SKD}|
).
```

对每个 U 使用同一个**不读取 GT**的确定性 hash 截断至 `m_q`，再由 T* 排序。

该诊断只用于排除“某个 retriever 只是产生了更多 unique candidates”的解释，不替代主结果。

---

## 10. U vs M：Qwen-Raw 也要做自己的 matched direct control

对于每个 retriever `S`：

```math
M_S(q) = Direct_S Top|U_S(q)|.
```

因此分别得到：

```text
Qwen-Raw: U_raw vs M_raw
B13:      U_B13 vs M_B13
Ssup:     U_sup vs M_sup
SKD:      U_kd vs M_kd
```

并在**同一个固定 T***下比较：

```text
U−M R@10
U−M R@20
U−M CR@50
U−M RawRecall
implicit / explicit
source-group bootstrap
```

这样可以回答：

> evidence candidate composition 的收益是 pretrained Qwen 原本就有，还是经过 Student training 后才增强/削弱？

特别关注 explicit：如果 Qwen-Raw / B13 / Ssup / SKD 对 explicit 的 U−M 表现变化明显，需要单独分析，不通过 weighted RRF 遮蔽。

---

## 11. Candidate Source 的低成本反事实拆分

对每个新的 Student S1（Ssup、SKD）保留：

```text
D0 = Direct(B13)
E0 = Evidence(B13)
D1 = Direct(S1)
E1 = Evidence(S1)
```

构造：

```text
U00 = D0 ∪ E0
U10 = D1 ∪ E0
U01 = D0 ∪ E1
U11 = D1 ∪ E1
```

用同一个 T* 排序。

Qwen-Raw 再作为额外 read-only reference：

```text
D_raw
E_raw
U_raw = D_raw ∪ E_raw
```

如果需要进一步拆 Qwen→Student 的变化，可以另外报告：

```text
D_B13 ∪ E_raw
D_raw ∪ E_B13
```

但这两项只作为 source attribution diagnostic，不成为新主实验臂，也不扩大训练矩阵。

这一步回答：

- Qwen-Raw → B13 的收益来自 direct 还是 evidence？
- B13 → S1 的收益来自 direct 还是 evidence？
- evidence retrieval 是否因 Student task-specific training 受到损伤？

---

## 12. Training-side Student→Teacher Candidate Analysis

在 train-only queries 上，对：

```text
Qwen-Raw
B13
Ssup
SKD
```

生成相同 branch budget 的 candidate pools，用同一个 T* 统一打分。

报告：

- pair universe overlap；
- Jaccard；
- Qwen-only / B13-only / Ssup-only / SKD-only；
- known-positive coverage；
- candidate source direct/text/image；
- T* high-score assumed-negative novelty；
- high-score unknown 的 target hubness；
- current Teacher 下 old/new negative rank；
- known-positive vs assumed-negative margin；
- strict exact-Direct100 outside + matched-M outside positives。

Qwen-Raw 的作用是多提供一层参照：

> 当前 hard-negative 搜索空间究竟是 pretrained backbone 原本就能看到，还是 Student training 后才产生？

不能把 Teacher high-score unknown 称为 verified negative；semantic false-negative 风险继续保持 unknown。

---

## 13. 结果判读

### 13.1 新增 Qwen-Raw 后的核心判读表

| 观察 | 支持的结论 | 不能推出什么 |
|---|---|---|
| B13 > Qwen-Raw | 原 Student task-specific learning 有价值 | 不能说明 Teacher/KD 有价值 |
| B13 ≈ Qwen-Raw | 早期 Student 学习没有产生明显 retrieval 增益 | 不等于 Student 架构必然无效 |
| B13 < Qwen-Raw | 旧 Student 可能破坏了 pretrained retrieval geometry | 不能因为后续 Teacher 强就忽略这个问题 |
| Ssup > B13 | natural deployment-like supervision 有效 | 不能归因于 Teacher KD |
| SKD > Ssup | Teacher soft ranking supervision 有额外贡献 | 需继续确认是 fixed-pool ranking 还是 admission |
| SKD fixed-U ↑，full-lake RawRecall ≈ | 学会已有candidate排序，但 admission 未改善 | 不算 scalable retrieval 成功 |
| SKD full-lake RawRecall ↑ 且 fixed-T* R@10 ↑ | Teacher knowledge transfer改善candidate generator获得较强支持 | 仍未证明 Student→Teacher feedback 有效 |
| Qwen-Raw evidence 比 Student evidence 更强 | Student training可能损伤跨模态 retrieval | 不应通过降低 evidence 权重掩盖 |
| Student exact ↑、ANN ≈/↓ | ANN/index contract 可能成为瓶颈 | 不能直接归因训练失败 |

### 13.2 是否触发 T2

本轮仍不自动训练 T2。

只有当某个 S1（不强制必须是 SKD）满足：

1. full-lake candidate admission 有稳定改善；
2. fixed T* 在该候选池上有实际 R@10 / CR@50 收益；
3. S1-only candidate 中存在旧 B13 没有的有用 hard negatives；
4. candidate diversity 没有严重 collapse；

才设计下一次单因素对照：

```text
同一个 T* parent
同一个 mining scorer
同一个 training budget
同一个 hard-list length

只改变：
B13 candidate generator
vs
S1 candidate generator
```

Qwen-Raw 不作为 T2 candidate generator 主臂，除非实验意外显示它比所有 Student 都更强；若发生这种情况，应优先重新审视 Student 训练，而不是继续闭环叙事。

---

## 14. 其他方向的优先级保持不变

### Dynamic Teacher refresh

R20 已证明单次 refresh 有效，但 R21 不与 Student KD 混跑。若 Student 实验失败且固定 B13 pool 上仍显示明显可重复的 fresh-hard空间，再单独考虑。

### Longer list

R19 已证明 list32 有价值，但暂不扫 64/128/256。只有新 candidate pool 显示大量高分 hard negatives 明显装不下，才单独预注册比较。

### Global dimension / relation head

仍不是当前第一优先。

### Fusion / adaptive RRF

继续暂缓。不允许通过 evidence 低固定权重退化成 direct-only。

### Evidence-conditioned `H(Q,T,E)`

只有未来显示 evidence candidate 自身包含有价值信号，而 QT Teacher 无法区分，才考虑。当前 Qwen-Raw baseline 只是 retriever baseline，不等价于 evidence-conditioned verification。

---

## 15. 必须导出的 R21 artifact

建议：

```text
work/stage1_optimization_r21_20260911/
```

### 15.1 Lineage / correctness

```text
PLAN_FROZEN.md
RESOLVED_CONFIG.json
INPUT_MANIFEST.json
CODE_HASH_MANIFEST.json
ENVIRONMENT.json
RUN_COMMANDS.md
COMPLETION_AUDIT.json
FAILURE_NOTES.json
STEP0_PARITY.json
ANN_SCORE_PARITY.json
QWEN_RAW_FEATURE_MANIFEST.json
QWEN_RAW_SCORE_PARITY.json
```

### 15.2 Student training

导出：

- train manifest；
- full natural candidate membership；
- known-positive registry；
- D2 Teacher soft-score cache；
- Ssup/SKD train history；
- supervised loss；
- KD loss；
- entropy / positive mass；
- gradient norm；
- actual optimizer param groups；
- final/mid checkpoint hash；
- batch-list order hash。

### 15.3 Full-lake retrieval

对：

```text
Qwen-Raw
B13
Ssup seed13/29
SKD seed13/29
```

全部保存：

- direct exact rankings；
- direct ANN rankings；
- QE rankings；
- ET rankings；
- U candidate membership；
- M candidate membership；
- candidate source；
- raw scorer score；
- fixed-T* score；
- positive ranks；
- exact-vs-ANN neighbor recovery；
- per-query candidate counts。

不要只保存 Top50；必须保存能够复算 RawRecall 和 exact/ANN gap 的完整 candidate membership。

### 15.4 独立统计

统一复算：

```text
Direct R@10/20/50/100
Direct Raw@100
U RawRecall
U R@10/20/CR@50
M RawRecall
M R@10/20/CR@50
U−M
implicit / explicit
per-query W/L/T
source-group paired bootstrap
seed-specific + seed mean
```

核心比较至少包括：

```text
B13 − Qwen-Raw
Ssup − B13
SKD − Ssup
SKD − Qwen-Raw
```

同时在 fixed-T* candidate-generator evaluation 中报告同样的 candidate admission 与最终 ranking 对比。

---

## 16. Codex 实施顺序

严格按：

```text
1. preflight
2. locate/fingerprint Qwen-Raw features
3. Qwen-Raw cosine/ANN parity smoke
4. freeze training pools
5. D2 final Teacher soft-score cache
6. Student loss/unit tests
7. freeze source/config hashes
8. train Ssup × 2
9. train SKD × 2
10. build Qwen-Raw / B13 / Ssup / SKD full-lake indexes
11. exact-vs-ANN evaluation
12. Direct + Evidence candidate generation
13. retriever-alone evaluation
14. fixed-T* reranking evaluation
15. U-vs-M + matched-cardinality analysis
16. candidate-source / hard-negative mechanism analysis
17. independent metric recomputation
18. completion audit
```

不要因为 Qwen-Raw baseline 加入，就重跑 Teacher、改变训练 loss、改变 candidate budget 或引入新的 fusion。

---

## 17. 本轮最终必须回答的科研问题

R21 最终不是只回答“哪个数字最高”，而要依次回答：

### Q1. 原 Student 到底有没有超过 pretrained Qwen？

```text
Qwen-Raw vs B13
```

重点看 full-lake exact/ANN、Direct Raw@100 和 U RawRecall，而不只看 frozen-pool排序。

### Q2. natural deployment-like supervised training 是否有额外价值？

```text
Ssup vs B13
```

### Q3. 强 Teacher 的 soft knowledge 是否真的有独立增益？

```text
SKD vs Ssup
```

### Q4. 提升发生在已有candidate排序，还是新的candidate admission？

同时比较：

```text
frozen old pool
full-lake RawRecall
fixed-T* candidate-generator evaluation
```

### Q5. 新 Student 是否同时改善 Direct 和 Evidence retrieval？

不能只看 overall；必须看：

```text
direct
text evidence
image evidence
implicit
explicit
```

### Q6. exact 表示能力和 ANN 部署能力是否一致？

如果 exact 好而 ANN 差，优先定位 index/score contract。

### Q7. 是否值得进入 Student→Teacher feedback？

只有 candidate generator 本身真正变强并给 T* 提供新的有用候选/难例时，才启动下一轮 T2 对照。

---

## 18. 论文叙事边界

如果实验成功，最理想、也最干净的 Stage1 机制链是：

```text
Qwen-Raw
    ↓
task-specific Student 学习改善 retrieval
    ↓
natural retrieval distribution supervision 进一步改善
    ↓
strong Teacher knowledge transfer 进一步改善
    ↓
full-lake candidate admission 提升
    ↓
同一个 Teacher 得到更好的候选池
```

但每一条箭头都必须由对应单因素比较支持。

特别禁止：

- 如果 B13 < Qwen-Raw，仍然只展示最终 SKD 而隐藏这个结果；
- 如果 SKD frozen-pool 更好但 RawRecall 不变，宣称 Student retrieval 已改善；
- 如果 Qwen-Raw 产生更多 candidates，直接比较未控制 candidate 数的 fixed-T*结果并称“表示更好”；
- 用 weighted RRF 让某一 retriever 看起来更优；
- 将 U−M 提升写成 Teacher 使用了 evidence 内容；
- 将 Qwen-Raw 的 RawRecall 与 Teacher R@10 直接比较为“未训练模型比训练模型更强”。

本轮最重要的新 baseline 问题是：

> **同一个 Qwen backbone 的 raw pretrained retrieval，到底离我们最终想要的 Join Discovery retriever 有多远；B13、Ssup 和 SKD 分别真正增加了什么。**
