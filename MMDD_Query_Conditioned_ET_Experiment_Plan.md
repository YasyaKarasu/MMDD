# MMDD Query-Conditioned E→T 实验文档

**状态：预注册实验计划，尚未执行。**  
**核心问题：在保持 target 侧静态 ANN 索引不变的前提下，让第二跳 E→T 显式读取 Q，是否能改善 evidence target admission，尤其是 strict direct-budget-outside positives？**

---

## 0. 背景与本轮定位

当前 Student 的第二跳本质上是：

\[
s(E,T)=u_E^\top R_{E,T}u_T
\]

其中：

\[
u_x=P_{\tau(x)}z_x
\]

因此对于同一个 evidence \(E\)，不同 query \(Q\) 下：

\[
T_1,T_2,\dots
\]

之间的相对排序不随 \(Q\) 改变。

虽然完整 path score 可以写成：

\[
p(Q,E,T)=s(Q,E)+s(E,T)
\]

但固定 \(Q,E\) 时：

\[
p(Q,E,T_1)-p(Q,E,T_2)
=
s(E,T_1)-s(E,T_2)
\]

也就是说，\(s(Q,E)\) 只对同一个 E 的所有 target 加上相同常数，无法改变该 E 内部的 target ranking。

这会造成一个结构性限制：

> 同一条 evidence 可能包含多个实体、属性或关联，但当前 E→T 不知道“当前 query 到底需要 evidence 中的哪一部分关系”。

本轮要验证的不是：

```text
Path loss 是否更复杂
```

而是一个更基础的问题：

> **第二跳显式读取 Q 是否本身有价值？**

---

## 1. 重要边界：这不是 C1@356→659 collapse 的直接解释

Bridge 已经发现：

```text
C1@356
→ C1@500
→ C1@659
```

期间 evidence retrieval 明显退化。

但：

```text
E→T 不读取 Q
```

这个结构限制在 step356 时就已经存在。

因此不能把它直接解释成：

> “356→659 collapse 的 root cause 就是 E→T 看不到 Q。”

本实验是一个独立结构性验证：

```text
现有健康 checkpoint
+
只改变第二跳 query construction
```

回答：

> 当前架构是否因为 query-independent E→T 而存在额外上限？

C1 collapse 的 QE / ET / P-R decomposition 仍然是独立机制诊断，不因本实验取消。

---

# 2. 核心设计：Target index 保持静态，只让查询向量依赖 (Q,E)

当前 E→T 可写为：

\[
s(E,T)
=
v_E^\top u_T
\]

其中：

\[
v_E=R_{E,T}^{\top}u_E
\]

而 \(u_T\) 是可预计算、可建立 ANN index 的 target vector。

本实验改成：

\[
\boxed{
v_{Q,E}=v_E+\Delta_\theta(Q,E)
}
\]

然后：

\[
\boxed{
s(T\mid Q,E)=v_{Q,E}^{\top}u_T
}
\]

关键性质：

```text
target vector u_T 不变
target ANN index 不变
只在线生成新的 query vector v_QE
```

因此：

> **不需要根据每个 Q 动态重建全湖索引。**

改变的是：

```text
过去：
一个 E 对所有 Q 共用同一个 E→T query vector / TopK

现在：
同一个 E 在不同 Q 下生成不同的 query vector
```

---

# 3. Parent checkpoint 与冻结范围

主实验统一使用：

```text
历史 B13 exact-replay 的最终健康 Student checkpoint
```

即 R27 已验证能够复现历史 B13 的最终 Student 状态。

原因：

1. 当前这是最健康、最可信的 Student reference；
2. 避免把 C1@659 evidence collapse 混入本实验；
3. 可以严格做 step0 parity；
4. Direct / Q→E / target index 均可冻结。

本轮默认：

```text
Frozen Qwen3-VL-Embedding-8B
Frozen P
Frozen all original R
Frozen Direct branch
Frozen Q→E branch
Frozen target vectors u_T
Frozen target ANN index
```

**只训练新增的 query-side residual adapter。**

不从 C1@659 开始。

不重新训练 B13。

不训练 Teacher。

不重新蒸馏 Student。

---

# 4. 新增模块

## 4.1 Base E→T query vector

对 text / image evidence 分别：

\[
v_E = R_{\tau(E),T}^{\top}u_E
\]

与当前实现完全一致。

---

## 4.2 Query-conditioned residual

定义：

\[
h(Q,E)=
[
u_Q;
u_E;
u_Q\odot u_E;
|u_Q-u_E|
]
\]

然后：

\[
\Delta_\theta(Q,E)
=
W_2\,\mathrm{GELU}(W_1h(Q,E)+b_1)+b_2
\]

最终：

\[
v_{Q,E}=v_E+\Delta_\theta(Q,E)
\]

建议：

```text
input dim = 4d
hidden dim = 256
output dim = d
activation = GELU
dropout = 0
```

其中：

```text
d = 当前 Student projection dimension
```

不要写死 1024；执行端从 checkpoint config 读取并写入 manifest。

---

## 4.3 零初始化要求

最后一层：

```text
W2 = 0
b2 = 0
```

因此 step0 必须满足：

\[
\Delta_\theta(Q,E)=0
\]

从而：

\[
v_{Q,E}=v_E
\]

即：

> **新增模块在训练前必须与原 B13 E→T 完全一致。**

这是 correctness gate，不是可选项。

---

# 5. 两个主要训练 arm

本轮不做大矩阵。

只训练两个 residual arm，每个两个 seed。

---

## Arm A：E-ONLY-RESIDUAL

作用：

> 控制“只是多了一层网络 / 更多参数”带来的收益。

使用与 QE arm **完全相同的 MLP 参数量和计算图**，但输入中的 Q 信息全部置零：

\[
h_E=
[
0;
u_E;
0;
0
]
\]

因此：

\[
v_E' = v_E+\Delta_{\theta_E}(E)
\]

它可以学习一个更好的 evidence→target query transform，但看不到 Q。

---

## Arm B：QE-RESIDUAL

主实验：

\[
h_{QE}=
[
u_Q;
u_E;
u_Q\odot u_E;
|u_Q-u_E|
]
\]

\[
v_{Q,E}=v_E+\Delta_{\theta_{QE}}(Q,E)
\]

其他：

```text
parent
training data
candidate lists
optimizer
updates
seed
batch order
negative lists
positive protection
```

与 E-ONLY 完全一致。

---

## 为什么不把 Q-additive 当主方案

不把下面这个作为主要方法：

\[
v_{Q,E}=v_E+\alpha v_Q
\]

因为它对应：

\[
s(T|Q,E)=s(E,T)+\alpha s(Q,T)
\]

本质上更接近：

```text
Direct + Evidence
```

而不是让 Q 决定：

> “当前应该如何理解 E。”

它可以作为一个 **evaluation-only sanity baseline**：

```text
Q-ADD
```

但不允许为了它调 \(\alpha\) 网格。

若执行，可固定：

```text
alpha = 1
```

并明确只作为诊断，不作为主方法。

---

# 6. Training data：只训练第二跳，不改变第一跳

## 6.1 第一阶段使用 verified train witness paths

主训练样本必须来自：

```text
train-known / verified witness
```

形成：

```text
(Q, E, positive target T+)
```

不要因为：

```text
(Q,T) 是正 target
```

就把该 query 检索到的每一个 evidence 都标成：

```text
(Q,E,T) positive
```

这会把错误 evidence 强行训练成正路径。

---

## 6.2 Candidate target list

为每个训练 `(Q,E)` 冻结一份 target candidate list。

建议：

```text
list size = 32
```

来源：

```text
原 B13 E→T full-lake retrieval 的 hard candidates
+
所有 train-known positive targets 的 closure protection
```

要求：

```text
candidate list 在两个 arms 之间逐 ID 相同
candidate order 相同
positive mask 相同
```

Unknown 可以作为 assumed negative。

所有 train-known positives 必须保护。

---

## 6.3 不重新 mining

本轮禁止：

```text
训练中动态 remining
Teacher mining
按 QE adapter 当前分数刷新 negative
```

否则：

```text
E-only
vs
QE-conditioned
```

将同时改变训练候选分布。

如果本轮证明 query conditioning 有效，再下一轮考虑 conditioned hard-negative mining。

---

# 7. Training objective

为了避免把：

```text
query conditioning
```

与：

```text
新的 path aggregator
```

混在一起，本轮直接训练第二跳 target ranking。

对每个 `(Q,E)`：

\[
\ell(T)=s(T\mid Q,E)
\]

使用当前项目已有的：

```text
multi-positive sum_probability listwise loss
```

正例为：

```text
当前 list 中所有 train-known positives
```

不得把缺席正例当 negative。

主损失：

\[
L_{ET-QC}
\]

本轮：

```text
KD = 0
Uniform = 0
new path loss = 0
new Direct loss = 0
new QE loss = 0
```

因为 Direct、QE、原始 P/R 全部冻结。

---

# 8. Optimizer 与训练预算

两个 arm × 两个 seed：

```text
2 arms × 2 seeds = 4 training jobs
```

推荐 seed：

```text
13
29
```

训练：

```text
3 epochs
```

保存：

```text
epoch 0
epoch 0.5
epoch 1
epoch 2
epoch 3
```

主结论使用：

```text
epoch 3
```

trajectory 只用于判断是否 undertrain / overfit。

不能从 5 个 checkpoint 中挑最高分冒充主结果。

---

## 8.1 Optimizer

新增 adapter：

```text
AdamW
```

不要继承 B13 optimizer state。

学习率不要做 grid。

建议单一预注册值：

```text
LR = 1e-4
weight_decay = 0.01
```

如果当前代码中小 MLP adapter 已有统一约定，则优先使用项目统一 adapter LR，但必须在执行前写死到 manifest。

不能看结果后调整。

---

# 9. 第一层评测：固定 evidence 的 E→T exact retrieval

这是本实验最重要的机制指标。

## 9.1 固定 evidence set

对每个 test query，固定同一批 evidence：

```text
优先使用 B13 Q→E 的冻结 retrieval 结果
```

三个模型：

```text
BASE
E-ONLY
QE-RESIDUAL
```

使用完全相同 evidence IDs。

这样第一跳完全不参与比较。

---

## 9.2 Full-lake exact target search

对每个 `(Q,E)`：

### BASE

\[
v_E
\]

### E-ONLY

\[
v_E+\Delta(E)
\]

### QE

\[
v_E+\Delta(Q,E)
\]

然后对同一份：

```text
全湖 target vectors u_T
```

做 exact inner-product ranking。

### 指标

```text
Target Recall@1
Recall@5
Recall@10
Recall@20
MRR
positive median rank
p90 positive rank
positive-vs-top-negative margin
```

主汇总：

```text
query-macro
```

另外报告：

```text
text evidence
image evidence
implicit
explicit
strict historical EO
```

---

# 10. 第二层评测：ANN compatibility

exact 完成后，对同一 query vector 使用：

```text
原 B13 target ANN index
```

不能重建一个 QE 专用 target index。

因为：

```text
target vectors u_T 没有变化
```

### 必须验证

```text
BASE / E-ONLY / QE
target index SHA 完全一致
target vector fingerprint 完全一致
index insertion order一致
HNSW M / efConstruction / efSearch一致
```

报告：

```text
exact vs ANN Recall
exact vs ANN membership overlap
latency p50 / p95
ANN query count
```

若：

```text
exact明显改善
ANN不改善
```

则问题优先归因：

```text
ANN approximation / query-vector geometry / search parameters
```

而不是 query conditioning 本身失败。

---

# 11. 第三层评测：End-to-end evidence candidate admission

第一跳仍固定为：

```text
B13 Q→E
```

只替换第二跳。

保持：

```text
same number of evidence per query
same targets-per-evidence
same dedup
same D1 / coverage E ranking
same Direct ranking
same U construction
same C100 budget
same frozen T0 reranker
```

因此唯一变化是：

```text
E→T candidate search query
```

---

## 11.1 主要指标

正式报告：

```text
E RawRecall
E R@10
U RawUnionRecall
U R@10 / R@20 / R@50
C100 candidate Recall
C100 + frozen T0 R@10 / R@20 / R@50
```

全部分：

```text
overall
implicit
explicit
```

同时保留：

```text
raw Qwen
historical B13
BASE adapter step0
```

---

# 12. Strict evidence-only funnel

这是本实验最关键的论文机制指标之一。

使用固定的历史双重排除 strict EO：

\[
EO_{\mathrm{fixed}}
=
G
\cap E_{\mathrm{B13}}
\setminus
(D_{\mathrm{ANN100}}\cup D_{\mathrm{exact100}})
\]

同时计算每个新 arm 自己的：

```text
model-own EO
```

但固定历史 EO 才用于 arm 间机制比较。

报告：

```text
strict EO total
第二跳召回
进入 E union
进入 U
进入 C100
T0 Top10
T0 Top20
T0 Top50
```

以及：

```text
rescued
dropped
net
```

特别检查：

> QE-RESIDUAL 是否真正救回 Direct100 外、原 E→T 找不到的 target。

---

# 13. Query-use mechanism diagnostic

仅仅 QE 比 BASE 高还不够。

必须证明：

> 模型真正使用了 Q，而不是 adapter 参数更多。

---

## 13.1 E-only capacity control

主比较：

```text
QE-RESIDUAL
vs
E-ONLY-RESIDUAL
```

同参数规模、同训练数据、同预算。

只有 QE 稳定优于 E-only，才能支持：

> “显式读取 Q 提供了额外信息。”

---

## 13.2 Query shuffle

固定：

```text
evidence E
candidate targets
target index
adapter checkpoint
```

仅把输入 adapter 的 query Q 换成：

```text
其他 source group 的 query
```

donor 不根据 target label 选择。

比较：

```text
Real-Q
vs
Shuffled-Q
```

QE arm 应当表现出明显差异。

E-only arm 应严格 invariant。

---

## 13.3 Direct-shortcut diagnostic

对于 QE 新救回的 target，报告它们在原 Direct QT 中的 rank：

```text
Direct rank <= 10
<= 100
> 100
```

尤其统计：

```text
strict direct-budget-outside rescued targets
```

如果新增收益几乎全部来自：

```text
本来 Direct 就非常靠前
```

则可能只是 adapter 偷学：

```text
Q→T shortcut
```

不能解释成 evidence relation selection 成功。

---

## 13.4 Same-E / different-Q diagnostic

找到测试集中：

```text
同一个 E
被多个 Q 召回
```

的情况。

比较 BASE 与 QE：

```text
同一个 E
不同 Q
→ target TopK 是否发生有意义变化
```

输出案例：

```text
Q1, E → TopK(T)
Q2, E → TopK(T)
```

并检查对应正 target 是否朝正确方向移动。

这是对“query-conditioned second hop”最直观的机制证明。

---

# 14. Adapter behavior diagnostic

导出：

```text
||v_E||
||Δ(Q,E)||
||Δ|| / ||v_E||
cos(v_E, v_QE)
```

分：

```text
text
image
positive witness
non-positive / unknown evidence
implicit
explicit
```

目的：

避免 adapter 训练成：

```text
巨大 residual 完全覆盖原 E→T
```

如果：

```text
||Δ|| >> ||v_E||
```

但性能提高，

不能直接说“保留原几何上的轻量修正”。

仍然可以是有效模型，但机制解释需要修改。

本轮不要为此增加 residual norm regularizer。

先观察。

---

# 15. Cache / indexing correctness

这是实现时最容易出 bug 的位置。

当前 query-vector cache 如果只按：

```text
evidence_id
```

或：

```text
(source_id, destination_type)
```

缓存，

加入 query conditioning 后会错误。

QE arm cache key 至少必须包含：

```text
query_id
evidence_id
evidence_type
destination_type
adapter checkpoint/version
```

即：

```text
(Q,E,T-type,model-version)
```

---

## 15.1 Target index identity 与 query-side identity 分离

过去完整 Student checkpoint SHA 变化时，index checker 可能认为必须重建 target index。

本轮必须拆成：

```text
target_index_identity
query_model_identity
```

因为：

```text
P / target-side R / u_T
全部冻结
```

QE adapter 更新不应迫使重新建 target index。

但是不能简单删除 index correctness check。

必须保存：

```text
target vector SHA/fingerprint
index manifest SHA
query adapter SHA
base Student SHA
```

---

# 16. Step0 correctness tests

训练前必须通过。

## C0. Score parity

因为 residual 最后一层零初始化：

```text
BASE E→T exact score
==
QE epoch0 E→T exact score
==
E-only epoch0 E→T exact score
```

默认 float32：

```text
atol = 1e-6
rtol = 1e-5
```

---

## C1. Ranking parity

固定小批量：

```text
TopK target IDs
```

必须与 BASE 完全一致。

---

## C2. End-to-end parity

epoch0：

```text
E candidates
U membership
C100
```

必须与 BASE 一致。

ANN tie 如存在，单独标记，不得偷偷容忍任意大差异。

---

## C3. Frozen branch check

训练前后都验证：

```text
Direct scores unchanged
Q→E scores unchanged
target vectors unchanged
original P/R unchanged
```

任何变化都属于实现错误。

---

# 17. 统计协议

主比较：

```text
QE − BASE
QE − E-ONLY
E-ONLY − BASE
```

两个 seed 先对同 query 聚合，再做：

```text
per-query W/L/T
source-group paired bootstrap
95% CI
```

不把两个 seed 当成双倍独立 query。

主指标：

```text
exact conditioned ET R@10
E RawRecall
U RawUnionRecall
C100 candidate Recall
C100+T0 R@10
strict EO Top10 retention
```

不要只报 overall。

必须同时：

```text
overall
implicit
explicit
text
image
```

---

# 18. 如何解释不同结果

## 情况 A

```text
QE > E-only > BASE
```

且：

```text
Real-Q > Shuffled-Q
strict EO 明显改善
```

支持：

> 当前 query-independent E→T 确实存在结构性上限，Q-conditioned second-hop 有实际价值。

下一轮可以考虑：

```text
conditioned hard-negative mining
target-level path training
Teacher path conditioning
Student distillation
```

---

## 情况 B

```text
E-only ≈ QE > BASE
```

说明：

> 主要收益来自额外 adapter capacity / 更好的 E→T transform，而不是 Q 信息本身。

不要写成 query conditioning 成功。

---

## 情况 C

```text
QE exact > BASE
ANN ≈ BASE
```

说明：

> 条件向量有价值，但 ANN search 对新的 query geometry 不够稳定。

下一轮优先：

```text
ANN/index diagnostics
```

而不是继续加复杂网络。

---

## 情况 D

```text
ET exact明显提高
但 U / C100 不提高
```

说明瓶颈可能在：

```text
Q→E evidence admission
per-E target budget
E target aggregation
C100 admission
```

需要沿 funnel 找损失位置。

---

## 情况 E

```text
QE 没有优于 E-only
且 shuffle-Q 几乎无影响
```

说明：

> 当前 adapter 没有从 Q 中学到额外有用信息。

竞争解释包括：

```text
现有 u_Q / u_E 已丢失细粒度绑定信息
witness supervision不足
训练 candidates不够困难
当前单向量表示不支持所需条件化
```

不能直接得出：

> “E→T 根本不需要 Q。”

---

## 情况 F

QE 提升，但新增正确 target 基本都是：

```text
Direct rank很高
```

说明：

> 模型可能主要学到了 Q→T shortcut。

此时不应直接进入论文主方法。

下一轮需要：

```text
更严格的 strict-EO training/evaluation
或 evidence-dependent interaction test
```

---

# 19. Execution gates

## G0：Base identity

B13 checkpoint、target vectors、target ANN index、Q→E rankings全部锁定。

---

## G1：Step0 parity

不通过则停止训练。

---

## G2：Training list correctness

检查：

```text
(Q,E)
candidate target IDs
positive masks
train-known closure
unknown-as-negative
```

两个 arm 完全一致。

---

## G3：Epoch1 health

必须满足：

```text
无 NaN / Inf
adapter gradients非零
原 P/R gradients = 0
Direct / QE frozen
```

但不能因为 epoch1 Recall 没涨就提前停。

---

## G4：Exact first

必须先完成：

```text
full-lake exact conditioned ET
```

再解释 ANN。

不能只看 HNSW 结果判断模型。

---

## G5：Query-use gate

如果：

```text
QE ≈ E-only
且
Real-Q ≈ Shuffled-Q
```

则停止扩大 query-conditioned 架构。

不自动加：

```text
更大 MLP
更多层
attention
cross-encoder
```

---

# 20. 本轮明确不做

```text
不解冻 Qwen
不更新原 P
不更新原 R
不重建 target embedding
不为每个 Q 动态构建 index
不做多跳
不训练 Teacher
不做 Teacher→Student KD
不做 dynamic hard-negative remining
不做 LR grid
不做 hidden-size grid
不做 residual-weight grid
不增加新的 fusion gate
不扩 Stage2
```

本轮只回答：

> **让第二跳显式读取 Q，本身是否有价值？**

---

# 21. 训练规模

正式训练：

```text
E-ONLY-RESIDUAL × 2 seeds
QE-RESIDUAL × 2 seeds
```

共：

```text
4 jobs
```

每个：

```text
3 epochs
```

BASE：

```text
0 training jobs
```

Q-ADD：

```text
可选 evaluation-only
0 training jobs
```

---

# 22. Artifact requirements

最终目录至少：

```text
EXECUTION_LEDGER.json
INPUT_LOCK.json
BASE_IDENTITY.json
TARGET_INDEX_IDENTITY.json

training/
  e_only/seed13/
  e_only/seed29/
  qe/seed13/
  qe/seed29/

step0_parity/
  exact_scores.jsonl.gz
  rankings.jsonl.gz
  summary.json

conditioned_et_exact/
  per_qe.jsonl.gz
  per_query.jsonl.gz
  summary.csv

conditioned_et_ann/
  per_qe.jsonl.gz
  per_query.jsonl.gz
  summary.csv

end_to_end/
  E_rankings/
  U_rankings/
  C100/
  T0_rankings/
  summary.csv

strict_eo/
  fixed_historical_eo.jsonl.gz
  own_eo.jsonl.gz
  funnel.csv

query_use/
  shuffle_q.jsonl.gz
  same_e_different_q.jsonl.gz
  direct_shortcut.csv

adapter_geometry/
  residual_norms.jsonl.gz
  summary.csv

statistics/
  wlt.csv
  source_group_bootstrap.csv

latency/
  ann_latency.json
  memory.json

RESULTS.md
LIMITATIONS.md
NEXT_DECISION.md
```

---

# 23. 最终报告必须回答

### Q1

在固定 evidence 下：

```text
QE-conditioned E→T
```

是否显著优于：

```text
原 E→T
```

？

### Q2

它是否显著优于参数量匹配的：

```text
E-only residual
```

？

### Q3

Query shuffle 后收益是否消失？

### Q4

收益主要来自：

```text
text
image
implicit
explicit
strict direct-budget-outside
```

中的哪一部分？

### Q5

exact 改善能否保留到 ANN？

### Q6

target index 是否真正做到：

```text
一次构建
所有 Q 复用
```

？

### Q7

新增 target 是否只是 Direct shortcut？

### Q8

同一个 evidence 在不同 Q 下，target ranking 是否真的发生了与任务一致的变化？

---

# 24. 每个结论统一使用因果格式

```text
Local fact：
真实源码 / 数据观察到了什么？

Hypothesis：
它说明什么？

Competing explanation：
还有什么解释？

Single-factor intervention：
本实验只改变什么？

Main metric：
主指标是什么？

Mechanism diagnostic：
如何确认模型真的使用了 Q？

Negative result means：
如果不提升，排除了什么、还剩什么解释？

Conclusion：
supported / weakened / unknown
```

---

# 25. 本轮科学成功标准

这轮不要求整体 R@10 一定大涨。

真正成功的结果是能够清楚回答：

> **“在完全不改变 target index、Direct、Q→E 和原 Student 几何的情况下，仅让第二跳 query vector 读取 (Q,E)，是否能恢复更多 query-specific target？”**

如果答案是 yes：

> 下一步才值得把 query conditioning 集成进正式 path-level Student / Teacher。

如果答案是 no：

> 再判断是 `Q/E 单向量信息不足`，还是 `训练监督不足`，而不是立刻扩大模型。

最重要的是：

> **本实验不牺牲全湖预处理能力：target 侧依旧是静态 ANN index；变化只发生在在线 query vector construction。**
