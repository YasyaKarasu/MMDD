# S2-COL-R2：Evidence-aware Column Selection（澄清版执行文档）

日期：2026-09-16  
状态：下一轮实验设计。本文档不是实验结果。  
目标：让 Codex 可以直接执行，不依赖它自行补全实验定义。

---

# 0. 一句话说明这一轮要解决什么

这一轮仍然只做 Stage2 的“列定位/列选择”，不做图片裁剪、文本 span 定位、逐行属性生成或最终 semantic-joinability。

但与 S2-COL-R1 不同，本轮不再主要优化“列打分头有多复杂”，而是回答：

> **Evidence 到底应该如何参与列选择？当前把至多 4 条 evidence 一次性拼到 reader 里，是否限制了 evidence 的有效利用？**

同时必须允许得到一个同样重要的结论：

> **如果 evidence 对列选择本身没有稳定增益，那么不要为了论文故事强行让 evidence 参与列选择；可以让 Q+T 先给出 Top-3 列假设，把 evidence 的主要作用后移到后续行级属性恢复。**

本轮不以“新结构一定要胜出”为前提。

---

# 1. R1 已经确认的事实：为什么现在要做这一轮

以下事实来自 S2-COL-R1 的真实运行产物，不是设计假设。

## 1.1 当前列定位已经不是“完全不会选列”

以 R1 的 C2（`tail_candidates_v1 + MLP`）为当前工作基线，两 seed 算术均值：

- Test / 正 evidence（O-O）：
  - Hit@1 ≈ 88.91%
  - Hit@2 ≈ 99.31%
  - Hit@3 ≈ 99.91%
  - MRR ≈ 0.9433
- Test / 自然 evidence（O-R）：
  - Hit@1 ≈ 83.46%
  - Hit@2 ≈ 98.67%
  - Hit@3 = 100.00%
  - MRR ≈ 0.9151

因此，“只保留 Top1”确实过早；Top2/Top3 可以保留大量 Top1 排错但仍有机会恢复的样本。

但是 Hit@3 在窄表上天然容易饱和，所以后续不能只盯着 Hit@3。

## 1.2 自然 evidence 的额外帮助很弱

R1 dev 上，同一 C2 checkpoint：

- O-O Hit@1 ≈ 88.61%
- O-R Hit@1 ≈ 83.40%
- No-E Hit@1 ≈ 82.66%
- Shuffled-E Hit@1 ≈ 82.71%

也就是说：

- 正 evidence 相比 No-E：约 +5.95 pp；
- 自然 evidence 相比 No-E：约 +0.74 pp；
- Shuffled-E 与 No-E 基本一样。

这说明：

1. reader **有能力**从质量较好的 evidence 中获益；
2. 目前自然 evidence 对列选择的稳定净增益很有限；
3. “自然 evidence 没作用”不能直接等价成“模型完全不会用 evidence”，因为训练分布、证据质量和融合方式都可能造成影响。

## 1.3 当前 evidence 并不是“简单 path Top4”

当前自然 evidence 的上游选择过程实际上是：

```text
evidence paths
→ exact-content dedup
→ 按 path score 保留 top_l=20
→ e2_row_coverage greedy selection
→ 最多保留 budget=4 条 evidence
→ Stage2 reader
```

所以这轮严禁把“把 Top4 改成 row coverage”作为新实验，因为 row coverage 已经存在。

此外，R1 冻结导出的 Stage2 evidence 本身已经 ≤4 条，因此 Stage2 的 `[:4]` 在这批正式输入上没有进一步删除 evidence。

## 1.4 本轮不是文本字符截断问题

R1 中保存的文本 evidence 最长约 800 字符，而 reader 在最多 4 条 evidence 时给每条文本的字符预算至少约 3000。

因此：

> 本轮没有证据表明“Stage2 reader 因文本太长截掉了关键信息”是当前主瓶颈。

不要在 R2 里优先调 12000 字符预算。

## 1.5 自然 evidence 为空时，要区分“没送到”与“不会用”

R1：

- dev：257/675 个 Q-T 的 O-R evidence 为空；
- test：220/644 个 Q-T 的 O-R evidence 为空。

其中多数是因为正确 T 没进入 Stage1 的自然 U，不是 Stage2 reader 自己丢了 evidence。

因此所有分析必须同时报告：

1. O-R full population；
2. O-R non-empty evidence subset。

**只有第 2 个子集能比较直接地诊断 evidence fusion。**

---

# 2. 本轮术语必须统一，禁止继续混用“Top-k”

下面这些变量含义完全不同，代码、日志、文件名中必须使用不同名字。

## 2.1 表候选数量

```text
n_target_candidates
```

这是 Stage1 给 Stage2 的 target table 数量。

本轮独立列实验给定正确 T，不优化这个变量。

## 2.2 Evidence 候选池大小

```text
m_evidence_pool
```

指在“昂贵 reader 真正读取之前”可供选择的 evidence 数量。

当前 Stage1 retention 的中间池通常是 exact-content dedup 后的 path-score Top20。

R2 主实验暂时**不扩大这个池**。

## 2.3 Reader 实际读取的 evidence 数量

```text
b_evidence_read
```

本轮固定：

```text
b_evidence_read = 4
```

如果自然 bundle 少于 4 条，就按真实数量读取；不得补 oracle evidence。

## 2.4 最终保留的候选列数

```text
k_column
```

评测：

```text
k_column ∈ {1, 2, 3, 5}
```

本轮新架构中的 prior shortlist 固定：

```text
m_column_shortlist = 3
```

即先由 Q+T 选 Top3 列，再让 evidence 决定是否需要在这 3 列之间重排。

这不是把最终输出强制为 3 列；最终仍然报告完整排序与 Hit@1/2/3/5。

---

# 3. 数据条件的精确定义

本轮所有实验必须基于同一个锁定的 Q-T population。

记：

- `Q`：query-by-example 表；
- `T`：正确 target table；
- `C(T)`：T 的全部候选列；
- `G(Q,T)`：已有标注认可的 gold bridge column 集合；
- `E`：evidence bundle。

## 3.1 O-O：positive/oracle evidence 条件

```text
Q + correct T + 已有 recovery 审计支持的 positive evidence
```

注意：

- 给定正确 T；
- 给定已有的正 evidence；
- **绝不把 gold column 名称、gold index 或 join_attribute 输入模型**；
- O-O 是组件能力诊断，不是在线部署结果。

## 3.2 O-R：natural retrieved evidence 条件

```text
Q + 同一个 correct T + 冻结 Stage1 自然检索实际给 T 带来的 evidence
```

关键约束：

- Stage1 retrieval 必须独立于 gold T/column 运行；
- 只有 T 自然出现在 Stage1 U 中时，才允许取它自然 retained 的 evidence；
- 如果 T 不在 U 中，`E=[]`；
- 如果 T 在 U 中但没有 evidence path，`E=[]`；
- **严禁因为 Stage2 已经知道“这张 T 是评测正表”，再以 T 为条件偷偷重新搜一遍 evidence。**

## 3.3 No-E

```text
Q + T + E=[]
```

用来测 Q+T 本身的列先验。

No-E 是模型输入条件，不是“目标没有 join”。

## 3.4 Shuffled-E

使用 R1 已有的 label-blind donor 规则：

- 不同 query；
- 不同 source table；
- 同 modality；
- 尽量匹配文本长度；
- 排除已知 positive witness。

Shuffled-E 只是扰动/合成负 evidence，不宣称每条 donor 在语义上都已被人工证明为负。

---

# 4. 本轮绝对不允许改动的内容

除非本文档某个实验臂明确写出，否则 Codex 不得自行修改：

1. 不训练或调参 Stage1 Teacher/Student；
2. 不重新设计 Stage1 target ranking；
3. 不改 B13/T0 或当前冻结 Stage1 retrieval 的候选逻辑；
4. 不做图片 crop；
5. 不做文本 span；
6. 不做属性值生成；
7. 不做 semantic-joinability；
8. 不引入 RRF/LSE 路径融合新实验；
9. Qwen reader backbone 完全冻结；
10. 继续使用 `tail_candidates_v1`，不要回退到 header marker；
11. target table、候选列、target rows 的可见内容保持 R1 合同；
12. 不把 dev/test 标签用于训练；
13. 不用 test 选择模型、超参或实验分支；
14. 不因为自然 evidence 为空而删掉 Q-T 样本；
15. 不自动把“非 gold 但看起来合理的列”改成 gold。

本轮的框架修改只发生在：

> **Stage2 列定位内部：Q/T 列先验如何形成，以及 evidence 如何对列先验进行验证/修正。**

这仍然符合原始方案中“先选择候选列，再做行级 evidence localization 和属性恢复”的总体流程。

---

# 5. 统一基线

当前工作基线记为：

```text
R1-C2
```

定义：

- reader layout：`tail_candidates_v1`
- frozen Qwen reader
- 所有当前 bundle evidence 一次性放进 reader
- candidate column marker 位于完整上下文之后
- MLP column head
- 训练条件：O-O
- 两个 seeds：13 / 29

R1-C2 可以作为“当前已完成工作点”。

但报告中不要写“C2 是显著最优结构”，因为 R1 的 C0/C1/C2 测试差异没有稳定证明这一点。

---

# 6. Phase A：先做一个不训练的新诊断，直接回答“一次拼接有没有明显互相干扰”

这一阶段只用 **dev**，禁止看 test 后再决定结构。

使用 R1-C2 已选 checkpoint。

仅在 O-R non-empty subset 上执行。

对于一个 evidence bundle：

```text
E = [E1, E2, ..., En], 1 <= n <= 4
```

执行以下 forward。

## A0. Bundle

保持 R1 原样：

```text
Reader(Q, T, [E1,...,En])
```

得到完整列排序。

## A1. Single-E

分别运行：

```text
Reader(Q,T,[E1])
Reader(Q,T,[E2])
...
Reader(Q,T,[En])
```

每个 pass 输出所有候选列 logits。

每个 pass 内对列做：

```text
logp_{c,j} = log_softmax(logits_j over columns)
```

禁止直接比较不同 pass 的原始 logit 绝对值。

额外生成两个**固定诊断聚合器**：

### A1-Mean

\[
score_c^{mean} = \frac{1}{n}\sum_j logp_{c,j}
\]

### A1-LME

\[
score_c^{lme} = \log \frac{1}{n}\sum_j \exp(logp_{c,j})
\]

这里 LME 是 log-mean-exp，不是 Stage1 path LSE，也不是本轮正式模型。

它们只是判断“逐条 evidence 再做简单集合聚合”是否已经比 concat 更有希望。

## A2. Leave-one-out

对每个 j 运行：

```text
Reader(Q,T,E without Ej)
```

统计：

- `harmful_evidence_count`：删除某条 E 后 Top1 从错变对；
- `helpful_evidence_count`：删除某条 E 后 Top1 从对变错；
- 每个 bundle 中是否存在明显 harmful item。

不要使用 gold 选择要删除哪条 evidence 来构造正式预测。

## A3. Evidence order perturbation

对 n>=2 的 bundle 生成固定、与标签无关的 3 个顺序：

1. 当前顺序；
2. 逆序；
3. 使用 `hash(query_id,target_id)` 决定的固定伪随机顺序。

统计同一 bundle 的 Top1 是否随顺序变化。

## A4. Oracle-single upper bound

只作为诊断：

> 如果允许事后看 gold，从 n 条 evidence 中选择“单独输入时最有利的一条”，Hit@1 能达到多少？

这个指标必须命名：

```text
oracle_single_evidence_upper_bound
```

严禁把它当成可部署方法、模型选择指标或论文主结果。

## Phase A 要输出什么

必须输出：

- concat Hit@1/MRR；
- Mean Hit@1/MRR；
- LME Hit@1/MRR；
- order-flip rate；
- helpful/harmful leave-one-out 统计；
- oracle-single upper bound；
- 按 evidence_count=1/2/3/4 分组；
- 按 text/image/mixed 分组。

### Phase A 如何解释

- 如果 `Single/Mean/LME` 明显好于 Bundle，说明 evidence 之间可能存在拼接干扰，值得训练分离式融合。
- 如果 Bundle 与这些方法差不多，不能再把“concat 本身”当成主要瓶颈。
- 如果 oracle-single 很高但任何 label-blind 聚合都不高，说明问题更可能是“如何识别有用 evidence”，而不是 reader 完全没能力读它。

---

# 7. Phase B：先排除训练分布不匹配——Flat-Mix 对照

R1 的 C2 只用 O-O 训练，但实际部署输入更接近 O-R。

在修改架构前，必须做一个**同结构、只改训练 evidence 分布**的对照。

## 7.1 先生成 train O-R

R1 中 train O-R 没有自然 retrieval 输入，必须补。

执行冻结的 Stage1 retrieval：

```text
train queries
→ frozen Stage1 pipeline
→ natural U and retained paths
```

然后，对于训练集中已知正 Q-T：

- 如果 T 自然出现在 U：读取它自然 retained 的 evidence；
- 如果 T 不在 U：O-R evidence = []；
- 不允许用 gold T 再 reretrieve；
- 不允许补 O-O evidence。

保存：

```text
FROZEN_NATURAL_TRAIN.jsonl.gz
```

并记录 Stage1 checkpoint / config / artifact hash。

## 7.2 B0：O-O control

使用 R1-C2 的训练 recipe：

```text
tail_candidates_v1
+ MLP
+ O-O only
```

如果 R1 cache/checkpoint/hash 完整一致，可直接复用 R1-C2，不要求无意义重跑。

## 7.3 B1：Flat-Mix

结构与 B0 完全相同，仍然一次性 concat bundle。

区别只有训练时每个基础 Q-T 每个 epoch 使用哪种 evidence：

```text
50% O-O
50% O-R
```

选择必须由：

```text
hash(seed, epoch, query_id, target_id)
```

确定，不能依据 gold position、evidence 是否命中或当前 loss 动态决定。

如果被分配到 O-R，而 O-R evidence 为空，就真的使用空 evidence。

### 公平性

B0/B1：

- 相同 base Q-T；
- 相同 epoch 数；
- 相同每 epoch sample visits；
- 相同 optimizer steps；
- 相同 seed；
- 相同 head 初始化；
- 相同 LR/weight decay；
- 相同 checkpoint selection rule。

因此 B1-B0 才能解释为“训练输入分布改变”的收益。

---

# 8. Phase C：主框架修改——Prior → Evidence Verify → Rerank

这是本轮最重要的结构实验。

核心思想不是“把更多 evidence 塞进去”，而是：

> **先用 Q+T 得到一个稳定的列先验，再让 evidence 只负责验证/修正少数候选列。**

原因：

- R1 No-E Hit@1 已经较高；
- R1 Hit@2/3 很高；
- 当前自然 evidence 的帮助与伤害大致会互相抵消。

因此 evidence 不一定适合从第一步就与 Q/T 全部混在一起做“从零选列”。

---

## 8.1 C0：No-E Prior

训练一个专门的 Q+T 列先验模型：

```text
Reader(Q,T,E=[])
→ tail candidate states
→ C2 同款 MLP
→ b_c
```

训练目标仍然是所有真实候选列上的 column CE。

记：

```text
b_c = prior logit for column c
```

训练时不输入任何 evidence。

这个模型不是为了替代 evidence，而是固定一个“没有 evidence 时模型认为哪列最合理”的 prior。

---

## 8.2 固定 shortlist

由 C0 Prior 生成：

```text
S3(Q,T) = Top3 columns according to b_c
```

规则：

```text
m_column_shortlist = min(3, number_of_columns)
```

生成 train/dev/test shortlist 时：

- 只使用 Prior prediction；
- 不看 gold；
- gold 不在 Top3 时，严禁把 gold 人工插进去。

必须记录：

```text
prior_admission@3
```

即 gold 是否进入 S3。

对于 prior 没把 gold 放进 Top3 的样本：

- downstream verifier 无法修正；
- 最终 joint Hit@1 按错误计；
- verifier 的条件准确率可以单独报，但不能从总分母删除该样本。

---

# 9. Phase C1：PVR-Bundle——先只测试“prior/rerank 分工”，仍然 concat evidence

这是一个很重要的中间对照。

它回答：

> 即使仍然把 evidence 一次性拼接，单纯把“列先验”和“evidence 修正”分开，是否已经比 Flat 模型更合理？

## 9.1 输入

对每个 Q-T：

### Prior feature

```text
h_c^0 = ReaderState(Q,T,[])
```

### Bundle evidence feature

```text
h_c^B = ReaderState(Q,T,[E1,...,En])
```

两者都是 tail candidate 的 open/close 拼接状态。

只对 `c ∈ S3` 做 rerank。

## 9.2 Correction head

先做共享投影：

\[
p_c^0=P(h_c^0),\qquad p_c^B=P(h_c^B)
\]

建议：

```text
P: Linear(2H,256) + LayerNorm + GELU
```

构造 correction feature：

\[
x_c=[p_c^0,\ p_c^B,\ p_c^B-p_c^0,\ p_c^B\odot p_c^0]
\]

再：

\[
\delta_c = MLP_{corr}(x_c)
\]

最终：

\[
g_c=b_c+\operatorname{softplus}(\beta)\cdot\delta_c
\]

其中：

- Prior 模型 C0 冻结；
- `beta` 是一个全局可学习标量；
- `softplus(beta)` 保证 evidence correction scale 非负；
- correction head 只负责对 prior 做修正。

如果 E=[]：

```text
delta_c = 0
g_c = b_c
```

不要对空 evidence 构造一个伪文本“no evidence”。

## 9.3 训练

只在 `prior_admitted=True` 的训练样本上优化 verifier CE：

\[
L_{col}=-\log
\frac{\exp(g_{gold})}
{\sum_{c\in S3}\exp(g_c)}
\]

若多 gold：

\[
L_{col}=-\log\sum_{c\in G\cap S3} softmax(g)_c
\]

Prior 不更新。

训练 evidence 分布使用 Phase B 的：

```text
50% O-O / 50% O-R
```

不要额外加入 No-E；O-R 自己会自然包含空 bundle。

---

# 10. Phase C2：PVR-Separate——真正测试“一次拼接 vs 逐条 evidence 后融合”

C2 与 C1 唯一希望改变的核心，是 evidence 表示方式。

Prior、shortlist、训练 population、evidence IDs、训练 schedule 都保持一致。

## 10.1 每条 evidence 单独跑 frozen reader

对 bundle：

```text
E=[E1,...,En], n<=4
```

运行：

```text
h_c^j = ReaderState(Q,T,[Ej])
```

注意：

- 一个 pass 仍然同时输出 S3 中所有候选列状态；
- **不是为每个 column 单独跑一次 9B reader**；
- 每个 evidence 一个 reader pass；
- 最多 4 个 evidence pass；
- Q/T 会在这些 pass 中重复出现，本轮先接受这个成本并测量它；
- 可以 batch 多个 evidence pass，但不得改变模型语义。

## 10.2 Evidence-item feature

共享投影：

\[
p_c^0=P(h_c^0),\qquad p_{c,j}=P(h_c^j)
\]

对每个 `(column c, evidence j)`：

\[
x_{c,j}
=
[p_c^0,\ p_{c,j},\ p_{c,j}-p_c^0,\ p_{c,j}\odot p_c^0,\ m_j]
\]

其中 `m_j` 是 learned modality embedding：

```text
text/image -> 16 dim
```

不要把 evidence ID、path rank、gold support flag 放入这个 feature。

计算：

\[
z_{c,j}=\phi(x_{c,j})\in R^{256}
\]

\[
a_{c,j}=w_a^\top z_{c,j}
\]

其中 `a_{c,j}` 是“这条 evidence 对候选列 c 的可用支持权重”，但在没有 support auxiliary loss 时，不要把它解释成校准概率。

## 10.3 Set fusion

对固定列 c，在 evidence 维度做 attention：

\[
\alpha_{c,j}
=
softmax_j(a_{c,j})
\]

\[
v_c
=
\sum_j\alpha_{c,j}z_{c,j}
\]

再：

\[
\delta_c=MLP_{corr}([p_c^0,v_c])
\]

最终：

\[
g_c=b_c+\operatorname{softplus}(\beta)\delta_c
\]

### 空 evidence

如果 `n=0`：

```text
delta_c = 0
g_c = b_c
```

不要让 softmax 在空集合上运行。

### 单 evidence

如果 `n=1`：

```text
alpha = 1
```

因此 C2 在单 evidence 时退化成单条 evidence correction。

## 10.4 为什么不用 evidence raw logits 直接 max/LSE

因为本轮想测试的是：

> 模型能否学习“哪条 evidence 对哪一列有用”，而不是手工规定所有 evidence 的作用只通过一个 scalar raw column score。

Phase A 的 Mean/LME 是诊断；C2 是可学习集合融合器。

---

# 11. Phase C 的关键指标：不要只看最终 Hit@1

Prior→rerank 结构必须额外报告以下指标。

## 11.1 PriorAdmission@3

\[
Admission@3
=
P(gold\in S3)
\]

这是 verifier 的理论 admission ceiling。

## 11.2 Correction rate

在：

```text
prior Top1 wrong
AND gold ∈ S3
```

的样本中，reranker 最终变成 Top1 正确的比例。

## 11.3 Damage rate

在：

```text
prior Top1 correct
```

的样本中，reranker 最终把 Top1 改错的比例。

## 11.4 Net correction

同时给 count 和 query-macro rate：

```text
corrected_cases - damaged_cases
```

自然 evidence 真正有用，至少应该表现为：

```text
Correction > Damage
```

而不是“既帮助一批，又伤害差不多一批，最后平均只涨一点”。

## 11.5 EvidenceGain

同一 population：

```text
EvidenceGain_H1 = H1(PVR, O-R) - H1(PriorOnly)
EvidenceGain_MRR = MRR(PVR, O-R) - MRR(PriorOnly)
```

同时在：

- O-R full；
- O-R non-empty；
- M>3 非饱和子集；

报告。

---

# 12. Phase D：Support auxiliary（条件执行，不允许 Codex 自己猜负例）

只有在审计确认存在足够多“明确到 evidence ID 的已审核 witness”后才执行。

最低要求：

```text
>= 500 train Q-T pairs
```

可以明确找到至少一个：

```text
positive evidence ID for gold column
```

否则：

```text
SKIP Phase D
```

并在报告中写“现有监督不足”，不要把 unknown evidence 自动当负例。

## 12.1 正例

`(Q,T,c*,E+)` 只有在已有 recovery review 明确表明：

> E+ 为 gold bridge attribute 提供了至少一个 query row 的可恢复证据

时才是 support positive。

## 12.2 不能使用的“负例”

以下都不能直接标 0：

- O-R 中没有 recovery annotation 的 evidence；
- 与 gold value 没有字符串匹配的 evidence；
- 未被当前人工/模型 review 覆盖的 evidence。

它们都是：

```text
unknown
```

## 12.3 合成负 support

允许使用现有 Shuffled-E donor 规则生成：

```text
E-
```

它只能命名为：

```text
synthetic_negative
```

## 12.4 Support ranking loss

对于已知 E+ 和 synthetic E-：

\[
L_{support}
=
softplus(-(a_{c^*,+}-a_{c^*,-}))
\]

总 loss：

\[
L=L_{col}+0.2L_{support}
\]

`0.2` 固定作为首轮值，不扫 lambda。

Phase D 只比较：

```text
PVR-Separate
vs
PVR-Separate + support ranking
```

其它内容完全相同。

---

# 13. 这一轮暂时不要做“扩大 evidence pool 到 20 再学 selector”

这是后续值得做的方向，但不要和本轮 evidence fusion 混在一起。

原因：

当前 baseline 的 evidence retention 已经是：

```text
dedup -> top20 -> row-coverage -> read4
```

如果这轮同时：

1. 换融合器；
2. 把 pool 从 read4 扩到 top20；
3. 再引入 selector；

即使结果变好也无法知道原因。

只有当：

```text
PVR-Separate > PVR-Bundle
```

或者 support score 的确表现出可用的证据区分能力后，下一轮再做：

```text
pre-retention pool20
→ column-aware cheap selector
→ fixed read budget4
→ PVR
```

届时要保证“昂贵 reader 真正读取的 evidence 数量仍是 4”。

---

# 14. 训练 recipe

## 14.1 Backbone

```text
Qwen reader frozen
eval()
requires_grad=False
```

禁止 LoRA。

## 14.2 Seeds

```text
13, 29
```

Phase A 无训练。

## 14.3 Epoch

Prior / Flat-Mix / PVR head：

```text
20 complete base-sample epochs
```

每 epoch 报：

- base samples seen；
- admitted samples；
- optimizer steps；
- train loss；
- dev metrics。

## 14.4 Optimizer

Prior / Flat MLP：

```text
AdamW
lr = 3e-4
weight_decay = 1e-4
grad_clip = 1.0
effective_batch = 32 Q-T
```

PVR correction/fusion：

```text
AdamW
lr = 1e-4
weight_decay = 1e-4
grad_clip = 1.0
effective_batch = 32 Q-T
```

首轮不扫 LR。

## 14.5 Checkpoint selection

只看 dev。

Prior：

```text
dev No-E query-macro MRR
tie: Hit@1
tie: earlier epoch
```

Flat-Mix：

```text
dev O-R full query-macro MRR
tie: O-R non-empty Hit@1
tie: earlier epoch
```

PVR：

```text
dev O-R full query-macro MRR
tie: O-R non-empty Hit@1
tie: lower Damage rate
tie: earlier epoch
```

test 只对已选 checkpoint 跑一次正式报告。

---

# 15. 主指标与统计

所有 Top1/Topk/MRR 必须来自同一确定性完整排序。

继续报告：

- query-macro Hit@1/2/3/5；
- MRR；
- pair-micro；
- column-count bucket；
- `M > k` non-saturated subset；
- modality；
- evidence_count；
- O-R empty/non-empty。

## 15.1 R2 主要比较口径

重点看：

```text
O-R non-empty:
Hit@1
MRR
Correction rate
Damage rate
Net correction
```

O-R full 仍然必须报告，因为它更接近真实系统供给。

Hit@2/3 继续报告，但不要因为其接近饱和就作为唯一模型选择依据。

## 15.2 Bootstrap

两个 seed 先对同一个 Q 的指标取平均，再按已有 `source_table_id/source group` 做 cluster paired bootstrap：

```text
10,000 replicates
95% CI
```

主要差值：

- Flat-Mix - R1-C2；
- PVR-Bundle - Flat-Mix；
- PVR-Separate - PVR-Bundle；
- PVR-Support - PVR-Separate。

不要只报绝对分数。

---

# 16. 计算成本必须报告

PVR-Separate 不属于“同算力无代价替换”。

每个 Q-T 记录：

- number of reader forwards；
- total reader input tokens；
- total image pixels processed；
- p50/p95 latency；
- peak VRAM；
- cache size；
- evidence_count；
- batch size。

预期：

- Bundle：约 1 次 evidence reader forward；
- Separate：最多 4 次 evidence reader forward，加一个可缓存的 No-E Prior feature。

Prior feature `h^0` 对同一个 Q-T 可以缓存，不应每个 epoch 重跑 frozen 9B。

如果 Separate 更准但贵很多，报告 cost-quality tradeoff，不要只报准确率。

---

# 17. Phase A/C 中严禁的实现捷径

Codex 不得：

1. 用 gold 在多个 evidence 中选“最好的一条”作为正式预测；
2. gold 不在 Prior Top3 时偷偷插入 gold；
3. O-R 为空时替换成 O-O；
4. 把 unknown natural evidence 当 support negative；
5. 为了让 PVR 好看而只评价 admitted samples；
6. 只报告条件准确率，不报告 joint accuracy；
7. 因为 Hit@3=100% 就声称 Stage2 join 已解决；
8. 把 evidence attention 权重直接解释为可信概率；
9. 把 PVR-Separate 称为“效率优化”；
10. 扩大 evidence pool 或改变 Stage1 retention 后仍声称只比较 fusion；
11. 使用 test 选择 m_column_shortlist；
12. 因为某个非 gold 列能恢复值就自动改标签。

---

# 18. 本轮最终需要回答的五个问题

最终报告必须逐条回答：

## Q1. R1 的自然 evidence 增益小，主要是不是训练分布不匹配？

看：

```text
Flat-Mix vs R1-C2
```

## Q2. “Q/T 先验 + evidence 修正”是否比“一开始就把所有东西混在一起选列”更合理？

看：

```text
PVR-Bundle vs Flat-Mix
```

## Q3. 一次性 concat evidence 是否真的是瓶颈？

看：

```text
Phase A
+
PVR-Separate vs PVR-Bundle
```

只有这两部分都支持，才能说 concat 是明确瓶颈。

## Q4. Evidence 到底是在“纠错”还是“制造等量新错误”？

看：

```text
Correction rate
vs
Damage rate
```

## Q5. 如果 evidence 依然没有稳定净增益，怎么办？

不要继续无止境调 evidence fusion。

如果最终出现：

- Prior Top3 admission 很高；
- PVR 对 Hit@1/MRR 没稳定净增益；
- Correction ≈ Damage；

则下一步系统应改为：

```text
Q+T Prior
→ 保留 Top2/Top3 columns
→ 对每个 column hypothesis 进入真正的 row-level evidence localization
→ 属性恢复
→ semantic join verification
```

也就是说：

> **evidence 的主要价值可能不在“决定是哪一列”，而在“给已经合理的列假设恢复每一行的属性值”。**

这也是一个完全有效的实验结论。

---

# 19. 实验执行顺序

严格按照下面顺序：

```text
Step 0  verify R1 artifacts / hashes / population
Step 1  Phase A dev-only inference diagnostics
Step 2  build frozen train O-R
Step 3  train/eval Flat-Mix
Step 4  train No-E Prior
Step 5  freeze Prior Top3 shortlist
Step 6  train/eval PVR-Bundle
Step 7  train/eval PVR-Separate
Step 8  audit support-label availability
Step 9  if >=500 valid train pairs: PVR-Support
Step 10 lock model selection
Step 11 run formal test once
Step 12 produce report + raw predictions + manifests + cost audit
```

不要因为某个中间结果“不好看”就跳过后续已经规定的核心对照。

Phase D 是唯一条件执行项。

---

# 20. 必须交付的 artifacts

```text
R2/
  README.zh-CN.md
  EXECUTION_MANIFEST.json
  SOURCE_AUDIT.json
  R1_BASELINE_REPLAY.json

  NATURAL_TRAIN/
    FROZEN_NATURAL_TRAIN.jsonl.gz
    MANIFEST.json

  PHASE_A/
    per_evidence_predictions.jsonl.gz
    leave_one_out.jsonl.gz
    order_perturbation.jsonl.gz
    RESULTS.json
    RESULTS.zh-CN.md

  PRIOR/
    checkpoints/
    predictions/
    metrics/
    shortlist.train.jsonl.gz
    shortlist.dev.jsonl.gz
    shortlist.test.jsonl.gz

  FLAT_MIX/
    checkpoints/
    predictions/
    metrics/

  PVR_BUNDLE/
    checkpoints/
    predictions/
    metrics/

  PVR_SEPARATE/
    checkpoints/
    predictions/
    metrics/

  PVR_SUPPORT/          # only if executed
    ...

  COST/
    reader_costs.json
    cache_costs.json

  ANALYSIS/
    paired_bootstrap.json
    flip_analysis.json
    subgroup_metrics.json
    error_cases.md

  FINAL_REPORT.zh-CN.md
```

每个 checkpoint manifest 至少记录：

- git/source hash；
- data manifest hash；
- reader layout；
- reader model identity；
- train condition schedule；
- seed；
- selected epoch；
- optimizer；
- learning rate；
- parameter count；
- input feature/cache hashes。

---

# 21. FINAL_REPORT 的结论约束

报告必须区分：

```text
planned
implemented
actually executed
actually evaluated
```

并明确写出：

- 哪些实验没有执行；
- 为什么没有执行；
- 哪些比较是严格受控比较；
- 哪些只是诊断；
- 哪些指标属于 oracle upper bound；
- 哪些结果只适用于 EntiTables implicit positive target column localization；
- 哪些结论不能外推到最终 Stage2 join success。

不要为了维护既有论文故事，把没有显著收益的实验解释成成功。

