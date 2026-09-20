# MMDD QCPATH-R1：受限条件化第二跳与真实路径 Teacher

**版本：1.0｜日期：2026-09-19｜性质：预先固定的执行规范，不是已完成实验报告。**

> **给执行模型的最高优先级要求：必须逐条遵守本文全部 MUST / MUST NOT / STOP 约束。不得根据个人判断“简化”“等价替换”“先跑起来”“为了效果更好”而修改训练、标签、输入、候选、评分或验收规则。没有执行的步骤必须标为未执行；缺少必要输入必须明确阻断，不得伪造、降级或静默回退。**
>
> 本文授权的是一个**有旧健康 parent 的局部可行性实验**，不是完整 fresh 重建。所有新增训练组合从原始 train 标注重新构造，不读取旧训练列表。历史模型仅按指定角色复用。**不允许把局部成功写成 fresh 成功，也不允许把 QT-only 改成主方法。**

---

## 0. 阅读顺序、规范效力与唯一允许的执行范围

### 0.1 必须先完整阅读，后编写或执行训练代码

依次读取：`EXPERIMENT_SPEC.zh-CN.md`（本文）→ `protocol.json` → `ACCEPTANCE_CHECKLIST.md` → `reference/contracts.py` 与测试 → `CODEX_PROMPT.md`。正文和配置互相约束；发现冲突必须 `BLOCKED_SPEC_CONFLICT`，不得自行决定取哪一个。参考代码只覆盖数学和规则内核，不是完整仓库实现。

原对话、旧实验文档、旧源码默认值，只能提供背景或查找资产。**本版明确规定的内容覆盖旧建议**，包括：同时采用全目标分母与受限残差、先 seed13、Teacher 在 A 门槛通过后才启动。不能恢复旧文档中的“先只改分母”“固定 Top32”“所有四个 adapter 作业直接跑完”等指令。

### 0.2 目标不是全面归因，而是获得可取舍的有效配方

本轮要回答两个顺序问题：

1. **A：**在静态目标向量、固定第一跳下，受限的 `(Q,E)→T` 是否比 BASE 与参数量匹配的 E-only 得到更好的实际第二跳候选？
2. **B：**仅当 A 通过，读取真实 `Q–E–T` 的 Teacher，能否在同一候选集上接近强 QT 对照，并交付 evidence-dependent 的实际增益？

A 同时修正监督竞争范围、残差限制和标签保护。因此 A 与旧失败实验的差异不能解释为单个修正的独立因果效应。**本轮唯一关于 Q 增量的训练对照，是新配方下的 A-QE 与 A-EONLY。**

### 0.3 执行状态机：不得越过门槛

```text
S0 输入、实现与数学验收
   ├─失败 → BLOCKED，保存缺项，不启动训练
   └─通过 → A-EONLY seed13 + A-QE seed13（最多各3 epoch）
               ├─A门槛失败 → STOP_NO_A_GAIN，结束本轮
               └─A门槛通过 → B-QT-CONT seed13 + B-QET-PATH seed13（各2 epoch）
                                  ├─B门槛失败 → STOP_NO_PATH_VALUE，结束本轮
                                  └─B门槛通过 → A两个分支seed29重复
                                                     ├─A29失败 → STOP_NOT_REPLICATED
                                                     └─A29通过 → B两个分支seed29重复
                                                                        └─汇总，结束
```

最多 **8 个正式训练作业**：A 4 个、B 4 个。最低只有 A seed13 的 2 个。实际重复的 epoch 不得突破预算。正确性单步探针不作为正式训练产物。

**本轮不授权 KD、完整 fresh 重建、Stage2、新特征方案、200K 训练、超参数扫描或追加训练。**这些工作只可在 `NEXT_DECISION.md` 中列为未执行，不得自动启动。这样避免“第一步刚有信号就顺便训练完整系统”。

### 0.4 不可更改清单

| ID | 必须保持的要求 |
|---|---|
| H01 | 冻结 Qwen3-VL-Embedding-8B；本轮不新增 Qwen 前向，不更新 backbone。 |
| H02 | Student 每个目标一个静态向量；只允许 Q→T 与 Q→E→T。 |
| H03 | A 只训练新增 adapter；冻结全部原 P/R、Direct、Q→E 和目标索引。 |
| H04 | A 全目标竞争，不得用 Top32、in-batch-only、随机子集或 sampled softmax 替代。 |
| H05 | A 残差相对上限 ρ=0.5；不是最终查询向量单位化。 |
| H06 | Teacher 正式路径分数必须来自真实 Q/E/T 输入；QT-only 仅对照。 |
| H07 | 不输入 GT join column、GT 恢复值、对象 ID/source ID/检索排名作为模型特征。 |
| H08 | 在线不读取 implicit/explicit、witness 标注或正目标集合路由。 |
| H09 | Unknown 是 assumed competitor，不称 confirmed negative；保护所有按本文定义的 train-known positives。 |
| H10 | 不加载历史训练列表、路径训练图、负例列表或 Teacher 分数作为新增训练数据。 |
| H11 | 不改已有 production Evidence 排序与 Equal admission；它们只分配候选预算，不代替最终 Teacher。 |
| H12 | 不用 weighted RRF、Q-additive、QT+Path 插值、固定低 evidence 权重或手工提权救结果。 |
| H13 | 不因缺特征删候选、删 query、删模态、复制 BASE 成两个 seed 或用假分数补齐。 |
| H14 | 不重做 CLEAN-R1 的 P/J 双任务或固定八槽缓存；不增加多个 Teacher 主干/输出头。 |
| H15 | 准确区分 planned / implemented / executed / evaluated；不得把测试通过称为真实训练完成。 |
| H16 | 失败就按状态机停止；不得以“还没解释完原因”为由新开训练分支。 |

---

## 1. 决策依据及证据边界

### 1.1 已有事实，不能在报告中歪曲

历史 `INTEGRATED_REVIEW.md` 记录：BASE 的条件化 ET exact R@10 为32.692%，E-only为0.433%，QE为0%；机制人口为939个已验证(Q,E)、462个implicit query，完整开发集为1198个query。QE残差/base范数比中位数约48倍；同一未进入14592条旧训练列表的目标，成为939个pair共同的exact第一名。它支持训练竞争范围与残差失控的风险，**不证明这两个因素是唯一根因**。

历史方案已使用零初始化输出层、冻结原 P/R、静态索引。因此不能把这些旧条件重新宣传成本轮的新贡献。

后续 EXP3 已做 Teacher target-level path 训练；不能再写“Teacher从来没有做path训练”。本轮改变的是**条件化三对象评分**，不是简单为原QE+ET再次追加epoch。

### 1.2 本文新增的预定工程选择

ρ=0.5、A门槛、B的4096-query上限、两epoch短程训练、支持对比权重等，是本轮预先选定的工作点，不是已验证最优值，不得写成既有事实。完整来源与本次检查范围见 `SOURCES_AND_BOUNDARIES.md`。

### 1.3 本轮不能宣称什么

不能宣称本轮从fresh获得B13级系统；不能宣称只改善ET就是最终方法成功；不能宣称Q敏感就是正确利用Q；不能宣称错误证据替换必然形成已确认负例；不能宣称sum-probability已经解决多正例Recall一致性；不能把dev探索置信区间当成未见测试集上的确定结论。

---

## 2. S0：锁定数据、模型、源码与资源

### 2.1 必须输出 `RESOLVED_INPUTS.json`

以下每个角色必须记录真实绝对路径、文件大小、SHA256、生成来源、用途、是否允许训练读取。路径必须由服务器现存内容定位，不能把报告里的路径当作已经存在。

| 角色 | 唯一解析规则 | 缺失处理 |
|---|---|---|
| `dataset_root` | 原QC-ET/B13实验同一20K规模数据快照，保留原split和对象库；表数可能不恰好20,000 | BLOCKED_DATASET |
| `raw_train_annotations` | 原始train qrels、query属性状态、query-specific witness/recovery标注 | BLOCKED_LABELS |
| `dev_annotations` | 原开发集，单独由评测进程读取 | BLOCKED_EVAL |
| `student_base` | 2026-09-16 QC-ET的INPUT_LOCK所指R27 B13 exact-replay最终健康Student，核验权重而非只看文件名 | BLOCKED_BASE；不得替换成C1@659或CLEAN-R1 |
| `frozen_object_embeddings` | 与BASE相同的原Qwen对象编码及prompt/tokenization/对象内容指纹 | BLOCKED_FEATURES |
| `target_vectors/index` | BASE生产目标向量与索引；源模态text/image必须指向同一静态目标空间 | 不存在实体但全部冻结输入齐全时允许用原构建器重建一次，见2.3 |
| `base_first_hop` | BASE完整Direct及20text+20image第一跳结果；可合法重算，但不能换算法 | BLOCKED_RETRIEVAL |
| `production_functions` | 原QC-ET调用的Evidence排序、retention、Equal、过滤与tie-break函数 | BLOCKED_PRODUCTION_RULES |
| `teacher_parent` | 原强T0：R22 T1-B seed13，历史SHA256 `ab0e3c3f85f006d2fdc4ba5194a0021680ab8fa1341441cb8eb003410ded68cc` | 只阻断B，A可以完成；不得用其他Teacher冒充 |
| `teacher_feature_store` | T0兼容的原对象细粒度特征和对象全局z，非CLEAN-R1新摘要 | 只阻断B；不得全湖重跑Qwen |

历史数量22,886目标、1198 dev query、599/599 implicit/explicit、939 probe pair、14592 train pair，仅是身份核对线索。**本轮重新枚举后数量变化必须解释，禁止为凑这些数抽样、复制、过滤。**不一致由输入或标签规则差异引起时记录；不能在未确认同一数据快照的情况下直接比较历史百分比。

### 2.2 最小源码锁定

记录仓库commit与dirty diff；记录实际导入的base模型、adapter、数据构造、loss、检索、admission、Teacher和评测模块路径及hash。使用小的 `SOURCE_LOCK.json` 即可，不做全仓库重构或全盘逐文件哈希。

`PRODUCTION_CONTRACT.json` 必须列出：每个保留函数的qualified name、签名、源文件hash、全部非默认实参、score space、retention规则、dedup key、排序/tie规则、候选来源顺序。**“沿用旧代码”四个字不构成规范；这些字段未解析完不得开始正式训练。**

### 2.3 索引身份与查询侧身份分开

`target_index_identity` 由目标ID顺序、目标向量字节hash、metric、维度、原HNSW参数和插入顺序决定；`query_model_identity` 由BASE与本轮adapter hash决定。adapter更新只能改变后者。

若原index文件缺失但原对象和配置齐全，可在A训练前重建一次：所有A分支共用该实例；记录这是新index实例，重新评BASE，不能强迫其逐ID等于历史ANN。不能在训练后为QE单独重建索引或调efSearch。

若原Student实际是低秩扩维index或text/image目标向量不同，而非本文full-R统一d维空间，则 `BLOCKED_UNSUPPORTED_BASE`。不得临时迁移模型结构。

### 2.4 资源合同

使用用户提供的A100；不自动换L40，不安装/升级核心库，不调用外部付费API。保存GPU型号/显存、torch/CUDA版本、可用RAM与磁盘。训练使用原环境，不把版本记忆当实际环境。

本轮新增实验目录上限100 GiB；保持文件系统至少200 GiB剩余空间；不复制完整已有Qwen缓存，不删除用户旧数据。A只保存所需冻结向量、排名、adapter；B仅缓存本轮实际用到的T0压缩对象，禁止完整hidden states复制。

若预估超预算，输出逐项字节估算并 `BLOCKED_RESOURCE`。不得缩小目标库、丢image或缩减候选充当资源优化。

---

## 3. 标签合同：所有训练组合重新生成

### 3.1 唯一数据根

从原始train标注重新建立：

- `G[q]`：全部train-known正确目标。
- `D[q]`：原始标注明确支持已有属性直接连接的目标。没有独立direct标注时，只允许使用原数据生成器已记录的explicit-query/direct事实，不能从QT分数推断。
- `W[q,t]`：原始query-specific标注确认提供连接所需信息的evidence ID集合。
- `Epos[e]`：从所有train-only W和原始无条件ET正标注取并集的已知关联目标。
- `P[q,e] = {t : e∈W[q,t]}`：当前条件下的正集。

`D[q]⊆G[q]`，全部G和W引用的目标、query、evidence必须能够解析；与原合法目标范围冲突必须阻断，不能静默删正例。

每条W记录保存原始文件定位、对应query/target/evidence、模态及标注等级。数据集原来使用何种自动审核等级就保留其术语；不得一律称人工金标。

**禁止来源回退：**当前Q/T没有witness时，不得用其他Q同一个T的evidence、同源图片、同URL或旧训练graph替代。它们可作为unknown候选，不能直接充当当前路径正例。

一条E支持某些example rows，并不证明它独自足够完成整张query的join。本轮W解释为“提供正确连接信息/路径贡献”，最终完整恢复仍属于Stage2。

### 3.2 全目标训练的 P / I / N 三分法

对每个非空正集的(q,e)：

\[
P_{q,e}=P[q,e],\quad
I_{q,e}=(G[q]\cup Epos[e])\setminus P_{q,e},\quad
N_{q,e}=\mathcal T_q^{legal}\setminus(P_{q,e}\cup I_{q,e}).
\]

其中 \(\mathcal T_q^{legal}\) 使用BASE原来的内容合法性/自表排除规则；不能读取dev/test标注。合法库中的所有目标都必须有静态向量。

- P：正例，进入分子和分母。
- I：已知相关但当前路径未确认；完全退出这个loss的分子分母。
- N：按约定使用的assumed competitors，不称确认负例。

若原标注有显式条件负例，只有不与P/I冲突时才放入N；冲突记录并阻断相应数据构造，不能静默按“正例优先”掩盖矛盾。

`P⊆legal`；`P,I,N`两两互斥；`P∪I∪N=legal`。合法范围外不能作为负例。“全目标分母”指完整合法库经过上述固定mask，而不是无视known positives。

**E-only与QE使用完全相同的P/I/N。**E-only看不到Q，是有意的信息移除对照；不替它重新定义P_E。这种设计不会把同E不同Q的未标注目标武断标负，也不保证有足够监督强迫Q选择；报告必须保留该限制。

### 3.3 去重、空值与训练人口

同一(q,e,t)重复标注合并；同一(q,e)只形成一个训练item，P含全部目标。P为空时不生成A训练item。N为空时 `inactive_no_competitor`，两臂同样排除并记录数量。

空列表`[]`、字段缺失、`null`语义不同。数据结构必须显式保存状态：`positive_ids: []`不得在prefetch/compact/expand后被省略并回退到G[q]。

文本和图像共用一个adapter，不按模态建立独立模型；模态由冻结BASE的u_E与v_E区分。全部合法已标注训练pair都参与A；不只选historical strict EO或容易pair。

输出：`train_pairs.jsonl.gz`、`label_counts.json`、`label_conflicts.jsonl`。这些是本轮新产物，不是旧list改名。A不需要存N的完整ID列表：固定table_ID顺序加稀疏P/I索引可精确恢复。

### 3.4 严格隔离评测标签

训练进程只获train label对象；dev标注仅由评测入口读取。目标库允许包含同一合法湖中供dev/test检索的对象内容，这属于既有transductive corpus设置，**不允许读取其评测相关性标签建立训练mask**。

本轮只用原dev作开发取舍，不读取test成绩。报告明确“开发集局部验证”；不宣称未见test泛化。原表正常列名与cell内容保留，不因恰好与GT字符串相同而删除；禁止的是额外注入标注字段、ID或答案。

---

## 4. A模型：唯一允许的公式

### 4.1 冻结向量的行向量约定

冻结BASE得到：

\[
u_q=P_{tab}(z_q),\quad u_e=P_{\tau(e)}(z_e),\quad
u_t=\mathrm{BASE.index\_vector}(z_t),\quad
v_e=\mathrm{BASE.relation\_query}(z_e,\tau(e),table).
\]

实际代码必须调用BASE投影/索引函数获得相同输出，不能绕过其已有normalization或role处理后再声称相同。对full-R行向量，`v_e = u_e @ R[e_type,table]`，不是`u_e @ R.T`；列向量书写等价为\(R^\top u_e\)。

原打分：`base_score = v_e @ U_T.T`。用非单位、非对称R测试；不能只测identity-R。

### 4.2 新adapter

\[
h_{QE}=[u_q;u_e;u_q\odot u_e;|u_q-u_e|],\qquad
h_E=[0;u_e;0;0],
\]
\[
\Delta_\theta(h)=W_2\operatorname{GELU}(W_1h+b_1)+b_2.
\]

`Linear(4d,256)→GELU→Linear(256,d)`；bias均启用；无dropout、无LayerNorm、无额外embedding、无gate。d从BASE读取，预期1024。W1/b1用固定torch初始化；W2/b2严格置零。每seed先初始化一个模型state，逐tensor复制给E-only与QE，不能依次随机初始化两个模型。

### 4.3 受限残差

\[
a=\min\left(1,\frac{0.5\|v_e\|_2}{\max(\|\Delta\|_2,10^{-12})}\right),
\quad \bar\Delta=a\Delta,
\quad v_{qe}=v_e+\bar\Delta.
\]

保留a对Δ的梯度；不得detach a。norm在float32计算。v_e范数小于等于1e-12视为无效输入并阻断，不能自动补随机方向。Δ=0时输出必须等于BASE，且对W2的初始梯度必须可达。

严格禁止：在合成后unit-normalize、裁剪最终分数、给Δ单独加经验权重、让ρ可训练、ρ扫描、额外norm penalty、Uniform、QT-additive、冻结W2导致完全不学习。

### 4.4 打分空间

\[
\ell(q,e,t)=v_{qe}^\top u_t.
\]

训练温度固定1.0；直接用raw inner-product。不得调用B13的`10*sigmoid(raw)`监督变换，不得继承CLEAN-R1的0.07，也不得为了“概率”先sigmoid再softmax。

检索、exact评测、ANN查询必须调用同一个`make_second_hop_query`；无train/eval两套投影公式。

---

## 5. A训练：完整目标分母及确切归约

### 5.1 单item损失

\[
L_i=\operatorname{LSE}_{t\in P_i\cup N_i}\ell_i(t)
-\operatorname{LSE}_{t\in P_i}\ell_i(t).
\]

保留sum_probability，不在本轮同时更换为逐正例pairwise损失。它会偏向容易正例，该限制报告即可，不为此新增分支。每个正例的分子必须存在，不因旧Top32缺席而漏掉。

### 5.2 完整分母、分块规则与性能

整个合法目标矩阵常驻GPU可用时，直接`V_QE @ U_T.T`。目标库过大时只允许**数值等价的chunked全分母**：块内LSE，再对块LSE做logaddexp；分子同样全局归约。训练时必须保留adapter梯度，不能先detach全部分数。

**不得逐块算softmax/CE后平均，不得用“每块包含正例的部分”当完整loss。**参考测试覆盖无正例块、ignore目标、多个正例分散于不同块和梯度一致性。

不保存全epoch全部score矩阵；每个step释放。无需哈希排序全部目标；table_ID排序只做一次，P/I查表用稀疏索引。禁止逐query对22K或200K对象执行SHA256全排序。

### 5.3 query-macro训练权重

令训练有Q_tr个不同query，N_pair个有效(q,e)，q对应m_q个item。固定权重：

\[
w_i=\frac{N_{pair}}{Q_{tr}m_{q_i}}.
\]

这样一整个epoch中每个query总权重相等。每个logical batch含b个item时：

\[
L_{batch}=\frac1b\sum_{i\in batch} w_iL_i.
\]

不得除以当前batch的权重和；不得把分母变成正例个数；不得把文本/图像各平均后再无记录地重加权。

### 5.4 顺序：每个item每epoch一次

先按UTF-8对象ID排序建立规范item表。每个epoch用独立 `random.Random(seed_token)` 对evidence组排序、组内query排序；对各组round-robin依次取一个剩余item直到耗尽。每个item恰好一次、不重复、不丢尾batch。它让同一E的不同Q分散进入更新，避免整段只训练某个条件；总曝光均衡由5.3的query权重定义，不声称每个E曝光相同。

seed_token由一次SHA256(`QCPATH-R1|A-order|seed|epoch`)前8字节得到整数。不得用Python内置hash，亦不为每个候选生成hash。

同seed两臂共用完全相同的epoch item顺序、P/I、权重和batch边界。数据预取不得改变消费顺序。

### 5.5 优化器和预算

| 参数 | 固定值 |
|---|---|
| seed顺序 | 13先；29仅在A13+B13均通过后 |
| epoch | 3；主终点固定epoch3，不选历史最高dev checkpoint |
| logical batch | 128个(q,e)item |
| microbatch | 32个item；尾项按实际b归约 |
| optimizer | fresh AdamW，仅adapter参数 |
| lr / betas / eps / weight_decay | 1e-4 / (0.9,0.999) / 1e-8 / 0.01 |
| scheduler / warm-up | 无 / 无 |
| clip_grad_norm | 1.0，在全部microbatch累计后、optimizer.step前一次 |
| 数值 | 参数、norm、matmul、loss均float32；关闭autocast与TF32 |
| KD / anchor / Uniform / 其他loss | 全部0 |

每个microbatch反传 `sum(w_i*L_i)/logical_b`；不是各自mean后平均。每个logical batch只zero_grad一次、step一次；不能把microbatch变成更多optimizer updates。

保存epoch0/1/2/3 state、optimizer、RNG、数据顺序位置；每个epoch完成后写真实统计。参考历史14592 pair时每epoch114updates、总342；本轮实际N变化时用`ceil(N/128)`，不能通过重复/截断强凑342。

### 5.6 安全停止，不以低loss替代效果

NaN/Inf、P/R更新、索引改变、ρ约束超差、关键样本丢失立即 `INVALID_RUN`。

epoch1/2仅做固定exact probe与冻结参数校验。任一arm的probe ET R@10比BASE低超过5个百分点，或其probe共同Top1占比超过25%，标`STOP_A_CATASTROPHIC`，两臂结束本轮；不恢复BASE作为该arm训练结果，不启动Teacher。这个门槛是资源止损，不是过拟合结论。

其余情况完成3epoch，不因epoch1没涨就自行停，不因epoch2最好就替换epoch3。

---

## 6. A评测：机制人口、自然检索与严格EO

### 6.1 冻结人口

开发集使用原全部query；预期1198，但实际以锁定split为准。机制probe：对每个dev q，只使用BASE第一跳自然召回、且原dev W中有当前正目标的(q,e)。probe在任何A训练开始前冻结；后续不按新模型召回/成功筛pair。N_probe为0则阻断机制评测，不能改用训练集。

ET Recall@K必须先每个pair求`|P∩TopK|/|P|`，再同query对pair求均值，最后对有probe的query宏平均。分别text/image；explicit没有适用probe时填NA与人口0，不填0%。

### 6.2 exact / ANN

所有exact排名使用相同合法目标矩阵、raw dot、分数降序、target ID升序tie-break。记录每个正目标的rank和Top50，不保存全湖分数。

A epoch0/3对probe做exact和ANN；epoch1/2只做exact。最终完整dev自然检索走原固定ANN，复用BASE Direct100和第一跳20text+20image，每个E扩20目标。不能为了新模型扩大per-E预算。

ANN只对返回候选做相同tie-break；原ANN没返回的并列对象不能伪装成exact一致。报告approximation差异；不得调参消除差异。

### 6.3 完整dev的四个候选对象

- `D_q`：BASE Direct100；两A臂完全相同。
- `E_q^a`：arm a全部自然第二跳Top20目标并集，经原production Evidence函数得到有意义的target排序。
- `U_q^a=D_q∪E_q^a`：集合诊断，无排序含义。
- `C_q^a=production_Equal(D_q,rank_E^a)[:100]`：固定预算Teacher入口。
- `M_q^a`：BASE direct ANN加深到恰好`|U_q^a|`个不同合法目标的matched direct；另保存exact matched direct作为诊断。只改变返回K，不改变efSearch等配置；无法返回足够合法ID时显式报告不足，不能拿exact补ANN主表。

**不能用QT-over-E的排序冒充Evidence排序，也不能把按ID排序的第二跳集合当E R@10。**若production Evidence函数不能解析，A正式自然评测阻断。

### 6.4 Raw Qwen与B13参考

raw Qwen使用同一冻结z和raw内积，独立生成D/E/U/C/M，使用相同20/20/20预算与同一production admission语义；已有纯编码可复用，不训练raw。必须做当前同协议评测，历史数字仅附录。

BASE即本轮锁定B13 healthy reference，只评一次，不复制成seed13/29两份独立训练结果。F-P等其他Student不加进本轮训练矩阵；可单独列历史参考，不混均值。

### 6.5 严格证据增量：双排除、固定cohort与新增cohort分开

\[
Z_q=G_q\setminus(D_{q,ANN100}\cup D_{q,exact100}).
\]

每个arm自己的strict discovered集合为`Z_q∩E_q^a`。冻结BASE cohort：`F_q=Z_q∩E_q^BASE`，用于固定分母的保留率。新增目标是`Z_q∩(E_q^a\E_q^BASE)`；丢失是`F_q\E_q^a`。不能只挑含strict target的pair后统计其全部正例。

历史207或213不写死；本轮从当前锁定BASE重算并保存。与历史集合分别命名，不合并。

### 6.6 必报指标

完整dev分别overall/implicit/explicit：D raw覆盖、E R@10/20与raw覆盖、U覆盖、C100覆盖、M覆盖、Student本身的C100候选准入排序R@10/20/50、strict新增/丢失及固定F保留。A阶段这些是Student/候选指标，**不能称最终Teacher路径结果**。

probe分别query-macro ET exact/ANN R@1/10/20/50。Top1集中率固定定义为`max_t count(pair的Top1=t)/全部probe pair数`，这是pair人口诊断，不冒称query-macro。正例深位次诊断：每pair先取其全部正例rank中位数，同q对pair值取均值，再报告q级值的均值/p50/p90。

ρ裁剪前/后范数比p50/p95和裁剪激活比例，最终在完整dev自然第一跳的全部(q,e)上计算，分别all/text/image，分母和缺项显式列出；不扩展成大量SVD或原因搜索。

A阶段不以QT Teacher分数选模型，不必跑QT Teacher来决定门槛。B阶段才建立正式最终R@10主表。

---

## 7. A门槛：一次计算，禁止事后改阈值

所有阈值使用fraction；0.01=1个百分点。以epoch3完整正式评测为准。所有gate所需指标必须是有限数且在[0,1]，NA/缺失不能转成0或跳过该条件。浮点阈值比较仅允许1e-12的数值容差，不允许可调epsilon。

| 条件 | 必须满足 |
|---|---|
| 完整性 | 两arm完成3epoch、全部correctness通过、无未解释人口变化 |
| QET确有局部收益 | `ET_exact_R10(QE) − ET_exact_R10(BASE) ≥ 0.010` |
| Q不是多参数的替身 | `ET_exact_R10(QE) − ET_exact_R10(EONLY) ≥ 0.005` |
| 隐式自然覆盖收益 | `U_implicit(QE) − U_implicit(BASE) ≥ 0.005` |
| 整体不退化 | `U_overall(QE) − U_overall(BASE) ≥ −0.0025` |
| 显式保护 | `U_explicit(QE) − U_explicit(BASE) ≥ −0.005` |
| 预算入口保护 | `C100_overall(QE) − C100_overall(BASE) ≥ −0.005` |

全部满足才`PASS_A`。这些是本轮工程取舍门槛，不是统计显著性断言。仍报告source-group bootstrap与W/L/T，不因为CI跨0就改阈值或增加seed先求翻盘。

A不过：完成A报告，停止。A过：固定QE epoch3为本seed的`S_cond`，进入B。不得根据某个Teacher随后表现改选A epoch1/2或E-only。

---

## 8. B前置：固定候选与真实路径，阻止候选变化冒充重排收益

### 8.1 B的主评测环境

seed13：A-QE seed13 epoch3自己的dev C100与retained真实路径。所有B scorers共享逐ID相同C100、相同路径数量、同一E内容。seed29：主own环境换成A-QE seed29；同时在seed13固定环境给出共同池复评，区分检索随机性与Teacher随机性。

BASE/raw own环境额外只在最终checkpoint各评一次相同B-QET与冻结T0，用于raw Qwen/B13/新Student同Teacher比较。它们不用于选epoch或改变门槛。

B训练环境仅在A13通过后从**原train query**新生成：Direct、第一跳与A-QE13第二跳，使用相同生产U/C/retention。A29/B29重复时复用这份B train pack，不根据A29重新挑训练候选；报告这是一份固定生成器训练环境，不是完全独立fresh数据谱系。

### 8.2 直接路径与evidence路径的唯一表示

对C100中每个t，都建立一个可评分的零跳候选`(q, EMPTY, t)`，不读取GT决定是否添加。它是原任务合法的0-hop路径类型，不等价于给所有implicit正目标标direct正例。

evidence路径只来自锁定production retained的真实(q,e,t)。同一个路径的multiplicity按production记录保留；相同三元组只前向一次、用整数m计入聚合，不能因重复优化改变数学含义。最大自然E ID数由20text+20image限制；不添加新的Teacher Top4/Top8/TopK。

**无自然evidence的t仅有零跳候选。不能复制一条E补位置；不能因为GT说implicit而把它删掉；不能按候选来源决定禁用/强制evidence。**

### 8.3 最终路径聚合

\[
f_0=f_\theta(q,\varnothing,t),\qquad f_e=f_\theta(q,e,t),
\]
\[
\boxed{S_\theta(q,t)=\log\!\left(\exp f_0+\sum_{e\in\mathcal E(q,t)}m_{qet}\exp f_e\right).}
\]

温度1.0，无归一化路径数、无额外QE分数、无QT插值、无阈值、无source权重。空E时S=f0。重复路径等价于对唯一f_e加`log(m)`后做LSE，m为正整数。

**零跳分数也必须由同一个正在训练的Teacher、同一head产生。禁止把外部冻结T0分数接到这里。**最终如果模型基本靠零跳获分、真实E没有效果，就按B门槛判失败，不通过降低零跳权重补救。

---

## 9. B模型：保留强T0的既有全局信息，不暗中删掉它

### 9.1 必须明确的实现事实

现有强T0并非纯`Transformer→Linear`：R19代码使用`scoring_head(local_REL + global_residual(Q,T))`，全局z分支与局部交互共用一个head。**本轮不能一面声称复用强T0，一面丢掉其global分支，或把512→512→1的既有head改成新线性层。**

本轮采用唯一的兼容扩展，仍然只有**一个共享Relation Transformer、一个共享既有scoring_head**；不新增第二个Teacher、P/J任务头或evidence gate。

### 9.2 确切前向

先使用T0自身的对象压缩器得到role-neutral局部token：`C_Q, C_E, C_T`；保留原schema/row kind、modality处理。Q的每个已有example row需有独立token组，不新增截行/跨行混合。若旧缓存本身缺行，不得伪称保留；本轮不重新编码全湖，阻断B并说明。

原全局对象表示：`g_x = LN_x(global_adapter_x(z_x))`。

原全局关系表示保持：

\[
G(q,t)=W_o\operatorname{GELU}\!\left(W_i[g_q;g_t;g_q\odot g_t;|g_q-g_t|;e_{tab,tab}]+b_i\right)+b_o.
\]

e非空时，输入序列为：

```text
[REL + type_pair(table,table)],
C_Q + modality(table) + role_query,
[SEP],
[g_E; C_E] + modality(type(E)) + role_bridge,
[SEP],
C_T + modality(table) + role_target
```

e为空时，**严格使用原T0 QT序列**：`[REL], C_Q, [SEP], C_T`，没有伪造空E token，也没有额外SEP。

\[
f_\theta(q,e,t)=H_\theta\!\left(F_\theta(X_{qet})_{REL}+G_\theta(q,t)\right).
\]

`[g_E;C_E]`中的g_E是现成T0全局表示，不是新的encoder或全局缓存。要求parent global_dim=model_dim=512；不匹配则BLOCKED，不能自行加投影。

局部Transformer3层、8heads、FFN2048、GELU、norm-first、dropout0.1；head保持parent原状（512→512→1，GELU/dropout）。这些来自现有R19兼容配置，不是CLEAN-R1 FFN1024/新head配置。

不加全局absolute position；保留角色和模态标识。`role_bridge`初始化为parent的query/target role向量算术均值，随后可训练；query/target role初值保持parent。

### 9.3 训练/冻结参数清单

冻结：Qwen；全部原对象token adapters、text/image poolers、table-token embeddings、modality embeddings、global adapters与global norms。以上可按parent身份离线压缩一次并复用。

可训练：Relation Transformer、原scoring_head、原global_residual_in/out、REL/SEP、原type_pair embedding、query/target/bridge role embeddings。不得解冻额外模块或添加新head。

两arm全部公共tensor从**同一个T0**精确复制；新增bridge role同样初始化。fresh AdamW，不继承T0 optimizer。这里叫`local_continuation`，不叫fresh Teacher。

### 9.4 冻结缓存只缓存冻结计算

B准备阶段必须先构造12.2的E-swap donor映射，缓存覆盖训练/评测全部候选、自然E、增强witness与swap donor，不能等评测缺特征时再删槽位。

cache可保存`C_x,g_x`，键含parent压缩器hash、原feature指纹、object ID、压缩配置；不缓存可训练role/modality以后新增的role、Transformer结果、REL或最终分数。训练开始后不允许复用旧Teacher logits。

pad只用于batch，无效位在attention中mask；三对象长度按真实记录构造，禁止固定每9个token为一个对象。train-time局部dropout由Transformer执行，不能缓存它。

### 9.5 B两个arm

| Arm | 前向和最终评分 |
|---|---|
| B-QT-CONT | 同一兼容代码，强制e=EMPTY；每target只取f0，**不乘路径数**；是匹配预算的QT续训对照 |
| B-QET-PATH | 上述真实三对象前向与8.3完整路径LSE；本轮唯一拟采用的最终Teacher |

冻结原T0 QT额外作为不训练对照。B-QT-CONT和B-QET-PATH训练query、候选、支持标注、epoch、优化器预算相同，但三对象前向次数不同；不宣称FLOPs匹配，也不把两arm差异归结为唯一结构因素。比较的是可部署配方；E-swap才检验真实E的必要作用。

**必须记录本节保留G(q,t)这个既有表征分支。它不是添加独立QT标量，但有忽略E的风险，必须通过B门槛验证；不允许为了“看起来纯路径”静默删掉它。**

---

## 10. B数据构造：自然竞争、同候选增强与支持边界

### 10.1 训练query数量和选择

只用train split。为控制投入，按原query级类型分别取最多2048个implicit、2048个explicit；不足取该类全部，不复制、不从另一类补齐。类内按规范ID列表用`random.Random(seed_token)`无放回shuffle一次取前缀，sampling seed固定20260919，与model seed无关。类型只用于离线训练覆盖，不进入模型或在线路由。

先选择query，再检查支持监督可用性；不得只挑有自然正确witness的容易query。两arm、两model seed使用同一选择。完整原train query数及选中分布必须导出。

### 10.2 固定目标候选

对每个选中q：

\[
C_q^{train}=C_{q,AQE13}^{natural}\cup G_q^{train}.
\]

所有train-known正目标全部加入，**不把候选总长强制裁成100**。未自然进入C的正目标标`injected_training_only=true`；dev评测绝不补入。

候选顺序通过一次固定局部PRNG洗牌；同seed两个arm完全一致。无正例位置特征。C100自然成员含direct/evidence竞争来源，不能只使用raw QT100或偷偷把U取成D。

不存在历史list回退，也不在训练中重新mining。每个candidate保存natural来源和原排名供审计，但不作为模型输入。所有q的顺序、subset与anchor的随机规则以18节为唯一准则。

### 10.3 每query两个训练视图

**Natural view：**使用原自然retained路径；补入的t若没有自然path则只有零跳。

**Augmented view：**若q有任何原标注W，按epoch固定循环选择一个标注`(t*,e*)`；将**同一e***添加给该query的**每一个候选t**，multiplicity按原已有路径m保留且额外加1。不是只给正t加witness。这样路径数量增量和“有新增E”不能直接泄漏目标标签。

已有same e时m←m+1，不执行第二次前向；自然E最多40个，增强视图允许额外1个标注E，不能为保持40而丢自然E。不同候选都增加同一条可评分候选路径，真实支持标签仍由W判断。Augmented不是自然召回，不进入主评测。

q没有W时仅Natural，不伪造Augmented。两个view共用完全相同目标ID集合。

### 10.4 目标正集、忽略集

view v的支持正目标：

\[
P_q^v=D_q\cup\{t\in G_q:W(q,t)\cap E^v(q,t)\ne\varnothing\}.
\]

`I_q^v=G_q\P_q^v`；`N_q=C_q^train\G_q`。

正集并入候选后取交集；P/I/N互斥。未观察到witness的implicit正确目标属于ignore，不被标负，也不自动提升为有支持正例。**空P的view直接标inactive；不能prefetch后回退到G。**

Target loss在P∪N上使用与A相同sum_probability LSE差。两arm使用相同的P/I/N；QT控制的信息少，不为它单独改标签。所有dev主Recall仍以完整G_q为分母，不只评可见支持的目标。

### 10.5 一个支持对比项，不增加新的支持head

对于原始标注确认依赖缺失属性的implicit anchor `(q,e*,t*)`，且e*∈W(q,t*)、t*∉D_q：

\[
L_{support}(q)=\operatorname{softplus}(1+f(q,\varnothing,t^*)-f(q,e^*,t^*)).
\]

固定margin1、权重0.2。它要求正确witness比断开的该条路径提供更高支持，不把t*改成目标负例，不声称其单条E已完成所有行join。没有这类明确anchor时该项为0。

每q每epoch最多一个anchor，与10.3同一个。不得另外挖大量“错误E”并假定为confirmed negative。B-QT-CONT使用同一函数但屏蔽E，必须把同一次forward的f0张量复用到两侧，使loss为常数且梯度相消。不能用两次独立dropout前向声称它们相同；记录该项梯度为0，不能给对照换其他额外loss。

### 10.6 query loss与跳过规则

若有a个有效view：`L_rank(q)=sum(valid_view_losses)/a`。a=0则无rank项，不用0稀释其他query；只有有效support项时仍可形成query训练项。q两者都没有则inactive。

\[
L(q)=L_{rank}(q)+0.2L_{support}(q).
\]

每logical batch对有效q均值；同一q自然/增强view数量不同不改变其总rank权重。对照与路径arm active query集合相同。

输出训练view计数、每view路径数、G/P/I/N、增强e*、自然/注入身份和空P处理。若所有自然view被意外清空、真实多E样本为0，正确性阻断，不允许用增强singleton继续并称完成自然path训练。

---

## 11. B训练预算、分块和验收

| 参数 | 固定值 |
|---|---|
| parent | 第2节固定T0，两个model seed相同权重起点 |
| epochs | 2；正式终点epoch2；epoch1只做短评与安全止损 |
| logical batch | 8个有效query |
| query microbatch | 1 |
| path前向microbatch | 默认8条路径，S0可在8/4/2/1中按显存选择，开训后固定 |
| optimizer | fresh AdamW，可训练参数仅9.3白名单 |
| lr/betas/eps/wd | 5e-5 / (0.9,0.999) / 1e-8 / 0.01 |
| clip_grad_norm | 1.0，logical batch累计完只执行一次 |
| scheduler/warmup | 无 |
| precision | float32，无autocast、TF32关闭 |
| dropout | parent原0.1，不为了对照方便关掉 |
| KD/其他loss | 0；只有10.6的目标loss与支持对比 |

每epoch对固定query集合做seed+epoch的PRNG洗牌，每个active q恰好一次；两arm同序。optimizer不重置epoch边界，阶段A/B优化器完全独立。

**路径分块只是计算方式。**必须先得到同一q全部candidate各自全部path logits，完成8.3 target LSE，再对同一q所有有效targets完成同一个target loss。不能每8条路径或每8张表分别CE再平均。

显存不足可对路径前向块使用激活checkpoint，保持dropout RNG重放；S0必须测试loss/gradient一致性。在初始化前选定microbatch，保存配置后不得中途换数值模式。若仍不足，BLOCKED_RESOURCE；不得裁路径或候选。

epoch0检查QT兼容输出；epoch1在固定全dev C100评估真实QET最终R@10。若比冻结T0 QT同池低超过10个百分点，`STOP_B_CATASTROPHIC`，停止两个B作业，不追加第二epoch。其余完成epoch2。

不能挑epoch1或别的历史Teacher冒充epoch2主终点。所有checkpoint保留，负结果仍交付。

---

## 12. B正式评测与E内容对照

### 12.1 主表：同池且所有候选可评分

每个own C100环境必须包含：

1. raw Qwen QT内积同池排序（不训练）。
2. 冻结T0 QT同池排序（不训练）。
3. B-QT-CONT epoch2同池排序。
4. B-QET-PATH epoch2真实路径LSE排序（主候选方法）。
5. B-QET-PATH epoch2 E-swap同池同路径槽数排序（内容对照）。

必须全部同query、同C100、相同全部正目标分母。不能给QT一个较难C、给Path一个较容易C；不能把无path目标从主表中删掉。另可给path-supported子集的辅助表，但主表不替换。

每行：query-macro R@10（主）、R@20、CR@50、overall/implicit/explicit、固定strict cohort Top10 retention、source-group paired区间、W/L/T、延迟/显存。表名写明`own_C100`或`fixed_C100`。

### 12.2 确定性的E-swap，不使用GT构造假负例

在当前评测split的自然第一跳evidence ID中，按模态分别建立排序的唯一ID环。对每个原e，沿环找第一个**不同content_key**的e'；没有不同内容时该模态E-swap不可执行，标`NA_no_distinct_donor`，B内容门槛不可通过。

每个(q,e,t)槽替换为(q,e',t)，t、槽数量、multiplicity、Q、Direct及C100完全不变。donor选择不读取GT，不从target属性挑E。无需rerun retrieval，Teacher只读取替换内容。

这是内容破坏对照，**不是已确认错误证据集**；donor偶然有效也保留。不得因某例没降分就依据GT换另一个donor。ID本身不输入模型。

不同模型、不同seed在同一环境使用同一donor映射。评估保持eval模式、关闭dropout。

### 12.3 strict指标固定分母

F_q使用6.5冻结BASE严格cohort，不使用各模型各自剩余的strict目标作分母。`R_strict@10`先每个|F_q|>0的q求`|Top10∩F_q|/|F_q|`，再query均值；另报目标对计数retained/total。所有方法共用相同F。

F为空则NA，不能填0并算差值。此时B证据增益门槛缺失，`BLOCKED_NO_STRICT_POPULATION`，不能自行改成explicit子集。

### 12.4 候选上限

\[
Oracle@10(q)=\min(10,|G_q\cap C_q|)/|G_q|.
\]

同时报C100覆盖；不能把C100覆盖等同于一般情况下的Oracle@10。它们都是候选诊断，不替代实际排序R@10。

---

## 13. B门槛与seed29重复

定义QT强参照为冻结T0和B-QT-CONT在**同一环境**各指标的较大值，仅用于严格门槛，不生成一个按query挑模型的oracle系统。

| 条件 | 必须满足 |
|---|---|
| 主效果接近强QT | `R10_overall(Path) ≥ max(R10_overall(T0),R10_overall(QTcont)) − 0.005` |
| 隐式不退 | `R10_implicit(Path) ≥ max(R10_implicit(T0),R10_implicit(QTcont))` |
| 严格证据增益 | `R10_strict(Path) ≥ max(R10_strict(T0),R10_strict(QTcont)) + 0.010` |
| 真实E内容有实用作用 | `R10_overall(Path_real) − R10_overall(Path_swap) ≥ 0.005` |
| 完整性 | epoch2终点、同池全人口、无非法fallback、所有测试通过 |

全部满足才`PASS_B`。它只是允许重复的工程门槛，不是“证据推理已被完全证明”。保留所有区间，不以局部正pair分数上升代替以上排名差值。

seed13 A/B都过才自动重复seed29；不问执行模型主观意见，也不根据接近阈值自行降低标准。A29/B29用相同门槛；任何失败就`STOP_NOT_REPLICATED`，不追加seed42/其他seed。

两个seed成功后先各自报告，再对同query平均得到汇总；不能选较好seed，也不能把1198×2当2396独立query。共同fixed_C100环境用于Teacher配对稳定性，own环境用于完整候选+重排表现，两张表分开。

**到此本轮强制结束。后续KD与fresh不自动执行。** `NEXT_DECISION.md`只能写“具备进入下一轮条件化KD/完整fresh验证的资格”或“停止该配方”，不得写未运行的最终成果。

---

## 14. 统计与主表定义

Recall@K(q)=`|G_q∩TopK_q|/|G_q|`，不是Hit@K。G为空的query不应在原任务人口中；若出现先报告输入异常，不能用0随意补齐。overall按全部合法query均值；implicit/explicit按原划分分别均值，不以两个分项简单平均替代overall（除非真实人数恰好相等）。

`RawUnionRecall`指U集合覆盖。`CR@50`在本协议指实际最终排序前50的目标覆盖/Recall，必须在输出注明定义，不与C100覆盖混淆。

对差值先逐query计算。paired bootstrap按原source-group抽样、有放回抽取与原group数相同的group，每个抽中group携带其全部query；每次求query均值差，不平均group均值。10,000次，bootstrap seed固定20260919，百分位95%区间。重复seed先按query平均再bootstrap。

W/L/T按逐query Recall差值，|Δ|≤1e-12为Tie；各分项W+L+T必须等于该分项query数。不能用overall的Tie填implicit/explicit表。

门槛只用于有限开发选择；所有比较标记探索性、未做多重比较校正。test不在本轮打开。

---

## 15. 必须通过的测试，不允许以“训练loss下降”替代

### 15.1 A数学与标签测试

| ID | 必测断言 |
|---|---|
| A-T01 | W2/b2=0时BASE/E-only/QE查询与raw score相同；真实向量probe同样检查。 |
| A-T02 | 非对称非单位R下，逐pair、矩阵、索引公式一致；transpose错误必能被检测。 |
| A-T03 | Δ=0初步梯度到W2非零；W1首步可为0，不能误判整个adapter断梯度。 |
| A-T04 | 任意输入输出残差比≤0.5+1e-6；极大Δ、极小Δ、Δ=0均有限。 |
| A-T05 | E-only替换Q后输出严格不变；QE替换Q在非零测试权重下有可测变化。 |
| A-T06 | 全矩阵与chunked全分母的loss和adapter梯度一致；有块无正例也可计算。 |
| A-T07 | ignore目标分数任意变化不影响loss，梯度为0；P不被忽略。 |
| A-T08 | 旧Top32之外的合成高分competitor在新loss中有正的降分梯度。 |
| A-T09 | empty P、empty N、标签重叠、不合法ID都有明确处理，不产生NaN继续。 |
| A-T10 | microbatch累计等于同logical batch一次计算；尾batch不丢。 |
| A-T11 | 每epoch每item恰好一次、两arm item/标签/权重顺序完全一致。 |
| A-T12 | BASE全部参数与目标vector/index训练前后hash不变；optimizer白名单准确。 |
| A-T13 | 缓存key包含q,e,modality,BASE,adapter版本；不同Q不能命中相同QE查询缓存。 |

### 15.2 B接口与训练集成测试

| ID | 必测断言 |
|---|---|
| B-T01 | e=EMPTY的step0 f0与原T0 QT真实输出近似相等（FP32 atol1e-5 rtol1e-5）。 |
| B-T02 | g_QT全局表征分支没有被删掉；只有一个shared Transformer与shared head。 |
| B-T03 | 实际triple forward读取三个对象；改变E内容可改变输出，改变ID不改变输出。 |
| B-T04 | variable长度/padding/role边界正确；没有固定九槽假设，Query row组不丢。 |
| B-T05 | Natural的输入路径来自真实检索，不是None/空数组统一回退。 |
| B-T06 | Augmented同一个e*给同query所有candidate；负candidate也能看到它。 |
| B-T07 | empty positive `[]`经prefetch和序列化保持空，不能恢复为G。 |
| B-T08 | target正例不等于每条E都正；未知替换E不把正确T翻成负例。 |
| B-T09 | path分块/target分块只分计算，不分softmax；与不分块loss/gradient一致。 |
| B-T10 | multiplicity优化前后S=logsumexp相同；空evidence时S=f0。 |
| B-T11 | QT-cont最终score没有加log(path_count)；主QET没有外部T0标量fallback。 |
| B-T12 | 真实一次“取query→Natural/Aug→prefetch→三对象forward→LSE→loss→backward→step→评测”集成通过。 |
| B-T13 | 冻结压缩缓存不包含任何可训练role/Transformer/head输出；跨checkpoint不混score。 |
| B-T14 | E-swap不改变candidate IDs/路径槽数/m、模态或GT分母；不重检索。 |

### 15.3 测试等级与证据

附带reference/tests仅是CPU合成数学测试。执行者必须补充实际仓库集成与真实特征探针，并记录调用了哪个生产函数。mock只能验证接口，不能替代真实推理。

正例注入、空集、非单位R、多正例、不同E模态、无path目标、重复路径、尾batch必须覆盖；只有shape测试不够。

正确性失败导致`INVALID_RUN`时修复实现允许，但不得顺便改变方法。保留失败run，另起attempt，从最后一个**证明未受错误影响**的完整边界恢复；无法证明则该arm从epoch0重启。最多一次因实现错误的正式重启；再次失败交付阻断报告，不连续重训。

---

## 16. 产物、日志、复算和打包

建议一个薄入口`src/run_stage1_qcpath_r1.py`加小模块，不建设通用实验平台，不搬动无关代码。以下命令是**需执行者接入的接口合同，本交付并未包含可直接训练仓库的现成CLI**：

```text
resolve-inputs → build-labels → test-contracts → prepare-a
→ train-a --arm EONLY|QE --seed 13
→ evaluate-a → decide-a
→ [PASS_A] prepare-b → train-b --arm QT_CONT|QET_PATH --seed 13
→ evaluate-b → decide-b
→ [PASS_A+B] repeat-seed29 → summarize → pack
```

每个阶段参数显式读取同一`protocol.json`及`RESOLVED_INPUTS.json`，不偷用函数默认值。不得实现`--force-pass`、`--fallback-qt`、`--allow-missing-features`绕过门槛。dry-run只能检查输入与预计作业，不生成成功回执。

最小目录：

```text
run/
  protocol.json
  RESOLVED_INPUTS.json
  SOURCE_LOCK.json
  PRODUCTION_CONTRACT.json
  compliance.json
  labels/{train_pairs.jsonl.gz,label_counts.json,label_conflicts.jsonl}
  frozen/{object_manifest,target_manifest,probe_pairs,first_hop,...}
  A/{EONLY,QE}/seed{13,29}/{config,trace,checkpoints,evaluation,...}
  B/{QT_CONT,QET_PATH}/seed{13,29}/{config,trace,checkpoints,evaluation,...}
  decisions/{A13,B13,A29,B29}.json
  reports/{RESULTS.md,LIMITATIONS.md,NEXT_DECISION.md,metrics.csv,paired.csv}
  tests/{cpu_report,production_report,real_probe_report}
  SHA256SUMS
```

每step日志最少：stage/arm/seed/epoch/update、消费item/query IDs指纹、实际batch大小、active视图数、loss分项、grad norm、学习率。epoch保存完整消费顺序一次即可，**不必每step重新hash全湖**。

A raw artifacts：每q的D、第一跳E、每E second-hop Top20及raw分数、Evidence ranking、U、C、M、retained path+m；probe正例ranks。B raw artifacts：每q/t的f0、每条唯一路径的eID/m/f_e、聚合S、全部最终排名；E-swap donor映射与对应分数；GT只在评测伴随文件中。

checkpoints保存模型、optimizer、RNG、completed update、协议hash；不得仅保存`best.pt`。旧BASE/T0不用重复打包完整权重，用hash与路径定位；新adapter和B可训练权重必须随结果包交付。

`RESULTS.md`列出全部实际终点及未通过gate。`LIMITATIONS.md`区分未执行/缺输入/资源阻断/实现失败/效果失败；`NEXT_DECISION.md`只做继续资格或停止决策，不生成未授权新训练矩阵。

归档manifest排除自身及最终zip，避免陈旧自引用。最终压缩包需在另一临时目录解开一次，检查文件hash和用于复算的排名/分数是否齐全。

---

## 17. 合规回执与禁止擅改（必须在开始和结束各检查一次）

执行者建立`compliance.json`，对H01–H16、A-T01–13、B-T01–14、每个gate给出：`required / status / evidence_path / actual_value / reason`。状态只允许`PASS, FAIL, BLOCKED, NOT_REACHED, NOT_APPLICABLE`；NOT_APPLICABLE只用于规范明确定义的情形，不能用来绕过必测项。

**再次强调：本文的每一项规划与要求都必须遵守。**

- 不得悄悄更换模型、库、split、prompt、缓存、分数空间或目标ID集合。
- 不得因实验失败而换成QT-only、削弱QT对照、加入GT路径评测或删掉困难query。
- 不得因资源紧张而换Top32分母、减少模态/候选、增加GT路由。
- 不得因某个epoch好看而事后改终点、门槛、seed或主指标。
- 不得因为“实现复杂”就把Natural换成空、把未知变负、把空P回退G、把未执行写完成。
- 不得在缺输入时假装读取到了服务器上的文件；必须报告具体缺哪个角色和允许执行到哪一步。

**允许的优化仅限不改变合同的实现优化：冻结计算缓存、稳定ID索引、矩阵运算、保持global分母的分块、固定顺序预取、同路径分数复用。优化前后必须有数学和真实小样本差分证据。**

本轮最终可接受的交付包括一个完整、可信的负结果。它不包括一份高分但违背协议的“成功报告”。


---

## 18. 随机性、顺序与边界的最终锁定表

本节消除“使用一个固定seed”仍可能留下的实现歧义。所有抽样输入先按UTF-8 ID规范排序；只对namespace做一次SHA256，取前8字节big-endian转非负整数，交给Python `random.Random`。不使用进程随机化的`hash()`。

| 用途 | namespace / 规则 |
|---|---|
| A item顺序 | `QCPATH-R1|A-order|{model_seed}|{epoch}`，再执行5.4的组内shuffle＋组间round-robin |
| B训练子集 | `QCPATH-R1|B-subset|20260919|{implicit_or_explicit}`；同类别候选q排序后shuffle取最多2048 |
| B目标候选顺序 | `QCPATH-R1|B-candidates|20260919|{query_id}`；`Cnatural∪G`去重排序后shuffle一次；所有epoch、两个arm、两个seed都复用 |
| B每epoch query顺序 | `QCPATH-R1|B-order|{model_seed}|{epoch}`；对当epoch有效q规范列表shuffle |
| B增强anchor | 将所有`(e,t)`原标注W去重并按`(e,t)`排序；初始offset为`namespace_seed(QCPATH-R1|B-anchor|20260919|q) % len(anchors)`；epoch e选`(offset+e-1)%len(anchors)`；同q两arm同anchor |
| E-swap | 不随机；同split同模态唯一ID有序环，找下一个不同content_key，见12.2 |
| bootstrap | 独立PRNG seed20260919，不消耗模型训练RNG |

训练前分别设置torch CPU/CUDA seed=model_seed；A两个arm复制同一初始化state，并在各自正式训练开始前恢复同一训练RNG起点。B两个arm复制同一parent及bridge角色，在各自训练开始前设置同一seed；由于前向路径数不同，随后dropout消耗量可不同，不冒称逐位随机轨迹匹配。数据抽样必须使用独立局部PRNG，不被模型前向消耗影响。

使用确定性算法并关闭TF32；运行前配置需要的CUDA确定性环境，记录实际设置。遇到不支持的确定性算子必须在S0报告，不能自动`warn_only`或偷偷关闭确定性继续。实际服务器算子若无法满足，标资源/实现阻断，不承诺跨GPU型号逐位一致。

S0的试前向、试反传、显存microbatch选择使用独立临时模型；结束后丢弃，正式作业重新从规定初值和RNG开始。不能把试跑的几步权重当epoch0。

阈值比较采用正文数值加最多1e-12浮点容差。输入指标有NaN/Inf/NA或超出[0,1]时门槛不可计算，必须BLOCKED；不得当0、当PASS或跳过此项。

**规范最后一条：任何执行到这里仍未被明确授权的训练、替换、调参、fallback或额外实验，都不属于本轮任务，不得自行开展。**


---

## 19. 资源与耗时记录的固定口径

每个正式job记录实际optimizer updates、logical/microbatch数、candidate打分数、唯一路径forward数、按multiplicity计的路径数、冻结缓存大小、峰值GPU allocated/reserved memory、CPU峰值RSS和实际wall time。前向与后向成本分别能测则分别报，不能用理论FLOPs冒充实测耗时。

最终checkpoint仅做一次统一延迟面板：从规范ID排序的完整dev取前min(128,N)个q，不按GT选；前min(16,N)个q先warm-up一次，不计统计。逐query计时边界显式同步CUDA，报告p50/p95/mean；所有方法相同query顺序、hardware、缓存驻留状态和路径/候选预算。

分别给出：

- **A second-hop latency**：给定已冻结第一跳E及对象向量，生成条件向量＋实际ANN查询＋原Evidence/Equal，明确不含第一跳和Qwen。
- **B rerank latency**：给定C100与自然路径，读取已驻留的冻结压缩特征＋真实Teacher评分＋LSE/排序，明确不含检索和Qwen。
- **Stage1-after-embedding latency（仅最终配对面板）**：从已给定query embedding开始，实际执行Direct/第一跳/第二跳/原admission/Teacher，不复用该query的结果排名缓存；可以复用静态索引和冻结对象特征。标明不含8B query编码，不能称含原始输入编码的全系统端到端时延。

首次冷加载磁盘到RAM/GPU的耗时单列；不能把一种模型计冷加载、另一种模型计warm结果。训练阶段不能因为耗时统计改变主模型随机数或训练顺序。延迟面板不用于调K、调efSearch、调ρ或挑checkpoint。

A/B关系的预算公平性：A匹配训练updates与相同候选分母；B匹配query与optimizer预算，QT与QET前向路径数不同，必须如实报告计算量，不能声称等FLOPs。

**至此规范封闭：必须按照既定步骤执行和交付，不得自行增加未授权的训练阶段。**
