# R13 修订版：以 target Recall 为主的 Stage 1 角色投影与见证学习

日期：2026-09-09。状态：**待审阅、未执行**。

本文件是完整替代稿，不是 diff，也不是已实施／预注册完成的声明。先由研究者审阅，再交给 Codex 实施。本次文档修订不启动训练、模型推理、实验测试、人工标注服务或外部 baseline 复现。文中的实验、测试、目录及新增接口均为后续实施规格；未明确列为现有实现的名称不代表仓库已有入口。

## 0. 依据、解释优先级与研究范围

优先级为：本轮用户研究意图 > `08_round2_context_update.md` > 本修订稿中的明确设计 > 旧 R13 / 01–07 的历史表述。历史实验结果保留原定义，不追溯改写。

本次全文读取旧 `stage1_optimization_r13_plan_20260909.md` 和 08；结合 01 的方案、02/04 的 R10–R12 审计、03 的旧计划、05 的指标／manifest，并复查 06 中模型、目标、打分、训练、检索及 R12 入口。07 的二阶段只用于保留后续研究边界。附件不是完整可运行仓库；历史 checkpoint、完整逐样本输出和实际候选文件没有随这些源码节选一起提供。本计划中的路径是来源定位，不意味着本轮访问了研究者电脑。

局部事实引用记为 [L1]–[L6]，文献记为 [R1]–[R6]，完整来源在末尾。本稿公式、实验臂、预算触发条件是**待验证的新建议**，不是附件中已证实的方法。

本轮的三个问题是：

1. query/target 共享低维 table 投影，是否造成可保留输入子空间不足或跨角色更新干扰？拆分角色后，完整部署流程的 target Recall 是否提高？
2. 不在线指定连接列，仅用训练侧既有见证记录改变路径信用分配，能否提高自然 ANN 下的 target Recall？哪些结构性歧义仍不能解决？
3. 若第二跳确实需要 query 上下文，轻量 `h(Q,E)` 能否在保持离线 target 向量、表级 ANN、0/1 跳与相同搜索预算的条件下提供增益？

本轮允许结论为“一阶段 Recall 改善，完整补属性机制尚待二阶段验证”。不要求先解决生成全空，不以独立最终 join 审核或大规模新属性负例标注作为全部 Stage 1 训练的门槛。

## 1. 更新后的任务定义与必须撤回的旧前提

### 1.1 二阶段是重排序，不是最终布尔准入

当前已评分候选按 `(-coverage, -mean_similarity, stage1_rank)` 排序；同 target 的已执行 direct/evidence 分支采用相同规则选较优分支，完全同分保留 direct。`joinable=False` 不删除已评分候选。[L1]

cell 相似阈值仍参与 coverage 的离散化，`min_row_coverage` 仍产生兼容性布尔字段；这两者不同。后者不是最终候选准入门槛。因此：

- 不再将“至少 3/5 行”定义为方法成功条件；1–2 行的真实恢复也可能影响排序。
- 不把旧阈值接受项的 precision 当本轮主成败标准。
- 支持行分布与 B=4 的处理限制仍要记录，但不得升级为 Recall 的硬替代目标。
- 固定 N 个唯一候选仅重排且不删表时，Recall@N 恒定；二阶段只能改善 K<N 的 Recall@K。

### 1.2 在线不提供具体 join column

在线输入只有 query-by-example 的可见表内容。Target 的可见 schema/cells 是正常对象内容，可独立编码；不在线指定 GT 隐藏属性、目标列、恢复值，不按预测列逐列发起检索，不新增 column-level 索引或显式列枚举。

训练可使用 `(q,t,attribute,column,row,evidence,value)` 记录构造辅助标签、分组或 Teacher 目标；这些字段不得加入 Student 的部署输入。一个 `(q,t)` 可有多个合法见证；构造时选中的属性不是其他属性的负标签。

### 1.3 方法身份

| 改动 | 身份 |
|---|---|
| 原三类型 P、full 有向 R、冻结骨干、离线 Teacher、原对象对评分 | 原主线 |
| 拆分 P_query/P_target，R 仍按原有类型对定义 | 角色感知 Student 表示扩展；仍是单对象可分解检索 |
| 仅训练期见证辅助监督 W | 监督目标扩展；在线函数与索引不变 |
| 输入空间的低秩角色残差 | 角色表示扩展的轻量版本 |
| `h(Q,E)` 第二跳 | 改变局部 primitive 的条件化检索扩展；仍表级单向量 ANN |
| KD-off、P 冻结、pure-direct、单模态 | 机制消融，不自动替代主方法 |
| 在线 Teacher、解冻骨干、额外跳数、列／行索引 | 不进入本轮首批 |

## 2. 已有证据与本轮假说，不把相关性写成根因

| 附件事实 | 本轮可使用的推断 | 仍不能推断 |
|---|---|---|
| C-candidates356 ValidPool 255/678，C-base356 222/678；额外 Student seeds 同向 | 固定 Teacher 下候选组成有条件收益；复用健康起点 | 该差值不是 target Recall 差值，也未隔离 KD |
| Edge 延长仍有 499 个 QE 正对，完整路径仅 46；text→table exact 受损 | 需定位 ET 目标邻域与其参数更新 | 不能仅凭这个分解确定共享 P 是根因 |
| Path-only QE 529→412、完整路径 255→186；最后选 step0 | Path 训练的首跳损伤与 edge 的主要损伤不同 | 不能把 C1 成果归功于 path |
| Teacher 对 hard candidates 的优势依关系变化 | Teacher 能力与同 hard-candidate KD-on/off 是必要归因控制 | 不足以宣布 Teacher 普遍更强或完全无用 |
| Student P 仅 table/text/image 三类；有向 R 已存在；Teacher 有 role embedding | 本轮检验的是压缩层的角色共享，而非首次加入方向 | 不能说现有系统“没有方向／角色信息” |
| Train/dev 的确认支持行分布不同，现有标签不穷尽支持 | 按支持行数、source、模态分层解释 Recall | 不能称两行支持天然无法完成任务 |

历史结果与来源见 [L3][L4]。首批不重做原有全景分析，也不重复未经新诊断支持的 sigmoid、函数正则或融合权重网格。

## 3. 数据、输入与监督边界

沿用 EntiTables-v9 的已物化划分：train-fit 11,390 queries；cal-fit 624；cal-check 616；dev 1,198；历史 R10 test regression 1,166。[L3][L4]

- **训练及训练候选 mask**：只读取 train-fit 的已物化监督。先限定来源，再建立全局正邻居；不能先读取全量 dev/test 真值再过滤。
- **无标签湖**：允许索引和挖候选，披露同湖 transductive 设置。
- **开发评价**：所有正式 Stage 1 臂在完整 dev 1,198 queries 上评价；不得以旧 96/32 富集样本替代总体。
- **诊断集**：从 train-fit 用 seed13 和 source-group 哈希锁定 8 个 edge batch、8 个 path batch及其固定正边；仅用于梯度、margin、Teacher 与损失资格检查，不称泛化集。
- **开发机制面板**：至多 128 个固定 query、128 个 evidence source，按模态/source 固定抽样，比较 exact 和 ANN。漏检计零；样本不足保留实际数量。
- **cal-fit/cal-check**：首批不新增校准器。若后续需拟合新 gate／尺度，另行登记，不能把 cal-check 反复当调参集。
- **历史 test**：只在本轮配方冻结后做回归，不用于选臂、延长或挑公式。

已有权重祖先与标签污染风险继续披露；修复当前输入不能净化旧 checkpoint。角色拆分、见证监督的首批结论均是固定 S0/Teacher 祖先下的条件干预。

### 3.1 多种标签的用途必须分开

| 标签 | 可用目的 | 禁止用途 |
|---|---|---|
| train-fit target 已知相关 | 多正 target ranking、known-positive mask | 宣称 qrels 穷尽全部合法目标 |
| 既有 `evidence_recoveries` | 正见证辅助训练、支持下界诊断 | 把未记录 evidence/行标成不支持 |
| 已独立确认属性特定不支持 | 该属性作用域内的诊断／后续条件头训练 | 标成对象对全局负例；压制其他合法属性 |
| 未标注 hard candidate | 明示的 ranking-only weak/unknown 对比 | BCE、概率校准、确认负例 precision |
| GT 列／属性／恢复值 | 训练损失分组、合法 privileged supervision、离线评价 | 在线 ANN query、索引对象内容或候选选择的额外真值输入 |

recoveries 的值只用于标签冲突核验和去重，不输入首批 Student，也不为首批新调用 8B 编码隐藏值。首批无需等待全部独立人工标注。

## 4. 唯一主评价口径、榜单与预算

### 4.1 Target Recall

对预定 query 集合 $\mathcal Q$，已知相关唯一目标集合为 $G_q$，同一完整榜单前 K 个唯一目标为 $\mathrm{TopK}(q)$：

$$
R_q@K=\frac{|G_q\cap\mathrm{TopK}(q)|}{|G_q|},\qquad
R@K=\frac{1}{|\mathcal Q|}\sum_{q\in\mathcal Q}R_q@K.
$$

**主要端点：完整 dev 的总体 target Recall@10。** 同时报告 implicit/explicit 的 query 宏平均 Recall@10，以及总体／分层 @20、@50。多正例保留全部相关目标；不能用 any-positive Hit@K、正对微平均或桶等权平均替换主指标。

预先登记 $|G_q|=0$ 的不可评分项；不得因模型未召回而排除 query。缺榜单记空榜，正例分母不变。现有数据若全部 query 有相关目标，主分母就是 1,198；若发现数据错误，只能对所有臂统一登记修复，不按结果过滤。

`CandidateRecall@N` 使用实际交付候选集合 $C_N(q)$：

$$
CR@N=\frac1{|\mathcal Q|}\sum_q\frac{|G_q\cap C_N(q)|}{|G_q|}.
$$

Stage 1 的 TopN 就是其交付集合时，数值上 `Stage1 Recall@N = CandidateRecall@N`；二者分别保存以明确流程含义。Stage 2 固定此集合后 `Stage2 Recall@N` 必须相同。@10 与 @20 是 N=50 榜单的真实前缀，不另行构造不同候选集。

### 4.2 主排序规则冻结，避免归因混乱

为不同时改变投影／监督与部署准入，**首批主读数沿用旧 R13 指定的 R12 D1 保留 + F1 union-direct**。F1 不是 pure-direct：Q→E→T 仍形成候选并集，所有候选补算 direct 分数，最终按该分数排序；但是它的最终排名不使用 evidence 通道分数作为独立票，必须如实注明。[L2][L3]

每个模型固定执行：

1. 正常 direct100 与 mixed 两跳搜索，形成唯一目标并集 U(q)。
2. 在同一 U(q) 上补算 raw direct score；保留每条 evidence path 及 D1 bundle。
3. 按 union-direct 得到唯一完整榜单，取 N=50，报告 @10/@20/@50。
4. **必报固定敏感性读数**：在完全同一个 U(q)、同一 D1 evidence 分数上计算等权 union-RRF，常数60；它用于判断 evidence 分数的作用，不按臂择优、不进入本轮主选模。
5. 同模型 pure-direct100 仅为消融。记录相对 pure-direct 与相对健康 S0 的 rescue/displace，按 query 的相关目标分母加权计算 Recall 差值。

这项选择保留已采用的健康部署参照，而不改用较弱的 RRF 起点制造“进步”。若研究者希望把等权 RRF 改为主部署规则，必须在审阅冻结时对**全部**臂统一改写本节、选择器和统计；不能在看结果后切换。F1 下 W 只改善路径责任而不改善实际目标 Recall，仍判本轮主端点未改善。二阶段可能利用这些证据，但不能预支该收益。

### 4.3 固定预算与集合

| 项目 | 首批设置 |
|---|---|
| 冻结骨干／Student | Qwen3-VL-Embedding-8B 缓存；D=4096、d=1024；full R |
| Q→T | 100 |
| Q→text / Q→image | 20 / 20 |
| 每个 E→T | 20 |
| 每 target 待保留 L / 保留 B | 20 / 4，D1 固定 |
| 交付 Stage 2 的 N | 50；本轮前半段不执行 Stage 2 |
| 主截断 K / 补充截断 | 10 / 20、50 |
| 后续 Stage 2 恢复目标数 M | 20；与 N、B、K 分开 |
| 单模态后续对照 | text40 或 image40，不加倍 evidence 总数 |
| ANN | 沿用同一后端、参数、插入顺序、种子和精度，冻结到 manifest |

无重复时每 query 至多 40 条 E、800 个 ET 返回位置和100个 direct位置；实际唯一目标不超过900，常更少。理论上是43个搜索向量（1个 QT、2个 QE、40个 ET），不是43次 Python/API 调用；`search_many` 可以批处理。应记录实测搜索向量数、去重候选数和补算数。

不同模型必须建立自己的投影索引及自然路径池；只复用相同的冻结骨干特征，不复用旧模型邻居 ID 冒充自然检索。同模型固定池的重排／分数诊断可以复用池，但与端到端自然 ANN 效果分表。

## 5. 公共 Student 形式、损失与起点

以下采用列向量记法。模态 $m\in\{text,image\}$：

$$
u_Q=P_Qz_Q,\quad u_T=P_Tz_T,\quad u_E=P_mz_E.
$$

三类评分是：

$$
s_{QT}=u_Q^\top R_{tt}u_T,\qquad
s_{QE}=u_Q^\top R_{tm}u_E,\qquad
s_{ET}=u_E^\top R_{mt}u_T.
$$

原模型 $P_Q=P_T=P_{table}$；角色模型分离它们。R 的键仍然是原有有序类型对，不在首批同时拆 R。Teacher 保持原有 source/destination role embedding、任务参数和打分缓存。

任意局部评分 $u_s^\top R u_d$ 的 ANN 查询向量为 $R^\top u_s$，索引向量为 $u_d$。不额外归一化，避免将 MIPS 偷换为 cosine。

### 5.1 S0 和公共优化器

S0：R12 `C-candidates seed13 step356`；若用 C2 `path_only step0`，必须验证 P/R、PCA anchor 和模型配置与 S0 一致，名称仍标 `c1_candidates356`。

角色／见证续训统一从 S0 参数开始，**统一新建 AdamW 状态**；这是配对的新阶段实验，不称逐位重现历史 edge 延长。使用旧 optimizer 的影响不与角色拆分同时变化。只有所有匹配臂都能按同一规则恢复状态时，才允许在审阅时统一改为恢复，且重新冻结协议。

公共设置：batch64，P 学习率 1e-6，R 学习率 1e-5，AdamW weight decay0.01，KD 权重0.3、温度1，BCE=0；首批不变 Teacher、损失温度、关系配额、骨干特征、D1、融合、ANN 参数。参数 anchor 使用完整链 PCA 与 identity R，不重置为 S0。

### 5.2 基础目标

定义多正例列表损失：

$$
CE^+(l,C,P)=\log\sum_{j\in C}\exp l_j-\log\sum_{j\in P}\exp l_j.
$$

Edge supervised logits 保持历史 $f(s)=10\sigma(s)$，KD 保持 raw logits；这不是统一分数空间。训练日志必须分别保存两者。单调变换本身不改变固定参数的边排序，但会改变梯度，不先把饱和定为退化根因。

$$
L_{edge}=CE^+(f(s),C,P)+0.3\,KL(P_T^{raw}\Vert P_S^{raw})+L_{anchor}.
$$

Path 保持：

$$
g(q,e,t)=s_{QE}(q,e)+s_{ET}(e,t),\qquad
e_t=\log\sum_{e\in\mathcal E_{qt}}\exp g(q,e,t),\qquad d_t=s_{QT}(q,t),
$$

$$
L_{path}=CE_D^++CE_E^++0.3(KL_D+KL_E)+L_{anchor}.
$$

CE_E/KL_E 只在列表中存在证据正 target 与至少一个 sampled 非正 target 时有效；无 evidence 是缺席，不是0分成员。目标列表的其他已知正例全部纳入正例或屏蔽，不能只保护 designated positive。

沿用实际 R12 双 anchor 形式并固定：$L_{anchor}=0.1A_{all}+0.1A_{E}$。其中 $A_{all}$ 为各 P 对原 PCA 的逐元素均方距离与各存储 R 对 I 的归一化平方范数之和；$A_E$ 另含四种 table/evidence R 的同一范数。实际已用关系和闲置关系数量单列；不能因角色数增加而无意增加 table 正则总权重。[L5]

路径训练与 D1 推理聚合存在既有差异，首批保留并记录；不以修改聚合来掩盖表示／监督的单因素效果。

## 6. 方向 B：角色投影的定义、能力与干扰

### 6.1 三种可比较设计

| 设计 | 定义 | Step0 | 增量参数（D4096,d1024） |
|---|---|---|---:|
| Shared | P_Q=P_T=P_table | S0 | 0 |
| Split | 独立 P_Q、P_T，text/image不变 | 两者都复制 S0.P_table | dD=4,194,304 |
| Residual | P_Q=P0+U_QV_Q；P_T=P0+U_TV_T；P0仍可训 | P0=S0.P_table，U_Q/U_T=0，V非零 | 2r(D+d)，r=16时163,840 |

Residual 的 V 初始为固定 seed 的非零正交方向，允许读取原 D 维输入；U=0 保持旧分数。不能把 U、V 同时置零，否则两者初始梯度可能都为零。可把 V 初始化在 S0 table 行空间的正交补内；这不要求后续始终正交，也不声称正交就代表有用语义。

三种设计的 ANN 形式完全一致：

| 搜索 | ANN query | index |
|---|---|---|
| Q→T | R_tt^T P_Q z_Q | P_T z_T |
| Q→E_m | R_tm^T P_Q z_Q | P_m z_E |
| E_m→T | R_mt^T P_m z_E | P_T z_T |

full R 下仍只需 target/text/image 三个对象级索引，不为每个 query 建索引；Q 不在 target 索引中额外占一条记录。同一表未来作 query 或 target 时，按**本次调用角色**选择投影，不能按数据集 ID 前缀或 qrels 身份硬编码。原 `embedding_role` 字段可辅助检查用途／内容指纹，不替代运行时角色参数。

### 6.2 有向 R 什么时候能吸收角色适配？

若仅在现有压缩空间内设 $P_Q=A_QP_0,P_T=A_TP_0$，则：

$$
P_Q^\top R_{tt}P_T=P_0^\top(A_Q^\top R_{tt}A_T)P_0.
$$

QE 的 A_Q 可吸收进 R_tm，ET 的 A_T 可吸收进 R_mt。因此在 full R 自由、纯双线性、无额外约束时，这不扩展函数类；它可能改变优化、anchor 和 weight decay 的归纳偏置。不能称“新发现了角色信息”。若 R 被限制为对角、低秩或强 identity 约束，等价关系可能不成立／代价不同；本轮固定 full R。

共享低维特征是多任务学习中的一种结构假设，而不是免费正确的事实。[R6] 真正的表达扩展来自不同的**输入子空间**：原模型所有表相关有效算子在 table 一端都受同一个不超过 d 维的行空间约束。独立 P_Q、P_T 允许两个行空间的并集超过 d，但每个角色仍只有 d 维。

逻辑反例（不是实验样本）：D=2,d=1，要表示 $s(q,t)=q_1t_2$。共享 P=[a,b]、标量 R 只能得到 $R\,P^\top P$，不能表示非对称矩阵 $e_1e_2^\top$；独立 P_Q=[1,0]、P_T=[0,1] 可以。对 d>1，原模型已有非对称 R，并不是所有共享 P 都对称；限制是左右有效子空间必须容纳于同一个 d 维空间。

当 d≥D 且 P 满列秩，或两个角色需要的特征本就能放入共同 d 维空间时，拆分未必增加相关表达能力。DPR 的独立编码器与 MDR 的共享编码器都曾在各自任务中有效，文献不支持“非对称任务必然要拆分”。[R1][R2]

### 6.3 Step0 恒等与训练公平

迁移必须逐分数验证 QT、两模态 QE/ET、path raw score，以及 exact topK。首次验证可共用旧 step0 索引以消除 ANN 重建随机性；随后各臂建自己的索引。

角色 anchor 用：

$$
A_{table}^{split}=\frac{\|P_Q-P_{PCA}\|_F^2+\|P_T-P_{PCA}\|_F^2}{2dD}.
$$

它替代原一个 table anchor，text/image/R不变；Residual 对**有效 P_Q/P_T**应用此式，不额外按因子增加一遍 anchor。

相同名义 LR 不代表相同有效更新：共享参数接收多个角色梯度之和，拆分后分别更新；Adam 的二阶状态、相消、正则和归一化均会改变。这是干预的一部分，须报告每角色有效 ΔP 范数、输入子空间外分量与分数／排名变化。不要私自给拆分臂乘2学习率或额外调权。

Split 新增 float32 权重约16MiB；计入梯度和两个 Adam moments 时额外约64MiB，不含框架、master weights等。实际仓库存储9个R而通常只用5个关系，报告“存储参数量”和“本轮获得梯度参数量”两种口径。Residual 可在部署前合并成两个线性矩阵，索引维数不变；训练／checkpoint 因子成本仍报告。

### 6.4 干扰与容量是不同假说

必须使用当前 loss 和实际一步更新，而不只观察梯度夹角或 P 漂移。

1. 在共同参数与 Adam 状态的临时副本上，分别施加 QE、ET、QT，以及 path-direct/path-evidence、ranking/KD 的一步更新；不把这些副本当正式模型。
2. 对不参与该 batch 的固定 train-fit 见证边，比较正负 margin；对固定 E 集合，用所有 target 的 exact 分数计算正确目标 rank/Recall。
3. 做参数块替换：仅应用 ΔP_table、仅 ΔP_text/image、仅 ΔR，及完整更新。非线性交互不能简单按贡献相加。
4. 在 Split 副本上仅令 query-role 更新、保持 target-role，检验是否阻断 QE→ET 的附带损伤；反向亦然。
5. 若边际损害立刻减少而两个行空间还几乎一致，更支持优化解耦。若需要可见的新增输入方向、去掉其正交分量后优势消失，才支持子空间贡献。
6. 对有效 W 和共同子空间做低秩逼近只能提供描述性证据。用 top-d SVD 压回共享空间失败，不证明所有共享模型都不可能成功；必要时另做同子空间角色适配的匹配对照，不能把它和 free-P 的比较混为同一能力证明。

PCGrad 讨论了冲突、梯度主导和曲率的共同作用；本轮借用其诊断动机，不直接加入梯度手术。[R4]

### 6.5 能否解释两类历史退化？

- Edge：QE 或 QT 更新通过共享 P_table 改变 target 端表示，可能使 ET 排名恶化；拆 table 角色对该机制有直接辨别力。ET 自身的坏监督、P_text 更新、R_mt 更新仍可能使它失败。
- Path：direct 分支同时使用 P_Q/P_T，且 P_Q 仍与 QE 共享。拆 table 不消除 QT-source 与 QE-source 的冲突，也不消除 path 负 target 对正确 QE 的错误责任。因此角色拆分可能只缓解 edge 退化，而对 path 无益；这是有信息的结果。
- Evidence 的 P_text/P_image 同时服务 QE destination 与 ET source。首批故意不拆它们。只有 table 拆分后，实际块更新明确定位到 evidence 投影跨角色损害，才值得另轮拆对应模态；不得一开始全拆导致不可归因。

## 7. 方向 A1：首选的训练期见证辅助目标 W

### 7.1 为什么先做 W，而不是先训练具名属性分类器？

局部高分并不保证同一个见证：

$$
(\exists w_1:A(Q,E,w_1))\land(\exists w_2:B(E,T,w_2))
\not\Rightarrow\exists w:A(Q,E,w)\land B(E,T,w).
$$

当前 target-only 损失可以用任意高分路径解释正 target。W 的目标是让**已有训练记录证明可用的路径**也承担学习信用，而不是要求在线选择连接列。

这里的结构性批评仅针对固定E的第二跳：`g(q,e,t1)-g(q,e,t2)=s(e,t1)-s(e,t2)`不依赖q。完整系统通过不同QE命中、QE权重及多个E聚合，本来可以产生依赖Q的目标排名；不能把整个系统说成query-independent。只有正确需求挤在同一E、而其他专一E无法在预算内替代时，这个限制才直接成为召回瓶颈。

首批不训练新的二元“属性支持概率”，不用独立错误属性标签来填满全部 negative quotas。先使用 train-fit recovery 正记录及现成 sampled target 对比，回答一个更小的问题：合法正见证改变信用后，target Recall 是否受益？这属于 privileged supervision 的运用，不是保证 Student 从缺失输入中恢复不可预测信息。[R3]

### 7.2 见证定义与数据结构

训练记录归为：

$$
w=(q,t,a,i),\qquad S_w^+=\{e: e\text{ 在已有记录中支持行 }i\text{ 的属性 }a\text{ 并对齐 }t\}.
$$

a 是训练侧桥接属性／等价列组的作用域，i 是原始 query row ID。它们只用于标签组织；Student forward 仅接收可见对象的 z_Q,z_E,z_T。不同合法 a 全部保留，不选一个 a 后将其他 a 的路径标负。

先按 `(q,t,a,i,e-content-key)` 去重。同一证据有多个合法行支持时可保留多个标签，但一次路径 forward 复用，不复制在线资产。分清“支持行并集”和现有唯一路由的可执行覆盖；本轮无三行硬要求。

必须从 train-fit 已物化 records 恢复 a 与 target-local column 的可靠映射。若记录没有属性作用域，不能凭列位置猜测；只能降为 `any-known-witness` 正例辅助并明确不检验同属性一致性。这个降级仅影响 W 的解释，不阻塞角色实验。

### 7.3 完整辅助评分与目标

训练列表为 C_q，已知正 target 集合 G_q，候选路径集合为 $\mathcal E_{qt}$。使用基础 path raw score g，不新增在线打分头。

对一个有可用正路径的 w：

$$
\widehat S_w^+=S_w^+\cap\mathcal E_{qt},\qquad
A_w=\operatorname{LME}_{e\in\widehat S_w^+}g(q,e,t),
$$

其中 $\operatorname{LME}(x_1,\ldots,x_n)=\log\sum_j\exp x_j-\log n$，温度固定1。归一化避免仅因某行有更多重复标注就奖励更高分。

仅对列表中非已知正 target、且实际有 candidate evidence 的目标形成 weak target 对比：

$$
C_q^- = \{t'\in C_q\setminus G_q:\mathcal E_{qt'}\ne\emptyset\},
$$

$$
B_q=\operatorname{LME}_{t'\in C_q^-}
\left[\operatorname{LME}_{e\in\mathcal E_{qt'}}g(q,e,t')\right].
$$

定义：

$$
\ell_w=\operatorname{softplus}(B_q-A_w),
$$

$$
L_W=\operatorname{mean}_{q\ \mathrm{eligible}}
\operatorname{mean}_{t\ \mathrm{eligible}}
\operatorname{mean}_{a\ \mathrm{eligible}}
\operatorname{mean}_{i\ \mathrm{eligible}}\ell_{(q,t,a,i)}.
$$

最终：

$$
L_{path+W}=L_{path}+0.1L_W.
$$

每层只对存在可用见证的项取均值；一个训练 batch 无有效 W 时返回可反传的0，并保存有效 query/pair/row 数。原 L_path 仍覆盖原 batch 的全部 query，不能只训练成功见证子集后省略其他 query 的评价。W 的系数0.1为首版固定设置，不随 dev 调整；记录其梯度贡献，若信号弱或过强作为负结果解释，不自动扫权重。

**这个目标与常规 posterior regularization 相关，但不是其完整算法复现，也不继承其保证。**它是支持分组的对比辅助损失：同一组中的证据可以替代，已知不同行／合法属性获得均衡的正信用。[R5]

### 7.4 正例、unknown 与信用边界

- 正 target 上未标注的其他 evidence 不进入 W 的负分母，不因缺少 recovery 被显式惩罚；其他合法属性的已知见证也不互为负例。
- C_q^- 是 ranking-only weak/unknown target，不是已确认非法 join；W 不给 E 或 ET 生成新的二元负标签。仍有漏标 target 的风险，不能称 PU 无偏估计。
- 基础 L_path/KD 仍可能间接压低 unknown；给字段取名 unknown 并不会消除这个效应。
- 负 target 路径中的局部 QE/ET 可以各自合法。W 的参数化仍可能向局部边传播不合适的惩罚，故需记录已知正局部边上的梯度。这正是尚未修复的结构限制，不在方法介绍中隐藏。
- 若只固定一个 E 对多个 T 做 softmax，`s(Q,E)` 在归一化中消掉，根本学不到首跳信用。上述跨路径对比一般不完全消掉，但仍须记录首跳有效梯度；不能因名为 path loss 就假设两跳都受到有效监督。
- 只凭训练正记录，可以验证正路径责任、支持下界与 Recall；**不能**验证“错误属性路径占比下降”或“模型能拒绝同实体错误属性”。后两者只在有独立作用域标签的集合上报告；标签缺失写 null，而非 membership complement。

### 7.5 训练与推理伪代码

以下是待实施伪代码，函数名不表示现有 CLI：

```python
# 只在训练数据管道加载 privileged witness metadata。
for batch in frozen_path_schedule:
    scores = score_original_visible_objects(batch, model)
    base = path_loss_with_known_positive_masks(scores, teacher_scores)
    terms_by_query = {}
    for q in batch.queries:
        negative_targets = present_evidence_targets(q) - train_fit_positive_targets(q)
        if not negative_targets:
            continue
        bq = nested_logmeanexp_negative_paths(q, negative_targets, scores)
        # 记录真实 q/t/a/row 层级，不将所有记录直接扁平平均。
        for witness_group in train_fit_groups(q):
            supported = witness_group.evidence_ids & candidate_evidence(q, witness_group.target)
            supported = unique_content(supported)
            if not supported:
                continue
            aw = logmeanexp(path_scores(q, supported, witness_group.target, scores))
            add_hierarchical_term(terms_by_query, witness_group, softplus(bq - aw))
    loss = base + 0.1 * hierarchical_mean_or_zero(terms_by_query)
    optimizer_step(loss)

# 部署没有 witness_group、GT column 或 recovery values。
indices = build_three_object_indices(model, frozen_visible_embeddings)
paths = retrieve_QT_and_QE_ET(query_visible_embedding, indices, fixed_budget)
return frozen_retention_and_union_direct_ranking(paths, N=50)
```

W 初版只使用共同训练列表已有的 E，不因 W 打开而额外插入 GT evidence。若 W 可用路径太少，报告资格率与有效更新数；不把空目标解释成见证机制被否定。后续采用训练标签补入正路径是合法的训练扩展，但必须在 control 和 W 中使用相同补入列表、记录额外 Teacher 打分，另行审阅，不进入这次单因素结果。

### 7.6 W 能修什么、不能修什么？

可能修正：容易／错误路径承担所有正 target 信用；一行或一个属性标注数量多而主导训练；正路径学习被 target-only 代理目标忽略。

无法结构性修正：同一 E 对不同 Q 需要不同 T 排序，而 s(E,T) 完全相同；未编码的实体—属性绑定；pooled z 已丢失的信息；共享 P 的梯度干扰。若不同 Q 的可见内容本就不能区分需求，训练 privileged labels 也不能给部署模型凭空补出区分信息。

特别地，任务不是要求模型猜出构造程序当时删掉了哪个属性。相同可见输入对应多个合法目标／见证时，应保留这些正例；不能把构造选择造成的标签差异硬解释成可预测的线上属性意图。

## 8. 方向 A2：潜在桥接状态——比较而不默认增加一组模型

### 8.1 正确的同状态组合

假设 K_w 个无具名潜状态，用 $a_k(Q,E)$、$b_k(E,T)$ 描述两条边在状态 k 下的兼容性。需要先对齐状态再边缘化：

$$
g_{same}=\log\sum_k\exp(a_k+b_k).
$$

不能用：

$$
\log\sum_k\exp a_k+\log\sum_j\exp b_j,
$$

后者包含 k≠j 的交叉解释。分别 max 也有相同问题。状态索引应由共享参数／统一代码本对齐；独立训练两个可任意置换的 latent heads 再按编号相加没有语义保证。

即使两跳选中了相同编号k，也只排除了“不同状态编号的交叉组合”，不保证同一实体、同一属性和值域；粗主题状态同样可能编号一致。

无列名不等于无约束。训练可用 a/row 支持对同一个 latent posterior 作多标签／集合监督，但同属性在不同实体上聚合、同实体不同属性对照和 source-group 泛化都要检查。单纯熵均衡只能防使用频率塌缩，不能保证状态代表属性，更不能排除主题／实体 ID 捷径。

### 8.2 ANN 能力必须如实核算

| 潜变量形式 | 是否等价于原 d 维单次 IP 搜索 | 代价／局限 |
|---|---|---|
| 每个 k 的 b_k 为线性内积；对 k 取 max/LSE | 一般不等价 | 对每 latent 搜索、合并后重算；first-hop 若也用 max/LSE，同样有问题 |
| target-independent 权重 π_k(Q,E)，共同 u_T，使用期望分数 Σπ_k b_k | 等价：h=Σπ_k R_k^T u_E，检索 h^T u_T | 它是期望，不是存在一个高分见证；混合会稀释少量强状态 |
| π_k 同时依赖 T | 一般不等价 | 需候选后打分或额外近似，不能隐称一次 ANN |
| 每个 T 有 K_w 个状态向量，把它们拼接 | 一次 IP 得到的是和，不是 max/LSE | 向量维数增大至 K_w d；不能冒称原分数的精确实现 |

max 的特殊实现可把 `(T,k)` 当扩展索引记录、采用块稀疏向量和 query 侧偏置做一次扩大维度的 MIPS；但索引变成 K_w 倍的记录，唯一 target topK 需处理重复命中和追加取数，已经改变检索粒度／预算，不进入本轮主线。

若采用 K_w=2 的多搜索近似，原 ET20 必须分为10+10，不能每 latent 各取20后仍称等预算；搜索向量数翻倍，合并去重后可能不足20，且不能保证召回真实 LSE top20。额外 over-fetch 必须单列预算扩展。

**本轮不首发离散 latent 网格。**它同时引入状态识别、对齐和近似搜索三个不确定性。先做 W；若确有 query-independent 第二跳歧义，优先下节只有一个条件检索向量的连续残差。

## 9. 方向 A3：推荐的条件性结构扩展 H

### 9.1 定义与完整评分

不读取列名标签，用可见对象压缩向量构造连续桥接状态。对每个 evidence 模态 m，设 r_h=16：

$$
p_Q=A_{Q,m}u_Q,\quad p_E=A_{E,m}u_E,\quad
w(Q,E)=p_Q\odot p_E,
$$

$$
h_m(Q,E)=R_{mt}^\top u_E+B_m w(Q,E),
$$

$$
s_{ET\mid Q}(Q,E,T)=h_m(Q,E)^\top u_T,\qquad
g_H(Q,E,T)=s_{QE}(Q,E)+s_{ET\mid Q}(Q,E,T).
$$

其中 $A_{Q,m},A_{E,m}\in\mathbb R^{r_h\times d}$，$B_m\in\mathbb R^{d\times r_h}$。QT、QE 原式保留；只改变第二跳 primitive。初始化 A 为 seed13 非零小矩阵，B=0，故 step0 所有分数等于 S0。B 与 A 不能同时全零。全部 P/R 仍可训练。

两个模态共新增 $2\times3dr_h=98,304$ 个参数（d1024,r_h16），不含可选偏置；初版不使用偏置。H 是具有 QE 乘性交互的低秩三元评分，不是已经被证明的实体—属性符号见证。它让 ET 排名可以随 Q 改变，但不保证完整事实绑定，也不保证多个 E 聚合时属于同一属性。

### 9.2 精确 ANN 分解及成本

目标索引仍为 $u_T=P_Tz_T$，**与当前 Q/E 无关**。对每个已召回 E 只构造一个 h，执行同一个 target 索引的 top20 搜索。因此模型评分严格是单次 IP；HNSW 等近似误差仍另测。

- 每 query 仍最多40个 ET 搜索向量，不新增 latent 检索或列检索。
- $A_Eu_E$、$R_{mt}^\top u_E$ 可离线或按模型缓存；$A_Qu_Q$ 每 query/模态计算一次，$B(p_Q\odot p_E)$ 每 QE 计算一次。
- 新缓存键必须含 `(query_visible_hash, evidence_visible_hash, model_hash, modality, score_protocol)`，不能继续只按 `(E,table)` 存条件 query 或条件 ET scores。
- 当前 06 的 `_relation_queries` 缓存的是 E 的查询向量，`search_many` 仍可对重复 E 发起搜索；没有证据说明已经缓存所有 ET 返回列表。因此不能虚构“条件化必然增加40次 ANN 调用”。要分别报告当前实现实际增加量，以及可跨 query 复用 ET 返回列表的工程上限对照。[L5]
- 若未来基线缓存 ET20 列表，其冷／热缓存命中可以节约查询；H 失去跨 Q 的这一复用。必须用同一 query 流分别测冷缓存、热缓存，不能只报模型前向成本。

### 9.3 训练、控制与停止条件

H 使用与 P-W 完全相同的训练 path candidates、known-positive masks、Teacher 分布和 $L_{path}+0.1L_W$。Teacher 仍是固定局部 Teacher；它没有变成三元 Teacher。这可能约束 H 的可学增量，记录监督/KD 对 B 的实际梯度，不能预设蒸馏帮忙或自动关 KD。

若启动 H，新增一组成对臂：

- **H-Eonly**：将 p_Q 的输入替换为训练开始前冻结的非零常量 c；其余矩阵、参数量、loss、初始化与调用路径相同。它是 E-only 的额外重参数化／容量控制。
- **H-QE**：使用真实可见 Q 的 u_Q。

两臂都从 S0 开始，B=0，固定178步，r_h16；新增矩阵 LR=1e-5、weight decay0.01，P/R公共设置不变。相对 H-Eonly 的优势才能较清楚地支持 query conditioning，而非仅新增参数。另与原 P-W 比较实际总体 Recall 和成本。

H-Eonly 的常量 c 为 train-fit 可见 u_Q 的均值，开始前固定，若范数过小则使用预登记 seed 的单位向量；不得按结果选择。它不依赖任何当前 Q，且其线性 ET 增量在 full R 下原则上可吸收进 R。

必要诊断：固定同一 E、改变真实 query 的可见内容，检查正 target 排名是否按已知见证分歧改变；额外做 Q 随机置换／置常量的输入敏感性，但“分数变化”不等于正确变化。优先使用确实共享 E、而所需 target/见证不同的 train-fit/source-group 诊断对；若没有这样的记录，只能称一般条件化检索实验，不能声称已验证特定歧义机制。

**不同Q共享E且标了不同正target，并不自动构成排序矛盾**：两个target可能对两者都合法，或同一个ET20名单已经容纳它们。若要声称局部ET存在不可化解的结构冲突，需有作用域可靠的反驳／支持依据，或证明在固定20个目标预算下共同名单不足以覆盖被检验的需求。只有已知正例时先做覆盖与条件化收益实验，不把qrels非membership当反证标签。

若 H 仅拟合 Teacher、B 几乎为零，或提升 list 指标但自然 Recall 不变，保留负结果。若依赖新增线索但成本超预算，报告质量—成本取舍，不免费归入原主线。

## 10. Task A：仅完成 Stage 1 必需的正确性与基线冻结

不把任务扩大成整个仓库工程整治。后续实施先完成：

1. 冻结 train-fit 监督、S0/PCA/Teacher、原始 embedding、对象可见内容及 path/edge 评分协议指纹。
2. 按第4节输出完整、去重、逐 query 的 Recall@10/20/50 与 CandidateRecall@50；每项保留命中ID及整数分子／分母。
3. 核对 known-positive mask 是实际构造结果，不是硬编码计数；修改不可见 split 标签不应影响训练监督输出。
4. 角色迁移 step0 验证所有五关系分数相同，原 d 维 ANN IP 与直接评分相符，索引／缓存包含用途角色和模型身份。
5. 从 S0 自然检索重新得到主协议的完整N50候选和 Recall。255/678只能作为旧路径读数，不当作本轮 Recall 起点。
6. 确认空 evidence 通道缺席、真实 E路径不丢失、GT不注入自然检索。旧 Task F 的跨 query 正例并集错误不能复制到新指标中。

必要合成测试：多个相关目标的宏 Recall；重复 target 去重；missed query记0；固定N重排的Recall@N守恒；仅改GT辅助字段不改变部署分数；role migration恒等；同一表两种调用用途；unknown不进confirmed BCE；空通道无票；不同角色／Q的缓存不串用。

本轮编写文档不执行这些测试。后续实施通过这一层即可开始 Stage 1，不等待 Stage 2 生成修复。

## 11. Task B：共享表示／信用的必要诊断及 KD 控制

### B0. 只读现有候选与当前目标的诊断

用第3节固定 train-fit batches，对 S0 与可用历史退化 checkpoint 做第6.4节的真实一步更新分析；分别记录 edge 与 path，不沿用 R11 的单批历史 KD 梯度代替当前目标。

每关系记录 raw score、sigmoid值及导数分布、ranking/KD梯度、实际 ΔP/ΔR、正边 exact rank、target hub频率。对同一固定 E 评价全部 target 的 exact 排名，排除第一跳样本变化造成的假条件收益。

Teacher 在**相同已冻结 hard candidates**上比较 Raw/PCA/Teacher，报告关系宏／实例微平均、多正邻居覆盖与难度分层。不能将 Raw 挖出的 hard list 上 Raw 较弱直接当 Teacher 普遍更强。用已有未参与训练的 cal-check/开发诊断池作描述性补充，不据此改训练标签。

首批只是记录 candidate 陈旧度和 source bias，不刷新候选／替换损失。若后续结构臂失败且诊断明确指向这些因素，再另轮单因素检验，不在本轮自动展开。

### B1. 补同 hard-candidates 的 KD-off

| 字段 | 规格 |
|---|---|
| 假说 | C-candidates 的收益可能主要来自 supervised hard candidates，而不是 KD |
| 唯一变量 | KD系数0.3→0 |
| 初始化 | 同一原PCA-1024、R=I，不从已蒸馏S0起步 |
| 候选、标签、顺序 | R12 candidates seed13的完整356个冻结batch；正例与完整mask相同 |
| Teacher | 同一checkpoint与完整pair分数；off臂不使用它作loss，但身份仍记录 |
| P/R | 全部按公共设置可训练 |
| 目标 | 第5节L_edge，仅KD项为0 |
| 固定更新 | 356，保存0/178/356，端点为356 |
| 主端点 | 自然ANN完整dev总体Recall@10；另报@20/@50及分层 |
| 必要诊断 | 五关系exact、ValidPool、支持责任，分母不变 |
| 成本 | 新Student训练；若完整缓存可复用，新增Teacher推理为0，不抹去原Teacher累计成本 |
| 分支 | 无KD增益就缩小知识转移主张；不阻塞角色/W研究，也不自动把KD-off改为主线 |

只有训练入口、loss、mask、参照、候选顺序与优化器轨迹可重放时才复用历史KD-on356。当前源码不是历史checkpoint代码的自动替身；不能核验时补一个同版本KD-on356控制（最多新增一个），而不是假装原结果可直接配对。此处是候选/KD归因控制，不与S0续训混成一个比较。

## 12. Task C：首先检验 table 角色拆分，两个 edge 续训臂

### C0. 统一训练列表

首批使用已完整Teacher打分的 R12 C-candidates seed13固定356-batch schedule，按原顺序循环重放；续训178步使用其前178个batch。这样不在表示对照中引入新的候选挖掘和Teacher补分。它是“在既有hard候选上的受控续训”，不是历史659/1318步数据轨迹的逐位复制。

### C1. 训练臂

| 字段 | C-S：Shared | C-R：Split |
|---|---|---|
| 假说／变量 | 同目标下共享P继续学习的控制 | 仅table投影分成Q/T两个角色 |
| 初始化 | S0 | S0，P_Q=P_T=S0.P_table |
| Teacher、候选、mask | 共同冻结的R12 Teacher/hard schedule；完整known-positive与KD mask | 完全相同 |
| 参数更新 | 原P/R均更新 | P_Q/P_T/P_text/P_image及原R均更新 |
| 目标 | L_edge | 同L_edge；table anchor按角色平均保持总权重 |
| 优化器 | 新AdamW，公共LR/WD | 同规则，不额外调LR |
| 更新数 | 178 | 178 |
| 主评价 | 自己的自然ANN池＋冻结F1，总体Recall@10 | 同预算、自己的索引／池 |
| 诊断 | 固定E的ET exact、各关系实际更新、QE、路径与支持 | 同上，增加角色子空间分析 |
| 成本 | 原模型 | 新增4,194,304参数；target索引条目／维度不变 |

保存0/45/89/178；45/89只做固定训练诊断，不据dev反复选点；0/178评价完整dev。共同S0正式起点只需一套参数等价基线；独立ANN重建波动另记录，不伪装为模型差异。

**成功解释：** C-R的总体Recall@10高于C-S且高于S0，并符合第15节分层／成本约定，才是新学习候选。若只高于退化C-S但低于S0，称损伤缓解。若Recall提高但证据路径退化，称Recall收益伴随evidence损伤，不称见证学习成功。

若拆分主要改善固定ET却不改善自然Recall，说明局部收益被QE、候选准入或direct评分限制，不把exact指标替代主端点。

## 13. Task D：独立的 path 见证监督对照，两个臂

与角色实验不串接：**P-S/P-W仍从共同S0出发**。不从C-R训后checkpoint启动W，否则同时改变初始化和监督，无法隔离W。

### D0. 冻结候选与有效监督

使用现有已物化的 train-fit `TargetExample` candidate records及R12 path候选构造规则，保留全部candidate IDs、原正例、evidence绑定与list宽度；按seed13确定性样本顺序物化178个batch，每batch64，数据不足时固定循环，不在线重采。

两臂用同一份列表。额外已知正target必须按train-fit完整正集合保护。现有列表若包含训练GT辅助补入的路径，登记其比例；两臂同等使用，不能称自然候选。主评测一律不补入。W只在现有列表中取recovery支持的交集，不因W开关改变candidate内容。

Teacher pair分数与真实candidate mask对齐；缺分先给出唯一pair数和费用，只有同一固定Teacher离线补齐后才计算KD，禁止0补缺或将局部Teacher mask冒充全列表。精确历史列表未能恢复时，使用现有物化记录建立并冻结R13新列表，两个控制都重跑，并标明不与旧path结果作逐位比较；不重做监督数据集。

### D1. 训练臂

| 字段 | P-S：target-only控制 | P-W：正见证辅助 |
|---|---|---|
| 假说／唯一变量 | 现有path目标的同版本对照 | 在相同图和分数上增加0.1L_W |
| 初始化 | S0 | S0 |
| Teacher／labels | 固定Teacher与train-fit目标标签 | 同Teacher；仅增加train-fit recovery分组监督 |
| 候选／mask | D0冻结列表、全部已知正target保护 | 完全相同；W负分母不含已知正target |
| P/R | 原共享三类P及原R均更新 | 相同 |
| loss | L_path | L_path+0.1L_W |
| 步数／optimizer | 178，新AdamW，公共设置 | 相同 |
| 主评价 | 完整dev自然ANN的总体Recall@10 | 同上，不在有W标签子集单独选模 |
| 诊断 | 每跳exact、责任、支持行分布、候选标签资格率 | 同上，W有效组／梯度覆盖、unknown率 |
| 增量成本 | 原path前向 | 复用已算path score做掩码聚合，无新增在线参数／索引 |

保存0/45/89/178，完整dev只评0/178。若W有效训练记录为0，不运行空辅助臂并宣布方法失败，而是记录“无有效监督资格”；角色任务照常完成。有效数据存在但分布窄，可执行诊断性训练，必须报告其范围，不引入64正/64负的统一门槛。

不默认添加旧 `path+0.1edge`。它是必要时的维护边控制，而不是本轮全部新方向的共同额外变量。

## 14. Task E：最多一个条件扩展，不同时展开所有方向

首批完成后，只允许从下面选择**一个**分支，最多新增两个178步训练臂。所有决定先写入条件触发记录，普通失败不靠改名重跑。依据若来自dev，则明确属于适应性开发，不称独立确认。

| 触发证据 | 唯一扩展 | 对照和归因 |
|---|---|---|
| C-R有Recall收益，P-W也有Recall收益，且二者诊断对应不同问题 | 补P-R与P-RW：从同一S0迁移Split，分别L_path和L_path+0.1L_W | 与原P-S/P-W合成同目标、同初始化、同候选的2×2。不能用edge阶段C-R充当path阶段P-R |
| C-R有效，需要检验更小参数是否足够 | C-residual：第6节r16输入空间残差，同C-S的178步edge续训 | 可复用完全相同配置的C-S/C-R；只比较一个rank，不扫r；无需同时做组合 |
| 正见证辅助仍受固定E对不同Q的目标歧义限制，且存在可见输入可区分的实例 | H-Eonly/H-QE，均含同一个W，见第9节 | 参数量与更新数匹配；只把两者差异归于条件输入，不扩大搜索 |
| Path更新明确破坏健康边，但W不解决；未见必要条件化证据 | P-preserve：L_path+0.1L_edge_hard，使用C1完整hard candidates/KD mask | 与P-S匹配；不是旧local-list KD维护的重复；额外edge前向列入成本 |

如果需要区分容量和重参数化，可用“同压缩子空间内角色适配”替代本轮Residual分支，但不能两个都做后挑最好。其full R函数等价性见6.2；新增参数量、anchor与更新几何必须披露。

### 14.1 组合的正确读法

只有 P-S、P-R、P-W、P-RW 全部保持同一 path schedule、Teacher、初始化评分、优化器政策和178步，才计算：

$$
\Delta_{interaction}=(R_{P-RW}-R_{P-R})-(R_{P-W}-R_{P-S}).
$$

正交互不保证独立确认，零交互也不否定两个改动可以相加。若只新增P-RW而没有P-R，就只能报告组合效果，不能识别角色×见证交互；本计划不采用该不完整归因。

角色拆分缓解表示共享／优化，W改变正路径信用；两者互不自动解决对方的问题。只有单因素证据和同目标配对都支持时才组合。

### 14.2 暂不自动触发的事项

text/image角色进一步拆分、Teacher重训或token/pooled对照、动态hard refresh、source-centering、raw/sigmoid改动、离散latent多索引、属性负例classifier、覆盖保留模型、在线Teacher、解冻骨干、额外跳数均放下一轮。当前记录相应诊断即可。

source bias可在同池离线重算进行不选模的检查：给每E所有T分数减同一个、由训练可见固定目标参照集估计的偏置，不改变该E的ET排序，只观察跨E聚合变化。它不能救回已掉出ET20的目标。首批不将该变换和角色/W混合。

## 15. Recall 选模、延长与负结果规则

### 15.1 固定端点是主要比较

预声明三个首批比较：

1. **C-R178 − C-S178**：完整dev、冻结主F1的总体Recall@10，检验角色表示干预。
2. **P-W178 − P-S178**：同口径，检验见证辅助监督。
3. **hard KD-on356 − hard KD-off356**：同口径，检验KD条件贡献；不是候选协议之间比较。

分别列S0、固定端点、开发selected三张表。已执行的45/89训练诊断不能在看到曲线后补做dev并挑最好。主比较仍固定178或356，selected不替换它。

### 15.2 首批质量与成本约定

候选进入“当前可用主配方”的条件：

- 总体Recall@10高于共同健康S0；宣称某干预有效还应高于其同预算控制。
- 相对S0，implicit和explicit Recall@10各下降不超过0.02；这是沿用历史取舍容忍度的**新计划质量护栏**，不是显著性标准。未通过者完整保留为取舍结果，不删除。
- 主搜索预算不增加；记录训练／Teacher补分／索引／在线全部成本。角色/W等轻量改动以同硬件冷/热查询的P95增幅≤10%为默认部署投入护栏；超过时报告成本取舍，不能用未计费缓存宣称通过。统计噪声大时先补测延迟，不调模型。

**路径诊断不是隐性主指标。**若Recall改善而QE/ET、ValidPool或支持下降，仍如实报告Recall增益，但标为evidence退化风险；不能自动声称“证据学习更有效”。不以ValidPool、nDCG或≥3行替代Recall选最终优胜臂。

### 15.3 最多一对延长

C或P中至多一个实验臂及其匹配控制延长到**累计356个新阶段更新**。默认投入条件：实验臂178步总体Recall@10比控制至少高0.005、且高于S0，并通过分层／成本约定；0.005是预先声明的计算投入差值，不是统计显著性阈值。

多个方向满足时，按178步总体Recall@10更高者优先；同分按@20、@50、更低在线实测成本、固定配置ID决定。两臂同样延长178步，保持候选协议；不因为对照退化就提前停对照而让实验臂继续。延长结果若下降，保留178/356两点，不改变主固定178步比较。

若实验臂只比退化对照好、但不及S0，不触发上述延长，结论为“损伤缓解”。若仅CR@50改善、R@10未改善，称候选池收益／后续重排空间，不称主端点成功；可以冻结此池做以后Stage2研究，但不能事后换主要指标。

### 15.4 开发checkpoint与配方选择

仅在实际完整评价过的0/178/356候选中，以预先定义的质量／成本约定确定可用集合，再按：

`总体Recall@10 → 总体Recall@20 → 总体Recall@50 → 较低实测在线成本 → 更早step → config_id`

选择。所有数值用未四舍五入值，比较容差1e-12。没有新臂比S0更好，就保留S0；不能把选回step0写成path或role训练成功。

B1 KD-off是消融，不能因其最高就默认为论文主方法；若它明确更好，报告Teacher贡献负结果并另外决定论文主张，主线身份变更需审阅，不通过隐性关KD掩盖。

NaN/Inf、label泄漏、错模型索引、role/cache串用、mask错误与score space不一致立即停止对应运行，作为实现失败。普通效果不佳保留到固定端点；不在本轮加大规模救援网格。

## 16. 统计、机制报告与实际计算清单

### 16.1 配对统计

总体／implicit／explicit Recall均先计算每query比例，再在各自预定query集合宏平均。按`source_table_id`做配对cluster bootstrap，10,000次、seed13，保存逐query的命中集合、分母、source与delta。

每次重采样复制所抽source中的全部query，再按query宏平均，不能先给每source均匀平均而无意改变estimand。若同一query有多个source关联，先在冻结清单定义其稳定group；不按输出改变分组。不得用正对净增数除以query数代替Recall差值。

路径指标按相应 `(q,t,a)` 或 `(q,t)` 定义独立分母，行级指标另报；不能把它们接成同一漏斗。漏检记0，不取两臂共同成功交集。

当前dev已被反复使用，所有区间是探索性开发区间；不做“预注册显著”声明。种子17/23只补唯一被冻结的新配方及其匹配控制，Teacher固定时称条件Student方差。不得将三Student种子当三条独立Teacher–Student训练链。

### 16.2 必要机制读数，不替代Recall

| 指标／分析 | 作用与边界 |
|---|---|
| QE已知正邻居Recall、any-positive命中 | 分开报告；前者对全部已知正邻居，后者不能冒充Recall |
| 固定E的ET exact rank/Recall | 将第二跳功能漂移与第一跳集合变化分离 |
| ANN–exact gap | 判断索引近似，不用exact替代主部署Recall |
| ValidPool、ValidB、ValidPath@K | 已确认路径下界，不是target Recall |
| 支持行0/1/2/3/4/5分布 | 保留属性作用域，无三行门槛；行并集与唯一路由分开 |
| 已知支持路径责任 | 正target的LSE责任分布、W覆盖情况；unknown责任不等于错误责任 |
| 错误属性／错误实体路径率 | 只在独立标签覆盖子集报告，缺标签写null并报告覆盖率 |
| Hub、分数范数、source bias | 竞争解释；不能仅相关就定根因 |
| Role实际更新与有效子空间 | 区分更新干扰、输入子空间、新增参数影响 |
| Rescue/displace、D100内外 | 对固定S0 D100和模型自身D100双口径报告；不使用GT在线分流 |

Target-level qrels不完整时，已知Recall是固定标注口径，不是所有合法目标的穷尽Recall。新增unknown候选不计入分子，也不自动称非法；不在单个实验臂运行后独自扩充qrels。后续统一独立补标应重评全部冻结臂。

### 16.3 首批预算与上限

| 工作 | 新Student更新预算 | 额外大模型成本 |
|---|---:|---|
| Task A/B0 | 0正式训练步；临时一步诊断另计 | 复用现有embedding；Teacher缺分单列 |
| C-S/C-R | 2×178=356 | 复用固定完整edge Teacher分数 |
| P-S/P-W | 2×178=356 | path候选缺失Teacher pair先估算、冻结后补分 |
| B1 KD-off | 356 | 已有edge分数可复用 |
| 必要KD-on同版本重放 | 最多356 | 不重训Teacher |
| 首批核心合计 | 1,068；需KD-on重放时1,424 | 不把固定Teacher补分视为免费 |
| 至多一对延长 | 2×178=356 | 同协议缓存复用 |
| Task E条件扩展 | 最多2×178=356 | 视共同候选是否缺分；先登记 |
| 条件Student种子复核 | 唯一配方与匹配控制，按冻结端点 | 配方冻结后单独批准投入，不扩全网格 |

以上是更新数，不是GPU小时。任何Teacher补分必须报告unique pairs、总occurrences、token数和实际时长。既有约914万补分pair属于历史成本，不重复计为新成本，也不能从方法总训练成本中删除。

**资源最小版：**先Task A/B0和C的两个178步臂；随后D的两个178步臂。先不做延长、Residual、H、组合或新Teacher。B1可以排在这四臂之后，但在宣称KD有效前必须完成。没有新人工负标签不阻塞这个最小版。

索引成本单列：构建时间、条目、维数、字节、常驻内存；在线成本分QT、QE、ET、角色／H向量构造、union-direct补算、保留、融合。记录query流、batch大小、冷/热缓存、P50/P95、硬件、重复测量方式；缺失为null，不写0。

## 17. Task F：后续 Stage 2 排序研究，不是本轮前置门槛

### 17.1 进入条件

满足以下条件即可准备后续Stage2：

- Stage1已有一个冻结且可复现的健康配方，允许仍为S0；不要求R13一定胜出。
- 该配方的自然候选、N50、K10/20/50、qrels分母、D1、实际搜索预算与评分字段已冻结。
- Stage1新臂若进入后续比较，其索引与候选已完整生成；与固定池读取实验严格分开。

**不要求Stage1先证明尚未执行的生成或最终重排成功。**进入Stage2后才完成生成链可观测性、必要oracle和归因审核。

### 17.2 保留原R13有价值的排序与生成设计

二阶段默认N50、M20、B4，列拒绝阈值None；真实evidence通道分数e_T用于表先验，不以`-final_rank`替代；`stage2_table_score`也不能覆盖成错误分数。按coverage、mean similarity、Stage1原排名排序；direct/evidence同分保留direct；`joinable`不筛选。

当前接口仍把未执行无分数候选放在`unattempted_candidates`。后续评测需明确实施完整N榜单：已评分部分按真实分数排序，未处理部分按原Stage1排名接尾，score仍为null；不把此策略误说成接口已完整实现。全部处理条件保留同一N个唯一target，Recall@N必须守恒。

主Stage2结果继续用预定完整query集合的总体Recall@10，分层与@20/@50并列。算力不足时先按source-group固定概率抽取自然样本，报告为样本估计；旧96/32富集集合只用于机制诊断，不替代自然总体。没有样本抽样权重时，不把富集样本裸均值推广到全dev。

### 17.3 后续oracle与反事实顺序

1. 在train-fit小型工程材料上保存原始completion、token数、finish reason和parse status，区分合法空值、解析失败与截断；当前全空根因仍未知。
2. 固定同一row/target/column，比较正常E、确认支持原资产、oracle行+span/ROI；oracle只作为诊断，不注入Stage1 Recall。
3. Oracle值离线填入后，检查固定scorer是否能改善目标排序；同预算支持与放宽预算的上限分别报告。
4. 固定N、实际列选择、M名单和生成row mask，比较empty-column、同generator的no-E、real-E；真实错误属性替换仅用独立标签。不能把M=0作为唯一no-E对照，因为会改变未处理状态。
5. 对新增grounded正确值做单target置空干预，其余候选分数固定，检验该值是否驱动排名；没有新增正确值的Recall变化仍仅为排序变化。
6. 同一 `(q,t,column,row)` 的text/image/mixed及删图实验，与重新检索text40/image40/mixed20+20分开；匹配实际token、窗口、ROI与生成成本。当前单条证据生成不等于同一行图文联合推理，联合读取须单列扩展。

后续生成/定位失败可以停止扩大全链生成，但**不能反向阻断已获授权的Stage1表示和Recall研究**。错误高分及qrels遗漏需独立审核；不恢复“至少3行”最终门槛，也不通过提升布尔阈值伪装修复排序。

## 18. 对旧 R13 Task A–F 的映射

| 旧任务 | 新处置 | 理由／新位置 |
|---|---|---|
| A：排序与评测迁移 | 拆分并改写 | Stage1 Recall、N/K、mask、role/index正确性前置至新A；Stage2完整榜单迁移后置新F |
| B：正确材料生成可评分值 | 推迟 | 保留观测、oracle、解析设计，不是Stage1通用门槛；新F |
| C：补属性→排名因果链 | 推迟并改指标 | 保留固定候选反事实，主指标改总体Recall@K，不再implicit nDCG；新F |
| D：coverage/mean评分诊断 | 推迟 | 新F条件诊断；不影响首批P/R学习 |
| E：path与维护边两臂 | 替换为角色C、见证D | 先验证两个核心假说；原维护边仅留为新E的一个条件分支 |
| F：支持保留与模态 | 分拆 | 正recovery直接服务训练W；独立错误属性标签与Stage2模态反事实后置；不先训练未知为负的支持classifier |

全篇删除旧implicit nDCG主端点、ValidPool最终选模优先级、生成成功后才扩大Stage1的普遍依赖。nDCG如为历史兼容保留，只能放附录，不影响选模、延长或主声明。

## 19. 模块修改与实现验收

### 19.1 已在附件看到的真实模块

| 模块 | 修改职责 |
|---|---|
| `src/mmdd_stage1/models.py` | `StudentJoinabilityModel.project/score_pairs/score_embedding_matrix/relation_query/index_vector`传递运行时Q/T角色；Split/Residual；条件H接口；导出有效投影与配置 |
| `src/mmdd_stage1/scoring.py` | edge/path批量评分中明确endpoint角色；复用投影时加入role key；暴露逐path raw scores供W；条件ET不得按(E,T)跨Q去重 |
| `src/mmdd_stage1/objectives.py` | W的层级正见证bag loss；mask、空集合和数值稳定性；不静默改原PathAggregator |
| `src/mmdd_stage1/training.py` | optimizer参数组、role平均anchor、固定step保存/评价、当前分支梯度与实际一步诊断；仅显式开关启用W |
| `src/mmdd_stage1/retrieval.py` | target索引统一P_T；query table用P_Q；search cache role-aware；H用(Q,E)查询target；主完整N输出与固定融合读数 |
| `src/mmdd_stage1/retrieval_aligned.py` | 仅确有需要时复用候选逻辑；首批不新增动态refresh、不把target-specific不支持升级为全局负 |
| `src/mmdd_stage1/row_support.py` | D1保持不变；仅补必要的属性作用域诊断，不把新支持预测器混入首批 |
| `src/run_stage1_r12_task_c.py`及extension/candidate/replay入口 | 作为实际历史参考，冻结／重放候选和Teacher；不覆盖R12结果 |

08还定位了`features.py`、`construction.py`；它们的完整依赖没有随06提供。实施时核对数据类／checkpoint加载器，保持OBJECT_TYPES仍为table/text/image，不为了角色拆分重建Teacher的9种relation key。`checkpoints.py`等导入模块要检查，但本轮不声称读过其完整实现。

### 19.2 需要新增但尚不存在的接口

建议集中增加 `src/mmdd_stage1/witness_supervision.py` 处理train-fit见证分组，及薄入口 `src/run_stage1_r13.py`；**这两个名称是待实现建议**。不得假装现在存在`--role-projection`、`--witness-weight`、`--query-conditioned-et`等CLI参数。先按真实解析器实施与核验，再在运行清单记录可执行命令。

角色接口建议显式传 `source_role/destination_role`，其中非table模态首批固定原行为；score/cache key包含可见内容指纹、对象类型、角色、模型版本。`score_pairs`当前按 `(object_type, object_id)` 缓存投影，必须同步改，不能只改`project()`后留旧cache键。[L5]

checkpoint配置保存projection mode、有效P、role anchors、全部R及可选H；保存原PCA参考。训练阶段身份与部署身份不能靠某固定query-ID清单推断。

### 19.3 后续实施必须通过的具体行为检查

- Shared迁移Split／Residual时，QT/QE/ET和完整path分数在step0相同；随机数影响的ANN与浮点容差单列。
- 分数矩阵、逐pair、ANN query/index三种路径返回一致raw score；更换调用角色真正使用对应P。
- R13训练候选／正例mask不受不可见split标签改变影响；unknown不进入confirmed BCE。
- W仅增加loss，不改变部署函数；关闭W时无辅助标签被读入forward；多个合法属性不互标负；同内容副本不增加标签权重。
- W的候选为空、只有正target、无有标注见证时正确跳过；有效样本数真实统计，不写死通过。
- H在B=0时等于原ET；相同E不同Q的缓存隔离；target索引不随Q/E变化。
- 自然评价不注入GT；同模型固定池与不同模型新索引的结果身份明确。
- Recall多正例、去重、空结果、N>K及Stage2固定集合守恒正确。

这些测试在后续执行授权后，用隔离的临时配置／合成对象完成；不调用真实生成器来验证线性代数性质，不加载秘密服务配置。输入合成性质测试不冒充真实样本实验。

## 20. 交付物、可写结论与少量待确认项

### 20.1 后续实验交付物

```text
work/stage1_optimization_r13_20260909/
  PLAN_FROZEN.json
  taskA_stage1_protocol/
  taskB_diagnostics_and_kd/
  taskC_role_projection/
  taskD_witness_supervision/
  taskE_conditional_extension/
  taskF_stage2_deferred/
  statistics/
  runs.jsonl
  RESULTS.md
```

`PLAN_FROZEN.json`登记本修订稿hash、具体模型／数据／候选／Teacher与完整依赖指纹、主榜单规则、N/K预算、固定steps、选模/延长规则。每臂保存单一变量、实际命令、父checkpoint、optimizer政策、有效参数数、所有候选与mask、逐query榜单、逐pair/row诊断、成本与停止原因。

主结果表每行至少为：`arm, parent, objective, projection_mode, step, all_R10, implicit_R10, explicit_R10, all_R20, CR50, delta_to_S0, delta_to_matched_control, cost`。另表报告诊断，不将它们揉成加权总分。

### 20.2 结果解释

- Recall无增益：主结果为负；即使ValidPool上升也不改口径。
- 比退化控制好、未超过S0：损伤缓解，不是新增学习收益。
- 角色拆分提高Recall且实际跨角色损害下降：支持角色共享干扰的条件性解释；不自动证明容量不足。
- W提高Recall和已知支持责任：支持训练见证监督有用；没有独立负标签时不声称学会排除错误属性。
- H-QE优于同参数H-Eonly、自然Recall改善且成本可接受：支持query-conditioned第二跳的价值；不称标量局部primitive已经足够。
- Recall改善但direct掩盖evidence损伤：如实报告；不能写为完整证据链改善。
- 只有后续正确值、可核对来源及成对干预共同解释排序提升时，才支持“多模态补缺失属性驱动join discovery”；Stage1当前不预支这一结论。

### 20.3 会实质改变执行设计的待确认项

1. **主部署排序**：本稿默认沿用旧R13的D1+F1，把等权RRF作为固定敏感性读数。若研究目标要求evidence score直接参与主榜单，应在审阅时统一改为RRF，而不是按结果切换。
2. **可恢复的训练身份**：需要S0、PCA参考、完整R12 hard batches及Teacher scores，以及path物化记录／历史配置。缺历史精确版本时按本稿做同版本匹配新控制；不宣称旧结果逐位配对。
3. **见证记录作用域**：确认train-fit recoveries可稳定映射到`(q,t,a,row,e)`。映射缺失只降低W的属性一致性解释；不需要先补全部人工负标签，也不阻塞C。

其余按本稿默认：full R、D4096/d1024、batch64、统一新AdamW状态、首批178步、固定F1主排名与N50、原Teacher固定、首批只拆table角色。

## 21. 来源与本轮文献核验

检索日期：2026-09-09。只列与本轮推导直接对应的原文／官方页面；这些研究提供机制参照，不是本任务上的效果证据。

### 附件来源

- **[L1]** `08_round2_context_update.md`：三类型P、已有有向R与Teacher角色、Stage2排序／保留规则、未处理候选的接口边界。
- **[L2]** `stage1_optimization_r13_plan_20260909.md`：旧R13全文，尤其3.1/3.3、5.1、Task A–F、12节；本修订稿替代其任务顺序和指标。
- **[L3]** `02_analysis.md`与`04_results_and_audits.md`：C1/edge/path退化、Teacher能力、F1/F3/F5及历史审计边界；引用已有复算，不声称本轮重跑。
- **[L4]** `05_metrics_and_protocols.json`：`.../taskC_training/c_candidates_seed13/manifest.json`、`.../candidate_quality/summary.json`、`.../statistics/summary.json`等内嵌来源；保留文件内路径身份。
- **[L5]** `06_stage1_implementation.md`：models/objectives/scoring/training/retrieval/row_support、R12 runner等静态源码；不自动等同历史checkpoint代码。
- **[L6]** `01_proposal.md`、`03_experiment_plans.md`、`07_stage2_implementation.md`：原始目标、旧预算与后续Stage2机制设计，冲突处以本轮更新为准。

### 文献

- **[R1]** Karpukhin et al. **Dense Passage Retrieval for Open-Domain Question Answering**，EMNLP 2020。正文3.1为独立query/passage encoder与可分解内积、离线passage索引。迁移：角色不对称表示可保持索引；局限：没有证明本项目必须分离低维P。<https://aclanthology.org/2020.emnlp-main.550/>
- **[R2]** Xiong et al. **Answering Complex Open-Domain Questions with Multi-Hop Dense Retrieval**，ICLR 2021；核对arXiv v2（2021-02-19），正文2.2。`q_t=g(q,p_1,...), p=h(p)`支持条件query和静态语料向量；其主体采用共享RoBERTa。迁移：H的索引结构；局限：原文重编码拼接文本，本方案只用冻结pooled向量的轻量组合，信息量和任务不同。<https://arxiv.org/html/2009.12756v2>
- **[R3]** Lopez-Paz et al. **Unifying Distillation and Privileged Information**，ICLR 2016；核对arXiv v3与作者页面。训练期额外解释信息、测试只用普通输入。迁移：训练见证可作目标／软监督；局限：不能把Student不可见的样本特定信息直接当可部署可预测信息。<https://arxiv.org/html/1511.03643v3>；<https://leon.bottou.org/papers/lopez-paz-2016>
- **[R4]** Yu et al. **Gradient Surgery for Multi-Task Learning**，NeurIPS 2020，正文2.2–2.3。迁移：检查梯度冲突、主导与实际更新损伤；局限：梯度夹角不构成本项目因果证明，本轮不自动采用PCGrad。<https://proceedings.neurips.cc/paper/2020/hash/3fe78a8acf5fda99de95303940a2420c-Abstract.html>
- **[R5]** Ganchev et al. **Posterior Regularization for Structured Latent Variable Models**，JMLR 11，2010，2001–2049。迁移：用训练结构约束潜解释，而非要求部署输入标签；局限：本稿W是具体对比辅助目标，不是该框架逐式复现。<https://jmlr.org/papers/v11/ganchev10a.html>
- **[R6]** Argyriou, Evgeniou and Pontil. **Multi-Task Feature Learning**，NIPS 2006。共享低维表示是相关任务的结构假设。迁移：将共享子空间当需检验的归纳偏置；局限：角色P/R的行空间推导是本稿线性代数分析，不是该文针对多模态join的定理。<https://papers.nips.cc/paper_files/paper/2006/hash/0afa92fc0f8a9cf051bf2961b06ac56b-Abstract.html>

**本稿到此结束。所有新增实验均未执行；下一步是审阅和冻结，而不是自动启动。**
