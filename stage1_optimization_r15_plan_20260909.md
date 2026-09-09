# MMDD Stage 1 R15：残差失败归因与 evidence-only discovery 审计

**版本：2026-09-09；依据：本次上传的 R13 与 R14 完整结果产物。**

本文是下一轮执行方案，不代表本文所列新实验已经执行。冻结 backbone、P/R 可训练、单对象 1024 维 ANN、0/1 跳、无在线 Teacher 的边界保持不变。

## 1. 本轮决策与优先级

R15 不再继续泛化地问“MLP 会不会更强”。已有 R14 表明：原线性 full/Eoff 的主指标接近，而加入线性或 GELU 残差的两个 full 配方均发生严重自然检索退化。下一步先区分 **实现/ANN问题、完整投影的优化问题、E-loss 与残差的交互**。同时，沿用冻结 B13，查清已有 evidence-only 候选为何未进入交付列表，保留论文优先证明的“路径新增发现 → 正确 join”主线。

顺序如下：

1. **G：零新增训练的数值与 exact 审计。** 不通过时，只修正可复现的实现问题，不解释容量结论。
2. **I：通过 G 后，补齐两个 Eoff 残差臂。** 仅新增 `L-Eoff`、`N-Eoff`，共 356 个 Student updates，检验 E-loss × 残差交互。
3. **C：冻结 B13 的候选来源与准入审计。** 不训练、不搜索融合权重，使用现成 F1、pure-direct、固定 equal-RRF。
4. **V：有单独验证预算时开展 Stage 2。** 先检查已导出的 evidence-only 样本，再做固定 N=50、K=10 的全 query 比较。V 不阻塞 G/I/C，也不能以小样本机制检查替代主端点。

不得自动进行 h/激活/学习率扫描、延长训练、role/W/H 扩展、Teacher 重训、hard refresh、解冻 backbone、column/row 全湖索引或多跳。

## 2. 已有事实及其边界

### 2.1 可重算的主结果

所有 Recall 数值为百分数；差值为百分点。主指标为 1,198 个 dev query 的宏平均 target Recall@10。

| Seed | S-full R@10 | S-Eoff R@10 | Eoff−full | 改善/下降/不变 query |
|---|---:|---:|---:|---:|
| 13 | 29.0067 | 29.0902 | +0.0835 | 3 / 2 / 1193 |
| 17 | 28.7980 | 29.0484 | +0.2504 | 7 / 3 / 1188 |
| 23 | 28.8815 | 29.0067 | +0.1252 | 5 / 3 / 1190 |

三 seed 平均 Eoff−full 为 +0.1530pp。先按 query 平均 seed 差值、再按 1,000 个 source group bootstrap 的探索性区间约为 [-0.2213,+0.5578]pp。此区间条件于固定 S0、Teacher、训练候选和这三个排列 seed；不是跨数据湖或完整训练流程的方差估计，也不是等效性证明。

| Seed13 配方 | R@10 | R@20 | 交付 Recall@50 |
|---|---:|---:|---:|
| S-full / B13 | 29.0067 | 35.0793 | 44.9082 |
| S-Eoff | 29.0902 | 35.0376 | 45.2282 |
| L-full | 0.0000 | 0.0000 | 0.5426 |
| N-full | 0.1669 | 0.3339 | 2.3790 |

不能将 N-full 比 L-full 高 0.1669pp 解读为非线性有效：二者相对 S-full 都是失败配方。

### 2.2 图像侧有比主 Recall 更明显的 E-loss 干预结果

678 个有 known witness 的 implicit 正对上：

| Arm | known image QE 正对 | known image QET 正对 | known text QET 正对 | 全部 known QET 正对 | distinct images | 最高频图像 query 数 |
|---|---:|---:|---:|---:|---:|---:|
| S0 | 110 | 61 | 214 | 255 | 9417 | 146 |
| S-full13 | 81 | 34 | 223 | 246 | 5165 | 639 |
| S-Eoff13 | 109 | 60 | 216 | 257 | 9405 | 144 |
| S-full17 | 87 | 37 | 222 | 248 | 5857 | 425 |
| S-Eoff17 | 109 | 60 | 216 | 257 | 9225 | 132 |
| S-full23 | 84 | 37 | 224 | 249 | 5554 | 489 |
| S-Eoff23 | 109 | 60 | 216 | 257 | 9387 | 147 |

text/image 正对可以重叠；这里不是路径出现次数。Seed13 的图像 QET 是 Eoff 恢复 29 对、丢失 3 对，净增 26 对。可支持“本次 E-loss continuation 包影响了图像邻域和路径行为”，但不能进一步断言是 image 子损失、KD、某个参数组或错误标签单独造成。Eoff 同时移除了 E-channel CE/KD，且共享 P_table 可传播间接影响。

### 2.3 发现空间没有消失，但当前 F1 会压掉这部分机会

B13 原始 ANN 候选池：

- 自身 direct100 的 target 宏 Recall：52.9563%。
- 完整 `direct100 ∪ evidence-targets` 的宏 Recall：69.6717%。
- 并集平均 289.374 个不同 target；这不是 N=50 的公平主结果。
- 213 个已知正 `(q,t)` 在自身 ANN-direct100 之外而在 evidence 候选中：179 implicit，34 explicit。
- 其中 82 个 implicit 正对有 known-witness QET，覆盖 79 个 query、75 个 source group；75 个有文本正确路径，11 个有图像正确路径，4 个兼有。
- 82 对的已知支持行数：1/2/3/4/5 行分别为 45/23/10/3/1 对。
- 这 82 对中，F1@10 收到 0 对，F1@50 收到 2 对；固定 equal-RRF@10 收到 4 对，@50 收到 45 对。

这些是 qrels/known-witness 意义下的候选与支持，不是新完成的属性恢复或正确 join。213 对目前按 ANN-direct100 定义；全部相应 exact rank 尚未导出，不得直接把 213/82 改写成 exact-direct100 之外的发现数。

## 3. 指标与因果估计对象

定义固定模型结构 f、训练 E 通道开关 e、部署规则 r 下：

$$Y(f,e;r)=\frac{1}{|\mathcal Q|}\sum_q\frac{|G_q\cap\operatorname{Top10}_{f,e,r}(q)|}{|G_q|}.$$

固定分母 G_q，不能改为 pair micro，也不能仅在获益 query、implicit 或发现子集上重新加权主结果。没有已知相关性的 target 不自动视作真实负例。

R14 已识别的两个条件效应是：

$$\tau_E(S)=Y(S,1;F1)-Y(S,0;F1),$$

$$\tau_{N-L\mid E=1}=Y(N,1;F1)-Y(L,1;F1).$$

前者是“已有 R12 evidence 训练基础上的短程 continuation E-loss 贡献”，不是“整个系统是否需要 evidence”。后者是“当前参数化、优化、正则与尺度处理下的激活配方效应”，不是理论表达能力的单独测量。

补齐 R15 两臂后计算：

$$I_L=[Y(L,1)-Y(L,0)]-[Y(S,1)-Y(S,0)],$$

$$I_N=[Y(N,1)-Y(N,0)]-[Y(S,1)-Y(S,0)],$$

$$I_{N-L}=[Y(N,1)-Y(N,0)]-[Y(L,1)-Y(L,0)].$$

这些仍然是在当前 regularization 与固定训练配置下的交互。不同因素不互斥：E 目标不适配、优化不稳定与表达能力不足可能同时存在。

## 4. G：先完成部署数值与 exact 闸门

### G1. 数值路径一致性

使用实际训练后的 L/N checkpoints，不只使用 step0 或随机小模型。对固定 train-fit 与 dev 面板，每个关系分别比较：

1. 训练/评分入口给出的 raw score；
2. 导出的 source query vector 与 destination vector 内积；
3. 对应缓存向量的 brute-force 内积；
4. ANN 返回 ID 上重算的 exact score；
5. checkpoint save/load 前后的上述结果。

五类有效关系必须全部覆盖：Q→T、Q→text、Q→image、text→T、image→T。包括 source/destination 两端，不能仅检查 table→table。

导出参数组实际内容：P/A/B/R 各自 lr、weight_decay、optimizer state、dtype、normalization、gradient clipping 的位置和范围。不能用一个顶层 `projection_lr` 字段代替新增 A/B 的实际分组证据。检查 c_tau 是否按原值保存/加载，是否发生 backbone/object embeddings 的二次归一化或残差遗漏，索引是否来自对应 checkpoint。

误差界依据 dtype 预先固定，不从 dev Recall 选择。FP32 相对/绝对误差使用明确的容差；混合精度另列容差，并报告并列分数/数值边界。

### G2. L-full 的合并恒等式测试

R14 实际结构为：

$$F_L(z)=Pz+B(Az/c_\tau)=\widetilde Pz,\qquad\widetilde P=P+BA/c_\tau.$$

直接从已训练的 L checkpoint 构造合并版。R、对象内容、任何对象局部归一化、dtype 和合法候选集合保持一致。比较完整模型、合并模型的投影向量、raw score、exact 排名与 Recall。ANN 若重新建索引，要区分数值等价与 HNSW 重新构建的随机差异。

这是零更新的强一致性测试，不是新模型或新训练配方。若不一致，先修实现；若一致，则证明失败不是“线性残差扩大了函数类以后容量太大”这一解释。

### G3. exact 与 ANN 分离

对 L/N 的 step0/45/89/178（本地存在则复用，不因缺少中间结果重训）评估：

- 全 dev exact Q→T Top100 与 natural ANN Top100；
- 固定 source/evidence 面板的四类 QE/ET exact TopK 与 ANN TopK；
- 精确正例 rank、positive margin、返回集合重合率；
- direct/image/text 的 hub 次数、distinct 数与向量几何。

现有结果包的 S0/B13 exact 审计不能替代这一检查。如果 exact 正常而 ANN 崩塌，归因先转到索引/几何与检索近似；若 exact 也崩塌，才继续考察训练的完整打分函数。

### G4. 残差与分数诊断

必须同时记录基础 P 输出、残差、完整 F 输出，不只看 P 权重漂移：

$$r_\tau=\operatorname{RMS}(F_\tau(z)-P_\tau z)/\operatorname{RMS}(P_\tau z).$$

报告 Az、Az/c、GELU(Az/c) 的分位数与 RMS，P/A/B 参数实际变化，完整 F 与 S0 的方向变化、均值向量、随机对象余弦分布、有效秩/谱以及五关系正例 margin。A 的权重范数变化不等于其参数更新范数。

进行一次已训练模型的 adapter-off 反事实：使用同一训练终点的 P/R，只置残差为零，并重新构建索引。这只是机制探针；由于 P/R 已与残差共同适配，不能把它当作独立训练的 S-full，也不能据此单独证明根因。

逐式导出训练 E 分数、edge 变换、LSE 温度、正例 mask、路径计数与聚合方式；对齐推理使用的 path score。计算 known-witness responsibility 时必须使用真实温度和保留/截断规则，不能临时假定 tau=1。

### G 的解释矩阵

| 观察 | 允许结论 | 下一步 |
|---|---|---|
| 已训练 forward 与缓存/IP 不一致 | 存在实现或部署分布错误 | 修正后原配置重评；必要时原配置重跑，不做容量结论 |
| forward 正常、exact 正常、ANN 严重退化 | 主要失效发生在检索近似/索引环节 | 做独立索引诊断，不宣称 MLP 无效 |
| exact 与 ANN 都差，adapter-off 明显恢复 | 残差参与的完整分数改变是重要中介 | 执行 I；根因仍需区分目标、尺度、正则 |
| exact 与 ANN 都差，adapter-off 仍差 | 还存在 P/R 共适配或基础分支漂移 | 执行 I，并保留分支共适配解释 |

G 不通过时不自动启动 I。验证通过不代表训练方法有效，只表示下一步效应具有可解释性。

## 5. I：仅补齐两个缺失单元

### 5.1 训练矩阵

| Arm | 投影 | E-channel CE/KD | 参数与推理 | 新 updates |
|---|---|---|---|---:|
| S-full13 | 原 P | 开 | 复用 B13 | 0 |
| S-Eoff13 | 原 P | 关 | 复用 R14 | 0 |
| L-full13 | Pz+B(Az/c) | 开 | 复用经 G 验证的 R14 | 0 |
| N-full13 | Pz+B GELU(Az/c) | 开 | 复用经 G 验证的 R14 | 0 |
| **L-Eoff13** | **Pz+B(Az/c)** | **关** | **P/A/B/R 可训练，完整 evidence 推理** | **178** |
| **N-Eoff13** | **Pz+B GELU(Az/c)** | **关** | **P/A/B/R 可训练，完整 evidence 推理** | **178** |

总新增正式训练：356 updates，0 次新 Teacher 推理。S0、Teacher、候选文件、11,390 query 的 batch 顺序、positive mask、178 updates、fresh AdamW、P/R/A/B 参数组与 R14 同 seed 臂完全匹配。

A/B 初始化和 c_tau 必须直接复用相应 L/N 的 step0 参数/缓存；不依赖“设相同 seed 大概一样”。B=0，A 非零；基础 P/R 复制同一 S0。

Eoff 只移除 E-channel CE/KD 的目标梯度。必须仍执行相同 evidence forward，保留 anchor 和原 optimizer 行为，完整构建 text/image/target 索引，仍执行 43 个 query-vectors/query 的 natural 检索。

本块不改变 R14 的 base-only anchor、不补做激活后尺度匹配、不修改 LR 或加入新的功能约束。这些都是潜在解释，但此次单因素比较必须先保持不变。

### 5.2 主指标与诊断

主 endpoint 仍为全 dev F1 target Recall@10，同时报告 @20、交付 @50、pure-direct、固定 equal-RRF、implicit/explicit。不要用已知路径数量替代主端点。

两个交互比较之外，必须报告 `L-Eoff−S-Eoff`、`N-Eoff−L-Eoff`、`N-Eoff−S-Eoff`。每对比较输出逐 query 赢/输/平、paired source-group bootstrap、正 target 进入/退出与候选来源；不要只输出汇总差。

所有训练端点同时输出 modality QE/ET exact/ANN、路径出现次数和独立正对数、known top path/责任质量、支持行数 0–5、全湖候选池与交付池、image/text/direct hub，以及 G4 的投影/score 诊断。

### 5.3 结果应如何解释

| 结果 | 支持的解释 | 不支持的过度解释 |
|---|---|---|
| L/N-Eoff 恢复健康，full 仍崩塌 | 当前 E-loss 与残差参数化存在强负交互 | image 本身没用；非线性本身无效；错误只在某个 image loss |
| L/N-Eoff 也崩塌 | E-loss 不是失败的必要条件 | 已排除训练目标/正则问题；已证明 frozen embedding 信息不够 |
| L-Eoff 与 N-Eoff 相近且优于 S-Eoff | 当前重参数化/优化配方有效，未建立额外非线性收益 | 线性 residual 增大了理论函数容量 |
| N-Eoff 稳定优于 L-Eoff、S-Eoff | 该条件下存在非线性配方价值 | evidence training 已被改进；多模态互补已建立 |
| N-full 好、N-Eoff 差，而 S-full≈S-Eoff | E-loss 的收益依赖投影结构，有正交互 | 该四/六臂结果已经定位到了唯一机制 |
| 训练 list 变好，exact/full-lake 变差 | 局部训练代理与部署检索不一致 | 单凭 train loss 已证实过拟合或错误负例根因 |

N−L 的小正差只有在两者不再坍塌且 N 超过 S/B13 时才具有方法投入意义。没有新完整候选时继续保留 B13 作为冻结参考，但不得仅为了论文故事把 Eoff 永久排除为科学对照。

对新配方追加 seed17/23 必须满足预声明的自然 R@10 收益门槛（参考 +0.5pp）、分层与机制/成本护栏；不因一个 failed model 比另一个 failed model 好一点而追加。若未来要声称“激活本身”的干净收益，另立一轮匹配完整投影约束、激活后固定尺度与有效步长；不要在本轮临时修补后继续与旧臂作单因素比较。

## 6. C：冻结模型，定位 evidence-only 从召回到交付的损失

### C1. 把四层集合明确分开

对每个模型 m、query q，保存：

- `D_m,100_ANN`：该模型自然 direct 检索的 100 个 target。
- `D_m,100_exact`：相同 score 的 exact Top100。
- `E_m`：40 个 evidence 各自 Top20 ET 的 target 并集。
- `U_m=D_m,100_ANN ∪ E_m`：未截断并集。
- `C_m,50^r`：某个预声明准入/排序规则 r 交给 Stage 2 的 50 个 target。
- `Top10_m^r`：Stage 1 截断，或明确标明经过 Stage 2 的最终 Top10。

“CandidateRecall@50”只指 C_50；另写 `RawUnionRecall`，不能把约 289 个候选的 69.67% 混称为 CR@50。

定义：

$$X_{m,q}=G_q\cap(E_{m,q}\setminus D^{exact}_{m,q,100}).$$

同时报告相对自身 direct、相对冻结 B13 direct 的新发现数；不同模型不能只用被自己搞坏的 direct 作为分母。这尤其重要：N-full 的自身 direct 崩塌后，仍可出现看起来不小的 E-only Recall，不能据此宣布发现能力强。

### C2. 82 对优先审计队列

使用随本报告提供的 `B13_evidence_only_known_witness_cases.csv`，不按最终成功与否重新挑选。

每对记录：query/source/target ID、natural 与 exact direct rank、模态、合法 witness ID、支持行数、现有 evidence 聚合 rank、F1/RRF 的 final rank、保留给 Stage 2 的具体 evidence ID、是否被已知 witness 支持、是否需要独立内容审核。

需要特别检查：虽然 raw pool 中存在正确 witness，但 path retention 可能丢掉它。正确路径可达不等于 Stage 2 实际收到正确 evidence。

所有 213 对都要补 exact direct rank；82 对是机制验证优先队列，不是主指标的新评测集合。未知其他 witness 和连接属性仍可合法，不能把没有已知标注的候选直接标错。

### C3. 不搜索融合权重，先复用已有入口

固定 B13。比较现成的：

- pure-direct Top50；
- 当前 F1 Top50（简单并集后按 direct score 排序的已有控制）；
- 已在 R13/R14 固定的 equal-RRF Top50。

不事后把 RRF@10 改成主指标；F1 的既有主结果保持可追溯。这里的问题是“交付 N=50 时哪批正确新发现得到保留”，而最终 K=10 的效果留给同一个 Stage 2 重排测量。

随机控制必须分清目标：

1. **池内随机**：固定 RRF 所保留的 direct100 内成员和外部槽位数 m_q，从 `E\D100` 均匀无放回采样 m_q 个替代外部成员。检验 evidence 候选内的排序价值。
2. **全湖随机扩池**：同样保留上述 direct 成员和 m_q 槽位，从合法湖内 `T\D100` 均匀采样。检验路径生成候选的价值。

m_q 仅由检索结果决定，不读取 GT 类型或相关性。随机种子预先固定，不取最好 seed。若做多个随机重复，报告平均及 Monte Carlo 变化，并单列其成本；不能称为 Student 重复。

候选质量在全 1,198 query 上按原定义报告。所有进入 Stage 2 的比较都固定 N=50，最终 K=10；检索 query-vector 数、返回对象数与延迟另外报告。相同 N 不等于相同检索成本，未做隔离延迟测试时不声称等计算预算优越。

### C4. 为什么这不等于放弃 evidence-only 主线

固定 direct score d，direct TopB 为 exact，且 K≤B 时：

$$\operatorname{TopK}_{d}(D^{exact}_B\cup E)=\operatorname{TopK}_{d}(T),$$

在统一合法目标集、统一 tie-break 的条件下成立。因为 exact TopK 已包含在 direct TopB 中，外部 target 不可能靠同一个 direct score 挤进 TopK。

这限制的是当前的准入/排序规则，而不是 0/1 跳、单对象 ANN 或 evidence 的信息价值。学习非线性 P 可以改变 d，但若依旧对同一模型采用上述规则，新增 evidence-only target 仍会被 direct TopK 主导。

## 7. V：最终机制验证（与 Stage 1 训练解耦）

当前 Stage 2 是重排序，不是 `joinable=False` 就删除候选；保持既有 `(coverage 降序, mean similarity 降序, Stage1 rank 升序)` 规则。不得重新引入“至少 3/5 行”硬门槛。

候选级机制链逐项验证：

1. target 位于同模型 exact direct100 之外；
2. 通过自然 Q→E→T 进入原始候选池；
3. 在固定 N=50 中被实际交付；
4. 被实际交付的 evidence 可支持 query row 的合法连接属性；
5. 恢复出的属性值正确，且能在 target 上形成正确连接；
6. 进入最终 Top10，或明确记录在哪一级失败。

已知 target qrel、known witness、coverage、`joinable` 布尔值都不能单独代替第 5 项。需有独立的值/实体/属性核验；多条合法连接属性允许并存。未核验结果标 unknown，不当作错误。

在固定候选集合上做相同模型的无 evidence 与经独立审核的错误 evidence 反事实，区分“路径找到了 target”和“evidence 值恢复帮助连接/排序”。图像结果需另核验目标信息是否已在可见文本中；否则不宣称图像提供了独立文本缺失信息。

79 个 query 的初步检查只能形成机制案例和失败分类。要宣布总体 Recall 提升，仍需全 query 或另行冻结、未参与调参的确认集。Stage 2 暂无预算时如实报告 V 未执行，不阻塞 G/I/C。

## 8. ANN、参数和成本边界

所有投影均为单对象函数：

$$u_x=F_{\tau(x)}(z_x),\quad q_{a\to b}=R_{\tau(a),\tau(b)}^\top F_{\tau(a)}(z_a),\quad v_b=F_{\tau(b)}(z_b).$$

故 raw score 为 `qᵀv`，destination 向量不依赖当前 query。P/R/A/B 可训练；backbone 冻结；Teacher 不在线。

- 输出/ANN 维度 1024；对象数和索引数量不增加。
- h=256 时三种类型共新增 `3×(4096×256+256×1024)=3,932,160` 参数。
- 未合并 residual 每对象增加 1,310,720 个矩阵乘加，相对原 P 的 4,194,304 个为 31.25%；这不是端到端延迟比例。
- L 可将 BA/c 合并到 P，部署投影矩阵乘法成本回到原结构；N 一般不能这样合并。
- destination 与静态 evidence 的投影可离线缓存；在线仍为 43 个 query-vectors/query。索引必须对应实际模型重新构建。
- 原始双线性关系块的 score matrix rank 仍 ≤1024；非线性只改变可从 z 提取的对象特征，不恢复 z 中不存在的信息，也不解决同一 E 的 ET 排序不受 Q 条件影响的问题。

G/I 不改变方法身份；C 只改变候选入口的比较口径/既有排序规则，必须单独命名；V 是完整机制验证。

## 9. 交付清单与可复现性

每个正式训练臂保存已冻结配置、真实 optimizer 参数组、父 checkpoint/初始化/Teacher/候选/顺序 hash，以及每次更新的分项 loss。输出 step0/45/89/178 的模型、对应索引 manifest 与几何诊断；不得只保留最好 step。

必须交付：

- `RESULTS.md`：事实、假说、竞争解释、干预、主 Recall、机制诊断、失败含义分别写明。
- `VALIDATION.json`、`COMPLETION_AUDIT.json`：区分代码正确、指标复算、exact 通过、统计支持、完整机制已验证五个层次。
- `per_query_metrics.jsonl.gz`：固定 query/source/G_q、三种已有部署规则与 K=10/20/50。
- `candidate_provenance.jsonl.gz`：D100 ANN/exact、E/U/C50、每个正 target 的自然/exact rank 与进入/退出。
- `witness_funnel.jsonl.gz`：按 query-target 去重和按路径次数的两套统计；text/image 允许重叠。
- `exact_relation_panels.json`：固定 source 对象集合、五关系 exact/ANN、分位数/正例 margin/hub。
- `projection_and_optimizer_diagnostics.jsonl.gz`：完整 F/base/residual、A/B 实际更新、实际参数组、梯度记录位置、训练真实聚合/温度与 responsibility。
- `source_snapshot/`：模型投影、训练 loss、optimizer 分组、索引构造、推理聚合的关键源码及 hash。只提供最终源码 hash 不足以独立审计训练后的数值路径。
- `evidence_only_verification.jsonl.gz`：82 对及所有新增对的逐级状态；Stage 2 未执行时明确 null，不伪造成功/失败。

本轮不得用 `pytest passed` 代替方法有效性，也不得在缺少 trained-model exact 的情况下把 ANN 崩塌直接称为 exact 表示崩塌。

## 10. 停止规则

G 发现具体实现错误：停止容量归因，修正原配置并重评，单独标记结果版本。

G 通过且 I 两臂同样失败：保存失败结果，停止投影扩容网格；依据真实几何/梯度再选择一个后续约束，不同时调整多个因素。

G 通过且 I 明确显示 E-loss × residual 负交互：下一轮才研究 target-only E 代理、跨模态责任分配、完整投影约束等单一干预，不回到“继续扩大 MLP”默认路线。

C 中 exact-direct100 外没有可靠路径：第一叙事缺少当前模型证据，保留结论边界；不能改用 ANN 漏召回来伪装扩大语义发现空间。

C 中存在路径但未交付：继续候选准入/保留的独立实验；不因 F1 对路径不敏感就否定 evidence。

V 中属性值或连接验证失败：这就是机制失败，不能用 Stage 1 target Recall 掩盖。

**最终原则：先证明测量通道能看到目标贡献，再区分训练目标效应与投影配方效应；不把崩塌配方的相对差值包装成表达能力证据。**
