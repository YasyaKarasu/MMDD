# R14 实验计划：分离 Recall 收益、证据分支训练贡献与部署排序限制

日期：2026-09-09。状态：**待审阅、未执行**。

本文件根据 R13 的四份产出报告提出下一轮实验。文档编写只做了附件阅读、已公布数值的算术复核与方法推导，没有运行 MMDD 模型、训练、ANN 检索、原仓库测试或标注服务。先审阅，再由 Codex 执行。以下新增任务、接口、目录和阈值都是拟议规格，不代表已经存在的 CLI 或已完成的实验。

## 0. 执行摘要

**R13 得到的是一个更好的 Stage 1 总体 Recall 开发配方，而不是角色拆分、见证监督或 Teacher 蒸馏的新正证据。** 当前保留 `p_s_target_only@178` 作为最新部署参考 B13；保留其父模型 S0，用于识别新增训练的真实增益。

下一轮不重复角色拆分／W 权重网格，不默认增加 latent 或 query-conditioned 第二跳。优先回答三个问题：

1. P-S 的 Recall 收益来自 direct 打分变化、自然候选集合变化，还是 evidence 分支的训练贡献？
2. F1 union-direct 是否使 evidence 训练的改进难以影响最终 Recall？两跳究竟带来了目标相关性增益，还是主要补偿 direct ANN 漏检？
3. 若实际更新显示 direct 分支损伤了共享投影上的证据检索，或旧 hard candidates 已不代表当前困难候选，哪一个单因素干预能改善完整自然检索 Recall？

核心顺序：**读取已有详细产物并做分解 → 一个训练分支消融 → 至多一个有条件优化 → 一组条件 Student 重复实验。** 不把生成全空或大规模新人工审核设为 Stage 1 启动门槛。

本轮继续保持冻结骨干、离线 Teacher、可训练 P/R、表级单向量 ANN、Q→T / Q→E→T、在线不指定具体 join column。所有正式主读数仍为全体 query 的 target Recall@10，@20/@50、implicit/explicit、证据诊断与成本同时报告。

---

## 1. 证据范围、身份与 R13 的可保留结论

### 1.1 实际材料

本轮实际阅读了四份产出报告全文，并核对上一轮计划中的任务定义、指标、目标、预算和选择条款：

- [S1] `RESULTS.md`：R13 固定端点、所选配方、部分诊断与置信区间。
- [S2] `COMPLETION_AUDIT.md`：任务完成声明、未触发分支与允许降级。
- [S3] `SELECTED_RECIPE.json`：冻结选择、checkpoint 路径和 SHA256。
- [S4] `VALIDATION.json`：自动验证通过声明与测试数量。
- [S5] `stage1_optimization_r13_plan_20260909_revised.md`：上一轮实际计划。

[S5] 本地文件 SHA256 为：

`32e268a141c7084b88090f205f263792916fc8344a97e87b0cd75c16ba7007d0`

与 [S2] 引用一致。这验证的是计划文件身份，不等于独立验证了运行、候选、统计或模型代码。

四份新附件没有给出完整逐 query 排名、详细 bootstrap、全部 B0 梯度表、完整中间 checkpoint 指标、pure-direct/RRF/rescue-displace 数值或全部成本表。报告说这些产物已存在，下一轮应优先读取，而不是重新生产一遍。本轮不能以绝对路径存在于报告中为由声称已访问原服务器。

当前 R13 新实现源码未随四份产出上传；旧 06/08 仅提供模块职责背景，不自动代表 R13 的精确实现。

### 1.2 已公布的固定端点

以下均为百分数；差值另用百分点。来源为 [S1]，不是本轮从逐 query 重算的结果。

| 配方 | 新阶段步数 | 总 R@10 | implicit R@10 | explicit R@10 | R@20 | CR@50 |
|---|---:|---:|---:|---:|---:|---:|
| S0 | 0 | 27.5668 | 17.5710 | 37.5626 | 33.3055 | 43.0509 |
| C-S：共享投影 edge 续训 | 178 | 28.4432 | 17.9883 | 38.8982 | 34.2863 | 44.8456 |
| C-R：拆分 table 角色 | 178 | 28.4432 | 18.3222 | 38.5643 | 34.4741 | 45.1377 |
| P-S：原 target-level path | 178 | 29.0067 | 17.4457 | 40.5676 | 35.0793 | 44.9082 |
| P-W：path + 0.1 W | 178 | 28.9232 | 17.4457 | 40.4007 | 35.0793 | 44.9082 |
| hard KD-on，从 PCA 起点 | 356 | 27.5668 | 17.5710 | 37.5626 | 33.3055 | 43.0509 |
| hard KD-off，从 PCA 起点 | 356 | 27.7755 | 17.4875 | 38.0634 | 33.3681 | 43.2596 |

主要已报告差值与探索性 95% source-cluster bootstrap 区间：

| 比较 | R@10 差值，百分点 | 95% 区间，百分点 | 本轮解释 |
|---|---:|---:|---|
| P-S − S0 | +1.4399 | [+0.0633, +2.7849] | 固定条件下的开发增益，尚无多 seed 或独立确认 |
| C-R − C-S | 0.0000 | [−0.5843, +0.5809] | 没有观察到总体优势，不是共享／拆分等效性证明 |
| P-W − P-S | −0.0835 | [−0.2542, 0.0000] | 本版 W 未改善主端点；上界显示为零也不作确认性显著结论 |
| hard KD-on − hard KD-off | −0.2087 | [−0.6399, +0.1702] | 没有建立 KD 的正增益，不能据此宣称所有 Teacher 无用 |

从六位小数表值复算的差值可能与报告使用未舍入值计算的差值相差 0.0001 个百分点。选模与统计必须使用原始值。

### 1.3 四个影响下一轮设计的事实

**第一，收益分层不均。** P-S 对 S0 的 explicit 增加 3.0050pp，implicit 下降约 0.1253pp。P-S 对同样追加178步的 C-S，总体多0.5635pp，但 implicit 少0.5426pp，explicit 多1.6694pp。后者没有随附件提供配对区间，且 edge/path 每步计算量不同，不能直接称显著或等算力优势。

**第二，已知支持没有同步提高。** 已知见证自然池覆盖：S0 37.61%，P-S 36.28%，P-W 36.73%。W 比其控制恢复约0.45pp，但仍低于S0约0.88pp，也没有报告出平均已知见证 LSE 责任的改善。[S1]

**第三，负结果对应的是具体实现。** W 使用 `any_known_witness_by_row`，没有稳定属性字段，独立错误属性／错误实体标签覆盖为0。故不能把它称为同属性见证学习被否定。角色拆分有非零新子空间更新，但总体Recall没有优势；这也不证明子空间假说在任何维度或训练路径下都不成立。[S1][S2]

**第四，历史对照、成本和验证均有边界。** 16个固定ET正对中的S0为5/16，C-R为6/16；附件未给出匹配C-S在此面板的值。R12 step1318缺失，历史edge使用R11 epoch2，且诊断用fresh AdamW与补齐的非打分anchor，不是重放历史轨迹。单遍C-R/C-S的P95为90.45/57.01ms，不足以断言角色拆分内在增加58.7%延迟。1863项pytest通过属于报告的自动验证结果，不是独立科学复现。[S1][S2][S4]

### 1.4 本轮禁止的推论

不得将“总Recall提高”直接写成“evidence分支更好”；implicit/explicit桶与实际推理分支不是同一个变量。不得用已知见证集合的补集构造错误属性率。不得将微小、单seed的KD-off优势转成无条件删除Teacher的主线变更。不得将未触发H写成H已被实验否定。

---

## 2. 必须新增的理论边界：F1 与 evidence 召回贡献

### 2.1 F1 是扩池后 direct 排序，不是 evidence 分数融合

记某个模型的 direct 分数为 d(q,t)，合法目标全集为 T，direct100候选为D(q)，两跳目标集合为E_T(q)，自然并集为：

$$
U(q)=D(q)\cup E_T(q).
$$

F1 按 d 在 U 上排序，取前N。固定 U 与 d 后，改变路径分数、source bias、见证责任或保留证据，不会直接改变F1目标榜单。若某实现改变了 U，那是候选集合干预，应另行记账。

### 2.2 精确 direct100 下的等价性质

若 D(q) 已是同一合法目标全集、同一 d、同一破同分规则的精确 top100：

$$
D^*_{100}(q)=\operatorname{Top100}_{t\in T}d(q,t),
$$

则对所有 K≤100：

$$
\operatorname{TopK}_{t\in D^*_{100}(q)\cup E_T(q)}d(q,t)
=
\operatorname{TopK}_{t\in T}d(q,t).
$$

证明：全局前K已全部在D*100中，而其余目标不可能按同一排序超过它们。这个性质不需要任何qrels或属性标签。

实际D100来自ANN，所以F1可能通过两跳找回direct ANN漏掉、但本来就有较高d的目标。这是有价值的搜索补充；却不同于“path分数把低direct分数、真实可连接的目标提升到前面”。

**重要限制：** 这个性质不否定 evidence 的训练作用，也不否定其为 Stage 2 提供补值材料。E分支的训练仍可能通过共享P改变d。它只界定固定模型、固定direct排序下在线扩池的影响。

### 2.3 对R14的实际约束

主F1保留以保证R13可比，等权union-RRF60继续作为固定敏感性读数；不在二者中逐臂择优。不为展示evidence而直接换成已知较弱的RRF主基线。

但若实测F1确实几乎等价于精确direct前缀，应明确报告它对证据分数学习的识别力有限。之后要研究evidence分数的准入收益，应作为一个单独的部署规则实验预先声明；不能无期限要求所有见证表示先在F1上获胜，才允许它们进入有证据分数的评价。

---

## 3. 数据、指标、预算与身份

### 3.1 数据边界

沿用R13冻结划分与物化监督。预期train-fit11390、cal-fit624、cal-check616、dev1198、历史R10 test1166；以实际冻结manifest核验为准。不因新结果改变query集、source-group、相关目标集合或过滤条件。

所有训练、全局known-positive mask、候选刷新和Teacher补分仅使用train-fit监督。共享无标签湖仍是transductive。cal/dev/test标签不进入训练候选排除逻辑。数据湖可见不等于其评测标签可用于训练。

在线输入只包含query-by-example的可见表内容；target可见schema/cells和evidence正文／像素为正常对象内容。不得输入GT隐藏列、恢复值、GT target column、query的implicit/explicit真值身份或来源答案捷径。

### 3.2 主要指标

预定query集合Q，已知相关目标集合G_q，TopK为前K个唯一合法target：

$$
R@K=\frac{1}{|Q|}\sum_{q\in Q}\frac{|G_q\cap\operatorname{TopK}(q)|}{|G_q|}.
$$

主要端点：**完整dev总体target Recall@10**。同时报告implicit/explicit、单／多正目标分层的R@10/20/50。分层是评价，不是线上分流。未命中、空输出列表或失败query按冻结规则计零，不取系统成功交集。

交付候选数N=50，主评价K=10；@20为第二截断。CR@50为实际50个唯一候选中已知正目标覆盖。固定N集合仅重排时，Recall@N必须不变。

### 3.3 不把CR@50−R@10叫可保证获得的重排增益

P-S的CR@50−R@10为15.9015pp，表示当前11–50名含有额外已知相关质量，不保证全都能塞进top10。

准确的已知标签oracle排序上限为：

$$
O@K\mid C_N
=\frac1{|Q|}\sum_q\frac{\min(K,|G_q\cap C_N(q)|)}{|G_q|}.
$$

同时可报告全湖已知标签的截断容量上限：

$$
O_{\mathrm{all}}@K=\frac1{|Q|}\sum_q\frac{\min(K,|G_q|)}{|G_q|}.
$$

真实候选／排序差距分别为O_all−O_pool与O_pool−R。它们是已知qrels下的诊断，不是可部署结果；不注入GT、不替代Recall主端点。

### 3.4 固定在线预算

| 项目 | 规格 |
|---|---|
| 底层 | 冻结Qwen3-VL-Embedding-8B及原序列化缓存 |
| Student | D4096、d1024、共享table/text/image线性P、full有向R |
| QT | 100个候选 |
| Q→text / Q→image | 20 / 20 |
| 每条E→T | 20 |
| 路径 | 仅0/1跳，表级目标索引 |
| L/B | 每target最多20条待保留路径，D1保留最多4条 |
| N/K | N50，主K10，另报20/50 |
| 主排序 | D1 + F1 union-direct |
| 固定敏感性 | 同池、同D1、等权union-RRF60 |
| 搜索预算 | 正常完整池43个搜索向量/query；记录实际数，不等同API次数 |

新增优化不得通过增加evidence条数、ET返回数、候选N或隐藏在线Teacher提高效果。不同模型重建各自索引；同模型固定池诊断复用原池。索引参数、插入顺序、随机种子、线程、精度、合法target过滤和tie-break均冻结。

---

## 4. Task A：先读已有详细产物，把R13结论从摘要推进到可归因

本任务不新增正式训练。以复用为主，缺失时只补这一轮决策必需的检索／打分诊断，不重做无关工程整治。审批本计划后才执行这些补测。

### A0. 必需产物与数据状态

优先读取R13报告明确引用的：

- `statistics/summary.json`
- `statistics/bootstrap.json`
- `statistics/mechanism_and_reproducibility_audit.json`
- 各arm的manifest所指向的逐query ranking／path-pool记录
- B0一步干预明细、固定E exact明细、Teacher关系分层记录
- `PLAN_FROZEN.json`、`PLAN_FROZEN_AMENDMENT.json`
- 所选B13/S0延迟profiles及其他arm的成本／准入判定记录

以上是R13来源定位，不保证每个文件名在任何工作区都存在；以manifest解析真实路径。若不存在，记录“report-declared but unavailable”，不要写成复核通过。

必须输出一张自足摘要：query数、各桶数、source-group数、正目标多重性；每臂完整F1/pure-direct/RRF值；每臂已知支持覆盖；各选择护栏与排除理由。特别补明C-R为何不在 `eligible_arms`，不能仅依据单遍90.45ms替它推测理由。

若存在R13所选配方与其他臂的缓存／代码身份冲突，只停止受影响比较；没有冲突不重做全部1863项测试。修改公共代码后再执行相关回归。

### A1. 三种在线读数与F1等价检验

对S0、C-S、C-R、P-S、P-W读取或生成：

1. 真实direct ANN100，仅在这100内以统一raw d重新排序的结果。
2. 真实D/E并集上的F1结果。
3. 同并集上的等权union-RRF60结果。

pure-direct消融保留相同target合法性与tie-break，不因关闭E改变QT投影、索引或raw d计算。

另对S0和B13做一次全target exact QT诊断（当前报告有22886个target，以manifest确认）。批量计算同一raw d的exact top100，保持原E候选不变，将exact D100与E合并后按d重排，验证第2节恒等性质。该exact诊断不进入主结果，也不改正式索引。

输出：F1−pure-direct的逐query加权rescued/displaced；所有F1新入选目标的exact direct rank；相同分数上的ANN topK overlap；变动来自候选遗漏、重算分数、过滤还是tie-break。超出恒等边界时先查这些差异，而不是直接叫新科学机制。

固定U与d，再置换evidence路径分数：F1目标榜单应不变；若变化，查明实际依赖链并登记真实定义。这里不比较补值后的Stage2。

### A2. 将S0→P-S的收益拆成候选效应和打分效应

记U_0、U_1为S0、P-S自己的自然目标并集，d_0、d_1为两模型的direct分数。只在离线交叉诊断中计算：

$$
R_{ij}=R@10\big(\operatorname{sort}_{d_j}U_i\big),\quad i,j\in\{0,1\}.
$$

每个U_i保留它自身的候选ID，不先裁为top50再扩充；所有评分均在同一合法target域内。d_j对别的模型候选重新打分属于交叉诊断，不是自然部署结果。

四格可得对称分解：

$$
\Delta_{pool}=\tfrac12[(R_{10}-R_{00})+(R_{11}-R_{01})],
$$

$$
\Delta_{score}=\tfrac12[(R_{01}-R_{00})+(R_{11}-R_{10})],
\qquad
\Delta_{pool}+\Delta_{score}=R_{11}-R_{00}.
$$

这是固定两个产物的可复算功能分解，不是训练因果中介的唯一解释。两个效应在不同路径上可能交互；不把它们叫“direct loss的因果贡献”和“E loss的因果贡献”。后者由Task B回答。

分别对整体、implicit、explicit、@20、@50做相同分解。多正目标query的rescued/displaced必须按1/|G_q|加权，不能把正对净增数直接除以query数。

### A3. 读取而不是重新猜测B0和W诊断

至少导出以下数值而非“已经做过”状态：

- direct、evidence、ranking、KD、P/R各块一步更新对固定正边margin及exact rank的影响；按实际8个锁定path batches报告方向分布。
- 区分fresh-AdamW临时干预与真实训练checkpoint的optimizer状态；前者不被表述为重演历史坍缩。若需验证当前训练过程的同一机制，最多补S0与P-S178两个状态的锁定一步诊断，不启动长训。
- C-S与C-R同面板exact；不能仅用S0 5/16、C-R 6/16宣布拆分修复ET。
- W资格query／pair／row／source-group数、全178步有效更新比例、weighted W梯度与总梯度比例、训练和自然池已知责任、同一正target内未标注路径的责任。
- 109/109非零梯度只证明首个batch接通计算图；不得替代上述全程覆盖和相对梯度。
- hard candidates逐关系当前排名、current ANN top20 unknown与旧候选的overlap、positive margin和Teacher正例概率／排序；现有683/6545多正TT列表不自动说明四种证据关系存在同样问题。

旧面板若只有16个ET正对，保留这个小样本范围；优先使用已经建立的128-source面板补读数，不从成功例扩大面板。独立错误属性标签为0时继续输出null。

### A4. 解释范围

若F1总收益主要是d变化，这不使R13收益无效，而是收缩归因。若E训练在固定F1下改善了d，也属于训练作用；没有补值或可信见证依据时，不写成属性恢复机制。

若B13的CR50和R10差距大，先输出第3.3节oracle上限及哪些正例在11–50，而不是承诺Stage2能把15.9pp全部追回。

A的详细导出不要求先修生成，也不要求新人工标签。关键原始文件缺失时，可完成可支持的算术与边界报告，但不能恢复不存在的bootstrap或一步因果结果。

---

## 5. Task B：训练期移除E损失，推理仍保留完整evidence分支

### 5.1 科学问题

R13的`p_s_target_only`不是“只训练direct”。这里的target-only是“用target标签、没有额外W”，基础目标仍有direct与evidence两个通道。

要隔离E分支训练作用，新增一个**E-loss-off训练消融**。它和pure-direct推理消融完全不同：训练后仍完整运行QT、QE、ET、D1、F1以及RRF敏感性读数。

### 5.2 完整损失

保留R13实际多正目标CE与raw-logit KD：

$$
CE^+(l,C,P)=\log\sum_{t\in C}e^{l_t}-\log\sum_{t\in P}e^{l_t}.
$$

$$
L_D=CE_D^++0.3KL_D,\qquad
L_E=CE_E^++0.3KL_E.
$$

没有证据正目标及至少一个weak/unknown对比目标时，E项按原资格规则为0。完整控制F：

$$L_F=L_D+L_E+L_{anchor}.$$

新增D：

$$L_D^{abl}=L_D+0\cdot L_E+L_{anchor}.$$

保留R13完整anchor与weight decay，不因关闭E项修改正则参考。这样证据专属参数可能仍受anchor/decay影响；必须记录，不能声称它们严格冻结。所有参数保持原requires_grad设置，不将D伪装成完整主方法。

### 5.3 训练臂规格

| 字段 | F：full-path控制 | D：E-loss-off消融 |
|---|---|---|
| 起点 | 原健康S0，非B13 | 完全相同S0 |
| 初始化 | 原共享P、full R | 逐张量相同 |
| 优化器 | 新AdamW；R13公共配置 | 完全相同，状态不继承 |
| 主变量 | L_D + L_E | 仅把E通道CE和KD权重同时置0 |
| Teacher | 同一冻结Teacher及完整cached scores | 保留同一身份；D通道仍KD |
| 候选 | R13同一178个path batch；seed13 | candidate IDs/positive masks/evidence bindings逐位相同 |
| 正例 | 完整train-fit已知正目标保护 | 相同 |
| unknown | 原ranking-only语义 | 相同；不升级为confirmed negatives |
| P/R | 共享P/R完整目标更新 | 保持可训练；无E监督更新的块如实记录 |
| 更新 | 178；保存0/45/89/178 | 相同 |
| 自然评价 | 0/178完整dev，独立索引 | 相同预算、独立索引 |
| 在线 | 完整evidence分支 | 完整evidence分支，非pure-direct部署 |
| 主端点 | 总Recall@10 | D−F固定178步的总Recall差 |
| 新参数／新索引 | 0 / 0 | 0 / 0 |

公共优化设置继承R13：batch64、P LR1e-6、R LR1e-5、AdamW WD0.01、KD温度1、BCE0；anchor和实际梯度裁剪／精度以核验后的R13配置冻结，不从报告缺失字段猜数值。path使用raw两跳和与原LSE，D1推理不变。

为了不改变候选构造及计算图随机行为，首版两臂都计算同一完整forward，只在loss汇合处乘0。测量实际训练成本；可省略的E反向成本属于消融成本，不能称总FLOPs严格相同。后续部署不保留用于对照的无效计算。

### 5.4 控制复用与解释

F只有在当前入口、loss、mask、候选顺序、Teacher、anchor、optimizer初态及数值设置与R13完全可核验时才复用P-S178。否则重跑F178并明确新版本；最多一个必要控制重放，不能以可见Recall碰巧相同代替身份验证。

解释规则：

- D约等于F：未观察到在已蒸馏S0之上新增E-loss的目标Recall贡献；不是证明S0从未需要evidence训练，也不是等效性证明。
- D优于F：当前E-loss包可能在该条件下有代价；尚不能区分E-CE、E-KD、其尺度、候选分布与anchor交互。
- F优于D：存在保留E-loss的训练价值，即使最终F1用direct分数；仍需A2判断主要表现为d还是候选。
- F更好但见证覆盖降低：保留“target召回收益和已知支持取舍”，不称见证正确性已提高。

D是机制消融，不能自动成为论文部署主方法。所有D的结果照常进入主结果表，不能因其不符合方法叙事隐藏。

---

## 6. Task C：至多一个优化方向，不同时启动两个

选择只依据预先锁定的train-fit诊断和预算，不据dev反复试路。需要从已有R13明细导出真实证据；“做过梯度分析”或“候选未刷新”不能自动满足条件。

### C-G：direct损失对共享table投影停止梯度

#### 假说与区别

假说：path训练中direct分支通过共享P_table提升可见表间匹配，却损伤QE／ET邻域。C-R只在edge目标下拆角色，未隔离这个path分支问题；而即使拆了Q/T，QT的query端与QE仍可能共享P_query。

该方法不是冻结P，也不是关闭direct。它仅改变direct loss的反传去向：direct仍通过R_tt学习，P_table仍由evidence目标和原anchor训练。它与多任务梯度干扰问题相关，但不是PCGrad的复现，也不以负夹角替代实际更新检验。[R2]

#### 评分与目标

记sg为stop-gradient。训练时direct分支改为：

$$
\widetilde d(q,t)=\operatorname{sg}(P_{table}z_q)^\top R_{tt}\operatorname{sg}(P_{table}z_t).
$$

forward数值与原d严格相同。E分支保持：

$$
g(q,e,t)=(P_{table}z_q)^\top R_{tm}P_mz_e
+(P_mz_e)^\top R_{mt}P_{table}z_t.
$$

$$
L_G=L_D(\widetilde d)+L_E(g)+L_{anchor}.
$$

部署使用普通d与g，无detach概念，不增加参数、搜索或索引。与F对照唯一变化是direct CE/KD对P_table的梯度被阻断。

#### 触发规则

在原锁定8个path诊断batch上，用不在该batch训练列表中的固定train-fit正邻居面板，比较实际AdamW一步更新。面板为每个source保留全部已知正例及20个固定weak/unknown对比，排除全部已知正邻居；以“正例分数均值减这20个对比的LME”为margin。先在关系内按source平均，再分别对QE两模态／ET两模态等权平均。比较组在读取R13诊断后、任何R14训练前冻结；必须由同一个关系组满足下列规则，不能逐batch挑最坏关系：

- direct分支的P_table更新在至少6/8个batch使预先固定的已知QE或ET正负margin下降；
- 阻断该路径的反事实在至少6/8个batch改善对应margin影响；
- 在至少两个source-group上出现，而非同一个重复source解释全部现象。

上述计数是工程投入条件，不是显著性检验；同时报告exact rank，若margin与exact方向不一致保留冲突。fresh AdamW结果只支持fresh优化条件下的干预，不声称证明历史真实轨迹。

若实际主要损伤来自E分支惩罚合法QE边，而不是D→P_table，**不要启动C-G**；该结构不针对那种原因。

#### 固定实验

从S0开始，共享P/R，和F使用相同178-step path schedule、Teacher、mask、LR、WD、anchor、聚合、seed13；不得同时加W、edge维护、关系权重或候选刷新。

主比较G178−F178；还必须比较当前最佳B13与健康S0。只保住支持但Recall不提高，称机制／质量取舍，不选为Recall优胜。

可能失败：P_table失去对direct任务有用的学习，explicit下降；问题原本不是D→P；E-loss本身偏差更大；或者F1对E改善不敏感。上述失败都不能以临时恢复0.5倍D梯度救援；若需比例实验进入下一轮。

实现测试：同参数下forward logits/loss值相同；direct项对P_table梯度为0而对R_tt非0；E项梯度未变；不要在全局投影cache中替换为detach张量，误伤E计算图。实际总梯度裁剪触发率与Adam状态变化单列。

### C-RF：一次健康Student难候选刷新

#### 触发与机制

仅在C-G条件不成立、且已有明细显示当前自然困难候选与旧训练列表明显错位时使用。对固定train-fit source面板，每关系独立比较当前S0 top20 unknown与旧hard子集：

- 至少一类关系上，半数以上source的top20 unknown有一半以上未出现在旧列表；
- 同一关系中，半数以上source的旧hard分数中位数低于当前第100名分数，且至少有一个当前unknown竞争者超过该source的某个已知正例分数；
- 该关系至少有16个可用source；记录实际source数、rank分布和known-positive排除结果，不能仅由“356步没有刷新”判断候选陈旧已造成伤害。

这是刷新投入的预定诊断条件，不是因果结论。若数据不支持或样本不足，不运行此臂。

#### 对照与唯一变量

与R13 C-S匹配，从S0开始，共享P/R、同178个edge实例顺序及关系配额、同完整Teacher KD和全部参数。**只将既有hard负候选配额改为S0当前ANN生成的候选**；随机／原采样配额和所有已知正例不变。

1. 读取原列表的hard/random quota元数据。无法可靠区分时不凭位置猜配额，记录不可执行；该分支停止，不重造另一种监督协议冒充单因素。
2. 对每个train-fit source和有向关系，使用S0离线ANN挖同类型候选，已知正邻居全部保护。
3. 首次挖掘深度设为 `max(200, 2*hard_quota)`，上限512；不足部分使用原列表已有unknown依冻结顺序回填，记录不足比例，不静默改变列表宽度。
4. positives、confirmed-label语义、list大小、实例次序、random配额、loss与梯度设置均与C-S相同。
5. 所有新候选作为ranking-only unknown，不是确认非法边；Teacher完整补分后再训练，不使用零填缺失或缩窄KD mask。
6. 刷新只在训练前完成一次，随后178步固定；称“一次快照刷新”，不称持续动态负例训练。训练后按新模型自己的索引自然评价。

该方案受STAR/ADORE关于静态难例风险的启发，但不复现ADORE的query-only训练，也不冻结本应可训练的P/R。[R1]

#### 成本和止损

控制为C-S178，只有完全匹配时复用；否则补C-S控制。额外离线索引／搜索、候选比较、Teacher新unique pairs、token数、GPU小时全部登记。拟议新增Teacher unique pair上限为1,000,000；执行前估算并冻结。超过时停止该条件分支等待另行资源批准，不通过关KD、少打部分Teacher或减少某臂候选数绕过上限。

主比较C-RF−C-S；超过C-S而仍低于B13时，只称edge候选优化信号，不自动更新当前最佳配方。不能接着再跑一段path直到超过B13。

失败方式：更hard的unknown含更多漏标正例；Teacher在新列表上仍弱；旧列表并未陈旧；更新破坏泛化；新列表难度提高但没有新的有效信息。记录Teacher分层差异，不一边换Teacher一边归因候选。

### C的选择与未触发

两个条件同时成立时优先C-G，因为不需新增Teacher补分且更直接对应R13 path的已知支持下降。只运行一个方向。两个条件都不成立时，保留A/B、完成重复性研究，明确“没有证据支持进一步优化分支”；不强制凑出第三臂。

---

## 7. Task D：条件Student重复性，而非再次大范围选臂

R13没有运行17/23。下一轮应在一对配方上补条件重复，不对所有旧臂做多seed。

- 若Task C提供了达到第9节预定投入条件的新完整主配方：冻结它与其匹配控制，分别运行seed17/23各178步。
- 否则：冻结F与D消融，分别运行seed17/23各178步，检验P-S收益和E-loss增量的方向稳定性。

所有续训seed共享同一个S0、同Teacher、同候选集合；seed只改变预先规定的batch排列等Student随机性。训练／候选内容不得根据seed输出重新挖掘。ANN构建seed固定，避免把索引随机性混入Student方差。

这不是独立Teacher链，也不是从PCA重新训练全部Stage1的重复性。报告每seed的自然R@10/20/50、implicit/explicit、与该seed匹配控制的差值、已知支持和成本。不能拿三seed中的最大值更新论文主结果，不能把每个seed复制一份query后当独立样本bootstrap。

只有共同协议的seed13可以并入三seed汇总。否则报告新对照身份及真实样本量，不把旧最好seed混入。主开发差值和source-cluster区间仍是探索性；模型随机性与query抽样不确定性分别呈现。

---

## 8. 实验矩阵、依赖与预算

| ID | 主假说／变量 | 起点 | 控制 | 新阶段步数 | 是否自动部署主候选 |
|---|---|---|---|---:|---|
| A | 已有结果的候选／score／F1分解 | 已保存S0与R13产物 | 同模型固定池／exact诊断 | 0 | 否 |
| B-F | 原完整path续训参考 | S0 | R13 P-S，身份核验后复用 | 178，通常不需新跑 | 是 |
| B-D | 训练只移除E通道CE/KD | S0 | B-F | 178 | 否，机制消融 |
| C-G | 只阻断D-loss→P_table | S0 | B-F | 178 | 条件触发，是 |
| C-RF | 只刷新旧hard候选配额 | S0 | C-S178 | 178 | 与C-G二选一，是 |
| D | 唯一配方对的Student条件方差 | 同S0 | 对应控制 | 2seeds×2arms×178 | 不按seed挑最好 |

通常新增：B-D178 + 一个条件优化178 + 四个seed复核712 = **1068步**。没有C分支时为 **890步**。必要控制重放最多两个178步，最高新增训练上限 **1424步**。这些不是GPU小时；模型前向、反向、Teacher补分、索引构建另列。无差别复制旧任务不消耗新矩阵预算。

本轮不自动延长至356/1318步，不运行角色残差、角色×W、H、多latent、外部baseline或新数据湖。普通负结果完成固定端点；NaN、label泄漏、错checkpoint/index、mask破坏等立即停止受影响臂。

**资源最小版：** A + B-D178，复用F控制；若不能复用则补F178。完成后即可回答本轮最关键的训练分支问题。再安排一组条件优化与seed重复，不为了跑满预算启动没有诊断依据的模型。

---

## 9. 选模、显著性、成本与结果分类

### 9.1 固定端点与健康参考

当前最佳部署参考B13为：

`p_s_target_only@178`

checkpoint来源：

`/home/oycy/MMDD/work/stage1_optimization_r13_20260909/taskD_witness_supervision/p_s_target_only/checkpoints/step_000178.pt`

SHA256：

`d71cd8566bf9d1236f50dc74d12ba3d18a04b3d74e932919f34a81c5a560febf`

路径是来源标识，执行前验证实际文件。S0保持原父模型身份；不把B13更名成S0造成相对改善混乱。

主要训练比较固定178步；0/45/89只用于参数／loss／锁定面板诊断，完整dev仅0/178，不在看曲线后追加每步dev选峰值。

### 9.2 新方法入围与投入规则

新的完整主候选需：

1. 总Recall@10超过其匹配控制，且超过冻结B13；低于B13不能称当前最佳。
2. implicit/explicit相对B13各下降不超过2pp，同时保留相对原S0的变化，防止逐轮放宽参照掩盖累计退化。
3. 主在线预算不增加；相同机器、query顺序、线程与批量设置下重复测量的P95不超过B13的1.10倍。

进入新增配方seed复核，使用首版相对匹配控制至少+0.5pp的投入阈值且高于B13；它不是统计显著性界限。没有达到时可以保留小收益读数，但按第7节把seed预算用于F/D归因，而不是临时降低阈值追模型。

主选择器保持：`总体R@10 → 总体R@20 → 总体CR@50 → 较低可比实测成本 → 更早step → config_id`。按原始数值，容差1e-12。无新优胜则保留B13。D消融不自动入选，KD-off不因旧表略高而默默替代主线。

### 9.3 不把诊断再变成隐性主门槛

已知见证覆盖、责任或固定ET提高但R@10不提高：报告机制代理／候选层收益，不能用它替代选模。

R@10提高但已知支持下降：可以报告并保留Stage1 Recall配方，但标明evidence退化风险；不得宣称见证学习改善或端到端补属性已证实。

只超过退化控制、未超过S0或B13：分别标为损伤缓解、局部阶段进步或非最佳结果。只在某个属性／implicit子集提高，不隐藏全体query读数。

### 9.4 配对统计

全query先计算R_q，再做宏平均。source-cluster bootstrap10000次、seed13；每次抽到source后复制其中全部query，仍按query宏平均，不改成source等权目标。

预声明：B-D−B-F；实际触发的C方法−其匹配控制；新完整配方−B13。没有新C时第三项不凑数。每项报告总体及两个分层的差、区间、query数、source数、多正目标分母。区间不是经过持续开发选择后的独立确认，也不自动包含Student随机性。

R13旧test已经是历史回归集；R14只对最终一次冻结配方作回归，必要的B13同协议对照也须在打开结果前固定，不能用该集挑配方。新封存source-group集才可能成为独立确认；本轮不擅自构建它，也不因缺少它停止Stage1开发。

### 9.5 成本

至少报告Student训练时间、processed examples、candidate occurrences、unique pairs、Teacher新增／历史成本、索引构建／条目／字节／RSS、在线QT/QE/ET/补算/D1/fusion、P50/P95、cold-like/hot协议。

可复用的缓存写清依赖key；不同checkpoint不复用错误邻居或向量。43搜索向量不代表总计算严格相等，union大小改变会改变补算与D1成本。延迟比较至少三次、交错运行B13与候选，保留原始分布；缺失phase字段填null，不填0。

角色拆分没有在本轮主矩阵中继续测；旧单遍90.45ms只作为报告事实，不以它推断新模型内在复杂度。

---

## 10. 如何处理W、角色与H：明确暂缓，不宣布理论被否定

### 10.1 角色拆分

本轮保留共享P。R13表明“在这个178步edge续训配置中，拆分未提高总体Recall”。新增输入方向占更新范数38.0%/31.1%不是任务信息保留率；共享P也可旋转出原S0子空间，且未提供对应C-S方向对照。

只有后来在**同一path目标**下出现明确角色冲突且梯度路由不足，或更大／不同分布显示压缩容量不足，才重新研究角色残差。不得仅因“可能训练不够”无限延长已无优势的拆分臂。

### 10.2 W的真正局限

R13的辅助目标把已知正见证的跨target对比分数抬高，但没有显式要求它击败同一个正target内的所有未标注路径；这些路径也不能因未标注就作为负例。因此“有非零W梯度但平均责任没提高”并不自相矛盾，也不能仅靠放大系数解决。

本轮先读资格与梯度明细，不扫0.1/1/10。可做的低成本准备是仅在train-fit已物化记录之间恢复属性作用域映射，优先用可靠qrel／构造映射键，保留一对多见证；无法恢复时继续写any-known-witness。不要以位置猜GT属性，也不读取dev标签给训练补字段。

新的“同实体错误属性”惩罚需要独立、作用域明确的负标签。属性a下不支持不等于E或ET全局不合法；不能用unknown补齐。

### 10.3 H未被R13测试，也未被R13反证

真正的结构扩展可保持target离线向量：

$$
h(Q,E)=R_{mt}^\top u_E+B[(A_Qu_Q)\odot(A_Eu_E)],
\qquad s(T\mid Q,E)=h(Q,E)^\top u_T.
$$

它仍是单次内积ANN，但改变了局部J(E,T)primitive，失去按E跨Q复用查询结果的能力。MDR提供了“问题＋已检索材料形成后续query、语料向量独立编码”的相关先例；这里是轻量表示迁移，不是其任务效果的保证。[R3]

R13的role/W无主端点增益，只能解释当时没有继续花预算，不能成为否定H的证据。将来H需满足自己的条件：真实可见Q需求差异、固定E的目标混淆、足够训练正链，以及能够识别E排序作用的预声明部署评价。若F1对ET候选的增量完全无敏感性，不应在冻结d下期待H自动提升F1。

本轮不执行H、不执行潜状态多次ANN、不增加column索引。出现上述清晰结构证据后另立两臂H-Eonly/H-QE比较，参数、初始分数、候选与搜索预算匹配。

### 10.4 不默认重训Teacher或改多正例loss

当前KD-on/off区间跨零；没有证据支持扩大Teacher容量。先利用已存在的关系／难度分层能力记录；若某关系KD明确有害，可另轮做该关系或E通道KD的单因素消融，但不能将R13的整体−0.2087pp解释成每一关系有害。

10.44%的TT列表多正、其他四关系在当前物化中单正，不说明真实邻域只有一个合法正例，也不自动说明多正loss是现有QE问题的主要根因。只有完整分层显示多正目标被sum-probability持续忽略，才比较逐正例目标，不把新loss加入当前C-G/C-RF。

---

## 11. 后续Stage2：已有健康候选即可进入，不依赖R14成功

B13本身已可作为Stage2固定输入参考。R14不要求先让它超过另一个阈值，才允许研究读取、定位或增广重排；也不要求Stage1先证明尚未执行的补值机制。

本轮主矩阵不执行Stage2。后续独立计划需要冻结：完整N50候选、M恢复预算、B4、列头、生成器与排序规则；已评分joinable=False仍参与重排。min_row_coverage布尔字段不恢复成最终准入门槛，cell阈值的评分离散化另行分析。

后续主端点同样是全query Recall@K，K<N；固定N集合Recall@N守恒。保留正确材料、无E、错误实体／属性、删图、移除新增正确值的配对归因。生成正确1–2行也可对排序有用，不恢复“至少3/5行才成功”的旧要求。

最终论文仍要分别验证：已有direct目标因evidence补属性而排名改善；以及evidence发现direct固定预算之外目标后的收益。一阶段Recall改善不能预支这两类完整机制。

---

## 12. 实现位置与可直接落实的伪代码

### 12.1 模块定位

下列职责依据既有源码附件；执行前核对R13真实代码版本与已落地role-aware接口。不要重复实现R13已完成的角色／缓存功能。

| 位置 | 本轮变化 |
|---|---|
| `src/mmdd_stage1/scoring.py` | 保留通道拆分；C-G只在direct分支局部detach投影结果；暴露d/e原始分数用于交叉重算 |
| `src/mmdd_stage1/objectives.py` | 显式记录L_D/L_E/KD/anchor；B-D仅关闭E包；不换聚合／温度 |
| `src/mmdd_stage1/training.py` | 冻结178-step schedule、公共optimizer、实际梯度去向／状态、checkpoint身份和主Recall选择 |
| `src/mmdd_stage1/retrieval.py` | A中的pure-direct重算／exact QT诊断、自然池与索引绑定；C-RF离线挖掘 |
| `construction.py`或R13已有候选物化模块 | 仅C-RF：保持原hard/random quota、全局positive mask、unknown语义与Teacher完整mask |
| `row_support.py` / `retrieval_aligned.py` | D1不改；导出路径、保留、候选准入与贡献记录 |
| 现有R13评估／closeout入口 | 增加自足摘要和缺失产物状态，不仅输出“已做” |

新增实验入口名称可建议为 `src/run_stage1_r14_*.py`，**待实现**。本文件不提供假装已有的命令／参数；Codex先核验现有入口和`--help`，再保存实际执行命令。不创建跨项目的通用实验框架。

### 12.2 C-G伪代码

```python
# Proposed implementation, not an existing API.
# Projection caches remain attached for evidence scoring.
uq = model.project_query_visible(q_embedding)
ut = model.project_target_visible(target_embeddings)

if stop_direct_gradient_to_table_projection:
    direct_logits = bilinear(uq.detach(), model.R_tt, ut.detach())
else:
    direct_logits = bilinear(uq, model.R_tt, ut)

# Must use the original attached projection tensors/model, not detached cache entries.
evidence_logits = score_original_evidence_paths(model, same_frozen_batch)
ld = direct_supervision_and_kd(direct_logits, frozen_teacher, full_positive_masks)
le = evidence_supervision_and_kd(evidence_logits, frozen_teacher, full_positive_masks)
loss = ld + evidence_loss_weight * le + original_anchor(model)
# evidence_loss_weight=1 for F/G; 0 for D ablation.
optimizer.zero_grad(set_to_none=original_setting)
loss.backward()
apply_original_gradient_clipping_if_enabled()
optimizer.step()
```

### 12.3 有限正确性验收

- 全queryRecall重算、唯一target去重、多正分母、同一N前缀；缺榜单不能被剔除。
- F1 exact-D100恒等性质；固定U/d改E分数不改变F1；同分规则一致。
- B-D只改变训练E项；部署仍产生43搜索向量的完整图；anchor不静默变更。
- C-G同参数forward恒等、direct→P梯度为0、E→P与direct→R非0，cache不被整体detach。
- C-RF known positives不变负，random quota不变，新Teacher分数完整且unknown不进BCE。
- 修改不可见dev/test标签不改变训练输入／候选／mask。
- 复用checkpoint／索引／Teacher／候选必须有身份验证；不以Recall相同代替hash或配置一致。

单元测试验证行为，不代替科学结果。测试数量不是实验质量评分，不为文案微调重跑昂贵模型。禁止读取、检查或修改受保护的`.env.openai`；不启动无关模型服务或数据集重建。外部服务和新标注调用不在本轮自动预算内。

---

## 13. 交付物与报告最低信息量

建议目录：`work/stage1_optimization_r14_20260909/`。名称为待实施产物，不表示当前已存在。

```text
PLAN_FROZEN.json
R13_DEPENDENCY_READOUT.json
TASK_DECISIONS.json
comparison_manifest.json
runs.jsonl
stage1_A_attribution/
  direct_vs_union_vs_rrf.json
  exact_direct_boundary.json
  pool_score_cross_matrix.json
  query_level_contributions.jsonl
  known_label_oracle_gaps.json
stage1_B_branch_ablation/
stage1_C_one_optimization/
stage1_D_student_variance/
statistics/
  summary.json
  bootstrap.json
  per_query_metrics.jsonl
  mechanism_table.json
  cost_profiles.json
SELECTED_RECIPE.json
RESULTS.md
COMPLETION_AUDIT.md
VALIDATION.json
```

`RESULTS.md`下一次必须直接展示，不只说“详见另一个文件”：

1. 每臂总体／implicit／explicit的F1 R@10/20/50，pure-direct与固定RRF的相应读数。
2. F1−pure-direct加权rescued/displaced；S0→B13四格候选／score分解。
3. F/D及实际C方法的固定端点差、区间、每seed差、与B13/S0的差。
4. 完整已知见证覆盖、训练W资格／责任的实际值，以及五关系exact/ANN表；无独立负标签的字段仍为null。
5. 所有候选入围／被排除的具体原因与所用成本测量身份，不能只给eligible ID列表。
6. 是否触发C及依据；未触发标“未执行”，不是“失败”。
7. 不同seed、query集、source-group、正目标分母、历史test身份和qrels不完整限制。
8. 真实新增Teacher、Student、索引、在线成本；阶段成本缺失null。

完整逐query与路径产物可压缩保存，报告给来源指针和hash。未来审阅包至少附summary、bootstrap、mechanism_table、cost_profiles及逐query差值，而不是仅附通过状态。

原始R13结果与选择文件只读保留；R14不回写其主指标、延长门槛、历史身份或负结果。

---

## 14. 最后决策树

**A显示主要是direct评分改善，B中D与F相当：** 将R13主张收缩为目标级排序训练收益；保留evidence作为候选／Stage2材料。优先查E目标为何无增量，不能称path见证学习已成功。

**A显示候选有增量，B中F稳定优于D：** E分支确有训练价值；再根据实际D→P干扰或候选错位只做C-G/C-RF之一。

**G提高Recall并减轻已知证据损伤：** 支持本配置下的分支梯度路由改进；不是一般多任务干扰全部解决，更不是属性补全已完成。

**C-RF提高Recall：** 支持一次候选更新的条件性收益；新Teacher打分、unknown比例与关系变化是成本和中介，不宣称所有历史坍缩都由staleness造成。

**证据排序改进但F1无敏感性：** 不再把F1上的零差值当作见证表示无价值的全部证据。下一轮单独研究固定池证据准入／Stage2重排，保持原F1为强参照，不扫权重造赢家。

**所有优化无收益：** 保留B13，输出F/D的负结果与真实限制。可以开展已经具备固定候选的Stage2研究；不追加模型容量、长程训练或数据湖规模来掩盖结论。

---

## 15. 少量真正会改变执行设计的缺失信息

1. R13详细statistics与逐query自然池能否读取；它们决定归因、准确的多正例oracle上限与训练控制复用。
2. B0实际干预数值及optimizer身份；它们决定C-G是否值得运行。若只有fresh状态，结论必须限定，不能推断真实历史更新原因。
3. R13冻结候选是否保留hard/random quota与完整Teacher身份；它们决定C-RF是否能形成单因素比较。

属性字段缺失不阻塞A/B/C；只限制后续属性特定W。独立确认集缺失不阻塞开发；只限制最终泛化主张。无需先提供全部原始图片、完整200K湖或新的外部baseline。

---

## 16. 参考来源及迁移边界

### 附件

[S1] `RESULTS.md`，R13 Stage-1 results，上传版本2026-09-09。

[S2] `COMPLETION_AUDIT.md`，R13 completion audit，上传版本2026-09-09。

[S3] `SELECTED_RECIPE.json`，R13所选P-S178及checkpoint身份。

[S4] `VALIDATION.json`，报告code SHA256 `92650218e8fb3a283925b23510cef7ed71b68c2c2b8c18aea55b7044558aa6e6`；本轮未独立重跑其1863项测试。

[S5] `stage1_optimization_r13_plan_20260909_revised.md`，实际文件SHA256与[S2]匹配。另参照08更新与06的旧模块职责，不能将旧代码当R13精确实现。

### 仅用于新实验机制的外部文献

核验日期：2026-09-09。以下没有用于补写附件未提供的数据。

[R1] Jingtao Zhan et al. **Optimizing Dense Retrieval Model Training with Hard Negatives**. SIGIR 2021. https://arxiv.org/html/2104.08051v1

借用静态难例可能与当前排名脱节、需要候选对照的动机；本轮一次快照刷新不等于ADORE，不继承其效果，也不冻结全部target表示。

[R2] Tianhe Yu et al. **Gradient Surgery for Multi-Task Learning**. NeurIPS 2020. https://proceedings.neurips.cc/paper/2020/hash/3fe78a8acf5fda99de95303940a2420c-Abstract.html

借用任务梯度可能相互干扰的分析视角。C-G是固定分支梯度路由，不是PCGrad；负夹角或梯度非零不直接证明因果。

[R3] Wenhan Xiong et al. **Answering Complex Open-Domain Questions with Multi-Hop Dense Retrieval**. ICLR 2021. https://arxiv.org/html/2009.12756v2

借用后续query可含先前检索内容、target向量独立编码的分解结构。H在本轮暂缓，不以文献成功取代本地证据。
