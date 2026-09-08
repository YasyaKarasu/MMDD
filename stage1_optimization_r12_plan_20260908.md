# R12 实验计划：恢复两跳检索学习，并验证证据如何补全属性与促成 join

日期：2026-09-08。状态：**待实现、待运行**。本文制定下一轮实验，不表示已启动训练、模型调用、数据扩建或 Stage 2 实验。

依据：[R11 独立审计](work/stage1_optimization_r11_20260908/REVIEW_20260908.md)、[离线复算](work/stage1_optimization_r11_20260908/review_20260908/audit_evidence.json)、[R11 原始结果](work/stage1_optimization_r11_20260908/FINAL.md)、[科研方案](方案.md)、[AGENTS.md](AGENTS.md)。建议新产物目录为 `work/stage1_optimization_r12_20260908/`；保留 R11 原文件和负结果。

## 1. 科学问题与当前判断

R12 要回答两个相连的问题：

1. 在冻结底层 encoder、保持 P/R 可训练和单对象 ANN 的条件下，训练能否提高“正确实体—属性证据—目标列”路径的召回，而不再把表示训练坏？
2. 被召回并保留的 evidence 是否确实补出了 query 中缺失的属性值，并使原本缺乏连接条件的表通过 join 验证？

R11 的起点不是“只差提高 R@10”。Raw 的 678 个隐式正对中，489 对第一跳找到正确 evidence，216 对形成完整路径，E2 仅保留 184 对，F5 最终保留正确路径的为 111 对。E2 比 E1 净增 4 个支持行，区间跨零；C/D 长程严重退化；F5 只给出 5/8 个固定 Raw D100 外的阶段一发现，D1 自身 D100 外的口径未另报。混合模态未超过等预算文本，属性独立标注及真实补值尚未完成。

同时保留未入选的取舍点：image40+F5 有 10 个 ValidDiscovery，但总 R@10=23.95%、相对 F1 下降 2.26pp，未通过 R11 质量门槛。图像问题需要研究“哪些证据值得准入以及能否补值”，不能仅从所选 image-only F4 的 0 个发现推断图像无用。

据此排序：**实验正确性 → 真实候选分布与两跳关系学习 → 属性/行支持 → 实际补值与最终 join**。成功可以是定位清晰的负结果；不能因为故事需要证据就把未知标签写成支持、把短程起点写成训练成功。

## 2. 保留的主方法与明确的扩展

主线保留 Qwen3-VL-Embedding 的冻结缓存、类型线性 P、有序关系 R、细粒度 Teacher 离线训练/打分、Student 在线单对象 ANN，路径只允许 `Q→T` 和 `Q→E→T`。所有主训练臂 P/R 均接受有效更新。目标候选与 evidence 候选生成不反传；同一轮比较用固定候选 ID。

以下需要明确标注，不能自动替代主方法：

| 设计 | 实验身份 |
| --- | --- |
| P 冻结、KD-off、direct-only、单模态 | 定位机制的消融或基线 |
| F5 预留 evidence 名额 | 相对 `方案.md` 原始等权 RRF 的候选准入扩展；保留原方案对照 |
| 全局行分配或 evidence 多行使用 | 相对方案唯一 argmax 路由的扩展，必须核算相同行级处理预算 |
| Oracle 目标列、GT 路由、正确 evidence | 分解上限，不能进入正式在线系统 |
| 在线 Teacher reranker、非线性 P、解冻骨干、增加跳数 | 不进入本轮首批矩阵 |

`方案.md` 的 direct/evidence target 正例定义先保留。implicit target 的 direct 命中是合法检索结果，但不作为证据补全的贡献；其归因由无 evidence、错误 evidence 和最终补值对照完成。

## 3. 数据、候选预算和防止适应性偏差

沿用 EntiTables-v9 的同湖 transductive 设置。train-fit=11,390、cal-fit=624、cal-check=616、dev=1,198，R10 test regression=1,166 query。每次构造前核验 query/source group 的不交叠；先限定监督来源再建立正邻居和候选标签。共享未标注对象可用于 ANN 与几何参照，不允许 dev/test 标签进入训练候选真值、负例过滤或 Teacher 监督。

cal-fit 只拟合冻结规则明确需要的标定；cal-check 用于一次性诊断其泛化，不把它反复用于选择参数。不存在独立确认集时照常完成开发研究，全部新增结论限定为探索性；不通过重切旧 test 或改 seed 恢复“盲测”身份。独立确认需要以后真正未查看、未进入训练祖先监督的 source groups，此轮不把扩建数据设为前置。

首批统一预算：`Q→T=100`；mixed `Q→text=20、Q→image=20`；每条 `E→T=20`；每 target 待保留候选上限 L=20、B=4；最终 K=10 为主，K=20/50 为诊断。text-only/image-only 各检索 40 条 E，保证同总 evidence 搜索预算。模型变化时建立自己的索引；同模型聚合/融合比较共享冻结路径池。

同时保存实际去重后 target/path 数量、补算 direct pair 数、行—证据打分数量。不能只以 ANN 调用数相同宣称总成本相同。F5 的每个 K 独立分配名额，K10 不从 K50 队列截取。

## 4. Task A：修正审计中发现的协议问题，重算受影响结果

本任务完成前，不开始新的长程训练。新增实现放在职责对应的 `src/mmdd_stage1/`、`src/mmdd_stage2/` 或现有入口中，保持直接实现，不搭建通用实验调度框架。

### A1. 监督与 mask

- 没有独立反驳依据的 corrupted E→T 标为 weak/unknown；可在明确声明的 ranking 抽样中使用，禁止进入 BCE、Brier/NLL 或 confirmed-negative 指标。
- 真正的负标签保存依据、对象和属性作用域。不能从 `Q,T` 非正推断 `E,T` 非正，也不能把“E 对当前属性不适用”升级为“Q 与 E 永远无关”。
- 审计调用实际扩池构造及 mask；从真实结果统计 local/global mask 的误负例。保存每 epoch 及总数；故意破坏 global mask 的测试必须使审计失败。
- 训练读取已物化 train-fit 监督；改变不可见 split 的标签不得改变 train-fit 的监督输出。历史 B0/B1 如用于比较，只重放，不净化旧权重后冒充 fresh 主线。

### A2. checkpoint 与训练观测

持久保存 `P_PCA` 和阶段起点 `P_stage_start`，分别记录相对二者的漂移。主臂正则参考固定为整链 PCA；阶段参考只作诊断，若作为训练正则则另列消融。R 参照仍为初始 identity。保存参照内容指纹、P/R 实际更新与梯度情况；重载前后同模型、同输入的分数及参照必须一致。

统一 edge/path 的湖检索回调与选择器。edge 的 `eval_epoch_zero=true` 必须产生真实 epoch0 checkpoint 和 metrics；保存指定 step 检查点，而非只在 epoch 末报告构造 list R@1。明确区分固定预算 `.last`、dev-selected、阶段初始化；接续哪一个由预登记的运行身份决定。

### A3. evidence 通道与干预

空集合返回缺席标志，不进入 evidence 排名或分位数。F1 的 union-direct 仍覆盖原 D/E union 中所有合法 target；删除 evidence 后的 target 是否保留 direct 补算必须与反事实定义一致。固定池删图解释“已有候选中 evidence 的贡献”，完整 text40/mixed 重检索解释模态替换的端到端收益，两种实验分别记录。

加三种真实行为测试：只有 direct 的目标不能得 evidence 票；移除全部 E 后 F5 回填 direct；exact-content 副本去重后不增加覆盖。不同内容同一行的冗余实验留给 Task D，不能用 exact-copy 测试代替。

### A4. 必须先做的重算

1. 在 R11 冻结池上重算 Raw、D1 short、text40、image40 的 F1–F5，以及删图干预；cal-fit 重拟 F4 尺度，再在 dev 选规则。保存旧/新 topK target IDs、改变的 query 数和全部指标。即使 F5 数值不变，也记录逐 query 检验依据。
2. 从已有 C1/C2/D1 的 edge epoch1、epoch2 和 path epoch0/1/2 checkpoint 重建各自索引并做相同漏斗，补共同端点；重载的纯推理权重不受 P anchor 修复影响。
3. 对同一固定 128 query/至多 128 evidence source，在 PCA 和 C2 训练后至少两个代表性 checkpoint 上做 ANN/exact，记录正邻居 Recall@K、any-positive Hit@K、topK overlap、并列分数与范数。不能用 Raw 的 exact 结果替训练后模型排除 ANN 问题。
4. 清理 D3 标签语义；若没有确认正负样本，保留旧排序读数为“构造负例映射对照”，停止新的概率校准，记录条件未满足。

验收：修复后的行为测试通过；每个重算指标有 query/pair ID 和整数分子；修正结论与 R11 原值并存。此任务只修正确性并补评估，不把新结果当作新训练增益。

## 5. Task B：先建立小型真实属性支持审计

假设 B：现有 evidence 分数和 E2 row cosine 混合了实体相关、属性可用性和背景相似性，缺少正确支持的判别依据。**先检验，再决定是否训练新的支持模型。**

第一批 256 条 row—attribute—evidence 候选，分为 train-fit/text、train-fit/image、dev/text、dev/image 四桶，各 64。每桶按固定 hash/source-group 抽样：16 条已有正 recovery 用于复核，24 条已召回且高分但未标注候选，16 条同实体的候选错误属性材料，8 条候选实体/值冲突材料。后两类是待审核候选身份，不预设其真值。材料不足时保持缺额和实际分母，不挪用 dev/test 补训练标签。每 source group 最多 4 条；保留抽样池、覆盖率及各层抽样概率。

实际审阅可见 row 全部列名-cell、目标候选列语义及原始正文/图片；记录 `(query_id,target_id,column_id,row_id,evidence_id)`，标签分为：确认正确实体属性值支持、确认同实体错误属性、确认实体或值冲突、信息不足。正例需具体值与可定位证据；错误属性需明确属性作用域。双人独立审阅至少 20%，争议保留 unknown 或裁决记录，不能用同一生成答案自行验证。

这一步必须真正读材料并记录依据；重新运行 R11 的 membership 脚本不算独立标注完成。若采用模型辅助，模型判断只作辅助并保存可检查依据，所用服务和配置在执行时登记；不自动读取受保护的 `.env.openai`。

输出：分桶确认支持率、unknown 率、抽样加权估计、row routing 的错误类型、图像能否独立表达所需属性。增采正例或难例有助于诊断，但不能把该样本的裸 precision 当全湖 precision。分析每 `(q,t)` 是否存在多种桥接属性，并为后续覆盖指标保留 column 维度。

新监督式支持模型仅在 train-fit 有至少 64 个确认正例、64 个确认负例且各覆盖至少 20 个 source group 时启动；训练侧以 source group 划分内部验证，dev 审计标签仅用于最终报告。达不到条件就保留无监督对照，记录“标签不足”，不能把 unknown 当负例来达到门槛。这是投入门槛，不是统计充分性保证。

## 6. Task C：定位训练为何损伤两跳，再做小规模优化

### C0. 先回答损伤发生在哪里

以 Task A 的现有 checkpoint 复评为依据，按五种有向关系分别画出/列出：构造 list R@1、多正邻居 Recall、全湖 Recall、GT 正边分数及 rank、完整路径 ValidPool。统计目的对象成为 topK hub 的频率、投影向量范数与方向变化、近邻保持率。

在固定 train-fit 小 batch 上，仅做可复现的梯度诊断：supervised ranking、KD、P/R 正则分别对三种 P 和五种 R 的梯度范数、两两夹角，以及它们对固定正负边 margin 的影响。固定原始 PCA 参照和阶段参照；不能以参数均方距离小或总 loss 下降替代这些功能指标。

优先检验的可证伪解释：

| 解释 | 支持它应出现的结果 | 反证或其他解释 |
| --- | --- | --- |
| 构造候选不代表部署难度 | 原 list 高分，真实 ANN 难列表/全湖差；改变候选组成改善真实路径 | 同候选诊断仍失败且替换候选无效 |
| 函数几何漂移损坏证据邻域 | 真实边 rank/hub/方向显著变化；温和函数约束改善两跳 | 约束只保住初始化，未学到正确新关系 |
| KD 与监督冲突 | 相应关系梯度长期相反，KD-off 在同候选同预算下改善 | R11 KD-off 仍失败，故 KD 不能单独解释全部坍缩 |
| ANN 近似误差 | exact 明显恢复训练后正例，ANN 丢失加剧 | exact 也差则回到模型/监督问题 |

### C1. 首批训练：一个对照、两个单因素臂

起点均为相同 PCA-1024、R=I；底层特征、seed13、batch64、样本次序、主 ranking/KD 设置固定。对照沿用 C2：supervised `sigmoid(s)/0.1`，Student/Teacher KD 使用 raw logits、KD 权重 0.3、温度 1、BCE=0；P/R 学习率 `1e-6/1e-5`、AdamW weight decay=0.01。仅纳入 Task A 明确修正后的配置；不静默将 sigmoid 传播到路径和或 KD。

| 臂 | 唯一算法变化 | 要回答的问题 |
| --- | --- | --- |
| C-base | 修正协议后的 C2 对照 | 在有效观测/锚点协议下重现或消除退化 |
| C-candidates | 用冻结 Raw ANN 的检索难例替换固定比例扩池负例，最终候选预算匹配 | 问题是否来自训练候选分布而非函数类 |
| C-function | 保留 C-base 候选，增加尺度标准化的函数几何正则 | 参数均方锚定是否保护不了实际检索关系 |

C-candidates 首版将可用负例名额的 1/2 分给同 source、同关系的 Raw ANN 高分候选，其余沿用原采样；排除 train-fit 已知全局正邻居。按固定名单补齐与 C-base 相同的每 list 候选数，无法补齐时报告实际规模。C-base 与实验臂都保存候选 ID；改变组成是唯一干预，不能同时改温度、loss 或关系权重。Teacher 为新增候选重新离线打分，三臂在相同完整候选 mask 上计算本轮声明的 KD，不使用过期 logits。

这里为完整候选补齐 Teacher 分数是三臂共享的 **R12 新训练协议**；C-base 因而不是 R11 只对原局部列表做 KD 的逐位复现，后者保留在 Task A 的历史重放。先估算离线 Teacher pair 总量并登记预算，比较新候选分布的收益以 C-base 为控制，不能把 KD 覆盖范围变化的收益混记给 C-candidates。

先在这些真实难列表上比较 frozen token Teacher、Raw 分数和 PCA Student。Teacher 若没有有效优势，记录负结果并在同候选下加入一次 KD-off 配对诊断；不能把这种情况下的失败归结为 Student 压缩能力不足，也不立即扩大 Teacher 结构。固定 Teacher 的每条 scored pair 计入成本。

C-function 使用训练可见无标签对象上的固定五关系 pair 集合，保留原参数 anchor，再增加：

`L_function = mean_relation mean_pair ((s_current(a,b)-s_PCA(a,b))/sigma_relation)^2`

其中 `sigma_relation` 是该固定 train-fit 参照集合上 PCA score 的标准差，开始前冻结；每关系 256 个 pair，包含正边邻域及随机对象，以固定轮转小批次计算。首版系数 0.1，记录实际梯度比例；这是一项待验证正则，不把它当成保持任何原始邻域都正确。若它仅选回 step0，则判“没有证明学习收益”。不按 dev 表现连续调整正则系数。

三臂先各运行 356 个 edge 更新，保存 `0/45/89/178/267/356`；每个点做同口径小诊断，0/178/356 做完整 dev 湖检索。若不足以定位首次损伤，使用已保存邻近 checkpoint 做离线评估，不额外长训。构造候选、Teacher 打分和索引成本各列，不能只按 Student 更新数宣称总计算相等。

其中至多一个新训练方向与 C-base 进入延长：选择参考末端 ValidPool、确认属性支持/实际路由覆盖及 direct 质量，固定规则见第 10 节。两臂运行相同累计 659/1,318 edge 更新共同检查点；若连续两次完整检索 ValidPool 低于起点 80%，停止该臂。提前停止的臂只在共同已完成步数比较，不能以短端点对另一臂完整训练。

最多允许一次事先登记的学习率救援：P/R 同时乘 0.1，其他不变，仅在所有首批臂都退化时启动，356 步后失败即收尾。此为 R12 新规则，不声称 R11 已执行这项救援。不继续扩大网格直到找到正结果。

### C2. 健康起点上的 path 学习

只有 edge 起点仍保有至少 80% 自身起点 ValidPool，才把它用于比较 path-only objective 与 `path + 0.1 × continuous edge objective`；后一项使用训练侧固定边表，旨在防止路径和掩盖其中一跳的损伤。两臂均 356 个 path 更新、batch64、相同 Q→E→T 候选，Direct/Evidence CE 同权、raw 两跳和及同一种聚合；额外 edge 前向成本单列。

三类难例分别生成：Q→E ANN 的 hard evidence、Q→T ANN 的 hard target、完整两跳和排序的 path-hard；有独立配额，不能从错误 target 的 paths 顺带抽 E 冒充 hard evidence。首版在训练前冻结；只有静态实验有稳定改善，才允许一次当前 Student hard refresh 对照，候选仍只使用 train-fit 监督判定已知正例。

不向训练图或部署召回池注入 dev GT evidence。若训练需要保证每个 positive 有支持路径，所有臂采用相同的、明确记录的训练标签候选构造；推理评估始终自然检索，并报告 teacher-forced 训练与自然检索的差异。

## 7. Task D：证据保留必须对齐行、属性和部署预算

先在 Raw 固定池完成，再补一个通过 Task C 的 P/R 可训练模型。没有健康模型时，Raw 可继续揭示聚合问题，但不能当作训练主线成功。最多三种策略：

| 策略 | 用途 |
| --- | --- |
| D0：E1 exact-content 去重后 top-B | 可解释的保留对照 |
| D1：R11 E2，修正空集合语义 | 历史未校准 row-coverage 对照 |
| D2：按实际 argmax 路由分组，再选择跨行证据 | 检验 soft 多行收益与实际唯一路由的不匹配 |

D2 仍只用可见 row 和冻结 evidence embedding 决定 `argmax row`，每条 evidence 对其他 row 的支持置零；每行优先保留质量最高的候选，再按其质量选择最多 B 个行的证据。使用与 D1 相同质量分数、top-L、内容去重和破同分规则。该臂只检验与部署一致的稀疏分配，不把它称为已具备属性判别能力。

若 Task B 的训练标签满足门槛，允许追加一个小型属性支持臂：在检索后的至多 L 个 evidence 上，用可见 row、目标 schema/候选列与 evidence 表示预测支持；单独报告其成本和相对 D2 的作用。GT 正确列不得作为正式选择输入。Teacher/schema 细粒度打分只允许离线监督；在线仍保持预登记的轻量实现，不能静默换成在线 Teacher rerank。

所有策略同时报告：

- GT evidence 行并集的 RowB、固定属性 RowB、ValidB、多行支持分布，全部按隐式正对汇总，未入池为零。
- 方案现有 argmax 的实际路由覆盖、确认支持证据的错误行分配率；再给 GT 二分匹配上限，不能把上限当实际成功。
- 已确认属性 precision、unknown 比例及其来源/分母；确认支持很少时保留区间和数量。
- 每 target/每 query 实际 evidence 数、不同内容数和行级处理次数；E2 的“均值变少”不能自动视作全系统节省。

固定池干预采用原始、删图、不同内容但同一行的冗余、同实体错误属性四组。后两组只使用独立确认材料；匹配模态、数量及分数区间，若必须回退则独立报告。exact-copy 只作为去重健全性检查。干预既记录固定队列上的证据变化，也记录重新聚合后 target 准入变化；只有系统分数/准入响应与属性支持一致，才能说系统利用了该信息。

## 8. Task E：准入、模态贡献与 direct 的净取舍

在 D 的保留策略冻结后，只比较：F1 union-direct、修正后的原方案等权 RRF、F5 half quota。F4 只作为 A 的历史修正，不继续扫描权重网格。三者使用同一 direct 补算范围，报告 Raw D100 外和各模型自身 D100 外的发现，防止候选边界变化造成假增益。

准入分解须保存每个 `(q,t)` 的原 D100 成员资格、各分支排名、最终 K 名单、选中 E IDs、正确属性行支持以及相对 F1 的 rescued/displaced。分别报告：

1. D100 内目标重排，以及其支持是否更充分。
2. D100 外的新增 target，但没有确认正确 evidence。
3. D100 外新增且有确认正确属性 evidence 的 ValidDiscovery。
4. 最终被正确补值并通过 join 的发现，留到 Task F。

text40/image40/mixed20+20 在相同总搜索预算、相同 K/B、相同保留/融合规则下比较。先冻结 Raw mixed 上选定规则并用于三种模态，再另表报告各自 dev 最佳值，防止把不同规则效果算作模态互补。image-only 若没有通过质量约束，照实列为取舍点，不自动放宽门槛。

固定池图像独有的 label support 只能支持局部贡献。建立互补主张需要在共同样本上同时看到：图像提供文本遗漏的正确属性/行；mixed 在成本匹配下增加正确值恢复或最终 join；删除/错误属性替换使相应收益消失。总体没优势但某类属性有效时，结论限定到该预定义属性桶，并报告其他桶的负结果和比例。

## 9. Task F：最小端到端机制验证

这是相对 R11 仅研究 Stage 1 的明确范围扩展，目标是补齐 `方案.md` 的关键证据链。先用冻结 Raw/D 策略即可做小样本实验，不把 Stage 1 训练成功设为所有机制验证的前置；训练主方法成功与完整机制成立分别报告。

第一批固定 128 个 dev query：96 个 implicit、32 个 explicit，按 source group 和标注模态/属性分层抽取，在查看新系统输出前锁定；另把所有当前已知 ValidDiscovery 案例列为案例审计，不能用案例富集样本估计总体效果。主样本缺组时列真实分母，不从成功样本补齐。运行前验证输入的可见字段；不把隐藏列真值、恢复答案或 provenance 放入模型输入。

先消除现有 Stage 2 选列实验可能利用列位置的捷径：在固定随机列排列下训练/评估，确保正确列索引同步更新；加入错误 target/无可用列例子，报告拒绝能力。不要复用“所有正确列都在第 0 列”的成功率作为语义选列依据。现有 verifier 如暂不能拒绝，登记局限并测 false positive，不能省略错误目标。

分两层做归因，避免一次更换太多东西：

**F-a：固定候选队列与目标列的诊断。** 在同一 R12 检索 topK 内，将 oracle 列作为显式上限条件，比较无 evidence、检索 E、等数量同实体错误属性 E、GT 可支持 E。先测路由、定位和生成值；oracle E 允许补入仅为读取上限，单独记额外成本，不能回填正式检索 Recall。

无 evidence 的读取诊断仍在相同行/列调用同一个生成器，只把证据上下文清空，用来测模型先验；另记录正式 pipeline 无 E 时保留空值的行为。不能用“禁止调用生成器，所以必然输出空值”作为模型依赖证据的证明。目标表可用于选列和最终 join 检测，但填值输入不得携带目标列的待匹配值、隐藏 query 真值或标注答案。

**F-b：实际完整链。** 使用真实 topK、真实选列、方案的 argmax 唯一路由、FOCUS 定位、生成、semantic join 验证；比较 F1 队列与 E 选出的融合队列，reader/生成器/验证阈值相同。Direct 分支单独验证，Evidence 分支只使用统一截断后的目标，不各自重新从全池取 topK。

必须记录下列主机制读数：

| 读数 | 定义及必要分母 |
| --- | --- |
| CorrectColumn | 在实际候选中选择正确桥接列/接受等价列的比例，另报 GT 表缺失和错误目标拒绝 |
| RoutedSupport | 路由到该行的 E 确认支持所选正确属性的行比例；全部 query 行与可恢复行两种分母 |
| CorrectValueRecovery | 输出与独立真值语义等价且有正确 evidence 支持的行数；空输出按未恢复，错误值单列 |
| FinalJoin | 生成后通过冻结 verifier 且与独立 join GT 一致；同时给 precision/recall、false positive |
| EvidenceEnabledJoin | 同候选、同目标/列控制下，无 evidence 失败而检索 evidence 成功，且新增正确值有证据支持 |
| OutsideDirectFinalJoin | 在 D100 外通过完整 evidence 分支实际发现并验证正确的 join |

“模型输出正确”与“由 evidence 支持”分开。无 evidence 也答对的值不能全归给多模态；错误属性替换后仍不变时，检查模型记忆或 target 泄漏。验证器自己的“通过”不等于独立 GT 正确，需人工复核全部新增成功及固定比例失败，避免同一模型自证。

首版保留一条 evidence 只能处理一个 row 的方案，B=4、query=5 时单 target 最多四行有 evidence，这一计算约束明确披露。只有实际路由明显落后于二分匹配上限，才追加一个等总 row-evidence 处理次数的全局分配对照；允许 E 分给多行、拼接多条 E 生成等作为后续扩展，不同时加入首批实验。

## 10. 选择、统计与停止条件

### 训练侧

主要科学端点是相同实际更新数的全湖 ValidPool；同时看每关系召回及确认属性路径。开发 checkpoint 使用显式字典序 `ValidPool整数分子 → RowB → ValidB整数分子 → direct R@10 → 更早step`，保留 R11 声明的选择目标以避免本轮再次无意更换标准；实际路由/属性支持另作机制约束和报告。固定端点、selected、初始化三张表，任何选中 step0 的阶段不宣称训练带来收益。

候选入围延长要求 ValidPool 不低于自身起点 80%，作为节约计算的门槛，不是函数类可行性的统计检验。主方法成功至少要求 P/R 有效更新、同预算完整路径与属性支持有受控改善，并保留 direct 质量。若只保住基线而无机制增益，写成稳定性修复。

### 最终组合

dev 主组合要求总 R@10 与 implicit R@10 不比同模型 F1 低超过 2pp；新增 explicit 不低超过 2pp 的独立约束，以暴露 R11 Raw F5 的 −3.01pp 取舍。这是 **R12 新门槛**，不能追溯改变 R11 的合格判定。全部门槛按未四舍五入值执行，容差 `1e-12`。

通过后按 `实际正确属性路由覆盖 → ValidPath → ValidDiscovery → 总R@10 → 更低实测成本 → 配置ID` 选 Stage 1 组合；属性/路由样本覆盖不足时暂不选“机制主方法”，使用全量 RowSupport 的探索性排序并清楚标记缺失。Task F 端到端实验只评价已冻结组合，不用这 128 个 query 反复调 reader 或 fusion。违反门槛但机制更好的点保留在取舍表，不能全删掉。

若没有融合策略通过全部质量条件，报告“未得到全面改进的主候选”，仍可冻结一个机制取舍点进行 Task F 的诊断；这种诊断不改变其未通过质量门槛的身份。

### 统计与复现

每次比较先对齐同一 query/pair/attribute 集合。R@K 是 query 宏平均；路径、RowB 以隐式正对为单位；属性指标先固定同一桥接属性；多正目标不能无权再次平均 query 路径均值。两系统的遗漏记零，不取二者成功交集。

首批预声明三个主比较：C 入围训练方向−C-base 的共同端点 ValidPool；D2−D1 的实际 RoutedSupport；冻结融合−F1 的 EvidenceEnabledJoin/OutsideDirectFinalJoin（后者有足够有效样本时报告）。按 source_table_id 配对 bootstrap，10,000 次、seed13，保留整数分子、差值和 95% 区间。开发选臂后的区间是探索性，不能当确认性显著性；案例富集审计不进入总体估计。

冻结最多一个新 P/R 配方后补 Student seeds17/23，并重复对应对照；Teacher 固定时只称条件 Student 方差。若要宣称完整 Teacher–Student 稳定，再补独立 Teacher seed 链，不能以三个 Student seed 代替。

若将来有封存独立集，提前冻结最多三个主张；使用 source-group 内配对交换、10,000 次、seed13 的随机化检验及 Holm 校正，bootstrap 用于效应区间。没有独立集时不强行生成确认性 p 值。

## 11. 工作量、依赖与条件扩展

| 顺序 | 工作 | 首批预算/触发条件 |
| --- | --- | --- |
| A | 正确性修复、已有模型与池重评 | 无新训练；优先修正会影响解释的产物 |
| B | 实际属性审计 | 256 条，来源/模态分层；可与 A 后半段并行 |
| C0 | 几何与分项梯度诊断 | 既有 checkpoint 与固定小 batch |
| C1 | C-base/C-candidates/C-function | 各356 edge更新；最多一个新方向与base成对延长 |
| C2 | path 与 path+edge | 健康起点才执行，两臂各356更新 |
| D/E | Raw 的保留、融合、模态实验 | 固定池三种保留、三种融合；候选变化时才重导池 |
| F | 端到端机制试验 | 冻结128 query，先诊断后完整链；全部模型/标注费用登记 |
| 复现 | 新训练臂和对应控制 seeds17/23 | 首个健康配方冻结后执行，不对全网格多种子 |

不把所有条件扩展一起启动：多正例 loss 仅在确认困难正例被集合概率目标忽略时比较 `sum_probability` 与 `mean_log_probability`；token/pooled Teacher 仅在 Student 目标稳定后同候选、同训练数据对照；关系加权只在分项梯度提示特定关系冲突时做单次配对；动态 refresh 仅在静态检索难例显示收益后执行一次。

本轮不优先扩大湖、加模型容量或加入在线重排。200K 的作用仍是规模/效率，未经精标的效果不替代高质量数据上的机制结论；跨 WDC 泛化留到 EntiTables 配方冻结后作为外部数据域检验，协议变化单独写清。

NaN/Inf、监督泄漏、secret/config 意外加载、打分空间不一致、错误 checkpoint/index 配对等立即停止对应 run 并记实现失败。普通负结果完整保留；没有满足条件的分支则标“未触发”并说明实测原因，不能用硬编码状态代替诊断。

## 12. 交付和最终可以写进论文的结论

每个任务保存：假设、实际命令/配置、输入与代码指纹、候选及 Teacher 身份、checkpoint 参照、更新数、逐 query/pair 指标、cost 和停止原因。真实 elapsed 未测量时写 null/未测，不能把缺省 0.0 当作零成本。报告索引时间/大小、在线 ANN、direct 补算、保留/融合、reader/定位/生成的 P50/P95，并说明硬件与重复测量条件。

预期文件：`PLAN_FROZEN.json`、`taskA_correctness/`、`taskB_attribute_audit/`、`taskC_training/`、`taskD_retention/`、`taskE_admission/`、`taskF_end_to_end/`、`runs.jsonl`、逐样本预测、`FINAL.md`。复用现有模块；新代码不进入 `scripts_old/` 或 `mmdd_dataset/`。实验新选项实现并核验 `--help` 后才保存可运行命令，不在计划中伪造尚不存在的 CLI。

相关测试在 MMDD 环境、隔离临时 cwd、合成数据/配置下运行；禁止读取、检查或改动 `.env.openai`。本轮修改训练/检索公共逻辑时运行完整 Stage 1 相关回归；Stage 2 变更补对应 verifier/oracle 测试。测试覆盖真实行为和实验完整性，不为可逆文案改动增加测试。

结论按证据强度分别写：

- 若只修复计数、锚点或空通道：这是正确性修复。
- 若训练稳定且自然检索的正确属性路径改善：支持可索引关系学习的 Stage 1 收益。
- 若只增加潜在覆盖或候选准入：只支持这一阶段的作用。
- 若正确值恢复增加、无/错 evidence 对照失去相应收益且最终 join 正确：才支持证据补属性机制。
- 若 mixed 在相同成本下胜过 text-only，并有图像提供独有正确属性及成对干预支撑：才支持多模态互补优势。
- 若某一环节失败，保留负结果和条件范围；不通过冻结 P、压低 evidence 权重、在线重排或挑选指标重新包装为原主方法成功。
