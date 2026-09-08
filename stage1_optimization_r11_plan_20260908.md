**R11 实验计划：修正训练监督，恢复可组合的证据检索，验证行与属性互补**

日期：2026-09-08。状态：待实现、待运行。本文件只制定计划，不表示实验已经启动。

计划依据：[R10独立审核](/home/oycy/MMDD/work/stage1_optimization_r10_20260907/REVIEW_20260908.md)、[审核数值证据](/home/oycy/MMDD/work/stage1_optimization_r10_20260907/review_20260908/audit_evidence.json)、[R10冻结结果](/home/oycy/MMDD/work/stage1_optimization_r10_20260907/FINAL.md)、[科研方案](/home/oycy/MMDD/方案.md)与[AGENTS.md](/home/oycy/MMDD/AGENTS.md)。

计划产物根目录：`work/stage1_optimization_r11_20260908/`。新增实现放在`src/mmdd_stage1/`和`src/`，配置放在`configs/`，测试放在`tests/`。复用明确的现有函数，保持算法路径直接，不为本轮增加通用实验框架。

**1. 研究目标与本轮边界**

本轮要回答：**在保持P/R可训练及单对象ANN检索的前提下，能否让正确属性证据更稳定地到达候选池、覆盖不同query行，并促成direct候选池之外的正确目标发现？**

主线保持冻结的Qwen3-VL-Embedding骨干、细粒度Teacher、线性投影P、有序关系R、独立对象索引，以及`Q→T`与`Q→E→T`两种路径。P冻结、KD-off、只有direct、只有单一模态均作为明确消融。在线Teacher重排、非线性P、额外跳数、骨干微调不进入本轮主矩阵。

本轮仅研究Stage 1，不安排Stage 2 reader、FOCUS、补值、生成或最终join验证。使用既有recovery记录评价“证据是否支持正确实体、属性和值”，不把它解释成模型已生成正确值。可以成立的结论限于证据召回、确认支持覆盖和证据促成的目标发现。

R10提供了以下研究起点，均不能预先认定原因已经查明：

| 观察 | R11待检验问题 |
| --- | --- |
| edge扩池两轮分别引入919/888次已知正例误作负例，涉及约2%的训练lists | 修正后是否缓解退化？影响是否足够大？ |
| 实际checkpoint选择未执行RowSupport/Recall破同分 | 按声明的规则选择后，哪些历史结论会改变？ |
| 少量test构造负边进入train/dev | 怎样确保所有训练标签和模型祖先来源可审计？ |
| E01候选池只保有14/646个隐式正对的正确路径，Raw为213/646 | 主要损失发生在Q→E、E→T，还是路径组合？ |
| p_frozen正确路径159对到达池，124对保留进top4，F2最终45对 | 表示、证据保留与目标准入各损失多少？ |
| p_frozen/F2只有3/646对达到至少3行确认覆盖 | 能否学到多行互补，而不只是至少一条强路径？ |
| Raw图像独有支持桶从29对到达池降至16对保留进top4 | 图像证据是否因分数尺度、重复文本或覆盖估计被淘汰？ |

本轮成功不以“最高总体Recall”为唯一判据。主结论需要可训练模型、正确证据覆盖、实际目标准入及代价共同支持；负结果和方法间取舍完整保留。

**2. 数据协议与模型来源**

先在已完成R10审核的EntiTables v9快照上执行。权威输入为`output_mm_joinability_entitables_20000_retry100_rounds5_qwen35_final_survivor_context_gaussian_v9/`。WDC和数据扩建不作为本轮前置条件，也不静默并入比较。

沿用source-group划分：train-fit 11,390个query、train-calibration 1,240、dev 1,198、R10原test 1,166。校验指纹后可复用无标签的冻结encoder特征、row embeddings、PCA基；序列化、模型版本和输入不变时不重算骨干。共享数据湖的未标注对象可被索引和挖掘，必须披露这是同湖的transductive检索设置。

训练入口只接收物化的train-fit监督文件。不得通过读取全量qrels/recoveries再过滤的方式生成训练监督；构造正邻居、corruption、hard negatives及Teacher目标均先限定监督来源。无标签语料的可见性不意味着dev/test监督可见。开发评估可读取dev标签，评估结果不得反向改写训练标签。

train-calibration按source group用seed13确定性拆为`cal_fit`和`cal_check`，各约一半；记录实际query和支持对数量。前者拟合分数映射，后者评价校准。P/R、Teacher和新增行支持预测器的监督训练只用train-fit；不得用cal_check反复选择映射。置信度训练样本不足的关系保留未校准标记，不用只有正例的数据拟合可解释为全候选概率的变换。

**历史诊断线与主实验分别记录。**

| 类别 | 初始化与Teacher | 可回答的问题 |
| --- | --- | --- |
| 历史诊断 | 固定同一个R10 C4 checkpoint及R10 T0 Teacher输出；注明其祖先链未被重新净化 | 在这个固定系统上，修正某一实现会产生什么变化？ |
| R11主实验 | Student从无监督PCA-1024、R=I初始化；Teacher从随机任务头出发，仅train-fit监督训练 | 在来源合规的新训练链中，方法能否学习有效证据关系？ |
| 历史参考 | Raw、R10 PCA、r5-repro、C4、p_frozen | 纵向比较与失败定位；历史Teacher迁移不冒充fresh训练 |

新主线不复用R10 hard pool的监督和Teacher logits。新候选ID应按train-fit协议重新产生；使用旧权重的历史诊断也必须让比较两臂共享同一份清理后标签。修复脚本不意味着旧权重吸收的信息已经消失。

R10 test已经参与错误分析并影响R11设计，因此之后称为`r10_test_regression`，保留原文件和ID，结果属于适应性研究的历史回归。不得通过重切它、改seed或更名恢复“盲测”声明。最终独立确认需要真正未查看、未进入训练祖先监督的source groups，且在最终规则冻结前封存。若本轮没有这类数据，完成dev研究和历史回归，结论注明“尚无独立确认”；不因此停止全部Stage 1工作，也不擅自启动新数据构建。

**3. 固定检索预算、指标与选择规则**

默认预算：`Q→T=100`、`Q→text=20`、`Q→image=20`、每条`E→T=20`，`B=4`，`K={10,20,50}`。保留完整路径池后再做聚合和融合。同一模型的比较共享同一池；比较不同模型候选能力时必须各建自身索引。索引参数、插入顺序和随机种子固定，记录近似检索的不确定性。带名额保留的融合按每个K独立分配，不能从K50结果截取K10。

对一个隐式正对`(q,t)`，记全部query行数为`m_q`，GT确认支持行集合为`G(q,t)`；所选证据共同确认支持的行集合为`S(q,t)`。支持必须属于同一目标对应的正确属性，不能将不同属性的行拼成覆盖。未标注证据不作为确认无效；所有以下覆盖都是确认标签上的下界。

| 指标 | 固定定义 |
| --- | --- |
| R@K | 每query正target召回率的宏平均；同时报告implicit/explicit、单/多正目标 |
| ValidPool | 隐式正对中，正确evidence的完整Q→E→T路径已进入池的比例 |
| ValidB | 同上，但至少一条正确evidence保留在该目标自身top-B；此时不做target top-K限制 |
| RowB | 对所有隐式正对平均`|S(q,t)|/m_q`，只施加B限制，不施加target K限制 |
| ValidPath@K,B | target入top-K且保留证据含至少一条正确路径的隐式正对比例 |
| RowSupport@K,B | 对所有隐式正对平均`1[t∈topK]·|S(q,t)|/m_q` |
| MultiRow2 / MultiRow3 | target入top-K且确认支持至少2/3行的隐式正对比例；另给0/1/2/3/4/5行分布 |
| RecoverableRow | 用`|G(q,t)|`替代`m_q`的覆盖；与全行口径并列，不更换主分母 |
| ValidDiscovery@K,B | `t∉D100`且`t∈final topK`，并有保留的正确evidence路径；报告数量及占全部隐式正对比例 |
| Rescue / Displace | 相对同模型direct top-K新增/丢失的正(q,t)对；有效路径救回另计 |
| Evidence质量 | 确认路径精度、支持行和属性一致性；未验证率、确认样本来源和分母同时报告 |
| 成本 | 实际更新数、样本/候选数、Teacher前向量、索引时间/大小、ANN/补算/聚合耗时及P50/P95 |

ValidPool、ValidB、RowB不限制每个query的最终target数量，是诊断上界，不称为可保证达到的top10表现。ValidDiscovery另外报告以固定Raw D100之外正对为分母的分层比较，避免不同模型的D100变化造成误读。

训练短程探针使用固定步数端点比较，epoch0必保存。主线延长训练的checkpoint选择在dev执行明确的字典序：`ValidPool → RowB → ValidB → direct R@10 → 更早step`，目的是先保住可用证据池，减少融合对训练诊断的干扰。此规则与R10不同，必须记录为R11新规则。

最终组合选择在固定K=10、B=4下进行：先要求总体和implicit R@10均不低于同模型、同候选评分范围的direct-only对照超过**2个百分点**；合格组合按`RowSupport → MultiRow2 → ValidPath → R@10 → 更低实测成本 → 固定配置ID`选择。2个百分点是预先约定的质量容忍值，不是统计显著性阈值。超出容忍但机制指标提高的组合仍在“机制/质量取舍”表报告，不作为全面改进方案。

离散有效对数和支持行数优先用整数分子比较；小数比较固定绝对容差`1e-12`，不按显示的四舍五入结果判平分。输出每级比较值和选择原因，并用合成例验证各级破同分。没有模型通过最终质量约束时，结论是“本轮未得到满足该取舍的主候选”，不自动改成冻结P或抑制evidence。

**4. Task A：修复实现与监督来源，建立可审计基线**

这是后续训练的必需前置，先完成代码和CPU检查。

- 全局正邻居：在train-fit范围为`(source_id, source_type, destination_type)`建立完整已知正集合。edge扩池排除集合中的额外正邻居；原列表中合法正邻居全部正确标正。监督与KD使用各自实际候选mask，不把Teacher只打过分的局部列表扩成伪Teacher目标。
- 解耦开关：`in_batch_negatives`只控制扩池；supervised edge ranking、KD、BCE各有明确权重。关闭扩池时仍可在原列表算同一ranking loss。旧行为只留作历史复现，不能静默改变历史实验解释。
- split来源：按train-fit自身qrels/recoveries及corruption出处生成训练标签。`Q,T`非正不能自动推出`E,T`确认负；无足够证据的corruption标为弱负/unknown，只有有独立冲突依据的边进入BCE。原构造弱负可进入明确标注的ranking采样，不能伪装成确定真值。
- 选择器：补齐RowB/RowSupport等指标，实现上述两套显式选择顺序；同时支持重放R10实际单指标选择作为历史对照，不能修改旧checkpoint来隐藏差异。
- 打分配置：区分基础关系分数、ranking loss输入变换、KD温度、BCE logit、两跳组合、跨路径聚合、推理score space。只增加实验确需的配置字段；所有缓存校验这些字段。
- 几何诊断：保存阶段起点和整链PCA参照。checkpoint重载不能使“相对最初PCA漂移”悄悄重置；参数漂移与功能漂移分开记录。

必要测试覆盖：跨list多正例不变负、原列表多正例mask、no-inbatch仍保留ranking、unknown不进BCE、改变不可见split标签不改变train-fit输出、各级破同分、同checkpoint训练/推理打分一致、F2无法准入D100外目标的边界、F5按K独立预算。使用小型合成文件，不读真实秘密配置。

基线重建：Raw、PCA初始化与历史参考在同一dev协议下导出完整池，按Q→E、E→T、完整路径、B保留、target K准入分解损失。增加独立于融合的Q→E正确证据召回和`E已正确召回`条件下的E→T正目标召回。

固定128个dev query作为exact-search诊断样本（implicit/explicit各64，按source与固定seed13抽取）；另固定最多128个有标签evidence source，覆盖text/image。对Q→T、Q→E、E→T比较ANN与exact inner product的命中与排名。该检查分离索引误差，不用oracle注入增加正式召回。

验收：已知训练正例误作扩池负例为0；训练监督无dev/test来源；配置/选择行为测试通过；基线逐query分子可重算。产出`taskA_protocol/{inputs,splits,label_provenance,selection_spec,baseline_funnel}.json`及对应结果说明。

**5. Task B：历史受控复测，量化修复的影响**

假设B：R10 edge训练中的错误负例是部分退化来源；也允许结果表明影响很小。

只运行两条历史诊断臂，不做大范围历史网格：

| 臂 | 扩池正例处理 | 其余条件 |
| --- | --- | --- |
| B0 | 复现当前list局部正例mask | 共享清理后train-fit标签、R10 C4初始化、R10 T0 Teacher及固定目标分布 |
| B1 | train-fit完整正邻居mask | 与B0相同 |

采用R10 E-T0的confidence训练设置、seed13、batch64、cap256。两个臂均保留相同的supervised ranking、KD和无BCE配置。执行2个edge epoch：若列表数仍为42,143，则每轮659个更新、合计1,318；列表数变化时以实际相同样本预算登记，不伪称精确历史复现。edge起点、每178步附近和epoch末评价；两臂再各运行相同178步path续训，查看修复影响能否保留。

固定候选基础池及样本顺序。修复导致可用扩池候选变化是本干预的一部分；随机抽样按`seed/epoch/step/list ID`确定，避免过滤改变随机数消耗后连带改变后续batch顺序。记录被排除数、实际负例数，不额外补采一批不同难度负例来混淆比较。

主要比较固定端点的ValidPool、Q→E/E→T、ValidB、RowB与P/R功能漂移；开发集选择结果另列。错误计数归零但性能未恢复时，记录“修复必要但不足”，继续Task C。B0/B1均属于历史条件下的因果诊断，不能作为来源干净的新主线。

**6. Task C：干净Teacher与排序尺度探针**

先建立一个干净token Teacher：沿用骨干缓存，任务头随机初始化，只用Task A的train-fit列表，raw logits、edge warm-up后path训练。固定hidden=512、3层、8个attention heads、text/image latents=16/24、dropout=0.1、batch=8、AdamW学习率`5e-5`、weight decay=0.01。第一轮上限为2个edge epoch加2个path epoch。edge checkpoint按`五种关系各自list R@1的等权平均 → 更低dev ranking loss → 更早epoch`选择；path按`evidence通道target-list宏平均R@1 → direct通道对应R@1 → 更低dev loss → 更早epoch`选择。候选列表在训练前冻结，保存每阶段固定端点。记录真实算力，不能只用Student步数声称等预算。

Teacher质量同时报告Q→E、E→T和属性一致路径判别；没有正负标签的关系不报AUROC。Teacher若只在简单构造负例上好，不宣称细粒度joinability已验证。若其固定dev候选上的evidence target R@1不优于Raw，C仍可作为优化诊断执行，但不把负蒸馏结果归因于Student压缩失败，也不启动额外Teacher结构长训。该Teacher固定后才比较以下Student探针，不能边换Teacher边比较分数空间。

假设C：概率压缩和梯度尺度不匹配是表示损伤的一部分；应通过单因素比较验证。

从相同PCA初始化出发，先运行3个edge-only探针，每臂固定178步，batch64，P学习率`1e-6`、R学习率`1e-5`、ranking权重1、KD权重0.3、KD温度1、BCE权重0。Student维度1024、full R、AdamW weight decay=0.01、anchor权重0.1（P与各关系同权），多正例loss固定`sum_probability`以避免同时更换监督公式；它不保证每个正邻居均匀学好，因此另报多正例召回。主线不开hard refresh或连续edge辅助项。后续raw path目标固定`s(Q,E)+s(E,T)`、全池LSE温度1、target温度1、Direct/Evidence CE同权；其余设置逐项写入配置，不在臂间变动。

| 臂 | supervised ranking的输入 | KD与推理 |
| --- | --- | --- |
| C0 | 原始关系logit `s` | 固定raw Teacher目标，Student KD也使用raw logits；推理使用raw关系分数 |
| C1 | `sigmoid(s)` | 同C0 |
| C2 | `sigmoid(s)/0.1` | 同C0 |

这组只测supervised ranking压缩与温度，**不是R10 E完整配方复现**；R10 Teacher/Student都转confidence的影响已保留在历史诊断中。三个臂不同时修改path组合、聚合、BCE或Teacher目标。

训练探针在step0/45/89/178记录：每个有序关系的logit范围、sigmoid落在`<0.01`和`>0.99`的比例、ranking/KD分项loss、固定小batch上的分项P/R梯度范数与夹角、固定对象的投影方向变化及近邻保持率。避免只根据参数均方漂移很小或loss下降判断几何被保护。

三个探针都保留固定178步结果。在178步ValidPool不低于自身起点80%的方案中，最多选两种排序方案，从同一干净起点分别运行完整2个edge epoch，再用完全相同的raw-score path目标续训356步；先验证edge阶段修复能否经path训练保留，不将C1的sigmoid压缩顺带传播进证据聚合。入围不足两种时不凑数。

另从同一PCA起点运行178步的raw-score **path-only** 对照，用于定位edge预热是否必要。它与edge+path算力不同，报告为机制消融，不宣称等算力优势。

若后续需要单独验证direct path的压缩问题，只在同一个path起点比较direct CE的`d`与`sigmoid(d)/0.1`，其余evidence CE、raw KD和推理不变；最多增加一组成对178步探针，登记为条件实验，不扩成全网格。

选择两种方案的顺序使用第3节的训练指标字典序。若所有方案的ValidPool都低于自身起点80%，先使用第10节唯一的学习率救援预算；仍无合格方案则保留PCA起点和path-only结果，继续做固定池证据实验，不强行延长。不能因此宣称Student函数类不足，也不自动把P冻结改为主方法。

**7. Task D：分离KD、P/R更新与校准，检验Teacher必要性**

在Task C确定的同一个干净配方上，做少量一次只改一个因素的对照：

| 臂 | 唯一变化 | 研究目的 |
| --- | --- | --- |
| D0 | Task C配方，P/R训练、KD开启 | 共享对照，不重复运行相同配置 |
| D1 | KD=0 | Teacher软分布是否改善正确关系，而不是仅改变loss？ |
| D2 | P冻结，R训练 | 定位投影更新的损伤；仅作消融 |
| D3 | 表示checkpoint固定，仅在cal_fit拟合关系仿射映射 | 边校准收益是否来自分数映射而非表示学习？ |

D1/D2先用与D0相同178步预算探针。确定一个最终可训练配方后，D1必补同样2个edge epoch和356个path更新以检验KD贡献；D2仅在短程ValidPool比D0高至少2个百分点时补同预算用于定位，不进入主候选选择。不同步数结果不放入同一端点效应表。

D3使用每关系`ell=a·s+b`、`a=softplus(alpha)+1e-6>0`的单调映射，先报告同关系ANN排名不变的校验，再报告路径排序和跨模态融合的变化。仅在cal_check有确认正负样本的关系报告Brier/NLL；只有正例的Q→E暂保留身份映射并标记未校准。不要以提高正例置信度代替区分错误证据。

可选一次BCE-on探针：仅对train-fit确认正、负各至少64个不同pair，且cal_check每类至少32个pair的关系启用；这是计算投入的样本下限，不是统计充分性的保证。在D0的raw ranking/KD保持不变的条件下增加BCE辅助项，权重固定0.1。与D3比较，分离“只校准”和“梯度进入P/R”。未满足标签条件时记录证据不足，不用unknown补成负例，也不把BCE跑完作为形式上的必需项。

Teacher必要性对照在选定Student目标后执行：token Teacher与pooled-object pair Teacher都从任务头随机初始化，使用相同训练列表、骨干输入、候选和固定更新预算，报告参数量及实际时耗。两者的主要差别为能否使用细粒度token交互，模型容量差异不能隐藏。先做各512个edge更新加256个path更新的成对探针，比较固定端点；只有token Teacher的dev evidence target R@1比pooled高至少2个百分点，且direct R@1下降不超过2个百分点时，才补同预算Student蒸馏比较。这是扩展触发规则，不是显著性结论。短程token探针不能换成已经长训的Teacher来比较。

**8. Task E：固定池下的证据保留与属性行覆盖**

假设E：正确证据到达池之后，按分数取top4可能重复覆盖同一行，或淘汰图像证据；按独立行支持选择可以提高覆盖。

先使用Raw及Task C/D中一个P/R可训练模型。固定完整池、每target前L=20个evidence候选和B=4。所有比较使用相同基础路径质量与同一分数映射协议；质量值只称强度，不在校准不足时称联合概率。固定两跳组合后再比较保留策略，不同时换min/product和LSE/LME。

| 臂 | 证据保留 | 属性条件 |
| --- | --- | --- |
| E0 | 按路径质量top4 | 无 |
| E1 | exact-content去重后top4 | 无 |
| E2 | 去重后按预测的新增行覆盖贪心选择 | row–evidence，复现G5思想 |
| E3，条件扩展 | E2加入目标可见schema条件 | 不提供GT桥接列、隐藏属性名或正确值 |

第一步固定target列表和target聚合分数，只改变保留证据，隔离B保留的影响。第二步才把预测覆盖分数用于target排序，沿用相同E候选，研究是否改善准入。两种结果分开，不能把重新检索带来的池变化当作保留策略收益。

可复用的覆盖目标为`coverage(E_set)=mean_i max_{e∈E_set}(quality_e·support_i,e)`；其目的是奖励新行，而非相同内容的重复票数。单条evidence可以支持多个不同query行；内容去重不能把这些合法支持删掉。记录内容重复、同一行重复和新行支持分别贡献多少。

E2的现有isotonic模型只回答“已知有用证据内部如何分配行”。R11先区分“证据是否支持任一行”与“支持哪一行”。每个模态只有在train-fit得到确认正/负QE各至少32个、合计覆盖至少16个source groups，且另有确认的行定位标签时，才训练对应支持预测器；在cal_fit拟合需要的分数映射，在cal_check报告外部表现。只有正QE或未恢复行时，不足以构成这些确认负标签。条件不满足则E2使用raw row–evidence cosine的无标签单调缩放作为定位强度，明确不称支持概率，不强行训练全候选判别头。E1/E2仍可在固定池上比较覆盖行为。

E3仅在E2的dev审计中确认至少10条实体相关但属性错误的路径、且来自至少5个source groups，并满足支持预测器的train-fit标签条件时执行；限一个轻量schema条件化支持模型。该数量是扩展触发条件，不是机制普遍性的证明。作为Stage 1聚合扩展单列成本和方法变化。输入只含query可见行、独立evidence和target可见schema，禁止GT列位置、recovery值及provenance。它不实现Stage 2列选择或值抽取。若采用列级内部候选，必须按同一个预测属性汇总不同query行，不能对每行各选不同属性后相加。

建立小型属性一致性审计：预先按source和模态抽样，目标预算256条候选路径，其中train-fit/dev各128条，每个split内text/image各64条；实际不足的桶按实数报告，不挪用dev样本凑训练标签。分别核查正确支持、同实体错误属性、同属性错误实体/值、重复同一行、互补新行。train-fit样本可转训练，dev样本只评估；盲于模型臂做独立核验，不能把Teacher高分自标为正确。没有完备依据的样本仍标unknown，报告覆盖率；本计划不自动调用外部标注API。

假设压力测试包括重复同一evidence、重复相同内容、同一行的多条证据、不同新行的证据，以及局部两条边都强但属性不一致。合成性质与真实任务收益分开报告。

阶段结果优先报告ValidB、RowB、MultiRow2/3、已标注可恢复行覆盖、确认错误属性率及text/image保留率。若只降低path-count correlation却损失正确行覆盖，不判为研究成功。

**9. Task F：证据促成的目标准入与多模态干预**

必须设置两个direct基线：D100内部的direct-only用于历史比较；对D/E并集全部补算相同direct分数的union-direct用于严格融合归因。F0/F2保留R10原D100语义，仅作历史参考；F1/F3/F4/F5共享并集及完整union-direct分数/排名、B、K，主归因在后四者中比较，并记录额外补算成本。F3/F5因此是R11的并集版本，不冒充原R10同名配置。

最小融合矩阵如下，不追加密集权重扫描：

| 臂 | 规则 | 用途 |
| --- | --- | --- |
| F0 | D100 direct-only | 历史基线 |
| F1 | union-direct-only | 相同评分覆盖的主对照 |
| F2 | R10 weighted RRF，D只排名原D100，D=1/E=.05 | 历史机制限制对照；D池满100时ValidDiscovery应为0 |
| F3 | D/E等权RRF | 无新增拟合的准入对照 |
| F4 | 共享映射下的direct/evidence线性融合，λ∈{.25,.5} | 最多两个dev候选；λ=0即F1 |
| F5 | 每个K预留一半evidence名额，去重后补齐 | 名额准入对照，按K独立计算 |

线性融合前各通道尺度及映射只由cal_fit固定，不能按每个测试query的GT归一化。默认对各通道使用`z=(score−q50)/max(q90−q10,1e-6)`的全局单调仿射缩放；分位数在该模型的cal_fit共同池上确定，所有融合臂共用，不逐臂重拟合。无evidence的目标，其E项固定为0并报告缺失比例。F1与F4的λ=0使用同一个D分数和破同分规则，保证排名完全一致。未经外部校准的强度不称概率。按第3节规则在dev选一个主候选，同时保留一个机制提高但质量下降的取舍点；后者不替代主候选。

ValidDiscovery要求D100之外、最终入选且具有正确属性路径，不能只要求`E@10有、D@10无`。同时测相对F1仍新增的正确目标，排除“扩大direct评分范围就能找回”的解释；保留原有正目标被挤出的完整统计。

多模态检验有两种互补设计：

- 全流程等总预算：text-only取40条E、image-only取40条E、text+image各20条E，每E→T均20；direct、最终K与B相同。统计去重后的实际路径数；不因mixed多一倍搜索预算而宣称模态互补。
- 固定池干预：对同一个池移除图像、用重复同一行证据替换图像、用已确认同实体错误属性证据替换；另保留原始混合池。替换尽量匹配模态、候选数量和原始分数区间，重新运行相同的保留与融合规则。报告配对支持变化以及系统分数/准入是否响应，不能把“删掉正确标签自然少覆盖”本身当作模型识别机制的证明。

模态GT桶只能用于评估分层，不能指导在线分配预算或选择证据。固定target队列的干预只解释证据保留质量；重新计算target准入的干预才解释Stage 1发现行为，二者分别报告。

**10. 执行顺序、规模与停止条件**

执行依赖为：`A → B历史诊断`，`A → C干净主线 → D → F`；E的Raw固定池实验在A完成后即可并行，随后补一个稳定可训练模型。F需使用已冻结的E保留策略。Teacher结构探针在学生目标稳定后运行，不阻塞Raw上的覆盖研究。

第一轮只用seed13筛选，禁止直接把所有维度做笛卡尔积。默认工作量为：

| 工作 | 上限或固定预算 |
| --- | --- |
| A实现与诊断 | CPU回归、基线池、128Q/最多128E exact检查，无模型训练 |
| B历史mask诊断 | 2臂，各2个edge epoch + 178步path |
| C干净token Teacher | 1条，最多2个edge epoch + 2个path epoch |
| C排序初筛 | 3个edge臂各178步；1个path-only臂178步 |
| C延长 | 最多2个配方，各2个edge epoch + 356步path；短程前缀可安全续用时不重复计费 |
| D消融 | KD-off、P-frozen各1个178步探针；必要的成对长程对照另登记 |
| E/F固定池 | 最多2个模型、4种保留策略、上述固定融合规则，不重复导出同池 |
| 条件扩展 | 最多1组path尺度成对探针、1个BCE探针、1个schema支持模型、1组Teacher结构成对探针 |

训练侧共享初始化、数据顺序、候选、batch size、优化器和实际更新数，端点对比与selected checkpoint各一张表。样本数变化时重新明确步数预算，不能一边重建列表一边仍声称完全复现旧659步。记录预处理/Teacher打分/索引开销，不事先编造GPU小时。

正常有限loss的178步探针按固定预算跑完；NaN/Inf、错误标签、缺失必要特征、打分空间不匹配等正确性失败立即停该run并记为实现失败。长程训练每个固定检查点记录证据池覆盖；若连续两个检查点ValidPool低于自身起点的80%，停止该臂的延长，保留最后端点和最佳checkpoint。这个阈值只控制计算浪费，不作为函数类被证伪的标准。

提前终止的臂不能与跑满的臂作等步数端点效应比较，只能比较共同已完成步数；最终质量差仍完整报告。最多允许一个事先登记的降低学习率救援配方，P/R学习率一起乘0.1，其他项不变；失败则收尾该设置，不能连续开新网格直到出现正结果。

**11. 复现、统计与主结论标准**

在dev冻结最多两个P/R可训练配方及一个明确消融后，补Student seeds17/23。Teacher、PCA或mining pool固定时明确称为条件Student方差；不以三个很接近的数值宣称整链稳定。若需要“完整Teacher–Student方法稳定”的结论，另补独立Teacher训练种子及其Student链；未做就限定措辞。

预先固定三组主比较：B1−B0以两个edge epoch末的ValidPool为主指标；E2−E1以同一个冻结模型和池上的RowB为主指标；dev选定最终融合−F1以ValidDiscovery@10,4为主指标，同时执行质量约束。端点和配置冻结后报告差值、分子、样本量及source-group配对bootstrap 95%区间（10,000次，seed13）。多正目标的路径指标按正对数聚合，不把query路径均值再次无权平均。

dev扫描区间只作探索性描述。若在独立确认集作多重主张，双侧p值来自source-group内整组交换两个系统输出的配对随机化检验，10,000次、seed13，以`(1+至少同样极端的次数)/10001`计算；对预声明的主比较使用Holm校正。bootstrap区间用于呈现效应大小，不从区间倒推出未经定义的p值。历史诊断不能通过加入显著性检验升级为干净主线结论。

只有同时满足以下证据时，才把组合描述为本轮支持的Stage 1方法改进：

- P/R确实接受有效训练更新，且相对干净初始化/同预算对照保留或改善正确路径池覆盖；选中阶段起点时不声称该阶段训练带来增益。
- 相对同池对照，提高正确属性证据的行覆盖或多行支持；改进不只来自增加重复路径。
- 相对union-direct出现有正确证据支持的新增目标，或明确证明证据提高了既有正确目标的支持充分性；这两种结论分别表述。
- 文本/图像的贡献有等预算或固定池干预支撑，同时报告rescued/displaced、implicit/explicit质量与成本。

没有独立确认集时，这些结论仍只能作为dev及历史回归支持的研究结果。任何Stage 1成功都不替代真实值恢复或最终joinability验证。

**12. 实现落点与交付清单**

主要实现落点：[scoring.py](/home/oycy/MMDD/src/mmdd_stage1/scoring.py)、[training.py](/home/oycy/MMDD/src/mmdd_stage1/training.py)、[selection.py](/home/oycy/MMDD/src/mmdd_stage1/selection.py)、[objectives.py](/home/oycy/MMDD/src/mmdd_stage1/objectives.py)、[row_support.py](/home/oycy/MMDD/src/mmdd_stage1/row_support.py)、[retrieval.py](/home/oycy/MMDD/src/mmdd_stage1/retrieval.py)、[train_stage1.py](/home/oycy/MMDD/src/train_stage1.py)。训练标签物化优先复用并修正现有Stage 1逻辑；如需新增R11入口，放在`src/`，不继续在`work/`写实验算法。

实施时先运行最相关的测试，再在修改训练/检索公共逻辑后运行Stage 1回归集合。所有测试从隔离`/tmp`工作目录、合成配置运行，使用MMDD环境；不可隐式加载`.env.openai`。计划中的命令和配置应由实际`--help`核实后保存，不在本文件把尚未实现的参数写成可直接运行的命令。

最终产物至少包含：

- `PLAN_FROZEN.json`：配置ID、假设、固定预算、选择顺序、种子、扩展触发条件和代码版本。
- `taskA_protocol/`：输入指纹、监督来源、模型祖先、split用途、baseline与exact诊断。
- `runs.jsonl`：每个臂的实际命令、参数、起点、Teacher、候选和缓存签名、步数、耗时、完成/停止原因。
- `metrics_per_query.jsonl`与路径诊断：目标召回、正确路径、行/属性覆盖、模态桶、准入归因及共同端点。
- 各任务`RESULTS.md`：该假设得到支持、不支持或证据不足；包含负结果及混杂限制。
- `FINAL.md`与机器摘要：历史复测、新主线、dev选择、旧test回归、独立确认（若有）分别列出；已有R10产物完整保留。

本计划的执行终点是完成上述受控判断并诚实报告结果。若修复后仍退化，应得到可定位的负结果；若只改善边校准而未改善正确属性证据，也应明确判为机制尚未成立。
