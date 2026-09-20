# MMDD FINAL_RERANK之后：三个有前置门槛的机制实验

## 总原则

先判断真实witness在哪里，再决定是否训练。以下不是同时启动的大矩阵；实验1立即做，实验2只在相应retention阶段确实丢witness时做，实验3只在verified witness已经retained而当前打分仍失败时做。

维持在线对照：冻结健康Student产生候选，固定production Equal C100，冻结原T0作QT-full/QT-on-P。Historical-B13只有一个reference；Healthy-B4与F-P分别保留两个真实seed的身份。开发过程仍报告overall、implicit、explicit，R@10主端点，R@20/50与strict funnel为机制指标。

原始dev集已多次用于选方案，不把它包装为新held-out确认。下一轮必须单独标明训练、开发、未见测试source group。原始raw Qwen基线由已有baseline工作流提供；缺少其ranking时标NA，不伪造本次复算数字。

## 实验1：补齐witness与真实edge质量审计——0 training、0 retrieval、0 ranking change

**Local fact**：本轮B13的736个C100正target中，114个无retained path，622个有path但witness未知；fixed207入池的101个全有path。现有“correct path”标签只是target为正，并未验证evidence。

**Hypothesis**：当前Path较弱可能因为输入bag缺少真实witness，而非仅仅模型不会利用witness。

**Competing explanation**：真实witness已经保留，但QE/ET打分、跨edge零点/尺度或target-bag目标缺失造成排序失败。

**Single-factor intervention**：只添加离线标签和pre-retention溯源，不改当前C100、retained paths、任何score或ranking。恢复原始own retrieval中每个target的全部已生成path、dedup后顺序、top20后集合、budget4后集合；连接已有离线annotation及原始Q/E/T内容。先覆盖B13全部101个已admit fixed207目标和它们的retained/preretained paths；同时验证主导Top10非GT竞争者的evidence，以避免只看正target的选择偏差。没有现成标签时，人工阅读时先隐藏模型分数与排名。

输入标签至少包含 `query_id,target_id,evidence_id,modality,witness_label,annotation_scope,annotation_source,completeness`。`witness_label`只能是verified_positive / verified_negative / unknown。验证的是支持该(q,t) join机制的三元witness；不能只因Q-E和E-T分别看似相关就默认是同一个有效桥接属性。可用已有独立edge annotation做补充，但与triadic witness真值分开。

**Main metric**：verified-witness-retention rate；分别以“至少有一个verified witness的pre-retention bag”为分母，以及以全部正target为分母报告；同时报告unknown覆盖率。本实验官方R@10/20/50必须与本轮逐query完全相同。

**Mechanism diagnostic**：逐target分类：无retained path；retained但annotation不完整；穷尽标注后无正确witness；verified witness在dedup/top20/budget4哪一步丢失；verified retained但rank>K；verified retained且rank≤K。再在有edge真值的候选列表中分别评估QE/ET；必须报告候选覆盖与unknown，不能把不完整标注下的所有竞争者称为负例。对verified retained目标观察正确witness的QE、ET、sum、LSE贡献和与相同模态竞争者的margin。

**Negative result means**：若大多数失败目标已有verified retained witness，retention不足解释被削弱，应进入实验3；若witness标签覆盖仍低，只能报告unknown，不能据此声称retention没有问题。若正确witness根本不在旧pre-retention bag，问题在更早检索/候选生成，本轮不再用final-ranking训练掩盖它。

交付：`WITNESS_DIAGNOSTIC.jsonl.gz`、各stage的path-membership hash、标签来源和覆盖表；绝不以GT替换路径，也不重排官方结果。

## 实验2：有证据才取消Student top20预筛——0 training、0 new retrieval

**Local fact**：生产retention先按既有path质量排序和content dedup，再topL20，然后Q-row greedy coverage选budget4。包中没有pre-retention或witness标注，因此现在还不知道top20是否实际裁掉真实witness。

**启动门槛**：实验1证实某些正确witness存在于旧pre-retention、经content dedup仍在、但在top20之外。若主要损失发生于更早retrieval或之后的greedy选择，不执行这个实验；不得因为文档列出了它就机械跑一遍。

**Hypothesis**：Student path质量预筛在Teacher重排之前就删除了有用witness；后续任何scorer只能处理剩余错误路径。

**Competing explanation**：扩宽预筛后greedy仍不会保留witness，或Teacher即使看到它仍然排序错误；也可能旧Student预筛在有效去噪，取消后反而更差。

**Single-factor intervention**：保持原始C100、已生成pre-retention paths、content dedup定义、Student pair分数、Q-row支持值、greedy算法及最终budget4不变；只把topL=20改成“该target全部已有dedup后paths”。不要重新retrieval，不加入未出现过的evidence，不重建E target ranking后重新admission。原路径数不足4时不补造；记录每个bag实际返回数，两臂支持集不同的pair单独报告。GT只用于诊断，不参与挑选。

如果多出的旧path没有Teacher分数，可用同一个冻结T0补算真实QE/ET并记录feature覆盖与pair数；这不是新retrieval，也不允许Student proxy。原C100哈希必须逐query不变。

**Main metric**：fixed C100下Teacher-LSE的query-macro R@10变化，分overall/implicit/explicit；同时在两臂共有支持集上报告排序变化，避免混合support和scoring。原QT-full和原QT-on-P只作锁定参考，另有新支持集时以明确新名字报告，不覆盖旧指标。

**Mechanism diagnostic**：top20之外verified witness被最终budget4选中的比例；原有正确witness被挤掉比例；最终真实witness总保留率；每query rescues/drops；新增路径的模态、Q-row覆盖及高分竞争者变化。先验证输入集合确实发生预期改变，再解释负结果。

**Negative result means**：若witness仍未进入budget4，则top20预筛不是足够解释，下一问题在greedy/coverage选择；若witness显著增加但R@10无益，则scorer/组合更可疑；若扩大后只引入噪声，保持原top20，不继续盲调更多L/budget网格。本实验不支持“所有retention方案都无效”的泛化结论。

## 实验3：只补target-level path监督，不同时换Teacher结构

**Local fact**：同一P中B13 QT R@10=42.01%、Teacher-LSE=8.89%；MAX/LME及局部softmax未修复。可见T1-B训练实现采用edge-listwise目标，没有最终target-path-bag目标。不过当前包没有该checkpoint的完整训练receipt，需补验。

**启动门槛**：实验1有足够verified-retained失败样本，且冻结Teacher forward/cache来源已通过小批真实QE/ET复算。训练数据必须有与dev/test按source group隔离的既有Student候选与path bags。缺这些输入时先补输入，不允许把本轮1198个dev query拿来训练。

**Hypothesis**：同source edge列表内部的正确排序，不保证跨evidence的logit零点、QE/ET相对尺度和LSE target排序正确；增加target-bag目标能使当前pair scorer学会服务最终target决策。

**Competing explanation**：即使真实witness存在，两次独立pair交互仍无法表达必要的Q-E-T一致性；或证据不含QT之外信息、正例标签稀疏、bag噪声过大。另一个控制解释是“只是继续训练了”，必须由等步数edge-only continuation排除。

### Parent与冻结对象

两臂均从同一个实际parent读取权重与优化器状态：

`fresh_lineage/T1-B/seed13/checkpoints/step_010536.pt`

SHA256：`ab0e3c3f85f006d2fdc4ba5194a0021680ab8fa1341441cb8eb003410ded68cc`。

读取前验证hash；不得换成较早A1/C3，也不得换成新随机Teacher。冻结Qwen特征、健康B13 Student、候选集合和路径选择。原T0 QT score始终冻结为评测控制；即使训练后Teacher的QT分数变化，也不能替换控制。

### Training candidates / bags / positives

从健康B13在**train split**已有own retrieval构造production Equal C100；path bags采用同一D1 top20/budget4产物。缓存不存在时，先由冻结B13一次性生成并锁定训练集候选，不得用待训练Teacher选择候选、挖新负例或做动态refresh。数据准备需单独计数，不能称为完全0 retrieval。

仅对 `G_train(q)∩P_train(q)` 非空的query计算target-bag监督；无admitted正例的query计为无可用bag，不注入GT target或oracle path。正例为GT target，**不是它的每条evidence path都被标成正edge**。其他已知正target全部纳入multi-positive mask，未知target保持既有unknown策略并明确false-negative风险。离线verified witness标签只诊断，首次试验不新增witness-edge loss，避免同时改两个监督因素。

固定bag最终分数仍为：

`Sθ(q,t) = logsumexp_{e∈B(q,t)} [sθ(q,e)+sθ(e,t)]`。

不加QT、不换aggregator、不换triadic Transformer、不加learned fusion、不改路径数。

### Single-factor intervention：两个等步数续训臂

A：`L_A=L_edge`，原Teacher edge-listwise continuation。

B：`L_B=L_edge + 1.0*L_target_path`。

其中

`L_target_path(q)=logsumexp_{t∈P_train(q)} Sθ(q,t) - logsumexp_{t∈G_train(q)∩P_train(q)} Sθ(q,t)`。

每个logical update两臂使用完全相同的原edge batch；B额外对预锁定query-bag batch反传目标loss。两臂采用同parent optimizer state、LR/weight decay、logical batch顺序、随机seed和update数。edge loss按原logical-batch定义平均，target loss按有效query平均。lambda=1为预先声明pilot值，不以dev搜索网格。训练预算为完整一遍预锁定有效train query-bags；记录实际query数、step数与edge-list暴露量，而不是任意200步并称为充分训练。若数值不稳定，报告失败；不要中途只对B调LR后继续称为单因素。

backprop必须穿过真实Teacher pair forward；不得把旧Teacher score cache当作可训练tensor。可复用冻结对象特征，不能跨参数更新复用失效的pair logits。用于内存分块的累积不能改变每query的LSE分母或正例mask。

首次只做一个训练随机seed的机制pilot，保留A/B两臂；有信号再复验训练seed和B4候选。原有两个Student endpoint seed不是这个训练pilot的两个seed。

**Main metric**：未见source-group上的同支持P query-macro R@10，核心对比B−A；并报告B−冻结Teacher-LSE及B−QT-on-P。overall与implicit必须同时报告，explicit作保护项；R@20/50和历史strict207只作相应split的机制分析，dev207不能充当held-out。

**Mechanism diagnostic**：verified-retained子集、无verified标签子集分别的收益；QE/ET单边质量变化；跨evidence logit offset/尺度变化；bag内witness贡献；每query正确target margin和rescued/dropped；相近QT分数区间内新增path score是否仍能区分正target。后者是条件判别诊断，不单独证明因果信息价值。若B仍显著弱于QT-on-P，不进行主方法替换或立刻开始fusion搜索。

**Negative result means**：B不优于等步数A，会削弱“简单增加target-level监督就能修好当前pair-sum架构”的解释；不能据此排除真正的三元交互Teacher、其他有效监督或不同witness质量。若只在训练/dev增益，则没有泛化证据。若B优于A但仍远低于QT，说明监督有作用，但仍不证明Path适合取代QT。

### 如何避免Student/Teacher循环依赖

Student在整个实验固定；候选与path membership训练前锁定；Teacher不改变candidate admission、路径选择或negative mining；本轮不蒸馏新Student。因而目标分数变化只来自Teacher监督干预，而不是候选分布同时改变。

## 不在本轮启动的方向

不重复已经失败的query-conditioned residual recipe；不再把MAX/LME/当前局部归一化当作未跑新实验；不同时试大量path budget、温度、模态权重和fusion权重。Student path learning须等待Teacher/监督目标有可验证收益；QT+Path fusion须等待在未见source上出现可重复的QT条件增量信号，而不只是几个rescue案例。

若完成上述诊断后仍是“QT-on-P识别绝大多数正确target，而任何合理且受监督的Path分数都无额外泛化收益”，允许将Evidence明确定位为candidate admission工具。现有方法的价值无需依赖“Evidence也必须主导final ranking”这个更强命题。
