# 下一步修复与重跑合同（建议交给执行代理）

输入依据：用户上传的 mmdd_s1_audit_20260918.tar.gz，以及本目录的独立复算和合成探针。
目的：先恢复 CLEAN-R1 的实际语义，不增加新架构、不调参、不把错误run结果覆盖掉。

## 0. 立即状态

暂停Teacher追加epoch及SUP/KD。保留现有全部产物为异常原run；不要删除或替换best.pt、原排名和报告。所有修改先在独立分支/目录进行。此文不是要求马上启动完整训练。

## 1. Teacher自然B接线

1. TeacherTrainer获得不可变的、本轮raw train query_rankings映射。
2. 每次PacketRequest和同步builder调用均传入当前query对应raw排名，不允许q_rank=None。
3. 当include_bundle=True时缺raw排名直接报错。合法raw列表确实为空与未提供排名必须是两种状态。
4. natural B严格由冻结raw text10/image10合并；难例刷新只改变HardRank，不改变raw bundle。
5. augmented只按原规范替换/追加该epoch选中witness，所有候选T共享同一个B。
6. 每epoch导出B长度直方图（0/1/2..20）、natural/augmented计数、原始B与实际B的ID摘要，不能只导出一个context_size平均值。

## 2. 预取必须保留空集合语义

1. `_expand_packets` 判断应为override is not None，而不是bool(override)。显式[]必须恢复成packet['positive_ids']=[]。
2. compact/expand保留excluded_ids、view、context与全部采样provenance，不能只保留ID列表。
3. 对B正集为空的样本，teacher_rank_term返回None或按现有约定的可微零，且不计入有效packet分母。masked positives既不为positive也不为negative。
4. 分别比较直接builder、workers=1预取、workers=4预取的完整packet；测试覆盖override缺失、None、[]、非空，natural/augmented、direct/implicit。
5. 测试必须对比mask、有效loss、skip、梯度，而不只是candidate IDs。严格等价性能优化不能修改上述任一项。

## 3. 难例刷新恢复原合同

1. Teacher D刷新池：本轮raw QT128与本轮raw evidence-target ranking前128的自然并集。不是rawQT128和自身的并集。
2. 明确预计算自然E的raw第二跳列表，保存L_E/R_E/U及来源，不能使用GT witness二跳列表替代自然E列表。
3. Teacher E刷新：每模态raw QE128的并集，在当前Teacher的P(Q,E)同一任务上重新评分，过滤W_Q后按同一scorer排序。不得先截断合并流后假装对完整union重排。
4. Teacher C刷新：仅为epoch3–6将被witness-cycle使用的(q,e)，取raw ET128∪rawQT128，在当前Teacher J(Q,T;{e})上评分；按(q,e)存储与读取，不能按e单独覆盖。排除任务正例/其他known positive。
5. epoch1–2 C保持raw ET列表作为初始HardRank，不把不同anchor的raw分数或Teacher P分数直接混合后全局排序。
6. 对有定义的名次交替流（初始E hard流），MakeList只按流顺序过滤/去重/取前16；对由同一scorer评分出的HardRank，预先规范为(score desc, ID asc)。后者不能再次破坏前者的名次语义。使用带明确类型的列表/独立入口即可，无需复杂注册器。
7. 每次刷新导出各任务真实评分调用数与候选来源数，C>0且使用J；记录训练当下模型完整hash。不要再用只有声明没有实际调用的anchor字段充当完成回执。

## 4. 修正Teacher选择与复评

1. 固定raw C100=RR(D100, R_E,100)，同一自然B，保存IDs顺序/集合hash。
2. 六轮必须使用完全相同C100/B评估；旧D100结果保留为historical diagnostic，不能继续命名为协议C100结果。
3. 原epoch3仍保留原选择记录。修正后六轮C100复评才可产生新选择记录；本包只有best权重，其他权重需从原服务器原run读，不得用best代替其他epoch。
4. 所有视图使用同一query集合和GT分母，源分组paired bootstrap保留重复权重。原始raw同池基线每轮固定。
5. 支持分层分别记录原始witness、bundle成员、实际输入缓存槽位状态；ID相交不是原文与tensor完整性的替代。

## 5. Student公式一致性（阻断后续训练）

1. score_packet显式接收packet类型D/E/C。不能使用P/J模式区分D/E，因为D和E都是P。
2. D查询=unit(A_D nu_Q)；E查询=unit(A_E nu_Q)；C查询=原规范u_C(Q,E)。
3. 所有destination key均为nu_x，不能在candidate侧再施加base_e、A_D或A_E。只有查询侧关系变换；nu_x本身已单位化。
4. 三种training logits均为query dot nu_x /0.07；ANN只使用query dot nu_x。每张T一个key，同时服务D/C；每个evidence一个key。
5. 全湖mining、离线bank、ANN/exact、loss、Teacher KD候选对齐使用同一评分约定。更新后任意非单位参数下均要验证，不只用单位初始化测试。
6. E-only loss必须对A_E产生正常梯度，不能误更新A_D（共享object encoder仍可更新）。C loss必须流经其相应conditional参数，不能把目标keys接到额外关系变换。
7. Student每个epoch训练结束后，以同一last模型重新生成keys和queries，供下一epoch挖掘。不得把epoch开始的keys与结束后的queries配对。
8. Student C刷新必须为下一epoch使用的每个(q,e)执行全湖exact top128，不能始终留用raw ET。W_Q的下一anchor以既定cycle确定，未知/其他known-positive仍正确mask。

## 6. 完整控制流与恢复

1. 明确RetrievalEngine.pipeline返回协议：当前返回外层{'ann': block}，run_split不得再次包一层。修复调用端直接使用该返回字典，并用真实run_split测试；不要只断言pipeline_both与pipeline的内部函数相同。
2. 运行一次小型fresh训练→epoch保存→dev→hard refresh→第二epoch取样的集成测试，分别覆盖SUP和KD。确保异常不发生在昂贵epoch结束后。
3. 真正需要resume时保存optimizer、scheduler/global step、所有RNG和sampler位置；只有weights+epoch的现有pt不足以声称精确恢复。
4. 模型cache identity使用完整state_dict/文件内容hash，不只每tensor前8个数。本次未发现hash碰撞，此项是防止未来误复用。

## 7. Query行与缓存复用

先导出query-only原始行数、序列化保留行数、独立行槽位数分布。当前统一九槽/截12行不等于后来用户要求的query全行保留。
- 对query实现z+schema+每行独立摘要，动态有效槽位/对象块边界。
- 若本轮所有query本来<=7且已全部保留，报告本轮无实际截行影响，不重复编码它们。
- 缓存内容不兼容时仅重建受影响query；目标/evidence缓存符合指纹者复用。
- 不因错误报告“行修订已完成”而把旧行组合冒充新表示，也不无依据重跑全湖Qwen。

## 8. 验收后才重跑

先修代码和合成/小型真实接线验证；全部通过后，另起新run目录，从同一预定fresh初始化重跑Teacher。不要继承旧epoch3或已受错误Student公式影响的epoch1权重。

保留原6epoch预算、网络、损失、seed与候选预算，不同时进行模型/超参搜索。每个epoch输出raw、P-QT、J-empty、J-natural同池结果、训练实际B分布和mask/skip。先判断修复Teacher是否真实学会任务，再启动完整SUP/KD。

禁止“为证明修复有效”临时切换最终主分数到P、插入dev GT候选、过滤无witness目标或改变分母。必要的协议修订单独记录，原异常run明确标注实现不合规，而不是成功执行后的方法失败。
