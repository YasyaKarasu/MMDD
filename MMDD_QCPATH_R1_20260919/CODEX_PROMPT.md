# 交给 Codex / 执行模型的完整指令

你要实现并执行的是附件中的 **MMDD QCPATH-R1 v1.0**，不是自行设计另一套实验。

## 先完整阅读，禁止只看摘要

完整读取 `EXPERIMENT_SPEC.zh-CN.md`、`protocol.json`、`ACCEPTANCE_CHECKLIST.md`、`SOURCES_AND_BOUNDARIES.md`，再阅读 `reference/contracts.py` 与测试。不要先开训练进程再补读规范。

**必须逐条遵守主文档全部规划和要求。不得擅自减少步骤、修改公式/损失/标签/候选/训练范围/终点/门槛，不得以提速、简化、补全、兼容或效果更好为理由改变实验语义。**正文和配置冲突时停止并指出冲突，不能自己挑一个。

## 你现在获得的授权

1. 在本地仓库定位指定B13、原强T0、原始数据集/GT、冻结编码及生产检索/admission。先执行无训练解析，保存真实路径/hash/源码及缺项。
2. 从原始train GT新建本轮监督，不读取旧训练list/path图/负例/Teacher logits。复用parent权重属于局部试验，不能称fresh。
3. 完成数学测试、生产函数测试和真实小样本集成。允许添加薄入口和小模块；不要构建通用平台或重构无关src。
4. A：E-only与QE两臂，固定ρ=.5、完整合法目标分母、冻结原P/R/Direct/QE/index、seed13、3epoch、logical128/micro32、FP32、raw inner-product。不加KD，不改温度，不改分数空间。
5. 只在协议门槛PASS_A时启动B。B保留原强T0既有局部+全局表征、同一Transformer和head，扩展真实Q/E/T路径；冻结压缩器。两臂QT续训对照/真实Path，各2epoch；本文规定的训练输入、支持mask、增强和LSE必须全部接入。
6. B主方法使用真实路径聚合。零跳候选由同一个Teacher计算；不得接入外部QT标量，不得发现Path差就退回QT-only。
7. 只有A13、B13都PASS才重复seed29；最多8个正式训练作业。阶段没到就写NOT_REACHED，不允许自动跑KD/fresh/Stage2/200K。
8. 保存raw rankings/scores/paths/masks/receipt与新训练权重，生成RESULTS/LIMITATIONS/NEXT_DECISION，重新解包校验结果。

## 必须避免已经发生过的错误

- 全目标分母不能退成Top32；分块不能各自softmax然后平均。
- Student训练、exact、ANN调用同一个条件向量函数；用非单位非对称R测方向。
- Query的全部实际example rows不能因固定槽数被截断或混成行组。
- empty `positive_ids=[]`不得被序列化当false丢掉并回退G。
- Natural必须真正读取自然path；不能传None后只训练空/单witness。
- Augmented的同一个e*必须给所有candidate，不能只给正target。
- target GT正例不代表每条evidence都正；unknown不升级成确认负例。
- 不能删缺特征候选来改评测人口，不能用proxy或旧cache伪装真实Teacher。
- QT-only对照不能更换候选池或被削弱来制造Path胜出。
- GT类型、ID、source、排名不能送入模型；在线不能按implicit/explicit分流。
- 不挑最好epoch/seed冒充预定终点；不降低门槛以继续训练。
- B的QT支持对比项必须复用同一个forward标量，不能用两次不同dropout后声称梯度相消。

## 遇到问题时

本地可读取的信息先自行定位，不重复询问用户已经提供的背景。确实缺少必要资产或规范矛盾则输出准确的BLOCKED状态和证据，继续独立且已授权的步骤；不要发明替代方案或追加训练。

效果门槛失败与实现错误必须区分。效果失败就停止该路线，不开始新一轮诊断/扫参。实现错误最多允许主规范规定的一次保留失败记录的修复重启，不能把污染checkpoint续训成“修好了”。

## 你的第一次输出

先给出：解析到的输入角色与缺项、A/B允许执行到哪一步、必须新增的最小文件、H01–H16合规映射。没有这一步和S0验收，不要启动正式训练。

## 你的最后一次输出

报告实际训练了哪些job/epoch、使用了什么parent/数据/候选/评分、哪些门槛通过或失败，并给出原始证据路径。明确哪些尚未执行。不要只说“已实现”或“测试通过”。

**最后再次强调：你必须遵守附件主规范的每一项规划和要求，不能自由发挥。一个可信、完整的负结果优于任何违背协议的高分结果。**
