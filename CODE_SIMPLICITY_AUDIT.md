# `src/` 代码简洁性审计报告

审计日期：2026-09-07

## 1. 审计目标

本次审计依据根目录 `AGENTS.md` 中对科研代码的要求，检查 `src/` 下当前实现是否存在：

- 冗余或重复实现；
- 多轮实验迭代遗留的不必要复杂度；
- 不必要的抽象、间接层和防御性设计；
- 职责混杂、依赖方向不清晰的设计；
- 已无实际用途的死代码；
- 遮蔽 pipeline 和算法主线的工程性代码。

所有建议均以**保持当前代码行为不变**为前提，不调整算法、实验协议、随机过程、指标定义、缓存格式、产物结构或历史实验结论。

## 2. 审计范围与方法

审计范围包括：

- `src/mmdd_dataset/` 数据集构建流程；
- `src/mmdd_stage1/` Teacher/Student、训练、检索、缓存和评估逻辑；
- `src/mmdd_stage2/` verifier、Oracle、reader cache 和训练逻辑；
- `src/` 根层的命令行入口、实验轮次脚本和结果汇总脚本；
- `tests/` 中与上述实现相关的调用点和行为测试。

当前 `src/` 共约 4.1 万行 Python 代码，其中根层入口和实验脚本约 2.3 万行，是复杂度增长最明显的区域。

本次仅进行了静态审计，没有修改源码，没有运行应用或测试，也没有访问 `.env.openai`。

## 3. 总体结论

当前实现的核心算法结构总体仍然清晰，主要问题不是算法本身过度设计，而是多轮实验迭代后形成了三类外围复杂度：

1. 通用计算逻辑残留在特定轮次的实验入口中，导致新脚本反向依赖旧脚本的私有函数。
2. 训练和评估入口同时承担参数处理、缓存准备、运行控制、选模和产物生成，主算法路径被大量编排代码包围。
3. 相同的指标记账、评估循环、缓存序列化、子进程执行和文件写入协议在多个位置重复维护。

没有发现足以支持大规模删除的死代码。历史实验脚本、兼容参数、缓存校验和恢复机制不能仅因当前主流程不直接使用就判定为冗余。

## 4. 主要发现

### 4.1 高优先级：历史实验入口已成为隐含基础库

位置：

- `src/run_stage1_r7_task_q.py:28`
- `src/run_stage1_r8_task_r2.py:28`
- `src/run_stage1_pipeline_unification_task_a.py:32`
- `src/run_stage1_r3_sweeps.py:254`

后续实验入口直接导入早期轮次脚本中的 `_path_pool`、`_query_values`、`_append_values`、`_finalize_records` 和 `_teacher_ensemble_metrics` 等私有函数。轮次脚本因此同时承担“历史实验入口”和“公共实现模块”两种职责。

这会产生以下问题：

- 阅读当前评估流程时必须跳转到旧实验脚本；
- 私有函数实际上已经成为无声明的公共 API；
- 修改旧实验脚本可能意外影响后续实验；
- `mmdd_stage1` 包不再完整表达 Stage 1 的公共计算能力。

建议将已经被多轮复用的计算函数移入 `mmdd_stage1` 中职责明确的模块。历史入口可保留原名称的兼容导出，避免破坏已有导入和复现实验。

行为保持要求：逐查询顺序、指标字段、coverage 定义、bootstrap 参数、基线名称和各轮选择门槛必须保持不变。不同轮次的实验政策不应为了共享实现而被统一。

### 4.2 高优先级：`train_stage1.py` 职责过多

位置：`src/train_stage1.py:822`

`run()` 同时负责：

- 参数默认值和合法性检查；
- 数据路径和 split 处理；
- FeatureStore 和冷热缓存规划；
- Teacher logits 缓存读取与补算；
- Teacher/Student 初始化；
- optimizer 构造；
- 四个训练阶段分发；
- epoch gating、ANN 构建和评估；
- checkpoint、history 和 selection 产物组装。

该函数约 700 多行。四阶段训练虽然仍然显式，但研究人员很难从入口快速识别“当前阶段实际执行了什么”。Student 初始化、维度检查、关系学习率和 optimizer 构造还在 edge/path 分支中重复出现。

建议按以下直接步骤整理，而不是引入通用训练框架：

1. 验证并规范化运行参数；
2. 加载训练、hard-negative 和 dev 数据；
3. 准备特征缓存与 Teacher logits；
4. 初始化对应模型和 optimizer；
5. 显式执行四个训练阶段之一；
6. 完成选模并写出产物。

行为保持要求：不能改变文件访问和随机种子设置顺序、Teacher 缓存生成时机、GPU 对象释放时机、epoch-zero 评估、hard-negative provenance 校验或产物字段。

### 4.3 高优先级：训练循环被重复的指标记账逻辑淹没

位置：

- `src/mmdd_stage1/training.py:873`
- `src/mmdd_stage1/training.py:1022`

Student edge/path 训练为每一类 loss 分别维护最终值列表和 pending tensor 列表，并逐个执行 append、flush 和 mean。Student path 一段需要同时维护九类指标，大量代码只是在搬运同构数据。

建议使用固定指标名到列表的局部字典，通过短循环完成初始化、追加、刷新和最终均值计算。模型评分、loss 公式、反向传播和每个阶段的显式结构仍应保留。

行为保持要求：

- 仍然每 100 步和 epoch 末尾刷新；
- 保持 `detach()` 后再转 CPU；
- 保持现有 batch 顺序；
- 保持当前按 batch loss 求平均的口径，不能改成按样本加权；
- history 中的字段名和数值类型保持不变。

### 4.4 中优先级：r7/r8 重复完整的检索组合评估循环

位置：

- `src/run_stage1_r7_task_q.py:185`
- `src/run_stage1_r8_task_r2.py:270`

两处都执行以下流程：

1. 按 recall k 和 batch 遍历查询；
2. 执行 zero/one-hop detailed retrieval；
3. 提取路径池；
4. 更新候选池哈希；
5. 遍历 aggregation 和 fusion 配置；
6. 重新排序路径并累计逐查询指标。

r8 已经直接复用 r7 的配置函数，因此这里属于同一协议的执行逻辑重复，而不是两个独立实验恰好结构相似。

建议提取一个接收 indices、examples、配置列表和 batch size 的评估函数，返回 records 和 pool hashes。checkpoint 加载、实验选择规则、报告文本和轮次特有门槛继续保留在各入口中。

行为保持要求：检索次数、batch 边界、配置遍历顺序、哈希分隔字节、计时范围和产物结构不能变化。

### 4.5 中优先级：raw 与 Student 评估重复转发同一协议参数

位置：

- `src/train_stage1.py:613`
- `src/train_stage1.py:640`

raw embedding baseline 与 Student checkpoint 的评估连续调用 `evaluate_student_retrieval()`，重复传入二十余项 recall、候选池、路径聚合和 fusion 参数。

这些参数定义的是同一评估协议。重复展开既增加视觉噪声，也容易在新增参数时只修改其中一处。

建议在调用处构造一次公共参数字典，raw 和 Student 分别只传入索引对象以及各自独有参数。

行为保持要求：raw 指标仍然只计算一次，Student 仍然接收 `identity_baseline_metrics`，参数值和调用顺序保持不变。

### 4.6 中优先级：Teacher 与 cosine logits 缓存重复维护相同结构

位置：

- `src/mmdd_stage1/teacher_logits.py:562`
- `src/mmdd_stage1/teacher_logits.py:599`
- `src/mmdd_stage1/teacher_logits.py:660`
- `src/mmdd_stage1/teacher_logits.py:686`

Teacher 评分和纯 frozen-embedding cosine 评分虽然来源不同，但后续都重复执行：

- 按实际候选数截断 direct/evidence logits；
- 使用 `replace()` 回填 `TargetExample`；
- 写入 aggregation 配置；
- 对变长 logits 执行 padding；
- 原子写入缓存文件。

建议只提取 target logits 回填和缓存 payload 编码的局部 helper，保留两条评分路径本身。

行为保持要求：缓存路径、fingerprint、dtype、padding、ensemble 版本、Teacher 模式字段及 cosine 独有的 `target_source` 必须保持不变。纯 cosine 路径不能因此重新依赖 Teacher 模型或 Teacher hidden tier。

### 4.7 中优先级：Student 逐对关系公式维护了两份

位置：

- `src/mmdd_stage1/models.py:624`
- `src/mmdd_stage1/models.py:663`

`score_embeddings()` 和 `score_pairs()` 都实现了相同的 full/lowrank 逐对关系公式。二者的区别主要是前者自行投影一对 embedding，后者先对唯一对象批量投影并按关系分组。

建议提取一个只接收“已投影 source vectors、destination vectors 和 relation key”的内部计算函数。两个公共入口和 `score_pairs()` 的唯一对象投影优化继续保留。

行为保持要求：关系分组、投影缓存、tensor shape、运算顺序和梯度必须保持一致。矩阵评分和 ANN query/index 公式不应顺带合并。

### 4.8 中优先级：相同的子进程执行包装复制四份

位置：

- `src/run_stage1_r6_task_f.py:18`
- `src/run_stage1_r6_task_f_tokens.py:18`
- `src/run_stage1_r6_task_e.py:15`
- `src/run_stage1_r7_task_p.py:15`

四个 `_run()` 函数执行相同的环境复制、`PYTHONPATH` 和 `PYTHONUNBUFFERED` 设置、`COMMAND` 日志追加以及同步 `subprocess.run(check=True)`。

建议共享一个普通的子进程执行函数，并显式传入仓库根目录。各实验的命令构造和完成条件继续留在原入口，不需要引入调度器或任务框架。

行为保持要求：解释器、cwd、环境变量、日志追加格式、stdout/stderr 合流、失败传播和命令执行顺序保持不变。使用 `/tmp` 工作目录或覆盖日志的其他执行器不能直接并入。

### 4.9 中优先级：WDC 提取阶段重复已有的原子分片写入流程

位置：

- 已有 helper：`src/mmdd_dataset/wdc_pipeline.py:578`
- `model_tasks` 重复实现：`src/mmdd_dataset/wdc_pipeline.py:1111`
- `model_results` 重复实现：`src/mmdd_dataset/wdc_pipeline.py:1148`

两处重复展开 `_new_shard -> write -> commit -> add_stage_shard`，并在异常时执行 `abort()` 后重新抛出。该流程与现有 `_write_outcomes()` 相同。

建议直接复用现有 helper，不再增加新的 writer 抽象。

行为保持要求：JSONL 顺序、空分片提交、异常传播、manifest 发布时点和 `after_shard` 调用顺序必须保持不变。存在 pending model tasks 时仍然不能发布 result shard。

### 4.10 中优先级：Student 与 raw ANN 重复加载持久化索引

位置：

- `src/mmdd_stage1/retrieval.py:313`
- `src/mmdd_stage1/retrieval.py:435`

两处都执行 HNSW 实例创建、`load_index()`、`set_ef()`、ID JSON 加载和对象数量校验。

建议提取一个返回 `(index, object_ids)` 的小型内部 helper。Student 和 raw 类自身的 manifest 校验、维度选择、lowrank relation key 以及 destination type 过滤继续独立保留。

行为保持要求：ID 顺序、错误类型和错误时机、HNSW 参数、manifest 校验及索引键保持不变。

### 4.11 低优先级：Stage 2 最佳 epoch 被裁决两次

位置：`src/mmdd_stage2/reader_cache.py:444` 和 `src/mmdd_stage2/reader_cache.py:460`

训练过程中已经通过 `(macro_accuracy, -column_nll, -epoch)` 维护最佳 key 和最佳权重，训练结束后又从 history 中按相同规则重新寻找最佳 epoch。

建议在更新 `best_state` 时同步保存 `best_epoch`，删除训练后的第二次选择。

行为保持要求：accuracy 优先、NLL 次优、较早 epoch 最优的比较顺序和权重复制时机必须保持一致。checkpoint 中的 selected epoch 必须继续与实际权重对应。

### 4.12 低优先级：Stage 2 重复维护相同的辅助指标定义

位置：

- 宏平均：`src/mmdd_stage2/reader_cache.py:391`、`src/run_stage2_round1.py:377`
- evidence modality bucket：`src/mmdd_stage2/oracle.py:63`、`src/mmdd_stage2/reader_cache.py:301`

两处宏平均都对各数据湖的 accuracy 做等权平均。两处 modality bucket 也都把证据分成 `text_only`、`image_only`、`text_image` 和 `unknown`。

建议复用同一辅助函数，使选模标准和报告标准显式共享定义。

行为保持要求：宏平均继续对数据湖等权，而不是按样本数加权；模态分桶继续先转换为集合，并保留现有桶名和 unknown 行为。

### 4.13 低优先级：语义负样本选择存在恒真过滤

位置：

- `src/mmdd_stage1/construction.py:134`
- 唯一调用：`src/mmdd_stage1/construction.py:387`

`_semantic_negative()` 接收 `allowed` 集合，并检查 `target_id in allowed`。当前唯一调用传入全部 targets，而 postings 本身也完全由这些 targets 构造，因此该条件在当前调用链中恒为真。

建议删除 `allowed` 参数、对应谓词以及仅用于该参数的 `target_id_set`。这只是删除冗余条件，不是删除整个语义负样本函数。

行为保持要求：positive/excluded target 过滤、postings 遍历顺序、分数累加顺序和同分选择规则保持不变。

### 4.14 低优先级：视图生成会枚举最终完全被截断的随机投影

位置：`src/mmdd_dataset/workload.py:102`

确定性的 entity-plus-attribute 和 entity-only 策略已经填满 `max_views` 时，代码仍然枚举宽度 2 到 4 的所有列组合、过滤包含 entity 的组合、执行 shuffle 并追加到 plans，最后才通过 `plans[:max_views]` 全部丢弃。

建议仅在 `len(plans) < max_views` 时执行随机投影阶段。

行为保持要求：确实需要随机投影补足时，仍使用原来的完整枚举和 shuffle；计划顺序、去重、ordinal 和 query view ID 保持不变。这里使用函数局部 RNG，跳过完全未使用的尾部随机计算不会影响其他调用。

### 4.15 低优先级：候选属性循环重复计算相同的有效行与阈值

位置：`src/mmdd_dataset/joinability.py:343`

`valid_rows` 只依赖当前 table 和 entity column，`required` 只依赖 `valid_rows` 及 BuildConfig，但二者目前在每个 candidate attribute 中重复计算。

建议在 candidate attribute 循环外计算一次，并继续由每个属性分别计算 recoveries。

行为保持要求：有效行顺序、`wiki_title` 清洗规则、恢复阈值公式和每个属性的 recovery 匹配逻辑保持不变。

### 4.16 低优先级：单表构建的多个早退分支重复构造相同结果字典

位置：

- `src/mmdd_dataset/joinability.py:685`
- `src/mmdd_dataset/joinability.py:697`
- `src/mmdd_dataset/joinability.py:722`
- `src/mmdd_dataset/joinability.py:784`

多个分支都返回由同一组局部列表组成的五字段字典。建议在函数开始时创建一次局部 `artifacts` 字典，各早退和正常出口直接返回它。

行为保持要求：每次调用必须创建独立容器；早退位置、失败原因、字段名、列表对象和记录顺序保持不变，不需要为此引入结果类。

## 5. 不建议作为简化处理的内容

以下设计虽然增加了代码量，但目前具有明确的实验正确性、性能或复现价值，不应仅以“工程代码过多”为由删除：

- 四个显式 Teacher/Student 训练阶段；
- 多种 path aggregation、fusion 和 initialization 消融；
- 两级 FeatureStore、冷热缓存和 embedding preload；
- Teacher logits sidecar 及其 checkpoint、候选列表和 aggregation 指纹；
- hard-negative provenance、split 和 round 校验；
- ANN index 与 corpus/checkpoint 的绑定校验；
- 原子文件写入、WDC 分片提交和中断恢复；
- 网络、模型 API、磁盘和外部文件边界上的校验；
- 历史实验入口、负结果和用于复现的固定路径配置。

以下看似重复的实现也不建议直接合并：

- Python 标量路径聚合与 Torch batch 路径聚合，二者涉及执行环境和数值精度差异；
- Stage 2 普通加载器与 Oracle 加载器，二者的缺失数据处理和扫描策略不同；
- 使用不同工作目录、日志模式或失败处理协议的子进程执行器；
- Student pair scoring、matrix scoring 和 ANN scoring，它们服务不同计算形态。

## 6. 死代码结论

本次没有确认可以安全整块删除的大面积死代码。

仅在 tests 中出现的 helper、名称中包含 `legacy` 的迁移代码、当前默认实验未启用的消融路径，以及已废弃但仍参与 CLI 兼容、配置序列化或 fingerprint 的参数，都不能仅凭静态调用次数判定为死代码。

当前可确认删除的内容主要是局部恒真条件、重复裁决和重复计算，而不是完整模块或完整实验入口。

## 7. 建议实施顺序

建议按以下顺序逐步整理，每一步单独提交并验证：

1. 将跨轮次复用的评估计算移入 `mmdd_stage1`，历史入口保留兼容导出。
2. 缩短 `train_stage1.run()`，明确数据准备、缓存准备、训练和产物四段主线。
3. 简化 `mmdd_stage1.training` 中的指标收集代码。
4. 合并 r7/r8 的检索组合评估循环。
5. 合并 raw/Student 的公共评估参数。
6. 整理 Teacher/cosine logits 缓存的共同序列化部分。
7. 整理 Student scoring 和 ANN 加载中的局部重复。
8. 最后处理 WDC、Stage 2 和数据构建中的低风险局部冗余。

不建议一次完成全部整理。跨模块大重构会使行为等价验证困难，也不符合科研代码优先保持实验可追溯性的要求。

## 8. 行为等价验证建议

每类整理应增加或运行针对性的等价测试：

- 训练：固定 seed 后比较采样顺序、逐 epoch history、loss、梯度和 selected checkpoint；
- 检索：比较逐查询 ranked IDs、scores、路径池哈希和融合指标；
- 缓存：比较路径、payload 字段、tensor dtype/shape、padding 和 fingerprint；
- 数据构建：比较完整 JSONL 记录、记录顺序、ID 和 manifest；
- Stage 2：比较非末轮最优、accuracy/NLL 同分时的 selected epoch 和实际 checkpoint 权重；
- CLI：比较参数默认值、错误信息、退出状态和产物路径。

特别需要补充的测试包括：

- `train_cached_scorer` 的非末轮最优及同分选择；
- Student full/lowrank 两个评分入口的数值和梯度等价；
- Teacher/cosine 变长候选 logits 缓存字段等价；
- ANN ID 数量不匹配时的错误行为；
- WDC 提取阶段 pending、写入失败和中断恢复；
- query view 在“确定性计划已足够”和“需要随机补足”两种情况下的完整输出等价。

## 9. 最终判断

当前代码最需要处理的是**实验迭代遗留的重复编排、重复记账和职责混杂**，不是重新设计核心算法。

优先整理通用评估逻辑的归属、`train_stage1.py` 的职责和训练循环的指标记账，可以显著改善 pipeline 的可读性，使以下科学主线重新成为代码中的第一视觉层级：

1. 构造 Query、Target 和 multimodal evidence；
2. 缓存冻结 encoder 的对象级与细粒度表示；
3. 训练 Teacher 的有向连接性；
4. 将连接性蒸馏到可索引 Student；
5. 执行 direct 和 evidence path 检索；
6. 聚合同一 Target 的多条 evidence path；
7. 在 Stage 2 中选择桥接列、分配并定位 evidence、恢复属性值；
8. 验证最终 semantic joinability。

其余局部重复可以在主结构清晰后逐步处理，无需引入新的框架化抽象。
