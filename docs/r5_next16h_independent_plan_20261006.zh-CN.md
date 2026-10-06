**建议：保留 Teacher，停止蒸馏救火；这 16 小时全自动执行，优先验证 Teacher 的证据效用、源表属性值恢复、图像增量与证据内容依赖。文本 span 降为最多 1.5–2 小时的机制探针，不默认全量运行。**

旧执行书是参考，不是已完成的实验。本次只新增 CPU 审计及本计划，没有运行新 GPU 实验，没有改动旧实验实现、R5 参数或原始产物。本文件是独立的实验决策与实现规格，不是声称所有命令已经可用的执行书。

日期采用环境时间 2026-10-06。预算是 16 小时墙钟、2×RTX4090。**用户已明确今天没有人工分析：零人工标注、零人工审核、零人工放行节点。** 所有分支依据预先固定的自动指标决策；不能把人工条件留作前置依赖，也不默认用LLM裁判替代人工。一个完整 dev+test recover 按已测约 3.5 小时估算，不把双卡当成单个串行进程自动加速两倍。

路径简称：S1=`work/stage1_entitables_r5_s13_sup_rerank/`；R5=`work/stage2_entitables_r5_sup_crop_on/`；A=`work/r5_independent_audit_20261006/`；F=`A/experiment_priority_audit/`。

**1．本次独立复算，改变了实验优先级。**

| 新检查 | 实际结果 | 对决策的影响 |
|---|---|---|
| 文本 span 真正有多大干预空间 | test 实际任务涉及 4,958 个不同文本资产，只有 1,248 个（25.17%）超过 192 tokens；中位数 174、95 分位 225、最大 746；超过 1,024 的为 0 | 当前证据已是短片段，没有验证长文滑窗的自然场景。不能按“长文定位”来安排主实验 |
| 192-token 截断可能节省多少 | 以原始通过 gate 的任务逐次文本输入累计，test 文本证据 tokens 为 13,362,071；截断最多去掉 450,902，即 3.37%；dev 为 3.98% | 这还没有包含固定 prompt、row、生成和图像 tokens。不能预设 joint 会有明显总成本收益 |
| 定位反而增加多少计算 | 在原始任务上按 query/row/attribute/asset 去重，test 约 16,105、dev 16,714 个长于 192 的定位单元 | 若逐单元做 V-feature forward，双 split 约新增 32,819 次定位前向；这是估算，实际受缓存和后续重试影响 |
| Teacher 选择能改变多少输入 | test 10,409 个 view 中 6,797 个有同模态多证据可选；Teacher 与旧 cosine(Q,E) 在 4,175 个 view、967/1,166 个 query 上选择不同 | Teacher 对照有足够干预空间，值得做；并非绝大多数输入完全相同的消融 |
| Teacher 是否已明显更会保留标注证据 | 在原袋含已标注 witness、且存在选择的 930 个 test view-row 中，Teacher 保留 426，cos(Q,E) 保留 447，cos(Q,E)+cos(E,T) 保留 351；dev 为 439/444/364，共 892 个单位 | 不能预设 Teacher 会赢。也不能看到两跳 cosine 较弱，就删掉更强的 Q–E cosine 对照 |
| 恢复值能否直接核对 | test 隐藏属性有 3,225 个非空源表 row-attribute 标准值；计划覆盖 2,667，带证据覆盖 2,647；最终 VALUE 644，规范化精确匹配 416，冲突 43 | “产生过值”不能替代“正确恢复”。当前全槽位精确匹配率 12.90%，已输出值中的精确匹配率 64.60% |

来源：F/SUMMARY.json、VIEW_SELECTION.csv、HIDDEN_ATTRIBUTE_SLOTS.csv；复算代码 `src/audit_r5_experiment_feasibility.py`。

token 统计用本地 Qwen3.5 tokenizer，对实际 reader 输入的前 3,000 字符计算；没有加载推理模型。token 缩减比只描述原始任务中的文本证据部分，不是整机提速估计。known-witness 统计是有限标注下的条件诊断，单位有同 query 相关性，不能当成独立样本精度或完整证据召回率。

源表对齐使用 `query_tables.source_table_id`、每行 `source_row_id`、`hidden_attributes.source_column_index`，不是拿目标表任意单元格当答案。6,620 个 dev+test row-attribute 单元无重复，其中 6,603 个标准值非空；额外与 4,604 条 `evidence_recoveries` 标准值核对，4,604/4,604 一致。test 有 224/583 个 implicit query 至少一个隐藏属性值精确匹配；这与“任意属性输出 VALUE”的 434/583 不是同一口径。

416/644 也不能直接写成语义正确率：不匹配包含 `Tokyo, Japan` 对 `Tokyo` 的粒度差异、缩写、真正错误等。test 228 个非精确匹配输出中，32 个等于该 query 同属性的其他行标准值。今天自动标记为“非精确匹配”和“其他行值碰撞”，不将它们全部判为语义错误，也不通过临时调相似度阈值消除差异。

补做的同池 Teacher 排名比较也已完成，不再占新 GPU 时间：

| test，同一个 SUP C150 | overall R@10 | 对 SUP 直接排序的增量与 95% CI |
|---|---:|---|
| SUP 的 Q–T 排序 | 46.7124% | — |
| Teacher FULL，alpha=.5 | 48.8422% | +2.1298 pp，[-0.4610,+4.7496] |
| 同一个 Teacher 的 F0 | 49.3997% | +2.6872 pp，[+0.0718,+5.2254] |

来源：S1 `seed13/eval/{dev,test}/native_sup/pools.jsonl.gz` 的 `all_U_QT_scores`，以及 F/TEACHER_VS_SUP_*.csv。FULL–F0 的已存 test overall 差是 -0.5575 pp，区间跨零；见 A/teacher_controls/CONTRASTS.csv。以上区间按 source group 配对 bootstrap，均为逐项、未多重比较校正。

这支持保留 Teacher 并认真验证其用途，但还没有证明完整证据分支优于直接分支。不能用弱一些、训练历史不同的独立 QT_ONLY Teacher，代替与实际 SUP retriever 的比较。

**2．旧计划哪些应保留，哪些应改。**

应保留：冻结 C30、上游排序、selector、属性、donor links；按 source group 取样；只用独立输出目录；同模态证据配额；保留 NULL、错误和完整分母；报告标准值匹配而非只看非空值。

需要修改的重点是实验定义，不是多加几个防御性判断：

1. **Teacher-vs-cosine 本身可以归因，但不能与原 R5 混为一谈。** 两个新臂都将每模态证据缩到一条，彼此配额是受控的；相对原 R5 的变化则同时包含证据袋缩减。称其为“固定计划下的证据选择扩展/消融”，不能自动改名为已验证的 R5 主方法。原 export 按已有 path slot 交接证据，并没有实现这个新选择器，见 `src/mmdd_stage1/export.py::_build_record`。
2. **保留强 cosine(Q,E)，同时补结构对照。** 旧 `experiments.py::choose_evidence` 只有整表 Q–E cosine。新增 Q–E+E–T cosine 用相同 donor 聚合；两者都报告。预算允许时，离线补已有 SUP `prepaths.jsonl.gz` 中的 `path_raw_score`；它回答 Teacher 是否超过当前学到的路径，而非仅超过未训练 embedding。不要为此重编码 row embeddings 或重训模型。
3. **文本必须有同长度 prefix192。** 原 prefix3000chars、lexical192、joint192 不足以区分截断效应和定位效应。旧代码 `mode=prefix` 是保留原文，并不是 prefix192；需要新增显式模式。
4. **自动值评估要跨臂对齐同一单位。** 旧 `audit_recovery` 按每臂的 task_id 抽一个 task，而 task_id 包含 evidence，臂间会抽到不同行/属性。固定 `(query_id, original_view_id, row_id, attribute)` 后再对齐各臂；全流水线指标则对齐最终 `(query_id,row_id,attribute)`，保留冲突/缺失。今天使用全量配对表，不依赖审查样本。
5. **成本先实测再决定全量。** 新 span 没有真实 GPU 验证。必须确保长于192的输入实际触发 hook；纯短文本 smoke 通过不代表实现已经工作。所有定位前向、retry、model load、score 成本要分别记录，不能只报告生成输入 tokens。
6. **不以表级 R@10 代替行定位正确性。** `matching.py::bridge_scores` 对每个恢复值做目标列值域匹配；同属性内置换恢复行的值不会改变这类计数。因此表级收益本身无法证明值绑定到了正确行。这是评估层次的区别，不应把 join discovery 写成已验证了 join 执行正确性。

**3．16 小时实验清单：自动值评估、Teacher、图像增量、内容控制；span 限时。**

**P0，必须：跨臂源表值评估与见证关联。CPU 分钟级运行，预留 0.5–1 小时补齐评估入口，纳入最初2小时实现预算。**

- 目的/claim：自动测量“恢复源表中缺失的连接属性值”，并把恢复值追溯到生成任务与输入证据。今天不交付人工判定的语义正确率或证据蕴含率。
- 输入/脚本：F/HIDDEN_ATTRIBUTE_SLOTS.csv、各臂 plans/recovery/scores、dataset 的 source_tables/query_tables/evidence_recoveries。将已有 CPU 槽位审计改为接受任意 `run_root` 与固定 population；Teacher/span probe 的 packet 输出也转换成相同评估键。旧 audit-recovery 只作覆盖/成本参考。
- **主指标 RowAttrEM**：每 query 中规范化值与对应源表值精确相等的槽位数 / 该 query 的全部预定非空标准值槽位数，再对 query 求均值。另报微平均计数。未计划、NULL、冲突、parse error、fallback 均计0，不能从分母删除；probe 的分母限定为预先冻结的条件诊断集。
- **辅助指标**：已输出VALUE中的精确匹配率；非精确匹配VALUE数 / 全部标准值槽位数（NonMatchOutputRate）；NULL/冲突率；其他行值碰撞率；known-witness输入命中率；最终R@10/nDCG@10及成本。NonMatchOutputRate不是语义错误率或幻觉率。
- **见证关联指标 KnownWitnessEM**：值精确匹配，并且产生该最终值的某条claim所对应的实际生成task输入包含该query-row-attribute已标注witness对应的资产。沿 `bridge slot.claim_ids → predictions.unit_id/task_id → task.evidence_ids` 关联。不能只看某个plan里有witness，就把另一个task产生的答案算成被支持。新probe直接按其实际packet关联；span可能只保留该资产的一部分，资产ID命中不能证明支持片段保留下来。
- KnownWitnessEM只表示“标准值匹配且输入含已有见证”，不能当成逐答案蕴含验证；未命中也不等于证据错误。已有witness是数据集已有标注，不将其升级为本次人工复核结果。source值/witness仅用于离线评估，不写入模型输入。
- 配对统计：以source group重采样10,000次；RowAttrEM保持query宏平均，VALUE精确匹配率每次重采样后重算分子/分母；分母为0时报告NA。输出每臂全量槽位表、差异表和CI，保留失败。用固定规则自动列出增益/损害案例索引，不等待阅读或人工结论。
- 自动报告将RowAttrEM增益、KnownWitnessEM和最终检索增益分别展示；它们共同提供机制证据，但不自动等同于“模型因这条证据推得这个答案”。内容依赖另由P4检验。没有精确增益就如实报告；不能用模型裁判把主指标改判成正结果。

**P1，最高 GPU 优先级：Teacher 的证据效用探针；有信号才扩到完整恢复。小实验 1–2 小时，扩展两卡并行约 4–5 小时。**

- 目的/claim：关系 Teacher 是否比相关性排序更能选到“能为当前 query 行恢复指定属性”的证据。保留 Teacher 作为 reranker 不以本实验成功为前提；把它写成独立科学贡献需要本实验支持。
- 第一层排名：复用已经完成的 SUP/F0/FULL/SWAP 同 C150 表，不再训练、不再 alpha 扫描。
- 第二层机制单位：在 dev 的“计划确实选中隐藏属性，且同模态至少两个 evidence”的 view 中按 source group 固定选 128 组；每组固定一个 query、一个 view，保留其全部五行。选择 query/view 的稳定 hash 不依赖恢复成败或哪种方法获胜。当前有 423 个符合条件的 dev groups，test 有 421 个，足够取样。
- 这是明确的**正确属性已被选中条件下的诊断集**，用 gold 识别待评属性只发生在 benchmark 构造；gold 值、witness、target cells 都不传给选择器或生成器。结论不能外推为全体 query 的无条件增益。
- 三个必要选择策略：Teacher residual；cos(Q,E)；cos(Q,E)+平均 cos(E,T)。均在原 bag 内每个已有模态取一条，tie 按 evidence_id，donor 聚合一致。原 bag 可作参考，但不能把其多证据预算当公平同预算比较。
- **节省算力的做法：只运行这 128×5 个共同 row/view 的不同所选 packet。** 多策略选择相同 packet 时只推理一次；缓存必须包含相同 row、column_name、evidence 顺序与内容、prompt、model、crop/config。不能把单条 evidence 的答案拼成未运行过的多证据 packet 结果。
- probe 固定为 PREFIX_PACKET 一次生成，不运行依赖前臂结果的 singleton 重试；报告为局部恢复探针。所有臂使用相同 gate 规则，若选 E 导致 gate 拒绝，保留在分母并单列；不能挑各臂成功任务重叠集。
- 主指标是P0定义的RowAttrEM，按source group配对；辅以已输出值的精确匹配率、已标注witness保留率、KnownWitnessEM和成本。不要把640行当640个独立query。
- **自动扩展门**：dev Teacher−Q–E cosine 的RowAttrEM点估计至少 +3 pp，且NonMatchOutputRate增量不超过 +1 pp；两个条件均满足才扩展。这是保守的预算分配规则，不是语义正确性或论文显著性标准。所有三个策略均报告；条件未满足就停止完整恢复扩展，保留Teacher reranker，不等人工判断。
- 扩展固定 **Teacher vs Q–E cosine** 为主比较，两卡各跑一臂完整 dev+test（已有 prepare/recover/score/evaluate 框架，但需正确性汇总修复）。不要因 test 上 path cosine 更弱而临时换主 baseline。固定 C30、selector、属性、alpha；加入原 R5 作为较大证据袋参考。
- 论文支持标准：确认集RowAttrEM增量的配对95% CI下界>0，并报告KnownWitnessEM及implicit/overall/explicit最终R@10、nDCG@10、区间与代价。如果只有条件机制收益而无端到端收益，只支持条件claim；不得把标准值匹配优势改写成未测的全面证据理解优势。区间跨零写不确定。

**P2，必做的系统归因：固定上游的 text-only recovery。单卡预算 2–3.5 小时，另留评分时间。**

- 目的/claim：图像是否带来文本证据不能提供的正确恢复与 join discovery。这比再加 entity-only/attribute-only span 更直接检验“多模态”的含量。
- 对照：原 R5 的 text+image，和只移除 recovery 输入图像的 text-only；C30、排序、selector、属性计划、文字内容与最终打分不变。包含移除图像后没有 E 的单位，以 NULL 留在分母。
- 数据/脚本：旧 `prepare --arm text_only` 可作为实现起点；`run_stage2.py recover/score/evaluate`；P0 的正确性统计扩到此臂。原 R5 recovery 作为冻结参照。这里测试的是生成阶段图像贡献；Stage-1/selector 已看过图像，不能命名为整个系统的 no-image。
- 预先定义两个统计人口：所有query；原计划含图像的query。以implicit RowAttrEM与R@10为核心，并完整报告explicit损失。自动计算“完整臂精确匹配而text-only不匹配”的槽位数及反方向计数，并标记实际生成task是否含已有图像witness。只在含图query看效果时必须标明条件。
- 成功：完整臂相对text-only的RowAttrEM有正向配对区间，并同时呈现下游指标。它证明固定上游下图像输入的增量效果；“答案来自图像哪个区域、文本完全不能回答”等更细主张本日不验证。区间跨零为不确定；为负则报告损失。不能把不匹配输出直接判为图像幻觉。
- 不默认再跑 image-only。text-only 已经回答图像的增量问题，image-only 则是另一个成本更高的问题。

**P3，保留但限时：文本 span 小型受控探针，开发/smoke/运行合计最多 1.5–2 小时。**

- 目的/claim：固定192-token预算下，行—属性联合定位是否比简单截断和词汇定位更保留正确支持。不要宣称长文或滑窗效果；本轮没有相应长度的数据。
- 在与 P1 相同类型、文本长度>192的原 view 内按 source group 预选最多128组；不足就全取并报告实际数。各臂使用同一原证据和同一前3000字符，图像配置不动。不能按原 R5 是否生成成功来筛样。
- 四臂：原片段；prefix192；lexical192；joint192。原片段可在 prompt/config 完全一致时复用原 packet 结果。短文本<=192不用重跑三遍；全体成本/效果报告仍需包含这些无变化单位。
- 主比较joint192−lexical192；prefix192分离截断效应，原片段检验删信息的代价。报告RowAttrEM、NonMatchOutputRate、额外forward和完整恢复墙钟。可附标准值/实体锚在span中的字面出现率，明确它只是字符串覆盖诊断，不是蕴含或正确绑定；词数/字符数不能当成token数。
- 真实 GPU smoke 必须触发长文本定位 forward，核对hook、offset、所选span与恢复解析。实现两小时内仍不可用，就停止；CPU合成测试不能替代它。
- **自动扩展门**：dev joint−lexical的RowAttrEM至少 +5 pp，NonMatchOutputRate增量不超过 +1 pp，且根据实测速率能在预留时段内完成固定test样本；同时满足才确认，否则停止。门槛只分配预算，最终结论仍看确认区间。不能通过临时放宽匹配规则使门通过；只减少tokens且墙钟变慢不算提速。
- 若想宣称效率，必须预先定义质量可接受范围并用区间验证，同时测得端到端时间节省；小样本无法排除1 pp下降时，不能硬写成非劣。不得把已定位文本的生成时间作为整个方法时间。
- entity-only/attribute-only 仅在 joint 已稳定超过 lexical 后再考虑；本16h默认不排。也不从原文档临时扩到整篇网页，这会改变可见信息而混入另外一个实验。

**P4，自动化条件下升级为必做的小实验：生成阶段的证据内容控制，约 0.5–1.5 小时。**

人工支持审查取消后，用受控干预补上“输出是否依赖证据”这一层。采用与P1不同的独立固定hash，dev/test各最多64个source groups，每组一个query-row-attribute；只要求原R5计划包含该单位的已有witness，不按原生成成败筛选。它是明确的“已检索到标注见证”的条件诊断集，不外推到全部query。两split最多128个单位、三个臂最多384次生成，通常远小于完整recover。

三个臂固定query row、attribute、prompt指令、parser、模型与生成参数：REAL为原packet；EMPTY不提供evidence内容；SWAP将packet换成另一source group的自然packet，保持模态数量，文本token总数尽量在±20%内，图像分辨率仍受同一配置限制。交换按固定hash，不查模型输出；优先使用同规范化属性、标准值不同的供体，拒绝相同实体锚/共享资产及当前单位的已标注witness。标签仅用于构造评估控制，不传给生成器。若没有合格供体，该单位仅参与REAL/EMPTY；SWAP的固定可用人口及排除原因在推理前记录。

REAL/EMPTY/SWAP全部直接进入同一packet生成入口，绕过外层`NO_EVIDENCE`跳过和实体gate，禁用singleton重试。EMPTY须实际调用模型；否则测到的是手写跳过规则。REAL也必须走这个入口，不能拿gate处理过的旧R5输出直接作参照。EMPTY输入更短，回答的是移除输入的效果；SWAP才用于减弱长度和模态预算差异的解释。

自动报告每query的RowAttrEM、VALUE率、NonMatchOutputRate以及REAL−EMPTY、REAL−SWAP的source-group配对区间。REAL明显优于两者时，支持“标注见证条件下生成结果依赖当前证据内容”。EMPTY也能答对提示query信息/模型先验足以回答部分单位。SWAP不一定是完全不含答案的负例，自动约束无法穷尽其语义，因此其输出率不能直接命名为幻觉率。

本实验不能判定每一个REAL答案确实被该证据蕴含，也不能单独证明Teacher的贡献。所有结果自动生成，区间跨零就标记不确定；不增设人工检查。实现超时则优先保留REAL/EMPTY并如实记录SWAP未完成，不能将未运行的内容对照算作成功。

**4．双卡排程与止损。**

| 墙钟 | GPU0 | GPU1 | 同时进行的自动工作 |
|---|---|---|---|
| 0–2h | 必要时真实模型smoke | 必要时span smoke | 补齐最小入口；冻结样本、指标、代码；P0原R5基线汇总 |
| 2–4h | P1 Teacher三策略packet probe | P2 text-only完整恢复，最长接续至5.5h | 自动评估P1、写扩展门结果；汇总成本 |
| 4–5.5h | P3四臂span probe，达到上限停止 | 完成P2 | 自动评估P2/P3，固定可完成的确认人口 |
| 5.5–7h | P4 REAL/EMPTY/SWAP小型内容对照 | 仅当P3通过门且能按时完成，做固定test确认；否则空闲 | 生成槽位差异、配对区间和P4结果 |
| 7–12h | 仅当P1通过自动门，完整Teacher recovery | 同一固定人口完整Q–E cosine recovery | 自动score、P0跨臂汇总；P1未过门则两臂均不启动 |
| 12–16h | 只完成已开任务 | 只完成已开任务 | 全量审计、bootstrap、成本和报告；至少留3h收尾 |

真实实验可能比表中快，但不得用省下的时间无限加调参。先用固定 source-group pilot 测包含定位/重试的耗时，再预测完整臂；若超预算，**在读取确认集效果前**固定共同子集，或放弃整组确认。不能只报告截止时先完成的 query。13h 后不开新 GPU 任务。每卡一个 Qwen 进程；score 和统计另计，别把3.5h recover当成全实验成本。

P1 probe的3pp、P3的5pp以及NonMatchOutputRate的+1pp上限是这里提出的预算分配阈值，不是观察新结果后再调的门槛。dev用于取舍；test已历史暴露，只能称冻结方法下的确认/回归，不能伪装新holdout。所有已跑臂和不显著结果都报告。

执行者按固定条件写 `AUTO_DECISIONS.json`：记录每个门的指标、阈值、通过与否、剩余时间、最终固定人口及跳过原因，然后自动继续下一项。无需另做复杂调度框架，也不等待用户选择扩展臂。GPU/接口失败保留日志并标记该项blocked/incomplete；在该项预算内可修复与重跑，超时自动跳到独立实验。未完成臂不比较部分结果、不伪造成功，也不改变原始R5。P1不通过则跳过其7–12h的完整恢复对照；仍完成P0/P2/P4与限时P3后提前收尾，不启动KD或新超参搜索。

**5．执行前最小代码缺口，不能当成已经实现。**

| 缺口 | 可复用代码 | 本次状态/验收 |
|---|---|---|
| 源表row-attribute对齐、原R5精确值审计 | 新增 `src/audit_r5_experiment_feasibility.py` | 已在真实原始产物上运行；任意run/probe评估、query宏平均和配对区间仍需补 |
| 三策略固定packet探针与安全复用 | `recovery.py` 的 make_task/Generator，`experiments.py::choose_evidence` | 待实现一个小入口，如 `src/run_r5_mechanism_probe.py`；不要另造整套pipeline |
| Q–E+E–T cosine 基线 | 已有冻结 `z`；本次CPU审计中已写公式 | 选择公式已离线运行；正式恢复臂/日志接入待补 |
| prefix192 | `text_span.py` | 待实现；旧prefix模式不截192 |
| 同row/attribute跨臂EM、claim-task-witness关联 | 原始plans/recovery和新槽位审计 | 待实现；不能用plan级witness命中冒充生成task命中 |
| REAL/EMPTY/SWAP统一packet入口 | `recovery.py` 的Generator与本地数据集标注 | 待接入；三个臂均真实调用模型，统一绕过外层gate与singleton |
| 自动扩展门与结果报告 | 固定CSV/JSON和已有bootstrap | 待接入简单条件分支；无人工放行、无模型裁判主指标 |
| 真GPU span/localizer兼容性与成本 | 旧 `text_span.py`、旧tests | 本次未验证；仅在真实长文本forward通过后计入可执行范围 |
| 正式Teacher/cosine/text-only恢复 | 旧prepare和run_stage2入口 | 有代码起点，没有本次实验结果；各臂需独立目录和身份记录 |

必要测试围绕实验正确性：同packet才允许复用；跨臂同row/attribute对齐且NULL留分母；prefix192与joint192预算一致；witness沿实际生成task关联；EMPTY确实调用模型而非被跳过。已有冻结产物不修改，生产数据不写gold进catalog/模型输入；`.env.openai`不访问。运行从隔离`/tmp`目录，使用MMDD环境和本地模型。

自动输出新实验根下的REPORT.zh-CN.md、METRICS.csv、CONTRASTS.csv、SLOT_DIFFS.csv与AUTO_DECISIONS.json，包含固定范围、所有臂、配对区间、标准值匹配漏斗、实际墙钟/显存/forward、按规则抽取的案例索引。每个claim逐项标记supported/inconclusive/contradicted：只对明确对应的自动指标作判断；“逐答案证据蕴含”“语义等价”“图像区域支持”直接标为not_evaluated。区间均为逐项95%，若作多个检验合并的显著性宣称须校正。报告必须落盘。

**6．论文如何随结果取舍。**

- 保留的研究问题：缺少连接属性时，如何从异构对象关系找到证据，再形成可检查的行—属性值，用于发现连接。这比“几个模型串联后R@10更高”更有可检验内容。
- Teacher 是保留的组件；“学到有向关系”“细粒度交互”“比相似度更会选证据”是不同 claim，不能因为保留组件就全部成立。这里验证前两者的部分后果，并不单独证明方向性或token交互必要性。
- Teacher probe失败时，不临时改为盲目重训Teacher或再救KD；保留其reranker角色，诚实降低证据选择创新的强度。
- span失败时，删除它作为主贡献的承诺；不为了凑创新点强跑entity/attribute乘积消融。短证据本来就可能不需要额外定位。
- 若Teacher效用、标准值恢复与图像增量都没有支撑，就不能靠改措辞宣称创新问题已经解决。应保留负结果、收窄主张，并审视数据任务与验证方法的贡献；新颖性最终还需要相关工作比较，不能由这一日实验自动保证。

这一天不安排 KD、selector 重训、alpha/融合调参、所有模态组合、从零接入新湖或多seed训练。优先得到能够决定论文主张去留的结果，再锁定方法写作。
