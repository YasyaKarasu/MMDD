**结论：保留 Teacher，退出蒸馏主线；冻结 R5 作为参照，再用两天完成机制补证，然后封版写作。现在直接把原《方案》封版不合适。**

R5 有可信的系统效果，也有相当明确的 implicit 桥接收益；但没有证明“统一局部关系经蒸馏形成更好的多模态召回空间”这条主线。Teacher 可以保留为在线关系重排器及证据选择器，但其存在本身不构成创新，必须用受控实验证明其对证据可用性或正确属性恢复的作用。不要把未证实的蒸馏换成同样未证实的“Teacher 推理”措辞。

本报告依据原始分数、CSV、交接文件、训练缓存与代码独立复算。日期以环境的 2026-10-06 UTC 为准；按提问中的 10-05 计算也不改变决策。建议 10-08 晚锁定方法和主表，最迟 10-09；10-18 前剩余时间用于写作、误差分析和复核。

路径简称：

- S1：`work/stage1_entitables_r5_s13_sup_rerank/`。
- R5：`work/stage2_entitables_r5_sup_crop_on/`。
- R4：`work/stage2_entitables_r4_crop_on/`。
- A：`work/r5_independent_audit_20261006/`，本次新增复算结果及 README 中的重现命令。
- B：`/home/oycy/baselines/qwen35/runs/mm_joinability_v9_shared/test/`。实际目录为 `baselines`，不是 `baseline`。

**1．数据是可复算的；报告中的科学结论需要重新限定。**

两轮各有 1,198 dev、1,166 test 查询，分别为 599/599、583/583 implicit/explicit；每轮 9,456 条 query-policy 记录。直接从 `scores/{dev,test}.jsonl` 和正 qrels 重算 R/P/nDCG，逐条对照 PER_QUERY、METRICS，并重做全部原有 CONTRASTS：最大误差均为 0。四份 dev/test handoff 的 SHA256 与 manifest 一致。来源：A/AUDIT.json、A/README.md、`src/audit_r5_artifacts.py`。

train/dev/test 的 source group 分别为 8,644/1,000/995，两两交集为零。这排除了该层面的分组穿越，但不等于审计过所有实体、内容重复或历史调参污染。S1/protocol.json 明示 `test_role=historically_exposed_regression_after_freeze`；RESULTS.md 也明确 test 已历史暴露。论文必须如实标注，不能称其为 unseen holdout。Bootstrap 只量化固定模型条件下的查询抽样不确定性；本轮只有 seed13，不能据此声称跨训练种子稳定。所有区间均为未做多重比较校正的逐项 95% 区间。

**1.1 三个门与端到端提升不矛盾，但原论证链确实有断点。**

| S1 正式门 | dev | test | 独立解释 |
|---|---:|---:|---|
| candidate gain | −9.3907 pp | −10.3202 pp | KD 的 C150 gold coverage 比同预算 MatchedDirectC 更差 |
| Teacher E-content | +0.5287 pp | +0.9434 pp | KD 池上 implicit Real−Swap；均值达标，统计并不显著 |
| KD reranked R@10 增量 | +1.0434 pp | +0.8005 pp | 均值略正，但置信区间跨零 |
| KD−SUP C150 coverage | −7.4638 pp | −7.8330 pp | 违反 −0.5 pp 下限，导致 KD gate FAIL |

来源：S1/reports/DECISION.json；门实现 `src/mmdd_stage1/pipeline.py:2034`；门阈值 S1/protocol.json 的 acceptance_pp。

E-content 的 dev CI 为 [−1.2907,+2.3411] pp，test 为 [−0.5994,+2.5087] pp。PASS 的实现没有要求 CI 下界大于零。KD 增量的 dev/test CI 分别为 [−1.0042,+3.0757]、[−1.1761,+2.7423] pp。来源：S1/seed13/eval/{dev,test}/SUMMARY.json。

正式门主要针对 `native_kd`、未缩放 Real（alpha=1）。实际 R5 handoff 是 `native_sup`、alpha=0.5、SUP 池续训 Teacher；R4 是 KD 池、另一续训 Teacher、alpha=1。S1/stage2_handoff/stage1_gate.json 的 `stage2_allowed=true` 只是认证可交接，其 performance_gate_note 明确不改变性能门结论。

所以“门失败，但换用 SUP 后端到端变好”在数值上完全可以同时成立。真正不能继续声称的是：**证据扩展在同预算下提高召回 → 这种能力成功蒸馏到 Student → 因此下游提升。** R5 的下游提升不能倒过来证明已经失败的这两环。

换成 SUP 也没有自动挽救“证据扩展优于直接检索”的 claim：SUP test C150 coverage=85.3917%，MatchedDirectC=89.3225%，差 −3.9308 pp；dev 差 −2.8790 pp。另一方面，SUP 比 KD 保留了更多证据可达目标：test E coverage 86.0206% 对 46.2979%，差 39.7227 pp。E coverage 是通过某条路径到达 gold target，不是路径证据正确率。来源：S1/seed13/eval/{dev,test}/SUMMARY.json 的 candidate。

**1.2 端到端主表：R5 有提升，但“R@10 显著胜过 R4”不成立。**

下表均为 test、百分数，完整各 policy/dev/test 见 A/RECOMPUTED_METRICS.csv。

| population / BIDF | R@5 | nDCG@5 | R@10 | nDCG@10 | R@15 | nDCG@15 | R@20 | nDCG@20 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| R4 overall | 42.610 | 32.601 | 53.731 | 36.228 | 59.034 | 37.652 | 62.450 | 38.471 |
| R5 overall | 45.297 | 34.656 | 54.889 | 37.805 | 61.049 | 39.448 | 63.951 | 40.144 |
| R4 implicit | 33.076 | 23.656 | 45.369 | 27.702 | 52.544 | 29.631 | 58.519 | 31.065 |
| R5 implicit | 37.250 | 27.055 | 48.199 | 30.704 | 57.090 | 33.091 | 61.521 | 34.158 |
| R4 explicit | 52.144 | 41.547 | 62.093 | 44.754 | 65.523 | 45.673 | 66.381 | 45.877 |
| R5 explicit | 53.345 | 42.256 | 61.578 | 44.905 | 65.009 | 45.805 | 66.381 | 46.130 |

我按相同 source group 对 R5−R4 重新做了 10,000 次 paired bootstrap：

| test 对比 | 增量 pp | 95% CI pp | W/L/T |
|---|---:|---|---|
| overall R@5 | +2.6872 | [+0.8018,+4.5494] | 86/52/1028 |
| overall R@10 | +1.1578 | [−0.9139,+3.1712] | 80/65/1021 |
| overall nDCG@10 | +1.5769 | [+0.3304,+2.8364] | 232/200/734 |
| implicit R@10 | +2.8302 | [−0.0283,+5.6709] | 47/29/507 |
| implicit nDCG@10 | +3.0023 | [+1.2705,+4.7814] | 129/95/359 |
| implicit R@20 | +3.0017 | [+0.2555,+5.7947] | 48/29/506 |

来源：A/R5_MINUS_R4.csv。implicit R@10 的区间很接近零，也必须按跨零报告。dev BIDF overall/implicit R@10 从 54.8762/44.3100% 到 57.1992/49.2905%，增量区间分别 [+0.1277,+4.5365]、[+1.8333,+8.1127] pp；dev overall nDCG@10 的跨轮区间则跨零。

**1.3 最能支撑论文的是同池内桥接贡献，而不是跨轮总指标。**

| R5 test policy | overall R@10 | implicit R@10 | explicit R@10 | overall nDCG@10 |
|---|---:|---:|---:|---:|
| STAGE1 | 48.8422 | 45.5403 | 52.1441 | 32.2888 |
| BRIDGE_RRF60 | 50.4145 | 48.8565 | 51.9726 | 32.9043 |
| VISIBLE_IDF_RRF60 | 53.2590 | 44.5969 | 61.9211 | 36.6859 |
| BIDF_RRF60 | 54.8885 | 48.1990 | 61.5780 | 37.8047 |

- R5 implicit BIDF−VISIBLE：R@10 +3.6021 pp，CI [+1.9219,+5.3603]，W/L/T=29/4/550；nDCG@10 +3.5125 pp，CI [+2.2226,+4.7951]。dev 对应 R@10 +5.3144 pp，CI [+3.262,+7.492]。
- implicit BRIDGE−STAGE1：R@10 +3.3162 pp，CI [+1.7694,+5.0145]。证明桥接分支的作用不是仅仅修复 VISIBLE 的负作用。
- overall BIDF−STAGE1 为 +6.0463 pp，其中 VISIBLE−STAGE1 已贡献 +4.4168 pp；在这条加法分解中约 73% 来自可见值匹配，不能把全部 +6.05 pp 都归因于多模态恢复。
- explicit BIDF−VISIBLE：R@10 −0.3431 pp，nDCG@10 −1.2746 pp，后者 CI [−2.0293,−0.6226]，有明确负作用。implicit 上 BRIDGE 的 R@10=48.8565% 还高于 BIDF 的 48.1990%；这不支持“统一融合对所有任务都更好”。

来源：R5/evaluation/METRICS.csv、CONTRASTS.csv。VISIBLE 仍使用读过证据的 Stage-1 排名，不能称为“整个系统不看证据”的消融；它隔离的是固定上游条件下恢复桥接值参与最终打分的增量。

桥接增量的跨轮差分 `(BIDF−VISIBLE)_R5−(BIDF−VISIBLE)_R4` 在 test implicit R@10 为 +2.2870 pp，CI [+0.4752,+4.1809]，nDCG@10 为 +2.6194 pp，CI [+1.4752,+3.8112]。这支持“R5 配置下桥接更有效”，但不隔离 SUP、Teacher 续训、alpha、selector 训练样本中的任何一个因素。

**1.4 候选池、恢复与成本。**

同一轮四个 policy 的 C30 集合完全一致，31–50 名尾部也一致；因此同轮消融可比较，R@30/50 不受 Stage-2 重排改变。跨轮 test 的 1,166 个 C30 没有一个相同，平均 Jaccard=0.51263；dev 只有 1/1,198 相同，Jaccard=0.50362。test implicit 的 C30 recall 上限由 64.9514% 到 69.4397%，C50 由 73.0417% 到 80.0457%。这是重要的上游变化。

Stage-2 的配置除输入路径外相同，但 selector 重新训练：R4/R5 fit pairs=5,940/5,999，holdout pairs=678/687；训练入池的 6,618/6,686 对中，有证据的对从 3,909 到 6,559。因此 R4→R5 并非冻结 verifier 后只换候选排序。来源：两轮 config.json、jobs/FUNNEL.json、head/history.json；A/POOL_COMPARISON.csv。

test implicit 发出至少一个 VALUE 的 query 从 167/583（28.64%）到 434/583（74.44%）；至少一个候选获得正 bridge score 的 query 从 128 到 363；gold target 获得正 bridge score 的 query 从 67 到 199。唯一 query-row 产生过非空值的数量从 388/2,915 到 957/2,915（13.31%→32.83%）。这些是输出覆盖/匹配统计，**不是正确值恢复率**。同一个非空值可能错误，也可能碰巧匹配目标值域；`recovered_rows` 还会跨属性重复计数。

补充检查：test C30 中已有 gold targets，其保留 evidence 命中已标注 witness 的比例由 83/812=10.22% 到 225/835=26.95%。这是有限 witness 标注下的命中统计，不是 evidence precision。S1 export manifest 的 43.2153% 则是另一口径：dev 所有 implicit 正对中，进入前十且带**任意** evidence 的对 293/678；代码没有判断那条 evidence 是否正确。来源：A/HANDOFF_EVIDENCE.json；`src/mmdd_stage1/export.py:296`。

R5 dev 恢复成功 1,198，test 1,165 成功、1 例输出长度超限后 fallback，仍计入 1,166 的评估分母。selector 训练 loss 0.5403→0.2355；holdout Hit@1 81.64%→87.38%，epoch15 峰值 88.58%，最终固定使用 epoch20。无需为了这点波动重训；这个 holdout 是训练内、正确目标表入池后的选列监控，不代表全系统桥接成功率。

R5 recover 存储的 dev/test seconds 合计 12,457.62s=3.46h，model_inputs 合计 76,856；R4 对应约 1.62h、33,225 次输入。R5 的恢复覆盖增长也伴随约 2.31 倍生成输入，不能写成无成本提升。日志的 `forwards/crops/no_crop` 是图像定位累计计数，不是生成请求：dev 为 4,395/1,858/2,537，最终 test 行为两 split 累计 8,825/3,678/5,147，故 test 增量为 4,430/1,820/2,610，约 41.08% 接受 crop。不能相加两行累计数字。R5 的 test selector features 另耗 5,039s=1.40h；3.5h recover 不等于全系统总运行时间。来源：A/AUDIT.json、R5/recover.log、features_test.log。

**1.5 外部 baseline：优势成立，但不是“全面胜出”。**

已核对 baseline 与 R5 同为 1,166 test query、22,886 target tables、1,229 正对；MosaicJoin 原始 `all_query_results.csv` 独立重建后，汇总最大误差 1.44e−15。其余五个方法读取各自 table-level summary，未完整审计训练调参与重放。来源：A/BASELINE_AUDIT.json、BASELINES.csv、MOSAIC_REPLAY.csv。

| test 系统 | overall R@10 | overall nDCG@10 | implicit R@10 | explicit R@10 |
|---|---:|---:|---:|---:|
| Q+ DeepJoin | 29.52 | 15.75 | 8.78 | 50.26 |
| Q+ FREYJA | 34.06 | 20.09 | 11.18 | 56.95 |
| Q+ MosaicJoin | 43.28 | 25.02 | 19.50 | 67.07 |
| Q+ Snoopy | 34.22 | 19.32 | 10.29 | 58.15 |
| Q+ TabSketchFM | 24.56 | 21.75 | 1.26 | 47.86 |
| Q+ WarpGate | 34.68 | 19.45 | 10.35 | 59.01 |
| R5 BIDF | 54.89 | 37.80 | 48.20 | 61.58 |

对 MosaicJoin，overall R@10 +11.6067 pp，CI [+8.079,+15.122]；implicit +28.7021 pp，CI [+24.510,+32.901]。但 explicit R@10 −5.4889 pp，CI [−10.588,−0.519]。来源：A/R5_MINUS_MOSAIC.csv。

这些 baseline 消费 `materialized/qplus_tables`，adapter.log 明示 `mock:False attribute_source=proposal`，不是纯原始表格方法。只能称“共享多模态 Q+ 增强后的 baseline 系统”。输入信息、恢复预算和训练方式尚未全部控制，不能把上述系统优势直接归因于 Teacher，更不能凭此宣布某个基础模型有创新。

**2．原《方案》哪些成立，哪些需要删除或改写。**

| 方案主张 | R5 判定 | 需要怎样表述 |
|---|---|---|
| 统一有向 J(a→b)，有别于相似度 | 架构表达了有向打分；“优于相似度”“关系可组合”未充分验证 | 写成建模假设；补同预算 cosine/方向性对照才能提升为贡献证据 |
| Teacher 的跨对象细粒度交互提供知识 | 保留 Teacher 有系统依据；token 交互独立贡献未覆盖 | 分开验证 Teacher 整体、直接分支、证据内容和 token/global 支路 |
| Teacher→Student 蒸馏成功 | 不支持 | 主方法改成监督训练的可索引 retriever + 在线 Teacher；KD 放消融/负结果 |
| 在线仅 Student，不运行 Teacher | 与 R5 不一致 | R5 的表排序来自在线 Teacher 评分缓存；论文计入该重排成本 |
| Q→E→T 优于直接检索，增加同预算候选召回 | 在当前正式比较上不支持，SUP 也输 MatchedDirectC | 保留其向下游供给证据的作用，不宣称同预算 candidate gain |
| Evidence 路径对排序的净内容贡献 | alpha=.5 的证据不足 | Real−Swap 与 FULL−F0 必须分开报告 |
| 表—列选择、桥接值生成帮助 implicit discovery | 有同池最终指标支持；正确值因果链尚未闭合 | 报告发出值、正确值、被证据支持的值、最终命中四层漏斗 |
| 每条 evidence 唯一分配给一个 query row | R5 未实现该流程 | 实际对 view 中每行生成 packet，并可能做 image singleton；不能写成已验证的唯一分配算法 |
| 图像 entity-map × attribute-map 联合定位 | 与 R5 实现不符 | R5 为 row-context 提示下的 attribute-token V-feature consensus，代码没有独立 entity heatmap 相乘 |
| 文本 1024/128 窗口、192-token joint span | R5 配置未启用 | 当前 R5 使用每 evidence 前 3,000 字符；新 text_span 代码不能冒充 R5 实测 |
| 选择唯一最高表—列对 | 实际不同 | R5 branch_budget=10、per_table_cap=3，多个属性 view，而非单一最高对 |
| evidence 表级 prior 完全不含 direct | 与 R5 不符 | export/stage2_table_score 包含 f0+alpha×residual，须改公式 |
| 二阶段提升“joinability/join discovery” | 支持表级候选重排；完整 join 执行正确性未测 | 当前指标是 binary-qrel 的表级 R/nDCG，不是 join-result precision/recall 或完整键映射正确率 |
| 2K+20K 联训、200K scalability、跨湖泛化 | 本轮没有覆盖 | 不写成已验证结论，另列研究范围/补充实验 |

依据：《方案》49–84、92–147、315–403、519–641、684–764；`src/mmdd_stage1/models.py:390`、`train.py:1098`、`export.py`；`src/mmdd_stage2/{recovery,localizer,selector,stage1,matching,visible}.py`；R5/config.json。

尤其要注意：Teacher 当前非空 bag 的分数是 `f0(Q,T)+LME[h(Q,E)+h(E,T)]`；Student 的非空 bag 是 `LME[s(Q,E)+s(E,T)]`。因此不能原样保留“Teacher 与 Student 蒸馏同一种两个局部关系的组合”的公式。LME 本身对重复同分路径不累积计数收益，也不能仅凭它宣称“多条路径自动覆盖不同 rows”。

本次已经免费完成的 Teacher 固定 SUP C150 控制（alpha=.5）：

| test 对照 | overall R@10 | implicit R@10 |
|---|---:|---:|
| FULL | 48.8422 | 45.5403 |
| 同 Teacher 的 F0 | 49.3997 | 45.9691 |
| 同 Teacher 的 SWAP | 48.4991 | 45.1973 |
| 已存独立 QT_ONLY Teacher | 42.5243 | 39.4225 |

FULL−QT_ONLY overall +6.3179 pp，CI [+4.491,+8.196]，支持保留一个较好的 Teacher 重排器；但独立 QT_ONLY 不保证接受过相同续训，不能把差异全部归为证据机制。FULL−F0 overall −0.5575 pp，CI [−1.766,+0.648]；implicit −0.4288 pp，CI [−2.200,+1.377]。FULL−SWAP implicit +0.3431 pp，CI [−1.382,+2.076]。dev 的 FULL−F0 implicit +1.4469 pp 也跨零。来源：A/teacher_controls/{METRICS,CONTRASTS}.csv。这说明 Teacher 值得保留与其多模态内容推理贡献尚未证实，可以同时成立。

alpha=.5 按任务书“dev implicit 最大且 overall 不低于 alpha=0”选出，符合预定规则。dev implicit 从 alpha=0 的 44.10% 到 .5 的 45.55%；test 却由 45.97% 到 45.54%，test .25=46.83%。不要看完 test 后改选 .25；应报告这个泛化不一致。来源：S1/alpha_scan.txt、R5 任务书第 6 节的 alpha 选择规则。

**2.1 最后一次蒸馏诊断：发现的是目标/监督问题，不是“Student 完全不会训练”。**

1. R5 没有重新从 SUP 续训后的 Teacher 蒸馏。它继承 R4 的 SUP/KD Student，双方都按 SUP 选出的 epoch2 比较。实际 KD 缓存来自 R4 `TB_CQET/end.pt`，state=d434d423…；R5 SUP Teacher state=a4c115fc… 是之后的另一模型。因此 KD FAIL 不能冒充“最新 R5 Teacher 的蒸馏实验”。来源：R4 Stage-1 `seed13/teacher_logits_cache/identity.json`、`SELECTION_FREEZE.json`、S1 handoff。
2. 旧 KD Teacher 在 native_kd dev 池上、续训前 Real R@10=40.6928%，Student Direct=49.7148%，差 −9.0220 pp，CI [−11.7925,−6.2061]。这是部署候选分布上的 Teacher 劣势，不是与 Student 完全同目标的训练上界，但提示软监督并不天然优质。来源：`work/stage1_r4_teacher_sp/eval/dev/SUMMARY.json` 的 init。
3. 我读取了实际 scores.pt：12,630 个 direct list 平均 1,112 项，temperature=10 后 top1 平均概率只有 0.765%，gold 概率总质量 0.743%，归一化熵 96.34%；9,984 个有正例 evidence list 平均 1,041 项，top1=1.065%，gold mass=1.010%，归一化熵 95.46%。所以不能沿用旧轮“Teacher softmax 几乎 one-hot”的诊断；这一轮已经很软。单正例、lambda=1 时 CE+KL 的最优目标等于 `(one_hot+p_T)/2`，相当大质量被分给长尾。过度平滑是可疑因素，不是已完成因果归因。来源：A/KD_CACHE_PROBE.json。
4. evidence KD 含 Teacher 的 f0，却要求 Student 的纯 QE+ET 去拟合；实际训练原始 bag 上 Teacher evidence 与 direct logit 的平均相关系数 0.9508。再加上目标表级蒸馏没有唯一约束每条 QE/ET 的语义，低 list loss 不保证全湖逐跳检索几何正确。这是与“直接略升、E coverage 巨降”一致的机制解释，需受控实验才能区分各因素。
5. 优化没有表现成完全未更新：第三轮平均 SUP direct/evidence loss=2.1653/2.3066，KD 臂=2.8103/2.9682；KD 两项 loss 有值，参数范数变化。也已经实现 Student logit_scale=20、全列表 Teacher 打分、256 random negatives、cosine schedule。不能再不看当前实现就建议“加温度、加负例”作为新修复。来源：两条 `NATIVE_C2_{SUP,KD}/train.attempt_001.jsonl`、train.py、losses.py。

我的取舍是停止把 KD 当投稿前主线救火项目。若研究兴趣需要，未来可分别做 residual-only evidence KD、QE/ET 边监督保持、温度/权重控制；这些是新假设，不应承诺半天必然救回。单次 Student 训练日志约 5.7 分钟，昂贵部分是重新 Teacher 打分、全湖评估及下游验证；不能用训练时间冒充完整实验时间。

**2.2 如何保住创新含量。**

建议把中心问题改写为：**在连接键缺失时，如何检索并判断能够为具体行恢复桥接属性的多模态证据，并将这些可检查的属性值用于 join discovery。**

值得保留并验证的三个贡献候选：

1. 有向异构关系 Teacher 对“是否提供可恢复桥接证据”的建模。检索器负责效率，Teacher 负责关系/证据效用，不再承诺知识已无损蒸馏。必须证明它比同袋同预算 cosine 更会选择有用证据，不能只证明架构更复杂。
2. 从对象路径到 `(query row, attribute, value, evidence)` 的可追溯属性见证，以及由缺失/冲突处理和多行匹配约束的连接验证。这里的研究点是从相关性到可恢复、可验证的连接条件，而不是一般“检索后调用大模型”。但目前只做了生成和目标值域匹配，尚不能称为正确性保证。
3. 行—属性条件的多模态定位，在固定预算下提高正确恢复或降低成本。如果正文把 joint localization 当主要创新，就必须补实验；仅将已有 FOCUS、RATA、RRF 组件组合起来，不足以宣称这些组件本身是原创。

若 Teacher-vs-cosine 和定位对照都失败，仍可保留 Teacher 作为系统组件、如实报告其排序价值，但不能说创新问题已经解决。届时需要缩小主张至有证据支持的属性见证框架，并以任务定义、数据构造、误差分析和验证方法支撑论文；是否足够新颖还需与最近相关工作逐一比较，现有实验不能代替文献新颖性审查。

**3．投稿前补什么：先做归因实验，不再开展大规模调参。**

以下按价值/成本排序；每项设置最多一天的硬上限，GPU 时间按单卡完整 dev+test recover≈3.5h 估计。两张 4090 可以各跑一个独立臂，但不能无依据假设同一个串行 recover 自动减半。第一项的计算部分和 Teacher 排序控制已由本次审计完成。

| 优先级与目的 / claim | 数据与脚本 | 时间预算 | 预先固定的成功/失败标准 |
|---|---|---|---|
| P0 属性见证漏斗与盲评：闭合“证据→正确值→连接” | 冻结 R5 plans/recovery/scores；`run_stage2_experiments.py audit-recovery` 导出；按 source group 无标签抽样，人工核对 100–150 个 row-attribute-evidence 单元，保留 null/失败 | CPU 分钟；人工 3–5h。gold row-value 对齐需补一个审计脚本，优先用已有源表/标注核对 | 分别给 assignment/entity 正确率、证据支持率、非空率、正确恢复率及其分母；只有正确且获证据支持的恢复能解释最终改善，才支持完整机制。若只增加 VALUE 或错误值匹配，就降低 claim，不把输出覆盖当正确率 |
| P1 Teacher 选择证据 vs cosine：保住 Teacher 的方法贡献 | 同一 R5 C30、属性、donor links、自然 bag；`prepare --arm teacher/cosine`，每模态各选一条，再 `run_stage2.py recover/score/evaluate`；用已有冻结 embedding，无需训练 | 两卡并行两臂；按各≤3.5h recover，连准备/打分/审查约 4–6h；先 50–100 source groups smoke | 预注册主要指标为 implicit 正确恢复率/证据支持率，paired CI 下界>0；最终 R/nDCG 无实质退化。若只胜错误 Swap，未胜同预算 cosine，不支持 Teacher 内容选择创新。固定 selector 已看过原 E，结论只覆盖其后的恢复阶段 |
| P2 定位消融：决定能否保留“联合定位”的创新 | 同源样本、同前3000字符，原 prefix vs `span_lexical` vs `span_joint`；使用现有 prepare/recover/compare。联合项若获益再在同样本做 entity-only/attribute-only | 两卡先 100–200 source groups；含实现检查/人审约 4–8h，硬上限 1天。测每 query 总时间后才扩全量 | joint 相比 lexical 的正确恢复或支持率有正向 paired CI；或预注册非劣界下（例如正确恢复降幅不超过1 pp）总时间/token 明显下降。额外定位 forwards 计入总成本；只压缩输入 token 而墙钟变慢，不能称提速。若无效，放弃该创新主张 |
| P3 模态与 crop 贡献：判别“多模态”及视觉定位是否必要 | `prepare --arm text_only/image_only` 与冻结原 R5；图像原图-only 与 crop-on 共用相同计划/恢复 prompt。现成 R4 crop_off 只把 concentration 阈值设2，仍做定位 forward | 两个模态臂可两卡并行，4–6h；crop 子集 1–3h，按预算二选一或最多一天 | 预定义有图证据样本，不按成功案例筛选；证明 image 给出文本拿不到的正确值和连接，或 crop 在同图像上提高正确恢复。若效应跨零，报告不确定。用于效果的“拒绝所有crop”不能当免定位计算的成本基线 |
| P4 未暴露小型外部验证：降低历史 test 复用风险 | 优先已有、确未参与调参的 AbeBooks/WDC 标注子集；100–200 query，冻结 Teacher/Student/alpha/selector；复用本地编码模型，仅补必要缓存和推理 | 只有数据、标签和入口现成才排入6–10h；数据接入不齐就不承诺一天完成 | 选样和主指标先冻结，报告全样本及区间，方法在外部任务仍有桥接方向性收益。不能把已看过的 EntiTables test 随机切一块重命名成新 holdout；无法完成时明确限制泛化 claim |

P0/P1 必做。P2 仅在计划把定位列为主要创新时必做；否则 P3 更适合验证现有系统。P4 对投稿可信度很重要，但实际数据接入成本未经本次审计，不能伪装成已确认的几小时任务。已有 `docs/r5_teacher_text_span_16h_gemini_task.zh-CN.md` 及相应脚本可复用；它是执行书，不是已完成的实验结果。

如果只允许再投入一个完整 GPU 时段，我选择 P1，而不是换 Teacher 的训练超参数。如果允许两个时段，再加入 P2 的受控子集或 P3；同时人工完成 P0。成功标准和样本必须在运行前写下，旧 test 仅作回归，不用于追选 alpha、阈值或融合策略。不要按 implicit/explicit 的 gold 类型在线分流；它们是分析标签，不是可直接使用的路由器。

**最终建议：现在冻结实验参照和范围，不冻结错误的故事。** 保留 Teacher；把 KD 从核心贡献中移走；把新增预算集中在“Teacher 是否让证据更有用、恢复值是否正确、定位是否有独立贡献”三个问题。已有数据足以证明系统在 implicit join discovery 上有价值，但尚不足以让原方案的每一环成为论文贡献。
