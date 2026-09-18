# 交给 Codex 的执行指令

请在现有 MMDD 仓库实现并执行 `MMDD Stage1 CLEAN-R1`。唯一规范为本包 `EXPERIMENT_SPEC.zh-CN.md` 和 `clean_r1.json`；不要回到历史实验脚本推测默认行为。本轮只做 Stage1，在现有原始带 GT 的 EntiTables 20K 数据集上，从零生成训练组合并训练。

先完整读规范。它不是让你提出更多备选方案：模型尺寸、损失、语义、采样、预算、阶段顺序和最终对照已经确定。你只需要实现、测试、执行和保存可复算产物。遇到环境路径可按规范的确定规则解决；遇到数据冲突或数学合同冲突，输出精确报错与位置，不擅自换算法。

## 必须保持的决定

1. 不加载任何历史训练 checkpoint、optimizer、PCA、Teacher logits、B13/C3/Rxx候选/训练列表、历史rankings。默认parent=null。已有纯冻结Qwen缓存只有在输入内容、模型和处理指纹完全匹配且可计算本轮摘要时才可复用；从原始数据枚举对象全集，不从旧缓存/旧训练清单枚举。
2. 数据集本来有GT。读取原始query_tables/data_lake_tables/bridge_assets/qrels/evidence_recoveries，保留原始split，不重新让模型标注，不把这个任务误做成从无标签网页建GT。
3. Teacher是一种共享模型：一套输入投影、一个三层Relation Transformer、一个共享线性readout。P/J是task token，不是两个独立模型、两个encoder或两个训练谱系。禁止实现Fbase/Fbridge/gate四网络。
4. P是potential检索；J是当前输入证据支持。implicit正target仍是P正例，但空E的J负例；未知非空E不自动标负。线上所有query同样执行，不读implicit/explicit标签进行路由。
5. Student使用每对象9slot（global+8固定摘要）的单query attention pooling，输出1024维单key；第二跳按规范低秩三元乘积生成Q条件查询向量，target索引静态。不建行/列全湖索引。
6. Qwen只需一次性冻结编码。只存float32 z和8×4096 float16摘要，不保存全长H、不缓存所有层、不训练resampler、不在每epoch反复跑8B。按实际表+text+image+query总数核算60GiB特征/100GiBworkspace预算，不复制cache给各模型。
7. 只训练一个fresh Teacher和同架构Student的SUP/KD两个必要实验臂。seed13，T/S都6epoch，禁止扩充矩阵或“效果差就再试几种参数”。Teacher训练完成后冻结；Student只刷新候选，不能反过来更新Teacher。
8. 训练packet、正例排除、unknown masks、multi-positive loss和KD必须与reference匹配。Teacher的singleton J监督与Student C蒸馏输入一致；禁止把20条E的bundle logits蒸馏给Q-only Student。
9. 所有候选召回结果是本轮新算；每个Student epoch重新从上一epoch自身last在全湖挖hard negatives，不长期固定一份小列表。
10. 最终C100通过指定round-robin预算调度产生，Teacher统一读取自然B_Q20，用J输出一个target logit；不把D/C/QE logits相加或LSE，不weighted RRF，不用P分数兜底替换J，不丢无arrival path目标。
11. Stage2不训练、不执行、不重构。只输出既有最大表—列乘积选择下一轮所需的r_T和完整evidence/provenance接口。

## 实现步骤

A. 确认原始数据目录与现有Qwen目录，输出INPUT_INVENTORY和resolved_config。当前包不包含原始湖/Qwen，不能把示例合成数据当真实实验。
B. 在独立 `src/mmdd_stage1_clean/` 下实现规范列出的少量模块和CLI。可复用原始artifact读取/target引用解析/序列化内容定位函数，但不能import历史run脚本的常量/工作目录/checkpoint loader。
C. 先运行本包11项CPU数学参考测试，再增加规范15项验收涉及的真实GT/cache/ANN/协议测试。不能只运行reference就声称集成测试全部通过。
D. 建立对象全集、压缩冻结cache和本轮raw排名；构建原始GT→D/E/C/B组合；实现并检查所有hash和allowlist。
E. 按规范顺序训练T，冻结best；从同一个本轮student_init分别训练S-SUP和S-KD，optimizer分别清零；内部按规范刷新hard negatives。使用一张A100顺序执行，不启动多模型并发抢显存。不使用L40。
F. 冻结best选择与配置后再读取test GT评测；输出raw baseline、各臂整体/implicit/explicit Recall、ANN/exact、same-pool J/P扰动、strict EO与资源数据。
G. 打包源码diff、配置、执行回执、loss/logits、逐query排名、统计脚本及REPORT。不得只写RESULTS.md。不许用缺失为0、drop query、drop target掩盖输入错误。

`reference_core.py`已经定义数学与张量参考，生产代码需要实现相同语义；如工程上批处理不同，可增加等价性测试，但不能改变归一化、mask、loss reduction或候选排序。命令入口 `python -m mmdd_stage1_clean ...` 当前不是旧仓库已有功能，需要由本轮实现。实现后按规范第11节完整运行。

报告明确区分：真正训练的模型、只做重放/扰动的诊断、未执行的200K效率与Stage2。若某一步失败，保留失败产物及原因；不得改名把部分流程写成完整成功，也不得自动找一个历史checkpoint“先跑通”。
