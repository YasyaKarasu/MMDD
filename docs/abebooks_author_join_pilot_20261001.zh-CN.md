# AbeBooks 固定候选湖作者连接副本：实施结果

日期：2026-10-01。

**交付定位更正：该 13-query 版本不满足模型训练和评测需求，不能作为本次数据集优化的最终交付。** 它只能保留为数据审核案例；此前生成 Stage-1 文件只证明格式兼容，不能证明样本规模或划分合理。train 9、dev 3、test 1 的划分不应启动正式实验。后续应从完整原始记录和审核缓存扩展标注并重建任务，不能继续在该子集上调参。

已经按 [技术报告](abebooks_small_dataset_technical_report_20261001.zh-CN.md) 的第一条路线生成独立副本：

[`dataset/abebooks_author_join_pilot_20261001`](../dataset/abebooks_author_join_pilot_20261001/REPORT.md)

以 `abebooks_joinability_no4_disjoint_20260930` 为输入。原目录全部文件的 SHA-256 在处理前后相同；没有改动原版、52 表历史版本、原图片或模型。副本中的元数据是物理复制，外部图片仍引用原路径，本次只读使用。源表、候选表、素材目录、实体及抽取记录逐字节保留；原查询、恢复标注、qrels、划分、catalog 和旧报告归档在副本的 `provenance/pre_author_curation/`。

## 1. 实际交付规模

| 项目 | 原输入 | 审核后主任务 |
|---|---:|---:|
| 源表 / 源记录 | 130 / 1,289 | 130 / 1,289 |
| 候选表 | 174 | 174 |
| 素材 | 5,441 | 5,441 |
| 查询 | 118 | 13 |
| implicit / explicit 查询 | 59 / 59 | 8 / 5 |
| qrels 正例记录 | 120 | 13 |
| 不同源组 | — | 13 |
| 主任务恢复证据记录 | 168，包含多个属性 | 20，仅作者 |
| 主任务不同作者行值事实 | 43 个作者事实待审核 | 16 |

原有 21 个作者 implicit 查询和 6 个作者 explicit 查询全部进入审核。其他 91 个查询记录为 `non_author_task_deferred`，不把未完成审核解释为“已证实错误”。作者查询中排除 13 个 implicit 和 1 个 explicit；未重组查询行、未删除候选、未按模型分数选择样本。

沿用原划分：train 为 5 implicit + 4 explicit，dev 为 2 implicit + 1 explicit，test 只有 1 implicit。**这是受控案例集，测试集不能承担稳定泛化或统计显著性结论。** 历史暴露没有通过重新划分被消除。

8 个 implicit 查询均有 5 行，其中各 2 行通过完整值证据复核；全部标为 `partial_recovery_verifiable`。没有任何一个被标为“五行完整可恢复”。5 个 explicit 查询只有可见值连接资格，不宣称其作者全集经过了多模态复核；其中两个正例只覆盖 2/5、3/5 行，其余三个覆盖 5/5 行。

## 2. 修正内容与判定范围

任务限定为：**补充当前抓取书目记录的缺失字段，使用保守规范化后的完整作者字段等值连接。** 每个正例至少为两行提供非空新增字段，而且执行产生的每一对行都必须对应正确记录。作者字段相同但对应其他书籍的结果不合格；不把这种关系改称同一本书属性恢复。

源记录标识只用于离线判定连接结果，实际执行连接只比较作者值。在判断前检查目标每个单元格确实对应所声明的源投影；随后枚举实际连接行对，并检查目标提供的非空增补值。来自同一源表并不足以成为正例。

规范化单独写入 `audit/author_matching_keys.jsonl`，没有改变表格编码输入。规则包括占位符拒绝、大小写及标点处理、明确的单姓氏 `Family, Given` 形式倒置；不会把 `Joe Kaplan, Ryan Dunn` 倒置成另一串姓名，不补全缩写，不做模糊别名归并。碰撞与原值映射另行保存。

对全部 27 个作者查询枚举了 27 × 174 = 4,698 个 query-target 组合。主任务保留部分为 13 × 174 = 2,262 个：13 个正例、2,249 个负例、0 个未判定。其中 12 个负例具有作者值重叠，但会连接到其他书目记录；其余 2,237 个没有完整作者值重叠。本次没有发现需要新增的合格正例，未为了制造多正例而将所有作者重叠自动标正。

正、负、未判定三态写入 `audit/candidate_judgments.jsonl`。构建器支持补全多个合格目标及对应恢复路径；未判定目标和正例进入负采样禁用清单，有未判定目标的查询不进入主任务。本次负例只对这里定义的完整作者等值连接任务有效，不代表目标通过任何其他连接键都不相关。

## 3. 证据复核

重新查看了原 58 条作者证据记录：47 张图片、11 条文本，涉及 43 条源行事实。审核者是 Codex 的图片/文本内容检查，**不是独立人工标注，也不是新模型实验**。可复现审核输入为 [内容审核记录](../configs/abebooks_author_evidence_review_20261001.jsonl)，每条绑定素材内容哈希。

| 证据判定 | 记录数 |
|---|---:|
| 完整字段值受支持 | 34 |
| 源值只有作者列表的一部分 | 16 |
| 姓氏或缩写不完整 | 3 |
| 作者字符串重复或有歧义 | 3 |
| 封面明确是编辑角色 | 1 |
| 角色未判定 | 1 |

34 条可用证据对应 27 个不同源行值事实，其中主任务采用 20 条证据、16 个事实；其余随不合格查询保留为未使用项。全部 168 条历史恢复标注均有使用状态和原因，110 条其他属性标注不进入本轮监督。

例如，源值 `Ford, William` 对应封面同时出现 William Ford 和 William Topp；`C. J. Date` 的封面同时列有 Hugh Darwen。不能继续按完整作者字段恢复成功计数。书名直接含 `Cohen, Kackie; Daniels, Andrew` 的查询也记录了可见泄漏。

这些修正不会把原始卖家声明改写成真实世界的权威作者目录；未审核行、作者省略、别名及编辑角色仍有明确边界。

## 4. 连接诊断与独立性

| 条件 | implicit 正确行对 / 产生行对 | 范围 |
|---|---:|---|
| 原始隐藏值补回 | 40 / 40 | 8 × 5 行，结构 Oracle |
| 内容复核值回放 | 16 / 16 | 16 个已复核行，query 行覆盖 40% |
| 不提供作者值 | 0 / 0 | 只能说明该等值连接没有键；precision 不定义 |
| 已复核行之间固定循环交换作者值 | 0 / 16 | 两行互换，产生错误记录连接 |

回放使用内容复核的 `observed_value`，独立扫描目标行执行连接，再与审核行对比较。没有用来源 ID 执行连接，也没有先过滤错误行对再计数。

例如 test 查询 `query_d30f5d4da66593ca` 中，James Faure Walker 对应行恢复后可补入该抓取记录的 `publication_year=2006`，Don Jones 对应行可补入 `publication_year=2004`。两行不会相互连接，也不会扩张到目标全部行。增补价格等字段只表示当时抓取记录的值，不代表图书的永久实体属性。

未发现保留查询的源组、源行、实体标识、规范化书名作者组合及批准证据内容跨 split 重复。附属素材仍有 18 个跨 split 重复内容组；另外记录 52 对文本近重复候选，包含完全相同文本对，不能与 18 组直接相加。图片使用 dHash 距离 ≤2、文本使用五词片段 Jaccard ≥0.90 筛查近重复；这些是候选提示，不是全面排除所有近重复的证明。素材不因这些诊断被删除，完整检索库仍共享。

以上只完成数据资格审核、结构 Oracle 和证据值回放。**尚未执行新 Direct 检索、属性生成器或端到端检索证据恢复实验，不支持训练收益或证据必需性的结论。** 后续 P1 应在同一副本、同一候选湖与标签上继续受控比较。

## 5. 验证与使用

实现入口是 [curate_abebooks_dataset.py](../src/curate_abebooks_dataset.py)，构造逻辑是 [abebooks_curation.py](../src/mmdd_dataset/abebooks_curation.py)。新目录必须不存在；命令会拒绝覆盖原输入或既有输出。构建方法见 [src/README.md](../src/README.md)。

从隔离工作目录运行相关测试：

```bash
cd /tmp
conda run -n MMDD python -m pytest \
  /home/oycy/MMDD/tests/test_abebooks_curation.py \
  /home/oycy/MMDD/tests/test_src_dataset_builder.py \
  /home/oycy/MMDD/tests/test_abebooks_source_rebuild.py \
  /home/oycy/MMDD/tests/test_stage1_construction.py -q
```

测试结果：**53 passed**。同时实际导出了 [Stage-1 数据](../work/abebooks_author_join_pilot_20261001/stage1/target_lists.jsonl)：5,628 个对象、495 个 edge 列表、13 个 target 列表，候选 corpus 包含 174 张表和 5,441 个素材。只进行了离线准备，没有编码、训练或 API 调用。

独立 [最终校验](../work/abebooks_author_join_pilot_20261001/FINAL_VALIDATION.json) 确认：原输入哈希未变；所有保留工件数量和内容一致；qrels、query target IDs、catalog、划分与恢复行对应一致；Stage-1 的正例和负采样符合新标签；实际表编码输入与原来的对应查询/目标视图一致，未混入来源 ID 或审计答案。

后续实验请使用新的数据目录与缓存命名空间，不复用旧监督列表。完整逐项审计可从 [副本报告](../dataset/abebooks_author_join_pilot_20261001/REPORT.md)、[构建统计](../dataset/abebooks_author_join_pilot_20261001/CURATION.json) 和 [查询审核表](../dataset/abebooks_author_join_pilot_20261001/audit/query_audit.jsonl) 进入。没有发布、提交大数据产物或覆盖历史实验。
