# Stage-1 代码精简与架构整理方案（2026-09-01）

## 背景

Stage-1 经过 r1–r9 多轮实验迭代，核心包 `src/mmdd_stage1/` 与 CLI `src/train_stage1.py` 已积累明显的重复实现、职责错位与"多套等价代码路径"。本文档只做**代码结构**层面的清理清单，逐项给出证据、改法、风险与验证方式。

两条硬约束：

1. **行为保持**。r9 已冻结主协议（`logsumexp` + evidence-top-k 4 + `weighted_rrf` evidence 权重 0.05，两个 lake 一致，见 `stage1_optimization_r9_plan_20260901.md:21`）。在跑实验期间，本文档所有改动必须是重构而非语义变更：同样输入必须产出同样 checkpoint、同样指标、同样 JSON 字段。任何会改变数值的项都单独标注。
2. **遵守 AGENTS.md 的研究代码优先级**。不为了消除少量重复而引入抽象层；只有当重复已经造成"改一处忘另一处"的真实双维护风险时才合并。因此下面每项都注明了"为什么值得改"，纯风格问题不列入。

优先级：**P0** = 已存在双维护正确性风险；**P1** = 明显冗余、增加阅读与修改成本；**P2** = 结构不优美，可择机处理。

---

## P0-1 训练循环与 dev 目标函数各写一遍损失组合

**现状与证据**

- 四个训练入口：`src/mmdd_stage1/training.py:694`（`train_teacher_edges`）、`:755`（`train_teacher_paths`）、`:830`（`train_student_edges`）、`:984`（`train_student_paths`）。
- 四个 dev 侧目标函数：`training.py:494`（`_teacher_edge_objective`）、`:518`（`_teacher_path_objective`）、`:538`（`_student_edge_objective`）、`:620`（`_student_path_objective`）。

dev 目标函数与对应训练步内的损失组合是**同一套公式的第二份实现**（listwise CE + 蒸馏 KL + anchor 项的加权求和）。改动任一侧的权重语义、开关条件或新增损失项，都必须记得同步改另一侧，否则 dev 门控评的就不是训练在优化的目标——这是本轮最危险的一处。

**建议改法**

把每个 stage 的损失组合抽成**一个**函数，返回"总损失 + 各分项标量"的字典，训练步与 dev 评估都调用它；训练步 `backward()` 总损失，dev 只读分项。不需要引入基类或框架，四个 stage 各一个 `*_objective` 函数、训练循环直接调用即可。

**风险与验证**

零数值风险（同一公式只留一份）。验证：`conda run -n MMDD python -m pytest tests/test_stage1_training_control.py tests/test_stage1_models.py -q`，并对同一 seed 跑 1 epoch 小规模训练，比对 history JSON 的每个 loss 分项逐位相等。

---

## P0-2 `_path_distillation_losses` 手写行切片

**现状与证据**

`src/mmdd_stage1/training.py:305` 起，对四组 `ListScores` 按行掩码逐字段重建，形如：

```python
student = TargetScores(
    direct=ListScores(
        student.direct.logits[row_mask],
        student.direct.candidate_mask[row_mask],
        student.direct.positive_indices[row_mask],
        None if student.direct.positive_mask is None else student.direct.positive_mask[row_mask],
    ), ...)
```

约 40 行重复。`positive_mask` 是多正例支持（见 `b53bcec`）后新加的字段，这种展开写法意味着**今后每加一个 `ListScores` 字段都要在这里补 4 处**，漏一处即静默错位（掩码后张量行数仍一致，不会报错）。

**建议改法**

在 `ListScores` 上加一个 `select(row_mask)` 方法（返回按行筛选后的同类型对象），`_path_distillation_losses` 改为四次 `.select(row_mask)`。字段增删自动跟随 dataclass 定义。

**风险与验证**

零数值风险。验证：现有 path 蒸馏相关测试 + 断言 `select` 后各字段 shape 与旧代码一致。

---

## P0-3 两份不同的召回指标实现

**现状与证据**

- `src/mmdd_stage1/evaluation.py:65` `_retrieval_metrics`：基于 `_recall` / `_reciprocal_rank` / `_channel_metrics`，输入是 channel 字典。
- `src/mmdd_stage1/teacher_rerank.py:85` `_retrieval_metrics`：基于 `statistics.fmean`，输入是排序列表。
- `RECALL_KS = (10, 20, 30, 40, 50)` 在 6 处独立定义：`evaluation.py:18`（`DEFAULT_RECALL_KS`）、`teacher_rerank.py:18`、`sweep_stage1_fusion.py:26`、`run_stage1_r6_sweeps.py:37`、`run_stage1_r7_task_q.py:34`、`summarize_stage1_r5_final.py:16`，另有 `train_stage1.py` 内的字面量。

两份 recall@k / mrr@k 各自计算同一个报告指标。空排序、并列分数、k 大于候选数这些边界情形一旦两侧处理不同，Teacher 重排的数字与主评估的数字就不可比——而论文表格正是把它们并排放。

**建议改法**

保留 `evaluation.py` 一份实现并导出，`teacher_rerank.py` 改为调用它（把排序列表适配成它的输入即可）。`RECALL_KS` 只在 `evaluation.py` 定义一次，其余模块 import。

**风险与验证**

两份实现若边界行为本就一致则无数值变化；先写一个对照测试，用同一批含并列分数与空排序的样例喂两份实现，确认输出一致后再删除其一。验证：`pytest tests -q` 全量。

---

## P0-4 Student 打分公式在 `scoring.py` 里被重新实现

**现状与证据**

`src/mmdd_stage1/scoring.py:_score_student_candidate_rows` 绕过模型方法，自己写关系打分：

```python
if student.relation_param == "full":
    score_matrix = query_vectors @ student.relations[relation_key] @ candidate_vectors.T
else:
    score_matrix = query_vectors @ candidate_vectors.T
    score_matrix = score_matrix + ((query_vectors @ student.relation_as[relation_key])
                   @ (candidate_vectors @ student.relation_bs[relation_key]).T)
```

而 `src/mmdd_stage1/models.py:624` `score_embeddings`、`:643` `score_pairs`、`:705` `relation_query`、`:717` `index_vector` 已经各自实现了同一关系形式。lowrank 分支里"是否包含单位矩阵项"这类细节改动，必须同时改模型和 scoring，否则训练用的分数与检索/ANN 用的分数会静默分叉。

**建议改法**

`_score_student_candidate_rows` 改为调用 `student.score_embeddings`（若批形状不匹配，给 `score_embeddings` 补一个矩阵形式的重载或让它接受 [Q, D] × [C, D]），删除本地公式。

**风险与验证**

需要确认两处公式当前**逐位等价**再替换：先加断言测试对随机权重比较两条路径的输出（`torch.allclose`，容差按 float32 设），通过后再删。此项改完后 ANN 检索指标应完全不变。

---

## P1-1 `train_stage1.py` 有三套默认值来源

**现状与证据**

- argparse 默认值（`parse_args()`，`src/train_stage1.py:1629` 起）；
- `run()` 开头的 `optional_defaults` 字典（`train_stage1.py:827`，随后 `:875` 循环回填）；
- 24 处 `getattr(args, ..., default)` 兜底，例如：

```python
recall_ks=tuple(getattr(self.args, "train_eval_ks", None) or getattr(self.args, "recall_ks", (10, 20, 30, 40, 50))),
```

全文件 68 处 `raise ValueError`，其中约 90 行是对 argparse 已能约束的参数做二次校验。

同一个参数的有效默认值要读三个地方才能确定，`getattr` 兜底还会掩盖拼错的属性名（拼错时静默取默认值而非报错）。

**建议改法**

以 argparse 为唯一默认值来源：删除 `optional_defaults` 回填，把其中确实需要的默认值移到对应 `add_argument(default=...)`；`getattr(args, name, default)` 改为直接 `args.name`。保留那些**跨参数一致性**校验（如 top-k 与候选数、gate 与 split 的关系），删除纯粹重复 argparse `type`/`choices` 的检查——后者属于 AGENTS.md 所说的"推测性防御编程"。

**风险与验证**

有行为风险：若某参数当前只在 `optional_defaults` 中有默认值而 argparse 未设，直接删会变成 `None`。改法是逐参数迁移，改完后 dump 一份 `vars(args)` 快照与改动前对比，要求完全一致。验证：`pytest tests/test_stage1_training_control.py -q` + 一次 dry-run 参数快照比对。

---

## P1-2 `train_stage1.py` 终端 payload 两处逐字重复

**现状与证据**

`history_payload` 与 `selection` 两个字典重复相同子块（`path_aggregation`、`fusion`、`teacher_rerank_gate`、`per_dataset_gate`、`gate_unsatisfied`、`best_epoch`、`best_metrics`、`mining_round`、`stop_reason`）。其中 `path_aggregation` 的 4 键字面量出现在三处：`src/mmdd_stage1/training.py:1202`（`checkpoint()`）、`src/train_stage1.py:1490`、`:1579`。

下游 summarize 脚本按键名读这些字段，任一处漏改就会出现"history 与 selection 不一致"的产物。

**建议改法**

把 `path_aggregation` 字典的构造收进一个 `aggregator` 上的方法或 `training.py` 里的小函数（`aggregation_metadata(aggregator)`），三处调用它；`history_payload` / `selection` 的共享部分构造一次后复用同一个字典对象。

**风险与验证**

零风险（键名与值不变）。验证：比对改动前后两份 JSON 产物字节相等。

---

## P1-3 `sample_balanced_epoch` 与 `_sample_balanced_count` 同一算法两份

**现状与证据**

`src/mmdd_stage1/training.py:41` 与 `:84`：分组、`n_d ** alpha` 权重、最大余数法配额、按配额循环重排取样——除了"总数是 `len(examples)` 还是显式 `count`"以及 `_sample_balanced_count` 末尾不 shuffle 之外完全相同（后者由调用方 `sample_mixed_epoch` 在 `:158` 统一 shuffle）。

两者都是采样逻辑，dataset 混采比例的任何调整都要改两遍，且只有 `sample_balanced_epoch` 被测试覆盖（`tests/test_stage1_models.py:2703`），`_sample_balanced_count` 仅间接经 `sample_mixed_epoch` 覆盖（`tests/test_stage1_training_control.py:263`）。

**建议改法**

保留 `_sample_balanced_count(examples, count, rng, alpha)` 为唯一实现，`sample_balanced_epoch` 变成 `_sample_balanced_count(examples, len(examples), rng, alpha)` 后 shuffle，并保留其中"单 dataset 或 alpha==1 时直接整体 shuffle"的快捷分支（该分支保证 alpha==1 时与历史采样序列一致）。

**风险与验证**

有采样序列风险：两份实现的 RNG 调用次数必须一致，否则同 seed 下 epoch 组成变化，正在跑的实验不可复现。务必先写"同 seed 下新旧实现输出列表完全相等"的对照测试（含单 dataset、alpha=0、alpha=1、alpha=0.5 四种情形）再替换。

---

## P1-4 `FeatureStore` 五层重叠缓存

**现状与证据**

`src/mmdd_stage1/features.py` 中 `FeatureStore` 同时维护：`_eager` 全量字典、manifest `_index` + LRU `_cache`、`_hot_keys`/`_hot_cache` 频次缓存、`_preloaded_features`/`_preloaded_embeddings` 连续矩阵、`_teacher_index`，共 10 余个状态字段。另有：

- 根目录 `teacher_manifest.jsonl` 解析块与分片 `teacher_paths` 解析块两段约 35 行近似代码；
- `teacher_dimension()` / `has_teacher_features()` 带旧版缓存布局的兼容回退分支；
- `preload_embeddings` 只是 `preload_embedding_matrix` 的薄包装；
- `ObjectFeatures.for_scoring` 静默丢弃 `row_embeddings`。

每次取特征都要推断"命中哪一层"，新增一种特征时要考虑五层的交互。`for_scoring` 静默丢字段是真实隐患：需要行级 embedding 的下游拿到的是空值而非报错。

**建议改法**

按当前实际用途裁剪到两层：**全量预载矩阵**（训练主路径）+ **manifest 惰性读取 + LRU**（大 lake / 评估路径）。`_hot_cache` 若在现有实验里已被预载矩阵覆盖，删除之（先加计数确认命中率）。两段 teacher manifest 解析合并为一个按路径列表工作的函数。删除已无对应数据的旧布局回退分支。`for_scoring` 改为显式：要么保留 `row_embeddings`，要么在丢弃时对需要它的调用方报错。

**风险与验证**

删缓存层不改变返回值语义，但会影响耗时；删旧布局回退分支前需确认 `data/` 下无旧版缓存目录。验证：`pytest tests -q`，并在一个 lake 上跑评估比对指标不变、记录耗时变化。

---

## P1-5 三个 Student 类鸭子类型但签名不兼容

**现状与证据**

`src/mmdd_stage1/models.py`：

- `:434` `StudentJoinabilityModel.index_vector(destination_embedding, destination_type, source_type=None)`（`:717`）
- `:734` `IdentityStudentJoinabilityModel.index_vector(destination_embedding, destination_type)`（`:769`）
- `:775` `ProjectedIdentityStudentJoinabilityModel.index_vector(destination_embedding, destination_type)`（`:822`）

三者被同一批检索/ANN 代码当作可互换的"Student"使用，却没有共同基类或 `Protocol`，且 `index_vector` 形参不一致。调用方一旦传 `source_type=`，两个 identity 变体直接 `TypeError`——只在运行到特定分支时才暴露。

同时 `relation_param` 的 full/lowrank 分支在 6 处重复：`:582`（`relation_parameters`）、`:592`（`relation_residual_squared_norm`）、`:634`（`score_embeddings`）、`:691`（`score_pairs`）、`:713`（`relation_query`）、`:724`（`index_vector`）。

**建议改法**

定义一个 `StudentScorer` `Protocol`（`relation_query` / `index_vector` / `score_embeddings` 三个方法），统一 `index_vector` 签名为带 `source_type` 的形式，两个 identity 变体接受并忽略该参数。full/lowrank 不必强行合并成一个表达式（那会让公式更难读，违背研究代码可读性优先），但可把"取该 relation key 的参数组"收成一个内部方法，让 6 处分支变成 1 处分支 + 5 处直接用参数。

**风险与验证**

签名统一属接口变更但不改数值。验证：`pytest tests/test_stage1_models.py tests/test_stage1_identity_probe.py -q`，并用 identity 探针脚本确认恒等基线指标不变。

---

## P1-6 `compress` / `compress_many` 与表池化包装

**现状与证据**

`src/mmdd_stage1/models.py:205` `compress` 与 `:255` `compress_many` 各自实现表池化 + `table_token_embeddings` 逻辑；`:54` `structural_table_pool` 只是 `:27` `structural_table_pool_with_groups` 的薄包装（丢弃 group 返回值）。

**建议改法**

`compress` 实现为 `compress_many([features])[0]`（或反向，取当前被主路径使用、性能更优的那个作为唯一实现）。`structural_table_pool` 若调用点很少，直接让调用方用 `_with_groups` 并忽略第二个返回值，删除包装。

**风险与验证**

`compress` 与 `compress_many` 在 batch=1 时须数值等价（注意 padding/mask 差异）。先写 `torch.allclose` 对照测试。

---

## P1-7 通用小工具在 14+ 处各写一遍

**现状与证据**

- `_write_json` / `_write_jsonl` 在 `src/` 下有 17 处独立定义（`diagnose_stage1_evidence.py:30,38`、`run_stage1_r8_task_r2.py:42`、`run_stage1_pipeline_unification_task_a.py:215`、`run_stage1_r6_sweeps.py:94`、`run_stage1_r3_sweeps.py:35`、`run_stage1_r7_task_q.py:108`、`build_stage1_training_data.py:20` 等），而 `src/mmdd_dataset/utils.py:52,69` 与 `src/mmdd_stage1/selection.py:127` 已提供同功能实现。
- `_mean` 三处：`run_stage1_pipeline_unification_task_a.py:115`、`run_stage1_r6_sweeps.py:197`、`mmdd_stage1/training.py:468`。
- `mrr@50` 作为硬编码字符串散布在约 15 个 summarize/evaluate 脚本中。

这里要按 AGENTS.md 的尺度分辨：`_mean` 这种一行函数**不值得**统一（重复成本低于引入依赖的成本），但 JSON/JSONL 写出涉及编码、换行、原子性、目录创建等一致性要求，且产物直接进论文表格，值得只留一份。

**建议改法**

把 `write_json` / `write_jsonl` 定为 `mmdd_stage1` 的公开工具（见 P2-2 关于放哪个模块），新脚本直接 import；已有的一次性 round 脚本**不回溯改动**（它们的产物已固化，改动只增加风险）。`mrr@50` 这类指标键名统一由 `evaluation.py` 的常量生成（与 P0-3 的 `RECALL_KS` 一起处理）。`_mean` 保持现状。

**风险与验证**

只影响新代码路径。验证：`pytest tests -q`。

---

## P2-1 冻结协议之外的多余可选分支

**现状与证据**

- `PATH_AGGREGATIONS` 有 7 种（`src/mmdd_stage1/objectives.py:9`），r9 只用 `logsumexp` + top-k 4。其中 `comb_mnz` 甚至不是显式分支，而是 7 路判断末尾的无名 `else`（`objectives.py:93-97`），公式为 `sum * count`，从命名上完全看不出来。
- 融合模式 5 种，校验块与错误消息在 `src/mmdd_stage1/retrieval.py:662` 与 `:1061` 逐字重复；r9 只用 `weighted_rrf`。
- 引用情况：`comb_mnz` 只被 `run_stage1_r6_sweeps.py`、`run_stage1_pipeline_unification_task_a.py` 两个历史扫描脚本与两个测试引用；`power_mean` / `softmax_weighted_mean` 同样只有历史扫描脚本引用。

**建议改法**

不删除这些模式（它们是 r6/r8 消融结果的可复现依据，删了旧脚本就跑不起来），但做两件事：把 `else` 分支改成显式 `elif self.evidence_aggregation == "comb_mnz"` 并在末尾 `raise ValueError`（现在拼错的聚合名会静默走 comb_mnz 路径——这是真实的静默错误）；把两处融合模式校验合并为一个模块级 `FUSION_MODES` 常量 + 一个校验函数，消息只写一份。同时在文档字符串里标注哪些是 r9 主协议、哪些是消融保留项。

**风险与验证**

`comb_mnz` 显式化会把"拼错聚合名"从静默变为报错，属于期望的行为改变，需确认现有脚本传的都是合法名（已 grep 确认）。验证：`pytest tests/test_stage1_r6_retrieval.py tests/test_stage1_pipeline_unification.py -q`。

---

## P2-2 模块职责错位

**现状与证据**

- `src/mmdd_stage1/protocol.py:23` `validate_r6_readonly_invariants`：轮次专用（硬编码 `{"entitables", "wdc"}`、要求 image evidence）的校验，放在描述通用 split 规则的模块里，唯一生产调用方是 `src/summarize_stage1_r6.py:510`。
- `src/mmdd_stage1/workflow.py` 从 `.retrieval` 导入 `checkpoint_fingerprint`、从 `.selection` 导入 `write_json`——通用工具寄居在按主题命名的模块中。

**建议改法**

`validate_r6_readonly_invariants` 移到 `summarize_stage1_r6.py`（连同其测试一并移动），`protocol.py` 只保留通用的 `validate_protocol_split`。`checkpoint_fingerprint` 与 `write_json` 移入一个 `mmdd_stage1/artifacts.py`（或复用现有最贴合的模块），`retrieval.py` / `selection.py` 改为 re-export 以免动到旧脚本。

**风险与验证**

纯移动。验证：`pytest tests -q` + `python -c "import summarize_stage1_r6"` 之类的导入检查。注意按 AGENTS.md 要求，检查须在隔离的临时工作目录、使用合成配置运行，避免隐式加载 `.env.openai`。

---

## P2-3 `data.py` 加载器的三重回退与废弃格式分支

**现状与证据**

`src/mmdd_stage1/data.py:158`、`:317` 等处的 provenance 合并写成三重回退：

```python
teacher_logit_mode=(
    record.get("teacher_logit_mode")
    or metadata.get("teacher_edge_logit_mode")
    or metadata.get("teacher_logit_mode")
),
```

`load_edge_examples`（`:130`）与 `load_target_examples`（`:186`）各写一遍这套 record→metadata 合并；后者另有约 90 行内联的多正例校验与两处废弃格式拒绝分支（`:215` 的 `positive_target_id`、`:248` 的合并 `teacher_logits`）。

**建议改法**

把"record 优先、metadata 兜底"的取值收成一个小函数 `_provenance(record, metadata, record_key, *metadata_keys)`，两个加载器共用（这是三键回退逻辑，不是一行重复，值得合并）。多正例校验抽成 `_resolve_positive_targets(...)` 返回 `positive_target_ids`，让 `load_target_examples` 主体回到可一屏读完的长度。废弃格式分支**保留**——它们防止用旧数据文件静默产出错误监督信号，属于 AGENTS.md 明确要保留的实验正确性检查。

**风险与验证**

零数值风险。验证：`pytest tests/test_stage1_models.py -q` 及数据加载相关测试；另用一份现有训练 JSONL 加载后比对 dataclass 字段逐一相等。

---

## P2-4 `src/` 下一次性轮次脚本堆积

**现状与证据**

`src/` 下 `run_stage1_*` / `summarize_stage1_*` / `evaluate_stage1_*` / `audit_stage1_*` 共 40 个文件，每轮新增一对（脚本 + 对应一次性测试），与核心包平铺在同一目录。

**建议改法**

不改动、不删除已有脚本（产物可复现性依赖它们）。仅约定后续新增放入 `src/experiments/`（或 `src/rounds/`）子目录，保持 `src/` 顶层只有核心包与长期入口。这是**约定变更**，不是重构工作量。

**风险与验证**

无。

---

## 建议执行顺序

1. **P0-3 → P0-4 → P0-2 → P0-1**：先消除双维护正确性风险。P0-3 与 P0-4 都要求"先写等价性对照测试，通过后再删其一"，这是整套计划的安全阀。
2. **P1-3 → P1-1 → P1-2**：涉及采样序列与参数默认值，必须在两次实验之间的空窗期做，并用快照比对确认。
3. **P1-4 → P1-5 → P1-6 → P1-7**：结构性清理，风险低。
4. **P2-\***：择机。

每一步单独提交，提交信息说明"行为保持"及其验证方式。

## 验证命令

```bash
conda run -n MMDD python -m pytest tests -q
```

单项验证用对应测试文件（各项已注明）。涉及数值的项（P0-4、P1-3、P1-6）一律先加 `torch.allclose` / 逐位相等的对照测试，确认等价后再删除旧实现——不要反序。

## 明确不做的事

- 不改 r9 冻结协议的任何默认值：`logsumexp`、evidence-top-k 4、`weighted_rrf` evidence 权重 0.05。
- 不删除 r6/r8 消融用到的聚合与融合模式，不改已有 round 脚本及其产物。
- 不为消除少量重复引入基类、注册表或插件层（AGENTS.md）；不加生产级并发、容错或可扩展性机制。
- 不删除保护实验正确性的校验：协议 split 检查、废弃数据格式拒绝、teacher checkpoint sha256 校验、指纹校验。
- 不改 Stage-2 与 `mmdd_dataset` 代码。


